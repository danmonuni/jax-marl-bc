import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
print(f"JAX running on platform: {jax.default_backend()} ({jax.devices()})")
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scipy.stats as sp_stats

from envs.env_ksmf import RBCKSEnv
from algos.nn import ActorCritic
from algos.make_train import make_train
from exps.utils import simulate, save_results, gini

import time

ALPHA = 0.36
BETA  = 0.95

# Indices into params_history to use as snapshots (set after training based on NUM_UPDATES)
N_SNAPSHOTS = 4


def params_at(params_history, idx):
    """Index into a pytree with a leading NUM_UPDATES dimension."""
    return jax.tree.map(lambda x: x[idx], params_history)


def run_ks_experiment(n_agents=20, name="ks_limit"):
    print(f"Running KS experiment: {name} (n_agents={n_agents})")

    kappas  = np.ones(n_agents)
    lambdas = np.ones(n_agents)

    env = RBCKSEnv(
        n_agents=n_agents,
        kappas=kappas,
        lambdas=lambdas,
        alpha=ALPHA,
        delta=0.025,
        beta=BETA,
        max_steps=500,
        k_init=1,
        obs_vars=("capital","mean_capital","aggregate_state","kappa","lambda"),
    )

    config = {
        "NUM_ENVS": 20,
        "ROLLOUT_LEN": 200,
        "TOTAL_TIMESTEPS": 10_000_000,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 20,
        "LR": 3e-4,
        "GAMMA": 0.95,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.00,
        "HIDDEN_DIMS": (64, 64),
        "ACTIVATION": "tanh",
    }

    t0 = time.perf_counter()
    train_fn = make_train(env, config)
    out = train_fn(jax.random.PRNGKey(123))
    t1 = time.perf_counter()
    print(f"training time: {(t1 - t0)/60:.4f} mins")

    metrics       = out["metrics"]
    params_final  = out["params"]
    params_hist   = out["params_history"]   # pytree with leading [NUM_UPDATES] dim

    save_results(metrics, {}, "results/ks", name)

    num_updates      = config["NUM_UPDATES"]  # set by make_train
    steps_per_update = config["ROLLOUT_LEN"] * config["NUM_ENVS"]

    # Four log-spaced snapshot indices (0-based); last one is always the final update
    snap_idxs = np.unique(np.round(
        np.logspace(0, np.log10(num_updates - 1), N_SNAPSHOTS)
    ).astype(int).clip(0, num_updates - 1))
    snap_steps = (snap_idxs + 1) * steps_per_update

    net = ActorCritic(action_dim=env.act_dim, hidden_dims=config["HIDDEN_DIMS"])

    # Simulate at each snapshot
    recs = []
    for idx in snap_idxs:
        p = params_at(params_hist, int(idx))
        recs.append(simulate(env, net, p, jax.random.PRNGKey(7), n_steps=2000))

    rec_before = recs[0]
    rec_after  = recs[-1]

    # ------------------------------------------------------------------ Figure
    os.makedirs("results/ks", exist_ok=True)
    fig = plt.figure(figsize=(18, 7))
    gs_outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # ---- Left: 2×2 law-of-motion scatters
    gs_left = gs_outer[0].subgridspec(2, 2, hspace=0.55, wspace=0.45)
    for panel_i, (idx, step, rec) in enumerate(zip(snap_idxs, snap_steps, recs)):
        ax = fig.add_subplot(gs_left[panel_i // 2, panel_i % 2])
        K   = np.array(rec["K"])
        agg = np.array(rec["agg_state"])
        Kt, Kt1, agt = K[:-1], K[1:], agg[:-1]
        slope, intercept, r_value, *_ = sp_stats.linregress(Kt, Kt1)
        ax.scatter(Kt[agt == 1], Kt1[agt == 1], s=2, alpha=0.3,
                   color="tab:orange", label="good")
        ax.scatter(Kt[agt == 0], Kt1[agt == 0], s=2, alpha=0.3,
                   color="tab:blue",   label="bad")
        x_r = np.linspace(Kt.min(), Kt.max(), 100)
        ax.plot(x_r, slope * x_r + intercept, color="red", lw=1.2,
                label=f"$R^2={r_value**2:.3f}$")
        ax.set_xlabel("$K_t$",     fontsize=8)
        ax.set_ylabel("$K_{t+1}$", fontsize=8)
        ax.set_title(f"Step $\\approx {step:.0e}$", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # ---- Centre: wealth histograms before / after
    gs_mid = gs_outer[1].subgridspec(2, 1, hspace=0.55)
    for ax, rec, label in zip(
        [fig.add_subplot(gs_mid[0]), fig.add_subplot(gs_mid[1])],
        [rec_before, rec_after],
        ["Untrained", "Trained"],
    ):
        # Avoid auto-reset boundary: skip every max_steps-th entry (k returns to k_init)
        k_vals = rec["ks"][-(env.max_steps):-1].flatten()
        g = gini(k_vals)
        ax.hist(k_vals, bins=40, density=True, color="steelblue", alpha=0.75)
        ax.set_title(f"{label}  (Gini={g:.3f})", fontsize=9)
        ax.set_xlabel("Capital $k^i$", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    # ---- Right: MPC scatter before / after, coloured by employment
    gs_right = gs_outer[2].subgridspec(2, 1, hspace=0.55)
    for ax, rec, label in zip(
        [fig.add_subplot(gs_right[0]), fig.add_subplot(gs_right[1])],
        [rec_before, rec_after],
        ["Untrained", "Trained"],
    ):
        # Exclude episode-end reset steps (every max_steps-th step)
        tail = slice(-2 * env.max_steps, None)
        mask = np.ones(2 * env.max_steps, dtype=bool)
        mask[env.max_steps - 1::env.max_steps] = False
        W   = rec["wealths"][tail][mask].flatten()
        CF  = rec["c_fracs"][tail][mask].flatten()
        emp = rec["emp_states"][tail][mask].flatten()

        ax.scatter(W[emp == 1], CF[emp == 1], s=1, alpha=0.3,
                   color="tab:orange", label="employed")
        ax.scatter(W[emp == 0], CF[emp == 0], s=1, alpha=0.3,
                   color="tab:blue",   label="unemployed")
        ax.set_title(label,                     fontsize=9)
        ax.set_xlabel("Wealth $a^i$",            fontsize=8)
        ax.set_ylabel("Cons. frac. $\\hat{c}^i$", fontsize=8)
        ax.legend(fontsize=7, markerscale=4)
        ax.tick_params(labelsize=7)

    plt.savefig(f"results/ks/{name}_fig4.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved results/ks/{name}_fig4.png")


if __name__ == "__main__":
    run_ks_experiment()
    
    
