import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import numpy as np
import matplotlib.pyplot as plt
from envs.env_std import RBCKLEnv
from algos.nn import ActorCritic
from algos.make_train import make_train
from exps.utils import simulate, save_results, lorenz, gini

import time

ALPHA = 0.36
BETA  = 0.95
B     = 5.0

N_GRID = 3   # n → n×n agents, n κ values, n λ values

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]   # one per spread experiment (always 3)


# ── Grid builders ─────────────────────────────────────────────────────────────

def _make_kappa_spreads(n):
    """Three κ configurations for an n-point grid: homogeneous, moderate, wide."""
    return [
        [1.0] * n,
        list(np.linspace(0.8, 1.2, n)),
        list(np.linspace(0.0, 2.0, n)),
    ]


def _make_lambda_vals(n):
    """n λ values spread symmetrically around 1.0."""
    return list(np.linspace(0.98, 1.02, n))


def _spread_label(kv):
    unique = sorted(set(round(v, 2) for v in kv))
    if len(unique) == 1:
        return f"[{unique[0]:.2f} | {unique[0]:.2f}]"
    return "[" + " | ".join(f"{v:.2f}" for v in unique) + "]"


def _build_arrays(kappa_vals, lambda_vals):
    """n×n grid: outer loop over κ, inner over λ → n² agents."""
    kappas  = np.array([k for k in kappa_vals for _ in lambda_vals], dtype=np.float32)
    lambdas = np.array([l for _ in kappa_vals for l in lambda_vals], dtype=np.float32)
    return kappas, lambdas


def _make_env(kappas, lambdas):
    return RBCKLEnv(
        n_agents=len(kappas),
        kappas=kappas,
        lambdas=lambdas,
        alpha=ALPHA,
        delta=0.025,
        beta=BETA,
        b=B,
        rho=0.9,
        sigma=0.01,
        max_steps=500,
        k_init=0.1,
        obs_vars=("capital", "mean_capital", "labour", "mean_labour",
                  "TFP", "kappa", "lambda"),
    )


# ── Training config ───────────────────────────────────────────────────────────

CONFIG = {
    "NUM_ENVS":        20,
    "ROLLOUT_LEN":     200,
    "TOTAL_TIMESTEPS": 1_000_000,
    "UPDATE_EPOCHS":   10,
    "NUM_MINIBATCHES": 40,
    "LR":              3e-4,
    "GAMMA":           0.95,
    "GAE_LAMBDA":      0.95,
    "CLIP_EPS":        0.2,
    "VF_COEF":         0.5,
    "ENT_COEF":        0.00,
    "HIDDEN_DIMS":     (64, 64),
    "ACTIVATION":      "tanh",
}


# ── Experiment runner ─────────────────────────────────────────────────────────

def run_general_experiment(name="hetero_rbc", n_grid=N_GRID):
    kappa_spreads = _make_kappa_spreads(n_grid)
    lambda_vals   = _make_lambda_vals(n_grid)
    spread_labels = [_spread_label(ks) for ks in kappa_spreads]

    n_agents = n_grid * n_grid
    print(f"Running General experiment: {name}  "
          f"(3 runs × {n_agents} agents  [{n_grid}×{n_grid} grid])")
    os.makedirs("results/general", exist_ok=True)

    results = []
    net = ActorCritic(action_dim=2, hidden_dims=CONFIG["HIDDEN_DIMS"])

    for i, kv in enumerate(kappa_spreads):
        kappas, lambdas = _build_arrays(kv, lambda_vals)
        env      = _make_env(kappas, lambdas)
        t0 = time.perf_counter()    
        train_fn = make_train(env, CONFIG)
        out      = train_fn(jax.random.PRNGKey(42 + i * 100))
        t1 = time.perf_counter()
        print(f"training time: {(t1 - t0)/60:.4f} mins")

        rec = simulate(env, net, out["params"], jax.random.PRNGKey(i), n_steps=2000)
        results.append((kappas, lambdas, rec, out["metrics"]))
        save_results(out["metrics"], rec, "results/general",
                     f"{name}_n{n_grid}_spread{i}")

    _plot_figure5(results, spread_labels, n_grid, name)


# ── Plotting ──────────────────────────────────────────────────────────────────

def _stationary_slice(arr, burn_frac=0.25):
    cut = int(arr.shape[0] * burn_frac)
    return arr[cut:]


def _plot_figure5(results, spread_labels, n_grid, name):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # ── Panel 1: Lorenz curves, one per κ-spread experiment ──────────────────
    ax = axes[0]
    for i, ((_, __, rec, ___), label) in enumerate(zip(results, spread_labels)):
        ks_stat = _stationary_slice(rec["ks"]).flatten()
        lx, ly  = lorenz(ks_stat)
        g       = gini(ks_stat)
        step    = max(1, len(lx) // 30)
        ax.plot(lx[::step], ly[::step],
                color=COLORS[i], marker="s", markersize=4, markevery=1, lw=1.5,
                label=f"{label}  Gini={g:.2f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("Cumulative share of population")
    ax.set_ylabel("Cumulative share of wealth")
    ax.legend(title="Cap. prod. ($\\kappa^i$)", fontsize=8, title_fontsize=9)

    # ── Panel 2: Policy scatter from wide-spread experiment ───────────────────
    # Colored by κ group; shows how capital productivity shapes savings behaviour.
    ax = axes[1]
    kappasW, _, recW, __ = results[2]
    unique_kappas = sorted(set(kappasW.tolist()))
    pcmap = plt.get_cmap("tab10")
    cutW  = int(recW["wealths"].shape[0] * 0.25)

    for j, kv in enumerate(unique_kappas):
        idx   = np.where(np.isclose(kappasW, kv))[0]
        w_all = recW["wealths"][cutW:, idx].flatten()
        c_all = recW["c_fracs"][cutW:, idx].flatten()
        n_samp = min(600, len(w_all))
        ridx   = np.random.default_rng(j).choice(len(w_all), n_samp, replace=False)
        ax.scatter(w_all[ridx], c_all[ridx],
                   s=6, alpha=0.5, color=pcmap(j),
                   label=f"$\\kappa^i = {kv:.2f}$")

    ax.set_xlabel("Wealth ($a$)")
    ax.set_ylabel("Cons. frac. $c$")
    ax.legend(fontsize=8)

    # ── Panel 3: Capital distributions – one curve per agent (moderate spread) ─
    # n²=n_grid² unique (κ,λ) pairs → genuinely distinct distributions.
    # Legend labels the four grid corners spanning the full (κ,λ) range.
    ax = axes[2]
    kappasM, lambdasM, recM, _ = results[1]
    ks_stat = _stationary_slice(recM["ks"])          # (T', n²)
    n_agents = kappasM.shape[0]                      # = n_grid²

    # Corner indices in the row-major n×n grid
    corners = {0, n_grid - 1, n_agents - n_grid, n_agents - 1}
    tab20   = plt.get_cmap("tab20")
    legend_handles = []

    for i in range(n_agents):
        col = tab20(i / n_agents)
        ax.hist(ks_stat[:, i].flatten(), bins=50, density=True,
                alpha=0.55, color=col)
        if i in corners:
            legend_handles.append(
                plt.Line2D([0], [0], color=col, linewidth=3,
                           label=f"$\\kappa^{{{i+1}}}$={kappasM[i]:.2f}, "
                                 f"$\\lambda^{{{i+1}}}$={lambdasM[i]:.2f}")
            )

    ax.legend(handles=legend_handles, fontsize=7)
    ax.set_xlabel("Capital ($k$)")
    ax.set_ylabel("Density")

    plt.tight_layout()
    out_path = f"results/general/{name}_n{n_grid}_figure5.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Figure saved → {out_path}")


if __name__ == "__main__":
    run_general_experiment()
    
