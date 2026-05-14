import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import numpy as np
import matplotlib.pyplot as plt
from envs.env_std import RBCKLEnv
from algos.make_train import make_train
from exps.utils import save_results

import time

ALPHA = 0.36
BETA  = 0.95
B     = 5.0


def run_rbc_experiment(delta, name, c_target, l_target):
    print(f"Running RBC experiment: {name} (delta={delta})")

    env = RBCKLEnv(
        n_agents=1,
        kappas=[1.0],
        lambdas=[1.0],
        alpha=ALPHA,
        delta=delta,
        beta=BETA,
        b=B,
        rho=0.9,
        sigma=0.01,
        max_steps=1000,
        k_init=1,
        obs_vars=("capital",),
    )

    config = {
        "NUM_ENVS": 20,
        "ROLLOUT_LEN": 200,
        "TOTAL_TIMESTEPS": 1_000_000,
        "UPDATE_EPOCHS": 10,
        "NUM_MINIBATCHES": 10,
        "LR": 3e-4,
        "GAMMA": 0.95,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "HIDDEN_DIMS": (64, 64),
        "ACTIVATION": "tanh",
    }

    train_fn = make_train(env, config)
    out = train_fn(jax.random.PRNGKey(42))
    metrics = out["metrics"]

    save_results(metrics, {}, "results/rbc", name)

    # c_frac_env / l_env: shape [NUM_UPDATES, NUM_ENVS]
    c_env = np.array(metrics["c_frac_env"])
    l_env = np.array(metrics["l_env"])

    steps_per_update = config["ROLLOUT_LEN"] * config["NUM_ENVS"]
    env_steps = np.arange(1, len(c_env) + 1) * steps_per_update

    c_mu, c_sd = c_env.mean(-1), c_env.std(-1)
    l_mu, l_sd = l_env.mean(-1), l_env.std(-1)

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(env_steps, c_mu, color="tab:blue",   label=r"$\hat{c}$")
    ax.fill_between(env_steps, c_mu - c_sd, c_mu + c_sd, color="tab:blue",   alpha=0.25)
    ax.axhline(c_target, color="tab:blue",   linestyle="--", label=r"$c^*$")

    ax.plot(env_steps, l_mu, color="tab:orange", label=r"$\hat{\ell}$")
    ax.fill_between(env_steps, l_mu - l_sd, l_mu + l_sd, color="tab:orange", alpha=0.25)
    ax.axhline(l_target, color="tab:orange", linestyle="--", label=r"$\ell^*$")

    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(r"Labor supply $\hat{\ell}$, Cons. frac. $\hat{c}$")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", color="lightgray", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(name)

    plt.tight_layout()
    plt.savefig(f"results/rbc/{name}_policy.png", dpi=150)
    plt.close()
    print(f"Saved results/rbc/{name}_policy.png")


if __name__ == "__main__":
    c_star_1 = 1 - ALPHA * BETA
    l_star_1 = ALPHA / (B * (1 - (1 - ALPHA) * BETA) + ALPHA)
    print(f"Analytical targets (delta=1.0): c*={c_star_1:.4f}, l*={l_star_1:.4f}")

    t0 = time.perf_counter()
    run_rbc_experiment(delta=1.0,   name="rbc_textbook", c_target=c_star_1, l_target=l_star_1)
    t1 = time.perf_counter()
    print(f"rbc_textbook execution time: {(t1 - t0)/60:.4f} mins")
    run_rbc_experiment(delta=0.025, name="rbc_typical",  c_target=0.1611,   l_target=0.1222)
    t2 = time.perf_counter()
    print(f"rbc_typical execution time: {(t2 - t1)/60:.4f} mins")