import os
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from envs.env_std import RBCKLEnv
from algos.nn import ActorCritic
from algos.make_train import make_train
from exps.utils import simulate, save_results, lorenz, gini

# Parameters from Paper Table 1
ALPHA = 0.36
BETA = 0.95
B = 5.0

def run_general_experiment(name="hetero_rbc_9"):
    print(f"Running General experiment: {name} (9 agents, 3x3 grid)")
    
    # Grid of productivities from paper Section 4.3
    gv = [0.98, 1.0, 1.02]
    kappas = np.array([k for k in gv for _ in gv], dtype=np.float32)
    lambdas = np.array([l for _ in gv for l in gv], dtype=np.float32)
    # prods for coloring
    prods = kappas * lambdas

    env = RBCKLEnv(
        n_agents=9, 
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
        obs_vars=("capital","mean_capital","labour","mean_labour","TFP","kappa","lambda")
    )

    config = {
        "NUM_ENVS": 8, 
        "ROLLOUT_LEN": 500,
        "TOTAL_TIMESTEPS": 400_000, 
        "UPDATE_EPOCHS": 4, 
        "NUM_MINIBATCHES": 4,
        "LR": 3e-4, 
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "VF_COEF": 0.5,
        "ENT_COEF": 0.01,
        "HIDDEN_DIMS": (64, 64),
        "ACTIVATION": "tanh"
    }

    train_fn = make_train(env, config)
    out = train_fn(jax.random.PRNGKey(789))
    
    metrics = out["metrics"]
    params = out["params"]
    
    # Diagnostics
    net = ActorCritic(action_dim=env.act_dim, hidden_dims=config["HIDDEN_DIMS"])
    rec = simulate(env, net, params, jax.random.PRNGKey(0), n_steps=2000)
    
    save_results(metrics, rec, "results/general", name)
    
    # General Specific Plots (Figure 5 reproduction)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Capital Paths colored by productivity
    cmap = plt.get_cmap("RdYlGn")
    norm_prods = (prods - prods.min()) / (prods.max() - prods.min() + 1e-8)
    
    for i in range(9):
        axes[0].plot(rec["ks"][:, i], color=cmap(norm_prods[i]), alpha=0.7)
    axes[0].set_title("Capital Accumulation by Agent")
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Capital $k^i$")
    
    # 2. Lorenz Curve
    wealth_final = rec["wealths"][-1].flatten()
    lx, ly = lorenz(wealth_final)
    g = gini(wealth_final)
    axes[1].plot(lx, ly, label=f"Gini={g:.3f}")
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1].set_title("Lorenz Curve (Wealth)")
    axes[1].set_xlabel("Cumulative share of population")
    axes[1].set_ylabel("Cumulative share of wealth")
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f"results/general/{name}_hetero.png")
    plt.close()

if __name__ == "__main__":
    run_general_experiment()
