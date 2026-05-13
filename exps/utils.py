import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

def smooth(x, w=50):
    x = np.array(x)
    if len(x) < w: return x
    return np.convolve(x, np.ones(w)/w, mode="valid")

def gini(values):
    v = np.sort(np.abs(np.array(values).flatten()))
    n = len(v)
    if n == 0: return 0.0
    s = v.sum() + 1e-12
    return (2 * np.dot(np.arange(1, n + 1), v) / (n * s)) - (n + 1) / n

def lorenz(values):
    v = np.sort(np.abs(np.array(values).flatten()))
    if len(v) == 0: return np.array([0, 1]), np.array([0, 1])
    return (np.r_[0, np.arange(1, len(v) + 1) / len(v)],
            np.r_[0, np.cumsum(v) / (v.sum() + 1e-12)])

def mpc_curve(w, cf, n_bins=40):
    if len(w) == 0: return np.array([]), np.array([])
    lo, hi = np.percentile(w, 2), np.percentile(w, 98)
    if lo == hi: hi += 1e-5
    bins = np.linspace(lo, hi, n_bins + 1)
    cx = 0.5 * (bins[:-1] + bins[1:])
    cy = []
    for i in range(n_bins):
        mask = (w >= bins[i]) & (w < bins[i+1])
        if np.sum(mask) > 0:
            cy.append(np.median(cf[mask]))
        else:
            cy.append(np.nan)
    return cx, np.array(cy)

def simulate(env, net, params, key, n_steps=2000):
    """Deterministic (mean-action) rollout for diagnostics."""
    obs, state = env.reset(key)
    rec = dict(ks=[], wealths=[], c_fracs=[], K=[], ls=[], reward=[], emp_states=[], agg_state=[])

    for _ in range(n_steps):
        obs_mat = jnp.stack([obs[a] for a in env.agents])
        pi, _ = net.apply(params, obs_mat)
        mu = pi.loc  # deterministic mean action

        acts_dict = {a: mu[i] for i, a in enumerate(env.agents)}

        key, sk = jax.random.split(key)
        obs, state, rews, dones, _ = env.step(sk, state, acts_dict)

        rec["ks"].append(np.array(state.ks))
        rec["wealths"].append(np.array(state.wealths))
        rec["c_fracs"].append(np.array((mu[:, 0] + 1) / 2))
        rec["ls"].append(np.array((mu[:, 1] + 1) / 2) if mu.shape[1] > 1 else np.array(state.ls))
        rec["K"].append(float(jnp.mean(state.ks)))
        rec["reward"].append(float(np.mean([rews[a] for a in env.agents])))
        rec["emp_states"].append(
            np.array(state.emp_states) if hasattr(state, "emp_states")
            else np.ones(len(env.agents), dtype=int)
        )
        rec["agg_state"].append(
            int(state.agg_state) if hasattr(state, "agg_state") else 1
        )

        if bool(dones["__all__"]):
            key, sk = jax.random.split(key)
            obs, state = env.reset(sk)

    return {k: np.array(v) for k, v in rec.items()}

def save_results(metrics, rec, folder, name):
    os.makedirs(folder, exist_ok=True)
    
    # metrics["loss"] is (total_loss, (value_loss, loss_actor, entropy))
    # stacked over updates and epochs.
    # We take the mean over epochs for plotting.
    if isinstance(metrics["loss"], tuple):
        total_loss = metrics["loss"][0] # Shape [NUM_UPDATES, UPDATE_EPOCHS, NUM_MINIBATCHES] ? 
        # Actually it depends on the scans. 
        # Let's be safe and take the mean over all but the first dimension.
        while total_loss.ndim > 1:
            total_loss = total_loss.mean(axis=-1)
    else:
        total_loss = metrics["loss"]

    # Plot training metrics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].plot(smooth(total_loss))
    axes[0].set_title("Total Loss")
    
    reward = metrics["step_reward"]
    while reward.ndim > 1:
        reward = reward.mean(axis=-1)
    axes[1].plot(smooth(reward))
    axes[1].set_title("Step Reward")
    
    ep_ret = metrics["returned_episode_returns"]
    while ep_ret.ndim > 1:
        ep_ret = ep_ret.mean(axis=-1)
    axes[2].plot(smooth(ep_ret))
    axes[2].set_title("Episode Return")
    
    plt.tight_layout()
    plt.savefig(f"{folder}/{name}_training.png")
    plt.close()
    
    # Save raw metrics
    # np.savez(f"{folder}/{name}_data.npz", metrics=metrics, rec=rec)
    print(f"Results saved to {folder}/{name}_*")
