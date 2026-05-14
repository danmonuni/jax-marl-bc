import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import time
import numpy as np
import jax
import jax.numpy as jnp

from envs.env_ksmf import RBCKSEnv
from envs.env_std import RBCKLEnv
from algos.nn import ActorCritic
from exps.utils import simulate, simulate_seeds

N_AGENTS = 4
N_STEPS  = 120   # covers >2 episodes (max_steps=50)


# ── Env / net factories ───────────────────────────────────────────────────────

def make_ks_env():
    return RBCKSEnv(
        n_agents=N_AGENTS,
        kappas=np.ones(N_AGENTS, np.float32),
        lambdas=np.ones(N_AGENTS, np.float32),
        alpha=0.36, delta=0.025, beta=0.95,
        max_steps=50, k_init=1.0,
        obs_vars=("capital", "mean_capital", "aggregate_state", "kappa", "lambda"),
    )


def make_kl_env():
    return RBCKLEnv(
        n_agents=N_AGENTS,
        kappas=np.ones(N_AGENTS, np.float32),
        lambdas=np.ones(N_AGENTS, np.float32),
        alpha=0.36, delta=0.025, beta=0.95,
        b=5.0, rho=0.9, sigma=0.01,
        max_steps=50, k_init=0.1,
        obs_vars=("capital", "mean_capital", "labour", "mean_labour",
                  "TFP", "kappa", "lambda"),
    )


def make_net_params(env):
    net    = ActorCritic(action_dim=env.act_dim, hidden_dims=(32, 32))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,)))
    return net, params


# ── Shape tests ───────────────────────────────────────────────────────────────

def test_ks_output_shapes():
    env = make_ks_env()
    net, params = make_net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(42), n_steps=N_STEPS)

    assert rec["ks"].shape        == (N_STEPS, N_AGENTS), rec["ks"].shape
    assert rec["wealths"].shape   == (N_STEPS, N_AGENTS)
    assert rec["c_fracs"].shape   == (N_STEPS, N_AGENTS)
    assert rec["ls"].shape        == (N_STEPS, N_AGENTS)
    assert rec["K"].shape         == (N_STEPS,)
    assert rec["reward"].shape    == (N_STEPS,)
    assert rec["emp_states"].shape == (N_STEPS, N_AGENTS)
    assert rec["agg_state"].shape  == (N_STEPS,)
    print("PASS  test_ks_output_shapes")


def test_kl_output_shapes():
    """RBCKLEnv has act_dim=2 and no emp_states / agg_state fields."""
    env = make_kl_env()
    net, params = make_net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(42), n_steps=N_STEPS)

    assert rec["ks"].shape      == (N_STEPS, N_AGENTS)
    assert rec["c_fracs"].shape == (N_STEPS, N_AGENTS)
    # ls comes from the action (not state.ls) when act_dim > 1
    assert rec["ls"].shape      == (N_STEPS, N_AGENTS)
    # agg_state falls back to scalar constant 1
    assert rec["agg_state"].shape == (N_STEPS,)
    print("PASS  test_kl_output_shapes")


# ── Value-sanity tests ────────────────────────────────────────────────────────

def test_ks_value_sanity():
    env = make_ks_env()
    net, params = make_net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(7), n_steps=N_STEPS)

    assert np.all(rec["ks"]    > 0),                    "capital should be positive"
    assert np.all(rec["c_fracs"] >= 0),                 "c_fracs lower bound"
    assert np.all(rec["c_fracs"] <= 1),                 "c_fracs upper bound"
    assert np.all(np.isfinite(rec["reward"])),           "rewards must be finite"
    assert np.all(np.isin(rec["emp_states"], [0, 1])),  "emp_states ∈ {0,1}"
    assert np.all(np.isin(rec["agg_state"],  [0, 1])),  "agg_state ∈ {0,1}"
    print("PASS  test_ks_value_sanity")


def test_kl_value_sanity():
    env = make_kl_env()
    net, params = make_net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(7), n_steps=N_STEPS)

    assert np.all(rec["ks"]    > 0)
    assert np.all(rec["c_fracs"] >= 0) and np.all(rec["c_fracs"] <= 1)
    assert np.all(np.isfinite(rec["reward"]))
    print("PASS  test_kl_value_sanity")


# ── Determinism ───────────────────────────────────────────────────────────────

def test_determinism():
    env = make_ks_env()
    net, params = make_net_params(env)

    rec1 = simulate(env, net, params, jax.random.PRNGKey(99), n_steps=N_STEPS)
    rec2 = simulate(env, net, params, jax.random.PRNGKey(99), n_steps=N_STEPS)

    np.testing.assert_array_equal(rec1["ks"],     rec2["ks"])
    np.testing.assert_array_equal(rec1["reward"], rec2["reward"])
    print("PASS  test_determinism")


def test_different_seeds_differ():
    env = make_ks_env()
    net, params = make_net_params(env)

    rec1 = simulate(env, net, params, jax.random.PRNGKey(1), n_steps=N_STEPS)
    rec2 = simulate(env, net, params, jax.random.PRNGKey(2), n_steps=N_STEPS)

    assert not np.allclose(rec1["ks"], rec2["ks"]), "different seeds must diverge"
    print("PASS  test_different_seeds_differ")


# ── simulate_seeds (vmap) ─────────────────────────────────────────────────────

def test_simulate_seeds_shapes():
    env    = make_ks_env()
    net, params = make_net_params(env)
    N_SEEDS = 3
    keys = jax.random.split(jax.random.PRNGKey(0), N_SEEDS)

    rec = simulate_seeds(env, net, params, keys, n_steps=N_STEPS)

    assert rec["ks"].shape     == (N_SEEDS, N_STEPS, N_AGENTS)
    assert rec["reward"].shape == (N_SEEDS, N_STEPS)
    print("PASS  test_simulate_seeds_shapes")


def test_simulate_seeds_matches_single():
    """Each slice of simulate_seeds must equal the corresponding simulate call."""
    env    = make_ks_env()
    net, params = make_net_params(env)
    N_SEEDS = 2
    keys = jax.random.split(jax.random.PRNGKey(5), N_SEEDS)

    batch = simulate_seeds(env, net, params, keys, n_steps=N_STEPS)
    for i in range(N_SEEDS):
        single = simulate(env, net, params, keys[i], n_steps=N_STEPS)
        np.testing.assert_allclose(
            batch["ks"][i], single["ks"], rtol=1e-5,
            err_msg=f"seed {i}: simulate_seeds slice != simulate",
        )
    print("PASS  test_simulate_seeds_matches_single")


# ── Timing ────────────────────────────────────────────────────────────────────

def _python_loop_simulate(env, net, params, key, n_steps=2000):
    """Reference Python-loop implementation for timing comparison."""
    obs, state = env.reset(key)
    rec = dict(ks=[], wealths=[], c_fracs=[], K=[], ls=[], reward=[], emp_states=[], agg_state=[])
    for _ in range(n_steps):
        obs_mat = jnp.stack([obs[a] for a in env.agents])
        pi, _   = net.apply(params, obs_mat)
        mu      = pi.loc
        acts    = {a: mu[i] for i, a in enumerate(env.agents)}
        key, sk = jax.random.split(key)
        obs, state, rews, dones, _ = env.step(sk, state, acts)
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
        rec["agg_state"].append(int(state.agg_state) if hasattr(state, "agg_state") else 1)
        if bool(dones["__all__"]):
            key, sk = jax.random.split(key)
            obs, state = env.reset(sk)
    return {k: np.array(v) for k, v in rec.items()}


def test_timing():
    env = make_ks_env()
    net, params = make_net_params(env)
    key = jax.random.PRNGKey(0)
    n   = 500

    # Warm-up: first call triggers XLA compilation
    simulate(env, net, params, key, n_steps=n)

    t0 = time.perf_counter()
    simulate(env, net, params, key, n_steps=n)
    t_scan = time.perf_counter() - t0

    t0 = time.perf_counter()
    _python_loop_simulate(env, net, params, key, n_steps=n)
    t_loop = time.perf_counter() - t0

    speedup = t_loop / t_scan
    print(f"TIMING  loop={t_loop:.3f}s  scan={t_scan:.3f}s  speedup={speedup:.1f}x")
    assert speedup > 1.0, f"scan ({t_scan:.3f}s) should beat loop ({t_loop:.3f}s)"


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_ks_output_shapes()
    test_kl_output_shapes()
    test_ks_value_sanity()
    test_kl_value_sanity()
    test_determinism()
    test_different_seeds_differ()
    test_simulate_seeds_shapes()
    test_simulate_seeds_matches_single()
    test_timing()
    print("\nAll tests passed.")
