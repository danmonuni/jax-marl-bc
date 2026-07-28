"""Randomized starting capital: per-env draw, persisted across episode resets."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

from jmbc.envs import RBCKSEnv

N_AGENTS = 200
K_INIT = 1.0
SIGMA = 0.3
LOW, HIGH = 3.0, 20.0
MAX_STEPS = 5


def _env(dist="lognormal", sigma=SIGMA, max_steps=MAX_STEPS,
         resample="per_env"):
    return RBCKSEnv(
        n_agents=N_AGENTS,
        kappas=np.ones(N_AGENTS, np.float32),
        lambdas=np.ones(N_AGENTS, np.float32),
        max_steps=max_steps, k_init=K_INIT, k_init_dist=dist,
        k_init_resample=resample, k_init_sigma=sigma,
        k_init_low=LOW, k_init_high=HIGH,
        obs_vars=("capital", "labour", "mean_capital", "aggregate_state"),
    )


def test_constant_is_the_old_behavior():
    """dist='constant' must reproduce the pre-refactor behavior exactly."""
    env = _env(dist="constant")
    _, st = env.reset_mat(jax.random.PRNGKey(0))
    np.testing.assert_array_equal(np.asarray(st.ks),
                                  np.full(N_AGENTS, K_INIT, np.float32))
    print("PASS  test_constant_is_the_old_behavior")


def test_uniform_respects_its_support():
    """The reference U(low, high) initialization, per agent."""
    env = _env(dist="uniform")
    keys = jax.random.split(jax.random.PRNGKey(1), 8)
    _, st = jax.vmap(env.reset_mat)(keys)
    ks = np.asarray(st.ks)
    assert ks.min() >= LOW and ks.max() <= HIGH, (ks.min(), ks.max())
    # spans most of the support, and averages near its midpoint
    assert ks.min() < LOW + 3 and ks.max() > HIGH - 3, (ks.min(), ks.max())
    mid = 0.5 * (LOW + HIGH)
    assert abs(ks.mean() - mid) < 1.5, ks.mean()
    print(f"PASS  test_uniform_respects_its_support  "
          f"(min={ks.min():.2f}, mean={ks.mean():.2f}, max={ks.max():.2f})")


def test_draw_differs_across_parallel_envs():
    """vmapped envs get independent populations, not one shared draw."""
    env = _env(dist="uniform")
    keys = jax.random.split(jax.random.PRNGKey(7), 4)
    _, st = jax.vmap(env.reset_mat)(keys)
    ks = np.asarray(st.ks)                        # [4, n]
    assert ks.shape == (4, N_AGENTS)
    for i in range(1, 4):
        assert not np.allclose(ks[0], ks[i]), f"env {i} shares env 0's draw"
    # dispersion WITHIN an env too (not a per-env constant)
    assert ks[0].std() > 0.05 * (HIGH - LOW), f"no cross-agent spread: {ks[0].std()}"
    print(f"PASS  test_draw_differs_across_parallel_envs  (std={ks[0].std():.3f})")


def test_lognormal_draw_is_mean_preserving():
    """E[k_i] = k_init for every sigma, so K_0 is not shifted by dispersion."""
    env = _env(dist="lognormal")
    keys = jax.random.split(jax.random.PRNGKey(11), 32)
    _, st = jax.vmap(env.reset_mat)(keys)
    mean_k = float(np.asarray(st.ks).mean())
    assert abs(mean_k - K_INIT) < 0.02, f"mean k_0 = {mean_k:.4f} != {K_INIT}"
    print(f"PASS  test_lognormal_draw_is_mean_preserving  (mean={mean_k:.4f})")


def test_k_init_persists_across_auto_reset():
    """An in-episode auto-reset restarts the SAME population; only a fresh
    reset_mat draws a new one."""
    env = _env(dist="uniform")
    key = jax.random.PRNGKey(5)
    _, st = env.reset_mat(key)
    k0 = np.asarray(st.k_init_vec).copy()

    acts = jnp.zeros((N_AGENTS, env.act_dim), jnp.float32)
    saw_reset = False
    for t in range(MAX_STEPS + 2):
        key, sk = jax.random.split(key)
        _, st, _, done = env.step_mat(sk, st, acts)
        if bool(done):
            saw_reset = True
            # the reset state's capital is the original draw, not a new one
            np.testing.assert_allclose(np.asarray(st.ks), k0, rtol=1e-6)
        np.testing.assert_allclose(np.asarray(st.k_init_vec), k0, rtol=1e-6)
    assert saw_reset, "episode never terminated; auto-reset path untested"

    # a genuinely new reset_mat call DOES redraw
    _, st2 = env.reset_mat(jax.random.PRNGKey(6))
    assert not np.allclose(np.asarray(st2.k_init_vec), k0)
    print("PASS  test_k_init_persists_across_auto_reset")


def test_per_episode_redraws_at_every_reset():
    """The reference behavior: each episode starts a NEW population."""
    env = _env(dist="uniform", resample="per_episode")
    key = jax.random.PRNGKey(5)
    _, st = env.reset_mat(key)
    k0 = np.asarray(st.ks).copy()

    acts = jnp.zeros((N_AGENTS, env.act_dim), jnp.float32)
    saw_reset = False
    for _ in range(MAX_STEPS + 2):
        key, sk = jax.random.split(key)
        _, st, _, done = env.step_mat(sk, st, acts)
        if bool(done):
            saw_reset = True
            ks = np.asarray(st.ks)
            assert not np.allclose(ks, k0), "per_episode reused the old population"
            assert ks.min() >= LOW and ks.max() <= HIGH, (ks.min(), ks.max())
    assert saw_reset, "episode never terminated; nothing tested"
    print("PASS  test_per_episode_redraws_at_every_reset")


def test_vec_and_dict_paths_agree_under_dispersion():
    """The legacy dict path must persist k_init_vec on auto-reset too — the
    jaxmarl base step() would otherwise redraw it and silently diverge."""
    from test_vec_path import DictOnly
    from jmbc.algos import ActorCritic
    from jmbc.diagnostics import simulate

    env = _env(dist="uniform", max_steps=20)
    net = ActorCritic(action_dim=env.act_dim, hidden_dims=(16, 16))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,)))
    rec_v = simulate(env, net, params, jax.random.PRNGKey(11), n_steps=60)
    rec_d = simulate(DictOnly(env), net, params, jax.random.PRNGKey(11), n_steps=60)
    assert rec_v["done"].any(), "no auto-reset in window; nothing tested"
    for k in rec_v:
        np.testing.assert_allclose(rec_v[k], rec_d[k], rtol=1e-6, atol=1e-7,
                                   err_msg=f"channel {k}")
    print("PASS  test_vec_and_dict_paths_agree_under_dispersion")


if __name__ == "__main__":
    test_constant_is_the_old_behavior()
    test_uniform_respects_its_support()
    test_draw_differs_across_parallel_envs()
    test_lognormal_draw_is_mean_preserving()
    test_k_init_persists_across_auto_reset()
    test_per_episode_redraws_at_every_reset()
    test_vec_and_dict_paths_agree_under_dispersion()
    print("\nAll k_init randomization tests passed.")
