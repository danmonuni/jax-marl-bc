"""Diagnostics correctness checks that do not require training."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

from jmbc.envs import RBCKLEnv, RBCKSEnv
from jmbc.algos import ActorCritic
from jmbc.diagnostics import simulate
from jmbc.diagnostics.economic import (
    resource_residual, analytical_rbc_compare, den_haan_stat, euler_errors,
)
from jmbc.diagnostics.distributional import gini, top_shares

N_AGENTS = 6
N_STEPS = 300


def _net_params(env):
    net = ActorCritic(action_dim=env.act_dim, hidden_dims=(32, 32))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,)))
    return net, params


def _rbc_env(delta=0.025):
    return RBCKLEnv(
        n_agents=N_AGENTS, kappas=np.ones(N_AGENTS), lambdas=np.ones(N_AGENTS),
        delta=delta, max_steps=100, k_init=1.0,
        obs_vars=("capital", "mean_capital"),
    )


def test_resource_identity_holds():
    """Aggregate resource constraint is an accounting identity (~0) for any policy."""
    env = _rbc_env()
    net, params = _net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(1), n_steps=N_STEPS)
    out = resource_residual(rec, env.delta, burn_frac=0.2)
    assert out["resource_mean_rel"] < 1e-2, out
    print("PASS  test_resource_identity_holds", out)


def test_resource_identity_ks():
    env = RBCKSEnv(
        n_agents=N_AGENTS, kappas=np.ones(N_AGENTS), lambdas=np.ones(N_AGENTS),
        delta=0.025, max_steps=100, k_init=1.0,
        obs_vars=("capital", "mean_capital", "aggregate_state"),
    )
    net, params = _net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(2), n_steps=N_STEPS)
    out = resource_residual(rec, env.delta, burn_frac=0.2)
    assert out["resource_mean_rel"] < 1e-2, out
    print("PASS  test_resource_identity_ks", out)


def test_analytical_targets():
    """Closed-form targets only returned for the textbook delta=1 case."""
    env = _rbc_env(delta=1.0)
    net, params = _net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(3), n_steps=N_STEPS)
    cmp = analytical_rbc_compare(rec, env.alpha, env.beta, env.b, 1.0)
    assert abs(cmp["c_star"] - (1 - env.alpha * env.beta)) < 1e-9
    assert 0 < cmp["l_star"] < 1
    # delta != 1 -> no closed form
    assert analytical_rbc_compare(rec, env.alpha, env.beta, env.b, 0.025) is None
    print("PASS  test_analytical_targets", {k: round(v, 4) for k, v in cmp.items()})


def test_euler_finite_and_bounded():
    env = _rbc_env()
    net, params = _net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(4), n_steps=N_STEPS)
    out = euler_errors(rec, env.beta)
    assert np.isfinite(out["euler_mean_abs"])
    assert 0.0 <= out["constrained_frac"] <= 1.0
    print("PASS  test_euler_finite_and_bounded", {k: round(v, 4) for k, v in out.items()})


def test_distributional_basics():
    assert abs(gini(np.ones(100))) < 1e-6          # perfect equality
    assert gini([0] * 99 + [1]) > 0.9              # near-perfect inequality
    sh = top_shares(np.ones(100))
    assert abs(sh["top_0.1_share"] - 0.1) < 1e-6   # equal -> top 10% holds 10%
    print("PASS  test_distributional_basics")


def test_den_haan_runs():
    env = RBCKSEnv(
        n_agents=N_AGENTS, kappas=np.ones(N_AGENTS), lambdas=np.ones(N_AGENTS),
        delta=0.025, max_steps=100, k_init=1.0,
        obs_vars=("capital", "mean_capital", "aggregate_state"),
    )
    net, params = _net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(5), n_steps=N_STEPS)
    out = den_haan_stat(rec)
    assert np.isfinite(out["ks_lom_r2"])
    assert np.isfinite(out["den_haan_max_pct"])
    print("PASS  test_den_haan_runs", {k: round(v, 3) for k, v in out.items()})


if __name__ == "__main__":
    test_resource_identity_holds()
    test_resource_identity_ks()
    test_analytical_targets()
    test_euler_finite_and_bounded()
    test_distributional_basics()
    test_den_haan_runs()
    print("\nAll diagnostics tests passed.")
