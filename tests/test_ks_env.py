"""KS environment semantics: aggregate risk and employment calibration."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

from jmbc.envs import RBCKSEnv
from jmbc.envs.env_ksmf import KS_PZ, KS_PUU, KS_PEU
from jmbc.algos import ActorCritic
from jmbc.diagnostics import simulate

N_AGENTS = 50
N_STEPS = 1500


def _env(max_steps=2000):
    return RBCKSEnv(
        n_agents=N_AGENTS,
        kappas=np.ones(N_AGENTS, np.float32),
        lambdas=np.ones(N_AGENTS, np.float32),
        max_steps=max_steps, k_init=1.0,
        obs_vars=("capital", "labour", "mean_capital", "aggregate_state"),
    )


def _net_params(env):
    net = ActorCritic(action_dim=env.act_dim, hidden_dims=(32, 32))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,)))
    return net, params


def test_conditional_chain_is_valid():
    """Transition tables are proper probabilities and rows of P_z sum to 1."""
    for M in (KS_PZ, KS_PUU, KS_PEU):
        m = np.asarray(M)
        assert np.all(m >= 0) and np.all(m <= 1), m
    np.testing.assert_allclose(np.asarray(KS_PZ).sum(axis=1), 1.0, rtol=1e-6)
    print("PASS  test_conditional_chain_is_valid")


def test_aggregate_state_switches():
    """The common aggregate shock must actually move: with p_switch = 1/8 per
    step, both states should be visited many times in 1500 steps. (The old
    per-agent majority-vote construction froze the aggregate state forever.)"""
    env = _env()
    net, params = _net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(3), n_steps=N_STEPS)
    agg = np.asarray(rec["agg_state"])
    n_switch = int(np.sum(agg[1:] != agg[:-1]))
    frac_good = float(np.mean(agg))
    assert n_switch > 50, f"aggregate state nearly frozen: {n_switch} switches"
    assert 0.25 < frac_good < 0.75, f"agg state occupancy off: {frac_good:.2f}"
    print(f"PASS  test_aggregate_state_switches  (switches={n_switch}, good%={frac_good:.2f})")


def test_unemployment_rates_match_calibration():
    """Unemployment should track u_bad=0.10 / u_good=0.04 by aggregate state."""
    env = _env()
    net, params = _net_params(env)
    rec = simulate(env, net, params, jax.random.PRNGKey(4), n_steps=N_STEPS)
    emp = np.asarray(rec["emp_states"])          # [T, N]
    agg = np.asarray(rec["agg_state"])           # [T]
    u = 1.0 - emp.mean(axis=1)
    u_good = float(u[agg == 1].mean())
    u_bad = float(u[agg == 0].mean())
    assert abs(u_good - 0.04) < 0.02, f"u_good={u_good:.3f}"
    assert abs(u_bad - 0.10) < 0.03, f"u_bad={u_bad:.3f}"
    print(f"PASS  test_unemployment_rates_match_calibration  (u_g={u_good:.3f}, u_b={u_bad:.3f})")


if __name__ == "__main__":
    test_conditional_chain_is_valid()
    test_aggregate_state_switches()
    test_unemployment_rates_match_calibration()
    print("\nAll KS env tests passed.")
