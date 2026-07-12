"""Deterministic (mean-action) rollouts for diagnostics.

Records, in addition to the raw state, the quantities needed by the economic
probes: per-agent consumption ``cons``, per-agent gross capital return ``R``
(1 - delta + marginal product), aggregate output ``Y`` and incoming mean
capital ``k_in_mean``. Returns are computed from the *incoming* state and the
labour used this period so that the Euler timing lines up:

    1/c_t = beta * E_t[ (1/c_{t+1}) * R_{t+1} ].
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np


def _simulate_jax(env, net, params, key, n_steps=2000):
    """Pure-JAX deterministic rollout returning JAX arrays (vmap-composable).

    Uses the array interface (reset_mat/step_mat) when the env provides it, so
    trace size stays independent of n_agents; falls back to the dict API.
    """
    use_vec = hasattr(env, "step_mat")
    if use_vec:
        obs_mat0, state = env.reset_mat(key)
    else:
        obs0, state = env.reset(key)
        obs_mat0 = jnp.stack([obs0[a] for a in env.agents])
    alpha, delta = env.alpha, env.delta
    kappas, lambdas = env.kappas, env.lambdas
    # Column of the capital observation, for the finite-difference MPC probe.
    cap_col = env.obs_vars.index("capital") if "capital" in env.obs_vars else None

    def _step(carry, _):
        obs_mat, state, key = carry
        pi, _ = net.apply(params, obs_mat)
        mu = pi.loc  # [n_agents, act_dim]

        # Match the env's action rescaling + clipping so the recorded
        # consumption/labour are consistent with the realized dynamics.
        c_frac = jnp.clip((mu[:, 0] + 1) / 2, 0.01, 0.99)
        labour_t = (
            jnp.clip((mu[:, 1] + 1) / 2, 0.01, 0.99) if mu.shape[1] > 1 else state.ls
        )

        # Gross return from the INCOMING state (capital k_t, TFP A_t) and the
        # labour used this period -> R_t = 1 - delta + alpha * Y_t / K_t * kappa.
        Kc = jnp.maximum(jnp.mean(kappas * state.ks), 1e-8)
        Ll = jnp.maximum(jnp.mean(lambdas * labour_t), 1e-8)
        Y = state.A * Kc ** alpha * Ll ** (1 - alpha)
        r_i = alpha * (Y / Kc) * kappas
        R_i = 1.0 - delta + r_i
        k_in_mean = jnp.mean(state.ks)

        # Policy-based MPC: bump this period's capital by eps (prices held
        # fixed, mean-field), re-evaluate the policy, and difference realized
        # consumption: mpc_i = [c(k+eps) - c(k)] / [dw], dw = R_i * eps.
        if cap_col is not None:
            eps = 0.05 * jnp.maximum(state.ks, 0.1)
            obs_pert = obs_mat.at[:, cap_col].add(eps)
            pi_p, _ = net.apply(params, obs_pert)
            c_frac_p = jnp.clip((pi_p.loc[:, 0] + 1) / 2, 0.01, 0.99)
            dw = R_i * eps

        key, sk = jax.random.split(key)
        if use_vec:
            obs_mat_n, state, rew_arr, done = env.step_mat(sk, state, mu)
        else:
            acts = {a: mu[i] for i, a in enumerate(env.agents)}
            obs_d, state, rews, dones, _ = env.step(sk, state, acts)
            obs_mat_n = jnp.stack([obs_d[a] for a in env.agents])
            rew_arr = jnp.stack([rews[a] for a in env.agents])
            done = dones["__all__"]

        cons = c_frac * state.wealths  # c_t = c_frac * a_t
        if cap_col is not None:
            mpc = (c_frac_p * (state.wealths + dw) - cons) / dw
        else:
            mpc = jnp.full((env.num_agents,), jnp.nan)
        record = {
            "ks": state.ks,
            "wealths": state.wealths,
            "c_fracs": c_frac,
            "cons": cons,
            "mpc": mpc,
            "ls": labour_t,
            "R": R_i,
            "Y": Y,
            "k_in_mean": k_in_mean,
            "K": jnp.mean(state.ks),
            "reward": jnp.mean(rew_arr),
            # True on an auto-reset step: the recorded state is a fresh reset,
            # so economic accounting probes must skip it.
            "done": done,
            "emp_states": getattr(state, "emp_states", jnp.ones(env.num_agents, jnp.int32)),
            "agg_state": getattr(state, "agg_state", jnp.array(1, jnp.int32)),
        }
        return (obs_mat_n, state, key), record

    _, rec = jax.lax.scan(_step, (obs_mat0, state, key), None, length=n_steps)
    return rec


def simulate(env, net, params, key, n_steps=2000):
    """Deterministic (mean-action) rollout compiled with lax.scan -> numpy dict."""
    return jax.tree.map(np.array, _simulate_jax(env, net, params, key, n_steps))


def simulate_seeds(env, net, params, keys, n_steps=2000):
    """Run ``simulate`` over multiple seeds in parallel with vmap.

    keys: Array [n_seeds, 2]. Returns rec with a leading n_seeds dim everywhere.
    """
    rec = jax.vmap(lambda k: _simulate_jax(env, net, params, k, n_steps))(keys)
    return jax.tree.map(np.array, rec)
