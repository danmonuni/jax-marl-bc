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


def _init_carry(env, key):
    """(obs_mat, state, key) initial carry via the fastest available interface."""
    if hasattr(env, "reset_mat"):
        obs_mat0, state = env.reset_mat(key)
    else:
        obs0, state = env.reset(key)
        obs_mat0 = jnp.stack([obs0[a] for a in env.agents])
    return obs_mat0, state, key


def _make_step(env, net, params):
    """Build the recording step function. ``params`` may be a tracer, so the
    same compiled program serves every training snapshot."""
    use_vec = hasattr(env, "step_mat")
    alpha, delta = env.alpha, env.delta
    kappas, lambdas = env.kappas, env.lambdas

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
        record = {
            "ks": state.ks,
            "wealths": state.wealths,
            "c_fracs": c_frac,
            "cons": cons,
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

    return _step


def _simulate_jax(env, net, params, key, n_steps=2000):
    """Pure-JAX deterministic rollout returning JAX arrays (vmap-composable)."""
    step = _make_step(env, net, params)
    carry = _init_carry(env, key)
    _, rec = jax.lax.scan(step, carry, None, length=n_steps)
    return rec


_RUNNER_CACHE = {}


def _segment_runner(env, net):
    """Jitted (params, carry, length) -> (carry, rec) segment executor.

    Cached per (env, net): one compile serves every snapshot and segment (a
    second one for a remainder segment of different length).
    """
    cache_key = (id(env), id(net))
    if cache_key not in _RUNNER_CACHE:
        from functools import partial

        @partial(jax.jit, static_argnames=("length",))
        def run(params, carry, length):
            step = _make_step(env, net, params)
            return jax.lax.scan(step, carry, None, length=length)

        _RUNNER_CACHE[cache_key] = run
    return _RUNNER_CACHE[cache_key]


def simulate(env, net, params, key, n_steps=2000, max_chunk_bytes=2.56e8):
    """Deterministic (mean-action) rollout -> numpy dict.

    Runs in segments sized so the on-device record buffer stays under
    ``max_chunk_bytes`` (~12 float channels x n_agents per step), streaming
    each segment to host RAM. Device memory for diagnostics is therefore flat
    in ``n_steps`` — large-population evals cannot OOM the accelerator.
    """
    per_step_bytes = 4 * 12 * max(env.num_agents, 1)
    chunk = int(max(1, min(n_steps, max_chunk_bytes // per_step_bytes)))
    run = _segment_runner(env, net)
    carry = _init_carry(env, key)
    parts = []
    done = 0
    while done < n_steps:
        length = min(chunk, n_steps - done)
        carry, rec = run(params, carry, length=length)
        parts.append(jax.tree.map(np.asarray, rec))  # blocks; offloads to host
        done += length
    if len(parts) == 1:
        return parts[0]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def simulate_seeds(env, net, params, keys, n_steps=2000):
    """Run ``simulate`` over multiple seeds in parallel with vmap.

    keys: Array [n_seeds, 2]. Returns rec with a leading n_seeds dim everywhere.
    """
    rec = jax.vmap(lambda k: _simulate_jax(env, net, params, k, n_steps))(keys)
    return jax.tree.map(np.array, rec)
