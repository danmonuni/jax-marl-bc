"""Array-interface episode logging: the vector counterpart of JaxMARL's
LogWrapper, for envs exposing ``reset_mat`` / ``step_mat``.

Tracks per-agent episode returns as one [n_agents] array instead of an
n_agents-entry dict, so graph size stays independent of n_agents.
"""
from __future__ import annotations

from typing import Any

import chex
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class VecLogState:
    env_state: Any
    episode_returns: chex.Array           # [n_agents]
    returned_episode_returns: chex.Array  # [n_agents]


class VecLogWrapper:
    """Same episode-return semantics as jaxmarl LogWrapper, on arrays."""

    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self, key):
        obs, env_state = self._env.reset_mat(key)
        n = self._env.num_agents
        return obs, VecLogState(env_state, jnp.zeros((n,)), jnp.zeros((n,)))

    def step(self, key, state: VecLogState, acts):
        obs, env_state, reward, done = self._env.step_mat(key, state.env_state, acts)
        new_return = state.episode_returns + reward
        d = done.astype(jnp.float32)
        state = VecLogState(
            env_state=env_state,
            episode_returns=new_return * (1.0 - d),
            returned_episode_returns=(
                state.returned_episode_returns * (1.0 - d) + new_return * d
            ),
        )
        info = {"returned_episode_returns": state.returned_episode_returns}
        return obs, state, reward, done, info
