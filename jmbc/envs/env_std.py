import jax, jax.numpy as jnp, numpy as np
from jax import lax, vmap
from flax import struct
import chex
from typing import Dict
from functools import partial

# JaxMARL imports
from jaxmarl.environments.multi_agent_env import MultiAgentEnv, State
from jaxmarl.environments.spaces import Box
from jaxmarl.wrappers.baselines import LogWrapper

@struct.dataclass
class RBCKLState(State):          # inherits: done: Array, step: int
    ks:      chex.Array           # [n] capitals
    z:       chex.Array           # () log-tech AR(1)
    A:       chex.Array           # () = exp(z)
    wealths: chex.Array           # [n]
    incomes: chex.Array           # [n]
    ls:      chex.Array           # [n] labour chosen last step
    key:     chex.PRNGKey

class RBCKLEnv(MultiAgentEnv):
    """
    RBC with endogenous labour — proper JaxMARL subclass.

    Actions : Dict[agent -> Array([2])] ∈ [-0.99, 0.99]
    Rescaled: (x+1)/2 → frac ∈ [0.005, 0.995]

    Notes:
    - CTRolloutManager is discrete-only; use vmap(env.step) for batching.
    - Decorate step_env / get_obs / reset with @partial(jax.jit, static_argnums=(0,)).
    """

    def __init__(self, n_agents, kappas, lambdas,
                 alpha=0.36, delta=0.025, beta=0.95, b=5.0,
                 rho=0.9, sigma=0.01, max_steps=1000, k_init=0.1,
                 obs_vars=("wealth",)):
        super().__init__(n_agents)
        self.kappas    = jnp.asarray(kappas,  jnp.float32)
        self.lambdas   = jnp.asarray(lambdas, jnp.float32)
        self.alpha, self.delta, self.beta = alpha, delta, beta
        self.b, self.rho, self.sigma = b, rho, sigma
        self.max_steps, self.k_init  = max_steps, k_init
        self.obs_vars  = tuple(obs_vars)
        self.obs_dim   = len(obs_vars)
        self.act_dim   = 2
        self.agents    = [f"agent_{i}" for i in range(n_agents)]
        # Required by MultiAgentEnv and LogWrapper
        self.observation_spaces = {a: Box(-jnp.inf, jnp.inf, (self.obs_dim,))
                                   for a in self.agents}
        self.action_spaces      = {a: Box(-0.99, 0.99, (self.act_dim,))
                                   for a in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        key, sk = jax.random.split(key)
        ks  = jnp.full((self.num_agents,), self.k_init, jnp.float32)
        z   = jnp.zeros((), jnp.float32)
        ls  = jnp.full((self.num_agents,), 0.33, jnp.float32)
        w   = ks * (1.0 - self.delta + self.alpha)
        state = RBCKLState(done=jnp.bool_(False), step=0,
                           ks=ks, z=z, A=jnp.exp(z), wealths=w,
                           incomes=jnp.zeros(self.num_agents, jnp.float32),
                           ls=ls, key=sk)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: RBCKLState) -> Dict[str, chex.Array]:
        """Returns Dict[agent -> obs_vector]. Called on auto-reset by base step()."""
        mat = self._obs_matrix(state)
        return {a: mat[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state: RBCKLState,
                 actions: Dict[str, chex.Array]):
        """Economic transition. Base class step() wraps this with auto-reset."""
        acts    = jnp.stack([actions[a] for a in self.agents])  # [n, 2]
        c_fracs = jnp.clip((acts[:,0]+1)/2, 0.01, 0.99)
        ls      = jnp.clip((acts[:,1]+1)/2, 0.01, 0.99)

        K = jnp.maximum(jnp.mean(self.kappas*state.ks), 1e-8)   # Eq.(1)
        L = jnp.maximum(jnp.mean(self.lambdas*ls),      1e-8)
        Y = state.A * K**self.alpha * L**(1-self.alpha)           # Eq.(2)
        rs = self.alpha*(Y/K)*self.kappas                         # Eq.(3)
        ws = (1-self.alpha)*(Y/L)*self.lambdas
        w  = jnp.maximum(ws*ls + rs*state.ks + (1-self.delta)*state.ks, 1e-8)  # Eq.(4)
        cs = c_fracs*w;  ks_new = jnp.maximum((1-c_fracs)*w, 1e-8)
        rews = jnp.clip(jnp.log(jnp.maximum(cs,1e-8)) + self.b*jnp.log(jnp.maximum(1-ls,1e-8)),-1e5, 1e10)

        key, sk = jax.random.split(state.key)
        z_new   = self.rho*state.z + self.sigma*jax.random.normal(sk)
        step_new = state.step + 1
        done     = jnp.bool_(step_new >= self.max_steps)

        new_state = RBCKLState(done=done, step=step_new,
                               ks=ks_new, z=z_new, A=jnp.exp(z_new),
                               wealths=w, incomes=ws*ls+rs*state.ks, ls=ls, key=key)
        rews_d = {a: rews[i]  for i, a in enumerate(self.agents)}
        dones  = {a: done     for a in self.agents}
        dones["__all__"] = done
        return self.get_obs(new_state), new_state, rews_d, dones, {}

    def _obs_matrix(self, state):
        n, parts = self.num_agents, []
        for v in self.obs_vars:
            if   v == "capital":      parts.append(state.ks)
            elif v == "mean_capital": parts.append(jnp.full((n,), jnp.mean(state.ks)))
            elif v == "wealth":       parts.append(state.wealths)
            elif v == "income":       parts.append(state.incomes)
            elif v == "labour":       parts.append(state.ls)
            elif v == "mean_labour": parts.append(jnp.full((n,), jnp.mean(state.ls)))
            elif v == "TFP":          parts.append(jnp.full((n,), state.A))
            elif v == "kappa":        parts.append(self.kappas)
            elif v == "lambda":       parts.append(self.lambdas)
            else: raise ValueError(f"Unknown obs var: {v}")
        return jnp.stack(parts, axis=-1)