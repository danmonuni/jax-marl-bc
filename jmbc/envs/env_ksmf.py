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

KS_A_GOOD = jnp.float32(1.02);  KS_A_BAD = jnp.float32(0.98)
KS_L_EMP  = jnp.float32(1.11);  KS_L_UNE = jnp.float32(0.0)


def build_ks_P():
    """
    4-state Markov transition matrix.
    States: 0=(Bad,Unemp)  1=(Bad,Emp)  2=(Good,Unemp)  3=(Good,Emp)

    Calibration (Krusell & Smith 1998):
      p_gg = p_bb = 7/8,  u_g=0.04,  u_b=0.10
      avg unemp spell: d_ub=2.5, d_ug=1.5

    Common bug — wrong p_eu formula:
      WRONG: p_eu = p_uu*(1-u)/u    → gives values > 1
      RIGHT: p_eu = u*(1-p_uu)/(1-u)  ← steady-state condition
    Stationary: (BU=0.05, BE=0.45, GU=0.02, GE=0.48)  ✓
    """
    p_gg=7/8; p_bb=7/8; p_bg=1/8; p_gb=1/8
    u_g=0.04; u_b=0.10
    p_uu_bb=1-1/2.5;  p_eu_bb=u_b*(1-p_uu_bb)/(1-u_b)   # ← correct formula
    p_uu_gg=1-1/1.5;  p_eu_gg=u_g*(1-p_uu_gg)/(1-u_g)
    p_uu_bg=u_g; p_eu_bg=u_g
    p_uu_gb=u_b; p_eu_gb=u_b
    P = jnp.array([
        [p_bb*p_uu_bb,    p_bb*(1-p_uu_bb), p_bg*p_uu_bg,    p_bg*(1-p_uu_bg)],
        [p_bb*p_eu_bb,    p_bb*(1-p_eu_bb), p_bg*p_eu_bg,    p_bg*(1-p_eu_bg)],
        [p_gb*p_uu_gb,    p_gb*(1-p_uu_gb), p_gg*p_uu_gg,    p_gg*(1-p_uu_gg)],
        [p_gb*p_eu_gb,    p_gb*(1-p_eu_gb), p_gg*p_eu_gg,    p_gg*(1-p_eu_gg)],
    ], jnp.float32)
    return P

KS_P = build_ks_P()


def ks_stationary_distribution(P=None):
    """Stationary distribution over (Bad/Good x Unemp/Emp) for verification.

    Returns dict with the four masses and the implied unemployment rates.
    Calibration target: u_bad=0.10, u_good=0.04.
    """
    P = KS_P if P is None else P
    v, e = np.linalg.eig(np.array(P).T)
    s = np.abs(e[:, np.argmax(v.real)])
    s /= s.sum()
    return {
        "BU": float(s[0]), "BE": float(s[1]),
        "GU": float(s[2]), "GE": float(s[3]),
        "u_bad": float(s[0] / (s[0] + s[1])),
        "u_good": float(s[2] / (s[2] + s[3])),
    }

@struct.dataclass
class RBCKSState(State):
    ks:         chex.Array        # [n]
    agg_state:  int               # 0=bad  1=good
    emp_states: chex.Array        # [n]  0=unemp  1=emp
    wealths:    chex.Array        # [n]
    incomes:    chex.Array        # [n]
    ls:         chex.Array        # [n] exogenous labour
    A:          chex.Array        # ()
    key:        chex.PRNGKey


class RBCKSEnv(MultiAgentEnv):
    """
    Krusell-Smith variant (exogenous labour, 1-D action).
    Same JaxMARL interface as RBCKLEnv.
    """
    def __init__(self, n_agents, kappas, lambdas,
                 alpha=0.36, delta=0.025, beta=0.95,
                 max_steps=500, k_init=1.0,
                 obs_vars=("capital","labour","mean_capital","aggregate_state")):
        super().__init__(n_agents)
        self.kappas    = jnp.asarray(kappas,  jnp.float32)
        self.lambdas   = jnp.asarray(lambdas, jnp.float32)
        self.alpha, self.delta, self.beta = alpha, delta, beta
        self.max_steps, self.k_init = max_steps, k_init
        self.obs_vars  = tuple(obs_vars)
        self.obs_dim   = len(obs_vars)
        self.act_dim   = 1
        self.agents    = [f"agent_{i}" for i in range(n_agents)]
        self.P         = KS_P
        self.observation_spaces = {a: Box(-jnp.inf, jnp.inf, (self.obs_dim,))
                                   for a in self.agents}
        self.action_spaces      = {a: Box(-0.99, 0.99, (self.act_dim,))
                                   for a in self.agents}

    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key):
        key, sk1, sk2 = jax.random.split(key, 3)
        ks  = jnp.full((self.num_agents,), self.k_init, jnp.float32)
        emp = (jax.random.uniform(sk2,(self.num_agents,)) > 0.04).astype(jnp.int32)
        ls  = jnp.where(emp==1, KS_L_EMP, KS_L_UNE)
        state = RBCKSState(done=jnp.bool_(False), step=0,
                           ks=ks, agg_state=1, emp_states=emp,
                           wealths=jnp.zeros(self.num_agents, jnp.float32),
                           incomes=jnp.zeros(self.num_agents, jnp.float32),
                           ls=ls, A=KS_A_GOOD, key=sk1)
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: RBCKSState) -> Dict[str, chex.Array]:
        mat = self._obs_matrix(state)
        return {a: mat[i] for i, a in enumerate(self.agents)}

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self, key, state: RBCKSState, actions: Dict[str, chex.Array]):
        acts    = jnp.stack([actions[a] for a in self.agents])
        c_fracs = jnp.clip((acts[:,0]+1)/2, 0.01, 0.99)
        K = jnp.maximum(jnp.mean(self.kappas*state.ks),  1e-8)
        L = jnp.maximum(jnp.mean(self.lambdas*state.ls), 1e-8)
        Y = state.A * K**self.alpha * L**(1-self.alpha)
        rs = self.alpha*(Y/K)*self.kappas
        ws = (1-self.alpha)*(Y/L)*self.lambdas
        w  = jnp.maximum(ws*state.ls + rs*state.ks + (1-self.delta)*state.ks, 1e-8)
        cs = c_fracs*w;  ks_new = jnp.maximum((1-c_fracs)*w, 1e-8)
        rews = jnp.clip(jnp.log(jnp.maximum(cs,1e-8)), -1e5, 1e10)

        key, sk = jax.random.split(state.key)
        idx   = 2*state.agg_state + state.emp_states          # ∈ {0,1,2,3}
        def _next(sk_i, s): return jax.random.choice(sk_i, 4, p=self.P[s])
        new_combined = vmap(_next)(jax.random.split(sk, self.num_agents), idx)
        new_agg = (jnp.mean(new_combined//2) > 0.5).astype(jnp.int32)
        new_emp = new_combined % 2
        new_ls  = jnp.where(new_emp==1, KS_L_EMP, KS_L_UNE)
        A_new   = jnp.where(new_agg==1, KS_A_GOOD, KS_A_BAD)

        step_new = state.step + 1
        done     = jnp.bool_(step_new >= self.max_steps)
        new_state = RBCKSState(done=done, step=step_new,
                               ks=ks_new, agg_state=new_agg, emp_states=new_emp,
                               wealths=w, incomes=ws*state.ls+rs*state.ks,
                               ls=new_ls, A=A_new, key=key)
        rews_d = {a: rews[i]  for i, a in enumerate(self.agents)}
        dones  = {a: done     for a in self.agents}
        dones["__all__"] = done
        return self.get_obs(new_state), new_state, rews_d, dones, {}

    def _obs_matrix(self, state):
        n, parts = self.num_agents, []
        for v in self.obs_vars:
            if   v == "capital":          parts.append(state.ks)
            elif v == "mean_capital":     parts.append(jnp.full((n,), jnp.mean(state.ks)))
            elif v == "labour":           parts.append(state.ls)
            elif v == "mean_labour":     parts.append(jnp.full((n,), jnp.mean(state.ls)))
            elif v == "kappa":            parts.append(self.kappas)
            elif v == "lambda":           parts.append(self.lambdas)
            elif v == "wealth":           parts.append(state.wealths)
            elif v == "income":           parts.append(state.incomes)
            elif v == "aggregate_state":  parts.append(jnp.full((n,), jnp.float32(state.agg_state)))
            else: raise ValueError(f"Unknown obs var: {v}")
        return jnp.stack(parts, axis=-1)