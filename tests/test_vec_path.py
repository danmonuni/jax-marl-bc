"""The vector (array) fast path must be equivalent to the JaxMARL dict path."""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax
import jax.numpy as jnp

from jmbc.envs import RBCKSEnv
from jmbc.algos import make_train, ActorCritic
from jmbc.diagnostics import simulate

N_AGENTS = 4


class DictOnly:
    """Proxy hiding the vector interface, forcing the legacy dict path."""

    def __init__(self, env):
        self.__dict__["_e"] = env

    def __getattr__(self, name):
        if name in ("step_mat", "reset_mat"):
            raise AttributeError(name)
        return getattr(self._e, name)


def _env():
    return RBCKSEnv(
        n_agents=N_AGENTS,
        kappas=np.ones(N_AGENTS, np.float32),
        lambdas=np.ones(N_AGENTS, np.float32),
        max_steps=50, k_init=1.0,
        obs_vars=("capital", "labour", "mean_capital", "aggregate_state"),
    )


def _train_cfg():
    return {
        "NUM_ENVS": 2, "ROLLOUT_LEN": 20, "TOTAL_TIMESTEPS": 60,
        "UPDATE_EPOCHS": 2, "NUM_MINIBATCHES": 2, "LR": 3e-4,
        "GAMMA": 0.99, "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2,
        "VF_COEF": 0.5, "ENT_COEF": 0.0, "ANNEAL_LR": False,
        "MAX_GRAD_NORM": 0.5, "HIDDEN_DIMS": (16, 16), "ACTIVATION": "tanh",
        "LOG_EVERY": 0,
    }


def test_train_vec_equals_dict():
    """Same seed -> same trained parameters through either interface."""
    env = _env()
    out_vec = make_train(env, _train_cfg())(jax.random.PRNGKey(3))
    out_dict = make_train(DictOnly(env), _train_cfg())(jax.random.PRNGKey(3))
    for pv, pd in zip(jax.tree_util.tree_leaves(out_vec["params"]),
                      jax.tree_util.tree_leaves(out_dict["params"])):
        np.testing.assert_allclose(np.asarray(pv), np.asarray(pd),
                                   rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(out_vec["metrics"]["step_reward"]),
        np.asarray(out_dict["metrics"]["step_reward"]), rtol=1e-5, atol=1e-6)
    print("PASS  test_train_vec_equals_dict")


def test_rollout_vec_equals_dict():
    """Diagnostic rollouts identical through either interface."""
    env = _env()
    net = ActorCritic(action_dim=env.act_dim, hidden_dims=(16, 16))
    params = net.init(jax.random.PRNGKey(0), jnp.zeros((env.obs_dim,)))
    rec_v = simulate(env, net, params, jax.random.PRNGKey(11), n_steps=120)
    rec_d = simulate(DictOnly(env), net, params, jax.random.PRNGKey(11), n_steps=120)
    for k in rec_v:
        np.testing.assert_allclose(rec_v[k], rec_d[k], rtol=1e-6, atol=1e-7,
                                   err_msg=f"channel {k}")
    print("PASS  test_rollout_vec_equals_dict")


if __name__ == "__main__":
    test_train_vec_equals_dict()
    test_rollout_vec_equals_dict()
    print("\nAll vec-path tests passed.")
