"""num_minibatches need not divide the batch per update.

The batch is rollout_len x num_envs x n_agents, so in a sweep over n_agents a
fixed num_minibatches hits indivisible cells. The remainder is dropped from the
tail of a fresh permutation each epoch; num_minibatches itself stays exact so
every sweep cell takes the same number of gradient steps.
"""
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import jax

from jmbc.envs import RBCKSEnv
from jmbc.algos import make_train

ROLLOUT_LEN = 20


def _env(n_agents):
    return RBCKSEnv(
        n_agents=n_agents,
        kappas=np.ones(n_agents, np.float32),
        lambdas=np.ones(n_agents, np.float32),
        max_steps=50, k_init=1.0,
        obs_vars=("capital", "labour", "mean_capital", "aggregate_state"),
    )


def _cfg(**over):
    cfg = {
        "NUM_ENVS": 1, "ROLLOUT_LEN": ROLLOUT_LEN, "TOTAL_TIMESTEPS": 40,
        "UPDATE_EPOCHS": 2, "NUM_MINIBATCHES": 7, "LR": 3e-4,
        "GAMMA": 0.99, "GAE_LAMBDA": 0.95, "CLIP_EPS": 0.2,
        "VF_COEF": 0.5, "ENT_COEF": 0.0, "ANNEAL_LR": False,
        "MAX_GRAD_NORM": 0.5, "HIDDEN_DIMS": (16, 16), "ACTIVATION": "tanh",
        "LOG_EVERY": 0,
    }
    cfg.update(over)
    return cfg


def test_indivisible_batch_trains():
    """Indivisible splits run, and drop fewer than num_minibatches samples."""
    for n_agents, nmb in [(2, 7), (3, 7), (5, 16)]:
        batch = ROLLOUT_LEN * n_agents
        assert batch % nmb != 0, f"case ({n_agents}, {nmb}) must be indivisible"

        train = make_train(_env(n_agents), _cfg(NUM_MINIBATCHES=nmb))
        c = train.config
        assert c["BATCH_SIZE"] == batch
        assert c["MINIBATCH_SIZE"] == batch // nmb
        assert c["USED_BATCH_SIZE"] == c["MINIBATCH_SIZE"] * nmb
        assert 0 < c["DROPPED_PER_EPOCH"] < nmb, c["DROPPED_PER_EPOCH"]

        out = train(jax.random.PRNGKey(0))
        loss = np.asarray(out["metrics"]["total_loss"])
        assert np.all(np.isfinite(loss)), loss
    print("PASS  test_indivisible_batch_trains")


def test_divisible_batch_uses_whole_batch():
    """Backward compatibility: an exactly-divisible split drops nothing, so the
    permutation slice is a no-op and MINIBATCH_SIZE is what ``-1`` used to infer.
    """
    for n_agents, nmb in [(4, 8), (4, 2), (10, 20), (1, 5)]:
        batch = ROLLOUT_LEN * n_agents
        assert batch % nmb == 0, f"case ({n_agents}, {nmb}) must be divisible"

        c = make_train(_env(n_agents), _cfg(NUM_MINIBATCHES=nmb)).config
        assert c["DROPPED_PER_EPOCH"] == 0
        assert c["USED_BATCH_SIZE"] == c["BATCH_SIZE"] == batch
        assert c["MINIBATCH_SIZE"] == batch // nmb
    print("PASS  test_divisible_batch_uses_whole_batch")


def test_too_many_minibatches_raises():
    """More minibatches than samples is a config error, not an empty split."""
    try:
        make_train(_env(2), _cfg(NUM_MINIBATCHES=41))  # batch = 40
    except ValueError as e:
        assert "exceeds the batch per update" in str(e), str(e)
    else:
        raise AssertionError("num_minibatches > batch_size did not raise")
    print("PASS  test_too_many_minibatches_raises")


if __name__ == "__main__":
    test_indivisible_batch_trains()
    test_divisible_batch_uses_whole_batch()
    test_too_many_minibatches_raises()
    print("\nAll minibatch-split tests passed.")
