"""PPO training loop (JaxMARL-compatible), with rich in-loop diagnostics.

Training dynamics are unchanged from the original implementation: the extra
metrics (approx-KL, clip fraction, explained variance, grad norm, action
saturation, per-component losses) are read-only quantities computed from values
already present in the loop, so a fixed seed reproduces the original policy.

``make_train`` copies the incoming config so derived keys (NUM_ACTORS, ...) do
not leak back into the caller — important for sweeps that reuse a base config.
"""
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple, Dict, Any

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.wrappers.baselines import LogWrapper
from ..envs.vec import VecLogWrapper
from .nn import ActorCritic


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Dict[str, Any]


def batchify(x: dict, agent_list, num_envs, num_agents):
    """Stack agent observations/rewards/dones into [num_envs, num_agents, ...]."""
    x = jnp.stack([x[a] for a in agent_list], axis=1)  # [num_envs, num_agents, ...]
    return x


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    """Split batched actions [num_envs, num_agents, AD] back into a per-agent dict."""
    return {a: x[:, i] for i, a in enumerate(agent_list)}


def make_train(env: MultiAgentEnv, config: dict):
    """Return a train function. Network init happens eagerly (outside JIT)."""
    config = dict(config)  # defensive copy: do not mutate the caller's dict
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    # TOTAL_TIMESTEPS counts *sequential* env steps (one step = all NUM_ENVS
    # environments advancing once in parallel), so the training length is
    # independent of how many envs run in parallel.
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LEN"]
    # NUM_MINIBATCHES need NOT divide the flat batch. The remainder
    # (< NUM_MINIBATCHES samples) is dropped from the tail of a *fresh random
    # permutation* every epoch, so no sample is systematically excluded.
    # NUM_MINIBATCHES is kept exact rather than snapped to a divisor: every
    # sweep cell then takes the same number of gradient steps per update, which
    # is what makes cells comparable when n_agents (and so the batch) varies.
    config["BATCH_SIZE"] = config["NUM_ACTORS"] * config["ROLLOUT_LEN"]
    config["MINIBATCH_SIZE"] = config["BATCH_SIZE"] // config["NUM_MINIBATCHES"]
    if config["MINIBATCH_SIZE"] < 1:
        raise ValueError(
            f"num_minibatches={config['NUM_MINIBATCHES']} exceeds the batch per "
            f"update ({config['BATCH_SIZE']} = rollout_len {config['ROLLOUT_LEN']}"
            f" x num_envs {config['NUM_ENVS']} x n_agents {env.num_agents}): "
            f"each minibatch would be empty. Lower num_minibatches or raise "
            f"num_envs / rollout_len / n_agents."
        )
    config["USED_BATCH_SIZE"] = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
    config["DROPPED_PER_EPOCH"] = config["BATCH_SIZE"] - config["USED_BATCH_SIZE"]

    # Vector fast path: when the env exposes the array interface, skip the
    # per-agent dict round-trip entirely (trace size / runtime independent of
    # n_agents). Dynamics and RNG stream are identical to the dict path.
    use_vec = hasattr(env, "step_mat")
    config["VEC_INTERFACE"] = use_vec
    n_agents = env.num_agents
    env = VecLogWrapper(env) if use_vec else LogWrapper(env)

    act_dim = env.action_space(env.agents[0]).shape[0]
    network = ActorCritic(
        action_dim=act_dim,
        activation=config.get("ACTIVATION", "tanh"),
        hidden_dims=config.get("HIDDEN_DIMS", (64, 64)),
    )
    init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    @jax.jit
    def _train_core(rng, network_params):
        # 1. OPTIMIZER & TRAIN STATE
        if config.get("ANNEAL_LR", False):
            tx = optax.chain(
                optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
                optax.adam(config["LR"], eps=1e-5),
            )

        train_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx,
        )

        # 2. INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # 3. TRAINING LOOP
        def _update_step(runner_state, update_i):
            train_state, env_state, last_obs, rng = runner_state

            # 3.1 COLLECT TRAJECTORIES
            def _env_step(carry, _):
                train_state, env_state, last_obs, rng = carry

                if use_vec:
                    obs_batch = last_obs                      # [E, n, d] already
                else:
                    obs_batch = batchify(last_obs, env.agents, config["NUM_ENVS"], n_agents)
                obs_flat = obs_batch.reshape((config["NUM_ACTORS"], -1))

                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs_flat)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                action_reshaped = action.reshape((config["NUM_ENVS"], n_agents, -1))

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                if use_vec:
                    obsv, env_state, reward_b, done_env, info = jax.vmap(env.step)(
                        rng_step, env_state, action_reshaped,
                    )
                    done_b = jnp.broadcast_to(
                        done_env[:, None], (config["NUM_ENVS"], n_agents)
                    )
                else:
                    env_act = unbatchify(action_reshaped, env.agents, config["NUM_ENVS"], n_agents)
                    obsv, env_state, reward, done, info = jax.vmap(env.step)(
                        rng_step, env_state, env_act,
                    )
                    reward_b = batchify(reward, env.agents, config["NUM_ENVS"], n_agents)
                    done_b = batchify(done, env.agents, config["NUM_ENVS"], n_agents)

                transition = Transition(
                    done=done_b,
                    action=action_reshaped,
                    value=value.reshape((config["NUM_ENVS"], n_agents)),
                    reward=reward_b,
                    log_prob=log_prob.reshape((config["NUM_ENVS"], n_agents)),
                    obs=obs_batch,
                    info=info,
                )
                return (train_state, env_state, obsv, rng), transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["ROLLOUT_LEN"]
            )

            # 3.2 ADVANTAGE (GAE)
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = (last_obs if use_vec else
                              batchify(last_obs, env.agents, config["NUM_ENVS"], n_agents))
            last_obs_flat = last_obs_batch.reshape((config["NUM_ACTORS"], -1))
            _, last_val = network.apply(train_state.params, last_obs_flat)
            last_val_reshaped = last_val.reshape((config["NUM_ENVS"], n_agents))

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val_reshaped)

            # Explained variance of the value function (computed once per update).
            y_true = targets.reshape(-1)
            y_pred = traj_batch.value.reshape(-1)
            var_y = jnp.var(y_true)
            explained_var = 1.0 - jnp.var(y_true - y_pred) / (var_y + 1e-8)

            # 3.3 UPDATE NETWORK (EPOCHS & MINIBATCHES)
            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()

                        # Read-only PPO health diagnostics.
                        approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
                        clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > config["CLIP_EPS"]).astype(jnp.float32))

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        aux = (value_loss, loss_actor, entropy, approx_kl, clip_frac)
                        return total_loss, aux

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    grad_norm = optax.global_norm(grads)
                    train_state = train_state.apply_gradients(grads=grads)
                    value_loss, loss_actor, entropy, approx_kl, clip_frac = aux
                    mb_metrics = {
                        "total_loss": total_loss,
                        "value_loss": value_loss,
                        "policy_loss": loss_actor,
                        "entropy": entropy,
                        "approx_kl": approx_kl,
                        "clip_frac": clip_frac,
                        "grad_norm": grad_norm,
                    }
                    return train_state, mb_metrics

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                batch_size = config["BATCH_SIZE"]
                # Slicing the permutation (not the batch) drops the remainder
                # from a freshly shuffled order each epoch: the excluded samples
                # differ every time, so the truncation is unbiased.
                permutation = jax.random.permutation(_rng, batch_size)[
                    : config["USED_BATCH_SIZE"]
                ]

                batch = (traj_batch, advantages, targets)
                flat_batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[3:]), batch)
                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), flat_batch)
                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(
                        x,
                        [config["NUM_MINIBATCHES"], config["MINIBATCH_SIZE"]]
                        + list(x.shape[1:]),
                    ),
                    shuffled_batch,
                )

                train_state, mb_metrics = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), mb_metrics

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            rng = update_state[-1]

            # Mean PPO health metrics over (epochs, minibatches).
            ppo = jax.tree.map(lambda x: x.mean(), loss_info)

            # Action saturation: fraction of actions pinned near the [-1, 1] bounds.
            action_saturation = (jnp.abs(traj_batch.action) > 0.98).mean()

            metrics = {
                # losses / RL health
                "total_loss": ppo["total_loss"],
                "value_loss": ppo["value_loss"],
                "policy_loss": ppo["policy_loss"],
                "entropy": ppo["entropy"],
                "approx_kl": ppo["approx_kl"],
                "clip_frac": ppo["clip_frac"],
                "grad_norm": ppo["grad_norm"],
                "explained_variance": explained_var,
                "action_saturation": action_saturation,
                # rollout summaries
                "step_reward": traj_batch.reward.mean(),
                "returned_episode_returns": traj_batch.info["returned_episode_returns"].mean(),
                # policy means (keep NE dim for std-band plotting)
                "c_frac_env": ((traj_batch.action[..., 0] + 1) / 2).mean(axis=(0, 2)),
                "l_env": (
                    ((traj_batch.action[..., 1] + 1) / 2).mean(axis=(0, 2))
                    if act_dim > 1
                    else jnp.zeros(config["NUM_ENVS"])
                ),
            }

            # Progress heartbeat (host callback; read-only, no effect on the
            # training dynamics or the RNG stream).
            log_every = int(config.get("LOG_EVERY", 0) or 0)
            if log_every > 0:
                def _log():
                    jax.debug.print(
                        "[train] update {i}/{n}  reward={r:.4f}  approx_kl={kl:.5f}  "
                        "c_frac={c:.3f}  entropy={e:.3f}",
                        i=update_i + 1, n=config["NUM_UPDATES"],
                        r=metrics["step_reward"], kl=metrics["approx_kl"],
                        c=metrics["c_frac_env"].mean(), e=metrics["entropy"],
                    )
                jax.lax.cond((update_i + 1) % log_every == 0, _log, lambda: None)

            return (train_state, env_state, last_obs, rng), (metrics, train_state.params)

        final_state, (metrics, params_history) = jax.lax.scan(
            _update_step, (train_state, env_state, obsv, rng),
            jnp.arange(config["NUM_UPDATES"]),
        )

        return {
            "params": final_state[0].params,
            "metrics": metrics,
            "params_history": params_history,
        }

    def _prep(rng):
        rng, init_rng = jax.random.split(rng)
        return rng, network.init(init_rng, init_x)

    def train(rng):
        rng, network_params = _prep(rng)
        return _train_core(rng, network_params)

    def lower(rng):
        """AOT: trace/lower now, compile explicitly, run later.

        Same rng split as ``train`` -> identical results; lets the caller time
        and announce the trace / XLA-compile / run phases separately from the
        host, with no change to the compiled program itself.
        """
        rng, network_params = _prep(rng)
        lowered = _train_core.lower(rng, network_params)

        def compile_():
            compiled = lowered.compile()
            return lambda: compiled(rng, network_params)

        return compile_

    # Expose derived config for callers (e.g. recorder, plotting axes).
    train.config = config
    train.network = network
    train.lower = lower
    return train
