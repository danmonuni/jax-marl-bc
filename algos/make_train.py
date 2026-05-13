import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from typing import NamedTuple, Dict, Any
from functools import partial

from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jaxmarl.wrappers.baselines import LogWrapper
from algos.nn import ActorCritic

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
    x = jnp.stack([x[a] for a in agent_list], axis=1) # [num_envs, num_agents, ...]
    return x

def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_agents):
    """Split batched actions [num_envs, num_agents, AD] back into a dictionary per agent."""
    return {a: x[:, i] for i, a in enumerate(agent_list)}

def make_train(env: MultiAgentEnv, config: dict):
    """
    Returns a train function. Network init happens eagerly (outside JIT) to avoid the core training loop is JIT-compiled internally.
    """
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["ROLLOUT_LEN"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["ROLLOUT_LEN"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env)

    # Build network and dummy input outside of train so init runs eagerly (not inside JIT).
    act_dim = env.action_space(env.agents[0]).shape[0]
    network = ActorCritic(
        action_dim=act_dim,
        activation=config.get("ACTIVATION", "tanh"),
        hidden_dims=config.get("HIDDEN_DIMS", (64, 64))
    )
    init_x = jnp.zeros(env.observation_space(env.agents[0]).shape)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    @jax.jit
    def _train_core(rng, network_params):
        """JIT-compiled training loop. Receives pre-initialised params."""

        # 1. BUILD OPTIMIZER & TRAIN STATE
        if config.get("ANNEAL_LR", False):
            tx = optax.chain(
                optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
                optax.adam(config["LR"], eps=1e-5)
            )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # 2. INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset)(reset_rng)

        # 3. TRAINING LOOP
        def _update_step(runner_state, _):
            train_state, env_state, last_obs, rng = runner_state

            # 3.1 COLLECT TRAJECTORIES
            def _env_step(carry, _):
                train_state, env_state, last_obs, rng = carry

                # {agent: [NE, OD]} -> [NE, N, OD]
                obs_batch = batchify(last_obs, env.agents, config["NUM_ENVS"], env.num_agents)
                # [NE, N, OD] -> [NE*N, OD]
                obs_flat = obs_batch.reshape((config["NUM_ACTORS"], -1))

                # Select actions
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, obs_flat)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # [NE*N, AD] -> [NE, N, AD] -> {agent: [NE, AD]}
                action_reshaped = action.reshape((config["NUM_ENVS"], env.num_agents, -1))
                env_act = unbatchify(action_reshaped, env.agents, config["NUM_ENVS"], env.num_agents)

                # Step environment
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    rng_step, env_state, env_act,
                )

                transition = Transition(
                    done=batchify(done, env.agents, config["NUM_ENVS"], env.num_agents),
                    action=action.reshape((config["NUM_ENVS"], env.num_agents, -1)),
                    value=value.reshape((config["NUM_ENVS"], env.num_agents)),
                    reward=batchify(reward, env.agents, config["NUM_ENVS"], env.num_agents),
                    log_prob=log_prob.reshape((config["NUM_ENVS"], env.num_agents)),
                    obs=obs_batch,
                    info=info,
                )
                return (train_state, env_state, obsv, rng), transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, (train_state, env_state, last_obs, rng), None, config["ROLLOUT_LEN"]
            )

            # 3.2 CALCULATE ADVANTAGE (GAE)
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = batchify(last_obs, env.agents, config["NUM_ENVS"], env.num_agents)
            last_obs_flat = last_obs_batch.reshape((config["NUM_ACTORS"], -1))
            _, last_val = network.apply(train_state.params, last_obs_flat)
            last_val_reshaped = last_val.reshape((config["NUM_ENVS"], env.num_agents))

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

            # 3.3 UPDATE NETWORK (EPOCHS & MINIBATCHES)
            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Value loss (clipped)
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Actor loss (clipped)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                # Flatten [T, NE, N, ...] -> [T*NE*N, ...]
                batch_size = config["ROLLOUT_LEN"] * config["NUM_ACTORS"]
                permutation = jax.random.permutation(_rng, batch_size)

                batch = (traj_batch, advantages, targets)
                flat_batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[3:]), batch)

                shuffled_batch = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), flat_batch)

                minibatches = jax.tree.map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch
                )

                train_state, losses = jax.lax.scan(_update_minibatch, train_state, minibatches)
                return (train_state, traj_batch, advantages, targets, rng), losses

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            rng = update_state[-1]

            metrics = {
                "loss": jax.tree.map(lambda x: x.mean(), loss_info),
                "step_reward": traj_batch.reward.mean(),
                "returned_episode_returns": traj_batch.info["returned_episode_returns"].mean(),
                # mean over T and agents, keep NE dim for std-band computation
                "c_frac_env": ((traj_batch.action[..., 0] + 1) / 2).mean(axis=(0, 2)),
                "l_env": (
                    ((traj_batch.action[..., 1] + 1) / 2).mean(axis=(0, 2))
                    if act_dim > 1
                    else jnp.zeros(config["NUM_ENVS"])
                ),
            }

            return (train_state, env_state, last_obs, rng), (metrics, train_state.params)

        # 4. RUN SCAN OVER UPDATES
        final_state, (metrics, params_history) = jax.lax.scan(
            _update_step, (train_state, env_state, obsv, rng), None, config["NUM_UPDATES"]
        )

        return {"params": final_state[0].params, "metrics": metrics, "params_history": params_history}

    def train(rng):
        """Eagerly initialise network params then dispatch to JIT-compiled core."""
        rng, init_rng = jax.random.split(rng)
        network_params = network.init(init_rng, init_x)
        return _train_core(rng, network_params)

    return train
