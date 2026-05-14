import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal, glorot_uniform
import distrax
from typing import Sequence

class ActorCritic(nn.Module):
    action_dim: int
    hidden_dims: Sequence[int] = (64, 64)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            act_fn = nn.relu
        else:
            act_fn = nn.tanh

        trunk_x = x
        for dims in self.hidden_dims:
            #orthoglonal initialiser does not work on METAL
            trunk_x = act_fn(nn.Dense(dims, kernel_init=orthogonal(jnp.sqrt(2)))(trunk_x))
            #trunk_x = act_fn(nn.Dense(dims, kernel_init=glorot_uniform())(trunk_x))
        
        actor_mean = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01))(trunk_x)
        #actor_mean = nn.Dense(self.action_dim, kernel_init=glorot_uniform())(trunk_x)
        actor_logtstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        critic = nn.Dense(1, kernel_init=orthogonal(1.0))(trunk_x)
        #critic = nn.Dense(1, kernel_init=glorot_uniform())(trunk_x)
        return pi, jnp.squeeze(critic, axis=-1)
