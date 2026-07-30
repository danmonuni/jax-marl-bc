"""Per-agent starting capital, shared by both environments.

Capital at t=0 is drawn once per parallel environment and then persisted in the
env state, so that in-training episode auto-resets restart the *same*
population rather than resampling one. Only ``reset_mat`` draws; ``step_mat``'s
auto-reset path reuses ``state.k_init_vec``.

Distributions (``EnvConfig.k_init_dist``):

``constant``
    Every agent starts at ``k_init``. The pre-randomization behavior, and the
    default: it consumes no PRNG key, so existing runs stay bit-reproducible.
``uniform``
    ``U(k_init_low, k_init_high)``, the form used in the reference KS
    implementation (``np.random.uniform(10, 70)``). The reference bounds span
    0.25x-1.75x of ITS steady state (K* ~ 40 at beta=0.99); the defaults here
    keep those proportions at this repo's beta=0.95, where K* ~ 11.7. Bounds
    are calibration-specific and must be rescaled if beta moves, not copied.
``lognormal``
    Mean-preserving spread of dispersion ``k_init_sigma`` around ``k_init``
    (mu = -sigma^2/2, so E[k_i] = k_init for every sigma), so sweeping sigma
    varies inequality without also moving aggregate K_0.
"""
import jax
import jax.numpy as jnp

DISTS = ("constant", "uniform", "lognormal")


def draw_k_init(key, n_agents: int, k_init: float, dist: str = "constant",
                sigma: float = 0.0, low: float = 0.0, high: float = 0.0):
    """[n] starting capital drawn from ``dist``."""
    if dist == "constant":
        return jnp.full((n_agents,), k_init, jnp.float32)
    if dist == "uniform":
        return jax.random.uniform(key, (n_agents,), jnp.float32,
                                  minval=low, maxval=high)
    if dist == "lognormal":
        draw = jax.random.lognormal(key, jnp.float32(sigma), (n_agents,), jnp.float32)
        return (draw * k_init * jnp.exp(-0.5 * sigma ** 2)).astype(jnp.float32)
    raise ValueError(f"Unknown k_init_dist: {dist!r} (expected one of {DISTS})")


def split_and_draw_k_init(key, n_agents: int, k_init: float,
                          dist: str = "constant", sigma: float = 0.0,
                          low: float = 0.0, high: float = 0.0):
    """(key for the rest of reset, k_init_vec [n]).

    ``dist`` is a static Python string, so the branch resolves at trace time.
    For ``constant`` the key is passed through UNCONSUMED, keeping the RNG
    stream byte-identical to the pre-randomization code so existing runs stay
    exactly reproducible.
    """
    if dist == "constant":
        return key, draw_k_init(key, n_agents, k_init, dist, sigma, low, high)
    key, sk = jax.random.split(key)
    return key, draw_k_init(sk, n_agents, k_init, dist, sigma, low, high)


def validate(dist: str, sigma: float, low: float, high: float) -> None:
    """Fail loudly at build time on a spec that would silently do nothing."""
    if dist not in DISTS:
        raise ValueError(f"Unknown k_init_dist: {dist!r} (expected one of {DISTS})")
    if dist == "uniform" and not 0 <= low < high:
        raise ValueError(
            f"k_init_dist='uniform' needs 0 <= k_init_low < k_init_high "
            f"(got low={low}, high={high})"
        )
    if dist == "lognormal" and sigma <= 0:
        raise ValueError(
            f"k_init_dist='lognormal' needs k_init_sigma > 0 (got {sigma}); "
            f"use k_init_dist='constant' for a degenerate population"
        )
