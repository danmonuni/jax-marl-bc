"""Construct environments from an :class:`EnvConfig`.

Also hosts the heterogeneity-grid builders shared by the "general" driver and
sweeps (moved from the old ``exps/general.py``).
"""
from __future__ import annotations

from typing import List, Sequence

import numpy as np

from ..config.schema import EnvConfig
from . import init_capital
from .env_std import RBCKLEnv
from .env_ksmf import RBCKSEnv


def _resolve_weights(weights, n_agents: int) -> np.ndarray:
    """None -> ones(n_agents); a list -> validated float array."""
    if weights is None:
        return np.ones(n_agents, np.float32)
    arr = np.asarray(list(weights), np.float32)
    if arr.shape[0] != n_agents:
        raise ValueError(
            f"weights length {arr.shape[0]} != n_agents {n_agents}"
        )
    return arr


def build_env(cfg: EnvConfig):
    """Build an RBCKLEnv ("rbc") or RBCKSEnv ("ks") from config."""
    n = int(cfg.n_agents)
    kappas = _resolve_weights(cfg.kappas, n)
    lambdas = _resolve_weights(cfg.lambdas, n)
    init_capital.validate(cfg.k_init_dist, cfg.k_init_sigma,
                          cfg.k_init_low, cfg.k_init_high)
    k_spec = dict(k_init_dist=cfg.k_init_dist, k_init_sigma=cfg.k_init_sigma,
                  k_init_resample=cfg.k_init_resample,
                  k_init_low=cfg.k_init_low, k_init_high=cfg.k_init_high)

    if cfg.kind == "rbc":
        return RBCKLEnv(
            n_agents=n, kappas=kappas, lambdas=lambdas,
            alpha=cfg.alpha, delta=cfg.delta, beta=cfg.beta, b=cfg.b,
            rho=cfg.rho, sigma=cfg.sigma, max_steps=cfg.max_steps,
            k_init=cfg.k_init, obs_vars=tuple(cfg.obs_vars), **k_spec,
        )
    if cfg.kind == "ks":
        return RBCKSEnv(
            n_agents=n, kappas=kappas, lambdas=lambdas,
            alpha=cfg.alpha, delta=cfg.delta, beta=cfg.beta,
            max_steps=cfg.max_steps, k_init=cfg.k_init,
            obs_vars=tuple(cfg.obs_vars), **k_spec,
        )
    raise ValueError(f"Unknown env kind: {cfg.kind!r} (expected 'rbc' or 'ks')")


# ── Heterogeneity-grid helpers (from old exps/general.py) ─────────────────────

def kappa_spreads(n: int) -> List[List[float]]:
    """Three kappa configurations on an n-point grid: homogeneous/moderate/wide."""
    return [
        [1.0] * n,
        list(np.linspace(0.8, 1.2, n)),
        list(np.linspace(0.0, 2.0, n)),
    ]


def lambda_values(n: int) -> List[float]:
    """n lambda values spread symmetrically around 1.0."""
    return list(np.linspace(0.98, 1.02, n))


def make_grid_arrays(kappa_vals: Sequence[float], lambda_vals: Sequence[float]):
    """n x m grid: outer loop over kappa, inner over lambda -> per-agent arrays."""
    kappas = np.array([k for k in kappa_vals for _ in lambda_vals], np.float32)
    lambdas = np.array([l for _ in kappa_vals for l in lambda_vals], np.float32)
    return kappas, lambdas
