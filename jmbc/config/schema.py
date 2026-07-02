"""Typed configuration schema — the single source of truth.

Dataclasses are used as an OmegaConf *structured* schema: YAML files in
``configs/`` and CLI dotlist overrides are merged onto these defaults and
validated against the declared types. ``to_train_dict`` adapts the training
hyperparameters to the UPPERCASE dict that :func:`jmbc.algos.make_train`
consumes, keeping a single authoritative definition.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EnvConfig:
    """Economic environment parameters (one environment instance)."""

    kind: str = "rbc"                       # "rbc" | "ks"
    n_agents: int = 1
    alpha: float = 0.36
    beta: float = 0.95
    delta: float = 0.025
    b: float = 5.0                          # labour-disutility weight (RBC only)
    rho: float = 0.9                        # TFP AR(1) persistence (RBC only)
    sigma: float = 0.01                     # TFP AR(1) innovation std (RBC only)
    max_steps: int = 1000
    k_init: float = 0.1
    obs_vars: List[str] = field(default_factory=lambda: ["capital"])
    # Per-agent heterogeneity. None -> filled with ones of length n_agents.
    kappas: Optional[List[float]] = None    # capital productivity weights
    lambdas: Optional[List[float]] = None   # labour productivity weights
    # Used only by the "general" heterogeneous-grid driver.
    n_grid: int = 3


@dataclass
class TrainConfig:
    """PPO training hyperparameters."""

    num_envs: int = 20
    rollout_len: int = 200
    # Sequential env steps (each steps all num_envs in parallel) — training
    # length is independent of num_envs. Total collected transitions =
    # total_timesteps * num_envs.
    total_timesteps: int = 50_000
    update_epochs: int = 10
    num_minibatches: int = 10
    lr: float = 3e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01
    anneal_lr: bool = False
    max_grad_norm: float = 0.5


@dataclass
class NetConfig:
    """Actor-critic network architecture."""

    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "tanh"                # "tanh" | "relu"


@dataclass
class DiagConfig:
    """Diagnostics toggles and sampling."""

    n_snapshots: int = 4                    # training snapshots for diagnostics
    sim_steps: int = 2000                   # rollout length used for diagnostics
    economic: bool = True                   # Euler errors, resource residual, etc.
    distributional: bool = True             # Gini, Lorenz, top shares, MPC
    burn_frac: float = 0.5                  # fraction of rollout discarded as burn-in


@dataclass
class LogConfig:
    """Where structured results are written."""

    out_dir: str = "results"
    run_name: Optional[str] = None          # None -> auto timestamped id


@dataclass
class RunConfig:
    """Execution controls."""

    seed: int = 0
    device: str = "auto"                    # "auto" | "cpu" | "gpu"


@dataclass
class ExperimentConfig:
    """A complete experiment specification."""

    exp: str = "rbc"
    env: EnvConfig = field(default_factory=EnvConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    net: NetConfig = field(default_factory=NetConfig)
    diag: DiagConfig = field(default_factory=DiagConfig)
    log: LogConfig = field(default_factory=LogConfig)
    run: RunConfig = field(default_factory=RunConfig)


@dataclass
class SweepConfig:
    """A meta-experiment: a base experiment plus axes to sweep over.

    ``axes`` maps dotted config paths (e.g. ``env.n_agents``) to lists of
    values. The runner evaluates the Cartesian product. ``overrides`` are
    applied to every cell before the axis values.
    """

    base_exp: str = "rbc"
    axes: Dict[str, List[Any]] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    repeats: int = 1
    name: str = "sweep"
    collect_diagnostics: bool = True   # also tabulate Euler/Gini per cell
    save_cell_runs: bool = False       # full per-cell output dir + figures
                                       # (distributional/economic/training health)


def to_train_dict(cfg: ExperimentConfig) -> dict:
    """Adapt structured train/net config to the UPPERCASE dict make_train uses."""
    t, n = cfg.train, cfg.net
    return {
        "NUM_ENVS": int(t.num_envs),
        "ROLLOUT_LEN": int(t.rollout_len),
        "TOTAL_TIMESTEPS": int(t.total_timesteps),
        "UPDATE_EPOCHS": int(t.update_epochs),
        "NUM_MINIBATCHES": int(t.num_minibatches),
        "LR": float(t.lr),
        "GAMMA": float(t.gamma),
        "GAE_LAMBDA": float(t.gae_lambda),
        "CLIP_EPS": float(t.clip_eps),
        "VF_COEF": float(t.vf_coef),
        "ENT_COEF": float(t.ent_coef),
        "ANNEAL_LR": bool(t.anneal_lr),
        "MAX_GRAD_NORM": float(t.max_grad_norm),
        "HIDDEN_DIMS": tuple(int(x) for x in n.hidden_dims),
        "ACTIVATION": str(n.activation),
    }
