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
    # Starting capital, drawn once per parallel env and persisted across
    # in-training episode resets. "constant" = every agent at k_init (the
    # pre-randomization default). See jmbc/envs/init_capital.py.
    k_init_dist: str = "uniform"            # "constant" | "uniform" | "lognormal"
    # When a new population is drawn. "per_episode": redrawn at every reset, as
    # the reference implementation does -- the policy then sees num_envs x
    # n_episodes distinct starting populations rather than just num_envs.
    # "per_env": drawn once per parallel env and reused on every reset, i.e. one
    # fixed initial wealth distribution per simulation. Costs ~0.3% of training
    # either way (the reset body is computed every step regardless).
    k_init_resample: str = "per_episode"    # "per_episode" | "per_env"
    k_init_sigma: float = 0.0               # lognormal dispersion
    # Uniform support, bracketing the steady state as the reference KS
    # initialization U(10, 70) does. The reference spans 0.25x-1.75x of K* ~ 40
    # (its beta=0.99 steady state); these bounds keep those proportions at THIS
    # repo's beta=0.95, where K* ~ 11.7. Rescale again if beta moves:
    #   r* = 1/beta - 1 + delta ;  K* = (alpha/r*)^(1/(1-alpha)) * L
    k_init_low: float = 3.0
    k_init_high: float = 20.0
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
    log_every: int = 100                    # progress print every N updates (0 = silent)


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
    # Evaluate the trained policy on a LARGER population than it was trained
    # with (policy inputs are per-agent + aggregates -> size-agnostic).
    # None -> same as env.n_agents. Requires kappas/lambdas = null (homogeneous).
    n_agents: Optional[int] = None
    economic: bool = True                   # Euler errors, resource residual, etc.
    distributional: bool = True             # Gini, Lorenz, top shares
    burn_frac: float = 0.5                  # fraction of rollout discarded as burn-in


@dataclass
class LogConfig:
    """Where structured results are written."""

    out_dir: str = "runs"
    run_name: Optional[str] = None          # None -> auto timestamped id
    save_raw: bool = True                   # persist raw snapshot rollouts (npz)
    # Subsample of agents kept in the SAVED per-agent series (None = all).
    # In-process diagnostics always use every agent; aggregate channels
    # (K, Y, agg_state, ...) are always saved in full. Keeps rollouts.npz
    # transferable for large n_agents (2000 agents full ~ 5 GB).
    save_agents: Optional[int] = None


@dataclass
class RunConfig:
    """Execution controls."""

    seed: int = 0
    device: str = "auto"                    # "auto" | "cpu" | "gpu"
    # Preallocate the GPU pool (75%) instead of growing on demand: fewer
    # fragmentation OOMs for big solo runs; keep false to share the GPU.
    prealloc: bool = False


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
    values. The runner evaluates the Cartesian product (or, with ``paired``,
    zips equal-length axes into a single trajectory of cells, e.g. a
    constant-product agents/envs tradeoff). ``overrides`` are applied to
    every cell before the axis values.

    ``figures`` selects which benchmark graphs are rendered from the timing
    table (see :func:`jmbc.plots.make_sweep_figures`):
    "auto" | "walltime" | "throughput" | "speedup" | "phase" | "tradeoff".
    """

    base_exp: str = "rbc"
    axes: Dict[str, List[Any]] = field(default_factory=dict)
    overrides: Dict[str, Any] = field(default_factory=dict)
    # How many times each cell is re-run, and with which RNG seeds. Give
    # ``seeds`` explicitly ([0, 1, 2]) when the repeats are the sample over
    # which timings get a mean and a standard deviation — the seed then
    # appears in results.csv and cells are collapsed in results_summary.csv.
    # ``repeats`` (legacy) is equivalent to seeds = run.seed + [0..repeats-1].
    repeats: int = 1
    seeds: Optional[List[int]] = None
    name: str = "sweep"
    # Parent directory of the sweep's output dir (<out_dir>/<name>/).
    # "benchmarks" for throwaway timing scans; "runs" for a kept experiment.
    out_dir: str = "benchmarks"
    # Skip (cell, seed) pairs already present in the output dir's results.csv
    # and keep their rows. results.csv is flushed after every cell, so a
    # multi-hour scan that loses its Colab session resumes where it stopped.
    resume: bool = False
    collect_diagnostics: bool = True   # also tabulate Euler/Gini per cell
    save_cell_runs: bool = False       # full per-cell output dir + figures
                                       # (distributional/economic/training health)
    paired: bool = False               # zip axes instead of Cartesian product
    # One timed run per cell: the AOT phase timer already separates trace /
    # XLA-compile / run, so time_s is steady-state without training twice.
    # True = legacy double-run split (second full run measures run_only_s).
    benchmark: bool = False
    figures: List[str] = field(default_factory=lambda: ["auto"])
    # CSV of baseline timings (``method``, ``n_agents``, ``time_hours`` or
    # ``time_s`` columns) overlaid on time figures and used as the numerator
    # of the "speedup" figure. Path is relative to the repo root.
    reference_csv: Optional[str] = None
    # n_agents * num_envs product along which the "tradeoff" figure slices.
    tradeoff_product: Optional[int] = None


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
        "LOG_EVERY": int(t.log_every),
        "HIDDEN_DIMS": tuple(int(x) for x in n.hidden_dims),
        "ACTIVATION": str(n.activation),
    }
