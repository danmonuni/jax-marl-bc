"""Config loading: merge structured defaults + YAML + CLI dotlist overrides.

Loading never imports JAX, so the device can be resolved *before* JAX is
imported (see :func:`setup_device`). This replaces the per-file
``os.environ["JAX_PLATFORM_NAME"] = "cpu"`` hack scattered across the old
``exps/`` scripts.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf

from .schema import ExperimentConfig, SweepConfig

# repo_root/configs  (this file is repo_root/jmbc/config/loader.py)
CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


def _root(config_root: Optional[str]) -> Path:
    return Path(config_root) if config_root else CONFIG_ROOT


def load_config(
    exp: Optional[str] = None,
    overrides: Optional[Sequence[str]] = None,
    config_root: Optional[str] = None,
) -> DictConfig:
    """Build an ExperimentConfig as an OmegaConf object.

    Merge order (later wins): structured defaults < base.yaml < [extends
    parent] < exp/<exp>.yaml < CLI dotlist overrides (e.g. ``env.n_agents=1000``).

    An exp file may declare ``extends: <name>`` (one level) to inherit another
    exp file and override only what differs — e.g. ``ks_local`` = ``ks`` with a
    shorter training budget.
    """
    root = _root(config_root)
    cfg = OmegaConf.structured(ExperimentConfig)

    base = root / "base.yaml"
    if base.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(base))

    if exp:
        exp_file = root / "exp" / f"{exp}.yaml"
        if not exp_file.exists():
            raise FileNotFoundError(f"No experiment config: {exp_file}")
        exp_conf = OmegaConf.load(exp_file)
        parent = exp_conf.pop("extends", None)
        explicit_exp = "exp" in exp_conf
        if parent:
            parent_file = root / "exp" / f"{parent}.yaml"
            if not parent_file.exists():
                raise FileNotFoundError(f"extends: no such experiment: {parent_file}")
            parent_conf = OmegaConf.load(parent_file)
            explicit_exp = explicit_exp or "exp" in parent_conf
            cfg = OmegaConf.merge(cfg, parent_conf)
        cfg = OmegaConf.merge(cfg, exp_conf)
        if not explicit_exp:
            cfg.exp = exp

    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))

    return cfg


def load_sweep(
    sweep: str,
    overrides: Optional[Sequence[str]] = None,
    config_root: Optional[str] = None,
) -> DictConfig:
    """Load a sweep spec from ``configs/sweep/<sweep>.yaml``."""
    root = _root(config_root)
    cfg = OmegaConf.structured(SweepConfig)
    sweep_file = root / "sweep" / f"{sweep}.yaml"
    if not sweep_file.exists():
        raise FileNotFoundError(f"No sweep config: {sweep_file}")
    cfg = OmegaConf.merge(cfg, OmegaConf.load(sweep_file))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def parse_cli(argv: Sequence[str], key: str = "exp") -> Tuple[Optional[str], List[str]]:
    """Split argv into the (exp|sweep) selector and dotlist overrides.

    ``["exp=ks", "env.n_agents=1000"]`` -> ("ks", ["env.n_agents=1000"]).
    The selector token is ``<key>=<value>`` with no dot in the key.
    """
    selector: Optional[str] = None
    overrides: List[str] = []
    for tok in argv:
        if "=" not in tok:
            continue
        lhs, rhs = tok.split("=", 1)
        if lhs == key:
            selector = rhs
        else:
            overrides.append(tok)
    return selector, overrides


def setup_device(device: str, prealloc: bool = False) -> None:
    """Resolve the JAX platform *before* JAX is imported.

    "auto" leaves JAX to pick the best backend (GPU on Colab T4, else CPU).
    Call this prior to importing any jax-dependent module.

    ``prealloc=True`` lets JAX grab its standard 75% GPU pool up front:
    contiguous, fragmentation-free — the right mode for one big training run.
    False (default) grows on demand so several processes can share the GPU.
    """
    device = (device or "auto").lower()
    if device == "cpu":
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    elif device in ("gpu", "cuda"):
        os.environ["JAX_PLATFORM_NAME"] = "gpu"
    # "auto": do not constrain the platform.
    os.environ.setdefault(
        "XLA_PYTHON_CLIENT_PREALLOCATE", "true" if prealloc else "false"
    )
