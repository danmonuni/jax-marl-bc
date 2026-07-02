"""Structured, reproducible run output.

Each run gets ``<out_dir>/<exp>/<run_id>/`` containing:
    config.yaml        resolved configuration
    metrics.csv        per-update training metrics
    diagnostics.json   economic + distributional probes across snapshots
    timing.json        compile/run split, throughput, device
    figures/*.png      figures
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from omegaconf import OmegaConf


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _jsonable(obj):
    """Recursively coerce numpy scalars/arrays to JSON-friendly types."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


class RunRecorder:
    def __init__(self, out_dir: str, exp: str, run_name: Optional[str] = None):
        self.run_id = run_name or _timestamp()
        self.dir = Path(out_dir) / exp / self.run_id
        self.fig_dir = self.dir / "figures"
        self.fig_dir.mkdir(parents=True, exist_ok=True)

    # ── writers ───────────────────────────────────────────────────────────────
    def save_config(self, cfg) -> None:
        OmegaConf.save(cfg, self.dir / "config.yaml")

    def save_metrics(self, metrics_np: Dict[str, np.ndarray]) -> None:
        # Keep only equal-length 1-D series (per-update); align on the shortest.
        series = {k: np.asarray(v).ravel() for k, v in metrics_np.items()
                  if np.asarray(v).ndim == 1}
        if not series:
            return
        n = min(len(v) for v in series.values())
        cols = {k: v[:n] for k, v in series.items()}
        try:
            import pandas as pd
            df = pd.DataFrame(cols)
            df.insert(0, "update", np.arange(n))
            df.to_csv(self.dir / "metrics.csv", index=False)
        except ImportError:  # pragma: no cover - pandas is a normal dep
            header = "update," + ",".join(cols)
            rows = np.column_stack([np.arange(n)] + [cols[k] for k in cols])
            np.savetxt(self.dir / "metrics.csv", rows, delimiter=",",
                       header=header, comments="")

    def save_diagnostics(self, summary: dict) -> None:
        with open(self.dir / "diagnostics.json", "w") as f:
            json.dump(_jsonable(summary), f, indent=2)

    def save_timing(self, timing: dict) -> None:
        with open(self.dir / "timing.json", "w") as f:
            json.dump(_jsonable(timing), f, indent=2)

    def figure_path(self, name: str) -> str:
        return str(self.fig_dir / name)


# ── timing helpers ────────────────────────────────────────────────────────────

def _total_env_steps(train_fn) -> int:
    c = getattr(train_fn, "config", {})
    try:
        return int(c["NUM_UPDATES"] * c["ROLLOUT_LEN"] * c["NUM_ENVS"])
    except Exception:
        return 0


def run_and_time(train_fn, rng) -> tuple:
    """Run once, blocking on the result. Returns (out, timing).

    Wall time includes JIT compilation (matches the original scripts' report).
    """
    import jax
    t0 = time.perf_counter()
    out = train_fn(rng)
    out = jax.block_until_ready(out)
    wall = time.perf_counter() - t0
    steps = _total_env_steps(train_fn)
    return out, {
        "wall_time_s": wall,
        "env_steps": steps,
        "throughput_steps_per_s": (steps / wall) if wall > 0 else None,
        "device": str(jax.devices()[0].platform),
    }


def benchmark_time(train_fn, rng) -> tuple:
    """Run twice to split compile vs steady-state run time (for benchmarks)."""
    import jax
    t0 = time.perf_counter()
    out = jax.block_until_ready(train_fn(rng))
    compile_plus_run = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = jax.block_until_ready(train_fn(rng))
    run_only = time.perf_counter() - t0

    steps = _total_env_steps(train_fn)
    return out, {
        "compile_plus_run_s": compile_plus_run,
        "run_only_s": run_only,
        "compile_time_s": max(compile_plus_run - run_only, 0.0),
        "env_steps": steps,
        "throughput_steps_per_s": (steps / run_only) if run_only > 0 else None,
        "device": str(jax.devices()[0].platform),
    }
