"""Structured, reproducible run output.

Each run gets ``<out_dir>/<exp>/<run_id>/`` containing:
    config.yaml        resolved configuration
    metrics.csv        per-update training metrics
    diagnostics.json   economic + distributional probes across snapshots
    timing.json        compile/run split, throughput, device
    rollouts.npz       raw snapshot rollouts (every recorded channel, all
                       snapshots stacked on a leading axis) — the complete
                       experimental record from which every figure and
                       diagnostic can be recomputed ex post (jmbc.analyze)
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


_T0 = time.perf_counter()


def phase(msg: str) -> None:
    """Host-side orchestration log line with elapsed wall time since import.

    Used for everything OUTSIDE the compiled program (config, env build,
    tracing, XLA compilation, diagnostics, IO) — zero effect on the XLA graph.
    """
    print(f"[jmbc +{time.perf_counter() - _T0:7.1f}s] {msg}", flush=True)


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

    def save_rollouts(self, recs, idxs, steps_per_update: int,
                      max_agents: Optional[int] = None) -> None:
        """Persist the raw snapshot rollouts: one array per channel with a
        leading snapshot axis, plus the snapshot -> training-step mapping.

        ``max_agents``: keep only the first ``max_agents`` agents in per-agent
        channels (aggregate channels are always complete) so the file stays
        transferable at large n_agents.
        """
        n = recs[0]["ks"].shape[-1]
        arrays = {}
        for k in recs[0]:
            a = np.stack([np.asarray(r[k]) for r in recs])
            if max_agents is not None and a.ndim >= 3 and a.shape[-1] == n:
                a = a[..., : int(max_agents)]
            arrays[k] = a
        arrays["snap_update_idxs"] = np.asarray(idxs, np.int64)
        arrays["snap_env_steps"] = (np.asarray(idxs, np.int64) + 1) * int(steps_per_update)
        if max_agents is not None:
            arrays["saved_agents"] = np.asarray(min(int(max_agents), n), np.int64)
        np.savez_compressed(self.dir / "rollouts.npz", **arrays)

    def figure_path(self, name: str) -> str:
        return str(self.fig_dir / name)


def load_rollouts(run_dir):
    """Inverse of ``save_rollouts``: return (recs, idxs, env_steps)."""
    with np.load(Path(run_dir) / "rollouts.npz") as z:
        idxs = z["snap_update_idxs"]
        env_steps = z["snap_env_steps"]
        keys = [k for k in z.files
                if not k.startswith("snap_") and k != "saved_agents"]
        recs = [{} for _ in range(len(idxs))]
        # Decompress each channel once; .copy() so no slice pins the full
        # stacked array (a view would keep every channel's whole buffer alive,
        # blowing memory as snapshots x file size).
        for k in keys:
            a = z[k]
            for s in range(len(idxs)):
                recs[s][k] = a[s].copy()
    return recs, idxs, env_steps


def device_report(requested: str) -> str:
    """One-line banner of the device JAX *actually* resolved.

    Import jax lazily: this must run after setup_device() pinned the platform.
    Catches the classic Colab failure of silently falling back to CPU.
    """
    import jax
    devs = jax.devices()
    kinds = ", ".join(sorted({f"{d.platform}:{d.device_kind}" for d in devs}))
    line = (f"[device] requested={requested} -> backend={jax.default_backend()} "
            f"({len(devs)} device(s): {kinds})")
    if requested in ("gpu", "cuda") and jax.default_backend() == "cpu":
        line += "  ** WARNING: GPU requested but JAX resolved CPU **"
    return line


# ── timing helpers ────────────────────────────────────────────────────────────

def _total_env_steps(train_fn) -> int:
    c = getattr(train_fn, "config", {})
    try:
        return int(c["NUM_UPDATES"] * c["ROLLOUT_LEN"] * c["NUM_ENVS"])
    except Exception:
        return 0


def run_and_time(train_fn, rng) -> tuple:
    """Run once, blocking on the result. Returns (out, timing).

    When the train fn exposes the AOT ``lower`` hook, the trace / XLA-compile /
    run phases are timed and announced separately (all host-side; the compiled
    program is identical). Total wall time still includes compilation.
    """
    import jax
    t0 = time.perf_counter()
    steps = _total_env_steps(train_fn)
    timing = {"device": str(jax.devices()[0].platform)}

    if hasattr(train_fn, "lower"):
        phase("tracing train program (building the XLA graph) ...")
        compile_ = train_fn.lower(rng)
        t1 = time.perf_counter()
        timing["trace_time_s"] = t1 - t0
        phase(f"traced in {t1 - t0:.1f}s; XLA-compiling ...")
        run = compile_()
        t2 = time.perf_counter()
        timing["compile_time_s"] = t2 - t1
        phase(f"compiled in {t2 - t1:.1f}s; running "
              f"({steps} sequential env steps) ...")
        out = jax.block_until_ready(run())
        t3 = time.perf_counter()
        timing["run_time_s"] = t3 - t2
        phase(f"training ran in {t3 - t2:.1f}s")
    else:  # fallback: opaque compile+run (external train fns)
        out = jax.block_until_ready(train_fn(rng))

    wall = time.perf_counter() - t0
    timing["wall_time_s"] = wall
    timing["env_steps"] = steps
    run_s = timing.get("run_time_s", wall)
    timing["throughput_steps_per_s"] = (steps / run_s) if run_s > 0 else None
    return out, timing


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
