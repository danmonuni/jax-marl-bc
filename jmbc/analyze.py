"""Ex-post analysis: recompute diagnostics + figures from a saved run.

    python -m jmbc.analyze runs/ks/<run_id> [more run dirs ...]

Consumes only the persisted experimental record (``rollouts.npz`` +
``config.yaml`` + ``metrics.csv``) — no training, no GPU, no parameters — so
figures and diagnostics can be iterated on locally after a run (e.g. one done
on Colab). Outputs overwrite ``diagnostics.json`` and ``figures/`` in place.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

from .diagnostics.economic import economic_report
from .diagnostics.distributional import distributional_report
from .recorder import load_rollouts
from . import plots
from .plots.ks_semantics import render_ks_figures


def analyze_run(run_dir: str) -> dict:
    run_dir = Path(run_dir)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    recs, idxs, env_steps = load_rollouts(run_dir)
    burn = float(cfg.diag.burn_frac)
    steps_per_update = int(env_steps[0] // (idxs[0] + 1))

    snapshots = []
    for idx, rec in zip(idxs, recs):
        snapshots.append({
            "update_idx": int(idx),
            "economic": economic_report(rec, cfg.env, burn),
            "distributional": distributional_report(rec, burn),
        })
    summary = {
        "snapshot_indices": [int(i) for i in idxs],
        "num_updates": int(idxs[-1]) + 1,
        "snapshots": snapshots,
        "final": snapshots[-1],
    }

    from .recorder import _jsonable
    import json
    with open(run_dir / "diagnostics.json", "w") as f:
        json.dump(_jsonable(summary), f, indent=2)

    fig_dir = run_dir / "figures"
    fig_dir.mkdir(exist_ok=True)
    plots.plot_distributional_snapshots(summary, steps_per_update,
                                        str(fig_dir / "distributional.png"))

    metrics_csv = run_dir / "metrics.csv"
    if metrics_csv.exists():
        import pandas as pd
        df = pd.read_csv(metrics_csv)
        metrics_np = {c: df[c].to_numpy() for c in df.columns if c != "update"}
        plots.plot_training_health(metrics_np, str(fig_dir / "training_health.png"))

    if cfg.env.kind == "ks":
        render_ks_figures(recs, np.asarray(env_steps, float),
                          fig_dir, burn_frac=burn)

    print(f"[analyze] {run_dir}: {len(recs)} snapshots -> diagnostics.json + figures/")
    return summary


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m jmbc.analyze <run_dir> [<run_dir> ...]")
        raise SystemExit(2)
    for d in argv:
        analyze_run(d)


if __name__ == "__main__":
    main()
