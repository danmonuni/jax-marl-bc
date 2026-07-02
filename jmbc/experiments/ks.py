"""Krusell-Smith driver: one run + the law-of-motion / wealth / MPC figure."""
from __future__ import annotations

import numpy as np

from .. import plots
from ..recorder import RunRecorder
from .common import run_single


def run(cfg, out_dir: str, run_id: str) -> dict:
    rec = RunRecorder(out_dir, "ks", run_id)
    res = run_single(cfg, recorder=rec)

    # KS figure 4: reuse the snapshot rollouts computed for diagnostics.
    recs = res["recs"]
    idxs = res["idxs"]
    if recs is not None and len(recs) >= 1:
        snap_steps = (np.asarray(idxs) + 1) * res["steps_per_update"]
        max_steps = int(cfg.env.max_steps)
        plots.plot_ks_fig4(recs, idxs, snap_steps, max_steps,
                           rec.figure_path("ks_fig4.png"))
    print(f"[ks] {res['timing']['wall_time_s']:.1f}s -> {rec.dir}")
    return res
