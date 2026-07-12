"""Krusell-Smith driver: one run + the KS semantic figure set.

Raw snapshot rollouts are persisted (``rollouts.npz``) so every figure and
diagnostic can be recomputed ex post with ``python -m jmbc.analyze``.
"""
from __future__ import annotations

import numpy as np

from ..plots.ks_semantics import render_ks_figures
from ..recorder import RunRecorder
from .common import run_single


def run(cfg, out_dir: str, run_id: str) -> dict:
    rec = RunRecorder(out_dir, "ks", run_id)
    res = run_single(cfg, recorder=rec)

    recs = res["recs"]
    idxs = res["idxs"]
    if recs is not None and len(recs) >= 1:
        if getattr(cfg.log, "save_raw", True):
            rec.save_rollouts(recs, idxs, res["steps_per_update"])
        snap_steps = (np.asarray(idxs) + 1) * res["steps_per_update"]
        render_ks_figures(recs, snap_steps, int(cfg.env.max_steps), rec.fig_dir,
                          burn_frac=float(cfg.diag.burn_frac))
    print(f"[ks] {res['timing']['wall_time_s']:.1f}s -> {rec.dir}")
    return res
