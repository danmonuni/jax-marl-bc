"""Krusell-Smith driver: one run + the KS semantic figure set.

Raw snapshot rollouts are persisted (``rollouts.npz``) so every figure and
diagnostic can be recomputed ex post with ``python -m jmbc.analyze``.
"""
from __future__ import annotations

import numpy as np

from ..plots.ks_semantics import render_ks_figures
from ..recorder import RunRecorder, phase
from .common import run_single


def run(cfg, out_dir: str, run_id: str) -> dict:
    rec = RunRecorder(out_dir, "ks", run_id)
    res = run_single(cfg, recorder=rec)

    recs = res["recs"]
    idxs = res["idxs"]
    if recs is not None and len(recs) >= 1:
        if getattr(cfg.log, "save_raw", True):
            rec.save_rollouts(recs, idxs, res["steps_per_update"],
                              max_agents=cfg.log.save_agents)
            size_mb = (rec.dir / "rollouts.npz").stat().st_size / 1e6
            phase(f"raw rollouts saved: rollouts.npz ({size_mb:.1f} MB)")
        snap_steps = (np.asarray(idxs) + 1) * res["steps_per_update"]
        phase("rendering KS figures (lom_evolution, wealth_heatmap, mpc, fig4) ...")
        render_ks_figures(recs, snap_steps, rec.fig_dir,
                          burn_frac=float(cfg.diag.burn_frac))
    print(f"[ks] {res['timing']['wall_time_s']:.1f}s -> {rec.dir}")
    return res
