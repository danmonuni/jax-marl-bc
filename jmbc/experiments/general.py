"""Heterogeneous-RBC driver: three kappa-spread runs + the combined figure 5."""
from __future__ import annotations

import numpy as np
from omegaconf import OmegaConf

from .. import plots
from ..envs import build_env, kappa_spreads, lambda_values, make_grid_arrays
from ..recorder import RunRecorder
from .common import run_single


def _spread_label(kv):
    unique = sorted(set(round(float(v), 2) for v in kv))
    if len(unique) == 1:
        return f"[{unique[0]:.2f} | {unique[0]:.2f}]"
    return "[" + " | ".join(f"{v:.2f}" for v in unique) + "]"


def run(cfg, out_dir: str, run_id: str) -> dict:
    n_grid = int(cfg.env.n_grid)
    spreads = kappa_spreads(n_grid)
    lam_vals = lambda_values(n_grid)
    labels = [_spread_label(kv) for kv in spreads]

    fig5_results = []
    runs = []
    for i, kv in enumerate(spreads):
        kappas, lambdas = make_grid_arrays(kv, lam_vals)
        sub = OmegaConf.merge(cfg, {"env": {
            "n_agents": int(len(kappas)),
            "kappas": [float(x) for x in kappas],
            "lambdas": [float(x) for x in lambdas],
        }})
        env = build_env(sub.env)
        rec = RunRecorder(out_dir, "general", f"{run_id}_spread{i}")
        res = run_single(sub, env=env, recorder=rec, seed=int(cfg.run.seed) + i * 100)
        rec_final = res["recs"][-1]
        fig5_results.append((kappas, lambdas, rec_final, res["out"]["metrics"]))
        runs.append(res)
        print(f"[general:spread{i}] {res['timing']['wall_time_s']:.1f}s -> {rec.dir}")

    combined = RunRecorder(out_dir, "general", f"{run_id}_figure5")
    plots.plot_general_fig5(fig5_results, labels, n_grid,
                            combined.figure_path("figure5.png"))
    print(f"[general] combined figure -> {combined.dir}")
    return {"runs": runs, "labels": labels}
