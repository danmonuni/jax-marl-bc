"""RBC driver: textbook (delta=1.0, closed form) and typical (delta=0.025)."""
from __future__ import annotations

from omegaconf import OmegaConf

from .. import plots
from ..recorder import RunRecorder
from .common import run_single


def _analytical_targets(alpha, beta, b):
    c_star = 1.0 - alpha * beta
    l_star = alpha / (b * (1.0 - (1.0 - alpha) * beta) + alpha)
    return c_star, l_star


def run(cfg, out_dir: str, run_id: str) -> dict:
    alpha, beta, b = cfg.env.alpha, cfg.env.beta, cfg.env.b
    c1, l1 = _analytical_targets(alpha, beta, b)

    variants = [
        ("textbook", 1.0, c1, l1),
        ("typical", 0.025, 0.1611, 0.1222),
    ]
    results = {}
    for name, delta, c_t, l_t in variants:
        sub = OmegaConf.merge(cfg, {"env": {"delta": float(delta)}})
        rec = RunRecorder(out_dir, "rbc", f"{run_id}_{name}")
        res = run_single(sub, recorder=rec)
        plots.plot_rbc_policy(
            res["out"]["metrics"], res["steps_per_update"], c_t, l_t,
            rec.figure_path(f"{name}_policy.png"), title=f"rbc_{name}",
        )
        print(f"[rbc:{name}] {res['timing']['wall_time_s']:.1f}s -> {rec.dir}")
        results[name] = res
    return results
