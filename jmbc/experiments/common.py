"""Shared single-run machinery used by every driver and by the sweep runner.

``run_single`` builds the env, trains, times the run, computes diagnostics, and
(optionally) writes structured outputs + standard figures. Bespoke per-figure
work is left to the drivers, which reuse the rollouts returned here.
"""
from __future__ import annotations

from typing import Optional

import jax

from .. import plots
from ..algos import make_train
from ..config.schema import to_train_dict
from ..diagnostics import compute_diagnostics, metrics_to_numpy
from ..envs import build_env
from ..recorder import RunRecorder, run_and_time, benchmark_time


def run_single(
    cfg,
    env=None,
    recorder: Optional[RunRecorder] = None,
    seed: Optional[int] = None,
    do_diagnostics: bool = True,
    do_figures: bool = True,
    benchmark: bool = False,
    diag_seed: int = 7,
) -> dict:
    seed = int(cfg.run.seed) if seed is None else int(seed)
    env = build_env(cfg.env) if env is None else env

    train_fn = make_train(env, to_train_dict(cfg))
    rng = jax.random.PRNGKey(seed)
    timer = benchmark_time if benchmark else run_and_time
    out, timing = timer(train_fn, rng)

    cfgd = train_fn.config
    net = train_fn.network
    steps_per_update = cfgd["ROLLOUT_LEN"] * cfgd["NUM_ENVS"]

    metrics_np = metrics_to_numpy(out["metrics"])

    summary, recs, idxs = None, None, None
    if do_diagnostics:
        summary, recs, idxs = compute_diagnostics(
            env, cfg.env, net, out["params_history"], cfg.diag, diag_seed
        )

    if recorder is not None:
        recorder.save_config(cfg)
        recorder.save_metrics(metrics_np)
        recorder.save_timing(timing)
        if summary is not None:
            recorder.save_diagnostics(summary)
        if do_figures:
            plots.plot_training_health(metrics_np, recorder.figure_path("training_health.png"))
            if summary is not None:
                plots.plot_economic_snapshots(
                    summary, cfg.env.kind, steps_per_update,
                    recorder.figure_path("economic.png"),
                )
                plots.plot_distributional_snapshots(
                    summary, steps_per_update, recorder.figure_path("distributional.png"),
                )

    return {
        "out": out,
        "metrics_np": metrics_np,
        "timing": timing,
        "summary": summary,
        "recs": recs,
        "idxs": idxs,
        "net": net,
        "env": env,
        "config": cfgd,
        "steps_per_update": steps_per_update,
    }
