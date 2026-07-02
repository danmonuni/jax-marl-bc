"""Meta experiment / benchmark runner.

    python -m jmbc.sweep sweep=scaling

Evaluates the Cartesian product of ``axes`` for a base experiment, timing each
cell (compile vs run split, throughput) and optionally tabulating end-of-run
diagnostics. Writes ``benchmarks/<name>/results.csv`` plus scaling figures.
A ``method`` column is included so the original implementation can be appended
later and overlaid as "standard vs JaxMARL-BC".
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from omegaconf import OmegaConf

from .config import load_config, load_sweep, parse_cli, setup_device


def _overrides_to_dotlist(d: Dict) -> List[str]:
    return [f"{k}={v}" for k, v in d.items()]


def _extract_diag_scalars(summary) -> Dict[str, float]:
    """Pull a few headline scalars from the final-snapshot diagnostics."""
    if not summary:
        return {}
    final = summary.get("final", {})
    econ = final.get("economic", {})
    dist = final.get("distributional", {})
    out = {}
    if "euler" in econ:
        out["euler_mean_abs"] = econ["euler"].get("euler_mean_abs")
    if "ks_forecast" in econ:
        out["ks_lom_r2"] = econ["ks_forecast"].get("ks_lom_r2")
        out["den_haan_max_pct"] = econ["ks_forecast"].get("den_haan_max_pct")
    if "capital_gini" in dist:
        out["capital_gini"] = dist["capital_gini"]
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    sweep_name, overrides = parse_cli(argv, key="sweep")
    if sweep_name is None:
        print("usage: python -m jmbc.sweep sweep=<name> [key=value ...]")
        raise SystemExit(2)

    scfg = load_sweep(sweep_name, overrides)
    base_over = _overrides_to_dotlist(OmegaConf.to_container(scfg.overrides, resolve=True))

    # Resolve device from the base experiment config before importing jax.
    base_cfg = load_config(scfg.base_exp, base_over)
    setup_device(base_cfg.run.device)

    from .experiments.common import run_single
    from .plots import make_benchmark_figures
    from .recorder import RunRecorder
    import pandas as pd

    axes = OmegaConf.to_container(scfg.axes, resolve=True) or {}
    keys = list(axes)
    value_lists = [axes[k] for k in keys]
    combos = list(itertools.product(*value_lists)) if keys else [()]

    out_dir = Path("benchmarks") / scfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    total = len(combos) * scfg.repeats
    n = 0
    for combo in combos:
        for rep in range(scfg.repeats):
            n += 1
            dotlist = base_over + [f"{k}={v}" for k, v in zip(keys, combo)]
            cfg = load_config(scfg.base_exp, dotlist)
            tag = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo))
            print(f"[sweep {n}/{total}] {tag} rep={rep}")
            recorder = None
            if scfg.save_cell_runs:
                slug = "_".join(f"{k.split('.')[-1]}{v}" for k, v in zip(keys, combo))
                recorder = RunRecorder(str(out_dir / "cells"), scfg.base_exp,
                                       f"{slug or 'base'}_rep{rep}")
            res = run_single(
                cfg,
                recorder=recorder,
                seed=int(cfg.run.seed) + rep,
                do_diagnostics=bool(scfg.collect_diagnostics),
                do_figures=bool(scfg.save_cell_runs),
                benchmark=True,
            )
            row = {"method": "jaxmarl-bc", "base_exp": scfg.base_exp, "repeat": rep}
            for k, v in zip(keys, combo):
                row[k.split(".")[-1]] = v
            row["n_agents"] = int(cfg.env.n_agents)
            row["num_envs"] = int(cfg.train.num_envs)
            row["total_timesteps"] = int(cfg.train.total_timesteps)
            row.update(res["timing"])
            row.update(_extract_diag_scalars(res["summary"]))
            rows.append(row)
            t = res["timing"]
            print(f"    run={t.get('run_only_s', t.get('wall_time_s')):.2f}s "
                  f"throughput={t.get('throughput_steps_per_s', 0):.3e} steps/s "
                  f"device={t.get('device')}")

    df = pd.DataFrame(rows)
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}  ({len(df)} rows)")

    figs = make_benchmark_figures(df, axes, str(out_dir))
    for p in figs:
        print(f"  figure -> {p}")


if __name__ == "__main__":
    main()
