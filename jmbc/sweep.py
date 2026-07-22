"""Meta experiment / benchmark runner.

    python -m jmbc.sweep sweep=scaling

Evaluates the Cartesian product of ``axes`` (or their zip, with ``paired``)
for a base experiment, timing each cell (compile vs run split, throughput) and
optionally tabulating end-of-run diagnostics. Writes
``benchmarks/<name>/{results.csv,sweep.yaml}`` plus the figures selected by
the sweep's ``figures`` list (walltime / throughput / speedup / phase /
tradeoff — see :func:`jmbc.plots.make_sweep_figures`). A ``method`` column is
included, and ``reference_csv`` timings (e.g. the original implementation's
digitized CPU times) are overlaid as extra methods and used for speedups.
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


def build_combos(axes: Dict[str, List], paired: bool = False) -> List[tuple]:
    """Cells to evaluate: Cartesian product of the axes, or — ``paired`` —
    the zip of equal-length axes (e.g. a constant n_agents*num_envs cut)."""
    keys = list(axes)
    value_lists = [axes[k] for k in keys]
    if not keys:
        return [()]
    if paired:
        lengths = {len(v) for v in value_lists}
        if len(lengths) > 1:
            raise ValueError(
                f"paired sweep needs equal-length axes, got "
                f"{ {k: len(v) for k, v in axes.items()} }")
        return list(zip(*value_lists))
    return list(itertools.product(*value_lists))


def load_reference(path_str: str):
    """Baseline timing table (``method`` + ``time_hours``/``time_s`` columns).

    Relative paths resolve against the CWD first, then the repo root, so the
    same sweep YAML works from Colab checkouts and local shells alike.
    """
    import pandas as pd
    from .config.loader import CONFIG_ROOT

    p = Path(path_str)
    if not p.exists():
        p = CONFIG_ROOT.parent / path_str
    if not p.exists():
        raise FileNotFoundError(f"reference_csv not found: {path_str}")
    return pd.read_csv(p, comment="#")


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


def replot(scfg, results_dir: Path) -> None:
    """Re-render the sweep's figures from a saved results.csv — no training,
    no JAX. Lets figure kinds be added/tweaked ex post (mirrors jmbc.analyze).
    """
    import pandas as pd
    from .plots import ensure_time_column, make_sweep_figures

    csv_path = results_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"no results.csv under {results_dir}")
    df = ensure_time_column(pd.read_csv(csv_path))
    print(f"replotting from {csv_path}  ({len(df)} rows)")
    if scfg.reference_csv:
        ref = load_reference(str(scfg.reference_csv))
        df = pd.concat([df, ref], ignore_index=True)
    axes = OmegaConf.to_container(scfg.axes, resolve=True) or {}
    figs = make_sweep_figures(
        df, axes, str(results_dir), figures=list(scfg.figures),
        tradeoff_product=(int(scfg.tradeoff_product)
                          if scfg.tradeoff_product else None),
    )
    for p in figs:
        print(f"  figure -> {p}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    sweep_name, overrides = parse_cli(argv, key="sweep")
    if sweep_name is None:
        print("usage: python -m jmbc.sweep sweep=<name> [key=value ...] "
              "[replot=<benchmarks/name dir>]")
        raise SystemExit(2)

    replot_dir = None
    for tok in list(overrides):
        if tok.startswith("replot="):
            replot_dir = tok.split("=", 1)[1]
            overrides.remove(tok)

    scfg = load_sweep(sweep_name, overrides)
    if replot_dir is not None:
        replot(scfg, Path(replot_dir))
        return
    base_over = _overrides_to_dotlist(OmegaConf.to_container(scfg.overrides, resolve=True))

    # Resolve device from the base experiment config before importing jax.
    base_cfg = load_config(scfg.base_exp, base_over)
    setup_device(base_cfg.run.device, bool(base_cfg.run.prealloc))

    from .experiments.common import run_single
    from .plots import ensure_time_column, make_sweep_figures
    from .recorder import RunRecorder, device_report
    import pandas as pd

    print(device_report(base_cfg.run.device))

    axes = OmegaConf.to_container(scfg.axes, resolve=True) or {}
    keys = list(axes)
    combos = build_combos(axes, paired=bool(scfg.paired))

    out_dir = Path("benchmarks") / scfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(scfg, out_dir / "sweep.yaml")

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
                benchmark=bool(scfg.benchmark),
            )
            # Tag the method by resolved backend so a CPU-forced run of this
            # same framework overlays as its own series (e.g. via
            # reference_csv) instead of averaging into the GPU numbers under
            # the same "jaxmarl-bc" label. GPU (the default target) keeps the
            # bare name so every existing sweep's figures are unaffected.
            device_tag = res["timing"].get("device") or "unknown"
            method = "jaxmarl-bc" if device_tag == "gpu" else f"jaxmarl-bc-{device_tag}"
            row = {"method": method, "base_exp": scfg.base_exp, "repeat": rep}
            for k, v in zip(keys, combo):
                row[k.split(".")[-1]] = v
            row["n_agents"] = int(cfg.env.n_agents)
            row["num_envs"] = int(cfg.train.num_envs)
            row["total_timesteps"] = int(cfg.train.total_timesteps)
            row.update(res["timing"])
            row.update(_extract_diag_scalars(res["summary"]))
            rows.append(row)
            t = res["timing"]
            run_s = t.get("run_only_s", t.get("run_time_s", t.get("wall_time_s")))
            print(f"    run={run_s:.2f}s "
                  f"throughput={t.get('throughput_steps_per_s', 0):.3e} steps/s "
                  f"device={t.get('device')}")

    df = ensure_time_column(pd.DataFrame(rows))
    csv_path = out_dir / "results.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {csv_path}  ({len(df)} rows)")

    if scfg.reference_csv:
        ref = load_reference(str(scfg.reference_csv))
        print(f"overlaying reference timings: {scfg.reference_csv} "
              f"({ref['method'].nunique()} method(s))")
        df = pd.concat([df, ref], ignore_index=True)

    figs = make_sweep_figures(
        df, axes, str(out_dir), figures=list(scfg.figures),
        tradeoff_product=(int(scfg.tradeoff_product)
                          if scfg.tradeoff_product else None),
    )
    for p in figs:
        print(f"  figure -> {p}")


if __name__ == "__main__":
    main()
