"""Meta experiment / benchmark runner.

    python -m jmbc.sweep sweep=<name>      # reads configs/sweep/<name>.yaml

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


def _dot_value(v) -> str:
    """Render a YAML-loaded override value in OmegaConf dotlist syntax.

    Python repr is not that syntax: ``None`` must round-trip as ``null`` and
    ``True`` as ``true``, or the merge assigns the *strings* "None"/"True" and
    the structured schema rejects them (e.g. env.kappas: Optional[List[float]]).
    """
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(_dot_value(x) for x in v) + "]"
    return str(v)


def _overrides_to_dotlist(d: Dict) -> List[str]:
    return [f"{k}={_dot_value(v)}" for k, v in d.items()]


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


def resolve_seeds(scfg, base_seed: int) -> List[int]:
    """The RNG seeds each cell is re-run under.

    Explicit ``seeds: [0, 1, 2]`` wins; otherwise the legacy ``repeats: N``
    means base_seed + 0..N-1, which is what the runner always did implicitly.
    """
    seeds = OmegaConf.to_container(scfg.seeds, resolve=True) if scfg.seeds else None
    if seeds:
        return [int(s) for s in seeds]
    return [base_seed + r for r in range(int(scfg.repeats))]


def write_results(df, axes: Dict[str, List], out_dir: Path, write_raw: bool = True):
    """Write the per-seed table and its per-cell mean/std collapse.

    results.csv         one row per (cell, seed) — the raw sample
    results_summary.csv one row per cell — mean/std/sem/min/max + n_seeds

    ``write_raw=False`` regenerates only the summary (replot reads results.csv
    and must not rewrite its own input).
    """
    from .plots import summarize_repeats

    if write_raw:
        csv_path = out_dir / "results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\nWrote {csv_path}  ({len(df)} rows)")

    axis_cols = [a.split(".")[-1] for a in axes]
    summary = summarize_repeats(df, axis_cols)
    sum_path = out_dir / "results_summary.csv"
    summary.to_csv(sum_path, index=False)
    print(f"Wrote {sum_path}  ({len(summary)} cells)")
    return summary


def print_summary(summary, axis_cols: List[str]) -> None:
    """Console rendering of the per-cell timing statistics."""
    cols = [c for c in ["method", *axis_cols] if c in summary.columns]
    print("\nper-cell timing (mean +- sample std over seeds)")
    for _, r in summary.iterrows():
        cell = " ".join(f"{c}={r[c]}" for c in cols)
        n = int(r["n_seeds"])
        std = float(r.get("time_s_std", float("nan")) or float("nan"))
        ok = n > 1 and std == std                    # NaN-safe: n>=2 and finite
        std_txt = f"{std:8.2f}s" if ok else "     n/a"
        cv = f"  cv={100 * std / r['time_s_mean']:.1f}%" \
            if ok and r["time_s_mean"] else ""
        print(f"  {cell:<44} {r['time_s_mean']:10.2f}s +- {std_txt}  "
              f"(n={n}){cv}")


def cell_config(scfg, combo: Sequence):
    """The fully resolved experiment config for one cell, exactly as the
    runner builds it: sweep ``overrides`` first, then the cell's axis values.

    Public so a sweep can be *validated* — printed, diffed against another
    sweep — before committing hours of GPU time to it. Imports no JAX.
    """
    base_over = _overrides_to_dotlist(
        OmegaConf.to_container(scfg.overrides, resolve=True))
    keys = list(OmegaConf.to_container(scfg.axes, resolve=True) or {})
    return load_config(
        scfg.base_exp,
        base_over + [f"{k}={v}" for k, v in zip(keys, combo)])


def _cell_key(axis_cols: Sequence[str], values: Sequence, seed) -> tuple:
    """Identity of one timed run: its axis values plus its seed.

    Stringified because the same cell arrives as YAML ints when running and as
    numpy/pandas scalars when read back from results.csv on resume.
    """
    def norm(v):
        try:                                    # 10, 10.0, "10" -> "10"
            return str(int(float(v)))
        except (TypeError, ValueError):
            return str(v)
    return tuple(norm(v) for v in (*values, seed))


def load_done(out_dir: Path, axis_cols: Sequence[str]):
    """Rows already in ``out_dir/results.csv``, and the set of (cell, seed)
    keys they cover. Returns ([], set()) when there is nothing to resume from.
    """
    import pandas as pd

    csv_path = out_dir / "results.csv"
    if not csv_path.exists():
        return [], set()
    df = pd.read_csv(csv_path)
    missing = [c for c in (*axis_cols, "seed") if c not in df.columns]
    if missing:
        print(f"[resume] ignoring {csv_path}: no {', '.join(missing)} column "
              f"(pre-multiseed file) — every cell will be re-run")
        return [], set()
    rows = df.to_dict("records")
    done = {_cell_key(axis_cols, [r[c] for c in axis_cols], r["seed"])
            for r in rows}
    print(f"[resume] {csv_path}: {len(rows)} run(s) already timed")
    return rows, done


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
    axes = OmegaConf.to_container(scfg.axes, resolve=True) or {}
    summary = write_results(df, axes, results_dir, write_raw=False)
    print_summary(summary, [k.split(".")[-1] for k in axes])
    if scfg.reference_csv:
        ref = load_reference(str(scfg.reference_csv))
        df = pd.concat([df, ref], ignore_index=True)
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

    out_dir = Path(str(scfg.out_dir)) / scfg.name
    out_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(scfg, out_dir / "sweep.yaml")

    seeds = resolve_seeds(scfg, int(base_cfg.run.seed))
    axis_cols = [k.split(".")[-1] for k in keys]
    print(f"[sweep] {len(combos)} cells x {len(seeds)} seed(s) {seeds} "
          f"-> {out_dir}")

    rows, done = load_done(out_dir, axis_cols) if scfg.resume else ([], set())
    total = len(combos) * len(seeds)
    n = 0
    for combo in combos:
        for rep, seed in enumerate(seeds):
            n += 1
            if _cell_key(axis_cols, combo, seed) in done:
                print(f"[sweep {n}/{total}] "
                      + ", ".join(f"{c}={v}" for c, v in zip(axis_cols, combo))
                      + f" seed={seed} — already done, skipping")
                continue
            cfg = cell_config(scfg, combo)
            tag = ", ".join(f"{k.split('.')[-1]}={v}" for k, v in zip(keys, combo))
            print(f"[sweep {n}/{total}] {tag} seed={seed} (rep {rep})")
            recorder = None
            if scfg.save_cell_runs:
                slug = "_".join(f"{k.split('.')[-1]}{v}" for k, v in zip(keys, combo))
                recorder = RunRecorder(str(out_dir / "cells"), scfg.base_exp,
                                       f"{slug or 'base'}_seed{seed}")
            res = run_single(
                cfg,
                recorder=recorder,
                seed=seed,
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
            row = {"method": method, "base_exp": scfg.base_exp,
                   "repeat": rep, "seed": seed}
            for k, v in zip(keys, combo):
                row[k.split(".")[-1]] = v
            row["n_agents"] = int(cfg.env.n_agents)
            row["num_envs"] = int(cfg.train.num_envs)
            row["total_timesteps"] = int(cfg.train.total_timesteps)
            row.update(res["timing"])
            # Agent-transitions/s: the throughput that varies with the axis.
            # timing's throughput_steps_per_s counts SEQUENTIAL env steps, held
            # fixed across cells by design; each such step advances n_agents
            # actors. Recorded, not just derived at plot time.
            thr = res["timing"].get("throughput_steps_per_s")
            if thr:
                row["transitions_per_s"] = thr * int(cfg.env.n_agents)
            row.update(_extract_diag_scalars(res["summary"]))
            rows.append(row)
            t = res["timing"]
            run_s = t.get("run_only_s", t.get("run_time_s", t.get("wall_time_s")))
            print(f"    run={run_s:.2f}s "
                  f"throughput={t.get('throughput_steps_per_s', 0):.3e} steps/s "
                  f"device={t.get('device')}")
            # Flush after every cell: a multi-hour scan that loses its session
            # keeps everything timed so far, and `resume: true` picks up here.
            ensure_time_column(pd.DataFrame(rows)).to_csv(
                out_dir / "results.csv", index=False)

    df = ensure_time_column(pd.DataFrame(rows))
    summary = write_results(df, axes, out_dir)
    print_summary(summary, [k.split(".")[-1] for k in keys])

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
