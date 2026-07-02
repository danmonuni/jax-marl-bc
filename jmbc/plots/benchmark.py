"""Benchmark / scaling plots ("standard" vs "JaxMARL-BC").

All functions read a tidy DataFrame with at least a ``method`` column, so a
future ``standard`` (original implementation) series overlays automatically
once its rows are appended to the benchmark CSV.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .style import apply_style, COLORS


def _series_iter(df, group_col):
    if group_col and group_col in df.columns:
        for i, (key, sub) in enumerate(df.groupby(group_col)):
            yield str(key), sub, COLORS[i % len(COLORS)]
    else:
        yield "jaxmarl-bc", df, COLORS[0]


def plot_metric_vs(df, x: str, y: str, path: str, group_col: str = "method",
                   loglog: bool = True, ylabel: Optional[str] = None,
                   title: Optional[str] = None) -> None:
    """Generic x-vs-y benchmark plot, one line per ``group_col`` value."""
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for label, sub, color in _series_iter(df, group_col):
        sub = sub.sort_values(x)
        ax.plot(sub[x], sub[y], "o-", color=color, label=label)
    if loglog:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if group_col in df.columns and df[group_col].nunique() > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_benchmark_figures(df, axes, out_dir: str, group_col: str = "method") -> list:
    """Auto-generate throughput & wall-time plots for each swept axis.

    ``axes`` is the dict of swept config paths -> values; the per-axis column
    name in the DataFrame is the leaf (e.g. env.n_agents -> n_agents).
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for axis in axes:
        col = axis.split(".")[-1]
        if col not in df.columns or df[col].nunique() < 2:
            continue
        if "throughput_steps_per_s" in df.columns:
            p = str(out / f"throughput_vs_{col}.png")
            plot_metric_vs(df, col, "throughput_steps_per_s", p, group_col,
                           ylabel="env steps / s", title=f"Throughput vs {col}")
            paths.append(p)
        time_col = "run_only_s" if "run_only_s" in df.columns else "wall_time_s"
        if time_col in df.columns:
            p = str(out / f"walltime_vs_{col}.png")
            plot_metric_vs(df, col, time_col, p, group_col,
                           ylabel="seconds", title=f"Wall time vs {col}")
            paths.append(p)
    return paths
