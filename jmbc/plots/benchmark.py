"""Benchmark / scaling plots ("standard" vs "JaxMARL-BC").

All functions read a tidy DataFrame with at least a ``method`` column, so a
baseline series (e.g. the original CPU implementation, digitized or re-run)
overlays automatically once its rows are concatenated to the benchmark table.

``make_sweep_figures`` is the config-driven dispatcher: the sweep YAML's
``figures`` list picks which graphs get rendered from the same timing table.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from .style import apply_style, COLORS

OURS = "jaxmarl-bc"

# Steady-state per-cell time, in preference order: double-run split
# (benchmark_time), AOT phase split (run_and_time), total wall time.
TIME_COLS = ("run_only_s", "run_time_s", "wall_time_s")


def ensure_time_column(df):
    """Add a canonical ``time_s`` column (steady-state run seconds).

    Rows from heterogeneous sources (double-run cells, single timed runs,
    reference CSVs with ``time_hours``) are normalized so every figure reads
    one column.
    """
    df = df.copy()
    if "time_s" not in df.columns:
        df["time_s"] = np.nan
    if "time_hours" in df.columns:
        df["time_s"] = df["time_s"].fillna(df["time_hours"] * 3600.0)
    for c in TIME_COLS:
        if c in df.columns:
            df["time_s"] = df["time_s"].fillna(df[c])
    return df


def _series_iter(df, group_col):
    if group_col and group_col in df.columns:
        keys = list(df[group_col].unique())  # first-seen order: ours first
        for i, key in enumerate(keys):
            yield str(key), df[df[group_col] == key], COLORS[i % len(COLORS)]
    else:
        yield OURS, df, COLORS[0]


def plot_metric_vs(df, x: str, y: str, path: str, group_col: str = "method",
                   loglog: bool = True, ylabel: Optional[str] = None,
                   title: Optional[str] = None) -> None:
    """Generic x-vs-y benchmark plot, one line per ``group_col`` value.

    Repeats (and any other swept axis) are averaged per x value; series with
    no finite y data (e.g. a reference CSV without throughput) are skipped.
    """
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    n_series = 0
    for label, sub, color in _series_iter(df, group_col):
        sub = sub.dropna(subset=[x, y])
        if sub.empty:
            continue
        agg = sub.groupby(x)[y].mean().sort_index()
        ax.plot(agg.index, agg.values, "o-", color=color, label=label)
        n_series += 1
    if loglog:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if n_series > 1:
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_speedup(df, x: str, path: str, ours: str = OURS,
                 title: Optional[str] = None) -> bool:
    """Speedup of ``ours`` over every other method at matching ``x`` values.

    speedup(x) = time_reference(x) / time_ours(x), from the ``time_s`` column.
    Returns False (and renders nothing) when no reference method shares x
    values with ours.
    """
    import matplotlib.pyplot as plt
    df = ensure_time_column(df).dropna(subset=[x, "time_s"])
    ours_t = df[df["method"] == ours].groupby(x)["time_s"].mean()
    refs = [m for m in df["method"].unique() if m != ours]
    series = []
    for m in refs:
        ref_t = df[df["method"] == m].groupby(x)["time_s"].mean()
        common = ours_t.index.intersection(ref_t.index)
        if len(common):
            series.append((m, ref_t[common] / ours_t[common]))
    if not series:
        return False

    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    # Direct-label only the headline series (largest final speedup); the
    # legend identifies the rest — end labels would collide.
    top = max(range(len(series)), key=lambda i: series[i][1].values[-1])
    for i, (label, sp) in enumerate(series):
        color = COLORS[(i + 1) % len(COLORS)]  # color follows the reference
        ax.plot(sp.index, sp.values, "o-", color=color, label=label)
        if i == top:
            ax.annotate(f"{sp.values[-1]:,.0f}x", (sp.index[-1], sp.values[-1]),
                        textcoords="offset points", xytext=(5, 6), fontsize=8)
    ax.axhline(1.0, color="0.6", linestyle="--", linewidth=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(x)
    ax.set_ylabel("speedup (reference time / ours)")
    ax.set_title(title or "Speedup vs reference implementation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_phase_diagram(df, path, x: str = "n_agents", y: str = "num_envs",
                       value: str = "time_s", ours: str = OURS,
                       title: Optional[str] = None) -> bool:
    """Phase diagram of absolute run time over the (x, y) mesh.

    Cells are the mean over repeats; axes are categorical (one row/column per
    swept value, log-spaced grids stay readable). Sequential colormap, log
    color scale, every cell direct-labeled with its time.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    df = ensure_time_column(df)
    if "method" in df.columns:
        df = df[df["method"] == ours]
    df = df.dropna(subset=[x, y, value])
    if df.empty or df[x].nunique() < 2 or df[y].nunique() < 2:
        return False
    piv = df.groupby([y, x])[value].mean().unstack(x).sort_index()

    apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    Z = np.ma.masked_invalid(piv.values)
    mesh = ax.pcolormesh(Z, cmap="viridis",
                         norm=LogNorm(vmin=Z.min(), vmax=Z.max()),
                         edgecolors="white", linewidth=1.5)
    ax.set_xticks(np.arange(piv.shape[1]) + 0.5,
                  [f"{v:g}" for v in piv.columns])
    ax.set_yticks(np.arange(piv.shape[0]) + 0.5,
                  [f"{v:g}" for v in piv.index])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(title or f"Run time (s): {y} vs {x}")
    ax.grid(False)

    log_mid = (np.log(Z.min()) + np.log(Z.max())) / 2
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isfinite(v):
                continue
            txt = f"{v:.2g}" if v < 100 else f"{v:,.0f}"
            dark_cell = np.log(v) < log_mid
            ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center",
                    fontsize=7, color="white" if dark_cell else "black")
    fig.colorbar(mesh, ax=ax, label="steady-state run time (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def plot_tradeoff(df, product: int, path, x: str = "n_agents",
                  y: str = "num_envs", ours: str = OURS,
                  title: Optional[str] = None) -> bool:
    """Run time along the constant-product cut ``x * y == product``.

    The agents/envs tradeoff at fixed batch width: each tick is one split of
    the same n_agents * num_envs budget.
    """
    import matplotlib.pyplot as plt
    df = ensure_time_column(df)
    if "method" in df.columns:
        df = df[df["method"] == ours]
    df = df.dropna(subset=[x, y, "time_s"])
    sub = df[(df[x] * df[y]).round().astype(int) == int(product)]
    if sub.empty:
        return False
    agg = sub.groupby([x, y])["time_s"].mean().reset_index().sort_values(x)

    apply_style()
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(agg[x], agg["time_s"], "o-", color=COLORS[0])
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(agg[x], [f"{a:g}×{e:g}" for a, e in zip(agg[x], agg[y])])
    ax.minorticks_off()
    ax.set_xlabel(f"{x} × {y}  (constant product = {product:g})")
    ax.set_ylabel("steady-state run time (s)")
    ax.set_title(title or f"Agents/envs tradeoff at {x}·{y} = {product:g}")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


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
        time_col = next((c for c in ("time_s",) + TIME_COLS if c in df.columns),
                        None)
        if time_col:
            p = str(out / f"walltime_vs_{col}.png")
            plot_metric_vs(df, col, time_col, p, group_col,
                           ylabel="seconds", title=f"Wall time vs {col}")
            paths.append(p)
    return paths


def make_sweep_figures(df, axes, out_dir: str, figures: List[str],
                       tradeoff_product: Optional[int] = None,
                       group_col: str = "method") -> list:
    """Config-driven figure dispatch: the sweep YAML's ``figures`` list picks
    which graphs are rendered from the timing table.

    Kinds: "auto" (legacy per-axis throughput+walltime), "walltime",
    "throughput" (one figure per swept axis, one line per method),
    "speedup" (ours vs each reference method on matching x), "phase"
    (n_agents x num_envs heatmap of absolute time), "tradeoff" (time along
    n_agents * num_envs == tradeoff_product). Unavailable data degrades to a
    skip message, never an error.
    """
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = ensure_time_column(df)
    axis_cols = [a.split(".")[-1] for a in axes
                 if a.split(".")[-1] in df.columns
                 and df[a.split(".")[-1]].nunique() >= 2]
    paths: list = []

    def _skip(kind, why):
        print(f"  figure '{kind}' skipped: {why}")

    for kind in figures:
        if kind == "auto":
            paths += make_benchmark_figures(df, axes, out_dir, group_col)
        elif kind == "walltime":
            for col in axis_cols:
                p = str(out / f"walltime_vs_{col}.png")
                plot_metric_vs(df, col, "time_s", p, group_col,
                               ylabel="steady-state run time (s)",
                               title=f"Run time vs {col}")
                paths.append(p)
        elif kind == "throughput":
            if "throughput_steps_per_s" not in df.columns:
                _skip(kind, "no throughput column"); continue
            for col in axis_cols:
                p = str(out / f"throughput_vs_{col}.png")
                plot_metric_vs(df, col, "throughput_steps_per_s", p, group_col,
                               ylabel="transitions / s",
                               title=f"Throughput vs {col}")
                paths.append(p)
        elif kind == "speedup":
            col = axis_cols[0] if axis_cols else "n_agents"
            p = str(out / f"speedup_vs_{col}.png")
            if plot_speedup(df, col, p):
                paths.append(p)
            else:
                _skip(kind, "no reference method with matching x values")
        elif kind == "phase":
            p = str(out / "phase_time.png")
            if plot_phase_diagram(df, p):
                paths.append(p)
            else:
                _skip(kind, "needs >=2 values on both n_agents and num_envs")
        elif kind == "tradeoff":
            if not tradeoff_product:
                _skip(kind, "tradeoff_product not set"); continue
            p = str(out / f"tradeoff_product{int(tradeoff_product)}.png")
            if plot_tradeoff(df, int(tradeoff_product), p):
                paths.append(p)
            else:
                _skip(kind, f"no cells with n_agents*num_envs == {tradeoff_product}")
        else:
            _skip(kind, "unknown figure kind")
    return paths
