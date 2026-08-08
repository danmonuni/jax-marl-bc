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

# Reader-facing names for the parallelism axes; the DataFrame columns keep the
# raw config leaf names. Sentence case, per the paper's plotting standards.
AXIS_LABELS = {
    "n_agents": "Number of agents, $n$",
    "num_envs": "Number of environments, $E$",
}


def axis_label(col: str) -> str:
    return AXIS_LABELS.get(col, col)


def ensure_time_column(df):
    """Normalize the derived metric columns every figure reads.

    ``time_s`` — canonical steady-state run seconds. Rows from heterogeneous
    sources (double-run cells, single timed runs, reference CSVs with
    ``time_hours``) are reconciled onto one column.

    ``transitions_per_s`` — agent-transitions per second. THE throughput
    measure for an agent-scaling sweep, and not the same thing as
    ``throughput_steps_per_s``: the latter counts *sequential env steps*
    (NUM_UPDATES x ROLLOUT_LEN x NUM_ENVS), which is held fixed across cells
    by design, so it is just time_s inverted and says nothing about how much
    work the device absorbed. Each sequential step advances n_agents actors,
    hence the n_agents factor. Derived here rather than only at write time so
    an older results.csv replots correctly.
    """
    df = df.copy()
    if "time_s" not in df.columns:
        df["time_s"] = np.nan
    if "time_hours" in df.columns:
        df["time_s"] = df["time_s"].fillna(df["time_hours"] * 3600.0)
    for c in TIME_COLS:
        if c in df.columns:
            df["time_s"] = df["time_s"].fillna(df[c])

    if {"throughput_steps_per_s", "n_agents"} <= set(df.columns):
        derived = df["throughput_steps_per_s"] * df["n_agents"]
        df["transitions_per_s"] = (derived if "transitions_per_s" not in df.columns
                                   else df["transitions_per_s"].fillna(derived))
    return df


# Timing/rate columns that get mean/std/sem/min/max in results_summary.csv.
STAT_COLS = ("time_s", "run_time_s", "wall_time_s", "compile_time_s",
             "trace_time_s", "throughput_steps_per_s", "transitions_per_s")
# Cell identity carried through aggregation unchanged (constant within a cell).
ID_COLS = ("base_exp", "device", "n_agents", "num_envs", "total_timesteps",
           "env_steps")


def summarize_repeats(df, axis_cols, group_col: str = "method"):
    """Collapse the per-seed rows of a sweep to one row per cell.

    Returns a table keyed by ``group_col`` + the swept axes, carrying
    ``<col>_mean/_std/_sem/_min/_max`` for every timing column plus ``n_seeds``
    and the seed list. Standard deviations are the SAMPLE std (ddof=1), so a
    single-seed cell reports NaN rather than a misleading 0.
    """
    import pandas as pd  # noqa: F401  (groupby machinery)

    df = ensure_time_column(df)
    keys = [c for c in [group_col, *axis_cols]
            if c in df.columns and c is not None]
    if not keys:
        raise ValueError("summarize_repeats needs at least one grouping column")
    value_cols = [c for c in STAT_COLS if c in df.columns]
    g = df.groupby(keys, dropna=False)

    out = g[value_cols].agg(["mean", "std", "sem", "min", "max"])
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    out.insert(0, "n_seeds", g.size())
    if "seed" in df.columns:
        out.insert(1, "seeds", g["seed"].agg(
            lambda s: ",".join(str(int(v)) for v in sorted(s.dropna().unique()))))
    for c in ID_COLS:                       # identity, constant within a cell
        if c in df.columns and c not in keys:
            out[c] = g[c].first()
    return out.reset_index().sort_values(keys)


def _mean_std(sub, x: str, y: str):
    """Per-x mean, sample std (0 where undefined) and repeat count."""
    stat = sub.groupby(x)[y].agg(["mean", "std", "count"]).sort_index()
    return stat["mean"], stat["std"].fillna(0.0), stat["count"]


def _yerr(mean, std, loglog: bool):
    """Symmetric std whiskers, with the lower arm kept strictly positive so a
    log y-axis can render it (only bites when std >= mean, i.e. never for
    well-behaved timings)."""
    lo = np.minimum(std.values, mean.values * (1 - 1e-3)) if loglog else std.values
    return np.vstack([np.clip(lo, 0.0, None), std.values])


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
    Multi-seed cells additionally get +-1 sample-std whiskers, and the caption
    records how many seeds the mean is over.
    """
    import matplotlib.pyplot as plt
    apply_style()
    fig, ax = plt.subplots(figsize=(6, 4.2))
    n_series, seed_counts = 0, set()
    for label, sub, color in _series_iter(df, group_col):
        sub = sub.dropna(subset=[x, y])
        if sub.empty:
            continue
        mean, std, count = _mean_std(sub, x, y)
        seed_counts.update(int(c) for c in count.values)
        if (count > 1).any() and float(std.max()) > 0:
            ax.errorbar(mean.index, mean.values, yerr=_yerr(mean, std, loglog),
                        fmt="o-", color=color, label=label,
                        capsize=3, elinewidth=1, markersize=4)
        else:
            ax.plot(mean.index, mean.values, "o-", color=color, label=label)
        n_series += 1
    if loglog:
        ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(axis_label(x))
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    if seed_counts and max(seed_counts) > 1:
        reps = (f"{min(seed_counts)}-{max(seed_counts)}"
                if len(seed_counts) > 1 else f"{max(seed_counts)}")
        ax.annotate(f"mean $\\pm$ 1 s.d. over {reps} seeds",
                    xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7,
                    color="0.35")
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
    ax.set_ylabel("Speedup (reference time / ours)")
    ax.set_title(title or "Speedup vs reference implementation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


def _cell_fmt(v: float) -> str:
    if v < 100:
        return f"{v:.2g}"
    if v < 10_000:
        return f"{v:,.0f}"
    e = int(np.floor(np.log10(v)))
    return f"{v / 10**e:.1f}e{e}"


def plot_phase_diagram(df, path, x: str = "n_agents", y: str = "num_envs",
                       value: str = "time_s", ours: str = OURS,
                       cbar_label: str = "Steady-state run time (s)",
                       title: Optional[str] = None,
                       dpi: Optional[int] = None) -> bool:
    """Phase diagram of ``value`` (run time, throughput, ...) over the (x, y)
    mesh.

    Cells are the mean over repeats; axes are categorical (one row/column per
    swept value, log-spaced grids stay readable). Sequential colormap, log
    color scale, every cell direct-labeled with its value.

    ``dpi`` overrides the house 150 for print-resolution paper deliverables.
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
    ax.set_xlabel(axis_label(x))
    ax.set_ylabel(axis_label(y))
    ax.set_title(title or "Steady-state run time (s)")
    ax.grid(False)

    log_mid = (np.log(Z.min()) + np.log(Z.max())) / 2
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isfinite(v):
                continue
            dark_cell = np.log(v) < log_mid
            ax.text(j + 0.5, i + 0.5, _cell_fmt(v), ha="center", va="center",
                    fontsize=7, color="white" if dark_cell else "black")
    fig.colorbar(mesh, ax=ax, label=cbar_label)
    fig.tight_layout()
    fig.savefig(path, **({"dpi": dpi} if dpi else {}))
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
    ax.set_xlabel(f"Agents $n$ × environments $E$  "
                  f"(constant batch width $W = {product:g}$)")
    ax.set_ylabel("Steady-state run time (s)")
    ax.set_title(title or f"Agents/environments split at $n·E = {product:g}$")
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
                           ylabel="Env steps / s", title=f"Throughput vs {col}")
            paths.append(p)
        time_col = next((c for c in ("time_s",) + TIME_COLS if c in df.columns),
                        None)
        if time_col:
            p = str(out / f"walltime_vs_{col}.png")
            plot_metric_vs(df, col, time_col, p, group_col,
                           ylabel="Seconds", title=f"Wall time vs {col}")
            paths.append(p)
    return paths


def make_sweep_figures(df, axes, out_dir: str, figures: List[str],
                       tradeoff_product: Optional[int] = None,
                       group_col: str = "method") -> list:
    """Config-driven figure dispatch: the sweep YAML's ``figures`` list picks
    which graphs are rendered from the timing table.

    Kinds: "auto" (legacy per-axis throughput+walltime), "walltime",
    "throughput" (sequential env steps/s -- fixed by design in a pure
    agent-scaling sweep, so mostly a sanity check), "transitions"
    (agent-transitions/s, the throughput that actually scales),
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
                               ylabel="Steady-state run time (s)",
                               title=f"Run time vs {col}")
                paths.append(p)
        elif kind == "throughput":
            if "throughput_steps_per_s" not in df.columns:
                _skip(kind, "no throughput column"); continue
            for col in axis_cols:
                p = str(out / f"throughput_vs_{col}.png")
                plot_metric_vs(df, col, "throughput_steps_per_s", p, group_col,
                               ylabel="Sequential env steps / s",
                               title=f"Sequential throughput vs {col}")
                paths.append(p)
        elif kind == "transitions":
            if "transitions_per_s" not in df.columns:
                _skip(kind, "needs throughput_steps_per_s and n_agents"); continue
            for col in axis_cols:
                p = str(out / f"transitions_vs_{col}.png")
                plot_metric_vs(df, col, "transitions_per_s", p, group_col,
                               ylabel="Agent-transitions / s",
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
                continue
            if "transitions_per_s" in df.columns:
                p = str(out / "phase_throughput.png")
                if plot_phase_diagram(
                        df, p, value="transitions_per_s",
                        cbar_label="Agent-transitions / s",
                        title="Total throughput (agent-transitions/s)"):
                    paths.append(p)
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
