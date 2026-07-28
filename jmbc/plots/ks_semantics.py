"""Krusell-Smith semantic figures: how the trained economy evolves over training.

All functions are pure numpy + matplotlib and consume the raw snapshot rollouts
(``recs``: list of channel dicts, one per training snapshot) plus the snapshot
-> env-step mapping, so they can be regenerated ex post from ``rollouts.npz``
via ``python -m jmbc.analyze`` without JAX or the trained parameters.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

import numpy as np

from .style import apply_style
from ..diagnostics.distributional import stationary_slice
from ..diagnostics.economic import ks_forecast_rule, den_haan_stat


def _fmt_steps(s: float) -> str:
    return f"{s:.1e}".replace("e+0", "e").replace("e+", "e")


def _interior_mask(rec, burn_frac: float):
    """Stationary-slice mask excluding auto-reset steps."""
    done = stationary_slice(rec["done"], burn_frac).astype(bool)
    return ~done


# ── 1. Aggregate law of motion: evolution over training ──────────────────────

def plot_ks_lom_evolution(recs: List[dict], snap_steps: Sequence[float], path: str,
                          burn_frac: float = 0.5):
    """Left: fitted K_{t+1} = a_s + b_s K_t lines per snapshot (colour = training
    progress; solid = good state, dashed = bad). Middle: one-step R^2 and the
    Den Haan dynamic-forecast error vs training steps. Right: fitted (a, b)."""
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from matplotlib import colors as mcolors
    apply_style()

    S = len(recs)
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(0, max(S - 1, 1))

    fits, r2s, dh_mean, dh_max = [], [], [], []
    for rec in recs:
        rules, r2 = ks_forecast_rule(rec, burn_frac)
        dh = den_haan_stat(rec, burn_frac)
        fits.append(rules)
        r2s.append(r2)
        dh_mean.append(dh["den_haan_mean_pct"])
        dh_max.append(dh["den_haan_max_pct"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))

    ax = axes[0]
    K_all = np.concatenate(
        [stationary_slice(np.asarray(r["K"]), burn_frac).ravel() for r in recs])
    lo, hi = np.percentile(K_all, 1), np.percentile(K_all, 99)
    xr = np.linspace(lo, hi, 50)
    # Faint scatter of the final snapshot as ground truth.
    Kf = stationary_slice(np.asarray(recs[-1]["K"]), burn_frac).ravel()
    ax.scatter(Kf[:-1], Kf[1:], s=2, alpha=0.12, color="0.55", zorder=1,
               label="final rollout")
    for s, rules in enumerate(fits):
        col = cmap(norm(s))
        ax.plot(xr, rules[1][0] + rules[1][1] * xr, color=col, lw=1.4, zorder=2)
        ax.plot(xr, rules[0][0] + rules[0][1] * xr, color=col, lw=1.4, ls="--", zorder=2)
    ax.plot(xr, xr, color="k", lw=0.8, ls=":", alpha=0.6, label="45°")
    sm = cm.ScalarMappable(norm=mcolors.Normalize(snap_steps[0], snap_steps[-1]), cmap=cmap)
    fig.colorbar(sm, ax=ax, label="training env steps")
    ax.set_xlabel("$K_t$"); ax.set_ylabel("$K_{t+1}$")
    ax.set_title("Aggregate law of motion (solid good / dashed bad)")
    ax.legend(loc="upper left")

    ax = axes[1]
    ax.plot(snap_steps, r2s, marker="o", color="tab:blue", label="one-step $R^2$")
    ax.set_ylim(min(0.0, np.nanmin(r2s)) - 0.02, 1.02)
    ax.set_xscale("log")
    ax.set_xlabel("Training env steps"); ax.set_ylabel("$R^2$", color="tab:blue")
    ax2 = ax.twinx()
    ax2.plot(snap_steps, dh_mean, marker="s", color="tab:red", label="Den Haan mean %")
    ax2.plot(snap_steps, dh_max, marker="^", color="tab:red", ls="--", alpha=0.6,
             label="Den Haan max %")
    ax2.set_yscale("log"); ax2.set_ylabel("dynamic forecast error (%)", color="tab:red")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_title("Forecasting quality of the LoM")

    ax = axes[2]
    for state, ls, lab in ((1, "-", "good"), (0, "--", "bad")):
        ax.plot(snap_steps, [f[state][1] for f in fits], ls=ls, marker="o",
                color="tab:green", label=f"slope $b$ ({lab})")
        ax.plot(snap_steps, [f[state][0] for f in fits], ls=ls, marker="o",
                color="tab:purple", label=f"intercept $a$ ({lab})")
    ax.set_xscale("log")
    ax.set_xlabel("Training env steps"); ax.set_ylabel("coefficient")
    ax.set_title("LoM coefficients $K' = a_s + b_s K$")
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── 2. Wealth distribution through training (heatmap) ────────────────────────

def plot_ks_wealth_heatmap(recs: List[dict], snap_steps: Sequence[float], path: str,
                           burn_frac: float = 0.5, n_bins: int = 60):
    """x = training progress (snapshots), y = wealth, colour = stationary density.

    Each column is the stationary (post burn-in, reset steps excluded) cross-
    sectional wealth histogram of the policy at that training snapshot.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    apply_style()

    ws = []
    for rec in recs:
        w = stationary_slice(rec["wealths"], burn_frac)
        keep = _interior_mask(rec, burn_frac)
        ws.append(w[keep].ravel())
    all_w = np.concatenate(ws)
    lo, hi = np.percentile(all_w, 0.5), np.percentile(all_w, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, n_bins + 1)
    dens = np.stack([np.histogram(w, bins=bins, density=True)[0] for w in ws], axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    x_edges = np.arange(len(recs) + 1)
    floor = max(dens[dens > 0].min() if (dens > 0).any() else 1e-6, 1e-6)
    mesh = ax.pcolormesh(x_edges, bins, dens, cmap="magma",
                         norm=mcolors.LogNorm(vmin=floor, vmax=dens.max() + 1e-12))
    means = [w.mean() for w in ws]
    ax.plot(np.arange(len(recs)) + 0.5, means, color="cyan", marker="o",
            markersize=3.5, lw=1.2, label="mean wealth")
    ax.set_xticks(np.arange(len(recs)) + 0.5)
    ax.set_xticklabels([_fmt_steps(s) for s in snap_steps], rotation=45, fontsize=7)
    ax.set_xlabel("Training env steps (snapshots)")
    ax.set_ylabel("Wealth (cash-on-hand)")
    ax.set_title("Stationary wealth distribution through training")
    ax.legend(loc="upper left")
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="density (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Shared driver: everything the KS experiment (or jmbc.analyze) renders ────

def render_ks_figures(recs: List[dict], snap_steps: Sequence[float],
                      fig_dir, burn_frac: float = 0.5):
    """Render the full KS figure set into ``fig_dir``. Numpy-only."""
    from .figures import plot_ks_fig4

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    snap_steps = np.asarray(snap_steps, float)

    plot_ks_lom_evolution(recs, snap_steps, str(fig_dir / "ks_lom_evolution.png"),
                          burn_frac)
    plot_ks_wealth_heatmap(recs, snap_steps, str(fig_dir / "ks_wealth_heatmap.png"),
                           burn_frac)

    # fig4 wants exactly 4 snapshots for its 2x2 LoM grid.
    S = len(recs)
    sel = sorted(set(np.linspace(0, S - 1, 4).round().astype(int))) if S > 4 else range(S)
    plot_ks_fig4([recs[i] for i in sel], [i for i in sel],
                 snap_steps[list(sel)], str(fig_dir / "ks_fig4.png"), burn_frac)
