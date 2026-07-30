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


def _fmt_steps(s: float) -> str:
    return f"{s:.1e}".replace("e+0", "e").replace("e+", "e")


def _interior_mask(rec, burn_frac: float):
    """Stationary-slice mask excluding auto-reset steps."""
    done = stationary_slice(rec["done"], burn_frac).astype(bool)
    return ~done


# ── Wealth distribution through training (heatmap) ───────────────────────────

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

    plot_ks_wealth_heatmap(recs, snap_steps, str(fig_dir / "ks_wealth_heatmap.png"),
                           burn_frac)

    # fig4 wants exactly 4 snapshots for its 2x2 LoM grid.
    S = len(recs)
    sel = sorted(set(np.linspace(0, S - 1, 4).round().astype(int))) if S > 4 else range(S)
    plot_ks_fig4([recs[i] for i in sel], [i for i in sel],
                 snap_steps[list(sel)], str(fig_dir / "ks_fig4.png"), burn_frac)
