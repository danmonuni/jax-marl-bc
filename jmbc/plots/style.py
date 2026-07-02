"""Shared publication plotting style and helpers."""
from __future__ import annotations

import numpy as np

COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]


def apply_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.grid": True,
        "grid.color": "0.9",
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "legend.fontsize": 8,
        "legend.frameon": False,
        "lines.linewidth": 1.6,
    })


def smooth(x, w=50):
    x = np.asarray(x, dtype=float)
    if len(x) < w or w <= 1:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")
