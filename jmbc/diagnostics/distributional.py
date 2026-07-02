"""Distributional / inequality diagnostics."""
from __future__ import annotations

from typing import Dict

import numpy as np


def stationary_slice(arr, burn_frac=0.5):
    """Drop the first ``burn_frac`` of a [T, ...] array as burn-in."""
    arr = np.asarray(arr)
    cut = int(arr.shape[0] * burn_frac)
    return arr[cut:]


def gini(values) -> float:
    v = np.sort(np.abs(np.asarray(values).flatten()))
    n = len(v)
    if n == 0:
        return 0.0
    s = v.sum() + 1e-12
    return float((2 * np.dot(np.arange(1, n + 1), v) / (n * s)) - (n + 1) / n)


def lorenz(values):
    v = np.sort(np.abs(np.asarray(values).flatten()))
    if len(v) == 0:
        return np.array([0, 1]), np.array([0, 1])
    return (
        np.r_[0, np.arange(1, len(v) + 1) / len(v)],
        np.r_[0, np.cumsum(v) / (v.sum() + 1e-12)],
    )


def mpc_curve(w, cf, n_bins=40):
    """Median consumption fraction by wealth bin (a proxy MPC curve)."""
    w = np.asarray(w).flatten()
    cf = np.asarray(cf).flatten()
    if len(w) == 0:
        return np.array([]), np.array([])
    lo, hi = np.percentile(w, 2), np.percentile(w, 98)
    if lo == hi:
        hi += 1e-5
    bins = np.linspace(lo, hi, n_bins + 1)
    cx = 0.5 * (bins[:-1] + bins[1:])
    cy = []
    for i in range(n_bins):
        mask = (w >= bins[i]) & (w < bins[i + 1])
        cy.append(np.median(cf[mask]) if np.sum(mask) > 0 else np.nan)
    return cx, np.array(cy)


def top_shares(values, quantiles=(0.01, 0.1)) -> Dict[str, float]:
    """Share of total wealth held by the top q fraction of agents."""
    v = np.sort(np.abs(np.asarray(values).flatten()))[::-1]  # descending
    n = len(v)
    total = v.sum() + 1e-12
    out = {}
    for q in quantiles:
        k = max(1, int(round(q * n)))
        out[f"top_{q:g}_share"] = float(v[:k].sum() / total)
    return out


def dist_summary(values) -> Dict[str, float]:
    """Mean / std / median / p10 / p90 of a distribution."""
    v = np.asarray(values).flatten()
    if len(v) == 0:
        return {}
    return {
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "p10": float(np.percentile(v, 10)),
        "p90": float(np.percentile(v, 90)),
    }


def distributional_report(rec, burn_frac=0.5) -> Dict[str, object]:
    """Full distributional diagnostic bundle on the stationary capital slice."""
    ks = stationary_slice(rec["ks"], burn_frac).flatten()
    out = {
        "capital_gini": gini(ks),
        "capital": dist_summary(ks),
    }
    out.update(top_shares(ks))
    if "wealths" in rec:
        out["wealth_gini"] = gini(stationary_slice(rec["wealths"], burn_frac).flatten())
    return out
