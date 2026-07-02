"""Orchestrate diagnostics across training snapshots.

Simulates the trained policy at log-spaced snapshots of ``params_history`` and
runs the economic and distributional probes at each, so we can show quantities
(Euler error, R^2, Den Haan stat, Gini) *improving over training*.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import jax
import numpy as np

from .rollout import simulate
from .economic import economic_report
from .distributional import distributional_report


def snapshot_indices(num_updates: int, n_snapshots: int) -> np.ndarray:
    """Log-spaced, unique update indices in [0, num_updates-1] (last included)."""
    if num_updates <= 1:
        return np.array([0])
    idxs = np.unique(
        np.round(np.logspace(0, np.log10(num_updates - 1), n_snapshots))
        .astype(int)
        .clip(0, num_updates - 1)
    )
    if idxs[-1] != num_updates - 1:
        idxs = np.append(idxs, num_updates - 1)
    return idxs


def params_at(params_history, idx: int):
    """Index a pytree carrying a leading NUM_UPDATES dimension."""
    return jax.tree.map(lambda x: x[idx], params_history)


def compute_diagnostics(
    env, env_cfg, net, params_history, diag_cfg, seed: int = 7,
) -> Tuple[Dict, List[dict], np.ndarray]:
    """Return (summary, recs, idxs).

    summary  : JSON-serializable dict with a per-snapshot list and a 'final'.
    recs     : simulated rollouts at each snapshot (for bespoke figures).
    idxs     : the snapshot update indices.
    """
    num_updates = int(jax.tree_util.tree_leaves(params_history)[0].shape[0])
    idxs = snapshot_indices(num_updates, diag_cfg.n_snapshots)
    key = jax.random.PRNGKey(seed)

    recs: List[dict] = []
    snapshots: List[dict] = []
    for idx in idxs:
        rec = simulate(env, net, params_at(params_history, int(idx)), key, diag_cfg.sim_steps)
        recs.append(rec)
        snap: Dict[str, object] = {"update_idx": int(idx)}
        if diag_cfg.economic:
            snap["economic"] = economic_report(rec, env_cfg, diag_cfg.burn_frac)
        if diag_cfg.distributional:
            snap["distributional"] = distributional_report(rec, diag_cfg.burn_frac)
        snapshots.append(snap)

    summary = {
        "snapshot_indices": [int(i) for i in idxs],
        "num_updates": num_updates,
        "snapshots": snapshots,
        "final": snapshots[-1],
    }
    return summary, recs, idxs


def metrics_to_numpy(metrics: Dict) -> Dict[str, np.ndarray]:
    """Reduce per-update training metrics to 1-D numpy series for CSV/plots."""
    out = {}
    for k, v in metrics.items():
        a = np.asarray(v)
        while a.ndim > 1:
            a = a.mean(axis=-1)
        out[k] = a
    return out
