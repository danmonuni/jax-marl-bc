"""Standard training-health and diagnostics-over-snapshots figures."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

from .style import apply_style, smooth

# Metrics shown in the training-health grid (key -> title), if present.
_HEALTH_KEYS = [
    ("total_loss", "Total loss"),
    ("value_loss", "Value loss"),
    ("policy_loss", "Policy loss"),
    ("entropy", "Entropy"),
    ("approx_kl", "Approx KL"),
    ("clip_frac", "Clip fraction"),
    ("explained_variance", "Explained variance"),
    ("grad_norm", "Grad norm"),
    ("action_saturation", "Action saturation"),
    ("step_reward", "Step reward"),
    ("returned_episode_returns", "Episode return"),
]


def plot_training_health(metrics_np: Dict[str, np.ndarray], path: str) -> None:
    import matplotlib.pyplot as plt
    apply_style()
    keys = [(k, t) for k, t in _HEALTH_KEYS if k in metrics_np]
    n = len(keys)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, (k, title) in zip(axes, keys):
        y = np.asarray(metrics_np[k]).ravel()
        ax.plot(smooth(y, min(50, max(2, len(y) // 10))))
        ax.set_title(title)
        ax.set_xlabel("update")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _snap_steps(summary: dict, steps_per_update: int) -> np.ndarray:
    idxs = np.asarray(summary["snapshot_indices"])
    return (idxs + 1) * steps_per_update


def plot_economic_snapshots(summary: dict, env_kind: str, steps_per_update: int, path: str) -> None:
    """Euler error (+ KS forecasting quality) vs training steps."""
    import matplotlib.pyplot as plt
    apply_style()
    steps = _snap_steps(summary, steps_per_update)
    snaps: List[dict] = summary["snapshots"]

    euler = [s.get("economic", {}).get("euler", {}).get("euler_mean_abs", np.nan) for s in snaps]
    is_ks = env_kind == "ks"
    ncol = 3 if is_ks else 1
    fig, axes = plt.subplots(1, ncol, figsize=(4.6 * ncol, 3.4), squeeze=False)
    axes = axes[0]

    axes[0].plot(steps, euler, "o-", color="#d62728")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training steps")
    axes[0].set_ylabel("mean |Euler residual|")
    axes[0].set_title("Euler-equation accuracy")

    if is_ks:
        r2 = [s.get("economic", {}).get("ks_forecast", {}).get("ks_lom_r2", np.nan) for s in snaps]
        dh = [s.get("economic", {}).get("ks_forecast", {}).get("den_haan_max_pct", np.nan) for s in snaps]
        axes[1].plot(steps, r2, "s-", color="#1f77b4")
        axes[1].set_xscale("log")
        axes[1].set_xlabel("training steps")
        axes[1].set_ylabel(r"law-of-motion $R^2$")
        axes[1].set_title("KS forecast fit")
        axes[2].plot(steps, dh, "^-", color="#2ca02c")
        axes[2].set_xscale("log")
        axes[2].set_xlabel("training steps")
        axes[2].set_ylabel("Den Haan max error (%)")
        axes[2].set_title("KS dynamic-forecast error")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_distributional_snapshots(summary: dict, steps_per_update: int, path: str) -> None:
    """Gini and top wealth shares vs training steps."""
    import matplotlib.pyplot as plt
    apply_style()
    steps = _snap_steps(summary, steps_per_update)
    snaps = summary["snapshots"]

    gini = [s.get("distributional", {}).get("capital_gini", np.nan) for s in snaps]
    top1 = [s.get("distributional", {}).get("top_0.01_share", np.nan) for s in snaps]
    top10 = [s.get("distributional", {}).get("top_0.1_share", np.nan) for s in snaps]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    axes[0].plot(steps, gini, "o-", color="#9467bd")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("training steps")
    axes[0].set_ylabel("capital Gini")
    axes[0].set_title("Inequality over training")

    axes[1].plot(steps, top1, "s-", label="top 1%")
    axes[1].plot(steps, top10, "^-", label="top 10%")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("training steps")
    axes[1].set_ylabel("wealth share")
    axes[1].set_title("Top wealth shares")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
