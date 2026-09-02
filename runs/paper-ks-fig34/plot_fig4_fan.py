"""Figure 4, fallback form: the capital distribution through training drawn
from ``diagnostics.json`` instead of ``rollouts.npz``.

The heatmap version (``plot_fig34.py``) needs every agent-step of every
snapshot. When only the run's diagnostics survive, what is left per snapshot is
a five-number summary of the cross-section -- mean, std, median, p10, p90 --
which is enough for a fan chart: the median through training with the p10-p90
band around it. Less of the distribution than the heatmap shows, but it is the
same quantity from the same run, and the shape of the argument (dispersion
opens up as the policy learns, then settles) survives.

    python runs/paper-ks-fig34/plot_fig4_fan.py                  # -> <run>/figures/4-fan.png
    python runs/paper-ks-fig34/plot_fig4_fan.py run=<dir> out_dir=<dir>

Prefer the heatmap whenever ``rollouts.npz`` exists. This exists because that
file is the one artefact the run does not ship, and it can be lost.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plot_fig34 import LABELS, RC, COLORS  # noqa: E402  same house style

#: Sequential env steps per update, from the protocol: rollout_len 200 x
#: num_envs 12 = 2400. Snapshots are recorded by update index, so this is what
#: turns them into the x axis figure 4 uses.
STEPS_PER_UPDATE = 200 * 12


def load_summary(path: Path) -> dict:
    """Per-snapshot capital summary out of diagnostics.json."""
    d = json.loads(path.read_text())
    snaps = d["snapshots"]
    cap = [s["distributional"]["capital"] for s in snaps]
    return {
        "updates": np.array([s["update_idx"] for s in snaps], float),
        "mean":    np.array([c["mean"] for c in cap]),
        "median":  np.array([c["median"] for c in cap]),
        "p10":     np.array([c["p10"] for c in cap]),
        "p90":     np.array([c["p90"] for c in cap]),
        "gini":    np.array([s["distributional"]["capital_gini"] for s in snaps]),
    }


def plot(s: dict, path: Path) -> None:
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    mpl.rcParams.update(RC)

    steps = s["updates"] * STEPS_PER_UPDATE / 1e6

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(steps, s["p10"], s["p90"], color=COLORS["hist"], alpha=0.45,
                    linewidth=0, label="p10-p90 of the cross-section")
    ax.plot(steps, s["median"], color=COLORS["bad"], lw=2.2, label="Median")
    # Mean over median, dashed: in a right-skewed cross-section the gap between
    # them IS the inequality the figure is about, so both are drawn.
    ax.plot(steps, s["mean"], color=COLORS["employed"], lw=2.0, ls="--",
            label="Mean")

    ax.set_xlim(steps[0], steps[-1])
    ax.set_ylim(bottom=0)
    ax.set_xlabel(LABELS["steps"], fontsize=14)
    ax.set_ylabel(LABELS["k_i"], fontsize=14)
    ax.set_title("Stationary capital distribution through training", fontsize=16)
    ax.legend(loc="lower right", fontsize=12, frameon=True, facecolor="white",
              framealpha=0.9, edgecolor="0.8")
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    opt = dict(a.split("=", 1) for a in argv if "=" in a)
    run = Path(opt.get("run", HERE / "results/ks/sigma_0.00_seed_8"))
    if not run.is_absolute():
        run = REPO_ROOT / run
    diag = run / "diagnostics.json"
    if not diag.exists():
        raise SystemExit(f"no diagnostics.json in {run}")

    out_dir = Path(opt.get("out_dir", run / "figures"))
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    s = load_summary(diag)
    p = out_dir / "4-fan.png"
    plot(s, p)
    print(f"source   {diag}  ({len(s['updates'])} snapshots)\n"
          f"trained  capital median {s['median'][-1]:.3f}  mean {s['mean'][-1]:.3f}  "
          f"Gini {s['gini'][-1]:.3f}\n"
          f"->       {p}")


if __name__ == "__main__":
    main()
