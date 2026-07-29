"""Paper figures 3 and 4, from this experiment's record. No training, no JAX.

Reads one run directory's ``rollouts.npz`` (every recorded snapshot, all
steps, all agents) and writes ``3.png`` and ``4.png``.

    python runs/ks-fig34-rerun/plot_fig34.py                    # -> <run>/figures/
    python runs/ks-fig34-rerun/plot_fig34.py out_dir=runs/final-paper-runs
    python runs/ks-fig34-rerun/plot_fig34.py run=<dir> window=50

What differs from the previous generation (``runs/final-paper-runs/ref/``):

  * **Cross-sectional histograms are time-averaged over the last WINDOW=50
    steps** of the evaluation instead of pooling every agent-step of the
    stationary half. A point is now one agent -- its mean capital (fig. 3) or
    wealth (fig. 4) over the last 50 simulated periods -- so the distribution
    is the cross-section of a settled economy rather than the cross-section
    convolved with each agent's own time variation. This is figure 6's
    ``steady_state_capital`` convention, narrowed to a window.
  * **Figure 4's columns are evenly spread over training** (linearly spaced
    snapshots on a linear step axis) instead of 12 log-spaced ones crowded
    into the first decade.
  * Figure 3's law-of-motion panels fit **one line per aggregate state**
    (good solid orange / bad dashed blue, one R^2 each) rather than a single
    pooled line, as in ``ref/3.png``.

Everything else -- panels, quantities, colours, sizes -- is as before.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent          # runs/ks-fig34-rerun/
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))          # jmbc isn't pip-installed here

from jmbc.diagnostics.distributional import gini, stationary_slice  # noqa: E402
from jmbc.plots.style import apply_style                            # noqa: E402

#: The averaging window for every cross-sectional histogram, in simulated
#: periods counted back from the end of each evaluation rollout.
WINDOW = 50

#: Histogram resolution. The pooled histograms of ref/*.png had ~10^5 samples
#: and could afford 40-60 bins; a time-averaged column has n_agents (=200)
#: samples, so both are binned coarser -- fig. 4 especially, where 60 bins put
#: ~3 agents in a bin and the heatmap turned to speckle.
N_BINS_FIG3 = 30
N_BINS_FIG4 = 24

BURN_FRAC = 0.5      # for the panels that still pool the stationary half

#: Channels figure 3's consumption-policy panels need, per snapshot.
POLICY_CHANNELS = ("wealths", "c_fracs", "emp_states", "done")


# ── data ──────────────────────────────────────────────────────────────────────
def find_run(explicit: str | None) -> Path:
    """The run directory to plot: an explicit ``run=`` path, else the single
    thing under ``results/`` that has a rollouts.npz."""
    if explicit:
        run = Path(explicit)
        if not run.is_absolute() and (REPO_ROOT / run).exists():
            run = REPO_ROOT / run
        if not (run / "rollouts.npz").exists():
            raise SystemExit(f"no rollouts.npz in {run}")
        return run

    results = HERE / "results"
    cands = sorted(p.parent for p in results.rglob("rollouts.npz"))
    if not cands:
        raise SystemExit(
            f"no run found under {results}/. Run rerun_fig34.py first, or "
            "pass run=<path> (e.g. a directory pulled back from Drive).")
    if len(cands) > 1:
        raise SystemExit("more than one run under results/, pass run=<path>:\n  "
                         + "\n  ".join(str(c) for c in cands))
    return cands[0]


def load_record(path: Path, window: int, panels: int = 4) -> dict:
    """Pull only what the two figures need out of rollouts.npz.

    ``jmbc.recorder.load_rollouts`` materialises every channel of every
    snapshot at once; here each channel is decompressed, reduced to the
    window mean (or the two snapshots the policy panels need), and dropped —
    so this stays cheap if SIM_STEPS or the snapshot count is ever raised.
    """
    with np.load(path) as z:
        env_steps = z["snap_env_steps"].astype(float)
        S = len(env_steps)
        K, agg = z["K"], z["agg_state"]              # [S, T] aggregates, small
        T = K.shape[1]
        if T < window:
            raise SystemExit(f"eval rollouts are {T} steps, shorter than the "
                             f"{window}-step averaging window")
        # Figure 3's four stages: log-spaced over training, as in ref/3.png,
        # and starting at snapshot 0. The record is linearly spaced, so a
        # plain log pick would start at the second snapshot and every panel
        # would show an already-converged law of motion -- the untrained end
        # of the comparison is the whole point of the panel.
        sel = np.unique(np.round(
            np.logspace(0, np.log10(S), panels)).astype(int).clip(1, S) - 1)

        done_win = z["done"][:, -window:]             # [S, w]
        out = {
            "env_steps": env_steps, "sel": sel, "n_snapshots": S, "n_steps": T,
            "K": K, "agg_state": agg,
            # per-agent time averages over the window, per snapshot: [S, n]
            "k_bar": _window_mean(z["ks"][:, -window:], done_win),
            "w_bar": _window_mean(z["wealths"][:, -window:], done_win),
        }
        # Untrained (first) and trained (last) snapshots, in full, for the
        # consumption-policy scatters.
        out["policy"] = {s: {c: z[c][s] for c in POLICY_CHANNELS}
                         for s in (0, S - 1)}
    out["n_agents"] = out["k_bar"].shape[-1]
    return out


def _window_mean(a: np.ndarray, done: np.ndarray) -> np.ndarray:
    """Per-agent mean over the window, auto-reset steps excluded. [S,w,n]->[S,n]"""
    keep = ~done.astype(bool)
    if keep.all():
        return a.mean(axis=1)
    return np.stack([x[m].mean(axis=0) if m.any() else x.mean(axis=0)
                     for x, m in zip(a, keep)])


def fmt_steps(s: float) -> str:
    return f"{s:.1e}".replace("e+0", "e").replace("e+", "e")


# ── figure 3 ──────────────────────────────────────────────────────────────────
def plot_fig3(d: dict, window: int, path: Path) -> None:
    """Law-of-motion scatters (4 training stages), time-averaged capital
    histograms, consumption-policy scatters -- untrained vs trained."""
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import scipy.stats as sp_stats
    apply_style()

    fig = plt.figure(figsize=(18, 7))
    gs_outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    # A -- aggregate law of motion, one fit per aggregate state.
    gs_left = gs_outer[0].subgridspec(2, 2, hspace=0.55, wspace=0.45)
    for i, s in enumerate(d["sel"][:4]):
        ax = fig.add_subplot(gs_left[i // 2, i % 2])
        K = stationary_slice(d["K"][s], BURN_FRAC)
        agg = stationary_slice(d["agg_state"][s], BURN_FRAC)
        Kt, Kt1, agt = K[:-1], K[1:], agg[:-1]
        for state, colour, ls, lab in ((1, "tab:orange", "-", "good"),
                                       (0, "tab:blue", "--", "bad")):
            m = agt == state
            ax.scatter(Kt[m], Kt1[m], s=4, alpha=0.4, color=colour, label=lab)
            if m.sum() < 3:
                continue
            slope, intercept, r, *_ = sp_stats.linregress(Kt[m], Kt1[m])
            x_r = np.linspace(Kt[m].min(), Kt[m].max(), 100)
            ax.plot(x_r, slope * x_r + intercept, color=colour, ls=ls, lw=1.2,
                    label=f"{lab} $R^2 = {r ** 2:.3f}$")
        ax.set_xlabel("$K_t$", fontsize=8)
        ax.set_ylabel("$K_{t+1}$", fontsize=8)
        ax.set_title(f"Step $\\approx {d['env_steps'][s]:.0e}$", fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # B -- cross-sectional capital, time-averaged over the last `window` steps.
    gs_mid = gs_outer[1].subgridspec(2, 1, hspace=0.55)
    for ax, s, label in zip(
        [fig.add_subplot(gs_mid[0]), fig.add_subplot(gs_mid[1])],
        [0, d["n_snapshots"] - 1], ["Untrained", "Trained"],
    ):
        k_bar = d["k_bar"][s]
        ax.hist(k_bar, bins=N_BINS_FIG3, density=True, color="steelblue",
                alpha=0.75)
        ax.set_title(f"{label}  (Gini={gini(k_bar):.3f})", fontsize=9)
        ax.set_xlabel(f"Capital $k^i$ (mean of last {window} steps)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    # C -- consumption policy, unchanged: every agent-step of the stationary
    # half, so the policy function itself is traced out rather than averaged.
    gs_right = gs_outer[2].subgridspec(2, 1, hspace=0.55)
    for ax, s, label in zip(
        [fig.add_subplot(gs_right[0]), fig.add_subplot(gs_right[1])],
        [0, d["n_snapshots"] - 1], ["Untrained", "Trained"],
    ):
        rec = d["policy"][s]
        # emp_states[t] is recorded from the post-step state, i.e. the draw for
        # period t+1; the employment that entered the policy's observation for
        # (wealths[t], c_fracs[t]) is emp_states[t-1]. Colour by the lag.
        W = stationary_slice(rec["wealths"][1:], BURN_FRAC)
        CF = stationary_slice(rec["c_fracs"][1:], BURN_FRAC)
        emp = stationary_slice(rec["emp_states"][:-1], BURN_FRAC)
        keep = ~stationary_slice(rec["done"][1:], BURN_FRAC).astype(bool)
        W, CF, emp = W[keep].ravel(), CF[keep].ravel(), emp[keep].ravel()
        ax.scatter(W[emp == 1], CF[emp == 1], s=1, alpha=0.3,
                   color="tab:orange", label="employed")
        ax.scatter(W[emp == 0], CF[emp == 0], s=1, alpha=0.3,
                   color="tab:blue", label="unemployed")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Wealth $a^i$", fontsize=8)
        ax.set_ylabel("Cons. frac. $\\hat{c}^i$", fontsize=8)
        ax.legend(fontsize=7, markerscale=4)
        ax.tick_params(labelsize=7)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── figure 4 ──────────────────────────────────────────────────────────────────
def plot_fig4(d: dict, window: int, path: Path) -> None:
    """x = training env steps, y = wealth, colour = density of the per-agent
    wealth averaged over the last `window` steps."""
    import matplotlib.pyplot as plt
    from matplotlib import colors as mcolors
    apply_style()

    ws = d["w_bar"]                                   # [S, n]
    steps = d["env_steps"]
    lo, hi = np.percentile(ws, 0.5), np.percentile(ws, 99.5)
    if hi <= lo:
        hi = lo + 1e-6
    bins = np.linspace(lo, hi, N_BINS_FIG4 + 1)
    dens = np.stack([np.histogram(w, bins=bins, density=True)[0] for w in ws],
                    axis=1)                            # [n_bins, S]

    # Column edges: midway between consecutive snapshots, the end ones
    # extended by half a gap. Snapshots are evenly spaced over training, so
    # the axis is the actual step count, linear -- no categorical columns.
    mids = 0.5 * (steps[:-1] + steps[1:])
    x_edges = np.concatenate([[2 * steps[0] - mids[0]], mids,
                              [2 * steps[-1] - mids[-1]]])

    fig, ax = plt.subplots(figsize=(9, 5))
    floor = max(dens[dens > 0].min() if (dens > 0).any() else 1e-6, 1e-6)
    mesh = ax.pcolormesh(x_edges, bins, dens, cmap="magma",
                         norm=mcolors.LogNorm(vmin=floor,
                                              vmax=dens.max() + 1e-12))
    ax.plot(steps, ws.mean(axis=1), color="cyan", marker="o", markersize=3.5,
            lw=1.2, label="mean wealth")
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_xlabel("Training env steps")
    ax.set_ylabel(f"Wealth (cash-on-hand, mean of last {window} steps)")
    ax.set_title("Stationary wealth distribution through training")
    ax.legend(loc="upper left")
    ax.grid(False)
    fig.colorbar(mesh, ax=ax, label="density (log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── driver ────────────────────────────────────────────────────────────────────
def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    opt = dict(a.split("=", 1) for a in argv if "=" in a)
    run = find_run(opt.get("run"))
    window = int(opt.get("window", WINDOW))
    # Default: beside the record it came from. Pass out_dir=runs/final-paper-runs
    # to write the paper's copies instead.
    out_dir = Path(opt.get("out_dir", run / "figures"))
    if not out_dir.is_absolute():
        out_dir = (REPO_ROOT / out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    d = load_record(run / "rollouts.npz", window)
    plot_fig3(d, window, out_dir / "3.png")
    plot_fig4(d, window, out_dir / "4.png")

    k_bar = d["k_bar"][-1]
    print(f"figures -> {out_dir}/3.png, {out_dir}/4.png\n"
          f"  source   {run}  ({d['n_snapshots']} updates x {d['n_steps']} "
          f"steps x {d['n_agents']} agents)\n"
          f"  window   last {window} steps, per-agent time average\n"
          f"  fig 3    stages at env steps "
          f"{', '.join(fmt_steps(d['env_steps'][s]) for s in d['sel'])}\n"
          f"  trained  capital Gini={gini(k_bar):.3f}  K_mean={k_bar.mean():.3f}")


if __name__ == "__main__":
    main()
