"""Lognormal kappa sweep, rerun on top of RANDOMIZED starting capital.

Third iteration of ``runs/ks-wealth-lognormal-random/`` (v2). The economic
design is unchanged and deliberately so -- kappa ~ LogNormal(mu, sigma) with
mean fixed to 1 via mu = -sigma^2/2, drawn as genuine i.i.d. samples
(``np.random.default_rng(seed)``), sigma swept 0 (homogeneous) to 1 (wide
spread), a list of seeds at every sigma so the spread across draws is
visible. Every (sigma, seed) cell is trained and kept.

What is NEW is underneath: agents no longer all start at the same capital.
Until 2026-07 every simulation began with k_i = k_init = 1.0 for all i. The
environment now draws per-agent starting capital, independently per parallel
env, and (by default) redraws it at every episode reset -- the initialization
the reference KS implementation used. This sweep is the first full-scale test
of that change, which is why it is a new experiment rather than an edit to
v2: its cells are NOT comparable to v2's, since the initial condition
differs.

Starting capital here is U(3, 20) per agent. Those bounds are the reference
implementation's U(10, 70) rescaled to this repo's calibration: the reference
spans 0.25x-1.75x of ITS steady state (K* ~ 40 at beta=0.99), and at the
beta=0.95 used here K* ~ 11.7. Note this starts the economy NEAR its steady
state, whereas the old constant k_init=1.0 started it at ~0.085x -- so the
burn-in has less transient to absorb, not more.

Usage (same config.yaml + CLI-dotlist convention as the sibling experiments):
    python sweep_lognormal_random_3.py
    python sweep_lognormal_random_3.py n_agents=100 device=cpu sigmas=[0.0,0.5] seeds=[0,1]
    python sweep_lognormal_random_3.py k_init_dist=constant   # v2's initialization
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # jmbc isn't pip-installed in this repo yet

DEFAULT_CONFIG_PATH = HERE / "config.yaml"


@dataclass
class SweepConfig:
    base_exp: str = "ks_n200"      # configs/exp/<base_exp>.yaml -- same protocol
                                    # as the sibling calibration scripts, only
                                    # n_agents/num_envs are overridden below
    n_agents: int = 500
    num_envs: Optional[int] = 8    # W = num_envs * n_agents = 4,000 at n=500,
                                    # inside the 4k-20k compute-plateau the
                                    # paper's own scaling mesh measured (see
                                    # sibling scripts' "Batch width" note)
    sigmas: List[float] = field(
        default_factory=lambda: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    sim_steps: int = 5000
    device: str = "gpu"
    total_timesteps: Optional[int] = None   # None -> base_exp's own budget
    out_dir: str = "results"
    save_raw: str = "all"          # every (sigma, seed) cell's raw rollout +
                                    # trained params is kept, not just some
                                    # subset -- see module docstring
    # Starting-capital initialization -- the point of this third iteration.
    # base_exp (ks_n200 -> ks) pins k_init_dist: constant to keep the older
    # runs reproducible, so these are overridden explicitly per cell.
    k_init_dist: str = "uniform"       # "constant" | "uniform" | "lognormal"
    k_init_low: float = 3.0            # U bounds: the reference U(10, 70)
    k_init_high: float = 20.0          # rescaled to K* ~ 11.7 at beta=0.95
    k_init_resample: str = "per_episode"   # "per_episode" | "per_env"


def load_sweep_config(argv: Optional[List[str]] = None) -> SweepConfig:
    """structured defaults < config.yaml (or ``config=<path>``) < CLI dotlist
    -- identical merge order to jmbc.config.load_config."""
    argv = sys.argv[1:] if argv is None else argv
    dotlist = [a for a in argv if "=" in a and not a.startswith("config=")]
    config_arg = next((a.split("=", 1)[1] for a in argv if a.startswith("config=")), None)
    config_path = Path(config_arg) if config_arg else DEFAULT_CONFIG_PATH

    cfg = OmegaConf.structured(SweepConfig)
    if config_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def build_kappas(n_agents: int, sigma: float, seed: int) -> np.ndarray:
    """I.i.d. LogNormal(mu, sigma) draw, mean renormalized to 1.

    Unlike the quantile-mesh sibling (``ks-wealth-calibration``), kappas here
    are genuine independent draws -- ``np.random.default_rng(seed)`` -- so
    the realized cross-sectional shape varies with ``seed`` at fixed sigma,
    not just with sigma itself. ``mu = -sigma^2/2`` keeps E[LogNormal] = 1 in
    the continuum limit; renormalized by the empirical mean afterwards for
    the same reason as the sibling script (the finite-n draw's mean deviates
    from that analytic mean, more so at high sigma and small n_agents).
    ``sigma <= 0`` is still the exact homogeneous baseline (no seed
    dependence -- there is nothing to sample).
    """
    if sigma <= 0:
        return np.ones(n_agents, np.float64)
    rng = np.random.default_rng(seed)
    mu = -0.5 * sigma ** 2
    kappas = rng.lognormal(mean=mu, sigma=sigma, size=n_agents)
    return kappas / kappas.mean()


def run_cell(kappas: np.ndarray, seed: int, sweep: SweepConfig):
    """Train one KS run at this kappa vector; return (rec, cfg, params, network)."""
    from jmbc.config import load_config, to_train_dict, setup_device

    n_agents = len(kappas)
    overrides = [
        f"env.n_agents={n_agents}",
        "diag.n_snapshots=1",
        f"diag.sim_steps={sweep.sim_steps}",
        f"run.seed={seed}",
        f"run.device={sweep.device}",
        # Starting capital: override base_exp's pinned `constant` (see README).
        f"env.k_init_dist={sweep.k_init_dist}",
        f"env.k_init_low={sweep.k_init_low}",
        f"env.k_init_high={sweep.k_init_high}",
        f"env.k_init_resample={sweep.k_init_resample}",
    ]
    if sweep.total_timesteps:
        overrides.append(f"train.total_timesteps={sweep.total_timesteps}")
    if sweep.num_envs:
        overrides.append(f"train.num_envs={sweep.num_envs}")
    cfg = load_config(sweep.base_exp, overrides)
    cfg.env.kappas = kappas.tolist()
    setup_device(cfg.run.device, bool(cfg.run.prealloc))  # before jax import

    import jax
    from jmbc.envs import build_env
    from jmbc.algos import make_train
    from jmbc.diagnostics import simulate
    from jmbc.experiments.common import _print_launch_summary
    from jmbc.recorder import phase

    env = build_env(cfg.env)
    train_fn = make_train(env, to_train_dict(cfg))
    _print_launch_summary(cfg, env, train_fn, int(cfg.run.seed))

    rng = jax.random.PRNGKey(int(seed))
    phase("training ...")
    out = train_fn(rng)
    phase("training done -- simulating steady-state rollout ...")
    rec = simulate(env, train_fn.network, out["params"],
                   jax.random.PRNGKey(7), n_steps=sweep.sim_steps)
    phase("simulation done")
    return rec, cfg, out["params"], train_fn.network


def steady_state_capital(rec) -> np.ndarray:
    """Per-agent time-averaged capital over the stationary window (auto-reset
    steps excluded) -- the 'capital = wealth' convention used throughout
    jmbc.diagnostics."""
    from jmbc.diagnostics import stationary_slice
    ks = stationary_slice(rec["ks"], 0.5)
    keep = ~stationary_slice(rec["done"], 0.5).astype(bool)
    return ks[keep].mean(axis=0)


def evaluate_cell(sigma: float, seed: int, sweep: SweepConfig):
    """Train + simulate at this (sigma, seed); return (row, artifacts)."""
    from jmbc.diagnostics import gini, top_shares, economic_report

    kappas = build_kappas(sweep.n_agents, sigma, seed)
    print(f"\n=== sigma={sigma:.3f} seed={seed}  (mu={-0.5*sigma**2:.4f}, "
          f"mean kappa={kappas.mean():.4f}, "
          f"min/max={kappas.min():.3f}/{kappas.max():.3f}) ===")
    rec, cfg, params, network = run_cell(kappas, seed, sweep)
    k_bar = steady_state_capital(rec)
    stats = {"capital_gini": gini(k_bar),
             **top_shares(k_bar, quantiles=(0.01, 0.1, 0.5))}
    econ = economic_report(rec, cfg.env, burn_frac=0.5)

    row = {
        "sigma": sigma, "seed": seed, "mu": -0.5 * sigma ** 2,
        "kappa_mean": float(kappas.mean()), "kappa_std": float(kappas.std()),
        "kappa_min": float(kappas.min()), "kappa_max": float(kappas.max()),
        # The initialization under test -- recorded per row so a results.csv
        # is interpretable on its own, without the config that produced it.
        "k_init_dist": str(cfg.env.k_init_dist),
        "k_init_low": float(cfg.env.k_init_low),
        "k_init_high": float(cfg.env.k_init_high),
        "k_init_resample": str(cfg.env.k_init_resample),
        **stats,
        "K_mean": econ["steady_state"]["K_mean"],
        "C_mean": econ["steady_state"]["C_mean"],
        "euler_mean_abs": econ["euler"]["euler_mean_abs"],
        "resource_mean_rel": econ["resource"]["resource_mean_rel"],
    }
    print(f"  capital_gini={stats['capital_gini']:.3f}  "
          f"top_10%={stats['top_0.1_share']:.3f}  "
          f"top_1%={stats['top_0.01_share']:.3f}  K_mean={row['K_mean']:.3f}")

    artifacts = {"rec": rec, "params": params, "network": network,
                "kappas": kappas, "delta": float(cfg.env.delta),
                "k_init_dist": str(cfg.env.k_init_dist),
                "k_init_low": float(cfg.env.k_init_low),
                "k_init_high": float(cfg.env.k_init_high),
                "k_init": float(cfg.env.k_init)}
    return row, artifacts


def save_raw(raw_dir: Path, tag: str, artifacts: dict):
    """Persist the full rollout + trained params/network + kappas under
    results/raw/<tag>/ -- everything needed to regenerate a different figure
    for this cell later without retraining."""
    import pickle
    from flax import serialization

    d = raw_dir / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(d / "rollout.npz", **artifacts["rec"])
    np.save(d / "kappas.npy", artifacts["kappas"])
    with open(d / "params.msgpack", "wb") as f:
        f.write(serialization.to_bytes(artifacts["params"]))
    with open(d / "network.pkl", "wb") as f:
        pickle.dump(artifacts["network"], f)
    with open(d / "meta.json", "w") as f:
        json.dump({"delta": artifacts["delta"]}, f)


def main():
    sweep = load_sweep_config()
    print(f"config: {OmegaConf.to_yaml(sweep)}")
    n_cells = len(sweep.sigmas) * len(sweep.seeds)
    print(f"NOTE: {len(sweep.sigmas)} sigmas x {len(sweep.seeds)} seeds = "
          f"{n_cells} cells -- time the first cell before trusting any "
          f"total-runtime estimate for the rest (see sibling experiment's "
          f"README).")

    out_dir = Path(sweep.out_dir)
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    raw_dir = out_dir / "raw"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    from jmbc.plots import apply_style
    apply_style()

    rows: List[dict] = []
    cells: Dict[Tuple[float, int], dict] = {}
    for sigma in [float(s) for s in sweep.sigmas]:
        for seed in [int(s) for s in sweep.seeds]:
            row, artifacts = evaluate_cell(sigma, seed, sweep)
            rows.append(row)
            cells[(sigma, seed)] = artifacts
            pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)  # checkpoint every cell
            if sweep.save_raw != "none":
                save_raw(raw_dir, f"sigma_{sigma:.2f}_seed_{seed}", artifacts)
            _plot_cell(sigma, seed, artifacts, fig_dir)

    df = pd.DataFrame(rows)
    print(f"\nswept {len(rows)} cells -> {out_dir / 'results.csv'}")
    _plot_comparison(df, cells, out_dir)


def _plot_cell(sigma: float, seed: int, artifacts: dict, fig_dir: Path):
    """Per-cell 'steady state dashboard': kappa profile, aggregate capital
    and consumption paths (does it actually settle into a stationary
    regime?), the aggregate KS shock (bad/good switching), the wealth
    distribution's Lorenz curve, and its histogram."""
    import matplotlib.pyplot as plt
    from jmbc.diagnostics import stationary_slice, lorenz, gini

    rec, kappas = artifacts["rec"], artifacts["kappas"]
    n = len(kappas)
    k_bar = steady_state_capital(rec)
    burn = int(len(rec["K"]) * 0.5)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    ax = axes[0, 0]
    order = np.argsort(kappas)
    ax.plot((np.arange(n) + 0.5) / n * 100, kappas[order], color="tab:blue")
    ax.set_xlabel("agent rank (%)"); ax.set_ylabel("kappa")
    ax.set_title(f"kappa profile (sigma={sigma:.2f}, seed={seed}, mean={kappas.mean():.3f})")

    ax = axes[0, 1]
    t = np.arange(len(rec["K"]))
    ax.plot(t, rec["K"], color="tab:blue", lw=0.7)
    ax.axvline(burn, color="0.6", ls="--", lw=1, label="burn-in cutoff")
    # Where the economy STARTED: the band the per-agent k_0 draw spans. The
    # point of this iteration is that K converges to the same steady state
    # from anywhere in this band, so it is worth seeing on the path itself.
    # Drawn UNDER the path's own y-limits: a band far wider than the settled
    # K would otherwise rescale the axis and flatten the path into a line.
    ylim = ax.get_ylim()
    if artifacts["k_init_dist"] == "uniform":
        lo, hi = artifacts["k_init_low"], artifacts["k_init_high"]
        ax.axhspan(lo, hi, color="tab:orange", alpha=0.12,
                   label=f"k_0 ~ U({lo:g}, {hi:g})")
    else:
        ax.axhline(artifacts["k_init"], color="tab:orange", ls=":", lw=1,
                   label=f"k_0 = {artifacts['k_init']:g}")
    ax.set_ylim(ylim)
    ax.set_xlabel("step"); ax.set_ylabel("aggregate capital K"); ax.legend(fontsize=7)
    ax.set_title("aggregate capital path")

    ax = axes[0, 2]
    cons_agg = np.asarray(rec["cons"]).mean(axis=1)
    ax.plot(t, cons_agg, color="tab:green", lw=0.7)
    ax.axvline(burn, color="0.6", ls="--", lw=1)
    ax.set_xlabel("step"); ax.set_ylabel("mean consumption")
    ax.set_title("aggregate consumption path")

    ax = axes[1, 0]
    if "agg_state" in rec:
        agg = stationary_slice(rec["agg_state"], 0.5)
        ax.plot(np.arange(len(agg)), agg, color="tab:red", lw=0.5)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["bad", "good"])
        ax.set_xlabel("step (post burn-in)"); ax.set_title("aggregate KS shock")
    else:
        ax.axis("off")

    ax = axes[1, 1]
    x, y = lorenz(k_bar)
    ax.plot(x, y, color="tab:blue"); ax.plot([0, 1], [0, 1], "k:", lw=0.8)
    ax.set_xlabel("cumulative share of agents"); ax.set_ylabel("cumulative share of wealth")
    ax.set_title(f"Lorenz curve (Gini={gini(k_bar):.3f})")

    ax = axes[1, 2]
    ax.hist(k_bar, bins=40, color="tab:blue", alpha=0.8)
    ax.set_xlabel("steady-state capital"); ax.set_ylabel("count")
    ax.set_title("wealth distribution")

    fig.suptitle(f"sigma = {sigma:.2f}, seed = {seed}", fontsize=13)
    fig.tight_layout()
    tag = f"sigma_{sigma:.2f}_seed_{seed}"
    fig.savefig(fig_dir / f"{tag}_steady_state.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {fig_dir / f'{tag}_steady_state.png'}")


def _plot_comparison(df, cells: Dict[Tuple[float, int], dict], out_dir: Path):
    """Cross-cell comparison: inequality stats vs sigma (one point per seed,
    so the spread at each sigma is visible), overlaid Lorenz curves and
    overlaid kappa profiles across every (sigma, seed) cell, colored by
    sigma."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from jmbc.diagnostics import lorenz

    sigmas = sorted(df["sigma"].unique())
    cmap = plt.get_cmap("viridis")
    colors = {s: cmap(i / max(1, len(sigmas) - 1)) for i, s in enumerate(sigmas)}
    jitter_rng = np.random.default_rng(0)  # x-axis jitter only, purely visual

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    for label, col, marker, color in [
        ("capital Gini", "capital_gini", "o", "tab:blue"),
        ("top 10% share", "top_0.1_share", "s", "tab:orange"),
        ("top 1% share", "top_0.01_share", "^", "tab:green"),
    ]:
        jitter = jitter_rng.uniform(-0.015, 0.015, size=len(df))
        ax.scatter(df["sigma"] + jitter, df[col], marker=marker, color=color,
                   alpha=0.55, s=28, label=label)
    ax.set_xlabel("sigma"); ax.set_ylabel("share / Gini")
    ax.set_title("Inequality vs. sigma (one point per seed)"); ax.legend(fontsize=8)

    ax = axes[1]
    for (sigma, seed), artifacts in cells.items():
        k_bar = steady_state_capital(artifacts["rec"])
        x, y = lorenz(k_bar)
        ax.plot(x, y, color=colors[sigma], alpha=0.35, lw=1)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="perfect equality")
    ax.set_xlabel("cumulative share of agents"); ax.set_ylabel("cumulative share of wealth")
    ax.set_title("Lorenz curves (all seeds, colored by sigma)")
    ax.legend(handles=[Line2D([0], [0], color="k", ls=":", lw=0.8, label="perfect equality")],
              fontsize=7)

    ax = axes[2]
    for (sigma, seed), artifacts in cells.items():
        kappas = artifacts["kappas"]
        n = len(kappas)
        order = np.argsort(kappas)
        ax.plot((np.arange(n) + 0.5) / n * 100, kappas[order],
               color=colors[sigma], alpha=0.35, lw=1)
    ax.set_xlabel("agent rank (%)"); ax.set_ylabel("kappa")
    ax.set_title("Input kappa profiles (all seeds, colored by sigma)")
    ax.legend(handles=[Line2D([0], [0], color=colors[s], lw=2, label=f"sigma={s:.2f}")
                       for s in sigmas], fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
