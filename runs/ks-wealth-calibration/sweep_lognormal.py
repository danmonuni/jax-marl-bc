"""Grid-search heterogeneous capital productivity as a lognormal spread.

Design (see README.md for the full writeup and Aldo's original guidance):
rather than calibrating kappa's cross-sectional SHAPE against an external
empirical curve (the sibling ``runs/ks-heterogeneous-{returns,wealth}/``
experiments, both anchored to Xavier 2021), this experiment generates kappa
directly from a LogNormal(mu, sigma) with mean fixed to 1 (mu = -sigma^2/2)
and sweeps sigma over a grid from 0 (homogeneous) to 1 (wide spread).

This is deliberately NOT a calibration against a target statistic -- fitting
the wealth Gini is "a thorny, poorly-identified quantity" (Aldo) and isn't
the point here. The point is to demonstrate the software's capability across
a spread of heterogeneity and let the resulting steady states speak for
themselves; picking whichever sigma(s) look best for the paper is a
visual/qualitative call made from the saved figures, not an optimization
objective.

Sampling. Agent kappas are NOT drawn i.i.d. from LogNormal(mu, sigma) -- that
would inject sampling noise into every cell, on top of the RL training's own
seed variance (a "did this run just get an unlucky draw" confound that would
make sigma harder to read off the results). Instead, agent rank is placed on
a REGULAR grid of quantiles, u_i = (i + 0.5) / n, and each agent's kappa is
that regular grid inverted through the lognormal's quantile function (ppf):
deterministic, exactly reproducible, and by construction the smoothest
possible n-point mesh that conforms to the LogNormal(mu, sigma) shape
without draw-to-draw surprises. (Same regular-quantile-mesh convention as the
Beta-distorted grid in the sibling ks-heterogeneous-wealth experiment.)

Usage (same config.yaml + CLI-dotlist convention as the sibling experiments):
    python sweep_lognormal.py
    python sweep_lognormal.py n_agents=100 device=cpu sigmas=[0.0,0.5]
"""
from __future__ import annotations

import json
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

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
    sim_steps: int = 5000
    seed: int = 0
    device: str = "gpu"
    total_timesteps: Optional[int] = None   # None -> base_exp's own budget
    out_dir: str = "results"
    save_raw: str = "all"          # small grid (few cells): keep every
                                    # cell's raw rollout, not just the "best"
                                    # -- there is no "best", every sigma is a
                                    # deliberate design point


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


def build_kappas(n_agents: int, sigma: float) -> np.ndarray:
    """Deterministic regular-quantile-mesh LogNormal(mu, sigma) draw, mean 1.

    u_i = (i + 0.5) / n on a regular grid (not i.i.d. sampling -- see module
    docstring); mu = -sigma^2/2 keeps E[LogNormal] = 1 in the continuum
    limit. Renormalized by the empirical mean afterwards since the finite-n
    quantile mesh's mean deviates slightly from that analytic mean (more so
    at high sigma, where the mesh's extreme points sit further out on the
    heavy right tail) -- same normalize-after-construct convention as the
    sibling Beta-distortion experiment's ``shape / shape.mean()``.
    """
    if sigma <= 0:
        return np.ones(n_agents, np.float64)
    from scipy.stats import lognorm
    u = (np.arange(n_agents) + 0.5) / n_agents
    mu = -0.5 * sigma ** 2
    kappas = lognorm.ppf(u, s=sigma, scale=np.exp(mu))
    return kappas / kappas.mean()


def run_cell(kappas: np.ndarray, sweep: SweepConfig):
    """Train one KS run at this kappa vector; return (rec, cfg, params, network)."""
    from jmbc.config import load_config, to_train_dict, setup_device

    n_agents = len(kappas)
    overrides = [
        f"env.n_agents={n_agents}",
        "diag.n_snapshots=1",
        f"diag.sim_steps={sweep.sim_steps}",
        f"run.seed={sweep.seed}",
        f"run.device={sweep.device}",
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

    rng = jax.random.PRNGKey(int(cfg.run.seed))
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


def evaluate_sigma(sigma: float, sweep: SweepConfig):
    """Train + simulate at this sigma; return (row, artifacts)."""
    from jmbc.diagnostics import gini, top_shares, economic_report

    kappas = build_kappas(sweep.n_agents, sigma)
    print(f"\n=== sigma={sigma:.3f}  (mu={-0.5*sigma**2:.4f}, "
          f"mean kappa={kappas.mean():.4f}, "
          f"min/max={kappas.min():.3f}/{kappas.max():.3f}) ===")
    rec, cfg, params, network = run_cell(kappas, sweep)
    k_bar = steady_state_capital(rec)
    stats = {"capital_gini": gini(k_bar),
             **top_shares(k_bar, quantiles=(0.01, 0.1, 0.5))}
    econ = economic_report(rec, cfg.env, burn_frac=0.5)

    row = {
        "sigma": sigma, "mu": -0.5 * sigma ** 2,
        "kappa_mean": float(kappas.mean()), "kappa_std": float(kappas.std()),
        "kappa_min": float(kappas.min()), "kappa_max": float(kappas.max()),
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
                "kappas": kappas, "delta": float(cfg.env.delta)}
    return row, artifacts


def save_raw(raw_dir: Path, tag: str, artifacts: dict):
    """Persist the full rollout + trained params/network + kappas under
    results/raw/<tag>/ -- everything needed to regenerate a different figure
    for this sigma later without retraining."""
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
    if sweep.n_agents >= 1000:
        print(f"NOTE: n_agents={sweep.n_agents} -- time the first evaluation "
              f"before trusting any total-runtime estimate for the rest of "
              f"the sweep (see sibling experiments' README).")

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

    rows, cells = [], {}
    for sigma in [float(s) for s in sweep.sigmas]:
        row, artifacts = evaluate_sigma(sigma, sweep)
        rows.append(row)
        cells[sigma] = artifacts
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)  # checkpoint every cell
        if sweep.save_raw != "none":
            save_raw(raw_dir, f"sigma_{sigma:.2f}", artifacts)
        _plot_cell(sigma, artifacts, fig_dir)

    df = pd.DataFrame(rows)
    print(f"\nswept {len(rows)} sigma values -> {out_dir / 'results.csv'}")
    _plot_comparison(df, cells, out_dir)


def _plot_cell(sigma: float, artifacts: dict, fig_dir: Path):
    """Per-sigma 'steady state dashboard': kappa profile, aggregate capital
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
    ax.set_title(f"kappa profile (sigma={sigma:.2f}, mean={kappas.mean():.3f})")

    ax = axes[0, 1]
    t = np.arange(len(rec["K"]))
    ax.plot(t, rec["K"], color="tab:blue", lw=0.7)
    ax.axvline(burn, color="0.6", ls="--", lw=1, label="burn-in cutoff")
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

    fig.suptitle(f"sigma = {sigma:.2f}", fontsize=13)
    fig.tight_layout()
    fig.savefig(fig_dir / f"sigma_{sigma:.2f}_steady_state.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {fig_dir / f'sigma_{sigma:.2f}_steady_state.png'}")


def _plot_comparison(df, cells: dict, out_dir: Path):
    """Cross-sigma comparison: inequality stats vs sigma, overlaid Lorenz
    curves and overlaid kappa profiles, colored by sigma -- the "which run is
    nicest" view."""
    import matplotlib.pyplot as plt
    from jmbc.diagnostics import lorenz

    sigmas = sorted(cells.keys())
    cmap = plt.get_cmap("viridis")
    colors = {s: cmap(i / max(1, len(sigmas) - 1)) for i, s in enumerate(sigmas)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.plot(df["sigma"], df["capital_gini"], "o-", label="capital Gini")
    ax.plot(df["sigma"], df["top_0.1_share"], "s-", label="top 10% share")
    ax.plot(df["sigma"], df["top_0.01_share"], "^-", label="top 1% share")
    ax.set_xlabel("sigma"); ax.set_ylabel("share / Gini")
    ax.set_title("Inequality vs. sigma"); ax.legend(fontsize=8)

    ax = axes[1]
    for sigma in sigmas:
        k_bar = steady_state_capital(cells[sigma]["rec"])
        x, y = lorenz(k_bar)
        ax.plot(x, y, color=colors[sigma], label=f"sigma={sigma:.2f}")
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="perfect equality")
    ax.set_xlabel("cumulative share of agents"); ax.set_ylabel("cumulative share of wealth")
    ax.set_title("Lorenz curves"); ax.legend(fontsize=7)

    ax = axes[2]
    for sigma in sigmas:
        kappas = cells[sigma]["kappas"]
        n = len(kappas)
        order = np.argsort(kappas)
        ax.plot((np.arange(n) + 0.5) / n * 100, kappas[order],
               color=colors[sigma], label=f"sigma={sigma:.2f}")
    ax.set_xlabel("agent rank (%)"); ax.set_ylabel("kappa")
    ax.set_title("Input kappa profiles"); ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out_dir / 'comparison.png'}")


if __name__ == "__main__":
    main()
