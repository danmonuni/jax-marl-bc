"""Calibrate heterogeneous capital productivity against the STATIONARY WEALTH
DISTRIBUTION (Gini / top shares), not against a return curve directly.

Design (see README.md for the full writeup): Xavier (2021)'s digitized
return-by-wealth-percentile bars are interpolated into a smooth, monotonic
quantile function ``Q(p) -> annual return (%)`` (PCHIP, so it's the real
shape, not a step function). Agent rank is mapped to a percentile through a
BETA-DISTORTED uniform grid rather than a plain one:

    u_i        = (i + 0.5) / n                      # plain uniform grid
    p_i        = Beta(a, b).ppf(u_i)                 # distorted percentile
    shape_i    = Q(clip(p_i))                        # look up Xavier's curve there
    kappa_i    = scale * shape_i / mean(shape)        # mean(kappa) = scale

Three free parameters -- ``scale`` (absolute level), ``a``/``b`` (Beta shape,
(1,1) = no distortion = literally Xavier's own curve) -- are searched by
Bayesian optimization (same `skopt.gp_minimize` machinery as the sibling
``ks-heterogeneous-returns`` experiment) to match target statistics of the
EMERGENT stationary wealth distribution (default: top-10% wealth share),
computed with the existing `jmbc.diagnostics.gini`/`top_shares` -- no new
diagnostic code needed, unlike the returns-curve design.

Usage (same config.yaml + CLI-dotlist convention as the sibling experiment):
    python calibrate_wealth_distortion.py
    python calibrate_wealth_distortion.py n_agents=200 device=cpu
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

CSV_PATH = HERE / "xavier_return_by_wealth_percentile.csv"
DEFAULT_CONFIG_PATH = HERE / "config.yaml"
PERIODS_PER_YEAR = 4  # see sibling experiment's README "Caveats" -- same assumption


@dataclass
class CalibrationConfig:
    base_exp: str = "ks_n200"
    n_agents: int = 1000
    num_envs: Optional[int] = 8
    bo_calls: int = 50
    bo_init_points: int = 8        # 3-D search: a slightly bigger initial design
                                   # than the 1-D sibling experiment's 5
    scale_bounds: List[float] = field(default_factory=lambda: [0.3, 3.0])
    a_bounds: List[float] = field(default_factory=lambda: [0.1, 10.0])
    b_bounds: List[float] = field(default_factory=lambda: [0.1, 10.0])
    # Target statistics of the STATIONARY WEALTH distribution to match.
    # Keys must be gini/top_shares output names: "capital_gini",
    # "top_0.1_share", "top_0.01_share". 0.70 is the top-10% figure already
    # cited in the paper's text (Xavier/US data); a Gini target isn't
    # independently cited anywhere in this repo -- if you add one, source it
    # yourself rather than trusting a default here.
    targets: Dict[str, float] = field(default_factory=lambda: {"top_0.1_share": 0.70})
    sim_steps: int = 5000
    seed: int = 0
    device: str = "gpu"
    total_timesteps: Optional[int] = None
    out_dir: str = "results"
    save_raw: str = "best"


def load_calib_config(argv: Optional[List[str]] = None) -> CalibrationConfig:
    argv = sys.argv[1:] if argv is None else argv
    dotlist = [a for a in argv if "=" in a and not a.startswith("config=")]
    config_arg = next((a.split("=", 1)[1] for a in argv if a.startswith("config=")), None)
    config_path = Path(config_arg) if config_arg else DEFAULT_CONFIG_PATH

    cfg = OmegaConf.structured(CalibrationConfig)
    if config_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def load_target_buckets():
    """Same CSV/format as the sibling experiment -- see its README for the
    digitization note and the 0-20% flat-extrapolation convention."""
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(r for r in f if not r.lstrip().startswith("#")):
            rows.append((float(row["bucket_low"]), float(row["bucket_high"]),
                        float(row["mean_return_pct"])))
    rows.sort(key=lambda r: r[0])
    if rows[0][0] != 0.0 or rows[-1][1] != 100.0:
        raise ValueError(f"target buckets must cover 0-100%, got "
                         f"{rows[0][0]:g}-{rows[-1][1]:g}")
    return rows


def build_target_quantile_fn(buckets):
    """Monotonic quantile function Q(p in [0,1]) -> annual return (%),
    PCHIP-interpolated through each bucket's midpoint -- the real shape of
    Xavier's curve, not a step function over discrete buckets."""
    from scipy.interpolate import PchipInterpolator
    mids = np.array([(lo + hi) / 2.0 / 100.0 for lo, hi, _ in buckets])
    rets = np.array([ret for _, _, ret in buckets])
    return PchipInterpolator(mids, rets, extrapolate=True)


def build_kappa_shape(quantile_fn, n_agents: int, a: float, b: float) -> np.ndarray:
    """shape_i = Q(Beta(a,b).ppf(u_i)), normalized to mean 1.

    u_i is a plain uniform grid (agent rank); Beta(a,b).ppf distorts it --
    (a,b)=(1,1) is the identity (Beta(1,1) = Uniform), recovering literally
    Xavier's own curve with no distortion. Clipped away from the [0,1]
    boundary before evaluating Q (PCHIP extrapolation can go non-positive
    right at the edges) and floored at a small positive value -- kappa must
    stay strictly positive, it's a productivity multiplier.
    """
    from scipy.stats import beta as beta_dist
    u = (np.arange(n_agents) + 0.5) / n_agents
    p = np.clip(beta_dist.ppf(u, a, b), 1e-3, 1 - 1e-3)
    shape = np.clip(quantile_fn(p), 0.1, None)
    return shape / shape.mean()


def run_cell(kappas: np.ndarray, calib: CalibrationConfig):
    """Train one KS run at this kappa vector; return (rec, delta, params, network)."""
    from jmbc.config import load_config, to_train_dict, setup_device

    n_agents = len(kappas)
    overrides = [
        f"env.n_agents={n_agents}",
        "diag.n_snapshots=1",
        f"diag.sim_steps={calib.sim_steps}",
        f"run.seed={calib.seed}",
        f"run.device={calib.device}",
    ]
    if calib.total_timesteps:
        overrides.append(f"train.total_timesteps={calib.total_timesteps}")
    if calib.num_envs:
        overrides.append(f"train.num_envs={calib.num_envs}")
    cfg = load_config(calib.base_exp, overrides)
    cfg.env.kappas = kappas.tolist()
    setup_device(cfg.run.device, bool(cfg.run.prealloc))

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
                   jax.random.PRNGKey(7), n_steps=calib.sim_steps)
    phase("simulation done")
    return rec, float(cfg.env.delta), out["params"], train_fn.network


def steady_state_capital(rec) -> np.ndarray:
    """Per-agent time-averaged capital over the stationary window (auto-reset
    steps excluded) -- the same 'capital = wealth' convention as the rest of
    this repo (distributional.py's capital_gini)."""
    from jmbc.diagnostics import stationary_slice
    ks = stationary_slice(rec["ks"], 0.5)
    keep = ~stationary_slice(rec["done"], 0.5).astype(bool)
    return ks[keep].mean(axis=0)


def score_targets(k_bar: np.ndarray, targets: Dict[str, float]):
    """RMS relative error between simulated and target statistics of the
    stationary wealth distribution. Always reports capital_gini + top-10%/
    top-1% shares regardless of which ones are in ``targets``."""
    from jmbc.diagnostics import gini, top_shares
    stats = {"capital_gini": gini(k_bar), **top_shares(k_bar, quantiles=(0.01, 0.1))}
    errs, detail = [], {}
    for name, target_val in targets.items():
        sim_val = stats[name]
        detail[f"target_{name}"] = target_val
        errs.append((sim_val - target_val) / target_val)
    score = float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan")
    return score, stats, detail


def evaluate_params(scale, a, b, quantile_fn, calib):
    """Train + simulate at this (scale, a, b); return (row, artifacts)."""
    shape = build_kappa_shape(quantile_fn, calib.n_agents, a, b)
    kappas = scale * shape
    print(f"\n=== scale={scale:.4f}  a={a:.4f}  b={b:.4f}  "
          f"(mean kappa={kappas.mean():.3f}, min/max={kappas.min():.3f}/"
          f"{kappas.max():.3f}) ===")
    rec, delta, params, network = run_cell(kappas, calib)
    k_bar = steady_state_capital(rec)
    score, stats, detail = score_targets(k_bar, calib.targets)

    row = {"scale": scale, "a": a, "b": b, "score": score,
          "capital_gini": stats["capital_gini"],
          "top_0.1_share": stats["top_0.1_share"],
          "top_0.01_share": stats["top_0.01_share"], **detail}
    print(f"  score (RMS rel. error) = {score:.4f}   "
          f"capital_gini={stats['capital_gini']:.3f}  "
          f"top_10%={stats['top_0.1_share']:.3f}  top_1%={stats['top_0.01_share']:.3f}")
    for name, target_val in calib.targets.items():
        print(f"    target[{name}] = {target_val:.3f}  vs  sim = {stats[name]:.3f}")

    artifacts = {"rec": rec, "params": params, "network": network,
                "delta": delta, "kappas": kappas, "shape": shape}
    return row, artifacts


def bo_search(quantile_fn, calib, on_row=None):
    """Bayesian optimization over (scale, a, b) -- same GP+EI machinery as
    the sibling experiment, natively extended to 3 dimensions (skopt handles
    this directly; no other change to the algorithm)."""
    from skopt import gp_minimize
    from skopt.space import Real

    rows: List[dict] = []

    def objective(x):
        scale, a, b = (float(v) for v in x)
        row, artifacts = evaluate_params(scale, a, b, quantile_fn, calib)
        rows.append(row)
        if on_row:
            on_row(row, artifacts)
        return row["score"]

    dims = [
        Real(*calib.scale_bounds, prior="log-uniform", name="scale"),
        Real(*calib.a_bounds, prior="log-uniform", name="a"),
        Real(*calib.b_bounds, prior="log-uniform", name="b"),
    ]
    print(f"=== BO: {calib.bo_calls} calls ({calib.bo_init_points} initial design) "
          f"over scale in {calib.scale_bounds}, a in {calib.a_bounds}, "
          f"b in {calib.b_bounds} (all log-uniform) ===")
    gp_minimize(objective, dims, n_calls=int(calib.bo_calls),
               n_initial_points=int(calib.bo_init_points),
               acq_func="EI", random_state=int(calib.seed))

    best_row = min(rows, key=lambda r: r["score"])
    return rows, best_row


def main():
    calib = load_calib_config()
    print(f"config: {OmegaConf.to_yaml(calib)}")
    if calib.n_agents >= 1000:
        print(f"NOTE: n_agents={calib.n_agents} -- time the first evaluation "
              f"before trusting any total-runtime estimate for the rest of "
              f"the search (see sibling experiment's README).")

    buckets = load_target_buckets()
    quantile_fn = build_target_quantile_fn(buckets)

    out_dir = Path(calib.out_dir)
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    rows = []
    raw_dir = out_dir / "raw"
    best_score = [float("inf")]

    def save_raw(tag: str, artifacts: dict):
        import pickle, json, shutil
        from flax import serialization

        d = raw_dir / tag
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(d / "rollout.npz", **artifacts["rec"])
        np.save(d / "kappas.npy", artifacts["kappas"])
        np.save(d / "shape.npy", artifacts["shape"])
        with open(d / "params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(artifacts["params"]))
        with open(d / "network.pkl", "wb") as f:
            pickle.dump(artifacts["network"], f)
        with open(d / "meta.json", "w") as f:
            json.dump({"delta": artifacts["delta"]}, f)

    def checkpoint(row, artifacts=None):
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)
        if artifacts is None or calib.save_raw == "none":
            return
        idx = len(rows) - 1
        if calib.save_raw == "all":
            save_raw(f"eval_{idx:03d}", artifacts)
        if row["score"] < best_score[0]:
            best_score[0] = row["score"]
            save_raw("best", artifacts)
            print(f"  -> new best, raw rollout + params saved to {raw_dir / 'best'}")

    _, best_row = bo_search(quantile_fn, calib, on_row=checkpoint)

    df = pd.DataFrame(rows)
    print(f"\nbest (scale, a, b) = ({best_row['scale']:.4f}, {best_row['a']:.4f}, "
          f"{best_row['b']:.4f})  score={best_row['score']:.4f} -> {out_dir/'results.csv'}")

    _plot_results(df, buckets, quantile_fn, calib, out_dir)


def _plot_results(df, buckets, quantile_fn, calib, out_dir):
    import matplotlib.pyplot as plt
    from jmbc.diagnostics import lorenz

    best_idx = df["score"].idxmin()
    best = df.loc[best_idx]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))

    ax = axes[0]
    sc = ax.scatter(np.arange(len(df)), df["score"], c=np.arange(len(df)), cmap="viridis")
    ax.scatter(best_idx, best["score"], marker="*", s=250, facecolors="none",
              edgecolors="red", linewidths=1.5)
    ax.set_xlabel("evaluation order"); ax.set_ylabel("score (RMS rel. error)")
    ax.set_title("Calibration convergence")

    ax = axes[1]
    raw_best_dir = Path(out_dir) / "raw" / "best"
    if (raw_best_dir / "rollout.npz").exists():
        # Reconstruct k_bar for the actual best evaluation on disk (not
        # necessarily this in-memory `best` row if save_raw=="none").
        import json
        from jmbc.diagnostics import stationary_slice
        rec = dict(np.load(raw_best_dir / "rollout.npz"))
        meta = json.load(open(raw_best_dir / "meta.json"))
        ks = stationary_slice(rec["ks"], 0.5)
        keep = ~stationary_slice(rec["done"], 0.5).astype(bool)
        k_bar = ks[keep].mean(axis=0)
        x, y = lorenz(k_bar)
        ax.plot(x, y, color="tab:blue", label="simulated Lorenz (best)")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="perfect equality")
        # Mark each top-q-share target as the single point a Lorenz curve
        # hitting it exactly would pass through: (1-q, 1-target), with
        # dotted guides to both axes -- clearer than a full-width line.
        for name, target_val in calib.targets.items():
            if name.startswith("top_"):
                q = float(name.split("_")[1])
                tx, ty = 1 - q, 1 - target_val
                ax.plot([tx, tx, 0], [0, ty, ty], color="tab:red", ls=":", lw=1)
                ax.plot(tx, ty, "x", color="tab:red", markersize=9,
                       label=f"target {name}={target_val:.2f}")
        ax.legend(fontsize=7)
    else:
        ax.text(0.5, 0.5, "save_raw='none': no rollout to plot", ha="center", va="center")
    ax.set_xlabel("cumulative share of agents"); ax.set_ylabel("cumulative share of wealth")
    ax.set_title(f"Lorenz curve (Gini={best['capital_gini']:.3f})")

    ax = axes[2]
    p_plot = np.linspace(0.01, 0.99, 200)
    ax.plot(p_plot * 100, quantile_fn(p_plot), color="0.5", ls=":",
           label="Xavier 2021 (undistorted)")
    shape = build_kappa_shape(quantile_fn, 200, best["a"], best["b"])
    rank_pct = (np.arange(200) + 0.5) / 200 * 100
    ax.plot(rank_pct, best["scale"] * shape, color="tab:orange",
           label=f"best kappa profile (a={best['a']:.2f}, b={best['b']:.2f}, "
                 f"scale={best['scale']:.2f})")
    ax.set_xlabel("agent rank (%)"); ax.set_ylabel("kappa (or Xavier return %, undistorted)")
    ax.set_title("Input shape: distorted vs. raw Xavier curve")
    ax.legend(fontsize=7)

    fig.tight_layout()
    fig.savefig(out_dir / "calibration_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out_dir / 'calibration_fit.png'}")


if __name__ == "__main__":
    main()
