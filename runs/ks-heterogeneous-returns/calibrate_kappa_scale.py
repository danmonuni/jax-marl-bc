"""Calibrate the capital-productivity (kappa) scale against Xavier (2021).

Design (see README.md for the full writeup): the RELATIVE shape of kappa
across the n agents is hardcoded directly from
``xavier_return_by_wealth_percentile.csv`` (agent rank i -> the return-on-
wealth value of the percentile bucket i belongs to). The only free parameter
is a single scalar, ``k_multiplier``, applied to that whole shape:

    kappa_i = k_multiplier * base_vector_i

For each candidate ``k_multiplier`` this script trains a full KS run, then
computes the STEADY-STATE (time-averaged, i.e. one long stationary rollout
averaged over time -- the "temporal average" reading of steady state) return
of every agent, re-ranks agents by their own steady-state capital (not by the
a-priori kappa assignment), bins them into the same percentile buckets as the
source chart, and compares against the target vector via the elementwise
ratio simulated/target ("close to 1" = good fit). ``k_multiplier`` is swept
over a grid and scored by the RMS deviation of that ratio from 1.

Usage (same merge convention as jmbc.config.load_config: structured defaults
< config.yaml < CLI dotlist overrides -- later wins):
    python calibrate_kappa_scale.py
    python calibrate_kappa_scale.py n_agents=200 device=cpu
    python calibrate_kappa_scale.py config=my_variant.yaml k_grid=[0.2,0.3]

Run from anywhere with the ``jmbc`` package importable (repo installed
editable -- ``pip install -e .`` from the repo root, the Colab convention --
or PYTHONPATH pointing at the repo root, handled automatically below).
"""
from __future__ import annotations

import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]  # runs/ks-heterogeneous-returns -> repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # jmbc isn't pip-installed in this repo yet

CSV_PATH = HERE / "xavier_return_by_wealth_percentile.csv"
DEFAULT_CONFIG_PATH = HERE / "config.yaml"
PERIODS_PER_YEAR = 4          # ASSUMPTION: delta=0.025 matches the standard
                              # quarterly RBC/KS calibration -> annualize by
                              # compounding 4 periods. Confirm before trusting
                              # absolute return levels (see README "Caveats").


@dataclass
class CalibrationConfig:
    base_exp: str = "ks_n200"     # configs/exp/<base_exp>.yaml -- same protocol
                                  # as ks_n20/ks_n200/ks_n2000, only n_agents
                                  # differs, and this script overrides that
    n_agents: int = 1000
    mode: str = "es"              # "es" (adaptive search) | "grid" (flat sweep)
    k_grid: List[float] = field(
        default_factory=lambda: [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5])
    es_iters: int = 12            # additional evaluations after the initial guess
    es_step0: float = 0.3         # initial log-space step stddev (multiplicative)
    es_grow: float = 1.5          # step *= es_grow on an improving move
    es_shrink: float = 0.7        # step *= es_shrink on a rejected move
    es_k0: Optional[float] = None  # None -> 1/mean(base_vector) naive guess
    sim_steps: int = 5000
    seed: int = 0
    device: str = "gpu"           # "gpu" | "cpu" | "auto"
    total_timesteps: Optional[int] = None  # None -> base_exp's own budget
    out_dir: str = "results"      # relative to this file, or an absolute path


def load_calib_config(argv: Optional[List[str]] = None) -> CalibrationConfig:
    """structured defaults < config.yaml (or ``config=<path>`` override) <
    CLI dotlist overrides -- identical merge order to jmbc.config.load_config."""
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
    """Read the source CSV: bucket_low, bucket_high, mean_return_pct, source.

    The CSV owns the full 0-100% coverage, including the 0-20% bucket (not in
    the original chart -- flat-extrapolated from the 20-40% bar; see
    README.md). Nothing is synthesized here; the file is the single source of
    truth for the target vector.
    """
    rows = []
    with open(CSV_PATH) as f:
        for row in csv.DictReader(r for r in f if not r.lstrip().startswith("#")):
            rows.append((float(row["bucket_low"]), float(row["bucket_high"]),
                        float(row["mean_return_pct"]), row["source"]))
    rows.sort(key=lambda r: r[0])
    if rows[0][0] != 0.0 or rows[-1][1] != 100.0:
        raise ValueError(
            f"target buckets must cover 0-100%, got "
            f"{rows[0][0]:g}-{rows[-1][1]:g}"
        )
    return rows


def bucket_agent_counts(buckets, n_agents: int):
    """Largest-remainder allocation of n_agents across bucket widths, so
    counts sum to exactly n_agents while staying proportional to width."""
    widths = np.array([hi - lo for lo, hi, _, _ in buckets], dtype=float)
    raw = widths / widths.sum() * n_agents
    counts = np.floor(raw).astype(int)
    remainder = n_agents - counts.sum()
    order = np.argsort(-(raw - counts))  # largest fractional remainder first
    counts[order[:remainder]] += 1
    if (counts == 0).any():
        raise ValueError(
            f"n_agents={n_agents} leaves an empty percentile bucket "
            f"(counts={counts.tolist()}) -- the finest bucket (99-100%) needs "
            f"at least 1 agent per 1% of population; use n_agents >= 100."
        )
    return counts


def build_base_vector(buckets, counts) -> np.ndarray:
    """Ascending per-agent target-return vector: agent 0 = lowest bucket.

    Units: plain percentage-point numbers straight from the CSV (3.6 means
    3.6%, not 0.036 and not 3.6%-as-360). ``kappa_i = k_multiplier *
    base_vector_i`` uses these as-is; k_multiplier is exactly the free
    parameter that reconciles this arbitrary numeric scale (raw target
    percentages, O(3-8)) with whatever scale kappa actually needs to be at
    (O(1), to stay comparable to the homogeneous kappa≡1 baseline) --
    there's no separate /100 anywhere because there doesn't need to be one.
    """
    return np.concatenate([
        np.full(c, ret) for (_, _, ret, _), c in zip(buckets, counts)
    ])


def run_cell(k_multiplier: float, base_vector: np.ndarray, calib: CalibrationConfig):
    """Train one KS run at this kappa scale; return the final rollout dict."""
    from jmbc.config import load_config, to_train_dict, setup_device

    n_agents = len(base_vector)
    kappas = (k_multiplier * base_vector).tolist()
    overrides = [
        f"env.n_agents={n_agents}",
        "diag.n_snapshots=1",           # only need the final steady state
        f"diag.sim_steps={calib.sim_steps}",
        f"run.seed={calib.seed}",
        f"run.device={calib.device}",
    ]
    if calib.total_timesteps:
        overrides.append(f"train.total_timesteps={calib.total_timesteps}")
    cfg = load_config(calib.base_exp, overrides)
    cfg.env.kappas = kappas
    setup_device(cfg.run.device, bool(cfg.run.prealloc))  # before jax import

    import jax
    from jmbc.envs import build_env
    from jmbc.algos import make_train
    from jmbc.diagnostics import simulate
    from jmbc.experiments.common import _print_launch_summary
    from jmbc.recorder import phase

    env = build_env(cfg.env)
    train_fn = make_train(env, to_train_dict(cfg))
    _print_launch_summary(cfg, env, train_fn, int(cfg.run.seed))  # confirms
    # NUM_ENVS/n_agents/batch shapes actually being used -- don't guess from
    # how slow it feels, read it here every cell.

    rng = jax.random.PRNGKey(int(cfg.run.seed))
    phase("training ...")
    out = train_fn(rng)
    phase("training done -- simulating steady-state rollout ...")
    rec = simulate(env, train_fn.network, out["params"],
                   jax.random.PRNGKey(7), n_steps=calib.sim_steps)
    phase("simulation done")
    return rec, float(cfg.env.delta)


def steady_state_return_by_bucket(rec, delta: float, counts):
    """Time-average per agent over the stationary window, rank by realized
    capital (matching the paper's existing 'capital = wealth' convention,
    see distributional.py's capital_gini), bin into ``counts``, annualize."""
    from jmbc.diagnostics import stationary_slice

    burn = 0.5
    R = stationary_slice(rec["R"], burn)            # [T, n] gross per-period return
    ks = stationary_slice(rec["ks"], burn)           # [T, n] capital held
    keep = ~stationary_slice(rec["done"], burn).astype(bool)
    R, ks = R[keep], ks[keep]

    r_period = R.mean(axis=0) - (1.0 - delta)        # per-agent net marginal return
    r_annual_pct = 100.0 * ((1.0 + r_period) ** PERIODS_PER_YEAR - 1.0)
    k_bar = ks.mean(axis=0)                          # per-agent steady-state capital

    order = np.argsort(k_bar)                        # ascending: poorest -> richest
    r_sorted = r_annual_pct[order]
    edges = np.cumsum(counts)
    groups = np.split(r_sorted, edges[:-1])
    return np.array([g.mean() for g in groups]), k_bar


def score_fit(simulated_by_bucket, target_by_bucket, chart_mask):
    """RMS deviation of simulated/target ratio from 1, over buckets marked
    ``source=chart`` in the CSV (any row marked otherwise, e.g. a future
    non-chart extrapolation, is reported but excluded from the score)."""
    ratio = simulated_by_bucket[chart_mask] / target_by_bucket[chart_mask]
    return ratio, float(np.sqrt(np.mean((ratio - 1.0) ** 2)))


def evaluate_k(k, base_vector, counts, target_by_bucket, chart_mask,
               bucket_labels, calib) -> dict:
    """Train + simulate at this k_multiplier; return one flattened results
    row (shared by both the grid sweep and the ES search)."""
    print(f"\n=== k_multiplier = {k:.4f} "
          f"(mean kappa = {k * base_vector.mean():.3f}) ===")
    rec, delta = run_cell(k, base_vector, calib)
    simulated_by_bucket, k_bar = steady_state_return_by_bucket(rec, delta, counts)
    ratio, score = score_fit(simulated_by_bucket, target_by_bucket, chart_mask)

    from jmbc.diagnostics import gini, top_shares
    held_out = {"capital_gini": gini(k_bar), **top_shares(k_bar)}

    row = {"k_multiplier": k, "score_rms_ratio_minus_1": score, **held_out}
    for lbl, sim_r, tgt_r in zip(bucket_labels, simulated_by_bucket, target_by_bucket):
        row[f"sim_return_pct[{lbl}]"] = sim_r
        row[f"target_return_pct[{lbl}]"] = tgt_r

    print(f"  score (RMS |ratio-1|) = {score:.4f}   "
          f"held-out capital_gini = {held_out['capital_gini']:.3f}  "
          f"top_10%_share = {held_out['top_0.1_share']:.3f}")
    for lbl, sim_r, tgt_r in zip(bucket_labels, simulated_by_bucket, target_by_bucket):
        print(f"    [{lbl:>7}] sim={sim_r:6.2f}%  target={tgt_r:5.2f}%  "
              f"ratio={sim_r / tgt_r:5.2f}")
    return row


def es_search(base_vector, counts, target_by_bucket, chart_mask, bucket_labels,
             calib, on_row=None):
    """(1+1) evolution strategy, Rechenberg's 1/5 success rule.

    k_multiplier is a strictly-positive scale, so steps are multiplicative in
    log-space: propose k_best * exp(step * randn()). Accept if the score
    improves (grow the step -- we're moving the right way, be bolder);
    reject otherwise (shrink the step -- overshot or wrong direction, be more
    cautious). Direction falls out of which side of k_best last improved;
    scale is the adaptively-sized step; the randomness is the proposal noise
    itself. Cheap (no new dependency), and well suited to an expensive
    (~minutes/eval), unknown-shape 1-D objective -- see README "Search
    strategy" for why this was chosen over e.g. Bayesian optimization.
    """
    rng = np.random.default_rng(calib.seed)
    k_best = float(calib.es_k0) if calib.es_k0 else 1.0 / base_vector.mean()
    step = float(calib.es_step0)

    print(f"=== ES: initial guess k = {k_best:.4f} ===")
    best_row = evaluate_k(k_best, base_vector, counts, target_by_bucket,
                          chart_mask, bucket_labels, calib)
    rows = [best_row]
    if on_row:
        on_row(best_row)

    for i in range(calib.es_iters):
        k_try = k_best * float(np.exp(step * rng.standard_normal()))
        print(f"\n--- ES iteration {i + 1}/{calib.es_iters}  "
              f"(current best k={k_best:.4f}, step={step:.3f}) ---")
        row = evaluate_k(k_try, base_vector, counts, target_by_bucket,
                         chart_mask, bucket_labels, calib)
        rows.append(row)
        if on_row:
            on_row(row)

        improved = row["score_rms_ratio_minus_1"] < best_row["score_rms_ratio_minus_1"]
        if improved:
            k_best, best_row, step = k_try, row, step * calib.es_grow
            print(f"  IMPROVED ({row['score_rms_ratio_minus_1']:.4f} < "
                  f"previous best) -> step x{calib.es_grow:g} = {step:.3f}")
        else:
            step *= calib.es_shrink
            print(f"  rejected ({row['score_rms_ratio_minus_1']:.4f} >= "
                  f"best {best_row['score_rms_ratio_minus_1']:.4f}) -> "
                  f"step x{calib.es_shrink:g} = {step:.3f}")

    return rows, best_row


def main():
    calib = load_calib_config()
    print(f"config: {OmegaConf.to_yaml(calib)}")
    if calib.n_agents >= 1000:
        print(f"NOTE: n_agents={calib.n_agents} is well past the n=200 the "
              f"~3 min/cell estimate elsewhere is based on -- time the first "
              f"evaluation before trusting any total-runtime estimate for "
              f"the rest of the {'search' if calib.mode == 'es' else 'grid'}.")

    buckets = load_target_buckets()
    counts = bucket_agent_counts(buckets, calib.n_agents)
    base_vector = build_base_vector(buckets, counts)
    target_by_bucket = np.array([b[2] for b in buckets])
    chart_mask = np.array([b[3] == "chart" for b in buckets])
    bucket_labels = [f"{int(lo)}-{int(hi)}" for lo, hi, _, _ in buckets]

    out_dir = Path(calib.out_dir)
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    import pandas as pd
    rows = []

    def checkpoint(row):
        """Overwrite results.csv after every evaluation -- a long ES/grid
        run (hours at n_agents=1000) shouldn't lose everything to a
        disconnect or Ctrl-C partway through."""
        rows.append(row)
        pd.DataFrame(rows).to_csv(out_dir / "results.csv", index=False)

    if calib.mode == "es":
        _, best_row = es_search(base_vector, counts, target_by_bucket,
                                chart_mask, bucket_labels, calib, on_row=checkpoint)
    elif calib.mode == "grid":
        for k in [float(x) for x in calib.k_grid]:
            checkpoint(evaluate_k(k, base_vector, counts, target_by_bucket,
                                  chart_mask, bucket_labels, calib))
        best_row = min(rows, key=lambda r: r["score_rms_ratio_minus_1"])
    else:
        raise ValueError(f"unknown mode: {calib.mode!r} (expected 'es' or 'grid')")

    df = pd.DataFrame(rows)
    print(f"\nbest k_multiplier = {best_row['k_multiplier']:.4f} "
          f"(score={best_row['score_rms_ratio_minus_1']:.4f}) -> {out_dir/'results.csv'}")

    _plot_results(df, bucket_labels, target_by_bucket, chart_mask, out_dir)


def _plot_results(df, bucket_labels, target_by_bucket, chart_mask, out_dir):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # Colored by evaluation order rather than connected in k-order: the ES
    # search doesn't visit k monotonically, so a line would be misleading;
    # the color progression still shows the search converging over time.
    sc = ax1.scatter(df["k_multiplier"], df["score_rms_ratio_minus_1"],
                     c=np.arange(len(df)), cmap="viridis")
    best_idx = df["score_rms_ratio_minus_1"].idxmin()
    ax1.scatter(df["k_multiplier"][best_idx], df["score_rms_ratio_minus_1"][best_idx],
               marker="*", s=250, facecolors="none", edgecolors="red", linewidths=1.5)
    fig.colorbar(sc, ax=ax1, label="evaluation order")
    ax1.set_xlabel("k_multiplier"); ax1.set_ylabel("RMS |simulated/target - 1|")
    ax1.set_title("Calibration score vs. kappa scale")

    best = df.loc[best_idx]
    x = np.arange(len(bucket_labels))
    sim = np.array([best[f"sim_return_pct[{lbl}]"] for lbl in bucket_labels])
    ax2.plot(x, target_by_bucket, "o-", label="target (Xavier 2021)")
    ax2.plot(x, sim, "s--", label=f"simulated (k={best['k_multiplier']:g})")
    if (~chart_mask).any():
        ax2.plot(x[~chart_mask], target_by_bucket[~chart_mask], "x", color="gray",
                markersize=10, label="excluded from fit score")
    ax2.set_xticks(x, bucket_labels, rotation=45, fontsize=7)
    ax2.set_ylabel("annual return (%)")
    ax2.set_title("Best-fit return-by-wealth-percentile curve")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "calibration_fit.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"figure -> {out_dir / 'calibration_fit.png'}")


if __name__ == "__main__":
    main()
