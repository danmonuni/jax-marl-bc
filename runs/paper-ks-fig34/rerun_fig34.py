"""One KS run, recorded densely enough to redraw paper figures 3 and 4.

Reruns a single cell of the lognormal-kappa sweep behind figure 6 (kept
locally, reproducible from ``configs/``) -- by default its ``sigma=0.0
seed=8`` cell -- and keeps the WHOLE training trajectory instead of only the
trained policy's rollout: the policy is evaluated at ``n_snapshots`` training
updates and every step of every one of those rollouts is persisted.

Why a separate experiment rather than a replot of the sweep's output: the
sweep saves one rollout per cell, the trained policy's. Figures 3 and 4 are
about how the economy CHANGES over training (untrained vs trained law of
motion, wealth distribution through training), so they need the snapshots the
sweep never recorded. Everything else -- economy, hyperparameters, kappa
construction, capital initialization -- is the sweep cell's, declared
explicitly in ``config.yaml`` and re-asserted by ``verify_protocol()`` before
training, so it cannot drift.

The one deliberate difference from ``jmbc.run``'s own diagnostics: snapshots
are spaced LINEARLY over training. The stock loop
(``jmbc/diagnostics/report.py:20``) is log-spaced, which is right for a dozen
snapshots spanning orders of magnitude and wrong for an evenly sampled
picture of training.

Usage (same config.yaml + CLI-dotlist convention as the sibling experiments):
    python runs/paper-ks-fig34/rerun_fig34.py
    python runs/paper-ks-fig34/rerun_fig34.py device=cpu n_snapshots=4
    python runs/paper-ks-fig34/rerun_fig34.py sigma=0.4 seed=3   # any sweep cell
    python runs/paper-ks-fig34/rerun_fig34.py protocol.train.num_envs=8

Writes ``results/ks/sigma_<S>_seed_<K>/``:
    config.yaml     the resolved config -- the record states its own protocol
    metrics.csv     per-update training metrics
    timing.json     trace/compile/run split, throughput, device
    rollouts.npz    every channel, snapshots stacked on a leading axis
    kappas.npy      the cell's kappa vector
    params.msgpack, network.pkl   the trained policy, to resimulate later
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # jmbc isn't pip-installed in this repo yet

DEFAULT_CONFIG_PATH = HERE / "config.yaml"

#: The sweep cell's resolved hyperparameters, spelled out rather than
#: inherited from base_exp -- same discipline (and same values) as
#: ``sweep_lognormal_random_3.py``'s PROTOCOL. Nested, not dotted-flat, so a
#: CLI dotlist override addresses an entry naturally.
PROTOCOL: Dict[str, Any] = {
    "env": {
        "n_agents":        200,
        "alpha":           0.36,
        "beta":            0.95,
        "delta":           0.025,
        "max_steps":       200,
    },
    "train": {
        "num_envs":        12,
        "rollout_len":     200,
        "total_timesteps": 128000,   # sequential steps = 640 updates x 200
        "update_epochs":   4,
        "num_minibatches": 64,
        "lr":              3.0e-4,
        "gamma":           0.95,
        "gae_lambda":      0.95,
        "clip_eps":        0.2,
        "vf_coef":         0.5,
        "ent_coef":        0.0,
        "anneal_lr":       True,
        "max_grad_norm":   0.5,
    },
    "net": {
        "activation":      "tanh",
    },
}


@dataclass
class RunConfig:
    base_exp: str = "ks_n200"      # supplies only what `protocol` does not pin
    protocol: Dict[str, Any] = field(default_factory=lambda: dict(PROTOCOL))
    # The cell. sigma is the kappa log-dispersion of the sweep, NOT env.sigma
    # (which is the RBC TFP innovation and inert in KS); sigma = 0 is the
    # homogeneous baseline, kappa_i = 1 for every i.
    sigma: float = 0.0
    seed: int = 8
    # The record -- the reason this experiment exists.
    n_snapshots: int = 30          # evaluations, spaced LINEARLY over training
    sim_steps: int = 200           # steps per evaluation (the cell's own value)
    device: str = "gpu"
    out_dir: str = "results"
    save_policy: bool = True       # keep params/network to resimulate later
    # Starting capital: as the sweep ran it.
    k_init_dist: str = "uniform"       # "constant" | "uniform" | "lognormal"
    k_init_low: float = 10.0
    k_init_high: float = 70.0
    k_init_resample: str = "per_episode"


def load_run_config(argv: Optional[List[str]] = None) -> RunConfig:
    """structured defaults < config.yaml (or ``config=<path>``) < CLI dotlist
    -- identical merge order to jmbc.config.load_config."""
    argv = sys.argv[1:] if argv is None else argv
    dotlist = [a for a in argv if "=" in a and not a.startswith("config=")]
    config_arg = next((a.split("=", 1)[1] for a in argv if a.startswith("config=")), None)
    config_path = Path(config_arg) if config_arg else DEFAULT_CONFIG_PATH

    cfg = OmegaConf.structured(RunConfig)
    if config_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def build_kappas(n_agents: int, sigma: float, seed: int) -> np.ndarray:
    """I.i.d. LogNormal(mu, sigma) draw, mean renormalized to 1.

    Copied from the sweep so a cell reproduced here is bit-for-bit the cell it
    ran: ``mu = -sigma^2/2`` keeps E[kappa] = 1, the finite-n draw is then
    renormalized by its empirical mean, and ``sigma <= 0`` short-circuits to
    the exact homogeneous baseline (no seed dependence -- nothing to sample).
    """
    if sigma <= 0:
        return np.ones(n_agents, np.float64)
    rng = np.random.default_rng(seed)
    kappas = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma, size=n_agents)
    return kappas / kappas.mean()


def flatten(d, prefix: str = "") -> Dict[str, Any]:
    """{'train': {'lr': 3e-4}} -> {'train.lr': 3e-4} for dotlist overrides."""
    out: Dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if hasattr(v, "items"):
            out.update(flatten(v, key + "."))
        else:
            out[key] = v
    return out


def _fmt(v) -> str:
    """OmegaConf dotlist rendering (bools lowercase, floats unabbreviated)."""
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def build_overrides(run: RunConfig) -> List[str]:
    """Every override applied to base_exp, in one place.

    PROTOCOL first (the full single-run spec), then the initialization, the
    seed and the recording -- so the deliberate choices are visible as exactly
    that, rather than buried among inherited defaults.
    """
    return (
        [f"{k}={_fmt(v)}" for k, v in flatten(run.protocol).items()]
        + [
            f"env.k_init_dist={run.k_init_dist}",
            f"env.k_init_low={run.k_init_low}",
            f"env.k_init_high={run.k_init_high}",
            f"env.k_init_resample={run.k_init_resample}",
            f"run.seed={int(run.seed)}",
            f"run.device={run.device}",
            f"diag.sim_steps={int(run.sim_steps)}",
            f"diag.n_snapshots={int(run.n_snapshots)}",   # provenance only:
            # the snapshot indices are chosen below, not by jmbc's diag loop
            "log.save_raw=true",
            f"log.save_agents={int(run.protocol['env']['n_agents'])}",
        ]
    )


def verify_protocol(cfg, run: RunConfig) -> None:
    """Fail loudly if the resolved config disagrees with PROTOCOL.

    The overrides above should make this impossible, which is the point: it is
    a standing assertion that this really is the sweep's cell, so a future edit
    to configs/exp/ks*.yaml (or to the schema) cannot quietly change what runs.
    """
    bad = []
    for key, want in flatten(run.protocol).items():
        got = OmegaConf.select(cfg, key)
        if str(got) != str(want):
            bad.append(f"  {key}: resolved {got!r}, protocol says {want!r}")
    if bad:
        raise SystemExit("resolved config does not match the declared "
                         "protocol:\n" + "\n".join(bad))


def snapshot_indices(num_updates: int, n_snapshots: int) -> np.ndarray:
    """Linearly spaced update indices, first and last always included.

    Contrast ``jmbc.diagnostics.report.snapshot_indices``, which is log-spaced.
    """
    n = int(np.clip(n_snapshots, 1, num_updates))
    return np.unique(np.linspace(0, num_updates - 1, n).round().astype(int))


def run_cell(run: RunConfig):
    """Train the cell, then evaluate every snapshot. -> (cfg, recs, idxs, ...)"""
    from jmbc.config import load_config, setup_device, to_train_dict

    n_agents = int(run.protocol["env"]["n_agents"])
    kappas = build_kappas(n_agents, float(run.sigma), int(run.seed))
    cfg = load_config(run.base_exp, build_overrides(run))
    verify_protocol(cfg, run)
    if float(run.sigma) > 0:                     # homogeneous stays `null`,
        cfg.env.kappas = kappas.tolist()         # which builds ones()
    setup_device(cfg.run.device, bool(cfg.run.prealloc))  # before jax import

    import copy
    import jax
    from jmbc.algos import make_train
    from jmbc.diagnostics import metrics_to_numpy
    from jmbc.diagnostics.report import params_at
    from jmbc.diagnostics.rollout import simulate
    from jmbc.envs import build_env
    from jmbc.experiments.common import _print_launch_summary
    from jmbc.recorder import device_report, phase, run_and_time

    print(device_report(cfg.run.device))
    env = build_env(cfg.env)
    train_fn = make_train(env, to_train_dict(cfg))
    _print_launch_summary(cfg, env, train_fn, int(cfg.run.seed))

    phase("training ...")
    out, timing = run_and_time(train_fn, jax.random.PRNGKey(int(run.seed)))

    # Evaluation env: reset-free over the whole rollout, exactly how jmbc's own
    # diagnostics build it (report.py:55) -- restated because that code path is
    # entangled with the log-spaced snapshot selection this script replaces.
    sim_steps = int(run.sim_steps)
    eval_cfg = copy.deepcopy(cfg.env)
    eval_cfg.max_steps = max(sim_steps + 1, int(cfg.env.max_steps))
    eval_env = build_env(eval_cfg)

    params_history = out["params_history"]
    num_updates = int(jax.tree_util.tree_leaves(params_history)[0].shape[0])
    idxs = snapshot_indices(num_updates, int(run.n_snapshots))
    phase(f"recording {len(idxs)} of {num_updates} updates (linearly spaced) "
          f"x {sim_steps}-step reset-free eval rollouts ...")

    key = jax.random.PRNGKey(7)          # the diag seed every jmbc run uses
    recs = []
    for i, idx in enumerate(idxs):
        if i % 10 == 0 or i == len(idxs) - 1:
            phase(f"  snapshot {i + 1}/{len(idxs)} "
                  f"(update {int(idx) + 1}/{num_updates})")
        recs.append(simulate(eval_env, train_fn.network,
                             params_at(params_history, int(idx)), key, sim_steps))
    phase("recording done")

    steps_per_update = int(train_fn.config["ROLLOUT_LEN"]
                           * train_fn.config["NUM_ENVS"])
    return {
        "cfg": cfg, "recs": recs, "idxs": idxs, "kappas": kappas,
        "timing": timing, "steps_per_update": steps_per_update,
        "metrics": metrics_to_numpy(out["metrics"]),
        "params": out["params"], "network": train_fn.network,
    }


def save(res: dict, run: RunConfig, out_root: Path, tag: str) -> Path:
    """Persist the run under ``<out_root>/ks/<tag>/``."""
    import pickle

    from flax import serialization
    from jmbc.recorder import RunRecorder, phase

    rec = RunRecorder(str(out_root), "ks", tag)
    phase(f"writing outputs -> {rec.dir}")
    rec.save_config(res["cfg"])
    rec.save_metrics(res["metrics"])
    rec.save_timing(res["timing"])
    rec.save_rollouts(res["recs"], res["idxs"], res["steps_per_update"],
                      max_agents=int(res["cfg"].log.save_agents))
    np.save(rec.dir / "kappas.npy", res["kappas"])
    with open(rec.dir / "meta.json", "w") as f:
        json.dump({"sigma": float(run.sigma), "seed": int(run.seed),
                   "delta": float(res["cfg"].env.delta),
                   "n_snapshots": len(res["idxs"]),
                   "sim_steps": int(run.sim_steps),
                   "snapshot_spacing": "linear"}, f, indent=2)
    if run.save_policy:
        with open(rec.dir / "params.msgpack", "wb") as f:
            f.write(serialization.to_bytes(res["params"]))
        with open(rec.dir / "network.pkl", "wb") as f:
            pickle.dump(res["network"], f)
    size_mb = (rec.dir / "rollouts.npz").stat().st_size / 1e6
    phase(f"raw rollouts saved: rollouts.npz ({size_mb:.1f} MB)")
    return rec.dir


def steady_state_capital(rec, window: int) -> np.ndarray:
    """Per-agent capital averaged over the last ``window`` steps, auto-reset
    steps excluded -- the quantity figure 3's histogram bins."""
    ks, done = rec["ks"][-window:], rec["done"][-window:].astype(bool)
    return ks[~done].mean(axis=0) if not done.all() else ks.mean(axis=0)


def main() -> None:
    run = load_run_config()
    print(f"config: {OmegaConf.to_yaml(run)}")
    print("single-run protocol (= the sweep cell; asserted by verify_protocol):")
    for k, v in flatten(run.protocol).items():
        print(f"  {k:<24} {v}")
    print(f"  {'env.k_init_dist':<24} {run.k_init_dist} "
          f"({run.k_init_low:g}, {run.k_init_high:g}) resampled {run.k_init_resample}")
    print(f"  {'env.kappas':<24} "
          f"{'homogeneous (kappa_i = 1)' if run.sigma <= 0 else f'LogNormal(sigma={run.sigma})'}")
    print(f"RECORD: {run.n_snapshots} snapshots, LINEARLY spaced over training "
          f"(jmbc's own diagnostics loop is log-spaced), {run.sim_steps} "
          f"reset-free steps each, every agent kept.\n")

    out_root = Path(run.out_dir)
    if not out_root.is_absolute():
        out_root = HERE / out_root

    tag = f"sigma_{float(run.sigma):.2f}_seed_{int(run.seed)}"
    res = run_cell(run)
    run_dir = save(res, run, out_root, tag)

    from jmbc.diagnostics import gini
    k_bar = steady_state_capital(res["recs"][-1], min(50, int(run.sim_steps)))
    print(f"\ndone -> {run_dir}\n"
          f"  {len(res['idxs'])} snapshots x {run.sim_steps} steps x "
          f"{res['recs'][0]['ks'].shape[-1]} agents"
          f" | training {res['timing'].get('run_time_s', float('nan')):.1f}s\n"
          f"  trained (last 50 steps): capital Gini={gini(k_bar):.3f}  "
          f"K_mean={k_bar.mean():.3f}")


if __name__ == "__main__":
    main()
