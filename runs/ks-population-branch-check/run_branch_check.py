"""Population-size branch check: does n_agents/num_envs change which Gini
regime a homogeneous (kappas=null) KS training run lands in?

Context. `runs/ks-correctness/ks_n200_top/` (n_agents=200, num_envs=32,
kappas=null) settled into a persistent bimodal, moderate-high-Gini (~0.42)
wealth distribution after an early training-time Gini spike. The new
`runs/ks-wealth-lognormal-random/` sweep uses n_agents=500, num_envs=8 (kept
from the `ks-wealth-calibration` sibling's own batch-width choice) and its
sigma=0 cells (5 seeds tried) all landed in a much lower-Gini (~0.02-0.09)
regime instead -- while every sigma>=0.04 cell in that same sweep landed
back in the ks_n200_top-like high-Gini regime. That is consistent with two
things both being true at once: (a) there being two distinct basins a
training run can settle into even at exactly zero heterogeneity, and (b) the
n_agents/num_envs choice changing which basin is typical.

This script isolates population size as the only varying factor: same
base_exp, same kappas=null (homogeneous), same total_timesteps, several
seeds at each of the two population/batch-width combinations already in
play elsewhere in this repo:

    n200_e32  n_agents=200, num_envs=32   (matches ks_n200_top exactly)
    n500_e8   n_agents=500, num_envs=8    (matches ks-wealth-lognormal-random's default)

`diag.n_snapshots` is raised (12, matching ks_n200_top) so each run records
the full Gini-over-training trajectory, not just the final checkpoint --
enough to see whether/when a run passes through the transient spike and
which level it settles at afterwards, the same view ks_n200_top's own
figures gave.

Each run goes through the standard `jmbc.run` CLI path (same driver that
produced every `runs/ks-correctness/*` folder), so outputs land in the usual
<out_dir>/ks/<run_name>/ layout: config.yaml, diagnostics.json, metrics.csv,
rollouts.npz, timing.json, figures/.

Usage:
    python run_branch_check.py
    python run_branch_check.py seeds=[0,1] device=cpu n_snapshots=3   # quick smoke test
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))  # jmbc isn't pip-installed in this repo yet

DEFAULT_CONFIG_PATH = HERE / "config.yaml"

# The two population/batch-width combinations being compared -- see module
# docstring for why exactly these two.
ARMS = [
    {"name": "n200_e32", "n_agents": 200, "num_envs": 32},
    {"name": "n500_e8", "n_agents": 500, "num_envs": 8},
]


@dataclass
class CheckConfig:
    base_exp: str = "ks_n200"     # configs/exp/ks_n200.yaml -- total_timesteps
                                   # and every other economic/training
                                   # hyperparameter come from here, unchanged,
                                   # for both arms
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    n_snapshots: int = 12          # full Gini-over-training trajectory, not
                                    # just the final checkpoint -- matches
                                    # ks_n200_top
    sim_steps: int = 5000
    total_timesteps: Optional[int] = None   # None -> base_exp's own budget
                                             # (128,000, unchanged, for both
                                             # arms); override only for a
                                             # quick smoke test
    device: str = "gpu"
    out_dir: str = "results"


def load_config(argv: Optional[List[str]] = None) -> CheckConfig:
    """structured defaults < config.yaml (or ``config=<path>``) < CLI dotlist
    -- identical merge order to jmbc.config.load_config."""
    argv = sys.argv[1:] if argv is None else argv
    dotlist = [a for a in argv if "=" in a and not a.startswith("config=")]
    config_arg = next((a.split("=", 1)[1] for a in argv if a.startswith("config=")), None)
    config_path = Path(config_arg) if config_arg else DEFAULT_CONFIG_PATH

    cfg = OmegaConf.structured(CheckConfig)
    if config_path.exists():
        cfg = OmegaConf.merge(cfg, OmegaConf.load(config_path))
    if dotlist:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(dotlist))
    return cfg


def main():
    cfg = load_config()
    print(f"config: {OmegaConf.to_yaml(cfg)}")
    n_runs = len(ARMS) * len(cfg.seeds)
    print(f"NOTE: {len(ARMS)} arms x {len(cfg.seeds)} seeds = {n_runs} runs -- "
          f"time the first run before trusting a total-runtime estimate for "
          f"the rest.")

    out_dir = Path(cfg.out_dir)
    if not out_dir.is_absolute():
        out_dir = HERE / out_dir

    from jmbc.run import main as jmbc_run_main

    for arm in ARMS:
        for seed in [int(s) for s in cfg.seeds]:
            run_name = f"{arm['name']}_seed{seed}"
            print(f"\n=== {run_name}  (n_agents={arm['n_agents']}, "
                  f"num_envs={arm['num_envs']}, seed={seed}) ===")
            overrides = [
                f"exp={cfg.base_exp}",
                f"env.n_agents={arm['n_agents']}",
                f"train.num_envs={arm['num_envs']}",
                f"diag.n_snapshots={cfg.n_snapshots}",
                f"diag.sim_steps={cfg.sim_steps}",
                f"diag.n_agents={arm['n_agents']}",
                f"run.seed={seed}",
                f"run.device={cfg.device}",
                f"log.out_dir={out_dir}",
                f"log.run_name={run_name}",
                "log.save_raw=true",
                f"log.save_agents={arm['n_agents']}",
            ]
            if cfg.total_timesteps:
                overrides.append(f"train.total_timesteps={cfg.total_timesteps}")
            jmbc_run_main(overrides)


if __name__ == "__main__":
    main()
