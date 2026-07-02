"""Unified experiment CLI.

    python -m jmbc.run exp=ks env.n_agents=1000 train.num_envs=64 run.device=cpu

``exp=<name>`` selects configs/exp/<name>.yaml; everything else is an OmegaConf
dotlist override. The device is resolved before JAX is imported.
"""
from __future__ import annotations

import sys
from typing import Optional, Sequence

from .config import load_config, parse_cli, setup_device


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    exp, overrides = parse_cli(argv, key="exp")
    if exp is None:
        print("usage: python -m jmbc.run exp=<rbc|ks|general> [key=value ...]")
        raise SystemExit(2)

    cfg = load_config(exp, overrides)
    setup_device(cfg.run.device)  # must precede any jax import

    # Import jax-dependent code only after the device is fixed.
    from .experiments import get_driver
    from .recorder import _timestamp

    run_id = cfg.log.run_name or _timestamp()
    print(f"== jmbc.run exp={cfg.exp} run_id={run_id} device={cfg.run.device} ==")
    driver = get_driver(cfg.exp)
    driver(cfg, cfg.log.out_dir, run_id)


if __name__ == "__main__":
    main()
