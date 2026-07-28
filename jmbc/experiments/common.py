"""Shared single-run machinery used by every driver and by the sweep runner.

``run_single`` builds the env, trains, times the run, computes diagnostics, and
(optionally) writes structured outputs + standard figures. Bespoke per-figure
work is left to the drivers, which reuse the rollouts returned here.
"""
from __future__ import annotations

from typing import Optional

import jax

from .. import plots
from ..algos import make_train
from ..config.schema import to_train_dict
from ..diagnostics import compute_diagnostics, metrics_to_numpy
from ..envs import build_env
from ..recorder import RunRecorder, run_and_time, benchmark_time, phase


def _k_init_desc(env_cfg) -> str:
    """One-line description of the starting-capital distribution."""
    d = env_cfg.k_init_dist
    if d == "constant":
        return f"{env_cfg.k_init:g} (constant)"
    spec = (f"U({env_cfg.k_init_low:g}, {env_cfg.k_init_high:g})" if d == "uniform"
            else f"LN(mean={env_cfg.k_init:g}, sigma={env_cfg.k_init_sigma:g})")
    return f"{spec} per agent, resampled {env_cfg.k_init_resample}"


def _print_launch_summary(cfg, env, train_fn, seed: int) -> None:
    """What is ACTUALLY about to run: values pulled from the built env, the
    derived training config and the initialized network — not the YAML."""
    import jax
    import numpy as np

    c = train_fn.config
    E, R, n = c["NUM_ENVS"], c["ROLLOUT_LEN"], env.num_agents
    seq_steps = c["NUM_UPDATES"] * R
    params = train_fn.network.init(
        jax.random.PRNGKey(0), jax.numpy.zeros((env.obs_dim,)))
    n_params = sum(int(np.prod(p.shape)) for p in jax.tree_util.tree_leaves(params))
    lr = f"{c['LR']:g}" + (" (annealed)" if c.get("ANNEAL_LR") else " (constant)")
    iface = "vector [n,d] arrays" if c.get("VEC_INTERFACE") else "jaxmarl per-agent dicts"

    # Device-memory forecast (fp32). Peak liveness in the update phase holds
    # ~4 full-batch tensors at once: stacked trajectory, flattened+shuffled
    # copy (the big single allocation that OOMs first), minibatch reshape,
    # and gather workspace. Per sample: obs + action + (value, reward,
    # log_prob, done, advantage, target, info) ~ 7 scalars.
    per_sample = (env.obs_dim + getattr(env, "act_dim", 1) + 7) * 4
    buf_gb = 4 * E * R * n * per_sample / 1e9
    upd_gb = c["MINIBATCH_SIZE"] * (sum(c["HIDDEN_DIMS"]) + 8) * 4 * 4 / 1e9
    mem_txt = f"est. device ~{buf_gb + upd_gb:.2f} GB (buffer {buf_gb:.2f} + update {upd_gb:.2f})"
    try:
        limit = jax.local_devices()[0].memory_stats().get("bytes_limit")
        if limit:
            pct = 100 * (buf_gb + upd_gb) / (limit / 1e9)
            mem_txt += f" | device {limit / 1e9:.1f} GB -> ~{pct:.0f}%"
            if pct > 80:
                mem_txt += "  ** >80%: likely OOM, raise num_minibatches / lower num_envs **"
    except Exception:
        pass

    rows = [
        ("economy", f"{cfg.env.kind} | {n} agents | obs[{env.obs_dim}]: "
                    f"{', '.join(env.obs_vars)}"),
        ("",        f"alpha {cfg.env.alpha} | beta {cfg.env.beta} | delta {cfg.env.delta}"
                    f" | episode {env.max_steps} steps | k_0 {_k_init_desc(cfg.env)}"),
        ("training", f"{c['NUM_UPDATES']:,} updates = {seq_steps:,} sequential steps"
                     f" | {E} envs x {n} agents -> {seq_steps * E * n:,} transitions"),
        ("",         f"batch/update {R * E * n:,} | minibatch {c['MINIBATCH_SIZE']:,}"
                     f" x {c['NUM_MINIBATCHES']} | {c['UPDATE_EPOCHS']} epochs"
                     f" | actors {c['NUM_ACTORS']:,}"),
        ("",         f"lr {lr} | gamma {c['GAMMA']} | clip {c['CLIP_EPS']}"
                     f" | ent {c['ENT_COEF']} | vf {c['VF_COEF']}"),
        ("network",  f"hidden {list(c['HIDDEN_DIMS'])} {c['ACTIVATION']}"
                     f" | {n_params:,} params | interface: {iface}"),
        ("memory",   mem_txt),
        ("diag",     f"{cfg.diag.n_snapshots} snapshots x {cfg.diag.sim_steps:,}"
                     f" eval steps (reset-free, chunked) | burn_frac {cfg.diag.burn_frac}"),
        ("run",      f"seed {seed} | backend {jax.default_backend()}"
                     f" | heartbeat every {c.get('LOG_EVERY', 0)} updates"),
    ]
    width = 74
    print("┌─ launch " + "─" * (width - 9))
    for label, text in rows:
        print(f"│ {label:<9}{text}")
    print("└" + "─" * width)


def run_single(
    cfg,
    env=None,
    recorder: Optional[RunRecorder] = None,
    seed: Optional[int] = None,
    do_diagnostics: bool = True,
    do_figures: bool = True,
    benchmark: bool = False,
    diag_seed: int = 7,
) -> dict:
    seed = int(cfg.run.seed) if seed is None else int(seed)
    if env is None:
        env = build_env(cfg.env)

    train_fn = make_train(env, to_train_dict(cfg))
    _print_launch_summary(cfg, env, train_fn, seed)
    rng = jax.random.PRNGKey(seed)
    timer = benchmark_time if benchmark else run_and_time
    out, timing = timer(train_fn, rng)

    cfgd = train_fn.config
    net = train_fn.network
    steps_per_update = cfgd["ROLLOUT_LEN"] * cfgd["NUM_ENVS"]

    metrics_np = metrics_to_numpy(out["metrics"])

    summary, recs, idxs = None, None, None
    if do_diagnostics:
        phase(f"diagnostics: {cfg.diag.n_snapshots} snapshots x "
              f"{cfg.diag.sim_steps}-step reset-free eval rollouts ...")
        summary, recs, idxs = compute_diagnostics(
            env, cfg.env, net, out["params_history"], cfg.diag, diag_seed
        )
        phase("diagnostics done")

    if recorder is not None:
        phase(f"writing outputs -> {recorder.dir}")
        recorder.save_config(cfg)
        recorder.save_metrics(metrics_np)
        recorder.save_timing(timing)
        if summary is not None:
            recorder.save_diagnostics(summary)
        if do_figures:
            plots.plot_training_health(metrics_np, recorder.figure_path("training_health.png"))
            if summary is not None:
                plots.plot_economic_snapshots(
                    summary, cfg.env.kind, steps_per_update,
                    recorder.figure_path("economic.png"),
                )
                plots.plot_distributional_snapshots(
                    summary, steps_per_update, recorder.figure_path("distributional.png"),
                )

    return {
        "out": out,
        "metrics_np": metrics_np,
        "timing": timing,
        "summary": summary,
        "recs": recs,
        "idxs": idxs,
        "net": net,
        "env": env,
        "config": cfgd,
        "steps_per_update": steps_per_update,
    }
