# JaxMARL-BC

Multi-agent reinforcement learning for macroeconomic models — Real Business
Cycle (RBC), heterogeneous RBC, and Krusell-Smith — on a **JaxMARL-compatible**
interface, trained with a pure-JAX PPO loop.

The package provides:

- a **typed, single-source-of-truth configuration** system (dataclasses +
  OmegaConf YAML) with CLI overrides;
- **rich economic & distributional diagnostics** that probe whether the trained
  policy behaves like a correct, realistic economy (Euler-equation errors,
  resource-constraint residual, analytical-solution comparison, Krusell-Smith
  forecasting R² and the Den Haan statistic, Gini / Lorenz / top shares);
- a **meta experiment / benchmark runner** that sweeps configurations across the
  three designs and produces publication-quality scaling/throughput figures.

## Install

```bash
pip install -e .
# or, on a Colab T4 GPU instance:
pip install -r requirements.txt   # uses jax[cuda12]
```

## Run an experiment

```bash
python -m jmbc.run exp=rbc                       # textbook + typical RBC
python -m jmbc.run exp=ks                         # Krusell-Smith
python -m jmbc.run exp=general                    # heterogeneous RBC grid

# Override anything via OmegaConf dotlist:
python -m jmbc.run exp=ks env.n_agents=1000 train.num_envs=64 run.device=gpu
```

`exp=<name>` selects `configs/exp/<name>.yaml`. The device (`run.device`:
`auto` | `cpu` | `gpu`) is resolved *before* JAX is imported; `auto` picks the
GPU on Colab and CPU otherwise.

### Outputs

Every run writes a self-contained, reproducible directory:

```
results/<exp>/<run_id>/
  config.yaml          # fully resolved configuration
  metrics.csv          # per-update training metrics (loss, KL, clip frac, ...)
  diagnostics.json     # economic + distributional probes across snapshots
  timing.json          # wall time, throughput, device
  figures/
    training_health.png      # PPO convergence panel
    economic.png             # Euler error (+ KS R²/Den Haan) vs steps
    distributional.png       # Gini & top shares vs steps
    <bespoke>.png            # rbc policy / ks_fig4 / figure5
```

## Sweep / benchmark

```bash
python -m jmbc.sweep sweep=scaling          # n_agents x num_envs scaling
python -m jmbc.sweep sweep=scaling_smoke     # quick end-to-end check
```

Writes `benchmarks/<name>/results.csv` (one row per cell, with a `method`
column so the original implementation can be appended and overlaid as
"standard vs JaxMARL-BC") and throughput / wall-time scaling figures. Timing
separates JIT **compile time** from steady-state **run time** so throughput is
measured fairly.

Define your own sweep in `configs/sweep/<name>.yaml`:

```yaml
name: my_sweep
base_exp: ks
axes:
  env.n_agents: [10, 100, 1000, 10000]
  train.num_envs: [32, 128]
overrides:
  train.total_timesteps: 50000   # sequential env steps, independent of num_envs
repeats: 1
collect_diagnostics: true   # also tabulate Euler / Gini per cell
save_cell_runs: false       # true -> full per-cell run dir + figures
                            # (wealth distribution, Euler panel, training health)
                            # under benchmarks/<name>/cells/
```

## Package layout

```
jmbc/
  config/      schema.py (dataclasses), loader.py (YAML + CLI merge)
  envs/        RBCKLEnv, RBCKSEnv (JaxMARL MultiAgentEnv), registry.build_env
  algos/       ActorCritic, make_train (PPO + in-loop health metrics)
  diagnostics/ rollout, economic, distributional, report (snapshots)
  plots/       style, training, figures (bespoke), benchmark (scaling)
  experiments/ rbc, ks, general drivers + common.run_single
  recorder.py  structured run output + timing helpers
  run.py       experiment CLI        sweep.py  meta/benchmark CLI
configs/       base.yaml, exp/*.yaml, sweep/*.yaml
notebooks/     quickstart.ipynb
```

## Notebook

`notebooks/quickstart.ipynb` walks through loading a config, training,
inspecting diagnostics inline, and running a mini scaling sweep.
