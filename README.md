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
python -m jmbc.run exp=ks                         # Krusell-Smith (full budget, Colab)
python -m jmbc.run exp=ks_local                   # same KS economy, ~5 min on CPU
python -m jmbc.run exp=general                    # heterogeneous RBC grid

# Override anything via OmegaConf dotlist:
python -m jmbc.run exp=ks env.n_agents=1000 train.num_envs=64 run.device=gpu
```

`exp=<name>` selects `configs/exp/<name>.yaml`; an exp file may declare
`extends: <other>` to inherit it and override only what differs (`ks_local`
is `ks` with a 5-minute training budget). The device (`run.device`:
`auto` | `cpu` | `gpu`) is resolved *before* JAX is imported; `auto` picks the
GPU on Colab and CPU otherwise.

### Outputs

Every run writes a self-contained, reproducible directory:

```
runs/<exp>/<run_id>/
  config.yaml          # fully resolved configuration
  metrics.csv          # per-update training metrics (loss, KL, clip frac, ...)
  diagnostics.json     # economic + distributional probes across snapshots
  timing.json          # wall time, throughput, device
  rollouts.npz         # RAW snapshot rollouts: every recorded channel
                       # (capital, wealth, consumption, MPC, employment,
                       # aggregate state, ...) at each training snapshot
  figures/
    training_health.png      # PPO convergence panel
    economic.png             # Euler error (+ KS R²/Den Haan) vs steps
    distributional.png       # Gini & top shares vs steps
    ks_lom_evolution.png     # aggregate law of motion through training
    ks_wealth_heatmap.png    # wealth distribution vs training (density heatmap)
    ks_mpc.png               # MPC by wealth/employment + through training
    <bespoke>.png            # rbc policy / ks_fig4 / figure5
```

`rollouts.npz` is the complete experimental record: every figure and every
diagnostic can be recomputed **ex post, without retraining and without a GPU**:

```bash
python -m jmbc.analyze runs/ks/<run_id>          # rebuild diagnostics + figures
```

The intended loop: train on Colab (GPU) or locally (CPU), sync the run
directory, then iterate on diagnostics/figures locally against the saved data.

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

## Repository map

Everything you edit lives in `jmbc/` + `configs/`; everything generated lands
in `runs/` (single experiments) or `benchmarks/` (sweeps), both git-ignored.

```
configs/                 WHAT to run
  base.yaml                shared defaults
  exp/{rbc,ks,general}     one file per experiment
  sweep/*.yaml             scaling / benchmark grids
jmbc/                    HOW it runs
  run.py                   CLI: train one experiment      -> runs/<exp>/<id>/
  sweep.py                 CLI: benchmark grid            -> benchmarks/<name>/
  analyze.py               CLI: recompute figures + diagnostics from a saved run
  config/                  typed schema + YAML/CLI loading
  envs/                    RBCKLEnv (RBC), RBCKSEnv (Krusell-Smith), registry
  algos/                   ActorCritic + PPO make_train (the RL kernel)
  experiments/             per-experiment drivers, common.run_single
  diagnostics/             rollout recorder, economic + distributional probes
  plots/                   style, training health, KS semantic figures, benchmarks
  recorder.py              run directory writer (config/metrics/rollouts/timing)
tests/                   env semantics, rollout, diagnostics checks
notebooks/quickstart     Colab walkthrough
paper.md                 the systems paper draft
.sources/                reference PDFs + the original pre-refactor PPO script
.archive/                superseded generated outputs (pre-env-fix results)
```

Data flow: `configs/` → `jmbc.run` → `runs/<exp>/<run_id>/` (complete record,
incl. `rollouts.npz`) → `jmbc.analyze` (iterate on figures/diagnostics ex post,
no retraining).

## Notebook

`notebooks/quickstart.ipynb` walks through loading a config, training,
inspecting diagnostics inline, and running a mini scaling sweep.
