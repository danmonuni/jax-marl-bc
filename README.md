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
# or, on a GPU instance (e.g. Colab T4):
pip install -r requirements.txt          # uses jax[cuda12] (pip-vendored CUDA)
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
                       # (capital, wealth, consumption, employment,
                       # aggregate state, ...) at each training snapshot
  figures/
    training_health.png      # PPO convergence panel
    distributional.png       # Gini & top shares vs steps
    ks_wealth_heatmap.png    # wealth distribution vs training (density heatmap)
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

A sweep is a grid of overrides over one base experiment. Write
`configs/sweep/<name>.yaml` (schema below), then:

```bash
python -m jmbc.sweep sweep=<name>                        # run the grid
python -m jmbc.sweep sweep=<name> train.total_timesteps=50000   # + CLI overrides
python -m jmbc.sweep sweep=<name> replot=benchmarks/<name>      # figures only,
                                                                # no training
```

Writes `benchmarks/<name>/{results.csv,sweep.yaml}` (one row per cell with all
timings and a canonical `time_s` column; `method` distinguishes series) plus
the figures selected by the sweep's `figures` list. Timing separates JIT
**compile time** from steady-state **run time** so throughput is measured
fairly. Because each run persists its own resolved `sweep.yaml` next to the
results, a completed sweep is self-documenting — the repository ships the
schema, not a library of presets.

Schema — every field with its default:

```yaml
name: my_sweep
base_exp: ks
axes:
  env.n_agents: [10, 100, 1000, 10000]
  train.num_envs: [32, 128]
overrides:
  train.total_timesteps: 50000   # sequential env steps, independent of num_envs
repeats: 1
paired: false               # true -> zip equal-length axes instead of the
                            # Cartesian product (e.g. a constant-product cut)
benchmark: false            # one timed run/cell; the AOT phase timer already
                            # splits compile from run (true = double-run split)
collect_diagnostics: true   # also tabulate Euler / Gini per cell
save_cell_runs: false       # true -> full per-cell run dir + figures
                            # (wealth distribution, Euler panel, training health)
                            # under benchmarks/<name>/cells/
figures: [auto]             # which graphs to render from the timing table:
                            # auto | walltime | throughput | speedup | phase | tradeoff
tradeoff_product: null      # slice for "tradeoff": n_agents * num_envs == product
reference_csv: null         # your own CSV of baseline timings, overlaid on the
                            # time figures & used as the numerator of "speedup".
                            # Columns: method, n_agents, time_hours | time_s
                            # (path relative to the repo root; none shipped)
```

## Repository map

Everything you edit lives in `jmbc/` + `configs/`; everything generated lands
in `runs/` (single experiments) or `benchmarks/` (sweeps), both git-ignored.

```
configs/                 WHAT to run
  base.yaml                shared defaults, inherited by every experiment
  exp/{rbc,ks,general}     the three base templates (no `extends`)
  exp/ks_local.yaml        derived preset: `extends: ks`, 5-minute CPU budget
  sweep/<name>.yaml        your benchmark grids (none shipped; see Sweep above)
  reference/*.csv          external baseline timings for `reference_csv`
                           (none shipped; supply your own)
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
llm-sim-dashboard/       Streamlit UI: describe a simulation in natural language,
                         an LLM maps it to a config, runs it, plots the results
runs/                    generated output; only paper-ks-fig34/ (the paper's
                         figures 3 & 4, complete record) ships — see runs/README.md
```

Data flow: `configs/` → `jmbc.run` → `runs/<exp>/<run_id>/` (complete record,
incl. `rollouts.npz`) → `jmbc.analyze` (iterate on figures/diagnostics ex post,
no retraining).

## LLM simulation dashboard

`llm-sim-dashboard/` is a Streamlit front end: describe an economy in natural
language, an LLM maps the description onto a validated `configs/` file, the
simulation runs through the same `jmbc.run` CLI, and the results come back as
figures. It shells out to a separate interpreter, so the dashboard's
dependencies stay independent of JAX.

```bash
pip install -r llm-sim-dashboard/requirements.txt
cp llm-sim-dashboard/.env.example llm-sim-dashboard/.env   # add your API key
./llm-sim-dashboard/run.sh
```

See [`llm-sim-dashboard/README.md`](llm-sim-dashboard/README.md) for the
provider options (OpenAI or a local Ollama model) and configuration.
