# JaxMARL-BC — LLM Simulation Dashboard

Describe a macroeconomic RL experiment in **plain language**; an LLM (OpenAI or
Ollama) translates it into JaxMARL-BC parameters, runs the JAX framework via a
CLI runner, streams the logs live, and renders the resulting figures, metrics,
diagnostics and on-disk artifacts — with a download button.

```
you: "Run a quick Krusell-Smith model with 50 agents on CPU."
 └─ LLM ── run_simulation(experiment=ks, n_agents=50, device=cpu, total_timesteps=30000)
     └─ writes configs/exp/dash-<id>.yaml   (extends: ks + your overrides)
         └─ python -m jmbc.run exp=dash-<id> log.run_name=dash-<id>
             └─ live logs ▸ figures + metrics.csv + diagnostics.json + timing.json + download
```

## Dynamic experiment configs

The framework's access point is `python -m jmbc.run exp=<name>`, which loads
`configs/exp/<name>.yaml`. Rather than pass an opaque dotlist of overrides, the
LLM's parameters are **materialized into a brand-new `configs/exp/<name>.yaml`**
that `extends` the closest base template and overrides only what the request
implies. That file is a persistent, human-editable, reproducible artifact you
can rerun by hand or tweak later. Generated files are prefixed `dash-` (so they
are excluded from the list of base templates) and can be pruned with
`config_gen.cleanup_generated()`.

| File | Responsibility |
|------|----------------|
| `settings.py` | Resolved config from env / `.env` (paths, interpreter, provider). |
| `sim_spec.py` | The NL→run contract: parameter catalogue, the LLM tool schema, and validation. |
| `config_gen.py` | Turns validated parameters into a new `configs/exp/<name>.yaml` (resolving `extends` chains + merging overrides). |
| `sim_runner.py` | The CLI/execution server: generates the config, launches the JAX subprocess by name, streams stdout, discovers output dirs. Also runnable standalone. |
| `llm_agent.py` | OpenAI / Ollama provider abstraction; both call the same `run_simulation` tool. |
| `results.py` | Loads `config.yaml` / `metrics.csv` / `diagnostics.json` / `timing.json` / figures; builds the download zip. |
| `app.py` | The Streamlit UI (interpret → run → results). |

## Setup

```bash
# 1. install the DASHBOARD deps (streamlit + an LLM client)
python -m pip install -r requirements.txt

# 2. configure
cp .env.example .env
#   - set OPENAI_API_KEY (for the OpenAI provider), and/or
#   - set OLLAMA_HOST / OLLAMA_MODEL (for a local Ollama), and
#   - set JMBC_PYTHON if jax + jmbc live in a different environment.
```

This dashboard lives inside the repo, at `jax-marl-bc/llm-sim-dashboard/`, so
`JMBC_REPO` defaults to the parent directory — no path config needed unless you
move it. The environment that runs the simulation needs `jmbc` installed
(`pip install -e .` from the repo root). By default that is the repo's own
`.venv` if present, otherwise the interpreter running the dashboard; set
`JMBC_PYTHON` to override.

## Run

```bash
./run.sh
# or explicitly:
python -m streamlit run app.py
```

Open the printed URL. In the sidebar you can toggle the **LLM provider**
(OpenAI ⇄ Ollama) live and see backend preflight status.

### Choosing a provider

- **OpenAI** — set `OPENAI_API_KEY` in `.env`. Uses native function calling.
- **Ollama** — run a local Ollama server and pick a **tool-calling** model
  (e.g. `llama3.1`, `qwen2.5`). Set `OLLAMA_HOST` / `OLLAMA_MODEL`.

## CLI server (no dashboard)

The runner is usable on its own — the LLM's job is just to produce this JSON:

```bash
python sim_runner.py \
  '{"experiment":"rbc","total_timesteps":20000,"device":"cpu"}'
```

It writes the generated `configs/exp/<id>.yaml`, streams the framework logs,
and prints the config path + output directory(ies).

## What the LLM can set

A curated, validated subset of the jmbc config (see `sim_spec.PARAMS`):
`experiment`, `n_agents`, `max_steps`, `total_timesteps`, `num_envs`,
`num_minibatches`, `update_epochs`, `lr`, `gamma`, `hidden_dims`,
`n_snapshots`, `sim_steps`, `device`, `seed`, plus an `extra_overrides` escape
hatch for any other dotted config path. Everything is range/enum-checked before
a subprocess is launched, and the proposed command is shown for confirmation —
the model configures the run, it never executes anything itself.

## Results shown

- **Timing** tiles (wall / run time, throughput, device).
- **Final-snapshot diagnostics** tiles (Euler errors, Gini, …).
- **Figures** (`training_health.png`, `distributional.png`, KS/RBC bespoke
  figures) rendered inline — the list in `results.HIDDEN_FIGURES` is skipped.
- **Training metrics** — `metrics.csv` as an interactive line chart + table.
- **Diagnostics across snapshots** — flattened table + per-metric chart.
- **Stored data** — the absolute run directory path, its file list, and a
  one-click **Download run (.zip)**.

Past runs on disk can be reopened from the **📂 Browse runs** tab, which finds
them by content (`results.discover_runs`) rather than at a fixed depth under
`runs/`, so records that don't sit at `runs/<exp>/<run_id>/` — e.g. the paper's
`paper-ks-fig34/results/ks/<cell>/` — show up too.

> Tip: for interactive use ask for a *quick* run (small `total_timesteps`, CPU).
> Full `ks` budgets (500k steps) are designed for a GPU and take a long time.
