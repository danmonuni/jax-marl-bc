# Figures 3 & 4, rerun on the v3 economy

One KS run — the `sigma=0.0, seed=8` cell of
[`runs/ks-wealth-lognormal-random-3/`](../ks-wealth-lognormal-random-3) — with
its **whole training trajectory** recorded, so the paper's figures 3 and 4 can
be redrawn from the same economy figure 6 already comes from.

## Why this is a run and not a replot

The sweep persists one rollout per cell: the trained policy's. Figures 3 and 4
are about how the economy *changes over training* — untrained vs trained law
of motion, wealth distribution across training snapshots — which no saved
artefact of that sweep contains. Retraining the cell is the only way to get
them, so this is a new experiment directory rather than a plotting change.

What it fixes: figures 3 and 4 currently come from
`runs/ks-correctness/ks_n200_top{,_replotted}` and `top`, which start every
agent at `k = 1` and use `num_envs=32` / `max_steps=5000`. Figure 6 comes from
the v3 sweep, which starts agents at `k_0 ~ U(10, 70)` per episode with
`num_envs=12` / `max_steps=200`. After this rerun all three figures describe
one economy.

## Protocol — a run *is* the sweep cell

Every hyperparameter of the sweep is restated in
[`config.yaml`](config.yaml) and re-asserted by `verify_protocol()` before
training, so an edit to `configs/exp/ks*.yaml` cannot quietly change what runs:

```
env    n_agents 200 | alpha 0.36 | beta 0.95 | delta 0.025 | max_steps 200
train  num_envs 12 | rollout_len 200 | total_timesteps 128000 (= 640 updates)
       update_epochs 4 | num_minibatches 64 | lr 3e-4 | gamma 0.95
       gae_lambda 0.95 | clip 0.2 | vf 0.5 | ent 0.0 | anneal_lr | grad_norm 0.5
net    tanh, hidden [64, 64]
k_0    U(10, 70) per agent, redrawn per episode
kappa  homogeneous (sigma = 0 short-circuits the LogNormal draw to ones)
seed   8
```

`sigma` here is the sweep's **kappa log-dispersion**, not `env.sigma` (the RBC
TFP innovation, which the KS environment does not read). `sigma=0.4 seed=3`
reruns any other cell of the sweep, with the same dense recording.

## The one deliberate difference: the record

| | sweep cell | here |
|---|---|---|
| evaluations | 12, **log**-spaced | 200, **linearly** spaced |
| steps each | 200 | 200 |
| agents kept | 200 | 200 |
| what is saved | the trained policy's rollout | every snapshot's rollout, all steps |

Linear rather than log because figure 4's x axis should cover training evenly
instead of bunching into its first decade. jmbc's own diagnostics loop
(`jmbc/diagnostics/report.py:20`) hard-codes log spacing, which is why this
directory carries its own recording loop rather than calling `jmbc.run`.

Evaluations are one uninterrupted **reset-free** episode even though training
episodes are 200 steps: the evaluator rebuilds the env with
`max_steps = sim_steps + 1` (`report.py:55`). Training is untouched by any of
this — snapshots are replayed after training ends.

`200 x 200 x 200` is ~70 MB — 200 snapshots is every ~3rd update, so figure
4's x axis is effectively continuous. Raising `sim_steps` thickens figure 3's
law-of-motion panels — they fit the post-burn-in half of one evaluation, ~100
points at 200 steps, against ~2500 in the old `ref/3.png` — at the price of
simulating past the 200-step episode the policy trained on.

## Running it

Colab (~2 min on a T4): [`notebooks/colab_ks_fig34_rerun.ipynb`](../../notebooks/colab_ks_fig34_rerun.ipynb).

```bash
python runs/ks-fig34-rerun/rerun_fig34.py                  # config.yaml as-is
python runs/ks-fig34-rerun/rerun_fig34.py device=cpu       # ~1 h on a laptop
python runs/ks-fig34-rerun/rerun_fig34.py sigma=0.4 seed=3 # another sweep cell
python runs/ks-fig34-rerun/rerun_fig34.py n_snapshots=60 sim_steps=1000
```

Output — `results/ks/sigma_0.00_seed_8/`:

| file | |
|---|---|
| `config.yaml` | the resolved config; the record states its own protocol |
| `rollouts.npz` | every channel, snapshots stacked on a leading axis |
| `metrics.csv` | per-update training metrics |
| `timing.json` | trace/compile/run split, throughput, device |
| `kappas.npy`, `meta.json` | the cell's kappa vector, sigma/seed/spacing |
| `params.msgpack`, `network.pkl` | the trained policy, to resimulate later |

`python -m jmbc.analyze results/ks/sigma_0.00_seed_8` recomputes the standard
diagnostics and KS figure set from that directory, no GPU needed.

## Plotting figures 3 and 4

```bash
python runs/ks-fig34-rerun/plot_fig34.py                        # -> <run>/figures/
python runs/ks-fig34-rerun/plot_fig34.py run=<pulled dir> out_dir=runs/final-paper-runs
```

With no arguments it finds the single run under `results/` and writes beside
it, which is what the Colab notebook does (the figures then sync to Drive with
the record). `out_dir=runs/final-paper-runs` writes the paper's `3.png` /
`4.png`.

It time-averages the cross-sectional histograms over the last 50 steps of each
evaluation — figure 6's `steady_state_capital` convention, so figure 3's Gini
becomes comparable with the Ginis in the sweep's `results.csv` — draws figure
4 on a real linear step axis, and fits one law of motion per aggregate state.

Sanity check against the sweep's own row for this cell (`sigma=0.0, seed=8`):
`capital_gini = 0.212`, `K_mean = 13.298`. Training should reproduce it — same
protocol, same kappas, same `PRNGKey(seed)` — so treat a large gap as a bug,
not as noise. Small differences are expected for two reasons: the statistic
here averages the last 50 steps rather than the stationary half, and the
evaluation env is rebuilt reset-free (`max_steps = 201`) where the sweep
evaluated in the training env, whose episode ends exactly at step 200.
