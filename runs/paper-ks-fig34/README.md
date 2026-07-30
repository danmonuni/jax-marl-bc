# Paper figures 3 & 4 — the Krusell-Smith run behind them

One KS run — the `sigma=0.0, seed=8` cell of the lognormal-kappa sweep that
produces figure 6 — with its **whole training trajectory** recorded, so
figures 3 and 4 are drawn from the same economy as figure 6.

This is the only run shipped in the repository: the protocol below, the two
plotting scripts, and the record of the training run under
`results/ks/sigma_0.00_seed_8/` (see the table further down). The sweep itself
is a local artefact, reproducible from `configs/` — this run is not, since it is
the training trajectory that had to be recorded once.

Everything ships except the raw `rollouts.npz`, which is 16 MB and would be the
bulk of a clone. The rendered figures, metrics and diagnostics are all here, so
the dashboard has a real run to display and the paper figures are readable as
committed; regenerate the raw arrays with `rerun_fig34.py` if you need
`plot_fig34.py` to redraw them from scratch.

## Why this is a run and not a replot

The sweep persists one rollout per cell: the trained policy's. Figures 3 and 4
are about how the economy *changes over training* — untrained vs trained law
of motion, wealth distribution across training snapshots — which no saved
artefact of that sweep contains. Retraining the cell is the only way to get
them, so this is a new experiment directory rather than a plotting change.

It also puts all three figures on one economy. Earlier drafts drew figures 3
and 4 from a KS run that started every agent at `k = 1` with `num_envs=32` /
`max_steps=5000`, while figure 6 comes from the sweep, which draws
`k_0 ~ U(10, 70)` per episode with `num_envs=12` / `max_steps=200`. This run
uses the sweep's protocol.

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
points at 200 steps, against ~2500 in the earlier generation — at the price of
simulating past the 200-step episode the policy trained on.

## Running it

Colab (~2 min on a T4): `notebooks/colab_ks_fig34_rerun.ipynb` (local only —
`notebooks/` is git-ignored).

```bash
python runs/paper-ks-fig34/rerun_fig34.py                  # config.yaml as-is
python runs/paper-ks-fig34/rerun_fig34.py device=cpu       # ~1 h on a laptop
python runs/paper-ks-fig34/rerun_fig34.py sigma=0.4 seed=3 # another sweep cell
python runs/paper-ks-fig34/rerun_fig34.py n_snapshots=60 sim_steps=1000
```

Output — `results/ks/sigma_0.00_seed_8/`. The record shipped here was made with
`n_snapshots=30` (not `config.yaml`'s default of 200): 30 linearly-spaced
evaluations x 200 steps x 200 agents, 16 MB. `meta.json` states what any given
record actually used.

| file | |
|---|---|
| `config.yaml` | the resolved config; the record states its own protocol |
| `rollouts.npz` | every channel, snapshots stacked on a leading axis — *not committed, see above* |
| `metrics.csv` | per-update training metrics |
| `timing.json` | trace/compile/run split, throughput, device |
| `diagnostics.json` | economic + distributional probes at every snapshot |
| `kappas.npy`, `meta.json` | the cell's kappa vector, sigma/seed/spacing |
| `params.msgpack`, `network.pkl` | the trained policy, to resimulate later |
| `figures/` | `3.png`, `4.png` and the standard per-run figure set |

`python -m jmbc.analyze results/ks/sigma_0.00_seed_8` recomputes the standard
diagnostics and KS figure set from that directory, no GPU needed — it is what
produced the `diagnostics.json` and the non-paper figures shipped here, so this
directory carries exactly the information any other jmbc run exposes.

## Plotting figures 3 and 4

```bash
python runs/paper-ks-fig34/plot_fig34.py                        # -> <run>/figures/
python runs/paper-ks-fig34/plot_fig34.py run=<pulled dir> out_dir=<dir>
python runs/paper-ks-fig34/plot_fig34.py mode=pooled            # one convention -> 3.png, 4.png
```

With no arguments it finds the single run under `results/` and writes beside
it, which is what the Colab notebook does (the figures then sync to Drive with
the record). `out_dir=<dir>` collects the paper's copies elsewhere.

Both figures draw a cross-section out of the last `WINDOW = 50` steps of each
evaluation, and there are two defensible ways to do that. `mode=both` (the
default) renders each figure twice, `<n>-time-averaged.png` and
`<n>-pooled.png`; naming one convention writes the plain `<n>.png`.

| | `timeavg` | `pooled` |
|---|---|---|
| a sample is | one agent's mean over the window | one agent-step |
| samples per histogram | 200 | 10 000 |
| keeps within-agent time variation | no | yes |
| equals | figure 6's `steady_state_capital`, windowed | averaging the window's per-step histograms; what the earlier generation of these figures did over the stationary half |
| looks like | spiky; figure 4 blocky | smooth; figure 4 continuous |

On this run they agree on the economics — trained capital Gini 0.215
(`timeavg`) vs 0.221 (`pooled`), same mean 13.12 — so the choice is
presentational.

Figure 4 is drawn on a real linear step axis, and figure 3 fits one law of
motion per aggregate state.

Sanity check against the sweep's own row for this cell (`sigma=0.0, seed=8`):
`capital_gini = 0.212`, `K_mean = 13.298`. Training should reproduce it — same
protocol, same kappas, same `PRNGKey(seed)` — so treat a large gap as a bug,
not as noise. Small differences are expected for two reasons: the statistic
here averages the last 50 steps rather than the stationary half, and the
evaluation env is rebuilt reset-free (`max_steps = 201`) where the sweep
evaluated in the training env, whose episode ends exactly at step 200.
