# `runs/`

Output directory. `python -m jmbc.run exp=<name>` writes one self-contained
directory per run here (`runs/<exp>/<run_id>/`: resolved config, metrics,
diagnostics, timing, raw rollouts and figures), and
`python -m jmbc.analyze runs/<exp>/<run_id>` rebuilds every figure and
diagnostic from that record without retraining.

Runs are generated output, not source — the inputs to every experiment are
`configs/` (what to run) and `jmbc/` (how it runs) — so this directory is
git-ignored and populated locally, with one exception.

## Sweeps

`python -m jmbc.sweep sweep=<name>` writes a *scan* rather than a single run,
into `<out_dir>/<name>/` (`out_dir: benchmarks` by default; the multi-seed
scaling scans set `out_dir: runs` so their record is kept here):

```
sweep.yaml           # the resolved sweep spec, as run
results.csv          # ONE ROW PER (cell, seed) — the raw timing sample
results_summary.csv  # ONE ROW PER CELL — mean/std/sem/min/max + n_seeds, seeds
*.png                # the figures named by the sweep's `figures` list
```

When a sweep declares `seeds: [0, 1, 2]`, each cell is trained once per seed;
`results_summary.csv` is the descriptive-statistics collapse of those repeats
(sample std, ddof=1 — a single-seed cell reports NaN, not 0), and the walltime
and throughput figures plot the mean with ±1 s.d. whiskers. Regenerate the
summary and the figures from a finished `results.csv` without retraining:

```bash
python -m jmbc.sweep sweep=<name> replot=runs/<name>
```

The two population-scaling scans are
[`configs/sweep/scan_agents_multiseed_gpu.yaml`](../configs/sweep/scan_agents_multiseed_gpu.yaml)
and its CPU twin — same protocol, `run.device` apart. Each config pins every
schema field it depends on, so a cell inherits nothing implicitly from
`configs/base.yaml` or `configs/exp/ks.yaml`.

## The one run in the repository

[`paper-ks-fig34/`](paper-ks-fig34) is the experiment behind the paper's
figures 3 and 4, and ships complete: its protocol and plotting scripts, plus
the full record of the run under
`paper-ks-fig34/results/ks/sigma_0.00_seed_8/`:

```
config.yaml          # fully resolved configuration
metrics.csv          # per-update training metrics
diagnostics.json     # economic + distributional probes across snapshots
timing.json          # wall time, throughput, device
rollouts.npz         # RAW snapshot rollouts: every recorded channel
kappas.npy, meta.json          # the cell's kappa vector, sigma/seed/spacing
params.msgpack, network.pkl    # the trained policy, to resimulate later
figures/
  3.png, 4.png                 # the paper's figures
  training_health.png          # PPO convergence panel
  distributional.png           # Gini & top shares vs steps
  ks_wealth_heatmap.png        # wealth distribution vs training
  ks_fig4.png                  # law of motion at four training snapshots
```

That is the same artifact set every other run writes, so the figures and every
diagnostic can be recomputed from it — no GPU, no retraining:

```bash
python -m jmbc.analyze runs/paper-ks-fig34/results/ks/sigma_0.00_seed_8
python runs/paper-ks-fig34/plot_fig34.py     # -> <run>/figures/3.png, 4.png
```

See [`paper-ks-fig34/README.md`](paper-ks-fig34/README.md) for the protocol,
why the run exists and how the two figures are drawn.
