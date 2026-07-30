# `runs/`

Output directory. `python -m jmbc.run exp=<name>` writes one self-contained
directory per run here (`runs/<exp>/<run_id>/`: resolved config, metrics,
diagnostics, timing, raw rollouts and figures), and
`python -m jmbc.analyze runs/<exp>/<run_id>` rebuilds every figure and
diagnostic from that record without retraining.

Runs are generated output, not source — the inputs to every experiment are
`configs/` (what to run) and `jmbc/` (how it runs) — so this directory is
git-ignored and populated locally, with one exception.

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
