# Diagnostic: does population size decide the Gini regime?

Small side-by-side experiment, not a sweep for the paper. It exists to
answer one question before committing more compute to
`runs/ks-wealth-lognormal-random/`: **is n_agents/num_envs a confound for
which Gini regime a homogeneous KS training run lands in?**

## Background

`runs/ks-correctness/ks_n200_top/` (n_agents=200, num_envs=32, kappas=null,
i.e. fully homogeneous) went through an early training-time Gini spike and
then settled into a persistent bimodal wealth distribution around
`capital_gini≈0.42` — roughly 40% of agents clustered at low wealth, the
rest spread out over a much wider, higher range.

`runs/ks-wealth-lognormal-random/`'s sigma=0 cells (n_agents=500,
num_envs=8, also kappas≡1, i.e. also fully homogeneous) instead landed in a
much lower-Gini regime (~0.02–0.09) in all 5 seeds tried — while every
sigma>=0.04 cell in that same sweep landed back in a ks_n200_top-like
high-Gini regime. Checking individual cells' raw rollouts confirmed the
high-Gini cells really do show the same two-cluster wealth histogram shape
as ks_n200_top, just reached via a different sigma value instead of via
population size.

So there appear to be (at least) two basins a training run can settle into
even at exactly zero kappa heterogeneity, and it's not yet established
whether population size (n_agents/num_envs) changes which basin is typical,
or whether the 5 sigma=0 seeds simply happened to all sample the low-Gini
one by chance.

## Design

Population size is the *only* thing that varies between the two arms;
`base_exp: ks_n200` keeps every other hyperparameter (including
`total_timesteps`) fixed and identical, and `env.kappas` stays `null`
(homogeneous) throughout — no lognormal spread here, that's the sibling
sweep's job.

| arm | n_agents | num_envs | matches |
|---|---|---|---|
| `n200_e32` | 200 | 32 | `ks_n200_top` exactly |
| `n500_e8` | 500 | 8 | `ks-wealth-lognormal-random`'s default |

6 seeds per arm by default (12 runs total). Each run goes through the
standard `jmbc.run` CLI path — the same driver that produced every
`runs/ks-correctness/*` folder — with `diag.n_snapshots=12` so the full
Gini-over-training trajectory is visible (not just the final checkpoint),
the same view `ks_n200_top`'s own figures gave.

```bash
python run_branch_check.py                                  # config.yaml as-is
python run_branch_check.py seeds=[0,1] device=cpu n_snapshots=3   # quick smoke test
```

## Outputs

Each of the 12 runs writes the standard `jmbc.run` output set under
`results/ks/<run_name>/`: `config.yaml`, `diagnostics.json` (per-snapshot
economic + distributional metrics — this is where the Gini-over-training
trajectory lives), `metrics.csv` (per-update training curves),
`rollouts.npz` (raw per-agent trajectories for every snapshot, so wealth
histograms can be recomputed), `timing.json`, and `figures/` (including
`distributional.png`'s Gini/top-share-vs-training-steps view and
`ks_wealth_heatmap.png`'s stationary-wealth-distribution-through-training
view — the same figures `ks_n200_top` has).

`<run_name>` is `{arm}_seed{seed}`, e.g. `n200_e32_seed0`, `n500_e8_seed3`.

## Reading the result

For each arm, look at the final-snapshot `capital_gini` across its 6 seeds
in each run's `diagnostics.json`. If `n200_e32`'s seeds land in the high-Gini
bimodal regime much more often than `n500_e8`'s, population size is a real
confound and the main sweep should either match `n_agents=200` or budget for
many more seeds at `n_agents=500` to reliably sample that regime. If both
arms show a similar mix of high/low outcomes across seeds, it's just seed
variance and the main sweep only needs more seeds, not a population change.
