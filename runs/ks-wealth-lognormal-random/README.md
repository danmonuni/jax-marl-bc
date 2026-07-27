# Showcase experiment: an i.i.d.-sampled lognormal spread of capital productivity

Sibling of `runs/ks-wealth-calibration/` (see that directory's README for
the full original writeup and Aldo Glielmo's 2026-07-25 guidance to drop the
Xavier-Gini calibration target and instead demonstrate the software's
capability across a spread of heterogeneity). Same underlying design --
`kappa ~ LogNormal(mu, sigma)` with mean fixed to 1 (`mu = -sigma^2/2`),
sigma swept over a grid from 0 (homogeneous) to 1 (wide spread) -- but this
experiment draws kappa as a genuine i.i.d. sample rather than placing it on
the sibling's deterministic regular-quantile mesh.

## Design

**Distribution.** Identical to the sibling: `kappa ~ LogNormal(mu, sigma)`,
`mu = -sigma^2/2` so `E[kappa] = 1` for every sigma. `sigma=0` is still the
exact homogeneous baseline (`kappa_i ≡ 1`), with no seed dependence -- there
is nothing to sample.

**Sampling -- i.i.d., not deterministic.** The sibling experiment places
agent kappas on a regular quantile mesh specifically to avoid injecting
sampling noise into the sigma comparison. This experiment does the opposite
on purpose: kappas are drawn as genuine independent samples,
`np.random.default_rng(seed).lognormal(mu, sigma, n_agents)`, renormalized
to exactly mean 1 afterwards (same normalize-after-construct convention as
the sibling). That means the realized cross-sectional *shape* of kappa --
not just its target moments -- now varies with `seed` at fixed `sigma`.

**The sweep grid.** Because one draw no longer pins down "what a given sigma
looks like", this experiment sweeps a list of **seeds** at every **sigma**,
not sigma alone: `sigmas x seeds` (default 6 x 5 = 30 cells). Each
`(sigma, seed)` pair deterministically fixes both the kappa draw and the RL
run (`run.seed = seed`), so every cell is exactly reproducible. Every cell in
the grid is trained and kept -- none is treated as more representative than
another; `results.csv` and `comparison.png` show the full spread of outcomes
across seeds at each sigma.

## Config

`base_exp: ks_n200` (only `env.n_agents`/`env.num_envs` overridden, same
convention as the sibling), `n_agents: 500`, `num_envs: 8` (batch width
`W = num_envs * n_agents = 4,000`, inside the compute plateau the paper's
own scaling mesh measured). `sigmas: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`,
`seeds: [0, 1, 2, 3, 4]` by default (30 cells) -- widen either list via CLI
dotlist. `save_raw: all` -- every cell's raw rollout + trained params/network
are kept, not just a subset.

```bash
python sweep_lognormal_random.py                                     # config.yaml as-is
python sweep_lognormal_random.py n_agents=100 device=cpu sigmas=[0.0,0.5] seeds=[0,1]  # quick smoke test
```

## Outputs

Structured under `results/`:

- `results.csv` -- one row per `(sigma, seed)` cell: `sigma`, `seed`, `mu`,
  realized `kappa_mean`/`kappa_std`/`kappa_min`/`kappa_max`, `capital_gini`,
  `top_0.1_share`, `top_0.01_share`, `K_mean`, `C_mean`, `euler_mean_abs`,
  `resource_mean_rel`. Checkpointed after every cell.
- `results/figures/sigma_<sigma>_seed_<seed>_steady_state.png` -- one
  dashboard per cell: kappa profile, aggregate capital path, aggregate
  consumption path, the aggregate KS (bad/good) shock, the Lorenz curve, and
  the wealth histogram.
- `results/comparison.png` -- cross-cell view: inequality (Gini/top shares)
  vs. sigma as one point per seed (so the spread at each sigma is visible),
  overlaid Lorenz curves, and overlaid kappa profiles, all colored by sigma.
- `results/raw/sigma_<sigma>_seed_<seed>/` -- full raw rollout
  (`rollout.npz`), the realized kappa vector (`kappas.npy`), trained params
  (`params.msgpack`), network definition (`network.pkl`), and `meta.json`
  (delta) for every cell -- enough to regenerate any figure later without
  retraining.

## Caveats

Same `n_agents=1000`+ cost caveats as the sibling apply, compounded by the
seed axis: total cells = `len(sigmas) * len(seeds)`, so time the first cell
before trusting a total-runtime estimate for the rest. No empirical target
is fit here (see "Design" above) -- `capital_gini`/`top_*_share` values in
`results.csv` are simply what each `(sigma, seed)` draw produces, and the
seed-to-seed spread at fixed sigma is itself part of what this experiment is
measuring, not noise to average away.
