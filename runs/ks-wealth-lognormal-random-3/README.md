# Lognormal kappa spread, rerun on randomized starting capital

Third iteration of `runs/ks-wealth-lognormal-random/` (referred to below as
**v2**). The economic design is deliberately identical to v2; what changed is
underneath it, in the environment's initialization.

## What changed, and why this is a new experiment

Until 2026-07 every simulation in this repo started every agent at the same
capital, `k_i = k_init` for all `i` (1.0 for KS). Starting capital is now
drawn **per agent**, independently **per parallel env**, and by default
redrawn **at every episode reset** — the initialization the reference KS
implementation uses (`np.random.uniform(10, 70, size=n)` inside its `reset`).

This is a new directory rather than an edit to v2 because **its cells are not
comparable to v2's**: the initial condition differs, so any change in the
reported Gini / top shares / `K_mean` confounds the kappa design with the
initialization. v2 stays on disk, untouched, as the constant-`k_init` record.

**Why U(3, 20).** The reference uses `U(10, 70)`, which spans roughly
0.25x–1.75x of *its* steady state — `K* ~ 40`, the standard KS calibration at
`beta = 0.99`. This repo runs `beta = 0.95`, where `K* ~ 11.7`:

```
r* = 1/beta - 1 + delta ;  K* = (alpha/r*)^(1/(1-alpha)) * L
```

Copying `U(10, 70)` across would start nearly every agent *above* steady
state (0.85x–6x) with a long one-directional transient. `U(3, 20)` preserves
the reference's proportions at this calibration. Worth noting this starts the
economy **near** its steady state, whereas the old constant `k_init = 1.0`
started it at ~0.085x — so burn-in has *less* transient to absorb than before,
not more.

## Design (unchanged from v2)

`kappa ~ LogNormal(mu, sigma)` with `mu = -sigma^2/2` so `E[kappa] = 1` for
every sigma, drawn as genuine i.i.d. samples
(`np.random.default_rng(seed).lognormal(...)`) and renormalized to exactly
mean 1 — not placed on the deterministic quantile mesh that
`runs/ks-wealth-calibration/` uses. Because one draw does not pin down what a
given sigma looks like, a list of **seeds** is swept at every **sigma**.
`sigma = 0` is the exact homogeneous baseline with no seed dependence. Every
`(sigma, seed)` cell is trained and kept; none is treated as more
representative than another.

## Config

`base_exp: ks_n200` (same protocol as v2 and the calibration siblings),
`n_agents: 500`, `num_envs: 8` (batch width `W = 4,000`, inside the compute
plateau the paper's scaling mesh measured). `sigmas: [0.0, 0.2, 0.4, 0.6,
0.8, 1.0]` x `seeds: [0, 1, 2, 3, 4]` = 30 cells.

`base_exp` resolves through `configs/exp/ks.yaml`, which **pins
`k_init_dist: constant`** so the older runs stay bit-reproducible. This sweep
therefore overrides the initialization explicitly per cell, from the
`k_init_*` fields in `config.yaml`.

```bash
python sweep_lognormal_random_3.py                      # config.yaml as-is
python sweep_lognormal_random_3.py n_agents=100 device=cpu sigmas=[0.0,0.5] seeds=[0,1]
python sweep_lognormal_random_3.py k_init_dist=constant # v2's initialization, for an A/B
```

That last form is the cleanest way to isolate the initialization: same kappa
design, same seeds, only `k_init_dist` differs.

## Outputs

Structured under `results/`, same layout as v2:

- `results.csv` — one row per `(sigma, seed)` cell: `sigma`, `seed`, `mu`,
  realized `kappa_mean`/`kappa_std`/`kappa_min`/`kappa_max`, the
  initialization actually used (`k_init_dist`, `k_init_low`, `k_init_high`,
  `k_init_resample` — recorded per row so the CSV is interpretable without
  the config that produced it), `capital_gini`, `top_0.1_share`,
  `top_0.01_share`, `K_mean`, `C_mean`, `euler_mean_abs`,
  `resource_mean_rel`. Checkpointed after every cell.
- `results/figures/sigma_<sigma>_seed_<seed>_steady_state.png` — per-cell
  dashboard: kappa profile, aggregate capital path (**with the `k_0` draw's
  support shaded**, so convergence to the same steady state from anywhere in
  that band is visible), aggregate consumption path, the aggregate KS shock,
  the Lorenz curve, and the wealth histogram.
- `results/comparison.png` — inequality vs. sigma (one point per seed),
  overlaid Lorenz curves, overlaid kappa profiles, colored by sigma.
- `results/raw/sigma_<sigma>_seed_<seed>/` — full rollout (`rollout.npz`),
  kappa vector (`kappas.npy`), params (`params.msgpack`), network
  (`network.pkl`), `meta.json` — enough to regenerate any figure without
  retraining.

## Caveats

Cells here are **not** comparable to v2's — see above. Total runtime scales as
`len(sigmas) * len(seeds)`; time the first cell before trusting an estimate
for the rest. No empirical target is fit (per Aldo Glielmo's 2026-07-25
guidance to demonstrate capability rather than chase the wealth Gini), so
`capital_gini` / `top_*_share` are simply what each draw produces, and the
seed-to-seed spread at fixed sigma is part of what is being measured, not
noise to average away.

One open calibration question: whether the reference intended the literal
`U(10, 70)` — which would imply moving this repo to `beta = 0.99` — or the
rescaled bounds used here. Worth confirming before these numbers go in the
paper.
