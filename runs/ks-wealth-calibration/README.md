# Showcase experiment: a deterministic lognormal spread of capital productivity

Sibling of `runs/ks-heterogeneous-{returns,wealth}/`, which both calibrate
kappa's cross-sectional *shape* against Xavier (2021)'s digitized
return-by-wealth-percentile data. This experiment drops the external target
entirely, per Aldo Glielmo's guidance (2026-07-25):

> il tema del Gini sulla ricchezza è un tema spinoso e difficile da studiare
> quindi neanche si sa stimare bene... la priorità non è quella di fittare il
> dato empirico ma quella di mostrare le capability del software, quindi
> vediamo quale run è più bella

i.e. matching an empirical wealth Gini is a poorly-identified target not
worth chasing here; the point of this run is to demonstrate the software's
capability across a spread of heterogeneity, and to pick whichever sigma(s)
turn out to look best for the paper — a qualitative call made from the saved
figures, not an optimization objective. Concretely: kappa is drawn from a
**LogNormal(mu, sigma)** with mean fixed to 1 (`mu = -sigma^2/2`, so the
economy stays comparable to the homogeneous `kappa≡1` baseline at every
sigma), and **sigma is swept over a grid from 0 to 1**.

## Design

**Distribution.** `kappa ~ LogNormal(mu, sigma)`, `mu = -sigma^2/2` so that
`E[kappa] = exp(mu + sigma^2/2) = 1` for every sigma — Aldo's instruction
exactly. `sigma=0` degenerates to the homogeneous baseline (`kappa_i ≡ 1`
for all agents), giving a built-in "no heterogeneity" anchor in every sweep.

**Sampling — deterministic, not i.i.d.** Rather than drawing each agent's
kappa as an independent sample (which would inject sampling noise into every
cell, on top of the RL training's own seed variance — a "did this cell just
get an unlucky draw" confound that makes sigma harder to read off the
results), agent rank is placed on a **regular grid of quantiles**,
`u_i = (i + 0.5) / n`, and each agent's kappa is that regular grid inverted
through the LogNormal's quantile function (`scipy.stats.lognorm.ppf`):

```
u_i     = (i + 0.5) / n                     # regular grid, agent rank
kappa_i = LogNormal(mu, sigma).ppf(u_i)      # deterministic quantile mesh
kappa_i = kappa_i / mean(kappa_i)            # renormalize to exactly mean 1
```

This is deterministic and exactly reproducible: the same sigma always
produces the same kappa vector, and the n-point mesh is by construction the
smoothest possible discretization that conforms to the LogNormal(mu, sigma)
shape without draw-to-draw surprises. The renormalization step corrects for
the finite-n quantile mesh's empirical mean deviating slightly from the
continuum-limit mean of 1 (more so at high sigma, where the mesh's extreme
points sit further out on the heavy right tail) — same
normalize-after-construct convention as the sibling Beta-distortion
experiment's `shape / shape.mean()`.

**The sweep.** `sigmas: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]` by default (6 cells).
For each sigma: build kappa, train a full KS run, simulate a long
steady-state rollout, and record inequality statistics (`capital_gini`,
`top_0.1_share`, `top_0.01_share`) plus the standard economic diagnostics
(Euler error, resource residual, steady-state K/C). No search, no
optimization — every cell is trained and kept.

## Config

`base_exp: ks_n200` (only `env.n_agents`/`env.num_envs` overridden, same
convention as the sibling scripts), `n_agents: 500`, `num_envs: 8` (batch
width `W = num_envs * n_agents = 4,000`, inside the compute plateau the
paper's own scaling mesh measured). `save_raw: all` — every sigma is a
deliberate design point, not a search, so every cell's raw rollout + trained
params/network are kept, not just a "best" one.

```bash
python sweep_lognormal.py                         # config.yaml as-is
python sweep_lognormal.py n_agents=100 device=cpu sigmas=[0.0,0.5]   # quick smoke test
```

## Outputs

Structured under `results/`:

- `results.csv` — one row per sigma: `sigma`, `mu`, realized `kappa_mean`/
  `kappa_std`/`kappa_min`/`kappa_max`, `capital_gini`, `top_0.1_share`,
  `top_0.01_share`, `K_mean`, `C_mean`, `euler_mean_abs`,
  `resource_mean_rel`. Checkpointed after every cell.
- `results/figures/sigma_<sigma>_steady_state.png` — one dashboard per
  sigma: kappa profile, aggregate capital path, aggregate consumption path,
  the aggregate KS (bad/good) shock, the Lorenz curve, and the wealth
  histogram — "does this cell actually settle into a stationary regime, and
  what does that regime look like."
- `results/comparison.png` — cross-sigma view: inequality (Gini/top shares)
  vs. sigma, overlaid Lorenz curves, and overlaid kappa profiles, all colored
  by sigma — the "which run is nicest" comparison.
- `results/raw/sigma_<sigma>/` — full raw rollout (`rollout.npz`), the
  kappa vector (`kappas.npy`), trained params (`params.msgpack`), network
  definition (`network.pkl`), and `meta.json` (delta) for every cell — enough
  to regenerate any figure later without retraining.

## Caveats

Same `n_agents=1000`+ cost caveats as the sibling experiments apply if the
grid is widened past the default `n_agents=500`: time the first cell before
trusting a total-runtime estimate for the rest. No empirical target is fit
here (see "Design" above) — do not read `capital_gini`/`top_*_share` values
in `results.csv` as calibrated-to-data figures; they are simply what each
sigma's heterogeneity produces.
