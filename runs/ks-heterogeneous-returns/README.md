# Showcase experiment: calibrating heterogeneous capital productivities

Goal: give the Krusell-Smith economy heterogeneous per-household capital
productivity (`kappa^i`), with the cross-sectional *shape* hardcoded directly
from real data (Xavier 2021's return-on-wealth-by-percentile curve, already
cited in the paper as `xavier2021`) and a single free scalar — the overall
scale of that shape — calibrated so the model's own emergent steady-state
return distribution reproduces the same curve. Background and the broader
feasibility analysis this builds on: `.reports/extension_report.md`.

## Data

`xavier_return_by_wealth_percentile.csv` — digitized from the bar chart the
user supplied (2026-07-22), attributed to Xavier (2021), "Wealth Inequality in
the US: The Role of Heterogeneous Returns" (SSRN 3915439):

| bucket | 0-20 | 20-40 | 40-60 | 60-70 | 70-80 | 80-90 | 90-95 | 95-97 | 97-99 | 99-100 |
|---|---|---|---|---|---|---|---|---|---|---|
| return (%) | 3.6 | 3.6 | 3.7 | 3.3 | 3.7 | 3.9 | 5.1 | 6.3 | 7.5 | 8.3 |

The chart itself has no bar below the 20th percentile; the CSV's `0-20` row is
a **flat extrapolation** of the `20-40` value (3.6%), since the model needs a
productivity for every agent and there's no data there. It's marked
`source=chart` like every other row, so — as the CSV currently stands — it's
treated as a real target and included in the fit score along with the other
nine (`chart_mask` in the script is what controls this per-row; give a row
`source=extrapolated` instead if a future revision should exclude it from
scoring while still reporting it).

The chart also has a dashed reference line at roughly 6.8%, read here as an
aggregate/wealth-weighted average return (not a percentile bucket) — recorded
as a comment in the CSV, not used by the script. It's a plausible secondary
sanity check (e.g. against the model's capital-weighted average return) if
useful later, but no claim is made about exactly what it measures.

## Design

**What's hardcoded (not fit):** the *relative* shape of capital productivity
across the population. Agents are assigned kappa by rank: agent 0 (lowest)
gets the target return of the lowest bucket, agent n-1 (highest) gets the
99-100% bucket's value, with agent counts per bucket proportional to bucket
width (largest-remainder rounding so counts sum exactly to `n_agents`; see
`bucket_agent_counts`). This is `base_vector` in the code — literally the
Xavier numbers themselves, repeated across each bucket's agent allocation.

**What's fit:** one scalar, `k_multiplier`. `kappa_i = k_multiplier *
base_vector_i`.

**Why this conflates input and output (on purpose, and why that's fine):**
the same empirical curve is used twice — once to shape the input (`kappa`'s
relative pattern), once as the target the emergent *output* (simulated
steady-state returns) is checked against. With only one free parameter and
10 target points, this is over-identified: no single `k_multiplier` can
make the simulated/target ratio exactly 1 at every bucket, because the
general-equilibrium mapping from `kappa` to realized returns (via `K_t =
mean(kappa_i k_i)`, prices, and each household's learned savings response) is
not a pure rescaling of the input shape. The score therefore doesn't chase a
perfect fit — it's `sqrt(mean((simulated/target - 1)^2))` over all 10
buckets, and **whatever residual remains at the best `k_multiplier` is itself
the finding**: it says whether the model amplifies, dampens, or roughly
preserves the input return-heterogeneity shape once general-equilibrium
feedback and learned behaviour are accounted for.

**Steady state = time average.** For each candidate `k_multiplier`: train a
full KS run, then simulate one long deterministic (mean-action) rollout and
average every agent's per-period return and capital over the *stationary*
window (post burn-in, auto-reset steps excluded) — the "temporal average"
reading of steady state, exactly the `stationary_slice` pattern already used
throughout `jmbc/diagnostics/`. Agents are then **re-ranked by their own
realized (simulated) mean capital**, not by the a-priori kappa assignment —
the comparison is about where an agent actually ends up, not where it started
labeled as.

**Held-out check, not a fit target.** Capital Gini and top-10%/top-1% wealth
shares (`jmbc.diagnostics.gini`/`top_shares`, already implemented, no new
code) are computed and reported for every `k_multiplier`, but never enter the
score. This keeps the earlier "should we fit Gini or fit returns?" debate
resolved as: fit returns (the input `kappa`'s own empirical analogue),
*report* Gini as an emergent, un-targeted validation — same logic as the
paper's own honest reporting of the homogeneous-kappa baseline's 25% vs the
real ~70% top-10% share.

## Caveats to sanity-check before trusting absolute numbers

- **Annualization.** The model's `delta = 0.025` matches the standard
  quarterly RBC/Krusell-Smith calibration, so one model period is assumed to
  be one quarter; the script annualizes each agent's net marginal return by
  compounding 4 periods (`(1+r)^4 - 1`). If periods should instead be read as
  annual, `PERIODS_PER_YEAR = 1` and the whole calibrated scale changes.
  Confirm this before reading too much into the fitted `k_multiplier`.
- **Gross vs. net return.** `rec["R"]` is the *gross* per-period return
  (`1 - delta + alpha*(Y/K)*kappa`); the script backs out the net marginal
  return `r = R - (1-delta)` before annualizing, on the assumption Xavier's
  "return on wealth" means a net rate, not gross-of-depreciation.
- **Rank stability.** Kappa is assigned by rank once, fixed for the whole
  run; the design assumes (very plausibly, since kappa directly drives
  returns) that an agent's kappa-rank and its eventual wealth-rank end up
  close, but the script re-ranks by realized wealth precisely so this doesn't
  have to be assumed.

## Config interface

`config.yaml` holds every script parameter (`base_exp`, `n_agents`, `k_grid`,
`sim_steps`, `seed`, `device`, `total_timesteps`, `out_dir`), loaded via the
same merge order as `jmbc.config.load_config`: structured dataclass defaults
< `config.yaml` < CLI dotlist overrides (later wins). No argparse flags —
overrides look exactly like the rest of the repo's CLI (`env.n_agents=1000`
style, just without the `env.` prefix since this script's config is its own
small `CalibrationConfig`, not `jmbc`'s `ExperimentConfig`):

```bash
python calibrate_kappa_scale.py                          # config.yaml as-is
python calibrate_kappa_scale.py n_agents=500 device=cpu   # dotlist override
python calibrate_kappa_scale.py config=variant.yaml       # a different file
```

`n_agents` must give every percentile bucket (finest = 99-100%, i.e. 1% of
the population) at least one agent — `n_agents >= 100`; the script raises a
clear error otherwise rather than silently producing empty-bucket NaNs.

The default `k_grid` (0.1-0.5) is centered on the value that would bring
`mean(kappa)` back to roughly 1 (matching the homogeneous baseline's
`kappa≡1` convention): the 10 raw target numbers average to ~3.9, so
`k_multiplier ≈ 1/3.9 ≈ 0.26` is the naive starting guess before any
general-equilibrium feedback is accounted for — the grid brackets it. Each
cell is a full training run (reuses `configs/exp/ks_n200.yaml`, the exact
protocol already validated for the paper's own n=200 correctness run, ~3 min
on a free Colab T4); 7 grid points is comfortably one Colab session. If the
score's minimum lands at a grid edge, extend the grid rather than trusting an
edge value; if it's smooth and unimodal, a second, narrower grid around the
best point is worth an extra pass before calling the fit final.

## Outputs

`results/results.csv` — one row per `k_multiplier`: fit score, per-bucket
simulated vs. target return, and the held-out Gini/top-share numbers.
`results/calibration_fit.png` — score-vs-scale curve, and the best-fit
simulated curve overlaid on the Xavier target curve.

## Running it locally

```bash
# Smoke test (CPU, tiny budget) -- validates the pipeline, not the economics:
python calibrate_kappa_scale.py n_agents=100 "k_grid=[0.2,0.3]" \
    sim_steps=300 device=cpu total_timesteps=2000 out_dir=/tmp/smoke
```
(quote list-valued overrides like `k_grid=[...]` — the shell would otherwise
glob-expand the brackets.) `jmbc` isn't `pip install -e`d in every local
environment; the script inserts the repo root onto `sys.path` itself, so it
runs from any cwd as long as this file stays two levels under the repo root.

## Running it on Colab

`notebooks/colab_ks_kappa_calibration.ipynb` — clone, `pip install -e .`,
mount Drive, run the grid, sync `results/` to
`MyDrive/jax-marl-bc-runs/ks-heterogeneous-returns/results/`, display the
fit. Same structure as this repo's other Colab notebooks
(`colab_scaling_benchmarks.ipynb`, `colab_train_ks_populations.ipynb`): run
locally into `runs/ks-heterogeneous-returns/results/` first (fast local
disk), copy to Drive only once the run finishes (a mid-run Drive write is
slower and a disconnect mid-copy is worse than a disconnect mid-run) — not
the direct-to-Drive `out_dir` this README originally suggested.

**Before this works, these files need to actually be on GitHub.** `runs/` is
blanket-gitignored in this repo (everywhere else under `runs/` is generated
output, not source) — `git clone` on a fresh Colab session will not see
`calibrate_kappa_scale.py`, `config.yaml`, or the CSV otherwise (README.md is
along for the ride but isn't needed at runtime). They're already
force-staged locally (`git add -f`, since a `git add -A` would skip them), so
a normal commit will pick them up:

```bash
git commit -m "Add kappa-scale calibration experiment"
git push
```
I staged them but didn't commit or push — that's a call for you to make. The
generated `results/` subfolder stays correctly gitignored either way (it's
still under `runs/`, and nothing above force-adds it), so Colab-run results
won't accidentally get committed back.
