# Showcase experiment: calibrating against the stationary wealth distribution

Sibling of `runs/ks-heterogeneous-returns/`, which turned out to be a
"convoluted" design (it used the same Xavier (2021) curve twice — once as
the fixed input shape, once as the fit target for the emergent output — with
a single free scalar chasing 10 target points at once). This experiment
fixes that by separating concerns properly: Xavier's data only shapes the
*cross-sectional profile* of capital productivity; the thing actually being
fit is the **stationary wealth distribution** (Gini / top shares) — the
phenomenon Krusell-Smith-style models exist to explain in the first place,
and a target that's independent of how the input was constructed.

## Design

**The input shape.** `xavier_return_by_wealth_percentile.csv` (same digitized
data as the sibling experiment) is interpolated into a smooth, monotonic
quantile function `Q(p) -> annual return (%)` via `scipy.interpolate.
PchipInterpolator` (monotonicity-preserving where the underlying data is
monotonic — Xavier's own digitized numbers have a genuine small dip at the
60-70% bucket, 3.3% vs 3.6-3.7% on either side, which PCHIP faithfully
reproduces rather than smoothing away).

**The distortion.** Agent rank is mapped to a percentile through a
Beta-distorted uniform grid instead of a plain one:
```
u_i     = (i + 0.5) / n                    # plain uniform grid (agent rank)
p_i     = Beta(a, b).ppf(u_i)               # distorted percentile
shape_i = Q(clip(p_i))                      # look up Xavier's curve there
kappa_i = scale * shape_i / mean(shape)     # mean(kappa) = scale
```
`(a, b) = (1, 1)` is Beta(1,1) = Uniform(0,1), whose `.ppf` is the identity —
so this recovers literally Xavier's own curve with zero distortion, confirmed
numerically (`beta.ppf(u, 1, 1) == u`). Away from `(1,1)`, the distortion
reshapes how densely agent ranks sample Xavier's curve — e.g. concentrating
more agents near the extremes or the middle — giving two continuous degrees
of freedom to amplify or dampen the raw empirical heterogeneity, independent
of `scale` (the absolute level, i.e. `mean(kappa)`).

**The fit.** Three free parameters — `scale`, `a`, `b` — are searched by
Bayesian optimization (`skopt.gp_minimize`, log-uniform priors, same
machinery as the sibling experiment, natively extended to 3 dimensions) to
match `targets`: default is `top_0.1_share: 0.70`, the top-10%-wealth-share
figure already cited in the paper's own text (Xavier/US data) — *not*
independently invented for this experiment. `capital_gini` can be added as a
second target, but no Gini figure is hardcoded here since none is
independently sourced anywhere in this repo yet; add your own citation
before trusting a default for it.

Scoring uses the **stationary wealth distribution** directly:
train → simulate → time-average each agent's capital over the stationary
window (`steady_state_capital`, same `stationary_slice` + auto-reset
exclusion convention as everywhere else in `jmbc/diagnostics/`) → `gini`/
`top_shares` on that (already-implemented, no new diagnostic code, unlike
the sibling experiment's return-by-percentile diagnostic which had to be
built from scratch). Score is the RMS relative error across `targets`.

## Why this is better-identified than the sibling experiment

The sibling design had one free parameter chasing 10 target points (the
input's own construction) — necessarily over-identified, and conflated with
its own answer. Here: 3 free parameters (`scale`, `a`, `b`) fit against
1-2 target statistics (`top_0.1_share`, optionally `capital_gini`) computed
from a genuinely different, emergent quantity (wealth, not returns). Still
not perfectly identified 1:1, but no longer circular, and the sibling
smoke-test evidence backs this up qualitatively: even at a tiny, barely-
trained budget, this design's score moved *meaningfully* across evaluations
(0.85 → 0.40, with Gini/top-share swinging from ~0.02/0.11 at low
heterogeneity to ~0.94-0.98 at high heterogeneity) — real, interpretable
structure, unlike the sibling experiment's smoke tests, which stayed stuck
in an uninformative regime regardless of `k_multiplier`.

## Config

Same `config.yaml` + CLI-dotlist convention as the sibling experiment
(structured defaults < `config.yaml` < CLI overrides). Key fields beyond the
shared training ones (`base_exp`, `n_agents`, `num_envs`, `sim_steps`,
`seed`, `device`, `total_timesteps`, `out_dir`, `save_raw` — see sibling
README for these):
- `scale_bounds`, `a_bounds`, `b_bounds` — log-uniform search ranges (all
  strictly-positive scale-like quantities).
- `bo_calls` / `bo_init_points` — total evaluations / initial random design.
  A 3-D search needs more initial points than the sibling's 1-D search
  (default 8 vs 5) to give the GP something to fit before acquisition-guided
  placement is informative.
- `targets` — `Dict[str, float]` of `jmbc.diagnostics.gini`/`top_shares`
  output names (`capital_gini`, `top_0.1_share`, `top_0.01_share`) to target
  values.

```bash
python calibrate_wealth_distortion.py                        # config.yaml as-is
python calibrate_wealth_distortion.py n_agents=200 device=cpu
python calibrate_wealth_distortion.py "targets={top_0.1_share: 0.70, capital_gini: 0.85}"
```

## Outputs

Same `results.csv` (now columns `scale`/`a`/`b`/`score` + simulated and
target stats) + `results/raw/{best,all}/` raw-data convention as the sibling
experiment (rollout, trained params/network, plus `kappas.npy`/`shape.npy`
for this experiment specifically).

`calibration_fit.png` — three panels: score-convergence scatter (colored by
evaluation order, best starred); the best evaluation's Lorenz curve with
each `top_*` target marked as the point a curve hitting it exactly would
pass through; and the best-fit distorted kappa profile plotted against the
raw (undistorted) Xavier curve, so you can see at a glance how much the
calibration actually reshaped the input.

## Caveats

Same annualization/gross-vs-net-return assumptions as the sibling experiment
apply wherever this design's outputs get compared back to Xavier's numbers
descriptively (the third plot panel) — they just no longer matter for the
fit score itself, since the score never touches the return curve, only
wealth. `n_agents=1000` cost caveats from the sibling README apply
unchanged.
