# The Wealth-Bimodality Phase Transition: A First-Principles Autopsy

The population-scaling runs (`runs/ks/ks_n20`, `ks_n200`, `ks_n2000`) show a
strange phenomenon: the wealth distribution starts unimodal, **splits into two
separated wealth classes** mid-training (Gini jumping from ~0.08 to ~0.7), and
then — in two of the three runs — **merges back** into a single mode by the
end. A "double phase transition, into and out of bimodality."

This report explains the phenomenon completely, from scratch: what was
measured, every explanation that was ruled out (with the tests that ruled them
out), the actual mechanism, why it appears *and* disappears during training,
and why one run (`ks_n200`) never escaped it. No prior knowledge of dynamical
systems or RL internals is assumed.

**The one-paragraph answer, up front.** Mid-training, the neural policy
develops a *cliff*: households below a wealth threshold consume ~27% of their
wealth per period, households above it consume ~1% — not because 1% is optimal
but because the network's output is *saturated* at its lower bound for
high-wealth inputs. A cliff-shaped consumption rule makes individual wealth
accumulation **bistable**: there are two self-sustaining wealth levels, one
poor and one rich, and every agent gets pulled to one or the other depending
on its employment luck. That is the bimodality. As training continues, the
policy smooths the cliff, the two levels merge into one, and the distribution
collapses back to unimodal — the second transition. Nothing was changed in the
economics; no heterogeneity in preferences (and **no stochastic betas** —
β = 0.99 is one deterministic number shared by every agent). The whole episode
is a transient property of *where the learning algorithm passes through
policy-space*, made possible by unnormalized observations feeding a saturating
network.

---

## Contents

1. [What was actually observed](#1-what-was-actually-observed)
2. [Background you need (5 concepts)](#2-background-you-need)
3. [Ruled-out explanations, with the tests](#3-ruled-out-explanations)
4. [The mechanism, step by step](#4-the-mechanism-step-by-step)
5. [Why it appears *and* disappears: a bifurcation along the learning path](#5-why-it-appears-and-disappears)
6. [Why `ks_n200` never got out](#6-why-ks_n200-never-got-out)
7. [Why the standard KS diagnostics missed it (almost)](#7-why-the-standard-diagnostics-missed-it)
8. [What to do about it](#8-what-to-do-about-it)
9. [Appendix: every number in this report, and how to recompute it](#9-appendix)

---

## 1. What was actually observed

Setup, in brief (details in `complexity-report.md`): a Krusell–Smith economy —
$n$ households, each with capital $k^i_t$, hit by idiosyncratic
employment shocks and one common good/bad productivity shock; all households
share **one** neural policy that maps each household's observation to its
consumption decision. Three runs were trained, identical in *everything*
(optimizer steps, learning-rate schedule, initial network weights, parallel
environments) except the population: $n = 20, 200, 2000$. Every run was
evaluated 12 times during training ("snapshots"): freeze the policy, simulate
a fresh 5000-step economy with **2000** agents, discard the first 2500 steps
as burn-in, and measure the wealth distribution on the rest.

The observations, from the saved figures and diagnostics:

| | early snapshots | mid-training (~update 1000) | final (update 2000) |
| :-- | :-- | :-- | :-- |
| `ks_n20` | one tight mode, mean k ≈ 1–2 | brief split (band at k≈65–78) | one mode, K = 28.0, **Gini 0.084** |
| `ks_n200` | one tight mode | splits, collapses, splits again | **still split**: K = 55, **Gini 0.819**, modes at k≈4 and k≈408 |
| `ks_n2000` | one tight mode | split: modes near k≈8 and k≈130–280, **Gini 0.71** | one mode, K = 26.8, **Gini 0.086** |

So: distribution unimodal → bimodal → unimodal (except `ks_n200`, stuck
bimodal). The inequality index spikes by an order of magnitude in the middle
and comes back. That is the "double phase transition."

## 2. Background you need

Five concepts, each in a few sentences. If these are familiar, skip to §3.

### 2.1 The consumption fraction (what the policy controls)

Each period a household has **cash-on-hand** $a$: this period's labour income
plus capital income plus its undepreciated capital,

$$a_t = \underbrace{w_t \ell_t}_{\text{wages}} + \underbrace{r_t k_t}_{\text{capital income}} + \underbrace{(1-\delta)k_t}_{\text{what's left of capital}}.$$

The policy outputs one number, the **consumption fraction**
$\hat c \in [0.01,\, 0.99]$: consume $\hat c \cdot a$, save the rest as next
period's capital:

$$k_{t+1} = (1 - \hat c)\, a_t .$$

Note the hard bounds: the raw network output $u \in [-1, 1]$ is rescaled as
$\hat c = \text{clip}\big(\tfrac{u+1}{2},\, 0.01,\, 0.99\big)$
(`jmbc/envs/env_ksmf.py`, `_step_core`). **$\hat c = 0.01$ means "the network
asked for the minimum consumption the environment allows."** Keep this in
mind; it is the smoking gun later.

### 2.2 Fixed points of the wealth map, stable and unstable

Fix the policy and (for the moment) the prices. Then next period's capital is
just a function of this period's: $k_{t+1} = f(k_t)$, the **wealth map**. A
**fixed point** is a wealth level that reproduces itself: $f(k^\*) = k^\*$ —
graphically, where the curve $f$ crosses the 45° line.

A fixed point is **stable** if wealth slightly above it shrinks back and
wealth slightly below it grows back (the curve crosses the 45° line from
above, slope < 1): nearby households get *pulled in*. It is **unstable** if
deviations amplify (slope > 1): households get *pushed away*, up or down. An
unstable fixed point is a **watershed**: land above it and you drift to the
next stable point above; land below and you drift down.

### 2.3 Bistability

If the wealth map crosses the 45° line **three times** — stable, unstable,
stable — the system is **bistable**: two self-sustaining wealth levels
coexist under the *same* policy and the *same* prices. Which one a household
ends up at depends only on which side of the watershed its shock history
leaves it. A population of identical agents then splits into two classes:
that is a **bimodal** distribution (two humps). With shocks occasionally
kicking agents across the watershed, both classes stay populated forever and
agents churn between them.

```text
 k'   (next-period capital)                        rich branch: save 99%
  |                                          ●S ← stable (GE brake: r has fallen)
  |                                     ____/
  |                               _____/
  |                          ____/
  |                      ___/  ← the CLIFF region
  |                 ●U _/         (unstable crossing = watershed)
  |        ________/
  |    ●S /   ← stable poor point (consume 27%)
  |   /
  +--/------------------------------------------  k  (this-period capital)
        the 45° line  k' = k
```

### 2.4 Saturation of a tanh network

The policy network squashes its inputs through tanh nonlinearities. tanh is
essentially flat outside $[-2, 2]$: once a unit's input is large, *the output
stops responding to the input at all*. The observations here include **raw
capital** — which ranges over 1 to 400 in these runs — with no normalization.
So for wealthy households the network's internals are pushed deep into the
flat region: the policy returns the *same extreme action for every rich
agent, regardless of how rich*. When that extreme action is $u = -1$, the
consumption fraction pins at the clip floor 0.01. This is called **action
saturation**, and the training code tracks exactly this (the fraction of
actions within 2% of the bounds — the `action_saturation` metric).

### 2.5 The Gini coefficient, and what a value of 0.8 means

Gini ∈ [0, 1] measures inequality: 0 = everyone identical, 1 = one agent owns
everything. A two-class distribution with ~90% of agents at $k \approx 4$ and
~10% at $k \approx 400$ puts ~92% of all wealth in the top decile ⇒ Gini
≈ 0.8. So the observed spike to 0.7–0.82 is not "somewhat more unequal" — it
is the signature of a **class split**, which is what makes it diagnostic.

## 3. Ruled-out explanations

Each of these was a plausible cause. Each was tested against the saved run
data (`rollouts.npz`; exact code in the Appendix) and excluded.

**(a) "Did we introduce stochastic / heterogeneous betas?" — No.**
The discount factor is a single scalar `env.beta = 0.99`, identical and
deterministic for every agent, and the RL discount `train.gamma = 0.99`
matches it. Heterogeneous betas were *proposed* in `complexity-report.md` as a
future model extension; they are not implemented anywhere. Likewise there is
no ex-ante heterogeneity of any kind in these runs: `kappas`/`lambdas` are
unset, which resolves to all-ones (`jmbc/envs/registry.py`,
`_resolve_weights`). Every agent is identical ex ante. Whatever splits them is
*generated*, not assumed.

**(b) "Is it a pooling artifact?" — No.** The reported distribution pools the
post-burn-in eval steps. If the economy were still *trending* during the eval
(e.g. wealth still growing), pooling early-poor and late-rich time steps would
fake a bimodal histogram even if the cross-section at each instant were a
single tight mode. Test: compute Gini at *individual time instants* and
compare with the pooled value. Result — indistinguishable (n2000 mid-training
snapshot: pooled 0.712 vs per-instant 0.711; n200 final: 0.809 vs 0.809). The
split exists *within each instant*: at any given moment, some agents are poor
and some are rich. Genuinely cross-sectional.

**(c) "Is the eval non-stationary?" — No.** Aggregate capital $K(t)$ over the
post-burn window is flat (n2000 snapshot 11: 5th–95th percentile of the
*time* variation is 60.5–65.6 around a median 63.5; it starts at 67 and ends
at 63). The system is at its stochastic steady state; the steady state itself
is bimodal.

**(d) "Are the two classes just frozen initial luck?" — No.** The rank
correlation between an agent's wealth at the start and at the end of the
2500-step measurement window is only ≈ 0.4. Agents *churn* between the
classes — poor agents occasionally break out upward, rich agents occasionally
crash down. This is exactly the signature of bistability with shocks (§2.3),
and rules out any story where the split is a one-off accident that then
freezes.

## 4. The mechanism, step by step

Now the positive case. All numbers are from `ks_n2000`, snapshot 11 (the
mid-training bimodal phase, ~update 1024); `ks_n200`'s final snapshot shows
the same structure even more extremely.

**Step 1 — measure the policy the agents actually follow.** The eval rollouts
record every agent's consumption fraction and wealth at every step. Group
agents by wealth decile and take the median consumption fraction in each:

| wealth decile | typical $k$ | consumption fraction $\hat c$ | median $k'/k$ |
| :-- | --: | --: | --: |
| 1 (poorest) | 5.1 | 0.300 | 1.0000 |
| 2–8 | 7.1 – 8.7 | 0.268 – 0.277 | 1.0000 |
| 9 | 126.6 | 0.024 | 1.0000 |
| 10 (richest) | 278.6 | **0.011** | 1.0000 |

Read this table three times; it contains the whole phenomenon.

**Step 2 — the cliff.** The consumption fraction is ≈ 0.27, flat, across
deciles 1–8 ($k \le 9$), then plunges to ≈ 0.01–0.02 by decile 9
($k \approx 127$). Somewhere between $k \approx 9$ and $k \approx 127$ the
policy falls off a cliff. And 0.011 ≈ the clip floor (§2.1): the rich are not
consuming 1% because some Euler equation says so — the actor is **saturated**
(§2.4), outputting its most extreme action for every observation in that
region. The `action_saturation` training metric independently confirms it:
it bumps upward around update ~850 and decays to ~0 by the end — precisely
bracketing the bimodal snapshots.

**Step 3 — the cliff makes the wealth map bistable.** Recall
$k' = (1-\hat c)\,a(k)$ with cash-on-hand roughly linear in $k$:
$a(k) \approx R k + w\ell$ where $R = 1 - \delta + r$ is the gross return.

- *Below the cliff*: $\hat c \approx 0.27$, so $k' \approx 0.73\,(Rk + w\ell)$.
  The slope $0.73 R < 1$: the map crosses the 45° line from above ⇒ a
  **stable poor fixed point**, at $k \approx 8$. Households there consume
  enough that wealth self-stabilizes at a low level.
- *Above the cliff*: $\hat c \approx 0.01$, so $k' \approx 0.99\,(Rk + w\ell)$,
  slope $0.99R > 1$ at the prevailing prices: wealth **compounds**. What stops
  it? General equilibrium: as the rich accumulate, aggregate capital rises,
  the marginal product of capital — the interest rate $r$ — falls, until
  $0.99\,R = 1$. That happens around $k \approx 280$: a **stable rich fixed
  point** created by the price system, not by the policy.
- *At the cliff*: in between, the map crosses the 45° line from below — the
  **unstable watershed** (§2.2).

The empirical check is the last column of the table: median $k'/k = 1.0000$
in *every* decile. Poor agents are sitting at a fixed point; rich agents are
sitting at a *different* fixed point; both reproduce themselves. Textbook
bistability.

**Step 4 — shocks populate both classes and mix them.** Idiosyncratic
unemployment (zero labour income for a spell) knocks some agents below the
watershed; lucky long employment spells push some above it. Hence: two humps,
both permanently populated, with churn (rank correlation 0.4, §3d). The
population of ex-ante *identical* agents has endogenously split into a
consuming class and an accumulating class.

**Step 5 — the Gini follows mechanically.** Two classes at $k \approx 8$ vs
$k \approx 130$–280 with roughly a 80/20 split ⇒ top decile holds ~half the
wealth ⇒ Gini ≈ 0.7 (§2.5). At `ks_n200`'s final snapshot (modes 4 vs 408)
the same arithmetic gives ≈ 0.82. The Gini spike *is* the class split, seen
through a summary statistic.

## 5. Why it appears *and* disappears

The mechanism above explains one bimodal snapshot. The "double phase
transition" is about the *training trajectory*: why does the cliff exist only
in the middle of training?

Think of training as a continuous path $\theta(t)$ through the space of
network weights. At each point of the path, the induced wealth map has some
number of fixed points — 1 or 3. The number changes at sharp boundaries (in
dynamical-systems language, *saddle-node bifurcations*: a stable/unstable pair
of fixed points is born or annihilated as the curve tangentially touches the
45° line). The training path **enters** the 3-fixed-point region and later
**exits** it. Both crossings are "phase transitions" of the induced economy,
even though $\theta$ moves smoothly. Concretely, in three acts:

- **Act I (early, snapshots 1–10 for n2000).** The freshly initialized policy
  consumes heavily everywhere ($\hat c$ ≈ 0.3–0.8 at all wealth levels). One
  fixed point, low: everyone hovers at $k \approx 1$–2. Unimodal, poor,
  Gini ≈ 0.07.
- **Act II (middle, around updates 500–1200).** The dominant early learning
  signal is "saving is undervalued" — the batch is full of poor agents whose
  returns to capital are high. The policy responds by lowering $\hat c$, and
  crucially it learns this in the *crudest representable form*: push the
  actor output down hard as wealth-related inputs grow, until — with raw
  $k$-values of order 100 flowing into tanh units — the output **saturates**
  at $u = -1$ for all high-wealth observations (§2.4). Why doesn't training
  immediately fix the exaggeration? Because PPO's gradient weights states by
  *visitation*: before any agent is rich, there is literally no data at high
  $k$, so nothing penalizes an absurd "consume 1% forever" rule up there.
  The saturated region is unsupervised until agents start living in it. The
  moment the cliff forms, the map is bistable, the rich class populates, and
  the distribution splits.
- **Act III (late, final snapshots for n20/n2000).** Now the rich basin *is*
  visited: ~10–20% of the training batch lives at high wealth, where
  over-saving is genuinely suboptimal (returns have been competed down;
  $\beta R < 1$; a log-utility household should consume more). Gradient
  signal finally flows to that region, the actor unsaturates, the cliff
  smooths into a monotone, gently-rising consumption schedule — the rich and
  poor fixed points move toward each other, annihilate against the watershed,
  and a **single** fixed point remains at $k \approx 27$. The distribution
  snaps back to unimodal (Gini 0.085); the second transition.

Every piece of this narrative is checkable against the recorded metrics and
is consistent: the reward and value-loss transients around updates 300–900,
the action-saturation bump and decay, the snapshot timing of the split, and
the final decile tables with no cliff.

## 6. Why `ks_n200` never got out

`ks_n200` ended training *inside* Act II: final Gini 0.819, rich mode pinned
at the 0.01 clip floor at $k \approx 408$, Euler-equation error 7× larger
than its siblings (0.074 vs ~0.01), and a heatmap that oscillates — split at
snapshot 9, merged at 10, split again at 11–12: a policy wobbling back and
forth *across* the bifurcation boundary from update to update.

Two things can be said rigorously, thanks to the equal-learning protocol
(`complexity-report.md` §9.7) under which these runs were built:

1. **It is not a training-budget artifact.** All three runs took *exactly*
   the same 32,000 gradient steps from *bit-identical* initial weights on the
   same learning-rate schedule with the same per-update aggregate information.
   `ks_n200` is not "less trained" — its optimization path genuinely
   traversed policy space differently and had not exited the bistable window
   when the budget ended.
2. **It is not yet attributable to $n = 200$ specifically.** One seed per
   cell. Escaping Act II depends on gradient signal from the rich region,
   which is noisy and could plausibly be slower at some population sizes —
   or this could be seed luck. The cheap discriminating experiment is
   multi-seed replication (attack vector E in `complexity-report.md`: `vmap`
   over seeds is nearly free); systematically stuck seeds at $n = 200$ would
   make this a real finding about population size and learning dynamics.

## 7. Why the standard diagnostics missed it (almost)

Through the *entire* episode — Gini swinging 0.07 → 0.71 → 0.09 — the
Krusell–Smith law-of-motion $R^2$ stayed at 0.9999 in every snapshot of every
run. This is Den Haan's classic critique made vivid: **approximate
aggregation is blind to the cross-section.** Aggregate capital tomorrow is
forecastable from aggregate capital today almost regardless of how wealth is
distributed, so an aggregate-fit statistic cannot certify a solution's
distributional content. (The Den Haan long-horizon error did twitch: 41% for
the stuck `ks_n200` vs 16–19% for the healthy runs — a partial flag.) The
practical lesson for the paper: distributional diagnostics — Gini and top
shares *per snapshot*, the wealth heatmap through training, and the
consumption-by-wealth-decile table of §4 — are not optional extras; here they
were the only instruments that saw the phenomenon at all.

## 8. What to do about it

Depending on whether the goal is to *eliminate* the artifact or *study* it:

**To suppress (recommended for the baseline results):**

1. **Normalize wealth observations** (divide own-capital and mean-capital by
   the steady-state scale before they enter the network). This attacks the
   root cause — tanh saturation at $k = \mathcal{O}(100)$ — and plausibly
   removes the cliff-forming failure mode entirely. One-line change in
   `_obs_matrix`.
2. **Keep late-training exploration alive** — a small entropy floor or a
   slower entropy collapse. Escaping a saturated basin needs gradient signal;
   a near-deterministic policy generates almost none.
3. **Rethink the 0.01 consumption floor.** The clip both enables the runaway
   (99% saving forever is representable and consequence-free until GE brakes
   it) and hides the actor's saturation behind a valid-looking action.
4. **Multi-seed everything** (vector E): the `ks_n200` outcome shows single
   seeds can land inside a transient window at the end of the budget; error
   bars across seeds make the phase structure visible instead of confusing.

**To study (it is genuinely interesting):** the transient phase is an
*endogenous class formation* mechanism — Gini 0.8 and a bimodal wealth
distribution out of ex-ante identical agents, produced by a boundedly-rational
(saturated) consumption rule plus general equilibrium. The empirical wealth
distribution's bimodality/heavy tail is usually obtained by *assuming*
heterogeneous patience; here a *learning trajectory* passes through it
naturally. The runs already contain everything needed (`rollouts.npz`
snapshots at 12 training stages); §9 reproduces every figure of merit.

## 9. Appendix

**Runs**: `runs/ks/ks_n{20,200,2000}` — equal-learning protocol
(`configs/exp/ks_n*.yaml`): $U = 2000$ updates × ($K{=}4$ epochs × $M{=}4$
minibatches), $E = 32$ envs, $T = 64$ rollout, seed 0, eval at 2000 agents,
5000 steps, burn 2500.

**Headline diagnostics** (`diagnostics.json`, final snapshot):

| run | capital Gini | mean K | Euler err | LOM $R^2$ | Den Haan max |
| :-- | --: | --: | --: | --: | --: |
| ks_n20 | 0.084 | 28.0 | 0.010 | 0.9999 | 19.3% |
| ks_n200 | **0.819** | 55.2 | **0.074** | 0.9999 | **41.2%** |
| ks_n2000 | 0.086 | 26.8 | 0.013 | 0.9999 | 16.5% |

**Key mid-training numbers** (`ks_n2000`, snapshot 11 ≈ update 1024): Gini
pooled 0.712 vs per-instant 0.711; $K(t)$ time-median 63.5 (p5–p95:
60.5–65.6); poor cluster $k \approx 8$, $\hat c \approx 0.27$; rich cluster
$k \approx 127$–279, $\hat c \approx 0.011$–0.024; median $k'/k = 1.0000$ in
all deciles; wealth rank-correlation over 2500 steps ≈ 0.40.

**Recompute everything** from a run directory:

```python
import numpy as np

def gini(v):
    v = np.sort(np.abs(v.flatten())); n = len(v); s = v.sum() + 1e-12
    return (2*np.dot(np.arange(1, n+1), v) / (n*s)) - (n+1)/n

z  = np.load('runs/ks/ks_n2000/rollouts.npz', allow_pickle=True)
s  = 10                              # snapshot index (0-based): 10 = mid-training
ks = z['ks'][s, 2500:]               # [time, saved_agents] post-burn capital
cf = z['c_fracs'][s, 2500:]          # consumption fractions actually played
w  = z['wealths'][s, 2500:]          # cash-on-hand

print('pooled', gini(ks), 'vs instant', np.mean([gini(ks[t]) for t in range(0, 2500, 100)]))

q = np.quantile(ks, np.linspace(0, 1, 11))          # decile table of §4
for i in range(10):
    m = (ks >= q[i]) & (ks < q[i+1])
    print(f'decile {i}: k~{np.median(ks[m]):8.1f}  c_frac={np.median(cf[m]):.3f}  '
          f"k'/k={np.median((1-cf[m])*w[m]/np.maximum(ks[m],1e-8)):.4f}")
```

**Related documents**: `complexity-report.md` (§9.3 gradient-noise anatomy,
§9.7 equal-learning protocol, §10 attack vectors A–J referenced above).
