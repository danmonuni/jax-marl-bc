# JaxMARL-BC: Hardware-Accelerated Multi-Agent Reinforcement Learning for Heterogeneous Macroeconomic Models

**Daniele Montagnani**
Università La Sapienza di Roma
*Draft — 2 July 2026*

## Abstract

Multi-agent reinforcement learning (MARL) has recently been proposed as a bridge between heterogeneous-agent general-equilibrium models and agent-based models: boundedly rational agents learn equilibrium behaviour by optimising rewards, removing both the rational-expectations fixed point of the former and the hand-crafted decision rules of the latter. The reference implementation of this programme, MARL-BC (Gabriele et al., 2026), demonstrates the idea on real-business-cycle (RBC) economies but is built on CPU-bound tooling (PettingZoo, Stable-Baselines3) that requires hours of wall-clock time for hundreds of agents, placing empirically calibrated experiments — thousands of households with realistic return heterogeneity (Xavier, 2021) — out of reach. We present **JaxMARL-BC**, a pure-JAX reimplementation of the MARL-BC framework on the JaxMARL interface in which environment dynamics, PPO training, and evaluation rollouts compile to a single fused XLA program. On a free-tier Colab T4 GPU the system sustains up to 1.8 × 10⁵ environment transitions per second at wall-clock times that are near-invariant to the number of parallel environments, and it trains economies whose *sequential* training length is independent of batch width by construction. Equally important, we pair the simulator with an **economic verification suite** — Euler-equation residuals, aggregate resource-constraint sentinels, closed-form comparisons, Krusell–Smith forecast statistics, and distributional probes computed across training snapshots — and show that it detects genuine defects invisible to RL-level metrics: a degenerate aggregate-shock construction that silently removes business cycles from the Krusell–Smith economy as the population grows, and an error in the closed-form labour-supply solution reported in the original MARL-BC paper (its Eq. 13), which we correct analytically and confirm numerically. We argue that such diagnostics should be a standard component of machine-learning-based economic simulators, on par with throughput benchmarks.

**Keywords:** multi-agent reinforcement learning, JAX, real business cycle, Krusell–Smith, heterogeneous agents, GPU acceleration

---

## 1 Introduction

Macroeconomic models with rich household heterogeneity face a well-known computational dilemma. Heterogeneous-agent general-equilibrium (GE) treatments — Aiyagari (1994), Krusell and Smith (1998), and their HANK descendants — impose rational expectations and require solving a fixed point over the cross-sectional distribution, which sharply limits the heterogeneity that can be modelled. Agent-based models scale to arbitrary populations but require the modeller to specify behavioural rules directly. Gabriele, Glielmo and Taboga (2026) propose MARL-BC as a synthesis: households are independent PPO/SAC learners embedded in an RBC production economy, behaviour emerges from reward maximisation, and the framework provably recovers the representative-agent RBC solution at *n* = 1 and the Krusell–Smith mean-field limit at *n* ≫ 1.

The bottleneck of this programme is computational. MARL-BC's reference implementation steps a Python environment once per transition and trains with CPU-based Stable-Baselines3; the original paper reports roughly two hours to train its largest configuration (529 agents, ~5 × 10⁷ steps) on a single CPU. Yet the scientifically interesting regime is precisely the large-*n* one: matching the *empirical* joint distribution of wealth and returns — where moving from the 20th to the 99th wealth percentile raises average annual returns from 3.6% to 8.3% (Xavier, 2021) — requires thousands of heterogeneous households and long horizons.

A parallel literature has shown that end-to-end JAX compilation removes exactly this bottleneck in adjacent domains: PureJaxRL (Lu et al., 2022) for single-agent RL, JaxMARL (Rutherford et al., 2023) for multi-agent benchmarks, JAX-LOB (Frey et al., 2023) for limit-order-book simulation, and EconoJax (Ponse et al., 2024) for an AI-Economist-style economy. This paper applies the same design discipline to the MARL-BC class of models, and adds a component those systems papers lack: a battery of *economic-correctness* diagnostics.

Our contributions are:

1. **A pure-JAX reimplementation of the MARL-BC framework** (RBC with endogenous labour, heterogeneous-productivity RBC, and a Krusell–Smith variant) as `jaxmarl.MultiAgentEnv` subclasses, trained by a PPO loop expressed entirely in `jax.lax.scan` so that environment stepping, advantage estimation, and optimisation compile to one XLA program. Environments, algorithm, configuration, diagnostics, and benchmarking total ~2,400 lines of Python.
2. **An economic verification suite** computed at log-spaced training snapshots: Den Haan–Marcet-style Euler-equation residuals with borrowing-constraint filtering, an exact aggregate resource-constraint sentinel, closed-form policy comparison for the analytically solvable RBC, Krusell–Smith law-of-motion R² and Den Haan (2010) dynamic-forecast statistics, and distributional probes (Gini, Lorenz, top wealth shares, propensity-to-consume curves).
3. **Verification results.** The suite certifies the RBC economies (resource identity satisfied to ~10⁻⁸ relative error; learned policies within 1.8 and 0.7 percentage points of the closed-form consumption and labour choices) and *detects two substantive defects*: (i) a majority-vote aggregate-shock construction whose business cycle degenerates as *n* grows — for *n* ≥ 20 the aggregate state never switches, silently converting the Krusell–Smith economy into a no-aggregate-risk economy; (ii) an α ↔ (1−α) transposition in the closed-form labour supply of the original paper's Eq. 13, which we correct by derivation and confirm by direct numerical optimisation of the environment's own dynamics.
4. **A reproducible benchmark harness** (typed configuration with single-source-of-truth schema, Cartesian sweep runner, compile-time/run-time separation) under which training length is measured in *sequential* environment steps, independent of the number of parallel environments — making scaling comparisons well-posed — and which sustains 1.8 × 10⁵ transitions/s on a free Colab T4.

Section 2 specifies the economic environments. Section 3 describes the system design. Section 4 presents the verification methodology and results, Section 5 the performance results, and Section 6 limitations and the road to empirically calibrated 10⁴-agent economies.

## 2 Economic environments

We implement the MARL-BC economy exactly as specified by Gabriele et al. (2026, §2). At each period *t*, *n* households indexed by *i* hold capital *k*ᵗⁱ and are endowed with fixed capital and labour productivities (κⁱ, λⁱ). Effective aggregates are population means,

> K_t = (1/n) Σᵢ κⁱ kᵗⁱ,  L_t = (1/n) Σᵢ λⁱ ℓᵗⁱ,  (1)

production is Cobb–Douglas,

> Y_t = A_t K_t^α L_t^{1−α},  (2)

and factor markets are competitive, so individual returns and wages are proportional to marginal products,

> rᵗⁱ = α (Y_t / K_t) κⁱ,  wᵗⁱ = (1−α) (Y_t / L_t) λⁱ.  (3)

Household wealth is

> aᵗⁱ = wᵗⁱ ℓᵗⁱ + rᵗⁱ kᵗⁱ + (1−δ) kᵗⁱ.  (4)

The household's action is a consumption *fraction* ĉᵗⁱ ∈ [0.01, 0.99] (and, where labour is endogenous, a labour supply ℓᵗⁱ in the same clipped interval), so that cᵗⁱ = ĉᵗⁱ aᵗⁱ and k^i_{t+1} = (1−ĉᵗⁱ) aᵗⁱ. The per-period reward is the log-utility flow

> Rᵗⁱ = log cᵗⁱ + b log(1 − ℓᵗⁱ).  (5)

Three experimental designs instantiate this template:

- **RBC** (*n* = 1, κ = λ = 1): technology follows log A_t = ρ log A_{t−1} + σ ε_t. With full depreciation (δ = 1) the model has a closed-form solution used for verification (§4.2); with δ = 0.025 it corresponds to the standard quarterly calibration.
- **General** (heterogeneous RBC): *n* households on a grid of (κⁱ, λⁱ) values, generating heterogeneous returns and wages and, endogenously, wealth inequality.
- **Krusell–Smith** (exogenous labour, one-dimensional action): a two-state aggregate technology A_t ∈ {0.98, 1.02} and idiosyncratic employment ℓᵗⁱ ∈ {0, 1.11} jointly follow the four-state Markov chain of Krusell and Smith (1998), calibrated so that unemployment is 4% (10%) in good (bad) times and both regimes last eight periods on average.

Throughout, β = γ = 0.95: the households' economic discount factor and the RL discount coincide, so the learning objective *is* the household problem.

## 3 System design

### 3.1 Everything is a compiled tensor program

Both environments are stateless `flax.struct` dataclasses with `step_env`/`reset` as pure jitted functions on the JaxMARL `MultiAgentEnv` interface, so any JaxMARL-compatible algorithm can train them unchanged. Population structure is expressed as array dimensions, not Python objects: one environment step for *n* households is a fixed sequence of `O(n)` vector operations (two reductions for the aggregates, elementwise price/wealth/transition updates), and `jax.vmap` batches this across parallel environments. The PPO loop (shared-parameter actor–critic, as in the original's "social learning" configuration; 64×64 tanh MLP) follows the PureJaxRL pattern: rollout collection, generalised advantage estimation, and minibatched epochs are nested `lax.scan`s inside a single jitted `train` function. A full training run — including parameter snapshots for diagnostics — is one XLA computation; Python is re-entered only afterwards.

Two consequences matter for the experiments below. First, wall-clock cost is governed by the *sequential* depth of the program (number of PPO updates × rollout length), while the batch width (parallel environments × households) is absorbed by the accelerator until it saturates. Second, because evaluation rollouts (§4) are also `lax.scan` programs, diagnostics add seconds, not minutes.

### 3.2 Configuration and benchmarking discipline

All experiment parameters live in typed dataclasses validated by OmegaConf; YAML files and CLI dot-list overrides merge onto this schema (defaults < `base.yaml` < experiment file < CLI), and every run archives its fully resolved configuration alongside metrics, diagnostics, and timing. A sweep runner evaluates the Cartesian product of declared axes and tabulates one row per cell.

Two measurement conventions avoid classic benchmarking artefacts:

- **Compile/run separation.** Each benchmark cell executes the jitted program twice; the second, compilation-free timing defines throughput. JIT compilation (0.9–1.8 s for the configurations reported here) would otherwise dominate small cells and vanish in large ones, distorting scaling curves.
- **Sequential-step semantics.** `total_timesteps` counts sequential environment steps: one step advances all parallel environments at once, so training length — the number of gradient updates and the data-generating horizon — is *invariant* to the parallelism axis. Sweeps over `num_envs` therefore compare equal sequential work (weak scaling); total collected transitions scale as `total_timesteps × num_envs` and are recorded per cell so that comparisons at equal data are also possible. Under the alternative (and common) convention of fixed total transitions, wall-clock time *falls* as environments are added — a correct but easily misread consequence of fewer sequential updates.

## 4 Economic verification

RL-level metrics (returns, losses, KL divergences) certify that optimisation worked, not that the learned economy is *correct*. We therefore evaluate deterministic (mean-action) policy rollouts at log-spaced training snapshots and subject them to probes with known ground truth. All statistics discard a 50% burn-in and, where accounting identities are evaluated, mask auto-reset boundaries.

### 4.1 Probes

**Resource-constraint sentinel.** Summing (4) over households and using (1)–(3), mean wealth must satisfy ā_t = Y_t + (1−δ) k̄_t *exactly*. This identity holds by construction, so its residual is a sentinel for implementation error in the recorded quantities (timing misalignment, wrong aggregation weights). Measured: 5.6 × 10⁻⁸ (RBC) to 1.7 × 10⁻⁸ (KS) mean relative residual — float32 rounding, as required.

**Euler-equation residuals.** For log utility the intertemporal first-order condition is 1/c_t = β E_t[(1/c_{t+1}) R_{t+1}] with R_{t+1} = 1 − δ + r_{t+1}. We report the distribution of the per-realisation residual e_t = β (c_t/c_{t+1}) R_{t+1} − 1 in consumption units (Den Haan and Marcet, 1994, in spirit), excluding steps where consumption falls below a floor (borrowing-constrained agents, for whom the unconstrained condition need not hold) and reporting the excluded fraction. Return timing is aligned so that R_{t+1} is the realised gross return on capital chosen at *t*.

**Closed-form comparison (textbook RBC).** With δ = 1 the model solves in closed form. The savings rate out of wealth is αβ, hence the optimal consumption fraction is ĉ\* = 1 − αβ. For labour supply, combining the intratemporal condition b c_t/(1−ℓ) = (1−α)Y_t/ℓ with c_t = (1−αβ)Y_t yields

> ℓ\* = (1−α) / [ b(1−αβ) + (1−α) ] = 0.1629  (α = 0.36, β = 0.95, b = 5).  (6)

We note that Eq. 13 of Gabriele et al. (2026) instead reports ℓ\* = α/[b(1−(1−α)β)+α] = 0.1552, which is the solution of the transposed economy Y = K^{1−α}L^α and does not satisfy the model's own optimality conditions. We confirmed (6) independently of the derivation by numerically maximising discounted welfare over constant policies on the implemented environment dynamics: the optimum is (ĉ, ℓ) = (0.6580, 0.1628), matching (6) to the third decimal and strictly dominating the Eq. 13 value in achieved welfare (−72.605 vs −72.632).

**Krusell–Smith forecast quality.** Following Krusell and Smith (1998) we regress the aggregate capital law of motion per aggregate state and report R², together with the stricter Den Haan (2010) statistic: the rule is iterated on its *own* predictions through the realised shock sequence and the maximal percentage deviation from the simulated path is reported.

**Distributional probes.** Gini coefficient, Lorenz curves, top-1%/10% wealth shares, and consumption-fraction-by-wealth curves, tracked across snapshots.

### 4.2 Verification results

Trained policies (PPO, 10⁶ collected transitions for RBC; 10⁷ for KS) yield:

| Probe | RBC textbook (δ=1) | RBC typical (δ=0.025) | Krusell–Smith |
|---|---|---|---|
| Resource residual (rel.) | 1.7 × 10⁻⁸ | ~10⁻⁸ | 5.6 × 10⁻⁸ |
| Euler mean abs. residual | 0.056 | 0.037 | 0.037 |
| Constrained fraction | 0.0 | 0.0 | 0.003 |
| ĉ learned vs ĉ\* | 0.676 vs 0.658 | — | — |
| ℓ learned vs ℓ\* (Eq. 6) | 0.170 vs 0.163 | — | — |
| Capital Gini | — | — | 0.098 |

Three observations. First, the single learned household reproduces the closed form to within 1.8 (consumption) and 0.7 (labour) percentage points — the labour error being *half* of what a comparison against the original Eq. 13 would report. Second, Euler residuals are nearly constant across time in the RBC runs (p99 ≈ median), indicating a small *systematic* intertemporal wedge — slight over-consumption relative to optimum — rather than noise; this motivates reporting signed means alongside absolute ones. Third, the RBC economies pass every probe with no probe-specific tuning.

### 4.3 Defects detected by the suite

The suite's purpose is falsification, and it succeeded twice.

**Degenerate aggregate risk in the KS environment.** The implementation draws each household's next (aggregate, employment) state independently from its own row of the joint transition matrix and defines the economy's aggregate state as the *majority vote* of the households' aggregate components. Because each household's aggregate draw is an independent Bernoulli with persistence 7/8, the vote concentrates exponentially in *n*: simulating the mechanism for 5,000 periods yields switch rates of 0.015 at *n* = 5 (true value: 0.125) and *exactly zero* for *n* ∈ {20, 100, 1000, 10000}. For any population of practical interest the "business-cycle" economy has no business cycle, all agents face only idiosyncratic risk, and the KS law-of-motion regression degenerates to a single-state autoregression (measured R² = 0.904; literature-standard values exceed 0.9999). No RL metric flags this — training converges, rewards are sensible — but the Den Haan statistic (446%, against <1% for acceptable solutions) and the frozen aggregate-state trace expose it immediately. The correct construction draws the aggregate transition once from its two-state marginal chain and household transitions conditionally on the realised aggregate pair, as in Krusell and Smith (1998).

**Closed-form transposition.** The Eq. 13 issue of §4.1 propagated from the original paper into our diagnostics and reference figures before being caught by cross-checking the analytical target against direct numerical optimisation — a reminder that verification suites must themselves be verified against independent computation, not only against published formulas.

We report both defects with fixes in hand rather than silently repairing them: the first affects the interpretation of all majority-vote KS results (including, potentially, mean-field claims in downstream work), and the second affects the original paper's Figure 3 reference line.

## 5 Performance

All measurements use `float32`, PPO with rollout length 200, 10 epochs, and the compile-excluded timing convention of §3.2. On a Colab T4 (free tier), with 10 households and a fixed budget of 4 × 10⁵ collected transitions per cell:

| Parallel envs | Wall-clock (s) | Throughput (transitions/s) |
|---:|---:|---:|
| 2 | 21.8 | 1.8 × 10⁴ |
| 8 | 7.3 | 5.5 × 10⁴ |
| 32 | 2.8 | 1.4 × 10⁵ |
| 64 | 2.2 | 1.8 × 10⁵ |

Throughput scales near-linearly to 32 environments and begins to saturate at 64 (1.28× from a further doubling), reflecting fixed sequential costs (10 optimisation epochs per update) and rising occupancy. Under the sequential-step semantics the same table reads as weak scaling: stepping 64 environments costs within a factor ~1.1 of stepping 2 on CPU smoke tests, so additional parallel data is nearly free until the device saturates. Compilation costs 0.9–1.8 s per configuration and is incurred once per shape.

For orientation rather than as a controlled comparison: the original CPU-based implementation reports roughly two hours for its largest experiment (529 agents, ~5 × 10⁷ transitions ≈ 7 × 10³ transitions/s); the harness records a `method` column precisely so that the original implementation can be run under identical budgets and overlaid as a *standard vs JaxMARL-BC* series. We deliberately refrain from quoting a speed-up factor until that head-to-head is executed on matched hardware and budgets.

## 6 Limitations and future work

**Pending environment fix.** The KS aggregate-shock correction of §4.3 is designed but not yet merged; all KS quantitative results above should be read as characterising the *defective* economy and as evidence for the diagnostics, not as economic results. Post-fix, the Den Haan statistic must be recomputed on episode-segmented, log-transformed capital paths (the current estimate is additionally contaminated by auto-reset boundaries within evaluation rollouts).

**Statistical scope of Euler diagnostics.** Per-realisation residuals conflate approximation error with realised expectational noise; they bound solution accuracy but do not decompose it. Orthogonality-based Den Haan–Marcet tests and signed-mean reporting are straightforward extensions.

**Toward empirically calibrated inequality.** The scientific target motivating the engineering is a Krusell–Smith economy with ~10⁴ households whose return heterogeneity is disciplined by the Survey-of-Consumer-Finances evidence of Xavier (2021) — a wealth-return gradient from 3.6% to 8.3% — via the κⁱ distribution of Eq. (3), asking whether learned savings behaviour reproduces the observed top-10% wealth share. The throughput results of §5 indicate this is feasible on commodity hardware once the aggregate-shock fix lands.

**Comparative training.** The framework trains with PPO only; the original reports SAC as more sample-efficient in the multi-agent regime. A jitted SAC under the same interface is mechanical future work.

## 7 Conclusion

JaxMARL-BC turns the MARL-BC research programme into a compiled tensor program: economies that took hours on CPU tooling train in seconds on a free GPU, training length is well-defined independently of parallelism, and every run ships with an audit of its own economic validity. The audit is not decoration — on first contact it found a business cycle that wasn't there and a textbook formula that was wrong, including in the peer-reviewed source. As machine-learning simulators enter economics, we consider this the paper's transferable lesson: *report the physics checks, not only the learning curves.*

## References

- Aiyagari, S. R. (1994). Uninsured idiosyncratic risk and aggregate saving. *Quarterly Journal of Economics*, 109(3), 659–684.
- Den Haan, W. J. (2010). Assessing the accuracy of the aggregate law of motion in models with heterogeneous agents. *Journal of Economic Dynamics and Control*, 34(1), 79–99.
- Den Haan, W. J., & Marcet, A. (1994). Accuracy in simulations. *Review of Economic Studies*, 61(1), 3–17.
- Frey, S., Li, K., Nagy, P., Sapora, S., Lu, C., Zohren, S., Foerster, J., & Calinescu, A. (2023). JAX-LOB: A GPU-accelerated limit order book simulator to unlock large-scale reinforcement learning for trading. *Proceedings of the 4th ACM International Conference on AI in Finance (ICAIF)*.
- Gabriele, F., Glielmo, A., & Taboga, M. (2026). Heterogeneous RBCs via deep multi-agent reinforcement learning. *Proceedings of the 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS)*. arXiv:2510.12272.
- Krusell, P., & Smith, A. A. (1998). Income and wealth heterogeneity in the macroeconomy. *Journal of Political Economy*, 106(5), 867–896.
- Lu, C., Kuba, J., Letcher, A., Metz, L., Schroeder de Witt, C., & Foerster, J. (2022). Discovered policy optimisation. *Advances in Neural Information Processing Systems*, 35.
- Ponse, K., Plaat, A., van Stein, N., & Moerland, T. M. (2024). EconoJax: A fast & scalable economic simulation in JAX. arXiv:2410.22165.
- Rutherford, A., Ellis, B., Gallici, M., et al. (2023). JaxMARL: Multi-agent RL environments and algorithms in JAX. arXiv:2311.10090.
- Xavier, I. (2021). Wealth inequality in the US: The role of heterogeneous returns. SSRN Working Paper 3915439.
