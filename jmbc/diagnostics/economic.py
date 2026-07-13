"""Economic-accuracy diagnostics: does the trained policy behave like a correct,
realistic economy?

- Euler-equation errors (Den Haan-Marcet style): how well the consumption
  first-order condition is satisfied along the stationary path.
- Resource-constraint residual: aggregate accounting identity (a correctness
  sentinel; should be ~0 by construction).
- Steady-state summary: stationary moments of K, C, L.
- Analytical RBC comparison: deviation of the learned policy from the closed
  form (valid for the depreciation = 1 textbook case).
- Krusell-Smith forecasting quality: law-of-motion R^2 and the Den Haan
  statistic (max dynamic-forecast error).
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .distributional import stationary_slice


# ── Euler equation ────────────────────────────────────────────────────────────

def euler_errors(rec, beta: float, burn_frac: float = 0.5,
                 c_floor: float = 1e-4) -> Dict[str, float]:
    """Log-utility consumption Euler residual along the stationary path.

    resid_t = beta * (c_t / c_{t+1}) * R_{t+1} - 1, in consumption units
    (0 = exactly satisfied). Steps where either c_t or c_{t+1} falls below
    ``c_floor`` are treated as constrained/degenerate (e.g. unemployed agents at
    the borrowing limit, where the unconstrained Euler need not hold). These are
    excluded from the error statistics and reported via ``constrained_frac``.
    """
    c = stationary_slice(rec["cons"], burn_frac)
    R = stationary_slice(rec["R"], burn_frac)
    if c.shape[0] < 3:
        return {"euler_mean_abs": float("nan"), "constrained_frac": float("nan")}
    c_t, c_t1, R_t1 = c[:-1], c[1:], R[1:]
    resid = beta * (c_t / np.maximum(c_t1, 1e-12)) * R_t1 - 1.0

    interior = (c_t > c_floor) & (c_t1 > c_floor) & np.isfinite(resid)
    constrained_frac = float(1.0 - interior.mean())
    a = np.abs(resid[interior])
    if a.size == 0:
        return {"euler_mean_abs": float("nan"), "constrained_frac": constrained_frac}
    return {
        "euler_mean_abs": float(np.mean(a)),
        "euler_median_abs": float(np.median(a)),
        "euler_p99_abs": float(np.percentile(a, 99)),
        "euler_log10_mean_abs": float(np.log10(np.mean(a) + 1e-12)),
        "constrained_frac": constrained_frac,
    }


# ── Resource constraint ───────────────────────────────────────────────────────

def resource_residual(rec, delta: float, burn_frac: float = 0.5) -> Dict[str, float]:
    """Aggregate identity: mean(a_t) should equal Y_t + (1-delta)*mean(k_t).

    Auto-reset steps (where the recorded state is a fresh reset) are excluded.
    """
    w = stationary_slice(rec["wealths"], burn_frac).mean(axis=1)
    Y = stationary_slice(rec["Y"], burn_frac)
    k_in = stationary_slice(rec["k_in_mean"], burn_frac)
    if "done" in rec:
        keep = ~stationary_slice(rec["done"], burn_frac).astype(bool)
        w, Y, k_in = w[keep], Y[keep], k_in[keep]
    resid = w - (Y + (1.0 - delta) * k_in)
    rel = np.abs(resid) / np.maximum(np.abs(w), 1e-12)
    return {
        "resource_mean_abs": float(np.mean(np.abs(resid))),
        "resource_mean_rel": float(np.mean(rel)),
    }


# ── Steady state ──────────────────────────────────────────────────────────────

def steady_state(rec, burn_frac: float = 0.5) -> Dict[str, float]:
    K = stationary_slice(rec["K"], burn_frac)
    cons = stationary_slice(rec["cons"], burn_frac)
    ls = stationary_slice(rec["ls"], burn_frac)
    return {
        "K_mean": float(np.mean(K)),
        "K_std": float(np.std(K)),
        "C_mean": float(np.mean(cons)),
        "L_mean": float(np.mean(ls)),
    }


# ── Analytical RBC comparison ─────────────────────────────────────────────────

def analytical_rbc_compare(
    rec, alpha: float, beta: float, b: float, delta: float, burn_frac: float = 0.5
) -> Optional[Dict[str, float]]:
    """Compare learned (c_frac, labour) to the closed form.

    Closed form is exact only for the textbook delta = 1 case:
        c* = 1 - alpha*beta
        l* = alpha / (b*(1 - (1-alpha)*beta) + alpha)
    Returns None when delta is far from 1 (no simple closed form).
    """
    if abs(delta - 1.0) > 1e-6:
        return None
    c_star = 1.0 - alpha * beta
    l_star = alpha / (b * (1.0 - (1.0 - alpha) * beta) + alpha)
    c_hat = float(np.mean(stationary_slice(rec["c_fracs"], burn_frac)))
    l_hat = float(np.mean(stationary_slice(rec["ls"], burn_frac)))
    return {
        "c_star": c_star,
        "l_star": l_star,
        "c_hat": c_hat,
        "l_hat": l_hat,
        "c_abs_err": abs(c_hat - c_star),
        "l_abs_err": abs(l_hat - l_star),
    }


# ── Krusell-Smith forecasting quality ─────────────────────────────────────────

def ks_forecast_rule(rec, burn_frac: float = 0.5):
    """Fit the aggregate law of motion K_{t+1} = a_s + b_s * K_t per agg state.

    Fit on the stationary slice only: the transient from k_init spans a wide,
    nearly deterministic K range that a line fits trivially, inflating R^2.
    Returns (rules, r2_overall) where rules maps state in {0,1} -> (a, b).
    """
    K = stationary_slice(np.asarray(rec["K"]), burn_frac).flatten()
    agg = stationary_slice(np.asarray(rec["agg_state"]), burn_frac).flatten()
    Kt, Kt1, st = K[:-1], K[1:], agg[:-1]
    rules = {}
    for s in (0, 1):
        m = st == s
        if m.sum() >= 2 and np.std(Kt[m]) > 1e-9:
            b, a = np.polyfit(Kt[m], Kt1[m], 1)
        else:  # degenerate: fall back to identity
            a, b = 0.0, 1.0
        rules[int(s)] = (float(a), float(b))
    # Overall R^2 on the pooled one-step prediction.
    pred = np.array([rules[int(s)][0] + rules[int(s)][1] * k for k, s in zip(Kt, st)])
    ss_res = np.sum((Kt1 - pred) ** 2)
    ss_tot = np.sum((Kt1 - Kt1.mean()) ** 2) + 1e-12
    return rules, float(1.0 - ss_res / ss_tot)


def den_haan_stat(rec, burn_frac: float = 0.5) -> Dict[str, float]:
    """Den Haan statistic: max/mean dynamic-forecast error (%) of the agg rule.

    The forecasting rule is iterated forward on its own predictions (feeding the
    actual aggregate-state sequence) and compared to the realized capital path,
    both starting from the first stationary (post burn-in) period.
    """
    rules, r2 = ks_forecast_rule(rec, burn_frac)
    K = stationary_slice(np.asarray(rec["K"]), burn_frac).flatten()
    agg = stationary_slice(np.asarray(rec["agg_state"]), burn_frac).flatten()
    if K.size < 3:
        return {"ks_lom_r2": r2, "den_haan_max_pct": float("nan"),
                "den_haan_mean_pct": float("nan")}
    K_hat = np.empty_like(K)
    K_hat[0] = K[0]
    for t in range(K.size - 1):
        a, b = rules[int(agg[t])]
        K_hat[t + 1] = a + b * K_hat[t]
    err = 100.0 * np.abs(K_hat - K) / np.maximum(np.abs(K), 1e-12)
    return {
        "ks_lom_r2": r2,
        "den_haan_max_pct": float(np.max(err)),
        "den_haan_mean_pct": float(np.mean(err)),
    }


# ── Aggregator ────────────────────────────────────────────────────────────────

def economic_report(rec, env_cfg, burn_frac: float = 0.5) -> Dict[str, object]:
    """Run the economic probes appropriate to the env kind."""
    out: Dict[str, object] = {}
    out.update({"euler": euler_errors(rec, env_cfg.beta, burn_frac)})
    out.update({"resource": resource_residual(rec, env_cfg.delta, burn_frac)})
    out.update({"steady_state": steady_state(rec, burn_frac)})

    if env_cfg.kind == "rbc":
        cmp = analytical_rbc_compare(
            rec, env_cfg.alpha, env_cfg.beta, env_cfg.b, env_cfg.delta, burn_frac
        )
        if cmp is not None:
            out["analytical_rbc"] = cmp
    if env_cfg.kind == "ks":
        out["ks_forecast"] = den_haan_stat(rec, burn_frac)
    return out
