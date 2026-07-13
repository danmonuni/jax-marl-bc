"""Bespoke per-experiment figures (RBC policy, KS fig4, general fig5).

Ported from the original ``exps/`` scripts and restyled. Each function takes
already-simulated data plus a save path so the driver controls IO.
"""
from __future__ import annotations

from typing import List

import numpy as np

from .style import apply_style, COLORS
from ..diagnostics.distributional import gini, lorenz, stationary_slice


def plot_rbc_policy(metrics, steps_per_update, c_target, l_target, path, title=""):
    """Consumption/labour policy convergence vs training steps with std-bands."""
    import matplotlib.pyplot as plt
    apply_style()

    c_env = np.asarray(metrics["c_frac_env"])
    l_env = np.asarray(metrics["l_env"])
    env_steps = np.arange(1, len(c_env) + 1) * steps_per_update
    c_mu, c_sd = c_env.mean(-1), c_env.std(-1)
    l_mu, l_sd = l_env.mean(-1), l_env.std(-1)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(env_steps, c_mu, color="tab:blue", label=r"$\hat{c}$")
    ax.fill_between(env_steps, c_mu - c_sd, c_mu + c_sd, color="tab:blue", alpha=0.25)
    ax.axhline(c_target, color="tab:blue", linestyle="--", label=r"$c^*$")
    ax.plot(env_steps, l_mu, color="tab:orange", label=r"$\hat{\ell}$")
    ax.fill_between(env_steps, l_mu - l_sd, l_mu + l_sd, color="tab:orange", alpha=0.25)
    ax.axhline(l_target, color="tab:orange", linestyle="--", label=r"$\ell^*$")
    ax.set_xscale("log")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Training steps")
    ax.set_ylabel(r"Labor supply $\hat{\ell}$, Cons. frac. $\hat{c}$")
    ax.legend()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_ks_fig4(recs, snap_idxs, snap_steps, path, burn_frac=0.5):
    """Law-of-motion scatters, wealth histograms, consumption-policy scatter
    (before/after).

    Every panel uses only the stationary (post burn-in) slice of the eval
    rollouts, so the transient from k_init does not pollute the LoM fits,
    histograms, or policy scatters.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import scipy.stats as sp_stats
    apply_style()

    rec_before, rec_after = recs[0], recs[-1]
    fig = plt.figure(figsize=(18, 7))
    gs_outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38)

    gs_left = gs_outer[0].subgridspec(2, 2, hspace=0.55, wspace=0.45)
    for panel_i, (idx, step, rec) in enumerate(zip(snap_idxs, snap_steps, recs)):
        ax = fig.add_subplot(gs_left[panel_i // 2, panel_i % 2])
        K = stationary_slice(np.asarray(rec["K"]), burn_frac)
        agg = stationary_slice(np.asarray(rec["agg_state"]), burn_frac)
        Kt, Kt1, agt = K[:-1], K[1:], agg[:-1]
        slope, intercept, r_value, *_ = sp_stats.linregress(Kt, Kt1)
        ax.scatter(Kt[agt == 1], Kt1[agt == 1], s=2, alpha=0.3, color="tab:orange", label="good")
        ax.scatter(Kt[agt == 0], Kt1[agt == 0], s=2, alpha=0.3, color="tab:blue", label="bad")
        x_r = np.linspace(Kt.min(), Kt.max(), 100)
        ax.plot(x_r, slope * x_r + intercept, color="red", lw=1.2, label=f"$R^2={r_value**2:.3f}$")
        ax.set_xlabel("$K_t$", fontsize=8); ax.set_ylabel("$K_{t+1}$", fontsize=8)
        ax.set_title(f"Step $\\approx {step:.0e}$", fontsize=8)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)

    gs_mid = gs_outer[1].subgridspec(2, 1, hspace=0.55)
    for ax, rec, label in zip(
        [fig.add_subplot(gs_mid[0]), fig.add_subplot(gs_mid[1])],
        [rec_before, rec_after], ["Untrained", "Trained"],
    ):
        k_vals = stationary_slice(np.asarray(rec["ks"]), burn_frac).flatten()
        g = gini(k_vals)
        ax.hist(k_vals, bins=40, density=True, color="steelblue", alpha=0.75)
        ax.set_title(f"{label}  (Gini={g:.3f})", fontsize=9)
        ax.set_xlabel("Capital $k^i$", fontsize=8); ax.set_ylabel("Density", fontsize=8)
        ax.tick_params(labelsize=7)

    gs_right = gs_outer[2].subgridspec(2, 1, hspace=0.55)
    for ax, rec, label in zip(
        [fig.add_subplot(gs_right[0]), fig.add_subplot(gs_right[1])],
        [rec_before, rec_after], ["Untrained", "Trained"],
    ):
        # emp_states[t] is recorded from the post-step state, i.e. the draw for
        # period t+1; the employment that entered the policy's observation for
        # (wealths[t], c_fracs[t]) is emp_states[t-1]. Colour by the lag.
        W = stationary_slice(np.asarray(rec["wealths"])[1:], burn_frac)
        CF = stationary_slice(np.asarray(rec["c_fracs"])[1:], burn_frac)
        emp = stationary_slice(np.asarray(rec["emp_states"])[:-1], burn_frac)
        if "done" in rec:  # drop auto-reset steps (none in reset-free evals)
            keep = ~stationary_slice(np.asarray(rec["done"])[1:], burn_frac).astype(bool)
            W, CF, emp = W[keep], CF[keep], emp[keep]
        W, CF, emp = W.flatten(), CF.flatten(), emp.flatten()
        ax.scatter(W[emp == 1], CF[emp == 1], s=1, alpha=0.3, color="tab:orange", label="employed")
        ax.scatter(W[emp == 0], CF[emp == 0], s=1, alpha=0.3, color="tab:blue", label="unemployed")
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Wealth $a^i$", fontsize=8)
        ax.set_ylabel("Cons. frac. $\\hat{c}^i$", fontsize=8)
        ax.legend(fontsize=7, markerscale=4); ax.tick_params(labelsize=7)

    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stationary_slice(arr, burn_frac=0.25):
    cut = int(arr.shape[0] * burn_frac)
    return arr[cut:]


def plot_general_fig5(results, spread_labels, n_grid, path):
    """Lorenz curves, policy scatter, per-agent capital distributions."""
    import matplotlib.pyplot as plt
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    ax = axes[0]
    for i, ((_, __, rec, ___), label) in enumerate(zip(results, spread_labels)):
        ks_stat = _stationary_slice(rec["ks"]).flatten()
        lx, ly = lorenz(ks_stat); g = gini(ks_stat)
        step = max(1, len(lx) // 30)
        ax.plot(lx[::step], ly[::step], color=COLORS[i], marker="s", markersize=4,
                lw=1.5, label=f"{label}  Gini={g:.2f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, lw=1)
    ax.set_xlabel("Cumulative share of population")
    ax.set_ylabel("Cumulative share of wealth")
    ax.legend(title="Cap. prod. ($\\kappa^i$)", fontsize=8, title_fontsize=9)

    ax = axes[1]
    kappasW, _, recW, __ = results[2]
    unique_kappas = sorted(set(kappasW.tolist()))
    pcmap = plt.get_cmap("tab10")
    cutW = int(recW["wealths"].shape[0] * 0.25)
    for j, kv in enumerate(unique_kappas):
        idx = np.where(np.isclose(kappasW, kv))[0]
        w_all = recW["wealths"][cutW:, idx].flatten()
        c_all = recW["c_fracs"][cutW:, idx].flatten()
        n_samp = min(600, len(w_all))
        ridx = np.random.default_rng(j).choice(len(w_all), n_samp, replace=False)
        ax.scatter(w_all[ridx], c_all[ridx], s=6, alpha=0.5, color=pcmap(j),
                   label=f"$\\kappa^i = {kv:.2f}$")
    ax.set_xlabel("Wealth ($a$)"); ax.set_ylabel("Cons. frac. $c$"); ax.legend(fontsize=8)

    ax = axes[2]
    kappasM, lambdasM, recM, _ = results[1]
    ks_stat = _stationary_slice(recM["ks"])
    n_agents = kappasM.shape[0]
    corners = {0, n_grid - 1, n_agents - n_grid, n_agents - 1}
    tab20 = plt.get_cmap("tab20")
    legend_handles = []
    for i in range(n_agents):
        col = tab20(i / n_agents)
        ax.hist(ks_stat[:, i].flatten(), bins=50, density=True, alpha=0.55, color=col)
        if i in corners:
            legend_handles.append(plt.Line2D(
                [0], [0], color=col, linewidth=3,
                label=f"$\\kappa^{{{i+1}}}$={kappasM[i]:.2f}, $\\lambda^{{{i+1}}}$={lambdasM[i]:.2f}"))
    ax.legend(handles=legend_handles, fontsize=7)
    ax.set_xlabel("Capital ($k$)"); ax.set_ylabel("Density")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
