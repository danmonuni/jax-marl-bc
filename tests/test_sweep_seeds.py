"""Multi-seed sweeps: seed resolution and the per-cell mean/std collapse.

No JAX, no training — these exercise the pure bookkeeping around a sweep so a
9-hour scan cannot come back with a mis-shaped statistics table.
"""
import math

import pandas as pd
import pytest
from omegaconf import OmegaConf

from jmbc.config.schema import SweepConfig
from jmbc.plots import summarize_repeats
from jmbc.sweep import _cell_key, load_done, resolve_seeds


def _scfg(**kw):
    return OmegaConf.merge(OmegaConf.structured(SweepConfig), kw)


# ── seed resolution ───────────────────────────────────────────────────────

def test_explicit_seeds_win_over_repeats():
    assert resolve_seeds(_scfg(seeds=[3, 7, 11], repeats=1), base_seed=0) == [3, 7, 11]


def test_repeats_fall_back_to_base_seed_offsets():
    """The legacy path: seed = run.seed + rep, exactly what the runner did."""
    assert resolve_seeds(_scfg(repeats=3), base_seed=5) == [5, 6, 7]


def test_default_is_a_single_seed():
    assert resolve_seeds(_scfg(), base_seed=0) == [0]


# ── per-cell statistics ───────────────────────────────────────────────────

def _table(times_by_n, method="jaxmarl-bc"):
    rows = []
    for n, times in times_by_n.items():
        for seed, t in enumerate(times):
            rows.append({"method": method, "base_exp": "ks", "seed": seed,
                         "n_agents": n, "num_envs": 1, "device": "gpu",
                         "time_s": t, "run_time_s": t,
                         "throughput_steps_per_s": 100000.0 / t})
    return pd.DataFrame(rows)


def test_one_row_per_cell_with_mean_std_and_seed_list():
    df = _table({10: [10.0, 12.0, 14.0], 20: [20.0, 22.0, 24.0]})
    s = summarize_repeats(df, ["n_agents"]).set_index("n_agents")

    assert len(s) == 2
    assert list(s["n_seeds"]) == [3, 3]
    assert s.loc[10, "seeds"] == "0,1,2"
    assert s.loc[10, "time_s_mean"] == pytest.approx(12.0)
    assert s.loc[10, "time_s_std"] == pytest.approx(2.0)         # ddof=1
    assert s.loc[10, "time_s_sem"] == pytest.approx(2.0 / math.sqrt(3))
    assert s.loc[10, "time_s_min"] == pytest.approx(10.0)
    assert s.loc[10, "time_s_max"] == pytest.approx(14.0)


def test_single_seed_cell_reports_nan_std_not_zero():
    """n=1 has no dispersion estimate; a 0 would read as 'perfectly repeatable'."""
    s = summarize_repeats(_table({10: [10.0]}), ["n_agents"])
    assert int(s["n_seeds"].iloc[0]) == 1
    assert math.isnan(s["time_s_std"].iloc[0])


def test_methods_are_not_averaged_together():
    """A CPU series overlaid on a GPU one must stay two rows per cell."""
    df = pd.concat([_table({10: [10.0, 12.0]}),
                    _table({10: [100.0, 120.0]}, method="jaxmarl-bc-cpu")])
    s = summarize_repeats(df, ["n_agents"]).set_index("method")
    assert len(s) == 2
    assert s.loc["jaxmarl-bc", "time_s_mean"] == pytest.approx(11.0)
    assert s.loc["jaxmarl-bc-cpu", "time_s_mean"] == pytest.approx(110.0)


def test_identity_columns_survive_aggregation():
    s = summarize_repeats(_table({10: [10.0, 12.0]}), ["n_agents"])
    assert s["device"].iloc[0] == "gpu"
    assert int(s["num_envs"].iloc[0]) == 1


# ── resume ────────────────────────────────────────────────────────────────

def test_cell_key_matches_across_yaml_ints_and_csv_floats():
    """A cell written as 10 and read back from CSV as 10.0 is the same cell —
    otherwise every resume would silently re-run the whole sweep."""
    assert _cell_key(["n_agents"], [10], 0) == _cell_key(["n_agents"], [10.0], 0.0)
    assert _cell_key(["n_agents"], [10], 0) != _cell_key(["n_agents"], [10], 1)
    assert _cell_key(["n_agents"], [10], 0) != _cell_key(["n_agents"], [20], 0)


def test_load_done_reports_finished_runs(tmp_path):
    _table({10: [1.0, 2.0], 20: [3.0]}).to_csv(tmp_path / "results.csv", index=False)
    rows, done = load_done(tmp_path, ["n_agents"])
    assert len(rows) == 3
    assert _cell_key(["n_agents"], [10], 1) in done
    assert _cell_key(["n_agents"], [20], 1) not in done   # seed 1 never ran


def test_load_done_on_a_fresh_directory(tmp_path):
    assert load_done(tmp_path, ["n_agents"]) == ([], set())


def test_load_done_ignores_a_pre_multiseed_results_csv(tmp_path):
    """An old results.csv has no `seed` column; resuming off it would mismatch
    every key, so it is discarded outright rather than half-trusted."""
    _table({10: [1.0]}).drop(columns=["seed"]).to_csv(
        tmp_path / "results.csv", index=False)
    assert load_done(tmp_path, ["n_agents"]) == ([], set())


def test_time_s_is_derived_when_absent():
    """Aggregation runs on ensure_time_column output, so a table carrying only
    run_time_s (what an older results.csv has) still summarizes."""
    df = _table({10: [10.0, 12.0]}).drop(columns=["time_s"])
    s = summarize_repeats(df, ["n_agents"])
    assert s["time_s_mean"].iloc[0] == pytest.approx(11.0)
