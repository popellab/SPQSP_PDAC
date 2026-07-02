"""Tests for workflows.plot_calibration_trajectories.

Verifies the figure structure (axes count, labels, legend entries,
saved file) using synthetic observable_df + duck-typed cal target
fixtures. No SBI run / qsp_sim dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # headless backend for CI

from workflows.plot_calibration_trajectories import plot_calibration_trajectories


@dataclass
class _Empirical:
    median: float
    ci95_lower: float
    ci95_upper: float
    units: str


@dataclass
class _Observable:
    units: str


@dataclass
class _Target:
    observable_id: str
    empirical_data: _Empirical
    observable: _Observable = field(default_factory=lambda: _Observable(units=""))


def _make_observable_df(observable_ids: List[str], n_sims: int = 50, n_t: int = 5,
                        seed: int = 0) -> pd.DataFrame:
    """Synthetic posterior-trajectory frame: bell-curve-ish convergence
    toward the observed median at t=0."""
    rng = np.random.default_rng(seed)
    rows = []
    for obs_id in observable_ids:
        # Synthetic target value at t=0 is 100; trajectories noisy at -50,
        # tighter at 0.
        t_axis = np.linspace(-50.0, 0.0, n_t)
        true_curve = 100.0 - 5.0 * (-t_axis)  # rises from -150 → 100
        for i in range(n_sims):
            noise = rng.normal(0, 5.0, n_t) * (1 + np.linspace(2, 0, n_t))
            vals = true_curve + noise
            for t, v in zip(t_axis, vals):
                rows.append(
                    {
                        "observable_id": obs_id,
                        "sample_index": i,
                        "t_to_diagnosis_days": float(t),
                        "value": float(v),
                    }
                )
    return pd.DataFrame(rows)


def _make_target(obs_id: str, median: float = 100.0) -> _Target:
    return _Target(
        observable_id=obs_id,
        empirical_data=_Empirical(
            median=median,
            ci95_lower=median * 0.7,
            ci95_upper=median * 1.4,
            units="cell/mm^2",
        ),
        observable=_Observable(units="cell/mm^2"),
    )


def test_returns_figure_with_one_axis_per_target() -> None:
    obs_df = _make_observable_df(["cd8_density", "treg_density"])
    targets = [_make_target("cd8_density"), _make_target("treg_density")]
    fig = plot_calibration_trajectories(obs_df, targets, n_cols=2)
    # 1 row × 2 cols; both panels populated, no trailing axis_off needed.
    visible = [ax for ax in fig.axes if ax.get_visible() and ax.has_data()]
    assert len(visible) == 2
    titles = sorted(ax.get_title().split("\n", 1)[0] for ax in visible)
    assert titles == ["cd8_density", "treg_density"]


def test_grid_layout_with_three_targets_two_cols() -> None:
    """3 targets at n_cols=2 → 2 rows × 2 cols, last panel hidden."""
    obs_ids = ["a", "b", "c"]
    obs_df = _make_observable_df(obs_ids)
    targets = [_make_target(o) for o in obs_ids]
    fig = plot_calibration_trajectories(obs_df, targets, n_cols=2)
    # 4 axes total; 3 visible-with-data, 1 axis_off.
    assert len(fig.axes) == 4
    populated = [ax for ax in fig.axes if ax.has_data()]
    assert len(populated) == 3


def test_observable_df_units_label_and_xlabel() -> None:
    obs_df = _make_observable_df(["cd8_density"])
    fig = plot_calibration_trajectories(obs_df, [_make_target("cd8_density")])
    ax = next(a for a in fig.axes if a.has_data())
    assert "time to diagnosis" in ax.get_xlabel().lower()
    assert ax.get_ylabel() == "cell/mm^2"


def test_observed_marker_drawn_at_t0() -> None:
    """The observed-CI errorbar lives at x=0 (diagnosis) on each panel."""
    obs_df = _make_observable_df(["cd8_density"])
    target = _make_target("cd8_density", median=42.0)
    fig = plot_calibration_trajectories(obs_df, [target])
    ax = next(a for a in fig.axes if a.has_data())
    # ErrorbarContainer's first child is the line collection of points.
    found_obs_at_zero = False
    for container in ax.containers:
        # Errorbar containers expose .lines — look for one with x=0 marker.
        if hasattr(container, "lines"):
            line, _, _ = container.lines if len(container.lines) == 3 else (container.lines[0], None, None)
            xs, ys = line.get_data()
            if len(xs) == 1 and xs[0] == 0.0 and ys[0] == 42.0:
                found_obs_at_zero = True
                break
    assert found_obs_at_zero


def test_log_y_sets_log_scale() -> None:
    obs_df = _make_observable_df(["cd8_density"])
    fig = plot_calibration_trajectories(
        obs_df, [_make_target("cd8_density")], log_y=True
    )
    ax = next(a for a in fig.axes if a.has_data())
    assert ax.get_yscale() == "log"


def test_missing_target_panel_blank_with_warning(capsys: pytest.CaptureFixture[str]) -> None:
    """A target without rows in observable_df gets a blank panel + printed warning."""
    obs_df = _make_observable_df(["cd8_density"])
    # Second target has no posterior data.
    targets = [_make_target("cd8_density"), _make_target("missing_obs")]
    fig = plot_calibration_trajectories(obs_df, targets)
    captured = capsys.readouterr()
    assert "missing_obs" in captured.out
    # First panel populated, second blank (axis_off).
    populated = [ax for ax in fig.axes if ax.has_data()]
    assert len(populated) == 1


def test_save_to_path(tmp_path: Path) -> None:
    obs_df = _make_observable_df(["cd8_density"])
    out = tmp_path / "subdir" / "trajectories.png"
    fig = plot_calibration_trajectories(
        obs_df, [_make_target("cd8_density")], output_path=out
    )
    assert out.exists()
    assert out.stat().st_size > 0


def test_empty_targets_raises() -> None:
    empty_df = pd.DataFrame(
        columns=["observable_id", "sample_index", "t_to_diagnosis_days", "value"]
    )
    with pytest.raises(ValueError, match="empty"):
        plot_calibration_trajectories(empty_df, [])


def test_missing_observable_df_columns_raises() -> None:
    bad = pd.DataFrame({"foo": [1.0]})
    with pytest.raises(ValueError, match="missing required columns"):
        plot_calibration_trajectories(bad, [_make_target("a")])


def test_unsupported_ci_method_raises() -> None:
    obs_df = _make_observable_df(["a"])
    with pytest.raises(ValueError, match="ci_method"):
        plot_calibration_trajectories(
            obs_df, [_make_target("a")], ci_method="bonferroni"
        )


def test_ci_level_out_of_range_raises() -> None:
    obs_df = _make_observable_df(["a"])
    with pytest.raises(ValueError, match="ci_level"):
        plot_calibration_trajectories(
            obs_df, [_make_target("a")], ci_level=1.5
        )


def test_observable_id_fallback_to_id() -> None:
    """Targets without observable_id but with an ``id`` attr still resolve."""

    @dataclass
    class _LegacyTarget:
        id: str
        empirical_data: _Empirical
        observable: _Observable = field(
            default_factory=lambda: _Observable(units="cell")
        )

    target = _LegacyTarget(
        id="legacy_obs",
        empirical_data=_Empirical(
            median=10.0, ci95_lower=5.0, ci95_upper=15.0, units="cell"
        ),
    )
    obs_df = _make_observable_df(["legacy_obs"])
    fig = plot_calibration_trajectories(obs_df, [target])
    ax = next(a for a in fig.axes if a.has_data())
    assert "legacy_obs" in ax.get_title()


def test_phase_post_scenario_observed_marker_uses_index_values() -> None:
    """``phase='post_scenario'`` with a target.empirical_data.index_values
    set places the observed marker at index_values[0], not at 0."""

    @dataclass
    class _PostEmpirical:
        median: float
        ci95_lower: float
        ci95_upper: float
        units: str
        index_values: list[float]

    @dataclass
    class _PostTarget:
        calibration_target_id: str
        empirical_data: _PostEmpirical
        observable: _Observable = field(
            default_factory=lambda: _Observable(units="day")
        )

    target = _PostTarget(
        calibration_target_id="day_28_endpoint",
        empirical_data=_PostEmpirical(
            median=42.0,
            ci95_lower=30.0,
            ci95_upper=60.0,
            units="day",
            index_values=[28.0],
        ),
    )
    rows = []
    for sample_idx in range(20):
        for t in (0.0, 14.0, 28.0, 42.0):
            rows.append({
                "observable_id": "day_28_endpoint",
                "sample_index": sample_idx,
                "t_to_diagnosis_days": t,
                # Vary across time so the panel doesn't trip the
                # scalar-detection path (which renders a histogram
                # instead of a trajectory band + errorbar).
                "value": 50.0 + 0.1 * sample_idx + 0.5 * t,
            })
    obs_df = pd.DataFrame(rows)
    fig = plot_calibration_trajectories(
        obs_df, [target], phase="post_scenario"
    )
    ax = next(a for a in fig.axes if a.has_data())
    # Errorbar lands at x=28 (the cal target's index_values[0]).
    found_at_28 = False
    for c in ax.containers:
        if hasattr(c, "lines") and c.lines:
            xs, ys = c.lines[0].get_data()
            if len(xs) == 1 and xs[0] == 28.0 and ys[0] == 42.0:
                found_at_28 = True
                break
    assert found_at_28, "observed marker should be at x=28 (index_values[0])"
    # x-axis label reflects phase.
    assert "since diagnosis" in ax.get_xlabel().lower()


def test_phase_post_scenario_falls_back_to_rightmost_data_point() -> None:
    """When index_values is None, post-scenario marker lands at the last
    data point (rightmost time in the sim band)."""

    @dataclass
    class _PostEmpirical:
        median: float
        ci95_lower: float
        ci95_upper: float
        units: str
        index_values: None

    @dataclass
    class _PostTarget:
        calibration_target_id: str
        empirical_data: _PostEmpirical
        observable: _Observable = field(
            default_factory=lambda: _Observable(units="dimensionless")
        )

    target = _PostTarget(
        calibration_target_id="end_of_scenario",
        empirical_data=_PostEmpirical(
            median=1.0,
            ci95_lower=0.5,
            ci95_upper=2.0,
            units="dimensionless",
            index_values=None,
        ),
    )
    rows = []
    for sample_idx in range(10):
        for t in (0.0, 50.0, 100.0):
            rows.append({
                "observable_id": "end_of_scenario",
                "sample_index": sample_idx,
                "t_to_diagnosis_days": t,
                "value": 1.0 + 0.01 * sample_idx + 0.005 * t,
            })
    obs_df = pd.DataFrame(rows)
    fig = plot_calibration_trajectories(
        obs_df, [target], phase="post_scenario"
    )
    ax = next(a for a in fig.axes if a.has_data())
    # Errorbar lands at the rightmost time = 100.0
    found_at_100 = False
    for c in ax.containers:
        if hasattr(c, "lines") and c.lines:
            xs, _ = c.lines[0].get_data()
            if len(xs) == 1 and xs[0] == 100.0:
                found_at_100 = True
                break
    assert found_at_100, "observed marker should fall back to rightmost time"


def test_phase_post_scenario_silently_disables_fraction() -> None:
    """``phase='post_scenario'`` + ``time_axis='fraction'`` falls back to
    absolute (post-scenario sims share a fixed time grid)."""
    target = _make_target("x", median=50.0)
    rows = [
        {"observable_id": "x", "sample_index": s, "t_to_diagnosis_days": t,
         "value": 50.0}
        for s in range(5) for t in (0.0, 30.0, 60.0)
    ]
    obs_df = pd.DataFrame(rows)
    fig = plot_calibration_trajectories(
        obs_df, [target], phase="post_scenario", time_axis="fraction"
    )
    ax = next(a for a in fig.axes if a.has_data())
    # xlabel should NOT be the fraction label.
    assert "fraction of evolve" not in ax.get_xlabel().lower()


def test_phase_invalid_raises() -> None:
    obs_df = _make_observable_df(["a"])
    with pytest.raises(ValueError, match="phase"):
        plot_calibration_trajectories(obs_df, [_make_target("a")], phase="weird")


def test_scalar_observable_renders_as_histogram() -> None:
    """Per-sim constant values (e.g., a derived doubling time) render
    as a histogram of per-sim values + observed CI95 vertical band,
    not as a trajectory band."""
    rows = []
    for sample_idx in range(50):
        # Constant per sim, varied across sims — the scalar signature.
        sim_value = 60.0 + 2.0 * sample_idx
        for t in (0.0, 30.0, 60.0, 90.0):
            rows.append({
                "observable_id": "doubling_time",
                "sample_index": sample_idx,
                "t_to_diagnosis_days": t,
                "value": sim_value,
            })
    obs_df = pd.DataFrame(rows)
    target = _make_target("doubling_time", median=120.0)
    fig = plot_calibration_trajectories(obs_df, [target])
    ax = next(a for a in fig.axes if a.has_data())
    # Histogram path uses ylabel "posterior sim count" instead of units.
    assert "sim count" in ax.get_ylabel().lower()
    # No errorbar (no observed-CI containers); axvspan is the observed CI band.
    has_axvspan = any(p.get_label() == "observed 95% CI" for p in ax.patches)
    assert has_axvspan, "expected observed 95% CI axvspan on scalar panel"


def test_scenario_per_target_appended_to_panel_title() -> None:
    """``scenario_per_target={obs_id: scen_name}`` shows up in panel title."""
    obs_df = _make_observable_df(["cd8_density"])
    fig = plot_calibration_trajectories(
        obs_df,
        [_make_target("cd8_density")],
        scenario_per_target={"cd8_density": "gvax_neoadjuvant_zheng2022"},
    )
    ax = next(a for a in fig.axes if a.has_data())
    assert "gvax_neoadjuvant_zheng2022" in ax.get_title()


def test_calibration_target_id_used_when_present() -> None:
    """maple's CalibrationTarget exposes ``calibration_target_id`` —
    the resolver should pick it up first."""

    @dataclass
    class _MapleTarget:
        calibration_target_id: str
        empirical_data: _Empirical
        observable: _Observable = field(
            default_factory=lambda: _Observable(units="cell/mm^2")
        )

    target = _MapleTarget(
        calibration_target_id="cd8_density",
        empirical_data=_Empirical(
            median=63.0, ci95_lower=40.0, ci95_upper=100.0, units="cell/mm^2"
        ),
    )
    obs_df = _make_observable_df(["cd8_density"])
    fig = plot_calibration_trajectories(obs_df, [target])
    ax = next(a for a in fig.axes if a.has_data())
    assert "cd8_density" in ax.get_title()


def test_time_axis_fraction_normalizes_to_unit_interval() -> None:
    """``time_axis='fraction'`` rescales each sim's burn-in to [-1, 0]."""
    # Hand-craft a frame with two sims that have different t_diag.
    rows = []
    # sim 0: t_diag = 100 → t_to_diag in {-100, -50, 0}
    for t in (-100.0, -50.0, 0.0):
        rows.append({"observable_id": "x", "sample_index": 0,
                     "t_to_diagnosis_days": t, "value": 50.0 - t * 0.5})
    # sim 1: t_diag = 500 → t_to_diag in {-500, -250, 0}
    for t in (-500.0, -250.0, 0.0):
        rows.append({"observable_id": "x", "sample_index": 1,
                     "t_to_diagnosis_days": t, "value": 50.0 - t * 0.1})
    obs_df = pd.DataFrame(rows)
    target = _make_target("x", median=50.0)

    fig = plot_calibration_trajectories(obs_df, [target], time_axis="fraction")
    ax = next(a for a in fig.axes if a.has_data())
    assert "fraction of evolve duration" in ax.get_xlabel().lower()
    # x-axis should span [-1, 0] (with a little headroom from set_xlim)
    xmin, xmax = ax.get_xlim()
    assert xmin <= -1.0 <= xmin + 0.1
    assert xmax >= 0.0


def test_time_axis_invalid_raises() -> None:
    obs_df = _make_observable_df(["a"])
    with pytest.raises(ValueError, match="time_axis"):
        plot_calibration_trajectories(
            obs_df, [_make_target("a")], time_axis="weird"
        )


def test_log_y_auto_engages_when_ci_spans_orders_of_magnitude() -> None:
    """``log_y='auto'`` uses log scale when ci95_upper / ci95_lower >= 10."""
    obs_df = _make_observable_df(["lognormal_target"])
    target = _Target(
        observable_id="lognormal_target",
        empirical_data=_Empirical(
            median=100.0, ci95_lower=10.0, ci95_upper=1000.0, units="cell/mm^2"
        ),
    )
    fig = plot_calibration_trajectories(obs_df, [target], log_y="auto")
    ax = next(a for a in fig.axes if a.has_data())
    assert ax.get_yscale() == "log"


def test_log_y_auto_skips_when_ci_is_narrow() -> None:
    """Narrow symmetric CI leaves linear y."""
    obs_df = _make_observable_df(["narrow_target"])
    target = _Target(
        observable_id="narrow_target",
        empirical_data=_Empirical(
            median=100.0, ci95_lower=80.0, ci95_upper=120.0, units="cell/mm^2"
        ),
    )
    fig = plot_calibration_trajectories(obs_df, [target], log_y="auto")
    ax = next(a for a in fig.axes if a.has_data())
    assert ax.get_yscale() == "linear"


def test_log_y_auto_engages_for_tight_log_symmetric_ci() -> None:
    """Tight CI that is symmetric in log space (hallmark of lognormal)
    triggers log even when ratio < 10×. Hits Liu et al. CD8 density
    (median 63, CI [30, 132], ratio 4.4× — log-symmetric)."""
    obs_df = _make_observable_df(["liu_like"])
    target = _Target(
        observable_id="liu_like",
        empirical_data=_Empirical(
            median=63.0, ci95_lower=30.0, ci95_upper=132.0, units="cell/mm^2"
        ),
    )
    fig = plot_calibration_trajectories(obs_df, [target], log_y="auto")
    ax = next(a for a in fig.axes if a.has_data())
    assert ax.get_yscale() == "log"


def test_log_y_auto_skips_log_asymmetric_fraction() -> None:
    """Stromal-fraction-like CI (asymmetric in log space) stays linear
    even though it's strictly positive."""
    obs_df = _make_observable_df(["stromal_like"])
    target = _Target(
        observable_id="stromal_like",
        empirical_data=_Empirical(
            median=0.668, ci95_lower=0.424, ci95_upper=0.863, units="dimensionless"
        ),
    )
    fig = plot_calibration_trajectories(obs_df, [target], log_y="auto")
    ax = next(a for a in fig.axes if a.has_data())
    assert ax.get_yscale() == "linear"


def test_log_y_auto_handles_missing_ci() -> None:
    """When CI bounds are absent, auto falls back to linear (no crash)."""

    @dataclass
    class _NoCI:
        median: float
        units: str

    @dataclass
    class _BadTarget:
        observable_id: str
        empirical_data: _NoCI
        observable: _Observable = field(
            default_factory=lambda: _Observable(units="cell")
        )

    obs_df = _make_observable_df(["x"])
    target = _BadTarget(
        observable_id="x",
        empirical_data=_NoCI(median=10.0, units="cell"),
    )
    fig = plot_calibration_trajectories(obs_df, [target], log_y="auto")
    ax = next(a for a in fig.axes if a.has_data())
    assert ax.get_yscale() == "linear"


def test_log_y_invalid_string_raises() -> None:
    obs_df = _make_observable_df(["a"])
    with pytest.raises(ValueError, match="log_y"):
        plot_calibration_trajectories(
            obs_df, [_make_target("a")], log_y="lol"
        )


def test_maple_list_empirical_schema() -> None:
    """maple stores median as ``[m]`` and CI95 as ``[[lo, hi]]`` — the
    extractor should unwrap both correctly."""

    @dataclass
    class _MapleEmpirical:
        median: list[float]
        ci95: list[list[float]]
        units: str

    @dataclass
    class _MapleTarget:
        calibration_target_id: str
        empirical_data: _MapleEmpirical
        observable: _Observable = field(
            default_factory=lambda: _Observable(units="dimensionless")
        )

    target = _MapleTarget(
        calibration_target_id="m1_m2_ratio",
        empirical_data=_MapleEmpirical(
            median=[0.518], ci95=[[0.30, 0.85]], units="dimensionless"
        ),
    )
    obs_df = _make_observable_df(["m1_m2_ratio"])
    fig = plot_calibration_trajectories(obs_df, [target])
    ax = next(a for a in fig.axes if a.has_data())
    # Errorbar lands at obs_median=0.518 — find the marker x=0 with y=0.518.
    found = False
    for c in ax.containers:
        if hasattr(c, "lines") and c.lines:
            line = c.lines[0]
            xs, ys = line.get_data()
            if len(xs) == 1 and xs[0] == 0.0 and abs(ys[0] - 0.518) < 1e-9:
                found = True
                break
    assert found, "observed marker at t=0 with median=0.518 not found"
