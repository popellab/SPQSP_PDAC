"""Unit tests for scripts.clinical_endpoints.

Each endpoint function is exercised on hand-crafted synthetic trajectories
for which the expected output is analytically derivable. The death /
progression / response thresholds are varied where it matters so the tests
exercise the decision boundaries and not just defaults.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from clinical_endpoints import (  # noqa: E402
    CR_DIAMETER_CM_DEFAULT,
    N_C_LETHAL_DEFAULT,
    PROGRESSION_INCREASE_DEFAULT,
    dfs_at_landmark,
    os_at_landmark,
    recist_category_at_day,
    time_to_progression,
    tumor_diameter_at_day,
    tumor_volume_doubling_time,
)


def _cells_for_diameter_cm(d_cm: float, cells_per_cm3: float = 1.0e9) -> float:
    """Return cancer-cell count whose sphere-volume-equivalent diameter is d."""
    v_cm3 = (4.0 / 3.0) * math.pi * (d_cm / 2.0) ** 3
    return v_cm3 * cells_per_cm3


def _trajectory(
    time_days: np.ndarray,
    c1_cells: np.ndarray | None = None,
    v_t_uL: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a (time, species_dict) pair of raw-float arrays in canonical units.

    Time is in days, V_T.C1 in cell, V_T in milliliter (canonical) — the
    helper accepts ``v_t_uL`` for caller convenience and converts to mL.
    At least one of ``c1_cells`` or ``v_t_uL`` must be provided.
    """
    species: dict[str, np.ndarray] = {}
    if c1_cells is not None:
        species["V_T.C1"] = np.asarray(c1_cells, dtype=float)
    if v_t_uL is not None:
        species["V_T"] = np.asarray(v_t_uL, dtype=float) * 1e-3  # µL → mL
    assert species, "must supply at least one of c1_cells / v_t_uL"
    return np.asarray(time_days, dtype=float), species


# ---------------------------------------------------------------------------
# OS at landmark
# ---------------------------------------------------------------------------


class TestOSAtLandmark:
    def test_alive_when_burden_below_threshold(self):
        t = np.array([0.0, 180.0, 365.0])
        c1 = np.array([1.0e8, 5.0e8, 1.0e9])  # far below 1e11
        time, sp = _trajectory(t, c1_cells=c1)
        result = os_at_landmark(time, sp, landmark_days=365.0)
        assert np.all(result == 1.0)

    def test_dead_when_burden_exceeds_threshold(self):
        t = np.array([0.0, 180.0, 365.0])
        c1 = np.array([1.0e9, 5.0e10, 5.0e11])
        time, sp = _trajectory(t, c1_cells=c1)
        result = os_at_landmark(time, sp, landmark_days=365.0)
        assert np.all(result == 0.0)

    def test_alive_at_earlier_landmark_dead_later(self):
        """Crossing at day 200: OS=1 at day 180, OS=0 at day 365."""
        t = np.array([0.0, 180.0, 200.0, 365.0])
        c1 = np.array([1.0e9, 5.0e10, N_C_LETHAL_DEFAULT * 1.1, 5.0e11])
        time, sp = _trajectory(t, c1_cells=c1)
        early = os_at_landmark(time, sp, landmark_days=180.0)
        late = os_at_landmark(time, sp, landmark_days=365.0)
        assert np.all(early == 1.0)
        assert np.all(late == 0.0)

    def test_custom_lethal_threshold(self):
        t = np.array([0.0, 180.0])
        c1 = np.array([1.0e9, 5.0e10])
        time, sp = _trajectory(t, c1_cells=c1)
        # Default: alive. Lower threshold → dead.
        alive = os_at_landmark(time, sp, landmark_days=180.0)
        dead = os_at_landmark(
            time, sp, landmark_days=180.0, n_c_lethal=1.0e10
        )
        assert np.all(alive == 1.0)
        assert np.all(dead == 0.0)

    def test_nan_when_landmark_beyond_sim(self):
        t = np.array([0.0, 180.0])
        c1 = np.array([1.0e9, 2.0e9])
        time, sp = _trajectory(t, c1_cells=c1)
        result = os_at_landmark(time, sp, landmark_days=365.0)
        assert np.all(np.isnan(result))


# ---------------------------------------------------------------------------
# DFS at landmark
# ---------------------------------------------------------------------------


class TestDFSAtLandmark:
    def test_disease_free_when_no_growth_from_nadir(self):
        t = np.linspace(0.0, 365.0, 10)
        # Flat diameter ≈ 2 cm throughout
        c1 = np.full_like(t, _cells_for_diameter_cm(2.0))
        v_t = np.full_like(t, (4.0 / 3.0) * math.pi * (2.0 / 2.0) ** 3 * 1e3)
        time, sp = _trajectory(t, c1_cells=c1, v_t_uL=v_t)
        result = dfs_at_landmark(time, sp, landmark_days=365.0)
        assert np.all(result == 1.0)

    def test_progressed_when_diameter_grows_above_threshold(self):
        t = np.array([0.0, 100.0, 200.0, 365.0])
        # Nadir at day 100 (1.0 cm), grow to 2.0 cm (100% increase) by 365 —
        # well above 20% + 0.5cm buffer.
        diameters = np.array([1.2, 1.0, 1.5, 2.0])
        v_t_uL = (4.0 / 3.0) * math.pi * (diameters / 2.0) ** 3 * 1e3
        c1 = v_t_uL * 1e3 * 1e9 / 1e3  # same cell count if density 1e9/cm^3
        time, sp = _trajectory(t, c1_cells=c1, v_t_uL=v_t_uL)
        result = dfs_at_landmark(time, sp, landmark_days=365.0)
        assert np.all(result == 0.0)

    def test_marginal_growth_below_threshold_still_dfs(self):
        t = np.array([0.0, 100.0, 200.0, 365.0])
        # Nadir 1.0, grow to 1.1 cm — 10% growth, below 20% threshold.
        diameters = np.array([1.2, 1.0, 1.05, 1.1])
        v_t_uL = (4.0 / 3.0) * math.pi * (diameters / 2.0) ** 3 * 1e3
        c1 = np.full_like(t, _cells_for_diameter_cm(1.1))
        time, sp = _trajectory(t, c1_cells=c1, v_t_uL=v_t_uL)
        result = dfs_at_landmark(time, sp, landmark_days=365.0)
        assert np.all(result == 1.0)

    def test_dead_patient_is_not_dfs(self):
        t = np.array([0.0, 100.0, 365.0])
        c1 = np.array([1.0e9, N_C_LETHAL_DEFAULT * 2, N_C_LETHAL_DEFAULT * 3])
        v_t = np.array([1.0, 10.0, 100.0]) * 1e3  # growing large volumes
        time, sp = _trajectory(t, c1_cells=c1, v_t_uL=v_t)
        result = dfs_at_landmark(time, sp, landmark_days=365.0)
        assert np.all(result == 0.0)


# ---------------------------------------------------------------------------
# Tumor diameter at landmark
# ---------------------------------------------------------------------------


class TestTumorDiameterAtDay:
    def test_exact_landmark_returns_interpolated_value(self):
        t = np.array([0.0, 180.0, 365.0])
        # 1 cm sphere: V = 4/3·π·(0.5)³ ≈ 0.5236 cm³ = 0.5236e3 µL
        v_t = np.array([0.5236e3, 4.188e3, 33.51e3])  # 1, 2, 4 cm
        c1 = np.zeros_like(v_t)
        time, sp = _trajectory(t, c1_cells=c1, v_t_uL=v_t)
        result = tumor_diameter_at_day(time, sp, landmark_days=180.0)
        assert result[0] == pytest.approx(2.0, rel=1e-2)

    def test_linear_interpolation_between_timepoints(self):
        t = np.array([0.0, 100.0, 200.0])
        v_t = np.array([0.5236e3, 4.188e3, 14.137e3])  # 1, 2, 3 cm
        time, sp = _trajectory(t, c1_cells=np.zeros_like(v_t), v_t_uL=v_t)
        # Diameter is not linear in V, but we interpolate diameter linearly
        # over time: so at t=150, expected = (2 + 3)/2 = 2.5 cm.
        result = tumor_diameter_at_day(
            time, sp, landmark_days=150.0
        )
        assert result[0] == pytest.approx(2.5, rel=1e-2)

    def test_falls_back_to_c1_when_v_t_absent(self):
        t = np.array([0.0, 100.0])
        c1 = np.array([_cells_for_diameter_cm(1.0), _cells_for_diameter_cm(2.0)])
        time, sp = _trajectory(t, c1_cells=c1)
        result = tumor_diameter_at_day(time, sp, landmark_days=100.0)
        assert result[0] == pytest.approx(2.0, rel=1e-2)

    def test_nan_when_dead_before_landmark(self):
        """Crossing n_c_lethal before the landmark censors the diameter to NaN
        — the simulator keeps integrating past 'death', and the uncensored
        values show up as anatomically impossible 50+ cm tails in the CI95."""
        t = np.array([0.0, 200.0, 365.0, 730.0])
        # C1 crosses 1e11 at t=365; patient is "dead" by day 730.
        c1 = np.array([1e9, 1e10, 1e11, 5e12])
        v_t_uL = np.array([0.5236e3, 4.188e3, 33.51e3, 1e6])  # 1, 2, 4, ~120 cm
        time, sp = _trajectory(t, c1_cells=c1, v_t_uL=v_t_uL)
        result = tumor_diameter_at_day(
            time, sp, landmark_days=730.0, n_c_lethal=1e11
        )
        assert np.isnan(result[0])

    def test_returns_value_when_alive_at_landmark(self):
        """Dead-before-landmark censoring does NOT fire when the patient
        hasn't crossed n_c_lethal by the landmark day."""
        t = np.array([0.0, 100.0, 200.0])
        c1 = np.array([1e9, 1e10, 5e10])  # stays below 1e11
        v_t_uL = np.array([0.5236e3, 4.188e3, 14.137e3])  # 1, 2, 3 cm
        time, sp = _trajectory(t, c1_cells=c1, v_t_uL=v_t_uL)
        result = tumor_diameter_at_day(
            time, sp, landmark_days=200.0, n_c_lethal=1e11
        )
        assert result[0] == pytest.approx(3.0, rel=1e-2)


# ---------------------------------------------------------------------------
# RECIST
# ---------------------------------------------------------------------------


class TestRECIST:
    def _build(self, diameters_cm):
        t = np.linspace(0.0, 90.0, len(diameters_cm))
        v_t = (4.0 / 3.0) * math.pi * (np.asarray(diameters_cm) / 2.0) ** 3 * 1e3
        c1 = np.zeros_like(t)
        return _trajectory(t, c1_cells=c1, v_t_uL=v_t)

    def test_pr_fires_on_30_percent_decrease(self):
        # Baseline 2 cm → day-90 1.3 cm (35 % decrease).
        time, sp = self._build([2.0, 1.8, 1.5, 1.3])
        result = recist_category_at_day(time, sp, landmark_days=90.0)
        assert result["PR"][0] == 1.0
        assert result["CR"][0] == 0.0
        assert result["PD"][0] == 0.0
        assert result["SD"][0] == 0.0

    def test_sd_on_marginal_decrease(self):
        # Baseline 2 cm → day-90 1.8 cm (10 % decrease; < 30 %).
        time, sp = self._build([2.0, 1.95, 1.85, 1.8])
        result = recist_category_at_day(time, sp, landmark_days=90.0)
        assert result["SD"][0] == 1.0
        assert sum(result[c][0] for c in ("CR", "PR", "PD")) == 0.0

    def test_pd_on_20_percent_rise_plus_5mm(self):
        # Nadir 1.0 cm (day 30), landmark 1.55 cm → 55 % above nadir,
        # absolute 0.55 cm > 0.5 cm buffer → PD.
        time, sp = self._build([1.5, 1.0, 1.2, 1.55])
        result = recist_category_at_day(time, sp, landmark_days=90.0)
        assert result["PD"][0] == 1.0

    def test_cr_when_diameter_below_01cm(self):
        time, sp = self._build([2.0, 0.5, 0.1, 0.05])
        result = recist_category_at_day(time, sp, landmark_days=90.0)
        assert result["CR"][0] == 1.0


# ---------------------------------------------------------------------------
# Time to progression
# ---------------------------------------------------------------------------


class TestTTP:
    def test_censored_when_no_progression(self):
        t = np.linspace(0.0, 365.0, 10)
        diameters = np.full_like(t, 2.0)
        v_t = (4.0 / 3.0) * math.pi * (diameters / 2.0) ** 3 * 1e3
        time, sp = _trajectory(t, c1_cells=np.zeros_like(t), v_t_uL=v_t)
        result = time_to_progression(time, sp)
        assert result[0] == pytest.approx(365.0, rel=1e-3)

    def test_first_crossing_detected(self):
        # Nadir = 1.0 (t=100). PD requires d ≥ 1.2*nadir AND d-nadir ≥ 0.5;
        # both fire when d reaches 1.5 (at t=200).
        t = np.array([0.0, 100.0, 200.0, 300.0, 365.0])
        diameters = np.array([1.2, 1.0, 1.5, 1.9, 2.5])
        v_t = (4.0 / 3.0) * math.pi * (diameters / 2.0) ** 3 * 1e3
        time, sp = _trajectory(t, c1_cells=np.zeros_like(t), v_t_uL=v_t)
        result = time_to_progression(time, sp)
        assert result[0] == pytest.approx(200.0, abs=1e-3)

    def test_absolute_increase_gate_stops_small_growth(self):
        """Proportional rise alone does not trigger PD without 0.5cm absolute."""
        # 25% rise but absolute < 5mm: nadir=0.2, land=0.25. The 20% threshold
        # fires (0.25 ≥ 0.24) but absolute 0.05 < 0.5. Should NOT progress.
        t = np.array([0.0, 100.0, 200.0])
        diameters = np.array([0.3, 0.2, 0.25])
        v_t = (4.0 / 3.0) * math.pi * (diameters / 2.0) ** 3 * 1e3
        time, sp = _trajectory(t, c1_cells=np.zeros_like(t), v_t_uL=v_t)
        result = time_to_progression(time, sp)
        assert result[0] == pytest.approx(200.0, rel=1e-3)  # censored at end

    def test_custom_progression_threshold(self):
        t = np.array([0.0, 100.0, 200.0])
        diameters = np.array([1.0, 1.0, 1.05])
        v_t = (4.0 / 3.0) * math.pi * (diameters / 2.0) ** 3 * 1e3
        time, sp = _trajectory(t, c1_cells=np.zeros_like(t), v_t_uL=v_t)
        # Default: 20 % + 0.5 cm buffer → not progressed.
        default = time_to_progression(time, sp)[0]
        # Looser progression (2 %, no buffer) → progressed at day 200.
        looser = time_to_progression(
            time,
            sp,
            progression_increase=0.02,
            min_absolute_increase_cm=0.0,
        )[0]
        assert default == pytest.approx(200.0, rel=1e-3)  # censored at end
        assert 100.0 < looser <= 200.0


# ---------------------------------------------------------------------------
# Tumor volume doubling time (parity with existing YAML)
# ---------------------------------------------------------------------------


class TestVDT:
    def test_exponential_trajectory_returns_true_vdt(self):
        t = np.linspace(0.0, 100.0, 11)
        true_vdt = 50.0  # days
        c1 = 1.0e9 * np.exp(t * math.log(2.0) / true_vdt)
        time, sp = _trajectory(t, c1_cells=c1)
        result = tumor_volume_doubling_time(time, sp)
        assert result[0] == pytest.approx(true_vdt, rel=1e-6)

    def test_nan_on_shrinking_tumor(self):
        t = np.array([0.0, 100.0])
        c1 = np.array([1.0e9, 5.0e8])
        time, sp = _trajectory(t, c1_cells=c1)
        result = tumor_volume_doubling_time(time, sp)
        assert np.all(np.isnan(result))
