"""Clinical-endpoint extractors for PDAC QSP simulations.

These functions take a simulation trajectory (raw-float time array + species
dict of raw-float arrays in canonical model_structure units) and return a
single scalar (broadcast over the time vector for interface consistency with
the calibration-target wrapper) representing a clinically meaningful endpoint.

All functions share the same signature:

    fn(time, species_dict, **kwargs) -> np.ndarray  # broadcast over time

``time`` arrives as a numpy array in days, ``species_dict`` maps species
qualified names (e.g. ``"V_T.C1"``) to numpy arrays on the same time axis in
each species' canonical unit (``cell`` for counts, ``milliliter`` for V_T,
``nanomolarity`` for cytokines, etc.). ``**kwargs`` covers per-endpoint knobs
(landmark day, thresholds, etc.).

Endpoints produced:

- :func:`os_at_landmark` — binary (alive=1) at a landmark day.
- :func:`dfs_at_landmark` — binary (disease-free survival) at a landmark day;
  patient must be alive AND not progressed from nadir.
- :func:`tumor_diameter_at_day` — diameter (cm) at a landmark day.
- :func:`recist_category_at_day` — one-hot encoding of {CR, PR, SD, PD} at a
  landmark day, from diameter change vs. pre-treatment baseline.
- :func:`time_to_progression` — days until first ≥20 % diameter rise from
  nadir; censored at stop_time when no progression observed.
- :func:`tumor_volume_doubling_time` — exponential-fit VDT across the run
  (legacy; mirrors the existing ``tumor_doubling_time_PDAC_deriv001`` YAML).

Death and progression defaults are QSP-IO convention:

- ``N_C_LETHAL_DEFAULT = 1e11`` cancer cells — equivalent to a ~10 cm-diameter
  sphere at ~1e9 cells/cm³. Above this the patient is considered dead.
- ``PROGRESSION_INCREASE_DEFAULT = 0.20`` — 20 % increase in longest tumor
  diameter from nadir (RECIST 1.1 PD criterion).
- ``RESPONSE_DECREASE_DEFAULT = 0.30`` — 30 % decrease in diameter triggers PR
  (RECIST 1.1 PR criterion).
- ``CR_DIAMETER_CM_DEFAULT = 0.1`` cm — below this the target lesion is
  considered to have disappeared (CR).

Override any of the defaults per call via keyword arguments so that the
calibration-target YAML can expose them as ``inputs`` for extraction provenance.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

N_C_LETHAL_DEFAULT: float = 1.0e11
"""Cancer-cell count above which the patient is considered dead.

Rationale: spherical tumor at ~1e9 cells/cm³ with 10 cm diameter has volume
(4/3)π(5)³ ≈ 524 cm³ → ~5e11 cells. The 1e11 threshold is conservative,
corresponding to ~6 cm diameter and matching the lethal-burden convention used
in the Jafarnejad / Popel QSP-IO model family.
"""

PROGRESSION_INCREASE_DEFAULT: float = 0.20
"""Minimum fractional diameter increase from nadir to declare RECIST PD."""

RESPONSE_DECREASE_DEFAULT: float = 0.30
"""Minimum fractional diameter decrease from baseline to declare RECIST PR."""

CR_DIAMETER_CM_DEFAULT: float = 0.1
"""Diameter below which the target lesion is considered absent (RECIST CR)."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_magnitude(qty, target_unit: str) -> np.ndarray:
    """Coerce ``qty`` (raw float / numpy array in canonical units) to the
    requested ``target_unit``.

    species_dict / time arrays now arrive as raw floats in their canonical
    model_structure units (no Pint), so this is a unit-aware scale rather
    than a Pint conversion. Canonical units are: ``cell`` for counts,
    ``day`` for time, ``milliliter`` for compartment volumes,
    ``nanomolarity`` for cytokines.
    """
    arr = np.asarray(qty, dtype=float)
    if target_unit == "microliter":
        # V_T canonical = milliliter; µL = mL × 1000
        return arr * 1000.0
    # cell, day are canonical → no scale needed
    return arr


def _tumor_diameter_cm(tumor_volume_uL: np.ndarray) -> np.ndarray:
    """Diameter (cm) of a sphere with volume ``tumor_volume_uL`` (microliters).

    1 µL = 1e-3 cm³, so diameter = 2·(3·V·1e-3 / 4π)^(1/3). Clamped at 0 for
    numerical safety on non-positive volumes.
    """
    v_cm3 = np.clip(tumor_volume_uL, 0.0, None) * 1e-3
    return 2.0 * np.cbrt(3.0 * v_cm3 / (4.0 * math.pi))


def _compartment_volume_uL(
    species_dict: Mapping[str, np.ndarray],
) -> np.ndarray | None:
    """Return V_T volume (µL) from species_dict if available, else ``None``.

    QSP-IO model exposes the tumor compartment volume as ``"V_T"``. When the
    compartment volume is not in the species dict (some scenarios expose only
    species concentrations), the caller must fall back to cell-count-based
    metrics.
    """
    vol = species_dict.get("V_T")
    if vol is None:
        return None
    return _get_magnitude(vol, "microliter")


def _diameter_from_species(
    species_dict: Mapping[str, np.ndarray],
    cells_per_cm3: float = 1.0e9,
) -> np.ndarray:
    """Derive tumor diameter (cm) from ``V_T`` or fall back to ``V_T.C1``.

    Prefer the true tumor-compartment volume when exposed. Fall back to a
    cell-count-based estimate assuming ``cells_per_cm3`` cell density — used
    only for scenarios that don't include V_T as a species.
    """
    v_uL = _compartment_volume_uL(species_dict)
    if v_uL is not None:
        return _tumor_diameter_cm(v_uL)
    c1 = _get_magnitude(species_dict["V_T.C1"], "cell")
    v_cm3 = np.clip(c1, 0.0, None) / cells_per_cm3
    return 2.0 * np.cbrt(3.0 * v_cm3 / (4.0 * math.pi))


def _interp_at(
    time_days: np.ndarray,
    values: np.ndarray,
    landmark_days: float,
) -> float:
    """Linear-interpolate ``values`` at ``landmark_days``.

    Returns ``nan`` when the landmark exceeds the simulated time range (the
    caller decides how to treat that — typically NaN endpoint).
    """
    if time_days.size == 0:
        return float("nan")
    if landmark_days < time_days[0] or landmark_days > time_days[-1]:
        return float("nan")
    return float(np.interp(landmark_days, time_days, values))


def _first_crossing(
    time_days: np.ndarray,
    values: np.ndarray,
    threshold: float,
    *,
    rising: bool,
) -> float | None:
    """First time at which ``values`` crosses ``threshold``.

    ``rising=True`` finds the first time ``values >= threshold`` when starting
    below; ``rising=False`` finds the first time ``values <= threshold`` when
    starting above. Returns ``None`` if no crossing occurs within the series.
    Uses linear interpolation on the bracketing interval.
    """
    if values.size < 2:
        return None
    if rising:
        mask = values >= threshold
    else:
        mask = values <= threshold
    if not mask.any():
        return None
    idx = int(np.argmax(mask))
    if idx == 0:
        return float(time_days[0])
    t0, t1 = time_days[idx - 1], time_days[idx]
    v0, v1 = values[idx - 1], values[idx]
    if v1 == v0:
        return float(t1)
    frac = (threshold - v0) / (v1 - v0)
    return float(t0 + frac * (t1 - t0))


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def os_at_landmark(
    time,
    species_dict: Mapping[str, np.ndarray],
    *,
    landmark_days: float,
    n_c_lethal: float = N_C_LETHAL_DEFAULT,
) -> np.ndarray:
    """Return 1.0 (alive) or 0.0 (dead) at ``landmark_days``.

    Dead := ``V_T.C1`` exceeds ``n_c_lethal`` at any t ≤ landmark. Evaluated
    on the time grid, not interpolated — monotonicity of tumor burden over a
    simulation interval is not guaranteed, so "dead by landmark" is the
    maximum-over-trajectory criterion. Returns NaN when the simulation did
    not reach the landmark.
    """
    t = _get_magnitude(time, "day")
    c1 = _get_magnitude(species_dict["V_T.C1"], "cell")
    if t.size == 0 or landmark_days > float(t[-1]):
        value = float("nan")
    else:
        mask = t <= landmark_days
        value = 0.0 if (c1[mask].max() >= n_c_lethal) else 1.0
    return np.full_like(t, value)


def dfs_at_landmark(
    time,
    species_dict: Mapping[str, np.ndarray],
    *,
    landmark_days: float,
    n_c_lethal: float = N_C_LETHAL_DEFAULT,
    progression_increase: float = PROGRESSION_INCREASE_DEFAULT,
    min_absolute_increase_cm: float = 0.5,
) -> np.ndarray:
    """Disease-free survival at ``landmark_days``.

    Returns 1.0 when the patient is alive AND has not progressed from the
    post-treatment nadir by the landmark, else 0.0.

    Progression follows RECIST 1.1: both ``diameter ≥ nadir · (1 +
    progression_increase)`` AND ``diameter − nadir ≥ min_absolute_increase_cm``
    must hold. The absolute-increase gate guards against spurious PD calls on
    small absolute diameters where percentage-only thresholds trigger easily.

    NaN when the simulation does not cover the landmark.
    """
    t = _get_magnitude(time, "day")
    c1 = _get_magnitude(species_dict["V_T.C1"], "cell")
    if t.size == 0 or landmark_days > float(t[-1]):
        return np.full_like(t, float("nan"))

    alive_mask = t <= landmark_days
    dead_by_landmark = bool((c1[alive_mask] >= n_c_lethal).any())
    if dead_by_landmark:
        return np.full_like(t, 0.0)

    diameter = _diameter_from_species(species_dict)
    d_upto = diameter[alive_mask]
    if d_upto.size < 2:
        return np.full_like(t, 1.0)
    nadir = float(d_upto.min())
    if nadir <= 0:
        return np.full_like(t, 1.0)
    progressed = bool(
        (
            (d_upto >= nadir * (1.0 + progression_increase))
            & (d_upto - nadir >= min_absolute_increase_cm)
        ).any()
    )
    return np.full_like(t, 0.0 if progressed else 1.0)


def tumor_diameter_at_day(
    time,
    species_dict: Mapping[str, np.ndarray],
    *,
    landmark_days: float,
    n_c_lethal: float = N_C_LETHAL_DEFAULT,
) -> np.ndarray:
    """Tumor diameter (cm) at ``landmark_days``, linearly interpolated.

    Returns NaN when the patient is already dead at ``landmark_days`` —
    ``V_T.C1`` crossed ``n_c_lethal`` at any t ≤ ``landmark_days``. Diameter
    values past death are mathematically valid simulator outputs but
    biologically meaningless (no one is measuring the tumor of a dead
    patient), and letting them through inflates posterior-predictive CI95
    tails to anatomically impossible values (e.g. 50+ cm). The uncensored
    trajectory is still available on the raw simulator output if a caller
    needs it.
    """
    t = _get_magnitude(time, "day")
    diameter = _diameter_from_species(species_dict)
    if t.size == 0 or landmark_days > float(t[-1]):
        value = float("nan")
    else:
        c1 = _get_magnitude(species_dict["V_T.C1"], "cell")
        mask = t <= landmark_days
        dead_before = bool(mask.any() and c1[mask].max() >= n_c_lethal)
        value = float("nan") if dead_before else _interp_at(t, diameter, landmark_days)
    return np.full_like(t, value)


def recist_category_at_day(
    time,
    species_dict: Mapping[str, np.ndarray],
    *,
    landmark_days: float,
    baseline_day: float = 0.0,
    progression_increase: float = PROGRESSION_INCREASE_DEFAULT,
    response_decrease: float = RESPONSE_DECREASE_DEFAULT,
    cr_diameter_cm: float = CR_DIAMETER_CM_DEFAULT,
    min_absolute_increase_cm: float = 0.5,
) -> dict[str, np.ndarray]:
    """One-hot {CR, PR, SD, PD} at ``landmark_days`` relative to baseline.

    Each key in the returned dict maps to a Pint-dimensionless array (1.0 on
    the matching category, 0.0 elsewhere) broadcast over the time vector.

    - CR: D(landmark) ≤ ``cr_diameter_cm``
    - PR: D(landmark) ≤ D(baseline) · (1 − ``response_decrease``) AND not CR
    - PD: D(landmark) ≥ nadir · (1 + ``progression_increase``) AND
          D(landmark) − nadir ≥ ``min_absolute_increase_cm`` (RECIST 1.1)
    - SD: otherwise
    """
    t = _get_magnitude(time, "day")
    diameter = _diameter_from_species(species_dict)
    d_base = _interp_at(t, diameter, baseline_day)
    d_land = _interp_at(t, diameter, landmark_days)

    if np.isnan(d_base) or np.isnan(d_land):
        cats = {k: float("nan") for k in ("CR", "PR", "SD", "PD")}
    else:
        mask = (t >= baseline_day) & (t <= landmark_days)
        nadir = float(diameter[mask].min()) if mask.any() else d_land
        cr = d_land <= cr_diameter_cm
        pd = (
            (d_land >= nadir * (1.0 + progression_increase))
            and (d_land - nadir >= min_absolute_increase_cm)
        )
        pr = (not cr) and (d_land <= d_base * (1.0 - response_decrease))
        sd = not (cr or pr or pd)
        cats = {
            "CR": 1.0 if cr else 0.0,
            "PR": 1.0 if pr else 0.0,
            "SD": 1.0 if sd else 0.0,
            "PD": 1.0 if pd else 0.0,
        }
    return {k: np.full_like(t, v) for k, v in cats.items()}


def time_to_progression(
    time,
    species_dict: Mapping[str, np.ndarray],
    *,
    baseline_day: float = 0.0,
    progression_increase: float = PROGRESSION_INCREASE_DEFAULT,
    min_absolute_increase_cm: float = 0.5,
    censoring_day: float | None = None,
) -> np.ndarray:
    """Time (days) from ``baseline_day`` to first RECIST PD.

    Progression (RECIST 1.1) = diameter ≥ running-nadir · (1 +
    ``progression_increase``) AND diameter − running-nadir ≥
    ``min_absolute_increase_cm``. The crossing time is linearly interpolated
    within the bracketing interval. When no progression is observed by
    ``censoring_day`` (or end of trajectory if ``censoring_day`` is ``None``),
    returns ``censoring_day`` as the censored value.
    """
    t = _get_magnitude(time, "day")
    diameter = _diameter_from_species(species_dict)
    if censoring_day is None:
        censoring_day = float(t[-1]) if t.size else float("nan")

    mask = (t >= baseline_day) & (t <= censoring_day)
    if not mask.any() or diameter[mask].size < 2:
        return np.full_like(t, float("nan"))

    t_win = t[mask]
    d_win = diameter[mask]
    # Running-nadir progression: at each time the threshold is
    # min_prior_d * (1 + inc) AND d - min_prior_d >= min_abs_increase.
    # Walk the trajectory and return the first time both conditions fire.
    nadir_running = np.minimum.accumulate(d_win)
    progressed = (
        (d_win >= nadir_running * (1.0 + progression_increase))
        & (d_win - nadir_running >= min_absolute_increase_cm)
    )
    if progressed.any():
        idx = int(np.argmax(progressed))
        # Interpolate within the bracketing interval for a smoother TTP.
        if idx == 0:
            value = float(t_win[0])
        else:
            nadir_prev = float(nadir_running[idx - 1])
            thr = max(
                nadir_prev * (1.0 + progression_increase),
                nadir_prev + min_absolute_increase_cm,
            )
            v0, v1 = float(d_win[idx - 1]), float(d_win[idx])
            if v1 == v0:
                value = float(t_win[idx])
            else:
                frac = (thr - v0) / (v1 - v0)
                value = float(t_win[idx - 1] + frac * (t_win[idx] - t_win[idx - 1]))
                value = float(np.clip(value, float(t_win[idx - 1]), float(t_win[idx])))
    else:
        value = censoring_day
    return np.full_like(t, value)


def tumor_volume_doubling_time(
    time,
    species_dict: Mapping[str, np.ndarray],
) -> np.ndarray:
    """VDT from exponential fit between first and last timepoints.

    Mirrors the existing ``tumor_doubling_time_PDAC_deriv001`` YAML — kept
    here for symmetry with the other endpoints and to allow an all-endpoints
    smoke test from one module.
    """
    t = _get_magnitude(time, "day")
    c1 = _get_magnitude(species_dict["V_T.C1"], "cell")
    if t.size < 2 or c1[0] <= 0 or c1[-1] <= 0:
        return np.full_like(t, float("nan"))
    dt = t[-1] - t[0]
    if dt <= 0:
        return np.full_like(t, float("nan"))
    r = (math.log(c1[-1]) - math.log(c1[0])) / dt
    if r <= 0:
        return np.full_like(t, float("nan"))
    vdt = math.log(2.0) / r
    return np.full_like(t, vdt)
