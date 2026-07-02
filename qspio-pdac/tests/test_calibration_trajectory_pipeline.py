"""Tests for workflows.calibration_trajectory_pipeline.

Builds synthetic per-sim trajectory binaries + duck-typed cal targets,
runs the full assembler → evaluator → plotter pipeline, verifies the
output figure is non-empty. No qsp_sim / SBI dependency.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from workflows.calibration_trajectory_pipeline import (
    collect_required_trajectory_columns,
    run_calibration_trajectory_pipeline,
)


@dataclass
class _Const:
    name: str
    value: float
    units: str


@dataclass
class _Empirical:
    median: float
    ci95_lower: float
    ci95_upper: float
    units: str


@dataclass
class _Observable:
    code: str
    units: str
    constants: list[_Const]
    species: list[str] = field(default_factory=list)


@dataclass
class _Target:
    observable_id: str
    observable: _Observable
    empirical_data: _Empirical


def _pack_v2(
    arr: np.ndarray, dt: float, t_end: float, n_sp: int, n_comp: int, n_rules: int
) -> bytes:
    n_t, n_cols = arr.shape
    assert n_cols == n_sp + n_comp + n_rules
    header = struct.pack(
        "<IIQQQQdd", 0x51535042, 2, n_t, n_sp, n_comp, n_rules, dt, t_end
    )
    return header + arr.astype("<f8").tobytes()


def _write_sim(
    traj_dir: Path,
    sample_index: int,
    arr: np.ndarray,
    *,
    dt: float,
    t_end: float,
    n_sp: int,
    n_comp: int,
    n_rules: int,
) -> Path:
    p = traj_dir / f"sim_{sample_index:09d}.bin"
    p.write_bytes(_pack_v2(arr, dt, t_end, n_sp, n_comp, n_rules))
    return p


def _make_density_target() -> _Target:
    code = (
        "def compute_observable(time, species_dict, constants):\n"
        "    return species_dict['V_T.CD8'] / species_dict['V_T']\n"
    )
    return _Target(
        observable_id="cd8_density",
        observable=_Observable(
            code=code,
            units="cell/mL",
            constants=[],
            species=["V_T.CD8", "V_T"],
        ),
        empirical_data=_Empirical(
            median=100.0, ci95_lower=80.0, ci95_upper=140.0, units="cell/mL"
        ),
    )


def test_pipeline_end_to_end(tmp_path: Path) -> None:
    """Three sims, one cal target — verifies plot file is written."""
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    species_names = ["V_T.CD8", "V_T.Treg", "V_T.M1"]
    comp_names = ["V_T", "V_LN"]
    rule_names = ["density_rule"]
    n_sp = len(species_names)
    n_comp = len(comp_names)
    n_rules = len(rule_names)
    n_cols = n_sp + n_comp + n_rules

    rng = np.random.default_rng(42)
    for sim_idx in range(3):
        # 3 timepoints per sim. Column order: species, then comps, then rules.
        # CD8 (col 0), Treg (1), M1 (2), V_T (3), V_LN (4), density_rule (5)
        arr = np.zeros((3, n_cols), dtype="f8")
        arr[:, 0] = [10.0, 50.0, 100.0]  # CD8 grows
        arr[:, 1] = [5.0, 25.0, 50.0]    # Treg grows
        arr[:, 2] = [3.0, 15.0, 30.0]    # M1 grows
        arr[:, 3] = [1.0, 1.0, 1.0]      # V_T (mL)
        arr[:, 4] = [10.0, 10.0, 10.0]   # V_LN
        arr[:, 5] = arr[:, 0] / arr[:, 3]  # density_rule
        arr += rng.normal(0, 0.5, arr.shape)
        _write_sim(
            traj_dir, sim_idx, arr,
            dt=10.0, t_end=20.0,
            n_sp=n_sp, n_comp=n_comp, n_rules=n_rules,
        )

    target = _make_density_target()
    out_path = tmp_path / "out.png"
    fig = run_calibration_trajectory_pipeline(
        traj_dir=traj_dir,
        species_names=species_names,
        compartment_names=comp_names,
        rule_names=rule_names,
        calibration_targets=[target],
        output_path=out_path,
        title="smoke",
    )
    assert out_path.exists()
    assert out_path.stat().st_size > 1000  # png has substance
    # Figure has one populated panel.
    populated = [ax for ax in fig.axes if ax.has_data()]
    assert len(populated) == 1


def test_pipeline_empty_traj_dir_raises(tmp_path: Path) -> None:
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()
    target = _make_density_target()
    with pytest.raises(ValueError, match="no per-sim trajectory binaries"):
        run_calibration_trajectory_pipeline(
            traj_dir=traj_dir,
            species_names=["V_T.CD8", "V_T.Treg", "V_T.M1"],
            compartment_names=["V_T", "V_LN"],
            rule_names=["density_rule"],
            calibration_targets=[target],
        )


def test_collect_required_trajectory_columns() -> None:
    """Helper aggregates ``observable.species`` across multiple targets."""
    t1 = _make_density_target()
    t2 = _Target(
        observable_id="treg_fraction",
        observable=_Observable(
            code="def compute_observable(time, species_dict, constants):\n"
                 "    return species_dict['V_T.Treg'] / species_dict['V_T.CD4']\n",
            units="dimensionless",
            constants=[],
            species=["V_T.Treg", "V_T.CD4"],
        ),
        empirical_data=_Empirical(0.3, 0.2, 0.4, "dimensionless"),
    )
    cols = collect_required_trajectory_columns([t1, t2])
    assert cols == sorted(["V_T.CD8", "V_T", "V_T.Treg", "V_T.CD4"])


def test_collect_required_handles_targets_without_species() -> None:
    """Gracefully ignores targets that don't expose ``observable.species``."""

    @dataclass
    class _Naked:
        observable_id: str

    t1 = _make_density_target()
    cols = collect_required_trajectory_columns([t1, _Naked(observable_id="x")])
    assert cols == sorted(t1.observable.species)
