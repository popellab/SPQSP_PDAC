"""Schema + smoke-dispatch tests for the cross-scenario calibration targets
shipped under calibration_targets/cross_scenario/.

Schema validation: every YAML in the directory parses cleanly via maple's
CrossScenarioCalibrationTarget Pydantic model.

Smoke dispatch: every YAML loads through the qsp-hpc-tools cross_scenario
loader + composer worker against a synthetic scenario_meta containing only
the test_statistic_ids each YAML references. Catches obvious wiring
mistakes (typo'd scenario name, missing test_statistic_id, broken composer
code) without needing a real SBI run.

This is the pdac-build-side analogue of tests/test_prediction_targets.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from maple.core.calibration import CrossScenarioCalibrationTarget
from qsp_hpc.batch.cross_scenario_worker import compute_cross_scenario_statistics
from qsp_hpc.calibration import load_cross_scenario_targets

CROSS_SCENARIO_DIR = Path(__file__).parent.parent / "calibration_targets" / "cross_scenario"


def _yaml_paths() -> list[Path]:
    return sorted(CROSS_SCENARIO_DIR.glob("*.yaml"))


def _yaml_id(path: Path) -> str:
    return path.stem


@pytest.mark.parametrize("path", _yaml_paths(), ids=_yaml_id)
def test_schema_valid(path: Path) -> None:
    """Every YAML round-trips through CrossScenarioCalibrationTarget."""
    data = yaml.safe_load(path.read_text())
    target = CrossScenarioCalibrationTarget(**data)
    assert target.cross_scenario_target_id == path.stem


def test_loader_parses_all_yamls() -> None:
    """The qsp-hpc-tools loader produces one row per YAML."""
    df = load_cross_scenario_targets(CROSS_SCENARIO_DIR)
    yaml_ids = {p.stem for p in _yaml_paths()}
    df_ids = set(df["cross_scenario_target_id"])
    assert df_ids == yaml_ids


@pytest.mark.parametrize("path", _yaml_paths(), ids=_yaml_id)
def test_smoke_dispatch_with_synthetic_inputs(path: Path) -> None:
    """For each YAML, build a synthetic scenario_meta keyed on the
    test_statistic inputs the YAML references and confirm the composer
    runs without error producing a finite scalar.

    Skipped for targets that declare timeseries inputs — those need a
    full sim_df fixture, out of scope for a wiring smoke test.
    """
    data = yaml.safe_load(path.read_text())
    target = CrossScenarioCalibrationTarget(**data)

    # All-scalar smoke only.
    if any(inp.input_kind == "timeseries" for inp in target.observable.inputs):
        pytest.skip("timeseries-input targets need sim_df fixtures (separate test)")

    df = load_cross_scenario_targets(CROSS_SCENARIO_DIR)
    df = df[df["cross_scenario_target_id"] == path.stem].reset_index(drop=True)

    # Build scenario_meta with exactly the per-scenario test_statistics
    # this YAML references; populate with values near 1.0 so the composer
    # produces a finite result.
    inputs_spec = json.loads(df.iloc[0]["inputs_json"])
    scenario_meta: dict = {}
    species_units: dict = {}
    for inp in inputs_spec:
        scen = inp["scenario"]
        tsid = inp["test_statistic_id"]
        meta = scenario_meta.setdefault(
            scen, {"x_raw": [], "observable_names": []}
        )
        if tsid not in meta["observable_names"]:
            meta["observable_names"].append(tsid)
        species_units[tsid] = "dimensionless"

    n_sims = 4
    for scen, meta in scenario_meta.items():
        n_cols = len(meta["observable_names"])
        # Values in [1.0, 1.5] keep ratios well-conditioned for any
        # composer that divides one by another.
        meta["x_raw"] = np.linspace(1.0, 1.5, n_sims * n_cols).reshape(n_sims, n_cols)

    out = compute_cross_scenario_statistics(scenario_meta, df, species_units)
    assert out.shape == (n_sims, 1)
    assert np.all(np.isfinite(out)), f"Composer produced NaN for {path.stem}: {out}"


def test_sbi_runner_cfg_field_default_empty() -> None:
    """RunConfig.cross_scenario_targets defaults to [] so existing
    workflows are byte-identical to pre-D3+ behavior."""
    # Import from workflows.run_config (the heavy-import-free split) so this
    # smoke test doesn't pull torch + sbi just to read a dataclass default.
    from workflows.run_config import RunConfig

    cfg = RunConfig()
    assert cfg.cross_scenario_targets == []
