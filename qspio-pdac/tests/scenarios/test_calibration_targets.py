"""Tests for calibration target YAML schema validation.

Validates all calibration target YAMLs in calibration_targets/ against
the CalibrationTarget Pydantic model from maple.

Usage:
    pytest tests/scenarios/test_calibration_targets.py
"""

import pytest
import yaml
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
CAL_TARGETS_DIR = REPO_ROOT / "calibration_targets"
MODEL_STRUCTURE_FILE = REPO_ROOT / "model_structure.json"


try:
    from maple.core.calibration import CalibrationTarget, CrossScenarioCalibrationTarget
    from maple.core.calibration.submodel_target import SubmodelTarget
    from maple.core.model_structure import ModelStructure

    HAS_MAPLE = True
except ImportError:
    HAS_MAPLE = False

SUBMODEL_DIR = CAL_TARGETS_DIR / "submodel_targets"
CROSS_SCENARIO_DIR = CAL_TARGETS_DIR / "cross_scenario"
SKIP_FILES = {"reference_values.yaml", "parameter_groups.yaml", "submodel_config.yaml", "submodel_priors.yaml",
              "compare_inference_results.yaml"}
SKIP_DIRS = {"papers", "retired", "excluded", "to-review", "predictions"}


def _get_calibration_target_files():
    """Collect CalibrationTarget YAMLs (excluding submodel_targets/, cross_scenario/, and non-target files)."""
    if not CAL_TARGETS_DIR.exists():
        return []
    files = []
    for f in sorted(CAL_TARGETS_DIR.rglob("*.yaml")):
        if f.name in SKIP_FILES:
            continue
        if SUBMODEL_DIR.exists() and f.is_relative_to(SUBMODEL_DIR):
            continue
        if CROSS_SCENARIO_DIR.exists() and f.is_relative_to(CROSS_SCENARIO_DIR):
            continue
        if any(d in f.parts for d in SKIP_DIRS):
            continue
        files.append(f)
    return files


def _get_cross_scenario_target_files():
    """Collect CrossScenarioCalibrationTarget YAMLs from cross_scenario/."""
    if not CROSS_SCENARIO_DIR.exists():
        return []
    return sorted(
        f for f in CROSS_SCENARIO_DIR.rglob("*.yaml")
        if f.name not in SKIP_FILES
        and not any(d in f.parts for d in SKIP_DIRS)
    )


def _get_submodel_target_files():
    """Collect SubmodelTarget YAMLs from submodel_targets/."""
    if not SUBMODEL_DIR.exists():
        return []
    return sorted(
        f for f in SUBMODEL_DIR.rglob("*.yaml")
        if f.name not in SKIP_FILES
        and not any(d in f.parts for d in SKIP_DIRS)
    )


@pytest.mark.skipif(not HAS_MAPLE, reason="maple not installed")
class TestCalibrationTargetSchema:
    """Validate calibration target YAMLs against Pydantic schema."""

    @pytest.fixture(scope="class")
    def model_structure(self):
        if not MODEL_STRUCTURE_FILE.exists():
            pytest.skip(f"model_structure.json not found: {MODEL_STRUCTURE_FILE}")
        return ModelStructure.from_json(MODEL_STRUCTURE_FILE)


    @pytest.mark.slow
    @pytest.mark.parametrize(
        "yaml_file",
        _get_calibration_target_files(),
        ids=lambda p: p.stem,
    )
    def test_calibration_target_validates(self, yaml_file, model_structure):
        """Each CalibrationTarget YAML must pass schema validation."""
        data = yaml.safe_load(yaml_file.read_text())
        CalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "yaml_file",
        _get_cross_scenario_target_files(),
        ids=lambda p: p.stem,
    )
    def test_cross_scenario_target_validates(self, yaml_file, model_structure):
        """Each CrossScenarioCalibrationTarget YAML must pass schema validation."""
        data = yaml.safe_load(yaml_file.read_text())
        CrossScenarioCalibrationTarget.model_validate(data, context={"model_structure": model_structure})

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "yaml_file",
        _get_submodel_target_files(),
        ids=lambda p: p.stem,
    )
    def test_submodel_target_validates(self, yaml_file):
        """Each SubmodelTarget YAML must pass schema validation."""
        if yaml_file.stem == "TGFb_50_PDAC_deriv001":
            pytest.xfail("maple validator incorrectly requires non-zero bootstrap spread")
        data = yaml.safe_load(yaml_file.read_text())
        SubmodelTarget.model_validate(data)

    def test_unique_calibration_target_ids(self):
        """All calibration_target_id values must be unique across YAMLs."""
        target_ids = {}
        all_files = _get_calibration_target_files() + _get_submodel_target_files()
        for f in all_files:
            data = yaml.safe_load(f.read_text())
            tid = data.get("calibration_target_id") or data.get("target_id")
            if tid is None:
                continue
            if tid in target_ids:
                pytest.fail(
                    f"Duplicate target id '{tid}' in "
                    f"{f.name} and {target_ids[tid]}"
                )
            target_ids[tid] = f.name

    def test_at_least_one_target_exists(self):
        """There should be at least one calibration target YAML."""
        files = _get_calibration_target_files()
        assert len(files) > 0, f"No YAML files found in {CAL_TARGETS_DIR}"
