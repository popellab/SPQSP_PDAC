"""Regression checks for the neoadjuvant Cy/GVAX/nivolumab regimen."""

from __future__ import annotations

import math
from pathlib import Path
import re
import subprocess

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO = REPO_ROOT / "scenarios/gvax_nivo_neoadjuvant_zheng2022.yaml"
DRUG_METADATA = REPO_ROOT / "resources/cpp/drug_metadata.yaml"
QSP_SIM = REPO_ROOT / "cpp/sim/build/qsp_sim"
PARAM_XML = REPO_ROOT / "resources/cpp/param_all.xml"


def test_scenario_represents_majority_arm_b_regimen() -> None:
    scenario = yaml.safe_load(SCENARIO.read_text())
    dosing = scenario["dosing"]

    assert dosing["patientWeight"] == 70
    assert dosing["patientBSA"] == 1.9
    assert dosing["cyclophosphamide_dose"] == 380
    assert dosing["cyclophosphamide_schedule"] == [6, 0, 1]
    assert dosing["nivolumab_dose"] == 3.0
    assert dosing["nivolumab_schedule"] == [6, 0, 1]
    assert dosing["GVAX_dose"] == 5.0e8
    assert dosing["GVAX_schedule"] == [7, 0, 1]

    metadata = yaml.safe_load(DRUG_METADATA.read_text())["drugs"]["nivolumab"]
    assert metadata["dose_basis"] == "per_weight"
    assert dosing["nivolumab_dose"] * dosing["patientWeight"] == 210.0


def test_matlab_runner_does_not_rescale_scenario_doses() -> None:
    source = (REPO_ROOT / "scripts/run_median_simulation.m").read_text()

    assert "nivo_dose_per_kg" not in source
    assert "dosing_fields = fieldnames(dosing_cfg);" in source
    assert "field_value = dosing_cfg.(field_name);" in source
    assert "dosing_args = [dosing_args, field_name, field_value]" in source


@pytest.mark.skipif(not QSP_SIM.exists(), reason="qsp_sim binary is not built")
def test_cpp_runner_applies_expected_nivolumab_amount_and_timing(tmp_path: Path) -> None:
    output = tmp_path / "trajectory.csv"
    result = subprocess.run(
        [
            str(QSP_SIM),
            str(PARAM_XML),
            str(output),
            "8",
            "4",
            "--scenario",
            str(SCENARIO),
            "--drug-metadata",
            str(DRUG_METADATA),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    log = result.stdout + result.stderr

    assert re.search(r"t=6d\s+cyclophosphamide@V_C\.Cy", log)
    assert re.search(r"t=7d\s+GVAX@V_ID\.GVAX_cells", log)
    match = re.search(r"t=6d\s+nivolumab@V_C\.aPD1\s+amount=([0-9.eE+-]+)", log)
    assert match, log

    expected_moles = 3.0 * 70.0 / 1.436e8
    assert math.isclose(float(match.group(1)), expected_moles, rel_tol=1e-5)
