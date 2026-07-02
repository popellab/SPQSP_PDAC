"""MATLAB ↔ C++ trajectory parity regression for the in-repo PDAC model.

Drives the harness from `qsp_codegen.parity`: exports a 365-day MATLAB
trajectory from the live ``immune_oncology_model_PDAC.m`` using the same
``resources/cpp/param_all.xml`` the C++ side consumes, runs the in-repo
``qsp_sim`` binary over the same window, and fails if any matched species
diverge past rtol=5% / atol=1e-6.

Skipped when MATLAB or the built binary are missing (CI won't have either
without extra setup); local runs guarded by ``pytest -m slow``.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from qsp_codegen.parity import (
    compare,
    run_cpp_trajectories,
    run_matlab_trajectories,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
QSP_SIM = REPO_ROOT / "cpp/sim/build/qsp_sim"
PARAM_XML = REPO_ROOT / "resources/cpp/param_all.xml"
SBML = REPO_ROOT / "PDAC_model.sbml"
DRUG_METADATA = REPO_ROOT / "resources/cpp/drug_metadata.yaml"
HEALTHY_STATE = REPO_ROOT / "resources/cpp/healthy_state.yaml"
MATLAB_MODEL_SCRIPT = "immune_oncology_model_PDAC"

STOP_TIME_DAYS = float(os.environ.get("PARITY_STOP_TIME", "365"))
RTOL = 0.05
ATOL = 1e-6

# Dosed scenario with evolve_to_diagnosis on both sides (stop_time from YAML).
SCENARIO_YAML = REPO_ROOT / "scenarios/gvax_nivo_neoadjuvant_zheng2022.yaml"
SCENARIO_STOP_DAYS = 30.0

# All scenarios consumed by the multi-scenario SBI workflow — parity is
# meaningful for each because evolve_to_diagnosis ICs + scenario stop_time +
# dose plan are all on the cross-validation path. Listed as (yaml_path,
# stop_time_days, test_id) tuples so pytest can parametrize cleanly.
SBI_SCENARIOS = [
    (REPO_ROOT / "scenarios/baseline_no_treatment.yaml", 1.0, "baseline_no_treatment"),
    (REPO_ROOT / "scenarios/clinical_progression.yaml", 365.0, "clinical_progression"),
    (REPO_ROOT / "scenarios/gvax_neoadjuvant_zheng2022.yaml", 21.0, "gvax_neoadjuvant"),
    (REPO_ROOT / "scenarios/gvax_nivo_neoadjuvant_zheng2022.yaml", 30.0, "gvax_nivo"),
]


@pytest.fixture(scope="module")
def matlab_binary() -> str:
    path = shutil.which("matlab")
    if path is None:
        pytest.skip("MATLAB not on PATH — cannot run parity")
    return path


@pytest.fixture(scope="module")
def qsp_sim_binary() -> Path:
    if not QSP_SIM.exists():
        pytest.skip(
            f"{QSP_SIM} not built. Run `make refresh-cpp` first."
        )
    return QSP_SIM


@pytest.mark.slow
def test_365d_default_ic_parity(tmp_path_factory, matlab_binary, qsp_sim_binary):
    """Full-trajectory parity from XML-default ICs, no treatments, no evolve.

    With the cpp-codegen assignment-rule fix (qsp-codegen PR #1), PDAC's
    365-day default-IC comparison is expected to be *clean*: every
    common species matches within rtol=5% / atol=1e-6. This test is a
    regression guard against either a MATLAB-side module edit that
    diverges from the generated ODE, or a codegen bug in `qsp-codegen`
    that recurs (e.g., another assignment-rule species whose output path
    drifts).
    """
    out_dir = tmp_path_factory.mktemp("parity")
    cpp_csv = out_dir / "cpp.csv"
    matlab_csv = out_dir / "matlab.csv"

    run_cpp_trajectories(
        qsp_sim=qsp_sim_binary,
        param_xml=PARAM_XML,
        out_csv=cpp_csv,
        t_end_days=STOP_TIME_DAYS,
    )
    run_matlab_trajectories(
        matlab_model_dir=REPO_ROOT,
        matlab_model_script=MATLAB_MODEL_SCRIPT,
        sbml_path=SBML,
        param_xml=PARAM_XML,
        out_csv=matlab_csv,
        stop_time=STOP_TIME_DAYS,
        matlab_binary=matlab_binary,
    )

    passed, report = compare(str(matlab_csv), str(cpp_csv), rtol=RTOL, atol=ATOL)
    assert passed, f"Parity FAILED — full report:\n{report}"


@pytest.mark.slow
@pytest.mark.parametrize(
    "scenario_yaml,stop_days",
    [(s[0], s[1]) for s in SBI_SCENARIOS],
    ids=[s[2] for s in SBI_SCENARIOS],
)
def test_sbi_scenario_parity(
    tmp_path_factory, matlab_binary, qsp_sim_binary, scenario_yaml, stop_days
):
    """Scenario parity: evolve_to_diagnosis + scenario YAML on both sides.

    Runs once per SBI scenario. Exercises every axis the default-IC test
    misses — the scenario YAML dose schedule must land at the same times
    with the same amounts on both sides, the evolved ICs must match, and
    the post-dose dynamics must not diverge.

    Covers the 4 scenarios that feed the multi-scenario SBI workflow:
    - baseline_no_treatment (burn-in-only snapshot)
    - clinical_progression (365 d, no doses)
    - gvax_neoadjuvant (Cy + GVAX, 21 d)
    - gvax_nivo (Cy + GVAX + nivo, 30 d — the previously-broken stiff one)
    """
    if not scenario_yaml.exists():
        pytest.skip(f"scenario not found: {scenario_yaml}")
    out_dir = tmp_path_factory.mktemp(f"scen_parity_{scenario_yaml.stem}")
    cpp_csv = out_dir / "cpp.csv"
    matlab_csv = out_dir / "matlab.csv"

    run_cpp_trajectories(
        qsp_sim=qsp_sim_binary,
        param_xml=PARAM_XML,
        out_csv=cpp_csv,
        t_end_days=stop_days,
        scenario_yaml=scenario_yaml,
        drug_metadata_yaml=DRUG_METADATA,
        evolve_to_diagnosis_yaml=HEALTHY_STATE,
    )
    # Pin MATLAB's sample grid to the C++ grid so the comparison is
    # row-aligned (no linear-interp artifacts at bolus discontinuities,
    # which otherwise cause spurious "post-bolus value at pre-bolus
    # timestamp" failures for jump-target species like V_C.aPD1).
    cpp_times = np.loadtxt(cpp_csv, delimiter=",", skiprows=1, usecols=0)
    cpp_times_csv = out_dir / "cpp_times.csv"
    np.savetxt(cpp_times_csv, cpp_times, fmt="%.12g")

    run_matlab_trajectories(
        matlab_model_dir=REPO_ROOT,
        matlab_model_script=MATLAB_MODEL_SCRIPT,
        sbml_path=SBML,
        param_xml=PARAM_XML,
        out_csv=matlab_csv,
        stop_time=stop_days,
        scenario_yaml=scenario_yaml,
        evolve_function_name="evolve_to_diagnosis",
        output_times_csv=cpp_times_csv,
        matlab_binary=matlab_binary,
    )

    passed, report = compare(str(matlab_csv), str(cpp_csv), rtol=RTOL, atol=ATOL)
    assert passed, f"Scenario parity FAILED ({scenario_yaml.name}):\n{report}"
