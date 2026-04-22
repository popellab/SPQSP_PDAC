"""Compare MATLAB SimBiology and C++ CVODE ODE trajectories.

Requires MATLAB + a local pdac-build/ sibling directory with startup.m and
immune_oncology_model_PDAC.m. The qsp_sim binary must be built
from PDAC/qsp/sim/ (see its CMakeLists.txt).

Run locally with:
    pytest PDAC/sim/tests/test_ode_vs_matlab.py -v --run-matlab
"""

import subprocess
import sys
from pathlib import Path

import pytest

import os as _os

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent.parent
# The MATLAB .m source the test drives must match the branch/worktree whose
# SBML was exported to regenerate the C++ codegen under test. Otherwise the
# two engines simulate different models (see the Fix-2 equivalence bug,
# 2026-04-21 — MATLAB ran main-branch modules while C++ ran the
# worktree-exported SBML, producing spurious 5.8× divergence). The
# pdac-build Makefile's cpp-equivalence target exports PDAC_BUILD_DIR so the
# worktree wins; fall back to the sibling pdac-build/ only if unset.
PDAC_BUILD = Path(_os.environ.get("PDAC_BUILD_DIR", REPO_ROOT.parent / "pdac-build"))
ODE_BUILD = REPO_ROOT / "PDAC" / "qsp" / "sim" / "build"
PARAM_XML = REPO_ROOT / "PDAC" / "sim" / "resource" / "param_all.xml"
# Both MATLAB and C++ simulate from the same XML so parameter divergence
# isn't confounded with solver divergence. Override via PDAC_PARAM_XML env var.
PARAM_XML = Path(_os.environ.get("PDAC_PARAM_XML", str(PARAM_XML)))
# MATLAB loads the model via sbmlimport from the same SBML the C++ codegen
# reads, so both sides simulate the exact same ODE structure.
SBML_PATH = REPO_ROOT / "PDAC" / "qsp" / "PDAC_model.sbml"
MATLAB_EXPORT_SCRIPT = TESTS_DIR / "export_matlab_trajectories.m"

DUMP_BIN = ODE_BUILD / "qsp_sim"
DRUG_META = REPO_ROOT / "PDAC" / "sim" / "resource" / "drug_metadata.yaml"
SCENARIO_DIR = PDAC_BUILD / "scenarios"

sys.path.insert(0, str(TESTS_DIR))
sys.path.insert(0, str(REPO_ROOT / "PDAC" / "codegen"))


# --- Staleness / sync guards -------------------------------------------------
# These wrap the shared sync_checks module (used by PDAC/codegen/check_sync.py
# too) so config drift surfaces as a clear failure instead of a bogus
# trajectory divergence.

def _assert_check(check_fn, **kwargs):
    ok, msg = check_fn(**kwargs)
    if not ok:
        if msg.startswith("skip:"):
            pytest.skip(msg[5:].strip())
        pytest.fail(msg)


def test_sbml_newer_than_matlab_model_script():
    from sync_checks import check_sbml_newer_than_matlab
    _assert_check(check_sbml_newer_than_matlab,
                  sbml=SBML_PATH,
                  matlab_script=PDAC_BUILD / "scripts" / "immune_oncology_model_PDAC.m")


def test_generated_ode_newer_than_sbml():
    from sync_checks import check_codegen_newer_than_sbml
    _assert_check(check_codegen_newer_than_sbml, sbml=SBML_PATH)


def test_dump_binary_newer_than_ode_cpp():
    from sync_checks import check_binary_newer_than_codegen
    _assert_check(check_binary_newer_than_codegen, dump_bin=DUMP_BIN)


def test_param_xml_contains_snippet_names():
    from sync_checks import check_param_xml_contains_snippet
    _assert_check(check_param_xml_contains_snippet, param_xml=PARAM_XML)


def test_priors_csv_names_in_param_xml():
    """Catch orphan rows in pdac-build/parameters/pdac_priors.csv.

    Each prior row becomes a sampled column in params.csv which the C++
    worker sets via the XML template. Names absent from param_all.xml
    abort the worker with ParamNotFoundError — this test catches them
    at check-time instead.
    """
    from sync_checks import check_priors_csv_names_in_param_xml
    _assert_check(check_priors_csv_names_in_param_xml, param_xml=PARAM_XML)


@pytest.fixture(scope="session")
def run_matlab(request):
    if not request.config.getoption("--run-matlab"):
        pytest.skip("Requires --run-matlab flag (local only)")


@pytest.fixture(scope="session")
def matlab_trajectories(run_matlab, tmp_path_factory):
    csv_path = tmp_path_factory.mktemp("matlab") / "matlab_trajectories.csv"
    matlab_cmd = (
        f"output_csv='{csv_path}'; pdac_build_dir='{PDAC_BUILD}'; "
        f"param_xml='{PARAM_XML}'; sbml_path='{SBML_PATH}'; "
        f"run('{MATLAB_EXPORT_SCRIPT}')"
    )
    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True, text=True, timeout=300,
        cwd=str(PDAC_BUILD),
    )
    assert result.returncode == 0, f"MATLAB failed:\n{result.stderr}"
    assert csv_path.exists(), "MATLAB did not produce trajectory CSV"
    return csv_path


@pytest.fixture(scope="session")
def cpp_trajectories(tmp_path_factory):
    if not DUMP_BIN.exists():
        pytest.skip(f"qsp_sim not built: {DUMP_BIN}")
    if not PARAM_XML.exists():
        pytest.skip(f"param_all.xml not found: {PARAM_XML}")

    csv_path = tmp_path_factory.mktemp("cpp") / "cpp_trajectories.csv"
    result = subprocess.run(
        [str(DUMP_BIN), str(PARAM_XML), str(csv_path), "365", "0.1"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"C++ ODE failed:\n{result.stderr}"
    assert csv_path.exists()

    # Any "QSP param not found" warning means the XML is missing a name the
    # generated C++ asked for, and QSPParam.cpp silently defaulted it to 0.
    # That produces diverged trajectories with no loud signal, so treat it
    # as a hard failure.
    missing = [line for line in result.stderr.splitlines()
               if "QSP param not found" in line]
    assert not missing, (
        f"{len(missing)} QSP params silently defaulted to 0 during the C++ run. "
        f"Refresh param_all.xml from qsp_params_xml_snippet.xml.\n"
        f"First few:\n  " + "\n  ".join(missing[:5])
    )
    return csv_path


def _load_traj(path):
    """Load a trajectory CSV as (times, values_by_species_name)."""
    import numpy as np
    with open(path) as f:
        header = f.readline().strip().split(",")
    data = np.genfromtxt(path, delimiter=",", skip_header=1)
    times = data[:, 0]
    # Normalize "V_C.nCD4" ↔ "V_C_nCD4" so MATLAB and C++ keys align.
    values = {h.replace(".", "_"): data[:, i + 1] for i, h in enumerate(header[1:])}
    return times, values


def test_time_grids_match(matlab_trajectories, cpp_trajectories):
    """MATLAB and C++ must sample at the exact same time points.

    Both sides configure dt=0.1, tend=365 — a mismatch means someone
    edited OutputTimes on one side without the other.
    """
    import numpy as np
    t_m, _ = _load_traj(matlab_trajectories)
    t_c, _ = _load_traj(cpp_trajectories)
    assert len(t_m) == len(t_c), (
        f"Time grids differ in length: MATLAB={len(t_m)}, C++={len(t_c)}"
    )
    max_diff = float(np.abs(t_m - t_c).max())
    assert max_diff < 1e-9, f"Time grids misaligned: max diff = {max_diff}"


def test_initial_conditions_match(matlab_trajectories, cpp_trajectories):
    """Every species in C++ must start at the same value MATLAB does.

    If this fails, the trajectory-match failure is uninformative — we'd be
    comparing trajectories of two different ODEs. This isolates "model+IC
    setup" from "integration" so drift is attributed to the right side.
    """
    _, vm = _load_traj(matlab_trajectories)
    _, vc = _load_traj(cpp_trajectories)
    common = sorted(set(vm) & set(vc))
    bad = []
    for k in common:
        a, b = vm[k][0], vc[k][0]
        denom = max(abs(a), abs(b), 1e-12)
        rd = abs(a - b) / denom
        if rd > 1e-6:
            bad.append((k, a, b, rd))
    assert not bad, (
        f"{len(bad)} species have mismatched ICs at t=0 (rtol > 1e-6). "
        f"First few:\n  " +
        "\n  ".join(f"{k}: MATLAB={a:.6e}, C++={b:.6e}, rdiff={rd:.2e}"
                    for k, a, b, rd in bad[:5])
    )


# --- Event-exercising scenario -----------------------------------------------
# With V_T.C1 = 0.1 at t=0 (below the 0.5*cell threshold), both SBML events
# must fire on the very first integration step: Event_1 resets V_T.C1 to
# 0.01*cell, Event_2 resets V_T.K to 0.01*cell. If the C++ codegen isn't
# wiring events correctly, V_T.K stays at its initial 10300 and the MATLAB
# vs C++ trajectory diverges immediately.

EVENT_STOP_TIME = 30.0  # days; short scenario keeps MATLAB fast


def _override_param_xml(base: Path, overrides: dict, dst: Path) -> None:
    """Copy base param_all.xml to dst with leaf-tag values overridden."""
    import re
    text = base.read_text()
    for name, val in overrides.items():
        pat = re.compile(rf"(<{re.escape(name)}>)[^<]*(</{re.escape(name)}>)")
        new_text, n = pat.subn(rf"\g<1>{val}\g<2>", text, count=1)
        if n == 0:
            raise ValueError(f"Tag <{name}> not found in {base}")
        text = new_text
    dst.write_text(text)


@pytest.fixture(scope="session")
def event_param_xml(tmp_path_factory):
    dst = tmp_path_factory.mktemp("event_scenario") / "param_event.xml"
    _override_param_xml(PARAM_XML, {"V_T_C1": "0.1"}, dst)
    return dst


@pytest.fixture(scope="session")
def event_cpp_trajectories(event_param_xml, tmp_path_factory):
    if not DUMP_BIN.exists():
        pytest.skip(f"dump_trajectories not built: {DUMP_BIN}")
    csv_path = tmp_path_factory.mktemp("cpp_event") / "cpp_trajectories.csv"
    result = subprocess.run(
        [str(DUMP_BIN), str(event_param_xml), str(csv_path),
         str(EVENT_STOP_TIME), "0.1"],
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, f"C++ ODE failed:\n{result.stderr}"
    return csv_path


@pytest.fixture(scope="session")
def event_matlab_trajectories(run_matlab, event_param_xml, tmp_path_factory):
    csv_path = tmp_path_factory.mktemp("matlab_event") / "matlab_trajectories.csv"
    matlab_cmd = (
        f"output_csv='{csv_path}'; pdac_build_dir='{PDAC_BUILD}'; "
        f"param_xml='{event_param_xml}'; sbml_path='{SBML_PATH}'; "
        f"stop_time={EVENT_STOP_TIME}; "
        f"run('{MATLAB_EXPORT_SCRIPT}')"
    )
    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True, text=True, timeout=180,
        cwd=str(PDAC_BUILD),
    )
    assert result.returncode == 0, f"MATLAB failed:\n{result.stderr}"
    return csv_path


def test_event_does_not_fire_at_t0_when_trigger_already_satisfied(event_cpp_trajectories):
    """SBML trigger.initialValue defaults to true, so events must NOT fire
    at t=0 just because the condition happens to hold — only on a later
    false→true transition. V_T.C1=0.1 starts below the 0.5 threshold, so
    the trigger is already satisfied; V_T.K should stay at its initial
    10300 through the whole simulation.

    This regression-tests the initialValue=true semantics. Previously the
    C++ codegen initialized _event_triggered[i] to false, which caused
    spurious firing at t=0 even when the trigger was already satisfied.
    """
    _, vc = _load_traj(event_cpp_trajectories)
    vtk = vc["V_T_K"]
    assert vtk[0] > 1000, (
        f"V_T.K at t=0 should be its initial 10300, got {vtk[0]:.3e}"
    )
    # No false→true transition occurs in this scenario, so V_T.K should
    # evolve only via its own rate law — not be reset by Event_2.
    assert vtk[-1] > 1000, (
        f"V_T.K at t_end should still be near 10300 (event must NOT fire), "
        f"got {vtk[-1]:.3e}"
    )


def test_event_trajectories_match(event_matlab_trajectories, event_cpp_trajectories):
    """With events firing at t=0 on both sides, trajectories should still agree."""
    from compare_trajectories import compare

    passed, report = compare(
        str(event_matlab_trajectories),
        str(event_cpp_trajectories),
        rtol=0.05,
        atol=1e-6,
    )
    print(report)

    for line in report.split("\n"):
        if "Failures:" in line:
            failures = int(line.split(":")[1].strip().split(" ")[0])
            total_line = next(l for l in report.split("\n") if "Comparisons:" in l)
            total = int(total_line.split(":")[1].strip())
            pass_rate = 1 - failures / total
            print(f"Pass rate: {pass_rate:.1%}")
            assert pass_rate > 0.95, (
                f"Event scenario pass rate too low: {pass_rate:.1%}. "
                f"MATLAB and C++ may be handling the event differently."
            )


# --- Scenario (dosing) validation -------------------------------------------
# Drive both sides from the same scenario YAML. C++ reads the YAML + drug
# metadata and applies boluses as `V_C.aPD1 += N mol` at scheduled times;
# MATLAB builds a SimBiology dose_schedule via schedule_dosing.m and passes
# it to sbiosimulate. Trajectories should agree post-dose.

SCENARIO_YAML = TESTS_DIR / "scenarios" / "nivo_single_day7.yaml"
HEALTHY_YAML = REPO_ROOT / "PDAC" / "sim" / "resource" / "healthy_state.yaml"


@pytest.fixture(scope="session")
def scenario_cpp_trajectories(tmp_path_factory):
    """C++ scenario trajectories run from the evolved diagnosis state."""
    if not DUMP_BIN.exists():
        pytest.skip(f"dump_trajectories not built: {DUMP_BIN}")
    if not SCENARIO_YAML.exists():
        pytest.skip(f"scenario not found: {SCENARIO_YAML}")
    csv_path = tmp_path_factory.mktemp("cpp_scenario") / "cpp_trajectories.csv"
    result = subprocess.run(
        [str(DUMP_BIN), str(PARAM_XML), str(csv_path), "30", "0.1",
         "--scenario", str(SCENARIO_YAML),
         "--drug-metadata", str(DRUG_META),
         "--evolve-to-diagnosis", str(HEALTHY_YAML)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"C++ ODE failed:\n{result.stderr}"
    assert "[dose]" in result.stderr, (
        f"No boluses applied. stderr:\n{result.stderr}"
    )
    return csv_path


@pytest.fixture(scope="session")
def scenario_matlab_trajectories(run_matlab, tmp_path_factory):
    if not SCENARIO_YAML.exists():
        pytest.skip(f"scenario not found: {SCENARIO_YAML}")
    csv_path = tmp_path_factory.mktemp("matlab_scenario") / "matlab_trajectories.csv"
    matlab_cmd = (
        f"output_csv='{csv_path}'; pdac_build_dir='{PDAC_BUILD}'; "
        f"param_xml='{PARAM_XML}'; sbml_path='{SBML_PATH}'; "
        f"scenario_yaml='{SCENARIO_YAML}'; stop_time=30; "
        f"evolve_to_diagnosis_enabled=true; "
        f"run('{MATLAB_EXPORT_SCRIPT}')"
    )
    result = subprocess.run(
        ["matlab", "-batch", matlab_cmd],
        capture_output=True, text=True, timeout=600,
        cwd=str(PDAC_BUILD),
    )
    assert result.returncode == 0, f"MATLAB failed:\n{result.stderr}"
    return csv_path


def test_dose_applied_in_cpp(scenario_cpp_trajectories):
    """After day 7, V_C.aPD1 should be nonzero in C++ (post-nivo bolus)."""
    _, vc = _load_traj(scenario_cpp_trajectories)
    assert vc["V_C_aPD1"][69] == 0, (
        f"V_C.aPD1 should be 0 just before day 7, got {vc['V_C_aPD1'][69]:.3e}"
    )
    assert vc["V_C_aPD1"][70] > 0, (
        f"V_C.aPD1 should be nonzero at day 7 (post-nivo bolus), "
        f"got {vc['V_C_aPD1'][70]:.3e}"
    )


def test_scenario_trajectories_match(scenario_matlab_trajectories, scenario_cpp_trajectories):
    """MATLAB dose_schedule and C++ YAML-driven boluses must land at the
    same times with the same amounts → trajectories agree post-dose.

    Both sides run evolve_to_diagnosis (reading healthy_state.yaml) to
    reach the diagnosis state before applying the nivo bolus. This
    avoids the step-size-zero crash that previously required skipping
    this test.
    """
    from compare_trajectories import compare

    passed, report = compare(
        str(scenario_matlab_trajectories),
        str(scenario_cpp_trajectories),
        rtol=0.05,
        atol=1e-6,
    )
    print(report)
    for line in report.split("\n"):
        if "Failures:" in line:
            failures = int(line.split(":")[1].strip().split(" ")[0])
            total_line = next(l for l in report.split("\n") if "Comparisons:" in l)
            total = int(total_line.split(":")[1].strip())
            pass_rate = 1 - failures / total
            print(f"Pass rate: {pass_rate:.1%}")
            assert pass_rate > 0.95, (
                f"Scenario pass rate too low: {pass_rate:.1%}. "
                f"MATLAB and C++ disagree on dose timing/amount or downstream dynamics."
            )


def test_V_T_matches(matlab_trajectories, cpp_trajectories):
    """V_T (dynamic tumor volume) must match MATLAB pointwise.

    V_T is the clinical tumor-volume quantity (cancer + immune + stroma +
    collagen). It drives diagnosis-criterion decisions and many
    concentration expressions inside the ODE, so drift here indicates
    either a codegen bug in the V_T rule or a divergence in its species
    inputs. Tighter tolerance than the full trajectory match (which
    absorbs near-zero species noise).
    """
    import numpy as np
    t_m, vm = _load_traj(matlab_trajectories)
    t_c, vc = _load_traj(cpp_trajectories)
    assert "V_T" in vm, "MATLAB CSV has no V_T column"
    assert "V_T" in vc, "C++ CSV has no V_T column"
    v_m = vm["V_T"]
    v_c = vc["V_T"]
    denom = np.maximum(np.maximum(np.abs(v_m), np.abs(v_c)), 1e-9)
    rel_diff = np.abs(v_m - v_c) / denom
    i_worst = int(np.argmax(rel_diff))
    assert rel_diff.max() < 0.01, (
        f"V_T drift too large: max rel diff = {rel_diff.max():.2e} at "
        f"t={t_c[i_worst]:.1f}d (MATLAB={v_m[i_worst]:.3e}, "
        f"C++={v_c[i_worst]:.3e})"
    )


def test_trajectories_match(matlab_trajectories, cpp_trajectories):
    """Compare MATLAB and C++ trajectories within tolerance.

    rtol=5% / atol=1e-6 tolerates solver-config differences and near-zero
    noise; tightening requires aligning MATLAB and CVODE tolerances.
    """
    from compare_trajectories import compare

    passed, report = compare(
        str(matlab_trajectories),
        str(cpp_trajectories),
        rtol=0.05,
        atol=1e-6,
    )
    print(report)

    lines = report.split("\n")
    for line in lines:
        if "Matched species:" in line:
            matched = int(line.split(":")[1].strip().split("/")[0])
            assert matched > 100, f"Too few species matched: {matched}"
        if "Failures:" in line:
            failures = int(line.split(":")[1].strip().split(" ")[0])
            total_line = next(l for l in lines if "Comparisons:" in l)
            total = int(total_line.split(":")[1].strip())
            pass_rate = 1 - failures / total
            print(f"Pass rate: {pass_rate:.1%}")
            assert pass_rate > 0.95, f"Pass rate too low: {pass_rate:.1%}"


# --- Evolve-to-diagnosis stand-alone (C++ only) ----------------------------
# Verify the C++ evolve_to_diagnosis lands at the expected diameter and the
# resulting state is reasonable (V_T.C1 plausible, V_T volume matches target).

def test_evolve_to_diagnosis_reaches_target():
    """C++ evolve_to_diagnosis must reach the target V_T diameter (3.2 cm)
    within the biological-plausibility window (120–7300 d)."""
    if not DUMP_BIN.exists():
        pytest.skip(f"dump_trajectories not built: {DUMP_BIN}")
    if not HEALTHY_YAML.exists():
        pytest.skip(f"healthy_state.yaml not found: {HEALTHY_YAML}")

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        csv_path = f.name
    result = subprocess.run(
        [str(DUMP_BIN), str(PARAM_XML), csv_path, "1", "0.1",
         "--evolve-to-diagnosis", str(HEALTHY_YAML)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"evolve failed:\n{result.stderr}"
    assert "diagnosis at" in result.stderr, (
        f"evolve_to_diagnosis did not reach diagnosis.\nstderr:\n{result.stderr}"
    )
    # Parse t_diag and diameter from the summary line:
    #   evolve_to_diagnosis: t_diag=857 d, diameter=3.20022 cm
    import re
    m = re.search(
        r"evolve_to_diagnosis: t_diag=(\d+) d, diameter=([\d.]+) cm",
        result.stderr)
    assert m, f"Could not parse evolve summary from stderr:\n{result.stderr}"
    t_diag = int(m.group(1))
    diam = float(m.group(2))
    assert 120 <= t_diag <= 7300, (
        f"t_diag={t_diag} d is outside the plausibility window [120, 7300]"
    )
    assert 3.1 < diam < 3.5, f"Diameter {diam} cm is not near the 3.2 cm target"


# --- Evolved IC match (MATLAB vs C++) ----------------------------------------

def test_evolved_initial_conditions_match(
    scenario_matlab_trajectories, scenario_cpp_trajectories
):
    """Species values at t=0 (post-evolve, pre-dose) must agree between
    MATLAB and C++. Both sides read healthy_state.yaml, evolve to diagnosis,
    then start the scenario. If ICs diverge, trajectory comparison is
    uninformative — this test isolates that.
    """
    _, vm = _load_traj(scenario_matlab_trajectories)
    _, vc = _load_traj(scenario_cpp_trajectories)
    common = sorted(set(vm) & set(vc))
    bad = []
    for k in common:
        a, b = vm[k][0], vc[k][0]
        denom = max(abs(a), abs(b), 1e-12)
        rd = abs(a - b) / denom
        if rd > 0.05:
            bad.append((k, a, b, rd))
    bad.sort(key=lambda x: -x[3])
    assert not bad, (
        f"{len(bad)} species have mismatched evolved ICs at t=0 (rtol > 5%). "
        f"Worst:\n  " +
        "\n  ".join(f"{k}: MATLAB={a:.6e}, C++={b:.6e}, rdiff={rd:.2e}"
                    for k, a, b, rd in bad[:10])
    )


# --- Healthy-state sanity (C++ only, no MATLAB) ------------------------------

def test_healthy_state_cell_counts():
    """set_healthy_populations must produce cell counts matching the YAML
    densities × V_tumor_mm3 (from D_cell=17 µm, tumor_cells=1e6).

    Pure C++ test: parse the verbose stderr from --evolve-to-diagnosis.
    """
    if not DUMP_BIN.exists():
        pytest.skip(f"dump_trajectories not built: {DUMP_BIN}")
    if not HEALTHY_YAML.exists():
        pytest.skip(f"healthy_state.yaml not found: {HEALTHY_YAML}")

    import math, re, tempfile, yaml
    with open(HEALTHY_YAML) as f:
        hs = yaml.safe_load(f)

    D_cell_um = 17.0  # from param_all.xml
    vol_cell_um3 = (4.0 / 3.0) * math.pi * (D_cell_um / 2.0) ** 3
    V_tumor_mm3 = 1e6 * vol_cell_um3 / 1e12 * 1e3

    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        csv_path = f.name
    result = subprocess.run(
        [str(DUMP_BIN), str(PARAM_XML), csv_path, "1", "0.1",
         "--evolve-to-diagnosis", str(HEALTHY_YAML)],
        capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"evolve failed:\n{result.stderr}"

    # Parse "[healthy] CD8 = 128.622 cells" lines from stderr
    reported = {}
    for m in re.finditer(
        r"\[healthy\]\s+(\S+)\s+=\s+([\d.e+\-]+)\s+cells", result.stderr
    ):
        reported[m.group(1)] = float(m.group(2))

    dens = hs["cell_densities_per_mm3"]
    ratios = hs["ratios"]
    TAM_total = dens["TAM_total"] * V_tumor_mm3
    APC_total = dens["APC_total"] * V_tumor_mm3

    expected = {
        "C1":    1e6,
        "K":     1e6,
        "CD8":   dens["CD8"]  * V_tumor_mm3,
        "Treg":  dens["Treg"] * V_tumor_mm3,
        "Th":    dens["Th"]   * V_tumor_mm3,
        "MDSC":  dens["MDSC"] * V_tumor_mm3,
        "Mac_M1": ratios["m1_m2"] * TAM_total / (1 + ratios["m1_m2"]),
        "Mac_M2": TAM_total / (1 + ratios["m1_m2"]),
        "iCAF":  dens["iCAF"]  * V_tumor_mm3,
        "myCAF": dens["myCAF"] * V_tumor_mm3,
        "apCAF": dens["apCAF"] * V_tumor_mm3,
        "qPSC":  dens["qPSC"]  * V_tumor_mm3,
        "cDC1":  ratios["f_cDC1"] * (1 - ratios["apc_mat"]) * APC_total,
        "cDC2":  (1 - ratios["f_cDC1"]) * (1 - ratios["apc_mat"]) * APC_total,
        "mcDC1": ratios["f_cDC1"] * ratios["apc_mat"] * APC_total,
        "mcDC2": (1 - ratios["f_cDC1"]) * ratios["apc_mat"] * APC_total,
    }

    bad = []
    for name, exp in expected.items():
        got = reported.get(name)
        if got is None:
            bad.append(f"{name}: not in stderr output")
            continue
        rdiff = abs(got - exp) / max(abs(exp), 1e-12)
        # Tolerance 1e-4: stderr prints ~6 significant digits, so truncation
        # noise up to ~1e-6 is expected; 1e-4 catches real formula errors.
        if rdiff > 1e-4:
            bad.append(f"{name}: expected={exp:.6e}, got={got:.6e}, rdiff={rdiff:.2e}")
    assert not bad, (
        f"set_healthy_populations cell counts do not match YAML math:\n  "
        + "\n  ".join(bad)
    )


# --- Evolve rejection guards (C++ only) --------------------------------------

def test_evolve_rejects_when_max_days_exceeded():
    """evolve_to_diagnosis must reject (exit code 2) when the target diameter
    cannot be reached within the max_days cap.
    """
    if not DUMP_BIN.exists():
        pytest.skip(f"dump_trajectories not built: {DUMP_BIN}")

    import tempfile, yaml
    # Write a temp YAML with max_days=100. Normal evolution to 3.2 cm takes
    # ~857 d, so this will time out and reject.
    with open(HEALTHY_YAML) as f:
        hs = yaml.safe_load(f)
    hs["evolve"]["max_days"] = 100
    tmp_yaml = Path(tempfile.mktemp(suffix=".yaml"))
    with open(tmp_yaml, "w") as f:
        yaml.dump(hs, f)
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        csv_path = f.name
    result = subprocess.run(
        [str(DUMP_BIN), str(PARAM_XML), csv_path, "1", "0.1",
         "--evolve-to-diagnosis", str(tmp_yaml)],
        capture_output=True, text=True, timeout=120,
    )
    tmp_yaml.unlink(missing_ok=True)
    assert result.returncode == 2, (
        f"Expected exit code 2 for exceeded max_days, got {result.returncode}.\n"
        f"stderr:\n{result.stderr}"
    )
    assert "REJECTED" in result.stderr, (
        f"Expected REJECTED message in stderr:\n{result.stderr}"
    )


def test_evolve_rejects_too_fast_diagnosis():
    """evolve_to_diagnosis must reject when the tumor reaches the target
    diameter faster than min_days (biologically implausible growth).
    """
    if not DUMP_BIN.exists():
        pytest.skip(f"dump_trajectories not built: {DUMP_BIN}")

    import tempfile, yaml
    # Write a temp healthy_state.yaml with min_days cranked to 10000 d so the
    # normal ~857 d evolution triggers the guard.
    with open(HEALTHY_YAML) as f:
        hs = yaml.safe_load(f)
    hs["evolve"]["min_days"] = 10000
    tmp_yaml = Path(tempfile.mktemp(suffix=".yaml"))
    with open(tmp_yaml, "w") as f:
        yaml.dump(hs, f)
    with tempfile.NamedTemporaryFile(suffix=".csv") as f:
        csv_path = f.name
    result = subprocess.run(
        [str(DUMP_BIN), str(PARAM_XML), csv_path, "1", "0.1",
         "--evolve-to-diagnosis", str(tmp_yaml)],
        capture_output=True, text=True, timeout=600,
    )
    tmp_yaml.unlink(missing_ok=True)
    assert result.returncode == 2, (
        f"Expected exit code 2 for too-fast diagnosis, got {result.returncode}.\n"
        f"stderr:\n{result.stderr}"
    )
    assert "REJECTED" in result.stderr
