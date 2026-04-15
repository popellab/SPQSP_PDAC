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

TESTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TESTS_DIR.parent.parent.parent
PDAC_BUILD = REPO_ROOT.parent / "pdac-build"
ODE_BUILD = REPO_ROOT / "PDAC" / "qsp" / "sim" / "build"
PARAM_XML = REPO_ROOT / "PDAC" / "sim" / "resource" / "param_all.xml"
# Both MATLAB and C++ simulate from the same XML so parameter divergence
# isn't confounded with solver divergence. Override via PDAC_PARAM_XML env var.
import os as _os
PARAM_XML = Path(_os.environ.get("PDAC_PARAM_XML", str(PARAM_XML)))
# MATLAB loads the model via sbmlimport from the same SBML the C++ codegen
# reads, so both sides simulate the exact same ODE structure.
SBML_PATH = REPO_ROOT / "PDAC" / "qsp" / "PDAC_model.sbml"
MATLAB_EXPORT_SCRIPT = TESTS_DIR / "export_matlab_trajectories.m"

DUMP_BIN = ODE_BUILD / "qsp_sim"

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
