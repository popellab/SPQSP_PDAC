"""Compare MATLAB SimBiology and C++ CVODE ODE trajectories.

Requires MATLAB + a local pdac-build/ sibling directory with startup.m and
immune_oncology_model_PDAC.m. The dump_trajectories binary must be built
from PDAC/sim/tests/ode_compile/ (see its CMakeLists.txt).

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
ODE_BUILD = TESTS_DIR / "ode_compile" / "build"
PARAM_XML = REPO_ROOT / "PDAC" / "sim" / "resource" / "param_all.xml"
MATLAB_EXPORT_SCRIPT = TESTS_DIR / "export_matlab_trajectories.m"

sys.path.insert(0, str(TESTS_DIR))


@pytest.fixture
def run_matlab(request):
    if not request.config.getoption("--run-matlab"):
        pytest.skip("Requires --run-matlab flag (local only)")


@pytest.fixture
def matlab_trajectories(run_matlab, tmp_path):
    csv_path = tmp_path / "matlab_trajectories.csv"
    matlab_cmd = (
        f"output_csv='{csv_path}'; pdac_build_dir='{PDAC_BUILD}'; "
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


@pytest.fixture
def cpp_trajectories(tmp_path):
    dump_bin = ODE_BUILD / "dump_trajectories"
    if not dump_bin.exists():
        pytest.skip(f"dump_trajectories not built: {dump_bin}")
    if not PARAM_XML.exists():
        pytest.skip(f"param_all.xml not found: {PARAM_XML}")

    csv_path = tmp_path / "cpp_trajectories.csv"
    result = subprocess.run(
        [str(dump_bin), str(PARAM_XML), str(csv_path), "365", "0.1"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"C++ ODE failed:\n{result.stderr}"
    assert csv_path.exists()
    return csv_path


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
