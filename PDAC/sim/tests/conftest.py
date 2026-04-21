"""Shared fixtures for PDAC simulation tests."""

import csv
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import pytest

SIM_DIR = Path(__file__).resolve().parent.parent
BINARY = SIM_DIR / "build" / "bin" / "pdac"
DEFAULT_PARAMS = SIM_DIR / "resource" / "param_all_test.xml"

STEPS = 10
SEED = 42


def pytest_addoption(parser):
    parser.addoption(
        "--run-matlab", action="store_true", default=False,
        help="Run MATLAB-dependent ODE validation tests (local only)",
    )


def has_gpu():
    """Check if a CUDA GPU is available."""
    try:
        subprocess.run(
            ["nvidia-smi"], capture_output=True, check=True, timeout=5
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return False


requires_gpu = pytest.mark.skipif(not has_gpu(), reason="No GPU available")
requires_binary = pytest.mark.skipif(
    not BINARY.exists(), reason=f"Binary not found at {BINARY}"
)


def run_simulation(binary, param_file, run_dir, steps=STEPS, seed=SEED,
                   grid_out=0, interval=1, timeout=120):
    """Run the simulation and return (result, run_dir)."""
    result = subprocess.run(
        [
            str(binary),
            "-p", str(param_file),
            "-s", str(steps),
            "--seed", str(seed),
            "-G", str(grid_out),
            "-oi", str(interval),
        ],
        cwd=str(run_dir),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return result


# ============================================================================
# Session-scoped fixtures: run sim once, share across all tests
# ============================================================================

@pytest.fixture(scope="session")
def _check_gpu_and_binary():
    """Gate for all GPU tests."""
    if not has_gpu():
        pytest.skip("No GPU available")
    if not BINARY.exists():
        pytest.skip(f"Binary not found at {BINARY}")


@pytest.fixture(scope="session")
def run_no_io(_check_gpu_and_binary, tmp_path_factory):
    """Simulation with no I/O output."""
    run_dir = tmp_path_factory.mktemp("no_io")
    result = run_simulation(BINARY, DEFAULT_PARAMS, run_dir, grid_out=0)
    assert result.returncode == 0, f"Sim failed:\n{result.stderr[-500:]}"
    return result, run_dir


@pytest.fixture(scope="session")
def run_full_io(_check_gpu_and_binary, tmp_path_factory):
    """Simulation with full I/O every step."""
    run_dir = tmp_path_factory.mktemp("full_io")
    result = run_simulation(BINARY, DEFAULT_PARAMS, run_dir, grid_out=3, interval=1)
    assert result.returncode == 0, f"Sim failed:\n{result.stderr[-500:]}"
    return result, run_dir


@pytest.fixture(scope="session")
def run_abm_only(_check_gpu_and_binary, tmp_path_factory):
    """Simulation with ABM-only output."""
    run_dir = tmp_path_factory.mktemp("abm_only")
    result = run_simulation(BINARY, DEFAULT_PARAMS, run_dir, grid_out=1, interval=1)
    assert result.returncode == 0, f"Sim failed:\n{result.stderr[-500:]}"
    return result, run_dir


@pytest.fixture(scope="session")
def run_no_io_repeat(_check_gpu_and_binary, tmp_path_factory):
    """Second no-I/O run with same seed for determinism check."""
    run_dir = tmp_path_factory.mktemp("no_io_repeat")
    result = run_simulation(BINARY, DEFAULT_PARAMS, run_dir, grid_out=0)
    assert result.returncode == 0, f"Sim failed:\n{result.stderr[-500:]}"
    return result, run_dir


@pytest.fixture(scope="session")
def run_interval_io(_check_gpu_and_binary, tmp_path_factory):
    """Simulation with I/O every 5 steps."""
    run_dir = tmp_path_factory.mktemp("interval_io")
    result = run_simulation(BINARY, DEFAULT_PARAMS, run_dir, grid_out=3, interval=5)
    assert result.returncode == 0, f"Sim failed:\n{result.stderr[-500:]}"
    return result, run_dir


@pytest.fixture(scope="session")
def run_single_step(_check_gpu_and_binary, tmp_path_factory):
    """Edge case: single simulation step."""
    run_dir = tmp_path_factory.mktemp("single_step")
    result = run_simulation(BINARY, DEFAULT_PARAMS, run_dir, steps=1, grid_out=3, interval=1)
    assert result.returncode == 0, f"Sim failed:\n{result.stderr[-500:]}"
    return result, run_dir


# ============================================================================
# Helpers for reading outputs
# ============================================================================

def find_timing_csv(run_dir):
    """Find timing CSV (timing.csv or timing_*.csv)."""
    outputs = Path(run_dir) / "outputs"
    matches = sorted(outputs.glob("timing*.csv"))
    return matches[0] if matches else None


def read_timing_rows(run_dir):
    """Read timing CSV and return list of dicts."""
    path = find_timing_csv(run_dir)
    if not path:
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def find_stats_csv(run_dir):
    """Find stats CSV."""
    outputs = Path(run_dir) / "outputs"
    matches = sorted(outputs.glob("stats_*.csv"))
    return matches[0] if matches else None


def read_stats_rows(run_dir):
    """Read stats CSV and return list of dicts."""
    path = find_stats_csv(run_dir)
    if not path:
        return []
    with open(path) as f:
        return list(csv.DictReader(f))
