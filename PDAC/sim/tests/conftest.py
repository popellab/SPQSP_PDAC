"""Shared fixtures for PDAC simulation tests."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

SIM_DIR = Path(__file__).resolve().parent.parent
BINARY = SIM_DIR / "build" / "bin" / "pdac"
DEFAULT_PARAMS = SIM_DIR / "resource" / "param_all_test.xml"


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


@pytest.fixture
def binary():
    """Path to the pdac binary. Skips if not built."""
    if not BINARY.exists():
        pytest.skip(f"Binary not found at {BINARY}")
    return BINARY


@pytest.fixture
def param_file():
    """Path to the default test parameter file."""
    if not DEFAULT_PARAMS.exists():
        pytest.skip(f"Param file not found at {DEFAULT_PARAMS}")
    return DEFAULT_PARAMS


@pytest.fixture
def run_dir():
    """Temporary directory for a simulation run. Cleaned up after test."""
    d = tempfile.mkdtemp(prefix="pdac_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


def run_simulation(binary, param_file, run_dir, steps=10, seed=42,
                   grid_out=0, interval=1, timeout=120):
    """Run the simulation and return (exit_code, stdout_text, run_dir).

    grid_out: 0=no IO, 1=ABM only, 2=PDE+ECM only, 3=all
    """
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
