"""Tests that don't require a GPU — just verify the build artifacts."""

from conftest import BINARY, SIM_DIR


def test_binary_exists():
    """The simulation binary should exist after a successful build."""
    assert BINARY.exists(), f"Binary not found at {BINARY}"
    assert BINARY.stat().st_size > 0, "Binary is empty"


def test_param_file_exists():
    """Default test parameter file should exist."""
    param = SIM_DIR / "resource" / "param_all_test.xml"
    assert param.exists(), f"Param file not found at {param}"


def test_lz4_vendored():
    """Vendored LZ4 source should be present."""
    assert (SIM_DIR / "third_party" / "lz4" / "lz4.h").exists()
    assert (SIM_DIR / "third_party" / "lz4" / "lz4.c").exists()
