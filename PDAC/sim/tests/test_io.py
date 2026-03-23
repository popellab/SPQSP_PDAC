"""Tests for I/O output format validation."""

import struct
from pathlib import Path

import pytest

from conftest import requires_gpu, requires_binary, run_simulation


STEPS = 5
SEED = 42


@requires_gpu
@requires_binary
class TestPDEOutput:
    """Verify PDE LZ4 output files."""

    @pytest.fixture(autouse=True)
    def _run(self, binary, param_file, run_dir):
        self.run_dir = run_dir
        self.result = run_simulation(
            binary, param_file, run_dir,
            steps=STEPS, seed=SEED, grid_out=2, interval=1
        )
        assert self.result.returncode == 0

    def test_pde_files_created(self):
        pde_dir = self.run_dir / "outputs" / "pde"
        files = sorted(pde_dir.glob("*.pde.lz4"))
        assert len(files) > 0, "No PDE LZ4 files found"

    def test_pde_lz4_header(self):
        pde_dir = self.run_dir / "outputs" / "pde"
        files = sorted(pde_dir.glob("*.pde.lz4"))
        for f in files:
            with open(f, "rb") as fp:
                magic = fp.read(4)
            assert magic == b"PDE1", f"{f.name}: bad magic {magic!r}"

    def test_pde_lz4_metadata(self):
        """Header contains valid grid dimensions and substrate count."""
        pde_dir = self.run_dir / "outputs" / "pde"
        f = sorted(pde_dir.glob("*.pde.lz4"))[0]
        with open(f, "rb") as fp:
            magic = fp.read(4)
            gx, gy, gz, ns, raw_sz, comp_sz = struct.unpack("<6i", fp.read(24))
        assert gx > 0 and gy > 0 and gz > 0, f"Invalid grid: {gx}x{gy}x{gz}"
        assert ns == 10, f"Expected 10 substrates, got {ns}"
        assert raw_sz == ns * gx * gy * gz * 4, "raw_sz doesn't match grid × substrates × float32"
        assert 0 < comp_sz <= raw_sz, f"Invalid compressed size: {comp_sz}"

    def test_pde_files_non_empty(self):
        pde_dir = self.run_dir / "outputs" / "pde"
        for f in pde_dir.glob("*.pde.lz4"):
            assert f.stat().st_size > 28, f"{f.name} is too small (header only?)"


@requires_gpu
@requires_binary
class TestABMOutput:
    """Verify ABM LZ4 output files."""

    @pytest.fixture(autouse=True)
    def _run(self, binary, param_file, run_dir):
        self.run_dir = run_dir
        self.result = run_simulation(
            binary, param_file, run_dir,
            steps=STEPS, seed=SEED, grid_out=1, interval=1
        )
        assert self.result.returncode == 0

    def test_abm_files_created(self):
        abm_dir = self.run_dir / "outputs" / "abm"
        files = sorted(abm_dir.glob("*.abm.lz4"))
        assert len(files) > 0, "No ABM LZ4 files found"

    def test_abm_lz4_header(self):
        abm_dir = self.run_dir / "outputs" / "abm"
        files = sorted(abm_dir.glob("*.abm.lz4"))
        for f in files:
            with open(f, "rb") as fp:
                magic = fp.read(4)
            assert magic == b"ABM1", f"{f.name}: bad magic {magic!r}"

    def test_abm_lz4_metadata(self):
        """Header contains valid agent count and column count."""
        abm_dir = self.run_dir / "outputs" / "abm"
        f = sorted(abm_dir.glob("*.abm.lz4"))[0]
        with open(f, "rb") as fp:
            magic = fp.read(4)
            n_agents, n_cols, raw_sz, comp_sz = struct.unpack("<4i", fp.read(16))
        assert n_agents > 0, f"No agents in output"
        assert n_cols == 8, f"Expected 8 columns, got {n_cols}"
        assert raw_sz == n_agents * n_cols * 4, "raw_sz doesn't match agents × cols × int32"
        assert 0 < comp_sz <= raw_sz, f"Invalid compressed size: {comp_sz}"

    def test_abm_can_decompress(self):
        """Verify the compressed data can be decompressed (if lz4 is installed)."""
        try:
            import lz4.block
        except ImportError:
            pytest.skip("lz4 Python package not installed")

        abm_dir = self.run_dir / "outputs" / "abm"
        f = sorted(abm_dir.glob("*.abm.lz4"))[0]
        with open(f, "rb") as fp:
            fp.read(4)  # magic
            n_agents, n_cols, raw_sz, comp_sz = struct.unpack("<4i", fp.read(16))
            compressed = fp.read(comp_sz)

        decompressed = lz4.block.decompress(compressed, uncompressed_size=raw_sz)
        assert len(decompressed) == raw_sz
