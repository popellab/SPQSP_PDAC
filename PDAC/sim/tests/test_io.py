"""Tests for I/O output format validation — uses shared simulation runs."""

import struct
from pathlib import Path

import pytest


# ============================================================================
# PDE output format
# ============================================================================

class TestPDEOutput:
    def test_pde_lz4_header(self, run_full_io):
        _, run_dir = run_full_io
        for f in sorted((run_dir / "outputs" / "pde").glob("*.pde.lz4")):
            with open(f, "rb") as fp:
                magic = fp.read(4)
            assert magic == b"PDE1", f"{f.name}: bad magic {magic!r}"

    def test_pde_lz4_metadata(self, run_full_io):
        _, run_dir = run_full_io
        f = sorted((run_dir / "outputs" / "pde").glob("*.pde.lz4"))[0]
        with open(f, "rb") as fp:
            magic = fp.read(4)
            gx, gy, gz, ns, raw_sz, comp_sz = struct.unpack("<6i", fp.read(24))
        assert gx > 0 and gy > 0 and gz > 0, f"Invalid grid: {gx}x{gy}x{gz}"
        assert ns == 10, f"Expected 10 substrates, got {ns}"
        assert raw_sz == ns * gx * gy * gz * 4
        assert 0 < comp_sz <= raw_sz

    def test_pde_files_non_empty(self, run_full_io):
        _, run_dir = run_full_io
        for f in (run_dir / "outputs" / "pde").glob("*.pde.lz4"):
            assert f.stat().st_size > 28, f"{f.name} too small"

    def test_pde_can_decompress(self, run_full_io):
        try:
            import lz4.block
        except ImportError:
            pytest.skip("lz4 Python package not installed")

        _, run_dir = run_full_io
        f = sorted((run_dir / "outputs" / "pde").glob("*.pde.lz4"))[0]
        with open(f, "rb") as fp:
            fp.read(4)  # magic
            gx, gy, gz, ns, raw_sz, comp_sz = struct.unpack("<6i", fp.read(24))
            compressed = fp.read(comp_sz)

        decompressed = lz4.block.decompress(compressed, uncompressed_size=raw_sz)
        assert len(decompressed) == raw_sz


# ============================================================================
# ABM output format
# ============================================================================

class TestABMOutput:
    def test_abm_lz4_header(self, run_full_io):
        _, run_dir = run_full_io
        for f in sorted((run_dir / "outputs" / "abm").glob("*.abm.lz4")):
            with open(f, "rb") as fp:
                magic = fp.read(4)
            assert magic == b"ABM1", f"{f.name}: bad magic {magic!r}"

    def test_abm_lz4_metadata(self, run_full_io):
        _, run_dir = run_full_io
        f = sorted((run_dir / "outputs" / "abm").glob("*.abm.lz4"))[0]
        with open(f, "rb") as fp:
            magic = fp.read(4)
            n_agents, n_cols, raw_sz, comp_sz = struct.unpack("<4i", fp.read(16))
        assert n_agents > 0
        assert n_cols == 8
        assert raw_sz == n_agents * n_cols * 4
        assert 0 < comp_sz <= raw_sz

    def test_abm_can_decompress(self, run_full_io):
        try:
            import lz4.block
        except ImportError:
            pytest.skip("lz4 Python package not installed")

        _, run_dir = run_full_io
        f = sorted((run_dir / "outputs" / "abm").glob("*.abm.lz4"))[0]
        with open(f, "rb") as fp:
            fp.read(4)  # magic
            n_agents, n_cols, raw_sz, comp_sz = struct.unpack("<4i", fp.read(16))
            compressed = fp.read(comp_sz)

        decompressed = lz4.block.decompress(compressed, uncompressed_size=raw_sz)
        assert len(decompressed) == raw_sz

    def test_abm_agent_types_valid(self, run_full_io):
        """All agent type IDs should be 0-6."""
        try:
            import lz4.block
            import struct as s
        except ImportError:
            pytest.skip("lz4 not installed")

        _, run_dir = run_full_io
        f = sorted((run_dir / "outputs" / "abm").glob("*.abm.lz4"))[0]
        with open(f, "rb") as fp:
            fp.read(4)
            n_agents, n_cols, raw_sz, comp_sz = s.unpack("<4i", fp.read(16))
            data = lz4.block.decompress(fp.read(comp_sz), uncompressed_size=raw_sz)

        # Parse int32 array, check first column (type_id)
        ints = s.unpack(f"<{n_agents * n_cols}i", data)
        for i in range(n_agents):
            type_id = ints[i * n_cols]
            assert 0 <= type_id <= 6, f"Agent {i}: invalid type_id {type_id}"
