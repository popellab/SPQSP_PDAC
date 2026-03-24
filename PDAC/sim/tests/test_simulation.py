"""GPU integration tests — verify simulation runs and produces valid output."""

import csv
from pathlib import Path

import pytest

from conftest import STEPS, read_timing_rows, read_stats_rows, find_timing_csv


# ============================================================================
# Basic simulation (no I/O)
# ============================================================================

class TestSimulationNoIO:
    def test_finished_message(self, run_no_io):
        result, _ = run_no_io
        assert "Simulation finished" in result.stdout

    def test_timing_csv_exists(self, run_no_io):
        _, run_dir = run_no_io
        assert find_timing_csv(run_dir) is not None

    def test_timing_csv_row_count(self, run_no_io):
        _, run_dir = run_no_io
        rows = read_timing_rows(run_dir)
        assert len(rows) == STEPS

    def test_timing_csv_columns(self, run_no_io):
        _, run_dir = run_no_io
        rows = read_timing_rows(run_dir)
        for col in ["step", "total_ms", "pde_ms", "qsp_ms", "abm_ms"]:
            assert col in rows[0], f"Missing column: {col}"

    def test_timing_no_nan(self, run_no_io):
        _, run_dir = run_no_io
        text = find_timing_csv(run_dir).read_text()
        assert "nan" not in text.lower(), "NaN found in timing CSV"

    def test_step_times_positive(self, run_no_io):
        _, run_dir = run_no_io
        for row in read_timing_rows(run_dir):
            assert float(row["total_ms"]) > 0

    def test_no_output_files(self, run_no_io):
        _, run_dir = run_no_io
        pde_dir = run_dir / "outputs" / "pde"
        abm_dir = run_dir / "outputs" / "abm"
        pde_files = list(pde_dir.glob("*")) if pde_dir.exists() else []
        abm_files = list(abm_dir.glob("*")) if abm_dir.exists() else []
        assert len(pde_files) == 0, "PDE output should not exist with grid_out=0"
        assert len(abm_files) == 0, "ABM output should not exist with grid_out=0"


# ============================================================================
# Full I/O simulation
# ============================================================================

class TestSimulationWithIO:
    def test_finished_message(self, run_full_io):
        result, _ = run_full_io
        assert "Simulation finished" in result.stdout

    def test_timing_csv_row_count(self, run_full_io):
        _, run_dir = run_full_io
        rows = read_timing_rows(run_dir)
        assert len(rows) == STEPS

    def test_layer_timing_has_io_instrumentation(self, run_full_io):
        _, run_dir = run_full_io
        layer_csv = run_dir / "outputs" / "layer_timing.csv"
        assert layer_csv.exists()
        text = layer_csv.read_text()
        assert "io_abm_collect" in text
        assert "io_pde_export" in text

    def test_stats_csv_agent_counts_positive(self, run_full_io):
        _, run_dir = run_full_io
        rows = read_stats_rows(run_dir)
        if not rows:
            pytest.skip("No stats CSV found")
        last_row = rows[-1]
        total = sum(float(last_row.get(k, 0)) for k in last_row if k != "step")
        assert total > 0, "All agent counts are zero in final step"

    def test_pde_files_created(self, run_full_io):
        _, run_dir = run_full_io
        pde_dir = run_dir / "outputs" / "pde"
        files = list(pde_dir.glob("*.pde.lz4"))
        assert len(files) > 0

    def test_abm_files_created(self, run_full_io):
        _, run_dir = run_full_io
        abm_dir = run_dir / "outputs" / "abm"
        files = list(abm_dir.glob("*.abm.lz4"))
        assert len(files) > 0


# ============================================================================
# ABM-only output
# ============================================================================

class TestABMOnly:
    def test_exits_cleanly(self, run_abm_only):
        result, _ = run_abm_only
        assert result.returncode == 0

    def test_abm_files_created(self, run_abm_only):
        _, run_dir = run_abm_only
        abm_dir = run_dir / "outputs" / "abm"
        files = list(abm_dir.glob("*.abm.lz4"))
        assert len(files) > 0

    def test_no_pde_files(self, run_abm_only):
        _, run_dir = run_abm_only
        pde_dir = run_dir / "outputs" / "pde"
        pde_files = list(pde_dir.glob("*")) if pde_dir.exists() else []
        assert len(pde_files) == 0


# ============================================================================
# Interval I/O
# ============================================================================

class TestIntervalIO:
    def test_exits_cleanly(self, run_interval_io):
        result, _ = run_interval_io
        assert result.returncode == 0

    def test_pde_file_count_matches_interval(self, run_interval_io):
        _, run_dir = run_interval_io
        pde_dir = run_dir / "outputs" / "pde"
        files = list(pde_dir.glob("*.pde.lz4")) + list(pde_dir.glob("*.npy"))
        expected = STEPS // 5 + 1  # +1 for step 0
        assert len(files) == expected, f"Expected {expected}, got {len(files)}"


# ============================================================================
# Determinism
# ============================================================================

class TestDeterminism:
    def test_same_seed_same_timing(self, run_no_io, run_no_io_repeat):
        _, run_dir1 = run_no_io
        _, run_dir2 = run_no_io_repeat
        rows1 = read_timing_rows(run_dir1)
        rows2 = read_timing_rows(run_dir2)
        assert len(rows1) == len(rows2), "Different number of steps"

    def test_same_seed_similar_agent_counts(self, run_no_io, run_no_io_repeat):
        """Same seed should produce similar results. GPU atomic scheduling
        is non-deterministic, so we allow up to 10% difference."""
        _, run_dir1 = run_no_io
        _, run_dir2 = run_no_io_repeat
        stats1 = read_stats_rows(run_dir1)
        stats2 = read_stats_rows(run_dir2)
        if not stats1 or not stats2:
            pytest.skip("No stats CSV found")
        for key in stats1[-1]:
            if key == "step":
                continue
            v1, v2 = float(stats1[-1][key]), float(stats2[-1][key])
            # Skip small counts — percentage is meaningless for 0 vs 1
            if max(abs(v1), abs(v2)) < 10:
                continue
            denom = max(abs(v1), abs(v2))
            pct_diff = abs(v1 - v2) / denom
            assert pct_diff < 0.10, (
                f"{key} differs by {pct_diff:.1%}: {v1} vs {v2}"
            )


# ============================================================================
# Numerical sanity
# ============================================================================

class TestNumericalSanity:
    def test_timing_values_finite(self, run_full_io):
        _, run_dir = run_full_io
        for row in read_timing_rows(run_dir):
            for key in ["total_ms", "pde_ms", "qsp_ms", "abm_ms"]:
                val = float(row[key])
                assert val == val, f"NaN in {key} at step {row['step']}"  # NaN != NaN
                assert val != float('inf'), f"Inf in {key} at step {row['step']}"

    def test_step_times_reasonable(self, run_full_io):
        """No single step should take more than 60 seconds at test grid size."""
        _, run_dir = run_full_io
        for row in read_timing_rows(run_dir):
            assert float(row["total_ms"]) < 60000, (
                f"Step {row['step']} took {row['total_ms']}ms"
            )

    def test_agent_counts_non_negative(self, run_full_io):
        _, run_dir = run_full_io
        rows = read_stats_rows(run_dir)
        if not rows:
            pytest.skip("No stats CSV")
        for row in rows:
            for key, val in row.items():
                if key == "step":
                    continue
                assert float(val) >= 0, f"Negative value: {key}={val} at step {row['step']}"


# ============================================================================
# Edge cases
# ============================================================================

class TestEdgeCases:
    def test_single_step(self, run_single_step):
        result, run_dir = run_single_step
        assert result.returncode == 0
        rows = read_timing_rows(run_dir)
        assert len(rows) == 1
