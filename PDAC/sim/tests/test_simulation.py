"""GPU integration tests — verify the simulation runs and produces valid output."""

import csv
from pathlib import Path

import pytest

from conftest import requires_gpu, requires_binary, run_simulation


STEPS = 10
SEED = 42


@requires_gpu
@requires_binary
class TestSimulationNoIO:
    """Simulation with no I/O output (pure compute)."""

    @pytest.fixture(autouse=True)
    def _run(self, binary, param_file, run_dir):
        self.run_dir = run_dir
        self.result = run_simulation(
            binary, param_file, run_dir, steps=STEPS, seed=SEED, grid_out=0
        )

    def test_exits_cleanly(self):
        assert self.result.returncode == 0, (
            f"Simulation failed:\nstdout: {self.result.stdout[-500:]}\n"
            f"stderr: {self.result.stderr[-500:]}"
        )

    def test_simulation_finished_message(self):
        assert "Simulation finished" in self.result.stdout

    def test_timing_csv_exists(self):
        csvs = list((self.run_dir / "outputs").glob("timing_*.csv"))
        assert len(csvs) == 1, f"Expected 1 timing CSV, found {len(csvs)}"

    def test_timing_csv_row_count(self):
        csvs = list((self.run_dir / "outputs").glob("timing_*.csv"))
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == STEPS, f"Expected {STEPS} rows, got {len(rows)}"

    def test_timing_csv_no_nan(self):
        csvs = list((self.run_dir / "outputs").glob("timing_*.csv"))
        text = csvs[0].read_text()
        assert "nan" not in text.lower(), "NaN found in timing CSV"

    def test_timing_csv_columns(self):
        csvs = list((self.run_dir / "outputs").glob("timing_*.csv"))
        with open(csvs[0]) as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames
        for expected in ["step", "total_ms", "pde_ms", "qsp_ms", "abm_ms"]:
            assert expected in cols, f"Missing column: {expected}"

    def test_step_times_positive(self):
        csvs = list((self.run_dir / "outputs").glob("timing_*.csv"))
        with open(csvs[0]) as f:
            for row in csv.DictReader(f):
                assert float(row["total_ms"]) > 0, f"Step {row['step']} has non-positive total_ms"

    def test_no_pde_output(self):
        pde_dir = self.run_dir / "outputs" / "pde"
        if pde_dir.exists():
            files = list(pde_dir.iterdir())
            assert len(files) == 0, "PDE output should not exist with grid_out=0"

    def test_no_abm_output(self):
        abm_dir = self.run_dir / "outputs" / "abm"
        if abm_dir.exists():
            files = list(abm_dir.iterdir())
            assert len(files) == 0, "ABM output should not exist with grid_out=0"


@requires_gpu
@requires_binary
class TestSimulationWithIO:
    """Simulation with full I/O every step."""

    @pytest.fixture(autouse=True)
    def _run(self, binary, param_file, run_dir):
        self.run_dir = run_dir
        self.result = run_simulation(
            binary, param_file, run_dir,
            steps=STEPS, seed=SEED, grid_out=3, interval=1
        )

    def test_exits_cleanly(self):
        assert self.result.returncode == 0, (
            f"Simulation failed:\nstdout: {self.result.stdout[-500:]}\n"
            f"stderr: {self.result.stderr[-500:]}"
        )

    def test_simulation_finished_message(self):
        assert "Simulation finished" in self.result.stdout

    def test_timing_csv_row_count(self):
        csvs = list((self.run_dir / "outputs").glob("timing_*.csv"))
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == STEPS

    def test_layer_timing_has_io_instrumentation(self):
        layer_csv = self.run_dir / "outputs" / "layer_timing.csv"
        assert layer_csv.exists(), "layer_timing.csv not found"
        text = layer_csv.read_text()
        assert "io_abm_collect" in text, "io_abm_collect not in layer timing"
        assert "io_pde_export" in text, "io_pde_export not in layer timing"

    def test_stats_csv_agent_counts_positive(self):
        csvs = list((self.run_dir / "outputs").glob("stats_*.csv"))
        if not csvs:
            pytest.skip("No stats CSV found")
        with open(csvs[0]) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) > 0, "Stats CSV is empty"
        # Check that at least some agent type has positive counts
        last_row = rows[-1]
        total = sum(float(last_row.get(k, 0)) for k in last_row if k != "step")
        assert total > 0, "All agent counts are zero in final step"


@requires_gpu
@requires_binary
class TestSimulationIOInterval:
    """Simulation with I/O at an interval."""

    @pytest.fixture(autouse=True)
    def _run(self, binary, param_file, run_dir):
        self.run_dir = run_dir
        self.interval = 5
        self.result = run_simulation(
            binary, param_file, run_dir,
            steps=STEPS, seed=SEED, grid_out=3, interval=self.interval
        )

    def test_exits_cleanly(self):
        assert self.result.returncode == 0

    def test_pde_file_count_matches_interval(self):
        pde_dir = self.run_dir / "outputs" / "pde"
        if not pde_dir.exists():
            pytest.skip("No PDE output directory")
        files = list(pde_dir.glob("*.pde.lz4")) + list(pde_dir.glob("*.npy"))
        expected = STEPS // self.interval + 1  # +1 for step 0
        assert len(files) == expected, (
            f"Expected {expected} PDE files, got {len(files)}"
        )
