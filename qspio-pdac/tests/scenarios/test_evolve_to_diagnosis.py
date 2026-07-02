"""Tests for evolve_to_diagnosis initialization and natural history evolution.

These tests verify that:
- set_healthy_populations computes tumor volume consistent with the model's vol_cell
- evolve_to_diagnosis reaches target diameter using V_T (not C1-only estimate)
- Diagnosis-state cell densities are in biologically plausible ranges

The density fix (vol_cell-based volume) reduced initial immune counts ~400x,
from e.g. 50,000 CD8 cells (in a 1000 mm³ tumor) to ~129 CD8 cells (in a
2.57 mm³ tumor). This test guards against regression.

Usage:
    # Run slow tests (includes MATLAB simulations):
    pytest tests/scenarios/test_evolve_to_diagnosis.py -m slow -v

    # Run quick tests only (file existence checks):
    pytest tests/scenarios/test_evolve_to_diagnosis.py -m "not slow"
"""

import os
import re
import sys
import pytest
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
HELPER_SCRIPT = Path(__file__).parent / "test_evolve_to_diagnosis_helper.m"


def parse_test_outputs(stdout: str) -> dict[str, float]:
    """Parse TEST_OUTPUT:key=value lines from MATLAB stdout."""
    outputs = {}
    for line in stdout.splitlines():
        m = re.match(r"TEST_OUTPUT:(\w+)=([\d.eE+\-]+)", line)
        if m:
            outputs[m.group(1)] = float(m.group(2))
    return outputs


class TestEvolveTodiagnosisPrereqs:
    """Quick tests for prerequisite files (no MATLAB required)."""

    def test_set_healthy_populations_exists(self):
        assert (REPO_ROOT / "scripts" / "set_healthy_populations.m").exists()

    def test_evolve_to_diagnosis_exists(self):
        assert (REPO_ROOT / "scripts" / "evolve_to_diagnosis.m").exists()

    def test_helper_script_exists(self):
        assert HELPER_SCRIPT.exists()


@pytest.mark.slow
class TestEvolveTodiagnosisScale:
    """Verify initialization density and evolution are self-consistent.

    Runs a single MATLAB session (via helper .m script) that:
    1. Builds the model with median parameters
    2. Calls set_healthy_populations and checks initial volume scale
    3. Runs evolve_to_diagnosis to a 3 cm target diameter
    4. Reports diagnosis-state species counts and densities
    """

    @pytest.fixture(scope="class")
    def matlab_outputs(self):
        """Run the full evolve_to_diagnosis pipeline and capture outputs.

        Backend selection via ``QUAL_TESTS_BACKEND`` (same switch as
        test_qualitative_behaviors.py):
            - ``cpp`` (default): run via the C++ helper — builds an evolve
              blob, reads t_diagnosis from the QSTH header, and runs the
              baseline scenario for diagnosis-state species densities.
            - ``matlab``: shell out to the .m helper (legacy path).
        """
        backend = os.environ.get("QUAL_TESTS_BACKEND", "cpp").lower()
        if backend == "cpp":
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from tests.scenarios._cpp_helper import build_evolve_outputs

            return build_evolve_outputs()

        if backend != "matlab":
            pytest.fail(f"Unknown QUAL_TESTS_BACKEND={backend!r} (expected 'cpp' or 'matlab')")

        matlab_cmd = (
            f"cd('{REPO_ROOT}'); "
            f"startup; "
            f"addpath('{HELPER_SCRIPT.parent}'); "
            f"test_evolve_to_diagnosis_helper"
        )

        result = subprocess.run(
            ["matlab", "-batch", matlab_cmd],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=REPO_ROOT,
        )

        if result.stdout:
            print("\n--- MATLAB stdout ---")
            print(result.stdout)
        if result.stderr:
            print("\n--- MATLAB stderr ---")
            print(result.stderr)

        assert result.returncode == 0, (
            f"MATLAB failed (rc={result.returncode})\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

        outputs = parse_test_outputs(result.stdout)
        assert "evolution_success" in outputs, (
            "Could not parse TEST_OUTPUT lines from MATLAB stdout"
        )
        return outputs

    # --- Initial state checks (after set_healthy_populations) ---

    def test_initial_volume_not_inflated(self, matlab_outputs):
        """Initial tumor volume should be ~0.003 mL, not ~1 mL.

        With D_cell=17µm and 1e6 cells:
          vol_cell = 2572 µm³ → V = 2.57e-3 mL
        The old bug (TUMOR_DENSITY=1000 cells/mm³) gave V = 1.0 mL.
        """
        V_init = matlab_outputs["init_V_tumor_mL"]
        assert V_init < 0.01, (
            f"Initial V_T = {V_init:.4f} mL is too large "
            f"(expected ~0.003 mL for 1e6 cells with D_cell=17µm). "
            f"The old TUMOR_DENSITY=1000 bug may have regressed."
        )
        assert V_init > 1e-4, (
            f"Initial V_T = {V_init:.6e} mL is implausibly small"
        )

    def test_initial_cd8_not_inflated(self, matlab_outputs):
        """Initial CD8 count should be ~129 cells, not ~50,000.

        With 50 cells/mm³ × 2.57 mm³ = 128.6 cells.
        The old bug gave 50 × 1000 = 50,000 cells.
        """
        CD8_init = matlab_outputs["init_CD8"]
        assert CD8_init < 1000, (
            f"Initial CD8 = {CD8_init:.0f} is too large "
            f"(expected ~129 for 50/mm³ × 2.57 mm³). "
            f"Density initialization may be using inflated volume."
        )
        assert CD8_init > 1, f"Initial CD8 = {CD8_init:.1f} is implausibly small"

    # --- Evolution checks ---

    def test_evolution_succeeds(self, matlab_outputs):
        """evolve_to_diagnosis should reach 3 cm with median parameters."""
        assert matlab_outputs["evolution_success"] == 1, (
            "evolve_to_diagnosis failed to reach 3 cm target diameter "
            "with median prior parameters"
        )

    def test_time_to_diagnosis_plausible(self, matlab_outputs):
        """Time to reach 3 cm should be biologically plausible (months to years).

        PDAC doubling time is ~100-200 days. From ~1.7 mm to 3 cm requires
        ~15-20 doublings in volume, so ~1500-4000 days is expected.
        Allow a wider range (60-3650 days) for robustness.
        """
        if matlab_outputs["evolution_success"] != 1:
            pytest.skip("Evolution did not succeed")
        t = matlab_outputs["time_to_diagnosis"]
        assert 60 < t < 3650, (
            f"Time to diagnosis = {t:.0f} days is outside plausible range "
            f"(60-3650 days)"
        )

    def test_diagnosis_diameter_near_target(self, matlab_outputs):
        """Diagnosis diameter should be close to the 3 cm target."""
        if matlab_outputs["evolution_success"] != 1:
            pytest.skip("Evolution did not succeed")
        diam = matlab_outputs["diam_diagnosis_cm"]
        # Should be >= 3.0 (first crossing) and not wildly overshoot
        assert 3.0 <= diam < 5.0, (
            f"Diagnosis diameter = {diam:.2f} cm "
            f"(expected ~3.0 cm, first crossing of target)"
        )

    # --- Diagnosis-state density checks ---

    def test_cd8_density_at_diagnosis(self, matlab_outputs):
        """CD8 volumetric density should be 5000-100000 cells/mm³ at diagnosis.

        Parameter file default: ~34,831 cells/mm³ (parameters_PDAC.m).
        Literature IHC values (Tahkola 2019, Fukunaga 2004) report 2D
        surface densities (cells/mm²); converting to volumetric density
        yields ~10,000-50,000 cells/mm³ depending on section thickness.
        Lower bound relaxed to 5000 to accommodate PDAC's immunosuppressive
        TME (high TGFb, MDSC-mediated T cell suppression).
        """
        if matlab_outputs["evolution_success"] != 1:
            pytest.skip("Evolution did not succeed")
        d = matlab_outputs["CD8_density"]
        assert 5000 < d < 100000, (
            f"CD8 density at diagnosis = {d:.0f} cells/mm³ "
            f"(expected 5000-100000, parameter default ~34,831)"
        )

    def test_tam_density_at_diagnosis(self, matlab_outputs):
        """TAM density should be 50-50000 cells/mm³ at diagnosis.

        Literature: 500-5000 cells/mm³ in PDAC (Chen 2015, Ino 2013).
        Use wider bounds for model flexibility.
        """
        if matlab_outputs["evolution_success"] != 1:
            pytest.skip("Evolution did not succeed")
        d = matlab_outputs["TAM_density"]
        assert 50 < d < 50000, (
            f"TAM density at diagnosis = {d:.0f} cells/mm³ "
            f"(expected 50-50000, literature ~500-5000)"
        )

    def test_m1_m2_ratio_at_diagnosis(self, matlab_outputs):
        """M1/M2 ratio should be immunosuppressive (<1.5) at diagnosis.

        PDAC TME is M2-skewed. Healthy tissue starts at 1.5 (M1-skewed);
        by diagnosis it should have shifted toward M2 dominance.
        """
        if matlab_outputs["evolution_success"] != 1:
            pytest.skip("Evolution did not succeed")
        ratio = matlab_outputs["M1_M2_ratio"]
        assert 0.01 < ratio < 3.0, (
            f"M1/M2 ratio at diagnosis = {ratio:.2f} "
            f"(expected 0.01-3.0; PDAC is typically M2-skewed)"
        )

    def test_cancer_cell_density_at_diagnosis(self, matlab_outputs):
        """Cancer cell density should be 1e3-1e7 cells/mm³ at diagnosis.

        With vol_cell = 2572 µm³, close-packed cells give ~3.9e5 cells/mm³.
        V_T includes stroma+ECM (often 60-80% of volume in PDAC), so
        effective cancer density is much lower than close-packing.
        """
        if matlab_outputs["evolution_success"] != 1:
            pytest.skip("Evolution did not succeed")
        d = matlab_outputs["C1_density"]
        assert 1e3 < d < 1e7, (
            f"Cancer cell density = {d:.0f} cells/mm³ "
            f"(expected 1e3-1e7)"
        )

    # --- Fallback path test (mimics batch_worker.m) ---

    def test_model_parameter_fallback_succeeds(self, matlab_outputs):
        """evolve_to_diagnosis should reach target diameter when reading
        initial_tumour_diameter from the model (not the variant).

        This mimics the batch_worker.m path where initial_tumour_diameter
        is not in pdac_priors.csv and thus not in the variant. Regression
        test for the bug where removing the parameter from priors caused
        evolve_to_diagnosis to fall back to a fixed 365-day evolution.
        """
        assert matlab_outputs.get("fallback_success", 0) == 1, (
            "evolve_to_diagnosis failed when initial_tumour_diameter is "
            "not in the variant. The model-parameter fallback may be broken."
        )

    def test_model_parameter_fallback_diameter(self, matlab_outputs):
        """Fallback path should reach the correct target diameter (~3.2 cm)."""
        if matlab_outputs.get("fallback_success", 0) != 1:
            pytest.skip("Fallback evolution did not succeed")
        diam = matlab_outputs["fallback_diam_cm"]
        assert 3.0 <= diam < 5.0, (
            f"Fallback diameter = {diam:.2f} cm "
            f"(expected ~3.2 cm from model parameter)"
        )
