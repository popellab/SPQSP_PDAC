"""Coverage-style migration of qualitative-bound tests (gh #57).

Sister file to ``test_qualitative_behaviors.py``. Where the legacy file
asserts that a single prior-median θ satisfies each bound, this file
runs N classifier-filtered prior draws and asserts that
``coverage_threshold`` of them satisfy the bound.

A failure here distinguishes three modes that the prior-median test
collapses into one:
  - **structural deficiency** — ~0% coverage means no θ in support
    can satisfy the bound; the model is mis-specified.
  - **prior-structure** — high p95 / low p50 vs the bound: model can
    reach the bound region but most prior mass is on the wrong side.
  - **post-cal regression** — needs a stage-2 source (gh #58).

Each migrated test mirrors the corresponding legacy test in
``test_qualitative_behaviors.py`` but operates on arrays of length
``n_valid`` instead of scalars.

Source defaults to "prior" (classifier-filtered). N defaults to 100
(override via ``QUAL_COVERAGE_N_DRAWS``).
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from tests.scenarios._coverage import (
    DEFAULT_COVERAGE_THRESHOLD,
    assert_coverage,
    assert_relational_coverage,
)
# Deprecated suite (memory: project_pr63_qual_coverage.md, 2026-04-30).
# Marked slow at the module level so CI's `-m "not slow"` skips it without
# touching individual xfails. Sister file test_qualitative_behaviors.py is
# also marked slow for the same reason.
pytestmark = pytest.mark.slow

N_DRAWS_DEFAULT = int(os.environ.get("QUAL_COVERAGE_N_DRAWS", "100"))
SOURCE = os.environ.get("QUAL_COVERAGE_SOURCE", "prior")


@pytest.fixture(scope="module")
def coverage_outputs():
    """Build the dict-of-arrays once per session via build_coverage_outputs.

    All tests in this module read from the same simulation result —
    scenarios run once across N draws, never once per test.
    """
    from tests.scenarios._cpp_helper import (
        _RESTRICTION_CLASSIFIER_DIR,
        _require_qsp_sim,
        build_coverage_outputs,
    )

    _require_qsp_sim()
    if SOURCE != "prior":
        pytest.skip(
            f"QUAL_COVERAGE_SOURCE={SOURCE} not yet wired (only 'prior' supported "
            "until stage-2 posterior path is added in gh #58)"
        )
    if not (_RESTRICTION_CLASSIFIER_DIR / "classifier.pkl").exists():
        raise RuntimeError(
            f"restriction classifier not found at {_RESTRICTION_CLASSIFIER_DIR}. "
            "It should be checked into the repo as a test fixture; see "
            "tests/fixtures/restriction_classifier/README.md for regen steps."
        )

    return build_coverage_outputs(
        n_draws=N_DRAWS_DEFAULT,
        seed=2025,
        use_restriction=True,
        include_treatment=True,
        cache=True,
    )


def _safe_div(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-30) -> np.ndarray:
    """Element-wise a / max(b, eps), preserving NaN."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return np.where(np.isfinite(a) & np.isfinite(b), a / np.maximum(b, eps), np.nan)


def _exh_fraction(cd8: np.ndarray, cd8_exh: np.ndarray) -> np.ndarray:
    return _safe_div(cd8_exh, cd8 + cd8_exh)


# ===========================================================================
# Sanity floor
# ===========================================================================

def test_n_valid_meets_floor(coverage_outputs):
    """At least 70% of requested draws must produce valid sims across all
    three scenarios. Below that, per-bound coverage estimates get noisy."""
    n_req = int(coverage_outputs["n_draws_requested"])
    n_valid = int(coverage_outputs["n_valid"])
    floor = 0.7
    frac = n_valid / max(n_req, 1)
    assert frac >= floor, (
        f"only {n_valid}/{n_req} ({frac*100:.1f}%) draws produced valid sims "
        f"across all 3 scenarios; coverage estimates unreliable"
    )


# ===========================================================================
# TIER 1: Baseline TME snapshots (diagnosis)
# ===========================================================================

# ===========================================================================
# TIER 2: Temporal monotonicity / tolerance bounds (90d untreated)
# ===========================================================================

class TestProgressionTolerances:
    """Coverage versions of the structural temporal bounds.

    These are 'kind: structural' bounds — relational invariants no single
    experiment quantifies, expressing 'PDAC does not spontaneously
    regress' or 'exhaustion does not improve under chronic antigen.'
    """

    def test_tumor_dip_tolerance(self, coverage_outputs):
        """C1_min should not drop far below C1 at d0 (no spontaneous regression)."""
        prog = coverage_outputs["prog"]
        d0 = coverage_outputs["prog_d0"]
        c1_min = np.asarray(prog["C1_min"])
        c1_d0 = np.asarray(d0["C1"])
        ratio = _safe_div(c1_min, c1_d0)
        assert_coverage(
            ratio, lower=0.95, threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="tumor_dip_tolerance", units="dimensionless",
        )

    def test_tumor_no_regression(self, coverage_outputs):
        """Tumor C1_min over 90d ≥ tol × C1 at d0 (monotonicity floor)."""
        prog = coverage_outputs["prog"]
        d0 = coverage_outputs["prog_d0"]
        c1_min = np.asarray(prog["C1_min"])
        c1_d0 = np.asarray(d0["C1"])
        ratio = _safe_div(c1_min, c1_d0)
        assert_coverage(
            ratio, lower=0.99, threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="tumor_no_regression_tolerance", units="dimensionless",
        )

    def test_exhaustion_worsening_tolerance(self, coverage_outputs):
        """CD8 exhaustion fraction must not improve d0 → d90."""
        d0 = coverage_outputs["prog_d0"]
        d90 = coverage_outputs["prog_d90"]
        frac_d0 = _exh_fraction(np.asarray(d0["CD8"]), np.asarray(d0["CD8_exh"]))
        frac_d90 = _exh_fraction(np.asarray(d90["CD8"]), np.asarray(d90["CD8_exh"]))
        delta = frac_d0 - frac_d90
        assert_coverage(
            delta, upper=0.05, threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="exhaustion_worsening_tolerance", units="dimensionless",
        )

    def test_ecm_concentration_stability(self, coverage_outputs):
        """ECM concentration ratio (d90 / d0) stays in [lower, upper]."""
        d0 = coverage_outputs["prog_d0"]
        d90 = coverage_outputs["prog_d90"]
        ecm0 = np.asarray(d0["collagen"])
        ecm90 = np.asarray(d90["collagen"])
        vt0 = np.asarray(d0["V_T"])
        vt90 = np.asarray(d90["V_T"])
        conc0 = _safe_div(ecm0, vt0)
        conc90 = _safe_div(ecm90, vt90)
        ratio = _safe_div(conc90, conc0)
        assert_coverage(
            ratio, lower=0.5, upper=1.5,
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="ecm_concentration_stability", units="dimensionless",
        )


# ===========================================================================
# TIER 3: Treatment-response bounds (GVAX+nivo)
# ===========================================================================

class TestTreatmentResponseCoverage:
    """Coverage versions of the treatment-effect bounds. All require
    include_treatment=True in build_coverage_outputs."""

    def test_checkpoint_release_fold(self, coverage_outputs):
        """k_C_Tcell_eff at tx_d14 must be ≥ 2× prog_d0 (PD-1 release)."""
        prog = coverage_outputs.get("prog_d0", {})
        tx14 = coverage_outputs.get("tx_d14", {})
        if "k_C_Tcell_eff" not in prog or "k_C_Tcell_eff" not in tx14:
            pytest.skip("k_C_Tcell_eff observable missing")
        k_base = np.asarray(prog["k_C_Tcell_eff"])
        k_tx = np.asarray(tx14["k_C_Tcell_eff"])
        fold = _safe_div(k_tx, k_base)
        assert_coverage(
            fold, lower=2.0, threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="checkpoint_fold_increase_min", units="dimensionless",
        )

    def test_cd8_increases_d0_to_d14(self, coverage_outputs):
        """CD8 should expand from tx_d0 to tx_d14 under GVAX+nivo (relational)."""
        d0 = coverage_outputs.get("tx_d0", {})
        d14 = coverage_outputs.get("tx_d14", {})
        if "CD8" not in d0 or "CD8" not in d14:
            pytest.skip("CD8 series at tx_d0/d14 not available")
        assert_relational_coverage(
            np.asarray(d14["CD8"]), np.asarray(d0["CD8"]),
            op=">", threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="cd8_infiltration_increases_d14",
        )

    def test_ecm_treatment_tolerance(self, coverage_outputs):
        """ECM at tx_d21 should not exceed 1.5× untreated d30 ECM."""
        tx21 = coverage_outputs.get("tx_d21", {})
        prog30 = coverage_outputs.get("prog_d30", {})
        if not tx21 or "collagen" not in tx21 or "collagen" not in prog30:
            pytest.skip("ECM treatment-comparison data unavailable")
        ecm_tx = np.asarray(tx21["collagen"])
        ecm_prog = np.asarray(prog30["collagen"])
        ratio = _safe_div(ecm_tx, ecm_prog)
        assert_coverage(
            ratio, upper=1.5, threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="ecm_treatment_tolerance", units="dimensionless",
        )


# ===========================================================================
# TIER 4: GVAX+nivo monotonic responses (no YAML bound; pure structural
# relational checks). These are 'kind: structural' invariants — fold-changes
# and direction-of-change predictions for the treatment cascade.
# ===========================================================================

class TestTreatmentMonotonicity:
    """Coverage versions of the legacy TestTreatmentResponse tier."""

    def test_total_cd8_increases_d0_to_d21(self, coverage_outputs):
        """Total CD8 (active + exhausted) should expand under GVAX+nivo."""
        d0 = coverage_outputs.get("tx_d0", {})
        d21 = coverage_outputs.get("tx_d21", {})
        if not d0 or "CD8" not in d0 or "CD8" not in d21:
            pytest.skip("CD8 series unavailable")
        total_d0 = np.asarray(d0["CD8"]) + np.asarray(d0.get("CD8_exh", np.zeros_like(d0["CD8"])))
        total_d21 = np.asarray(d21["CD8"]) + np.asarray(d21.get("CD8_exh", np.zeros_like(d21["CD8"])))
        assert_relational_coverage(
            total_d21, total_d0, op=">",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="total_cd8_increases_tx_d0_to_d21",
        )

    @pytest.mark.xfail(strict=True, reason="M1/M2 polarization cluster; tracked in #34")
    def test_m1_m2_ratio_improves_under_treatment(self, coverage_outputs):
        """M1/M2 ratio should rise from tx_d0 to tx_d21 (IFNg-driven repolarization)."""
        d0 = coverage_outputs.get("tx_d0", {})
        d21 = coverage_outputs.get("tx_d21", {})
        if not d0 or "Mac_M1" not in d0:
            pytest.skip("Mac series unavailable")
        ratio_d0 = _safe_div(np.asarray(d0["Mac_M1"]), np.asarray(d0["Mac_M2"]))
        ratio_d21 = _safe_div(np.asarray(d21["Mac_M1"]), np.asarray(d21["Mac_M2"]))
        assert_relational_coverage(
            ratio_d21, ratio_d0, op=">",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="m1_m2_ratio_improves_under_treatment",
        )

    def test_exhaustion_fraction_decreases_under_treatment(self, coverage_outputs):
        """CD8 exhaustion fraction should fall from tx_d0 to tx_d21 (PD-1 blockade)."""
        d0 = coverage_outputs.get("tx_d0", {})
        d21 = coverage_outputs.get("tx_d21", {})
        if not d0 or "CD8" not in d0:
            pytest.skip("CD8 series unavailable")
        frac_d0 = _exh_fraction(np.asarray(d0["CD8"]), np.asarray(d0["CD8_exh"]))
        frac_d21 = _exh_fraction(np.asarray(d21["CD8"]), np.asarray(d21["CD8_exh"]))
        assert_relational_coverage(
            frac_d0, frac_d21, op=">",  # d0 > d21 (decreased)
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="exhaustion_fraction_decreases_under_treatment",
        )


# ===========================================================================
# TIER 5: GVAX cascade structural fixes (regression guards from #44 / #47).
# Each is a pure structural relational check protecting against silent
# re-breakage of the cDC1 antigen-transfer / niche-scaled-cap fixes.
# ===========================================================================

class TestGvaxCascadeFixes:
    """Coverage versions of the GVAX cascade structural-regression guards."""

    @pytest.mark.xfail(strict=True, reason="GVAX cascade fires only in upper tail; tracked in #60")
    def test_gvax_drives_mcDC1_at_least_2x(self, coverage_outputs):
        """V_LN.mcDC1 should rise ≥ 2× under GVAX (cDC1 antigen-transfer fires)."""
        d0 = coverage_outputs.get("tx_d0", {})
        d21 = coverage_outputs.get("tx_d21", {})
        if not d0 or "VLN_mcDC1" not in d0:
            pytest.skip("V_LN.mcDC1 unavailable")
        ratio = _safe_div(np.asarray(d21["VLN_mcDC1"]), np.asarray(d0["VLN_mcDC1"]))
        assert_coverage(
            ratio, lower=2.0, threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="gvax_drives_mcDC1_2x", units="dimensionless (fold-change)",
        )

    @pytest.mark.xfail(strict=True, reason="GVAX cascade fires only in upper tail; tracked in #60")
    def test_gvax_drives_cd8_infiltration_at_least_2x(self, coverage_outputs):
        """V_T.CD8 should rise ≥ 2× under GVAX+Cy+nivo (cascade propagates to V_T)."""
        d0 = coverage_outputs.get("tx_d0", {})
        d21 = coverage_outputs.get("tx_d21", {})
        if not d0 or "CD8" not in d0:
            pytest.skip("CD8 series unavailable")
        ratio = _safe_div(np.asarray(d21["CD8"]), np.asarray(d0["CD8"]))
        assert_coverage(
            ratio, lower=2.0, threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="gvax_drives_cd8_2x", units="dimensionless (fold-change)",
        )

    @pytest.mark.xfail(strict=True, reason="GVAX cascade fires only in upper tail; tracked in #60")
    def test_gvax_expands_th_pool_vs_untreated(self, coverage_outputs):
        """V_T.Th under GVAX (tx_d21) should exceed untreated (prog_d21) by ≥ 1.10×."""
        tx21 = coverage_outputs.get("tx_d21", {})
        prog21 = coverage_outputs.get("prog_d21", {})
        if not tx21 or "Th" not in tx21 or "Th" not in prog21:
            pytest.skip("Th series at tx_d21/prog_d21 unavailable")
        ratio = _safe_div(np.asarray(tx21["Th"]), np.asarray(prog21["Th"]))
        assert_coverage(
            ratio, lower=1.10, threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="gvax_expands_Th_pool_vs_untreated",
            units="dimensionless (tx_d21 / prog_d21)",
        )

    @pytest.mark.xfail(strict=True, reason="GVAX cascade fires only in upper tail; tracked in #60")
    def test_cy_depletes_tregs_under_gvax(self, coverage_outputs):
        """V_T.Treg should fall to ≤ 0.80× from tx_d0 to tx_d21 (Cy depletion)."""
        d0 = coverage_outputs.get("tx_d0", {})
        d21 = coverage_outputs.get("tx_d21", {})
        if not d0 or "Treg" not in d0:
            pytest.skip("Treg series unavailable")
        ratio = _safe_div(np.asarray(d21["Treg"]), np.asarray(d0["Treg"]))
        assert_coverage(
            ratio, upper=0.80, threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="cy_depletes_tregs_under_gvax",
            units="dimensionless (tx_d21 / tx_d0)",
        )

    def test_th_treg_ratio_rises_under_treatment(self, coverage_outputs):
        """Th/Treg ratio should rise from tx_d0 to tx_d21 (Th expansion + Cy)."""
        d0 = coverage_outputs.get("tx_d0", {})
        d21 = coverage_outputs.get("tx_d21", {})
        if not d0 or "Th" not in d0:
            pytest.skip("Th/Treg series unavailable")
        ratio_d0 = _safe_div(np.asarray(d0["Th"]), np.asarray(d0["Treg"]))
        ratio_d21 = _safe_div(np.asarray(d21["Th"]), np.asarray(d21["Treg"]))
        assert_relational_coverage(
            ratio_d21, ratio_d0, op=">",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="th_treg_ratio_rises_under_treatment",
        )


# ===========================================================================
# TIER 6: Untreated progression direction (parametrized, multi-species).
# Mirrors legacy TestProgressionDirection / TestProgressionTimeSeries.
# ===========================================================================

PROGRESSION_INCREASES = [
    "C_total", "V_T", "collagen", "Treg", "Mac_M2", "MDSC",
    "Mac_M1", "iCAF", "myCAF",
]
# Legacy CD8 and IFNg are xfailed (ECM-exclusion mechanism, see legacy
# docstring for test_species_increases_d0_to_d90). Mirror the xfail.
PROGRESSION_INCREASES_XFAIL = ["CD8", "IFNg"]


@pytest.mark.parametrize("species", PROGRESSION_INCREASES)
def test_species_increases_d0_to_d90(coverage_outputs, species):
    """Coverage version of TestProgressionDirection.test_progression_increases.

    Each species should grow over the 90d untreated progression in the
    majority of valid prior draws.
    """
    d0 = coverage_outputs.get("prog_d0", {})
    d90 = coverage_outputs.get("prog_d90", {})
    if species not in d0 or species not in d90:
        pytest.skip(f"{species} unavailable")
    assert_relational_coverage(
        np.asarray(d90[species]), np.asarray(d0[species]), op=">",
        threshold=DEFAULT_COVERAGE_THRESHOLD,
        name=f"prog_d0_to_d90_{species}_increases",
    )


@pytest.mark.parametrize("species", PROGRESSION_INCREASES_XFAIL)
@pytest.mark.xfail(
    reason=(
        "Legacy xfail mirrored: ECM-barrier exclusion drops CD8 over 90d "
        "after the LOX-crosslinking shield fix (2026-04-22). IFNg is "
        "downstream. Coverage test inherits the same expected failure; "
        "revisit after M1/M2 polarization fix (#34)."
    ),
    strict=False,
)
def test_species_increases_d0_to_d90_xfail(coverage_outputs, species):
    d0 = coverage_outputs.get("prog_d0", {})
    d90 = coverage_outputs.get("prog_d90", {})
    if species not in d0 or species not in d90:
        pytest.skip(f"{species} unavailable")
    assert_relational_coverage(
        np.asarray(d90[species]), np.asarray(d0[species]), op=">",
        threshold=DEFAULT_COVERAGE_THRESHOLD,
        name=f"prog_d0_to_d90_{species}_increases",
    )


@pytest.mark.xfail(strict=True, reason="myCAF dominance shortfall; tracked in #59")
def test_mycaf_dominates_icaf_at_d0(coverage_outputs):
    """myCAF > iCAF at diagnosis (TGFb-rich PDAC favors myCAF phenotype)."""
    d0 = coverage_outputs.get("prog_d0", {})
    if "myCAF" not in d0 or "iCAF" not in d0:
        pytest.skip("CAF subtype series unavailable")
    assert_relational_coverage(
        np.asarray(d0["myCAF"]), np.asarray(d0["iCAF"]), op=">",
        threshold=DEFAULT_COVERAGE_THRESHOLD,
        name="mycaf_dominates_icaf_at_d0",
    )


# ===========================================================================
# TIER 7: Treatment vs untreated comparisons (tx_d21 vs prog_d30).
# ===========================================================================

class TestTreatmentVsUntreated:
    """Coverage versions of TestTreatmentVsUntreated."""

    def test_treatment_slows_tumor_C_total(self, coverage_outputs):
        """C_total should be lower under treatment at d21 vs untreated d30."""
        tx = coverage_outputs.get("tx_d21", {})
        prog = coverage_outputs.get("prog_d30", {})
        if not tx or "C_total" not in tx:
            pytest.skip("C_total comparison unavailable")
        assert_relational_coverage(
            np.asarray(tx["C_total"]), np.asarray(prog["C_total"]), op="<",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="treatment_slows_C_total",
        )

    def test_treatment_reduces_volume(self, coverage_outputs):
        """V_T at tx_d21 should be smaller than untreated d30."""
        tx = coverage_outputs.get("tx_d21", {})
        prog = coverage_outputs.get("prog_d30", {})
        if not tx or "V_T" not in tx:
            pytest.skip("V_T comparison unavailable")
        assert_relational_coverage(
            np.asarray(tx["V_T"]), np.asarray(prog["V_T"]), op="<",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="treatment_reduces_V_T",
        )

    def test_treatment_reduces_m2(self, coverage_outputs):
        """Mac_M2 should be lower under treatment at d21 vs untreated d30."""
        tx = coverage_outputs.get("tx_d21", {})
        prog = coverage_outputs.get("prog_d30", {})
        if not tx or "Mac_M2" not in tx:
            pytest.skip("Mac_M2 comparison unavailable")
        assert_relational_coverage(
            np.asarray(tx["Mac_M2"]), np.asarray(prog["Mac_M2"]), op="<",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="treatment_reduces_Mac_M2",
        )

    def test_treatment_reduces_tregs(self, coverage_outputs):
        """Treg should be lower under treatment at d21 vs untreated d30."""
        tx = coverage_outputs.get("tx_d21", {})
        prog = coverage_outputs.get("prog_d30", {})
        if not tx or "Treg" not in tx:
            pytest.skip("Treg comparison unavailable")
        assert_relational_coverage(
            np.asarray(tx["Treg"]), np.asarray(prog["Treg"]), op="<",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="treatment_reduces_Treg",
        )


# ===========================================================================
# TIER 8: Treatment cascade ordering (temporal structure of the GVAX dose).
# ===========================================================================

class TestTreatmentCascadeOrdering:

    def test_gvax_cells_decay_d7_d14_d21(self, coverage_outputs):
        """V_ID.GVAX_cells should decay monotonically d7 → d14 → d21."""
        d7 = coverage_outputs.get("tx_d7", {})
        d14 = coverage_outputs.get("tx_d14", {})
        d21 = coverage_outputs.get("tx_d21", {})
        if not d7 or "VID_GVAX_cells" not in d7:
            pytest.skip("GVAX_cells series unavailable")
        g7 = np.asarray(d7["VID_GVAX_cells"])
        g14 = np.asarray(d14["VID_GVAX_cells"])
        g21 = np.asarray(d21["VID_GVAX_cells"])
        # require g7 > g14 AND g14 > g21 jointly across draws
        finite = np.isfinite(g7) & np.isfinite(g14) & np.isfinite(g21)
        satisfied = finite & (g7 > g14) & (g14 > g21)
        frac = float(satisfied.mean())
        if frac < DEFAULT_COVERAGE_THRESHOLD:
            raise AssertionError(
                f"[gvax_cells_decay_d7_d14_d21] coverage {frac*100:.1f}% < "
                f"threshold {DEFAULT_COVERAGE_THRESHOLD*100:.0f}%\n"
                f"  bound: GVAX_cells d7 > d14 > d21\n"
                f"  N: {g7.size} ({int(finite.sum())} finite); satisfying: "
                f"{int(satisfied.sum())}/{g7.size}"
            )

    @pytest.mark.xfail(strict=True, reason="mAPC peak heterogeneity; tracked in #62")
    def test_mapc_peak_at_d14(self, coverage_outputs):
        """V_ID.mAPC peaks transiently around d14: d14 > d0 AND d14 > d21."""
        d0 = coverage_outputs.get("tx_d0", {})
        d14 = coverage_outputs.get("tx_d14", {})
        d21 = coverage_outputs.get("tx_d21", {})
        if not d14 or "VID_mAPC" not in d14:
            pytest.skip("mAPC series unavailable")
        m0 = np.asarray(d0["VID_mAPC"])
        m14 = np.asarray(d14["VID_mAPC"])
        m21 = np.asarray(d21["VID_mAPC"])
        finite = np.isfinite(m0) & np.isfinite(m14) & np.isfinite(m21)
        satisfied = finite & (m14 > m0) & (m14 > m21)
        frac = float(satisfied.mean())
        if frac < DEFAULT_COVERAGE_THRESHOLD:
            raise AssertionError(
                f"[mapc_peak_at_d14] coverage {frac*100:.1f}% < "
                f"threshold {DEFAULT_COVERAGE_THRESHOLD*100:.0f}%\n"
                f"  bound: mAPC d14 > d0 AND mAPC d14 > d21\n"
                f"  N: {m0.size} ({int(finite.sum())} finite); satisfying: "
                f"{int(satisfied.sum())}/{m0.size}"
            )

    def test_ifng_increases_by_d14(self, coverage_outputs):
        """IFNg should increase from tx_d0 to tx_d14 (CD8 reactivation)."""
        d0 = coverage_outputs.get("tx_d0", {})
        d14 = coverage_outputs.get("tx_d14", {})
        if not d0 or "IFNg" not in d0:
            pytest.skip("IFNg series unavailable")
        assert_relational_coverage(
            np.asarray(d14["IFNg"]), np.asarray(d0["IFNg"]), op=">",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="ifng_increases_tx_d0_to_d14",
        )


# ===========================================================================
# TIER 9: Treatment time-series comparisons against untreated baseline.
# Mirrors legacy TestTreatmentTimeSeriesComparison.
# ===========================================================================

class TestTreatmentTimeSeriesComparison:

    def test_k_C_Tcell_eff_higher_with_treatment(self, coverage_outputs):
        prog = coverage_outputs.get("prog_d0", {})
        tx = coverage_outputs.get("tx_d14", {})
        if "k_C_Tcell_eff" not in prog or "k_C_Tcell_eff" not in tx:
            pytest.skip("k_C_Tcell_eff observable unavailable")
        assert_relational_coverage(
            np.asarray(tx["k_C_Tcell_eff"]), np.asarray(prog["k_C_Tcell_eff"]), op=">",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="k_C_Tcell_eff_higher_with_treatment",
        )

    def test_ifng_higher_with_treatment(self, coverage_outputs):
        prog = coverage_outputs.get("prog_d10", {})  # closest to tx_d14
        tx = coverage_outputs.get("tx_d14", {})
        if not prog or "IFNg" not in prog:
            pytest.skip("IFNg comparison unavailable")
        assert_relational_coverage(
            np.asarray(tx["IFNg"]), np.asarray(prog["IFNg"]), op=">",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="ifng_higher_with_treatment_d14_vs_progd10",
        )

    def test_cd8_higher_with_treatment(self, coverage_outputs):
        prog = coverage_outputs.get("prog_d10", {})
        tx = coverage_outputs.get("tx_d14", {})
        if not prog or "CD8" not in prog:
            pytest.skip("CD8 comparison unavailable")
        assert_relational_coverage(
            np.asarray(tx["CD8"]), np.asarray(prog["CD8"]), op=">",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="cd8_higher_with_treatment_d14_vs_progd10",
        )

    def test_tumor_lower_with_treatment_at_d21(self, coverage_outputs):
        prog = coverage_outputs.get("prog_d21", {})
        tx = coverage_outputs.get("tx_d21", {})
        if "C_total" not in prog or "C_total" not in tx:
            pytest.skip("C_total comparison unavailable")
        assert_relational_coverage(
            np.asarray(tx["C_total"]), np.asarray(prog["C_total"]), op="<",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="tumor_lower_with_treatment_at_d21",
        )

    def test_treg_lower_with_treatment_at_d21(self, coverage_outputs):
        prog = coverage_outputs.get("prog_d21", {})
        tx = coverage_outputs.get("tx_d21", {})
        if "Treg" not in prog or "Treg" not in tx:
            pytest.skip("Treg comparison unavailable")
        assert_relational_coverage(
            np.asarray(tx["Treg"]), np.asarray(prog["Treg"]), op="<",
            threshold=DEFAULT_COVERAGE_THRESHOLD,
            name="treg_lower_with_treatment_at_d21",
        )


