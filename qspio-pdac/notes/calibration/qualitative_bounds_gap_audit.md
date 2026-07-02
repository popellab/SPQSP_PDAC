# Qualitative Behaviors Test Audit — Biological Gaps

**Date:** 2026-04-22
**Scope:** Audit of `tests/qualitative_bounds.yaml` (20 bounds) and `tests/scenarios/test_qualitative_behaviors.py` (~85 tests) against the biology implemented in the PDAC QSP model.

## Current Coverage (summary)

| Area | Bounds / tests |
|---|---|
| CD8 exhaustion (PD-1/PD-L1) | exhaustion fraction, worsening tolerance, checkpoint fold-change |
| Tregs | CD8:Treg ratio, Treg fraction, TGFb coupling |
| Macrophages | M2:M1 ratio, IL-10 / IFNg repolarization |
| CAFs / stroma | iCAF / myCAF / apCAF fractions, CAF:cancer ratio, stromal fraction, ECM fold-change |
| Tumor dynamics | monotonic growth, no spontaneous regression, diameter bounds |
| Treatment | GVAX+nivo CD8 infiltration, exhaustion reduction, M1/M2 shift |
| Hill functions | ~20 regulatory Hills audited for sensitivity |

## Gaps (tiered by OBED / virtual-patient-generation impact)

### Tier A — critical (breaks model if missing)

1. **CD4+ T-helper composition & Th1 priming.** No bounds on CD4:CD8 ratio (PDAC literature ~0.6–1.0) or CD4+/CD3 fraction (~30–40%). Without Th1 help the model cannot sustain IFNg after GVAX.
2. **Baseline cytokine concentrations.** Only VEGF has a clinical range. IL-12, IL-10, TGFb, IFNg, IL-6, IL-1 baselines are unconstrained — these are anchors for every downstream Hill function.
3. **PD-L1 expression frequencies.** No bounds on PD-L1+ fraction on cancer cells (~10–50%), APCs (~30–60%), or macrophages. This is the drug target — wrong baseline invalidates the checkpoint axis.
4. **Chemokine concentrations + quantitative recruitment kinetics.** CXCL12 blockade → CD8 infiltration is only tested directionally. No bounds on CCL2/CCL5/CXCL9-10/CXCL12 magnitudes.
5. **ECM pore size / stiffness.** `phi_collagen` is bounded, but `d_pore` (~50–200 nm) and `stiffness` (~10–100 kPa) — the actual inputs to `H_pore_T`, `H_pore_APC`, `H_stiff_Texh` — are not.

### Tier B — high (affects outcome predictions)

6. **IL-6 autocrine → iCAF maintenance; iCAF↔myCAF plasticity.** Static CAF composition misses plasticity.
7. **Hypoxia.** No bounds on O₂ baseline, hypoxic volume fraction (30–60% in PDAC), or HIF–VEGF coupling.
8. **FOLFIRINOX arm.** No monotherapy or combo tests, no DAMP/ICD dynamics — required for OBED arm comparison.
9. **Nivolumab PK & tumor penetration.** Current checkpoint release is effectively instant.
10. **Adaptive resistance.** No TIM-3 / LAG-3 / TIGIT upregulation under sustained PD-1 blockade.

### Tier C — medium (sensitivity/robustness)

- GVAX cell kinetics (peak d3–d7, ~5d half-life); CD8 peak timing (d5–d14)
- apCAF → P1 sequestration → Treg induction quantitation
- ECM remodeling under immune activation

## Revised scope after self-review

Initial draft had 14 Tier A bounds; review against "would this actually fail a
broken model?" and "is there a defensible PDAC source?" collapsed it to 7.

**Keep as qualitative bounds** (drafted in `qualitative_bounds_tier_a_drafts.yaml`):

| Bound | Why it stays |
|---|---|
| `cd4_fraction_range_diagnosis` | Defensible range from Carstens 2017 |
| `cd4_cd8_ratio_range_diagnosis` | Tightened to 0.5–1.5; stable landmark across scoring methods |
| `il12_tme_max_diagnosis` | Ceiling only — PDAC is IL-12-low; upper bound *would* fail broken models |
| `pdl1_positive_cancer_fraction_tps5` | Drug-target baseline with explicit TPS cutoff |
| `cxcl12_blockade_cd8_increase_min` | Perturbation/fold-change test; robust to absolute-concentration uncertainty |
| `ecm_pore_size_range_diagnosis` | Direct H_pore_T / H_pore_APC input |
| `ecm_stiffness_range_diagnosis` | Direct H_stiff_Texh / H_stiff_fib input |

**Demoted to deferred calibration targets** (belong in
`notes/calibration/pdac_cytokine_chemokine_targets.csv`, not bounds):

- IFNg, TGFb, IL-10, IL-6, CXCL12, CCL2 absolute TME concentrations — 3–4
  orders-of-magnitude μM bands are too wide to fail realistic models.
- IFNg specifically is now modeled as a linear sub-saturating spatial-coverage
  slope (main post-2026-04 refactor), so an absolute μM bound is even less
  meaningful.

**Dropped outright:**

- APC-PD-L1 positive fraction minimum — no defensible PDAC-specific source
  found; revisit if pan-cancer DC-PD-L1 multiplex data surfaces.

## Recommended next steps

1. Fill verbatim `evidence_snippet` + DOI for the 7 surviving drafts
   (maple extraction or manual), then promote to `tests/qualitative_bounds.yaml`.
2. Add demoted absolute-concentration entries to
   `notes/calibration/pdac_cytokine_chemokine_targets.csv` if not already
   present; extract via the staged-extraction pipeline.
3. Tier B #8 (FOLFIRINOX qualitative tests) stays deferred to the
   clinical-outcomes sprint already scoped.

## References

- Current bounds file: `tests/qualitative_bounds.yaml`
- Schema + snapshot checker: `tests/qualitative_bounds.py`
- Test suite: `tests/scenarios/test_qualitative_behaviors.py`
- Related: `notes/calibration/pdac_cytokine_chemokine_targets.csv` (deferred cytokine targets)
