# Qualitative Bounds — Current State (2026-04-22)

Literature-backed structural bounds the PDAC QSP model must satisfy. If violated, SBI calibration alone will not fix the issue — they are sanity checks upstream of posterior conditioning.

**Sources**: `tests/qualitative_bounds.yaml` (20 bounds, provenance + evidence), `tests/qualitative_bounds.py` (schema, loader, snapshot checker), `tests/test_qualitative_bounds.py` (17 fast schema/provenance tests), `tests/scenarios/test_qualitative_behaviors.py` (~102 behavior tests, 85 pass / 15 xfail / 1 xpass / 1 skip at last run, 60 s).

Values below are evaluated against the cached posterior-median base snapshot at `tests/scenarios/.sim_outputs.json` (last written by a MATLAB slow-test run).

## Part 1 — Snapshot bounds (evaluated from a diagnosis-state species dict)

14 of 20 bounds are evaluable without time series. `check_snapshot_bounds()` returns a `BoundCheckResult` per bound. **12 pass, 2 fail** on the cached posterior-median base.

| Bound | Lower | Upper | Units | Value | Status | Source |
|---|---|---|---|---|---|---|
| `cd8_exhaustion_fraction_min_diagnosis` | 0.15 | — | dimensionless | 0.389 | PASS | Shen 2017 (mean 0.306 ± 0.168; floor ≈ mean − 1 SD) |
| `cd8_exhaustion_fraction_range` | 0.10 | 0.65 | dimensionless | 0.389 | PASS | Shen 2017 (same PD-1+/CD8 data, range form) |
| `cd8_treg_ratio_max_diagnosis` | — | 15.0 | dimensionless | **1.7e-4** | PASS (trivially) | Golesworthy 2022, Carstens 2017 (literature mean ≈ 3) |
| `stromal_fraction_min_diagnosis` | 0.40 | — | dimensionless | 0.810 | PASS | Erkan 2012 (stroma > 80% of tumor volume) |
| `ecm_fold_change_range_diagnosis` | 5.0 | 20.0 | dimensionless | **0.995** | **FAIL** | Xu 2017 MPM (tumor/normal collagen ≈ 10×) |
| `apc_maturation_max_diagnosis` | — | 0.50 | dimensionless | **3.5e-6** | PASS (trivially) | Hegde 2020 (DC paucity / poor maturation) |
| `clinical_vegf_range` | 1000 | 200 000 | pg/mL | 47 910 | PASS | Haslene-Hox 2014 TIF, Hogendorf 2014 portal blood |
| `caf_cancer_ratio_range` | 1.0 | 5.0 | dimensionless | 2.28 | PASS | Erkan 2012 + Elyada 2019 (inferred) |
| `mycaf_fraction_min` | 0.40 | — | dimensionless | 0.586 | PASS | Elyada 2019 flow cytometry (45 ± 4%) |
| `icaf_fraction_range` | 0.20 | 0.65 | dimensionless | 0.413 | PASS | Elyada 2019 (44 ± 4%) |
| `apcaf_fraction_max` | — | 0.15 | dimensionless | 0.0033 | PASS | Elyada 2019 (10 ± 1%) |
| `m2_m1_ratio_range` | 1.0 | 5.0 | dimensionless | **11.63** | **FAIL** | Dias Costa 2022, Ino 2013 (stromal M2:M1 ≈ 1.9) |
| `treg_fraction_min` | 0.05 | — | dimensionless | 0.853 | PASS (trivially; very high) | Hiraoka 2006 (34.6% of CD4+) |
| `tumor_diameter_range` | 2.5 | 4.0 | cm | 3.20 | PASS | AJCC T2; Takahashi 2018 (median > 2 cm) |

### Failures worth discussing

- **`ecm_fold_change_range_diagnosis` — 0.995 vs [5, 20]**. The model at posterior medians shows essentially no collagen fold-change between tumor and baseline normal pancreas. Literature requires ~10×. Either the baseline-normal reference value used in the compute function is too high or the tumor-state collagen accumulation is undershooting. Flagged in the `test_all_snapshot_bounds_pass` xfail (`ecm_fold_change and m2_m1_ratio out of range at posterior medians`).
- **`m2_m1_ratio_range` — 11.6 vs [1, 5]**. Excess M2 polarization; also flagged by `TestCellComposition::test_m2_polarized` (xfail: "M2:M1 = 11.6 at posterior medians, needs joint calibration"). Stromal ratio in Dias Costa 2022 is ≈ 1.9. The upper bound of 5 already allowed a ~2.5× margin and is still breached.

### Passes that look suspicious (worth a collaborator eyeball)

- **`cd8_treg_ratio_max_diagnosis` = 1.7e-4** — technically passes the upper bound, but the ratio is four orders of magnitude below the literature mean of ~3. This is the same signal as the `m2_m1` failure from the T-cell compartment: Tregs dominating CD8.
- **`treg_fraction_min` = 0.85** — passes the ≥ 5% floor, but Tregs making up 85% of T cells is far above Hiraoka 2006 (34.6% of CD4 compartment). There is no upper bound on this observable; adding one would turn this into a hard failure.
- **`apc_maturation_max_diagnosis` = 3.5e-6** — trivially passes ≤ 0.5; effectively zero mature APCs. Consistent with the Hegde 2020 DC-paucity story but extreme.

Together these three "trivial passes" + the two hard failures all point at a posterior-median state that is over-immunosuppressed: too many Tregs, too few CD8, too many M2, and too little mature APC. This is what the xfail comment "needs joint calibration" refers to.

## Part 2 — Behavior bounds (time-series / perturbation, MATLAB-driven)

Remaining 6 bounds require longitudinal simulation:

| Bound | Criterion | Source |
|---|---|---|
| `tumor_dip_tolerance` | `C1_min / C1_d0 ≥ 0.95` over 90-d untreated progression | Haeno 2012 (exponential, monotonic growth) |
| `tumor_no_regression_tolerance` | `C1_min / C1_d0 ≥ 0.99` (tighter numerical tolerance) | Haeno 2012 |
| `exhaustion_worsening_tolerance` | 90-d change in CD8 exhaustion fraction ≤ 0.05 (must not improve) | Wherry 2011 |
| `ecm_concentration_stability` | ECM(d90)/ECM(d0) ∈ [0.5, 1.5] | Erkan 2012 (persistent desmoplasia) |
| `ecm_treatment_tolerance` | ECM(treated)/ECM(untreated) ≤ 1.5 at d21 | Erkan 2012 (no direct immune effect on ECM) |
| `checkpoint_fold_increase_min` | `k_C_Tcell_eff(nivo)/k_C_Tcell_eff(no nivo) ≥ 2` | Hui 2017 (PD-1 → IL-2 down 63%) |

These are checked by dedicated tests (mostly in `TestTemporalDynamics`, `TestTreatmentResponse`, `TestProgressionTimeSeries`, `TestTreatmentCascadeOrdering`). All pass at last run **except**:

- **`TestTemporalDynamics::test_worsening_exhaustion`** — currently XPASS. Declared `xfail` with reason "exhaustion drops slightly over 90 d (0.33 → 0.28)" but the current run passes, meaning the test is stricter than reality on the current build. Candidate for removing the xfail mark.
- **`TestTreatmentResponse::test_m1_m2_ratio_improves`** — XFAIL. "M1/M2 ratio essentially flat (0.739 → 0.738) under GVAX+nivo" — the model is not producing the expected repolarization shift.

## Part 3 — Adjacent behavior tests (not one-per-bound, but driven by the YAML)

`test_qualitative_behaviors.py` contains ~85 additional tests that load thresholds from the YAML and check directional, cascade, and Hill-function behaviors. Highlights:

- **`TestBaselineTME` (6 tests)**: all pass. Dense stroma, M2 dominance, high CD8 exhaustion, low CD8:Treg, myCAF-dominant stroma, poor APC maturation.
- **`TestDirectionalPerturbations` (5 tests)**: all pass. Blocking TGFb, CXCL12, IL-10, ECM, and IFNg each move the expected observable in the expected direction.
- **`TestHillFunctionAudit`**: sensitivity audit of 16 Hill functions. 4 PASS (`H_IL12`, `H_CXCL12_Texcl`, `H_pore_T`, `H_stiff_Texh`, `H_DAMP`), 11 XFAIL with documented reasons. The three TGFb-family Hills (`H_TGFb`, `H_TGFb_Teff`, `H_TGFb_APC`) are xfailed because the submodel-calibrated `k_TGFb_sec_myCAF` puts `[TGFb] ≈ 17 nM ≫ EC50 ≈ 0.07–0.14 nM`, so they saturate — biologically correct for desmoplastic PDAC, but the audit test expects a graded response.
- **`TestProgressionTimeSeries` (7 tests)** and **`TestTreatmentCascadeOrdering` (5 tests)**: all pass. Tumor, CD8, Mac_M1, IFNg, iCAF, myCAF all grow over 90 d; GVAX decay, mAPC transient peak, checkpoint release, d14 CD8 infiltration, d14 IFNg increase all check out.
- **`TestTreatmentVsUntreated` (5 tests)** and **`TestTreatmentTimeSeriesComparison` (5 tests)**: all pass. GVAX+nivo reduces tumor, M2, Treg; raises CD8, IFNg, k_C_Tcell_eff.

## Part 4 — Provenance

All 20 bounds carry a verbatim `evidence_snippet` and a `primary_data_source` with a DOI. No PMID-only entries. 10 distinct DOIs, with Erkan 2012 (`10.1038/nrgastro.2012.115`) supporting 4 bounds, Elyada 2019 (`10.1158/2159-8290.CD-19-0094`) supporting 3, and Shen 2017 + Hanahan/Weinberg + Haeno 2012 each supporting 2. `validate_all_provenance()` cross-checks year and first-author surname against CrossRef; covered by `tests/test_qualitative_bounds.py`.

## Part 5 — TL;DR for collaborators

1. **Structural bounds as a whole are holding**: 12/14 snapshot bounds pass on the current posterior median; the full ~85 behavior tests run green modulo documented xfails.
2. **Two live failures**: ECM fold-change (tumor/normal ≈ 1 instead of 10) and M2:M1 ratio (11.6 instead of 1–5). Both are flagged as "needs joint calibration" rather than structural model bugs.
3. **Three "passing but extreme" observables** (CD8:Treg ≈ 0, Treg fraction 0.85, mature APC ≈ 0) indicate an over-immunosuppressed posterior median — the same failure mode viewed from the T-cell and DC compartments. Consider adding upper bounds on `treg_fraction` and a lower bound on `cd8_treg_ratio` to convert these into hard failures.
4. **TGFb-family Hill xfails are by design**: PDAC desmoplastic TGFb saturates those Hills, and the audit test is the piece that is biologically loose, not the model.
5. **Exhaustion-worsening test is now XPASS** — the xfail mark is stale; remove next time someone is editing the file.
