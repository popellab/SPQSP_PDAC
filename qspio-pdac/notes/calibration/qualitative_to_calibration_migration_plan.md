---
date: 2026-04-27
status: Phase A + B + C + E1 + mechanistic-prior schema landed. Phase D scoped down (see Phase D section); full retirement parked as Phase D-2. Phase E2 still gated on the clinical-backbone sprint.
---

# Migration plan — qualitative tests → single calibration framework

## TL;DR

The repository currently maintains three parallel validation frameworks:

1. **Calibration targets + parameter_audit PPC** (literature distribution fits)
2. **Qualitative bounds + snapshot checker** (literature inequality thresholds)
3. **Qualitative behavior tests + coverage tier** (structural/dynamics/ordering invariants)

The goal of this migration is to **collapse all three into one
framework** centered on calibration targets, with a small auxiliary
restriction-phase filter for parameter-validity preconditions. The
single framework is principled (Bayesian inference using literature as
data, not as priors), reviewer-defensible (one canonical source of
truth for "what should the model do"), and avoids the prior-tuning
methodology debates that consumed the previous ~2 weeks of work.

The migration is phased:
- **Phase A (1-2 days)**: drop redundant bounds; surviving bounds become cal targets. Bounds YAML retires.
- **Phase B (re-scoped 2× on 2026-04-27, ~half day)**: original plan was multi-timepoint d3/d7/d14 cal targets, but no PDAC trial samples patients mid-neoadjuvant. First reframe (higher-n + paired Heumann targets) also dead — Heumann mIHC is TLA-specific, not non-LA bulk tumor like Li 2022. Final scope: 3-4 mechanistic-prior cascade-ordering targets only (`epistemic_basis: mechanistic`). The Heumann mIHC sheet is parked for the future TLA / B-cell module sprint.
- **Phase C (10 min)**: delete Hill function audit tests (no replacement). The audit was a *diagnostic* layer with `_GRADED_XFAIL`/`_SWITCH_XFAIL` carve-outs that admitted we couldn't agree on its categorical thresholds. The earlier plan to convert it into a restriction-phase filter would have *added* a fourth validation framework just as we were collapsing three into one; "saturated Hills are unidentifiable" is a training-efficiency argument, not a statistical-correctness one. Posterior + PRCC + audit `contraction_local_shell` already surface unidentifiable parameters. Drop the audit; if a Hill-activation report turns out to be useful, it becomes a PPC-adjacent script post-Phase-D.
- **Phase D (scoped down, ~1 day done; full retirement deferred)**: original plan was to migrate counterfactual perturbations + regression guards into cross-scenario fold-change cal targets, deleting all of `test_qualitative_behaviors.py` / `test_qualitative_coverage.py`. Two findings reshaped this:
  1. The 5 perturbation scenarios (`pert_tgfb`, `pert_cxcl12`, ...) are **MATLAB-only** runtime parameter overrides in `test_qualitative_behaviors_helper.m`, not scenario YAMLs. Migrating them to cal targets requires per-scenario parameter-override support in the C++ simulator + scenario loader (qsp-hpc-tools surgery, ~3-4 days).
  2. The cross-scenario observable architecture (Option B in design discussion) requires extending `compute_test_statistics_batch` to accept multiple sim_dfs simultaneously — another ~3-4 days of qsp-hpc-tools work.
  Done now: delete genuinely-redundant tests (TestQualitativeBehaviorPrereqs file-existence checks), commit the mechanistic-prior schema (separately landed in maple), ship two canonical mechanistic targets. Survivors in `test_qualitative_behaviors.py` / `test_qualitative_coverage.py` are reframed as **model-wiring + regression-guard tests** (TestDirectionalPerturbations, TestCascadeCoherence, TestGvaxCascadeFixes, TestTreatmentVsUntreated, TestTreatmentMonotonicity, TestTemporalDynamics, TestTreatmentResponse, TestProgressionDirection/TimeSeries, TestTreatmentCascadeOrdering, TestTreatmentTimeSeriesComparison, TestProgressionTolerances, TestTreatmentResponseCoverage, TestParameterSensitivity). They survive as a small "model behavior unit-test" layer rather than a parallel validation framework. Full migration to cal targets is gated on the perturbation-scenario + cross-scenario-observable infrastructure work and has been parked as a future sprint (call it "Phase D-2") rather than blocking this migration.
- **Phase E (1 week now + deferred backbone sprint)**: clinical outcomes (RECIST, ORR, natural-history tumor diameter, time-to-progression). Convert the low-confound prediction-only YAMLs into CalibrationTargets now; defer DFS/OS until the surgery+FFX clinical-backbone sprint exists.

End state (revised after Phase D scope-down): bounds YAML and bounds-style
literature-inequality validation are retired (Phase A done). Hill audit
is retired (Phase C done). Mechanistic-prior cal target schema landed in
maple (epistemic_basis field). Cal-target PPC + mechanistic priors are
the validation framework for literature/biological constraints.
`test_qualitative_behaviors.py` + `test_qualitative_coverage.py` survive
as model-wiring + regression-guard unit tests (perturbation responses,
cascade coherence, treatment-cascade ordering, treatment vs untreated
comparisons) — these aren't a "validation framework" in the same sense
as the retired bounds YAML; they're just normal tests of model behavior.
Phase D-2 (full retirement of those tests via perturbation-scenario +
cross-scenario-observable infrastructure) is parked as a future sprint.
No new restriction-filter module.

## Motivation

The prior-tuning debate over the past sprint surfaced a structural
issue: maintaining bounds and cal targets as separate validation roles
creates double-dipping, Goodhart-on-coverage-metric, and OBED
miscalibration risks. The cleanest resolution is to make every literature
observation play exactly one statistical role — and that role should be
calibration target / likelihood data, not prior.

A second observation: the qualitative-test surface is ~5x larger than
the bounds YAML (132 tests vs 24 bounds). Many of these tests are
counterfactual / structural / dynamic invariants that PPC cannot replace
but that *can* be folded into calibration as cross-scenario or
multi-timepoint cal targets. By doing so, the model's structural
correctness becomes part of its likelihood, not part of a separate test
suite to maintain.

Companion methodology notes:
- `cd8_treg_fix_proposal.md` — path A (sanity check + fix root causes) vs metric tuning

## Mechanistic-prior cal targets

A mid-migration scope refinement (2026-04-27): not every constraint we
want to encode in the unified framework has a primary literature
quantification. Many are *mechanistic invariants* — "untreated PDAC
does not regress", "PD-1 blockade increases T-cell killing capacity",
"TGFb blockade reduces iCAF fraction" — well-known to QSP modelers
but not directly measured per-tumor in any published cohort.

To accommodate these without forcing fabricated CIs or tangential
citations, the `CalibrationTarget` schema in `maple` gains an
`epistemic_basis: Literal["literature", "mechanistic"]` field
(default `"literature"`):

- **literature** (default): existing behavior. Requires
  `primary_data_source` with DOI; `value_in_snippet` checks enforced.
- **mechanistic**: `primary_data_source` may be null. DOI/title/
  author/year/snippet validators skip. `key_assumptions` must
  document the mechanistic rationale and explain the chosen
  distribution width. Targets live in
  `calibration_targets/mechanistic/<scenario>/` to keep them
  visibly distinct from literature targets.

Constraints that keep this honest:

- **Mechanistic is not a backdoor for unverified citations.** If a
  paper exists, use `literature`. The `feedback_no_fabricated_citations`
  rule still binds.
- **Distributions must be deliberately wide.** Mechanistic targets
  nudge the posterior, not pin it. CI95 spanning ~OOM is normal.
- **Separate directory.** `calibration_targets/mechanistic/` is the
  canonical home, parallel to scenario directories. Reviewers can
  see at a glance which targets are literature-grounded vs intuition.
- **Audit visibility.** `parameter_audit.py` tags rows with
  `epistemic_basis` so PPC pass/fail at literature targets can be
  read separately from mechanistic targets.

Mechanistic targets that ship with this migration:

| Target file | Observable | CI95 | Replaces (Group D bound) |
|---|---|---|---|
| `mechanistic/clinical_progression/tumor_no_regression_mechanistic_PDAC_deriv001.yaml` | running `min_t(C1)/C1(0)` over 90d untreated | [1.0, 5.0] | `tumor_no_regression_tolerance` (also subsumes `tumor_dip_tolerance`, which has a looser 0.95 floor) |
| `mechanistic/clinical_progression/exhaustion_worsens_untreated_mechanistic_PDAC_deriv001.yaml` | `frac_exh(d90)/frac_exh(d0)` over 90d untreated | [1.0, 3.0] | `exhaustion_worsening_tolerance` |
| `mechanistic/clinical_progression/ecm_concentration_stability_mechanistic_PDAC_deriv001.yaml` | `(collagen/V_T)(d90) / (collagen/V_T)(d0)` over 90d untreated | [0.5, 1.5] | `ecm_concentration_stability` |
| `mechanistic/gvax_nivo_neoadjuvant/checkpoint_release_fold_mechanistic_PDAC_deriv001.yaml` | `k_C_Tcell_eff(d14)/k_C_Tcell_eff(d0)` under nivo | [2.0, 10.0] | `checkpoint_fold_increase_min` |

### Mechanistic targets parked behind infrastructure (not yet implemented)

These bounds map naturally to mechanistic-prior cal targets *but
require cross-scenario observable infrastructure* (Option B / Phase
D-2 — extension of `compute_test_statistics_batch` to accept multiple
sim_dfs and pass `reference_species_dicts` to observables). They
remain as inlined-threshold pytest assertions in
`test_qualitative_behaviors.py` / `test_qualitative_coverage.py`
until that infrastructure lands.

| Future target | Observable | Approx CI95 | Currently inlined as |
|---|---|---|---|
| `mechanistic/gvax_nivo_neoadjuvant/ecm_treatment_tolerance_mechanistic` | `(collagen/V_T)_treated(d21) / (collagen/V_T)_untreated(d30)` cross-scenario | [0.5, 1.5] | `ecm_tol = 1.5` in TestTreatmentVsUntreated (replaces `ecm_treatment_tolerance` Group D bound) |
| `mechanistic/<perturbation_scenario>/icaf_under_tgfb_blockade_mechanistic` (and 4 sibling perturbation→downstream targets) | downstream species ratio under blockade vs baseline | wide directional priors | TestDirectionalPerturbations — additionally needs the perturbation-scenario YAML primitive (currently MATLAB-only runtime overrides) |
| `mechanistic/<perturbation_scenario>/cascade_*` (4 targets) | coupled-pathway covariation under perturbation | wide directional priors | TestCascadeCoherence — same dependency on perturbation-scenario YAMLs |

Total deferred cohort: **~10 mechanistic targets** parked behind
Phase D-2 infrastructure. Each is identified by an inlined-threshold
or perturbation-scenario test that survives the migration as a
"model-wiring" assertion.

## End state vision

Three things remain:

1. **`calibration_targets/` directory** — distribution fits for literature
   observations. Includes:
   - Static-snapshot targets (existing) — TME densities, ratios at diagnosis
   - Treatment-endpoint targets (existing) — d21 GVAX/nivo observables
   - Multi-timepoint targets (new) — cascade ordering at d3/d7/d14/d21
   - Cross-scenario fold-change targets (new) — perturbation responses
   - Clinical-outcome targets (new) — RECIST CR/PR/SD/PD, ORR, natural-history tumor diameter, time-to-progression. DFS/OS deferred to clinical-backbone sprint (see Phase E)

2. **`scripts/parameter_audit.py` PPC report** — single source of truth for "did the model match literature?" Posterior-median PPC + tier-coverage statistics.

3. **Existing `evolve_to_diagnosis` restriction classifier** (unchanged) — the only hard prior gate, and a numerical necessity (likelihood is undefined when the simulator initialization fails). No new restriction-filter module is added in this migration.

Retired:
- `tests/qualitative_bounds.yaml` and `tests/qualitative_bounds.py`
- `tests/test_qualitative_bounds.py`
- `tests/scenarios/test_qualitative_behaviors.py`
- `tests/scenarios/test_qualitative_coverage.py`
- `tests/scenarios/_coverage.py` (coverage assertion helpers)
- ~132 qualitative tests

## Current state inventory

132 qualitative tests across 3 files:

| File | Tests | Role |
|---|---:|---|
| `tests/test_qualitative_bounds.py` | 19 | bounds YAML schema + snapshot checker tests |
| `tests/scenarios/test_qualitative_behaviors.py` | 68 | single-point checks on posterior-median sim |
| `tests/scenarios/test_qualitative_coverage.py` | 49 | coverage-tier checks on prior-predictive ensemble |

Disposition by category:

| Category | N | Destination |
|---|---:|---|
| Bounds-infra tests | 22 | DELETE with bounds YAML (Phase A) |
| Static-snapshot biology (TestBaselineTME, TestCellComposition, TestBaselineSnapshots) | 23 | DROP — already covered by cal target PPC (Phase A) |
| Multi-timepoint dynamics + ordering (TestProgressionDirection, TestTreatmentCascadeOrdering, etc.) | 25 | Partially migrate to revised Phase B (mechanistic ordering + paired/ratio cal targets) + Phase E1; remainder reframed as model-wiring guards (no source data exists for full retirement) |
| Hill function operating regimes (TestHillFunctionAudit + variants) | 7 | Migrate to restriction-filter module (Phase C) |
| Counterfactual perturbations (TestDirectionalPerturbations, TestCascadeCoherence) | 9 | Migrate to cross-scenario fold-change cal targets (Phase D) |
| Regression guards (TestGvaxCascadeFixes single-point + coverage versions) | 11 | Migrate to cross-scenario or multi-timepoint cal targets (Phase D) |
| Treatment time series + monotonicity | 13 | Some migrate to multi-timepoint cal targets (Phase B), some to Phase D |
| Misc independent (prereqs, parameter sensitivity smoke, VEGF clinical range) | 6 | DELETE or migrate as appropriate |
| Treatment-vs-untreated comparisons | 17 | Multi-scenario fold-change cal targets (Phase D) |

Total: 22 + 23 + 25 + 7 + 9 + 11 + 13 + 6 + 17 = 133 (rounding from earlier
counts; minor double-counts across single-point + coverage variants of the
same biology).

## Phase A — bounds YAML retirement (1-2 days)

**Status**: ready to start.

### Steps

1. Drop 14 Group A bounds from `tests/qualitative_bounds.yaml` (redundant with existing cal targets) plus 2 Group B bounds (`caf_cancer_ratio_range`, `tumor_diameter_range`).
2. Promote 1 Group B cal target:
   - `tumor_diameter_baseline_PDAC_deriv001.yaml` — observable tumor diameter at diagnosis from V_T, Tran 2022 (n=315) Table 2 direct measurements (median 30 mm, IQR 23-35 mm). Moved out of `excluded/` and renamed.
   - `caf_cancer_ratio_range` is dropped without replacement: no primary paper directly reports per-tumor CAF:cancer-cell ratio, and `stromal_fraction_PDAC_deriv001` + the CAF subtype targets + the cellularity reference constant already implicitly constrain the ratio. Constructing a vague [1, 5] derived target would not add likelihood signal.
3. Delete the 23 static-snapshot tests (TestBaselineTME, TestCellComposition, TestBaselineSnapshots).
4. Inline thresholds in 25 KEEP/INLINE tests that use `BOUNDS[...]` purely as a numerical-threshold registry. (Some of these will be migrated in Phase B/D and deleted entirely; inlining is a temporary compatibility step.)
5. Delete `tests/test_qualitative_bounds.py` and `TestSnapshotBoundsChecker`.
6. Delete `tests/qualitative_bounds.yaml` and `tests/qualitative_bounds.py` once no test imports remain.
7. Re-run `parameter_audit.py`; verify the 2 new Group B targets appear in PPC.

### Deliverables

- 16 bounds removed (14 Group A + 2 Group B)
- 1 new cal target (tumor_diameter_baseline; promoted from excluded/)
- 23 tests deleted
- 22 tests deleted (bounds infra)
- ~25 tests inlined-threshold (temporary)
- bounds YAML + py + tests file gone

### Estimated test count after Phase A

132 → 87 (45 deletions: 22 infra + 23 PPC-covered).

## Phase B — mechanistic-prior cascade-ordering targets (~half day)

**Status**: re-scoped twice on 2026-04-27 after Heumann supp recon. Final scope is much smaller than the original plan or the first reframe.

### Two failed assumptions

**First assumption (original plan)**: multi-timepoint d3/d7/d14 cascade-ordering targets sourced from literature. **Dead** because no PDAC immunotherapy trial samples patients between priming and surgery. All three NCT02451982 papers (Zheng 2021, Li 2022, Heumann 2023) collect mIHC at a single surgery time point (~d14). The MEMORY note "Heumann supp S3/S4/S5 paired targets" referred to per-patient paired *biomarkers* (Sup Tables 3/4/5 stratify outcomes by biomarker), not paired timepoints.

**Second assumption (first reframe)**: even without time-series data, Heumann's per-patient mIHC sheet would supply higher-n single-marker and paired/ratio targets (B1 + B2) at d~14. **Also dead** because Heumann's mIHC is **TLA-specific** — every column except col 14 measures cell fractions inside lymphoid aggregates. Li 2022 measured **non-LA bulk tumor** (the ~90% compartment that maps to V_T). The two studies sample different tissue compartments. Heumann's TLA data would only be calibration-relevant once a TLA / B-cell module exists in the QSP model — planned but not in this sprint.

### Final scope: B3 only (mechanistic-prior cascade-ordering)

Encode the temporal-ordering claim ("mAPC peak precedes CD8 peak precedes tumor regression") under the `epistemic_basis: mechanistic` schema (no literature data, deliberately wide CIs, lives in `calibration_targets/mechanistic/`). These are the only B-phase targets that can ship today.

Candidates:

- `mapc_peak_before_cd8_peak_mechanistic` — observable returns `argmax_t(mAPC_total_T) - argmax_t(CD8)` over the neoadjuvant window; CI95 `[-14, -1]` days (mAPC peaks before CD8 peaks)
- `cd8_peak_before_tumor_min_mechanistic` — `argmax(CD8) - argmin(C1)`; CI95 `[-21, -1]` days
- (optional) `mapc_rise_precedes_cd8_rise_mechanistic` — fold-change ordering at d7 vs d14

These compose with the existing GVAX / GVAX+nivo scenarios; no new scenarios, no new infra.

### Estimated work

~3-4 new mechanistic cal target YAMLs. ~half day. Schema already supports.

### Tests retired by Phase B

- TestTemporalDynamics (3) — covered by B3 mechanistic ordering
- TestProgressionDirection (2) — already covered by mechanistic `tumor_no_regression` cal target shipped with Phase A
- module-level test_species_increases_d0_to_d90 + xfail (2) — covered by mechanistic monotonicity targets

Tests that *would have been retired* by B1/B2 if the data had been usable, but instead survive as model-wiring assertions (no source data exists):

- TestTreatmentCascadeOrdering (5 single-point + 3 coverage)
- TestTreatmentTimeSeriesComparison (5 + 5 coverage)
- TestProgressionTimeSeries (3) — folds into Phase E1 instead
- TestTreatmentMonotonicity (3 coverage)

≈ 7 tests retired (vs the original 31). Test count 87 → ~80.

### Heumann supp inventory (for reference)

The Heumann supp PDFs/xlsx files in `notes/data-extraction/41467_2023_*` deliver:

- `MOESM4 :: Efficacy Cohort` — per-patient DFS/OS, age, tumor size, adjuvant chemo regimen (n=40). **Routes to Phase E2**, gated on clinical-backbone sprint.
- `MOESM4 :: mIHC data` — per-patient TLA-specific mIHC (n=40, 13 markers). **Not extractable today** without a TLA module.
- `MOESM4 :: Matched Cohort` — historical-control matching for Arm C urelumab.
- `MOESM6 :: Sup Tables 3/4/5` — biomarker stratifications of OS/DFS. **Routes to E2.**
- `MOESM6 :: Sup Fig 3-9` — KM curve underlyings, scatter plots. **Routes to E2.**

Once a TLA / B-cell module lands, the Heumann mIHC sheet becomes the high-priority data source for calibrating that compartment — flagged for the future TLA-module sprint.

## Phase C — delete Hill audit tests (10 min)

**Goal**: drop the 9 Hill audit tests. No replacement.

### Why no filter

The original plan called for porting the categorical predicates
(`graded` ⇒ in [1/(1+5^n), 1−1/(1+5^n)]; `switch_high` ⇒ > 0.5;
`constitutive_pdac` ⇒ > 0.8) into a `hill_audit_filter` and
AND-ing it into the restriction classifier's label. Two reasons that
plan was rejected:

1. **Framework multiplication.** The migration's whole goal is
   collapsing three validation layers (cal-target PPC + bounds YAML +
   qual tests) into one (cal-target PPC). Adding a *fourth* layer
   (restriction-filter categorical predicates) reintroduces the
   problem the migration was designed to solve.
2. **Methodological framing was wrong.** "Saturated Hills are
   unidentifiable, so filter them out" is a training-efficiency
   argument, not a statistical-correctness one. The posterior already
   handles unidentifiable parameters by giving them wide marginals;
   PRCC + the audit's `contraction_local_shell` flag them.
   Saturation/dead-zone Hills are diagnostic information, not prior-
   support violations.

The audit tests themselves were diagnostic, not load-bearing: the
`_GRADED_XFAIL` / `_SWITCH_XFAIL` carve-outs admit we couldn't agree
on the categorical thresholds in the first place. Better diagnostic
in the same direction would be a single PPC-adjacent script that
prints Hill activation values across posterior draws and flags
saturated/dead axes — out of scope for this migration; deferrable
to post-Phase-D.

### Tests deleted by Phase C

- `TestHillFunctionAudit` (4 single-point + 3 coverage)
- `TestTreatmentHillFunctions` (1)
- `TestHillFunctionIQR` (1)
- Plus their supporting helpers (`HILL_AUDIT` registry,
  `_sensitive_range`, EC50 prior loaders, `_HILL_COVERAGE_XFAIL`).

= 9 tests retired. Test count 56 → 47.

## Phase D — counterfactual + regression guards as cross-scenario cal targets (1-3 weeks)

**Goal**: fold counterfactual perturbation tests (TGFb blockade, CXCL12 blockade, etc.) and historical regression guards (cDC1 antigen transfer, niche-scaled CAF carrying capacity) into the calibration target framework.

### Required infrastructure

#### 1. Perturbation scenario YAMLs

New scenarios in `scenarios/`:

- `tgfb_blockade_baseline.yaml` — sets `f_active_TGFb` (or equivalent TGFb-availability parameter) to ~0
- `cxcl12_blockade_baseline.yaml` — disables CXCL12 secretion
- `il10_blockade_baseline.yaml` — disables IL10 production / signaling
- `ecm_blockade_baseline.yaml` — disables collagen secretion
- `ifng_blockade_baseline.yaml` — disables IFNg secretion / signaling
- (and corresponding treatment-on perturbation scenarios if regression guards require them)

Each is mechanically a copy of `baseline_no_treatment.yaml` with one parameter overridden. Authoring is fast; the engineering question is whether the scenario YAML schema supports parameter overrides cleanly (it should — most scenarios already use variants).

#### 2. Cross-scenario observables — RESOLVED 2026-04-27 (Option D3+)

**Decision**: see `notes/calibration/cross_scenario_observable_spike.md`
for the full design spike. Original D1/D2 framing was superseded by
**Option D3+** (post-process composer over the aligned scenario_meta
dict, with a timeseries-capable variant). D3+ avoids per-scenario
cal-target schema changes, doesn't invalidate the test-stats CSV cache
key, and ships in ~5 days instead of D2's ~1-2 weeks.

**Status (2026-04-27):**
- ✅ maple `CrossScenarioCalibrationTarget` schema landed (popellab/maple#26)
- ✅ qsp-hpc-tools loader + composer worker landed (jeliason/qsp-hpc-tools#62)
- ✅ pdac-build sbi_runner wiring (scalar inputs end-to-end) + first
  canonical target landed (#81). `tumor_diameter_fold_d90_gvax_vs_untreated`
  composes two existing within-scenario mechanistic priors into a
  cross-scenario directional rail.
- ⏳ **#82** — Timeseries-input HPC plumbing. Composer worker supports
  timeseries; sbi_runner raises `NotImplementedError` at compose time
  until `run_hpc()` ferries sim_dfs back alongside x. ~1-2 days, mostly
  qsp-hpc-tools side. Doesn't block the scalar cohort.
- ⏳ **#83** — C++ simulator parameter-override support. Unblocks the 5
  perturbation scenarios (tgfb/cxcl12/il10/ecm/ifng blockade) and ~10
  deferred mechanistic counterfactual targets. ~3-4 days for the
  override mechanism + ~1-2 days authoring scenarios and cal targets.
- ⏳ **#84** — Author additional cross-scenario cal targets. Tracking
  issue with candidate lists across scalar (available today),
  timeseries (gated on #82), and perturbation (gated on #83) cohorts.

#### 3. Cross-scenario cal target YAMLs

For each counterfactual / regression guard test, author a cal target with:
- Scenario list (e.g., `[baseline_no_treatment, tgfb_blockade_baseline]`)
- Observable code: ratio or fold-change between the two scenarios at same time
- Empirical data: literature-grounded vague distribution (e.g., median 2.0 with CI95 [1.2, 5.0] for "blockade increases iCAF fraction")
- Source: literature paper for the counterfactual claim

Estimate: ~25-30 new cal target YAMLs.

### Tests deleted by Phase D

- TestTreatmentResponse (3) — Hill function relational checks
- TestGvaxCascadeFixes (6 + 5 coverage)
- TestDirectionalPerturbations (5)
- TestCascadeCoherence (4)
- TestTreatmentVsUntreated (5 + 4 coverage)
- TestProgressionTolerances (4 coverage)
- TestTreatmentResponseCoverage (3 coverage)
- TestQualitativeBehaviorPrereqs (4) — fixture file existence; no longer needed once tests file deletes
- TestParameterSensitivity (1)
- TestClinicalSpeciesRanges (1)
- module-level test_n_valid_meets_floor + test_mycaf_dominates_icaf_at_d0 (2)

≈ 47 tests retired. Test count 47 → 0.

After Phase D: tests/scenarios/test_qualitative_behaviors.py and
test_qualitative_coverage.py and _coverage.py all delete.

## Phase E — clinical outcomes as cal targets (1 week now; rest deferred)

**Goal**: convert the prediction-only YAMLs in
`calibration_targets/predictions/<scenario>/` into proper
CalibrationTargets so RECIST / ORR / time-to-progression / tumor
diameter participate in the likelihood, not just PPC sanity. Bring
clinical outcomes under the unified framework.

**Constraint**: the surgery + adjuvant-FFX confound analysis in
`notes/calibration/clinical_outcomes_calibration_scoping.md` (2026-04-22)
stands. Fitting Heumann-style DFS/OS to the current
`*_longterm` scenarios — which model neither resection nor adjuvant
chemo — would force QSP immune-potency parameters to absorb survival
benefit that mechanistically belongs to the surgery and chemo
backbone. So Phase E is split:

- **E1 (now, ~1 week)**: low-confound subset only. Untreated natural
  history + pre-surgery RECIST windows. ~12-15 new cal targets.
- **E2 (deferred)**: treated DFS/OS and post-surgery endpoints. Stays
  prediction-only until the clinical-backbone sprint
  (surgery event + lumped adjuvant FFX, ~3-4 weeks per scoping note)
  unlocks them.

### E1 — confound-light conversions (do now)

#### Untreated natural-history (no confound)

Scenario: `clinical_progression_longterm` (no surgery, no adjuvant — by
design models untreated PDAC).

| Prediction-only YAML | New empirical anchor | Candidate source |
|---|---|---|
| `recist_cr_at_day_90.yaml` | ~0% (untreated PDAC essentially never CRs at day 90) | Surveillance-window literature; van Veldhuisen 2020 staging-change cohorts |
| `recist_pr_at_day_90.yaml` | ~0-2% | same |
| `recist_sd_at_day_90.yaml` | majority fraction (e.g. 50-70%) | same |
| `recist_pd_at_day_90.yaml` | substantial fraction (e.g. 25-45%) | same |
| `tumor_diameter_day_365.yaml` | follows VDT trajectory; lognormal anchored on Ahn 2017 VDT and 90-day fraction surviving | derived from Ahn 2017 + natural-history mOS |
| `tumor_diameter_day_730.yaml` | strongly truncated by death threshold; vague distribution | same; effectively a sanity bound on growth law |
| `time_to_progression.yaml` | mTTP ~3-6 months for untreated locally advanced/metastatic PDAC | Conroy 2011 placebo-equivalent arms; consider untreated-cohort data |

These constrain `k_C_growth`, the growth-law saturation, and the
diameter-from-volume mapping — all already weakly constrained by Ahn
2017 `tumor_doubling_time` (existing). Adding the RECIST distribution
forces the natural-history trajectory to match the *category fractions*,
not just the mean VDT.

#### Pre-surgery / non-resected ORR — DROPPED (decision 2026-04-27)

The original plan called for 4-8 vague-prior treated ORR cal targets
on the `gvax_neoadjuvant_longterm` / `gvax_nivo_neoadjuvant_longterm`
scenarios, anchored on Lutz 2011 / Le 2015 / Wu 2024 metastatic-PDAC
trials with high translation_sigma. **Dropped** for two reasons:

1. **Population mismatch is too large.** Available trial data is
   metastatic, multi-line-pretreated, performance-status-degraded,
   often combined with a chemo backbone. The simulator scenarios are
   resectable, treatment-naive, immunotherapy-alone counterfactuals
   that don't appear in any single trial. Cranking translation_sigma
   wide enough to absorb that drift is just hiding the fact that the
   literature isn't really informing the prior. No published QSP-IO
   work I'm aware of calibrates against trial data from a
   substantially different population this way.
2. **The mechanistic-prior schema already covers the directional
   claim.** "Treated d90 fold should be at most comparable to
   untreated, not dramatically reduced (PDAC is cold)" is a
   biological invariant, not a population estimate.

**Replacement** (option 2 from the 2026-04-27 discussion): two
mechanistic-prior cal targets shipped with this update, mirroring
the within-scenario `tumor_diameter_fold_d90_untreated` target on
each treated longterm scenario:

- `mechanistic/gvax_neoadjuvant_longterm/tumor_diameter_fold_d90_gvax_mechanistic_PDAC_deriv001.yaml`
  — median 1.05, CI95 [0.80, 1.35], directional "GVAX provides at
  most modest growth suppression."
- `mechanistic/gvax_nivo_neoadjuvant_longterm/tumor_diameter_fold_d90_gvax_nivo_mechanistic_PDAC_deriv001.yaml`
  — median 1.00, CI95 [0.75, 1.30], directional "checkpoint blockade
  adds incremental benefit on top of GVAX priming."

**Prediction-only YAMLs stay where they are.** The existing
`calibration_targets/predictions/gvax_*_longterm/recist_*.yaml` continue
to ride the OBED prediction pipeline as PPC sanity rails. They are
not promoted to CalibrationTargets; the audit reads them as
"does the posterior produce sane treated-arm RECIST distributions?"
without claiming the literature constrains the prior. Real ORR
calibration is gated on the clinical-backbone sprint (E2) when
scenarios actually match the published trial designs.

### E2 — survival + post-surgery endpoints (deferred)

Stay prediction-only until the clinical-backbone sprint:

- `os_at_{12,24,36}mo.yaml` for all three longterm scenarios
- `dfs_at_{12,24}mo.yaml` for the two treated longterm scenarios
- Post-surgery `time_to_progression.yaml` for treated scenarios

Sources to anchor once the backbone exists:
- PRODIGE 24 (mFFX adjuvant) — anchors `k_adjuvant_kill`
- Heumann 2023 arms — once surgery + FFX nuisances are in the model,
  the immunotherapy increment is identifiable
- Zheng 2021 historical adjuvant-chemo control — cross-cohort anchor

The scoping note (2026-04-22) recommends a 3-4 week clinical-backbone
sprint covering: surgery event, lumped adjuvant FFX, neoadjuvant FFX +
SBRT proxy, gem-nabP. Phase E2 follows that sprint.

### E1 work breakdown — STATUS: complete (2026-04-27)

1. ✅ Untreated natural-history mechanistic targets shipped:
   `clinical_progression_longterm/{tumor_diameter_fold_d90_untreated,
   tumor_diameter_day_365, tumor_diameter_day_730,
   time_to_progression}_mechanistic_PDAC_deriv001.yaml`. The 4 binned
   RECIST one-hot targets in the original plan were replaced by 1
   continuous d90 diameter fold-change target (information-preserving;
   smooth NPE observable). See PR #79.
2. ✅ Treated directional mechanistic priors shipped on each treated
   longterm scenario:
   - `gvax_neoadjuvant_longterm/tumor_diameter_fold_d90_gvax_mechanistic`
   - `gvax_nivo_neoadjuvant_longterm/tumor_diameter_fold_d90_gvax_nivo_mechanistic`
3. ✅ Treated-ORR vague-prior cal targets dropped (see "Pre-surgery /
   non-resected ORR" section above for rationale). Prediction-only
   YAMLs in `calibration_targets/predictions/gvax_*_longterm/` retain
   their PPC-rail role.
4. Pending: re-run `parameter_audit.py` once the next stage-2 promotion
   is ready, to verify the new targets surface in PPC and don't
   dominate the likelihood.
5. Pending: spot-check that the natural-history d90 distribution is
   recoverable by the existing untreated trajectory — if not, that's a
   structural bug to fix, not a calibration issue.

### Tests deleted by Phase E

None directly — clinical outcomes are not in the qualitative-test
suite today (they were only ever in the prediction-only OBED pipeline).
Phase E is net-additive to the cal target inventory.

### Estimated cal targets added — actual numbers

- E1 untreated natural-history: 4 mechanistic-prior cal targets (the
  4-binned-RECIST + 3-continuous projection collapsed to 1 continuous
  fold-d90 + 2 long-horizon diameter rails + TTP).
- E1 treated directional: 2 mechanistic-prior cal targets (1 per
  treated longterm scenario).
- **E1 total: 6 new cal targets** (vs the 12-15 originally projected;
  the gap is the dropped treated-ORR vague-prior cohort).
- E2 (deferred): ~12 additional after clinical-backbone sprint.

## Phase ordering and dependencies

```
Phase A (bounds) ─┐
                  │
                  ├─ independent of B/C/D/E1 ─→ ship first
                  │
Phase B (multi-tp)│ ─ no architectural change required
                  │
Phase C (Hill)    │ ─ touches restriction classifier; independent of B
                  │
Phase D (xscene)  │ ─ requires schema decision (D1 vs D2);
                  │   also requires Phases A+B done so cal target
                  │   inventory is stable
                  │
Phase E1 (clin)   │ ─ independent of A-D; can run in parallel
                  │
Phase E2 (surv)   │ ─ blocked on clinical-backbone sprint
                  │   (surgery + lumped FFX modules)
```

Critical path: A → B → D. Phase C can run in parallel with B. Phase
E1 can run in parallel with any of A-D since it operates on a separate
scenario family. Phase E2 is gated on the clinical-backbone sprint
(out of scope for this migration).

## Open design decisions

1. **D1 vs D2 (cross-scenario observable architecture)** — needs a
   1-2 day design spike before Phase D starts. The decision affects
   how 25-30 cal targets are authored and how the simulator
   parallelizes.

2. **Restriction filter granularity** — the Hill audit filter could be
   binary (any saturation rejects) or graded (count of saturated Hills
   contributes to a soft score). Binary is simpler; graded gives the
   classifier more signal. Default to binary; revisit if the prior is
   over-restricted post-Phase-C.

3. **Translation_sigma for vague cal targets** — Phase B and Phase D
   add many new cal targets with literature-grounded but vague
   distributions (CI95 spans an order of magnitude or more). The
   translation_sigma mechanism is designed for in-vitro→in-vivo drift;
   it may need re-tuning for these "literature inequality" targets to
   avoid them dominating the likelihood with their wide tails. Audit
   per-source-translation-sigma after Phase B/D.

4. **Should we also fold the existing submodel targets into a unified
   cal target schema?** Out of scope for this migration but worth
   asking: the SubmodelTarget schema is parallel to CalibrationTarget,
   and the audit handles both. They could plausibly unify. Defer to a
   later sprint.

5. **Phase E1 untreated RECIST anchors.** ~~Original wording deprecated.~~
   Resolved 2026-04-27: shipped as a single continuous d90 diameter
   fold-change observable rather than 4 binned RECIST one-hot targets
   (information-preserving; smoother NPE likelihood). CI95 derived
   from the Ahn 2017 VDT distribution rather than RECIST category
   fractions, since the latter has no canonical PDAC source.

## Risks

- **Phase D engineering creep**: cross-scenario observable architecture
  could expand if it reveals issues in the cpp simulator or the audit
  pipeline. Budget +1 week buffer.
- **Sim cost increase**: cross-scenario cal targets multiply simulation
  cost per theta. With ~30 perturbation cal targets and ~5 unique
  perturbation scenarios, sim cost roughly doubles. The simulation pool
  cache helps mitigate but expect retraining cycles to be slower.
- **Vague-cal-target overfit**: if many new cal targets have vague CI95s
  (~OOM), they may not constrain the posterior much but still add
  computational overhead. Worth a sensitivity check after Phase B.
- **Test coverage gap**: deleting all qualitative test files removes a
  layer of CI-level model validation. Cal target PPC + restriction
  filters are the replacement; if either fails to catch regressions
  that the qualitative tests caught, we'd find out late. Mitigation:
  during each phase, run `parameter_audit.py` and verify previously-flagged
  pathologies (e.g., the M2/M1 cascade) still surface in PPC failures.
- **Clinical-outcome confound bias (Phase E)**: even E1's vague treated-ORR
  targets risk attributing ORR signal to the wrong QSP parameters when
  the underlying trial design includes surgery / FFX that the scenario
  doesn't model. Mitigation: keep treated-ORR translation_sigma high
  in E1, treat E1 as primarily a *natural-history* fit with
  treated-ORR as PPC-adjacent rails, and gate any tightening on the
  clinical-backbone sprint completing.

## Success criteria

The migration succeeds when all of the following hold:

1. `tests/qualitative_bounds.yaml` is deleted.
2. `tests/scenarios/test_qualitative_behaviors.py` is deleted.
3. `tests/scenarios/test_qualitative_coverage.py` is deleted.
4. `parameter_audit.py` PPC report covers ≥ 95% of the literature
   observations previously encoded as bounds + qualitative test
   thresholds.
5. The restriction classifier training accepts the new Hill audit
   filter without unduly restricting valid prior draws (>50% acceptance
   rate on prior).
6. End-to-end NPE training on the unified cal target set produces a
   posterior whose median simulations satisfy the structural-correctness
   properties previously checked by qualitative tests (verified by
   spot-checking ~10 randomly-selected ex-qual-test predicates).

## References

- Methodology debate (path A vs metric tuning): `notes/calibration/cd8_treg_fix_proposal.md`
- Structural Treg-CD8-M2-APC vortex (motivating example): `notes/calibration/structural_treg_vortex.md`
- Stage 1 value analysis (composite prior is net-negative): `notes/calibration/stage1_value_analysis.md`
- k_PSC_to_apCAF audit overfit (parameterizer artifact case study): `notes/calibration/k_PSC_to_apCAF_audit_overfit.md`
- Clinical-outcomes calibration scoping + surgery/FFX confound: `notes/calibration/clinical_outcomes_calibration_scoping.md`

## Running stage 2 with the C++ + HPC backend

The canonical stage-2 entry point is `workflows/sbi_runner.py` (CppSimulator + HPCJobManager). It uses the in-repo cpp/ tree (post-PR-#46) on HPC.

### One-time setup

1. **Clone pdac-build on HPC** at `cpp.repo_path` (per the project-local `.qsp-hpc/credentials.yaml` override):
   ```bash
   ssh hpc 'cd /home/jeliaso2/qsp-projects && \
            git clone git@github.com:jeliason/qspio-pdac.git pdac-build'
   ```
2. **Authorize the HPC SSH key for the popellab GitHub org** (one-time UI step; popellab/qsp-codegen has SAML SSO enforced). Settings → SSH keys → Configure SSO → authorize popellab. Without this, `ensure_cpp_binary` fails to clone qsp-codegen during the cmake build.
3. **Project-local credentials** at `pdac-build/.qsp-hpc/credentials.yaml` (gitignored) overrides the global config to point cpp at the in-repo layout:
   ```yaml
   cpp:
     repo_path: /home/jeliaso2/qsp-projects/pdac-build
     sim_subdir: cpp/sim
     binary_path: /home/jeliaso2/qsp-projects/pdac-build/cpp/sim/build/qsp_sim
     template_path: /home/jeliaso2/qsp-projects/pdac-build/resources/cpp/param_all.xml
     branch: main
   ```
   These keys all default to pdac-build conventions in qsp-hpc-tools `>= main @ <PR-#58 + #59 merge>`, so the override is mostly redundant but explicit.

### Pre-launch checklist

Before each launch:

```bash
make hpc-status         # confirm HPC repo branch/SHA matches main, no drift warnings
make refresh-cpp        # rebuild local cpp/sim/build/qsp_sim if model changed
make check-cpp-sync     # priors CSV ↔ cpp XML ↔ species_units.json
```

A model edit (parameters_PDAC.m, core/modules/*.m) requires `make refresh-cpp` to regenerate cpp/qsp/ode/ + resources/cpp/param_all.xml + the local binary. Bump `MODEL_VERSION` in `sbi_runner.py` so a fresh sim pool is keyed off the new model.

### Launch

```bash
# Smoke run (CSV-only prior, 100 sims, all params, fastest path)
# (classifier filter is now on by default — see Cfg.restriction_classifier_dir)
python -m workflows.sbi_runner \
    --skip-submodel-sync \
    --num-total-simulations 100 \
    --no-prcc-tier

# Production run (composite prior, 5k sims, default PRCC tier, classifier on)
python -m workflows.sbi_runner

# To opt out of restriction classifier (unfiltered prior — slower training, more NaNs):
python -m workflows.sbi_runner --no-restriction-classifier

# PRCC-tiered subsetting + custom NPE capacity
python -m workflows.sbi_runner --prcc-tier 1 --hidden-features 192 --num-transforms 9
```

Sweeps over NPE hyperparameters live under `workflows/sweep/` (scripts: `run_sweep.sh`, `generate_configs.py`, `collate_sweep.py`). Delta GPU offload for NPE training is in `workflows/delta/` (`run_delta_training.sh`, `run_delta_sampling.sh`).

### What `sbi_runner` does on each launch

1. Builds the prior — composite (load_composite_prior_log: submodel YAML + CSV) by default, or CSV-only with `--skip-submodel-sync`.
2. Initializes a `CppSimulator` per scenario (4 active per `DEFAULT_SCENARIOS`: `baseline_no_treatment`, `clinical_progression`, `gvax_neoadjuvant_zheng2022`, `gvax_nivo_neoadjuvant_zheng2022`). Each scenario's cal-target dirs are now lists (literature dir + mechanistic-prior dir where present).
3. Calls `ensure_cpp_binary` on HPC: git fetch + ff main on `cpp.repo_path`; pip-install qsp-codegen into `hpc_venv_path`; cmake configure (with `-DPython3_EXECUTABLE` pointing at the venv) + cmake build qsp_sim. First build is ~3-5 min (SUNDIALS + yaml-cpp fetch); incremental builds are ~zero cost.
4. Submits SLURM array jobs per scenario; theta pool drawn from the prior, rejection-sampled by the restriction classifier.
5. Aggregates per-scenario observables across the joint NaN filter, transforms via Gaussian copula, trains NPE.
6. Writes `figures/sbi_<scenarios>_<tier>/` with diagnostics, posterior samples, and the `cfg.json` snapshot.

### Stage 1 (when you want it back)

`--skip-submodel-sync` bypasses the static composite prior. To use it instead, drop the flag — the workflow loads `calibration_targets/submodel_targets/submodel_priors.yaml` (the cached output of stage-1 inference). To regenerate that yaml, run `python scripts/parameter_audit.py` which invokes `qsp_inference.audit.report.run_audit → run_comparison` to do component-wise NPE inference on each submodel target.

## Next steps

Migration phases not yet shipped:

- **Phase B (twice-revised)** — DONE. 3 cascade-ordering targets shipped 2026-04-27 (commit 0553fab: `mapc_peak_precedes_cd8_peak` for both treated arms + `tumor_min_follows_cd8_peak_gvax_nivo`). Arm A parity completion `tumor_min_follows_cd8_peak_gvax_mechanistic` shipped 2026-04-28, closing the priming-cascade-of-three on both treated scenarios. Optional `mapc_rise_precedes_cd8_rise` not pursued — the peak-time targets already encode the same ordering claim with cleaner geometry. Heumann mIHC sheet remains parked for the future TLA / B-cell module sprint.
- **Phase E1** — DONE (2026-04-27). Shipped 4 untreated natural-history mechanistic cal targets on `clinical_progression_longterm` (PR #79) plus 2 treated directional mechanistic priors on `gvax_neoadjuvant_longterm` / `gvax_nivo_neoadjuvant_longterm`. Treated-ORR vague-prior cohort dropped (population mismatch too large for translation_sigma to honestly bridge); prediction-only YAMLs retain their PPC-rail role. E2 (DFS/OS for treated arms) stays gated on the clinical-backbone sprint.
- **Phase D-2** (full retirement of the surviving qual tests) — parked. Requires (a) C++-simulator-level parameter-override support for the 5 perturbation scenarios currently as MATLAB-only runtime overrides, and (b) cross-scenario observable architecture in qsp-hpc-tools (`compute_test_statistics_batch` accepting multiple `sim_dfs`). Both ~3-4 days qsp-hpc-tools work each. Decide whether to do this when the "model wiring tests as a parallel layer" cost actually starts biting.
- **Mechanistic targets queued for Phase D-2**: 1 cross-scenario (`ecm_treatment_tolerance`) + 5 directional-perturbation + 4 cascade-coherence targets, all currently inlined as pytest assertions. Listed in the "Mechanistic targets parked behind infrastructure" table.

Infrastructure cleanups:

- **`scripts/hpc_status.py`** still hardcodes the `SPQSP_PDAC` label even when `cpp.repo_path` points at pdac-build. Cosmetic; rename the display string to read from the configured path.
- **Decommission HPC SPQSP_PDAC checkout** at `/home/jeliaso2/SPQSP_PDAC` once stage-2 runs from pdac-build's in-repo cpp/ are verified end-to-end. Free space, eliminate the legacy build path entirely.
- **MATLAB rsync codepath** in qsp-hpc-tools (`sync_codebase`) is no longer used by C++ jobs. It still serves any remaining MATLAB workflows. Track which workflows still use it; deprecate when none remain.
- **Binary-in-hash issue** ([qsp-hpc-tools #56](https://github.com/jeliason/qsp-hpc-tools/issues/56)) — `compute_pool_id_hash` doesn't include the C++ binary, so a model-edit-without-MODEL_VERSION-bump silently serves stale draws. Track manually for now.

Calibration follow-ups:

- **Wider mechanistic-prior coverage** — the 4 mechanistic targets shipped are placeholders for the Group D bounds. As the migration matures, candidates for more mechanistic priors will surface from PPC failures the literature targets can't explain.
- **Stage-1 / submodel_priors.yaml refresh** — `submodel_priors.yaml` was generated before this migration and may not reflect the latest submodel targets. Re-run `python scripts/parameter_audit.py` (with `--invalidate <param>` if needed) when ready to update the static composite prior.

## Branch state at plan creation

- Current branch: `ci/skip-slow-by-default`
- Behind origin/main by 3 commits
- Unstaged: `tests/qualitative_bounds.{py,yaml}` (the 2 added bounds — both Group A drops in Phase A)
- This plan and the accompanying audit note are net-additive notes; safe to commit independently of the implementation work.
