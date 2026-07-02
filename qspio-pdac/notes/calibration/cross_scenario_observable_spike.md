---
status: design spike (Phase D-2 prerequisite)
date: 2026-04-27
author: Claude (with Joel)
owner_decision_required: yes
supersedes_section: qualitative_to_calibration_migration_plan.md §"Phase D – Cross-scenario observables — DESIGN DECISION" (the original D1/D2 framing)
revision: 2026-04-27 v2 — D3 extended to D3+ (timeseries-capable); owner decisions resolved; D2 rejected outright (subsumed by D3+)
implementation_status: 2026-04-27 v3 — schema (popellab/maple#26 merged) + loader+composer (jeliason/qsp-hpc-tools#62 merged) + scalar-input sbi_runner wiring + 1 canonical target shipped on pdac-build feat/cross-scenario-observables. Timeseries-input HPC plumbing (run_hpc returning sim_dfs alongside x) is the next sub-task; the composer worker fully supports timeseries inputs but sbi_runner refuses them at compose time until that bridge exists.
---

# Cross-scenario observable design spike

## TL;DR

**Recommendation: Option D3 (post-process composer over the aligned scenario_meta dict), with a timeseries-capable variant (D3+) that handles the same use cases as D2.** D3 emerges from re-reading the actual sbi_runner code: scenarios are *already* sample_index-aligned and concatenated row-for-row before the joint NaN filter. A cross-scenario observable is therefore a function over the aligned `{scen_name: x_scen}` dict (scalar inputs) or over the aligned `{scen_name: sim_df}` dict (timeseries inputs), evaluated **after** per-scenario `compute_test_statistics_batch` has run. No simulator-side change, no per-scenario cal-target schema change. ~3-4 days of plumbing for scalar-only D3; ~5 days for the timeseries-capable D3+. Detailed cost / risk comparison against D1 and D2 below.

D2 is rejected once D3+ is on the table: D3+ handles every observable shape D2 handles (including timeseries-level cross-scenario compositions), at a fraction of the engineering cost and without invalidating the test-stats CSV cache key or touching maple's calibration_target schema.

D1 (simulator-side derived species; run blocked + unblocked inside one sim) is rejected: it doubles per-sim cost, doesn't fit cleanly into the C++ scenario-YAML model, and the "derived species" abstraction is awkward when the perturbation is a parameter override rather than a species transformation.

## Why this spike is needed

The original migration plan (qualitative_to_calibration_migration_plan.md §"Phase D — Required infrastructure 2") proposed two options:
- **D1**: simulator-side derived species (one combined counterfactual sim emits ratio observable)
- **D2**: cal-target schema extension (observable receives a dict of `{scenario: species_dict}`)

…and recommended D2 with "~1-2 weeks engineering". That estimate predated the actual scenario-alignment plumbing that landed in sbi_runner.py during Phase B (see commit `0553fab` and the `shared_idx` block at sbi_runner.py:688-720). With that alignment in place, a third option opens up that the original framing missed.

## What we actually need a cross-scenario observable to do

The Phase D-2 cohort (~25-30 cal targets) has two flavors:

1. **Counterfactual fold-changes**: "iCAF fraction with TGFb blockade vs baseline" — observable = ratio of iCAF density between two scenarios at the same time point and same θ.
2. **Treatment-vs-untreated comparisons**: "tumor diameter at d90 with GVAX vs without" — observable = ratio across `gvax_neoadjuvant_longterm` and `clinical_progression_longterm` at the same θ.

Both reduce to: **given θ, evaluate observable f on `species_dict_A` from scenario A and `species_dict_B` from scenario B, return a scalar**.

Crucially, this is a function over **already-computed scenario outputs**, not over raw simulation timeseries. The species_dict could be replaced by a precomputed within-scenario observable from each side. That observation is what makes D3 feasible.

## Architectural facts the spike depends on

(Verified by reading the code today — see line refs.)

1. **Per-scenario CppSimulators run independently** with the same theta pool seed. `sbi_runner.py:573` instantiates one `CppSimulator` per scenario; `sbi_runner.py:610` and `:627` run them sequentially. Each emits its own `(theta, x_scen)` and `target_df_scen`.

2. **Cross-scenario row alignment is already a hard invariant** post-Phase B. `sbi_runner.py:688-720` intersects `sample_index` across scenarios; after this block, `scenario_meta[scen]['x_raw']` rows are the same θ across scenarios.

3. **Cal-target YAMLs are scoped to one scenario today.** `yaml_loader.load_calibration_targets()` (qsp-hpc-tools yaml_loader.py:99) returns a single `test_stats_df` and is called per-scenario via `CppSimulator.test_stats_csv`. Schema has no scenario field.

4. **`compute_test_statistics_batch`** (qsp-hpc-tools derive_test_stats_worker.py:121) takes one `sim_df` (one scenario's parquet) and one `test_stats_df`. It emits `(n_sims, n_test_stats)`. No multi-sim_df form exists.

5. **Test-stats CSV is part of the pool cache key.** Per the comment at yaml_loader.py:37-43, "the HPC test-stats pool is keyed by `compute_test_stats_hash(csv)` and any schema change invalidates every existing derived-stats cache." A schema change (D2) re-derives the stats column for every existing pool — manageable but painful.

6. **Per-scenario joint NaN filter is `axis=0` over rows.** Cross-scenario observables computed from per-scenario results inherit the joint NaN mask automatically — if either scenario's input went NaN for a θ, the cross-scenario observable is NaN for that row, which is the desired semantic.

## Three options, scored

### D1 — Simulator-side derived species

**Idea**: Run a "counterfactual" scenario YAML that internally simulates two model copies (blocked, unblocked) on the same θ, emit `ratio` as a derived species column in the parquet. Cal target is normal-shape, scoped to the new scenario.

**Touchpoints**:
- C++ qsp_sim: requires running two parameter sets in one sim and emitting derived columns. **Major rework**.
- Scenario YAML schema: needs a "counterfactual_overlay" block.
- `evolve_to_diagnosis` cache: would need to handle two parameter sets per θ.

**Cost**: ~2 weeks, mostly C++ side. **Doubles per-sim wall-clock** for the counterfactual scenarios.

**Pros**: cal-target schema unchanged.
**Cons**: cost; "derived species" is the wrong primitive for parameter overrides; HPC pool layout breaks (two parameter sets per row).

**Rejected.**

### D2 — Schema extension: cal target receives `{scenario: species_dict}` (rejected; subsumed by D3+)

**Idea**: Cal-target YAML grows a `scenarios: [A, B]` field. Observable signature changes to `compute_observable(time_dict, species_dict_dict, _constants, ureg)`. Test-stats CSV grows a `scenarios` column. `compute_test_statistics_batch` accepts a dict of `{scen: sim_df}`.

**Touchpoints** (concrete file list):
- `maple/src/maple/core/calibration/calibration_target_models.py` — schema change to allow scenario list; observable validator update.
- `qsp-hpc-tools/qsp_hpc/calibration/yaml_loader.py` — add scenarios column; wrapper-code generator handles dict-of-dicts; CSV column set widens (cache invalidation, see fact #5).
- `qsp-hpc-tools/qsp_hpc/batch/derive_test_stats_worker.py:121` — `compute_test_statistics_batch` accepts `Dict[str, pd.DataFrame]` and the `species_plan` becomes per-scenario.
- `qsp-hpc-tools/qsp_hpc/cpp/cpp_simulator.py` — needs a multi-scenario simulator construct, OR we keep per-scenario simulators and the worker is invoked outside them. Probably the latter, but then `CppSimulator.test_stats_csv` no longer fully describes the work.
- `pdac-build/workflows/sbi_runner.py:632-720` — joint NaN filter has to know which observables are cross-scenario.
- Audit (`scripts/parameter_audit.py` & `qsp-inference`'s PPC sections): assumes test stats are per-scenario.

**Cost**: ~1-2 weeks, touching 4 repos.

**Pros**: clean abstraction, reusable once paid for.
**Cons**: cache invalidation across all existing pools; multi-repo schema migration; per-scenario cal targets pay zero cost for the new field but still see the schema change.

### D3 — Post-process composer (recommended)

**Idea**: Cross-scenario observables are a separate object type — `CrossScenarioObservable` — composed *after* per-scenario `compute_test_statistics_batch` has run. They consume the aligned `scenario_meta[*]['x_raw']` dict (sample-index-aligned per fact #2) and emit one column appended to the concatenated `x` matrix.

**Schema sketch** (new YAML directory: `calibration_targets/cross_scenario/<observable_id>.yaml`):

```yaml
cross_scenario_target_id: tumor_diameter_fold_gvax_vs_untreated_d90
inputs:
  - scenario: clinical_progression_longterm
    test_statistic_id: tumor_diameter_day_90  # must exist in that scenario's per-scenario cal targets OR be added as a prediction target
    role: denominator
  - scenario: gvax_neoadjuvant_longterm
    test_statistic_id: tumor_diameter_day_90
    role: numerator
observable:
  code: |
    def compute(inputs, ureg):
        return inputs['numerator'] / inputs['denominator']
  units: dimensionless
empirical_data:
  median: [1.05]
  ci95: [[0.80, 1.35]]
  sample_size: nan
provenance: ...  # same calibration_target provenance schema
```

**Touchpoints** (concrete and minimal):
- New file: `qsp-hpc-tools/qsp_hpc/calibration/cross_scenario_loader.py` — loads the YAMLs, emits a `cross_scenario_df` parallel to `test_stats_df`.
- New function: `compute_cross_scenario_statistics(scenario_meta, cross_scenario_df, ureg) -> np.ndarray (n_sims, n_cross_targets)`. Reads each input row out of `scenario_meta[scen]['x_raw']` by `test_statistic_id`, calls the composer, fills NaN if any input is NaN.
- `pdac-build/workflows/sbi_runner.py`: insert the composer call between alignment (`sbi_runner.py:720`) and the per-observable NaN audit, then concatenate cross-scenario columns onto each scenario's `x_raw`. Or, cleaner, treat cross-scenario columns as their own pseudo-scenario block in the final concatenation at sbi_runner.py:860.
- `parameter_audit.py` / qsp-inference PPC: cross-scenario observables come through as additional columns in the same long-format CSV, identified by a `is_cross_scenario` marker. Audit's coverage tables don't need to know the difference; PRCC / shap views can group them separately.
- `maple`: optional — if we want maple-extracted provenance for cross-scenario YAMLs, we'd add a sibling `CrossScenarioCalibrationTarget` model. Can defer; first round can be hand-authored since perturbation literature claims are usually inequality-shaped rather than primary data.

**Cost**: ~2-3 days plumbing + ~1 day per ~10 cal targets. Total Phase D-2 work shrinks because the infra cost is small and the YAML authoring is the same regardless of option.

**What changes that originally drove the D2 recommendation**:
- The original argument for D2 was "Option D1 doubles simulation cost; D2 is the only clean abstraction". D3 was missed because it's neither simulator-side (D1) nor schema-extension (D2) — it lives in the post-process layer that didn't exist when the migration plan was authored. Phase B added the cross-scenario alignment block (`shared_idx`) for the joint NaN filter, and that same alignment is exactly what cross-scenario observables need.

**Pros**:
- Per-scenario cal-target schema, yaml_loader, derive worker, and pool cache keys are **unchanged** (no cache invalidation).
- Cross-scenario observables don't need to re-run any simulation — they're a function over already-computed per-scenario outputs.
- Composer is testable in isolation (pure function on a small dict).
- Maps naturally onto Phase B / E1's existing per-scenario "tumor_diameter_day_90" cal targets — they double as inputs to the cross-scenario observable.

**Cons / risks**:
- Scalar-only D3 requires every input to a cross-scenario observable to already exist as a per-scenario test_statistic_id (calibration or prediction). For perturbation scenarios specifically, this means we'd need a `tumor_diameter_day_90` etc. as a *prediction* target in the perturbation scenario, not a calibration target — which is fine since the perturbation scenario has no measurement. **D3+ removes this constraint** (timeseries inputs are pulled directly from the scenario's sim_df). **Action**: spec out the prediction-target side for scalar inputs; use timeseries inputs when no natural per-scenario summary exists.
- The YAML "calibration_target" provenance schema doesn't quite fit since the input is a derived ratio not a primary measurement. We can either (a) extend `CalibrationTarget` with an optional `composition` block (small change to maple) or (b) define a sibling `CrossScenarioCalibrationTarget` model. (b) is cleaner.

### D3+ — Timeseries-capable composer (recommended for Phase D-2)

**Idea**: D3 with the cross-scenario YAML schema declaring `input_kind` per input. `input_kind: test_statistic` (default) pulls a precomputed scalar from `scenario_meta[scen]['x_raw']` (scalar D3); `input_kind: timeseries` pulls the full per-scenario sim_df and exposes `{species_name: Quantity[T]}` to the composer, mirroring `compute_test_statistics_batch`'s species_dict but composed across scenarios.

**Schema sketch** (extends the D3 example):

```yaml
cross_scenario_target_id: time_to_tumor_50pct_fold_gvax_vs_untreated
inputs:
  - scenario: clinical_progression_longterm
    input_kind: timeseries
    required_species: [V_T.C1, V_T]
    role: untreated
  - scenario: gvax_neoadjuvant_longterm
    input_kind: timeseries
    required_species: [V_T.C1, V_T]
    role: treated
observable:
  code: |
    def compute(inputs, ureg):
        # inputs[role] = {'time': Quantity[T], 'V_T.C1': Quantity[T], 'V_T': Quantity[T] or scalar}
        t_untreated = _time_to_threshold(inputs['untreated'], 0.5, ureg)
        t_treated   = _time_to_threshold(inputs['treated'],   0.5, ureg)
        return t_treated / t_untreated
  units: dimensionless
empirical_data:
  median: [0.6]
  ci95: [[0.3, 1.0]]
provenance: ...
```

**Touchpoints (delta on top of scalar D3)**:
- `cross_scenario_loader.py`: parse `input_kind`; collect per-scenario `required_species` unions; emit a `cross_scenario_df` with one row per cross-target carrying `(role, scenario, input_kind, required_species, ts_id_if_scalar)` per input.
- `compute_cross_scenario_statistics(scenario_meta, cross_scenario_df, species_units, ureg)`: for each θ-row, build a per-scenario species_dict (timeseries inputs) and/or per-scenario scalar (test_statistic inputs), dispatch composer, write to output column. Reuse the species-resolution plan from `derive_test_stats_worker.py:194-228` directly — same `('series', 'param', 'template', 'missing')` strategies. ~80 lines.
- `sbi_runner.py`: retain `scenario_meta[scen]['sim_df_path']` (or sim_df itself) past the per-scenario `compute_test_statistics_batch` call. One-liner change — sim_dfs are already in memory during that call; we just don't drop the reference.
- HPC parallelism: parquets are already pool-scoped (qsp-hpc-tools PRs #46/#47); cross-scenario derive can run remote-side as an additional derive step paralleling per-scenario derive. Adds ~1 day to wire as a remote step; locally the loop is straightforward.

**Cost vs scalar D3**: ~+1-2 days. Total ~5 days infra.

**What D3+ buys you**:
- Every observable shape D2 handles, including time-to-event, AUC, peak-time-shift, fold-of-derived-quantity-at-arbitrary-time. The composer has the same view of timeseries data as a per-scenario `compute_observable` does — just multiplexed across scenarios.
- No per-scenario cal-target schema change. No maple `CalibrationTarget` schema change. No test-stats CSV cache invalidation.
- Memory cost: the per-scenario sim_dfs are already loaded for `compute_test_statistics_batch`. Holding the references through the cross-scenario pass is free; the `process_single_batch` loop in derive_test_stats_worker.py:315 already streams batch-by-batch, so very large N stays bounded.

**Cons / risks**:
- Cross-scenario YAML schema is now non-trivial (two input_kinds, per-input required_species, role-keyed dispatch). Need a small Pydantic model + tests.
- Composer code lives in YAML, like per-scenario observables — same gotchas (Pint discipline, NaN handling). Mitigated by reusing the existing `_generate_wrapper_code` pattern from yaml_loader.py:325.
- Provenance: same as scalar D3 — sibling `CrossScenarioCalibrationTarget` model in maple.

## Decision matrix

| Axis | D1 | D2 | D3 (scalar) | D3+ (timeseries) |
|---|---|---|---|---|
| Engineering cost | ~2 weeks (C++) | ~1-2 weeks (4 repos) | ~3-4 days | ~5 days |
| Per-sim wall-clock impact | 2× counterfactual scenarios | none | none | none |
| Pool cache invalidation | yes (new scenario) | yes (test-stats CSV column set) | no | no |
| Multi-repo schema migration | C++ side | maple + qsp-hpc-tools + pdac-build + audit | qsp-hpc-tools + pdac-build | qsp-hpc-tools + pdac-build (+ tiny maple sibling model) |
| Timeseries-level cross-scenario composers | partial | yes | no | yes |
| Affects existing per-scenario cal targets | no | yes (schema field, even if unused) | no | no |
| Time to first working cross-scenario target | ~3 weeks | ~2 weeks | ~3-4 days | ~5 days |

D3+ dominates D1 and D2 on every axis and dominates scalar-only D3 on the timeseries row at +1-2 days. Recommended path is D3+ directly (skip the scalar-only intermediate; the API surface is the same and the +1-2 days buys flexibility for the unknown future cross-scenario observable).

## Recommended path

1. **Land D3+** as a ~5-day sprint:
   1. Sibling YAML schema in maple (`CrossScenarioCalibrationTarget`) — supports `input_kind: test_statistic | timeseries` per input.
   2. `cross_scenario_loader.py` + `compute_cross_scenario_statistics` in qsp-hpc-tools (one new file, ~230 lines — scalar resolution + timeseries species_dict construction reusing the species_plan pattern from `derive_test_stats_worker.py:194-228`).
   3. Wire into `sbi_runner.py` between alignment and concatenation (~30 lines), retaining sim_df references through the cross-scenario pass for timeseries inputs.
   4. Add the perturbation scenarios as **prediction-only** scenarios (the 5 already in the migration plan: tgfb, cxcl12, il10, ecm, ifng blockade); they emit no per-scenario cal targets but their sim_dfs feed cross-scenario observables.
   5. Author 2-3 canonical cross-scenario cal targets to validate both input kinds:
      - Scalar: tumor diameter d90 GVAX vs untreated (composes existing per-scenario test stats).
      - Timeseries: time-to-tumor-50% fold ratio (no natural per-scenario summary; needs full timeseries).
      - Counterfactual: iCAF fraction TGFb-blockade vs baseline (timeseries from a perturbation scenario).

2. **D2 is rejected**, not deferred — D3+ subsumes it.

3. **Update qualitative_to_calibration_migration_plan.md** §"Phase D — Required infrastructure 2" to point at this spike, mark D2 as superseded by D3+, and re-cost Phase D-2 from "1-3 weeks" down to "~1.5 weeks" (~5 days infra + ~3-4 days authoring 25-30 cal targets, half of which already exist in the migration plan as inlined thresholds).

## Owner decisions (resolved 2026-04-27)

1. **Provenance for cross-scenario targets**: ✅ Mechanistic-prior pattern (epistemic_basis=mechanistic, vague CI95) is appropriate at cross-scenario scale. Same epistemic move as the Phase E1 directional mechanistic priors.

2. **Perturbation scenarios as prediction-only**: ✅ Blockade scenarios contribute no per-scenario cal targets; they participate in cross-scenario observables only. Their sim_dfs (timeseries inputs) and any per-scenario prediction-target test_statistics (scalar inputs) feed the composer.

3. **HPC budget impact of 5 sequential perturbation scenarios**: ✅ Acceptable. Warm evolve cache means each adds a few minutes at production N.

4. **Scope of timeseries inputs**: ✅ Adopt D3+ (timeseries-capable) directly rather than landing scalar-only D3 first. The +1-2 days buys flexibility for unanticipated cross-scenario observable shapes.

## References

- Migration plan: `notes/calibration/qualitative_to_calibration_migration_plan.md` §Phase D
- Aligned scenario_meta plumbing: `pdac-build/workflows/sbi_runner.py:688-720` and `:860`
- Per-scenario test-stats worker: `qsp-hpc-tools/qsp_hpc/batch/derive_test_stats_worker.py:121`
- Cal target loader: `qsp-hpc-tools/qsp_hpc/calibration/yaml_loader.py`
- Existing within-scenario fold targets shipped in Phase E1: `calibration_targets/mechanistic/{gvax,gvax_nivo}_neoadjuvant_longterm/tumor_diameter_fold_d90_*_mechanistic.yaml` — same shape but observable scoped to one scenario; D3 cross-scenario versions would compute the ratio across two scenarios for the same θ instead.
