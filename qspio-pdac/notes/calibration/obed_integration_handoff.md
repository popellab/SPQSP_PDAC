# OBED integration — session handoff (2026-04-20)

Option D of the OBED plan: compute clinical-endpoint predictions at the
promoted stage-2 posterior, feed the audit's new OBED section, attribute
endpoint uncertainty to parameters. Five phases total, two merged.

## TL;DR

- **Phase 1 (merged)** — `scripts/clinical_endpoints.py`: OS/DFS landmarks,
  tumor-diameter landmarks, RECIST 1.1 categories, TTP, VDT. 22 unit tests.
- **Phase 2 (merged in PR #20)** — prediction-only observable mechanism
  separate from `CalibrationTarget`. Schema + codegen + 36 YAMLs across
  three long-horizon scenarios. Long-horizon variants split out from the
  short-horizon clinical scenarios to keep existing qualitative-behavior
  tests green.
- **Phase 3 (merged in qsp-hpc-tools PR #51)** —
  `CppSimulator.simulate_with_parameters(theta, *, prediction_targets=None)`,
  `load_prediction_targets` / `hash_prediction_targets` in
  `qsp_hpc.calibration`. Theta-hashed suffix pool keyed by theta + cal
  hash + pred hash. Output `pa.Table` has named `ts:<target_id>` columns.
  Calibration-CSV schema preserved byte-stable (HPC test-stats cache
  hashes unaffected); `is_prediction_only` back-filled at concat time.
  22 new tests, full suite 634 passed.
- **Phase 4 (TODO)** — one-shot PPC run: load
  `figures/sbi_stage2_current/posterior_samples.csv`, simulate each of the
  3 long-horizon scenarios at those thetas, write
  `posterior_predictive_clinical.csv` next to the posterior samples.
- **Phase 5 (TODO)** — audit OBED section reads the PPC file, reports CI95
  per endpoint, uses Spearman(θ_post, endpoint_value) for param attribution.

Open task list: `#4` (PPC run), `#5` (audit section).
Follow-up issue #19 tracks surgery + adjuvant FOLFIRINOX modeling (Path C)
— needed before the long-horizon predictions compare directly to clinical
OS/DFS.

## What's in `main` after PR #20

| Path | Purpose |
|---|---|
| `scripts/clinical_endpoints.py` | Reference endpoint functions (pure Python, Pint-aware) |
| `scripts/prediction_target.py` | Pydantic schema for `PredictionTarget` (no `empirical_data`) |
| `scripts/generate_prediction_yamls.py` | Codegen: inlines endpoint + helpers into self-contained YAMLs |
| `calibration_targets/predictions/<scenario>/*.yaml` | 36 generated YAMLs (12 endpoints × 3 scenarios) |
| `scenarios/clinical_progression.yaml` | stop_time bumped 365 → 1095 |
| `scenarios/gvax_neoadjuvant_longterm.yaml` | New 1095 d variant of GVAX-alone arm |
| `scenarios/gvax_nivo_neoadjuvant_longterm.yaml` | New 1095 d variant of GVAX+nivo arm |
| `tests/test_clinical_endpoints.py` | 22 unit tests on endpoint functions |
| `tests/test_prediction_targets.py` | 148 tests: schema, executability, semantics, codegen drift |
| `tests/scenarios/test_calibration_targets.py` | `predictions` added to `SKIP_DIRS` |

Endpoints defined (per scenario):

- `os_at_{12,24,36}mo` — binary alive/dead; death = `V_T.C1 ≥ 1e11` cells
- `dfs_at_{12,24}mo` — binary; alive + no RECIST PD from running nadir
- `tumor_diameter_day_{365,730}` — cm, sphere equivalent from `V_T`
- `recist_{cr,pr,sd,pd}_at_day_90` — one-hot indicators from day-0 baseline
- `time_to_progression` — days, interpolated first RECIST PD, censored at 1095

Death / progression defaults (hardcoded in the endpoint functions, overridable
via kwargs in the codegen):

- `N_C_LETHAL_DEFAULT = 1.0e11` cells
- `PROGRESSION_INCREASE_DEFAULT = 0.20`
- `RESPONSE_DECREASE_DEFAULT = 0.30`
- `CR_DIAMETER_CM_DEFAULT = 0.1`
- `min_absolute_increase_cm = 0.5` (RECIST 1.1 AND-gate)

## Key design decisions already made

1. **Prediction-only YAMLs are separate** from CalibrationTarget — no
   empirical_data / primary_data_source / provenance. Lives in
   `calibration_targets/predictions/<scenario>/`. Schema home is pdac-build;
   upstream to maple later if it proves useful.
2. **Codegen over inline** — endpoint source lives in
   `clinical_endpoints.py`; YAMLs are regenerated from it. Drift-guard
   test re-runs the generator and diffs.
3. **Long-horizon scenarios are a separate file**, not a stop_time change
   on the short ones. Reason: bumping stop_time on
   `gvax_nivo_neoadjuvant_zheng2022` broke the qualitative-behavior
   comparison tests (numerical solver differences over the 30 d → 1095 d
   horizon shift). Short scenarios match published d21 endpoints; long
   scenarios are prediction-only.
4. **Clinical_progression extended in-place** to 1095 d (no qualitative
   tests there, only the single tumor_doubling_time CalibrationTarget
   which still evaluates over the full trajectory).
5. **No surgery, no adjuvant FOLFIRINOX** — issue #19. Long-horizon
   predictions therefore capture the pharmacodynamic effect of the
   neoadjuvant regimen alone, not trial OS/DFS directly.

## Phase 3 — CppSimulator.simulate_with_parameters (MERGED — qsp-hpc-tools #51)

Goal: a method that takes an explicit `theta_matrix` (not drawn from prior)
and returns test-stat outputs at those thetas, caching in a suffix pool.
Mirrors `QSPSimulator.simulate_with_parameters` (MATLAB backend) per the
`notes/calibration/sbi_cpp_backend_handoff.md` note (which flags it as
`~50 LOC + a theta-hashed suffix pool`).

Additional requirement specific to Phase D: the method must accept
**prediction targets alongside calibration targets** so prediction-only
observables are computed at the same thetas and written to the same
test_stats DataFrame. Suggested signature:

```python
def simulate_with_parameters(
    self,
    theta: np.ndarray,          # (N, n_params)
    *,
    prediction_targets: Path | None = None,  # dir for prediction YAMLs
) -> tuple[np.ndarray, pa.Table]:
    ...
```

Where to work:

- `qsp-hpc-tools/qsp_hpc/simulation/cpp_simulator.py` — new method.
- `qsp-hpc-tools/qsp_hpc/calibration/yaml_loader.py` — add
  `load_prediction_targets(yaml_dir)` and `hash_prediction_targets(yaml_dir)`
  with the same DataFrame shape as `load_calibration_targets` but:
  - `median`, `ci95_lower`, `ci95_upper` all `NaN`
  - New column `is_prediction_only=True`
  - No empirical_data access — read `prediction_target_id`, `observable.code`,
    `observable.constants`, `observable.units`, `index_values`, `species`.
- Optional sanity test: a small cpp-sim smoke run at 10 posterior thetas
  on the `clinical_progression` scenario, confirming the 12 new prediction
  columns appear in the output test_stats DataFrame.

**Per feedback memory:** use `git worktree add` in qsp-hpc-tools rather
than branching in the primary checkout, since this is a multi-file
sibling-repo change.

## Phase 4 — one-shot PPC run

Script that:
1. Loads `figures/sbi_stage2_current/posterior_samples.csv` (2000 thetas,
   177 params — the columns of the CSV are parameter names matching the
   priors CSV; need to align to the simulator's theta order).
2. For each of the 3 long-horizon scenarios, invokes
   `CppSimulator.simulate_with_parameters(theta)` with the calibration
   targets (existing) and prediction targets (new).
3. Writes `figures/sbi_stage2_current/posterior_predictive_clinical.csv`
   — 2000 rows × N endpoints. Named to advertise "this is the PPC data
   the audit reads".

Depends on Phase 3 landing.

## Phase 5 — audit OBED section

New section in `qsp-inference/src/qsp_inference/audit/report.py` (see
`_section_stage2_npe` for the established pattern). Rendered after Stage 2
shifts, before the cross-model appendix.

Inputs (all already on disk when sbi_run_path is set):
- `posterior_samples.csv` — parameter draws at posterior
- `posterior_predictive_clinical.csv` — endpoint values at those draws (Phase 4)
- Existing diagnostics CSVs

Tables to emit:
1. **Predictive uncertainty** — per endpoint: CI95 width, p05/p50/p95, fraction
   of simulations in failure states (e.g. `os_at_12mo` < 1 is dead; censor
   fractions for TTP).
2. **Driver attribution** — for each endpoint, top-k parameters by absolute
   Spearman correlation on `(posterior_samples[param], predictions[endpoint])`.
   Filter to `prcc_significant=True` params by default.
3. **Stopping criterion headline** — "N of 36 endpoints have CI95 width > X%
   of their median" → rank order and flag which clinical decisions those
   endpoints feed.

Store the per-endpoint CI table and per-(endpoint,param) correlation table in
`notes/calibration/` as CSV artifacts so downstream analysis doesn't have
to re-parse the audit markdown.

## Resume commands (next session)

```bash
# 1. Check that Phase 2 is still healthy.
.venv/bin/pytest tests/test_clinical_endpoints.py tests/test_prediction_targets.py

# 2. Scout CppSimulator state.
grep -n "simulate_with_parameters\|run_hpc\|NotImplementedError" \
    /Users/joeleliason/Projects/qsp-hpc-tools/qsp_hpc/simulation/cpp_simulator.py

# 3. Start Phase 3 in a worktree.
cd /Users/joeleliason/Projects/qsp-hpc-tools
git worktree add ../qsp-hpc-tools-cpp-ppc feature/cpp-simulate-with-parameters

# 4. The QSPSimulator counterpart is in the same dir — mirror its pattern.
grep -n "simulate_with_parameters\|_posterior_predictive_pool" \
    /Users/joeleliason/Projects/qsp-hpc-tools/qsp_hpc/simulation/qsp_simulator.py
```

## Related notes in this directory

- `notes/calibration/sbi_cpp_backend_handoff.md` — status of the C++ backend,
  including the open PPC gap that Phase 3 closes.
- `notes/calibration/sbi_tier_flow_sweep.md` — the NPE sweep that picked
  `h128_t5_s100k` as the promoted run (feeds Phase 4).
- `notes/calibration/calibration_strategy.md` — the broader calibration
  workflow that frames OBED as the validation step.
