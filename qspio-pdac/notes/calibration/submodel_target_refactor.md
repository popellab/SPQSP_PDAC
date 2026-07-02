# SubmodelTarget Schema Refactor

## Motivation

The SubmodelTarget schema in maple has accumulated fields that are either unused, redundant, or belong elsewhere. After filling out 4 YAMLs (EC50_GMCSF, k_APC_mature_ID, k_IL2_deg, k_NK_kill), these pain points emerged:

1. **Priors don't belong in the data model** — they should be assembled downstream from observation code
2. **Uncertainty/n are awkwardly nested** inside Input — every numeric value should be its own flat Input
3. **role, source_access, extraction_method** are unused noise
4. **sample_size** should reference an input (it's paper data with provenance)
5. **observation_code signature** keeps 2-arg `(inputs, sample_size)` — sample_size resolved from `sample_size_input`
6. **InputType enum** had too many escape hatches (`assumed_value`, `inferred_estimate`, etc.)

## Schema Changes (in submodel_target.py)

### Remove classes/enums
- [x] `PriorDistribution` enum
- [x] `Prior` class
- [x] `Uncertainty` class
- [x] `InputRole` enum
- [x] `ExtractionMethod` enum (the one in submodel_target.py)
- [x] `SourceAccess` enum

### Simplify InputType enum
- [x] Remove `ASSUMED_VALUE` — too easy to bypass snippet validation
- [x] Remove `PROXY_MEASUREMENT` — conversion happens in observation_code, input is still a measurement
- [x] Remove `EXPERIMENTAL_CONDITION` — these are direct measurements from methods sections
- [x] Remove `INFERRED_ESTIMATE` — not comfortable offloading qualitative interpretation to LLM
- [x] Add `UNIT_CONVERSION` — dimensionless conversion factors (e.g., IQR-to-SD, pM-per-nM)
- [x] Add `REFERENCE_VALUE` — normalization/reference constants (e.g., V_T_ref, tumor_cell_density)
- Final enum: `DIRECT_MEASUREMENT`, `UNIT_CONVERSION`, `REFERENCE_VALUE`

### Simplify Input class
- [x] Remove `uncertainty`, `n`, `role`, `extraction_method`, `source_access`
- [x] Add `rationale: Optional[str]` — required for `unit_conversion` and `reference_value` inputs
- Input keeps: `name`, `value`, `units`, `input_type`, `rationale`, `source_ref`, `source_location`, `value_snippet`, `table_excerpt`

### Simplify Parameter class
- [x] Remove `prior` field
- Parameter keeps: `name`, `units`

### Modify ErrorModel
- [x] Replace `sample_size` + `sample_size_rationale` with `sample_size_input: str` (required, references input name)
- [x] Restore `observable: Optional[Observable]` field (was accidentally removed)
- [x] Restore `ObservableType` enum and `Observable` class
- [x] Keep `uses_inputs` as-is (currently Optional)

### Make optional
- [ ] `code_julia` on `AlgebraicModel` and `CustomODEModel` → Optional

### Keep as-is
- `data_rationale`, `submodel_rationale` — required
- `formula` on AlgebraicModel — required
- `observable` on ErrorModel
- `uses_inputs` on ErrorModel

## Validator Changes

### Deleted (referenced removed `param.prior` or old `sample_size`)
- [x] `validate_prior_reflects_translation_uncertainty` — used `param.prior`
- [x] `validate_algebraic_prior_predictive` — used `param.prior` and `InputRole.TARGET`
- [x] `validate_sample_size_list_length` — used old `entry.sample_size` list field
- [x] `validate_algebraic_model_output_units` — used `get_prior_median(param.prior)`
- [x] `validate_prior_predictive_scale` — used `param.prior`

### Fixed (updated to use `sample_size_input`)
- [x] `validate_observation_sd_units_for_likelihood` — resolves sample_size from inputs via `entry.sample_size_input`
- [x] `validate_observation_code_execution` — same
- [x] `warn_observation_sd_unreasonable` — same

### Updated (reference old enum values)
- [x] `validate_input_values_in_snippets` — skip `UNIT_CONVERSION`/`REFERENCE_VALUE` (not `ASSUMED_VALUE`/`EXPERIMENTAL_CONDITION`)
- [x] `validate_no_hardcoded_constants` — updated guidance messages

### Added
- [x] `validate_input_refs` — check that `sample_size_input` references an existing input
- [x] `validate_non_measurement_inputs_have_rationale` — require `rationale` for `unit_conversion`/`reference_value`
- [x] `validate_no_assumed_or_uncertainty_inputs` — reject inputs with `assumed` in name; reject uncertainty-smelling names on non-measurement types

### Clean up references
- [x] `__all__` list — removed `InputRole`, `PriorDistribution`, etc.; added `TableExcerpt`, kept `Observable`/`ObservableType`

## Test file (test_submodel_target_validators.py)
- [x] Remove imports of deleted classes (`InputRole`, `ExtractionMethod`, `Prior`, `PriorDistribution`)
- [x] Update `minimal_input` fixture (remove `role`, `extraction_method`)
- [x] Delete `minimal_prior` fixture
- [x] Update `minimal_parameter` fixture (remove `prior`)
- [x] Update `minimal_measurement` fixture (add `sample_size_input`)
- [x] Update `make_algebraic_target` helper (remove `prior`/`role`/`extraction_method`, add `sample_size_input`, add `test_n` input, add `value_snippet`)
- [x] Update all inline data dicts (remove `prior`/`role`/`extraction_method`, add `sample_size_input`, add `test_n` input, add `value_snippet`)
- [x] Delete `TestPriorPredictiveScale` class (validator removed)
- [x] Rework `test_skips_experimental_condition` → `test_skips_unit_conversion`
- [x] All 46 tests pass, all 209 maple tests pass

## Remaining Work

### julia_translator.py (separate task)
- Heavily coupled to `Prior`/`PriorDistribution` (10 references to `.prior`)
- Needs interface change: accept priors from external source (priors CSV)
- NOT part of this refactor — do separately

### exceptions.py
- [ ] Check if `PriorParameterError`, `PriorScaleError`, `SampleSizeError` are used outside deleted validators — if not, remove

### YAML files (pdac-build/to-review/)
- [x] Add sample_size as explicit Input entries where needed
- [x] Remove `prior` blocks from calibration.parameters
- [x] Remove `role`, `extraction_method`, `source_access`, `uncertainty`, `n` from inputs
- [x] Update `input_type: assumed_value` → appropriate new type or remove input entirely
- [x] Remove translation uncertainty assumptions from observation_code (SD must be data-derived)
- [x] Update observation_code to return bootstrap samples (see Bootstrap Observation Code below)

### YAML files (pdac-build/calibration_targets/submodel_targets/)
- [ ] Update `input_type: assumed_value` inputs — most are uncertainty factors that should be removed from inputs entirely and derived in observation_code
- [ ] Update `input_type: experimental_condition` → `direct_measurement` with snippet
- [ ] Update `input_type: proxy_measurement` → `direct_measurement`
- [ ] Update `input_type: inferred_estimate` → `direct_measurement` with snippet
- [ ] Update observation_code to return bootstrap samples

### Prompt file (submodel_target_prompt.md)
- [ ] Update InputType documentation to reflect new enum values

### shared_models.py (CalibrationTarget schema)
- [ ] Align InputType enum with SubmodelTarget changes (separate PR)

## FigureExcerpt (completed)

Added `FigureExcerpt` class to `submodel_target.py`, parallel to `TableExcerpt`, for inputs
read from figures where values cannot be validated by text matching.

### Schema additions
- [x] `FigureExcerpt` class: `figure_id`, `value`, `description`, `context`
- [x] `figure_excerpt` field on `Input` (optional, alongside `value_snippet` and `table_excerpt`)
- [x] `validate_input_values_in_snippets` accepts `figure_excerpt` as valid provenance, emits warning for manual review
- [x] Added to `__all__` exports

### Usage
Inputs read from figures (scatter plots, bar charts, dose-response curves) use `figure_excerpt`
instead of `value_snippet`. The internal validator accepts this as provenance but warns that
manual review is needed. The external snippet validator should also flag these for manual review.

## Bootstrap Observation Code

### Motivation

The old observation_code returned `{value, sd, sd_uncertain, n}` — a summary statistics dict
that forced every extractor to make ad hoc choices about how to compress their data. This led to:
- Hardcoded uncertainty assumptions (e.g., `sd_log = 1.0` for "translation uncertainty")
- Fragile analytical formulas (d2 range-to-SD conversions on n=2 data)
- Inconsistent treatment across targets

The new interface: **observation_code always returns a 1D numpy array of bootstrap samples**.
The framework derives value (median), SD, CI95, and distribution shape from the samples.

### Design

**observation_code signature** (unchanged):
```python
def derive_observation(inputs, sample_size, rng, n_bootstrap):
    # ... parametric bootstrap ...
    return samples  # 1D np.array, shape (n_bootstrap,)
```

**Framework responsibilities** (downstream, not in YAML):
- `value` = np.median(samples)
- `sd` = np.std(samples)
- CI95 = np.percentile(samples, [2.5, 97.5])
- Likelihood fitting (optional): fit normal/lognormal to samples, or use empirical

**Removed fields from error_model return**:
- `sd_uncertain` — gone, the bootstrap SD is what it is
- `n` — derived from `sample_size_input`, no longer in return dict

**Kept fields on error_model**:
- `observation_code` — now returns 1D array (required)
- `uses_inputs` — which inputs it references (required)
- `sample_size_input` — references input name for sample size (required)
- `likelihood` — optional override; if omitted, framework infers from samples
- `n_bootstrap` — optional, default 10000

**Multi-timepoint models**: observation_code returns 1D array per call. If `evaluation_points`
has multiple entries, the framework calls observation_code once per point (passing the point
via an argument or input). Each call returns its own sample array.

### Parametric bootstrap patterns by data type

**Directly reported mean±SD** (e.g., Han2009 CD80 = 90±8%, n=6):
```python
samples = rng.normal(loc=mean, scale=sd, size=n_bootstrap)
```

**Lognormal from multiple point estimates** (e.g., Guthridge2006 two EC50s):
```python
log_vals = np.log([ec50_1, ec50_2])
mu, sigma = np.mean(log_vals), np.std(log_vals, ddof=1)
samples = rng.lognormal(mean=mu, sigma=sigma, size=n_bootstrap)
```

**Cross-study synthesis** (e.g., Lotze1985 + Konrad1990 half-lives):
```python
log_vals = np.log([t_half_1, t_half_2, t_half_3])
mu, sigma = np.mean(log_vals), np.std(log_vals, ddof=1)
samples = rng.lognormal(mean=mu, sigma=sigma, size=n_bootstrap)
```

**Count data** (e.g., Bhat2007 NK killing frequency, mean=4):
```python
samples = rng.poisson(lam=mean_count, size=n_bootstrap)
```

### Schema changes needed
- [x] Update `validate_observation_code_execution` to check return type is ndarray
- [x] Replaced `validate_observation_sd_units_for_likelihood` → `validate_observation_bootstrap_samples` (checks spread > 0)
- [x] Replaced `warn_observation_sd_unreasonable` → `warn_observation_cv_unreasonable` (checks CV from samples)
- [x] Replaced `validate_observation_code_return_signature` → `validate_observation_code_returns_array_not_dict` (rejects old dict format)
- [x] Updated signature check from 2 args to 4 args (inputs, sample_size, rng, n_bootstrap)
- [x] Added `n_bootstrap` optional field to ErrorModel (default 10000)
- [x] Made `likelihood` optional on ErrorModel (framework infers from samples)
- [x] Guarded `warn_likelihood_distribution_unknown` against None likelihood
- [x] Update observation_code docstring/description in ErrorModel
- [x] Update all 4 to-review YAMLs with bootstrap observation_code
- [x] Test DI: added VALID_OBSERVATION_CODE constants to test file (single source of truth for signature)
- [ ] Update calibration_targets/submodel_targets/ YAMLs

## Translation Sigma Rubric (yaml_to_prior.py)

Replaced `estimated_translation_uncertainty_fold` with deterministic 8-axis rubric.
Components add in quadrature, floor=0.15.

| Axis | Values (sigma) |
|---|---|
| indication_match | exact=0, related=0.2, proxy=0.5, unrelated=1.0 |
| species mismatch | same=0, different=0.3 |
| source_quality | clinical=0, human_in_vitro=0.1, animal_in_vivo=0.3, animal_in_vitro=0.4, review=0.2, textbook=0.3, non_peer=0.5 |
| perturbation_type | pathological_state=0, physiological_baseline=0.1, pharmacological=0.25, genetic=0.4 |
| tme_compatibility | high=0, moderate=0.15, low=0.5 |
| measurement_directness | direct=0, single_inversion=0.15, steady_state_inversion=0.3, proxy_observable=0.5 |
| temporal_resolution | timecourse=0, endpoint_pair=0.1, snapshot_or_equilibrium=0.2 |
| experimental_system | clinical_in_vivo=0, animal_in_vivo=0.15, ex_vivo=0.1, in_vitro_coculture=0.15, in_vitro_primary=0.2, in_vitro_cell_line=0.3 |

### Schema additions (maple shared_models.py)
- [x] `MeasurementDirectness` enum (4 levels)
- [x] `TemporalResolution` enum (3 levels)
- [x] `ExperimentalSystem` enum (6 levels)
- [x] All 3 fields required on `SourceRelevanceAssessment`
- [x] `perturbation_relevance`, `tme_compatibility`, `tme_compatibility_notes` also made required
- [x] Test DI: `DEFAULT_SOURCE_RELEVANCE` + `make_source_relevance(**overrides)` factory
- [x] 46/46 submodel target tests pass, 185/185 maple core tests pass
- [x] 4 to-review YAMLs populated with new fields
- [ ] calibration_targets/submodel_targets/ YAMLs need new fields when next touched

## Completed Snippet Fixes (prerequisite work)

- [x] k_APC_mature_ID — all table_excerpts + snippets verified (17 checks, score 1.00)
- [x] EC50_GMCSF — snippets verified, Bhattacharya2015 removed (fabricated data)
- [x] k_NK_kill — snippet verified from Bhat2007 PDF (score 1.00)
- [x] k_IL2_deg — snippets verified from PubMed abstracts, DOI corrected
- [x] Local snippet validator with PubMed abstract fallback (`validate_snippets_local.py`)
- [x] Paper text search tool (`search_paper_text.py`)
- [x] Prior comparison tool (`compare_priors.py`)
