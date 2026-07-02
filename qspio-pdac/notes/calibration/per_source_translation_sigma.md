# Per-Source Translation Sigma Design

## Problem

The current `SubmodelTarget` schema has a single `source_relevance: SourceRelevanceAssessment` at the top level, producing one translation sigma (σ_trans) that inflates the likelihood uniformly across all measurements in a target. This works for single-paper derivations but breaks down for **multi-paper derivations** where different inputs come from sources with different translation quality.

Example: `q_nCD8_LN_in` (naive CD8 LN entry rate) is derived from:
- **Mandl 2012** (mouse): LN exit rate from anti-L-selectin blockade experiments
- **Autissier 2010** (human): naive CD8 counts in peripheral blood

The mouse exit rate has a species translation gap; the human blood counts don't. A single `source_relevance` can't express this.

## Design: Per-Source Relevance Assessment

### Schema Changes

Move `source_relevance` from `SubmodelTarget` onto `PrimaryDataSource` and `SecondaryDataSource`:

```python
class PrimaryDataSource(BaseModel):
    doi: Optional[str]
    source_tag: str
    source_relevance: SourceRelevanceAssessment  # NEW — was on SubmodelTarget
    ...

class SecondaryDataSource(BaseModel):
    doi: Optional[str]
    source_tag: str
    source_relevance: SourceRelevanceAssessment  # NEW — was on SubmodelTarget
    ...
```

Remove `source_relevance` from `SubmodelTarget` top level (no backward compat — migrate all existing YAMLs).

### Translation Sigma Computation

Per-measurement sigma computed from the sources of its inputs:

1. For each measurement in `error_model`, collect `uses_inputs` → look up each input's `source_ref` → find that source's `source_relevance`
2. Compute `sigma_trans_i = compute_translation_sigma(source_relevance_i)` for each **unique source** (deduplicate by `source_ref` — two inputs from the same paper share a correlated translation gap, don't double-count)
3. Combine unique source sigmas in **quadrature**: `sigma_trans_measurement = sqrt(Σ sigma_i²)` — each source introduces an independent translation gap (different species, different indication, etc.)

### Why Quadrature (Not Max)

Each source's translation gap is independent:
- Mandl 2012 has a mouse→human species gap
- Autissier 2010 has a healthy→PDAC indication gap
- These are different uncertainties that compound, not alternatives

Max would be correct if you had to pick one source's assessment as representative. Quadrature is correct when each source contributes independent translation uncertainty to the derived parameter.

### Why Deduplicate by Source

Two inputs from the same paper (e.g., mean and SD from Table 1 of Mandl 2012) share the same translation gap — the species mismatch applies once, not twice. Group inputs by `source_ref`, compute sigma once per unique source, then combine across sources.

### Where Sigma Gets Applied

Currently (line 952 of `submodel_inference.py`):
```python
sigma_total = jnp.sqrt(entry.sigma**2 + tl.sigma_trans**2)
```

Changes to per-entry sigma:
```python
sigma_total = jnp.sqrt(entry.sigma**2 + entry.sigma_trans**2)
```

`sigma_trans` moves from `TargetLikelihood` to `ErrorModelEntry`. Each measurement gets its own translation sigma based on which sources its inputs come from.

### Downstream Changes Required

1. **`submodel_target.py`**: Remove top-level `source_relevance`, add to data source models
2. **`shared_models.py`**: No changes (SourceRelevanceAssessment stays the same)
3. **`submodel_inference.py`**: `sigma_trans` moves to `ErrorModelEntry`, new `resolve_per_measurement_sigma()` function
4. **`yaml_to_prior.py`**: `compute_translation_sigma` called per-source, composed per-measurement
5. **All existing YAML targets**: Migrate `source_relevance` from top level into `primary_data_source`
6. **Validators in `submodel_target.py`**: Update validators that reference `self.source_relevance`

### Example: q_nCD8_LN_in

```yaml
primary_data_source:
  doi: "10.1073/pnas.1211717109"
  source_tag: Mandl2012
  source_relevance:
    indication_match: proxy
    indication_match_justification: "Mouse LN transit data as proxy for human PDAC..."
    species_source: mouse
    species_target: human
    source_quality: primary_animal_in_vivo
    perturbation_type: pharmacological  # anti-L-selectin blockade
    perturbation_relevance: "Entry blockade reveals exit rate..."
    tme_compatibility: low
    tme_compatibility_notes: "Healthy mouse LN, not tumor-draining..."
    measurement_directness: single_inversion
    temporal_resolution: timecourse
    experimental_system: animal_in_vivo

secondary_data_sources:
  - doi: "10.1182/blood-2009-08-239442"
    source_tag: Autissier2010
    source_relevance:
      indication_match: proxy
      indication_match_justification: "Healthy human blood counts as proxy for PDAC..."
      species_source: human
      species_target: human
      source_quality: primary_human_clinical
      perturbation_type: physiological_baseline
      perturbation_relevance: "Unperturbed baseline counts..."
      tme_compatibility: moderate
      tme_compatibility_notes: "Blood counts, not tumor-draining LN..."
      measurement_directness: direct
      temporal_resolution: snapshot_or_equilibrium
      experimental_system: clinical_in_vivo

inputs:
  - name: q_nCD8_LN_out
    value: 1.8
    units: "1/day"
    source_ref: Mandl2012        # → mouse sigma ≈ 0.55

  - name: nCD8_blood
    value: 350e6
    units: "cell"
    source_ref: Autissier2010    # → human sigma ≈ 0.30

calibration:
  forward_model:
    type: algebraic
    formula: q_in = q_out * ratio_LN_blood

  error_model:
    - name: naive_CD8_entry_rate
      uses_inputs: [q_nCD8_LN_out, nCD8_blood]
      # sigma_trans = sqrt(0.55² + 0.30²) ≈ 0.63
      # (two unique sources, independent translation gaps)
```

### Parameters That Would Use Multi-Paper Derivations

- **Trafficking rates** (`q_nCD8_LN_in`, `q_nCD8_P_out`): Derived from exit rates (kinetic studies) + compartment counts (clinical hematology)
- **Parameters requiring unit conversion from a second source**: e.g., Ki67+ fraction (paper A) → absolute proliferation rate requires cell cycle duration (paper B)
- **Steady-state inversions**: Any parameter derived by solving a balance equation using measured values of other terms from different studies

### Migration Plan

1. Implement schema changes in maple (PrimaryDataSource, SecondaryDataSource get source_relevance)
2. Update inference code (per-measurement sigma resolution)
3. Migrate all ~44 existing submodel target YAMLs (move source_relevance into primary_data_source)
4. Update staged_extraction.py prompts to guide multi-paper derivations
5. Update validators
