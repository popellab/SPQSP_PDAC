# LLM Expert Elicitation for Misspecification Detection

*Date: 2026-03-07*

## Motivation

Iterative prior construction using calibration data constitutes empirical Bayes — the data is used twice (once to shape the prior, again in the posterior), leading to overconfident posteriors. To avoid this, we need a source of domain knowledge that is independent of the specific calibration targets.

## Approach

Use LLM "experts" as a panel (analogous to the SHELF protocol for structured expert judgment) to elicit:

1. **Hypothetical observed values** for each model observable at diagnosis
2. **Uncertainty ranges** (approximate CI95) reflecting inter-patient and methodological variability
3. **Qualitative relationships** between observables (optional, for structural validation)

These elicited values serve as **synthetic calibration targets** for misspecification detection (Phase 1 of the workflow in `misspecification_detection_framework.md`). Real calibration data is reserved exclusively for Phase 2 (actual calibration) and Phase 3 (held-out validation).

## Elicitation Protocol

### Prompting Strategy

Each observable is presented to the LLM with:
- **Observable definition**: What is being measured (e.g., "density of CD8+ T cells in PDAC tumor tissue at diagnosis, cells/mm²")
- **Biological context**: PDAC TME characteristics (cold tumor, desmoplastic, immunosuppressive)
- **Instruction**: Reason from physiology and mechanism, NOT from specific published measurements. The goal is "what would a knowledgeable immunologist estimate without looking up a specific paper?"

### Panel Composition

Multiple LLM queries with varied framing to capture uncertainty in expert judgment:
- **Optimistic immunologist**: Emphasizes immune presence even in cold tumors
- **Pessimistic immunologist**: Emphasizes immunosuppression and stromal barriers
- **Balanced pathologist**: Emphasizes what is typically seen on IHC slides
- **Systems biologist**: Emphasizes mass balance and stoichiometric constraints

The median across panel members becomes the elicited point estimate; the range becomes the elicited CI.

### Key Constraint

The prompt must explicitly instruct: "Do NOT cite specific numbers from published studies. Reason from first principles about what is physiologically plausible." This reduces (but cannot eliminate) indirect leakage from training data.

## Current Baseline Observables (27 targets)

These are the observables for which we need LLM-elicited values:

### Cell Densities (cells/mm²)
- `cancer_cell_density_baseline` — PanCK+ tumor cells
- `cd8_density_baseline` — CD8+ T cells (effector + exhausted)
- `cd4_helper_density` — CD4+ Th cells (Th1 + Th2 + Th17)
- `treg_density_baseline` — Foxp3+ regulatory T cells
- `cdc1_density_baseline` — CLEC9A+ cDC1 dendritic cells
- `cdc2_density_baseline` — CD1c+ cDC2 dendritic cells
- `tam_density` — Total tumor-associated macrophages (M1 + M2)
- `mdsc_density` — MDSCs (neutrophil/eosinophil proxy)
- `total_til_density` — Total TILs (CD8 + Th + Treg)

### Fractions and Ratios (dimensionless)
- `cd8_exhausted_fraction_shen2017` — PD-1+ fraction of CD8+
- `th_exhausted_fraction` — PD-1+ fraction of CD4+
- `cd8_treg_ratio` — CD8+/Treg ratio
- `cd8_treg_ratio_golesworthy2022` — CD8+/Treg (alternative source)
- `treg_fraction_cd4` — Treg/(Treg+Th) of CD4+ cells
- `treg_fraction_cd4_hiraoka2006` — FOXP3+/CD4+ (alternative source)
- `treg_fraction_baseline` — Foxp3+/CD3+
- `m1_m2_ratio` — M1/M2 macrophage ratio
- `m2_macrophage_fraction` — CD163+/total macrophages
- `apc_maturation_ratio` — DC-LAMP+/total DC
- `icaf_fraction_of_caf` — iCAF/total CAF
- `mycaf_fraction_of_caf` — myCAF/total CAF
- `apcaf_fraction_of_caf` — apCAF/total CAF
- `apcaf_fraction_of_caf_elyada2019` — apCAF (alternative source)

### Stromal and ECM
- `stromal_fraction` — Stromal volume fraction of tumor
- `psc_activation_ratio` — Activated PSC fraction
- `collagen1_protein_level` — Collagen fold-change vs normal pancreas

### Other
- `vegf_tissue_concentration` — VEGF-A (pg/mg protein)

## Implementation

The elicitation is implemented as a structured prompt → response → parse pipeline:

1. Single comprehensive prompt with all 27 observables
2. Response parsed into a CSV: `notes/inference/llm_elicitation/llm_elicited_targets.csv`
3. CSV format matches calibration target structure for direct use in `sbi_loo_predictive_check`

## Usage in Workflow

```python
# Load LLM-elicited targets instead of real calibration targets
obs_llm = load_llm_elicited_targets('notes/inference/llm_elicitation/llm_elicited_targets.csv')

# Prior predictive D² against LLM-elicited data (no real data touched)
sr_prior, fig = sbi_self_reference_null(x_train_copula, obs_llm, observable_names)

# After NPE training on LLM-elicited data:
# Posterior predictive D² against LLM-elicited data
sr_post, fig = sbi_self_reference_null(x_pp_copula, obs_llm, observable_names)
```

## Limitations

- LLM training data includes the same publications → indirect leakage (mitigated by "reason from mechanism" prompting)
- LLM uncertainty may be poorly calibrated (systematic overconfidence or underconfidence)
- No guarantee that LLM estimates span the same space as real clinical measurements
- Panel composition is somewhat arbitrary

## References

- O'Hagan, A. et al. (2006). Uncertain Judgements: Eliciting Experts' Probabilities. Wiley. (SHELF protocol)
- Gelman, A. et al. (2020). Bayesian Workflow. arXiv:2011.01808
- See `notes/diagnostics/misspecification_detection_framework.md` for the full three-phase workflow
