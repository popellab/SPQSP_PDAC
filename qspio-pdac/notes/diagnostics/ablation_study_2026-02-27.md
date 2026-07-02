# Ablation Study: Noise vs Parameter Count

**Date:** 2026-02-27
**Context:** Disentangling what caused the diagnostic differences between the SA97 run (97 params, 30 obs, no noise, Feb 22) and the allparams run (193 params, 35 obs, noise, Feb 27).

---

## Design

2×2 factorial using the same simulation pool (baseline_allparams_v1, 2500 raw → ~1100 valid sims, 35 observables):

| Config | Params | Noise | Description |
|--------|--------|-------|-------------|
| A | 193 | on | Current allparams run (baseline) |
| B | 193 | off | Isolates noise effect at full dimensionality |
| C | 96 | on | SA97-overlap subset with noise |
| D | 96 | off | SA97-overlap subset without noise (closest to original SA97) |

All 4 configs share the same simulations, train/test split, and 35 observables. Only the training pipeline varies (noise injection and theta column subsetting).

---

## Results

| Config | Params | Noise | R² med | R²>0.5 | R²>0.1 | MMD p | Prior fail | PPC | Corr | Piling |
|--------|--------|-------|--------|--------|--------|-------|------------|-----|------|--------|
| 193 + noise | 193 | yes | -0.002 | 2 | 28 | **0.740** | 3 | 30/35 | 0 | 3 |
| 193, no noise | 193 | no | -0.005 | 3 | 43 | **0.044** | 3 | 30/35 | 1 | 10 |
| 96 + noise | 96 | yes | 0.022 | 1 | 30 | **0.755** | 3 | 30/35 | 0 | 2 |
| 96, no noise | 96 | no | 0.062 | 4 | 37 | **0.033** | 3 | 30/35 | 4 | 12 |

---

## Effect Decomposition

### MMD is entirely driven by noise injection

- With noise: MMD p = 0.74–0.76 (passes), regardless of param count
- Without noise: MMD p = 0.03–0.04 (fails), regardless of param count

Parameter count has essentially zero effect on MMD. The noise broadens the training distribution enough to encompass the observations. The noise injection is the correct modeling choice — real calibration targets carry measurement uncertainty, and the NPE should know about it. The SA97 MMD failure (p=0.017) was partly artifactual: the model was being held to an unrealistic standard of matching exact point estimates.

However, the model still fails MMD without noise (p=0.033–0.044) regardless of param count. There is real model-data tension that observation noise is just large enough to paper over. With more precise calibration targets (tighter CIs), the misspecification would reappear.

### R² is driven by param count, slightly hurt by noise

| | No noise | Noise | Δ (noise effect) |
|---|---------|-------|-------------------|
| 96 params | 0.062 | 0.022 | -0.040 |
| 193 params | -0.005 | -0.002 | +0.003 |

The R² collapse from SA97 to allparams was primarily the parameter count doubling, with noise adding a smaller penalty at lower dimensionality. At 193 params the NPE is already at the noise floor so noise barely matters.

The 96-param no-noise config recovers 4 params with R²>0.5 and 37 with R²>0.1 — this represents the true information content of 35 baseline observables.

### Prior predictive failures are structural and invariant

All 4 configs produce exactly 3 prior predictive failures (p<0.05). Neither noise nor param count affects these. They are hard structural model problems:

1. **collagen1_protein_level** (p=0.000) — ECM module production/degradation balance caps collagen below observed values
2. **tumor_glucose_ratio** (p=0.022) — metabolism module depletes glucose too aggressively
3. **apcaf_fraction_of_caf_elyada2019** (p=0.042) — fibroblast module underproduces apCAFs

### Boundary piling and correlations are suppressed by noise

- Without noise: 10-12 params pile at boundaries, 1-4 strong posterior correlations
- With noise: 2-3 piling, 0 correlations

Noise injection widens posteriors (by design), which prevents them from hitting prior walls and masks structural non-identifiabilities. The SA97 run's 7 correlated pairs weren't "resolved" in the allparams run — they were hidden by wider posteriors.

---

## Structural Conclusions

### The model has ~4-6 genuinely identifiable parameters from diagnosis-snapshot data

The 96-param no-noise config (cleanest signal) recovers 4 params with R²>0.5. This is the real information content of 35 baseline observables about the parameter space. The rest are either insensitive or degenerate. This is a data limitation, not a model or NN limitation.

### The 3 prior predictive failures need model equation fixes

These don't respond to any training configuration change. They are structural ceilings:

1. **collagen1_protein_level**: The ECM module (fibroblast_module_enhanced.m) has a production/degradation balance that prevents collagen from reaching the observed level of ~9.8 µg/mg. The simulation maximum is 9.43 — a hard ceiling. Likely needs adjustment to k_ECM_myCAF_sec, k_ECM_deg, or the ECM carrying capacity logic.

2. **tumor_glucose_ratio**: The metabolism module (metabolism_module.m) depletes glucose too aggressively. Observed tumor/normal glucose ratio is 0.92, but the model median is ~0.33. The glucose QSS formula or the consumption rate parameters (k_glc_tumor, k_glc_immune) may need recalibration.

3. **apcaf_fraction_of_caf_elyada2019**: The fibroblast module underproduces antigen-presenting CAFs relative to the Elyada 2019 benchmark of 10.3%. Model median is ~1.4%. The k_PSC_to_apCAF rate or apCAF death rate may be miscalibrated.

### Production/degradation non-identifiabilities are real

The 4 correlated pairs in the 96-param no-noise config reflect fundamental limitations of snapshot data: steady-state observables constrain the ratio of production/degradation rates but not the absolute turnover. Time-course or perturbation data would be needed to break these degeneracies.

### The without-noise MMD failure reveals residual model-data tension

Even with the correct prior and all 193 parameters, the model's prior predictive distribution doesn't fully encompass the observed data (MMD p=0.03-0.04 without noise). This tension is concentrated in the 3 structurally failing observables — fixing those would likely resolve the MMD failure even without noise.

---

## Methodological Notes

### Observation noise is the correct default

Adding noise to training data is the right thing to do — it matches the actual data-generating process (noisy measurements of biological quantities). The SA97 run's treatment of calibration targets as exact values was incorrect. Future runs should always include noise injection.

### Prior truncation was removed

During this investigation, truncation bounds were removed from all 193 parameters in pdac_priors.csv. The priors are now plain lognormal (Normal in log-space). This eliminated a 162s → 0.05s sampling bottleneck (rejection sampling) and allows posteriors to explore beyond the previous artificial bounds. The bounded version is preserved as pdac_priors_bounded.csv for reproducibility with the existing simulation pool.

---

## Files

| File | Contents |
|------|----------|
| `figures/ablation_study/ablation_summary.csv` | Summary table (4 rows × 15 metrics) |
| `figures/ablation_study/recovery_r2_*.csv` | Per-parameter R² for each config |
| `workflows/ablation_study.py` | Ablation script |
| `parameters/pdac_priors.csv` | Updated priors (no truncation) |
| `scenarios/pdac_priors_bounded.csv` | Original priors (with truncation, for pool compatibility) |