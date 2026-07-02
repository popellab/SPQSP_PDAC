# NPE Diagnostics: All-Parameters Baseline Run

**Date:** 2026-02-27
**Context:** NPE trained on 1107 valid sims (193 params, 35 observables), baseline_no_treatment scenario
**Model version:** baseline_allparams_v1
**Key result:** MMD p = 0.747 (not misspecified); median R² = -0.008; 1 param R² > 0.5; 4 boundary-piling flags

---

## 0. Summary Dashboard

| Metric | Value (this run) | Previous (SA97) | Δ |
|--------|-----------------|-----------------|---|
| Training samples | 1107 (from 2500 raw, 56% NaN rate) | 2567 (from 5000, 49% NaN) | -1460 |
| Parameters | 193 (all) | 97 (SA-restricted) | +96 |
| Observables | 35 | 30 | +5 |
| MMD p-value | **0.747** | 0.017 | No longer misspecified |
| Recovery R² median | **-0.008** | 0.083 | Collapsed |
| Recovery R² > 0.5 | **1** / 193 | 8 / 97 | -7 |
| KS calibration rejects (p<0.05) | **21** / 193 | 8 / 97 | +13 |
| Boundary piling (>20%) | **4** / 193 | 10 / 97 | -6 |
| Observables out of range | **1** / 35 | 1 / 30 | Same |
| Prior predictive p < 0.05 | **3** / 35 | 7 / 30 | Improved |
| Prior predictive p < 0.10 | **5** / 35 | 9 / 30 | Improved |
| PPC obs in 90% CI | **30** / 35 | 21 / 30 | Improved |
| Strong posterior correlations | **0** | 7 pairs | All resolved |
| Learning curve (final R² median) | **-0.007** | 0.085 | Flat from start |
| Seed stability (median R² std) | **0.020** | 0.012 | Slightly noisier |
| Dim. sweep best median R² | **0.313 at D=10** | 0.242 at D=8 | Better peak |

---

## 1. Parameter Recovery (R²)

Only **1 of 193 parameters** has R² > 0.5 — a dramatic collapse from the SA97 run's 8/97.

### Top-recovered parameters

| Rank | Parameter | R² | R² (SA97) | Role |
|------|-----------|-----|-----------|------|
| 1 | k_M2_pol | **0.762** | 0.726 | M1→M2 polarization rate |
| 2 | k_cell_clear | 0.498 | 0.742 | Dead cell clearance |
| 3 | k_Mac_rec | 0.347 | 0.571 | Macrophage recruitment |
| 4 | APC0_T | 0.338 | 0.618 | Baseline APC density in tumor |
| 5 | k_vas_growth | 0.316 | — | Tumor growth rate |
| 6 | k_Treg_death | 0.283 | 0.550 | Treg death rate |
| 7 | k_lac_clear | 0.271 | — | Lactate clearance rate |
| 8 | k_O2_demand | 0.263 | — | Oxygen demand (hypoxia threshold) |
| 9 | k_iCAF_to_myCAF | 0.261 | — | iCAF→myCAF transition |
| 10 | k_lac_Csec | 0.259 | — | Lactate secretion by cancer cells |

**Notable changes from SA97:**
- k_M2_pol improved slightly (0.726 → 0.762) and is now the only R² > 0.5 parameter
- pO2_50 collapsed from 0.967 → 0.012 — was the best-recovered parameter in SA97
- k_PSC_to_apCAF dropped from 0.835 → 0.241
- k_ECM_myCAF_sec dropped from 0.534 → 0.062
- Several new parameters (k_vas_growth, k_lac_clear, k_lac_Csec, k_iCAF_to_myCAF) appeared in the top 10, all with moderate R²

### Bottom parameters (R² < -0.10)

| Parameter | R² | Note |
|-----------|-----|------|
| k_CXCL9_sec | -0.121 | CXCR3 ligand production |
| k_CCL5_sec_iCAF | -0.106 | iCAF chemokine secretion |
| TGFb_50_NK | -0.105 | TGFb NK suppression threshold |
| k_vas_myCAF_sec | -0.101 | myCAF VEGF secretion |

~106 of 193 parameters have negative R² (55%) — posterior median performs worse than predicting the prior mean.

### Interpretation

The median R² of -0.008 means the NPE has essentially **zero average parameter recovery**. This is expected: 193 parameters with only 1107 training samples gives a sample-per-dimension ratio of 5.7, vastly below the O(d²) ≈ 37,249 heuristic. The model is massively underdetermined.

However, this is an improvement in one key way: by including all parameters, we avoid selection bias from the SA pre-filtering. The fact that k_M2_pol remains well-recovered (R² = 0.762) despite doubling the parameter space confirms it has a genuinely strong, identifiable signal.

---

## 2. Calibration (SBC/ECDF)

21 of 193 parameters fail the KS test at p < 0.05 (10.9%), up from 8/97 (8.2%) in SA97. Top rejects:

| Parameter | KS statistic | KS p-value |
|-----------|-------------|-----------|
| k_ArgI_sec | 0.179 | 0.0006 |
| k_P1_deg_ID | 0.154 | 0.005 |
| q_CD8_T_in | 0.150 | 0.007 |
| k_stroma_support | 0.143 | 0.012 |
| k_CXCL9_sec | 0.142 | 0.012 |
| k_cell_clear | 0.140 | 0.014 |
| k_Treg_CD8_exh | 0.140 | 0.015 |
| N_div_base | 0.145 | 0.010 |
| k_APC_mat_cDC1 | 0.138 | 0.016 |
| k_CCL5_sec_iCAF | 0.129 | 0.030 |

The KS band width at n_test=123 is 1.36/√123 ≈ 0.123, so parameters with KS < 0.12 are within expected noise. The top rejects (k_ArgI_sec, k_P1_deg_ID, q_CD8_T_in) are genuinely miscalibrated.

---

## 3. Coverage Check

**1 of 35 observables** falls outside the training simulation range:

| Observable | Observed | Training range | z-score |
|-----------|----------|---------------|---------|
| collagen1_protein_level | 9.84 | [0.001, 9.43] | +2.05 |

This is nearly in range (observed 9.84, max 9.43), a dramatic improvement over SA97 where the out-of-range observable was cdc1_fraction_of_dc (z = +5.58).

**High |z|-score observables (still in range):**

| Observable | z-score | Interpretation |
|-----------|---------|----------------|
| tumor_glucose_ratio | +2.35 | Model tends to produce lower glucose than observed |
| collagen1_protein_level | +2.05 | Just barely out of range |
| treg_fraction_cd4 | -2.74 | Model produces far higher Treg/CD4 fraction than observed |
| treg_fraction_cd4_hiraoka2006 | -1.89 | Same direction, different calibration target |
| th_exhausted_fraction | +1.43 | Model produces less Th exhaustion than observed |

The treg_fraction_cd4 z-score of -2.74 is the most concerning — the model systematically overproduces Tregs as a fraction of CD4 cells.

---

## 4. Boundary Piling

Only **4 parameters** show >20% piling (vs 10 in SA97):

| # | Parameter | Direction | Piling % | Prior [lower, median, upper] |
|---|-----------|-----------|----------|-------------------------------|
| 1 | VEGF_norm_50 | upper | 33.6% | [254.7, **500**, 981.5] |
| 2 | k_O2_demand | upper | 26.8% | [0.181, **0.29**, 0.465] |
| 3 | k_PSC_to_apCAF | upper | 22.4% | [0.0014, **0.01**, 0.074] |
| 4 | k_ECM_myCAF_sec | upper | 20.5% | [1.2e-9, **5e-9**, 2.0e-8] |

**All 4 pile at the upper bound.** The biological interpretation:
- **VEGF_norm_50 and k_O2_demand**: Model wants deeper hypoxia / higher VEGF baseline — consistent with SA97 findings
- **k_PSC_to_apCAF**: More apCAF activation — model needs to produce more apCAFs
- **k_ECM_myCAF_sec**: More ECM production by myCAFs

The dramatic reduction from 10 → 4 boundary-piling parameters is partly because the wider all-parameter prior space provides more room for the posterior. The SA97 run's extreme f_cDC1 piling (93.7%) and cDC maturation pilings (>84%) are absent here because those priors were likely widened or the additional parameters provide compensatory degrees of freedom.

### Action items

- **Widen VEGF_norm_50 upper bound** (33.6% piling, most extreme)
- **Widen k_O2_demand upper bound** (26.8% piling, consistent signal across runs)

---

## 5. Misspecification (MMD)

MMD p-value: **0.747** (not significant).

This is the single biggest improvement over SA97 (p = 0.017). With all 193 parameters, the prior predictive distribution is broad enough to encompass the observed data. The model is **not misspecified** at the joint level.

This doesn't mean the model is well-calibrated — it means the observed data is plausible under the model's prior. The SA97 run was misspecified because SA pre-filtering excluded parameters needed to match certain observables.

---

## 6. Prior Predictive p-values

**3 observables** have p < 0.05 (down from 7 in SA97):

| Observable | p-value | Tail | Interpretation |
|-----------|---------|------|----------------|
| collagen1_protein_level | 0.000 | upper | Observed 9.84 above virtually all sims |
| tumor_glucose_ratio | 0.022 | upper | Observed 0.92 above most sims — model depletes glucose more than reality |
| apcaf_fraction_of_caf_elyada2019 | 0.042 | upper | Observed 10.3% above most sims |

**2 more at p < 0.10:**
- treg_fraction_cd4 (p = 0.072, lower tail) — observed 11.1% well below model mean
- apcaf_fraction_of_caf (p = 0.078, upper tail)

### Compared to SA97

The 4 resolved failures (psc_activation_ratio, stromal_fraction, cancer_cell_density, tam_to_cdc1_ratio) were all stroma/DC-related. With 193 parameters, the model now has enough flexibility to produce plausible values for these observables from the prior alone.

The 3 remaining failures are structural:
1. **collagen1_protein_level** (p = 0.000): The ECM module consistently overproduces collagen. Prior max is 9.43, observed is 9.84. This is a hard model ceiling, not a prior width issue.
2. **tumor_glucose_ratio** (p = 0.022): The metabolism module depletes glucose too aggressively.
3. **apcaf_fraction_of_caf_elyada2019** (p = 0.042): The fibroblast module underproduces apCAFs relative to the Elyada 2019 benchmark.

---

## 7. LOO Predictive Check (Misspecification Decomposition)

D²_full = 30.39 (down from 143.5 in SA97). Top misspecification drivers:

| Observable | Influence (ΔD²) | % of total |
|-----------|-----------------|-----------|
| cdc2_density_baseline | 4.95 | 16.3% |
| cdc1_density_baseline | 4.06 | 13.4% |
| collagen1_protein_level | 3.59 | 11.8% |
| apcaf_fraction_of_caf_elyada2019 | 2.16 | 7.1% |
| icaf_fraction_of_caf_elyada2019 | 1.87 | 6.2% |
| cd4_helper_density | 1.77 | 5.8% |
| treg_fraction_cd4 | 1.67 | 5.5% |
| tumor_glucose_ratio | 1.32 | 4.3% |

Top 3 observables account for 41.5% of joint misspecification.

**Key change from SA97:** The D² dropped nearly 5x (143.5 → 30.39). The misspecification is now dominated by DC densities (cdc1, cdc2) and collagen, rather than the broad stroma/hypoxia/cDC1 issues that dominated the SA97 run. This indicates the additional parameters successfully absorbed much of the previous model-data tension.

---

## 8. Posterior Predictive Check

**30 of 35 observables** fall within the 90% PPC interval (up from 21/30 in SA97). 5 failures:

| Observable | Observed | 90% PPC CI | PPC p-value |
|-----------|----------|-----------|-------------|
| apcaf_fraction_of_caf | 0.064 | [0.0005, 0.057] | 0.092 |
| apcaf_fraction_of_caf_elyada2019 | 0.103 | [0.0005, 0.057] | 0.054 |
| treg_fraction_cd4 | 0.111 | [0.251, 1.000] | 0.039 |
| tumor_glucose_ratio | 0.924 | [0.048, 0.850] | 0.013 |
| collagen1_protein_level | 9.84 | [0.331, 8.457] | 0.000 |

### Interpretation

The 5 PPC failures align with the prior predictive failures:
- **collagen1_protein_level**: Hard structural ceiling — the model can't produce enough collagen even with optimized posteriors
- **tumor_glucose_ratio**: Model depletes glucose too much; posterior can't rescue this
- **treg_fraction_cd4**: Model overproduces Tregs as fraction of CD4; observed 11.1% is far below the posterior's 90% CI lower bound of 25.1%
- **apcaf fractions**: Model underproduces apCAFs relative to both calibration targets

---

## 9. Posterior Correlations

**No strong correlations** (|r| > 0.5) detected — down from 7 pairs in SA97.

This is likely a consequence of the poor overall parameter recovery: when the NPE can't constrain most parameters, the posteriors remain broad and prior-like, so correlated pairs don't emerge. The SA97 run's correlations (e.g., k_CD4_death/k_cell_clear at r = 0.906) reflected genuine structural non-identifiabilities that are now masked by the high-dimensional noise floor.

This shouldn't be interpreted as the non-identifiabilities being resolved — they're hidden by the curse of dimensionality.

---

## 10. Learning Curve

| Fraction | N_train | Median R² | R² > 0.5 |
|----------|---------|-----------|----------|
| 10% | 110 | -0.025 | 0 |
| 20% | 221 | -0.019 | 0 |
| 40% | 442 | -0.012 | 0 |
| 60% | 664 | -0.006 | 0 |
| 80% | 885 | -0.010 | 2 |
| 100% | 1107 | -0.007 | 1 |

The learning curve is **essentially flat** — no improvement from 10% to 100%. The R² never becomes meaningfully positive. This confirms that 1107 samples across 193 dimensions is massively insufficient. At this dimensionality, learning can't begin until ~O(d) ≈ 193 samples are available per parameter, requiring roughly 37,000+ simulations.

---

## 11. Seed Stability

Median R² std across 5 seeds: **0.020** (slightly higher than SA97's 0.012).

Top 10 most stable parameters:

| Parameter | R² (mean ± std) |
|-----------|-----------------|
| k_M2_pol | 0.740 ± 0.034 |
| k_cell_clear | 0.582 ± 0.031 |
| APC0_T | 0.352 ± 0.074 |
| k_vas_growth | 0.315 ± 0.037 |
| k_Treg_death | 0.305 ± 0.034 |
| k_lac_Csec | 0.292 ± 0.018 |
| k_O2_demand | 0.281 ± 0.016 |
| k_iCAF_to_myCAF | 0.263 ± 0.023 |
| k_lac_clear | 0.249 ± 0.051 |
| k_PSC_to_apCAF | 0.249 ± 0.013 |

k_M2_pol and k_cell_clear are the most robust — stable across both runs and seeds. APC0_T has the highest variability (std = 0.074) among the top 10, suggesting its recovery depends on which training samples are included.

---

## 12. Dimensionality Sweep

| D (n_params) | Median R² | R² > 0.5 |
|-------------|-----------|----------|
| 10 | **0.313** | 2 |
| 20 | 0.249 | 2 |
| 40 | 0.184 | 2 |
| 80 | 0.051 | 2 |
| 120 | 0.026 | 2 |
| 193 | -0.007 | 1 |

**Best performance at D=10** with median R² = 0.313 (vs 0.242 at D=8 in SA97). The sweep confirms the curse of dimensionality: reducing from 193 → 10 parameters improves median R² by 45x.

The consistent 2 parameters with R² > 0.5 across D=10 through D=120 (k_M2_pol and k_cell_clear) are the only truly robust recoveries. Everything else degrades as dimensionality increases.

---

## 13. Z-score / Contraction Analysis

The z-score vs contraction plot reveals which parameters the NPE actually learned about. Parameters with high contraction (posterior much narrower than prior) and z-score near 0 are well-calibrated.

**Highest contraction (most posterior narrowing):**

| Parameter | Contraction (median) | z-score (median) | Interpretation |
|-----------|---------------------|-------------------|----------------|
| VEGF_norm_50 | -3.65 | +0.01 | Posterior expands vs prior (negative = inflated) |
| k_O2_demand | -3.59 | -0.003 | Posterior expands vs prior |
| k_M2_pol | **+0.637** | -0.04 | Strong narrowing, well-centered |
| k_PSC_to_apCAF | +0.323 | -0.11 | Moderate narrowing |
| k_Treg_pro | -0.372 | +0.19 | Slight expansion |

The large negative contraction for VEGF_norm_50 and k_O2_demand means the NPE is producing *wider* posteriors than priors for these parameters — a sign of prior-posterior mismatch where the NPE can't reconcile the prior constraint with the training data (consistent with boundary piling). k_M2_pol has the strongest positive contraction (+0.637), meaning the posterior concentrates to ~36% of the prior width.

---

## 14. Summary: What Changed and Why

### Improvements over SA97
1. **MMD misspecification resolved** (p = 0.747 vs 0.017) — broader parameter space encompasses observations
2. **Fewer prior predictive failures** (3/35 vs 7/30) — additional parameters provide compensatory flexibility
3. **Better PPC** (30/35 vs 21/30 in 90% CI) — broader posteriors are more conservative
4. **Less boundary piling** (4 vs 10 flags) — wider prior space
5. **Higher peak dimensional sweep R²** (0.313 at D=10 vs 0.242 at D=8)

### Regressions from SA97
1. **Recovery collapsed** (median R² = -0.008 vs 0.083) — 193 parameters with 1107 sims is massively underdetermined
2. **More KS calibration failures** (21 vs 8) — NPE can't calibrate in high-D
3. **Fewer well-recovered parameters** (1 vs 8 with R² > 0.5)
4. **Learning curve flat from start** — no evidence of learning at this sample size
5. **Slightly less seed-stable** (std = 0.020 vs 0.012)

### Root Cause

The fundamental tradeoff: including all 193 parameters eliminates selection bias and resolves model misspecification, but the sample-to-dimension ratio (1107/193 ≈ 5.7) is far too low for the NPE to learn meaningful posterior structure. The NaN rate also increased (56% vs 49%), reducing usable training data.

---

## 15. Actionable Recommendations

### Immediate (prior/data fixes)
1. **Widen VEGF_norm_50 and k_O2_demand priors** — both pile at upper bounds across runs, with posterior expansion (negative contraction)
2. **Investigate collagen1_protein_level ceiling** — structural model limit (max sim = 9.43, obs = 9.84). The ECM secretion or degradation balance needs adjustment.
3. **Fix Treg/CD4 ratio** — z = -2.74 and PPC failure. Model systematically overproduces Tregs as fraction of CD4. Consider auditing `k_apCAF_Treg`, `k_Treg_act`, and the CCL5-mediated Treg recruitment pathway.
4. **Address glucose ratio** — tumor_glucose_ratio PPC fails. The metabolism module may deplete glucose too aggressively. Check `k_glc_tumor`, `k_glc_immune`, and `f_HIF_glycolysis` interactions.

### Strategic (architecture changes)
5. **Generate more simulations** — the 56% NaN rate is expensive. Target 10,000 raw sims to get ~4,400 valid, which would match the SA97 data regime at full dimensionality. Investigate why NaN rate increased (likely new parameter regions hitting stiff ODE boundaries).
6. **Use SA-informed parameter grouping** — the dimensionality sweep shows D=10 recovers well. Use the SA97 results to identify the top ~30-50 most sensitive parameters and fix the rest at prior medians. This bridges the SA97 and all-params approaches.
7. **Consider sequential NPE** — train an initial NPE on the top-10 parameters (where R² = 0.313), condition on those posteriors, then train a second NPE on the next tier. This staged approach can handle higher total dimensionality.
8. **Add time-course observables** — the 5 new observables helped (3 fewer prior predictive failures), but they're all diagnosis-snapshot values. Adding dynamic observables (fold-changes over treatment, time-to-event) would provide orthogonal information to break degeneracies.

---

## Appendix: Diagnostic Files

All diagnostic outputs are in `figures/sbi_baseline_allparams/`:

### CSV data
| File | Contents |
|------|----------|
| `diagnostics/summary.json` | Key metrics |
| `diagnostics/recovery_r2.csv` | Per-parameter R² (193 rows) |
| `diagnostics/boundary_piling.csv` | Posterior-vs-prior boundary analysis |
| `diagnostics/calibration_ks.csv` | SBC KS test results |
| `diagnostics/coverage.csv` | Observable coverage check |
| `diagnostics/prior_predictive_pvalues.csv` | Prior predictive p-values |
| `diagnostics/loo_predictive.csv` | LOO misspecification decomposition |
| `diagnostics/posterior_predictive.csv` | PPC results |
| `diagnostics/posterior_correlations.csv` | Posterior correlations (empty) |
| `diagnostics/learning_curve.csv` | R² vs training fraction |
| `diagnostics/seed_stability.csv` | 5-seed stability results |
| `diagnostics/dimensionality_sweep.csv` | R² vs parameter count |
| `diagnostics/z_score_contraction.csv` | Z-score and contraction per parameter |
| `posterior_summary.csv` | Posterior summary statistics |
| `posterior_samples.csv` | Raw posterior samples |

### Plots
| File | Contents |
|------|----------|
| `recovery.png` | R² histogram across 193 parameters |
| `posterior_vs_prior.png` | Top parameters posterior vs prior |
| `posterior_marginals.png` | All posterior marginal distributions |
| `posterior_correlations.png` | Correlation matrix |
| `posterior_pairs_top6.png` | Scatterplot of top 6 correlated pairs |
| `prior_predictive_pvalues.png` | Bar chart of prior predictive p-values |
| `prior_predictive_histograms.png` | Prior predictive coverage |
| `prior_vs_posterior_pvalues.png` | Prior vs posterior p-values comparison |
| `posterior_predictive_check.png` | PPC results |
| `calibration_ecdf.png` | SBC ECDF bands |
| `loo_predictive.png` | LOO influence bar chart |
| `z_score_contraction.png` | Z-score vs contraction scatter |
| `learning_curve.png` | R² vs training fraction |
| `seed_stability.png` | Stability across 5 seeds |
| `dimensionality_sweep.png` | R² vs parameter count |
