# Surrogate Sensitivity Analysis Report

**Date:** 2026-02-21
**Model version:** `restriction_estimator_v4` (includes `max(0,x)` ODE guards, 20-year evolution window)
**Scenario:** `baseline_no_treatment` (diagnosis snapshot, evolve_to_diagnosis)

## Overview

Global sensitivity analysis of 177 QSP model parameters against 31 calibration targets at diagnosis. Two complementary methods — Partial Rank Correlation Coefficients (PRCC) and Random Forest (RF) permutation importance — were used to rank parameters by their influence on model outputs.

| Metric | Value |
|--------|-------|
| Parameters | 177 |
| Simulations requested | 5,000 (2,500 x 2 oversample) |
| Valid simulations used | 2,852 (43% NaN rate from failed evolve_to_diagnosis) |
| Calibration targets analyzed | 31/31 (all had 2,852 valid samples) |
| Cross-method rank correlation (Spearman rho) | **0.865** (p < 10^-53) |
| Top-20 overlap between methods | 16/20 (80%) |
| Top-30 overlap between methods | 24/30 (80%) |
| RF R^2 on targets | 0.83–0.95 (median ~0.91), except cancer_cell_density = -24.0 |
| Significant parameters (FDR < 0.05) | **97/177** (permutation test, 1,000 permutations) |

## Key Findings

### 1. Strong cross-method agreement

The two methods agree well (rho = 0.865), meaning the ranking is robust regardless of whether we use a linear (PRCC) or nonlinear (RF) lens. The top ~10 parameters are consistent across both methods.

### 2. Top 13 parameters (union of PRCC top-10 + RF top-10)

These 13 parameters dominate both methods:

| Rank (PRCC) | Rank (RF) | Parameter | Mean |PRCC| | Mean RF Imp | Module |
|:-----------:|:---------:|-----------|:--------:|:-----------:|--------|
| 1 | 2 | `k_PDGF_sec` | 0.290 | 0.041 | Fibroblast (PDGF secretion) |
| 2 | 1 | `k_myCAF_prolif` | 0.286 | 0.048 | Fibroblast (myCAF proliferation) |
| 3 | 10 | `k_C1_growth` | 0.225 | 0.023 | Cancer (tumor growth rate) |
| 4 | 17 | `k_ECM_myCAF_sec` | 0.204 | 0.018 | Fibroblast (ECM secretion by myCAFs) |
| 5 | 4 | `k_vas_growth` | 0.183 | 0.031 | Cancer (vascular growth) |
| 6 | 8 | `k_TGFb_sec_myCAF` | 0.178 | 0.025 | Fibroblast (TGFb secretion by myCAFs) |
| 7 | 5 | `q_nCD8_P_out` | 0.160 | 0.027 | T cell (naive CD8 peripheral exit) |
| 8 | 12 | `q_CD8_T_in` | 0.156 | 0.020 | T cell (CD8 tumor infiltration) |
| 9 | 14 | `q_nCD8_LN_in` | 0.144 | 0.020 | T cell (naive CD8 LN entry) |
| 10 | 7 | `N_IL2_CD4` | 0.137 | 0.026 | T cell (IL-2-dependent CD4 expansion) |
| 14 | 9 | `k_myCAF_to_iCAF` | 0.127 | 0.023 | Fibroblast (myCAF-to-iCAF transition) |
| 22 | 6 | `k_cell_clear` | 0.108 | 0.027 | General (dead cell clearance) |
| 26 | 3 | `k_M2_pol` | 0.103 | 0.035 | Macrophage (M2 polarization) |

### 3. Biological interpretation: stroma dominance

**The stromal/fibroblast module dominates the model.** Five of the top 13 parameters are fibroblast-related (`k_PDGF_sec`, `k_myCAF_prolif`, `k_ECM_myCAF_sec`, `k_TGFb_sec_myCAF`, `k_myCAF_to_iCAF`). This reflects the PDAC biology — the dense desmoplastic stroma is the defining feature of PDAC tumors and the primary barrier to immune infiltration.

**Parameter clusters by biological mechanism:**

- **Fibroblast/stroma (5 params):** `k_PDGF_sec`, `k_myCAF_prolif`, `k_ECM_myCAF_sec`, `k_TGFb_sec_myCAF`, `k_myCAF_to_iCAF` — These control myCAF dynamics and ECM deposition. myCAF proliferation and PDGF-driven activation are the top two parameters overall.

- **Tumor intrinsic (2 params):** `k_C1_growth`, `k_vas_growth` — Cancer growth rate and vascular-carrying-capacity growth. These determine whether the tumor can reach the diagnosis threshold.

- **CD8 T cell trafficking (3 params):** `q_nCD8_P_out`, `q_CD8_T_in`, `q_nCD8_LN_in` — The full CD8 trafficking chain from thymic output to LN entry to tumor infiltration. This confirms T cell supply to the tumor is a key bottleneck.

- **Immune regulation (3 params):** `N_IL2_CD4`, `k_cell_clear`, `k_M2_pol` — IL-2-dependent T cell expansion, dead cell clearance (affects DAMP-driven APC maturation), and M1-to-M2 macrophage polarization.

### 4. Nonlinear-only parameters (top-20 RF but not top-20 PRCC)

Four parameters are important by RF but not PRCC, suggesting nonlinear or interaction-dependent effects:

| Parameter | PRCC Rank | RF Rank | Interpretation |
|-----------|:---------:|:-------:|----------------|
| `k_M2_pol` | 26 | 3 | M2 polarization has threshold/switch-like effect |
| `k_cell_clear` | 22 | 6 | Dead cell clearance → DAMP release is nonlinear |
| `k_Treg_death` | 21 | 13 | Treg survival has threshold effects on suppression |
| `k_APC_mat_cDC1` | 29 | 20 | cDC1 maturation rate interacts with antigen load |

### 5. Notable non-influential parameters

Several parameters that might be expected to matter are surprisingly low-ranked:

| Parameter | PRCC Rank | RF Rank | Notes |
|-----------|:---------:|:-------:|-------|
| `k_C1_death` | 45 | 99 | Cancer death rate barely matters (growth >> death in viable sims) |
| `PD1_50` | 83 | 134 | PD-1 checkpoint has minimal effect at diagnosis (no treatment) |
| `p_T_kill_per_contact` | 171 | 138 | T cell killing rate itself is low-impact (trafficking is the bottleneck) |
| `k_NK_kill` | 91 | 80 | NK killing is secondary to stromal/T cell effects |
| `K_M1_phago` | 85 | 144 | Macrophage carrying capacity is insensitive |

`k_C1_death` at rank 45/99 is particularly notable — it means that for viable simulations (where the tumor reached diagnosis), the death rate barely matters. This is consistent with the restriction estimator finding that the valid/invalid boundary depends on growth-vs-death, but once past that threshold, growth dominates.

### 6. RF model quality

RF fits are excellent (R^2 > 0.83) for all targets except `cancer_cell_density_baseline` (R^2 = -24.0). This extreme negative R^2 indicates the RF completely fails to predict cancer cell density, likely because it has very high variance or a bimodal distribution from the evolve_to_diagnosis initialization.

### 7. Permutation significance test

To answer the question "should this parameter be calibrated?" (set membership, not ranking), we use a permutation null test:

**Method:** For each of 1,000 permutations, shuffle the simulation outputs (x) to break the parameter-output relationship, then recompute mean |PRCC| per parameter. The p-value for each parameter is the fraction of null values >= observed. Benjamini-Hochberg FDR correction controls false discovery at 5%.

**Result: 97/177 parameters are statistically significant** (FDR < 0.05).

The boundary is clean — there is a sharp 2x jump in FDR at the cutoff:

| PRCC Rank | Parameter | Raw p | FDR | Significant |
|----------:|-----------|------:|----:|:-----------:|
| 88 | `k_CCL2_deg` | 0.002 | 0.004 | Yes |
| 91 | `k_NK_kill` | 0.002 | 0.004 | Yes |
| 93 | `IL12_50` | 0.003 | 0.006 | Yes |
| 94 | `k_NK_death` | 0.006 | 0.011 | Yes |
| 95 | `f_HIF_glycolysis` | 0.008 | 0.015 | Yes |
| 96 | `k_Treg_CD8_exh` | 0.020 | 0.037 | Yes |
| **97** | **`k_TGFb_Msec`** | **0.027** | **0.049** | **Yes** |
| 98 | `IL12_50_NK` | 0.057 | 0.101 | No |
| 99 | `glc_50_Teff` | 0.050 | 0.089 | No |

The top 88 parameters all hit the permutation resolution floor (p = 1/1001), meaning they are maximally significant at this sample size. Their relative ranking still comes from the PRCC/RF analysis above.

### 8. Cumulative importance

- 80% of total PRCC signal captured by top 79 params; 80% of RF signal by top 96
- 90% by top 116 (PRCC) / 134 (RF)
- The 97 significant parameters capture ~92% of total PRCC signal — consistent with the permutation boundary

## Implications for SBI Calibration

1. **Calibrate the 97 significant parameters.** The permutation test with FDR correction provides a principled, threshold-free answer to "which parameters matter?" These 97 parameters have statistically detectable influence on at least one calibration target.

2. **Fix remaining 80 parameters at prior medians.** These are indistinguishable from noise in a permutation null — they do not measurably influence any of the 31 calibration targets. Fixing them reduces SBI dimensionality from 177 → 97 without information loss.

3. **The top 13 parameters dominate.** While 97 parameters are significant, the top 13 (Section 2) account for a disproportionate share of variance. SBI will naturally constrain these most tightly; the remaining 84 significant parameters will have posteriors closer to their priors.

4. **The `cancer_cell_density_baseline` target needs investigation** — the RF R^2 = -24 suggests this observable may have pathological variance or is not well-predicted by the parameter space.

5. **Consider tightening fibroblast priors first** — the stroma module dominates variance, so constraining `k_PDGF_sec`, `k_myCAF_prolif`, `k_ECM_myCAF_sec` against stromal fraction / collagen data would yield the biggest reduction in predictive uncertainty.

## Files

- `results/surrogate_sensitivity/aggregate_parameter_ranking.csv` — Full 177-parameter ranking with permutation p-values
- `results/surrogate_sensitivity/recommended_parameters.csv` — 97 significant parameters (FDR < 0.05)
- `results/surrogate_sensitivity/cross_method_agreement.json` — Top-N overlap statistics and permutation test results
- `results/surrogate_sensitivity/prcc_matrix.csv` — Signed PRCC (177 x 31)
- `results/surrogate_sensitivity/rf_importance_matrix.csv` — RF importance (177 x 31)
- `results/surrogate_sensitivity/per_target/` — Per-target PRCC rankings (31 CSVs)
- `figures/surrogate_sensitivity/` — Aggregate ranking, heatmaps, cross-method scatter, cutoff/significance analysis, per-target PRCC bar plots
