---
format:
  html:
    theme:
      light: flatly
      dark: darkly
---

# Parameter Coverage Audit

**[============--------] 115/188 parameters covered (61%)**

---

## What should I extract next?

Ranked by impact (sigma × PRCC × effective-coverage). **#obs** = count of observables where this param is PRCC-significant (informs where to look for constraining data). **S1 eff** = how much submodel inference actually contracted the posterior from the prior; low values mean the existing target is ineffective and the param re-enters the priority list.

### New target — literature extraction

| # | Parameter | Type | Sigma | PRCC | #obs | Targets | Score | Models |
|--:|-----------|------|------:|-----:|----:|:-------:|------:|-------:|
| 1 | `k_P1_up_apCAF` | rate | 1.20 | #111 | 5 | --- | 0.52 | --- |
| 2 | `k_apCAF_death` | death | 1.15 (S1) | #80 | 4 | --- | 0.48 | --- |
| 3 | `y_Ag_per_cell` | rate | 1.50 | #135 | 2 | --- | 0.46 | --- |
| 4 | `d_crit_APC` | biophysical | 0.50 | #15 | 15 | --- | 0.46 | --- |
| 5 | `k_APC_mat_cDC2` | rate | 0.70 | #70 | 2 | --- | 0.45 | --- |
| 6 | `phi_cross_50` | EC50 | 2.00 |  | --- | --- | 0.45 | --- |
| 7 | `k_Treg_act` | activation | 1.00 | #109 | 5 | --- | 0.44 | --- |
| 8 | `k_CD4_death` | death | 0.50 | #25 | 10 | --- | 0.44 | --- |
| 9 | `k_iCAF_death` | death | 1.16 (S1) | #91 | 6 | --- | 0.44 | --- |
| 10 | `P1_50_apCAF` | EC50 | 1.20 | #125 | 7 | --- | 0.43 | --- |
| 11 | `k_Cy_Treg_deplete` | rate | 1.00 | #116 | 3 | --- | 0.41 | --- |
| 12 | `N_IL2_CD4` | count | 0.50 | #38 | 8 | --- | 0.40 | 6/6 |
| 13 | `N_div_costim` | count | 0.50 | #50 | 15 | --- | 0.37 | --- |
| 14 | `TGFb_baseline` | rate | 0.70 | #95 | 6 | --- | 0.36 | 6/6 |
| 15 | `k_Th_act` | activation | 0.70 | #99 | 5 | --- | 0.35 | --- |
| 16 | `N_div_base` | count | 0.50 | #71 | 14 | --- | 0.32 | --- |
| 17 | `K_apCAF_Treg` | rate | 1.00 | #134 | 2 | --- | 0.31 | --- |
| 18 | `k_vas_deg` | degradation | 0.50 | #77 | 1 | --- | 0.30 | --- |
| 19 | `k_apCAF_Treg` | rate | 1.00 | #136 | 3 | --- | 0.30 | --- |
| 20 | `k_IFNg_response` | rate | 1.00 |  | --- | --- | 0.30 | --- |
| | *...53 more* | | | | | |

### Improve existing — ineffective target (S1 eff < 30%)

These params have at least one submodel target, but stage-1 inference barely moved the posterior. Likely causes: target is wrong, forward-model code is wrong, or the prior needs widening. Extract new data, re-audit existing target, or widen the prior — not the same action as the *new target* list above.

| # | Parameter | Type | Sigma | PRCC | #obs | Targets | S1 eff | Score | Models |
|--:|-----------|------|------:|-----:|----:|:-------:|-----:|------:|-------:|
| 1 | `k_cell_clear` | degradation | 1.01 (S1) | #9 | 16 | 1 | 0% | 0.97 | --- |
| 2 | `TGFb_50_CAF_act` | EC50 | 1.48 (S1) | #81 | 8 | 4 | 0% | 0.87 | --- |
| 3 | `q_nCD8_LN_in` | trafficking | 0.83 (S1) | #8 | 18 | 1 | 0% | 0.80 | --- |
| 4 | `k_CD8_exh_death` | death | 1.02 (S1) | #53 | 6 | 1 | 0% | 0.74 | --- |
| 5 | `f_stroma_max` | fraction | 0.71 (S1) | #1 | 19 | 1 | 0% | 0.71 | --- |
| 6 | `n_CD4_clones` | count | 1.09 (S1) | #69 | 9 | 1 | 0% | 0.71 | 6/6 |
| 7 | `q_nCD8_P_out` | trafficking | 0.66 (S1) | #10 | 16 | 1 | 0% | 0.63 | --- |
| 8 | `n_IL1` | count | 0.79 (S1) | #40 | 13 | 2 | 0% | 0.63 | --- |
| 9 | `k_CAF_encounter` | rate | 1.03 (S1) | #78 | 13 | 1 | 0% | 0.62 | 1/6 |
| 10 | `k_stroma_support` | rate | 0.70 (S1) | #33 | 17 | 2 | 0% | 0.59 | --- |
| 11 | `ArgI_50_Treg` | EC50 | 1.02 (S1) | #86 | 7 | 1 | 0% | 0.57 | --- |
| 12 | `k_APC_mig` | rate | 0.79 (S1) | #58 | 10 | 1 | 0% | 0.55 | --- |
| 13 | `k_myCAF_death` | death | 1.11 (S1) | #59 | 13 | 1 | 0% | 0.55 | 1/6 |
| 14 | `k_CD8_death` | death | 0.63 (S1) | #31 | 18 | 2 | 0% | 0.53 | --- |
| 15 | `k_Treg_death` | death | 0.54 (S1) | #2 | 17 | 2 | 0% | 0.53 | --- |
| | *...37 more* | | | | | |

### Low-hanging fruit

EC50s → `direct_fit` (Hill curve from dose-response). Secretion rates → `batch_accumulation` (ELISA).

| Parameter | Type | Sigma | PRCC | Units |
|-----------|------|------:|-----:|-------|
| `TGFb_50_CAF_act` | EC50 | 1.48 | #81 | nanomolarity |
| `IL1_50` | EC50 | 0.99 | #42 | nanomolarity |
| `k_TGFb_sec_myCAF` | secretion | 0.87 | #18 | nanomolarity*liter/(cell*day) |
| `ArgI_50_Treg` | EC50 | 1.02 | #86 | nanomolarity |
| `TGFb_50_IL1R1` | EC50 | 0.80 | #72 | nanomolarity |
| `TGFb_50_APC` | EC50 | 0.91 | #41 | nanomolarity |
| `phi_cross_50` | EC50 | 2.00 |  | dimensionless |
| `P1_50_apCAF` | EC50 | 1.20 | #125 | molarity |
| `TGFb_50` | EC50 | 1.08 | #90 | nanomolarity |
| `k_col_myCAF_sec` | secretion | 0.65 | #4 | milligram/(cell*day) |
| `k_IL2_sec` | secretion | 0.66 | #30 | nanomole/cell/hour |
| `CXCL12_50_Texcl` | EC50 | 0.64 | #16 | nanomolarity |
| `IL12_50` | EC50 | 0.71 | #108 | picomolarity |
| `CD8_50_Tcap` | EC50 | 1.00 |  | cell |
| `apCAF_50_Tcap` | EC50 | 1.00 |  | cell |

### Parameter group gaps

Extracting one member constrains the whole group via partial pooling.

| Group | Coverage | Still needed |
|-------|:--------:|-------------|
| CAF_death_rates | 1/4 | `k_qpsc_death`, `k_iCAF_death`, `k_apCAF_death` |
| IL6_EC50s | 2/3 | `IL6_50_iCAF` |

*Remaining by type: **rate** 55 | **EC50** 34 | **secretion** 21 | **death** 15 | **degradation** 12 | **count** 12 | **fraction** 9 | **proliferation** 8 | **trafficking** 7 | **biophysical** 5 | **exhaustion** 4 | **activation** 3 | **polarization** 2 | **killing** 1*

---

## Are my existing targets working?

### Inference DAG

![Inference DAG](parameter_audit_report_files/inference_dag.svg)

### Weak targets

These have submodel targets but the data barely moved the prior. The target may be uninformative, or the translation sigma is too large.

| Parameter | Contraction | Targets |
|-----------|:-----------:|---------|
| `TGFb_50_CAF_act` | -348% | TGFb_50_CAF_act_bachem1992, caf_interconversion_biffi2019, k_psc_activation_myCAF_bachem1993, psc_activation_myCAF_biffi2019 |
| `k_qpsc_death` | -313% |  |
| `k_iCAF_death` | -176% |  |
| `q_nCD8_LN_in` | -175% | q_nCD8_LN_in |
| `k_apCAF_death` | -170% |  |
| `k_myCAF_death` | -152% | k_myCAF_death_kisseleva2012 |
| `n_IL1` | -147% | IL1_50_CXCL8_ciminieri2023, IL1_50_IL6_ciminieri2023 |
| `TGFb_50` | -137% | TGFb_50 |
| `n_Texh` | -81% | E_50_Texh_IFNg_saitakis2017, E_50_Texh_TNFa_saitakis2017 |
| `q_nCD8_P_out` | -75% | q_nCD8_P_out |
| `TGFb_50_APC` | -69% | TGFb_50_APC_PDAC_deriv001 |
| `k_CD8_death` | -60% | k_CD8_basal_exh, k_CD8_death |
| `IL6_50_M2` | -46% | IL6_50_M2_fernando2014 |
| `IL6_50_MDSC` | -46% | IL6_50_MDSC |
| `IL6_50_iCAF` | -42% |  |
| `k_APC_mature_ID` | -42% | k_APC_mature_ID_dc_maturation_kinetics |
| `n_fib` | -40% | E_50_fib_YAP_chang2025 |
| `TGFb_50_IL1R1` | -30% | caf_interconversion_biffi2019 |
| `k_APC_mig` | -26% | k_APC_mig |
| `n_CD4_clones` | -19% | n_CD4_clones |
| `EC50_GMCSF` | -19% | EC50_GMCSF_guthridge2006 |
| `k_Treg_death` | -15% | k_Treg_death, k_nCD8_death |
| `E_50_Texh` | -15% | E_50_Texh_IFNg_saitakis2017, E_50_Texh_TNFa_saitakis2017 |
| `k_Th_to_Treg` | -10% | k_Th_to_Treg |
| `q_Treg_T_in` | -9% | q_Treg_T_in |
| `TGFb_50_Teff` | -6% | TGFb_50_Teff |
| `k_CAF_encounter` | -6% | k_CAF_encounter |
| `f_stroma_max` | -3% | f_stroma_max_ogawa2021_PDAC_deriv001 |
| `rho_Th_max` | -3% | rho_Th_max_ihc_karakhanova2015 |
| `k_CD8_exh_death` | -3% | k_CD8_exh_death |
| `ArgI_50_Treg` | -3% | ArgI_50_Treg |
| `k_cell_clear` | -3% | k_cell_clear |
| `TGFb_50_col_sec` | -2% | TGFb_50_col_sec |
| `IL12_50` | -2% | IL12_50 |
| `k_stroma_support` | -1% | k_stroma_support, k_stroma_support_hwang2008_PDAC_deriv001 |
| `K_C_exh` | -1% | K_C_exh |
| `n_T_expansion_burst` | 1% | n_T_expansion_burst_Butz1998 |
| `k_MMP_M2` | 1% | k_MMP_M2 |
| `IL1_50` | 1% | IL1_50_CXCL8_ciminieri2023, IL1_50_IL6_ciminieri2023, IL1_50, caf_interconversion_biffi2019 |
| `k_CD8_basal_exh` | 3% | k_CD8_basal_exh |
| `k_iCAF_to_myCAF` | 4% | caf_interconversion_biffi2019, k_myCAF_to_iCAF |
| `E_50_fib` | 5% | E_50_fib_YAP_chang2025 |
| `k_CXCL12_deg` | 5% | k_CXCL12_sec_cancer, k_CXCL12_sec_iCAF_PDAC_deriv001, k_CXCL12_sec_iCAF_PDAC_deriv002 |
| `q_CD8_T_in` | 7% | q_CD8_T_in_Boissonnas2007_OT1_EG7_timecourse |
| `IL2_50_Treg` | 8% | k_IL2_cons |
| `k_apCAF_prolif` | 9% | k_apCAF_prolif |

> Joint MCMC had **24 divergences** — may indicate prior-model conflict.

### Posterior predictive coverage

**217/265** observables covered by 95% CI (82%)

![PPC Histograms](parameter_audit_report_files/ppc_coverage.png)

### Contraction summary

| Parameter | Prior Sigma | Post Sigma | Median | Contraction | n_eff | r_hat |
|-----------|:----------:|:---------:|-------:|:-----------:|------:|------:|
| `k_PSC_to_apCAF` | 1.00 | 0.02 | 0.05123 | **100%** | --- | --- |
| `k_PSC_to_myCAF` | 1.00 | 0.14 | 0.1118 | **98%** | --- | --- |
| `k_myCAF_prolif` | 1.00 | 0.21 | 0.7714 | **95%** | 4243 | 1.000 |
| `k_CCL5_sec_iCAF` | 1.20 | 0.27 | 5.579e-10 | **95%** | 4865 | 1.000 |
| `n_CD8_clones` | 1.00 | 0.25 | 14.22 | **94%** | 2771 | 1.001 |
| `N_IL2_CD8` | 0.50 | 0.16 | 5.98 | **90%** | 2657 | 1.000 |
| `APC0_cDC2_T` | 0.70 | 0.24 | 5.539e+05 | **88%** | 1623 | 1.000 |
| `k_APC_death` | 0.70 | 0.25 | 0.1913 | **87%** | --- | --- |
| `k_vas_Csec` | 0.70 | 0.28 | 0.002171 | **84%** | 1532 | 1.000 |
| `k_col_deg` | 1.20 | 0.50 | 0.004225 | **83%** | 6929 | 1.000 |
| `phi_col_ref` | 0.50 | 0.22 | 0.04621 | **80%** | 11931 | 1.000 |
| `n_pore` | 0.50 | 0.22 | 3.476 | **80%** | 3471 | 1.000 |
| `k_stiff_CD8_exh` | 1.00 | 0.46 | 0.0526 | **79%** | 2844 | 1.000 |
| `k_CAF_const` | 1.00 | 0.46 | 2.712e+05 | **79%** | 4599 | 1.000 |
| `E_ref` | 0.50 | 0.23 | 0.7018 | **79%** | 12683 | 1.000 |
| `k_CXCL12_sec_iCAF` | 1.20 | 0.56 | 2.552e-10 | **78%** | 9460 | 1.000 |
| `d_crit_T` | 0.50 | 0.23 | 4.644 | **78%** | 2649 | 1.000 |
| `k_CXCL12_sec_cancer` | 1.20 | 0.57 | 8.577e-11 | **78%** | 8781 | 1.000 |
| `k_vas_myCAF_sec` | 1.00 | 0.49 | 0.006392 | **76%** | 4955 | 1.000 |
| `k_C_CD8_exh` | 1.00 | 0.50 | 0.02779 | **75%** | 1277 | 1.001 |
| `CXCL12_50_Texcl` | 1.20 | 0.64 | 6.183 | **71%** | 1652 | 1.002 |
| `k_nTreg_death` | 1.20 | 0.67 | 0.001127 | **69%** | 3821 | 1.000 |
| `k_nCD8_death` | 1.20 | 0.67 | 0.001045 | **69%** | 3709 | 1.000 |
| `k_IL2_cons` | 1.50 | 0.84 | 2.999e-11 | **68%** | 4390 | 1.000 |
| `k_IL1_sec` | 1.00 | 0.56 | 4.396e-11 | **68%** | 3755 | 1.000 |
| `k_PSC_to_iCAF` | 1.00 | 0.57 | 0.02388 | **68%** | --- | --- |
| `k_iCAF_prolif` | 1.00 | 0.58 | 0.2384 | **66%** | 6238 | 1.000 |
| `rho_Treg_max` | 1.00 | 0.59 | 2.645e+07 | **65%** | 1477 | 1.000 |
| `k_IFNg_Tsec` | 0.70 | 0.42 | 4.527e-12 | **65%** | 2738 | 1.001 |
| `k_T_activation_per_APC` | 1.50 | 0.94 | 0.0001241 | **61%** | 4498 | 1.000 |
| `IL2_50` | 0.50 | 0.31 | 0.308 | **60%** | 2584 | 1.000 |
| `k_M1_phago` | 0.70 | 0.44 | 0.4691 | **60%** | 8165 | 1.000 |
| `Ve_T` | 0.50 | 0.32 | 0.3713 | **60%** | 2674 | 1.000 |
| `k_M2_pol` | 1.00 | 0.64 | 0.2333 | **59%** | 7565 | 1.000 |
| `k_C1_growth` | 0.70 | 0.45 | 0.008104 | **58%** | 2687 | 1.001 |
| `d_pore_ref` | 0.50 | 0.32 | 5.595 | **58%** | 11643 | 1.000 |
| `k_IL10_sec` | 1.00 | 0.65 | 1.158e-11 | **58%** | 2469 | 1.000 |
| `k_col_myCAF_sec` | 1.00 | 0.65 | 1.187e-08 | **57%** | 4821 | 1.001 |
| `phi_col_50_Kmax` | 0.70 | 0.46 | 1.578 | **57%** | 1100 | 1.000 |
| `rho_collagen` | 0.70 | 0.46 | 0.1027 | **57%** | 9406 | 1.000 |
| `k_TGFb_Tsec` | 0.70 | 0.46 | 1.168e-11 | **56%** | 1501 | 1.000 |
| `k_IL2_sec` | 1.00 | 0.66 | 2.283e-11 | **56%** | 2643 | 1.000 |
| `k_IL12_sec` | 0.70 | 0.49 | 7.439e-11 | **52%** | 1504 | 1.000 |
| `n_stiff` | 0.50 | 0.35 | 0.7152 | **51%** | 10685 | 1.000 |
| `CD28_CD8X_50` | 1.00 | 0.70 | 160.4 | **51%** | 3519 | 1.000 |
| `k_Th_pro_tumor` | 0.80 | 0.56 | 0.1744 | **50%** | 1995 | 1.000 |
| `APC0_cDC1_T` | 0.70 | 0.50 | 5.052e+05 | 49% | 1490 | 1.000 |
| `p1_50` | 1.00 | 0.72 | 0.0008667 | 48% | 4462 | 1.001 |
| `k_Mac_death` | 0.70 | 0.52 | 0.007742 | 45% | 8865 | 1.000 |
| `IL10_50` | 0.70 | 0.52 | 7.218 | 44% | 2585 | 1.000 |
| `VEGF_50` | 0.70 | 0.52 | 2032 | 44% | 1459 | 1.001 |
| `k_APC_mat_cDC1` | 0.70 | 0.52 | 3.255 | 44% | 1509 | 1.000 |
| `f_col_basal` | 0.50 | 0.38 | 0.5943 | 43% | 5436 | 1.000 |
| `k_Mac_rec` | 0.70 | 0.53 | 9.238e+05 | 43% | 9129 | 1.000 |
| `k_Treg_pro_tumor` | 0.70 | 0.54 | 0.1342 | 41% | 2251 | 1.000 |
| `k_CCL2_sec` | 0.70 | 0.54 | 1.287e-10 | 40% | 1509 | 1.000 |
| `k_IL2_deg` | 0.77 | 0.60 | 0.0922 | 39% | 2760 | 1.001 |
| `k_C1_death` | 0.70 | 0.55 | 0.0052 | 38% | 4878 | 1.000 |
| `k_CD8_pro` | 0.50 | 0.40 | 1.137 | 37% | 5617 | 1.001 |
| `k_IL6_sec_iCAF` | 0.70 | 0.56 | 9.325e-11 | 36% | 6499 | 1.000 |
| `K_M1_phago` | 1.00 | 0.82 | 1.58 | 33% | 7971 | 1.000 |
| `k_vas_growth` | 1.00 | 0.82 | 0.1347 | 33% | 1289 | 1.003 |
| `f_IL6_iCAF` | 0.70 | 0.57 | 1.597 | 33% | 2772 | 1.002 |
| `k_mAPC_death` | 0.50 | 0.42 | 1 | 30% | 4910 | 1.001 |
| `k_IL6_deg` | 0.50 | 0.43 | 5.036 | 26% | 7868 | 1.000 |
| `k_TGFb_sec_myCAF` | 1.00 | 0.87 | 2.057e-08 | 24% | 2576 | 1.001 |
| `k_MDSC_rec` | 0.70 | 0.61 | 9.865e+04 | 23% | 10274 | 1.000 |
| `k_MDSC_death` | 0.70 | 0.62 | 0.02469 | 21% | 8561 | 1.000 |
| `k_M1_pol` | 1.00 | 0.91 | 0.03409 | ~18%~ | 6800 | 1.001 |
| `k_IL1_deg` | 0.50 | 0.46 | 2.679 | ~16%~ | 3755 | 1.000 |
| `k_Treg_CD8_exh` | 1.00 | 0.94 | 0.08717 | ~12%~ | 2278 | 1.000 |
| `k_myCAF_to_iCAF` | 0.70 | 0.66 | 0.04931 | ~11%~ | 4739 | 1.000 |
| `k_vas_decay` | 1.00 | 0.95 | 0.003947 | ~10%~ | 2163 | 1.000 |
| `k_apCAF_prolif` | 1.00 | 0.95 | 0.04724 | ~9%~ | --- | --- |
| `IL2_50_Treg` | 0.50 | 0.48 | 0.3415 | ~8%~ | 4974 | 1.000 |
| `q_CD8_T_in` | 0.70 | 0.68 | 0.0003946 | ~7%~ | 6140 | 1.000 |
| `k_CXCL12_deg` | 0.50 | 0.49 | 1.705 | ~5%~ | 11667 | 1.000 |
| `E_50_fib` | 0.70 | 0.68 | 5.736 | ~5%~ | 5833 | 1.000 |
| `k_iCAF_to_myCAF` | 1.00 | 0.98 | 0.09895 | ~4%~ | 5200 | 1.000 |
| `k_CD8_basal_exh` | 1.00 | 0.98 | 0.05061 | ~3%~ | 4895 | 1.000 |
| `IL1_50` | 1.00 | 0.99 | 0.01711 | ~1%~ | --- | --- |
| `k_MMP_M2` | 1.20 | 1.20 | 1.078e-08 | ~1%~ | 6008 | 1.000 |
| `n_T_expansion_burst` | 0.70 | 0.70 | 799.3 | ~1%~ | 6111 | 1.000 |
| `K_C_exh` | 1.00 | 1.00 | 9.713e+07 | _-1%_ | 2533 | 1.000 |
| `k_stroma_support` | 0.70 | 0.70 | 1.049 | _-1%_ | 5947 | 1.000 |
| `IL12_50` | 0.70 | 0.71 | 14.76 | _-2%_ | 3517 | 1.000 |
| `TGFb_50_col_sec` | 0.70 | 0.71 | 0.07258 | _-2%_ | 5384 | 1.000 |
| `k_cell_clear` | 1.00 | 1.01 | 0.1034 | _-3%_ | 1337 | 1.000 |
| `ArgI_50_Treg` | 1.00 | 1.02 | 0.3537 | _-3%_ | 2872 | 1.000 |
| `k_CD8_exh_death` | 1.00 | 1.02 | 0.168 | _-3%_ | 4624 | 1.001 |
| `rho_Th_max` | 0.70 | 0.71 | 8.851e+06 | _-3%_ | 2463 | 1.000 |
| `f_stroma_max` | 0.70 | 0.71 | 1.403 | _-3%_ | 5642 | 1.000 |
| `k_CAF_encounter` | 1.00 | 1.03 | 1.018e+06 | _-6%_ | 3237 | 1.000 |
| `TGFb_50_Teff` | 0.70 | 0.72 | 0.06032 | _-6%_ | 5638 | 1.000 |
| `q_Treg_T_in` | 0.70 | 0.73 | 0.0001131 | _-9%_ | 2816 | 1.000 |
| `k_Th_to_Treg` | 0.70 | 0.73 | 0.02568 | _-10%_ | 4455 | 1.001 |
| `E_50_Texh` | 0.70 | 0.75 | 3.334 | _-15%_ | 6586 | 1.000 |
| `k_Treg_death` | 0.50 | 0.54 | 0.05778 | _-15%_ | 4621 | 1.000 |
| `EC50_GMCSF` | 0.70 | 0.76 | 0.00299 | _-19%_ | 2648 | 1.000 |
| `n_CD4_clones` | 1.00 | 1.09 | 168.1 | _-19%_ | 1204 | 1.002 |
| `k_APC_mig` | 0.70 | 0.79 | 1.311 | _-26%_ | 4906 | 1.000 |
| `TGFb_50_IL1R1` | 0.70 | 0.80 | 0.08345 | _-30%_ | --- | --- |
| `n_fib` | 0.50 | 0.59 | 0.4787 | _-40%_ | 4318 | 1.001 |
| `k_APC_mature_ID` | 0.70 | 0.83 | 2.808 | _-42%_ | 2289 | 1.001 |
| `IL6_50_iCAF` | 0.70 | 0.84 | 0.1192 | _-42%_ | 8024 | 1.000 |
| `IL6_50_MDSC` | 0.70 | 0.85 | 0.1225 | _-46%_ | 7663 | 1.000 |
| `IL6_50_M2` | 0.70 | 0.85 | 0.1237 | _-46%_ | 8080 | 1.000 |
| `k_CD8_death` | 0.50 | 0.63 | 0.1713 | _-60%_ | 6654 | 1.000 |
| `TGFb_50_APC` | 0.70 | 0.91 | 0.02381 | _-69%_ | 6728 | 1.000 |
| `q_nCD8_P_out` | 0.50 | 0.66 | 10.78 | _-75%_ | 751 | 1.000 |
| `n_Texh` | 0.50 | 0.67 | 0.4933 | _-81%_ | 5489 | 1.000 |
| `TGFb_50` | 0.70 | 1.08 | 0.04259 | _-137%_ | 6891 | 1.000 |
| `n_IL1` | 0.50 | 0.79 | 0.3135 | _-147%_ | 1928 | 1.000 |
| `k_myCAF_death` | 0.70 | 1.11 | 0.03246 | _-152%_ | 6363 | 1.000 |
| `k_apCAF_death` | 0.70 | 1.15 | 0.02524 | _-170%_ | 6567 | 1.000 |
| `q_nCD8_LN_in` | 0.50 | 0.83 | 0.06868 | _-175%_ | 1447 | 1.000 |
| `k_iCAF_death` | 0.70 | 1.16 | 0.02543 | _-176%_ | 7384 | 1.000 |
| `k_qpsc_death` | 0.50 | 1.02 | 0.01542 | _-313%_ | 5501 | 1.000 |
| `TGFb_50_CAF_act` | 0.70 | 1.48 | 0.002345 | _-348%_ | --- | --- |

### Prior vs posterior marginals

![Prior vs posterior marginals](parameter_audit_report_files/prior_vs_posterior_marginals.png)

---

## Component diagnostics

**83 components** (79 NUTS, 4 NPE) — **24 total divergences** — **217/265** PPC observables covered

### Components with issues

**NUTS components:**

| Params | Divergences | PPC | Parameters |
|:------:|:-----------:|:---:|-----------|
| 3 | **16** | 3/3 | `E_50_Texh`, `k_stiff_CD8_exh`, `n_Texh` |
| 5 | **5** | 12/12 | `K_well`, `k_iCAF_prolif`, `k_myCAF_prolif`, `k_qPSC_prolif`, `n_logistic` |
| 4 | **2** | 4/6 | `E_50_Texh`, `IFNg_max`, `TNFa_max`, `n_Texh` |
| 13 | **1** | 6/8 | `Emax_IL6_MDSC`, `IL6_50_M2`, `IL6_50_M2__z`, `IL6_50_MDSC`, `IL6_50_MDSC__z`, `IL6_50_iCAF`, +7 more |
| 2 | 0 | 5/6 | `k_CCL5_deg_invitro`, `k_CCL5_sec_iCAF` |
| 6 | 0 | 0/2 | `H_CCL2_MDSC_eff`, `H_IL6_MDSC_eff`, `k_MDSC_death`, `k_MDSC_rec`, `r_LLC_B16`, `total_cell_density` |
| 2 | 0 | 2/3 | `d_crit_T`, `n_pore` |
| 1 | 0 | 0/1 | `k_IL12_sec` |
| 4 | 0 | 1/2 | `k_IL6_deg`, `k_IL6_sec_basal`, `k_IL6_sec_iCAF`, `v_media_mL` |
| 1 | 0 | 0/1 | `k_TGFb_Tsec` |
| 10 | 0 | 0/1 | `CAF_death_rates__base`, `CAF_death_rates__tau`, `k_apCAF_death`, `k_apCAF_death__z`, `k_iCAF_death`, `k_iCAF_death__z`, +4 more |
| 3 | 0 | 2/3 | `IL1_50`, `max_iCAF_induction`, `n_IL1_iCAF` |
| 1 | 0 | 0/2 | `K_C_exh` |
| 6 | 0 | 2/4 | `k_Treg_death`, `k_nCD8_death`, `k_nCD8_death__z`, `k_nTreg_death`, `k_nTreg_death__z`, `naive_T_death_rates__base` |
| 10 | 0 | 4/6 | `TGFb_50`, `TGFb_50_APC`, `TGFb_50_APC__z`, `TGFb_50_Teff`, `TGFb_50_Teff__z`, `TGFb_50__z`, +4 more |
| 3 | 0 | 3/4 | `k_CAF_const`, `k_CAF_encounter`, `psc_frac_scale` |
| 2 | 0 | 0/3 | `F_inf`, `k_Th_to_Treg` |
| 3 | 0 | 1/3 | `f_M2_eff`, `k_MMP_M2`, `k_col_deg` |
| 1 | 0 | 0/1 | `k_cell_clear` |
| 4 | 0 | 4/6 | `CXCL8_fold_max`, `IL1_50`, `IL6_fold_max`, `n_IL1` |
| 9 | 0 | 4/5 | `CXCL12_secretion_rates__base`, `k_CXCL12_deg`, `k_CXCL12_sec_cancer`, `k_CXCL12_sec_cancer__z`, `k_CXCL12_sec_iCAF`, `k_CXCL12_sec_iCAF__z`, +3 more |
| 6 | 0 | 9/11 | `K_M1_phago`, `alpha_Hepa`, `alpha_Huh7`, `beta_pdac_phago`, `k_M1_phago`, `logit_H_Mac_C` |
| 2 | 0 | 0/1 | `A_Tcell_target_contact`, `p1_50` |
| 3 | 0 | 2/3 | `E_50_fib`, `YAP_max`, `n_fib` |
| 2 | 0 | 2/3 | `k_iCAF_to_myCAF`, `k_myCAF_to_iCAF` |
| 3 | 0 | 0/1 | `caf_density_per_mm2`, `cancer_density_per_mm2`, `f_stroma_max` |
| 3 | 0 | 2/4 | `alpha_endpoint_mass`, `k_stroma_support`, `rho_C1_baseline_density` |
| 6 | 0 | 5/8 | `E_ref`, `d_pore_ref`, `n_stiff`, `phi_col_ref`, `protein_per_wet_volume_pancreas`, `rho_collagen` |

<details><summary><b>16 divergences</b> — `E_50_Texh`, `k_stiff_CD8_exh`, `n_Texh`</summary>

| Parameter | n_eff | r_hat | Contraction | z_score |
|-----------|------:|------:|:-----------:|--------:|
| `E_50_Texh` | 3051 | 1.0002 | 23% | 0.05 |
| `k_stiff_CD8_exh` | 2844 | 1.0003 | 55% | 0.61 |
| `n_Texh` | 2645 | 1.0021 | 14% | 0.00 |

</details>

<details><summary><b>5 divergences</b> — `K_well`, `k_iCAF_prolif`, `k_myCAF_prolif`, `k_qPSC_prolif`, ...</summary>

| Parameter | n_eff | r_hat | Contraction | z_score |
|-----------|------:|------:|:-----------:|--------:|
| `K_well` | 4398 | 1.0001 | -11% | 0.01 |
| `k_iCAF_prolif` | 6238 | 0.9999 | 74% | 0.37 |
| `k_myCAF_prolif` | 4243 | 1.0003 | 68% | 1.69 |
| `k_qPSC_prolif` | 7363 | 0.9999 | 92% | 0.34 |
| `n_logistic` | 6073 | 0.9998 | 2% | 0.01 |

</details>

<details><summary><b>2 divergences</b> — `E_50_Texh`, `IFNg_max`, `TNFa_max`, `n_Texh`</summary>

| Parameter | n_eff | r_hat | Contraction | z_score |
|-----------|------:|------:|:-----------:|--------:|
| `E_50_Texh` | 6586 | 1.0004 | 24% | 0.20 |
| `IFNg_max` | 8107 | 0.9999 | 99% | 1.00 |
| `TNFa_max` | 7589 | 0.9999 | 91% | 0.69 |
| `n_Texh` | 5489 | 0.9999 | 70% | 0.61 |

</details>

<details><summary><b>1 divergences</b> — `Emax_IL6_MDSC`, `IL6_50_M2`, `IL6_50_M2__z`, `IL6_50_MDSC`, ...</summary>

| Parameter | n_eff | r_hat | Contraction | z_score |
|-----------|------:|------:|:-----------:|--------:|
| `Emax_IL6_MDSC` | 9783 | 1.0000 | 37% | 0.11 |
| `IL6_50_M2` | 8080 | 0.9999 | -18% | 0.37 |
| `IL6_50_M2__z` | --- | --- | --- | --- |
| `IL6_50_MDSC` | 7663 | 1.0003 | -889% | 1.57 |
| `IL6_50_MDSC__z` | --- | --- | --- | --- |
| `IL6_50_iCAF` | 8024 | 1.0003 | -183% | 0.41 |
| `IL6_50_iCAF__z` | --- | --- | --- | --- |
| `IL6_EC50s__base` | --- | --- | --- | --- |
| `IL6_EC50s__tau` | --- | --- | --- | --- |
| `baseline_arginase` | 9670 | 1.0000 | 38% | 0.11 |
| `k_MDSC_gen_baseline` | 11656 | 0.9999 | 59% | 0.07 |
| `max_arginase` | 11395 | 0.9998 | 87% | 0.82 |
| `n_IL6_M2` | 8878 | 1.0000 | 11% | 0.13 |

</details>

**NPE components:**

**1 params** (`k_APC_death`) — PPC 11/12

**5 params** (`TGFb_50_CAF_act`, `k_PSC_to_apCAF`, `k_PSC_to_iCAF`, `k_PSC_to_myCAF`, `k_myCAF_prolif`) — PPC 19/26

SBC failures:

| Parameter | Issue | KS p-value |
|-----------|-------|:----------:|
| `k_PSC_to_apCAF` | underconfident | 5.21e-20 |
| `k_PSC_to_iCAF` | miscalibrated | 4.30e-03 |
| `k_myCAF_prolif` | biased_low | 2.42e-21 |

**7 params** (`IL1_50`, `TGFb_50_CAF_act`, `TGFb_50_IL1R1`, `k_iCAF_prolif`, `k_iCAF_to_myCAF`, `k_myCAF_prolif`, +1 more) — PPC 11/11

SBC failures:

| Parameter | Issue | KS p-value |
|-----------|-------|:----------:|
| `k_iCAF_prolif` | miscalibrated | 2.20e-09 |
| `k_iCAF_to_myCAF` | miscalibrated | 3.23e-06 |
| `k_myCAF_prolif` | miscalibrated | 3.67e-05 |

### Healthy components (52)

52 components with 0 divergences, full PPC coverage, and passing SBC.

---

## What's left before stage 2?

**115/188** parameters have submodel targets (61%).

Of **113 PRCC-significant** parameters, **92** have targets (81%).

### Covered (115)

| Parameter | Type | Targets | Group | PRCC |
|-----------|------|:-------:|-------|-----:|
| `APC0_cDC1_T` | rate | 1 |  | #24 |
| `APC0_cDC2_T` | rate | 1 |  | #51 |
| `ArgI_50_Treg` | EC50 | 1 |  | #86 |
| `CD28_CD8X_50` | EC50 | 1 |  | #101 |
| `CXCL12_50_Texcl` | EC50 | 1 |  | #16 |
| `EC50_GMCSF` | EC50 | 1 |  | #214 |
| `E_50_Texh` | EC50 | 2 |  | #122 |
| `E_50_fib` | EC50 | 1 |  | #153 |
| `E_ref` | biophysical | 2 |  | #170 |
| `IL10_50` | EC50 | 1 |  | #57 |
| `IL12_50` | EC50 | 1 |  | #108 |
| `IL1_50` | EC50 | 4 |  | #42 |
| `IL2_50` | EC50 | 1 |  | #160 |
| `IL2_50_Treg` | EC50 | 1 |  | #164 |
| `IL6_50_M2` | EC50 | 1 | IL6_EC50s | #147 |
| `IL6_50_MDSC` | EC50 | 1 | IL6_EC50s | #115 |
| `K_C_exh` | rate | 1 |  | #106 |
| `K_M1_phago` | rate | 1 |  | #98 |
| `N_IL2_CD8` | count | 1 |  | #22 |
| `TGFb_50` | EC50 | 1 | TGFb_EC50_immune | #90 |
| `TGFb_50_APC` | EC50 | 1 | TGFb_EC50_immune | #41 |
| `TGFb_50_CAF_act` | EC50 | 4 |  | #81 |
| `TGFb_50_IL1R1` | EC50 | 1 |  | #72 |
| `TGFb_50_Teff` | EC50 | 1 | TGFb_EC50_immune | #204 |
| `TGFb_50_col_sec` | EC50 | 1 |  | #178 |
| `VEGF_50` | EC50 | 1 |  | #102 |
| `Ve_T` | rate | 1 |  |  |
| `d_crit_T` | biophysical | 1 |  | #167 |
| `d_pore_ref` | biophysical | 1 |  | #19 |
| `f_IL6_iCAF` | fraction | 1 |  | #52 |
| `f_col_basal` | fraction | 2 |  | #110 |
| `f_stroma_max` | fraction | 1 |  | #1 |
| `k_APC_death` | death | 1 |  | #45 |
| `k_APC_mat_cDC1` | rate | 1 |  | #54 |
| `k_APC_mature_ID` | rate | 1 |  | #205 |
| `k_APC_mig` | rate | 1 |  | #58 |
| `k_C1_death` | death | 1 |  | #36 |
| `k_C1_growth` | rate | 2 |  | #3 |
| `k_CAF_const` | rate | 1 |  | #82 |
| `k_CAF_encounter` | rate | 1 |  | #78 |
| `k_CCL2_sec` | secretion | 1 |  | #114 |
| `k_CCL5_sec_iCAF` | secretion | 1 |  | #179 |
| `k_CD8_basal_exh` | exhaustion | 1 |  | #61 |
| `k_CD8_death` | death | 2 |  | #31 |
| `k_CD8_exh_death` | death | 1 |  | #53 |
| `k_CD8_pro` | proliferation | 1 |  | #68 |
| `k_CXCL12_deg` | degradation | 3 |  | #65 |
| `k_CXCL12_sec_cancer` | secretion | 1 | CXCL12_secretion_rates | #44 |
| `k_CXCL12_sec_iCAF` | secretion | 2 | CXCL12_secretion_rates | #60 |
| `k_C_CD8_exh` | exhaustion | 1 |  | #66 |
| `k_IFNg_Tsec` | secretion | 1 |  | #105 |
| `k_IL10_sec` | secretion | 1 |  | #64 |
| `k_IL12_sec` | secretion | 1 |  | #113 |
| `k_IL1_deg` | degradation | 1 |  | #75 |
| `k_IL1_sec` | secretion | 1 |  | #39 |
| `k_IL2_cons` | rate | 1 |  | #35 |
| `k_IL2_deg` | degradation | 1 |  | #206 |
| `k_IL2_sec` | secretion | 1 |  | #30 |
| `k_IL6_deg` | degradation | 1 |  | #97 |
| `k_IL6_sec_iCAF` | secretion | 1 |  | #89 |
| `k_M1_phago` | killing | 1 |  | #166 |
| `k_M1_pol` | polarization | 2 |  | #46 |
| `k_M2_pol` | polarization | 3 |  | #20 |
| `k_MDSC_death` | death | 1 |  | #56 |
| `k_MDSC_rec` | trafficking | 1 |  | #55 |
| `k_MMP_M2` | rate | 1 |  |  |
| `k_Mac_death` | death | 3 |  | #14 |
| `k_Mac_rec` | trafficking | 2 |  | #12 |
| `k_PSC_to_apCAF` | rate | 1 |  | #63 |
| `k_PSC_to_iCAF` | rate | 1 |  | #104 |
| `k_PSC_to_myCAF` | rate | 4 |  | #85 |
| `k_TGFb_Tsec` | secretion | 1 |  | #87 |
| `k_TGFb_sec_myCAF` | secretion | 1 |  | #18 |
| `k_T_activation_per_APC` | activation | 1 |  | #26 |
| `k_Th_pro_tumor` | proliferation | 1 |  | #29 |
| `k_Th_to_Treg` | rate | 1 |  | #88 |
| `k_Treg_CD8_exh` | exhaustion | 1 |  | #93 |
| `k_Treg_death` | death | 2 |  | #2 |
| `k_Treg_pro_tumor` | proliferation | 1 |  | #7 |
| `k_apCAF_prolif` | proliferation | 1 |  | #17 |
| `k_cell_clear` | degradation | 1 |  | #9 |
| `k_col_deg` | degradation | 2 |  | #73 |
| `k_col_myCAF_sec` | secretion | 1 |  | #4 |
| `k_iCAF_prolif` | proliferation | 2 |  | #43 |
| `k_iCAF_to_myCAF` | rate | 2 |  | #23 |
| `k_mAPC_death` | death | 2 |  | #34 |
| `k_myCAF_death` | death | 1 | CAF_death_rates | #59 |
| `k_myCAF_prolif` | proliferation | 5 |  | #5 |
| `k_myCAF_to_iCAF` | rate | 2 |  | #37 |
| `k_nCD8_death` | death | 1 | naive_T_death_rates | #28 |
| `k_nTreg_death` | death | 1 | naive_T_death_rates | #83 |
| `k_stiff_CD8_exh` | exhaustion | 1 |  | #103 |
| `k_stroma_support` | rate | 2 |  | #33 |
| `k_vas_Csec` | secretion | 1 |  | #62 |
| `k_vas_decay` | rate | 1 |  | #49 |
| `k_vas_growth` | rate | 1 |  | #21 |
| `k_vas_myCAF_sec` | secretion | 1 |  | #155 |
| `n_CD4_clones` | count | 1 |  | #69 |
| `n_CD8_clones` | count | 1 |  | #32 |
| `n_IL1` | count | 2 |  | #40 |
| `n_T_expansion_burst` | count | 1 |  | #27 |
| `n_Texh` | count | 2 |  | #187 |
| `n_fib` | count | 1 |  | #177 |
| `n_pore` | count | 1 |  | #79 |
| `n_stiff` | count | 1 |  | #157 |
| `p1_50` | EC50 | 1 |  |  |
| `phi_col_50_Kmax` | EC50 | 1 |  | #96 |
| `phi_col_ref` | biophysical | 4 |  | #48 |
| `q_CD8_T_in` | trafficking | 1 |  | #6 |
| `q_Treg_T_in` | trafficking | 1 |  | #74 |
| `q_nCD8_LN_in` | trafficking | 1 |  | #8 |
| `q_nCD8_P_out` | trafficking | 1 |  | #10 |
| `rho_Th_max` | fraction | 1 |  | #84 |
| `rho_Treg_max` | fraction | 1 |  | #11 |
| `rho_collagen` | fraction | 1 |  | #67 |

### Stage 2 runs available

- **sbi_baseline_allparams** — 194 parameters
- **sbi_baseline_no_treatment** — 188 parameters
- **sbi_baseline_no_treatment_tier1** — 30 parameters
- **sbi_baseline_sa97** — 97 parameters
- **sbi_baseline_tier1** — 30 parameters
- **sbi_gvax_neoadjuvant_zheng2022_tier1** — 30 parameters
- **sbi_gvax_nivo_neoadjuvant_zheng2022_tier1** — 29 parameters
- **sbi_multi** — 188 parameters
- **sbi_multi_128x7_s2025** — 217 parameters
- **sbi_multi_128x7_s2029** — 217 parameters
- **sbi_multi_192x9_s2025** — 217 parameters
- **sbi_multi_256x11_s2025** — 217 parameters
- **sbi_multi_64x5** — 217 parameters
- **sbi_multi_tier1** — 29 parameters
- **sbi_multi_tier2** — 69 parameters
- **sbi_multi_tier3** — 116 parameters
- **sbi_multi_tier3_smallnn** — 116 parameters

---

## Appendix: Cross-model parameter comparison

Values used for the same parameter across QSPIO cancer models (TNBC, CRC, UM, NSCLC, HCC, PDAC). Parameters where our PDAC value differs substantially from other models may warrant validation.

| Parameter | Ref name | TNBC | CRC | UM | NSCLC | HCC | PDAC (ref) | PDAC (current) | Unit |
|-----------|----------|------|-----|-----|-------|-----|------------|:--------------:|------|
| `E_50_Texh` | `ECM_50_T_exh` | --- | --- | --- | --- | 1000000 | --- | **3.334** | nanomolarity |
| `N_IL2_CD4` |  | 7 - 10 | 8 | 8 | 5 - 11 | 8 | 7 - 10 | 8.5 | dimensionless |
| `N_IL2_CD8` |  | 10 - 12 | 11 | 11 | 9 - 13 | 8 | 10 - 12 | **5.98** | dimensionless |
| `TGFb_baseline` | `TGFbase` | 0.016 | 0.02 | 0.016 | 0.016 | 0.016 | 0.016 | 0.016 | nanomolarity |
| `Ve_T` |  | 0.37 | 0.37 | 0.37 | 0.75 | 1 | 0.017-0.529 | **0.3713** | dimensionless |
| `initial_tumour_diameter` |  | 2.5 | 2 | 1.1 | 2.64 | 5 | 1.43-3.03 9 | 3 | centimeter |
| `k_ArgI_sec` |  | 0.014 | 0.014 | 0.014 | 0.014 | 0.014 | 0.014 | 1.1e-10 | (mU*microliter)/cell/day |
| `k_C1_growth` |  | 0.0075 (Gompertzia | 0.005 (Logistic) | 0.002 (Logistic) | 0.013 | 0.012 | 0.0066 3 | **0.008104** | 1/day |
| `k_CAF_const` | `k_fib_const` | --- | --- | --- | --- | 2000000 | --- | **2.712e+05** | cell/milliliter |
| `k_CAF_encounter` | `k_fib_rec` | --- | --- | --- | --- | 1000000 | --- | **1.018e+06** | cell/(milliliter*day) |
| `k_CCL2_sec` |  | 1.7e-12 | 1.5e-12 | 1.7e-12 | 1.7e-12 | 1.7e-12 | 1.7e-12 | **1.287e-10** | nanomole/cell/day |
| `k_IFNg_Tsec` |  | 4e-13 | NA | NA | 4e-13 | 4e-13 | 4e-13 | **4.527e-12** | nanomole/cell/day |
| `k_IL10_sec` |  | 3e-12 | NA | 3e-12 | 3e-12 | 3e-12 | 3e-12 | **1.158e-11** | nanomole/cell/day |
| `k_IL12_Msec` |  | 5e-13 | NA | 5e-13 | 5e-13 | 5e-13 | 5e-13 | 5e-13 | nanomole/cell/day |
| `k_IL12_sec` |  | 8.5e-12 | NA | 8.5e-12 | 8.5e-12 | 8.5e-12 | 8.5e-12 | **7.439e-11** | nanomole/cell/day |
| `k_M1_pol` |  | 0.045 | NA | 0.001 | 0.02 | 0.02 | 0.05 18 | **0.03409** | 1/day |
| `k_MDSC_rec` | `k_MDSC_mig` | 11000 | 1.2 (1/day) | 11000 | 2000 | 25000 | 21000 11 | **9.865e+04** | cell/(milliliter*day) |
| `k_Mac_rec` | `k_Mac_mig` | 170000 | NA | 170000 | 200000 | 170000 | 175000 17 | **9.238e+05** | cell/(milliliter*day) |
| `k_NO_sec` |  | 4.8e-07 | 4.8e-07 | 4.8e-07 | 4.8e-07 | 4.8e-07 | 4.8e-07 | 4.8e-07 | nanomole/cell/day |
| `k_TGFb_Msec` |  | 2e-11 | NA | 2e-11 | 2e-11 | 2e-11 | 2e-11 | 2e-11 | nanomole/cell/day |
| `k_TGFb_Tsec` |  | 1.2e-10 | 1.2e-10 | 1.2e-10 | 1.2e-10 | 1.2e-10 | 1.2e-10 | **1.168e-11** | nanomole/cell/day |
| `k_Th_to_Treg` | `k_Th_Treg` | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | 0.022 | **0.02568** | 1/day |
| `k_col_deg` | `k_ECM_deg` | --- | --- | --- | --- | 0.035 | --- | **0.004225** | 1/day |
| `k_col_myCAF_sec` | `k_ECM_CAF_sec` | --- | --- | --- | --- | 3.45e-06 | --- | **1.187e-08** | nanomole/cell/day |
| `k_myCAF_death` | `k_CAF_death` | --- | --- | --- | --- | 0.0165 | --- | **0.03246** | 1/day |
| `k_qpsc_death` | `k_fib_death` | --- | --- | --- | --- | 0.0165 | --- | 0.005 | 1/day |
| `k_vas_Msec` |  | 1.7e-05 | NA | 1.7e-05 | 1.7e-05 | 1.7e-05 | 1.7e-05 | 0.05 | picogram/cell/day |
| `n_CD4_clones` | `n_T1_clones` | 63 | 63 | 17 | 92 | 100 | 21-208 16 | **168.1** | dimensionless |
| `n_CD8_clones` | `n_T0_clones` | 63 | 63 | 17 | 100 | 100 | 63 9 | **14.22** | dimensionless |
| `q_CD8_T_in` | `q_T0_T_in` | 5.8e-05 | 5.8e-05 | 0.000145 | 0.000145 | 0.00069 | 2.5-6.5% 14 | **0.0003946** | 1/(centimeter^3*minute) |
| `q_Treg_T_in` | `q_T1_T_in` | 5.8e-05 | 5.8e-05 | 0.000145 | 0.000145 | 0.0005973 | 0.0001305 | **0.0001131** | 1/(centimeter^3*minute) |
| `vol_myCAFcell` | `vol_CAFcell` | --- | --- | --- | --- | 1600 | --- | 3500 | micrometer^3/cell |
| `vol_qPSCcell` | `vol_Fibcell` | --- | --- | --- | --- | 1600 | --- | 1500 | micrometer^3/cell |

> **Bold** PDAC (current) values are posterior medians from submodel inference; plain values are prior medians.

### Fixed / derived parameters

Cancer-type-specific parameters not varied in SBI (constants, PK params, or derived quantities).

| Parameter | TNBC | CRC | UM | NSCLC | HCC | PDAC (ref) | Unit |
|-----------|------|-----|-----|-------|-----|------------|------|
| `APC_PDL1_base` | 5e4 - 2.7e5 | 270000 | 180000 | 5e4 - 2.7e5 | --- | 0-10e4 15 | molecule |
| `A_cell` | 907.9202768874502 | 907.9202768874502 | 907.9202768874502 | 897.2702777917808 | 500 | 572.555 | micrometer^2 |
| `C1_CD80_total` | 80000 | NA | 80000 | 80000 | 80000 | 80000 | molecule |
| `C1_CD86_total` | 860000 | NA | 860000 | 860000 | 860000 | 860000 | molecule |
| `C_CD47` | 100 - 700 | NA | 500 | 50 - 150 | 400 | 147-665 20 | molecule/micrometer^2 |
| `C_max` | 27000 (Gompertzian | NA | NA | 150000000000 | 250000000000 | 1e10 13 | cell |
| `ECM_50_T_mot` | --- | --- | --- | --- | 500000 | --- | nanomolarity |
| `ECM_MW` | --- | --- | --- | --- | 300000 | --- | gram/mole |
| `ECM_base` | --- | --- | --- | --- | 200000 | --- | nanomolarity |
| `ECM_density` | --- | --- | --- | --- | 1.2 | --- | gram/milliliter |
| `ECM_level` | --- | --- | --- | --- | 200000 | --- | nanomolarity |
| `ECM_max` | --- | --- | --- | --- | 5000000 | --- | nanomolarity |
| `M_PD1_total` | 1.5e3 - 6.2e4 | NA | 12500 | 1.5e3 - 6.2e4 | 1.5e3 - 6.2e4 | 1.5e3 - 6.2e4 | molecule |
| `PDL1_C1_baseline` | 5e4 - 2.7e5 | 180000 | 180000 | 5e4 - 2.7e5 | 5e4 - 2.7e5 | 5e4 - 2.7e5 | molecule |
| `T_CD28_total` | 184000 | 184000 | 184000 | 184000 | 184000 | 184000 | molecule |
| `T_CTLA4_syn` | 8000 | 8000 | 8000 | 8000 | 8000 | 8000 | molecule |
| `T_PD1_total` | 1.2e4 - 6.2e4 | 62000 | 62000 | 1.2e4 - 6.2e4 | 1.2e4 - 6.2e4 | 0-400 19 | molecule |
| `T_PDL1_total` | 79980 | 186000 | 186000 | 79980 | 79980 | 79980 | molecule |
| `V_LN` | 1112.647398 | 1112.647398 | 1112.647398 | 1374.4467859455344 | 50000 | 916.2979-1357.58 6 | millimeter^3 |
| `gamma_T_TCE` | NA | 0.522 | 0.522 | --- | --- | --- | dimensionless |
| `gamma_T_aCTLA4` | 0.522 | NA | NA | 0.2 | 0.718 | 0.446-0.918 | dimensionless |
| `gamma_T_aPD1` | 0.522 | NA | NA | 0.2 | 0.718 | 0.446-0.918 | dimensionless |
| `gamma_T_aPDL1` | 0.522 | 0.522 | NA | 0.2 | 0.718 | 0.446-0.918 | dimensionless |
| `k_ECM_fib_sec` | --- | --- | --- | --- | 1.15e-06 | --- | nanomole/cell/day |
| `k_IFNg_Thsec` | 80.29521142666667 | 4E-13 (nanomole/ce | 5E-13 (nanomole/ce | 80.29521142666667 | 80.29521142666667 | 80.29521142666667 | 1/day |
| `k_P0_d1` | 1e-07 | 1e-07 | 1e-07 | 1e-07 | --- | 1e-07 | molarity |
| `k_P1_d1` | 27 | 10 | 10 | 40 | 40 | 100 8 | nanomolarity |
| `k_caf_tran` | --- | --- | --- | --- | 0.2 | --- | 1/day |
| `k_in_PDL1` | 1 | 1 | 1 | 1 | 1 | 1 | 1/day |
| `k_out_PDL1` | 50000 | 50000 | 50000 | 50000 | 50000 | 50000 | molecule/day |
| `q_P_TCE` | NA | fitted | fitted | --- | --- | --- | liter/second |
| `q_P_aCTLA4` | fitted | NA | NA | --- | 1.74e-7 | --- | liter/second |
| `q_P_aPD1` | fitted | NA | NA | --- | 0.000000173655 | --- | liter/second |
| `q_P_aPDL1` | fitted | fitted | NA | --- | 0.000000173655 | --- | liter/second |
| `q_T_TCE` | NA | 8.52e-06 | 0.001 | --- | --- | --- | 1/second |
| `q_T_aCTLA4` | assumed to be cons | NA | NA | --- | 8.52e-6 | --- | milliliter/second |
| `q_T_aPD1` | assumed to be cons | NA | NA | --- | 8.52e-06 | --- | milliliter/second |
| `q_T_aPDL1` | assumed to be cons | 8.52e-05 | NA | --- | 8.52e-06 | --- | milliliter/second |
| `r_PDL2APC` | 0.07 | 0.07 | 0.07 | 0.07 | 0.025 | 0.07 | dimensionless |
| `r_PDL2C1` | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | dimensionless |
| `vol_cell` | 2572.4407845144424 | 2572.4407845144424 | 2572.4407845144424 | 2527.3112824468485 | 2325 | 2642.08 | micrometer^3/cell |
