---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 38px;
    color: #1a5276;
  }
  h2 {
    font-size: 32px;
    color: #2c3e50;
  }
  section.title {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title h1 {
    font-size: 48px;
  }
  section.title h2 {
    font-size: 28px;
    color: #666;
    font-weight: normal;
  }
  section.two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto 1fr;
    gap: 0 30px;
  }
  section.two-col h1 {
    grid-column: 1 / -1;
  }
  table {
    font-size: 18px;
  }
  img {
    display: block;
    margin: 0 auto;
  }
  .small {
    font-size: 18px;
  }
  .footnote {
    font-size: 14px;
    color: #888;
    position: absolute;
    bottom: 30px;
  }
  blockquote {
    border-left: 4px solid #2980b9;
    padding-left: 16px;
    font-style: italic;
    color: #555;
  }
  .highlight {
    color: #c0392b;
    font-weight: bold;
  }
  .good {
    color: #27ae60;
    font-weight: bold;
  }
  .warn {
    color: #e67e22;
    font-weight: bold;
  }
---

<!-- _class: title -->

# PDAC QSP Calibration Pipeline
## Sensitivity Analysis, Restriction Estimation, and NPE Diagnostics

Joel Eliason
Lab Meeting -- February 25, 2026

---

# Pipeline Overview

**Three-stage calibration pipeline** for 177-parameter PDAC QSP model, 31 calibration targets at diagnosis:

1. **Sensitivity analysis** -- Which parameters matter? (177 → 97)
2. **Restriction estimator** -- Can we pre-filter invalid combinations of parameters from their priors? (those that don't grow to the median tumor size within 10 years.)
3. **NPE calibration** -- Learn the full, joint posterior over 97 parameters

---

# Pipeline: Results Summary

| Stage | Method | Result |
|---|---|---|
| **1. Sensitivity analysis** | PRCC + RF importance (2,852 sims) | 97/177 parameters significant (FDR < 0.05) |
| **2. Restriction estimator** | Binary classifier on valid/invalid sims | Null result -- boundary too diffuse in 177D to classify well |
| **3. NPE calibration** | Neural Posterior Estimation (2,567 sims) | 8/97 params identifiable; 11 diagnostics run |

---

# NPE Diagnostics: Cheap Tests (No Retraining)

8 tests that use a single trained NPE:

- **Parameter recovery (R²)** -- can the network recover known parameters?
- **SBC calibration (KS test)** -- are posteriors well-calibrated?
- **Prior predictive p-values** -- can the prior generate observed data?
- **LOO misspecification decomposition** -- which observables drive misfit?

---

# NPE Diagnostics: Cheap + Moderate Tests

Cheap (continued):
- **Approximate posterior predictive check** -- does the posterior predict observations?
- **Posterior correlations** -- which parameters are non-identifiable?
- **Boundary piling** -- is the posterior pushing against prior bounds?
- **MMD misspecification test** -- global distribution mismatch

Moderate (require retraining):
- **Learning curve** -- does more data help?
- **Seed stability** -- is the posterior reproducible?
- **Dimensionality sweep** -- would fewer parameters improve recovery?

---

# Sensitivity Analysis: Method

![h:480](../figures/surrogate_sensitivity/aggregate_ranking.png)

---

# Sensitivity Analysis: How It Works

**What it measures:** PRCC (linear) and Random Forest (nonlinear) permutation importance of 177 parameters against 31 calibration targets.

**Statistical test:** Permutation null test (1000 perms) with Benjamini-Hochberg FDR correction determines significance.

**Goal:** Identify which of 177 parameters actually influence calibration targets, so we can fix the rest and reduce dimensionality for SBI.

---

# Sensitivity Analysis: Result

**Data:** 2,852 valid simulations (of 5,000 raw; 43% NaN rate). Two complementary methods rank parameters; a permutation test answers "calibrate or fix?"

**97/177 parameters significant** (FDR < 0.05).

| Metric | Value |
|---|---|
| Cross-method Spearman rho | 0.865 |
| Top-20 overlap | 16/20 (80%) |
| Top-30 overlap | 24/30 (80%) |
| RF R² on targets | 0.83--0.95 (median ~0.91) |

The 80 non-significant parameters are fixed at prior medians, reducing SBI dimensionality from 177 → 97.

---

# Sensitivity Analysis: Top 5 Parameters

| Rank (PRCC) | Rank (RF) | Parameter | Module |
|:-:|:-:|---|---|
| 1 | 2 | k_PDGF_sec | Fibroblast (PDGF secretion) |
| 2 | 1 | k_myCAF_prolif | Fibroblast (myCAF proliferation) |
| 3 | 10 | k_C1_growth | Cancer (tumor growth rate) |
| 4 | 17 | k_ECM_myCAF_sec | Fibroblast (ECM secretion) |
| 5 | 4 | k_K_g | Cancer (vascular growth) |

Top 5 are all stroma + tumor intrinsic growth.

---

# Sensitivity Analysis: Parameters 6--10

| Rank (PRCC) | Rank (RF) | Parameter | Module |
|:-:|:-:|---|---|
| 6 | 8 | k_TGFb_sec_myCAF | Fibroblast (TGFb secretion) |
| 7 | 5 | q_nCD8_P_out | T cell (naive CD8 peripheral exit) |
| 8 | 12 | q_CD8_T_in | T cell (CD8 tumor infiltration) |
| 9 | 14 | q_nCD8_LN_in | T cell (naive CD8 LN entry) |
| 10 | 7 | N_IL2_CD4 | T cell (IL-2 CD4 expansion) |

Also in union of top-13: k_myCAF_to_iCAF (14/9), k_cell_clear (22/6), k_M2_pol (26/3).

---

# Sensitivity Analysis: Biological Clusters

**The model is dominated by stroma and T cell trafficking:**

| Cluster | Key parameters |
|---|---|
| **Fibroblast/stroma** (5/13) | k_PDGF_sec, k_myCAF_prolif, k_ECM_myCAF_sec, k_TGFb_sec_myCAF, k_myCAF_to_iCAF |
| **CD8 trafficking** (3/13) | q_nCD8_P_out, q_CD8_T_in, q_nCD8_LN_in |
| **Tumor intrinsic** (2/13) | k_C1_growth, k_K_g |
| **Immune regulation** (3/13) | N_IL2_CD4, k_cell_clear, k_M2_pol |

Dense desmoplastic stroma is the defining feature of PDAC; the full thymus→LN→tumor trafficking chain matters -- the bottleneck is *getting T cells there*, not killing efficiency.

---

# Sensitivity Analysis: Notable Findings

**Notable absences:**
- p_T_kill_per_contact (rank 171) -- killing efficiency doesn't matter
- CXCR3/chemotaxis module all non-significant -- constitutive trafficking dominates
- k_C1_death (rank 45/99) -- once tumor is viable, death rate barely matters

**Nonlinear-only** (top RF but not PRCC):
- k_M2_pol (RF rank 3, PRCC rank 26) -- switch-like M2 polarization
- k_cell_clear (RF rank 6, PRCC rank 22) -- DAMP threshold for APC maturation

---

# Sensitivity Analysis: Other Highlights

- **apCAF→Treg axis** significant (k_apCAF_Treg rank 42) -- Elyada 2019 mechanism detectable at diagnosis
- **CXCL12 exclusion axis** significant (CXCL12_50_Texcl rank 23) -- Feig 2013 immune desert mechanism
- **Metabolism module** has real teeth: 9 parameters significant including glucose competition and hypoxia
- **n_CD8_clones** (rank 25) is a proxy for TMB -- immune response scales with neoantigen diversity
- **PD1_50** significant but low-ranked (rank 83) -- checkpoint axis wired correctly, will matter more under anti-PD-1

---

# Restriction Estimator: Problem

**Problem:** ~49% of simulations return NaN because `evolve_to_diagnosis` fails to grow the tumor to 3.2 cm within the evolution window. Wasted compute.

**Approach:** sbi `RestrictionEstimator` (Deistler et al. 2022) -- train a binary classifier on (θ, valid/invalid) labels, then use rejection sampling to only propose viable parameter sets.

**Classifier:** ResNet with 512 hidden features, 4 blocks. Trained with loss importance weights. 3 iterative refinement rounds.

---

# Restriction Estimator: Null Result

**Classifier accuracy:** ~65% (barely above 57% majority-class baseline).

| allowed_false_negatives | acceptance rate | effective rejection |
|---|---|---|
| 0.05 | ~0.03 | 97% |
| 0.10 | ~0.08 | 92% |
| 0.20 | ~0.18 | 82% |

**Why it failed:** The valid/invalid boundary is diffuse in 177D -- no sharp decision surface. k_C1_growth/k_C1_death ratio (AUC ~0.65) is the best single predictor, but insufficient.

**What we did instead:** Ran 5,000 sims, dropped NaN rows, worked with 2,567 valid sims.

---

# Quick Primer: What is NPE?

**Neural Posterior Estimation** learns the mapping from observations → parameter distributions.

1. **Sample** parameters θ from the prior: θ ~ p(θ)
2. **Simulate** the QSP model: x = f(θ) (MATLAB SimBiology)
3. **Train** a neural density estimator on (θ, x) pairs
4. **Condition** on real observed data x_obs → get posterior p(θ | x_obs)

Unlike optimization (`sbiofit`): we get the **full joint posterior** over all 97 parameters simultaneously, not just a point estimate.

Unlike MCMC: **amortized** -- train once, condition on any observation instantly.

> **Key question:** How do we know the posterior is correct? That's what the diagnostics answer.

---

# NPE Summary Dashboard

| Metric | Value | Assessment |
|--------|-------|------------|
| Median R² | **0.083** | Poor -- most params not individually recoverable |
| R² > 0.5 | 8 / 97 | Only ~8% well-recovered |
| MMD p-value | **0.017** | Misspecified (p < 0.05) |
| Prior predictive p < 0.05 | **7 / 30** | Model can't reproduce 7 observables |
| Boundary piling (>20%) | 10 / 97 | Prior misspecification for ~10% |
| PPC obs in 90% CI | 21 / 30 | 9 observables fail posterior predictive |
| Strong correlations (\|r\| > 0.5) | 7 pairs | Structural non-identifiabilities |
| Learning curve final R² | **0.085** | Saturated -- more data won't help |
| Seed stability (R² std) | **0.012** | Stable across seeds |
| Dim. sweep best R² | **0.242** (D=8) | 3x better than full D=97 |

---

<!-- _class: two-col -->

# Parameter Recovery (R²)

![h:450](../figures/sbi_baseline_sa97/recovery.png)

**What it measures:** Compare posterior median to known truth on held-out test sims. R² = 1 means perfect recovery; R² ≤ 0 means the network learns nothing beyond the prior.
**Result:** Median R² = 0.083. Only 8/97 parameters have R² > 0.5.

---

# Parameter Recovery: Well-Recovered (Top 4)

| Parameter | R² | Biological role |
|---|---|---|
| pO2_50 | **0.967** | O2 half-saturation (hypoxia threshold) |
| k_psc_activation_apCAF | 0.835 | apCAF activation rate |
| k_cell_clear | 0.742 | Dead cell clearance |
| k_M2_pol | 0.726 | M1→M2 polarization |

pO2_50 is by far the best-recovered -- hypoxia is the dominant identifiable axis.

---

# Parameter Recovery: Well-Recovered (5--8)

| Parameter | R² | Biological role |
|---|---|---|
| APC0_T | 0.618 | Baseline APC density in tumor |
| k_Mac_rec | 0.571 | Macrophage recruitment |
| k_Treg_death | 0.550 | Treg death rate |
| k_ECM_myCAF_sec | 0.534 | ECM secretion by myCAFs |

All 8 span hypoxia, stroma, and immune regulation -- the dominant biological axes.

---

<!-- _class: two-col -->

# Prior Predictive p-values

![h:450](../figures/sbi_baseline_sa97/prior_predictive_pvalues.png)

**What it measures:** What fraction of prior sims produce a value as extreme as the real observation? Small p → the model prior can't generate what we actually observe.
**Result:** 7/30 observables have p < 0.05.

---

# Prior Predictive Histograms

![h:520](../figures/sbi_baseline_sa97/prior_predictive_histograms.png)

---

# Prior Predictive: Failing Observables

| Observable | p | Issue |
|---|---|---|
| cdc1_fraction_of_dc | **0.000** | Observed 0.81 above all sims |
| psc_activation_ratio | **0.011** | Model over-activates PSCs |
| stromal_fraction | **0.014** | Stroma module too aggressive |
| tam_to_cdc1_ratio | **0.016** | Too many TAMs relative to cDC1 |
| cancer_cell_density | 0.020 | Cellularity formula mismatch |
| collagen1_protein | 0.030 | ECM overproduction |
| apcaf_fraction_of_caf | 0.035 | Too few apCAFs |

Two clusters: **stroma overproduction** and **DC subtype imbalance**. These are structural model issues, not prior width problems.

---

<!-- _class: two-col -->

# LOO Misspecification Decomposition

![h:420](../figures/sbi_baseline_sa97/loo_predictive.png)

**What it measures:** Mahalanobis distance D² of observation from training distribution. Remove each observable one at a time; influence = D²_full − D²_without_j. Captures **multivariate structure** unlike marginal p-values.

---

# LOO: Top Drivers of Misspecification

D²_full = 143.5. Top 3 observables account for 56.7%:

| Observable | ΔD² | % total |
|---|---|---|
| cdc1_fraction_of_dc | **36.5** | 25.4% |
| tumor_glucose_ratio | **25.4** | 17.7% |
| tumor_pO2_baseline | **19.5** | 13.6% |
| tam_to_cdc1_ratio | 12.4 | 8.7% |
| cdc1_density_baseline | 11.9 | 8.3% |

glucose and pO2 have marginal p > 0.1 but their **joint** relationship with DC observables drives misspecification.

---

# Boundary Piling (Top 5)

**What it measures:** Fraction of posterior mass in the outer 5% of the prior range. If >20%, the posterior is "pushing against the wall."

| Parameter | Dir. | Piling | Meaning |
|---|---|---|---|
| f_cDC1 | ↑ | **93.7%** | More cDC1 fraction |
| k_APC_mat_cDC2 | ↑ | **87.8%** | Faster cDC2 maturation |
| k_APC_mat_cDC1 | ↓ | **84.8%** | Slower cDC1 maturation |
| k_O2_demand | ↑ | **73.3%** | Deeper hypoxia |
| k_glc_immune | ↓ | 35.0% | Less immune glucose use |

---

# Boundary Piling (continued)

| Parameter | Dir. | Piling | Meaning |
|---|---|---|---|
| k_psc_const | ↑ | 30.6% | Higher PSC activation |
| k_cell_clear | ↓ | 26.5% | Slower clearance |
| k_APC_mig | ↑ | 24.1% | Faster APC migration |
| k_MDSC_rec | ↑ | 22.4% | More MDSC recruitment |
| k_mAPC_death | ↑ | 20.1% | Faster mAPC death |

---

# Boundary Piling: Interpretation

**DC biology is the strongest signal** (5 of 10 flagged parameters). The posterior wants more immature cDC1 in tumor but faster cDC2 maturation -- consistent with the PDAC "immune desert" phenotype.

**Action:** widen f_cDC1 prior (σ=0.3 → 0.6), k_O2_demand, k_APC_mat_cDC1/cDC2.

---

# Posterior vs Prior (Top 13 Parameters)

![h:520](../figures/sbi_baseline_sa97/posterior_vs_prior.png)

---

# Posterior vs Prior: Observations

- **pO2_50** has the sharpest posterior -- hypoxia threshold is the most identifiable axis
- **Fibroblast parameters** (k_PDGF_sec, k_myCAF_prolif, k_ECM_myCAF_sec) show moderate narrowing
- Most parameters show **minimal posterior narrowing** -- consistent with median R² = 0.083
- Visible shifts correspond to the 8 well-recovered (R² > 0.5) parameters

---

# Posterior Correlations

![h:520](../figures/sbi_baseline_sa97/posterior_correlations.png)

---

# Posterior Correlations: Interpretation

**What it measures:** Pearson correlation between posterior samples. Strong correlations (|r| > 0.5) reveal **structural non-identifiabilities** -- parameter combinations the data can't distinguish.

**7 strong pairs found.** All production/degradation pairs with positive correlation.

---

# Posterior Correlations: Non-Identifiable Pairs

| Param 1 | Param 2 | r | Mechanism |
|---|---|---|---|
| k_CD4_death | k_cell_clear | **+0.91** | CD4 clearance |
| k_vas_Csec | k_vas_deg | **+0.86** | VEGF steady-state |
| k_M2_pol | k_M1_pol | +0.69 | M1↔M2 balance |
| k_Treg_death | k_apCAF_Treg | +0.67 | Net Treg density |
| k_MDSC_rec | k_MDSC_death | +0.64 | MDSC steady-state |
| k_psc_act_apCAF | k_apCAF_death | +0.56 | Net apCAF density |
| k_Mac_rec | k_Mac_death | +0.53 | TAM steady-state |

Data constrains *steady-state levels* (ratios) but not *turnover rates*.
**Action:** reparameterize as (steady-state, turnover) → ~90 effective params.

---

<!-- _class: two-col -->

# Learning Curve

![h:380](../figures/sbi_baseline_sa97/learning_curve.png)

**What it measures:** Retrain NPE at 10%–100% of training data. If R² keeps rising, we need more sims. If it plateaus, the bottleneck is elsewhere.
**Saturates at ~1500 sims.** More data won't help -- bottleneck is information content of 30 observables about 97 parameters.

---

# Learning Curve: Data

| N | Med R² | R² > 0.5 | R² > 0 |
|---|---|---|---|
| 256 | 0.004 | 1 | 52 |
| 513 | 0.054 | 3 | 62 |
| 1026 | 0.063 | 7 | 70 |
| 1540 | 0.091 | 8 | 73 |
| 2053 | 0.084 | 9 | 76 |
| 2567 | 0.085 | 9 | 77 |

Number of well-recovered parameters (R² > 0.5) stabilizes at 8--9 from ~1500 sims onward.

To improve, we need: more informative observables (time-course, perturbation), fewer parameters (fix non-identifiable ones), or observables that break production/degradation degeneracies.

---

<!-- _class: two-col -->

# Seed Stability (5 Seeds)

![h:380](../figures/sbi_baseline_sa97/seed_stability.png)

**What it measures:** Retrain NPE with 5 random seeds, same data. High variance → unstable posterior.
Median R² std across seeds: **0.012**. Well-recovered params very stable (std < 0.025). **Seed instability is not a concern.** Ensembling unnecessary.

---

# Seed Stability: Per-Parameter

| Parameter | R² (mean ± std) |
|---|---|
| pO2_50 | 0.923 ± 0.045 |
| k_psc_activation_apCAF | 0.826 ± 0.003 |
| k_cell_clear | 0.737 ± 0.008 |
| k_M2_pol | 0.719 ± 0.003 |
| APC0_T | 0.595 ± 0.013 |
| k_ECM_myCAF_sec | 0.562 ± 0.024 |
| k_Mac_rec | 0.549 ± 0.013 |
| k_Treg_death | 0.535 ± 0.014 |

The top-8 parameters are highly reproducible across seeds -- the NPE learns a consistent posterior.

---

<!-- _class: two-col -->

# Dimensionality Sweep

![h:380](../figures/sbi_baseline_sa97/dimensionality_sweep.png)

**What it measures:** Retrain NPE with top-N parameters by SA rank (keeping all 30 observables). If fewer params → better R², the system is over-parameterized for the available data.
**Best median R² at D=8 (0.242), 3x the full D=97 (0.090).**

---

# Dimensionality Sweep: Data

| D (params) | Med R² | R² > 0.5 | R² > 0 |
|---|---|---|---|
| 8 | **0.242** | 1 | 8 |
| 15 | 0.190 | 1 | 15 |
| 20 | 0.172 | 3 | 20 |
| 30 | 0.170 | 6 | 29 |
| 50 | 0.128 | 7 | 45 |
| 97 (full) | 0.090 | 9 | 72 |

**Quality-vs-quantity tradeoff:** fewer params → higher per-parameter R², but more total params recovered at full D. **Sweet spot: D=30** — median R² doubles while recovering 6 params at R² > 0.5.

---

# Dimensionality Sweep: Interpretation

**Why D=30 is the sweet spot:**
- Median R² doubles (0.170 vs 0.090) -- each parameter better constrained
- Still recovers 6 of the 9 well-identified parameters
- 29/30 params have R² > 0 (nearly all positively informed)

**Surprise: pO2_50 only identifiable at full D**
- Best-recovered parameter overall (R² = 0.953) but PRCC rank > 50
- Identifiable through **nonlinear joint constraints**, not monotonic sensitivity
- PRCC-based parameter selection would miss it

**Implication:** Reducing to D=30 is tempting, but some parameters are identifiable only through high-dimensional interactions. Better strategy: keep all 97 but use sweep to prioritize prior tuning.

---

# Parameter-Observable Sensitivity (PRCC)

Which observables constrain the 8 well-recovered parameters?

| Parameter | R² | Key Observable(s) | PRCC |
|---|---|---|---|
| pO2_50 | 0.953 | hif1a_positivity_rate | +0.60 |
| k_psc_activation_apCAF | 0.821 | apcaf_fraction_of_caf | +0.79 |
| k_cell_clear | 0.744 | th_exhausted_fraction | **-0.96** |
| k_M2_pol | 0.732 | m1_m2_ratio / m2_macrophage_fraction | **-0.95 / +0.95** |
| APC0_T | 0.618 | tam_to_cdc1_ratio, cdc1_density | -0.75, +0.66 |
| k_Mac_rec | 0.571 | tam_density, tam_to_cdc1_ratio | +0.75, +0.69 |
| k_Treg_death | 0.550 | treg_fraction_cd4 | **-0.85** |
| k_ECM_myCAF_sec | 0.534 | collagen1, tumor_volume (7 total) | +0.64, +0.64 |

---

# PRCC: Why These Parameters Are Identifiable

**Pattern:** Well-recovered parameters have strong, near-exclusive PRCC relationships with 1--3 observables.

- **pO2_50** is the extreme case -- a single observable (|PRCC|=0.60) suffices for R²=0.95
- **k_M2_pol** and **k_cell_clear** have the strongest individual PRCCs (|0.95|) -- essentially "solved" by one observable each
- **k_ECM_myCAF_sec** has the broadest footprint (7 observables) -- ECM secretion affects collagen, volume, stroma, and CAF composition

**cdc1_fraction_of_dc** (the worst-failing observable) is driven by only 3 params: f_cDC1 (+0.82), k_APC_mat_cDC1 (-0.83), k_APC_mat_cDC2 (+0.80). These are exactly the 3 extreme boundary-piling params -- the data demands values outside the prior.

---

# Conflicting Observable Constraints

Some observables pull the same parameter in **opposite directions**. Combining PRCC (direction of influence) with z-score (direction of model-vs-reality mismatch) reveals which parameters are caught in a tug-of-war.

**Worked example -- k_ECM_myCAF_sec:**
- `stromal_fraction` (PRCC=+0.33, z=−6.4): model over-produces stroma → wants ECM secretion **lower**
- `psc_activation_ratio` (PRCC=−0.49, z=−6.4): model over-activates PSCs → wants ECM secretion **higher**

Both off by ~6σ, opposite pull directions. No single value satisfies both.

---

# Conflicting Constraints: Top 5

| Parameter | R² | Wants ↑ | Wants ↓ |
|---|---|---|---|
| k_ECM_myCAF_sec | 0.534 | psc_activation (z=−6.4) | stromal_fraction (z=−6.4) |
| k_C1_growth | 0.232 | stromal_fraction (z=−6.4) | psc_activation (z=−6.4) |
| k_ECM_deg | 0.129 | collagen1 (z=−2.8) | psc_activation (z=−6.4) |
| k_APC_mat_cDC2 | 0.468 | cdc1_fraction (z=+5.6) | apc_maturation (z=−1.2) |
| k_K_g | 0.463 | stromal_frac (z=−6.4) | tumor_pO2 (z=−1.0) |

**Stroma observables are the main source of conflict.** The model simultaneously over-produces stromal fraction AND over-activates PSCs, but these have opposite parameter dependencies. This is a structural model issue -- fixing one makes the other worse.

---

<!-- _class: two-col -->

# Approximate Posterior Predictive Check

![h:420](../figures/sbi_baseline_sa97/posterior_predictive_check.png)

**What it measures:** For each posterior sample, find k nearest neighbors in training set as proxy for model predictions. Compare to real observations.
**21/30 in 90% CI.** Same observables that fail prior predictive also fail PPC -- posterior **can't rescue** structural mismatch.

---

# PPC: Failing Observables

| Observable | Obs | 90% CI |
|---|---|---|
| cdc1_fraction_of_dc | 0.81 | [0.10, 0.43] |
| psc_activation_ratio | 0.57 | [0.91, 1.00] |
| tam_to_cdc1_ratio | 4.0 | [27, 1159] |
| collagen1_protein | 9.8 | [65, 310] |
| stromal_fraction | 0.67 | [0.94, 1.00] |
| apcaf_fraction_of_caf | 0.064 | [0.0003, 0.039] |
| cancer_cell_density | 996 | [3.4, 271] |
| cd8_exhausted_fraction | 0.73 | [0.10, 0.71] |
| apc_maturation_ratio | 0.014 | [0.016, 0.50] |

---

# Is the NPE the Bottleneck?

**No -- structural model issues dominate.** The R² improvement at lower D is expected from information theory (30 observables / fewer params = more info per param), not NPE failure.

Evidence:
- Even at D=8, 7/8 params have R² < 0.5 -- the signal isn't there in snapshot data
- The best-recovered params (pO2_50=0.95, k_cell_clear=0.74) aren't the most sensitive -- they're identifiable via **nonlinear fingerprints**, not monotonic sensitivity
- Seed stability is excellent (std=0.012) -- the NPE consistently learns what's learnable
- 7/30 observables fail prior predictive -- no estimator can calibrate to data the model can't generate

**Fix the forward model first.** After structural fixes, if R² is still low, *then* try a better NPE architecture.

---

# Why Not sbiofit?

A head-to-head SBI vs `sbiofit` comparison isn't useful yet:

| Issue | Impact |
|---|---|
| 7 failing observables | Both methods would struggle with the same broken targets |
| 97 parameters | `sbiofit` multistart is intractable at this dimensionality |
| Evaluation metric | Point estimate + asymptotic CIs vs full posterior -- not comparable |
| Non-identifiabilities | `sbiofit` would silently pick one mode; the 7 correlated pairs (Section 10c) would be invisible |

**After structural fixes**, a focused comparison on the 8 well-recovered parameters could be informative. Until then, the posterior correlations already demonstrate something `sbiofit` fundamentally cannot show.

---

# What the Diagnostics Tell Us

| Problems | Strengths |
|---|---|
| **Stroma module too aggressive**: stromal_fraction, collagen, PSC activation all fail | **Posteriors are stable**: seed std = 0.012 |
| **cDC1 fraction structurally wrong**: observed 0.81 exceeds all 5000 sims | **8 well-identified params** with clear biological signatures |
| **Most parameters unidentifiable**: only 8/97 from 30 snapshot observables | **Dim. sweep confirms** D=30 sweet spot (R² doubles) |
| **7 non-identifiable pairs**: production/degradation degeneracies | **21/30 observables pass PPC** |
| **Learning curve saturated**: more sims won't help | **Diagnostic infrastructure** in place for iteration |

---

# Next Steps

### Immediate
1. **Reparameterize 7 correlated pairs** as (steady-state, turnover) → ~90 effective params
2. **Widen priors** for f_cDC1 (σ=0.3 → 0.6), k_O2_demand, k_APC_mat_cDC1/cDC2
3. **Fix ~20 negative-R² parameters** at prior medians → ~77 params
4. **Consider D=30 focused calibration** — dim. sweep shows 2x R² improvement

### Medium-term
5. **Investigate cdc1_fraction_of_dc**: model structural change to APC subtype balance
6. **Address stromal_fraction mismatch** (z = −6.4): fibroblast module structural issue
7. **Add perturbation observables**: time-course or treatment-response data to break degeneracies

### Not needed
- ~~More simulations~~ (learning curve saturated at 1500)
- ~~Ensemble methods~~ (seed stability is excellent)

---

# Backup: Posterior Pairs (Top 6)

![h:520](../figures/sbi_baseline_sa97/posterior_pairs_top6.png)

---

# Backup: SBC Calibration ECDF

**What it measures:** Percentile rank of true θ in posterior should be Uniform(0,1) if well-calibrated. KS test detects deviations. Gray band = expected noise.

![h:380](../figures/sbi_baseline_sa97/calibration_ecdf.png)

<div class="small">8/97 parameters fail KS test at p < 0.05. Top reject: CCL2_50 (KS=0.123, p=0.0003).</div>

---

# Backup: Z-Score vs Contraction

**What it measures:** Z-score = (posterior mean − true) / posterior std. Should be ~N(0,1). Contraction = 1 − (posterior variance / prior variance). High contraction → data is informative.

![h:400](../figures/sbi_baseline_sa97/z_score_contraction.png)

---

# Backup: Cross-Method Agreement (SA)

![h:520](../figures/surrogate_sensitivity/cross_method_agreement.png)

---

# Backup: Cutoff Analysis (SA)

![h:520](../figures/surrogate_sensitivity/cutoff_analysis.png)

---

# Backup: PRCC Heatmap (Well-Recovered Params + DC Axis)

![h:420](../figures/sbi_baseline_sa97/prcc_heatmap_key_params.png)

<div class="small">PRCC values annotated where |PRCC| > 0.3. Parameters sorted by R²; observables sorted by max |PRCC|.</div>

---

# Backup: PRCC Heatmap (Full 97 x 31, Clustered)

![h:520](../figures/sbi_baseline_sa97/prcc_heatmap_full.png)
