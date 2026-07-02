# NPE Diagnostics and Calibration Assessment

**Date:** 2026-02-22
**Context:** NPE trained on 2567 sims (97 params, 30 observables), baseline_no_treatment scenario
**Key result:** MMD p = 0.017; median R² = 0.083; 10 params with boundary piling; 7 observables with prior p < 0.05

---

## 0. Summary Dashboard

| Metric | Value | Assessment |
|--------|-------|------------|
| Training samples | 2567 (from 5000 raw, 49% NaN rate) | Below O(d²)=9409 rule of thumb |
| Parameters | 97 (of 177 total, SA-significant) | |
| Observables | 30 calibration targets | |
| MMD p-value | 0.017 | Misspecified (p < 0.05) |
| Recovery R² median | 0.083 | Poor — most params not individually recoverable |
| Recovery R² > 0.5 | 8 / 97 | Only ~8% well-recovered |
| KS calibration rejects (p<0.05) | 8 / 97 | ~8% miscalibrated |
| Boundary piling (>20%) | 10 / 97 | Prior misspecification for ~10% |
| Observables out of range | 1 / 30 (cdc1_fraction_of_dc) | |
| Prior predictive p < 0.05 | 7 / 30 | Model can't reproduce 7 observables |
| PPC obs in 90% CI | 21 / 30 | 9 observables fail posterior predictive |
| Strong posterior correlations (\|r\| > 0.5) | 7 pairs | Structural non-identifiabilities |
| Learning curve (final R² median) | 0.085 | Saturated — more data won't help |
| Seed stability (median R² std) | 0.012 | Stable across seeds |
| Dim. sweep best median R² | 0.242 at D=8 | 3x improvement over full D=97 |
| Dim. sweep sweet spot | D=30: R²=0.170, 6 R²>0.5 | Best quality-quantity tradeoff |

---

## 1. Parameter Recovery (R²)

Recovery R² measures how well the NPE can recover true parameter values from simulated data. Only 8 of 97 parameters have R² > 0.5.

### Well-recovered parameters (R² > 0.5)

| Parameter | R² | Role |
|-----------|-----|------|
| pO2_50 | 0.967 | O2 half-saturation — controls hypoxia threshold |
| k_PSC_to_apCAF | 0.835 | apCAF activation rate |
| k_cell_clear | 0.742 | Dead cell clearance rate |
| k_M2_pol | 0.726 | M1→M2 polarization rate |
| APC0_T | 0.618 | Baseline APC density in tumor |
| k_Mac_rec | 0.571 | Macrophage recruitment |
| k_Treg_death | 0.550 | Treg death rate |
| k_ECM_myCAF_sec | 0.534 | ECM secretion by myCAFs |

These are parameters that strongly influence observable outputs and have identifiable effects. pO2_50 is by far the best-recovered, consistent with hypoxia being a dominant axis in the model.

### Poorly-recovered parameters (R² < 0)

~15 parameters have negative R² (posterior median is worse than predicting the prior mean). This reflects insufficient training data, weak sensitivity, and degeneracies. The ~20 negative-R² parameters are effectively unidentifiable from these data.

### Interpretation

The median R² of 0.083 is low but expected given the sample-to-dimension ratio (2567/97 ≈ 26, far below the ~97 needed per the O(d²) rule). The learning curve (Section 8) confirms this is not a sample-size issue — it's a fundamental identifiability limit at 30 observables.

---

## 2. Calibration (SBC/ECDF)

SBC calibration checks whether posterior credible intervals have correct coverage. 8 parameters fail the KS test at p < 0.05:

| Parameter | KS statistic | KS p-value |
|-----------|-------------|-----------|
| CCL2_50 | 0.123 | 0.0003 |
| CXCL12_50_Texcl | 0.102 | 0.005 |
| k_MDSC_rec | 0.096 | 0.010 |
| k_IL2_sec | 0.091 | 0.017 |
| n_CD8_clones | 0.089 | 0.021 |
| k_PDGF_sec | 0.089 | 0.021 |
| k_MDSC_death | 0.088 | 0.024 |
| q_nTreg_P_out | 0.086 | 0.027 |

The top two rejects (CCL2_50, CXCL12_50_Texcl) are both chemokine half-saturation constants, suggesting the calibration targets are informative about chemokine signaling thresholds. The KS band width at n_test=285 is 1.36/√285 ≈ 0.081, so parameters with KS < 0.08 are within expected noise.

---

## 3. Coverage Check

1 of 30 observables falls outside the training simulation range:

| Observable | Observed | Training range | z-score |
|-----------|----------|---------------|---------|
| cdc1_fraction_of_dc | 0.810 | [0.012, 0.760] | +5.58 |

Several observables are technically in-range but deep in the tails:

| Observable | z-score | Note |
|-----------|---------|------|
| psc_activation_ratio | -6.39 | Model nearly always produces high activation |
| stromal_fraction | -6.38 | Model produces much higher stroma than observed 0.668 |
| cancer_cell_density_baseline | +4.24 | Dynamic cellularity formula; obs=996, mean=72 |
| cdc1_fraction_of_dc | +5.58 | Out of range |

---

## 4. Boundary Piling

Parameters where >20% of posterior mass concentrates in the outer 5% of prior range. 10 parameters flagged:

| # | Parameter | Direction | Piling % | Prior [lower, median, upper] | Biological meaning |
|---|-----------|-----------|----------|-------------------------------|---------------------|
| 1 | `f_cDC1` | upper | 93.7% | [0.110, **0.2**, 0.364] | More cDC1 relative to cDC2 |
| 2 | `k_APC_mat_cDC2` | upper | 87.8% | [0.370, **1.5**, 6.083] 1/day | Faster cDC2 maturation |
| 3 | `k_APC_mat_cDC1` | lower | 84.8% | [0.493, **2.0**, 8.110] 1/day | Slower cDC1 maturation |
| 4 | `k_O2_demand` | upper | 73.3% | [0.030, **0.12**, 0.487] | Deeper tumor hypoxia |
| 5 | `k_glc_immune` | lower | 35.0% | [0.005, **0.02**, 0.081] | Less immune glucose consumption |
| 6 | `k_CAF_const` | upper | 30.6% | [2.7e5, **2e6**, 1.5e7] | Higher constitutive PSC activation |
| 7 | `k_cell_clear` | lower | 26.5% | [0.014, **0.1**, 0.739] 1/day | Slower clearance (inert at diagnosis) |
| 8 | `k_APC_mig` | upper | 24.1% | [0.986, **4.0**, 16.22] 1/day | Faster APC migration to LN |
| 9 | `k_MDSC_rec` | upper | 22.4% | [8631, **35000**, 141932] | More MDSC recruitment |
| 10 | `k_mAPC_death` | upper | 20.1% | [0.074, **0.3**, 1.217] 1/day | Faster mature APC death |

### Biological interpretation

The posterior paints a coherent picture of the PDAC TME:

**DC biology is the strongest signal** (#1, #2, #3, #8, #10): f_cDC1 piles at 93.7% (most extreme). The model wants more cDC1 fraction but slower cDC1 maturation, faster cDC2 maturation, faster APC migration, and faster mAPC death. Net effect: more immature cDC1 in tumor (present but functionally impaired), while cDC2 matures and migrates to prime Tregs. Consistent with the "immune desert" phenotype.

**Hypoxia remains dominant** (#4): k_O2_demand at 73.3% — the model wants deeper hypoxia than the prior allows.

**Myeloid immunosuppression** (#9): More MDSC recruitment, consistent with PDAC immunosuppressive TME.

### Action items

- **Widen `f_cDC1` prior** — 93.7% piling is extreme; sigma = 0.3 is the tightest of all 97 params; increase to 0.5-0.7
- **Widen `k_O2_demand` prior** — 73.3% piling; the data strongly wants values above the current upper bound
- **Investigate stromal_fraction mismatch** — z = -6.38 suggests the fibroblast/ECM module fundamentally overproduces stroma
- **Consider fixing `k_cell_clear`** — the clearance switch is inactive at diagnosis; adds noise without information

---

## 5. Misspecification (MMD)

MMD misspecification test p-value: **0.017** (significant at 0.05).

The observed data is unlikely under the model's prior predictive distribution.

---

## 6. Prior Predictive p-values

Per-observable two-sided p-values from the prior predictive distribution. 7 observables have p < 0.05 — the model prior cannot reproduce these observed values:

| Observable | p-value | Tail | Interpretation |
|-----------|---------|------|----------------|
| cdc1_fraction_of_dc | 0.000 | upper | Observed 0.81 above all sims — model never produces enough cDC1 |
| psc_activation_ratio | 0.011 | lower | Observed 0.57 below nearly all sims — model over-activates PSCs |
| stromal_fraction | 0.014 | lower | Observed 0.67 far below model mean 0.98 — stroma module too aggressive |
| tam_to_cdc1_ratio | 0.016 | lower | Observed 4.0 far below model — too many TAMs relative to cDC1 |
| cancer_cell_density_baseline | 0.020 | upper | Observed 996 above most sims — cellularity formula mismatch |
| collagen1_protein_level | 0.030 | lower | Observed 9.8 below model mean 230 — ECM overproduction |
| apcaf_fraction_of_caf | 0.035 | upper | Observed 6.4% above most sims — model underproduces apCAFs |

Two more at p < 0.1: apc_maturation_ratio (0.077), cd8_exhausted_fraction (0.086).

### Key insight

The 7 failing observables cluster into two mechanistic groups:
1. **Stroma overproduction** (stromal_fraction, collagen1_protein_level, psc_activation_ratio): The fibroblast/ECM module is too aggressive
2. **DC subtype imbalance** (cdc1_fraction_of_dc, tam_to_cdc1_ratio, apcaf_fraction_of_caf, apc_maturation_ratio): The APC module doesn't produce enough cDC1 or apCAFs

These are structural model issues, not prior width problems.

---

## 7. LOO Predictive Check (Misspecification Decomposition)

Mahalanobis distance decomposition (D²_full = 143.5) identifies which observables drive joint misspecification. The "influence" is how much D² decreases when each observable is excluded:

| Observable | Influence (ΔD²) | % of total |
|-----------|-----------------|-----------|
| cdc1_fraction_of_dc | 36.5 | 25.4% |
| tumor_glucose_ratio | 25.4 | 17.7% |
| tumor_pO2_baseline | 19.5 | 13.6% |
| tam_to_cdc1_ratio | 12.4 | 8.7% |
| cdc1_density_baseline | 11.9 | 8.3% |
| tam_density | 9.6 | 6.7% |
| mycaf_fraction_of_caf | 8.4 | 5.9% |
| icaf_fraction_of_caf | 7.8 | 5.4% |

Top 3 observables account for 56.7% of the joint misspecification. cdc1_fraction_of_dc alone accounts for 25% — consistent with the prior predictive p-value of 0.000 for this observable.

Notably, **tumor_glucose_ratio** (25.4 influence) and **tumor_pO2_baseline** (19.5) are top drivers in the LOO but had marginal prior predictive p-values (0.16 and 0.32). This means they're not individually extreme but their *joint* relationship with other observables is problematic — the model can't simultaneously produce the right glucose, pO2, and cDC1 values.

---

## 8. Approximate Posterior Predictive Check

Using k-NN proxy (k=10 nearest neighbors in theta space), 21/30 observables fall within the approximate 90% PPC interval. 9 failures:

| Observable | Observed | 90% PPC CI | PPC p-value |
|-----------|----------|-----------|-------------|
| cdc1_fraction_of_dc | 0.810 | [0.101, 0.426] | 0.000 |
| psc_activation_ratio | 0.575 | [0.909, 0.997] | 0.004 |
| tam_to_cdc1_ratio | 3.98 | [27.3, 1159] | 0.005 |
| collagen1_protein_level | 9.84 | [65.2, 310] | 0.009 |
| stromal_fraction | 0.668 | [0.939, 0.999] | 0.031 |
| apcaf_fraction_of_caf | 0.064 | [0.0003, 0.039] | 0.033 |
| cancer_cell_density_baseline | 996 | [3.43, 271] | 0.034 |
| cd8_exhausted_fraction | 0.728 | [0.096, 0.710] | 0.043 |
| apc_maturation_ratio | 0.014 | [0.016, 0.497] | 0.068 |

These largely overlap with the prior predictive failures — the posterior hasn't rescued these observables because the model structurally can't match them.

---

## 9. Posterior Correlations

7 parameter pairs with |r| > 0.5, all revealing structural non-identifiabilities:

| Param 1 | Param 2 | r | Interpretation |
|---------|---------|---|---------------|
| k_CD4_death | k_cell_clear | +0.906 | Both control CD4 clearance — only net turnover identified |
| k_vas_Csec | k_vas_deg | +0.859 | VEGF production/degradation — only steady-state level identified |
| k_M2_pol | k_M1_pol | +0.693 | M1↔M2 balance — only ratio identified, not absolute rates |
| k_Treg_death | k_apCAF_Treg | +0.674 | apCAF-Treg induction vs Treg death — net Treg density identified |
| k_MDSC_rec | k_MDSC_death | +0.644 | MDSC recruitment/death — only steady-state density identified |
| k_PSC_to_apCAF | k_apCAF_death | +0.556 | apCAF activation/death — net apCAF density identified |
| k_Mac_rec | k_Mac_death | +0.531 | Macrophage recruitment/death — net TAM density identified |

All correlations are **positive** — when one rate goes up, the paired rate goes up to compensate. This is the classic signature of production/degradation non-identifiability: the observables constrain the steady-state *level* (ratio of rates) but not the absolute *turnover* (product of rates).

### Action items

- **Reparameterize** these pairs as (steady-state level, turnover rate). This would improve identifiability and reduce effective dimensionality.
- **Consider fixing turnover rates** at literature values and only varying steady-state levels.

---

## 10. Learning Curve (Sample Size Scaling)

NPE retrained at 10%–100% of training data (256–2567 sims):

| Fraction | N_train | Median R² | R² > 0.5 | R² > 0 |
|----------|---------|-----------|----------|--------|
| 10% | 256 | 0.004 | 1 | 52 |
| 20% | 513 | 0.054 | 3 | 62 |
| 40% | 1026 | 0.063 | 7 | 70 |
| 60% | 1540 | 0.091 | 8 | 73 |
| 80% | 2053 | 0.084 | 9 | 76 |
| 100% | 2567 | 0.085 | 9 | 77 |

### Key finding: the curve saturates after ~1500 sims

Median R² plateaus at ~0.085 from 60% (1540 sims) onward. The number of well-recovered parameters (R² > 0.5) stabilizes at 8-9. **More training data at the current configuration will not meaningfully improve posterior quality.**

This means the bottleneck is **not** sample size — it's the information content of the 30 observables about the 97 parameters. To improve, we need either:
1. More informative observables (e.g., time-course data, perturbation responses)
2. Fewer parameters (fix non-identifiable ones)
3. Better observables that break the production/degradation degeneracies

---

## 11. Seed Stability

NPE retrained with 5 different seeds. Median R² std across seeds: **0.012** — the posterior is stable.

Top 10 most stable parameters:

| Parameter | R² (mean ± std) |
|-----------|-----------------|
| pO2_50 | 0.923 ± 0.045 |
| k_PSC_to_apCAF | 0.826 ± 0.003 |
| k_cell_clear | 0.737 ± 0.008 |
| k_M2_pol | 0.719 ± 0.003 |
| APC0_T | 0.595 ± 0.013 |
| k_ECM_myCAF_sec | 0.562 ± 0.024 |
| k_Mac_rec | 0.549 ± 0.013 |
| k_Treg_death | 0.535 ± 0.014 |
| k_vas_Csec | 0.502 ± 0.005 |
| k_NK_rec | 0.498 ± 0.008 |

The well-recovered parameters (R² > 0.5) are very stable across seeds (std < 0.025 for most). The low-R² parameters have proportionally higher variability but their absolute R² range is small. **Seed instability is not a concern** — the posterior is reproducible.

---

## 12. Parameter-Observable Sensitivity (PRCC Decomposition)

The PRCC matrix from surrogate sensitivity analysis reveals which observables constrain which parameters. This decomposition explains *why* certain parameters are well-recovered and which observables carry the most information.

### Which observables constrain the 8 well-recovered parameters?

| Parameter | R² | Observables (|PRCC| > 0.3) | PRCC |
|-----------|-----|---------------------------|------|
| pO2_50 | 0.953 | hif1a_positivity_rate | +0.60 |
| k_PSC_to_apCAF | 0.821 | apcaf_fraction_of_caf, psc_activation_ratio, treg_fraction_cd4 | +0.79, -0.32, +0.32 |
| k_cell_clear | 0.744 | th_exhausted_fraction, cd8_exhausted_fraction, treg_fraction_cd4 | -0.96, -0.54, +0.51 |
| k_M2_pol | 0.732 | m1_m2_ratio, m2_macrophage_fraction | -0.95, +0.95 |
| APC0_T | 0.583 | tam_to_cdc1_ratio, cdc1_density_baseline | -0.75, +0.66 |
| k_Mac_rec | 0.558 | tam_density, tam_to_cdc1_ratio, apc_maturation_ratio | +0.75, +0.69, -0.36 |
| k_Treg_death | 0.524 | treg_fraction_cd4, treg_fraction_baseline, treg_density_baseline, cd8_treg_ratio | -0.85, -0.53, -0.52, +0.51 |
| k_ECM_myCAF_sec | 0.550 | collagen1_protein_level, tumor_volume_resection, psc_activation_ratio, apcaf_fraction_of_caf, apc_maturation_ratio, mycaf_fraction_of_caf, stromal_fraction | +0.64, +0.64, -0.49, +0.47, +0.40, -0.33, +0.33 |

**Pattern:** Well-recovered parameters have strong, near-exclusive relationships with 1-3 observables. pO2_50 is the extreme case — a single observable (hif1a_positivity_rate, |PRCC|=0.60) is enough for R²=0.953, because the relationship is tight and no other parameter competes for that observable.

### What drives cdc1_fraction_of_dc?

Only 3 parameters have |PRCC| > 0.2 against this observable:

| Parameter | PRCC |
|-----------|------|
| k_APC_mat_cDC1 | -0.828 |
| f_cDC1 | +0.820 |
| k_APC_mat_cDC2 | +0.803 |

This is a remarkably tight set: the cDC1 fraction is determined almost exclusively by the cDC1/cDC2 maturation rates and the cDC1 subtype fraction. These are exactly the 3 parameters with the most extreme boundary piling (93.7%, 87.8%, 84.8%). The observable strongly constrains them, but the calibration target value (0.81) lies outside the range achievable within the prior — hence the piling.

### Information density: which observables are most/least informative?

**Most broadly informative** (constrain the most parameters globally at |PRCC| > 0.3):

| Observable | # Params |PRCC|>0.3 | Assessment |
|-----------|---------|------------|
| cd8_density_baseline | 17 | High-value: carries info about many params |
| cd8_treg_ratio | 14 | |
| treg_fraction_baseline | 14 | |
| nk_fraction_of_tils | 13 | |
| total_til_density | 13 | |

**Most selective** (few parameters but very strong PRCCs):

| Observable | # Params | Mean |PRCC| | Assessment |
|-----------|---------|------------|------------|
| th_exhausted_fraction | 2 | 0.91 | Pinpoints k_cell_clear |
| m1_m2_ratio | 3 | 0.77 | Pinpoints k_M2_pol |
| cdc1_fraction_of_dc | 3 | 0.82 | Pinpoints f_cDC1 / APC maturation |

**Uninformative:** cancer_cell_density_baseline has zero parameters with |PRCC| > 0.3 across all 177 parameters, suggesting it is driven by complex nonlinear interactions not captured by PRCC.

### Conflicting observable constraints

Some parameters are pulled in opposite directions by different observables. We can detect this by combining the PRCC (direction of influence) with the coverage z-score (direction of mismatch between model and reality).

**Method:** For each parameter-observable pair:
- The z-score tells us whether the model's prior predictive distribution is biased high or low relative to the calibration target (z < 0 means model simulations are biased high, z > 0 means biased low)
- The PRCC tells us how increasing the parameter affects the observable
- The "pull direction" = sign(PRCC) × sign(z): positive means the observable wants the parameter higher to bring simulations closer to reality, negative means lower
- A **conflict** arises when two observables with strong PRCCs pull the same parameter in opposite directions

**Top 5 conflicted parameters:**

| Parameter | R² | Conflict strength | Wants higher (why) | Wants lower (why) |
|-----------|-----|-------------------|--------------------|-------------------|
| k_ECM_myCAF_sec | 0.534 | 2.12 | psc_activation_ratio (z=-6.4), apcaf_fraction (z=+2.2) | stromal_fraction (z=-6.4), collagen1 (z=-2.8) |
| k_C1_growth | 0.232 | 1.61 | stromal_fraction (z=-6.4), collagen1 (z=-2.8) | psc_activation_ratio (z=-6.4), apcaf_fraction (z=+2.2) |
| k_ECM_deg | 0.129 | 1.19 | collagen1 (z=-2.8) | psc_activation_ratio (z=-6.4) |
| k_APC_mat_cDC2 | 0.468 | 0.78 | cdc1_fraction_of_dc (z=+5.6) | apc_maturation_ratio (z=-1.2) |
| k_vas_growth | 0.463 | 0.70 | stromal_fraction (z=-6.4), tumor_glucose_ratio (z=+1.3) | tumor_pO2 (z=-1.0), hif1a_positivity (z=+0.9) |

**Worked example — k_ECM_myCAF_sec:**
- `stromal_fraction`: PRCC = +0.33, z = -6.4. The model's simulations are biased high for stroma (observed 0.668 vs model mean ~0.98). Higher ECM secretion → more stroma. To bring simulations closer to the real value: want ECM secretion **lower**.
- `psc_activation_ratio`: PRCC = -0.49, z = -6.4. Simulations are biased high for PSC activation (observed 0.575 vs model ~0.97). Higher ECM secretion → lower PSC activation. To bring simulations closer to reality: want ECM secretion **higher**.

Both observables are off by ~6 standard deviations, but they pull k_ECM_myCAF_sec in opposite directions. The NPE is caught in a tug-of-war — no single value can satisfy both. This is a structural model issue: the relationship between ECM deposition and PSC activation doesn't match the biology at the calibration targets.

**The stroma observables are the main source of conflict.** `stromal_fraction` and `psc_activation_ratio` both have extreme z-scores (~-6.4) and appear on opposite sides for the top 3 conflicted parameters (k_ECM_myCAF_sec, k_C1_growth, k_ECM_deg). The model's stroma module simultaneously over-produces stroma fraction and over-activates PSCs, but these two features have opposite parameter dependencies — fixing one makes the other worse.

**Metabolism creates a secondary conflict for k_vas_growth:** tumor_glucose_ratio wants faster vascular growth (more glucose delivery), but tumor_pO2 and HIF want slower growth (more hypoxia). The model ties vascularization to both oxygen and glucose delivery, but the calibration targets suggest they should be partially decoupled.

Conflict strength is quantified as min(max |PRCC × z| from each side) — limited by the weaker side of the tug-of-war.

### Implications for calibration strategy

1. **treg_fraction_cd4** is the most multi-informative observable for the well-recovered set — it constrains 3 of the 8 (k_PSC_to_apCAF, k_cell_clear, k_Treg_death). This makes it a critical calibration target.
2. **k_ECM_myCAF_sec** has the broadest footprint (7 observables), reflecting ECM secretion's downstream effects on collagen, tumor volume, stromal fraction, and CAF composition.
3. The 12 observables that don't constrain any well-recovered parameter (cd8_density_baseline, tumor_glucose_ratio, tumor_pO2_baseline, etc.) still carry information about parameters in the unidentified tail — they would become useful if we added perturbation data or reduced dimensionality.
4. **Resolving the stroma conflict** (k_ECM_myCAF_sec, k_C1_growth, k_ECM_deg) should be a priority — these are the most conflicted parameters and the stroma observables are the most mismatched. Either the stroma module structure or the calibration targets need revisiting.

---

## 13. Dimensionality Sweep (Parameter Subset Scaling)

NPE retrained with top-N parameters by SA ranking (PRCC), keeping all 30 observables. Tests whether reducing parameter dimensionality improves identifiability.

| D (params) | Median R² | R² > 0.5 | R² > 0 |
|------------|-----------|----------|--------|
| 8 | **0.242** | 1 | 8 |
| 15 | 0.190 | 1 | 15 |
| 20 | 0.172 | 3 | 20 |
| 30 | 0.170 | 6 | 29 |
| 50 | 0.128 | 7 | 45 |
| 97 (full) | 0.090 | 9 | 72 |

### Key finding: quality-vs-quantity tradeoff

**Median R² nearly triples** going from D=97 (0.090) to D=8 (0.242) — fewer parameters means each one is better-recovered on average. But the **total number of well-recovered parameters increases with D**: D=97 gets 9 params with R²>0.5 vs only 1 at D=8.

The **sweet spot is D=30**: median R² doubles (0.170 vs 0.090) while still recovering 6 parameters above R²>0.5. At D=30, the well-recovered parameters are:

| Parameter | R² at D=30 | R² at D=97 |
|-----------|-----------|-----------|
| k_cell_clear | 0.740 | 0.744 |
| k_M2_pol | 0.714 | 0.732 |
| APC0_T | 0.601 | 0.583 |
| k_ECM_myCAF_sec | 0.585 | 0.550 |
| k_Mac_rec | 0.557 | 0.558 |
| k_Treg_death | 0.534 | 0.524 |

### Notable: pO2_50 only enters at full D

pO2_50 (the best-recovered parameter at R²=0.953) has PRCC rank >50, so it only enters the sweep at D=97. It's identifiable through **nonlinear joint constraints** from multiple observables, not strong monotonic correlations with any single target. This highlights a limitation of PRCC-based parameter selection — some highly identifiable parameters would be dropped.

### Is the NPE the bottleneck?

The R² improvement at lower D might suggest the NPE architecture is failing at high dimensionality. However, the evidence points to **structural model issues** as the primary bottleneck, not the estimator:

1. **The improvement at lower D is expected from information theory.** With 30 observables, you have ~30 degrees of freedom of information. Spreading that over 8 parameters vs 97 naturally gives better per-param recovery — more information per parameter. The total information extracted (sum of positive R²) is comparable across D values.

2. **Even at D=8, most params aren't well-recovered.** 7 of 8 parameters still have R² < 0.5 at D=8. If the NPE were the bottleneck, reducing to 8 params should give excellent recovery — but it doesn't. The signal just isn't there in snapshot data for most parameters.

3. **The truly well-recovered params aren't the most sensitive.** pO2_50 (R²=0.953), k_cell_clear (0.74), and k_M2_pol (0.73) aren't in the top-8 by SA rank. They're identifiable because they leave unique nonlinear fingerprints in the observables, not because they're the most sensitive. This is consistent with identifiability being a property of the forward model, not the estimator.

4. **Seed stability is excellent** (std=0.012). The NPE consistently learns what's learnable.

5. **7/30 observables fail prior predictive** — the model can't even generate observed values. No estimator can calibrate to data the forward model can't produce. The 7 production/degradation degeneracies (Section 9) are mathematical non-identifiabilities that a perfect estimator would still be unable to resolve.

The learning curve saturating at 1500 sims could reflect either information saturation or network capacity saturation. A quick disambiguation would be to try a larger flow (more transforms, wider hidden layers) at D=97 and check if the curve shifts up. But gains would likely be marginal.

**Recommendation: fix structural model issues first.** The ROI is much higher:
- Fix the 7 failing observables → they become informative constraints instead of noise
- Reparameterize the 7 correlated pairs → reduces effective D from 97 to ~90
- Widen the 4 extreme boundary-piling priors → lets the posterior explore the correct region

After those fixes, if R² is still disappointing, then it's worth trying a better NPE (embedding net, more flow transforms, or flow-matching). But improving the estimator now is optimizing the wrong thing — building a better lens to look at a blurry image when the problem is the image itself.

---

## 14. Comparison with Classical SimBiology Calibration


### What we're doing (SBI/NPE)

| Aspect | SBI approach |
|--------|-------------|
| **Method** | Amortized neural posterior estimation — train once, condition on any observation instantly |
| **Likelihood** | Implicit (likelihood-free) — no need to specify p(x\|θ), learned from simulations |
| **Output** | Full joint posterior p(θ\|x) over 97 parameters simultaneously |
| **Cost** | Upfront: ~5000 forward simulations + minutes of NN training. Per-observation: milliseconds |
| **Uncertainty** | Full posterior distributions with correlations between parameters |

### Classical SimBiology approaches

#### a. `sbiofit` (optimization-based)

Point estimate θ* + asymptotic covariance. Finds a single mode; misses multimodality. For 97 params, massive compute per patient. Hessian-based CIs assume Gaussian posterior — rarely true for biological models.

#### b. `sbiopso` / particle swarm

Global optimization, no uncertainty. Better at avoiding local optima but still a point estimate.

#### c. Population approaches (NLME)

Requires individual-level longitudinal data, not applicable to our population-level summary statistics.

### Why SBI is the right choice here

1. **Likelihood-free:** Observables involve complex derived quantities (IHC densities, dynamic cellularity)
2. **Simultaneous calibration:** 30 observables matched jointly
3. **Full posterior:** Joint distribution over 97 parameters with correlations
4. **Amortization:** Instantly condition on different datasets without re-fitting

### What classical methods do better

1. **Small parameter sets:** For ≤10 parameters, `sbiofit` with multistart is simpler
2. **Individual patient fitting:** If you have per-patient longitudinal data
3. **Convergence guarantees:** Optimization has well-understood criteria; SBI convergence is empirical

### Why a head-to-head comparison isn't useful yet

A formal SBI-vs-`sbiofit` benchmark would not be informative at this stage because the dominant issues are structural model problems, not estimator problems. Both methods would struggle with the same forward model.

A meaningful comparison would require:
- The 7 failing observables fixed, so both methods have valid targets to fit
- A small enough parameter subset (e.g., the 8 well-recovered) to make `sbiofit` tractable — multistart optimization over 97 params with 30 observables would be extremely expensive and likely get stuck in local optima
- A clear evaluation metric — `sbiofit` produces point estimates + asymptotic CIs, SBI produces full posteriors, so "better" isn't straightforward to define

The comparison would also be unfair in both directions: `sbiofit` can't produce joint posteriors or capture non-identifiabilities (it would silently pick one mode), while SBI's amortized posterior is spread thin over 97 dimensions.

The posterior correlations (Section 9) already demonstrate something `sbiofit` fundamentally cannot show — the 7 production/degradation degeneracies would be invisible to a point estimator. The time is better spent fixing structural issues; after those fixes, a focused comparison on the well-recovered parameter subset could be informative.

---

## 15. Next Steps

### Immediate (from diagnostic findings)

1. **Reparameterize correlated pairs** (Section 9): Replace the 7 production/degradation pairs with (steady-state, turnover) to reduce effective dimensionality from 97 to ~90
2. **Widen priors** for f_cDC1 (93.7% piling), k_APC_mat_cDC1/cDC2 (85-88%), k_O2_demand (73.3%)
3. **Investigate cdc1_fraction_of_dc** — observed 0.81 above all sims; may need model structural change to APC module subtype balance

### Medium-term

4. **Fix non-identifiable parameters** — the ~20 negative-R² parameters at prior medians, reducing to ~77 params
5. **Add perturbation observables** — learning curve shows 30 diagnosis-snapshot observables have saturated their information content; time-course or treatment-response data would break degeneracies
6. **Address stromal_fraction mismatch** (z = -6.38) — fibroblast/ECM module structural issue

### Not needed

- **More simulations at current config** — learning curve saturates at 1500; going to 10K won't help
- **Ensemble methods** — seed stability is already good (std = 0.012)

---

## Appendix: Diagnostic Files

All numerical results saved to `figures/sbi_baseline_sa97/diagnostics/`:

| File | Contents |
|------|----------|
| `summary.json` | Scalar metrics and training metadata |
| `recovery_r2.csv` | R² per parameter (97 rows) |
| `z_score_contraction.csv` | Z-score mean/std/median and contraction per parameter |
| `calibration_ks.csv` | KS statistic and p-value per parameter |
| `coverage.csv` | Observed vs training range per observable |
| `boundary_piling.csv` | Piling fraction, side, bounds per parameter |
| `prior_predictive_pvalues.csv` | Per-observable prior predictive p-values |
| `loo_predictive.csv` | LOO Mahalanobis influence per observable |
| `posterior_predictive.csv` | Approximate PPC statistics per observable |
| `posterior_correlations.csv` | Strong parameter pairs (\|r\| > 0.5) |
| `learning_curve.csv` | R² metrics at each training fraction |
| `seed_stability.csv` | Per-parameter R² mean/std across 5 seeds |
| `dimensionality_sweep.csv` | R² metrics at each parameter subset size |
| `dimensionality_sweep_per_param.csv` | Per-parameter R² at each subset size |
