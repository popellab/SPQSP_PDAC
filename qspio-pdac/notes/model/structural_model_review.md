# Structural Model Review: Observable Verification and Mechanism Brainstorm

**Date:** 2026-02-23
**Context:** Follow-up to NPE diagnostics report. Reviewed observable code (calibration target YAMLs), model module source (APC_module.m, fibroblast_module_enhanced.m, macrophage_module.m), and calibration target provenance to identify data issues vs structural model issues.

---

## 1. Observable Code Issues

### 1.1 cdc1_fraction_of_dc — ~~likely mapping error~~ REMOVED (redundant)

**Observable code** (`cdc1_fraction_of_dc_PDAC_deriv001.yaml`):

```python
cdc1 = species_dict['V_T.cDC1']   # immature only
cdc2 = species_dict['V_T.cDC2']   # immature only
frac = cdc1 / (cdc1 + cdc2)
```

**Data source:** Plesca 2022 (n=40 PDAC resections) measures **CLEC9A+** (cDC1) and **CD1c+CLEC10A+** (cDC2) by multiplex IHC. These are lineage markers that label both immature and mature cells of each subtype. The correct formula should be:

```python
(V_T.cDC1 + V_T.mcDC1) / (V_T.cDC1 + V_T.mcDC1 + V_T.cDC2 + V_T.mcDC2)
```

**Why it matters:** The boundary piling shows the model wants slow cDC1 maturation and fast cDC2 maturation. This means:

- Immature pool: lots of cDC1, few cDC2 → high fraction (what the code measures)
- Mature pool: few mcDC1, lots of mcDC2 → low fraction
- Including mature DCs could substantially change the ratio

The immature-only formula may be artificially inflating what the model *can* produce. With the correct formula, the gap could get larger or the boundary piling pattern could shift. Either way, this is a mapping error that needs fixing before drawing structural conclusions about the APC module.

**Resolution:** Target removed (moved to `excluded/`) and replaced with `cdc2_density_baseline` (Plesca 2022, 2.98 ± 0.597 cells/mm²). The fraction was redundant with the existing `cdc1_density_baseline` — if cdc1 density calibrates fine but the fraction fails, the issue is cDC2 density in the model. Constraining cDC2 density directly isolates the problem and avoids a derived ratio that compounds uncertainty.

### 1.2 tam_to_cdc1_ratio — ~~weak data source~~ REMOVED (redundant)

**Observable code** (`tam_to_cdc1_ratio_PDAC_deriv001.yaml`):

```python
tam_total = mac_m1 + mac_m2
cdc1_total = cdc1_immature + cdc1_mature   # V_T.cDC1 + V_T.mcDC1
ratio = tam_total / cdc1_total
```

**Data source:** Breaza 2025 (n=22 of 56, immunotype A only). Semi-quantitative: "20% CD68+ macrophages, 5% dendritic cells" in the inflammatory infiltrate, giving a ratio of 4. Key concerns:

1. **DC marker mismatch:** CD1a is not cDC1-specific — it labels Langerhans-like DCs and potentially some cDC2s. The experimental denominator is broader than the model's `cDC1 + mcDC1`, making the experimental ratio appear lower than the model-equivalent.
2. **No dispersion data:** sigma_log=0.5 is entirely assumed (CV ~50%).
3. **Selection bias:** Only the "hot" immunotype A subset is used (39% of cases). The model simulates the full spectrum.
4. **Sample size:** n=22 patients, no per-patient statistics.

**Resolution:** Target removed (moved to `excluded/`). The ratio is redundant because both `tam_density` (Sivagnanam 2021) and `cdc1_density_baseline` (Plesca 2022) are already independent calibration targets. Including the ratio would double-count the same constraints and distort the likelihood surface.

### 1.3 psc_activation_ratio — ~~modest denominator mismatch~~ FIXED (numerator + denominator)

**Observable code** (`psc_activation_ratio_PDAC_deriv001.yaml`):

```python
activated = iCAF + myCAF
total_fibroblasts = activated + apCAF + qPSC
ratio = activated / total_fibroblasts
```

**Data source:** Drifka 2016 (n=114 patients, 356 cores). Measures alpha-SMA+ area fraction by color deconvolution (DAB/(DAB+hematoxylin)) in periductal stromal ROIs.

**Denominator issue:** The experimental denominator includes all stained cells in stromal ROIs (fibroblasts + immune cells + endothelium), not just fibroblasts. This **dilutes** the measured activation fraction relative to the model's pure fibroblast ratio. The YAML notes this and estimates the bias as "modest (~10-20%)" because fibroblasts dominate periductal stroma.

Correcting for this: observed 0.572 / 0.85 (if ~15% non-fibroblast dilution) ≈ 0.67 true activation ratio. Still far below the model's ~0.97, but the gap narrows from 0.40 to 0.30. Not enough to resolve the discrepancy by itself.

**Also:** Only point estimates reported (0.572 in core, 0.503 at edge) — the CV=0.2 used for the calibration target distribution is entirely assumed.

**FIXED:** Updated observable to use myCAF / (iCAF + myCAF + apCAF + qPSC). The original numerator (iCAF + myCAF) incorrectly included iCAFs, which are FAP+/α-SMA-low and would not stain in Drifka's α-SMA assay. The corrected observable measures the myCAF fraction of total fibroblasts, which cross-validates against Ogawa 2021 ACTA2-dominant fibroblast fraction (71.7% of CAFs). The model should now target ~0.57 for myCAF/(total fibroblasts), not ~0.57 for (iCAF+myCAF)/(total fibroblasts) — a fundamentally different constraint that allows both high PSC activation AND moderate myCAF fraction.

### 1.4 stromal_fraction and cancer_cell_density — internally consistent, formulas correct

These two targets tell the same story from opposite directions:

| Target | Observed | Model | Implies cellularity |
|--------|----------|-------|---------------------|
| stromal_fraction | 0.668 | ~0.98 | ~0.33 (obs) vs ~0.02 (model) |
| cancer_cell_density | 996 cells/mm² | ~72 cells/mm² | ~0.23 (obs) vs ~0.02 (model) |

Both use volume-based formulas with consistent cell diameter constants (cancer 17 µm, CAF 15 µm, macrophage 20 µm, lymphocyte 7 µm) and ECM tissue density (150 mg/mL). The formulas are correct. The problem is real: the model produces too much stroma (mainly ECM) and too few cancer cells relative to the observed tissue composition.

### 1.5 collagen1_protein_level — data is solid, model overproduces

**Data source:** Xu 2017 (n=8). SHG/TPEF measured collagen pixel density: 2.09 ± 0.26 (normal) vs 20.85 ± 4.50 (PDAC), giving ~10x fold change. Small sample but the measurement technique (label-free multiphoton) is direct.

**Observable code:** Computes `V_T.ECM / (tumor_vol * tissue_density * normal_collagen_fraction)` using reference constants (normal pancreas collagen 5 µg/mg, tumor volume 10 mL, tissue density 1.04 g/cm³). Model produces ~230x fold change vs observed ~10x. This is a 23-fold overestimate of ECM accumulation. Confirms the fibroblast module is too aggressive in ECM production.

### 1.6 apcaf_fraction_of_caf — wide uncertainty bracket, model at lower boundary

**Data source:** Elyada 2019 (n=6 tumors, 962 CAFs) reports 20.9% at transcript level (scRNA-seq), but the YAML correctly notes that protein-level apCAFs are ~2-5% (Ogawa 2021, n=215). The calibration target uses a log-normal spanning 2% to 21% (median ~6.4%). This is well-constructed uncertainty propagation.

The model underproduces apCAFs. Given that apCAF activation is constitutive (`k_PSC_to_apCAF * V_T.qPSC`), the issue is likely coupled to the PSC activation problem: if nearly all qPSCs are activated to iCAF/myCAF (the dominant TGFb+PDGF-driven pathways), there are too few qPSC remaining to feed the constitutive apCAF pathway.

---

## 2. Structural Model Analysis

### 2.1 Stroma module (fibroblast_module_enhanced.m)

**Current architecture:**

| Process | Rate expression | Notes |
|---------|----------------|-------|
| qPSC → myCAF | `k_PSC_to_myCAF * qPSC * H_TGFb * H_PDGF` | TGFb + PDGF required |
| qPSC → iCAF | `k_PSC_to_iCAF * qPSC * (1-H_TGFb) * H_PDGF` | No TGFb, PDGF required |
| qPSC → apCAF | `k_PSC_to_apCAF * qPSC` | Constitutive |
| iCAF → myCAF | `k_iCAF_to_myCAF * iCAF * H_TGFb` | TGFb-driven |
| myCAF → iCAF | `k_myCAF_to_iCAF * myCAF * (1-H_TGFb)` | TGFb withdrawal |
| ECM secretion (myCAF) | `k_ECM_myCAF_sec * ((ECM_max-ECM_level)/ECM_max) * myCAF * (0.3+0.7*H_TGFb)` | Saturating, TGFb-enhanced |
| ECM secretion (qPSC) | `k_ECM_qpsc_sec * ((ECM_max-ECM_level)/ECM_max) * qPSC * (0.1+0.9*H_TGFb)` | Low basal |
| ECM degradation | `k_ECM_deg * V_T.ECM` | First-order only |
| Carrying capacity | `C_max = (K + k_stroma_support * aPSC_total) * (1 - ECM_level/(ECM_level + ECM_50_Kmax))` | CAFs increase, ECM decreases |
| qPSC supply | `k_CAF_const` (constitutive) | Boundary piles upper |

**Root cause analysis:**

The model produces psc_activation ~0.97 because activation is fast and partially irreversible — qPSC → myCAF/iCAF is driven by PDGF (which is abundant because myCAFs proliferate and secrete more PDGF in a positive feedback loop), and while myCAF ↔ iCAF interconversion exists, there is no CAF → qPSC reversion pathway. The constitutive supply `k_CAF_const` only partially compensates.

ECM is overproduced because myCAF secretion is strong, the only degradation is first-order (no enzyme-mediated degradation), and the ECM_max saturation is apparently not restrictive enough at the current parameter values.

The stromal_fraction is ~0.98 because ECM volume dominates total tissue volume. At 150 mg/mL ECM tissue density, even modest ECM mass translates to large volume.

**The stroma tug-of-war (from NPE diagnostics):** k_ECM_myCAF_sec is pulled in opposite directions because:
- Reducing ECM secretion → lower stromal_fraction and collagen (good) but lower stromal fraction also reduces the ECM-mediated PSC activation feedback, potentially lowering psc_activation (bad — it's already too high and this goes the wrong direction)
- Increasing ECM secretion → the psc_activation_ratio denominator math doesn't directly benefit

The core issue is that the model lacks independent knobs for "how much ECM" vs "how many activated CAFs." Both are controlled by the same upstream signal (TGFb + PDGF → PSC activation → myCAF → ECM secretion). The model needs a structural change to decouple these.

### 2.2 APC module (APC_module.m)

**Current architecture:**

| Process | Rate expression | Notes |
|---------|----------------|-------|
| cDC1 recruitment | `k_APC_death * f_cDC1 * APC0_T * V_T * (1-H_ECM_APC_mig) * (1+H_XCL1_cDC1)` | XCL1 boost from NK |
| cDC2 recruitment | `k_APC_death * (1-f_cDC1) * APC0_T * V_T * (1-H_ECM_APC_mig)` | No extra boost |
| cDC1/cDC2 death | Same `k_APC_death` for both | No subtype specificity |
| cDC1 maturation | `k_APC_mat_cDC1 * cDC1 * (1-(1-H_DAMP)*(1-H_IL12)) * (1-H_IL10) * (1-H_TGFb_APC)` | DAMP/IL12 OR, IL10/TGFb inhibit |
| cDC2 maturation | `k_APC_mat_cDC2 * cDC2 * (same maturation logic)` | Same signals, different rate |
| Migration (both) | `k_APC_mig * mcDC * (1-H_ECM_APC_mig)` | Same rate, both types |

**Key constraint:** The steady-state immature cDC1 fraction in tumor depends on:
- f_cDC1 (recruitment split)
- k_APC_mat_cDC1 vs k_APC_mat_cDC2 (differential maturation)
- k_APC_death (shared)
- H_XCL1_cDC1 (cDC1-specific boost)

The model has some asymmetry (XCL1 boost, different maturation rates) but shares death rates and migration rates between subtypes. This may be insufficient to reach cDC1 fractions of ~0.81 among all DCs (immature + mature).

---

## 3. Candidate Structural Mechanisms

### 3.1 Stroma: Missing active ECM degradation (MMPs)

**Biological basis:** Matrix metalloproteinases (MMP-2, MMP-9, MMP-14) actively degrade ECM. Their activity is regulated by:
- M1 macrophages (secrete MMP-9, MMP-12)
- Cancer cells (membrane-type MMP-14)
- TIMP-1/2 inhibitors from M2 macrophages and myCAFs

**Proposed mechanism:** Add MMP species and enzyme-mediated ECM degradation:

```
ECM degradation rate = k_ECM_deg * ECM + k_MMP_deg * ECM * MMP / (MMP + MMP_50)
MMP production = k_MMP_M1 * Mac_M1 + k_MMP_C * C1
MMP inhibition by TIMPs (from M2, myCAF)
```

**What it would fix:**
- Decouple ECM accumulation from CAF activation — you can have lots of activated CAFs but moderate ECM if MMPs are active
- Create immune-stroma feedback: more M1 → more MMP → less ECM → less immune exclusion
- Provide a treatment-response axis: immunotherapy → M1 polarization → ECM degradation
- Could reduce collagen fold-change from 230x toward 10x without changing CAF dynamics

**What it wouldn't fix:** psc_activation_ratio (that's a CAF dynamics problem, not an ECM problem).

**Testable prediction:** Inverse correlation between M1 macrophage fraction and collagen density across patients.

### 3.2 ~~Stroma: CAF deactivation / quiescent reversion~~ — REMOVED

**Status:** Removed from candidate mechanisms. CAF-to-qPSC reversion is primarily an in vitro observation (VDR agonist: Sherman 2014; mechanical unloading: Chronopoulos 2016) and has not been clearly demonstrated in vivo in established human PDAC. The Ogawa 2021 data (n=215) shows near-complete PSC activation with apCAFs "extremely limited in number," consistent with activation being effectively irreversible in the disease context. The psc_activation_ratio discrepancy (Drifka 0.575 vs expected >0.90) is more likely explained by the denominator dilution issue (Section 1.3) than by genuine reversion.

### 3.3 Stroma: Invert stroma-cancer carrying capacity relationship

**Biological basis:** Dense desmoplasia physically constrains tumor cellularity. High ECM → high interstitial pressure → reduced cancer cell proliferation and survival. The current model has `C_max = (K + k_stroma_support * aPSC_total) * (1 - ECM/(ECM + ECM_50_Kmax))`. The `k_stroma_support * aPSC_total` term *increases* carrying capacity with CAFs, while the ECM term *decreases* it. These are in tension.

**Proposed change:** Remove or reduce `k_stroma_support` so that ECM-mediated compression dominates. Alternatively, add a direct ECM-mediated growth inhibition term to the cancer growth rate.

**What it would fix:**
- stromal_fraction and cancer_cell_density: less cancer cell volume relative to stroma
- Would create a natural negative feedback: more ECM → fewer cancer cells → less PDGF → less CAF activation

**What it wouldn't fix:** psc_activation_ratio or collagen fold-change (those are upstream of carrying capacity).

**Testable prediction:** Strong negative correlation between tumor cellularity (PanCK+ density) and stromal density (alpha-SMA+ area) across patients. Testable in Sivagnanam 2021 (n=113).

### 3.4 DC: Differential subtype survival in the TME

**Biological basis:**
- cDC2s are more susceptible to TGFb-mediated apoptosis (Ghiringhelli 2005)
- cDC2s are more susceptible to IL-10-mediated functional suppression
- cDC1s maintain BATF3 expression and are more resistant to TME suppression
- The PDAC TME is rich in TGFb and IL-10, creating a selectively hostile environment for cDC2s

**Proposed mechanism:** Subtype-specific death rates:

```
cDC1 death: k_APC_death * cDC1                                  % baseline
cDC2 death: k_APC_death * cDC2 * (1 + f_TGFb_cDC2_death * H_TGFb)   % TGFb-enhanced
```

Or more simply, separate `k_cDC1_death` and `k_cDC2_death` parameters.

**What it would fix:**
- cdc1_fraction_of_dc: higher cDC1 fraction if cDC2 dies faster
- tam_to_cdc1_ratio: if total DCs shift toward cDC1, the ratio changes

**What it wouldn't fix:** Stroma issues.

**Testable prediction:** cDC1/cDC2 ratio correlates positively with TGFb expression across PDAC specimens. Higher TGFb → fewer cDC2 → higher cDC1 fraction.

### 3.5 DC: Monocyte-derived DC contamination of experimental cDC2 pool

**Biological basis:** CD1c (used by Plesca 2022 for cDC2) also labels monocyte-derived DCs (moDCs), which are abundant in PDAC tumors. moDCs are not modeled as a separate species. If moDCs inflate the experimental cDC2 count, the "true" conventional cDC2 count is lower and the actual cDC1 fraction is even higher than the measured 0.81.

This is a data interpretation issue rather than a model structural issue. It implies the calibration target may be approximately correct and the model just needs wider priors or a minor structural change.

**Resolution path:** Look for PDAC DC studies using more specific markers that exclude moDCs:
- CD5 positivity distinguishes bona fide cDC2 from moDC (Villani 2017)
- FceRI and CD14 identify moDCs
- CD88/CD89 (FACS-based) separate cDC2 from moDC

### 3.6 DC: Additional cDC1-specific recruitment signals

**Biological basis:** Beyond XCL1 (already modeled), cDC1 recruitment to tumors is promoted by:
- FLT3L (from NK cells and activated T cells)
- CCL4 (from NK cells)
- CCL5 (from activated CD8 T cells and some cancer cells; attracts cDC1 via CCR5)

The model already has CCL5 (from iCAFs in fibroblast_module_enhanced.m), but this drives Treg recruitment via CCL5_Treg_module.m. If CCL5 also recruits cDC1, it would increase the cDC1 pool.

**Proposed mechanism:** Add CCL5-mediated cDC1 recruitment (separate from the existing Treg recruitment):

```
cDC1 recruitment += k_CCL5_cDC1_rec * V_T * V_T.CCL5 / (CCL5_50_cDC1 + V_T.CCL5)
```

**What it would fix:** cdc1_fraction_of_dc (more cDC1 recruited) and tam_to_cdc1_ratio (better TAM/cDC1 balance).

**What it wouldn't fix:** Stroma issues.

**Consideration:** This adds parameters and complexity. Only pursue if the simpler fixes (correct observable, differential survival) are insufficient.

---

## 4. Diagnostics to Differentiate Mechanisms

| Question | Diagnostic approach | Data source |
|----------|-------------------|-------------|
| Is cdc1_fraction_of_dc a mapping error? | Fix observable to include mature DCs (mcDC1, mcDC2), re-run prior predictive | Code change only |
| Is tam_to_cdc1_ratio reliable? | Cross-validate against Plesca 2022 or Sivagnanam 2021 (same cohort, better markers) | Published supplementary data |
| Does MMP degradation matter for ECM? | Correlate M1 fraction with collagen density across patients | Sivagnanam 2021 multiplex IHC (n=113) |
| Is PSC activation truly irreversible? | Measure qPSC fraction at biopsy vs resection in matched patients | Paired biopsy-resection cohorts |
| Does stroma constrain or support cancer growth? | Correlate PanCK+ density with alpha-SMA+ area across patients | Sivagnanam 2021 (already have densities for both) |
| Is differential DC death the mechanism? | Correlate cDC1/cDC2 ratio with TGFb expression across specimens | Multi-parameter IHC or spatial transcriptomics |
| Are moDCs contaminating the cDC2 measurement? | Find PDAC studies using CD5 or FceRI to separate cDC2 from moDC | Literature search (Villani 2017 panel in PDAC) |

---

## 5. Additional Data to Collect

### Highest priority (resolves multiple questions)

1. **Plesca 2022 full supplementary data** — Already sourced for cDC1/cDC2 densities. If they also report macrophage densities (CD68+, CD163+), we get a properly-measured tam_to_cdc1_ratio from the same cohort with cDC1-specific markers (CLEC9A). Would either validate or invalidate the Breaza 2025 target.

2. **Sivagnanam 2021 Supplementary Table S5** — Already sourced for cancer_cell_density (PanCK+). Their n=113 multiplex IHC dataset likely has CD68 (TAMs), CD163 (M2), alpha-SMA (activated stroma), and multiple T cell markers. Could:
   - Cross-validate tam_to_cdc1_ratio
   - Test M1 vs collagen correlation (MMP hypothesis)
   - Test cellularity vs stromal density correlation (carrying capacity hypothesis)
   - Provide additional calibration targets from the same cohort

### Medium priority

3. **Ogawa 2021 (CCR)** — n=215 PDAC specimens with multiplex IHC for three fibroblast subtypes and collagen. Protein-level CAF subtype fractions would anchor the apCAF and myCAF/iCAF targets. Already cited as secondary source for apcaf_fraction_of_caf.

4. **Paired biopsy-resection specimens** — Any study measuring stromal activation at EUS biopsy vs resection in matched patients would test the CAF reversion hypothesis. If activation ratio is similar at both timepoints, reversion is minimal. If it's lower at resection than expected from pure activation kinetics, reversion is occurring.

### Lower priority (interesting, not blocking)

5. **MMP expression in PDAC** — MMP-2/9/14 levels correlated with collagen density. Would directly test whether enzyme-mediated degradation is a relevant pathway to model.

6. **cDC subtyping with moDC exclusion** — Studies using the Villani 2017 DC classification in PDAC. Would resolve whether moDC contamination explains the high cDC1 fraction.

---

## 6. Recommended Priority Order

### Immediate (code/data verification)

1. ~~**Fix cdc1_fraction_of_dc observable**~~ **REPLACED** — removed fraction target (redundant), added `cdc2_density_baseline` to directly constrain the cDC2 axis.

2. **Extract Sivagnanam 2021 Table S5** — pull all cell density columns to cross-validate existing targets and test correlations (M1 vs collagen, cellularity vs stroma).

3. **Search Plesca 2022 for macrophage data** — check whether the paper reports CD68+ densities alongside cDC1/cDC2 for a better tam_to_cdc1_ratio estimate.

### Short-term (structural model changes, in priority order)

4. **Add IL-6/STAT3 autocrine feedback on iCAF** — closes the missing positive feedback loop for iCAF maintenance (Biffi 2019). Lowest complexity change (2 params, modify 2 existing rates). See Section 7.7 for details.

5. **Add iCAF-mediated ECM degradation (FAP/MMP)** — most targeted fix for collagen fold-change and stromal_fraction. Pairs with #4 to create the complete iCAF remodeling axis. Moderate complexity (new reaction, 2-3 new parameters). See Section 7.2.

6. **Re-evaluate stroma-cancer carrying capacity** — test sensitivity of C_max to `k_stroma_support` parameter. May only need a parameter change, not a structural change.

~~Add CAF deactivation pathways~~ — removed. In vitro phenomenon (VDR agonist, mechanical unloading), not demonstrated in vivo in established PDAC. See Section 3.2.

### Medium-term (after re-running diagnostics)

7. **Add differential DC death rates** — only if cdc1_fraction is still problematic after fixing the observable mapping.

8. **Consider moDC as a third DC subtype** — only if data confirms moDC contamination is significant and the model needs to represent them explicitly.

9. **Add perturbation observables** — time-course or treatment-response data to break the 7 production/degradation degeneracies identified in posterior correlations.

---

## 7. Ogawa 2021 Deep Review: Stromal Heterogeneity and Model Implications

**Paper:** Ogawa Y et al. "Three Distinct Stroma Types in Human Pancreatic Cancer Identified by Image Analysis of Fibroblast Subpopulations and Collagen." *Clinical Cancer Research* 27(1):107-119, 2021. DOI: 10.1158/1078-0432.CCR-20-2298

**Why this paper matters:** n=215 treatment-naive PDACs with pixel-level multiplex IHC for fibroblast subtypes + collagen, linked to transcriptomics, survival, and CD8+ infiltration. Directly addresses the stroma tug-of-war identified in Section 2.1 and provides quantitative data for multiple candidate mechanisms from Section 3.

### 7.1 Key quantitative findings

**Desmoplastic composition (n=215, % of desmoplastic area):**

| Element | Mean | Median | IQR | Model species |
|---------|------|--------|-----|---------------|
| Collagen | 53.4% | 54.1% | 45.6–62.2% | V_T.ECM |
| ACTA2-dominant fibroblasts | 33.4% | 30.6% | 24.2–41.1% | V_T.myCAF |
| FAP-dominant fibroblasts | 13.0% | 12.1% | 8.2–17.4% | V_T.iCAF |
| ACTA2/FAP overlap | 0.2% | 0.1% | 0.05–0.2% | — |

**Noncancerous pancreas reference (n=147):**

| Element | Median |
|---------|--------|
| Collagen | 13.4% |
| ACTA2-dominant fibroblasts | 7.0% |
| FAP-dominant fibroblasts | 0.1% |

**CAF subtype fractions (fibroblast-only denominator):**

| Subtype | Fraction | Cross-validation |
|---------|----------|------------------|
| myCAF (ACTA2-dominant) | 71.7% | — new target |
| iCAF (FAP-dominant) | 27.9% | — new target |
| apCAF | ~0.4% (protein level, "extremely limited") | vs Elyada 2019 transcript-level 20.9% |

**Tumor cellularity:** 73% of tumors have cellularity ≤30%, implying stromal fraction ≥70%. Consistent with our stromal_fraction target of 0.668 (Bengtsson 2026).

**Three stroma types:**

| Type | n (%) | Defining feature | Survival |
|------|-------|-----------------|----------|
| C-stroma | 55 (25.6%) | Collagen-rich, low fibroblast activity | Better (HR 0.57, 95% CI 0.33–0.99) |
| F-stroma | 95 (44.2%) | FAP-dominant, CD8-excluding | Reference |
| A-stroma | 65 (30.2%) | ACTA2-dominant, diffuse fibroblasts | Worse (HR 1.65, 95% CI 1.06–2.58) |

### 7.2 Evidence for MMP-mediated ECM degradation (supports Section 3.1)

This is the single most important finding for the structural model. The F-stroma transcriptome signature (Fig. 3B, 295 upregulated genes) includes:

> **Upregulated in F-stroma:** ADAM12, ADAMTS12, CD14, COL1A2, COL3A1, COL5A1, COL6A1, COL6A2, COL6A3, COL8A1, COL10A1, COL11A1, COL12A1, FAP, FN1, LOX, **MMP2, MMP9, MMP11, MMP14**, PDPN, POSTN, etc.
>
> **Downregulated in F-stroma:** SERPINA3, SNORA11B, etc.

And in the C-stroma signature (406 downregulated genes):

> **Downregulated in C-stroma:** ACTA2, ADAM22, ADAM32, ADAMTS12, COL1A1, COL1A2, COL3A1, COL5A1, COL6A1, COL6A3, COL8A1, COL10A1, COL11A1, COL12A1, FAP, LOX, **MMP2, MMP9, MMP14**, PDPN, POSTN, LOX, SNAI2, SPARC, TIMP3, etc.

**Interpretation for the model:**

1. **MMP expression is stroma-type dependent.** F-stroma (FAP/iCAF-dominant) strongly upregulates MMP2, MMP9, MMP11, MMP14. C-stroma (collagen-rich) downregulates the same MMPs. This is exactly the pattern predicted by the "missing MMP degradation" hypothesis: iCAF-rich tumors have active ECM remodeling, while collagen-rich tumors have quiescent stroma with slow turnover.

2. **FAP itself is a collagenase.** The Discussion notes: "the membrane glycoprotein FAP can act as a peptidase/proteinase to degrade collagen proteins, suggesting opposite roles of ACTA2-dominant fibroblasts and FAP-dominant fibroblasts in collagen remodeling." This is mechanistically important — iCAFs don't just produce cytokines, they actively degrade ECM. The current model has no iCAF-mediated ECM degradation.

3. **LOX is co-regulated with collagen genes.** LOX (lysyl oxidase) crosslinks collagen and is upregulated in F-stroma and downregulated in C-stroma. This suggests collagen crosslinking (stabilization) and MMP degradation coexist in F-stroma, creating a dynamic remodeling state rather than simple accumulation.

4. **SERPINA3 is downregulated in F-stroma.** SERPINA3 (alpha-1-antichymotrypsin) is a serine protease inhibitor. Its downregulation in F-stroma further supports active proteolysis. In C-stroma, TIMP3 (tissue inhibitor of MMPs) is downregulated, suggesting low MMP activity overall rather than active inhibition.

**Implication for Section 3.1 (MMP mechanism):** The transcriptomic evidence strongly supports adding iCAF/FAP-mediated ECM degradation rather than (or in addition to) M1 macrophage-mediated MMP activity. The revised mechanism should be:

```
ECM degradation = k_ECM_deg * ECM                        % passive (existing)
                + k_MMP_iCAF * ECM * iCAF / (iCAF + K_iCAF_MMP)  % iCAF/FAP-mediated
                + k_MMP_M1 * ECM * Mac_M1 / (Mac_M1 + K_M1_MMP)  % M1-mediated (optional)
```

This creates the key decoupling mechanism: high iCAF activity → both cytokine secretion AND ECM degradation, preventing ECM runaway even with active stroma. The balance between myCAF ECM production and iCAF ECM degradation determines the stroma type (C vs F vs A).

### 7.3 Evidence for collagen-fibroblast decoupling (challenges current model)

A critical finding is the **absence of correlation between collagen and ACTA2-dominant fibroblasts in PDAC**:

> "No significant inverse correlation was observed between CD8^TC or CD8^TM and the proportion of collagen, FAP-dominant fibroblasts, or ACTA2-dominant fibroblasts (all r > −0.11; Supplementary Table S5)."

And from the noncancerous pancreas comparison:

- In **normal pancreas**: collagen and ACTA2-fib are positively correlated (r=0.54, inferred from the paper's statement that normal pancreas shows coordinate regulation)
- In **PDAC**: collagen and ACTA2-fib are NOT correlated (r = −0.11, P = 0.13)

**Implication for the model:** The current fibroblast module ties ECM production directly to myCAF activity (`k_ECM_myCAF_sec * myCAF * (0.3+0.7*H_TGFb)`). This hard coupling means the model cannot produce the observed decorrelation. The model structurally assumes "more myCAFs → more collagen," but Ogawa shows this relationship breaks down in PDAC.

This decorrelation could arise from:
1. **iCAF-mediated MMP degradation** (Section 7.2) — iCAFs counterbalance myCAF ECM production
2. **Collagen crosslinking and maturation** — old collagen persists even after the producing myCAFs die or deactivate
3. **ECM from non-fibroblast sources** — cancer cells and macrophages also contribute collagen in some contexts

Of these, mechanism (1) is the most supported by the transcriptomic data and the most tractable to implement.

### 7.4 FAP-dominant fibroblasts restrict CD8 infiltration (testable model prediction)

**Key correlation (Fig. 5C):** FAP-dominant fibroblast proportion inversely correlates with CD8^TC/CD8^TM ratio (r = −0.27, P < 0.001). ACTA2-dominant fibroblasts show NO correlation (r = −0.06, P = 0.40). Collagen shows a weak POSITIVE correlation (r = 0.19, P = 0.006).

**Current model mechanism:** CD8 exclusion is mediated by CXCL12 (secreted by myCAFs in the fibroblast module) and ECM (via `H_ECM_APC_mig` affecting APC migration and `H_CXCL12_Texcl` affecting T cell infiltration). The model attributes exclusion primarily to myCAFs (CXCL12) and ECM (physical barrier), NOT to iCAFs.

**The data says the opposite:** iCAFs (FAP-dominant) drive CD8 exclusion, while myCAFs (ACTA2-dominant) and collagen do not. This is a significant model-data conflict.

**Possible explanations:**
1. **CXCL12 may come from iCAFs, not myCAFs.** Some literature supports FAP+ fibroblasts as the main CXCL12 source. If so, the CXCL12 secretion reaction should be moved from myCAF to iCAF in the fibroblast module.
2. **iCAFs recruit immunosuppressive cells** (neutrophils, MDSCs) that indirectly exclude CD8+ T cells. The paper shows neutrophil infiltration is highest in F-stroma (P = 0.005, Fig. 5F).
3. **FAP itself may mediate exclusion** through ECM remodeling that creates a physical barrier distinct from bulk collagen density.

**Action item:** Check the CXCL12 secretion assignment in `fibroblast_module_enhanced.m`. If it's assigned to myCAF, consider reassigning to iCAF or splitting between both subtypes. This is a low-complexity change with potentially high impact on the CD8 infiltration axis.

### 7.5 apCAF quantification: protein vs transcript discrepancy resolved

Ogawa reports CD74+ stromal cells (the apCAF marker) are "extremely limited in number" at the protein level, with CD74-expressing stromal cells coexpressing FAP. No quantitative percentage is given, but the language strongly suggests <1% of total CAFs.

This resolves the Elyada 2019 transcript-protein discrepancy:
- **Transcript level (scRNA-seq):** 20.9% of CAFs in KPC mouse tumors (Elyada 2019)
- **Protein level (IHC):** <1% in human PDAC (Ogawa 2021, n=215)

Our calibration target (apcaf_fraction_of_caf) uses a log-normal spanning 2–21% with median 6.4%. Given the Ogawa protein-level data, the lower end of this range (~2%) is more appropriate. Consider tightening the target to median ~2%, CI95 [0.5%, 8%], using Ogawa as a secondary source to down-weight the Elyada transcript estimate.

### 7.6 Stroma type heterogeneity: implications for population modeling

The three stroma types have distinct survival signatures (Fig. 4A, log-rank P = 0.001):

| Comparison | HR | 95% CI | P |
|-----------|-----|--------|---|
| C-stroma vs F-stroma | 0.57 | 0.33–0.99 | 0.046 |
| A-stroma vs F-stroma | 1.65 | 1.06–2.58 | 0.026 |

In the multivariable model (Fig. 4B), stroma type remains significant after adjusting for tumor grade, cellularity, residual tumor, CD8+ density, tertiary lymphoid structures, and disease stage.

**Implication for SBI:** The current model generates a single parameter set per virtual patient, producing a single stroma "type." If the stroma module can reproduce the C/F/A spectrum through parameter variation alone (varying the balance of myCAF vs iCAF activation rates, MMP activity, and ECM secretion), this is fine — the NPE should learn the mixture. But if the model is structurally constrained to a single stroma state (e.g., always high ECM + high myCAF), it cannot represent C-stroma patients (high collagen, low fibroblast activity).

Adding iCAF-mediated MMP degradation (Section 7.2) would help: high iCAF → active remodeling (F-stroma-like), low iCAF + high myCAF → collagen accumulation (C-stroma-like), high myCAF + diffuse → A-stroma-like. The key is that the model needs at least two independent stroma-shaping axes, not just one.

### 7.7 Missing IL-6/JAK/STAT3 autocrine feedback on iCAF maintenance

**Biological basis (Biffi et al. 2019):** The iCAF phenotype is not simply the "absence of TGFb." It is actively driven and maintained by IL-1α/IL-6/JAK/STAT3 signaling:

1. Cancer cells secrete IL-1α → activates NF-κB in PSCs → LIF/IL-6 secretion
2. IL-6 (autocrine) and LIF (paracrine from cancer cells) → JAK/STAT3 activation in iCAFs
3. JAK/STAT3 → maintains iCAF transcriptional program (inflammatory cytokine secretion, FAP expression)
4. TGFb antagonizes this by downregulating IL-1R1 and activating the competing myCAF (SMAD2/3) program

The two pathways are mutually antagonistic: TGFb → SMAD2/3 → myCAF vs IL-1α/IL-6 → JAK/STAT3 → iCAF. This is the mechanistic basis for the Ogawa stroma types — F-stroma (JAK/STAT3 wins) vs A-stroma (SMAD wins).

**What the model currently implements:**

```matlab
% qPSC → iCAF (line 384): only "not-TGFb + PDGF"
k_PSC_to_iCAF * qPSC * (1 - H_TGFb) * H_PDGF

% myCAF → iCAF interconversion (line 401): only "not-TGFb"
k_myCAF_to_iCAF * myCAF * (1 - H_TGFb)

% iCAF → IL-6 secretion (line 515): one-directional, no feedback
k_IL6_sec_iCAF * iCAF
```

IL-6 is already a model species (`V_T.IL6`) with secretion by iCAFs, degradation, and downstream Hill functions (`H_IL6_MDSC`, `H_IL6_M2`). But it has **no feedback onto iCAF activation or maintenance**. The iCAF phenotype is purely a passive "not-TGFb" default.

**Consequence of the missing feedback:**
- iCAFs are structurally disadvantaged: TGFb (abundant in PDAC TME) constantly pushes toward myCAF, with no counterbalancing positive drive toward iCAF
- The model cannot produce stable iCAF-dominated states (F-stroma), only transient iCAF states when TGFb happens to be low
- The IL-6 autocrine loop that stabilizes iCAFs in vivo is absent — iCAFs secrete IL-6 into a void (only affecting MDSCs and M2 macrophages downstream)
- This likely contributes to myCAF overrepresentation and incorrect iCAF/myCAF balance

**Proposed implementation — IL-6 positive feedback on iCAF:**

Since `V_T.IL6` and its degradation kinetics already exist, the change requires only a new Hill function and modifications to two existing reaction rates:

```matlab
% New Hill function for IL-6 feedback on iCAF
H_IL6_iCAF = V_T.IL6 / (IL6_50_iCAF + V_T.IL6)    % new param: IL6_50_iCAF

% Modified qPSC → iCAF activation (add IL-6 boost)
% Old: k_PSC_to_iCAF * qPSC * (1-H_TGFb) * H_PDGF
% New: k_PSC_to_iCAF * qPSC * (1-H_TGFb) * H_PDGF * (1 + f_IL6_iCAF * H_IL6_iCAF)
%   f_IL6_iCAF: fold-increase in iCAF activation when IL-6 is saturating (~2-5x)

% Modified myCAF → iCAF interconversion (IL-6 promotes reversion)
% Old: k_myCAF_to_iCAF * myCAF * (1-H_TGFb)
% New: k_myCAF_to_iCAF * myCAF * (1-H_TGFb) * (1 + f_IL6_iCAF * H_IL6_iCAF)
```

**New parameters (2 for IL-6 feedback):**
- `IL6_50_iCAF` — EC50 for IL-6 feedback on iCAF (units: nM). Should be in the range of IL-6 concentrations the model produces. Can start at same order as `IL6_50_MDSC` or `IL6_50_M2`.
- `f_IL6_iCAF` — fold-enhancement of iCAF activation/interconversion at saturating IL-6 (dimensionless, ~2-5). This controls the strength of the autocrine loop.

#### Deferred: Contact-dependent TGFb scaling (spatial range encoding)

TGFb acts at juxtacrine/short-range paracrine distances (~cell diameters), while IL-6 is freely diffusible (~hundreds of µm). This spatial difference could be encoded by scaling effective TGFb on PSCs by a contact probability (e.g., `f_contact = C1 / (C1 + K_contact * aPSC_total)`). However, Ogawa 2021 Table 1 shows F-stroma (iCAF-dominant) has the **highest** tumor cellularity (34% >30%) while the naive contact model predicts the opposite. This is because (1) iCAFs degrade ECM → less stroma → higher cellularity (bidirectional causality), and (2) cancer cell differentiation state (not cellularity) determines TGFb vs IL-1α production (Somerville 2018). The population-level confounders make cellularity a poor proxy for juxtacrine contact. **Deferred** — the IL-6 autocrine feedback alone is sufficient for now, and adding contact scaling would introduce complexity without clear population-level validation.

**Total parameter count:** 2 new parameters (IL6_50_iCAF, f_IL6_iCAF), 1 new Hill function (H_IL6_iCAF), modifications to 4 existing reaction rate strings. No new species, no new reactions.

**What this fixes:**
1. **iCAF/myCAF balance:** Creates a competing positive feedback (IL-6 → iCAF → more IL-6) that counterbalances the TGFb → myCAF axis. The steady-state iCAF fraction becomes a tunable outcome of TGFb vs IL-6 levels rather than a structural default.
2. **Stroma type diversity:** Enables bistability — high IL-6 / low TGFb → stable iCAF state (F-stroma), high TGFb / low IL-6 → stable myCAF state (A-stroma). C-stroma emerges when both signals are low.
3. **Couples to MMP/ECM mechanism:** If iCAF-mediated ECM degradation is also added (Section 7.2), the IL-6 loop amplifies it: more iCAFs → more IL-6 → more iCAFs → more MMP/ECM degradation. This is the complete F-stroma phenotype.
4. **Treatment response axis:** JAK inhibitors (ruxolitinib) and IL-6 blockers (tocilizumab) are being explored in PDAC. The IL-6 feedback loop would make these therapies mechanistically meaningful in the model — they'd shift the iCAF/myCAF balance toward myCAF, potentially increasing ECM but reducing immunosuppressive cytokines.

**What it doesn't fix directly:** ECM overproduction (that still needs MMP degradation), psc_activation_ratio (needs CAF deactivation), CD8 exclusion attribution (needs CXCL12 reassignment).

**Implementation advantage:** This is the lowest-complexity structural change with the highest mechanistic payoff. IL-6 is already a model species with secretion, degradation, and Hill function infrastructure. The change adds 2 parameters, 1 Hill function, and modifies 2 existing reaction rate strings — no new species, no new reactions.

### 7.8 Revised mechanism priority based on Ogawa evidence and IL-6 analysis

The Ogawa data and the IL-6 gap analysis together revise the priority ordering from Section 6:

| Rank | Mechanism | Evidence strength | Complexity | Targets addressed |
|------|-----------|-------------------|------------|-------------------|
| 1 | **IL-6/STAT3 autocrine feedback on iCAF** | Strong — Biffi 2019, mechanistic basis for stroma types | Very low (2 params, modify 2 rates) | iCAF/myCAF balance, stroma type diversity |
| 2 | **iCAF-mediated ECM degradation (FAP/MMP)** | Strong — Ogawa transcriptome, FAP collagenase | Low-moderate (1 new reaction, 2-3 params) | collagen fold-change, stromal_fraction, ECM-CAF decoupling |
| 3 | **CXCL12 source reassignment (myCAF → iCAF)** | Strong — FAP-CD8 correlation, not ACTA2-CD8 | Minimal (move existing reaction) | CD8 exclusion attribution |
| 4 | **Tighten apCAF target** | Strong — Ogawa protein data | Data change only | apcaf_fraction_of_caf |
| 5 | **Stroma-cancer carrying capacity** | Moderate — cellularity data | Parameter change | stromal_fraction, cancer_cell_density |

~~CAF deactivation/reversion~~ removed — in vitro phenomenon, not demonstrated in vivo in established PDAC (see Section 3.2).

Mechanisms 1+2 together form a coherent iCAF remodeling axis: IL-6 feedback stabilizes iCAFs, and iCAFs degrade ECM via FAP/MMPs. This pair addresses the stroma tug-of-war from both sides (too many myCAFs AND too much ECM). Mechanism 3 is a near-trivial fix that corrects a likely attribution error.

**Recommended implementation order:** 1 → 3 → 2, then re-run NPE diagnostics before deciding on 4-5.

### 7.9 New calibration targets from Ogawa 2021

Two targets are ready for extraction (already drafted in `ogawa2021_ccr_digitization_results.md`):

1. **mycaf_fraction_of_caf** — median 0.717, IQR [0.519, 0.882], n=215. Strong mapping quality (ACTA2 → myCAF). Constrains the iCAF/myCAF balance.

2. **icaf_fraction_of_caf** — median 0.279, IQR [0.176, 0.373], n=215. Moderate mapping quality (FAP → iCAF, some overlap). Constrains the same axis from the complementary direction.

Additional targets considered but deferred:

3. **collagen_fraction_of_stroma** — median 54.1%. Skipped due to unit mismatch (area fraction vs model mass units). Could become viable if the model adds an area-based observable using ECM_level and cell cross-sections.

4. **psc_activation_ratio** — Ogawa does not directly report this, but the near-absence of FAP-fibroblasts in normal pancreas (0.1%) vs PDAC (12.1%) implies near-complete activation. Consistent with existing Drifka 2016 target; no new extraction needed.

### 7.10 Cross-validation with existing targets

| Existing target | Ogawa data point | Agreement |
|----------------|------------------|-----------|
| stromal_fraction (0.668) | 73% with cellularity ≤30% | Consistent (cellularity ~30% ↔ stroma ~70%) |
| psc_activation_ratio (0.575) | Near-absent qPSCs implied | Ogawa suggests higher activation (~0.90+), tension with Drifka 0.575 |
| apcaf_fraction_of_caf (0.064) | "extremely limited" protein expression | Ogawa supports lower end; target median could decrease |
| collagen1_protein_level (9.84x) | Collagen occupancy 54.1% vs normal 13.4% = ~4x increase in occupancy area | Different units (fold-change vs area fraction); not directly comparable but both show moderate increase, not 230x |

**Notable tension:** The psc_activation_ratio target (0.575 from Drifka 2016) and Ogawa's implication of near-complete activation (>90%) are in significant disagreement. Drifka's lower value likely reflects the denominator dilution issue noted in Section 1.3 (non-fibroblast cells in the experimental denominator). After correcting for ~15% dilution: 0.575/0.85 ≈ 0.67, still well below 0.90. This warrants further investigation — either the Drifka measurement captures something different from activation state, or there is genuine quiescent PSC persistence at levels higher than Ogawa implies.


⏺ The investigation reveals this is primarily a mechanism asymmetry, not just a priors issue. Here's the core problem:                                               
                                                                                                                                                                     
  The Compound Imbalance                                                                                                                                             
                                                                                                                                                                     
  ┌─────────────────────────┬───────────────────┬───────────────────────┬────────────────────────────┐                                                               
  │         Process         │       Treg        │          CD8          │            Bias            │                                                               
  ├─────────────────────────┼───────────────────┼───────────────────────┼────────────────────────────┤                                                               
  │ Baseline recruitment    │ 5.8e-5            │ 5.8e-5                │ 1:1                        │                                                               
  ├─────────────────────────┼───────────────────┼───────────────────────┼────────────────────────────┤
  │ Chemotactic recruitment │ CCL5→CCR5, H≈0.12 │ CXCL9→CXCR3, additive │ ~0.7:1                     │                                                               
  ├─────────────────────────┼───────────────────┼───────────────────────┼────────────────────────────┤                                                               
  │ Proliferation           │ 0.05/day          │ 1.0/day               │ 1:20                       │
  ├─────────────────────────┼───────────────────┼───────────────────────┼────────────────────────────┤
  │ Death                   │ 0.1/day           │ 0.1/day               │ 1:1                        │
  ├─────────────────────────┼───────────────────┼───────────────────────┼────────────────────────────┤
  │ Net intratumoral growth │ -0.05/day         │ +0.9/day              │ Tregs shrink, CD8s explode │
  ├─────────────────────────┼───────────────────┼───────────────────────┼────────────────────────────┤
  │ Clonal expansion burst  │ None              │ ~120x                 │ 1:120                      │
  └─────────────────────────┴───────────────────┴───────────────────────┴────────────────────────────┘

  The two dominant issues:

  1. Tregs have negative net growth in the tumor — k_Treg_pro (0.05) < k_Treg_death (0.1), so Tregs can only exist through continuous recruitment. Meanwhile CD8s
  grow at +0.9/day.

  2. CD8s have clonal expansion burst that Tregs lack — n_T_expansion_burst (~120x) in Teff_module.m has no equivalent in Treg_module.m.

  Secondary bottlenecks:
  - TGFb too low for Th→Treg conversion — TGFb_baseline = 0.016 nM, with TGFb_50 = 0.07 nM, so H_TGFb ≈ 0.19 (only 19% conversion efficiency)
  - CCL5 production too weak — steady-state ~0.04 nM vs CCL5_50_Treg = 0.3 nM (only 12% Hill saturation)

  These compound to produce the ~200x CD8 enrichment we observe. The question is which to fix — this is structural (missing Treg expansion mechanism) not just
  parameter tuning. What are your thoughts on the priorities?

### Resolution: Tumor-Local Treg Proliferation (2026-02-23)

**Fix:** Added logistic tumor-local Treg proliferation: `null -> V_T.Treg` at rate `k_Treg_pro_tumor * Treg * (1 - Treg / (rho_Treg_max * V_T + cell))`.

**Rationale:** Tumor-infiltrating Tregs are among the most proliferative immune cells in PDAC (Ki67+ 30-60%, Balli 2017 PMID:28007400, Plitas 2016 PMID:29089587). The model previously had zero tumor-local Treg proliferation — all expansion occurred in V_LN only.

**Parameters:**
- `k_Treg_pro_tumor`: median 0.3/day (Ki67-realistic). Prior: lognormal, sigma=0.7, 95% range [0.076, 1.18].
- `rho_Treg_max`: median 1e7 cells/mL (max Treg density from IHC data). Prior: lognormal, sigma=1.0, 95% range [1.4e6, 7.1e7].

**Carrying capacity:** `K = rho_Treg_max * V_T` (density-limited, grows with tumor). Prevents unbounded Treg growth during evolve_to_diagnosis while allowing biologically realistic proliferation rates (k_pro > k_death is permitted). ECM modulation was considered but rejected — `H_ECM_T_mot ≈ 0.93` in PDAC crushes effective capacity, and dense stroma actually creates a Treg-favorable niche (TGFb-rich), not an unfavorable one.

**Result:** CD8:Treg ratio dropped from ~208 (no tumor-local proliferation) to ~7.3 at median params. Qualitative test threshold set to <10; clinical target of <5 is for SBI calibration.
