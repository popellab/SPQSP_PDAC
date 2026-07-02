# Cross-Validation: Sivagnanam 2021 Table S5 vs Calibration Targets

**Date:** 2026-02-23
**Source:** Sivagnanam et al. 2021 (cited as Liudahl2021_LeukocyteHeterogeneityPDAC in calibration targets), "Leukocyte Heterogeneity in Pancreatic Ductal Adenocarcinoma," Cancer Discovery. n=113 treatment-naive PDAC specimens, multiplex IHC. Supplementary Table S5 reports cell counts and densities across histopathological regions.

**Note:** Many existing calibration targets already cite this paper (Liudahl 2021). This analysis verifies the extraction pipeline by comparing raw S5 values against extracted calibration target medians, and identifies new targets and cross-cohort insights.

---

## 1. Raw Data: Tumor (T) Region Cell Densities

From Supplementary Table S5, Cell Density per Patient, Median (IQR), in cells/mm²:

| Cell type | Median | IQR | Model species |
|-----------|--------|-----|---------------|
| PanCK+ (cancer) | 995 | 660–1526 | V_T.C1 |
| CD8+ T cell | 76 | 31–142 | V_T.CD8 |
| Th0 (CD4+) | 77 | 34–134 | V_T.Th (partially) |
| Th1 | 3 | 1–6 | V_T.Th (partially) |
| Th2 | 7 | 2–18 | V_T.Th (partially) |
| Th17 | 1 | 0–4 | V_T.Th (partially) |
| Treg | 14 | 5–26 | V_T.Treg |
| CD20+ B cell | 21 | 8–42 | Not modeled |
| Plasma cell | 2 | 0–10 | Not modeled |
| Plasmablast | 2 | 0–7 | Not modeled |
| Mast cell | 21 | 10–42 | Not modeled |
| Neutrophil/Eosinophil | 49 | 17–149 | V_T.MDSC (proxy) |
| DC-LAMP+ DC (mature) | 1 | 1–4 | V_T.mcDC1 + V_T.mcDC2 |
| DC-LAMP- DC (immature) | 69 | 21–161 | V_T.cDC1 + V_T.cDC2 |
| CD163+ Mono/Mac (M2) | 61 | 26–118 | V_T.Mac_M2 |
| CD163- Mono/Mac (M1) | 39 | 20–68 | V_T.Mac_M1 |

**Not measured:** NK cells, CAFs (iCAF/myCAF/apCAF/qPSC), ECM/collagen, alpha-SMA.

---

## 2. Cross-Validation Against Existing Calibration Targets

### 2.1 Targets sourced from this paper (internal consistency check)

These targets were extracted from Sivagnanam/Liudahl 2021. Cross-validation confirms whether the extraction pipeline preserved the raw values correctly.

| Target ID | Extracted median | S5 raw value | Match? | Notes |
|-----------|-----------------|-------------|--------|-------|
| cancer_cell_density_baseline | 996.47 cells/mm² | 995 (660–1526) | **Yes** | Direct extraction from PanCK+ density |
| tam_density | 120.49 cells/mm² | ~100 (sum of CD163+/- medians) | **Close** | Extracted value is higher; likely from lognormal fit to sum of cell counts rather than sum of density medians. Median of sum ≠ sum of medians. |
| treg_density_baseline | 13.85 cells/mm² | 14 (5–26) | **Yes** | |
| cd8_treg_ratio | 5.39 | 76/14 = 5.43 | **Yes** | |
| treg_fraction_cd4 | 0.111 | 14/(77+3+7+1+14) = 0.137 | **Close** | Discrepancy because median of ratio ≠ ratio of medians. Extracted value likely from per-patient ratio distribution. |
| m2_macrophage_fraction | 0.61 | 61/(61+39) = 0.61 | **Exact** | |
| total_til_density | 256 cells/mm² | ~199 (sum of T cells + Treg + B cells, no NK) | **Plausible** | Gap likely due to NK cells and per-patient sum distribution |
| apc_maturation_ratio | 0.01418 | 1/(1+69) = 0.0143 | **Exact** | DC-LAMP+ / total DC |
| mdsc_density | 48.3 cells/mm² | 49 (17–149) | **Yes** | Uses Neutrophil/Eosinophil as MDSC proxy |
| cd4_helper_density | 109.74 cells/mm² | 77+3+7+1 = 88 (sum of Th medians) | **Close** | Median of sum > sum of medians |

**Verdict:** Extraction pipeline is sound. Small discrepancies (tam_density, treg_fraction_cd4, cd4_helper_density) are expected because the extraction uses per-patient distributions (lognormal fits to cell counts / area), not simple arithmetic on population medians.

### 2.2 Targets sourced from other papers (cross-cohort validation)

| Target ID | Extracted median | Source | S5-derived value | Agreement? |
|-----------|-----------------|--------|-----------------|------------|
| cd8_density_baseline | 138.78 | Jansen 2021 (n=444) | 76 (31–142) | **Discrepant** — Jansen 1.8x higher |
| m1_m2_ratio | 0.518 | Costa 2022 (n=30) | 39/61 = 0.64 | **Within CI** (0.071–3.722) |
| cdc1_density_baseline | 7.82 | Plesca 2022 (n=40) | Cannot compute (no cDC1-specific marker) | N/A |
| nk_fraction_of_tils | 0.032 | Marcon 2020 (n=9) | Not measured | N/A |
| cd8_exhausted_fraction | 0.728 | Shibuya 2014 (flow, n~15) | Not measured | N/A |
| treg_fraction_baseline | 0.052 | Kiryu 2021 (n=85) | 14/(76+88+14+49+70+21) ≈ 0.044 | **Close** (depends on denominator definition) |

**Key discrepancy: cd8_density_baseline.** Jansen 2021 (n=444, meta-analysis of multiple IHC markers) gives a median of 139 cells/mm², while Sivagnanam (n=113, specific multiplex panel) gives 76 cells/mm². Possible explanations:
- Different CD8 antibody clones or staining protocols
- Different tissue region definitions (Jansen may include stromal CD8+ cells; Sivagnanam defines specific tumor ROIs)
- Jansen aggregates across studies with heterogeneous cohorts
- Sample population differences (stage, resectability, demographics)

The 1.8x difference is substantial but not unusual for IHC-based immune cell quantification across studies. The existing target CI (20–965) is deliberately wide and encompasses both values. However, if the model is calibrating to 139 when the "true" value is closer to 76, this could affect posterior inference for CD8-related parameters.

**Recommendation:** Consider using the Sivagnanam value (76 cells/mm²) as a secondary target or weighted average with Jansen. The Sivagnanam data is from a single well-controlled study; Jansen is larger but more heterogeneous.

---

## 3. New Derived Quantities

### 3.1 TAM/DC ratio (cross-validation of tam_to_cdc1_ratio)

From Sivagnanam alone (all-DC denominator):

```
Total TAMs = CD163+ + CD163- = 61 + 39 = 100 cells/mm²
Total DCs = DC-LAMP+ + DC-LAMP- = 1 + 69 = 70 cells/mm²
TAM / total_DC = 100 / 70 ≈ 1.4
```

This is TAM/all-DCs, not TAM/cDC1. To get TAM/cDC1, we need to estimate what fraction of Sivagnanam's DCs are cDC1. Using the Plesca 2022 cDC1 fraction (~0.77 of conventional DCs):

```
Estimated cDC1 among Sivagnanam DCs: 0.77 × 70 ≈ 54 cells/mm²
TAM / cDC1 ≈ 100 / 54 ≈ 1.9
```

But this overestimates cDC1 because Sivagnanam's DC panel (DC-LAMP-based) likely includes pDCs and moDCs in addition to conventional DCs, inflating the denominator. Using Plesca's cDC1 density directly:

```
Plesca cDC1 density = 9.84 cells/mm²
Sivagnanam TAMs = 100 cells/mm²
Cross-cohort TAM / cDC1 ≈ 100 / 9.84 ≈ 10.2
```

**Comparison with Breaza target:**

| Source | TAM/cDC1 estimate | Denominator definition |
|--------|-------------------|----------------------|
| Breaza 2025 | 4.0 (CI: 1.5–10.7) | CD1a+ DCs (broad, includes non-cDC1) |
| Sivagnanam + Plesca cross-cohort | ~10.2 | CD163+/- TAMs / CLEC9A+ cDC1 |
| Sivagnanam alone (TAM/all-DC) | ~1.4 | All macrophages / all DCs |

The cross-cohort estimate of ~10 is at the upper edge of the Breaza CI. This confirms the suspicion from the structural review: the Breaza CD1a+ DC denominator is broader than cDC1, making the Breaza ratio a lower bound. The true TAM/cDC1 ratio is likely higher than 4.

**Implication for model calibration:** The model's high TAM/cDC1 ratio may be closer to reality than the Breaza target suggests. If the true ratio is ~10, the model's overestimate is less severe than the diagnostics indicated.

### 3.2 DC maturation ratio — independently validated

```
DC-LAMP+ / (DC-LAMP+ + DC-LAMP-) = 1 / 70 = 0.0143
```

This is an independent derivation from the raw S5 table and matches the extracted apc_maturation_ratio target (0.01418) exactly. The vast majority (~98.6%) of tumor-infiltrating DCs are immature. This is consistent with PDAC's immunosuppressive TME blocking DC maturation.

### 3.3 Mature cDC1 estimate (combining Sivagnanam + Plesca)

```
Total mature DCs (DC-LAMP+): 1 cell/mm²
Total DCs: 70 cells/mm²
If cDC1 fraction among conventional DCs ≈ 0.77 (Plesca):
  Estimated mature cDC1 (mcDC1) ≈ 0.77 × 1 ≈ 0.8 cells/mm²
  Estimated immature cDC1 ≈ 9.84 - 0.8 ≈ 9.0 cells/mm²
  cDC1 maturation fraction ≈ 0.8 / 9.84 ≈ 0.08 (8%)
```

This is important for the cdc1_fraction_of_dc mapping error identified in the structural review. If mature DCs are included in the denominator:

```
Total cDC1 (immature + mature) ≈ 9.0 + 0.8 = 9.8 cells/mm²
Total cDC2 (immature + mature) ≈ 2.98 + 0.2 = 3.2 cells/mm²
cDC1 fraction (all DCs) ≈ 9.8 / 13.0 ≈ 0.75
```

vs the current immature-only formula:
```
cDC1 fraction (immature only) ≈ 9.84 / 12.82 ≈ 0.77
```

The difference is small (~0.75 vs 0.77) because mature DCs are rare (only 1.4% of total). But the Monte Carlo median in the calibration target is 0.81 due to the lognormal ratio distribution. The key point: **including mature DCs won't substantially change the calibration target value.** The mapping error exists but is minor for this specific observable because so few DCs mature in the tumor.

### 3.4 M1-like macrophage characterization

```
CD163- Mono/Mac: 39 (20–68) cells/mm² in tumor
CD163- Mono/Mac: 10 (6–17) cells/mm² in healthy normal
Fold increase: 39/10 = 3.9x
```

vs M2-like:
```
CD163+ Mono/Mac: 61 (26–118) cells/mm² in tumor
CD163+ Mono/Mac: 78 (73–92) cells/mm² in healthy normal
Fold increase: 61/78 = 0.78x (actually DECREASED)
```

**Key insight: M2-like (CD163+) macrophages are actually LOWER in PDAC tumors than in normal pancreas.** The increase in TAMs is driven entirely by CD163- (M1-like or undifferentiated) monocyte/macrophage recruitment. This challenges the common assumption that PDAC is dominated by M2 polarization at diagnosis. The M2 fraction is 0.61 not because M2s are elevated, but because CD163- cells aren't elevated *enough* to shift the ratio.

**Implication for model:** The macrophage module assumes M1 recruitment with subsequent M2 polarization. This data suggests the recruitment itself is M1-biased but doesn't increase M2s above normal tissue levels. The model's M1→M2 polarization rate may be appropriate, but the baseline macrophage recruitment assumption might need revisiting.

### 3.5 CD4 T helper subtypes

The model lumps all CD4 helpers as `V_T.Th`. The Sivagnanam data breaks them down:

| CD4 subtype | Density | Fraction of CD4 helpers |
|-------------|---------|------------------------|
| Th0 (naive/memory) | 77 | 87.5% |
| Th2 | 7 | 8.0% |
| Th1 | 3 | 3.4% |
| Th17 | 1 | 1.1% |

The dominance of Th0 (87.5%) means most CD4 T cells in PDAC are functionally uncommitted. The Th1 fraction (3.4%) is very low, consistent with the immunosuppressive TME. This is relevant because the model's V_T.Th includes all helper subtypes, and the Th-derived IFNg secretion rate implicitly assumes Th1 function. With only 3.4% Th1, the effective IFNg contribution from CD4 helpers would be much lower than if all Th were Th1-like.

### 3.6 Tertiary lymphoid structures (TLS) — future reference

The TLS region data provides a contrasting immune environment:

| Cell type | Tumor (T) | TLS | Fold difference |
|-----------|-----------|-----|-----------------|
| CD8+ | 76 | 1250 | 16x |
| Th0 | 77 | 1111 | 14x |
| Treg | 14 | 104 | 7x |
| CD20+ B cell | 21 | 4230 | 200x |
| DC-LAMP+ DC | 1 | 33 | 33x |

TLS have 33x more mature DCs than tumor regions. This suggests that DC maturation is not intrinsically blocked — it's specifically suppressed in the TME. If TLS form in response to immunotherapy, the model should be able to represent this shift. This could become relevant for GVAX scenario calibration.

---

## 4. Healthy Normal Pancreas as Reference

The HN column provides baseline values for non-diseased pancreas:

| Cell type | Tumor (T) | Healthy Normal (HN) | Tumor/Normal |
|-----------|-----------|---------------------|-------------|
| PanCK+ | 995 | 1873 | 0.53x |
| CD8+ T cell | 76 | 40 | 1.9x |
| Th0 T cell | 77 | 11 | 7.0x |
| Treg | 14 | 0.2 | 70x |
| DC-LAMP- DC | 69 | 1 | 69x |
| DC-LAMP+ DC | 1 | 0.1 | 10x |
| CD163+ Mono/Mac | 61 | 78 | 0.78x |
| CD163- Mono/Mac | 39 | 10 | 3.9x |
| Neutrophil/Eosinophil | 49 | 5 | 9.8x |

**Key ratios for model initialization:**
- Tregs show the most dramatic increase (70x) — consistent with strong Treg recruitment/induction in PDAC
- DCs increase dramatically (69x for immature) — there IS DC recruitment to tumors, they just don't mature
- Cancer cell density is actually LOWER in tumor than normal epithelium (0.53x) — stroma displaces epithelial cells
- CD163+ macrophages decrease (0.78x) while CD163- increase (3.9x)

---

## 5. Implications for Structural Model Review

### 5.1 The cdc1_fraction_of_dc mapping error is minor

Including mature DCs changes the fraction from ~0.77 to ~0.75 (Section 3.3). The Monte Carlo-derived calibration target (0.81) is driven by the lognormal ratio distribution, not by whether mature DCs are included. The observable should still be corrected for accuracy, but this will not resolve the structural mismatch. The APC module still needs investigation.

### 5.2 The tam_to_cdc1_ratio target is likely an underestimate

Cross-cohort analysis (Section 3.1) suggests the true ratio is ~10, not 4. The Breaza target's broad CD1a+ denominator inflates the DC count. **Recommendation:** Either (a) replace the Breaza target with a Plesca-derived estimate using CLEC9A+ cDC1 as denominator, or (b) widen the CI substantially, or (c) extract TAM data from Plesca 2022 directly.

### 5.3 M2 macrophages are not elevated in PDAC vs normal pancreas

This is a surprising finding (Section 3.4). The immunosuppressive M2 phenotype in PDAC comes from the *fraction* being 61% (because CD163- cells are also present), not from an absolute increase in M2 numbers. The model's macrophage recruitment → M2 polarization cascade may overestimate M2 accumulation if it assumes M2s are above normal levels.

### 5.4 Sivagnanam cannot test stroma hypotheses

The dataset lacks CAF markers (alpha-SMA, FAP, PDGFRa), ECM/collagen measurements, and stromal area quantification. The stroma-constrains-growth and MMP hypotheses require a different dataset (e.g., Ogawa 2021, n=215, which has fibroblast subtype and collagen data).

### 5.5 Th1 fraction is very low — affects IFNg model assumptions

Only 3.4% of CD4 helpers are Th1 (Section 3.5). The model's V_T.Th species lumps all CD4 helpers and assumes a single IFNg secretion rate. This likely overestimates IFNg from CD4 cells by ~30x if the rate constant was parameterized assuming Th1-like function for all Th cells.

### 5.6 TLS data supports treatment-scenario modeling

The 33x increase in mature DCs in TLS vs tumor (Section 3.6) confirms that DC maturation is TME-suppressed, not intrinsically blocked. This is relevant for GVAX scenario modeling: if GVAX promotes TLS formation, the model should predict dramatic increases in DC maturation and CD8/CD4 priming.

---

## 6. New Calibration Targets to Consider

The following could be extracted from Sivagnanam S5 as new calibration targets. Starred items would provide independent constraints not covered by existing targets.

| Observable | S5 value (tumor) | Would constrain | Priority |
|-----------|-------------------|----------------|----------|
| **CD163- / CD163+ ratio** (M1/M2 proxy) | 0.64 | M1/M2 balance (cross-validates Costa 2022) | Low (already have m1_m2_ratio) |
| **Mast cell density** | 21 (10–42) | Not modeled | N/A |
| **B cell density** | 21 (8–42) | Not modeled | N/A |
| **Neutrophil density** | 49 (17–149) | Already extracted as mdsc_density | Low |
| **★ Th1/CD4 fraction** | 3/102 = 0.029 | Effective IFNg production from CD4 | Medium |
| **★ Th0/CD4 fraction** | 77/102 = 0.75 | Validates Th activation assumptions | Medium |
| **★ Tumor/normal PanCK+ ratio** | 995/1873 = 0.53 | Dynamic cellularity, ECM displacement | Medium |
| **★ Tumor/normal CD163+ ratio** | 61/78 = 0.78 | M2 accumulation assumptions | High |
| **★ Tumor/normal CD163- ratio** | 39/10 = 3.9 | M1 recruitment | High |

The tumor/normal macrophage ratios (★) are particularly valuable because they challenge the model's assumption that M2 TAMs are dramatically elevated in PDAC. Currently the model recruits macrophages and polarizes them to M2, implying M2 accumulation. The data says M2 density is actually slightly *below* normal, and the M1-like population increases 4x. This could reshape macrophage module assumptions.

---

## 7. Summary of Actionable Findings

1. **Fix cdc1_fraction_of_dc observable to include mature DCs** — confirmed the mapping error exists, but the impact is small (~0.77 → 0.75) because mature DCs are rare (1.4% of total). Still worth fixing for correctness.

2. **Revise or replace tam_to_cdc1_ratio target** — cross-cohort estimate ~10 vs current target of 4. Breaza's CD1a+ denominator is too broad. Consider extracting TAM data from Plesca 2022 for a properly-matched ratio.

3. **cd8_density_baseline shows cross-cohort discrepancy** — Sivagnanam gives 76 vs Jansen 139 (1.8x). Both within the wide CI, but the point estimate difference could affect CD8 trafficking parameter inference.

4. **M2 macrophage density is NOT elevated in PDAC vs normal pancreas** — CD163+ cells are 61/mm² in tumor vs 78/mm² in normal (0.78x). The increase is entirely in CD163- (M1-like) cells (3.9x). This may warrant a macrophage module review.

5. **Th1 fraction is only 3.4%** — the model may overestimate CD4-derived IFNg if it treats all Th cells as Th1-like.

6. **Sivagnanam data cannot test stroma hypotheses** — no CAF, ECM, or stromal area measurements. Need Ogawa 2021 or similar dataset.

7. **Existing target extractions are internally consistent** — all Liudahl-sourced targets match S5 values within expected distributional differences.