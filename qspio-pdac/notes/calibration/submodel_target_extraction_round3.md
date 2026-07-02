# SubmodelTarget Extraction — Round 3

Date: 2025-03-13
Source: submodel_target_priorities.md — PRCC/OBED priority parameters

## Parameters

| Param | PRCC | OBED | Value | Units | Model role |
|-------|------|------|-------|-------|------------|
| f_stroma_max | 7 | 15 | 3.0 | dimensionless | Max CAF:cancer cell ratio (logistic carrying capacity) |
| TGFb_50_APC | 46 | 9 | 0.07 | nM | Half-max TGFb for inhibiting APC maturation |
| APC0_cDC1_T | 34 | 20 | 5.4e5 | cell/mL | Baseline cDC1 (BATF3+) density in tumor |
| ECM_max | 69 | 13 | 100 | mg/mL | Maximum ECM density (collagen carrying capacity) |
| f_ECM_50_APC_mig | 65 | 22 | 0.15 | dimensionless | Fraction of ECM_max for 50% APC migration inhibition |
| DAMP_50 | 106 | 7 | 1e4 | cell/day | Half-max tumor death rate for DAMP-mediated APC maturation |

---

## Extractability Assessment

| Param | Extractable? | Best paper | Approach |
|-------|-------------|------------|----------|
| **TGFb_50_APC** | YES | Fogel-Petrovic 2007 (PMID 18039529) | 3-point DC maturation dose-response (0.02, 0.2, 2 ng/mL) |
| **ECM_max** | YES | Desogere 2017 (DOI 10.1038/s41598-017-08838-6) | Hydroxyproline HPLC in mouse PDAC |
| **f_stroma_max** | YES | Lin et al. 2020 (DOI 10.1186/s13073-020-00776-9) | scRNA-seq CAF:cancer per-patient ratios |
| **APC0_cDC1_T** | DIFFICULT | No PDAC-specific cDC1 density | Cross-tissue bridging needed |
| **f_ECM_50_APC_mig** | DIFFICULT | No DC dose-response | Only 2-point DC data (Gunzer 2000) |
| **DAMP_50** | DIFFICULT | Li 2017 + Rovere-Querini 2004 | Multi-step bridging (death rate -> HMGB1 -> DC) |

---

## 1. TGFb_50_APC — RECOMMENDED FOR EXTRACTION

### Literature
- **Fogel-Petrovic et al. 2007** (PMID 18039529, Int Immunopharmacol)
  - Title: "Physiological concentrations of transforming growth factor beta1 selectively inhibit human dendritic cell function"
  - Human monocyte-derived DCs, TGFb1 at 0.02, 0.2, 2 ng/mL
  - 0.02 ng/mL (0.0008 nM): no effect on any marker
  - 0.2 ng/mL (0.008 nM): IL-12p70 suppressed significantly
  - 2 ng/mL (0.08 nM): complete maturation failure (CD86, HLA-DR, IL-12p70 all suppressed)
  - IC50 for CD86/HLA-DR: ~0.04-0.08 nM; for IL-12p70: ~0.008 nM
  - TGFb1 MW ~25 kDa (1 ng/mL = 0.04 nM)

- **Ohtani et al. 2009** (DOI 10.1111/j.1365-2567.2008.02919.x)
  - Human Mo-DCs + HPC-DCs, TGFb1 at 1, 10, 20, 100 ng/mL
  - Dose-dependent CD86 suppression; IC50 ~1-10 ng/mL (0.04-0.4 nM)
  - Higher concentrations than Fogel-Petrovic, less precise for low-dose range

### Forward model
`direct_fit` Hill equation: maturation_marker = 1 / (1 + (TGFb / TGFb_50)^n)

### Papers needed
- [ ] Fogel-Petrovic et al. 2007 (PMID 18039529) — PRIMARY

---

## 2. ECM_max — RECOMMENDED FOR EXTRACTION

### Literature
- **Desogere et al. 2017** (DOI 10.1038/s41598-017-08838-6, Sci Rep)
  - Title: "Molecular MR imaging of fibrosis in a mouse model of pancreatic cancer"
  - Hydroxyproline HPLC in orthotopic syngeneic mouse PDAC
  - Healthy pancreas: 523 +/- 14 ug/g hydroxyproline (n=5) -> ~4.1 mg/mL collagen
  - Mature tumor (d13-16): ~1121-1244 ug/g (n=18) -> ~9.7 mg/mL collagen
  - Conversion: HyP x 7.46 = collagen; tissue density ~1.05 g/mL

- **Bedossa et al. 1989** (PMID 2599283, Digestion)
  - Human pancreas, Sirius red colorimetric
  - Normal: 26.5 +/- 7.2 ug collagen/mg protein (n=11)
  - Chronic calcifying pancreatitis: 77.9 +/- 8.0 ug/mg protein (n=17) -> ~12 mg/mL est.

- **Imamura et al. 1995** (PMID 8532652, Pancreas)
  - Human PDAC hydroxyproline; PDAC ~3x normal (~12 mg/mL est.)

- **Cross-tissue ceiling:** Liver cirrhosis ~23-40 mg/mL (most fibrotic soft organ)

### Assessment
- Observed PDAC collagen: ~6-15 mg/mL at diagnosis
- Current ECM_max=100 is 2.5x higher than severe liver cirrhosis
- As a carrying capacity it controls logistic shape, not necessarily achieved
- ECM_max ~30-50 mg/mL would be more physiologically bounded

### Forward model
`algebraic` — collagen_mg_per_mL = hydroxyproline_ug_per_g * 7.46 * tissue_density / 1000

### Papers needed
- [ ] Desogere et al. 2017 (DOI 10.1038/s41598-017-08838-6) — PRIMARY
- [ ] Bedossa et al. 1989 (PMID 2599283) — SECONDARY (human, but no PDAC)

---

## 3. f_stroma_max — RECOMMENDED FOR EXTRACTION

### Literature
- **Lin et al. 2020** (DOI 10.1186/s13073-020-00776-9, Genome Med)
  - Title: "Single-cell transcriptome analysis of tumor and stromal compartments of pancreatic ductal adenocarcinoma"
  - scRNA-seq, n=10 primary PDAC tumors (~15,000 cells)
  - ETCs: mean 42.2% (range 1.7-85.1%)
  - CAFs: mean 21.7% (range 2.8-68.3%)
  - Mean CAF:cancer = 21.7/42.2 = ~0.51:1
  - Range: 0.03:1 to 40:1 (extreme variability)
  - Caveat: scRNA-seq underrepresents fibroblasts (dissociation bias)

- **TCGA 2017** (DOI 10.1016/j.ccell.2017.07.007)
  - Median neoplastic cellularity 18%, n=150
  - Implies ~80% non-cancer (immune + fibroblast + ECM + vasculature)

### Assessment
- Lin 2020 per-patient ratios provide a direct distribution for bootstrapping
- f_stroma_max represents the MAXIMUM ratio (carrying capacity), not the mean
- Current value of 3.0 is at the high end but within observed range

### Forward model
`algebraic` — CAF_cancer_ratio = CAF_fraction / cancer_fraction (per patient)

### Papers needed
- [ ] Lin et al. 2020 (DOI 10.1186/s13073-020-00776-9) — PRIMARY (need supplementary tables)

---

## 4. APC0_cDC1_T — DEFERRED

### Why deferred
No published study reports cDC1-specific (BATF3/XCR1/CD141) absolute density (cells/mm2 or cells/mL) for human PDAC. Available data:
- Pan-tumor cDC1 ~14 cells/mm2 (van der Hoorn 2024, NOT PDAC)
- PDAC has 79-fold fewer cDC1s than lung (Hegde 2020, mouse)
- Total DCs <=3/HPF in 96% of PDAC (Dallal 2002)
- Total DCs <4% of myeloid in PDAC (Steele 2020)

Cross-tissue bridging would introduce 10-100x uncertainty. Better constrained as a full-model calibration target using diagnosis-state mIHC (if such data can be found).

---

## 5. f_ECM_50_APC_mig — DEFERRED

### Why deferred
No DC-specific migration vs collagen concentration dose-response exists. Only:
- 2-point DC data: 1.67 vs 3.96 mg/mL (Gunzer & Friedl 2000)
- 3-point CTL data: 2, 4, 5 mg/mL with IC50 ~3-4 mg/mL (Sapudom 2021, preprint)
- DCs use podosome/MMP-dependent migration (Cougoule 2018), so CTL data is not directly transferable

Insufficient data for a submodel target. Leave to full-model calibration.

---

## 6. DAMP_50 — DEFERRED

### Why deferred
Parameter is in cell/day (death rate), not concentration. Constraining it requires a bridging calculation:
1. HMGB1 per dying cell: ~100 pg/cell (Rovere-Querini 2004)
2. HMGB1 EC50 for DC activation: ~10-100 ng/mL (Li 2017, Yang 2007)
3. Bridge through compartment volume and HMGB1 half-life

Additional complication: bell-shaped dose-response (Li 2017) — high HMGB1 suppresses DC maturation, which the Hill function doesn't capture.

Multi-step bridging with large compounding uncertainty. Leave to full-model calibration.

---

## Next steps

1. Obtain PDFs for extractable parameters:
   - `to-review/papers/FogelPetrovic2007/` — TGFb_50_APC
   - `to-review/papers/Desogere2017/` — ECM_max
   - `to-review/papers/Lin2020/` — f_stroma_max (need supplementary tables)
2. Extract targets one by one (TGFb_50_APC first — cleanest data)
3. Validate each with `mcp__maple__validate_target`