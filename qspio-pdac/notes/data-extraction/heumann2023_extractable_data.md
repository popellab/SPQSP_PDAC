t# GVAX PDAC Trials — Paper Family & Extractable Data

## Paper Family

### NCT02451982 (Resectable PDAC)

| # | Paper | Journal | DOI | Focus | Our Targets |
|---|-------|---------|-----|-------|-------------|
| 1 | **Zheng et al. 2021** | Clin Cancer Res 27(5):1278-1286 | 10.1158/1078-0432.CCR-20-2974 | GVAX ± Cy: TLA formation, DFS/OS (n=87) | None yet |
| 2 | **Li et al. 2022** | Cancer Cell 40(11):1374-1391 | 10.1016/j.ccell.2022.10.001 | GVAX ± nivo: multi-omic TME, mIHC (n=19 subset) | 18 YAMLs in `calibration_targets/gvax_nivo_neoadjuvant/` |
| 3 | **Heumann et al. 2023** | Nat Commun 14:3650 | 10.1038/s41467-023-39196-9 | GVAX ± nivo ± urelumab: CD8+CD137+, DFS/OS (n=40) | None yet |

### NCT03161379 (Borderline Resectable PDAC)

| # | Paper | Journal | DOI | Focus | Our Targets |
|---|-------|---------|-----|-------|-------------|
| 4 | **Agarwal et al. 2025** | Clin Cancer Res 31(15):3205-3214 | 10.1158/1078-0432.CCR-24-3403 | GVAX/Cy/nivo/SBRT: CD8 density, OS (n=31) | None yet |

**Important:** The calibration CSV (`scenarios/calibration_targets/pdac_calibration_targets.csv`) references "Zheng et al. 2022, DOI: 10.1016/j.ccell.2022.10.024" for treatment-response targets (rows 45-48). Both the year and DOI are wrong:
- The DOI `10.1016/j.ccell.2022.10.024` resolves to an NSCLC commentary (Ballesteros et al.)
- The title in the CSV ("Vaccine-induced intratumoral lymphoid aggregates...") matches **Zheng et al. 2021** (Clin Cancer Res, DOI: 10.1158/1078-0432.CCR-20-2974)
- But the actual mIHC data used in those targets comes from **Li et al. 2022** (Cancer Cell, DOI: 10.1016/j.ccell.2022.10.001)
- The existing YAML calibration targets already have the correct Li 2022 DOI — only the input CSV rows need fixing

---

## Zheng et al. 2021 (Clin Cancer Res)

**PMID:** 33277370 | **PMCID:** PMC7925374

### Study Design
- **Population:** Resectable PDAC, n=87 randomized (66 evaluable for efficacy)
- **Arms (1:1:1 randomization):**
  - **A**: GVAX alone (no cyclophosphamide) — n=29 (22 evaluable)
  - **B**: GVAX + IV cyclophosphamide — n=28 (23 evaluable)
  - **C**: GVAX + oral metronomic cyclophosphamide — n=30 (21 evaluable)
- **Note:** NO nivolumab in any arm. This is the original GVAX ± Cy study.

### Survival Data (Table 2 — all 87 patients)

| Arm | n | DFS Events | Median DFS mo (95% CI) | OS Events | Median OS mo (95% CI) |
|-----|---|-----------|------------------------|-----------|------------------------|
| A (GVAX alone) | 29 | 24 | 18.92 (13.87, 34.1) | 23 | 34.2 (21.6, 45.7) |
| B (GVAX+IV Cy) | 28 | 25 | 8.54 (2.66, 17.1) | 24 | 15.4 (13.2, 26.5) |
| C (GVAX+oral Cy) | 30 | 27 | 5.56 (2.67, 14.0) | 26 | 16.5 (11.3, 25.2) |

**Evaluable cohort (n=66, from Results text):**
- Arm A: median DFS 18.4 mo
- Arm B: median DFS 11.7 mo (p=0.29 vs A)
- Arm C: median DFS 7.2 mo (p=0.02 vs A)
- Arm A median OS: 35.0 mo (from abstract)
- Historical controls (adjuvant GVAX alone): median OS 24.8 mo

**Key finding:** Low-dose metronomic Cy in combination with GVAX results in inferior outcome vs GVAX alone.

### Tumor Size at Resection (Table 1 — n=66 evaluable)

| | All | Arm A | Arm B | Arm C |
|--|-----|-------|-------|-------|
| Tumor size cm, median (IQR) | 3.52 (1.16) | 3.21 (1.06) | 3.44 (1.03) | 3.94 (1.31) |

### TLA Data (Fig 4-5)
- TLAs observed in GVAX-treated resected tumors but not in unvaccinated historical controls
- Higher TLA density associated with longer OS (OS>24mo vs OS<15mo, Fig 5)
- TLA density is qualitative/semi-quantitative, not easily mapped to a calibration target

---

## Heumann et al. 2023 (Nat Commun)

**PMID:** 37339979 | **PMCID:** PMC10281953

### Study Design

- **Population:** Resectable PDAC, n=40 efficacy cohort
- **Arms:**
  - **A**: Cy-GVAX alone (n=16)
  - **B**: Cy-GVAX + nivolumab (n=14)
  - **C**: Cy-GVAX + nivolumab + urelumab/CD137 agonist (n=10)
- **Neoadjuvant window:** Single priming dose ~2 weeks before surgery
  - Time from treatment to surgery: Arm A median 12.5d (Q1 11, Q3 15), Arm B 14d (12.3, 15), Arm C 14d (11, 15.8)
- **Post-surgery:** 2nd priming → SOC adjuvant chemo → Priming #3-6 q28d → Follow-up/extended treatment
- **Data cutoff:** 25 May 2022

### Relationship to Other Papers

- Li et al. 2022 is the companion multi-omic paper. Li reports mIHC cell composition (CD8, CD4, Treg, NK, M1, M2, etc.) for Arms A (n=9) and B (n=10) — already digitized in `notes/li2022_digitization_results.md`, 18 calibration targets in `calibration_targets/gvax_nivo_neoadjuvant/`.
- Heumann 2023 reports **clinical outcomes** (DFS, OS) for expanded cohort and **TLA-specific immune data** (CD8+CD137+, CD8+GZMB+) — different axes from Li 2022.
- Zheng 2021 reports the earlier GVAX ± Cy arms (no nivo) with DFS/OS.

## Extractable Data

### 1. Disease-Free and Overall Survival (Table 2) — HIGH PRIORITY

| Arm | n | Median follow-up | DFS events | Median DFS mo (95% CI) | OS events | Median OS mo (95% CI) |
|-----|---|-----------------|------------|------------------------|-----------|------------------------|
| Cy-GVAX (A) | 16 | 23.1 | 12 | 13.90 (5.59, NR) | 12 | 23.59 (13.27, NR) |
| Cy-GVAX+PD-1 (B) | 14 | 26.1 | 13 | 14.98 (7.95, 44.09) | 11 | 27.01 (20.76, NR) |
| Cy-GVAX+PD-1+CD137 (C) | 10 | 31.6 | 6 | 33.51 (16.76, NR) | 4 | 35.55 (17.74, NR) |

**Hazard ratios (DFS):**
- B vs A: HR 1.09 (0.50–2.40), p=0.829
- C vs A: HR 0.55 (0.21–1.49), p=0.242
- C vs B: HR 0.51 (0.19–1.35), p=0.173

**Hazard ratios (OS):**
- B vs A: HR 1.11 (0.47–2.63), p=0.813
- C vs A: HR 0.59 (0.18–1.91), p=0.377
- C vs B: HR 0.53 (0.17–1.67), p=0.279

**Note:** Survival measured from date of first neoadjuvant study treatment; Cox regression utilized. NR = not reportable.

**Model mapping considerations:**
- Arms A and B map directly to our existing GVAX and GVAX+nivo scenarios
- Arm C (CD137 agonist) is not currently modeled
- Requires a long-running clinical scenario (18-36+ months), not the existing 21-day neoadjuvant snapshot
- DFS would need a tumor regrowth/recurrence endpoint; OS would need a death/survival endpoint
- KM curves in Fig 4 have number-at-risk tables that could be digitized for more granular survival data

### 2. CD8+CD137+ T Cell Density in TLAs (Fig 3b, text) — LOW PRIORITY (TLA-specific)

Mean density of CD8+CD137+ T cells within TLAs (% of all cells in ROI):
- Arm A (Cy-GVAX): 3.72%
- Arm B (Cy-GVAX+PD-1): 0.183%
- Arm C (Cy-GVAX+PD-1+CD137): 27.9%
- C vs A: p=0.007; C vs B: p=0.003

Overall median across arms: 0.41%. Patients with >0.41% CD8+CD137+ in TLAs had HR 0.30 (95% CI 0.11–0.86, p=0.026) for DFS.

**Model mapping:** TLA-specific — our model does not distinguish TLA from non-TLA tumor regions. Low mapping confidence.

### 3. CD8+GZMB+ T Cells in TLAs (Fig 3c) — LOW PRIORITY (TLA-specific)

Scatter plot data (% of all cells in ROI in TLA), by arm. Individual data points visible but not tabulated. General pattern: Arm C has the highest density, but not as dramatically different as CD137+.

**Model mapping:** TLA-specific. Same limitation as above.

### 4. CD8+CD137+GZMB+ Double-Positive in TLAs (Fig 3d) — LOW PRIORITY (TLA-specific)

Subset of activated cytotoxic T cells. Arm C significantly higher than A/B.

**Model mapping:** TLA-specific and double-positive subtype not modeled.

### 5. CD8+TIGIT+ T Cells (Fig 5a) — LOW PRIORITY

Exhaustion marker. Stratified by low vs high CD137+ groups:
- Low CD137+ group: ~5-10%
- High CD137+ group: ~4-6% (trend toward lower exhaustion with CD137 activation)

**Model mapping:** TIGIT not modeled; CD8 exhaustion captured via PD-1+ in existing targets.

### 6. CD8+CD137+ in Tumor Vicinity (Non-TLA) (Fig 5d) — MEDIUM PRIORITY

Activated T cells migrating from TLAs to tumor vicinity: 2-4% of cells in non-TLA tumor ROIs from Arm C, minimal in Arms A/B. This is closer to our model's bulk-tumor compartment.

**Model mapping:** Potentially maps to activated/effector CD8 fraction in V_T, but data only available for Arm C (not currently modeled).

### 7. Baseline Clinicopathologic Characteristics (Table 1) — CONTEXT ONLY

Useful metadata for scenario configuration (age, stage distribution, adjuvant chemo regimens), not direct calibration targets.

Key staging: pT-stage distribution across all arms approximately T1 ~20%, T2 ~70%, T3 ~7%.

## Recommended Targets to Create

### Immediate (map to existing model scenarios):

**None directly actionable for the 21-day neoadjuvant scenario.** The mIHC cell composition data from this trial is already captured via Li 2022 targets (the same patients). Heumann 2023 and Zheng 2021 add clinical outcomes and TLA-specific markers, not bulk-tumor mIHC on new axes.

### Future (requires clinical_progression scenario):

All survival targets below require a long-running scenario (18-36+ months) with tumor regrowth/recurrence and death endpoints. The existing `clinical_progression.yaml` runs 90 days — would need extending.

**From Zheng 2021 (GVAX alone, no Cy, no nivo — the cleanest GVAX arm):**

1. **Median DFS for GVAX alone** — 18.92 mo (95% CI 13.87, 34.1), n=29 (all), or 18.4 mo (n=22 evaluable)
   - Best GVAX-only DFS data; Cy was detrimental, so GVAX-alone arm is the reference

2. **Median OS for GVAX alone** — 34.2 mo (95% CI 21.6, 45.7), n=29, or 35.0 mo (n=22 evaluable)
   - This is remarkably long for resectable PDAC (typical mOS ~20-25 mo with adjuvant chemo alone)

**From Heumann 2023 (GVAX+nivo arms):**

3. **Median DFS for Cy-GVAX+nivo** — 14.98 mo (95% CI 7.95, 44.09), n=14
   - Maps to our `gvax_nivo_neoadjuvant` scenario (extended to clinical progression)

4. **Median OS for Cy-GVAX+nivo** — 27.01 mo (95% CI 20.76, NR), n=14
   - Note: worse than Zheng 2021 GVAX alone (34.2 mo), possibly due to Cy component

**Cross-paper consistency check:**
- Zheng 2021 GVAX alone (no Cy): DFS 18.92 mo, OS 34.2 mo
- Heumann 2023 Cy-GVAX (with Cy): DFS 13.90 mo, OS 23.59 mo
- The Cy component appears to worsen outcomes, consistent with Zheng 2021's finding that Cy was detrimental
- For GVAX-only calibration, use Zheng 2021 Arm A (no Cy)

### Not recommended:

- CD8+CD137+/GZMB+ in TLAs — TLA-specific, no model compartment
- CD8+TIGIT+ — TIGIT not modeled
- Arm C (urelumab) data from Heumann — CD137 agonist not modeled
- Arms B/C from Zheng 2021 — Cy was detrimental, not our target regimen

---

## Agarwal et al. 2025 (Clin Cancer Res)

**DOI:** 10.1158/1078-0432.CCR-24-3403 | **Trial:** NCT03161379

### Study Design

- **Population:** Borderline resectable (BR) PDAC, n=31 enrolled, 30 ITT
- **Treatment sequence:**
  1. Neoadjuvant mFOLFIRINOX (2-4 months)
  2. Cy/GVAX + nivolumab (2 doses, ~4 weeks apart)
  3. SBRT (5 fractions)
  4. Surgery
- **Key cohorts:**
  - 18 received at least one dose of GVAX/Cy/nivo
  - 17 eligible for surgery
  - 14 underwent definitive resection
  - 13 evaluable for CD8 density (1 MR excluded — insufficient tissue)
- **Historic controls:** 21 BR-PDAC patients from Hopkins Biorepository treated with mFOLFIRINOX + SBRT only (no GVAX/nivo), 2012-2015

### CD8+ T-cell Density — PRIMARY ENDPOINT (Fig 3, text)

**Mean CD8 density (cells/mm²):**
- Study (GVAX/Cy/nivo/SBRT): **266.7** cells/mm² (n=13)
- Historic controls (mFOLFIRINOX+SBRT): **199.9** cells/mm² (n=21)
- P=0.246 (two-sample t-test on log-transformed densities)
- **No significant difference** — primary endpoint not met

**Per-patient CD8 density (Fig 3B):**

| Subject | Path Response | CD8 density (cells/mm²) | Log CD8 |
|---------|-------------|------------------------|---------|
| 2 | CR | 65.73 | 4.19 |
| 6 | MR | 288.94 | 5.67 |
| 12 | MR | 514.80 | 6.24 |
| 20 | MR | 582.64 | 6.37 |
| 7 | MO | 188.39 | 5.24 |
| 19 | MO | 266.67 | 5.59 |
| 22 | MO | 161.23 | 5.08 |
| 23 | MO | 212.01 | 5.36 |
| 24 | MO | 363.69 | 5.90 |
| 25 | MO | 260.82 | 5.56 |
| 21 | PR | 375.24 | 5.93 |
| 301 | PR | 147.95 | 5.00 |
| 303 | PR | 38.38 | 3.65 |

(14th patient: subject 302, MR, excluded from CD8 — insufficient tissue)
(Subjects 1, 8, 14: liver met; Subject 11: pre-op met — no CD8 data)

**Computed summary statistics (n=13 study patients):**
- Mean: 266.65, SD: 160.72
- Median: 260.82, IQR: [161.23, 363.69]
- Geometric mean: 214.03
- Log mean: 5.366, Log SD: 0.771

**CD8 density by pathologic response:**
- Marked response (n=3): mean 462.13
- Moderate response (n=6): mean 242.14
- PR (n=3): mean 187.19
- pCR (n=1): 65.73

### Response and Survival

**ORR (RECIST, post-study-intervention imaging):**
- 31% (5/16, 95% CI 0.11–0.59): 5 PR, 8 SD, 3 PD

**Pathologic response (n=14 resected):**
- Complete response: 1 (7%)
- Marked response: 4 (29%)
- Moderate response: 6 (43%)
- Poor response: 3 (21%)

**Resection rates:**
- ITT: 47% (14/30, 95% CI 0.28–0.66)
- Treated cohort: 78% (14/18, 95% CI 0.52–0.94)
- R0: 93% (13/14, 95% CI 0.66–1.00)
- pCR: 3.3% (1/30, 95% CI 0.0008–0.17)

**Survival (ITT, n=30):**
- Median follow-up: 19.5 months
- 18-month OS rate: 67% (95% CI 0.51–0.87)
- **Median OS: 20.4 months (95% CI 18.2–NR)**
- Median DMFS: 13.6 months (95% CI 7.1–NR)
- Resected patients (n=14): median OS 30 months (95% CI 30–NR)

**Tumor size (Table 2, n=14 resected):**
- Median gross tumor size: 1.6 cm (IQR 0.6–2.7) — post-neoadjuvant chemo+SBRT

### Model Mapping Considerations

- **CD8 density** is the most directly mappable target: cells/mm² → V_T.CD8 + V_T.CD8_exh (via cell density conversion, same observable as `cd8_density_baseline_PDAC_deriv001.yaml`)
- **Confound: mFOLFIRINOX + SBRT** — both groups received prior chemo and radiation, which our model does not currently simulate. The GVAX/nivo effect is layered on top of chemo+SBRT.
- **Historic controls** could serve as a "chemo+SBRT only" reference for CD8 density (199.9 cells/mm²), but SBRT is not modeled
- **Borderline resectable** population differs from the resectable population in NCT02451982 — typically larger/more locally advanced tumors
- Per-patient data enables individual-level calibration or bootstrap resampling

### Infrastructure Gaps (DEFERRED)

**Status: Deferred** — the Agarwal scenario requires two unimplemented therapy modules. Focus calibration on NCT02451982 data (GVAX ± nivo without chemo/RT confounds) first.

**Missing modules:**
1. **FOLFIRINOX chemotherapy** — No chemo modules exist. FOLFIRINOX = 5-FU + leucovorin + irinotecan + oxaliplatin. Would need PK for each agent (or lumped), tumor cell kill rate, and myelosuppression effects. Patients received 2-4 months of mFOLFIRINOX before immunotherapy.
2. **SBRT (stereotactic body radiotherapy)** — No radiation module exists. 33 Gy in 5 fractions during cycle 2. Would need radiation dose-response (linear-quadratic model), immunogenic cell death / DAMP release, and abscopal-type immune activation effects.

**Workaround options (for future):**
- **Lumped pre-conditioning**: Single module applying fractional tumor kill + immune modulation to approximate combined mFOLFIRINOX+SBRT before immunotherapy starts
- **Initial condition adjustment**: Set smaller `initial_tumour_diameter` (~2 cm post-chemo, vs 3 cm at diagnosis) and accept approximate TME state
- **Full implementation**: Build FOLFIRINOX + SBRT modules (biggest lift, most accurate)

**Scenario YAML** (`gvax_nivo_sbrt_neoadjuvant_agarwal2025.yaml`) already exists with correct dosing schedule but notes SBRT is not modeled. The `evolve_to_diagnosis` function uses `initial_tumour_diameter` from the variant/parameter (default 3.2 cm), which doesn't account for prior mFOLFIRINOX tumor shrinkage (post-resection median was 1.6 cm).

---

## DOI Fix Applied

In `scenarios/calibration_targets/pdac_calibration_targets.csv`, rows 45-48 previously referenced a fabricated DOI:
> "Zheng et al. 2022 - Vaccine-induced intratumoral lymphoid aggregates ... (Cancer Cell, DOI: 10.1016/j.ccell.2022.10.024)"

**Fixed to:**
> "Li et al. 2022 - Multi-omic analyses of changes in the tumor microenvironment ... (Cancer Cell, DOI: 10.1016/j.ccell.2022.10.001)"

Also fixed `source_tag` in `calibration_targets/gvax_nivo_neoadjuvant/cd8_fold_increase_gvax_nivo_d21_PDAC_deriv001.yaml` from `Zheng2022_GVAXNivo` to `Li2022_CancerCell_PDAC_AntiPD1`.

## Summary

Data across four papers spans three distinct axes:

1. **TME immune composition at ~2 weeks post-treatment** (Li 2022) — 18 calibration targets extracted
2. **Clinical outcomes (DFS/OS)** (Zheng 2021, Heumann 2023, Agarwal 2025) — not yet extracted, requires long-running scenario infrastructure
3. **Absolute CD8 density post-treatment** (Agarwal 2025) — per-patient cells/mm² data, requires new scenario YAML

Key reference values:
- GVAX alone (no Cy, resectable): mDFS 18.9 mo, mOS 34.2 mo (Zheng 2021, n=29)
- Cy-GVAX+nivo (resectable): mDFS 15.0 mo, mOS 27.0 mo (Heumann 2023, n=14)
- mFOLFIRINOX+Cy/GVAX/nivo+SBRT (BR): mOS 20.4 mo, CD8 267 cells/mm² (Agarwal 2025, n=13)
- mFOLFIRINOX+SBRT only (BR controls): CD8 200 cells/mm² (Agarwal 2025, n=21)

The TLA-specific immune data has limited utility without a TLA compartment in the model.
