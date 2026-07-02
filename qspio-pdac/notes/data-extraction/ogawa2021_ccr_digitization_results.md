# Ogawa et al. 2021 — Stromal Heterogeneity in Human Pancreatic Cancer

**Full title:** Three Distinct Stroma Types in Human Pancreatic Cancer Identified by Image Analysis of Fibroblast Subpopulations and Collagen
**Authors:** Ogawa Y, Masugi Y, Abe T, Yamazaki K, Ueno A, Fujii-Nishimura Y, Hori S, Yagi H, Abe Y, Kitago M, Sakamoto M
**Journal:** Clinical Cancer Research 27(1):107-119, 2021
**DOI:** 10.1158/1078-0432.CCR-20-2298
**PDF:** `ccr.pdf`

## Study Design

- n=215 treatment-naive PDACs from 212 patients (Keio University Hospital, 1991-2016)
- Excluded: 47 preoperative chemo/chemoradiotherapy, 17 PDAC variants, 14 poor staining
- Whole-tissue multiplex fluorescence IHC (ACTA2 + FAP + h-caldesmon) + van Gieson collagen stain
- Pixel-by-pixel Naive Bayes classification of desmoplastic elements
- Total tumor area examined: 1.6×10⁴ mm² (average 75.4 mm² per tumor)
- Noncancerous pancreas reference: n=147 slides

## Key Quantitative Data

### Desmoplastic element proportions (n=215, % of desmoplastic area)

These are proportions of the desmoplastic compartment only (collagen + ACTA2-fib + FAP-fib + overlap = 100%), NOT of total tumor area.

| Element | Mean | Median | IQR | Model species |
|---|---|---|---|---|
| Collagen | 53.4% | 54.1% | 45.6–62.2% | V_T.ECM |
| ACTA2-dominant fibroblasts | 33.4% | 30.6% | 24.2–41.1% | V_T.myCAF |
| FAP-dominant fibroblasts | 13.0% | 12.1% | 8.2–17.4% | V_T.iCAF |
| ACTA2/FAP overlap | 0.2% | 0.1% | 0.05–0.2% | — |

**Source:** Results section, paragraph 1 of "Three distinct stroma types..."

### Noncancerous pancreas reference (n=147, occupancy rates)

| Element | Mean | Median | IQR |
|---|---|---|---|
| Collagen | 19.9% | 13.4% | 8.1–27.0% |
| ACTA2-dominant fibroblasts | 11.4% | 7.0% | 3.0–15.0% |
| FAP-dominant fibroblasts | 0.2% | 0.1% | 0.04–0.2% |

**Source:** Methods section, "Noncancerous areas" paragraph

### Stroma type distribution (n=215)

| Stroma type | n | Description |
|---|---|---|
| C-stroma (collagen-rich) | 55 (25.6%) | Highest collagen proportion |
| F-stroma (FAP-dominant fibroblast-rich) | 95 (44.2%) | Highest FAP-fib proportion (mean 18.8% vs 8.9% and 7.9% in C/A) |
| A-stroma (ACTA2-dominant fibroblast-rich) | 65 (30.2%) | Highest ACTA2-fib proportion |

### Tumor cellularity

- Low (≤30%): 158/215 (73%)
- High (>30%): 57/215 (27%)
- Implies stromal fraction ≥70% in ~73% of treatment-naive PDACs

### apCAF (CD74+ fibroblasts)

- Described as "extremely limited in number" (Supplementary Fig. S8)
- CD74-expressing stromal cells coexpressed FAP
- No quantitative percentage reported; likely <1% of total CAFs

## Derived Calibration Values

### CAF subtype fractions (fibroblasts only, excluding collagen)

Denominator = ACTA2-fib + FAP-fib + overlap = 33.4 + 13.0 + 0.2 = 46.6%

| CAF subtype | Fraction of total CAFs | Model mapping |
|---|---|---|
| myCAF (ACTA2-dominant) | 33.4/46.6 = **71.7%** | V_T.myCAF / (V_T.iCAF + V_T.myCAF + V_T.apCAF) |
| iCAF (FAP-dominant) | 13.0/46.6 = **27.9%** | V_T.iCAF / (V_T.iCAF + V_T.myCAF + V_T.apCAF) |
| apCAF | ~0.4% (estimated from "extremely limited") | V_T.apCAF / (V_T.iCAF + V_T.myCAF + V_T.apCAF) |

### Collagen fold-change (tumor vs normal)

- Tumor collagen occupancy: median 54.1% of desmoplastic area (but need raw occupancy rates, not proportions)
- Normal collagen occupancy: median 13.4% of tissue area
- These are not directly comparable denominators — tumor proportions are within desmoplasia, normal are of tissue area

### PSC activation ratio proxy

- In noncancerous pancreas: FAP-fib essentially absent (0.1%), ACTA2-fib present (7.0%)
- In PDAC: FAP-fib dramatically elevated (12.1%), ACTA2-fib elevated (30.6%)
- Implies near-complete PSC activation in established PDAC (consistent with literature: >90% PSCs are activated in PDAC)

## Mapping Considerations

### ACTA2-dominant → myCAF
- Strong match: αSMA-high is the defining marker for myCAF in Öhlund 2017 and Elyada 2019
- Paper confirms ACTA2-dominant fibroblasts are juxtacrine (close to well-differentiated glands), ECM-producing
- h-caldesmon exclusion ensures smooth muscle cells are NOT counted

### FAP-dominant → iCAF
- Reasonable match: FAP+ fibroblasts in this paper are paracrine, associated with poorly differentiated areas
- FAP is expressed by both iCAFs and some myCAFs in scRNA-seq data, but this paper's pixel classification separates them by dominant marker
- iCAF assignment is approximate — FAP-dominant cells may include some activated myCAFs

### apCAF
- This paper does NOT quantify apCAF as a separate subtype
- CD74+ (MHC-II+) stromal cells noted as "extremely limited" and coexpressing FAP
- For the model, apCAF can be set to ~0.5-2% based on Elyada 2019 KPC data (10.3±1.25%) scaled down for human tissue

### Collagen → ECM
- Collagen comprises >90% of ECM in PDAC stroma (cited in paper, ref 29)
- van Gieson stain measures collagen area fraction, a reasonable proxy for V_T.ECM level

### Denominator caveat
- The "proportion of desmoplastic elements" is normalized to collagen + fibroblasts only
- It does NOT include immune cells, vasculature, or tumor cells
- The raw "occupancy rates" (area fractions of total tumor tissue) would be more directly useful but are not reported as summary statistics — they are shown in scatter plots (Supplementary Fig. S3)

## Figures with Digitizable Data

- **Fig. 2B** — Bar chart: 215 tumors showing landscape of desmoplastic proportions (collagen, ACTA2-fib, FAP-fib). Stacked bars for each patient.
- **Fig. 2D** — Box plots: Proportions of each desmoplastic element by stroma type. Has individual data points.
- **Supplementary Fig. S3** — Scatter plots: Pairwise correlations of occupancy rates (raw area fractions, not proportions). These would give the actual tissue-level area fractions.
- **Fig. 5B** — CD8+ density in tumor center vs margin (cells/mm²). Already have CD8 targets from other sources.
- **Fig. 5C** — Scatter: FAP-fib proportion vs CD8 center/margin ratio. Not directly useful as a target.

## Planned Calibration Target YAMLs

### Target 1: `mycaf_fraction_of_caf`

**What it constrains:** The balance between myofibroblastic and inflammatory CAF phenotypes, which determines ECM production vs cytokine secretion in the fibroblast module.

**Observable:**
```python
def compute_observable(time, species_dict, constants, ureg):
    mycaf = species_dict['V_T.myCAF']
    total_caf = species_dict['V_T.iCAF'] + species_dict['V_T.myCAF'] + species_dict['V_T.apCAF']
    return (mycaf / total_caf).to('dimensionless')
```

**Empirical data:**
- Derived from ACTA2-dominant fibroblast proportion: mean 33.4%, median 30.6%, IQR 24.2–41.1%
- Renormalized to fibroblast-only denominator (excluding collagen): 33.4 / (33.4 + 13.0 + 0.2) = 71.7%
- Median: 0.717, CI95 approximated from IQR: Q1 = 24.2/46.6 = 0.519, Q3 = 41.1/46.6 = 0.882
- Note: IQR renormalization is approximate since each patient's denominator differs slightly

**Distribution code approach:** Use lognormal or beta fit from median + IQR. With n=215, IQR-to-CI95 conversion via normal approximation: σ ≈ IQR / 1.35, CI95 ≈ median ± 1.96σ.

**Key assumptions:**
- ACTA2-dominant fibroblasts (αSMA-high, h-caldesmon-negative) correspond to model myCAFs
- h-caldesmon exclusion removes smooth muscle contamination
- The small ACTA2/FAP overlap population (0.2%) is ignored

**Mapping quality:** Strong. αSMA is the canonical myCAF marker (Öhlund 2017, Elyada 2019). Paper confirms juxtacrine positioning and ECM association.

---

### Target 2: `icaf_fraction_of_caf`

**What it constrains:** Inflammatory CAF abundance relative to total CAFs, controlling IL-6/CCL5 secretion strength in the fibroblast module.

**Observable:**
```python
def compute_observable(time, species_dict, constants, ureg):
    icaf = species_dict['V_T.iCAF']
    total_caf = species_dict['V_T.iCAF'] + species_dict['V_T.myCAF'] + species_dict['V_T.apCAF']
    return (icaf / total_caf).to('dimensionless')
```

**Empirical data:**
- Derived from FAP-dominant fibroblast proportion: mean 13.0%, median 12.1%, IQR 8.2–17.4%
- Renormalized: 13.0 / 46.6 = 27.9%
- Median: 0.279, Q1 = 8.2/46.6 = 0.176, Q3 = 17.4/46.6 = 0.373

**Distribution code approach:** Same IQR-based approach as mycaf_fraction_of_caf.

**Key assumptions:**
- FAP-dominant fibroblasts correspond to model iCAFs
- FAP is expressed by both iCAFs and some myCAFs in scRNA-seq, but pixel-level dominant-marker classification separates them reasonably
- Paper shows FAP-dominant fibroblasts are paracrine, associated with poorly differentiated areas and restricted CD8 infiltration — consistent with iCAF inflammatory/immunosuppressive role

**Mapping quality:** Moderate. FAP is not exclusive to iCAFs but dominant-marker classification provides reasonable separation. The paper's functional characterization (CD8 exclusion, neutrophil recruitment) aligns with iCAF biology.

---

### Target 3: `collagen_fraction_of_stroma`

**What it constrains:** The proportion of ECM (collagen) vs cellular fibroblasts within the desmoplastic compartment. Constrains ECM production/degradation balance in the fibroblast module.

**Observable:**
```python
def compute_observable(time, species_dict, constants, ureg):
    ecm = species_dict['V_T.ECM']
    # ECM is in milligram units; CAFs are in cell units — these are not directly comparable
    # This target would need a unit-conversion constant to make the ratio meaningful
    # ALTERNATIVE: treat as a constraint on the steady-state ECM level relative to a reference
    ...
```

**Problem:** The model's V_T.ECM is in mass units (milligram), while CAFs are in cell counts. The paper's collagen proportion (54.1% of desmoplastic area) is an area fraction. There's no clean way to compute a dimensionless collagen/(collagen+fibroblasts) ratio from model species without an arbitrary area-per-cell or area-per-mg conversion factor.

**Decision:** Skip this as a direct calibration target. The collagen proportion is useful context (stroma is ~half collagen, ~half cells) but the unit mismatch makes it hard to map. ECM dynamics are better constrained indirectly through the CAF subtype balance + downstream effects (T cell exclusion via H_ECM_APC_mig).

---

### Target 4: `psc_activation_ratio`

**What it constrains:** Fraction of PSCs that are activated (differentiated into CAF subtypes) vs quiescent. Maps to (iCAF + myCAF + apCAF) / (iCAF + myCAF + apCAF + qPSC).

**Observable:**
```python
def compute_observable(time, species_dict, constants, ureg):
    activated = species_dict['V_T.iCAF'] + species_dict['V_T.myCAF'] + species_dict['V_T.apCAF']
    total = activated + species_dict['V_T.qPSC']
    return (activated / total).to('dimensionless')
```

**Empirical data:**
- Not directly reported in this paper as a single number
- Inferred from: normal pancreas has FAP-fib at 0.1% (essentially zero iCAFs) while PDAC has massive CAF expansion
- Literature consensus: >90% of PSCs are activated in established PDAC (Apte 2004, Erkan 2012)
- Reasonable point estimate: median ~0.95, CI95 ~[0.85, 0.99]

**Distribution code approach:** Use a beta distribution centered at 0.95 with bounds [0.85, 0.99] reflecting literature consensus rather than a single paper's data.

**Key assumptions:**
- qPSC (quiescent PSCs) are rare in established PDAC but not entirely absent
- The model's qPSC represents vitamin-A-storing, GFAP+, αSMA-negative stellate cells
- This target should be at diagnosis (baseline scenario), not post-treatment

**Mapping quality:** Moderate. No single paper gives an exact activated/total PSC ratio in human resections. The estimate is a literature synthesis. Consider marking as `secondary_data_sources` with multiple refs rather than a single primary source.

---

### Targets NOT built from this paper

**apcaf_fraction_of_caf:** Paper says CD74+ stromal cells are "extremely limited in number" but gives no percentage. Would need Elyada 2019 KPC data (10.3% by flow cytometry in mice) as primary source, with a translation uncertainty fold of ~3-5x given cross-species and method differences. Defer until we have a human-data source or decide to use mouse data with explicit translation caveat.

**stromal_fraction:** Would need raw occupancy rates (total desmoplastic area / total tumor area), which are in Supp Fig S3 scatter plots but not tabulated. The tumor cellularity data (≤30% in 73% of cases) gives an indirect bound but is too coarse for a calibration target. Skip for now.

**caf_density (cells/mm²):** Paper reports area fractions, not cell counts. No cell density data. Would need a different source (e.g., Liudahl 2021 if it reports fibroblast densities).

**collagen_content_total:** Same unit-mismatch issue as collagen_fraction_of_stroma. Skip.