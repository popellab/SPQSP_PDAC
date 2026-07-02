# IFNg CD8 T Cell Secretion Rate Literature Search

**Target parameter:** `k_IFNg_Tsec = 4e-12 nmol/cell/day`
**Current citations:** Autenshlyus 2021 (PMID 34059727), Borj 2017 (doi:10.15171/ijbms.2017.05)
**Goal:** Find papers with ELISA IFNg concentration + cell count + volume + time to back-derive per-cell rate
**Forward model type:** `batch_accumulation` — k = (conc_nM * vol_L) / (n_cells * time_days)
**IFNg MW = 17 kDa** → 1 ng/mL = 1 pg/uL = (1e-9 g/mL / 17000 g/mol) = 58.8 nM ≈ 0.059 nM per pg/mL

---

## Conversion formula

```
k [nmol/cell/day] = (conc_pg_mL * 1e-12 g/pg / (17000 g/mol) * 1e9 nmol/mol * vol_mL * 1e-3 L/mL)
                    / (n_cells * time_days)

Simplified:
k = conc_pg_mL * vol_mL / (n_cells * time_days * 17000 * 1e3)
  = conc_pg_mL * vol_mL / (n_cells * time_days * 1.7e7)

Or equivalently:
k [nmol/cell/day] = (conc_nM * vol_L) / (n_cells * time_days)
where conc_nM = conc_pg_mL * 0.05882  (1 pg/mL = 0.05882 nM for 17 kDa protein)
```

---

## Papers Reviewed

### 1. Biller et al. 2001 — BEST QUANTITATIVE SOURCE
**Title:** "Interferon-gamma secretion of peripheral blood CD8+ T lymphocytes in patients with bronchial asthma: In vitro stimulus determines cytokine production"
**Journal:** Clinical and Experimental Immunology
**DOI:** 10.1046/j.1365-2249.2001.01666.x
**PMID:** PMC1906207
**Source quality:** primary_human_in_vitro

**Experimental conditions:**
- Cell type: CD8+ T cells isolated by magnetic sorting (85 +/- 7% CD8+ purity)
- Cell concentration: 3 x 10^6 cells/mL
- Incubation: 24 hours at 37°C, 5% CO2
- Stimulants:
  - Anti-CD3 antibody OKT3, plate-coated, 20 ug/mL (overnight 4°C pre-coat)
  - PMA (10^-8 M) + calcium ionophore A23187 (10^-6 M)
- Culture volume per well: NOT EXPLICITLY STATED (only concentration given)

**IFNg concentrations measured by ELISA:**

| Condition | Asthma patients | Healthy controls |
|-----------|----------------|-----------------|
| Anti-CD3 (24h) | 359 +/- 51 pg/mL | 151 +/- 79 pg/mL |
| PMA/calcium-ionophore (24h) | 163 +/- 19 pg/mL | 352 +/- 104 pg/mL |

**Per-cell rate calculation (using 3e6 cells/mL, assuming 1 mL volume per well):**
- Anti-CD3, healthy controls: k = 151 pg/mL * 1 mL / (3e6 cells * 1 day * 1.7e7 g/mol * 1e-12 g/pg * 1e3 mL/L)
  = 151e-12 g/mL * 1e-3 L / (3e6 * 1 day * 17000 g/mol)
  = 151e-15 mol / (3e6 * 17000 * day)
  = 151e-15 / (5.1e10) nmol/cell/day
  ... let me redo this cleanly:

  conc = 151 pg/mL = 151e-12 g/mL
  vol = 1 mL (assumed; only density given)
  total IFNg in well = 151e-12 g
  in nmol = 151e-12 g / 17000 g/mol * 1e9 nmol/mol = 8.88e-6 nmol
  n_cells = 3e6 cells
  time = 1 day
  k = 8.88e-6 nmol / (3e6 cells * 1 day) = 2.96e-12 nmol/cell/day ~ **3e-12 nmol/cell/day**

- Anti-CD3, asthma patients: 359 pg/mL
  total = 359e-12 / 17000 * 1e9 = 21.1e-6 nmol
  k = 21.1e-6 / 3e6 = **7.0e-12 nmol/cell/day**

- PMA/ionophore, healthy controls: 352 pg/mL -> k = **6.9e-12 nmol/cell/day**
- PMA/ionophore, asthma patients: 163 pg/mL -> k = **3.2e-12 nmol/cell/day**

**Range: 3.0 – 7.0 x 10^-12 nmol/cell/day**
**Central estimate (healthy, anti-CD3): ~3e-12 nmol/cell/day**

CAVEAT: Culture volume is assumed 1 mL because only density (3x10^6 cells/mL) is given.
If actual volume is different (e.g., 200 uL in a 96-well plate at 3e6 cells/mL = 600,000 cells/well),
then k is proportionally lower: 0.2 mL * 3e-12 = **6e-13 nmol/cell/day** (lower bound, 96-well format).

Most likely interpretation: 3x10^6 cells/mL cultured at undisclosed volume; rate spans ~0.6 – 7e-12 nmol/cell/day
depending on assumed volume.

**Conclusion:** Consistent with current model value of 4e-12 nmol/cell/day. The key uncertainty is culture volume.

---

### 2. Autenshlyus et al. 2021 — CURRENT CITATION (PMID 34059727)
**Title:** "Cytokines in various molecular subtypes of breast cancer"
**Journal:** International Journal of Immunopathology and Pharmacology
**DOI:** 10.1177/20587384211034089
**PMC:** 8375341

**What they measured:**
- IFNg from tumor biopsies (tumor secretion) and peripheral blood cells
- Tumor: spontaneous 7.65–16.20 pg/mL; mitogen-induced 10.15–27.00 pg/mL
- Blood cells: 24h incubation, mitogen stimulation (PHA-P, PHA-M, ConA, LPS at 2-4 ug/mL)
- 1 mL whole blood + 4 mL DMEM-F12 medium = 5 mL total culture
- Detection limit: 2.0 pg/mL

**Assessment:** Reports whole blood cytokine production, NOT purified CD8+ T cells.
Cannot derive a clean per-cell CD8 secretion rate. The "blood cell" IFNg includes contributions from
CD4, CD8, NK cells, and potentially monocytes. Not directly usable for k_IFNg_Tsec.
The model citation appears to use this as supporting evidence that activated lymphocytes produce IFNg
at detectable levels (order of magnitude), not as a precise rate measurement.

**Notes:** A second related paper exists: Autenshlyus et al. 2021 Dokl Biochem Biophys (PMID 33895921)
on "Cytokine Production by Tumor Bioptate" — does not report IFNg.

---

### 3. Borj 2017 — CURRENT CITATION (doi:10.15171/ijbms.2017.05)
**Journal:** Iranian Journal of Basic Medical Sciences (IJBMS)
**DOI format note:** The DOI prefix 10.15171 is an older IJBMS format; current format is 10.22038/ijbms.
The article appears to be article 8086 in the IJBMS database (URL: ijbms.mums.ac.ir/article_8086).

**Status:** Could not access full text. PDF URL returned ECONNREFUSED. Web searches did not
reveal the article content, title, or quantitative IFNg data.

**Action needed:** User should locate this paper in Zotero or manually download from:
- https://ijbms.mums.ac.ir/article_8086.html
- Try CrossRef: https://doi.org/10.15171/ijbms.2017.05

---

### 4. Ngai et al. 2007 — CD4 vs CD8 IFNg secretion capacity
**Title:** "Gamma Interferon Responses of CD4 and CD8 T-Cell Subsets Are Quantitatively Different..."
**Journal:** Infection and Immunity
**DOI:** 10.1128/IAI.00024-07
**PMID:** 17307945
**Source quality:** primary_animal_in_vitro (mouse C57BL/6)

**Conditions:** 500,000 purified mouse CD8 T cells, 250 uL culture volume, 3-day incubation,
stimulated with infected APCs (Mycobacterium bovis BCG).

**Key finding:** CD4 T cells have "much greater IFNg secretion capacity than IFNg-positive CD8 T cells."
Calculated by: total IFNg (ELISA) / frequency of IFNg+ cells (ELISPOT) per million cultured cells.

**No absolute ng/mL values reported in text** (only figures). Cannot compute exact per-cell rate.
But useful for: (a) supporting that CD4 > CD8 for IFNg (consistent with r_IFNg=3), (b) confirming
the batch_accumulation methodology is appropriate.

---

### 5. Biller et al. 2001 (intracellular staining addendum)
Percentage IFNg+ cells (intracellular, 6h PMA/ionophore):
- Asthma: 29 +/- 8% CD8+ cells IFNg+
- Controls: 16 +/- 4% CD8+ cells IFNg+

This means only ~16% of "healthy" CD8 T cells are IFNg-secreting under polyclonal stimulation.
For antigen-specific effectors in tumor (model assumption: activated CD8 = effector), the fraction
would be higher. This supports using the anti-CD3 (more physiological TCR-like) rather than
PMA/ionophore values as the reference.

---

## Summary and Verdict

**Model value:** k_IFNg_Tsec = 4e-12 nmol/cell/day

**Best supported value from literature:**
- Biller 2001, anti-CD3, healthy controls: ~3e-12 nmol/cell/day (assuming 1 mL total volume)
- Biller 2001, PMA/ionophore, healthy controls: ~7e-12 nmol/cell/day
- Range across conditions and patients: 3 – 7e-12 nmol/cell/day

**Verdict:** The current model value of 4e-12 nmol/cell/day is in the center of the range
supported by Biller 2001. It is plausible but the culture volume assumption is the dominant
source of uncertainty.

**For SubmodelTarget extraction:**
- Biller 2001 is the best available source with all required fields except explicit volume.
  The missing volume (1 mL vs 96-well 200 uL format) introduces ~5x uncertainty.
- Autenshlyus 2021 and Borj 2017 cannot be used directly because:
  - Autenshlyus: whole blood, not purified CD8
  - Borj: inaccessible, content unknown

---

## Next Steps for Extraction

1. **Get Biller 2001 PDF** via Zotero: DOI 10.1046/j.1365-2249.2001.01666.x
   - Verify culture volume from Methods section
   - Extract table/figure values as direct_measurement inputs

2. **Get Borj 2017 PDF**: doi:10.15171/ijbms.2017.05
   - Check if it has explicit cell counts + volume + IFNg pg/mL

3. **Additional papers to consider:**
   - Sadiq et al. 2009 (PLOS NTD, leishmaniasis CD8) — has conditions but no absolute values
   - Corstjens 2008 (Clinical Biochemistry) — PBMC at 2e6 cells/mL in 200 uL, 6-day culture,
     reports threshold >100 pg/mL for positive response (antigen-specific, too dilute for rate)
   - Ngai 2007 (Infection and Immunity) — mouse data, values only in figures

4. **Good supplementary target:** If Biller 2001 Methods confirms volume = 1 mL (standard for T25
   flask or 24-well format at 3e6/mL density), this paper alone supports k ≈ 3e-12 nmol/cell/day,
   consistent with the current model default.
