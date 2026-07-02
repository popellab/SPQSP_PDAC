# Submodel Target Extraction: Chemotaxis EC50 Parameters

## Parameters

| Parameter | Current value | Units | Model role |
|-----------|--------------|-------|------------|
| `IFNg_50_CXCL9` | 0.05 nM | nanomolarity | EC50 in `H_IFNg_CXCL9 = IFNg/(IFNg_50_CXCL9 + IFNg)` gating CXCL9 production by C_total, Mac_M1, mcDC1 |
| `CXCL9_50_Trec` | 1.0 nM | nanomolarity | EC50 in `H_CXCL9_Trec = CXCL9/(CXCL9_50_Trec + CXCL9)` gating CD8, Th1, NK recruitment via CXCR3 |

## Literature Search Results

### IFNg_50_CXCL9 (IFNg EC50 for CXCL9/10 induction)

No paper reports a fitted EC50. Best candidates with dose-response data:

1. **Antonelli et al. 2010** (PMID 20299237) — Primary human thyrocytes, full dose-response curve for IFNg -> CXCL9 + CXCL10 by ELISA. Best candidate but behind paywall.
2. **Tager et al. 2008** (PMID 18046562, DOI 10.1007/s00384-007-0396-6) — Human colon epithelial cells, IFNg at 2/10/50 ng/mL (0.12/0.6/2.9 nM), CXCL10 ELISA. 3-point titration.
3. **Hardaker et al. 2004** (PMID 14597565) — Human airway smooth muscle, dose-dependent CXCL10 by IFNg.

Indirect constraint: IFNg receptor Kd ~ 0.1-1 nM; R&D Systems ED50 for antiviral activity 0.009-0.044 nM. Model value of 0.05 nM is plausible.

### CXCL9_50_Trec (CXCL9 EC50 for CXCR3 T cell recruitment)

Best candidates:

1. **Loetscher et al. 1998** (PMID 9842912) — CXCL9 Kd = 0.9-1.2 nM on CXCR3+ human T cells. Direct anchor.
2. **Piali et al. 1998** (PMID 9541591) — T cell adhesion onset <1 nM, max at 5-10 nM. Functional data.
3. **Smit et al. 2003** (PMID 12750173) — Human T cell transwell at 3/10/30/100 nM CXCL9. Dose-response figures.
4. **Dar & Knechtle 2007** (PMC2265907) — Full transwell dose-response curves for CXCL10/11, figures need digitization.
5. **Weng et al. 1998** (PMID 9660793) — CXCL9 Ki = 4.9 nM (competitive binding on CHO/CXCR3).

Synthesis: Kd range 0.9-5 nM for CXCL9 on CXCR3. Model value 1.0 nM at lower end, consistent with pooled CXCL9+CXCL10 signal.

## Recommended Extractions

### IFNg_50_CXCL9
- **Primary:** Antonelli et al. 2010 if full text accessible; otherwise Hardaker et al. 2004
- **Forward model:** `direct_fit` (hill curve), IFNg dose -> CXCL9/10 secretion
- **Additional QSP params:** k_CXCL9_sec (secretion rate) could be jointly constrained

### CXCL9_50_Trec
- **Primary:** Loetscher 1998 (binding Kd) + Piali 1998 (functional adhesion)
- **Forward model:** `direct_fit` (hill curve) if dose-response figure available; otherwise binding Kd as algebraic constraint
- **Note:** Binding Kd ~ EC50 for GPCR chemotaxis (within 2-5x typically)