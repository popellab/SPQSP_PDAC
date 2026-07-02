# Submodel Target Extraction Priorities

Date: 2026-03-13
Sources: PRCC sensitivity (aggregate_parameter_ranking.csv), OBED MI prior tightening (obed_mi_results.md)
Scope: Parameters added since pdac_changelog.md (PDAC-specific, less likely to have literature-derived values)

## Method

Cross-referenced three data sources:
1. **PRCC rank** — global sensitivity of each parameter across all observables (lower = more influential)
2. **OBED prior tightening rank** — IQR-dispersion reduction on log(V_final/V_baseline) when prior is halved (lower = more outcome-predictive)
3. **pdac_changelog.md** — identifies ~135 parameters added during PDAC development (vs ~60 inherited from QSPIO)

Parameters already constrained by submodel targets: EC50_GMCSF, k_APC_mature_ID, k_IL2_deg, k_NK_kill.

---

## Tier 1 — High PRCC + OBED (extract first)

Both globally sensitive and outcome-predictive. Highest value per unit extraction effort.

| Param | PRCC | OBED | Mechanism | Extraction approach |
|-------|------|------|-----------|-------------------|
| f_stroma_max | 7 | 15 | Max stromal volume fraction in tumor | Histomorphometry from resected PDAC (% stroma by area) |
| TGFb_50_APC | 46 | 9 | TGFb concentration for half-max APC maturation inhibition | In vitro DC maturation ± TGFb dose-response |
| APC0_cDC1_T | 34 | 20 | Baseline cDC1 density in tumor | mIHC BATF3+ DC counts in treatment-naive PDAC |
| ECM_max | 69 | 13 | Maximum ECM density (collagen carrying capacity) | Hydroxyproline assay or picrosirius red quantification |
| f_ECM_50_APC_mig | 65 | 22 | ECM fraction for half-max APC migration inhibition | Transwell migration assays through collagen gels |
| DAMP_50 | 106 | 7 | DAMP level for half-max APC activation | HMGB1/calreticulin dose-response on DC maturation |

## Tier 2 — High PRCC only (globally sensitive, not outcome-specific)

Top 30 PRCC parameters. Constrain these to reduce overall model uncertainty, even if they don't specifically predict treatment response.

| Param | PRCC | Mechanism | Extraction approach |
|-------|------|-----------|-------------------|
| k_Treg_death | 2 | Treg death rate in tumor | Annexin V/Ki67 turnover assays on tumor Tregs |
| k_ECM_myCAF_sec | 3 | myCAF ECM secretion rate | Collagen deposition assays from sorted αSMA+ CAFs |
| k_iCAF_to_myCAF | 5 | iCAF→myCAF interconversion rate | Time-lapse phenotyping of CAF cultures ± TGFb |
| rho_Treg_max | 10 | Treg carrying capacity in tumor | Max Treg density from mIHC across PDAC cohorts |
| VEGF_norm_50 | 12 | VEGF threshold for angiogenic normalization | In vivo vessel normalization dose-response |
| k_myCAF_to_iCAF | 13 | myCAF→iCAF interconversion rate | Time-lapse phenotyping of CAF cultures ± IL-1/TNFα |
| k_myCAF_prolif | 14 | myCAF proliferation rate | Ki67+ fraction in αSMA+ CAFs from PDAC tissue |
| k_Treg_pro_tumor | 15 | Tumor-local Treg proliferation | Ki67+ Tregs in PDAC TME (20-30% reported) |
| k_TGFb_sec_myCAF | 17 | myCAF TGFb secretion rate | ELISA on conditioned media from sorted myCAFs |
| k_T_activation_per_APC | 18 | T cell activation rate per mature APC | In vitro co-culture killing/proliferation assays |
| k_ECM_deg | 25 | ECM degradation rate | MMP activity assays, collagen turnover measurements |
| CXCL12_50_Texcl | 26 | CXCL12 threshold for T cell exclusion | Transwell T cell migration ± CXCL12 dose-response |
| n_T_expansion_burst | 27 | Clonal expansion burst size | CFSE dilution or barcode tracking in vivo |
| f_IL6_iCAF | 30 | IL-6 autocrine feedback strength on iCAFs | IL-6R blockade effect on iCAF phenotype maintenance |

## Tier 3 — OBED only (high outcome influence, low bulk sensitivity)

These parameters strongly predict treatment response but don't dominate global PRCC (which averages across all observables). High value for decision-relevant calibration.

| Param | OBED | PRCC | Mechanism | Extraction approach |
|-------|------|------|-----------|-------------------|
| v_T_search_volume | 1 | 125 | T cell search volume (encounter rate) | Two-photon intravital imaging of T cell motility |
| k_Tcell_ECM | 6 | 187 | ECM-driven T cell exhaustion rate | T cell function assays in 3D collagen matrices |
| rho_Th_max | 10 | - | Th carrying capacity in tumor | Max Th density from mIHC across PDAC cohorts |
| k_apCAF_prolif | 16 | - | apCAF proliferation rate | Ki67 in MHC-II+ CAFs |
| Cy_50_Treg | 17 | 190 | Cyclophosphamide EC50 for Treg depletion | Clinical PK/PD: Treg counts vs metronomic Cy dose |
| k_IL12_sec | 19 | 157 | IL-12 secretion rate by mDCs | ELISA on stimulated DC conditioned media |
| XCL1_50_cDC1 | 23 | 151 | XCL1 threshold for cDC1 recruitment | XCL1 chemotaxis dose-response on BATF3+ DCs |
| f_nTreg | 25 | - | Fraction of naive CD4 committed to Treg lineage | Thymic Treg output ratio (nTreg/total nCD4) |

## Inherited QSPIO Parameters — Literature Audit

Three inherited parameters appeared in OBED top 10 and had papers in `to-review/`. Reviewed 2026-03-13.

### k_IL2_sec (PRCC #28, OBED #3) — DIVERSITY SCALING ARTIFACT

Current value: 3.0e-5 nmol/cell/day

| Source | Rate (nmol/cell/day) | Method |
|--------|---------------------|--------|
| Han et al. 2010 (PMID: 20478744) | ~1e-9 (median), ~7e-9 (high) | Single-cell microengraving, PMA/iono stim |
| Busse et al. 2010 (PMID: 20133624) | ~2e-10 | Modeling param calibrated to FACS data |

**Current value is ~30,000x higher than direct measurements.** This is almost certainly a diversity scaling artifact: the model tracks one clone out of ~1.1e6, so per-cell rates were inflated to produce reasonable IL-2 concentrations at clone-level species counts. If diversity scaling is added to IL2 secretion reactions (×div_CD8), k_IL2_sec should drop to ~1e-9 nmol/cell/day to match Han et al.

**Action**: Do not extract a submodel target at the current parameterization — it would conflict with the compensated rate. Revisit when diversity scaling is implemented as a structural change.

### k_TGFb_Tsec (PRCC #43, OBED #8) — MILD CONFLICT

Current value: 1.2e-10 nmol/cell/day

| Source | Rate (nmol/cell/day) | Method |
|--------|---------------------|--------|
| Stockis et al. 2009 (PMID: 19830734) | 4e-12 to 1.6e-11 | ELISA on stimulated Treg clone supernatants |

Current value is ~10-30x higher than Stockis ELISA data. However, ELISA only captures soluble latent TGFb — it misses GARP-bound surface TGFb (the biologically active fraction in Tregs). The gap is plausibly explained by this measurement limitation, plus potential diversity scaling inflation.

**Action**: Soft lower bound at best. Low priority for extraction — the uncertainty is too large to meaningfully tighten the prior.

### Q_nTreg_thym (OBED #4) — ADEQUATELY CONSTRAINED

Current value: 1e6 cell/day (prior σ=1.0)

| Source | Estimate | Method |
|--------|----------|--------|
| Bains et al. 2009 (PMID: 19179300) | ~1.2e7 total CD4/day at age 65 | TREC + math modeling, extrapolated |
| den Braber et al. 2012 (PMID: 22365666) | ~2-3.4e6 total CD4/day at age 65 | Deuterium labeling + modeling |

Neither paper provides Treg-specific output. Deriving: total CD4 output (3-12e6) × nTreg fraction (5-10%) gives ~1.5e5 to 1e6 Treg/day. Current value sits at the upper end of this range. Den Braber peripheral division parameters (K_nT_pro, k_nT_death) are already in the model.

**Action**: No extraction needed. Papers don't add constraint beyond what's already incorporated.

### rho_Th_max — PAPERS IN to-review/

Papers in `to-review/rho_Th_max/` (Karakhanova 2015, Wartenberg 2018) contain PDAC-specific IHC data on Th densities. This is a new PDAC-specific parameter (OBED #10) worth extracting. See Tier 3 above.

---

## Notes

- **Stripped modules**: f_HIF_glycolysis (OBED #12), k_glc_tumor (PRCC #31), k_lac_clear (PRCC #59), pO2_50 (PRCC #84) are from hypoxia/metabolism modules that are currently inactive. Their PRCC/OBED signal is noise from prior sampling. Skip these.
- **OBED measures outcome-specific influence** (log tumor volume ratio under GVAX+nivo), while **PRCC measures global sensitivity** across all 30+ observables. Parameters can be OBED-important but PRCC-invisible if they affect treatment response through a narrow pathway.
- **Hill function species** in the OBED MI sweep (H_TGFb_Teff, H_CXCL12_Texcl, etc.) point to which parameter *combinations* matter but aren't directly extractable. The underlying parameters (TGFb_50_APC, CXCL12_50_Texcl, etc.) are what to target.
- **Diversity scaling**: Multiple inherited parameters (k_IL2_sec, k_IL2_cons, k_TGFb_Tsec, k_IFNg_Tsec) have values inflated by ~1e4-1e6 to compensate for clone-level tracking. Submodel targets for these should wait until diversity scaling is implemented structurally. See cytokine_tcell_module.m header comment.
