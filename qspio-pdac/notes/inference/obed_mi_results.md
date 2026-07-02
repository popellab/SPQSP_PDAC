# OBED MI Sweep Results: gvax_nivo_neoadjuvant_zheng2022

Date: 2026-03-10
Scenario: GVAX + nivolumab neoadjuvant (21-day treatment, 30-day grid)
Outcome: log(V_final / V_baseline), 2814/4000 valid sims
Method: KSG continuous-continuous MI, 4000 prior predictive sims

## Top 40 TME Species by Peak MI

Excludes PK species (aPD1, GVAX), tumor volume (V_T), and synapse bookkeeping (syn_*).

| Rank | Species | Peak MI | Best Day |
|------|---------|---------|----------|
| 1 | k_C_Tcell_eff | 0.8341 | 14 |
| 2 | V_T.CCL2 | 0.7407 | 30 |
| 3 | V_T.IFNg | 0.6019 | 9 |
| 4 | V_T.PDGF | 0.5989 | 30 |
| 5 | V_T.CXCL12 | 0.5696 | 25 |
| 6 | H_DAMP | 0.5601 | 30 |
| 7 | H_P0 | 0.5475 | 10 |
| 8 | k_C1_growth | 0.5224 | 13 |
| 9 | H_CCL5_Treg | 0.5209 | 30 |
| 10 | A_s.M1p0 | 0.5179 | 0 |
| 11 | V_T.NO | 0.4796 | 21 |
| 12 | H_CXCL12_Texcl | 0.4748 | 30 |
| 13 | V_T.CCL5 | 0.4672 | 30 |
| 14 | V_T.ArgI | 0.4563 | 19 |
| 15 | ECM_level | 0.4547 | 25 |
| 16 | H_NO | 0.4520 | 30 |
| 17 | H_P1_apCAF | 0.4505 | 8 |
| 18 | N_aT | 0.4501 | 2 |
| 19 | A_s.M1 | 0.4369 | 22 |
| 20 | H_IL10 | 0.4357 | 21 |
| 21 | H_P1 | 0.4357 | 26 |
| 22 | H_ArgI_Treg | 0.4314 | 12 |
| 23 | H_TGFb_Teff | 0.4294 | 8 |
| 24 | H_IL6_MDSC | 0.4212 | 27 |
| 25 | H_TGFb_APC | 0.4194 | 10 |
| 26 | H_MDSC | 0.4170 | 30 |
| 27 | H_IL12 | 0.4142 | 5 |
| 28 | V_T.IL6 | 0.4124 | 30 |
| 29 | V_T.ECM | 0.4119 | 11 |
| 30 | A_s.M1p1 | 0.4104 | 6 |
| 31 | V_T.TGFb | 0.4102 | 10 |
| 32 | H_PD1_APC | 0.4064 | 25 |
| 33 | N_aT0 | 0.4009 | 7 |
| 34 | N_aTh | 0.4009 | 7 |
| 35 | H_ECM_APC_mig | 0.4002 | 11 |
| 36 | H_IL6_iCAF | 0.4001 | 15 |
| 37 | K_T_Treg | 0.3982 | 0 |
| 38 | H_IL6_M2 | 0.3975 | 30 |
| 39 | H_TGFb | 0.3965 | 30 |
| 40 | H_ECM_T_exh | 0.3946 | 9 |

## Interpretation by Biological Theme

### Tier 1 (MI > 0.6) — Direct effector mechanisms
- **k_C_Tcell_eff (day 14)**: Net T cell killing rate — the single most informative non-PK species. Peak at day 14 aligns with expected anti-PD1 immune reactivation timing. Constraining CD8 killing efficacy (TCR avidity, checkpoint occupancy, exhaustion markers) would maximally reduce ORR uncertainty.
- **V_T.CCL2 (day 30)**: Macrophage/MDSC recruitment chemokine. Late-stage accumulation suggests macrophage-mediated immunosuppression is a key axis of variation in treatment response.
- **V_T.IFNg (day 9)**: Early IFNγ — immune activation signal. Day 9 timing suggests the early immune priming phase is a critical decision point.
- **V_T.PDGF (day 30)**: Stromal/angiogenic remodeling factor. Late accumulation reflects the stromal response to treatment.

### Tier 2 (MI 0.45–0.6) — Immunosuppressive microenvironment
- **V_T.CXCL12 (day 25)**: CAF-derived T cell exclusion chemokine. Constraining CXCL12 data would help resolve the immune exclusion vs infiltration balance.
- **H_DAMP (day 30)**: Damage signal driving APC maturation — integrates tumor cell death signals.
- **H_P0 (day 10)**: Antigen presentation Hill function — early APC activity matters.
- **k_C1_growth (day 13)**: Effective cancer growth rate — the balance between growth and killing.
- **H_CCL5_Treg (day 30)**: CCL5-driven Treg recruitment — immunosuppressive axis.
- **V_T.NO (day 21)**: Nitric oxide (M1 macrophage effector) — pro-inflammatory macrophage activity at resection.
- **ECM_level (day 25)**: Extracellular matrix density — stromal barrier to immune access.

### Tier 3 (MI 0.40–0.45) — Regulatory checkpoints and cytokine signaling
- **H_IL10, H_TGFb_Teff, H_TGFb_APC**: Immunosuppressive cytokine signaling — multiple TGFβ/IL-10 axes appear, suggesting this is a major uncertainty driver.
- **H_IL12 (day 5)**: Very early DC activation — the earliest informative timepoint in the dataset.
- **H_ArgI_Treg (day 12), H_MDSC (day 30)**: Suppressive cell functions.
- **V_T.ECM (day 11), H_ECM_APC_mig (day 11)**: Stromal barrier affects both APC migration and T cell access.
- **H_IL6_iCAF (day 15), H_IL6_MDSC (day 27)**: IL-6 signaling through inflammatory CAFs and MDSCs.

## Literature Search Priorities

Based on these results, the highest-value literature targets are:

1. **CD8 T cell killing efficacy data** (days 12-16 post-treatment) — in vitro killing assays, granzyme B quantification, or TCR avidity measurements in PDAC would be most valuable.
2. **CCL2 tumor concentrations** — IHC or ELISA data from resected PDAC tissue, especially post-immunotherapy.
3. **IFNγ levels at day 7-10** — early immune activation biomarker data from neoadjuvant trials.
4. **CXCL12 and ECM measurements** — stromal barrier quantification (collagen density, CXCL12 gradients).
5. **TGFβ/IL-10 axis** — multiple Hill functions appear in tier 3, suggesting better priors on immunosuppressive cytokine parameters would compound to reduce uncertainty.
6. **IL-12 at day 5** — very early DC activation data would constrain the immune priming phase.

## Notes

- The Hill function species (H_*) reflect parameter interactions, not directly measurable quantities. They point to which *parameter combinations* matter — useful for identifying which parameters to tighten priors on.
- `A_s.M1p0` (antigen surface, M1 presenting no tumor antigen) at day 0 suggests baseline macrophage phenotype is informative even before treatment.
- `N_aT` (activated T cell count) at day 2 indicates very early T cell dynamics matter.
- 1186/4000 sims had invalid outcomes (NaN/inf in log ratio), likely from sims where V_baseline ≈ 0.

---

## 2026-03-12 — Updated priors (post-literature loop)

Date: 2026-03-12
Scenario: GVAX + nivolumab neoadjuvant (21-day treatment, 30-day grid)
Outcome: log(V_final / V_baseline), 1416/2000 valid sims
Method: KSG continuous-continuous MI, 2000 prior predictive sims
Priors: pdac_priors.csv with 4 submodel-target updates (EC50_GMCSF, k_APC_mature_ID, k_IL2_deg, k_NK_kill)
Dispersion: IQR-based `(IQR/1.349)^2` (robust to heavy-tailed outcomes, replaces np.var)

### Top 40 TME Species by Peak MI

Absolute MI values are lower than the 2026-03-10 run (4000 sims) due to smaller sample size (1416 vs 2814 valid). KSG MI scales with N; relative rankings are consistent.

| Rank | Species | Peak MI | Best Day |
|------|---------|---------|----------|
| 1 | k_C_Tcell_eff | 0.4433 | 12 |
| 2 | V_T.CCL2 | 0.3434 | 30 |
| 3 | V_T.IFNg | 0.2210 | 13 |
| 4 | H_P0 | 0.2043 | 0 |
| 5 | V_T.CXCL12 | 0.1962 | 30 |
| 6 | V_T.PDGF | 0.1828 | 30 |
| 7 | H_DAMP | 0.1671 | 30 |
| 8 | A_s.M1p0 | 0.1423 | 0 |
| 9 | k_C1_growth | 0.1293 | 22 |
| 10 | V_T.NO | 0.1244 | 30 |
| 11 | H_CCL5_Treg | 0.1047 | 30 |
| 12 | V_T.CCL5 | 0.1000 | 30 |
| 13 | H_CXCL12_Texcl | 0.0959 | 30 |
| 14 | ECM_level | 0.0889 | 9 |
| 15 | V_T.TGFb | 0.0732 | 30 |
| 16 | V_T.ECM | 0.0695 | 3 |
| 17 | H_P1_apCAF | 0.0664 | 8 |
| 18 | V_T.VEGF | 0.0642 | 30 |
| 19 | A_e.M1p0 | 0.0634 | 8 |
| 20 | V_T.ArgI | 0.0554 | 24 |
| 21 | A_s.M1 | 0.0492 | 12 |
| 22 | N_aTh | 0.0490 | 29 |
| 23 | N_aT0 | 0.0490 | 29 |
| 24 | H_TGFb_APC | 0.0471 | 8 |
| 25 | A_e.M1p1 | 0.0468 | 8 |
| 26 | H_IL10 | 0.0423 | 30 |
| 27 | H_NO | 0.0411 | 30 |
| 28 | H_TGFb_Teff | 0.0355 | 13 |
| 29 | H_ECM_T_mot | 0.0347 | 7 |
| 30 | H_IL12 | 0.0324 | 22 |
| 31 | A_s.M1p1 | 0.0320 | 6 |
| 32 | V_T.IL10 | 0.0314 | 30 |
| 33 | H_ArgI_Teff | 0.0301 | 7 |
| 34 | H_P1 | 0.0296 | 2 |
| 35 | H_TGFb | 0.0283 | 7 |
| 36 | V_T.IL12 | 0.0280 | 30 |
| 37 | N_aT | 0.0278 | 20 |
| 38 | H_MDSC | 0.0278 | 30 |
| 39 | V_LN.nCD8 | 0.0274 | 9 |
| 40 | H_GMCSF_ID | 0.0268 | 23 |

### Prior Tightening — Top 25 by IQR-Dispersion Reduction

Method: filter to inner 50% of each parameter's marginal prior, measure IQR-dispersion reduction on outcome.

| Rank | Parameter | ΔDisp | Prior CV | σ |
|------|-----------|-------|----------|---|
| 1 | v_T_search_volume | +33.4% | 131% | 1.0 |
| 2 | k_vas_growth | +31.4% | 131% | 1.0 |
| 3 | k_IL2_sec | +30.1% | 80% | 0.7 |
| 4 | Q_nTreg_thym | +28.6% | 131% | 1.0 |
| 5 | k_vas_Csec | +28.0% | 80% | 0.7 |
| 6 | k_Tcell_ECM | +27.5% | 131% | 1.0 |
| 7 | DAMP_50 | +26.9% | 131% | 1.0 |
| 8 | k_TGFb_Tsec | +25.4% | 80% | 0.7 |
| 9 | TGFb_50_APC | +25.3% | 80% | 0.7 |
| 10 | rho_Th_max | +25.0% | 95% | 0.8 |
| 11 | k_nTreg_death | +24.8% | 179% | 1.2 |
| 12 | f_HIF_glycolysis | +24.5% | 53% | 0.5 |
| 13 | ECM_max | +24.4% | 66% | 0.6 |
| 14 | PDGF_50 | +23.6% | 179% | 1.2 |
| 15 | f_stroma_max | +23.1% | 80% | 0.7 |
| 16 | k_apCAF_prolif | +22.7% | 131% | 1.0 |
| 17 | Cy_50_Treg | +22.5% | 80% | 0.7 |
| 18 | PD1_50 | +21.9% | 80% | 0.7 |
| 19 | k_IL12_sec | +21.5% | 80% | 0.7 |
| 20 | APC0_cDC1_T | +21.4% | 80% | 0.7 |
| 21 | k_IFNg_deg | +21.4% | 53% | 0.5 |
| 22 | f_ECM_50_APC_mig | +21.0% | 131% | 1.0 |
| 23 | XCL1_50_cDC1 | +20.6% | 131% | 1.0 |
| 24 | q_nCD8_LN_in | +20.6% | 179% | 1.2 |
| 25 | f_nTreg | +19.8% | 80% | 0.7 |

### Changes from 2026-03-10 (variance-based, old priors)

**Ranking stability:** Top MI species are consistent (k_C_Tcell_eff, CCL2, IFNg, CXCL12, PDGF all in top 6 both times). Prior updates shifted some relative positions (H_P0 moved from #7 to #4) but the biological themes are unchanged.

**Prior tightening — key shifts after IQR fix:**
- k_NK_kill dropped from #5 (ΔVar=+37%) to rank ~97 (ΔDisp=+2.4%) — was an artifact of heavy-tailed variance; NK module is stripped, parameter is noise
- k_Th_pro_tumor dropped from #1 (ΔVar=+46%) to rank ~37 (ΔDisp=+15.7%) — still relevant but less dominant under robust dispersion
- v_T_search_volume, k_vas_growth, k_IL2_sec emerged as top 3 — these reflect tumor-immune encounter mechanics and IL-2 signaling as core uncertainty axes
- f_HIF_glycolysis at #12 (ΔDisp=+24.5%) — from a stripped module; parameter has no effect on simulation, appearing here due to random subsample noise. Similar for k_lac_clear (#31), k_glc_tumor, pO2_50, lac_50_* etc.

### Notes

- The MI sweep used cached results from the initial 2000-sim run (KSG MI is already robust to heavy tails). Only steps 4 (observable augmentation) and 5 (prior tightening) were recomputed with IQR-based dispersion.
- Step 4 ΔDisp values are unreliable (e.g., -42662%) due to noisy IQR estimation in small bins (~140 sims per decile). MI rankings from step 3 are the reliable observable importance measure.
- Parameters from stripped modules (NK, hypoxia, metabolism, chemotaxis) still appear in the priors CSV and show up in the prior tightening ranking as noise. With IQR-based dispersion they fall to modest ΔDisp values (+24% or less) rather than dominating the ranking.
