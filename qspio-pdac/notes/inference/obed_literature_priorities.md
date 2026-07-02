# OBED Prior Tightening — Literature Search Priorities

Date: 2026-03-10
Source: `figures/obed/prior_tightening_gvax_nivo_neoadjuvant_zheng2022.csv`
Method: Filter trajectory grid to inner 50% of each parameter's prior, measure outcome variance reduction (ΔVar on log(V_final/V_baseline))

## Tier A — Well-studied basic immunology rates

These have high ΔVar, uninformed priors (σ≥0.8), and abundant basic science literature.

| Rank | Parameter | ΔVar | Prior CV | σ | What it is | Data source type |
|------|-----------|------|----------|---|-----------|-----------------|
| 1 | k_Th_pro_tumor | 46% | 95% | 0.8 | CD4 T cell proliferation rate in TME | Ki67/BrdU assays, CFSE dilution |
| 2 | k_APC_mature_ID | 46% | 131% | 1.0 | DC maturation rate (GM-CSF-driven) | In vitro DC culture kinetics |
| 5 | k_NK_kill | 37% | 80% | 0.7 | NK cell killing rate | Chromium release / impedance assays |
| 6 | k_vas_decay | 35% | 80% | 0.7 | Cancer cell death rate (natural) | Growth/death kinetic measurements |
| 7 | k_CD8_basal_exh | 34% | 131% | 1.0 | CD8 basal exhaustion rate | PD-1/TIM-3/LAG-3 kinetic studies |
| 8 | k_IL10_sec | 34% | 80% | 0.7 | IL-10 secretion rate | ELISA from sorted cell populations |
| 9 | k_vas_growth | 33% | 131% | 1.0 | Cancer growth rate | PDAC doubling time data |
| 13 | k_IL2_deg | 32% | 53% | 0.5 | IL-2 degradation rate | Protein half-life (very well characterized) |
| 14 | k_IL2_cons | 30% | 80% | 0.7 | IL-2 consumption rate | Receptor binding / consumption kinetics |
| 15 | k_C_CD8_exh | 30% | 131% | 1.0 | CD8 T cell killing rate | Killing assay literature |

## Tier B — Constrainable from PDAC IHC/flow or in vitro dose-response

| Rank | Parameter | ΔVar | Prior CV | σ | What it is | Data source type |
|------|-----------|------|----------|---|-----------|-----------------|
| 3 | IL10_50_NK | 40% | 131% | 1.0 | IL-10 EC50 for NK suppression | Dose-response (in vitro NK + IL-10) |
| 4 | k_Tcell_ECM | 38% | 131% | 1.0 | ECM effect on T cell motility | Collagen gel migration assays |
| 12 | k_APC_recruit0 | 32% | 131% | 1.0 | Baseline APC recruitment rate | DC density in normal vs tumor tissue |
| 16 | IL12_50_NK | 29% | 80% | 0.7 | IL-12 EC50 for NK activation | IL-12 dose-response on NK cytotoxicity |
| 17 | EC50_GMCSF | 29% | 131% | 1.0 | GM-CSF EC50 for DC effects | GM-CSF dose-response (pharmaceutical data) |
| 20 | k_qpsc_death | 26% | 53% | 0.5 | Quiescent PSC death rate | Stellate cell culture data |
| 21 | k_iCAF_death | 26% | 53% | 0.5 | iCAF death rate | CAF subtype turnover studies |
| 22 | DAMP_50 | 26% | 131% | 1.0 | DAMP EC50 for APC activation | HMGB1/ATP dose-response on DCs |
| 24 | n_CD8_clones | 25% | 80% | 0.7 | CD8 clonotype diversity | TCR sequencing (growing literature) |

## Tier C — Trafficking/compartment rates (harder)

| Rank | Parameter | ΔVar | Prior CV | σ | What it is | Data source type |
|------|-----------|------|----------|---|-----------|-----------------|
| 10 | q_nCD8_LN_in | 33% | 179% | 1.2 | Naive CD8 entry rate to LN | Lymphocyte trafficking studies |
| 18 | k_NK_rec | 28% | 131% | 1.0 | NK recruitment to tumor | TME-specific trafficking data |
| 19 | f_ECM_50_APC_mig | 27% | 131% | 1.0 | ECM half-max for APC migration block | Transwell/collagen gel + DC migration |
| 23 | k_CCR5_Treg_rec | 26% | 179% | 1.2 | CCR5-mediated Treg recruitment | CCR5/CCL5 chemotaxis assays |
| 25 | q_nTreg_LN_in | 24% | 179% | 1.2 | Naive Treg LN entry rate | Treg trafficking data |

## Search Strategy

Highest bang-for-buck targets (high ΔVar + uninformed prior + accessible data):

1. **k_IL2_deg** (ΔVar=32%, σ=0.5) — IL-2 half-life is one of the most measured cytokine parameters. σ could drop to ~0.2.
2. **k_vas_growth** (ΔVar=33%, σ=1.0) — PDAC doubling time is clinically reported. σ=1.0 is very uninformed for a well-studied cancer.
3. **EC50_GMCSF** (ΔVar=29%, σ=1.0) — Pharmaceutical-grade dose-response data exists (sargramostim).
4. **k_NK_kill** / **k_C_CD8_exh** (ΔVar=37%/30%) — Killing assay data is abundant across cancer types.
5. **k_APC_mature_ID** (ΔVar=46%, σ=1.0) — In vitro DC maturation with GM-CSF is a standard protocol.
6. **k_Th_pro_tumor** (ΔVar=46%, σ=0.8) — CD4 Ki67+ data in solid tumors constrains this.
7. **IL10_50_NK** (ΔVar=40%, σ=1.0) — IL-10 dose-response on NK function is published.
8. **k_Tcell_ECM** (ΔVar=38%, σ=1.0) — T cell motility in collagen gels is measurable.

## Search Results

### k_IL2_deg (ΔVar=32%) — DONE, SubmodelTarget written
- **Current**: 0.2/min (t½ ~3.5 min), σ=0.5
- **Literature**: Alpha-phase t½ = 5-7 min (Lotze1985), 12.9 min (Konrad1990). Beta t½ = 85 min.
- **Tissue estimate**: t½ ~5-30 min (no renal clearance, receptor-mediated consumption separate)
- **Recommended**: σ → 0.3, median 0.1/min (t½ ~7 min), 95% CI [0.055, 0.18]/min
- **SubmodelTarget**: `k_IL2_deg_PDAC_deriv001.yaml`
- **Citations**: Lotze1985 (PMID:3871099), Konrad1990 (PMID:2317789), Konrad1994 (PMID:7955769)

### k_vas_growth (ΔVar=33%) — NOT directly constrainable
- **k_vas_growth is vasculature carrying capacity growth rate**, not cancer cell growth rate
- PDAC doubling time data constrains k_C1_growth (already has SubmodelTarget from Jang2015)
- k_vas_growth lacks direct clinical measurements; PDAC is hypovascular
- No SubmodelTarget written — keep σ=1.0 or look for angiogenesis kinetic data

### EC50_GMCSF (ΔVar=29%) — DONE, SubmodelTarget written
- **Current**: 0.5 nM, σ=1.0
- **Literature**: Receptor Kd (high-affinity α+β) = 30-100 pM; proliferation EC50 = 5.5 pM (Guthridge2006); standard DC culture uses 0.7-1.4 nM (saturating)
- **Recommended**: σ → 0.5, median 0.5 nM, 95% CI [0.18, 1.35] nM
- **SubmodelTarget**: `EC50_GMCSF_PDAC_deriv001.yaml`
- **Citations**: Guthridge2006 (DOI:10.1038/sj.emboj.7600948), Bhattacharya2015 (DOI:10.1189/jlb.4HI0315-098R), Zhan2019 (DOI:10.3389/fimmu.2019.02679)

### k_APC_mature_ID (ΔVar=46%) — DONE, SubmodelTarget written
- **Current**: 0.7/day (t½ ~24h), σ=1.0
- **Literature**: Phenotypic maturation (CD80/86/83) ~24h; functional peak (IL-12) ~6h; in vivo cDC2 proliferation 0.75/day (Villani2024)
- **Recommended**: σ → 0.6, median 1.0/day (t½ ~17h), 95% CI [0.30, 3.3]/day
- **SubmodelTarget**: `k_APC_mature_ID_PDAC_deriv001.yaml`
- **Citations**: Dudek2024 (DOI:10.1186/s12967-024-05368-4), Jansen2010 (DOI:10.1097/CJI.0b013e3181c0b106), Villani2024 (DOI:10.1084/jem.20220867)

### k_NK_kill (ΔVar=37%) — DONE, SubmodelTarget written
- **Current**: 0.15/day, σ=0.7
- **Literature**: ~4 kills/active NK/16h (Bhat2007); ~50% of NK cells are non-killers (Prager2019); population average ~3 kills/day
- **Note**: In vitro data uses 721.221 (MHC-I⁻, highly susceptible); PDAC targets are less susceptible. Translation uncertainty is high.
- **Recommended**: σ → 0.5, median 0.15/day, 95% CI [0.055, 0.41]/day
- **SubmodelTarget**: `k_NK_kill_PDAC_deriv001.yaml`
- **Citations**: Bhat2007 (DOI:10.1371/journal.pone.0000326), Choi2013 (DOI:10.1073/pnas.1221312110), Prager2019 (DOI:10.3389/fimmu.2019.03010)

### k_CD8_basal_exh (ΔVar=34%) — TODO
- **Current**: 0.05/day, σ=1.0
- Baseline exhaustion from chronic TCR stimulation. Look for: PD-1/TIM-3/LAG-3 upregulation kinetics on chronically stimulated CD8 T cells. In vitro chronic stimulation → exhaustion timecourse data.

### k_C_CD8_exh (ΔVar=30%) — TODO
- **Current**: 0.1/day, σ=1.0
- This is T cell exhaustion rate by cancer cells (not killing rate). Related to tumor-induced T cell dysfunction kinetics. Different literature than killing assays.

### k_Th_pro_tumor (ΔVar=46%) — TODO
- **Current**: 0.15/day, σ=0.8
- CD4 T cell proliferation rate in TME. Look for: Ki67+ fraction of CD4+FoxP3- T cells in PDAC or other solid tumors, combined with cell cycle duration estimates.

### IL10_50_NK (ΔVar=40%) — TODO
- **Current**: σ=1.0. Look for: IL-10 dose-response curves on NK cell cytotoxicity or IFNg production.

### k_Tcell_ECM (ΔVar=38%) — TODO
- **Current**: σ=1.0. Look for: T cell motility measurements in collagen gels of varying density.

---

## 2026-03-12 — Updated rankings (IQR-based dispersion, literature-informed priors)

Source: `figures/obed/prior_tightening_gvax_nivo_neoadjuvant_zheng2022.csv`
Method: Filter to inner 50% of each parameter's prior, measure **IQR-based dispersion reduction** (`(IQR/1.349)^2`) on log(V_final/V_baseline). Replaces `np.var` which was dominated by ~30 heavy-tailed outliers (kurtosis=48.8).

### What changed

The variance→IQR fix and prior updates (4 SubmodelTarget parameters merged) reshuffled the rankings significantly:

- **k_NK_kill** dropped from #5 (ΔVar=+37%) to rank ~97 (ΔDisp=+2.4%) — confirmed as noise from stripped module
- **k_Th_pro_tumor** dropped from #1 (ΔVar=+46%) to #37 (ΔDisp=+15.7%) — still relevant, less dominant
- **k_APC_mature_ID** dropped from #2 (ΔVar=+46%) to negative ΔDisp — literature prior (σ=0.6) already well-constrained
- **IL10_50_NK** dropped from #3 (ΔVar=+40%) to negative ΔDisp — NK module stripped
- New top parameters reflect tumor-immune encounter mechanics and signaling

### Updated Tier A — Top priorities for literature search

| Rank | Parameter | ΔDisp | Prior CV | σ | What it is | Search feasibility |
|------|-----------|-------|----------|---|-----------|-------------------|
| 1 | v_T_search_volume | +33.4% | 131% | 1.0 | Tumor volume searched per T cell per unit time | Hard — biophysical parameter, needs motility + scanning data |
| 2 | k_vas_growth | +31.4% | 131% | 1.0 | Vasculature carrying capacity growth rate | Moderate — angiogenesis kinetics, but not directly clinical |
| 3 | k_IL2_sec | +30.1% | 80% | 0.7 | IL-2 secretion rate by T cells | Good — ELISA from stimulated T cell cultures |
| 4 | Q_nTreg_thym | +28.6% | 131% | 1.0 | Thymic output of naive CD4 (nTreg) | Good — TREC assays, age-dependent thymic output literature |
| 5 | k_vas_Csec | +28.0% | 80% | 0.7 | Cancer cell VEGF secretion rate | Moderate — VEGF ELISA from tumor cell lines |
| 6 | k_Tcell_ECM | +27.5% | 131% | 1.0 | ECM effect on T cell motility | Moderate — collagen gel migration assays |
| 7 | DAMP_50 | +26.9% | 131% | 1.0 | DAMP EC50 for APC activation | Moderate — HMGB1/ATP dose-response on DCs |
| 8 | k_TGFb_Tsec | +25.4% | 80% | 0.7 | TGFβ secretion by T cells | Good — ELISA from Treg/Th cultures |
| 9 | TGFb_50_APC | +25.3% | 80% | 0.7 | TGFβ EC50 for APC inhibition | Good — TGFβ dose-response on DC maturation |
| 10 | rho_Th_max | +25.0% | 95% | 0.8 | Max Th proliferation rate | Good — CFSE/Ki67 data on CD4+ T cells |

### Updated search strategy (bang-for-buck)

Highest-value targets combining ΔDisp, prior uninformedness (σ), and data accessibility:

1. **k_IL2_sec** (ΔDisp=+30.1%, σ=0.7) — IL-2 secretion rates from activated T cells are routinely measured by ELISA. PDAC-specific T cell activation data exists from neoadjuvant trial correlatives.
2. **Q_nTreg_thym** (ΔDisp=+28.6%, σ=1.0) — Thymic output is well-characterized by TREC assays (Douek 1998, den Braber 2012). Age-dependent decline is a major source of inter-patient variability in CD4 availability.
3. **k_TGFb_Tsec / TGFb_50_APC** (ΔDisp=+25.4%/+25.3%, σ=0.7) — TGFβ axis appears twice in top 10. Constraining both the secretion rate and the APC inhibition EC50 would compound to reduce uncertainty.
4. **rho_Th_max** (ΔDisp=+25.0%, σ=0.8) — CD4 T cell proliferation kinetics from CFSE dilution assays. Related to the old k_Th_pro_tumor target but parameterized differently.
5. **PD1_50** (ΔDisp=+21.9%, σ=0.7) — PD-1/PD-L1 binding EC50 is pharmaceutical-grade data (pembrolizumab/nivolumab dose-response). Very well characterized.

### Completed SubmodelTargets (from 2026-03-10 iteration)

These 4 parameters now have literature-informed priors merged into pdac_priors.csv:
- **EC50_GMCSF** — median shifted 0.5 → 0.003 nM (picomolar receptor Kd)
- **k_APC_mature_ID** — median shifted 0.7 → 2.73/day (DC maturation in ~8h)
- **k_IL2_deg** — median shifted 0.2 → 0.091/min (well-characterized PK)
- **k_NK_kill** — median shifted 0.15 → 5.5/day (stripped module, now noise)
