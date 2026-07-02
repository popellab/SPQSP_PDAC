# Unconstrained Parameter Audit

Date: 2026-03-25

Parameters in `pdac_priors.csv` that have **neither** a submodel target **nor** membership in a hierarchical parameter group. These are varied by SBI but have no literature-derived constraint beyond their prior. Ordered by sigma (most uncertain first), grouped by module.

**Scope:** 217 total params in priors CSV. 31 have submodel targets, 15 are in hierarchical groups (some overlap). **171 unconstrained (79%).**

---

## sigma = 1.2 (13 params) — Highest priority

| Parameter | Units | Module | Notes |
|-----------|-------|--------|-------|
| `k_Treg_pro` | 1/day | T cell | Treg proliferation in tumor |
| `k_Treg_death` | 1/day | T cell | Treg death rate |
| `q_CD8_T_in` | 1/day | T cell | CD8 T cell tumor infiltration rate |
| `q_Treg_T_in` | 1/day | T cell | Treg tumor infiltration rate |
| `q_nCD8_LN_in` | 1/day | T cell | Naive CD8 LN infiltration |
| `q_nCD8_P_out` | 1/day | T cell | Naive CD8 peripheral egress |
| `q_nTreg_LN_in` | 1/day | T cell | Naive Treg LN infiltration |
| `q_nTreg_P_out` | 1/day | T cell | Naive Treg peripheral egress |
| `n_CD4_clones` | dimensionless | T cell | CD4 clone diversity (denominator in APC Hill functions) |
| `k_MMP_M1` | mL/(cell*day) | Fibroblast | M1 macrophage MMP-mediated ECM degradation |
| `k_CCR5_Treg_rec` | 1/day | CCL5/Treg | CCR5-mediated Treg recruitment by iCAF CCL5 |
| `K_apCAF_Treg` | cell | Fibroblast | apCAF carrying capacity for Treg induction |
| `P1_50_apCAF` | molarity | Fibroblast | Antigen EC50 for apCAF uptake |

**Extractability notes:**

- `k_Treg_pro` / `k_Treg_death`: In vitro Treg proliferation/survival data exists (CFSE assays, IL-2 withdrawal). Reasonable candidates.
- `q_CD8_T_in` / `q_Treg_T_in`: Hard — requires in vivo imaging or adoptive transfer kinetics. Best constrained indirectly via full-model T cell density targets.
- Trafficking rates (`q_nCD8_*`, `q_nTreg_*`): Same — in vivo kinetics, difficult to extract as submodel targets.
- `n_CD4_clones`: Could constrain from TCR sequencing diversity data in PDAC (Balachandran 2017).
- `k_MMP_M1`: Needs pre/post therapy biopsy collagen + M1 data. No clean in vitro assay.
- `k_CCR5_Treg_rec`: CCR5/CCL5 Treg migration assays exist (transwell). Moderate extractability.
- `K_apCAF_Treg` / `P1_50_apCAF`: Thin literature on apCAF-Treg interactions (Huang 2022).

---

## sigma = 1.0 (50 params) — High priority

### T cell module (15 params)

| Parameter | Units | Notes |
|-----------|-------|-------|
| `k_CD8_pro` | 1/day | CD8 T cell proliferation |
| `k_CD8_death` | 1/day | CD8 death in tumor |
| `k_stiff_CD8_exh` | 1/day | T cell collagen interaction rate |
| `k_CD8_mig` | 1/day | CD8 migration rate |
| `k_CD8_act` | 1/day | CD8 activation rate |
| `k_exh` | 1/day | T cell exhaustion rate |
| `K_C_exh` | cell | Cancer cell threshold for exhaustion |
| `K_C2_recruit` | cell | T cell recruitment carrying capacity |
| `K_APC_mig` | cell | Migration carrying capacity |
| `Q_nCD8_thym` | cell/day | Naive CD8 thymic output |
| `Q_nTreg_thym` | cell/day | Naive Treg thymic output |
| `CD28_CD8X_50` | dimensionless | CD28 co-stimulation EC50 |
| `R50_Treg` | dimensionless | Treg suppression EC50 |
| `ArgI_50_Treg` | dimensionless | Arginase I EC50 for Treg |
| `n_CD8_clones` | dimensionless | CD8 clone diversity (has target) |

### Macrophage module (14 params)

| Parameter | Units | Notes |
|-----------|-------|-------|
| `k_M1_pol` | 1/day | M1 polarization rate |
| `k_M2_pol` | 1/day | M2 polarization rate |
| `k_M1_death` | 1/day | M1 death rate |
| `k_M2_death` | 1/day | M2 death rate |
| `k_Mac_rec` | cell/(mL*day) | Macrophage recruitment |
| `K_M1_phago` | cell | Mac carrying capacity |
| `k_Mac_C_kill` | 1/day | Mac cancer cell killing |
| `IL10_50` | nanomolarity | IL-10 EC50 for M2 polarization |
| `k_IL10_sec_M2` | nM*L/(cell*day) | M2 IL-10 secretion |
| `k_IL10_sec_Treg` | nM*L/(cell*day) | Treg IL-10 secretion |
| `k_IL10_deg` | 1/day | IL-10 degradation |
| `k_IL12_sec` | pM*L/(cell*day) | IL-12 secretion |
| `k_IL12_deg` | 1/day | IL-12 degradation |
| `IL12_50` | picomolarity | IL-12 EC50 |

### APC / dendritic cell module (13 params)

| Parameter | Units | Notes |
|-----------|-------|-------|
| `APC0_cDC1_T` | cell/mL | cDC1 homeostatic density (tumor) |
| `APC0_cDC2_T` | cell/mL | cDC2 homeostatic density (tumor) |
| `APC0_cDC1_LN` | cell/mL | cDC1 homeostatic density (LN) |
| `APC0_cDC2_LN` | cell/mL | cDC2 homeostatic density (LN) |
| `k_APC_mat_cDC1` | 1/day | cDC1 maturation rate |
| `k_APC_mat_cDC2` | 1/day | cDC2 maturation rate |
| `k_APC_mig` | 1/day | APC migration to LN |
| `k_APC_death` | 1/day | APC death rate |
| `k_mAPC_death` | 1/day | Mature APC death rate |
| `DAMP_50` | 1/day | DAMP EC50 for APC maturation |
| `EC50_P1_mature` | molarity | Antigen EC50 for APC maturation |
| `k_P1_up` | 1/(cell*day) | Antigen uptake rate |
| `k_xpres` | 1/day | Cross-presentation rate |

### Metabolism / hypoxia module (9 params)

| Parameter | Units | Notes |
|-----------|-------|-------|
| `k_gluc_consumption` | 1/day | Glucose consumption rate |
| `k_gluc_supply` | 1/day | Glucose supply rate |
| `gluc_50` | dimensionless | Glucose EC50 for cancer growth |
| `k_lactate_prod` | 1/day | Lactate production rate |
| `k_lactate_clear` | 1/day | Lactate clearance rate |
| `lactate_50_immune` | dimensionless | Lactate EC50 for immune suppression |
| `k_O2_supply` | 1/day | Oxygen supply rate |
| `k_O2_consumption` | 1/day | Oxygen consumption rate |
| `HIF_50` | dimensionless | HIF activation threshold |

### Other sigma=1.0 params

| Parameter | Units | Module | Notes |
|-----------|-------|--------|-------|
| `k_PSC_to_iCAF` | 1/day | Fibroblast | Base iCAF activation rate |
| `k_PSC_to_apCAF` | 1/day | Fibroblast | apCAF activation rate |
| `K_stiff_niche` | cell | Fibroblast | Stiffness-dependent CAF niche |
| `k_CAF_encounter` | cell/(mL*day) | Fibroblast | PSC encounter rate |
| `k_CAF_const` | cell/mL | Fibroblast | Normal pancreas PSC density |
| `k_apCAF_Treg` | 1/day | Fibroblast | apCAF Treg induction rate |
| `k_NK_act` | 1/day | NK | NK activation rate |
| `k_NK_death` | 1/day | NK | NK death rate |
| `k_NK_rec` | cell/(mL*day) | NK | NK recruitment |
| `K_NK_C` | cell | NK | NK carrying capacity |
| `IL10_50_NK` | nanomolarity | NK | IL-10 EC50 for NK suppression |
| `k_GVAX_clearance` | 1/day | GVAX | GVAX clearance rate |
| `k_GMCSF_prod` | nM/cell/day | GVAX | GM-CSF production |
| `k_GMCSF_deg` | 1/day | GVAX | GM-CSF degradation |
| `VEGF_norm_50` | dimensionless | Angiogenesis | VEGF EC50 for vascular normalization |
| `PDL1_C1_baseline` | dimensionless | Checkpoint | Baseline PD-L1 on cancer |

---

## sigma = 0.7 (73 params) — Moderate priority

These include cytokine degradation rates, baseline densities, therapy parameters, and numerous Hill function EC50s. Many are well-estimated from literature but lack formal submodel targets. Full list omitted for brevity — see `pdac_priors.csv`.

**Notable sigma=0.7 params without targets:**
- `k_CCL5_sec_iCAF` — iCAF CCL5 secretion (estimated, no literature)
- `f_IL6_iCAF` — IL-6/LIF proxy fold-enhancement (not extractable, see fibroblast audit)
- `k_stroma_support` — Stromal support of cancer carrying capacity
- `phi_col_50_Kmax` — Collagen EC50 for K_max
- `k_C1_death` — Cancer cell death rate
- `k_TGFb_sec_C` — Cancer TGFb secretion
- All checkpoint module params (4)
- Multiple CXCL9/T recruitment params (7)

---

## Modules with 0% coverage

| Module | Params | Notes |
|--------|--------|-------|
| CXCL9/T recruitment | 7 | CXCL9 secretion, CXCR3 signaling, T cell recruitment. Literature exists (Mikucki 2015). |
| Metabolism/hypoxia | 9 | Glucose, lactate, oxygen, HIF. Warburg effect data available but mapping to model params is indirect. |
| Checkpoint (PD-1/PD-L1) | 4 | PD-L1 expression dynamics. Relevant for nivo scenario — prioritize when treatment calibration begins. |
| CCL5/Treg | 1 | `k_CCR5_Treg_rec` — CCR5-mediated Treg recruitment. Transwell migration data exists. |

---

## Outstanding fibroblast module params

From `notes/calibration/fibroblast_param_audit.md`. The fibroblast module has the **best coverage** (25/63 = 40% with targets, 53% including groups), but key gaps remain:

### No target, no group (11 params)

| Parameter | Value | Units | sigma | Priority | Notes |
|-----------|-------|-------|-------|----------|-------|
| `k_PSC_to_iCAF` | 0.2 | 1/day | 1.0 | High | Base iCAF activation rate. No IL-1-driven activation assay with time course. Constrained indirectly by CAF ratio calibration targets. |
| `k_PSC_to_apCAF` | 0.01 | 1/day | 1.0 | High | apCAF activation. Elyada 2019 apCAF ~2-5%. Thin literature. |
| `K_stiff_niche` | 1e5 | cell | 1.0 | Medium | Max CAFs sustained by stiff ECM alone. Theoretical, no direct data. |
| `k_CAF_encounter` | 1e6 | cell/(mL*day) | 1.0 | Low | PSC encounter rate. Sudo 2009 has density data. |
| `k_CAF_const` | 2e6 | cell/mL | 1.0 | Low | Normal pancreas PSC density. |
| `k_apCAF_Treg` | 0.03 | 1/day | 1.0 | Low | apCAF Treg induction. No literature backing. |
| `k_MMP_M1` | 1e-8 | mL/(cell*day) | 1.2 | Medium | Immune ECM degradation. Needs pre/post therapy collagen + M1 data. |
| `K_apCAF_Treg` | 1e7 | cell | 1.2 | Low | Estimated. |
| `P1_50_apCAF` | 1e-9 | M | 1.2 | Low | Estimated. |
| `k_stroma_support` | 1.0 | dimensionless | 0.7 | Medium | Stromal support of cancer carrying capacity. Rhim 2014/Ozdemir 2014. |
| `phi_col_50_Kmax` | 2.0 | dimensionless | 0.7 | Medium | Intentionally >1 (stroma depletion paradox). |

### No target, in group (8 params)

| Parameter | Value | Units | sigma | Group | Anchor target |
|-----------|-------|-------|-------|-------|---------------|
| `k_qpsc_death` | 0.0165 | 1/day | 0.5 | CAF_death_rates | k_myCAF_death (Kisseleva 2012) |
| `k_iCAF_death` | 0.0165 | 1/day | 0.5 | CAF_death_rates | k_myCAF_death |
| `k_apCAF_death` | 0.0165 | 1/day | 0.5 | CAF_death_rates | k_myCAF_death |
| `k_apCAF_prolif` | 0.05 | 1/day | 1.0 | CAF_prolif_rates | k_myCAF_prolif + k_iCAF_prolif |
| `TGFb_50_col_sec` | 0.07 | nM | 0.7 | TGFb_EC50_fib | TGFb_50_CAF_act (Bachem 1992) |
| `TGFb_50_IL1R1` | 0.07 | nM | 0.7 | TGFb_EC50_fib | TGFb_50_CAF_act |
| `IL6_50_iCAF` | 0.1 | nM | 0.7 | IL6_EC50s | IL6_50_M2 (Fernando 2014) |
| `k_CXCL12_sec_cancer` | 5e-11 | nM*L/(cell*day) | 1.2 | CXCL12_sec | k_CXCL12_sec_iCAF (2 targets) |

### Not extractable (documented)

| Parameter | sigma | Reason |
|-----------|-------|--------|
| `f_IL6_iCAF` | 0.7 | IL-6 proxy for LIF. Biffi 2019 S4B-D: IL-6 KO no effect on iCAF. Not extractable. |
| `IL6_50_iCAF` | 0.7 | Same — LIF proxy. In IL6_EC50s group, anchored by IL6_50_M2. |

---

## Recommended extraction priorities (cross-module)

### Tier 1 — High impact, likely extractable
1. **`k_Treg_pro`** (sigma=1.2) — CFSE proliferation assays with IL-2 stimulation
2. **`k_Treg_death`** (sigma=1.2) — IL-2 withdrawal survival curves
3. **`n_CD4_clones`** (sigma=1.2) — TCR sequencing diversity in PDAC (Balachandran 2017)
4. **`k_M1_pol`** / **`k_M2_pol`** (sigma=1.0) — IFNg/LPS and IL-4/IL-10 polarization kinetics

### Tier 2 — Moderate impact, data may exist
5. **`k_CCR5_Treg_rec`** (sigma=1.2) — Transwell migration assays with CCL5
6. **`APC0_cDC1_T`** / **`APC0_cDC2_T`** (sigma=1.0) — mIF density data from PDAC tissue
7. **`k_CD8_pro`** (sigma=1.0) — CFSE proliferation with anti-CD3/CD28
8. **CXCL9 module** (7 params) — Mikucki 2015, Chow 2019 for CXCR3 chemotaxis

### Tier 3 — Important for treatment scenarios
9. **Checkpoint params** (4, sigma=0.7) — Needed for nivo calibration
10. **GVAX params** (5, sigma=0.5-1.0) — Needed for GVAX scenario calibration
