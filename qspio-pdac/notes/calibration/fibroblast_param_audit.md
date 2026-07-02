# Fibroblast Module Parameter Audit

Date: 2026-03-18 (updated 2026-03-24)

Comprehensive audit of all parameters in `fibroblast_module_enhanced.m`. For each: current value, whether it has a submodel target, whether it's in `pdac_priors.csv`, and calibration status.

**All fibroblast params are now in `pdac_priors.csv`.** As of this audit, the 16 previously-fixed params (cell volumes, degradation rates, IL-1 pathway, TGFb_50 splits) were added. `collagen_base` and `collagen_MW` were removed from both the model and priors (dead parameters).

**Priors CSV reverted to base values.** An autouse pytest fixture (`tests/conftest.py`) and module-level call in `workflows/obed_workflow.py` were silently overwriting `pdac_priors.csv` with submodel target posteriors on every run. Both removed. The CSV now holds original literature defaults matching `parameters_PDAC.m`. Submodel target posteriors live in the YAML files and should be composed at SBI runtime.

---

## Parameters WITH Submodel Targets (30 params, excluding 1 removed)

| Parameter | Value | Units | Target Source | Contraction |
|-----------|-------|-------|---------------|-------------|
| `d_pore_ref` | 3.5→4.0 | μm | Wolf 2013 (Fig 1D, 4-pt power law) | 0.53 |
| `phi_col_ref` | 0.12→0.13 | dimensionless | Wolf 2013 + Rubiano 2018 (shared) | 0.69 |
| `d_crit_T` | 2.3→3.97 | μm | Wolf 2013 (Fig 7A, filter transmigration) | 0.97 |
| `n_pore` | 4→8.27 | dimensionless | Wolf 2013 (Fig 7A, 3-pt Hill fit) | 0.86 |
| `E_ref` | 1.0→0.91 | dimensionless | Rubiano 2018 (normal pancreas SSM) | 0.80 |
| `n_stiff` | 1.8→1.65 | dimensionless | Rubiano 2018 (collagen gel power law) | 0.34 |
| `k_col_myCAF_sec` | 5e-9→1.3e-8 | mg/(cell·day) | Illsley 2000 (OHpro HPLC) | 0.68 |
| `f_col_basal` | 0.5→0.566 | dimensionless | Illsley 2000 (TGFb fold-change) | 0.34 |
| `k_col_deg` | 0.007→0.0038 | 1/day | Decaris 2015 (hepatic collagen FSR) | 0.99 |
| `f_stroma_max` | 3.0→1.44 | dimensionless | Torphy 2018 (autopsy stromal fraction) | 0.80 |
| `k_PSC_to_myCAF` | 0.4 | 1/day | Bachem 1993 Fig 4a + Biffi 2019 Fig S6B (qPSC+TGFb activation ODE) | 0.23 |
| `k_myCAF_prolif` | 0.15 | 1/day | Bachem 1992 Table I (BrdU) + Bachem 1993 Fig 4a + Biffi 2019 S6K+5H+S6B (3 targets) | 0.40–0.64 |
| `k_myCAF_death` | 0.0165 | 1/day | Kisseleva 2012 (in vivo recovery) | ~0 |
| `k_TGFb_sec_myCAF` | 2e-11 | nM·L/(cell·day) | Bachem 1992 Table II (TGFb1 ELISA) | validated |
| ~~`PDGF_50_prolif`~~ | ~~0.2→0.18~~ | ~~nM~~ | ~~Bachem 1993 Fig 7b~~ | **REMOVED** — PDGF species eliminated from model |
| `k_CXCL12_sec_iCAF` | 1e-10→6.3e-10 | nM·L/(cell·day) | Saiman 2013 + Ene-Obong 2013 (2 targets) | -3.9 |
| `k_IL1_sec` | 1e-12→4.2e-11 | nmol/(cell·day) | Tjomsland 2013 (PC013 ELISA) | -254 |
| `TGFb_50_CAF_act` | 0.07→0.05 | nM | Bachem 1992 Fig 7b + Biffi 2019 5H+S6B (shared, weakly constrained) | validated |
| `E_50_Texh` | 4.0→4.5 | dimensionless | Saitakis 2017 (IFNg+TNFa vs stiffness, CD4 T cells) | -0.14 |
| `n_Texh` | **1.0→0.33** | dimensionless | Saitakis 2017 (shared w/ E_50_Texh). **NEW PARAM** | 0.98 |
| `E_50_fib` | 5.0→7.1 | dimensionless | Chang 2025 (YAP nuc/cyt in PDAC CAFs vs stiffness) | -0.75 |
| `n_fib` | **1.0→0.33** | dimensionless | Chang 2025 (shared w/ E_50_fib). **NEW PARAM** | 0.98 |
| `IL1_50` | 0.01→0.02 | nM | Ciminieri 2023 (IL-6+CXCL8 fold-change vs IL-1beta) | -2.9 |
| `n_IL1` | **1.0→0.5** | dimensionless | Ciminieri 2023 (shared w/ IL1_50). **NEW PARAM** | 0.74–0.83 |
| `k_iCAF_prolif` | 0.10 | 1/day | Biffi 2019 Fig S6K (CellTiter-Glo 5-day time course) | 0.10 |
| `k_iCAF_to_myCAF` | 0.1 | 1/day | Biffi 2019 Fig 5H (two-state iCAF↔myCAF ODE) | -0.18 |
| `k_myCAF_to_iCAF` | 0.05 | 1/day | Biffi 2019 Fig 5H (shared, weakly constrained) | 0.10 |

---

## Parameters WITHOUT Submodel Targets — In Priors

These are varied by SBI but have no literature-derived submodel constraint. Ordered by calibration priority.

### Priority 1: CAF interconversion + activation (unblocked by IL-1 addition)

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| `k_PSC_to_iCAF` | 0.2 | 1/day | 1.0 | Base rate before IL-6 amplification. No direct constraint yet. |
| `k_PSC_to_apCAF` | 0.01 | 1/day | 1.0 | Elyada 2019 apCAF ~2-5%. |

**Calibration target coverage:** Elyada 2019 KPC flow cytometry ratios (44:45:10) already extracted as calibration targets (Ogawa 2021 primary, `baseline_no_treatment/`). These constrain the steady-state equilibrium of all activation/interconversion/death rates jointly via the full model. `k_PSC_to_myCAF` additionally constrained by Bachem 1993 submodel target (see WITH table).

**Remaining gaps:** `k_PSC_to_iCAF` has no direct constraint (no IL-1-driven activation assay with time course). `k_PSC_to_apCAF` likewise unconstrained. Both are constrained indirectly via the CAF ratio calibration targets. `k_iCAF_to_myCAF` and `k_myCAF_to_iCAF` now have submodel targets from Biffi 2019 Fig 5H (two-state ODE).

### Priority 2: Coupled equilibrium params

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| `K_stiff_niche` | 1e5 | cell | 1.0 | Max CAFs sustained by stiff ECM alone. Theoretical, no direct data. |
| `k_stroma_support` | 1.0 | dimensionless | 0.7 | Stromal support of cancer carrying capacity. Rhim 2014/Ozdemir 2014. |
| `phi_col_50_Kmax` | 2.0 | dimensionless | 0.7 | Intentionally >1 (stroma depletion paradox). |

**Target data:** Stromal fraction at diagnosis (Torphy 2018 median=0.74, Vendittelli 2024 median=0.85). These constrain the equilibrium outcome of cancer+CAF+collagen dynamics jointly. f_stroma_max already extracted; remaining params need equilibrium submodel target.

### Priority 3: Immune-dependent ECM degradation

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| `k_MMP_M1` | 1e-8 | mL/(cell·day) | 1.2 | No literature constraint. Enables GVAX stroma remodeling. |

**Target data:** Needs paired pre/post therapy biopsy collagen + immune infiltration data. FOLFIRINOX pre/post most available; Zheng 2022 if stromal fraction measured.

### Priority 4: CAF proliferation rates — partially addressed

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| `k_apCAF_prolif` | 0.05 | 1/day | 1.0 | Elyada 2019 apCAF fraction maintained by low proliferation. Now in hierarchical group `CAF_prolif_rates` (delta_mu=-0.4, biased lower). |

`k_myCAF_prolif` and `k_iCAF_prolif` now have submodel targets (see WITH table). `k_apCAF_prolif` has no direct target but benefits from partial pooling via the `CAF_prolif_rates` hierarchical group. `PDGF_50_prolif` was removed with PDGF species.

### ~~Priority 5: Mechanosensing thresholds~~ DONE

All mechanosensing parameters now have submodel targets. Three new Hill coefficient parameters (n_Texh, n_fib, n_IL1) were added to the model — all show sub-linear response (n=0.3-0.6), meaning mechanosensing acts as a rheostat, not a binary switch. "Liu 2015" reference for E_50_fib was hallucinated; replaced with Chang 2025 (PDAC CAFs, exact indication match).

### Priority 6: PSC encounter / homeostasis

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| `k_CAF_encounter` | 1e6 | cell/(mL·day) | 1.0 | Sudo 2009 PSC density 145±9.38 cells/0.785mm². |
| `k_CAF_const` | 2e6 | cell/mL | 1.0 | Normal pancreas PSC density. |

**Target data:** Sudo 2009 (PMID: 19289585) PSC density in normal vs chronic pancreatitis pancreas. Could extract as SubmodelTarget if quantitative density data available.

### ~~Priority 7: CAF death rates~~ ADDRESSED (hierarchical group)

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| `k_qpsc_death` | 0.0165 | 1/day | 0.5 | ~~Hao 2015 PMID: 25225370~~ **Citation invalid** — paper is a renal fibrosis math model, not pancreatic biology. Now constrained via hierarchical group. |
| `k_iCAF_death` | 0.0165 | 1/day | 0.5 | Hierarchical group (partial pooling from k_myCAF_death target). |
| `k_myCAF_death` | 0.0165 | 1/day | 0.5 | **HAS TARGET** — Kisseleva 2012 in vivo recovery. Derived prior median=0.0168/day, confirms current value. |
| `k_apCAF_death` | 0.0165 | 1/day | 0.5 | Hierarchical group (partial pooling from k_myCAF_death target). |

**Resolved via hierarchical parameter group** (`parameter_groups.yaml`, group `CAF_death_rates`). All four death rates share a latent base rate with per-member deviations. The Kisseleva 2012 target anchors k_myCAF_death; the other three shrink toward the group mean via partial pooling. Biological ordering encoded in delta priors: qPSC biased lower (quiescent cells are long-lived, delta_mu=-0.5), myCAF biased slightly lower (apoptosis-resistant via BCL-XL/XIAP, delta_mu=-0.2), iCAF and apCAF centered on group mean. The Hao 2015 citation (PMID: 25225370) was found to be invalid — it's a mathematical model of renal interstitial fibrosis, not pancreatic biology.

### Priority 8: Cytokine dynamics (secretion + EC50s)

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| ~~`k_PDGF_sec`~~ | — | — | — | **REMOVED** — PDGF species eliminated. Matrix-bound, not freely diffusing. See DEVLOG 2026-03-20. |
| ~~`PDGF_50`~~ | — | — | — | **REMOVED** — PDGF Hill function removed from activation reactions. |
| `k_CXCL12_sec_iCAF` | 1e-10 | nM·L/(cell·day) | 1.2 | **HAS 2 TARGETS** — Saiman 2013 (proxy, median=2.3e-9) + Ene-Obong 2013 (exact, median=6.3e-10). Both >current. |
| `k_CXCL12_sec_cancer` | 5e-11 | nM·L/(cell·day) | 1.2 | Feig 2013. |
| `CXCL12_50_Texcl` | 2.0 | nM | 1.2 | **HAS TARGET** — CXCL12 dose-response on T cell migration. |
| `k_IL6_sec_iCAF` | 4e-11 | nM·L/(cell·day) | 0.7 | **HAS TARGET** — Saga 2018 LX-2 ELISA (ctrl 65 pg/mL, TNF-activated 243 pg/mL). Posterior median=7.0e-10, ~18× current. Elyada 2019 citation invalid (RNA only), Mace 2018 PMID wrong. |
| `IL6_50_MDSC` | 0.06 | nM | 0.7 | Recalibrated to avoid baseline saturation. |
| `IL6_50_M2` | 0.2 | nM | 0.7 | Recalibrated to match model IL-6 range. |
| `IL6_50_iCAF` | 0.1 | nM | 0.7 | IL-6 proxy for LIF/JAK/STAT3 feedback (Biffi 2019 S4B-D: IL-6 KO no effect; LIF is actual mediator). Not extractable as submodel target. |
| `f_IL6_iCAF` | 3.0 | dimensionless | 0.7 | 4x fold at saturating signal. IL-6 proxy for LIF. Constrained indirectly by iCAF density targets. |
| `k_vas_myCAF_sec` | 8.5e-6 | pg/(cell·day) | 1.0 | **HAS TARGET** — Erkan 2009 Fig 4C PSC VEGF ELISA. Posterior median=0.00145, ~170x higher. |
| `k_CCL5_sec_iCAF` | 1e-12 | nmol/(cell·day) | 0.7 | **HAS TARGET** — Kawka 2014 HPFB TNF-alpha time course (6-96h, n=6). Posterior median=2.84e-10, **284× current**. Original value was "estimated" with no source — genuinely too low. |

**Note:** Cytokine secretion/EC50 params are listed in `notes/calibration/pdac_cytokine_chemokine_targets.csv` as deferred targets. Strategy: constrain indirectly through cell density targets first, then extract submodel targets for cytokines that drive clinically relevant outputs (per PRCC ranking).

### Priority 9: apCAF tolerogenic presentation

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| `k_P1_up_apCAF` | 1.44 | 1/(cell·day) | 1.0 | ~0.1× professional APC uptake. |
| `k_apCAF_Treg` | 0.03 | 1/day | 1.0 | No literature backing. |
| `K_apCAF_Treg` | 1e7 | cell | 1.2 | Estimated. |
| `P1_50_apCAF` | 1e-9 | M | 1.2 | Estimated. |

**Target data:** Huang 2022 (apCAF co-culture with naive CD4 → Treg). Thin literature.

---

## Previously Fixed Parameters (now in priors with tight σ)

All 16 previously-fixed params were added to `pdac_priors.csv` with conservative sigmas.

### Physical/material constants (σ=0.3)

| Parameter | Value | Units | σ | Rationale |
|-----------|-------|-------|---|-----------|
| `rho_collagen` | 0.025 | g/mL | 0.3 | Material constant. Antoine 2014. |
| `vol_qPSCcell` | 1600 | μm³ | 0.3 | Encyclopedia of Life Sciences. |
| `vol_iCAFcell` | 1600 | μm³ | 0.3 | Inherited; no subtype-specific data. |
| `vol_myCAFcell` | 1600 | μm³ | 0.3 | Inherited. |
| `vol_apCAFcell` | 1600 | μm³ | 0.3 | Inherited. |

### Degradation rates / EC50s (σ=0.5–0.7)

| Parameter | Value | Units | σ | Rationale |
|-----------|-------|-------|---|-----------|
| `k_PDGF_deg` | 5.0 | 1/day | 0.5 | PMID: 2167330 (t½ ~2-5h). |
| `k_CXCL12_deg` | 2.0 | 1/day | 0.5 | Chemokine t½ 4-12h. |
| `k_IL6_deg` | 10.0 | 1/day | 0.5 | Castell 1988 (t½ ~100 min). |
| `PDGF_50_prolif` | 0.2 | nM | 0.7 | **HAS TARGET** — Bachem 1993 Fig 7b. Confirms current value (~0.18 nM). |
| `d_crit_APC` | 2.1 | μm | 0.5 | DCs more deformable; no arrest data. |

### IL-1 pathway (σ=0.5–1.0)

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| `k_IL1_sec` | 1e-12 | nmol/(cell·day) | 1.0 | **HAS TARGET** — Tjomsland 2013 PC013 supernatant ELISA. Posterior median=4.2e-11, ~42x higher. |
| `k_IL1_deg` | 6.93 | 1/day | 0.5 | t½ ~2.4h; typical pro-inflammatory cytokine. |
| `TGFb_50_IL1R1` | 0.07 | nM | 0.7 | Set equal to TGFb_50; needs independent constraint. |

`IL1_50` and `n_IL1` now have submodel targets (see WITH table). `k_IL1_sec` and `k_IL1_deg` remain unconstrained.

### TGFb_50 split params (σ=0.7)

| Parameter | Value | Units | σ | Notes |
|-----------|-------|-------|---|-------|
| `TGFb_50_CAF_act` | 0.07 | nM | 0.7 | **HAS TARGET** — Bachem 1992 Fig 7b. Posterior ~0.05 nM. |
| `TGFb_50_col_sec` | 0.07 | nM | 0.7 | Illsley 2000 has only 1 TGFb concentration (1 ng/mL) — cannot determine EC50 from single dose. Need TGFb dose-response on collagen secretion (McAnulty 1991?). |

---

## Coverage Summary

| Category | Total | Has SubmodelTarget | In Priors (no target) | Computed |
|----------|-------|--------------------|-----------------------|----------|
| ECM constitutive relations | 6 | 6 (100%) | 0 | 0 |
| Collagen dynamics | 4 | 3 (75%) | 1 | 0 |
| CAF carrying capacity | 4 | 1 (25%) | 3 | 0 |
| CAF activation/interconversion | 6 | 1 (17%) | 3 | 2 |
| CAF proliferation | 3 | 2 (67%) | 1 | 0 |
| CAF death | 4 | 1 (25%) | 3 | 0 |
| Cytokine secretion/EC50 | 10 | 3 (30%) | 7 | 0 |
| IL-1 pathway | 5 | 3 (60%) | 2 | 0 |
| TGFb EC50 split | 2 | 1 (50%) | 1 | 0 |
| Mechanosensing thresholds | 4 | 4 (100%) | 0 | 0 |
| PSC encounter | 2 | 0 (0%) | 2 | 0 |
| apCAF presentation | 4 | 0 (0%) | 4 | 0 |
| Cell volumes | 4 | 0 (0%) | 4 | 0 |
| Degradation rates/EC50s (prev. fixed) | 3 | 0 (0%) | 3 | 0 |
| Pore/stiffness (prev. fixed) | 2 | 0 (0%) | 2 | 0 |
| **Total** | **63** | **25 (40%)** | **36 (57%)** | **2 (3%)** |

All non-computed parameters are now in `pdac_priors.csv`. The "Fixed" category no longer exists — previously-fixed params were added with tight σ (0.3–0.5).

**New submodel targets added 2026-03-18:**
- `k_PSC_to_myCAF` + `k_myCAF_prolif` (joint) — Bachem 1993 Fig 4a activation time course
- `k_myCAF_death` — Kisseleva 2012 in vivo recovery dynamics
- `k_myCAF_prolif` — Bachem 1992 Table I BrdU incorporation
- `k_TGFb_sec_myCAF` — Bachem 1992 Table II TGFbeta1 production rate

**New submodel targets added 2026-03-20:**
- `k_CXCL12_sec_iCAF` (deriv001) — Saiman 2013 JS1 hepatic stellate cell CXCL12 ELISA time course (0-96h). Posterior median=2.34e-9 nM·L/(cell·day), 23x higher than current 1e-10. Also constrains `k_CXCL12_deg` (posterior 1.33/day vs prior 2.0/day).
- `k_CXCL12_sec_iCAF` (deriv002) — Ene-Obong 2013 primary PDAC aPSC CXCL12 ELISA (n=7 patients, exact indication). Posterior median=6.29e-10, 6.3x higher than current. Cell count/volume as nuisance params. Joint inference with deriv001 will compose the two estimates.
- `k_IL1_sec` — Tjomsland 2013 PC013 human PDAC supernatant IL-1alpha ELISA (48h steady-state). Posterior median=4.2e-11 nmol/(cell·day), 42x higher than placeholder 1e-12. Also constrains `k_IL1_deg` (posterior 2.59/day vs prior 6.93/day). Exact indication match, low translation sigma (0.38).

**New submodel targets added 2026-03-19:**
- `E_50_Texh` + `n_Texh` (new param) — Saitakis 2017 IFNg + TNFa vs stiffness in CD4 T cells
- `E_50_fib` + `n_fib` (new param) — Chang 2025 YAP nuc/cyt in PDAC CAFs vs stiffness
- `IL1_50` + `n_IL1` (new param) — Ciminieri 2023 IL-6 + CXCL8 fold-change vs IL-1beta in MRC-5
- All three Hill coefficients are sub-linear (n=0.3-0.6): mechanosensing is rheostat-like, not switch-like

**New submodel targets added 2026-03-25:**
- `k_vas_myCAF_sec` — Erkan 2009 Fig 4C PSC VEGF ELISA (primary human PSCs, PDAC tissue). Posterior median=0.00145 pg/cell/day, 170x higher than current 8.5e-6 placeholder. Normoxia only (model has no hypoxia-VEGF modifier on CAFs).
- `IL6_50_M2` (macrophage module, not fibroblast) — Fernando 2014 Fig 2A, 5-point IL-6 dose-response on BMDM arginase activity. Posterior median=0.223 nM, validates current 0.20 nM. Anchors IL6_EC50s hierarchical group.
- `f_IL6_iCAF` / `IL6_50_iCAF` — NOT extractable. Biffi 2019 S4B-D shows IL-6 KO has no effect on iCAF markers; actual mediator is LIF. Parameter notes updated to document proxy interpretation.

---

## Next Steps: Literature Targets to Find

### Tier 1 — High impact, data available or in hand

1. ~~**Elyada 2019 CAF ratios**~~ DONE — active calibration targets in `baseline_no_treatment/` (Ogawa 2021 primary).
2. ~~**Biffi 2019 organoid IL-1alpha concentration**~~ Biffi only has 2 concentrations (no dose-response). IL1_50 constrained instead by Ciminieri 2023.
3. ~~**Liu 2015 / Dupont 2011 YAP nuclear translocation vs stiffness**~~ "Liu 2015" was hallucinated. E_50_fib constrained by Chang 2025 (PDAC CAFs).
4. ~~**Saitakis 2017 T cell activation vs substrate stiffness**~~ DONE — E_50_Texh + n_Texh from IFNg/TNFa dose-response.
5. ~~**PSC proliferation with PDGF**~~ DONE (Bachem 1992 Table I → k_myCAF_prolif).
6. ~~**Bachem 1993 Fig 7 — PDGF dose-response on MFBLC**~~ DONE → `PDGF_50_prolif` extracted and validated.
7. ~~**Bachem 1992 Fig 7b — TGFb1-accelerated activation**~~ DONE → `TGFb_50_CAF_act` extracted and validated.

### Tier 2 — Moderate impact, data may exist

8. **PDGF secretion by PDAC cell lines** (ELISA) → constrains k_PDGF_sec.
9. ~~**CXCL12 ELISA in PDAC conditioned media**~~ DONE — Saiman 2013 JS1 HSC time course (proxy: hepatic stellate). Posterior median 2.34e-9, 23× higher than current 1e-10.
10. ~~**IL-1alpha ELISA from PDAC cell lines**~~ DONE — Tjomsland 2013 PC013 supernatant (exact indication, human primary). Posterior median 4.2e-11, 42x higher than placeholder 1e-12.
11. **Pre/post neoadjuvant therapy stromal fraction** → constrains k_MMP_M1 jointly with k_col_deg.
12. ~~**Kisseleva 2012 apoptosis snapshot**~~ SKIP — 2.6+/-0.7% caspase-3+ at day 7 measures transient immune-mediated killing during fibrosis resolution, not constitutive k_myCAF_death. Would conflict with deriv001 (full 30-day recovery trajectory).
13. ~~**Kisseleva 2012 adoptive transfer**~~ SKIP — 38x iHSC vs qHSC reactivation requires an iHSC/iCAF compartment not in current model. Valuable for future model extension (CAF reversion/memory).

### Tier 3 — Low impact or thin literature

14. **DC migration in 3D collagen** → constrains d_crit_APC. Wolf 2013 only did T cells.
15. **apCAF co-culture Treg conversion** → constrains k_apCAF_Treg, K_apCAF_Treg.
16. **Bachem 1993 Table 1 — TGFb1 differential proliferation** → qualitative bound on TGFb-proliferation interaction.
