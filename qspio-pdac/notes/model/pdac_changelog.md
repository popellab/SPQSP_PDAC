# PDAC Model Changelog

Changes to the core SimBiology QSP model (`core/`) since adaptation from the original QSPIO-HCC framework (ShumingZhang1997, Sep 2023) to PDAC immunotherapy modeling. PDAC adaptation began Jun 2025; the `core/` directory was created Jul 25, 2025 by restructuring the flat repo layout.

**Total: ~157 commits affecting `core/` across 7 development phases.**

---

## Phase 1: Repository Restructuring & Parameter Management (Jul 25 – Aug 22, 2025)

### Repository restructuring (33feb03, Jul 25)
Moved flat directories into `core/` subdirectories: `model/` → `core/modules/`, `@struct/` → `core/@struct/`, `postprocessing/` → `core/postprocessing/`, `visualization/` → `core/visualization/`, `PK_fitting/` → `core/PK_fitting/`. Pure rename/move, no content changes.

### SimBiology compatibility fixes (8f6f55c, Sep 5)
Fixed species/parameter naming across 7 modules for SimBiology compatibility:
- **MDSC_module.m**: Default cancer type `'chemo_sens'` → `'C1'`
- **Teff_module.m**: `H_PD1_chemo_sens` → `H_PD1_C1`; added weighted IL-2 consumption parameters (`w_IL2_CD8`, `w_IL2_Treg`, `w_IL2_aCD8`)
- **Treg_module.m**: Added weighted IL-2 consumption by Tregs
- **cabozantinib_module.m**, **nabpaclitaxel_module.m**: `chemo_sens`/`chemo_res` → `C1`/`C2`
- **checkpoint_module.m**: Added case-insensitive matching and error handling for cell type names
- **phagocytosis_module.m**: `'chemo_sens'` → `'C1'`

### T cell and cancer cell naming updates (97211e8, Aug 12)
Comprehensive species rename for PDAC clarity:
- **simbio_init.m**: `T1_exh` → `CD8_exh`; `H_PD1_C1` → `H_PD1_chemo_sens`; `H_CD28_C1` → `H_CD28_chemo_sens`
- **Teff_module.m**: Removed generic `ID` parameter; hardcoded `species_name = 'CD8'`, `antigen = 'P1'`; all `T1_exh` → `CD8_exh`
- **Treg_module.m**: Hardcoded `species_name = 'Treg'`, `antigen = 'P0'`; `Tregs_ = V_T.Treg`; IL-2 references `V_LN.T1` → `V_LN.CD8`
- **cancer_module.m**: Growth/death parameter names became species-dependent (`k_C_growth` → `k_{species_name}_growth`)
- **antigen_rate.m**: T cell killing reaction detection `k_C_T` → `k_C_CD8`; added default `R_T = '0*cell/day'`

### Parameter renaming across all modules (81beae0, Aug 12)
Reverted `C1`/`C2` back to `chemo_sens`/`chemo_res` and standardized T cell naming across ~19 files:
- **APC_module.m**: `nT0`/`nT1` → `nTreg`/`nCD8`; `n_T0_clones`/`n_T1_clones` → `n_Treg_clones`/`n_CD8_clones`
- **Th_module.m**: `k_T0_pro` → `k_CD4_pro`; `k_T0_death` → `k_CD4_death`; transport rates `q_T0_P_in` → `q_Treg_P_in`; differentiation `V_T.T → V_T.T0` → `V_T.T → V_T.Treg`
- **Treg_module.m**: Clone notes clarified as "self-antigen-specific T cell clones"
- **Treg_ADCC_module.m**: `V_T.T0`/`V_P.T0` → `V_T.Treg`/`V_P.Treg`
- **TCR_KPR_module.m**: Added `T0` → `n_Treg_clones`, `T1` → `n_CD8_clones` mapping
- **macrophage_module.m**: Exhaustion reaction `V_T.T1 → V_T.T1_exh` → `V_T.CD8 → V_T.CD8_exh`
- **fibroblast_module.m**: TGFb secretion from `V_T.T0` → `V_T.Treg`
- **cabozantinib_module.m**, **nabpaclitaxel_module.m**: `C1`/`C2` → `chemo_sens`/`chemo_res`
- **MDSC_module.m**: Default cancer type reverted to `'chemo_sens'`; Th-to-Treg reaction updated

### SimBiology select() syntax fix (ac3bbeb, Aug 22)
Fixed `howManyClones.m` and visualization functions for correct SimBiology `select()` usage.

### PSA visualization suite (Jul 27–31)
Added tornado plots, sensitivity heatmaps, parameter ranking, waterfall charts, patient stratification, and comprehensive QSP diagnostic plots in `core/visualization/`.

### Parameter management simplification (Jul 28 – Aug 22)
Iteratively simplified from complex multi-source YAML/JSON parameter loading (~1100 lines) down to streamlined hybrid loader (`load_parameters_hybrid.m`, ~505 lines). Removed 10+ redundant parameter management files.

### Batch/HPC system (Jul 29 – Oct 23)
Built SLURM cluster integration, refactored to generic batch system, then eventually extracted to `metadata/` and later to external `qsp-hpc-tools` package.

---

## Phase 2: GVAX Module & Core Module Updates (Sep 5, 2025)

### GVAX immunotherapy module (958aad1, Sep 5)
Created `core/modules/gvax_module.m` (179 lines) — GM-CSF-mediated vaccine immunotherapy:
- **GVAX_cells** species in tumor compartment with first-order clearance
- **GMCSF** species with production by GVAX cells and degradation
- **Hill function saturation**: `H_GMCSF = GMCSF / (EC50_GMCSF + GMCSF)`
- **APC recruitment**: `k_GMCSF_recruit * H_GMCSF * V_T`
- **APC maturation enhancement**: `k_GMCSF_mature * H_GMCSF * V_T.APC * H_IL12 * (1-H_IL10)`
- **Antigen dynamics**: `P1_GVAX` species tracking pure GVAX antigen with uptake by mAPCs, plus background antigen release to total P1 pool

---

## Phase 3: Parameter Metadata & Traceability (Sep – Oct 2025)

### UserData metadata on all module parameters (f631d49, Sep 15)
Added `param.UserData.derivedFrom` to every parameter definition across all modules, enabling traceability to literature sources. Pattern: `set(param, 'UserData', params.X.derivedFrom)`.

### Refined metadata pattern (3cf8525, Oct 2)
Standardized metadata across all modules to use structured `UserData.Source` (literature reference) + `UserData.derivedFrom` (parameter provenance), with clean `Notes` (description only). Key changes:
- **Th_module.m**: Added `k_CD4_pro` and `k_CD4_death` parameters with full metadata
- **Treg_module.m**: Split shared `k_CD4_pro`/`k_CD4_death` into separate `k_Treg_pro`/`k_Treg_death`; updated `f_nTreg` metadata
- **antigen_module.m**: Cleaned compartment Notes from literature refs to descriptions
- **cancer_module.m**: Reformatted parameter addition code for consistency

### derivedFrom added to PK parameters (35739dd, Sep 15)
Extended metadata tracking to all pharmacokinetic module parameter definitions.

---

## Phase 4: Enhanced Fibroblast/PSC Module (Oct 28 – Nov 5, 2025)

### PSC-based fibroblast module (0b1e71a, Oct 28)
Created `core/modules/fibroblast_module_enhanced.m` (375 lines) — pancreatic stellate cell dynamics for PDAC desmoplastic stroma:

**New species:** `qPSC` (quiescent), `aPSC` (activated), `ECM`, `PDGF`, `CXCL12`, `IL6`

**PSC dynamics:**
- Encounter/recruitment of resident qPSCs: `k_CAF_encounter * V_T * ((k_CAF_const - V_T.qPSC/V_T) / k_CAF_const)`
- Activation (AND logic, both signals required): `k_psc_activation * V_T.qPSC * H_TGFb * H_PDGF`
- aPSC proliferation (PDGF-driven): `k_apsc_prolif * V_T.aPSC * H_PDGF`
- Death reactions for both qPSC and aPSC (constitutive + clearance)

**ECM dynamics:**
- qPSC secretion: 10% basal + 90% TGFb-dependent, with ECM_max saturation
- aPSC secretion: high rate, TGFb-dependent, with ECM_max saturation
- Homeostatic degradation driving concentration toward ECM_base

**Cytokine dynamics:**
- PDGF: secreted by cancer cells, drives PSC activation and proliferation
- CXCL12: secreted by aPSCs, contributes to T cell exclusion
- IL-6: secreted by aPSCs, influences MDSC and M2 macrophage polarization

**TGFb integration:** Full TGFb dynamics with try-catch for module ordering compatibility

### TGF-beta secretion by aPSCs (a79f17b, Oct 28)
Added positive feedback loop — aPSCs as a major source of TGFb in PDAC stroma:
```matlab
set(reaction, 'ReactionRate', 'k_TGFb_sec_apsc * V_T.aPSC');
```
New parameter `k_TGFb_sec_apsc` creates the core PSC → TGFb → PSC activation positive feedback.

### ECM unit conversions (d860bca – 596e65c, Nov 4–5)
Iterative conversion of species units for physical consistency:
- **ECM**: `nanomole` → `milligram` (total mass)
- **PDGF**: `nanomolarity` → `picogram` (total amount)
- **TGFb**: `nanomolarity` → `picogram` (total amount) — later reverted to nM
- Hill functions updated to use concentration (divide by `V_T`): `(V_T.TGFb/V_T) / (TGFb_50 + V_T.TGFb/V_T)`
- PDGF degradation: `k_PDGF_deg * V_T.PDGF * V_T` → `k_PDGF_deg * V_T.PDGF` (first-order on total amount)
- CXCL12, IL-6 kept as nM (reverted from mass-based)

### ECM degradation fix to first-order (f269528, Jan 15, 2026)
Replaced homeostatic drive-toward-baseline with simple first-order MMP-mediated turnover:
```matlab
% Before: 'k_ECM_deg * (ECM_base - ECM_level) * V_T'  (net production/degradation toward ECM_base)
% After:  'k_ECM_deg * V_T.ECM'                        (first-order degradation by MMPs)
```
Direction changed from `null → V_T.ECM` to `V_T.ECM → null`.

---

## Phase 5: Initial Conditions & GVAX Extensions (Nov 2025)

### Preserve naive T cell pools during IC equilibration (6e9fd05, Nov 11)
Modified `core/model_compilation/initial_conditions.m` to create an `IC_NoActivation` variant that sets `k_CD8_act = 0` and `k_CD4_act = 0` during the burn-in equilibration. This prevents naive T cell depletion via spurious activation during equilibration, reflecting PDAC's immunologically "cold" baseline where T cells should remain naive.

### Preserve Treg initial values (4213be3 – 5c683c5, Nov 11)
Extended IC preservation to Tregs using an exclusion list approach — species on the list retain their parameter-defined initial values rather than being overwritten by the equilibration state.

### Low-dose cyclophosphamide for Treg depletion (46b7970, Nov 14)
Extended GVAX module with metronomic cyclophosphamide (CY) dosing for selective Treg depletion, supporting the GVAX + CY + nivolumab triple therapy combination. Added CY dosing to `schedule_dosing.m`.

### GVAX phenomenological calibration (5beca07, Nov 14)
Calibrated GVAX parameters for triple therapy simulation: adjusted `n_sites_APC`, `K_APC_mig`, `K_T_C`, `ECM_50_T_mot` in parameter files.

### Density-based initial conditions (e8d602c, Dec 16)
Major rewrite of `core/model_compilation/initial_conditions.m`:
- Added `BurnInDays` parameter (default: 0, skip burn-in entirely)
- IC parameters (`C1_density_initial`, `CD8_density_initial`, etc.) applied from variant before `set_baseline_populations()`
- With `BurnInDays = 0`: directly returns density-based ICs with no equilibration simulation
- Removed old tumor-volume-target approach that searched for time point where tumor reached target size
- Simplified state extraction to use final time point of equilibration

---

## Phase 6: Mechanistic Rework of T Cell Biology (Dec 2025 – Jan 2026)

### Mechanistic T cell activation and tumor killing (1e460dc, Dec 4; restored 72d2cfd, Dec 15)
Two fundamental mechanistic changes in `core/modules/Teff_module.m` and `core/modules/cancer_module.m`:

**T cell activation — APC-driven instead of competition-based:**
```matlab
% Before (Hill function on antigen):
'k_nCD8_act * H_mAPC * H_P1 * V_LN.nT'                     % consumed
'k_nCD8_act * H_mAPC * H_P1 * V_LN.nT * n_clones_tum'      % produced

% After (per-APC encounter rate with expansion burst):
'k_T_activation_per_APC * V_LN.mAPC * H_P1 * V_LN.nT'                                       % consumed
'k_T_activation_per_APC * V_LN.mAPC * H_P1 * V_LN.nT * n_T_expansion_burst * n_clones_tum'  % produced
```
Activation now scales linearly with APC number rather than saturating, with an explicit expansion burst size.

**Tumor killing — search-volume based instead of ratio-based:**
```matlab
% Before (effector:target ratio):
'k_C_Tcell_eff = k_C_Tcell * V_T.T/(K_T_C*C_total + V_T.T + cell)
              * V_T.T/(V_T.T + K_T_Treg*Tregs_ + cell)
              * (1-H_PD1_C1) * (1-H_TGFb_Teff) * (1-H_ECM_T_mot)'

% After (physical search volume):
'k_C_Tcell_eff = p_T_kill_per_contact * v_T_search_volume * V_T.T / V_T / cell
              * (1-H_ECM_T_mot)
              * V_T.T/(V_T.T + K_T_Treg*Tregs_ + cell)
              * (1-H_PD1_C1) * (1-H_TGFb_Teff)'
```
Killing now depends on physical search volume (volume scanned per T cell per day) and per-contact kill probability, rather than an empirical effector:target ratio.

New parameters: `k_T_activation_per_APC`, `n_T_expansion_burst`, `v_T_search_volume`, `p_T_kill_per_contact`

### Exhausted CD8 T cell death rate (1bb1fd9, Jan 8)
Added `k_CD8_exh_death` parameter and reaction in `core/modules/Teff_module.m`:
```matlab
'V_T.CD8_exh → null'  at rate  'k_CD8_exh_death * V_T.CD8_exh'
```
Prevents indefinite accumulation of exhausted CD8 T cells in the tumor compartment.

### Reparameterization and OR logic for signaling (9240b92, Jan 5)

**Fibroblast module — fraction-based ECM parameterization:**
- New dimensionless parameters `f_ECM_50_T_exh` and `f_ECM_50_T_mot` (0–1 scale)
- ECM half-max values derived via rules: `ECM_50_T_exh = f_ECM_50_T_exh * ECM_max`

**Macrophage module — OR logic for polarization signals:**
```matlab
% Before (additive, could exceed 1):
'k_M2_pol * V_T.Mac_M1 * (H_TGFb + H_IL10 + H_IL6_M2)'

% After (OR logic via probability complement, capped at 1):
'k_M2_pol * V_T.Mac_M1 * (1 - (1-H_TGFb)*(1-H_IL10)*(1-H_IL6_M2))'
```
Same OR logic applied to M2→M1 repolarization with IL-12 and IFN-gamma signals.

**simbio_init.m:**
Commented out density-based IC parameters (now handled in initial_conditions.m).

---

## Phase 7: New Mechanistic Modules — Chemokine Pathways (Jan – Feb 2026)

### CCL5/CCR5 Treg recruitment module (2bba490, Jan 7)
Created `core/modules/CCL5_Treg_module.m` (86 lines) — chemokine-mediated immunosuppressive Treg recruitment:
- **CCL5** species in tumor compartment (nanomolarity)
- **Secretion by cancer cells**: `k_CCL5_sec * C_total` (FOXP3-driven, per Tan et al. 2017)
- **First-order degradation**: `k_CCL5_deg * V_T.CCL5`
- **Hill function**: `H_CCL5_Treg = V_T.CCL5 / (V_T.CCL5 + CCL5_50_Treg)`
- **CCR5-dependent Treg recruitment**: `k_CCR5_Treg_rec * V_T * V_C.Treg * H_CCL5_Treg` (selective for Tregs over CD8s via CCR5 expression)

Modified `core/modules/Treg_module.m` to integrate CCL5 recruitment pathway.

### aPSC CCL5 secretion (00b967d, Jan 8)
Added activated PSCs as an additional CCL5 source in `core/modules/CCL5_Treg_module.m`:
```matlab
'null → V_T.CCL5'  at rate  'k_CCL5_sec_apsc * V_T.aPSC'
```
Links stromal biology to immune evasion: aPSCs recruit Tregs via CCL5, complementing cancer cell secretion.

### APC recruitment via CCL2 — PLANNED, NOT YET IMPLEMENTED
~~(9e22186, Feb 9)~~ *Audit note (Feb 12, 2026): This change is not present in the current codebase. `k_APC_recruit_T`, `k_APC_recruit_LN`, `CCL2_50_APC` do not exist as parameters. APC recruitment still uses relaxation dynamics: `k_APC_death*(APC0_T*V_T-V_T.APC)`. Entry moved to planned work.*

Planned: Replace relaxation-based APC dynamics with mechanistic CCL2-driven recruitment in `core/modules/APC_module.m`:
```matlab
% Current (single homeostatic reaction):
'k_APC_death * (APC0_T*V_T - V_T.APC)'  % net flux toward steady state

% Planned (separated recruitment + death):
% Recruitment (baseline + CCL2-enhanced):
'k_APC_recruit_T * V_T * (1 + V_T.CCL2/(CCL2_50_APC + V_T.CCL2))'
% Death:
'k_APC_death * V_T.APC'
```
New parameters needed: `k_APC_recruit_T`, `k_APC_recruit_LN`, `CCL2_50_APC`. Same separated recruitment/death pattern to be applied to lymph node APCs.

### M1 macrophage CCL2 secretion — PLANNED, NOT YET IMPLEMENTED
~~(e03cc8f, Feb 9)~~ *Audit note (Feb 12, 2026): `k_CCL2_Msec` does not exist as a parameter in the current codebase. Entry moved to planned work.*

Planned: Add M1 macrophages as a CCL2 source in `core/modules/macrophage_module.m`:
```matlab
'null → V_T.CCL2'  at rate  'k_CCL2_Msec * V_T.Mac_M1'
```
Would close the innate immune amplification loop: M1 macrophages secrete CCL2 → recruits APCs → enhances antigen presentation → activates adaptive immunity. Also creates a positive feedback loop for M1 macrophage-driven inflammation.

### Fix killing-dependent antigen release (Feb 12)
Fixed stale search string in `core/modules/antigen_rate.m`:
```matlab
% Before (no match — k_C_CD8 doesn't exist after Phase 6 reparameterization):
strfind(model.reaction(k).ReactionRate, 'k_C_CD8')

% After:
strfind(model.reaction(k).ReactionRate, 'k_C_Tcell_eff')
```
This restores T-cell killing-dependent antigen release. Previously, `R_T` defaulted to `0*cell/day` because the search found no matching reaction.

### Restore DAMP-mediated APC maturation with TGFb inhibition (Feb 12)
Replaced IL-12-only intratumoral APC maturation in `core/modules/macrophage_module.m` with DAMP + IL-12 OR logic and added TGFb inhibition:
```matlab
% Before (IL-12 only — chicken-and-egg in cold PDAC TME):
'k_APC_mat*V_T.APC*H_IL12*(1-H_IL10)'

% After (DAMP bootstrap + IL-12 amplification, dual inhibition):
'k_APC_mat*V_T.APC*(1-(1-H_DAMP)*(1-H_IL12))*(1-H_IL10)*(1-H_TGFb_APC)'
```

New parameters: `DAMP_50` (half-max death rate), `TGFb_50_APC` (half-max TGFb for APC inhibition).
New rules: `H_DAMP = R_Tcell/(R_Tcell+DAMP_50)`, `H_TGFb_APC = V_T.TGFb/(V_T.TGFb+TGFb_50_APC)`.

This closes two edges in the cancer immunity cycle:
- **DAMP bootstrap**: tumor death → R_Tcell → H_DAMP → APC maturation (no pre-existing immune activation required)
- **TGFb suppression**: PDAC's TGFb-rich stroma now inhibits DC function
- **R_Tcell** is no longer vestigial — it drives H_DAMP

### ECM impedes APC migration and recruitment (Feb 13)
Added `H_ECM_APC_mig` Hill function to model ECM barrier effects on DC trafficking in `core/modules/APC_module.m`:

**APC migration (out of tumor to LN):**
```matlab
% Before: 'k_APC_mig*V_T.mAPC'
% After:  'k_APC_mig*V_T.mAPC*(1-H_ECM_APC_mig)'
```

**APC recruitment (split from relaxation into separate reactions):**
```matlab
% Before (single relaxation reaction):
'k_APC_death*(APC0_T*V_T-V_T.APC)'

% After (separate recruitment + death):
'k_APC_death*APC0_T*V_T*(1-H_ECM_APC_mig)'   % recruitment (impeded by ECM)
'k_APC_death*V_T.APC'                          % death (first-order)
```

New equilibrium: `APC_eq = APC0_T * V_T * (1 - H_ECM_APC_mig)`. Identical to original at zero ECM; reduced APC density in desmoplastic stroma.

New parameters in `fibroblast_module_enhanced.m`: `f_ECM_50_APC_mig = 0.12` (12% of ECM_max — higher threshold than T cells at 6% because DCs use podosome/MMP-dependent migration; Renkawitz et al. 2009 PMID: 19650874).

### Tumor-derived CXCL12 secretion (Feb 13)
Added cancer cells as a second CXCL12 source in `core/modules/fibroblast_module_enhanced.m`:
```matlab
'null → V_T.CXCL12'  at rate  'k_CXCL12_sec_cancer * C_total'
```

New parameter: `k_CXCL12_sec_cancer = 5e-11 nM·L/(cell·day)` (~0.5x aPSC per-cell rate). Makes T cell exclusion partially tumor-autonomous rather than purely stroma-dependent (Feig et al. 2013, PMID: 24101459).

---

## Summary of New Species Added

| Species | Compartment | Module | Description |
|---------|-------------|--------|-------------|
| `GVAX_cells` | V_T | gvax_module | Irradiated GVAX vaccine cells |
| `GMCSF` | V_T | gvax_module | GM-CSF cytokine |
| `P1_GVAX` | V_T | gvax_module | GVAX-derived tumor antigen |
| `qPSC` | V_T | fibroblast_module_enhanced | Quiescent pancreatic stellate cells |
| `aPSC` | V_T | fibroblast_module_enhanced | Activated pancreatic stellate cells |
| `ECM` | V_T | fibroblast_module_enhanced | Extracellular matrix (collagen) |
| `PDGF` | V_T | fibroblast_module_enhanced | Platelet-derived growth factor |
| `CXCL12` | V_T | fibroblast_module_enhanced | CXCL12/SDF-1 (T cell exclusion) |
| `IL6` | V_T | fibroblast_module_enhanced | Interleukin-6 (MDSC/M2 effects) |
| `CCL5` | V_T | CCL5_Treg_module | CCL5/RANTES (Treg recruitment) |

## Summary of New Parameters Added

### gvax_module.m
- `k_GVAX_clearance` — GVAX cell clearance rate
- `k_GMCSF_prod` — GM-CSF production by GVAX cells
- `k_GMCSF_deg` — GM-CSF degradation rate
- `EC50_GMCSF` — GM-CSF half-max for APC effects
- `k_GMCSF_recruit` — GM-CSF-driven APC recruitment rate
- `k_GMCSF_mature` — GM-CSF-enhanced APC maturation rate

### fibroblast_module_enhanced.m
- `k_CAF_encounter` — qPSC encounter/recruitment rate
- `k_CAF_const` — Resident qPSC density constant
- `k_psc_activation` — TGFb/PDGF-dependent PSC activation rate
- `k_apsc_prolif` — aPSC proliferation rate (PDGF-driven)
- `k_qpsc_death`, `k_apsc_death` — PSC death rates
- `k_ECM_sec_q`, `k_ECM_sec_a` — ECM secretion rates (quiescent/activated)
- `ECM_max`, `ECM_base` — ECM capacity and baseline
- `k_ECM_deg` — ECM first-order degradation rate
- `f_ECM_50_T_exh`, `f_ECM_50_T_mot` — ECM half-max fractions for T cell effects
- `k_PDGF_sec`, `k_PDGF_deg`, `PDGF_50` — PDGF dynamics
- `k_CXCL12_sec`, `k_CXCL12_deg` — CXCL12 dynamics
- `k_IL6_sec`, `k_IL6_deg` — IL-6 dynamics
- `k_TGFb_sec_apsc` — aPSC TGFb secretion rate

### CCL5_Treg_module.m
- `k_CCL5_sec` — CCL5 secretion by cancer cells
- `k_CCL5_sec_apsc` — CCL5 secretion by aPSCs
- `k_CCL5_deg` — CCL5 degradation rate
- `CCL5_50_Treg` — CCL5 half-max for Treg recruitment
- `k_CCR5_Treg_rec` — CCR5-dependent Treg recruitment rate

### Teff_module.m (new/changed)
- `k_T_activation_per_APC` — Per-APC T cell activation rate (replaced `k_nCD8_act`)
- `n_T_expansion_burst` — Clonal expansion burst size
- `v_T_search_volume` — Physical search volume per T cell per day
- `p_T_kill_per_contact` — Per-contact kill probability
- `k_CD8_exh_death` — Exhausted CD8 T cell death rate
- `w_IL2_CD8`, `w_IL2_Treg`, `w_IL2_aCD8` — IL-2 consumption weights

### APC_module.m (new — APC maturation fix)
- `DAMP_50` — Half-maximal tumor death rate for DAMP-mediated APC maturation
- `TGFb_50_APC` — Half-maximal TGFb concentration for APC maturation inhibition
- `H_DAMP` — Hill function for DAMP signal (rule: `R_Tcell/(R_Tcell+DAMP_50)`)
- `H_TGFb_APC` — Hill function for TGFb inhibition (rule: `V_T.TGFb/(V_T.TGFb+TGFb_50_APC)`)

### APC_module.m (ECM impedes APC trafficking)
- `H_ECM_APC_mig` — Hill function for ECM inhibition of APC migration/recruitment (rule: `ECM_level/(ECM_50_APC_mig+ECM_level)`)

### fibroblast_module_enhanced.m (ECM APC migration + tumor CXCL12)
- `f_ECM_50_APC_mig` — Desmoplasia severity for 50% APC migration inhibition (0.12 = 12% of ECM_max)
- `ECM_50_APC_mig` — Derived EC50 (rule: `f_ECM_50_APC_mig * ECM_max`)
- `k_CXCL12_sec_cancer` — CXCL12 secretion rate by tumor cells (5e-11 nM·L/(cell·day))

### APC_module.m (planned, not yet implemented)
- `k_APC_recruit_T` — Baseline + CCL2-driven APC recruitment in tumor
- `k_APC_recruit_LN` — APC recruitment in lymph node
- `CCL2_50_APC` — CCL2 half-max for APC recruitment

### macrophage_module.m (new)
- `k_CCL2_Msec` — M1 macrophage CCL2 secretion rate

---

## Phase 8: CAF Subtype Split — aPSC → iCAF / myCAF / apCAF

Replaced the single `aPSC` (activated pancreatic stellate cell) species with three functionally distinct CAF subtypes based on Elyada 2019 and Biffi 2019:
- **iCAF** (inflammatory CAF): IL-6 and CCL5 secretion → immunosuppression
- **myCAF** (myofibroblastic CAF): ECM, CXCL12, TGFb, VEGF secretion → desmoplasia
- **apCAF** (antigen-presenting CAF): tolerogenic antigen presentation → Treg induction

### Species changes
| Species | Compartment | Module | Description |
|---------|-------------|--------|-------------|
| `aPSC` (removed) | V_T | fibroblast_module_enhanced | Replaced by three subtypes below |
| `iCAF` (new) | V_T | fibroblast_module_enhanced | Inflammatory CAFs |
| `myCAF` (new) | V_T | fibroblast_module_enhanced | Myofibroblastic CAFs |
| `apCAF` (new) | V_T | fibroblast_module_enhanced | Antigen-presenting CAFs |

### Parameter changes
**Removed** (11 old aPSC params): `k_apsc_prolif`, `k_apsc_death`, `vol_aPSCcell`, `k_ECM_apsc_sec`, `k_CXCL12_sec`, `k_IL6_sec_apsc`, `k_TGFb_sec_apsc`, `k_vas_apsc_sec`, `k_CCL5_sec_apsc`, `k_psc_activation`

**Added** (~22 new params):
- Activation: `k_PSC_to_myCAF` (0.6), `k_PSC_to_iCAF` (0.4), `k_PSC_to_apCAF` (0.01)
- Interconversion: `k_iCAF_to_myCAF` (0.1), `k_myCAF_to_iCAF` (0.05) — TGFb-driven plasticity (Biffi 2019)
- Proliferation: `k_myCAF_prolif` (0.15), `k_iCAF_prolif` (0.10)
- Death: `k_iCAF_death` (0.0165), `k_myCAF_death` (0.0165), `k_apCAF_death` (0.025)
- Secretion (renamed by source): `k_ECM_myCAF_sec`, `k_CXCL12_sec_myCAF`, `k_IL6_sec_iCAF`, `k_TGFb_sec_myCAF`, `k_vas_myCAF_sec`, `k_CCL5_sec_iCAF`
- apCAF mechanism: `k_P1_up_apCAF` (1.44), `k_apCAF_Treg` (0.5), `K_apCAF_Treg` (1e6), `P1_50_apCAF` (1e-12)
- Cell volumes: `vol_iCAFcell`, `vol_myCAFcell`, `vol_apCAFcell` (all 1600 um^3)

### Reaction changes (fibroblast_module_enhanced.m)
- **Activation**: Single `qPSC→aPSC` replaced by three reactions (qPSC→myCAF TGFb-driven, qPSC→iCAF anti-TGFb, qPSC→apCAF constitutive)
- **Interconversion**: New iCAF↔myCAF plasticity driven by TGFb
- **Proliferation**: Separate PDGF-driven proliferation for iCAF and myCAF (no apCAF proliferation)
- **Death**: Constitutive + tumor-eradication clearance for each subtype
- **Cytokine production**: ECM/CXCL12/TGFb/VEGF → myCAF; IL-6/CCL5 → iCAF
- **apCAF mechanism**: Antigen sequestration (P1 sink) + tolerogenic Th→Treg conversion

### CCL5_Treg_module.m
- Removed `k_CCL5_sec_apsc * V_T.aPSC` reaction (CCL5 secretion moved to fibroblast module as `k_CCL5_sec_iCAF * V_T.iCAF`)

### Initial conditions (set_healthy_populations.m)
- `aPSC_density = 10` → `iCAF_density = 3.5`, `myCAF_density = 6.0`, `apCAF_density = 0.5`

### Test statistics
- `apsc_density_baseline` → `caf_density_baseline` (sum of iCAF+myCAF+apCAF)
- `apsc_qpsc_ratio_baseline` → `caf_qpsc_ratio_baseline`
- `stromal_fraction_baseline`, `stromal_barrier_score_d30`: updated to use three subtypes
- All baseline test statistic target values updated to match PDAC literature:
  - Cell densities corrected ~5-10x downward (section thickness conversion fix)
  - Sources: Miksch 2019, Liu 2016, Tang 2014, Vayrynen 2021, Pang 2020, Wang 2020
  - Ratios made internally consistent with densities
  - Added `qpsc_density_baseline` and `total_immune_density_baseline` entries
  - Measurement time changed from day 7 to day 0 (diagnosis state)

### Priors (pdac_priors.csv)
- Removed 9 old aPSC priors, added ~15 new CAF subtype priors

---

## Phase 9: Species Rename & Hypoxia-HIF-VEGF Feedback (Feb 2026)

### Species rename: c_vas -> VEGF (Feb 13)
Renamed the `c_vas` (angiogenic factor) species to `VEGF` throughout the codebase for clarity. The species was already functionally acting as VEGF (produced by tumor/M2/stroma cells, drives vascular growth, targeted by anti-VEGFR2). Rate constant parameter names (`k_vas_Csec`, `k_vas_deg`, `k_vas_Msec`, `k_vas_myCAF_sec`) retained for continuity; only the species name and directly derived parameter names changed:
- `V_T.c_vas` -> `V_T.VEGF`
- `c_vas_50` -> `VEGF_50`
- `c_vas_conc_initial` -> `VEGF_conc_initial`

Files modified: cancer_module.m, macrophage_module.m, fibroblast_module.m, fibroblast_module_enhanced.m, nabpaclitaxel_module.m, simbio_init.m, parameters_PDAC.m, set_healthy_populations.m, sprint_data.m, pdac_priors.csv, baseline_no_treatment.csv, baseline_no_treatment_llm_input.csv

### Hypoxia-HIF-VEGF feedback module (Feb 13)
Created `core/modules/hypoxia_module.m` — models tumor hypoxia, HIF-1alpha signaling, and downstream immunosuppressive effects.

**Rationale**: PDAC is among the most hypoxic solid tumors (median pO2 ~2.7 mmHg vs ~50 mmHg normal). The model previously had vascular dynamics (K, VEGF) but no oxygen tension, no HIF signaling, and VEGF didn't feed back onto immune cells.

**New algebraic parameters** (quasi-steady-state, repeated assignment rules):
- `pO2`: tumor oxygen tension (mmHg). Supply proportional to K, discounted by vascular aberration (VEGF/VEGF_norm_50), consumed by metabolically active cells.
- `HIF`: HIF-1alpha activity (0-1). Inverse Hill function of pO2.

**Three downstream effects**:

1. **HIF-driven VEGF amplification** (positive feedback loop)
   - Reaction: `null -> V_T.VEGF` at rate `k_VEGF_HIF * C_total * HIF`
   - Creates: low pO2 -> HIF -> more VEGF -> aberrant vasculature -> lower pO2
   - Ref: Forsythe 1996 (PMID: 8599938)

2. **HIF-driven PD-L1 upregulation** (IFNg-independent)
   - Parallel reaction on cancer cell synapse (syn_CD8_C1.PDL1)
   - Constitutive — persists even with anti-PD-1 blocking IFNg-driven adaptive PD-L1
   - Separate ceiling r_PDL1_HIF=3 (vs IFNg ceiling r_PDL1_IFNg=6)
   - Applied to cancer cells only (not APCs)
   - Ref: Barsoum 2014 (PMID: 25378023), Noman 2014 (PMID: 25428506)

3. **HIF-driven M2 macrophage polarization**
   - Modified M1->M2 reaction to include `*(1-HIF)` in OR logic
   - Before: `k_M2_pol*V_T.Mac_M1*(1 - (1-H_TGFb)*(1-H_IL10)*(1-H_IL6_M2))`
   - After: `k_M2_pol*V_T.Mac_M1*(1 - (1-H_TGFb)*(1-H_IL10)*(1-H_IL6_M2)*(1-HIF))`

**New parameters** (8 total, 6 varied in SBI):

| Parameter | Value | Units | SBI? | Notes |
|-----------|-------|-------|------|-------|
| `pO2_normal` | 50 | dimensionless | No | Normal tissue pO2 (mmHg). Fixed. |
| `k_O2_demand` | 0.12 | dimensionless | Yes | Calibrated for baseline pO2 ~2.7 mmHg |
| `VEGF_norm_50` | 5000 | pg/mL | Yes | VEGF for half-max vascular aberration |
| `pO2_50` | 7 | dimensionless | Yes | pO2 for half-max HIF activation |
| `n_HIF` | 2 | dimensionless | No | Hill coefficient. Fixed. |
| `k_VEGF_HIF` | 3.3e-4 | pg/(cell*day) | Yes | ~3x basal k_vas_Csec |
| `k_PDL1_HIF` | 2.5e4 | molecule/day | Yes | ~50% of k_out_PDL1 |
| `r_PDL1_HIF` | 3 | dimensionless | Yes | Max fold PD-L1 from HIF |

**Assembly order**: After macrophage_module, before fibroblast_module_enhanced. Forward-references `aPSC_total` from fibroblast_module_enhanced (same pattern as APC_module using `ECM_level`).

---

## Phase 10: NK Cell Module (Feb 2026)

### NK cell innate killing module (Feb 13)
Created `core/modules/NK_module.m` (~100 lines) — antigen-independent innate killing of tumor cells by NK cells. Addresses Gap 6 (partial) from model gap analysis: adds a second killing loop alongside CD8 T cells that does not require antigen presentation or T cell priming.

**New species:**
| Species | Compartment | Module | Description |
|---------|-------------|--------|-------------|
| `NK` | V_T | NK_module | Tumor-infiltrating NK cells |
| `NK` | V_P | NK_module | Circulating NK cell reservoir |

**Hill functions** (repeated assignment rules):
- `H_IL10_NK = V_T.IL10/(IL10_50_NK + V_T.IL10)` — IL-10 suppression of NK function
- `H_TGFb_NK = V_T.TGFb/(TGFb_50_NK + V_T.TGFb)` — TGFb suppression of NK function

**Reactions** (7 total):
1. **Peripheral NK production**: `null -> V_P.NK` at rate `q_NK_P_in * NK_P_steady` (constant source)
2. **Peripheral NK turnover**: `V_P.NK -> null` at rate `q_NK_P_in * V_P.NK` (first-order, steady state at NK_P_steady)
3. **NK recruitment to tumor**: `null -> V_T.NK` at rate `k_NK_rec * V_T * (V_P.NK/(NK_P_steady+cell)) * (1-H_CXCL12_Texcl)` (scaled by peripheral pool, CXCL12 exclusion)
4. **NK death in tumor**: `V_T.NK -> null` at rate `k_NK_death * V_T.NK`
5. **NK clearance on tumor eradication**: `V_T.NK -> null` at rate `k_cell_clear * V_T.NK * (K_C2_recruit/(C_total^2+K_C2_recruit))`
6. **NK killing of C1**: `V_T.C1 -> V_T.C_x` at rate `k_NK_kill * V_T.C1 * V_T.NK/(V_T.NK+K_NK_C*C_total+cell) * (1-H_IL10_NK) * (1-H_TGFb_NK) * (1-H_MDSC)`
7. **NK-derived IFNg**: `null -> V_T.IFNg` at rate `k_IFNg_NK_sec * V_T.NK * (1-H_TGFb_NK)`

**Rule updates:**
- Rule 1 (tumor volume): appended `+V_T.NK*vol_NK/Ve_T`
- Rule 5 (antigen release rate): appended NK killing contribution (same pattern as macrophage phagocytosis)

**New parameters** (10 total, 7 varied in SBI):

| Parameter | Value | Units | SBI? | Notes |
|-----------|-------|-------|------|-------|
| `k_NK_rec` | 1e4 | cell/(mL*day) | Yes | ~50x lower than macrophage recruitment |
| `k_NK_death` | 0.02 | 1/day | Yes | Similar to macrophage turnover |
| `k_NK_kill` | 0.15 | 1/day | Yes | Lower than M1 phagocytosis (0.33) |
| `K_NK_C` | 2.0 | dimensionless | Yes | Same as K_M1_phago |
| `k_IFNg_NK_sec` | 2e-12 | nmol/(cell*day) | Yes | Similar to k_IFNg_Tsec |
| `IL10_50_NK` | 5.0 | nM | Yes | Moderate IL-10 sensitivity |
| `TGFb_50_NK` | 0.1 | nM | Yes | High TGFb sensitivity (Viel 2016) |
| `q_NK_P_in` | 0.1 | 1/day | No | Peripheral homeostasis rate |
| `NK_P_steady` | 2e8 | cell | No | ~200 NK/uL blood (Caligiuri 2008) |
| `D_NK` | 12 | um | No | NK diameter (Cichocki 2019) |
| `vol_NK` | derived | um^3/cell | No | 4/3*pi*(D_NK/2)^3 = ~905 um^3 |

**Initial conditions:**
- `NK_density_initial` = 5000 cell/mm^3 (diagnosis state, Lim 2017 PMID: 28424325)
- Healthy state: 20 cell/mm^3 (set_healthy_populations.m)

**Assembly order**: After macrophage_module, before hypoxia_module. Requires IL-10, TGFb, IFNg species (from Th_module/macrophage_module), H_CXCL12_Texcl (Teff_module placeholder, rule from fibroblast_module — forward reference), H_MDSC (from MDSC_module), C_total/K_C2_recruit/k_cell_clear (from simbio_init/cancer_module).

**What this does NOT include:**
- No NKG2D ligand / MHC-I explicit modeling (future work)
- No NK-specific checkpoint interactions
- No NK synapse compartment

---

## Phase 11: Metabolic Competition Module (Feb 2026)

### Glucose/Lactate metabolic competition module (Feb 14)
Created `core/modules/metabolism_module.m` (~120 lines) — models tumor glucose consumption (Warburg effect) and lactate accumulation in the PDAC TME. Addresses Rank 5 from model gap analysis: creates a positive feedback loop where tumor growth depletes glucose (impairing T cell function) and produces lactate (promoting M2 polarization).

**New species:**
| Species | Compartment | Type | Units | Description |
|---------|-------------|------|-------|-------------|
| `lactate` | V_T | ODE | micromolarity | Tumor lactate from Warburg metabolism |

**Algebraic parameters** (quasi-steady-state, repeatedAssignment):
- `glc` (dimensionless): Glucose supply/demand balance. Same QSS pattern as `pO2` in hypoxia_module. Supply proportional to `K`, discounted by VEGF-driven vascular aberration. Demand from tumor cells (high coefficient) and immune/stromal cells (low coefficient).

**Hill functions** (repeated assignment rules):
- `H_glc_Teff = glc^n / (glc_50^n + glc^n)` — positive Hill (1=full function, 0=impaired). Multiplies k_C_Tcell_eff rule and IFNg secretion.
- `H_lac_M2 = lactate / (lac_50_M2 + lactate)` — lactate M2 polarization drive. Added to M1→M2 OR logic.
- `H_lac_Teff = lactate / (lac_50_Teff + lactate)` — lactate T cell suppression (reserved, not yet connected).

**Reactions** (2 total):
1. **Lactate production**: `null -> V_T.lactate` at rate `k_lac_Csec * C_total` (Warburg metabolism)
2. **Lactate clearance**: `V_T.lactate -> null` at rate `k_lac_clear * V_T.lactate` (vascular washout + MCT export)

**Assembly-time modifications** (4):
1. `k_C_Tcell_eff` rule: appended `* H_glc_Teff` (glucose impairment of T cell killing)
2. CD8 IFNg secretion (`k_IFNg_Tsec`): appended `*H_glc_Teff`
3. Th IFNg secretion (`k_IFNg_Thsec`): appended `*H_glc_Teff`
4. M1→M2 polarization: replaced `*(1-HIF))` with `*(1-HIF)*(1-H_lac_M2))` (same strrep pattern as hypoxia_module)

**New parameters** (9 total, 6 varied in SBI):

| Parameter | Value | Units | SBI? | Notes |
|-----------|-------|-------|------|-------|
| `glc_normal` | 5.0 | dimensionless | No | Normal tissue glucose (~5 mM) |
| `k_glc_tumor` | 0.15 | dimensionless | Yes (σ=1.0) | Glucose demand per cancer cell |
| `k_glc_immune` | 0.02 | dimensionless | Yes (σ=0.7) | Glucose demand per immune cell |
| `glc_50_Teff` | 0.25 | dimensionless | Yes (σ=0.7) | Half-max glucose for T cell function |
| `n_glc_Teff` | 2 | dimensionless | No | Hill coefficient |
| `k_lac_Csec` | 5e-4 | µmol/(cell*day) | Yes (σ=0.7) | Warburg lactate production |
| `k_lac_clear` | 3.0 | 1/day | Yes (σ=0.7) | Lactate clearance (t½ ~5.5h) |
| `lac_50_M2` | 5000 | µM | Yes (σ=0.7) | Lactate for half-max M2 drive (=5 mM) |
| `lac_50_Teff` | 10000 | µM | Reserved | Lactate T cell suppression (=10 mM, not connected) |

**Expected baseline behavior** (C_total~5e4, K~1e4):
- Glucose: `glc ≈ 0.29` (~1.4 mM actual). Moderately depleted.
- `H_glc_Teff ≈ 0.57`. T cell killing/IFNg reduced to ~57% of normal.
- Lactate: `≈ 8333 µM` (8.3 mM) steady state. In PDAC range (5-15 mM).
- `H_lac_M2 ≈ 0.62`. Strong additional M2 polarization drive.

**Assembly order**: After hypoxia_module, before fibroblast_module_enhanced. Reuses `VEGF_norm_50` from hypoxia_module. Forward-references `aPSC_total` (fibroblast_module_enhanced).

**Unit note**: Lactate uses `micromolarity` (µM) in SimBiology because `millimolarity`/`millimole/liter` are not recognized by SimBiology's `addspecies`. Values are 1000× the mM equivalents (e.g., lac_50_M2=5000 µM = 5 mM).

---

### cDC1/cDC2 dendritic cell subtype split

**Rationale**: In PDAC, cDC1 (BATF3+ cross-presenting) are uniquely scarce (~79-fold less than lung adenocarcinoma; Hegde 2020, PMID: 31932165), creating a cross-presentation bottleneck that limits CD8 priming. Splitting DCs into cDC1 and cDC2 makes cDC1 abundance a gatekeeping variable for immune activation.

**Species changes**: Replaced generic `APC`/`mAPC` (4 species in V_T/V_LN) with 8 subtype species: `cDC1`, `cDC2`, `mcDC1`, `mcDC2` in both V_T and V_LN. Added `V_T.XCL1` (chemokine, nanomolarity).

**Functional routing**:
- CD8 T cell priming (`H_mAPC`): `V_LN.mcDC1` only (cross-presentation)
- Th/Treg priming (`H_APCh`, `H_APC`): `V_LN.mcDC2` only (MHC-II)
- IL-12 secretion: `V_T.mcDC1` only (cDC1 are primary IL-12 producers)
- Antigen uptake: `APC_total_T` (both subtypes, aggregate alias)
- GVAX migration: `V_ID.mAPC → V_LN.mcDC2` (GM-CSF recruits cDC2/moDCs)

**NK-XCL1-cDC1 positive feedback loop**: NK cells secrete XCL1 (suppressed by TGFb), which boosts cDC1 recruitment via `H_XCL1_cDC1`. cDC1 produce IL-12, which activates NK cells, closing the loop. This is the primary mechanism by which NK cells promote adaptive immunity in PDAC (Bottcher 2018, PMID: 30185328).

**New parameters**:

| Parameter | Value | Units | Prior | Description |
|-----------|-------|-------|-------|-------------|
| `f_cDC1` | 0.6 | dimensionless | Yes (σ=0.3) | cDC1 fraction of total DCs |
| `k_APC_mat_cDC1` | 2.0 | 1/day | Yes (σ=0.7) | cDC1 maturation rate |
| `k_APC_mat_cDC2` | 1.5 | 1/day | Yes (σ=0.7) | cDC2 maturation rate |
| `k_XCL1_NK` | 5e-12 | nanomole/cell/day | Yes (σ=1.0) | XCL1 secretion by NK |
| `k_XCL1_deg` | 2.0 | 1/day | Yes (σ=0.7) | XCL1 degradation |
| `XCL1_50_cDC1` | 0.1 | nM | Yes (σ=1.0) | XCL1 EC50 for cDC1 recruitment |

**Removed parameters**: `k_APC_mat` (replaced by subtype-specific rates).

**Backward compatibility**: Aggregate alias rules (`APC_total_T`, `mAPC_total_T`, `mAPC_total_LN`) sum subtypes for downstream modules. V_ID injection site species remain generic (`V_ID.APC`, `V_ID.mAPC`).

**Assembly order**: APC_module unchanged. NK_module adds XCL1 species and overwrites `H_XCL1_cDC1` default from simbio_init.

**Key references**: Hegde 2020 (PMID: 31932165), Broz 2014 (PMID: 25417636), Spranger 2017 (PMID: 28983061), Bottcher 2018 (PMID: 30185328).

---

## Qualitative Bounds Evidence Audit (Feb 2026)

Systematic audit and update of `tests/qualitative_bounds.yaml` — the file encoding literature-backed biological thresholds for structural model validation. Every bound now carries a required `evidence_snippet` field with a verbatim quote from the cited source.

### Schema change (metadata/qualitative_bounds.py)
`evidence_snippet` changed from `Optional[str]` (default `None`) to required `str`. All bounds must now carry a verbatim quote from the primary source.

### Evidence snippets added to all bounds
Added `evidence_snippet` fields to all 27 original bounds by navigating to full-text papers (PMC, PubMed, AACR journals) and extracting verbatim quotes that support each bound's numerical value.

### Numerical value corrections (3 bounds)

**checkpoint_fold_increase_min** — lower bound 100 → 2.0:
The original 100x fold-increase had no literature support. Hui 2017 (Science, doi:10.1126/science.aaf1292) shows PD-1 engagement reduces IL-2 secretion by ~63% in vitro, implying ~2.7x suppression. Blocking PD-1 should restore at least 2-fold function. Source changed from Topalian 2012 to Hui 2017.

**mycaf_fraction_min** — lower bound 0.50 → 0.40:
The original 50% floor sat at the boundary of the measurement (Elyada 2019: myCAFs 45.3 ± 4.1%). Reduced to 40% for a conservative floor approximately 1 SD below the mean. Source changed from Ogawa 2021 to Elyada 2019 (doi:10.1158/2159-8290.CD-19-0528).

**clinical_lactate_range** — lower bound 5000 → 2000 µM:
Halbrook 2018 (HR-MAS NMR, human PDAC tissue) shows lactate at 244 ± 155% of healthy tissue. With normal pancreas lactate ~1–2 mM, the low end of the PDAC range (~2.4 mM) falls below the previous 5 mM floor. Reduced to 2 mM to accommodate PDAC-specific variance.

### Reference upgrades (PDAC-specific sources)

| Bound | Old source | New source | Reason |
|-------|-----------|------------|--------|
| `lactate_min_diagnosis` | Walenta 2000 (cervical cancer) | Halbrook 2018 (PDAC HR-MAS NMR) | PDAC-specific data |
| `lactate_floor_progression` | Walenta 2000 (cervical cancer) | Halbrook 2018 (PDAC HR-MAS NMR) | PDAC-specific data |
| `clinical_lactate_range` | Walenta 2004 (cross-tumor) | Halbrook 2018 (primary) + Walenta 2004 (secondary) | PDAC-specific primary |
| `clinical_hif_range` | Shibaji 2003 (IHC only) | Koong 2000 (primary, pO2 electrode) + Shibaji 2003 (secondary) | Quantitative pO2 → Hill function derivation |
| `clinical_vegf_range` | Jain 2005 → Hogendorf 2014 | Holm 2016 (primary, TIF data) + Hogendorf 2014 (secondary) | TIF-to-plasma ratio justifies compartment range |

### Rationale improvements (3 bounds)

**ecm_concentration_stability**: Explicitly labeled as a structural guard — no direct longitudinal ECM measurements in PDAC exist over 90 days. The 0.5–1.5x range catches model bugs while the persistent desmoplastic phenotype argues against dramatic changes.

**ecm_treatment_tolerance**: Explicitly labeled as a structural guard — no study directly measures ECM changes under GVAX+nivolumab. The bound encodes mechanistic reasoning: checkpoint inhibitors target PD-1/PD-L1, not PSCs/CAFs.

**clinical_hif_range**: Now explicitly derives the 0.5–1.0 range from Koong 2000 pO2 data (0–5.3 mmHg) through the model's inverse Hill function: HIF = 1/(1+(pO2/pO2_50)^n) with pO2_50 ~5 mmHg.

### New bounds added (2)

**icaf_fraction_range** [0.20, 0.65] dimensionless:
iCAF fraction of total activated CAFs at diagnosis. Elyada 2019 (KPC model, flow cytometry, n=20): iCAFs 44.4 ± 3.9% of PDPN+ CAFs. Range accommodates inter-patient variability.

**apcaf_fraction_max** upper 0.25 dimensionless:
Maximum apCAF fraction of total activated CAFs. Elyada 2019: apCAFs 10.3 ± 1.25%, the smallest subtype. Ceiling of 25% ensures apCAFs remain a minority.

### Final state
29 bounds total (was 27), all with evidence_snippet, no duplicate IDs, organized across 6 categories: baseline_tme (7), cell_composition (8), clinical_range (4), temporal_dynamics (6), treatment_response (2), qss_consistency (2).
---

## Phase 12: Th Exhaustion Decoupling (Apr 2026)

### Motivation
Cluster B (`cdc1_density × th_exhausted_fraction`, prior pairwise φ = −0.483 at τ_cohort=1.0) was identified as the top remaining structural tension after the cohort random-effect prototype (PR #108). Tracing the model with `qsp_sim` revealed two issues:

1. **Th_exh fraction was structurally tied to bulk death.** The single Th→Th_exh reaction in `Th_module.m` ran at rate `k_CD4_death × V_T.Th` — i.e., all CD4 "death" was relabeled as exhaustion rather than being mechanistically driven by chronic antigen / immunosuppression. At steady state, Th_exh fraction reduced to `k_CD4_death / (k_CD4_death + k_cell_clear)` = 0.5 at default priors vs. observation 0.78.
2. **Bifurcation at `k_CD4_death ≈ k_Th_pro_tumor`.** Forcing `k_CD4_death` upward to match the Th_exh fraction observation pushed the Th compartment past the proliferation/death bifurcation, crashing CD4 density from 213 → 0.1 cells/mm² (vs `cd4_helper_density` obs 17). No (k_CD4_death, rho_Th_max) regime simultaneously satisfied both targets.

The handoff narrative attributed the tension to a cDC1 ↔ Th-priming coupling through `H_mAPC`, but the model code routes Th priming through `mcDC2` (not `mcDC1`), and Th_exh fraction at SS does not depend on APC dynamics at all. The actual mechanism was the absence of an exhaustion driver decoupled from death.

### Reaction changes (`core/modules/Th_module.m`)
Replaced the single death-relabel reaction with four parallel mechanisms mirroring `Teff_module.m`'s CD8 exhaustion routes:

| Reaction | Rate | Purpose |
|---|---|---|
| `V_T.Th -> null` | `k_CD4_death * V_T.Th` | Regular Th cell death (no longer routed to Th_exh) |
| `V_T.Th -> V_T.Th_exh` | `k_Th_basal_exh * V_T.Th * C_total/(C_total+K_C_exh+cell)` | Baseline exhaustion from chronic antigen |
| `V_T.Th -> V_T.Th_exh` | `k_TGFb_Th_exh * V_T.Th * H_TGFb_Th_exh` | TGFβ-driven chronic immunosuppression (cell-specific Hill) |
| `V_T.Th -> V_T.Th_exh` | `k_cell_clear * V_T.Th * (K_C2_recruit/(C_total^2 + K_C2_recruit))` | Antigen-clearance funneling (unchanged) |

`K_C_exh` was already defined in the CD8 exhaustion block. A dedicated `H_TGFb_Th_exh = V_T.TGFb / (V_T.TGFb + TGFb_50_Th_exh)` was added to `cytokine_tcell_module.m`, following the existing per-cell-type pattern (`H_TGFb_Teff`, `H_TGFb_NK`, `H_TGFb_APC`). Reusing the generic `H_TGFb` (which is shared with Th→Treg differentiation, M1→M2 polarization, and fibroblast activation) would have created cross-effect contamination during SBI.

### Parameter changes (`parameters/parameters_PDAC.m`, `pdac_priors.csv`)
| Parameter | Default | Units | σ (lognormal) | Notes |
|---|---|---|---|---|
| `k_Th_basal_exh` | 0.05 | 1/day | 1.0 | Mirrors `k_CD8_basal_exh` default and prior σ |
| `k_TGFb_Th_exh` | 0.3 | 1/day | 1.0 | Estimated; submodel target extraction pending — TGFβ-driven CD4 exhaustion / Treg-skewing rate from in vitro literature |
| `TGFb_50_Th_exh` | 0.14 | nM | 1.0 | Cell-specific TGFβ EC50 for CD4 exhaustion. Mirrors `TGFb_50_Teff` (CD8) default. Decouples Th-side TGFβ dose-response from generic `TGFb_50`. |

All three flagged `derivedFrom = {'estimated'}`. Submodel target extraction is the next step to anchor `k_TGFb_Th_exh` and `TGFb_50_Th_exh` against TGFβ in vitro Th-suppression / dysfunction kinetics.

### Calibration window
At default priors with these new parameters: Th_exh fraction 0.49 → 0.57 (target 0.78), CD4 density 213 → 0.40 (target 17). 2D feasibility grid sweeping `(k_Th_pro_tumor, k_TGFb_Th_exh)` finds a feasible point at **(0.4, 0.8) → CD4 density = 14, Th_exh frac = 0.75**, both within ~1.5σ of prior medians (`k_Th_pro_tumor` at +1.22σ, `k_TGFb_Th_exh` at +0.98σ). The proliferation/death bifurcation still produces a sharp transition along the grid, but the feasible region is now reachable from prior tail draws without extreme combinations.

Compared to an alternative design that reused the shared `H_TGFb`, the dedicated EC50 makes the feasible region easier to hit because (a) SBI has two degenerate knobs (`k_TGFb_Th_exh` rate × `TGFb_50_Th_exh` EC50) instead of one shared rate, and (b) Th-side TGFβ tuning no longer drags `TGFb_50` away from its Treg / macrophage / fibroblast roles.

### Files
- `core/modules/Th_module.m`: 1 reaction split into 4, with 2 new `addparameter` calls
- `core/modules/cytokine_tcell_module.m`: added `TGFb_50_Th_exh` parameter + `H_TGFb_Th_exh` Hill function
- `parameters/parameters_PDAC.m`: 3 new params
- `parameters/pdac_priors.csv`: 3 new rows
- `cpp/qsp/ode/`, `resources/cpp/param_all.xml`: regenerated via `make refresh-cpp`; `make parity` passes (47.16s, 2/2 trajectories match within tolerance)


---

## 2026-05-02: cDC1/cDC2 recruitment-target rename + prior widening

### Rationale
Local `qsp_sim` traces (sweeping `k_PSC_to_apCAF` 50× and `APC0_cDC1_T` 1800×) showed:
- The SBI-named "top driver" `k_PSC_to_apCAF` for `cdc1_density_baseline` and `exhaustion_worsens_untreated_mechanistic` has near-zero absolute leverage (~2% on cdc1, 0.3% on exhaustion ratio) — a Jacobian aliasing artifact under low effective rank, not a real driver.
- `cdc1_density` is in fact ~180× attenuated below `APC0_cDC1_T × V_T` because the V_T equation is **recruit + first-order death** (not homeostatic relaxation): steady-state realized density = `APC0 × V_T × (1-H_pore_APC) / (1 + maturation_drain/k_APC_death)` ≈ `APC0 × V_T × 0.02 × 0.2`. With H_pore_APC ≈ 0.98 and maturation drain ~5×, the realized density is decoupled from the "homeostatic" parameter label.

Literature review (Liu 2007/2009, Salmon 2016, Böttcher 2018, Broz 2014, Hegde 2020, Lin 2020, Kamath 2002, Cytlak 2024 — full citations and synthesis in `notes/model/cdc1_homeostasis_vs_recruitment.md`) confirms tumor cDC1/cDC2 are **recruitment-and-apoptosis dominated, not homeostatically defended**. PDAC specifically (Hegde 2020 = 79× cDC1 deficit driven by IL-6-induced apoptosis and FLT3L sequestration) makes the recruit-form V_T equation biologically correct — reformulating to homeostatic relaxation would elide the central PDAC DC phenotype. The V_T-vs-V_LN asymmetry in `core/modules/APC_module.m` is appropriate.

### Parameter rename
| Old name | New name | Reason |
|---|---|---|
| `APC0_cDC1_T` | `cDC1_recruit_target_T` | Clarifies the parameter is a *recruitment-target density* (`k_recruit / k_APC_death` numerator), not a realized density |
| `APC0_cDC2_T` | `cDC2_recruit_target_T` | Same |

`APC0_cDC1_LN` and `APC0_cDC2_LN` retained — the V_LN equation IS homeostatic relaxation and the parameter is correctly named there.

### Prior widening
| Parameter | Median | Old σ | New σ | Why |
|---|---|---|---|---|
| `cDC1_recruit_target_T` | 5e4 cell/mL | 0.7 | 2.0 | Stale σ assumed parameter ≈ realized density; with the recruit-target semantic, σ must span the recruitment-rate scale that produces realized density consistent with Plesca 2022 (7.82 cells/mm²) under the model's full attenuation chain. SBI fits the recruit-target via the realized-density observable; do not re-anchor a literature prior on this. |
| `cDC2_recruit_target_T` | 2e5 cell/mL | 0.7 | 2.0 | Same |

### Submodel targets excluded
Moved to `calibration_targets/submodel_targets/excluded/` with `parameter-semantic mismatch (moved 2026-05-02)` block in the README:
- `APC0_cDC1_T_PDAC_deriv001.yaml` — forward model used stale `H_ECM_APC_mig` name, assumed occupancy ~0.6 (actual H_pore_APC ≈ 0.98 at default ECM); doesn't account for maturation drain.
- `APC0_cDC2_T_PDAC_deriv001.yaml` — assumed `H_pore_APC ~ 0` at baseline; same realized-vs-recruit-target confusion.

The same Plesca 2022 data is anchored correctly by the full-model targets at `calibration_targets/baseline_no_treatment/cdc{1,2}_density_baseline_PDAC_deriv001.yaml`, which evaluate against the realized-density observable through the model's full attenuation chain.

### Files touched
- `core/modules/APC_module.m`: parameter-name references in `addparameter` calls and the two recruitment ReactionRate strings; docstring updates.
- `parameters/parameters_PDAC.m`: parameter-name updates + Notes/derivedFrom blocks.
- `parameters/pdac_priors.csv`: rename + σ widen.
- `calibration_targets/submodel_targets/{APC0_cDC1_T,APC0_cDC2_T}_PDAC_deriv001.yaml` → `excluded/`.
- `calibration_targets/submodel_targets/excluded/README.md`: 2026-05-02 entry.
- `cpp/qsp/ode/`, `resources/cpp/param_all.xml`, `model_structure.json`, `species_units.json`: regenerated via `make refresh-cpp`.
- New: `notes/model/cdc1_homeostasis_vs_recruitment.md` (literature review).
