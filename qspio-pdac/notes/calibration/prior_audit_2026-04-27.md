# Prior Audit — `parameters/pdac_priors.csv` (2026-04-27)

Audit of support / median / sigma for all 186 priors, with focus on recently-added or recently-modified parameters (IL-1 mechanism, ECM redesign, IFNg slope reparam, GVAX cascade, f_active_TGFb, IL-6 EC50 fixes, CAF subtypes, CCL5 axis).

**Convention.** All entries are lognormal with `dist_param1 = log(median)` and `dist_param2 = sigma_log`. 95% CI ≈ median × exp(±1.96·sigma). σ=0.5→×2.7, 0.7→×3.9, 1.0→×7, 1.2→×10.5, 1.5→×19.

---

## 1. Critical (fix before next SBI run)

### 1.1 `f_nTreg` — wrong distribution family for a fraction
- Current: `lognormal`, median=0.25, σ=0.7, bounds [0.01, 0.95].
- Problem: fraction in (0,1). Lognormal mass extends >1 before truncation; truncation at 0.95 creates a mass spike there.
- Suggested: `beta` with mean ≈0.05–0.10, or trunc-normal centered 0.05–0.10, σ≈0.03 on (0,0.5). Tregs are 5–10% of CD4+ in healthy tissue and 15–30% within the CD4 compartment in PDAC TME (Wang 2017 PMID 29162176; Tang 2014 PMID 24441136). Median 0.25 is high for peripheral; appropriate for tumor-infiltrating CD4 fraction — confirm intended compartment.
- Confidence: High on family change; medium on median (depends on compartment).

### 1.2 `f_active_TGFb` — σ=1.5 unbounded for a fraction
- Current: lognormal, median=0.1, σ=1.5. ~10% of mass above 1, unphysical.
- Biology: latent→active TGFβ conversion 1–10% physiologic, 10–30% under integrin-αvβ6/8 in fibrotic tissue (Munger 1999 PMID 10025395; Annes 2003 PMID 12810700). Median 0.1 OK.
- Suggested: beta(α=2, β=18) (mean 0.10) or trunc-lognormal σ≤0.7 with `upper_bound=1.0`.

### 1.3 `k_IFNg_response` — σ=1.5 too wide for a structurally critical slope
- Current: median=0.2 1/pM, σ=1.5 → 95% CI 0.01–3.7. Slope (per memory `project_ifng50_literature_gap`) gates M2→M1 polarization; high tail produces instantaneous polarization, low tail none.
- Suggested: σ=1.0 (or 1.2 if want more headroom). Median 0.2 OK.

### 1.4 `k_TGFb_sec_myCAF` — encoded `dist_param1` rounded; σ=1.5 wide; median high
- Current: median=1e-9 nM·L/cell/day, dist_param1=−20.72 (should be −20.7232658), σ=1.5 (×360 span — widest in the file).
- Yamazaki 2020 PMID 32001483: ~5 pg/cell/day myofibroblast TGFb ≈ 2e-10 nmol/cell/day (MW 25 kDa). Median ~5× high.
- Suggested: median=2e-10, σ=1.0; recompute `dist_param1` to full precision.

### 1.5 `k_cDC1_Ag_transfer` — σ=1.5, no upper bound
- Current: 1e-7 1/(cell·day), σ=1.5. Bimolecular rate × cDC1 × mcDC2 (each 10⁴–10⁵) at high tail → instantaneous transfer.
- Suggested: σ=1.0, `upper_bound=1e-6`. Median 1e-7 OK.

### 1.6 `k_T_activation_per_APC` — calibrated 100× from baseline; prior centers above literature
- Current: median=1e-4 1/(day·cell), σ=1.0. parameters_PDAC.m note: "Calibrated: 5× from baseline (1e-7)" but value is 1000× baseline.
- Miller 2004 (APC scan rate) × activation probability gives ~1e-7 to 1e-5 1/(day·cell). Prior 95% CI 1e-5–1e-3 sits **above** literature.
- Suggested: median=1e-6, σ=1.5; or keep 1e-4 but document calibration target.

### 1.7 `v_T_search_volume` — calibrated 30× above kinetic estimate
- Current: 1e-5 mm³/day, σ=1.0. Boissonnas 2007 (10 µm/min, persist ~2 min) → ~3e-7 mm³/day.
- Suggested: median=1e-6, σ=1.0.

### 1.8 `K_apCAF_Treg` — saturation constant at baseline density
- Current: 1e7 cells. apCAF baseline density (parameters_PDAC.m) ~1.2e7 cells/mL → Hill near saturation by default, kills identifiability of `k_apCAF_Treg`.
- Suggested: median=1e8 cells, σ=1.0. Or pivot to density units.

### 1.9 `phi_col_50_Kmax` — units / Hill form check
- Current: median=2.0 dimensionless. If Hill is `phi/(phi_col_50_Kmax + phi)` with phi a fraction <1, EC50 of 2.0 means never half-saturates. **Verify Hill form in `fibroblast_module_enhanced.m` before changing** — may be parameterized as `phi/phi_col_ref` ratio (in which case 2.0 is fine).

---

## 2. Moderate (citation needed / σ calibration)

### 2.1 CAF subtype death rates — over-correlated medians
- `k_qpsc_death`, `k_iCAF_death`, `k_apCAF_death`, `k_myCAF_death` all 0.0165 1/day. iCAFs are reversible/transient (~3 day timescale per Öhlund 2017 PMID 28530656).
- Suggested (hierarchical group still active): qpsc 0.005, iCAF 0.05, myCAF 0.015, apCAF 0.03 (σ 0.5–0.7).

### 2.2 `k_PSC_to_apCAF` — 40× slower than other PSC→subtype
- Current: 0.01 vs 0.4 (myCAF), 0.2 (iCAF). Dominguez 2020 PMID 31988145 reports apCAF emergence on similar timescale.
- Suggested: 0.05, σ=1.0.

### 2.3 `k_M1_pol`, `k_M2_pol` — medians 5–10× below in vitro
- Xue 2014 PMID 24530056: M0→M2 IL-4 ~24h (k≈0.5–1/day). Listed: 0.045/0.1. Submaximal Hill in tissue may justify, but flag for submodel target (per memory `project_il10_il6_tissue_targets`).

### 2.4 `k_Mac_death` — likely too low
- Current: 0.02 (t½ ~35d). Franklin 2014 PMID 24812208: TAM t½ 5–14d → k 0.05–0.15.
- Suggested: 0.05, σ=0.7. Already queued as submodel target.

### 2.5 `IL6_50_*` — three identical (0.015 nM), trans- vs classical signaling
- IL-6R Kd 0.01–0.05 nM (Heinrich 2003 PMID 12773095). Trans-signaling (iCAFs) ~10× lower affinity.
- Suggested: keep MDSC/M2 at 0.015 nM, raise iCAF to 0.05; or σ=1.0 across the three to allow ±10× spread. Low priority.

### 2.6 `Emax_IL1R1` — fold suppression
- Current: 9.0. Bhowmick 2009 PMID 19549774: TGFb ↓ IL-1R1 mRNA ~3–10×.
- Suggested: 5, σ=0.7.

### 2.7 `n_IL1`, `n_Texh`, `n_fib` Hill coefficients — σ=0.7 wide
- 95% CI ~0.25–4 — Hill ultrasensitivity not biologically supported.
- Suggested: σ=0.5, `upper_bound=4.0`.

### 2.8 ECM mechanical (`E_50_Texh`, `E_50_fib`, `E_ref`, `n_stiff`)
- Rice 2017 PMID 28369035: PDAC 5–10 kPa vs healthy ~1 kPa → ratio 5–10. EC50s of 4–5 (dim, kPa-scaled) place half-effect at PDAC operating point. **No change.**

### 2.9 `K_stiff_niche` (1e5 cells) — internally consistent
- Negligible vs cancer-paracrine until cancer cells gone (intended post-therapy fibrotic-scar mechanism). No published anchor; OK.

### 2.10 σ=1.2 on well-anchored published values
- `q_nCD8_LN_in`, `q_nCD8_P_out` (Autissier 2010), `k_nCD8_death`, `k_nTreg_death`: tighten to σ=0.5–0.7.

---

## 3. Minor / nitpicks

- `IL2_50` and `IL2_50_Treg` both 0.32 nM. Yu 2009 PMID 19748982: Tregs 10–100× higher affinity. Suggested IL2_50_Treg=0.01 nM, σ=0.5.
- `d_crit_APC=2.1` < `d_crit_T=2.3` — DCs are larger than T cells; likely should be reversed.
- `K_C_exh=1e6` cells but tumor 1e8–1e9 → Hill always saturated, kills shape identifiability. Raise to 1e8.
- `K_initial=10300` cell — verify still used; scenarios use `evolve_to_diagnosis`.
- `EC50_GMCSF` σ=1.103, `k_APC_mature_ID` σ=0.853 — odd residuals from prior calibration; cosmetic.
- `vol_*CAFcell` all 1600 µm³ — could be 2000–4000 (Wirtz 2009) but conversion-constant tightness OK.

---

## 4. Looks fine (no detail)

PK/checkpoint binding (kd_PD1_*, kon_*, kd_CTLA4_*, Chi_*, koff_*), antibody transport (Dk_*, SA_*, Q_LD), naive T pool (Q_nCD8_thym, K_nT_pro), TCR/synapse (A_syn, d_syn, TCR_tot_abs, phi_TCR, N_TCR), antigen processing (k_up, k_xP_deg, k_p_deg, k_on), Hill EC50s with literature anchors (CCL2_50, TGFb_50, TGFb_50_APC, TGFb_50_Teff, IL12_50, IL10_50, ArgI_50_*, NO_50_Teff, CXCL12_50_Texcl, CCL5_50_Treg), bulk cytokine secretion/degradation rates (σ 0.5–1.0), GVAX cascade caps (alpha_Th_cap=5, alpha_Treg_cap=2, CD8_50_Tcap, apCAF_50_Tcap), APC homeostatic densities (APC0_cDC*_T/LN at 0.2/0.8 split).

---

## 5. Cross-parameter consistency

1. iCAF turnover balance (k_PSC_to_iCAF + k_iCAF_to_myCAF + k_iCAF_death) gives t½ ~2d — consistent with Öhlund 2017. ✓
2. `k_PSC_to_apCAF` (0.01) << `k_apCAF_Treg` (0.03) — verify `k_apCAF_Treg` is *Treg induction per apCAF* (not depleting apCAF).
3. **`k_M2_pol` (0.1) / `k_M1_pol` (0.045) → SS M2:M1 = 2.2:1**, but `m1_m2_ratio_initial = 0.756` implies M1 ≈ M2 at IC. IC vs prior medians inconsistent — drift to SS will fight calibration targets at diagnosis.
4. `K_C_exh=1e6` saturates Hill 100–1000× — kills identifiability of `k_C_CD8_exh`.

---

## 6. Deep audit by subsystem

**Verification of `phi_col_50_Kmax` Hill form (resolves §1.9):** in `core/modules/fibroblast_module_enhanced.m:276` the rule is `C_max = (V_T.K + k_stroma_support * aPSC_total) * (1 - phi_collagen/(phi_collagen + phi_col_50_Kmax))`. Form is `phi/(phi+phi_col_50_Kmax)` with `phi_collagen` a true fraction (0.05–0.30 in PDAC). At `phi_col_50_Kmax=2.0` the barrier term is 2.5–13% — effectively disengaged. Matches Rhim 2014 / Ozdemir 2014 design intent (intentionally weak ECM barrier; stromal *support* dominates). **No unit error**, but median is far above operating range; if a real ECM barrier is wanted, push median toward 0.10–0.20. Recommend documenting intent in parameter Notes; downgraded from Critical (verify) → Minor (by design).

Coverage convention: parameters covered in detail in §1–§3 are tagged `(see §X)` and not repeated. All values quoted as `(median, σ)` where σ is `dist_param2`.

### 6.1 CAF / fibroblast

- `k_PSC_to_myCAF` (0.4, 1.0) — myCAF emergence in TGFβ-stimulated PSC cultures completes in 3–7 d (Öhlund 2017 PMID 28530656; Biffi 2019 PMID 30366930), k ≈ 0.1–0.3/day. Mildly fast. **⚠ shift median to 0.2; σ OK. (m)**
- `k_PSC_to_iCAF` (0.2, 1.0) — IL-1α/LIF-driven program, 2–4 d (Biffi 2019). **✓ OK.**
- `k_iCAF_to_myCAF` (0.1, 1.0) and `k_myCAF_to_iCAF` (0.05, 0.7) — phenotypic plasticity per Öhlund 2017 (TGFβ→myCAF, IL-1→iCAF, ~1-week phenotype half-life). Direction asymmetry appropriate for TGFβ-rich PDAC. **✓ OK.**
- `k_apCAF_Treg` (0.03, 1.0) — Treg-induction-per-apCAF (verified from module line 400: "Max Th->Treg conversion rate via apCAF"). Huang 2022 (PMID 35574056): apCAF→Treg ~10–20% over 5 d → k ≈ 0.02–0.04. **✓ OK; resolves §"Verification" item 4.**
- `k_P1_up_apCAF` (1.44, 1.0) — analogous to DC MHC-II loading (Sallusto 1995); cross-cell-type extrapolation. **⚠ widen σ to 1.2. (m)**
- `P1_50_apCAF` (1e-9 M, 1.2) — standard MHC-II EC50 (Demotz 1990, 1–10 nM). **✓ OK.**
- `f_IL6_iCAF` (3.0, 0.7) — IL-6 autocrine fold-enhancement; consistent with Biffi 2019 STAT3-inhibition data. **✓ OK.**
- `k_CAF_encounter` (1e6 cell/(mL·day), 1.0), `k_CAF_const` (2e6 cell/mL, 1.0) — PSC density 1–5×10⁶ cells/mL (Apte 1998 PMID 9462235). Encounter rate model-specific. **✓ OK.**
- `f_stroma_max` (3.0, 0.7) — PDAC stroma:cancer ratio 4–9:1 by volume (Whatcott 2015 PMID 25878334). Median low. **⚠ shift to 5.0. (m)**
- `k_stroma_support` (1.0, 0.7) — dimensionless coupling; couples tightly with `f_stroma_max` and `K`. **Verify identifiability.**
- `k_myCAF_prolif` (0.15), `k_iCAF_prolif` (0.10), `k_apCAF_prolif` (0.05), all σ=1.0. PSC doubling 2–4 d → k 0.17–0.35 in vitro (Apte 2004 PMID 15233793); slower in vivo. Ordering tracks expected myCAF > iCAF > apCAF. **✓ OK.**
- `vol_*CAFcell` (all 1600 µm³, σ=0.3) — Wirtz 2009 PMID 19238201: PSCs ~2000–3000 µm³ in 2D; activated myofibroblasts up to 4000+. Identical medians ignore subtype morphology. **⚠ differentiate: qPSC 1500, iCAF 2000, myCAF 3500, apCAF 2500. Low priority. (m)**
- `k_TGFb_sec_myCAF` — see §1.4. **(C)**
- `k_IL6_sec_iCAF` (4e-11 nmol·L/cell/day, 0.7) — Biffi 2019: ~17 ng/(mL·day) at 10⁵ cells/mL → ~2× high. **⚠ shift to 2e-11; σ OK. (m)**
- `k_CXCL12_sec_iCAF` (1e-10, 1.2) — Feig 2013 (PMID 24297891) confirms FAP+ CAFs as dominant source; no per-cell rate. σ=1.2 appropriate. **✓ OK.**
- `k_CCL5_sec_iCAF` (1e-10 nmol/cell/day, 1.2) — Tanaka 2018 (PMID 29467282): iCAF CCL5 ~3–10× M2 rate (M2 ~5e-12). Median is 2–7× high. **⚠ shift to 3e-11. (m)**
- `k_vas_myCAF_sec` (5e-3 pg/cell/day, 1.0) — myCAFs ~30–50% of cancer-cell rate; cancer dominates VEGF in PDAC (Erkan 2009). Mildly high. **⚠ shift to 2e-3; σ OK. (m)**

### 6.2 DC / APC

- `APC0_cDC1_T` (5.4e5 cell/mL, 0.7), `APC0_cDC2_T` (2.16e6, 0.7) — Hegde 2020 (PMID 32109377): total DC ~1–5% of CD45+ in PDAC; CD45+ ~10⁷/mL → cDC1 ~5×10⁴, cDC2 ~2×10⁵. **Tumor medians ~10× high.** LN values OK. **⚠ shift APC0_cDC1_T to 5e4, APC0_cDC2_T to 2e5; σ OK. (M)**
- `APC0_cDC1_LN` (2.4e5), `APC0_cDC2_LN` (9.6e5) — DCs enriched in LN. **✓ OK.**
- `k_APC_mat_cDC1` (2.0), `k_APC_mat_cDC2` (1.5), σ=0.7 — DAMP/PAMP-driven maturation 12–24h → k ≈ 1–2 (Mellman 2001). **✓ OK.**
- `DAMP_50` (1e4 cell/day, 1.0) — half-max at 10⁴ dying cells/day; no absolute threshold reported (Apetoh 2007 PMID 17704786). **✓ OK.**
- `TGFb_50_APC` (0.07 nM, 0.7) — canonical TGFβ EC50 0.05–0.1 nM (Heldin 1997). **✓ OK.**
- `k_APC_mig` (4.0, 1.2) — DC→LN migration within 24h (Martín-Fontecha 2003 PMID 14530375), k ≈ 1–4. **⚠ tighten σ to 0.7. (m)**
- `k_mAPC_death` (0.3, 1.2) — mature DC LN half-life 2–3 d (Kamath 2002), k ≈ 0.25–0.35. **⚠ tighten σ to 0.5. (m)**
- `k_APC_death` (0.15, 0.7) — immature DC half-life 5–7 d. **✓ OK.**
- `k_APC_recruit_ID` (1e5 cell/day, 1.0) — model construct, no anchor. **Verify.**
- `k_APC_mature_ID` (2.73, 0.85) — odd σ residual. **⚠ round σ to 0.7. (m, cosmetic)**
- `EC50_P1_mature` (0.1 nM, 1.0) — peptide MHC EC50 0.1–1 nM (Demotz 1990). **✓ OK.**
- `k_APC_mig_ID_LN` (0.7, 1.0) — slower than mature cDC, plausible. **✓ OK.**
- `K_APC_mig` (1e5 cell, 1.0) — Hill far below saturation given LN cDC ~10⁶. **✓ OK.**
- `k_APC_recruit0` (100 cell/day, 1.0) — steady state should be `APC0 × k_APC_death × V_T` ≈ 8e3–8e4 cells/day in mL-scale tumor. **Median 3 OOM low; verify if IC enforcement bypasses this.** **(M)**
- `k_cDC1_Ag_transfer` — see §1.5. **(C)**
- `y_Ag_per_cell` (1.7e-8 nmol/cell, 1.0) — total cellular protein ~200 pg = 6.7 fmol average peptides; per-tumor-antigen yield ~7e-11 to 7e-9 nmol/cell. Median 2–200× high. **⚠ shift to 1e-9; σ=1.5. (M)**
- `k_P1_deg_ID` (10/day, 1.0) — antigen DC half-life 1–2h (Delamarre 2005 PMID 15692050) → k 8–16. **✓ OK.**

### 6.3 T cells

- `k_Treg_pro` (0.05, 1.2) — Treg homeostatic proliferation 3–5%/day (Vukmanovic-Stejic 2006). **⚠ tighten σ to 0.7. (m)**
- `k_Treg_death` (0.1, 1.2) — half-life 5–10 d → k 0.07–0.14. **⚠ tighten σ to 0.5. (m)**
- `k_Treg_pro_tumor` (0.2, 0.7) — Plitas 2016 (PMID 27851914): 4× peripheral Ki67 in tumor Treg. **✓ OK.**
- `rho_Treg_max` (1e7 cell/mL, 1.0) — permissive cap (peak Treg 10⁵–10⁶/mL). **✓ OK.**
- `q_CD8_T_in`, `q_Treg_T_in` (both 5.8e-5 1/(cm³·min), 1.2) — identical medians ignore CCR4-driven Treg preference. **⚠ tighten σ to 0.7; consider Treg slightly higher. (m)**
- `k_CD8_pro`, `k_CD4_pro` (1.0, 0.5), `k_CD8_death`, `k_CD4_death` (0.1, 0.5) — post-activation doubling 8–12h; effector half-life 5–10 d. **✓ OK.**
- `k_Th_act` (10, 0.7), `k_Treg_act` (5, 1.0) — rapid (hours–day). **✓ OK.**
- `k_Treg_CD8_exh` (0.1, 1.0), `k_C_CD8_exh` (0.1, 1.0), `k_stiff_CD8_exh` (0.02, 1.0) — Sakaguchi 2008 / Kuczek 2019 (PMID 31412947). **✓ OK** (but `K_C_exh=1e6` saturates Hill, killing identifiability — see §3).
- `k_CD8_basal_exh` (0.05, 1.0), `k_CD8_exh_death` (0.15, 1.0) — Wherry 2011 (PMID 21739672): exhausted CD8 t½ 5–7 d → 0.1–0.14. **✓ OK** on death; verify basal.
- `N_IL2_CD4` (8.5, 0.5), `N_IL2_CD8` (11, 0.5) — Marchingo 2014 (PMID 25082697) division destiny; well-anchored. **✓ OK.**
- `n_CD8_clones` (120, 0.7), `n_CD4_clones` (100, 1.2) — PDAC TMB low, ~10–100 neoantigens (Balachandran 2017 PMID 29132146). Medians high; CD4 σ wider than needed. **⚠ shift both to 30, σ=1.0. (M)**
- `k_Th_to_Treg` (0.022, 0.7) — induced-Treg conversion 1–5%/day (Curotto de Lafaille 2009). **✓ OK.**
- `k_Th_pro_tumor` (0.15, 0.8), `rho_Th_max` (5e6 cell/mL, 0.7) — **✓ OK.**
- `R50_Treg` (0.31, 1.0) — Treg:CD8 ratio for half-suppression. **✓ OK.**
- `q_nCD8_LN_in`, `q_nCD8_P_out`, `k_nCD8_death`, `k_nTreg_death` — see §2.10. **(m)**
- `Q_nCD8_thym` (3e6 cell/day, 1.0) — Bains 2009 (PMID 19561105). **✓ OK.**
- `N_div_base` (2, 0.5), `N_div_costim` (3, 0.5) — Marchingo 2014. **✓ OK.**
- `n_T_expansion_burst` (800, 0.7) — burst 100–10000 (Buchholz 2013). **✓ OK.**
- `v_T_search_volume`, `k_T_activation_per_APC` — see §1.6, §1.7. **(C)**
- `p_T_kill_per_contact` (0.45, 0.7) — Halle 2016 (PMID 27692830): 30–50%. **✓ OK.**
- `K_C_exh` — see §3. **(M)**
- `f_nTreg` — see §1.1. **(C)**
- `CD8_50_Tcap` (1e5), `apCAF_50_Tcap` (5e6), `alpha_Th_cap` (5), `alpha_Treg_cap` (2) — GVAX-cascade caps; saturation by design. **✓ OK.**
- `CD28_CD8X_50` (200 mol/µm², 1.0) — Bromley 2001: synapse density 100–500. **✓ OK.**
- `p1_50` (2.6e-5 mol/µm², 1.5) — Purbhoo 2004 (PMID 15039774): 1–10 peptides/cell ≈ 1e-5 to 1e-4. **⚠ tighten σ to 1.0 (well-characterized). (m)**

### 6.4 Collagen / ECM mechanical

- `rho_collagen` (0.025 g/mL, 0.3) — fibrillar packing 0.02–0.03 g/mL (van der Rest 1991). **✓ OK.**
- `k_col_myCAF_sec` (5e-9 mg/cell/day, 1.2) — Bachem 1992 (PMID 1576110): activated PSC 1–10 ng/10⁵ cells/day = 1e-8 to 1e-7 mg/cell/day. Median 2–20× low. **⚠ shift to 5e-8; σ=1.0. (M)** — **BLOCKED until phi_cross_50 anchor is resolved: applied alone, this 10× bump worsens the phi_collagen attractor (already at 0.58 per `collagen_attractor_overshoot.md`).**
- `f_col_basal` (0.5, 0.5) — Ignotz 1986 (PMID 3711287): TGFβ stim 5–10× basal → basal fraction 0.1–0.2. Median high. **⚠ shift to 0.15. (M)**
- `k_col_deg` (0.007/day, 1.2) — collagen t½ ~100 d → k ~0.007 (Sivakumar 2008 PMID 18579773). **⚠ ANCHOR REVIEW (added 2026-04-27): Sivakumar 2008 measures osteoid (bone matrix), not pancreatic stromal collagen. The 100-day t½ target is what `phi_cross_50` was originally tuned against (per `parameters_PDAC.m:1840` comment), so re-anchoring `k_col_deg` would force a co-revisit of `phi_cross_50`. Pull PDAC stromal collagen turnover literature before changing either. (watch)**
- `k_MMP_M2` (1e-8 mL/(cell·day), 1.2) — recently added; no published anchor. σ appropriate. **✓ OK.**
- `phi_cross_50` (0.03, **σ widened 1.2 → 2.0 on 2026-04-27**) — LOX-crosslinking shield EC50; model-specific. **σ widening only; median unchanged.** Empirical context (`collagen_attractor_overshoot.md`): at the current median, phi_collagen sits at 0.58 (5× normal pancreas) and drives a 6× cDC1 density gap. **Median NOT shifted** because (a) per `feedback_no_prior_tuning`, shifting requires independent biological anchor, and (b) the 0.03 was a deliberate trade-off targeting Sivakumar's 60–100d in vivo collagen t½. Widened σ (CI95 ≈ [6e-4, 1.5]) gives SBI room to navigate via the cDC1 / maturation observables that are now in the loop. Real fix is a literature-anchored re-evaluation (queued; needs maple extraction targeting MMP-13 / TAM activity on mature crosslinked collagen — references like LaRue 2022 / Madsen 2013 cited in `parameters_PDAC.m` and CLAUDE.md should be **verified independently** before being relied on). (watch)
- `phi_col_50_Kmax` (2.0, 0.7) — by design weak barrier (see top of §6). **✓ OK; document intent. (m)**
- `phi_col_ref` (0.12, 0.5) — normal pancreas collagen 10–15% (Apte 2004). **✓ OK.**
- `d_pore_ref` (3.5 µm, 0.5), `n_pore` (4, 0.5) — Wolf 2013 (PMID 23509067): normal pore 2–5 µm. **✓ OK.**
- `d_crit_T` (2.3 µm), `d_crit_APC` (2.1 µm) — Lämmermann 2008 (PMID 18493255) shows DCs traverse smaller gaps via amoeboid + MMP cleavage. **Ordering likely correct after all; downgrade §3 flag. (m, document)**
- `E_ref` (1.0, 0.5), `n_stiff` (1.8, 0.5) — Levental 2007 (PMID 19913683): stiffness ~phi^1.5–2.0. **✓ OK.**
- `E_50_Texh`, `n_Texh`, `E_50_fib`, `n_fib` — see §2.7, §2.8.
- `K_stiff_niche` — see §2.9.

### 6.5 GVAX

- `k_GVAX_clearance` (0.1/day, 1.0) — irradiated allogeneic vaccine clearance, t½ 5–10 d. **✓ OK.**
- `k_GMCSF_prod` (1e-8 nM/cell/day, 1.0) — Jaffee 2001 (PMID 11208862): GVAX produces ~35 ng/10⁶ cells/day GM-CSF (MW 14 kDa) ≈ 2.5e-6 nmol/cell/day. **Possible unit mismatch** (nM/cell/day vs nmol/cell/day). **⚠ Verify units; may need shift to 2.5e-6. (M)**
- `k_GMCSF_deg` (5/day, 0.5) — half-life 3–4h → k ~4–6. **✓ OK.**
- `EC50_GMCSF` (2.94e-3 nM, 1.10) — receptor EC50 5–50 pM (Hercus 2009 PMID 19497856). **⚠ tighten σ to 0.7. (m)**
- `k_Cy_clear` (0.5/day, 0.7) — Cy t½ 6–8h (Bagley 1973) → k ~2–3. **⚠ shift median to 2.5; σ=0.5. (M)**
- `k_Cy_Treg_deplete` (0.5/day, 1.0) — Lutsiak 2005 (PMID 15894685): low-dose Cy depletes Treg ~50% in 4–7 d → k ~0.1–0.2. **⚠ shift median to 0.15. (M)**
- `Cy_50_Treg` (50 mg, 0.7) — Lutsiak 2005 used 100 mg/m². **✓ OK; verify mg total vs mg/m².**

### 6.6 Cytokines / chemokines (recently added or modified)

**IL-1 axis.**
- `k_IL1_sec` (1e-12 nmol/cell/day, 1.0) — cancer IL-1α ~10–100 pg/10⁶ cells/day (Apte 2006), 6e-13 to 6e-12 nmol/cell/day. **✓ OK.**
- `k_IL1_deg` (6.93/day, 0.5) — IL-1α t½ 2–6h. **✓ OK.**
- `IL1_50` (0.01 nM, 1.0) — IL-1R1 affinity 1–10 pM (Sims 2010). **✓ OK.**
- `n_IL1` — see §2.7. **(m)**
- `TGFb_50_IL1R1` (0.07 nM, 0.7) — assumed equal canonical TGFβ EC50; no direct measurement of IL-1R1 repression EC50. **Verify (no published anchor).**
- `Emax_IL1R1` — see §2.6. **(M)**

**CCL5 axis.**
- `k_CCL5_sec` (2e-12 nmol/cell/day, 0.7) — Mantovani 2004 ~1–10 ng/10⁶ cells/day (MW 8 kDa) = 1.25e-13 to 1.25e-12 nmol/cell/day. ~2× high. **⚠ shift to 1e-12; σ OK. (m)**
- `k_CCL5_deg` (1.5/day, 0.7) — chemokine t½ 6–12h. **✓ OK.**
- `k_CCL5_sec_iCAF` — see §6.1. **(m)**
- `CCL5_50_Treg` (0.15 nM, 0.7) — Schaerli 2002: CCR5 functional EC50 1–10 nM. Median low. **⚠ shift to 1.0 nM. (m)**
- `k_CCR5_Treg_rec` (1e-4 1/(cm³·min), 1.2) — trafficking rate. **✓ OK.**

**CXCL12 axis.**
- `k_CXCL12_sec_iCAF` — see §6.1.
- `k_CXCL12_sec_cancer` (5e-11 nmol·L/cell/day, 1.2) — half iCAF rate, plausible. **✓ OK.**
- `k_CXCL12_deg` (2/day, 0.5) — t½ 8–12h. **✓ OK.**
- `CXCL12_50_Texcl` (2.0 nM, 1.2) — CXCR4 EC50 1–10 nM (Crump 1997 PMID 9384584). **✓ OK.**

**IL-6 axis.**
- `k_IL6_sec_iCAF` — see §6.1. **(m)**
- `k_IL6_deg` (10/day, 0.5) — IL-6 t½ ~1h → k ~16. Mildly low. **⚠ shift to 16; σ OK. (m)**
- `IL6_50_*`, `f_IL6_iCAF` — see §2.5, §6.1.

**IFNγ arm.**
- `k_IFNg_deg` (7.68/day, 0.5) — t½ ~2h → k ~8. **✓ OK.**
- `k_IFNg_Tsec` (4e-12 nmol/cell/day, 0.7) — effector T IFNγ ~1–10 pg/cell/day (MW 17 kDa) = 6e-14 to 6e-13 nmol/cell/day. Median **6–60× high**. **⚠ shift to 2e-13. (M)**
- `k_IFNg_response` — see §1.3. **(C)**
- `r_PDL1_IFNg` (6.0, 0.7) — Spranger 2013 (PMID 23878289): 5–10× upregulation. **✓ OK.**

**VEGF.**
- `k_vas_Csec` (4e-3 pg/cell/day, 0.7) — Itakura 2000 (PMID 10963631): pancreatic cancer ~5–20 pg/10⁶ cells/h ≈ 0.12–0.48 pg/cell/day. Median **30–100× low**. **⚠ shift to 0.1; σ=0.7. (M)**
- `k_vas_Msec` (1.7e-5 pg/cell/day, 0.7) — same range issue if same scale applies. **⚠ shift to 0.05; σ=0.7. (M)**
- `k_vas_myCAF_sec` — see §6.1. **(m)**
- `k_vas_deg` (16.6/day, 0.5) — VEGF t½ 30–60 min → k 16–35. **✓ OK.**
- `VEGF_50` (2410 pg/mL, 0.7) — VEGFR2 EC50 1–10 ng/mL. **✓ OK.**
- `k_vas_growth` (0.15/day, 1.0), `k_vas_decay` (0.004/day, 1.0) — model abstractions. **✓ OK.**

---

## Summary table (flagged)

| Parameter | Current | Suggested | Severity |
|---|---|---|---|
| f_nTreg | lognormal 0.25, σ=0.7, [0.01,0.95] | beta or trunc-normal | Critical |
| f_active_TGFb | lognormal 0.1, σ=1.5 | beta(2,18) or σ≤0.7 + upper=1 | Critical |
| k_IFNg_response | 0.2, σ=1.5 | σ=1.0 | Critical |
| k_TGFb_sec_myCAF | 1e-9, σ=1.5 | 2e-10, σ=1.0 | Critical |
| k_cDC1_Ag_transfer | 1e-7, σ=1.5 | σ=1.0, upper=1e-6 | Critical |
| k_T_activation_per_APC | 1e-4, σ=1.0 | 1e-6, σ=1.5 (or doc calibration) | Critical |
| v_T_search_volume | 1e-5, σ=1.0 | 1e-6, σ=1.0 | Critical |
| K_apCAF_Treg | 1e7 cell, σ=1.2 | 1e8 cell, σ=1.0 | Critical |
| APC0_cDC1_T, APC0_cDC2_T | 5.4e5, 2.16e6 cell/mL | 5e4, 2e5 cell/mL | Moderate |
| k_APC_recruit0 | 100 cell/day | 1e4 cell/day (verify IC enforcement) | Moderate |
| y_Ag_per_cell | 1.7e-8 nmol/cell, σ=1.0 | 1e-9, σ=1.5 | Moderate |
| n_CD8_clones, n_CD4_clones | 120, 100 | 30 each, σ=1.0 | Moderate |
| k_col_myCAF_sec | 5e-9 mg/cell/day, σ=1.2 | 5e-8, σ=1.0 | Moderate |
| f_col_basal | 0.5, σ=0.5 | 0.15, σ=0.5 | Moderate |
| k_GMCSF_prod | 1e-8 nM/cell/day | verify units; possibly 2.5e-6 | Moderate |
| k_Cy_clear | 0.5/day | 2.5/day, σ=0.5 | Moderate |
| k_Cy_Treg_deplete | 0.5/day | 0.15/day | Moderate |
| Emax_IL1R1 | 9, σ=0.7 | 5, σ=0.7 | Moderate |
| k_IFNg_Tsec | 4e-12 nmol/cell/day | 2e-13 | Moderate |
| k_vas_Csec | 4e-3 pg/cell/day | 0.1 pg/cell/day | Moderate |
| k_vas_Msec | 1.7e-5 pg/cell/day | 0.05 pg/cell/day | Moderate |
| K_C_exh | 1e6 cell | 1e8 cell | Moderate |
| CAF subtype deaths | all 0.0165 | differentiate by subtype | Moderate |
| k_PSC_to_apCAF | 0.01, σ=1.0 | 0.05, σ=1.0 | Moderate |
| k_Mac_death | 0.02, σ=0.7 | 0.05, σ=0.7 | Moderate |
| k_PSC_to_myCAF | 0.4 | 0.2 | Minor |
| f_stroma_max | 3.0 | 5.0 | Minor |
| k_IL6_sec_iCAF | 4e-11 | 2e-11 | Minor |
| k_CCL5_sec_iCAF | 1e-10 | 3e-11 | Minor |
| k_vas_myCAF_sec | 5e-3 | 2e-3 | Minor |
| k_APC_mig | σ=1.2 | σ=0.7 | Minor |
| k_mAPC_death | σ=1.2 | σ=0.5 | Minor |
| k_APC_mature_ID | σ=0.85 | round to 0.7 | Minor |
| k_Treg_pro | σ=1.2 | σ=0.7 | Minor |
| k_Treg_death | σ=1.2 | σ=0.5 | Minor |
| q_CD8_T_in, q_Treg_T_in | σ=1.2 | σ=0.7 | Minor |
| p1_50 | σ=1.5 | σ=1.0 | Minor |
| k_CCL5_sec | 2e-12 | 1e-12 | Minor |
| CCL5_50_Treg | 0.15 nM | 1.0 nM | Minor |
| k_IL6_deg | 10/day | 16/day | Minor |
| EC50_GMCSF | σ=1.10 | σ=0.7 | Minor |
| vol_*CAFcell | all 1600 µm³ | differentiate (1500/2000/3500/2500) | Minor |
| k_P1_up_apCAF | σ=1.0 | σ=1.2 (cross-cell extrapolation) | Minor |
| IL2_50_Treg | 0.32 nM | 0.01 nM, σ=0.5 | Minor |
| q_nCD8_LN_in, q_nCD8_P_out | σ=1.2 | σ=0.5 | Minor |
| n_IL1, n_Texh, n_fib | σ=0.7 | σ=0.5, upper=4 | Minor |
| K_initial | check if still used | drop if vestigial | Minor |
| d_crit_APC vs d_crit_T | 2.1 < 2.3 | likely correct (DC amoeboid) — document | Minor |
| phi_col_50_Kmax | 2.0 dim, σ=0.7 | OK by design — document intent | Minor |
| dist_param1 (k_TGFb_sec_myCAF) | −20.72 (rounded) | recompute exactly | Nit |
| TGFb_50_IL1R1 | 0.07 nM | verify (no published anchor) | verify |
| k_APC_recruit_ID | 1e5 cell/day | verify (no published anchor) | verify |

---

## Verification still needed (before applying critical fixes)

1. ~~Hill form using `phi_col_50_Kmax` in `fibroblast_module_enhanced.m`.~~ **Resolved** — see top of §6 (form is `phi/(phi+phi_col_50_Kmax)`, design intent is intentionally weak barrier).
2. `f_nTreg` compartment (peripheral vs tumor-infiltrating CD4 fraction).
3. `K_initial` — still consumed by simulator under `evolve_to_diagnosis`?
4. ~~`k_apCAF_Treg` direction (Treg-per-apCAF rate, not apCAF depletion).~~ **Resolved** — verified Treg-induction-per-apCAF (module line 400).
5. `k_GMCSF_prod` — units (nM/cell/day vs nmol/cell/day) and whether `Cy_50_Treg` is mg total dose or mg/m².
6. `k_APC_recruit0` — does the IC enforcement bypass the rate, making the 100 cell/day median irrelevant?
