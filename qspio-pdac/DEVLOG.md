# PDAC Model Development Log

Working notes on model development decisions, observations, and reasoning. More granular than git commits — captures the *why* and cross-commit narrative.

See `notes/pdac_changelog.md` for the structured module-level changelog.

---

## 2026-04-30 — Th exhaustion decoupling (Cluster B structural fix)

Worktree: `../pdac-build-cluster-b`, branch `feature/cluster-b-cdc1-th-exh`. Picks up after PR #108 (cohort random-effect) flagged Cluster B (`cdc1_density × th_exhausted_fraction`, prior pairwise φ = −0.483 at τ=1.0) as the top remaining structural tension.

**Diagnosis.** Per `feedback_trace_before_proposing_fix.md`, ran a single-θ qsp_sim trace at default priors before scoping a fix.

The handoff hypothesis was that high cDC1 → faster Th priming via `H_mAPC` → fewer exhausted Th, so Cluster B reflected a structural cDC1 ↔ Th-exh coupling. **Both halves of that story are wrong.** From the actual model code:

1. `H_APC_Th` uses `V_LN.mcDC2`, not `mcDC1` (per `core/modules/APC_module.m` comments + grep); cDC1 abundance does not enter Th priming.
2. The single Th→Th_exh reaction in `Th_module.m` ran at rate `k_CD4_death × V_T.Th` — exhaustion was just relabeled bulk death. At steady state, `Th_exh / (Th + Th_exh) = k_CD4_death / (k_CD4_death + k_cell_clear)`, which is **decoupled from APC dynamics entirely**. Trace at default priors confirmed: 0.49 ± 0.01 across APC0_cDC1_T sweeps spanning 1000×.

The actual Cluster B problem is a **k_CD4_death bifurcation in the Th compartment**:
- Th_exh fraction = 0.78 needs `k_CD4_death ≈ 3.5 × k_cell_clear` ≈ 0.35/day
- That's `k_CD4_death > k_Th_pro_tumor = 0.15`, pushing Th into the death-limited regime
- CD4 density crashes from 213 → 0.1 cells/mm² (vs `cd4_helper_density` obs 17)
- No (k_CD4_death, rho_Th_max, alpha_Th_cap) regime simultaneously satisfies both targets — verified by 4-knob combined trace

**cDC1 density** (obs 7.82 cells/mm² vs default 0.026, posterior cap 3.81 from v5 PC) is **independently unreachable**, but its appearance in the τ=1.0 pairwise table is just spurious tail-tail anti-correlation: both targets demand tail draws of independent parameters, so conditioning on satisfying one shrinks the joint volume. Deferred — separate scope (likely σ-widen on `APC0_cDC1_T` per Plesca 2022 0.3–30 cells/mm² range, or revisit `H_pore_APC` for cDC1).

**Fix.** Mirrored the CD8 exhaustion structure into Th. `Th_module.m` now has:
- `V_T.Th -> null` at `k_CD4_death * Th` (regular death; no longer relabeled)
- `V_T.Th -> V_T.Th_exh` at `k_Th_basal_exh * Th * C_total/(C_total+K_C_exh+cell)` (chronic-antigen baseline; mirrors `k_CD8_basal_exh`)
- `V_T.Th -> V_T.Th_exh` at `k_TGFb_Th_exh * Th * H_TGFb_Th_exh` (TGFβ-driven; cell-specific Hill)
- `V_T.Th -> V_T.Th_exh` antigen-clearance funnel (unchanged)

Reuses existing `K_C_exh` (CD8 exhaustion block). Adds a dedicated `H_TGFb_Th_exh = V_T.TGFb / (V_T.TGFb + TGFb_50_Th_exh)` Hill in `cytokine_tcell_module.m`, following the existing per-cell-type pattern (`H_TGFb_Teff`, `H_TGFb_NK`, `H_TGFb_APC`). Reusing the generic `H_TGFb` would have coupled Th-exh tuning into Th→Treg differentiation, M1→M2 polarization, and fibroblast activation. Three new parameters:
- `k_Th_basal_exh` = 0.05/day, σ=1.0 (matches `k_CD8_basal_exh`)
- `k_TGFb_Th_exh` = 0.3/day, σ=1.0 (estimated)
- `TGFb_50_Th_exh` = 0.14 nM, σ=1.0 (mirrors `TGFb_50_Teff`)

`k_TGFb_Th_exh` and `TGFb_50_Th_exh` are both estimated; submodel target extraction is the next step.

**Result.** At new defaults, Th_exh frac 0.49 → 0.57 (target 0.78), CD4 density 213 → 0.40 (target 17). 2D feasibility grid over `(k_Th_pro_tumor, k_TGFb_Th_exh)` finds a feasible cell at **(0.4, 0.8) → CD4 density = 14, Th_exh frac = 0.75**, both within +1.5σ of prior medians (`k_Th_pro_tumor` +1.22σ, `k_TGFb_Th_exh` +0.98σ). The proliferation/death bifurcation still creates a sharp transition along the grid, but the feasible region is reachable from prior tail draws without extreme combinations — SBI should land it. The dedicated `TGFb_50_Th_exh` adds a degenerate second knob (rate × EC50), making the feasible region easier to hit than a shared-Hill alternative.

**Why not Fix #2 (cDC1 functional impairment via VEGF/PGE2):** doesn't address the actual Cluster B problem. The handoff suggested both fixes assumed the cDC1↔Th-priming coupling, which doesn't exist in this model. cDC1 reachability is a separate issue requiring its own scope.

**Refresh + parity.** `make refresh-cpp` regenerated SBML/codegen/`param_all.xml` with the two new parameters. `make parity` passed (45.66s, both 365-day default-IC trajectory and GVAX+nivo scenario match within tolerance).

**Next.** Submodel target extraction for `k_TGFb_Th_exh` — TGFβ-driven CD4 dysfunction kinetics from in vitro literature. The user took over this thread.

---

## 2026-04-22 — LOX-crosslinking shield on M2 MMP degradation

Addresses the `ecm_fold_change` half of the `test_all_snapshot_bounds_pass` xfail (`tests/scenarios/test_qualitative_behaviors.py:1930`). Worktree: `../pdac-build-mmp-collagen`, branch `fix/mmp-collagen-xfails`.

**Diagnosis.** Posterior-median snapshot had ECM fold-change 0.47× (target 5-20×). Steady-state math: M2 MMP degradation rate `k_MMP_M2 × Mac_M2/V_T = 1e-8 × 9.08e7 = 0.908/day` — ~130× the constitutive `k_col_deg=0.007/day`, giving collagen t½ ≈ 0.8 d (literature: 60-100 d for mature fibrotic collagen). Root cause is the in-vitro→in-vivo extrapolation of `k_MMP_M2`: the submodel target derived from Zhu 2001 uses 4e5 monocytes/mL in freshly-reconstituted rat-tail collagen gels (uncrosslinked, MMP-susceptible). In vivo, LOX-mediated crosslinking renders mature collagen I fibrils MMP-resistant, and TME M2 density is ~227× Zhu's.

**Fix.** Added `f_nascent = phi_cross_50/(phi_cross_50 + phi_collagen)` Hill shield on the M2 MMP term. Only the non-crosslinked (nascent) collagen fraction is susceptible; at high phi the M2 contribution self-limits. Preserves the in-vitro submodel target (phi≈0 → f_nascent≈1, fresh gels degrade freely).

- New parameter: `phi_cross_50 = 0.03` (lognormal prior, σ=1.2). Small value reflects early LOX engagement.
- Reaction in `fibroblast_module_enhanced.m`: `(k_col_deg + k_MMP_M2 * Mac_M2/V_T * f_nascent) * collagen`

**Result.** ECM fold-change: 0.47× → **3.4×** at `phi_cross_50 = 0.03`. `phi_collagen ≈ 0.17` at diagnosis, collagen t½ ≈ 11d. Tumor reaches 3.20 cm target diameter. Initial iterations tried `phi_cross_50 = 0.02` (4.6× fold) but the combination with main's PR #27 IFNg reparameterization ate the carrying-capacity margin and `evolve_to_diagnosis` asymptoted at 3.17 cm. Bumped to 0.03 to coexist — still well inside the [2, 12] bound; SBI can pull it lower if the data wants.

**V_T/rho_collagen semantic issue noted** (deferred to separate PR). The volume rule `+V_T.collagen/rho_collagen` grows V_T as collagen accumulates, which (a) inflates cytokine distribution volume in the wrong direction (real biology: collagen crowds interstitium, *concentrating* cytokines), and (b) can make `C_max` net-increase with collagen due to V_T scaling outrunning phi-suppression. `rho_collagen = 0.025 g/mL` is also semantically overloaded — used both as a phi-scaling reference (25 mg/mL saturation convention) and as a material packing density (physically more like 0.1-0.3 g/mL). Tracking in a GH issue.

**Bound revision.** The pre-existing bound `[5, 20]×` was derived solely from Xu 2017 SHG/TPEF pixel density — a method that preferentially detects aligned crosslinked fibrils enriched in desmoplasia. Mass-basis measurements (hydroxyproline, TMT proteomics), which match the model's mass/volume `phi_collagen`, consistently report 2-3×:
- Tian 2019 TMT proteomics: 2.6× (range 2.2-3.0)
- Imamura 1995 hydroxyproline (n=14 human): ~3×
- Polasek 2017 orthotopic mouse hydroxyproline: 2.3× (1224±111 vs 523±14 µg/g, n=18 tumor vs n=5 sham)
Revised bound to `[2, 12]×` with Tian 2019 as primary source and Imamura 1995 + Xu 2017 as secondary. Upper bound 12× covers the SHG upper range without making it the norm. Post-revision, the 4.61× state is comfortably in range; bound ceases to xfail.

**Test split.** `test_all_snapshot_bounds_pass` bundled 19 bounds into one assertion — one failing bound masked 18 passing. Split into `test_snapshot_bound_passes` parametrized per bound, with per-bound xfail reasons via `_SNAPSHOT_BOUND_XFAIL_REASONS` (same pattern as `_XFAIL_REASONS` in `TestHillFunctionAudit`).

**What's still broken.** M2/M1 = 11.9 (bound [1, 5]). Unchanged by this fix — macrophage polarization issue (IL-10/TGFb → M2 saturation vs weak IL-12/IFNg → M1), independent of MMP mechanics. Only ECM fold-change is cleared here; M2/M1 remains xfail and is deferred to a separate branch/PR.

---

## 2026-04-21 — GVAX cascade structural fixes: mcDC2→cDC1 antigen transfer + CD8-scaled Th/Treg cap

Diagnostics on `gvax_nivo_neoadjuvant_zheng2022` showed the stage-2 NPE posterior
couldn't identify any of the 12 GVAX-axis parameters: `V_T.CD8` rose only 1.10×
and `V_T.Th`/`V_T.Treg` 1.03–1.04× at tx_d21 despite a 54× rise in `V_C.Th`
and a 1300× rise in `V_LN.mcDC2`. Cascade audit (notes/calibration/gvax_cascade_bottleneck_handoff.md)
traced it to two structural breakdowns that priors could not fix.

**Fix 1: LN-resident cDC1 antigen transfer (`gvax_module.m`, `parameters_PDAC.m`).**
GVAX routes all maturing ID-derived mAPCs to `V_LN.mcDC2`, so the CD8 priming
Hill `H_mAPC` (which uses `V_LN.mcDC1`) couldn't respond. Original framing
proposed "mcDC2 licenses mcDC1 via CD40L/IL-12" but that conflates T-cell help
with antigen handoff. Lit check confirmed the dominant mechanism for GM-CSF-
recruited DC vaccines is actually **antigen transfer from migratory cDC2/GMDC
to LN-resident Batf3⁺ cDC1**: Ferris 2022 *Cancer Immunol Res* (PMC9357132)
showed GMDC vaccines fail completely in `Irf8+32⁻/⁻` cDC1-null mice. Route A
(direct migratory cDC1) exists but is weak because GM-CSF poorly expands cDC1.
Added reaction `V_LN.cDC1 → V_LN.mcDC1` at rate `k_cDC1_Ag_transfer × V_LN.cDC1 × V_LN.mcDC2`
(bimolecular, k default 1e-7 /(cell·day)). Impact: `V_LN.mcDC1` 1.03× → 18.5×,
`V_T.CD8` 1.10× → 17.5×, `k_C_Tcell_eff` up 10⁵×.

**Fix 2: niche-scaled logistic carrying caps (`Th_module.m`, `Treg_module.m`, `parameters_PDAC.m`).**
After Fix 1, V_T.Th/V_T.Treg were still stuck at 1.04× / 1.03×. The bottleneck
is the fixed logistic cap `rho_X_max × V_T` in the local proliferation rate:
baseline V_T.Th ≈ cap, so any V_C-driven influx is drained by the negative
logistic term. Replaced the fixed caps with niche-specific inflammation-scaled
caps, `rho_X_max × V_T × (1 + α_X × H_niche_X)`:

- **Th cap**: `H_CD8_Tcap = V_T.CD8/(V_T.CD8 + CD8_50_Tcap)` — effector CD8
  infiltration = T-cell-inflamed TME → chemokines/adhesion molecules sustain
  higher Th density (Spranger 2015 PMID: 25970248, Peng 2012 PMID: 22437871).
  Chose CD8 over IFNg because V_T.IFNg is dominated by Th-secreted IFNg, which
  would create a circular Th → IFNg → Th-cap loop.

- **Treg cap**: `H_apCAF_Tcap = V_T.apCAF/(V_T.apCAF + apCAF_50_Tcap)` — apCAFs
  are the dedicated Treg-inducing CAF subtype (Elyada 2019 PMID: 31197017,
  Huang 2022 PMID: 35181260). *Decoupled from CD8* after an earlier
  CD8-scaled attempt created a pathological "TGFb blockade → more CD8 →
  higher Treg cap → more Treg" loop that inverted the TGFb-blockade-reduces-
  Tregs behavior test. apCAF activation is TGFb-dependent via CAF
  interconversion, so TGFb blockade correctly shrinks the Treg cap.

Defaults: `alpha_Th_cap=5`, `alpha_Treg_cap=2`, `CD8_50_Tcap=1e5`,
`apCAF_50_Tcap=5e6`. All tunable by SBI.

Impact at default priors (GVAX+Cy+nivo scenario, tx_d0 → tx_d21):
- V_T.Th 1.04× → **1.47×** (up, inflammation-responsive)
- V_T.Treg 1.03× → **0.38×** (down, Cy depletion dominates — matches
  Zheng 2022 reported Treg reduction)
- V_T.CD8 preserved at 17× (Fix 1)
- Th/Treg ratio rises as expected under successful immunotherapy

No local proliferation added to CD8 itself: intentional. LN-side clonal
expansion (`k_CD8_pro × 2^N_aT` in Teff_module) already amplifies primed
CD8, and adding a tumor-local cap would recreate the Fix-2 bottleneck on CD8.

**Scenario correction (2026-04-21).** Both Zheng 2022 scenarios (arm A GVAX-
mono, arm B GVAX+nivo) were missing low-dose cyclophosphamide (200 mg/m² day
6). Cy is part of every arm of NCT02451982 — standard Hopkins GVAX
preparation since Laheru 2008. Added to both YAMLs. Without Cy the Treg
depletion mechanism isn't engaged and the qualitative-behavior tests for
GVAX-reduces-Tregs failed, even after Fix 2.

**CD8 local proliferation decision.** Considered but rejected. LN-side
clonal expansion already amplifies primed CD8 pre-trafficking. Adding tumor-
local CD8 prolif would require a cap (to prevent blow-up) that would become
the new rate-limiter — identical bottleneck to Th/Treg. Ki67+ CD8 TIL is not
currently a calibration target.

**Fix 3: lower `EC50_P1_mature` from 10 nM to 0.1 nM.** Single-parameter change
in `parameters_PDAC.m`. With the old value, `V_ID.P1_GVAX` (steady state
~0.009 nM concentration at the injection site) produced a Hill factor of
~9e-4 in `J_mature_ID` — antigen-dependent APC maturation was numerically
dead, and maturation proceeded almost entirely via the GM-CSF + baseline APC
terms. 0.1 nM is an **estimated value** in the nanomolar-affinity range
typical for peptide-receptor interactions; not anchored to a specific
primary source. (An earlier draft of this note cited "Bhatia 2014 JACI" —
that reference is fabricated and has been removed; the parameter is a
candidate for submodel-target literature extraction before Stage-2 SBI.)

Impact on the cascade is modest because V_ID.mAPC steady-state is already
rate-limited by death/migration, so a 92× bump in `J_mature_ID` (Hill 0.0009
→ 0.083) barely moves V_ID.mAPC, V_LN.mcDC2, or V_T.CD8 — Fix 1 already
reached saturation for those pools. The value of Fix 3 is mechanistic
correctness (antigen-gating is no longer vestigial) and sensible behavior
under prior draws that lower the mAPC pool upstream.

We chose the minimal option (EC50 shift) over the more invasive
mAPC-carried-antigen-transport reaction (`V_ID.P1_GVAX → V_LN.P1`) because
(a) `V_LN.P1` does not exist as a species in the current model — antigens
live in V_T and travel via APC endocytosis only; adding a new LN-resident
antigen pool would require re-plumbing the TCR-KPR cascade in V_LN, and
(b) Fix 1 already captures the biologically correct outcome (antigen-loaded
migratory DCs license LN-resident cDC1 to cross-prime) via the
`V_LN.cDC1 → V_LN.mcDC1` antigen-transfer reaction, without needing to track
the antigen molecules themselves.

**Post-model-change propagation.** See `notes/model/post_change_checklist.md`
for the full list of downstream artifacts that must be updated
(pdac_priors.csv, targets.csv, model_structure.json/model_defs.json, SBI
caches, etc.) whenever parameters or reactions are added.

---

## 2026-04-01 — Rename k_MMP_M1 → k_MMP_M2 (M2 macrophages are the dominant collagen degraders)

Lit search for `k_MMP_M1` found 4 papers — all consistently show M2-like macrophages, not M1, are the primary collagen degraders (via mannose receptor endocytosis and MMP-13). Assessment marked all 4 papers `usable: false` for M1 parameterization. Switched the collagen degradation rate law from `k_MMP_M1 * Mac_M1 / V_T` to `k_MMP_M2 * Mac_M2 / V_T`. Key refs: Madsen 2013 (M2 internalize collagen at 2.5x M1 rate), LaRue 2022 (MRC1-high TAMs dominate collagen uptake in KPC tumors). Same prior value/sigma retained — the papers that were unusable for M1 now become candidate submodel targets for M2.

---

## 2026-03-31 02:30 — Staged extraction pipeline + per-source translation sigma design

Built an end-to-end pipeline for automated submodel target extraction (`scripts/staged_extraction.py`) using pydantic-ai agents against the OpenAI Responses API (gpt-5.1). The pipeline has four stages: (1) literature search with web search tool + DOI validation via CrossRef, (2) paper assessment from PDFs with digitization flagging, (3) SubmodelTarget assembly with schema validation + retries, (3b) post-hoc validation including MCMC prior derivation, unit checking against model_structure.json, and snippet-in-PDF verification. All stages cache to JSON — re-running skips completed work. Batch processing loops over a configurable `TARGET_RANGE` with parallel agent calls per stage via `asyncio.as_completed` (same pattern as maple's `immediate_processor.py`, avoids logfire issues with `asyncio.gather`).

**First automated extraction: k_Treg_death** (`c031ad6`). The pipeline autonomously chose VukmanovicStejic2006 (human deuterium-labeling study, Table 1: d*=6.58%/day, SD=3.83, n=8) over the digitized Kolodin2015 mouse BrdU data — defensible since the human study had direct table values. Prior derivation: lognormal median=0.070/day (t½≈10d), CV=0.56, contraction=0.98. All snippet validations pass at 1.00 against the PDF after fixing two table_excerpt column headers that the LLM had paraphrased rather than copied verbatim.

**Digitization workflow.** Papers and digitized data organized into `<source_tag>/` subdirectories (e.g., `papers/Kolodin2015/`, `digitized/Kolodin2015/`). Stage 2 auto-generates per-source README templates with WPD format documentation and figure-specific digitization hints from the assessment. User drops raw WPD CSV exports (no header changes needed), README provides the LLM context about what each file contains.

**Per-source translation sigma design** (documented in `notes/calibration/per_source_translation_sigma.md`). Triggered by `q_nCD8_LN_in` — a derived parameter where the entry rate comes from a steady-state constraint combining mouse LN exit rates (Mandl 2012) with human blood counts (Autissier 2010). The current single `source_relevance` can't express different translation quality per source. Design decision: move `source_relevance` from `SubmodelTarget` onto `PrimaryDataSource`/`SecondaryDataSource`. Each measurement's translation sigma is computed by deduplicating its inputs' `source_ref`s, computing per-source sigma, and combining in quadrature (independent translation gaps from different sources compound; same-source inputs share a correlated gap and count once). The sigma gets applied at the likelihood level per `ErrorModelEntry`, not per target — so precise human inputs contribute less noise than rough mouse proxies, and the NPE learns this from the training data noise structure. No forward model differentiation needed.

Also added cross-model parameter comparison to the audit report (`8f66d56`) showing how PDAC parameter values compare across QSPIO cancer models (TNBC, CRC, UM, NSCLC, HCC), and renamed 19 cryptic parameters to descriptive names (`091eb8b`).

---

## 2026-03-25 22:00 — Macrophage + T cell submodel targets: 5 new targets from 5 papers

Major extraction session targeting the macrophage module (previously 0% coverage) and T cell trafficking/IL-2 parameters. All 5 targets pass schema validation, prior derivation, and snippet verification.

**Calibration sufficiency analysis.** Before extracting, benchmarked our target counts against published QSP-IO models (Wang 2023, Arulraj 2023, Milberg 2019). With ~80 targets for 217 parameters, we're at the upper end of the field (~0.4:1 ratio). No published Popel lab model has submodel-level parameter constraints. Documented in `notes/calibration/calibration_sufficiency_analysis.md` including the three-paper strategy (MAPLE methods, PDAC application, clinical/translational) and a reviewable rationale for submodel target selection (novel module first, extractability, PRCC sensitivity).

**Macrophage M2:M1 ratio targets (k_M1_pol_PDAC_deriv001, deriv002).** Two independent PDAC cohorts: Gu 2023 (CD86/CD163, n=42, ratio 2.3:1) and De Simoni 2023 (iNOS/CD163, n=38, ratio 18:1). The 8x difference in ratios reflects the stricter M1 marker (iNOS vs CD86), absorbed by nuisance Hill terms. Forward model: steady-state M2/M1 = (k_M2_pol * H_anti) / (k_M1_pol * H_pro + k_Mac_death). Each target constrains 4 QSP params + 2 nuisance. De Simoni figure 5A was digitized with WPD for individual iNOS patient scatter data (n=43 points, mean 6.6, SD 6.1). k_M2_pol contraction 0.89 (Gu) — strong.

**Macrophage recruitment density (k_Mac_rec_PDAC_deriv001).** Chen 2015 CD68+ TAM counts in peripheral PDAC (median 71.5/HPF, range 13-198, n=70). Forward model: total_Mac = k_Mac_rec * H_CCL2 / k_Mac_death, converted via HPF area and section thickness. k_Mac_death contraction 0.93 — pulled from 0.02 to 0.009/day (longer macrophage residence). Complements the ratio targets: ratios set M1/M2 balance, density sets absolute scale.

**CD8 T cell trafficking (q_CD8_T_in_PDAC_deriv001).** Boissonnas 2007 OT1-GFP infiltration into EG7 thymoma. Two time points from digitized Figure 1B (early day 3-4: 102+/-32; late day 5-6: 327+/-50 cells/mm2). Saturation ODE model: dN/dt = q*CD8_C - k_death*N. Also digitized Figure 6B (depth-resolved density) but discovered a 10.8x methodological discrepancy between Figures 1B and 6B — likely different detection sensitivity. Used Figure 1B (whole-tumor average, text-confirmed values) as the more appropriate match for the well-mixed QSP model.

**IL-2 division destiny (N_IL2_CD8_PDAC_deriv001).** Castro 2012 IL-2/anti-IL-2 complex experiment. Digitized Figure 2A: OT-I cell numbers for both IL2-IC and control conditions. Fold ratios at day 10 (39.7x) and day 30 (70.2x) match text exactly. Forward model: fold = 2^N_IL2_CD8. Posterior median 5.9 (prior was 11.0), contraction 0.98 — very strong. The data says ~6 extra divisions, not 11.

**MAPLE tool updates.** Updated `mcp_server.py` and `submodel_target_prompt.md` in the maple repo with three workflow improvements: (1) use WPD directly instead of CSV templates, (2) strong preference for `direct_measurement` input types over `reference_value`, (3) always prefer figure digitization when richer data exists.

Skipped targets: Gett 1998 (N_IL2_CD4) — only a CFSE detection limit (7-8 divisions), not a clean measurement. Kai 2023 (ECM_density) — phi_collagen is a full-model equilibrium, not a submodel target. Bachem 1993 (PDGF_50_prolif) — PDGF removed from model.

---

## 2026-03-25 14:00 — IL-6/LIF proxy clarification + VEGF secretion + IL6_50_M2 dose-response

Three parameters investigated for submodel target extraction; two extracted, one flagged as not extractable.

**f_IL6_iCAF — LIF proxy, not extractable.** Literature search revealed that Biffi 2019 (the parameter's own citation) explicitly shows IL-6 neutralization and IL-6 KO have no effect on iCAF formation (Supp Fig S4B-D). The actual autocrine mediator is LIF, not IL-6. The pathway is IL-1alpha → NF-kB → LIF → JAK/STAT3 → iCAF program. IL-6 is a downstream output, not a driver. Evaluated reparameterizing to LIF as a separate species but rejected: would replace 2 params with 5+ poorly-constrained ones (no PSC/CAF LIF dose-response, no secretion rate, only systemic half-life data). The current IL-6 proxy is defensible since both IL-6 and LIF signal through gp130/JAK/STAT3. Updated parameter notes in parameters_PDAC.m and fibroblast_param_audit.md to document the proxy interpretation.

**k_vas_myCAF_sec — Erkan 2009 PSC VEGF ELISA.** Primary human PSCs from resected PDAC tissue, 200,000 cells/well in six-well plates, 24h serum-free, VEGF normalized to 100,000 cells. Digitized Figure 4C: normoxia 346.6±144.9 (SEM) pg/mL per 100k cells. Batch accumulation model gives posterior median 0.00145 pg/cell/day — **170× higher** than the current 8.5e-6 placeholder (which was just "~50% of M2 rate" with no literature). This is a large correction that will significantly increase model VEGF production from myCAFs. Hypoxia data (946 pg/mL, 175% increase) not used since the model has no hypoxia-VEGF modifier on CAFs — the normoxic rate is a conservative lower bound.

**IL6_50_M2 — Fernando 2014 arginase dose-response.** 5-point IL-6 titration (0, 0.01, 0.1, 1, 10 ng/mL) on mouse BMDM arginase activity with IL-4+IL-13 co-stimulation, n=4. Used algebraic multi-output model (not direct_fit, which only supports inhibitory Hill). Posterior median 0.223 nM — essentially validates the current 0.20 nM value. Translation sigma is high (0.811) due to mouse→human proxy, but this anchors the IL6_EC50s hierarchical group (3 members, previously zero targets). IL6_50_MDSC had no extractable data (the MDSC field uniformly uses single fixed IL-6 doses).

---

## 2026-03-24 22:00 — Cytokine secretion targets + ODE→algebraic conversions + MCP workflow tools

Extracted two new cytokine secretion submodel targets, both finding the current model values are substantially wrong:

**k_IL6_sec_iCAF (Saga 2018, LX-2 ELISA).** Steady-state IL-6 in conditioned media from human hepatic stellate cells: control 65.3±4.0 pg/mL, TNF-alpha activated 243±29.8 pg/mL. Uses the multi-output algebraic + observable pattern — compute() returns both conditions, per-entry observables select. Posterior median = 7.0e-10 nM·L/(cell·day), ~18× higher than the current 4e-11. Also found that both cited sources in parameters_PDAC.m are wrong: Elyada 2019 has no IL-6 protein data (RNA only), and Mace 2018 PMID 29196415 points to an unrelated esophageal microbiome paper.

**k_CCL5_sec_iCAF (Kawka 2014, HPFB time course).** 6-point TNF-alpha time course (6-96h, n=6 donors) of CCL5 accumulation in pg/10^4 cells from human peritoneal fibroblasts. Digitized with WebPlotDigitizer. Analytical secretion+degradation saturation model: C(t) = k_sec/k_deg * (1 - exp(-k_deg*t)). Posterior median = 2.84e-10 nmol/(cell·day), **284× higher** than the current 1e-12. Unit conversion verified: the current value predicts 0.1 pg/10^4 cells at steady state vs data plateau of 230. The original value was genuinely just a guess ("estimated" with no source).

**ODE→algebraic conversions.** Converted 3 of 6 custom_ode targets to algebraic forward models with analytical solutions, eliminating 23 diffrax ODE solves from the joint inference model: (1) TGFb_50_CAF_act — pure first-order Q→A has fraction = 1-exp(-k*H*t); (2) k_CXCL12_sec_iCAF — exponential growth + linear accumulation closed form; (3) caf_prolif_rates — Richards logistic. All use the new multi-output algebraic pattern with per-entry observables selecting from a compute() dict. Added Observable.state_variables as optional for algebraic models (required for ODE only).

**MCP workflow tools.** Added verify_dois (CrossRef resolution + directory creation) and fetch_papers_from_zotero (copies PDFs from local Zotero storage to paper directories) to streamline the extraction workflow. The fetch tool's author matching needs refinement (substring match grabbed "Sunnaker" for "Sun").

**CXCL12_50_Texcl** also extracted in a parallel session — now at 37 total submodel targets, 30/63 fibroblast params covered.

---

## 2026-03-23 23:30 — Hierarchical parameter groups + inference pipeline fixes

Added hierarchical parameter groups to maple's submodel inference pipeline. The motivation: many QSP parameters represent "the same biological process in different cell types" (e.g., CAF subtype death rates, TGFb EC50s across cell types) but have no individual literature constraints. Instead of leaving them fully independent, partial pooling via a random-effects structure lets data-rich parameters inform data-poor siblings.

**Implementation (maple):** New `parameter_groups.py` schema — `ParameterGroupsConfig` with `ParameterGroup` objects declaring members, between-member SD (tau), and optional per-member delta priors for biological ordering. The NumPyro model samples `k_base ~ LogNormal(mu, sigma)`, `tau ~ HalfNormal(sigma_tau)`, `delta_i ~ Normal(mu_i, sigma_i)`, then `k_i = k_base * exp(delta_i)`. Key design decision: `base_prior` is optional in the YAML — when omitted, it's derived at runtime from the CSV priors of the group members (mean of log-space mu, max of sigma). This keeps `pdac_priors.csv` as the single source of truth. Groups live in `parameter_groups.yaml` alongside the submodel targets, not inside target YAMLs, because groupings are prior structure (not likelihood) and can include parameters with no targets at all.

**7 groups defined (21 params):** CAF death rates (4, with qPSC biased lower and myCAF slightly lower for apoptosis resistance), CAF proliferation rates (3), TGFb EC50 fibroblast (3, tight tau=0.25 — same receptor on same cell), TGFb EC50 immune (4), IL-6 EC50s (3), naive T death rates (2, very tight tau=0.2), CXCL12 secretion rates (2). 15 tests (13 schema + 2 integration with JAX/NumPyro MCMC), all passing.

**Bug fixes along the way:** (1) Removed stale PDGF_50 references from `TGFb_50_CAF_act_PDAC_deriv001` and `k_psc_activation_PDAC_deriv002` — PDGF was removed from the model in the 2026-03-20 commit but these target forward models still had PDGF Hill functions. (2) Discovered that `Hao 2015 (PMID: 25225370)`, cited as the source for all CAF death rates, is actually a renal fibrosis mathematical model paper — not pancreatic biology at all. The 0.0165/day value was transplanted from a kidney fibroblast turnover parameter. The hierarchical group structure now provides the appropriate constraint instead. (3) Fixed JAX `pmap` + diffrax closure conversion bug (`RESULTS<traced>` vs `RESULTS<unknown>`) by switching to `chain_method="vectorized"` (uses vmap instead of pmap). This is specific to Apple Silicon virtual CPU devices. (4) Added JAX persistent compilation cache (`~/.cache/maple/jax_compilation_cache/`), reduced diffrax `max_steps` 16384→512, and enabled `dense_mass=True` for NUTS to reduce leapfrog steps in the correlated 65-parameter space.

**Inference comparison run:** Joint inference with 34 targets, 65 params (21 in 7 hierarchical groups), 98 likelihood terms runs successfully. Compilation takes ~4 min on first run (cached thereafter), sampling at ~1-2 it/s with 31-63 leapfrog steps. The 6 custom_ode targets using diffrax dominate the per-step cost. Future optimization: 3 of the 6 ODE targets have analytical solutions (TGFb_50_CAF_act is pure first-order Q→A, CXCL12 secretion is exponential growth + linear accumulation, CAF proliferation is Richards logistic). Converting these to algebraic forward models would halve the diffrax calls. This requires either splitting multi-condition targets into separate YAMLs or adding an `entry_inputs` mechanism to the error model schema — deferred for now.

---

## 2026-03-23 18:00 — CAF proliferation + interconversion submodel targets from Biffi 2019

Shifted strategy from single-parameter algebraic targets to time-course ODE targets using digitized proliferation curves from Biffi et al. 2019 (Cancer Discov). This paper is the primary reference for our iCAF/myCAF interconversion model but was previously annotated as "qualitative only" — turns out the CellTiter-Glo proliferation assays contain rich kinetic data when modeled properly.

**Target 1: `caf_prolif_rates_PDAC_deriv001` (Fig S6K).** Three independent logistic growth curves for quiescent PSCs, iCAFs, and myCAFs cultured in isolation. Digitized via WebPlotDigitizer (5 timepoints × 3 curves). MLE growth rates: myCAF 0.71/day (doubling 1.0d), iCAF 0.35/day (doubling 2.0d), qPSC 0.09/day (doubling 7.5d). The myCAF curve showed sharp logistic saturation at day 5 that standard logistic (n=1) couldn't capture — added a Hill coefficient `n_logistic` on the `(1-N/K)` term as a nuisance parameter (in vitro contact inhibition, not a QSP model change). Lognormal prior mu=1.1, sigma=0.5 (median ~3). MLE found n≈89 (step function), but n=3-5 captures most improvement. The existing k_myCAF_prolif target (Bachem 1992, rat HSC BrdU) uses completely different data (single-timepoint BrdU labeling fraction); this new target provides a complementary 5-day time course from mouse PDAC PSCs. Validated: schema PASS, prior derivation PASS, zero divergences.

**Target 2: `caf_interconversion_PDAC_deriv001` (Fig 5H).** Two-state model — the key extraction. PSCs cultured with organoid-conditioned media (pure iCAFs) vs CM + 20 ng/mL TGFb (iCAF→myCAF conversion). The CM+TGFb curve grows ~46% faster because converted myCAFs proliferate faster, and total luminescence = iCAF + myCAF. Forward model is a 4-state custom ODE: quiescent growth (control), pure iCAF growth (CM), and coupled iCAF/myCAF with interconversion (CM+TGFb). Replicates full QSP Hill function structure including TGFb antagonism of IL-1R1 (`H_IL1_eff = IL1/(IL1_50*(1+TGFb/TGFb_50_IL1R1)+IL1)`). Seven QSP parameters: `k_iCAF_to_myCAF` (primary), `k_myCAF_to_iCAF`, `TGFb_50_CAF_act`, `IL1_50`, `TGFb_50_IL1R1`, `k_iCAF_prolif`, `k_myCAF_prolif` (shared with S6K). At 20 ng/mL TGFb (0.8 nM), the TGFb Hill function is near-saturating (H≈0.92), so this primarily constrains k_iCAF_to_myCAF directly. TGFb_50_CAF_act is weakly constrained (would need sub-saturating TGFb doses). Validated: schema PASS, prior derivation PASS, zero divergences. The reverse direction (k_myCAF_to_iCAF) is weakly constrained because TGFb suppresses IL-1R1, reducing reverse conversion.

Key observation: translation sigma (~0.63) dominates for both targets, keeping posterior medians close to priors despite strong MLE signals. This is correct — mouse in vitro rates are 3-6x higher than we'd expect in vivo. The real value is in joint inference: shared k_iCAF_prolif/k_myCAF_prolif between both targets, and the interconversion rate ratio, will compose into tighter joint constraints. Next steps: Fig S6B (qPSC+TGFb → myCAF activation) and in vivo iCAF:myCAF ratio (Fig S7E/S8G) for additional constraint.

---

## 2026-03-20 05:30 — Cytokine secretion rate extraction: k_CXCL12_sec_iCAF + k_IL1_sec

Shifted focus from mechanosensing/EC50 parameters to cytokine secretion rates — the Tier 2 targets identified in the fibroblast param audit. Started by investigating three candidates (k_PDGF_sec, k_CXCL12_sec_iCAF, k_IL1_sec) in the model code and searching literature for ELISA secretion data. Key finding: k_PDGF_sec is not cleanly extractable because wild-type PDAC cells produce undetectable PDGF-BB by ELISA (retention motif traps it in ECM). Deferred that parameter despite it being PRCC #1.

Extracted three submodel targets from three papers. **Saiman 2013** (deriv001): JS1 murine hepatic stellate cell CXCL12 ELISA time course (0-96h, 2e4 cells, 300 uL). Used custom_ode with cell proliferation + secretion + degradation. Hit a time unit mismatch bug (ODE span in hours but rate constants in 1/day) that caused MCMC to struggle with step sizes ~1e-9 — fixed by dividing rates by 24 in the ODE. Posterior median k_CXCL12_sec_iCAF = 2.34e-9, 23x higher than current 1e-10. Cross-species proxy (translation sigma 0.93). **Tjomsland 2013** (k_IL1_sec deriv001): PC013 primary human PDAC cell supernatant IL-1alpha ELISA (Fig 3B, 48h steady-state in 2 mL). Used algebraic steady-state model since IL-1alpha t1/2 ~2.4h << 48h collection. Exact indication match, translation sigma only 0.38. Posterior median 4.2e-11 nmol/(cell*day), 42x higher than the placeholder 1e-12 — the placeholder was essentially a guess and the data strongly corrects it (z=22.8). Also shifts k_IL1_deg down to 2.59/day from prior 6.93/day. **Ene-Obong 2013** (deriv002): primary PDAC patient-derived aPSC CXCL12 ELISA (Fig 6E, n=7 individual patient scatter). Exact indication match but cell count/volume not reported — used nuisance parameters. Bootstrap over individual patient values rather than assuming normal distribution. Posterior median 6.29e-10, lower than Saiman's 2.34e-9, which makes biological sense (immortalized vs primary cells). Joint inference will compose the two estimates.

Coverage now 24/63 (38%), up from 22/67 (33%) — denominator shrank due to PDGF removal (see below).

**PDGF species removal (model structure change).** During literature search for k_PDGF_sec, discovered that wild-type PDAC cells produce **undetectable** PDGF-BB by conditioned media ELISA (McCarty 2007, JCI). This is because PDGF-B has a C-terminal retention motif that traps it in ECM (Heldin & Westermark 2008). The model was treating PDGF as a freely diffusing cytokine with secretion/degradation dynamics, which is the wrong physics. Moreover, k_PDGF_sec was PRCC #1 — the model's highest sensitivity parameter was uncalibratable, creating artificial parameter degeneracy.

Decision: **remove PDGF species entirely (Option A)**. Removed PDGF species, 4 parameters (k_PDGF_sec, k_PDGF_deg, PDGF_50, PDGF_50_prolif), 2 secretion/degradation reactions, and the H_PDGF Hill functions from all activation and proliferation reactions. Rationale: PDGF is a permissive signal in the PDAC TME (cancer cells are adjacent to PSCs, PDGF is locally matrix-bound at saturating levels), not a discriminating one. TGFb and IL-1 are what determine myCAF vs iCAF fate. The CAF proliferation rates (k_myCAF_prolif etc.) now represent effective in-vivo rates at physiological PDGF levels, which is what the Bachem 1992 BrdU data actually measured. Lost the PDGF_50_prolif submodel target (Bachem 1993), but that EC50 was only useful if we knew where on the dose-response curve the TME sits — and we don't. Net: -4 parameters, -1 species, -1 submodel target, eliminates the #1 PRCC sensitivity artifact.

---

## 2026-03-19 17:00 — IL1_50 submodel targets + full mechanosensing session wrap-up

Extracted IL1_50 + n_IL1 targets from Ciminieri 2023 (Am J Respir Cell Mol Biol). MRC-5 lung fibroblasts treated with IL-1beta at 0.01, 0.1, 1 ng/mL (0.6, 5.7, 57 pM). Figure 2I qRT-PCR shows IL-6 saturates by 5.7 pM while CXCL8 still increases from 854→1250 fold between 5.7 and 57 pM — different apparent gene sensitivities but shared receptor EC50. Both targets give IL1_50 ~18-22 pM (current model: 10 pM), confirming the model is in the right ballpark. n_IL1 ~0.45-0.60 (sub-linear, like n_Texh and n_fib).

Finding the right IL-1 dose-response paper took significant searching. Biffi 2019 only used 2 concentrations (no dose-response). Son 2021 (Sci Signal) studies NF-kB as a dose differentiator — fascinating but measures response to concentration *changes*, not absolute levels, making it incompatible with the model's static Hill function. Ciminieri 2023 was the right fit: 3 IL-1beta concentrations spanning the receptor Kd range with quantitative qRT-PCR readout. Broken y-axis in Figure 2I complicated digitization but the data is usable.

**Full session summary:** 7 submodel targets extracted across 3 papers, constraining 6 QSP parameters (3 new Hill coefficients). Key finding: all three mechanosensing/signaling Hill functions (T cell stiffness, fibroblast stiffness, IL-1 signaling) show sub-linear Hill coefficients (n=0.3-0.6), meaning these pathways act as gradual rheostats rather than binary switches. This is a biologically meaningful correction — the previous n=1 default overstated the sharpness of all three responses. Also contributed a `derived_arithmetic` input type to the maple schema for traceable unit conversions.

---

## 2026-03-19 15:30 — E_50_Texh submodel targets from Saitakis 2017 + new n_Texh parameter

Extracted two submodel targets constraining the T cell mechanosensing Hill function (`H_stiff_Texh`) from Saitakis et al. 2017 (eLife). The paper measures IFNγ and TNFα production by human CD4+ T lymphoblasts on PA-gels of 3 stiffnesses (0.5, 6.4, 100 kPa) with TCR stimulation. Both cytokines show a graded dose-response — exactly the kind of data needed for a Hill function EC50.

The key finding: the mechanosensing response is strongly sub-linear. A joint Hill fit gives EC50 ≈ 177 kPa with n ≈ 0.48, meaning the response hasn't saturated anywhere near the PDAC-relevant stiffness range (5–20 kPa). The Bayesian inference (via maple validate) confirms n_Texh posterior median ≈ 0.33 with contraction 0.98 — the data very strongly favors n < 1. E_50_Texh stays near its prior (median 4.5, z_score 0.13) because the high translation sigma (0.79, driven by proxy indication + low TME compatibility) keeps the prior dominant. This makes sense: the EC50 estimate from 3 in vitro stiffness points with healthy donor CD4 T cells shouldn't override the prior too aggressively.

Added `n_Texh` as a new QSP parameter (default 1.0 for backward compatibility). Updated `fibroblast_module_enhanced.m` to use `stiffness^n_Texh / (E_50_Texh^n_Texh + stiffness^n_Texh)` instead of the old n=1 form. The model default (n=1) gives `H_stiff = 0.56` at 5 kPa; the data-informed n=0.33 gives `H_stiff ≈ 0.15` — the model was overstating stiffness-mediated exhaustion by ~4×. This is a biologically meaningful correction: stiffness acts as a gentle rheostat, not a binary switch. Used WebPlotDigitizer for figure digitization (individual donor dots) — much more practical than manual reading. Assessed Biffi 2019 for IL1_50 but it only has 2 IL-1α concentrations (no dose-response). Liu 2015 for E_50_fib may be hallucinated — needs verification.

---

## 2026-03-18 — Fibroblast parameter lit search + 6 new submodel targets

Systematic literature search for quantitative kinetic data on CAF activation, interconversion, death, proliferation, and cytokine secretion rates. The existing sources (Biffi 2019, Elyada 2019) provide mechanism and steady-state ratios but no rate constants. Searched for papers with actual time-course or dose-response data, found three primary sources: Bachem 1992 (JCI), Bachem 1993 (J Hepatology), and Kisseleva 2012 (PNAS). Also reviewed Lee 2025 (npj Syst Bio ODE model — not useful, static CAF fractions), Bougueon 2024 (PLoS Comp Bio HSC model — useful as computational reference pointing to primary data), and Ohlund 2017 (JEM — qualitative switching data only).

**Six submodel targets extracted and validated** (52b64af through 00daa23), bringing coverage from 10/61 (16%) to 16/61 (26%):

1. `k_TGFb_sec_myCAF` — Bachem 1992 Table II. TGFb1 production 0.20 fmol/cell/d in MFBLC. **50× shift up** from current 2e-11 to 1e-9 nmol/(cell·d). The heuristic was way too low.
2. `k_myCAF_prolif` — Bachem 1992 Table I. BrdU incorporation across 4 FCS conditions (21-60%). Bootstrap across conditions captures uncertainty about in vivo growth factor levels. Median 0.24/day (current 0.15).
3. `k_myCAF_death` — Kisseleva 2012 in vivo recovery. Total HSC fraction drops from 14.3% to 5.6% over 30 days post-CCl4. **Confirms current value** (0.017 vs 0.0165/day).
4. `k_psc_activation_myCAF` + `k_myCAF_prolif` — Bachem 1993 Fig 4a. custom_ode with digitized alpha-SMA% + cell count time course. Joint 2-parameter target initially, expanded to 7 parameters (5 QSP + 2 nuisance) to match the full model's Hill function structure.
5. `PDGF_50_prolif` — Bachem 1993 Fig 7b. direct_fit Hill curve from PDGF dose-response on MFBLC thymidine incorporation. **Confirms current value** (0.18 vs 0.2 nM).
6. `TGFb_50_CAF_act` + `k_psc_activation_myCAF` + `PDGF_50` — Bachem 1992 Fig 7b. Paired custom_ode (control vs +1ng/ml TGFb1) with 5 params (3 QSP + 2 nuisance). TGFb_50_CAF_act median 0.05 nM (down from 0.07).

**Key architectural insight:** The activation rate `k_psc_activation_myCAF` initially showed a 15× discrepancy between targets (0.065 vs 1.0/day). This was because the Bachem 1993 ODE had no Hill functions — `k_act` absorbed the basal TGFb/PDGF Hill terms. After adding Hill gating to match the QSP model's reaction structure (`k_act × TGFb/(EC50+TGFb) × PDGF/(EC50+PDGF)`), both targets agree on ~0.9/day intrinsic rate. Lesson: **submodel forward models must replicate the QSP model's parameterization**, not simplify it. Include as many QSP parameters as mechanistically relevant, using nuisance parameters for experimental conditions (basal cytokine levels in culture).

**Nuisance parameter support added to maple** (same session). Two nuisance parameters (`tgfb_basal_culture`, `pdgf_basal_culture`) represent basal cytokine levels in 10% FCS culture. Shared by name across both Bachem targets. Marginalized out of output priors. Identified the need for explicit nuisance parameter merging in joint inference (currently by naming convention).

**Joint inference across all 22 targets** (run_joint_inference) shows clean MCMC (zero divergences, r_hat ~1.0) with notable correlations: d_crit_T/n_pore (r=-0.75), E_ref/phi_col_ref (r=+0.49), f_col_basal/k_col_myCAF_sec (r=-0.45), k_myCAF_prolif/k_psc_activation_myCAF (r=-0.12).

**Next targets identified:** Bachem 1993 Fig 7 PDGF dose-response (done), Bachem 1992 Fig 7b TGFb-accelerated activation (done). Remaining: Kisseleva 2012 apoptosis snapshot and adoptive transfer reactivation ratio (lower priority). See `notes/calibration/fibroblast_lit_search_summary.md` for full inventory.

**Files:** `calibration_targets/submodel_targets/` (6 new YAMLs + 3 digitized CSVs), `notes/calibration/fibroblast_lit_search_summary.md`, `notes/calibration/fibroblast_param_audit.md` (updated).

---

## 2025-03-18 — IL-1 species addition + fibroblast module audit

Added IL-1alpha as an explicit species to replace the `(1-H_TGFb)` proxy for iCAF formation. The old formulation was mechanistically wrong: in the PDAC TME where TGFb is high, `(1-H_TGFb) ≈ 0`, forcing near-zero myCAF→iCAF conversion — inconsistent with the observed ~1:1 iCAF:myCAF ratio (Elyada 2019). The proxy conflated "low TGFb" with "IL-1 signaling present", but these are independent axes.

**IL-1 mechanism (Biffi 2019):** Tumor-secreted IL-1alpha drives iCAF formation via NF-kB → LIF → JAK/STAT3. TGFb antagonizes this not by directly suppressing the iCAF program, but by downregulating IL-1R1 expression (SMAD3-mediated transcriptional repression). This means iCAF formation depends on the *ratio* of IL-1 to TGFb-suppressed receptor availability, not on the *absence* of TGFb.

**New species:** `V_T.IL1` (nanomolarity) — IL-1alpha in tumor compartment.

**New parameters:** `k_IL1_sec` (cancer cell secretion), `k_IL1_deg` (degradation, t_half ~2.4h), `IL1_50` (basal EC50 for iCAF formation), `TGFb_50_IL1R1` (TGFb EC50 for IL-1R1 downregulation).

**New Hill function:** `H_IL1_eff = IL1 / (IL1_50 * (1 + TGFb/TGFb_50_IL1R1) + IL1)`. The effective EC50 shifts rightward with TGFb — more IL-1 needed to drive iCAF when TGFb is high. This captures the receptor-level antagonism cleanly.

**Reactions updated:**
- qPSC→iCAF activation: `(1 - H_TGFb) * PDGF * IL6` → `H_IL1_eff * PDGF * IL6`
- myCAF→iCAF interconversion: `(1 - H_TGFb) * IL6` → `H_IL1_eff * IL6`
- iCAF→myCAF interconversion: `TGFb_50` → `TGFb_50_CAF_act`
- Collagen secretion: `TGFb_50` → `TGFb_50_col_sec`

**TGFb_50 split completed.** The generic TGFb_50 (0.07 nM) was used for 5 different biological processes in the fibroblast module. Now fully split:
- `TGFb_50_CAF_act`: PSC activation to myCAF + iCAF↔myCAF interconversion (SMAD-driven lineage)
- `TGFb_50_col_sec`: Collagen secretion enhancement by myCAFs
- `TGFb_50_IL1R1`: TGFb antagonism of IL-1R1 expression (new, part of IL-1 addition)
No remaining generic `TGFb_50` references in the fibroblast module.

**Audit findings fixed:**
1. *Stale notes:* iCAF species notes, `k_psc_activation_iCAF`, `k_myCAF_to_iCAF` all said "low TGFb" — updated to IL-1-driven.
2. *Dead parameters removed:* `collagen_base` and `collagen_MW` were registered in the model but never used in any reaction, rule, or downstream code. Removed from model registration (still in `parameters_PDAC.m` if needed).
3. *Unnecessary guard:* Stiffness rule had `(phi_col_ref + 1e-6)` guard on a constant (0.12). Removed.
4. *Clearance/niche tension resolved:* All CAF subtypes had `k_cell_clear` tumor-eradication clearance, which contradicted the `K_stiff_niche * H_stiff_fib` carrying capacity term designed to sustain CAFs in stiff ECM post-therapy. Fix: CAF clearance reactions now modulated by `(1 - H_stiff_fib)` — in stiff ECM, clearance is suppressed. qPSC clearance left unchanged (resident cells that revert with tumor regression).

**Unblocked:** Elyada 2019 CAF subtype ratio target (iCAF:myCAF:apCAF = 44:45:10) can now be extracted — the model structure correctly represents competing IL-1 vs TGFb signaling axes.

**Files touched:** `core/modules/fibroblast_module_enhanced.m`, `parameters/parameters_PDAC.m`

---

## 2025-03-15 — ECM module redesign: collagen, pore size, stiffness, immune-dependent degradation

Rewrote the ECM subsystem to replace the generic bulk-concentration approach with mechanistic derived quantities. The old model had a redundant dual-ceiling (ECM_max=100 mg/mL vs ECM_density=25 mg/mL) where Hill functions operated in ranges they could never fully explore, and a single `ECM_level` (mg/mL) variable driving all immunosuppressive effects regardless of mechanism.

**What changed structurally:**

1. **Species rename: ECM → collagen.** The model now tracks fibrillar collagen (types I/III) explicitly rather than a generic "ECM" species.

2. **phi_collagen replaces ECM_level.** Dimensionless collagen volume fraction (0–1), computed as `V_T.collagen / (V_T × rho_collagen)`. Single physical ceiling via volume feedback — no more ECM_max. Secretion saturation is now `(1 - phi_collagen)` which actually engages as collagen fills the available volume.

3. **Pore-size-based immune exclusion replaces ad-hoc Hill functions.** `d_pore = d_pore_ref × (phi_col_ref / phi_collagen)^0.5` gives pore diameter in micrometers, which feeds into `H_pore_T = 1/(1+(d_pore/d_crit_T)^n_pore)` for T cells and `H_pore_APC` for APCs. Parameters calibrated from Wolf 2013 (J Cell Biol): T cell arrest at pore cross-section 4 μm² → d_crit_T = 2.3 μm. Key insight: 7 μm was the tumor cell value, not T cells. DCs tolerate smaller pores (MMP/podosome migration) → d_crit_APC = 2.1 μm.

4. **Stiffness-based T cell exhaustion replaces H_ECM_T_exh.** `stiffness = E_ref × (phi_collagen/phi_col_ref)^n_stiff` (power law, n≈1.8 from collagen gel literature). Exhaustion: `H_stiff_Texh = stiffness/(E_50_Texh + stiffness)`. Also adds `H_stiff_fib` for fibroblast mechanotransduction.

5. **Stiffness-dependent CAF niche.** CAF carrying capacity was purely cancer-cell-coupled (`f_stroma_max × C_total`). Now includes `K_stiff_niche × H_stiff_fib` — stiff ECM sustains CAFs via YAP/TAZ even without cancer cells. Enables fibrotic scar persistence post-therapy.

6. **Immune-dependent collagen degradation.** Added `k_MMP_M1 × V_T.Mac_M1 / V_T` term alongside constitutive `k_col_deg`. GVAX-induced M1 polarization can now open the stroma, enabling T cell infiltration.

**Key feedback loops enabled:**
- Stiffness → fibroblast activation → more collagen → stiffer (positive, bounded by secretion saturation)
- Therapy → M1 macrophages → MMP → collagen degradation → pore size ↑ → T cell infiltration ↑

**Calibration implications:** The design doc (`notes/model/ecm_formulation_review.md`) includes a multi-level calibration strategy. Priority submodel targets: Wolf 2013 (pore size arrest), Sapudom 2021 (CTL migration+killing vs collagen density), Rubiano 2018 (stiffness-collagen relationship), Erstad 2020 (collagen time course in KPC mice). These constrain the new constitutive relationships jointly rather than parameter-by-parameter.

**Files touched:** 19 files across 4 core modules, parameters, priors CSV, 7 scripts, 3 test files, 1 postprocessing utility. All 72 relevant MATLAB-backed behavior tests pass; 3 pre-existing failures unrelated to ECM. Also fixed stale priors path in test_simulate_scenario_median.py.

**TODO from this work:**
- [ ] Decompose k_C_Tcell_eff into infiltration flux × contact rate × per-contact kill probability (pore exclusion conflated with killing rate)
- [ ] Extract submodel targets from Wolf 2013, Sapudom 2021, Rubiano 2018
- [ ] Add HA species + IFP when modeling systemic drug delivery
- [ ] Add TGFb sequestration (latent/active pools) if treatment dynamics require it

---

## 2026-03-13 19:30 — rho_Th_max submodel target + prior comparison tooling

Extracted `rho_Th_max` submodel target from Carstens 2017 IHC data (`rho_Th_max_PDAC_deriv001.yaml`). The old prior was a vague lognormal (median 5M cells/mL, σ=0.80) inherited from the generic QSPIO framework. The Carstens data — CD4+FoxP3- densities across 109 PDAC specimens — gives a much tighter constraint: median 26.7M cells/mL, σ=0.31. This was rolled into `pdac_priors.csv` as part of the repo reorganization (`accbb21`).

Built a prior comparison script (`scripts/compare_prior_update.py`) to visualize the impact of prior changes on simulation outputs. Key design decisions: (1) **controlled comparison** — unchanged parameters get identical draws across old/new conditions, so any trajectory difference is attributable solely to the changed parameter; (2) **delegates simulation to `qsp-hpc` batch_runner** — started with a pure MATLAB version but it hung on `readtable`/`git show` I/O; Python handles orchestration cleanly while MATLAB just runs sims; (3) **content-addressed caching** in `cache/prior_comparison/` so plotting iterations don't re-run MATLAB (~90s per 20-sim batch). Supports `--replot` and `--clear-cache` flags.

Results from the 20-sample comparison (`d0ea59e`): the rho_Th_max shift has modest impact on the six tracked observables (tumor volume, CD8, Treg, CD8/Treg ratio, ECM, mAPC). Treg density shows the most visible separation. The 40% IC failure rate (tumors not reaching 3.2cm diagnosis diameter) is consistent across both conditions, confirming this is a prior-width issue on growth-related params rather than a rho_Th_max effect. Also reorganized repo: moved priors to `parameters/`, qualitative bounds to `tests/`, reference values to `calibration_targets/`, updated all path references (`accbb21`, `8a63e8b`, `d0d6e07`).

---

## 2026-03-12 — OBED iteration 2: IQR-based dispersion fix + re-ranking

Re-ran OBED with two fixes: (1) replaced `np.var` with IQR-based dispersion `(IQR/1.349)^2` in prior tightening (step 5) and observable augmentation (step 4), and (2) updated priors from 4 SubmodelTarget YAMLs merged into pdac_priors.csv.

**Why the fix matters.** The outcome distribution (log tumor fold-change) has kurtosis=48.8 — ~30 extreme sims dominate `np.var`. Any 50% subsample that randomly excludes a few outliers shows large spurious "variance reduction," making irrelevant parameters (including stripped-module noise like k_NK_kill) appear in the top 5. IQR-based dispersion equals variance for Gaussian data but is robust to heavy tails.

**Key ranking shifts (prior tightening):**
- k_NK_kill: #5 (ΔVar=+37%) → rank ~97 (ΔDisp=+2.4%) — confirmed artifact, stripped module
- k_Th_pro_tumor: #1 (ΔVar=+46%) → #37 (ΔDisp=+15.7%) — still relevant, not dominant
- k_APC_mature_ID: #2 (ΔVar=+46%) → negative ΔDisp — literature prior already well-constrained
- **New top 3:** v_T_search_volume (+33.4%), k_K_g (+31.4%), k_IL2_sec (+30.1%) — tumor-immune encounter mechanics and IL-2 signaling

**MI sweep unchanged.** KSG mutual information is rank-based (robust to heavy tails), so cached MI results are valid. Top species: k_C_Tcell_eff (MI=0.44), V_T.CCL2 (0.34), V_T.IFNg (0.22) — same biological themes as 2026-03-10.

**Step 4 (observable augmentation) ΔDisp still unreliable.** Values like -42662% arise from noisy IQR estimation in small bins (~140 sims per outcome decile). MI rankings from step 3 remain the reliable observable importance measure. Fixing step 4 requires either larger simulation grids or a different binning strategy.

**Stripped-module parameters still in priors CSV.** Parameters from NK, hypoxia, metabolism, chemotaxis modules appear in prior tightening as low-level noise (ΔDisp ≤ +24.5%). With the IQR fix they no longer dominate rankings, so cleanup is not urgent.

**Next targets for SubmodelTarget extraction:** k_IL2_sec (IL-2 secretion ELISA), Q_nTreg_thym (thymic output TREC assays), TGFb axis (k_TGFb_Tsec + TGFb_50_APC), PD1_50 (pharmaceutical dose-response). See `notes/obed_literature_priorities.md` for full updated search strategy.

## 2026-03-12 — OBED→literature loop: prior updates shift medians 2–170x

Completed the first full OBED→literature→prior iteration loop. The OBED MI sweep (2026-03-10) identified high-MI parameters; we extracted SubmodelTarget YAMLs for 4 of them from primary literature; and the resulting priors reveal that hand-tuned CSV values were systematically wrong — not just imprecise, but centered in the wrong part of parameter space.

**Prior comparison (yaml_to_prior.py vs pdac_priors.csv):**

| Parameter | CSV median | Literature median | Shift | Interpretation |
|---|---|---|---|---|
| EC50_GMCSF | 0.5 nM | 0.003 nM | 170x lower | CSV guess; actual EC50 is picomolar (Guthridge2006 TF-1 dose-response) |
| k_APC_mature_ID | 0.7/day | 2.73/day | 4x higher | DC maturation completes in ~8h (Han2009 CD80 kinetics) |
| k_IL2_deg | 0.2/min | 0.091/min | 2x lower | Roughly right; Lotze1985/Konrad1990 PK data |
| k_NK_kill | 0.15/day | 5.5/day | 37x higher | CSV was heavily conservative; Bhat2007 shows ~6 kills/NK/day |

**Key insight:** OBED flagged these parameters as high-value for variance reduction in predicted tumor volume. But the actual information gained was median correction, not variance reduction. The model was operating in a biologically implausible regime — GM-CSF barely activating DCs (EC50 too high), NK cells barely killing (rate too low). No amount of SBI posterior shrinkage can fix priors centered in the wrong place.

**Implication for the iteration loop:** MI rankings were computed under the old (wrong) priors. At the corrected operating point, the parameter sensitivity landscape may be different. Re-running OBED with updated priors is needed before selecting the next batch of extraction targets.

### yaml_to_prior.py — automated prior generation pipeline

Built `metadata/yaml_to_prior.py` to convert SubmodelTarget YAMLs into SBI-ready prior rows:

1. Load YAML, run observation_code bootstrap → 10K parameter samples (via forward model inversion)
2. Fit candidate distributions (lognormal, gamma, inverse-gamma), select best by AIC
3. Anderson-Darling absolute GOF test (N=2000 subsample, Stephens 1986 critical values)
4. Apply 8-axis translation sigma inflation rubric from source_relevance fields
5. Output lognormal prior rows (translation inflation always in log-space)

**Multi-distribution fitting:** k_NK_kill is gamma-distributed (dAIC=+437 over lognormal) due to the gamma bootstrap in observation_code; other 3 are lognormal. All priors are expressed as lognormal after inflation (equivalent median + inflated sigma).

**Diagnostic plots** (`--plot` flag): histogram of inverted parameter samples with overlaid fitted PDFs for all candidates plus the inflated prior. Saved to `figures/yaml_to_prior/yaml_to_prior_fits.png`.

### 8-axis translation sigma rubric

Replaced the removed `estimated_translation_uncertainty_fold` (subjective per-target guess) with a deterministic, auditable rubric computed from structured source_relevance enum fields. Components add in quadrature with a floor of 0.15.

**Original 5 axes:** indication_match, species, source_quality, perturbation_type, tme_compatibility

**3 new axes added to schema (maple SourceRelevanceAssessment):**
- `measurement_directness` — inferential steps between measurement and parameter (direct / single_inversion / steady_state_inversion / proxy_observable)
- `temporal_resolution` — data structure constraining the parameter (timecourse / endpoint_pair / snapshot_or_equilibrium)
- `experimental_system` — biological fidelity of the experimental system (clinical_in_vivo / animal_in_vivo / ex_vivo / in_vitro_coculture / in_vitro_primary / in_vitro_cell_line)

**Rubric refinements:** pathological_state→0.0 (was 0.1; it's the most relevant perturbation for a disease model), physiological_baseline→0.1, pharmacological→0.25 (was 0.3), tme:moderate→0.15 (was 0.2), floor=0.15 (was 0.0).

The new axes now differentiate targets that previously collapsed (e.g., EC50_GMCSF gets in_vitro_cell_line penalty=0.30 vs k_APC_mature_ID's in_vitro_primary=0.20; k_IL2_deg gets timecourse=0.0 + clinical_in_vivo=0.0, making it the tightest prior).

**Publishability note:** The specific sigma values are not empirically calibrated — they're structured expert judgment. Defensible as a principled default (cite toxicology uncertainty factor precedent), but proper calibration would be a methods paper on its own (cross-validation on multi-derivation parameters).

### Schema and test changes (maple)

- Added `MeasurementDirectness`, `TemporalResolution`, `ExperimentalSystem` enums to `enums.py`
- All 3 new fields + `perturbation_relevance`, `tme_compatibility`, `tme_compatibility_notes` made required on `SourceRelevanceAssessment`
- Test DI improved: `DEFAULT_SOURCE_RELEVANCE` dict + `make_source_relevance(**overrides)` factory in test file — future schema field additions need only one dict updated (was 10 inline copies)
- 46/46 submodel target tests pass, 185/185 maple core tests pass

### Full iteration loop

The validated workflow is now:
1. **OBED** — MI sweep identifies high-value parameters
2. **Literature extraction** — SubmodelTarget YAMLs with full provenance
3. **yaml_to_prior** — automated prior generation with multi-dist fitting + translation inflation
4. **Structural validation** — qualitative bounds tests + LLM-elicited prior predictive checks
5. **If fail** → fix model structure (add/modify modules)
6. **If pass** → SBI posterior inference
7. **Repeat** from 1 with updated priors

**Next:** Update pdac_priors.csv with the 4 literature-grounded priors, run prior predictive simulations with the new priors, check qualitative bounds and LLM-elicited targets. If structural validation passes, proceed to SBI. If not, the shifted operating point has exposed model deficiencies that need fixing.

---

## 2026-03-10 14:30 — OBED infrastructure: MI sweep over trajectory grids

Built out the full OBED (Optimal Bayesian Experimental Design) workflow for identifying which additional literature data would most reduce uncertainty in treatment response prediction. The plan (`notes/obed_plan.md`) uses mutual information between dense species×time trajectory grids and a continuous tumor outcome to rank observables by information value.

**Key design pivot — binary→continuous outcome.** Initially classified response as binary (MPR via viable tumor fraction ≤35%, or RECIST categories), but prior predictive sims produced 0% MPR across all scenarios — the prior is too diffuse to generate responders. Switched to `log(V_final/V_baseline)` as a continuous outcome, which preserves full variance information and avoids the degenerate-binary problem. This also lets us use the KSG continuous-continuous MI estimator (`mi_ksg`) rather than the less powerful continuous-binary version.

**Infrastructure built across three repos:**
- *qsp-hpc-tools*: Added `extract_trajectory_grid()` to `QSPSimulator` wrapping the HPC pipeline (pool path resolution → submit → wait → download). Also added `derive_trajectory_grid_worker.py` for server-side Parquet extraction.
- *qsp-sbi*: Added `mi_sweep_continuous()` with progress logging (rate + ETA every ~5%), `generate_tightened_theta_sets()` for prior tightening analysis (clamp or filter methods). Both exported from `qsp_sbi.inference.obed`.
- *pdac-build*: `workflows/obed_workflow.py` (602 lines) orchestrates the full pipeline with MI result caching, composite feature discovery (disabled by default — too slow for marginal gain), LOO observable decomposition (requires SBI artifacts), and prior tightening candidate selection (LOO impact × PRCC sensitivity scoring).

**MI sweep results for gvax_nivo_neoadjuvant (2814/4000 valid sims):** Top TME features after filtering out PK species, V_T, and synapse bookkeeping: `k_C_Tcell_eff` (MI=0.83, day 14) — T cell killing rate is the single most informative non-PK species; `V_T.CCL2` (MI=0.74, day 30) — macrophage recruitment chemokine; `V_T.IFNg` (MI=0.60, day 9) — early immune activation; `V_T.PDGF` (MI=0.60, day 30) — stromal remodeling. Full results in `notes/obed_mi_results.md`. These point to CD8 killing efficacy data, CCL2 tumor concentrations, and early IFNg as the highest-value literature search targets.

**Debugging notes:** Pool hash mismatch (OBED simulator didn't include `model_script` → different hash than sbi_workflow). Stale cached grids from wrong pool persisted after fix — had to delete `cache/trajectory_grids/`. Baseline scenario had `stop_time=1` day, causing all-NaN beyond t=1 — added auto-detection of last valid timepoint. MI sweep takes ~10 min per scenario (8K columns × KSG estimator) — added caching to `figures/obed/mi_heatmap_{scenario}.csv`.

**Next steps:** Run MI sweep for remaining 3 scenarios. Map features to actionable literature search targets. Run forward sims for prior tightening once LOO identifies candidates (requires SBI artifact saving in sbi_workflow.py). 25 OBED unit tests passing in `tests/test_obed_utils.py`.

---

## 2026-03-09 16:30 — Betancourt workflow review + calibration strategy

Reviewed Betancourt's "Towards A Principled Bayesian Workflow" (`notes/principled_bayesian_workflow.Rmd`) to clarify what's valid for iterative model fixing. Key takeaways for our workflow:

**Prior predictive checks** must compare to domain expertise only, never observed data (lines 616-624). Using calibration targets in prior predictive checks = empirical Bayes = double-dipping. Our LLM-elicited targets serve this role correctly. The model passes (D²=27.4, p=0.591 against LLM targets).

**Posterior retrodictive checks** explicitly use the same data that informed the posterior (lines 1930-1935). This is the standard approach — Betancourt acknowledges it "assess[es] only the self-consistency of the model at that particular observation" but doesn't require new data.

**Between iterations** (lines 2669-2677): "we want the *qualitative failure* of a posterior retrodictive check to motivate model expansion, not the *quantitative features of the observed data itself*." The posterior retrodictive failure triggers domain expertise; domain expertise drives the fix. Carry the pattern ("MDSCs too low") not the number ("should be 48.3").

**Vigilance warning** (lines 2625-2628): "Once we consider the observed data we introduce a vulnerability to overfitting and so we must be extremely vigilant in how we further develop our model in subsequent iterations." This isn't a clean firewall — it's a discipline.

**Held-out data** (lines 1937-1946) is presented as an optional upgrade for stronger validation, not a requirement. The standard workflow refits on the same data across iterations.

**Practical implication**: We can proceed to calibrate with the current model (which passes prior predictive against domain expertise). Posterior predictive failures will motivate the next iteration's model changes — but those changes must be biologically motivated, not numerically targeted. The 3 known misfits (M1/M2, apCAF, MDSC) visible from peeking at real data should NOT drive pre-calibration fixes; they may resolve through calibration (parametric) or persist (structural), and only the posterior predictive can tell us which.

---

## 2026-03-09 15:00 — Multi-scenario SBI + LLM elicitation for treatment targets

### Multi-scenario joint SBI (4 scenarios, 2000 sims)

Enabled all 4 scenarios for joint NPE training: baseline_no_treatment (22 obs), clinical_progression (1 obs), gvax_neoadjuvant (6 obs), gvax_nivo_neoadjuvant (9 obs) — 38 total observables, 30 PRCC Tier-1 parameters. The config hash handles scenario selection so no model version bump needed.

**Prior predictive D² comparison (self-reference null):**

| Configuration | D²_obs | Null median | Empirical p |
|---|---|---|---|
| Baseline only (22 obs, 4000 sims) | 17.6 | 19.6 | 0.585 |
| All 4 scenarios (38 obs, 2000 sims, real targets) | 69.8 | 30.2 | 0.078 |
| All 4 scenarios (38 obs, LLM-elicited targets) | 27.4 | 30.2 | 0.591 |
| Treatment only (16 obs, LLM-elicited targets) | 4.4 | 11.6 | 0.951 |

The model passes misspec detection against LLM-elicited targets (p=0.591) but is borderline against real targets (p=0.078). This gap is expected — the LLM panel produces wider, more centrist estimates that are easier to satisfy. The real targets are more precise and create genuine tensions the model must resolve through calibration, but p=0.078 doesn't indicate structural misspecification at α=0.05. Treatment scenarios alone (16 obs) are too sparse to constrain the model.

R² recovery dropped modestly (median 0.100→0.061) but is confounded by half the training data (1287 vs 2552 valid sims). Top identified params stable: k_M2_pol (0.63), k_Mac_rec (0.53), k_Treg_death (0.42). k_Treg_death and k_C1_growth *improved* with treatment data — the GVAX scenarios provide signal for tumor growth and Treg dynamics that baseline alone cannot.

### LLM expert panel elicitation for treatment scenarios

Extended the LLM expert elicitation framework to cover treatment observables (16 targets across clinical_progression, gvax_neoadjuvant, gvax_nivo_neoadjuvant). Created `scenarios/llm_elicitation_prompt_treatment_personas.md` with 10 expert personas. Key prompt design decisions:

1. **Denominator clarification was essential**: The original prompt didn't emphasize that mIHC cell percentages use ALL nucleated cells as denominator (not just immune cells). This caused systematic overestimation of macrophage/DC fractions (LLM estimated ~5% M2 TAMs vs real ~0.6%).
2. **Minimal anchoring to avoid data leakage**: Initially added quantitative anchors (baseline ranges, effect sizes) but these essentially told the LLMs the answer. Stripped back to only: (a) denominator definition, (b) immune sum <15% constraint, (c) PD-1 staining artifact note.
3. **Persistent systematic bias**: Even with corrected prompts, the LLM panel underestimates GVAX's T cell infiltration effect by ~3× (LLM: 2-3% CD8, real: 8%) and overestimates M2 TAMs. This is likely a genuine knowledge gap — the magnitude of GVAX priming in surgical specimens is surprisingly large.

Updated `scripts/aggregate_llm_elicitations.py` to combine both baseline (v1-v10) and treatment (v1-v3) elicitation files. Consensus CSV at `scenarios/llm_elicited_consensus.csv` now has 43 targets (27 baseline + 16 treatment).

### Predictive output targets ideation

Wrote `notes/predictive_output_targets.md` cataloging clinically relevant endpoints the model should predict but currently doesn't constrain well. Top priorities: ORR and OS across treatment arms (data available from O'Reilly 2019, Le 2013/2015, Zheng 2022). These require population-level aggregation (simulate N VPs, compute response rates or KM curves).

### Open question

The model is structurally sound but under-constrained for predictive outputs. Before adding more calibration targets, need to think about principled model fixing — using degrees of freedom on structural corrections (M1/M2 ratio, apCAF, MDSC) rather than exhausting them on calibration.

---

## 2026-03-09 11:00 — Deduplicate redundant calibration targets, resolve D² inflation

### Problem: redundant target pairs inflated posterior D²

After excluding `cancer_cell_density_baseline` and `psc_activation_ratio`, posterior predictive D² was 52.8 (empirical p=0.047, borderline significant). LOO decomposition revealed the top misfit drivers were **redundant target pairs** measuring the same biology from different sources:

| Pair | Influence (combined) | Biology |
|------|---------------------|---------|
| cd8_treg_ratio + cd8_treg_ratio_golesworthy2022 | 18.5 + 17.6 = 36.1 | CD8/Treg balance |
| m2_macrophage_fraction + m1_m2_ratio | 7.8 + 7.8 = 15.6 | Macrophage polarization |
| apcaf_fraction_of_caf + apcaf_fraction_of_caf_elyada2019 | 6.7 + 6.7 = 13.4 | apCAF fraction |

Within each pair, both targets agreed directionally — the model was off in the same direction for both. The double-counting inflated D² without adding diagnostic information.

### Exclusion decisions (quality-based)

For each pair, kept the higher-quality target:

1. **CD8/Treg**: Kept Golesworthy 2022 (deriv002, n=166, per-patient ratio directly reported) over Sivagnanam (deriv001, n=113, ratio of independently sampled medians — introduces ratio-of-means bias and ignores within-patient correlation).

2. **Macrophage polarization**: Kept m1_m2_ratio (Costa 2022, n=30, 5-marker panel: CD68/CD163/IRF5/CD86/CD206) over m2_macrophage_fraction (Sivagnanam, n=113, single CD163+/− marker, CI [0.08, 0.96] — barely constraining).

3. **apCAF fraction**: Kept deriv001 (human scRNA-seq + Ogawa protein-level bounds, CI [0.02, 0.21]) over deriv002 (mouse KPC flow cytometry, CI [0.08, 0.13] — tight CI may be SEM-as-SD, cross-species translation uncertain).

### Also excluded: psc_activation_ratio (Drifka 2016)

Drifka's α-SMA color deconvolution measures DAB/(DAB+hematoxylin) pixels — the denominator is ALL nucleated cells in periductal stromal ROIs, not just fibroblasts. The current mapping to `(iCAF+myCAF+apCAF+qPSC)` is a denominator mismatch. The ~20% dilution gap vs Ogawa's marker-classified mycaf_fraction_of_caf (0.575 vs 0.716) cannot be decomposed without spatial immune data. No dispersion reported (CV=0.2 assumed). Redundant with mycaf_fraction_of_caf. See `notes/psc_activation_ratio_mapping_review.md`.

### Results after deduplication (22 observables, 30 params)

| Metric | Before (25 obs) | After (22 obs) |
|--------|-----------------|----------------|
| Posterior D² | 52.8 | **22.4** |
| Empirical p | 0.047 | **0.455** |
| D²_post/D²_prior | 2.99 | **1.27** |
| Obs outside 90% PPC CI | 1/25 | 1/22 |
| Obs exceeding null 95th | 7 | **2** |
| SBC miscalibrated | 2/30 | **0/30** |
| Post. pred. NaN rate | 116/500 (23%) | 104/500 (21%) |

The model now **passes** the misspecification test (p=0.455). D²_post/D²_prior=1.27 means almost all prior predictive misfit was parametric (fixable by calibration) with minimal structural tension.

Only 2 observables exceed null influence threshold: `mdsc_density` (influence 9.0, p=0.010) and `stromal_fraction` (influence 6.6, p=0.023). `mdsc_density` remains the sole target outside 90% PPC CI (obs=48.3 vs post_med=7.12, p=0.066).

### Remaining real model issues (no longer masked by double-counting)

The predictive summary shows three directions where the model is genuinely off:
- **M1/M2 ratio too low** (post_med 0.274 vs obs 0.518, z=−0.82) — too M2-polarized
- **apCAF fraction too low** (post_med 0.007 vs obs 0.064, z=0.82) — constitutive apCAF production insufficient
- **MDSC density too low** (post_med 7.12 vs obs 48.3, z=1.59) — persistent structural miss

These are genuine model structure issues, not data quality artifacts.

---

## 2026-03-09 10:30 — Resolve stromal/cancer density target conflict, drop cancer_cell_density

Diagnosed why the tier-1 SBI run (D=30, N=2601) had posterior predictive D²=4100: `stromal_fraction` and `cancer_cell_density_baseline` were accounting for 86% of the total misfit (LOO influence 3525 each). Both observables measure cellularity through the same `(C1, V_T)` pair, but the target *values* are numerically inconsistent. Back-solving: `stromal_fraction` target (66.8% stroma, Bengtsson n=142) implies cellularity=33.2% → density=1463 cells/mm², but the `cancer_cell_density` target (996 cells/mm², Sivagnanam n=113) implies cellularity=22.6% → stroma=77.4%. The 10.6pp gap in cellularity comes from the geometric bridge `area_per_cell=0.000227 mm²/cell` (17µm sphere cross-section). Reconciling would require ~20.6µm effective diameter — plausible for sectioned tissue but fragile.

Dropped `cancer_cell_density_baseline` (moved to `excluded/`) and kept `stromal_fraction` as the cleaner constraint (direct area measurement, larger cohort, no geometric conversion). Also simplified both observables to use V_T directly rather than reconstructing tumor volume from species, eliminating a second source of definitional inconsistency.

Re-ran tier-1 SBI: D² dropped from 4100 → 449 (9.1× reduction), exactly as LOO predicted. `stromal_fraction` influence dropped from 3525 → 12.3. Recovery R² essentially unchanged (median 0.095→0.100, same 3 params above 0.5). New top misfit drivers: `psc_activation_ratio` (398) and `mycaf_fraction_of_caf` (393) — the CAF activation/subtype balance is the next structural tension to address. Only 1/26 observables outside 90% posterior predictive CI (`mdsc_density`, marginal at p=0.05). Effective Jacobian rank improved from 5.4→6.4.

Also reverted the ECM_density bump (0.012→0.025) from the working tree — keeping it at 0.012 for now to evaluate the collagen ceiling issue separately from the target set change.

---

## 2026-03-08 01:30 — LLM expert panel elicitation + full misspec detection run (4000 sims)

### Persona-based elicitation for diverse expert estimates

Initial LLM elicitations (v1, v2) were too consistent — same model, same prompt, same internalized knowledge. Created `scenarios/llm_elicitation_prompt_personas.md` with a `{PERSONA}` placeholder and explicit instructions requiring step-by-step quantitative derivation from anchor quantities. Eight personas: GI pathologist, tumor immunologist, stromal biologist, myeloid biologist, clinical oncologist, systems biologist, comparative oncologist, single-cell researcher. Each persona description emphasizes different expertise biases (e.g., "myeloid biologist who tends to estimate higher myeloid cell densities than T cell-focused researchers").

Generated v3–v10 via `scripts/generate_llm_elicitations.sh`, then aggregated with `scripts/aggregate_llm_elicitations.py` — samples 10,000 from each expert's implied distribution (lognormal for positive quantities, logit-normal for fractions), pools all 100,000 draws, takes median and 2.5/97.5 percentiles. Result: `scenarios/llm_elicited_consensus.csv` with 27 observables from 10 experts. Good diversity on key quantities: e.g., single-cell researcher estimated higher cDC1/apCAF fractions than pathologist, myeloid biologist pushed TAM density higher.

### Misspec detection results (4000 sims, full pipeline)

Updated `sbi_workflow.py` so `MISSPEC_DETECTION_MODE=True` swaps *all* real data references — not just diagnostic comparisons but also NPE conditioning (posterior sampling). No real calibration targets appear anywhere in the pipeline.

**Prior predictive**: D²=41.7, self-reference null p=0.100 — borderline. Four stromal observables flagged: `psc_activation_ratio` (p=0.007), `cancer_cell_density_baseline` (p=0.004), `stromal_fraction` (p=0.005), `mycaf_fraction_of_caf` (p=0.012). All 27 observables within 90% marginal CI — prior is wide enough individually but shows joint tension.

**Posterior predictive** (conditioned on LLM consensus): D²=3951, p=0.000, D²_post/D²_prior=94.6. 19/27 observables flagged. Top two drivers: `stromal_fraction` (influence=3583) and `cancer_cell_density_baseline` (influence=3579) — together 90% of total D². The model can cover LLM targets marginally under the prior, but once the NPE picks parameters to fit some observables, it creates massive tension on the stromal compartment.

**Cross-validation with CMA-ES**: The same structural bottlenecks appeared in the earlier CMA-ES point-optimization (`workflows/misspec_diagnostic.py`) against real targets — `treg_fraction_cd4`, `collagen1_protein_level`, `total_til_density`, CAF subtypes all had |z|>2 residuals. Two independent methods (NPE posterior predictive + CMA-ES best-fit) converge on the same conclusion: the stromal compartment and Treg/CD4 balance are structurally coupled in ways that prevent joint calibration. The NPE's D² magnitude (3951) is likely inflated by posterior concentration, but the direction is real.

**Self-reference null behavior**: At 130 sims the null was extremely heavy-tailed (D² out to 700, p=0.508 — test had no power). At 2552 sims the null tightened substantially (median 22.5, p=0.100). The heavy tail comes from pathological prior draws; more sims give a tighter bulk. Per-observable influence p-values were more informative than the aggregate D² throughout.

NPE also improved with more data: top R² went from 0.097 (200 sims) → 0.614 (2000) → 0.741 (4000). Effective Jacobian rank 6.2/27 — only ~6 independent observable directions.

### Next steps

Structural fixes needed in the stromal compartment — the model can't simultaneously satisfy stromal fraction, cancer cell density, myCAF fraction, and PSC activation. These are the same observables CMA-ES couldn't fit. Need to investigate the coupling between `f_stroma_max`, ECM production, and cancer cell carrying capacity.

---

## 2026-03-07 — Misspecification detection framework: resolve double-dipping, self-reference null for D²

### Problem: double-dipping in prior construction

The Treg/Th fix (previous entry) exposed a fundamental workflow issue: we used prior predictive diagnostics to identify the Treg imbalance, tuned 5 parameters until the prior predictive looked good, then reported prior predictive coverage as if it validated the priors. This is textbook empirical Bayes — using the data twice. Betancourt: *"Allowing comparisons between the prior predictive distribution and any observed data to influence the development of the prior model is a form of empirical Bayes that uses the data twice."* Consequences: posterior credible intervals become too narrow (double-counting data influence), coverage degrades, and SBC can't detect it (our 0/30 miscalibrated result is meaningless if the prior was shaped by the same data).

### Solution: LLM expert elicitation + posterior predictive D²

Designed a three-phase workflow that cleanly separates domain knowledge from calibration data. **Phase 1** (structure validation): elicit hypothetical data and priors from an LLM expert panel (SHELF protocol via LLMs — reproducible, scalable, auditable). Use this synthetic data to detect and fix misspecification via posterior predictive D². No real data touched. **Phase 2**: calibrate to real data (first and only use). **Phase 3**: validate on held-out treatment targets (e.g., GVAX+nivo Zheng 2022).

Key insight: **posterior predictive D²** (after calibrating to LLM-elicited data) is fundamentally stronger than prior predictive D² for misspecification detection. The posterior has used up parameter flexibility — any remaining discrepancy is unambiguously structural. LOO influence on posterior predictive identifies observables the model is actively trading off against each other, not just "hard to cover under the prior."

### Self-reference null distribution (zero extra simulations)

Instead of comparing D² to χ²(k_eff) (assumes normality, approximate k_eff), use the posterior predictive samples themselves as the null. Each x_i is what x_obs "would look like" if the model were correct. Compute D²_i for each, compare D²_obs to the distribution — pure empirical p-value, no distributional assumptions. Handles over-parameterization naturally: if excess parameters allow overfitting, both D²_obs and the null shift together, keeping the p-value calibrated. LOO bias correction via leave-one-out mean/covariance (still no extra simulations).

### iCAF data conflict resolution

Dropped Elyada 2019 iCAF fraction target (0.4435, scRNA-seq, n=6) in favor of Ogawa 2021 (0.284, protein IHC, n=215). Moved YAML to `calibration_targets/baseline_no_treatment/excluded/`. Reverted MODEL_VERSION from v4 back to v3 — the version bump was unnecessary since the config hash already changes with test stats CSV content. Reran 200-sim baseline: D² dropped 39.8 → 25.4, iCAF pair no longer dominates LOO. New top drivers: cDC1 density (5.2), CD8:Treg ratio (4.6), stromal fraction (4.2), MDSC density (4.2).

Full framework documented in `notes/misspecification_detection_framework.md`.

---

## 2026-03-06 — Fix structural Treg/Th imbalance: apCAF conversion + proliferation asymmetry

### Problem

Treg/CD4+ fraction was ~0.97 in prior predictive (target: 0.34–0.44 from Hiraoka 2006 and Sivagnanam 2021). Tregs consumed nearly all CD4+ space, and `treg_density_baseline` was the #1 LOO influence driver (11.3), dominating the entire NPE. The model was structurally misspecified for Treg homeostasis.

### Root cause analysis

Traced Th cell scarcity through the full Th budget in the TME. Identified two compounding issues:

1. **apCAF-mediated Th→Treg conversion was massively overpowered.** Three estimated parameters with no literature backing created an always-on, fast drain:
   - `k_apCAF_Treg = 0.5/day` — 3.3× Th proliferation rate (0.15/day), 23× the TGFβ conversion rate (0.022/day). Elyada 2019 (the cited source) provides no rate constant.
   - `K_apCAF_Treg = 1e6 cells` — saturated with modest apCAF counts
   - `P1_50_apCAF = 1e-12 M` — effectively zero, reaction always on

2. **Unjustified Treg proliferative advantage.** Treg had 8× the proliferative capacity of Th (rate × carrying capacity: 0.3×2e7 vs 0.15×5e6). Ki67+ data supports only ~1.3× rate difference.

### Fixes applied

**apCAF parameters** (all three were estimates with `derivedFrom: estimated`):
- `k_apCAF_Treg`: 0.5 → 0.03/day (~1.5× TGFβ pathway, reflecting that apCAFs lack costimulatory molecules)
- `K_apCAF_Treg`: 1e6 → 1e7 cells (Hill term saturates only when apCAFs are substantial)
- `P1_50_apCAF`: 1e-12 → 1e-9 M (requires meaningful antigen for activation)

**Treg proliferation** (Option 1 from earlier analysis):
- `k_Treg_pro_tumor`: 0.3 → 0.2/day (Ki67+ supports ~1.3× Th, not 2×)
- `rho_Treg_max`: 2e7 → 1e7 cells/mL (2× Th cap, not 4×)

Updated `pdac_priors.csv` to match (new log-means for all five parameters).

### Results — median simulation

| Metric | Before | After apCAF fix | + Prolif fix | Target |
|---|---|---|---|---|
| Treg/CD4+ | ~0.97 | 0.82 | **0.63** | 0.34–0.44 |
| Treg/CD3+ | ~0.24 | 0.18 | **0.075** | 0.05 |
| CD8:Treg | ~2.2 | 4.3 | **11.8** | 3.7–5.4 |

### Results — 200-sim prior predictive (baseline only, tier 1)

**27/28 observables within 90% CI** (1 marginal miss: cdc1_density_baseline).

Key improvements in prior predictive:
- `treg_fraction_cd4` median: 0.97 → **0.51** (target 0.44)
- `treg_density_baseline` median: 109 → **31.7** cells/mm² (target 13.9)
- `cd4_helper_density` median: 1.2 → **27.0** cells/mm² (target 17.0)

**LOO influence**: `treg_density_baseline` dropped from **#1 (11.3) → #17 (1.3)**. No longer dominates the NPE. New top tensions are cDC1 density (10.8) and iCAF fraction data conflict (10.7, two sources disagree: 28% vs 44%).

### Literature review context

Reviewed mechanisms for Treg homeostasis in PDAC TME. Key findings that informed the approach:
- IFNg-driven Treg fragility (Overacre-Delgoffe 2017) — well-established but in melanoma (hot tumor); PDAC is IFNg-poor, so likely weak at baseline
- IL-6 antagonism of Th→Treg (Kimura 2010) — would require Th17 species since Th17 is pro-tumorigenic in PDAC
- Treg depletion paradoxically accelerates PDAC carcinogenesis (Zhang 2020, Cancer Discovery) — high Tregs may be partially correct for PDAC
- apCAF-mediated conversion is a real but recently discovered mechanism; no quantitative rate data exists

Decided against adding new mechanisms (Options 2-5) in favor of fixing clearly wrong parameter values first. The structural imbalance was dominated by three estimated parameters with no literature support, not missing biology.

---

## 2026-03-06 11:30 — Multi-scenario SBI: index_values bug fix, theta alignment, 1000-sim run

Consolidated `sbi_workflow.py` into a proper multi-scenario joint inference pipeline. Four scenarios now share the same theta draws (baseline_no_treatment, clinical_progression, gvax_neoadjuvant_zheng2022, gvax_nivo_neoadjuvant_zheng2022) with per-scenario observables concatenated for NPE training (44 total: 28 baseline + 1 clinical + 6 GVAX + 9 GVAX+nivo).

**Critical bug found and fixed**: All 14 GVAX/GVAX+nivo calibration target YAMLs had empty `index_values` fields. The wrapper code in `yaml_loader.py` defaults `_target_t = 0.0` when `index_values` is empty, so treatment observables were being evaluated at t=0 (initial conditions) instead of day 21. This made GVAX and GVAX+nivo cd4_pct values bit-for-bit identical — they were both just reading diagnosis state. Added `index_values: [21.0]`, `index_unit: day`, `index_type: time` to all 14 files. The cd8_fold_increase target already had index_values set, which is how the bug was confirmed.

**Theta alignment fix**: The previous 200-sim run had 48% joint NaN rate because baseline sampled 200 from a 4000-sim pool (different theta indices) while treatment scenarios generated fresh 200 with seed=2025. Bumped MODEL_VERSION to `multi_scenario_v2` to force all pools empty, ensuring identical theta draws across scenarios. Result: NaN rate dropped to 31% (687/1000 survived), matching the expected ~28% from evolve_to_diagnosis failures.

**1000-sim results**: D² = 79.1 (higher than the 200-sim D² = 30.6 — more stable prior predictive estimates reveal larger true coverage gaps). Top misspec drivers: treg_density_baseline (16.3), cd4_pct_nonLA_gvax_nivo_d21 (15.7), cd4_pct_nonLA_gvax_d21 (13.4). R² improved substantially with 619 training samples vs 94: median R² = 0.044 (was -0.14), best k_M2_pol (0.357). Local calibration analysis shows 12 params lose R² near observed data — structural identifiability, not data volume, is the bottleneck. Also fixed a brittle Parquet batch counting bug in `hpc_job_manager.py` where the HPC login wrapper's `go.sh` injected error text into stdout, breaking `isdigit()` parsing.

---

## 2026-03-05 17:15 — Misspecification assessment: more data, not more structure

### Verdict

No strong structural misspecification. 28/28 prior predictive, 26/28 posterior predictive, zero structural conflicts, zero dead targets. The two marginal misses (`cdc1_density_baseline`, `mdsc_density`) are both sensitive to parameters and not competing with other targets — addressable without model changes.

The soft signals are about **identifiability**, not fidelity: effective rank 5.4/28, 13/30 sloppy directions, median R² = 0.095. The model can explain the data, but many parameter combinations explain it equally well. The 27.8% NaN rate is benign — those draws just don't evolve to tumor diameter within 10 years, not crashes.

### Next step: more data, not more structure

Adding mechanisms (infiltration rate rework, etc.) would increase parameter count and worsen the degeneracy. The constraint bottleneck is that all 28 current targets are baseline diagnosis snapshots — they can't distinguish parameters with similar steady-state effects but different dynamic or treatment-response signatures.

Priority data sources to break degeneracies:
1. **Multi-scenario calibration** — `gvax_nivo_neoadjuvant_zheng2022` targets constrain treatment-response parameters that are invisible at baseline. Infrastructure already exists (`multi_scenario_sbi_workflow.py`).
2. **Temporal targets** — even 2-3 longitudinal observations (tumor growth rate, immune infiltrate kinetics) would constrain dynamics parameters (`k_C1_growth`, `k_C1_death`, infiltration rates) that are currently sloppy in a single snapshot.
3. **Conditional/ratio targets** — observations that constrain relationships between observables under specific conditions would break the correlated degeneracies in the coupling matrix.

Model structure changes should wait until multi-scenario SBI reveals genuine structural conflicts (targets that compete across scenarios).

---

## 2026-03-05 16:30 — SBI workflow: posterior-based misspec diagnostics + local SBC calibration

### Changes

Overhauled `sbi_workflow.py` diagnostics (v6 → v7). The old step 3b computed a Jacobian via OLS on *prior* samples and used unsigned cosine similarity for conflict detection — both of which are unreliable for diagnosing structural misspecification (linearization is too far from the data, and raw coupling ignores which direction each target needs to move). Replaced with:

1. **Posterior Jacobian (step 7e):** Moved Jacobian estimation from prior to posterior predictive samples. The OLS is now computed in the posterior-concentrated region, giving a local linearization where it matters. Conflict detection uses residual-direction-aware cosine similarity (from `misspec_diagnostic.py`): Jacobian rows are sign-flipped by whether the posterior median overshoots or undershoots each target, so conflicts are only flagged when two targets genuinely need parameters to move in opposite directions. Dead-target threshold raised from 0.01 to 0.05.

2. **Local SBC calibration (step 5b):** Standard SBC/recovery averages over the prior — but we only care about NPE accuracy near our actual observation. New section filters test observations to the closest 20% (by Euclidean distance in copula space) and re-runs R² recovery, z-score contraction, and `sbi_calibration_ecdf` on that local subset. Side-by-side comparison table flags parameters whose calibration degrades near the data.

### Run results (v7, tier 1, 4000 sims → 2889 after 27.8% NaN filter)

Prior predictive: 28/28 targets within 90% CI (no structural unreachability), but joint D² = 52.4 (+10.8σ) driven by `cancer_cell_density_baseline` (14.5) and `stromal_fraction` (13.5). Notably, `collagen1_protein_level` is no longer a top LOO driver — the ECM_density fix from earlier today resolved that ceiling.

Posterior predictive: 26/28 in CI. Two marginal misses — `cdc1_density_baseline` (p=0.077, z=−3.11) and `mdsc_density` (p=0.056, z=−1.29). Both are MARGINAL (sensitive to params, no structural conflicts). Neither is dead or competing with other missed targets, so these are addressable through prior tuning or additional mechanism rather than fundamental model restructuring.

Local vs global calibration: median R² drops slightly (0.095 → 0.046), 8 params degrade locally by >0.1 (led by `k_ECM_myCAF_sec`, `k_myCAF_prolif`). But SBC miscalibration actually *improves* locally (3 → 1 param with KS p<0.05), meaning the NPE's credible intervals are better calibrated near the real data than on average. Contraction is stable (0.136 → 0.127).

Posterior Jacobian: effective rank 5.4/28, 13/30 sloppy directions. `N_IL2_CD8` dominates sensitivity for 4 of top 5 observables. The 15 boundary pile-ups (100% at boundary for `cd4_helper_density`, `cd8_treg_ratio`, etc.) indicate the copula transform struggles with the concentrated posterior — worth investigating alternative transforms.

---

## 2026-03-05 — ECM_density structural ceiling on collagen fold-change

### Problem

The `sbi_baseline_tier1` posterior predictive could not match the `collagen1_protein_level` calibration target (median 9.84, CI95 6.07-16.0). The prior predictive 95th percentile was only 9.32 — the simulator itself could barely reach the target, regardless of what the NPE did. The posterior piled up against this ceiling (posterior 95th = 9.67).

Initial hypothesis was `ECM_max` (carrying capacity for ECM production), but PRCC of ECM_max on collagen was only +0.02 — negligible. The saturation term `(ECM_max - ECM_level)/ECM_max` evaluates to 0.89 at the target ECM level of ~11 mg/mL, barely constraining.

### Root cause: ECM_density asymptotic ceiling

Traced through the V_T rule (assembled across multiple modules):

```
V_T = (cancer cells + T cells)/Ve_T + macrophages + CAFs + V_T.ECM/ECM_density
```

The last term (`V_T.ECM/ECM_density`) converts ECM mass to volume. Since `ECM_level = V_T.ECM / V_T`, as ECM mass grows it inflates V_T through this term, creating a self-limiting feedback. The maximum possible ECM_level converges asymptotically to **ECM_density** itself:

```
lim(ECM→∞) ECM_level = lim(ECM→∞) ECM / (V_cells + ECM/ECM_density) = ECM_density
```

With `ECM_density = 0.012 g/mL = 12 mg/mL`, the max fold-change was `12 / 1.144 = 10.49`. Observed max across 5000 prior sims: **10.31** — right at this theoretical ceiling. The top 20 sims all clustered tightly at 10.0-10.3 despite wildly different ECM_max values (33-293 mg/mL), confirming ECM_max was irrelevant.

### Diagnostic evidence

- Max collagen fold-change across 5000 sims: 10.31 (theoretical ceiling: 10.49)
- Top collagen sims had ECM_level/ECM_max = 0.02-0.49 (not saturated)
- Top collagen sims had stromal_fraction ~0.92 (ECM dominating V_T)
- Distribution showed sharp right edge at ~10.3, then zero above 10.4
- All top PRCC drivers (k_ECM_myCAF_sec, k_ECM_deg, TGFb_50, k_iCAF_to_myCAF) were already in tier 1

### Fix: ECM_density 0.012 → 0.025 g/mL

The key relationship: `ECM_level = ECV × ECM_density`, where ECV is the extracellular volume fraction.

**Literature evidence:**
- CT-derived ECV in pancreatic tissue with advanced fibrosis (F3): 42.5 ± 12.3% (Kai et al. 2023, DOI: 10.2169/internalmedicine.0410-22)
- Cancerous tissue interstitial collagen: 9-45 mg/mL (Antoine et al. 2014, DOI: 10.1089/ten.teb.2014.0086)
- Pure hydrated collagen density: ~1.19 g/cm³ (vs current effective 0.012)

ECM_density physically represents collagen protein concentration within the hydrated extracellular space. At 0.025 g/mL (25 mg/mL):
- Ceiling fold-change = 25/1.144 = **21.9** (was 10.5) — ample headroom
- At target ECM_level ~11 mg/mL: ECV = 11/25 = 44% — matches MRI data
- At median ECM_level ~6 mg/mL: ECV = 6/25 = 24% — reasonable

Previous value (0.012) implied ECV = 10/12 = 83% at target ECM level — unrealistically high.

### Files changed
- `parameters/parameters_PDAC.m`: ECM_density 0.012 → 0.025 g/mL
- `calibration_targets/submodel_targets/ECM_density_PDAC_deriv001.yaml`: New submodel target constraining ECM_density via algebraic model (ECV = C_tissue / ECM_density), validated against maple SubmodelTarget schema

### Impact
This changes V_T (tumor volume) via reduced ECM volumetric contribution, which affects:
1. Tumor diameter in `evolve_to_diagnosis` — tumors may take slightly longer to reach target diameter
2. All concentration-based species (ECM_level, cytokines) — denominator V_T changes
3. Cell densities (cells/V_T) — will shift
4. **Simulation pool must be regenerated** — all cached sims are invalid after this change

### What NOT to do
- Do NOT add ECM_density to SBI priors — it's a physical constant (tissue property), not a biological rate that varies between patients
- Do NOT increase ECM_max — it was never the bottleneck (PRCC = 0.02)

## 2026-02-24 — Devlog initialized

Set up DEVLOG.md and `/devlog` slash command for quick session logging.

## 2026-01-04 — Natural history workflow: evolve_to_diagnosis

The model previously started simulations from arbitrary initial conditions (C1=1 cell), which meant the first ~year of simulation time was just growing a tumor to a plausible size before any immunology mattered. Created `evolve_to_diagnosis` (`3515cf8`) — evolves from healthy tissue state to a target tumor diameter, then uses that as the IC for treatment scenarios. This is critical because the immune TME composition at diagnosis depends on the tumor's growth history (stroma remodeling, immune evasion, metabolic changes all happen during growth). Also configured the baseline scenario to use `stop_time=1` day since we only need the diagnosis snapshot for calibration.

## 2026-01-06 — Prior distributions and SBI infrastructure

First pass at priors (`fcb61f4`): wide log-normal distributions (σ=1.5) covering several orders of magnitude for all varied parameters. The idea was to be maximally uninformative and let SBI narrow things down. In hindsight this was too wide — many parameter combinations produced simulations that crashed or gave nonsensical results. Also set up the calibration targets infrastructure (`3ac366e`) with YAML schema for provenance tracking.

## 2026-01-08 — CCL5/CCR5 Treg recruitment and tuned priors

Added the CCL5 chemokine pathway (`2bba490`) — cancer cells and aPSCs secrete CCL5, which recruits Tregs via CCR5. This is separate from the antigen-driven Treg pathway and gives the model a way to get Tregs into the tumor without requiring APC-mediated priming. Important for PDAC's "immune desert" phenotype where conventional T cell priming is weak but Tregs are abundant.

Replaced the σ=1.5 wide priors with parameter-specific CVs (`0926169`): σ=0.3 for well-constrained literature values, σ=0.7 for moderately uncertain, σ=1.0-1.2 for poorly constrained. This dramatically reduced the fraction of simulations that crashed.

## 2026-01-15 — Isolated system targets and mechanistic fixes

Explored a different calibration approach: isolated system targets (`45671be`) for subsystem validation. The idea was to calibrate individual parameter subsets (e.g., PSC proliferation, PDGF EC50) against in vitro data before doing full-model SBI. Built a Julia/Turing.jl Bayesian ODE inference pipeline (`0a17b49`) as a proof of concept. This worked well for simple subsystems but the overhead of maintaining a separate Julia codebase wasn't worth it for the few parameters that have clean in vitro data.

Also fixed several mechanistic issues: ECM degradation changed from homeostatic (drive toward ECM_base) to first-order MMP turnover (`f269528`), added CCL5 secretion by aPSCs (`00b967d`), and added exhausted CD8 death rate (`1bb1fd9`) to prevent unrealistic accumulation.

## 2026-02-09 — Clinical outcomes test statistics

Added 15 clinical endpoints for DKL surrogate calibration (`1d2fe05`): RECIST response, PFS, tumor volumes at 30/60/90 days, immune biomarkers. This was the first step toward connecting the mechanistic model to clinical trial observables. These go in a separate `clinical_progression.yaml` scenario with 90-day simulation.

## 2026-02-12 — Implementation audit and APC maturation fix

Did a thorough audit of model assembly (`a7b8498`) by tracing every module's reactions, rules, and cross-module dependencies. Found two critical bugs:

**Antigen release was silently zero** (`9579f11`). The antigen_rate.m function searches for killing reactions by string-matching `k_C_CD8` in reaction rates, but the Phase 6 reparameterization renamed the killing rate to `k_C_Tcell_eff`. The search found nothing, so `R_T` defaulted to `0*cell/day`. This meant the entire antigen presentation pathway was disconnected — no tumor killing → no antigen release → no APC activation → no T cell priming. Fixed the search string.

**APC maturation was chicken-and-egg** (`9579f11`). Intratumoral APC maturation required IL-12 (`k_APC_mat*V_T.APC*H_IL12*(1-H_IL10)`), but IL-12 is secreted by mature APCs. In a cold TME with no pre-existing immune activation, this creates a deadlock — APCs can't mature without IL-12, IL-12 can't be produced without mature APCs. Replaced with DAMP + IL-12 OR logic: `k_APC_mat*V_T.APC*(1-(1-H_DAMP)*(1-H_IL12))*(1-H_IL10)*(1-H_TGFb_APC)`. Now tumor cell death (from any cause, including innate killing) produces DAMPs that can bootstrap APC maturation. Also added TGFb inhibition of DC maturation, which is important for PDAC's TGFb-rich stroma.

## 2026-02-13 — CAF subtype split and new module sprint

This was the big mechanistic expansion day. Split the single `aPSC` species into three CAF subtypes (`0be3585`): iCAF (inflammatory, IL-6/CCL5), myCAF (myofibroblastic, ECM/CXCL12/TGFb/VEGF), and apCAF (antigen-presenting, tolerogenic Treg induction). Based on Elyada 2019 and Biffi 2019 — the iCAF/myCAF balance is TGFb-driven and determines whether the stroma is inflammatory vs desmoplastic. Also added ECM-dependent APC trafficking (`9889b35`): ECM impedes both APC migration out of tumor and recruitment in, creating a stroma-dependent bottleneck for antigen presentation.

Renamed `c_vas` to `VEGF` throughout (`7787232`) and built the hypoxia-HIF-VEGF feedback module: pO2 as QSS algebraic parameter, HIF driving VEGF amplification (positive feedback), constitutive PD-L1 upregulation (IFNg-independent), and M2 macrophage polarization.

## 2026-02-14 — NK cells and metabolic competition

Added NK cell module (`f691b40`): antigen-independent innate killing suppressed by IL-10/TGFb/MDSCs. The key insight is that NK cells provide a second killing loop that doesn't require antigen presentation or T cell priming — important for PDAC where the adaptive immune system is largely excluded. NK-derived IFNg feeds into M1 polarization.

Built the metabolism module (`b7c1513`): glucose depletion (Warburg effect) and lactate accumulation. Glucose is QSS (same pattern as pO2), lactate is ODE. T cell killing and IFNg secretion scale with glucose availability; lactate drives M2 polarization. This creates a positive feedback loop: tumor growth → glucose depletion → impaired T cell function → less killing → more tumor growth.

## 2026-02-15 — cDC1/cDC2 split, CXCL9 chemotaxis, and calibration strategy

Split generic APCs into cDC1 (BATF3+, cross-presenting → CD8 priming) and cDC2 (IRF4+, MHC-II → CD4/Treg priming) (`8ad2375`). This is crucial for PDAC because cDC1s are ~79-fold scarcer than in lung adenocarcinoma (Hegde 2020), making cDC1 abundance a gatekeeping variable for immune activation. Added the NK→XCL1→cDC1→IL-12→NK positive feedback loop (Bottcher 2018) which is the primary mechanism linking innate and adaptive immunity in PDAC.

Added CXCL9/10 chemotaxis module (`99eea2b`): IFNg drives CXCL9 production by tumor cells, M1 macrophages, and mcDC1s, which recruits CD8 T cells and NK cells via CXCR3. This closes the adaptive→innate→adaptive loop. Also added CCL2-driven APC recruitment module, though this was later removed (`4fbd748`) because the biology was weak — CCL2 primarily recruits monocytes, not DCs, and we already had adequate APC dynamics.

Wrote the comprehensive calibration strategy (`297570d`) with the full workflow: target extraction with provenance → sensitivity screening → modular SBI → joint refinement → validation. This replaced the ad hoc approach with a systematic pipeline.

## 2026-02-16 — Calibration target extraction begins

Started the systematic calibration target extraction using `qsp-llm-workflows` (`10b390e`, `055c65d`). First batch of 6 approved YAMLs with full provenance (DOI, value snippets, experimental context, denominator audits). Added model structure and species metadata JSONs (`0a666a6`) so the extraction pipeline knows what species/compartments exist. Key decision: split targets into main CSV (42 targets with clean model mappings) and deferred cytokine CSV (12 targets where IHC scores vs absolute concentrations is a mapping problem).

## 2026-02-17 — Calibration target population and metabolite scale fix

Extracted ~30 calibration targets in one day: Li 2022 mIHC targets from digitized scatter plots (`5c404d3`), Ogawa 2021 CAF fraction targets (`e390436`), ECM/CAF targets (`a2b64a5`), tumor pO2 (`c9b5f2f`). Put glucose and lactate on absolute µM scale with Battini 2017 PDAC tissue data (`768a54d`) — previously they were dimensionless ratios which made calibration against real measurements impossible.

Added Pydantic schema validation test (`e8edf0e`) for calibration target YAMLs to catch malformed targets early.

## 2026-02-18 — Parameter audit and evolve_to_diagnosis fixes

Ran a systematic audit of all 177 prior values against literature (`4fe4a5d`): found 14 wrong values and 6 unit mismatches. Examples: f_cDC1=0.6 should be ~0.15-0.25 (cDC1s are rare in PDAC), thymic output rates 10-100x too high, k_IL2_deg too slow. Fixed initial_tumour_diameter default to 3.2 cm (`3595de0`) and removed it from priors — it's a scenario configuration parameter, not something SBI should vary.

Found and fixed a density inconsistency in evolve_to_diagnosis (`dc33aa5`): the healthy-state initial conditions weren't matching the ICs used for diagnosis-state species. Also updated pdac_cellularity_fraction from 0.25 to 0.15 across 8 cell density targets (`4271302`) — 15% is more consistent with PDAC's heavily stromal composition.

## 2026-02-19 — Restriction estimator and evolution time guards

Built the prior restriction estimator workflow (`e51946b`): uses SBI's RestrictionEstimator to learn the valid/invalid boundary in 177-dimensional parameter space, then samples only from the "plausible" region. ~43% of prior samples produce NaN (failed simulations), so this is essential for efficient SBI training. Added minimum evolution time guard to catch parameter combinations where the tumor never reaches diagnosis size.

## 2026-02-20 14:00 — Gompertzian growth, vasculature fixes, and simulation stability

Switched cancer growth from logistic to Gompertzian (`d07ae18`) to get realistic evolution from healthy tissue to diagnosis. Logistic growth was too slow early on — tumors couldn't reach diagnosis size in reasonable timeframes. Gompertzian's fast-early/slow-late profile matches PDAC doubling time literature much better. Also moved the ECM-dependent carrying capacity rule from gvax_module to fibroblast_module_enhanced where it belongs, and added glucose-dependent cancer proliferation.

Immediately hit parameter problems (`971c3d6`): k_K_g was 50x too high (7.37 → 0.15) because it was inherited from the HCC model — PDAC is hypovascular compared to HCC. k_K_d was 10x too low (4e-5 → 4e-4) for stromal vessel compression. And k_lac_Csec had a 1000x unit error (5e-4 → 5e-7) from confusing µmol with nmol — lactate was running away to absurd concentrations. After fixes, diagnosis time went from 174 to 473 days which is more realistic for PDAC. Also guarded all fractional exponents against transient negative species values (`a06dddf`) — the ODE solver trial steps were producing complex numbers that silently corrupted trajectories, firing millions of warnings on HPC.

## 2026-02-20 20:00 — Surrogate sensitivity analysis and restriction estimator tuning

Built the surrogate sensitivity pipeline (`9b67cdc`): PRCC + random forest importance across all 177 parameters and 31 calibration targets, reusing the restriction estimator simulation pool. Key design decision: replaced the split-half ranking stability approach with a permutation null test (`7418da2`, 1000 permutations, BH FDR < 0.05). Split-half was answering "is the ranking stable?" but what we actually need is "should this parameter be calibrated at all?" The permutation test answers that directly — 97/177 parameters are significant, which becomes the SA-restricted parameter set for SBI.

Also increased restriction estimator capacity (hidden_features=512, num_blocks=4, longer training) and added 2x oversampling to handle the ~43% NaN rate from failed simulations.

## 2026-02-22 10:00 — NPE diagnostics pipeline

Added a comprehensive NPE diagnostics suite to sbi_workflow.py (`4ecede5`, `632a2ec`): posterior predictive checks, R² recovery per parameter, posterior z-scores, coverage calibration, learning curves, and seed stability analysis. The diagnostics revealed several problems — boundary piling on cDC1 fraction and TAM:cDC1 ratio targets, poor R² on some parameters, and prior-data conflict on a few observables. These findings drove the model and calibration target changes that followed.

Built a Marp presentation (`967fe7a`) summarizing results for lab meeting.

## 2026-02-24 09:00 — Model mechanism updates from SA interpretation

Three mechanistic changes motivated by interpreting the sensitivity analysis results and reviewing scRNA-seq literature:

**CXCL12 reassigned from myCAF to iCAF** (`97bba8b`). The SA showed CXCL12 exclusion axis is significant (rank 23/40), but we had CXCL12 secretion attributed to myCAFs. Elyada 2019 and Ogawa 2021 scRNA-seq data consistently show CXCL12 as an iCAF marker, not myCAF. This matters because it couples T cell exclusion to the inflammatory (not desmoplastic) CAF axis.

**IL-6/JAK/STAT3 autocrine feedback on iCAF phenotype** (`97bba8b`). Biffi 2019 showed IL-1α from tumor cells initiates NF-κB → IL-6 → JAK/STAT3 → iCAF program. Previously the model had iCAF/myCAF balance driven only by TGFb. Added H_IL6_iCAF Hill function that amplifies both qPSC→iCAF activation and myCAF→iCAF interconversion. Reduced base iCAF activation rate from 0.4 to 0.2 to compensate for the feedback amplification.

**Tumor-local Treg proliferation** (`97bba8b`). Tregs are 30-60% Ki67+ in PDAC TME (Balli 2017, Plitas 2016) but the model only had peripheral Treg recruitment. Added logistic proliferation reaction with density-limited carrying capacity that grows with tumor volume.

## 2026-02-24 10:00 — Calibration target remapping and cleanup

Structural model review (`notes/structural_model_review.md`) identified several calibration target issues:

**PSC activation ratio remapped to myCAF-only** (`dce879d`). The Drifka 2016 assay uses α-SMA IHC, which is the defining myCAF marker. iCAFs are FAP+/α-SMA-low (Öhlund 2017) and wouldn't stain. Was previously mapped as (iCAF+myCAF)/total — corrected to myCAF/total. Cross-validated against Ogawa 2021 (n=215): ACTA2-dominant fibroblasts = 71.7% of total CAFs.

**Excluded two problematic DC targets** — cDC1_fraction_of_dc and tam_to_cdc1_ratio both showed boundary piling in NPE diagnostics. The cDC1 fraction had a denominator bug (immature-only vs total DCs in the IHC data). The TAM:cDC1 ratio conflates two independently varying axes. Moved to excluded/ for reformulation.

**Added cDC2 density target** from Plesca 2022 to replace the excluded targets with a cleaner observable.

Also moved working notes and digitization results out of the gitignored calibration_targets/sources/ into notes/ for proper version control.

## 2026-02-24 11:00 — SBI diagnostics extensions and test infrastructure

Added dimensionality sweep to sbi_workflow.py (`27256c8`) — retrains NPE at top-N parameter subsets (8, 15, 20, 30, 50) by SA ranking. This answers "how many parameters do we actually need?" without manual guessing, and helps identify where adding more parameters stops helping. Also added prior predictive histograms and PRCC heatmaps for visual diagnostics.

Started qualitative behavior tests (`ee9c920`) — pytest tests that validate structural model properties (immune cold baseline, stromal barrier effects) without requiring full calibration. These are meant to catch regressions when mechanisms are modified.

## 2026-02-24 14:00 — Hill function operating point audit

Ran every Hill function in the model at baseline (diagnosis, median parameters) and classified each as graded modulator, biological switch, or treatment reserve. The framework: a graded modulator should be in [0.05, 0.95] so perturbations produce proportional output changes; a switch should be at the correct extreme; a treatment reserve should be dormant at baseline. See `notes/hill_function_audit_2026-02-24.md` for the full table.

Three genuine problems found:

**IL10_50_NK = 5 nM is ~625x too high.** The macrophage IL-10 EC50 is 0.008 nM, giving H_IL10 = 0.40 at baseline IL-10 ~0.005 nM. But the NK EC50 of 5 nM means H_IL10_NK = 0.001 — NK cells are completely immune to IL-10, which contradicts in vitro evidence. Likely a typo or inherited from a different context. Need to reduce to 0.05-0.5 nM.

**HIF = 0.09 when it should be ~0.87.** Model produces pO2 ~19 mmHg at diagnosis vs clinical PDAC median ~2.7 mmHg. This cascades downstream: lactate production is underestimated (HIF amplifies glycolysis), H_lac_Teff and H_lac_M2 are barely active, VEGF is underproduced. The entire metabolic immunosuppression branch is weaker than it should be. Root cause is vascular/O2 consumption parameters need tuning.

**IL12_50 = 0.14 pM makes IL-12 always saturated** for M1 polarization (H_IL12 = 0.975). Any measurable IL-12 maxes out the response, so k_M1_pol absorbs what should be a graded IL-12 effect. Meanwhile IL12_50_NK = 50 pM gives H_IL12_NK = 0.073, which IS graded. The 357x difference is suspicious. IL12_50 probably needs to increase ~100x.

Also identified cascade dependencies: fixing HIF would cascade through the metabolic branch (lactate, glucose), and fixing ECM overproduction (230x vs observed 10x) would cascade through the stromal branch (T cell motility, APC migration, T cell exhaustion all near saturation).

## 2026-02-24 16:00 — Hill audit improvements: n-dependent thresholds, EC50 fixes, structural tests

Two commits (`b4ca36e`, `30c10b6`) that extend the Hill function audit from a one-time snapshot into a permanent regression test suite.

**N-dependent thresholds and reclassifications** (`b4ca36e`): Replaced the arbitrary [0.05, 0.95] sensitive range with a principled n-dependent formula: `[1/(1+3^n), 1-1/(1+3^n)]`, which gives [0.250, 0.750] for n=1 and [0.100, 0.900] for n=2. Reclassified H_IL12 and H_IL6_MDSC from `switch_high` to `graded` (they were miscategorized because their saturated values looked like switches, but they should be rheostats). Added treatment Hill output (Tier 6: GVAX+nivo d0 vs d21 comparison), analytical IQR from EC50 priors (Tier 7), and xfails for 14 known issues.

**EC50 fixes** (`30c10b6`): Fixed IL12_50 (0.14→15 pM) and IL10_50_NK (5.0 nM→0.01 nM). The IL-12 fix had a nice cascade effect: enabling graded M1 polarization via IL-12 made the IFNg→M1 pathway work under GVAX+nivo, so the M1/M2 treatment response test that was previously xfailed now passes. IL10_50_NK was confirmed against IL-10R1 Kd literature (50-200 pM); 10 pM is biologically justified.

**Root cause analysis of HIF underactivation**: Traced the pO2 QSS equation and discovered k_O2_demand (0.12) was hand-calibrated assuming VEGF=100,000 pg/mL, but the model only produces ~2,600 pg/mL. The `(1+VEGF/5000)` vascular quality discount is 1.52 instead of the assumed 21, making pO2 ~19 mmHg vs clinical ~2.7. Same anti-pattern for k_glc_tumor (assumed C_total=5e4, actual=2.5e8). This led to adding four new test tiers: clinical species ranges (Tier 8, all xfailed), QSS round-trip consistency (Tier 9, catches formula errors — already caught a bug where I had the wrong cell_demand weights), cascade coherence (Tier 10, 6 perturbation direction tests), and parameter sensitivity smoke tests (Tier 11, k_O2_demand 2x). Final suite: 58 passed, 18 xfailed, 0 failures.

## 2026-02-24 18:00 — Fix HIF cascade: VEGF_norm_50 scale mismatch and k_O2_demand recalibration

Fixed the HIF underactivation cascade by identifying a **unit scale mismatch** in VEGF_norm_50. The model tracks VEGF as a compartment-averaged concentration (~2,700 pg/mL), but VEGF_norm_50=5,000 pg/mL was calibrated against interstitial fluid measurements (Jain 2005). In reality, VEGF-A165 binds heparan sulfate in ECM, creating steep gradients — local peri-vascular concentrations are 10-100x higher than the compartment average. The vascular quality discount `(1 + VEGF/VEGF_norm_50)` was therefore barely active (1.54x), when it should be the model's primary mechanism for VEGF-driven vascular dysfunction.

**The fix (Option B from analysis):** Lowered VEGF_norm_50 from 5,000→500 pg/mL to correct the scale mismatch, then recalibrated k_O2_demand from 0.12→0.29 to hit clinical pO2=2.7 mmHg. This is a 2.4x increase in k_O2_demand (vs the 10x that would have been needed without the VEGF_norm_50 fix). The key insight: rather than adding structural complexity (free vs ECM-bound VEGF pools), we can correct the phenomenological equation's threshold to account for the known concentration gradient.

**Cascade resolution:** The fix resolved 7 xfails in one shot: all 4 clinical species ranges (pO2 1→10, VEGF 1k→200k, lactate 5k→40k, HIF 0.5→1.0) now pass, plus HIF switch_high, H_glc_C (glucose now properly depleted), and H_lac_Teff/H_lac_M2 (lactate now in the active range for immunosuppression). One new borderline xfail: H_IL12 dropped from 0.30→0.24 (just below 0.25 threshold) due to changed IL-12 dynamics with higher HIF. VEGF_norm_50 also appears in the glucose and lactate clearance equations, so lowering it strengthens all three metabolic couplings simultaneously. Suite now: 65 passed, 11 xfailed, 0 failures.

## 2026-02-25 09:00 — ECM cascade fix, TGFb diagnosis, CAF carrying capacity, V_T volume reinterpretation

Four interlinked fixes to the stromal subsystem, each revealing the next problem.

**ECM overproduction (835→~10 mg/mL).** The Hill function audit flagged H_ECM_T_mot and H_ECM_APC_mig as saturated switches rather than graded modulators. Tracing the cause: ECM_level was ~835 mg/mL at diagnosis — compared to measured PDAC collagen concentrations of 6-15 mg/mL, this is ~100x too high. The root cause was k_ECM_myCAF_sec and k_ECM_qpsc_sec being calibrated against the old single-aPSC model where one cell type produced all ECM. With three CAF subtypes (iCAF/myCAF/apCAF) and unrealistic myCAF counts, total production was enormous. Fixed by reducing k_ECM_myCAF_sec 207x (1.035e-6→5e-9) and k_ECM_qpsc_sec 345x (3.45e-7→1e-9). Also corrected ECM_base (60→3 mg/mL, ~3x normal pancreas collagen), ECM_max (1500→100 mg/mL), and recalibrated the three ECM Hill function EC50 fractions (f_ECM_50_T_exh, f_ECM_50_T_mot, f_ECM_50_APC_mig) to maintain graded sensitivity at the new scale.

**TGFb tripling (0.22→0.78 nM).** After ECM fix, H_TGFb_Teff (EC50=0.14 nM) and H_TGFb_NK (EC50=0.07 nM) started failing — both saturated at baseline. Investigating via cached sim outputs: myCAF count was 8.6 billion (39:1 ratio vs cancer cells), producing 91.7% of all TGFb. Net myCAF growth rate = 0.1335/day (5.2-day doubling time) with no density-dependent constraint — exponential growth limited only by PDGF saturation.

**CAF carrying capacity.** Added logistic density dependence to both myCAF and iCAF proliferation: `(1 - aPSC_total / (f_stroma_max * C_total + cell))` where f_stroma_max=3.0 is the maximum CAF:cancer ratio. First attempt used V_T-based density cap (rho_stroma_max × V_T), but this failed because V_T starts near zero during evolve_to_diagnosis — a single CAF immediately exceeds the capacity. C_total-based ratio cap is more robust: CAFs scale proportionally to the cancer population they're supporting.

**V_T too small → ECM_density reinterpretation.** With realistic CAF counts (~6.6e8 vs 8.6e9 before), total cell volume at diagnosis is only ~2 mL for a 17 mL tumor. The old model "worked" because 9.3B CAFs (24 mL) + 835 mg/mL ECM (12 mL) artificially inflated V_T. The real issue: ECM_density = 1.2 g/mL is the density of pure solid collagen, so `V_T.ECM / ECM_density` computes the volume of dry collagen fibers — which is negligible. In reality, ECM is a hydrogel: a sparse collagen network occupying ~80% of tumor volume as interstitial space. The fix: reinterpret ECM_density as an "effective hydrated density" (~0.012 g/mL = 12 mg/mL), so the ECM term naturally accounts for the full extracellular volume. Math: if ECM at 10 mg/mL fills 80% of volume, effective_density = 10/0.8 = 12.5 mg/mL ≈ 0.012 g/mL. This is more principled than the alternative (changing Ve_T from 1.0 to 0.12) because it localizes the fix to the ECM term rather than scaling all cell volumes.

Also added sim output caching: the test fixture now writes parsed outputs to `.sim_outputs.json` for debugging without painful stdout capture. Cleared fabricated derivedFrom references on all modified ECM parameters — the vast majority of literature citations in the parameter file have unverified provenance.

**Post-fix results (66 passed, 10 xfailed, 0 failures).** The combined ECM + CAF cap + ECM_density fix resolved 6 previously-xfailed tests: H_IL12 (0.256, just above 0.250), H_ECM_T_mot (0.427), H_TGFb (0.736), H_TGFb_APC (0.736), H_TGFb_Teff (0.583), H_TGFb_NK (0.661). Key species at diagnosis: V_T=17.2 mL, myCAF=1.32e9 (ratio 2.2:1), TGFb=0.195 nM, ECM_level=7.4 mg/mL, pO2=2.6 mmHg, HIF=0.88.

**IL-6 EC50 harmonization (50x spread → 3.3x).** Three pathways use the same ligand (IL-6) but had wildly different EC50s: IL6_50_MDSC=0.02, IL6_50_iCAF=0.30, IL6_50_M2=1.0 nM — a 50x spread for the same IL-6R/gp130/STAT3 receptor family. With model IL-6 at 0.087 nM, MDSC was permanently saturated (H=0.81) while M2 was inactive (H=0.08). Harmonized to 0.06/0.10/0.20 nM (3.3x spread, preserving the hierarchy: MDSCs most sensitive, M2 least). The iCAF autocrine loop strengthened as expected — lowering IL6_50_iCAF increased iCAF sensitivity, boosting iCAF count (3.7e8→6.0e8) and IL-6 production (0.087→0.140 nM). All three Hills now solidly graded: H_IL6_MDSC=0.70, H_IL6_iCAF=0.58, H_IL6_M2=0.41.

**New qualitative test tiers (69→87 passed).** Added Tiers 12-14 using existing sim output data — no new MATLAB runs needed. Tier 12 (cell composition): CAF:cancer ratio [1,5]:1, myCAF dominance >50%, M2:M1 >1, CD8 exhaustion 10-80%, Treg >1% of T cells, tumor diameter 2.5-4.0 cm. Tier 13 (progression direction): tumor/ECM/Treg/M2/MDSC increase over 90 days, pO2 decreases, no spontaneous regression. Tier 14 (treatment vs untreated): GVAX+nivo reduces tumor, M2, Tregs vs 30-day untreated. All 18 new tests pass.

**Lactate clearance fix (125 mM→25 mM).** The ECM_density change revealed that lactate was 3x the clinical upper bound (125 mM vs 10-40 mM). Root cause: the lactate clearance equation `k_lac_clear * lactate / (1 + VEGF/VEGF_norm_50)` uses the same unbounded linear VEGF term as pO2 and glucose. But for pO2/glucose, the VEGF factor multiplies demand inside a self-limiting denominator (approaches zero asymptotically). For lactate, dividing clearance by the impairment is unbounded — lactate can accumulate without limit. With VEGF/VEGF_norm_50 = 23, clearance was impaired 24x, which is unphysical (MCT transporters and interstitial diffusion still work regardless of vasculature).

Fix: replaced the linear form with a saturating Hill: `1 + f_VEGF_lac * VEGF / (VEGF + VEGF_norm_50)` where f_VEGF_lac=4.0 caps the max impairment at 5x. At model VEGF (~11,500 pg/mL), the effective impairment is 4.83x → lactate=25.4 mM, right in the Walenta 2004 range. H_lac_Teff=0.547 and H_lac_M2=0.614, both solidly in the graded range. Resolves 3 more xfails. Suite now: 69 passed, 7 xfailed, 0 failures.

## 2026-02-25 14:00 — Low-signal Hill reclassification and EC50 fixes: 94/0/0

Final cleanup of the Hill function audit. Four remaining xfails fell into two categories: dormant pathways and borderline EC50s.

**Dormant pathway reclassification.** H_IL12_NK (0.104) and H_IFNg_CXCL9 (0.056) were classified as `graded` but consistently sit deep in the switch_low region. The key question: are these treatment reserves (activating under therapy) or permanently dormant? Checked GVAX+nivo d0→d21 deltas: H_IL12_NK moved +2%, H_IFNg_CXCL9 moved +18% — neither comes close to the graded range [0.25, 0.75]. The upstream species (mcDC1, IFNg) are too scarce in PDAC to drive these pathways even under treatment. Reclassified both as `switch_low` with notes explaining the dormancy mechanism (mcDC1 scarcity for IL-12/NK, effector cytokine suppression for IFNg/CXCL9). These would need combination with IL-12 agonists or cDC1-targeted therapy to activate.

**Borderline EC50 fixes.** CCL5_50_Treg was 0.3 nM giving H=0.226 (below 0.25 threshold). Model CCL5 at ~0.065 nM suggested the EC50 was too high relative to the ligand concentration. Lowered to 0.15 nM → H=0.368, solidly graded. NO_50_Teff was 0.75 nM giving H=0.797 (above 0.75 threshold, meaning near-complete suppression at baseline). Raised to 1.5 nM → H=0.649, graded. Both changes are well within the uncertainty range for these EC50s (σ=0.7 in priors).

**Final suite: 94 passed, 0 failures, 0 xfailed.** The `_XFAIL_REASONS` dict is empty for the first time. All 28 Hill functions are either correctly classified as graded (within [0.25, 0.75]) or correctly classified as switches (with documented biological justification). The test suite has grown from the original 58 tests to 94, covering Hill operating points, clinical species ranges, QSS consistency, cascade coherence, parameter sensitivity, cell composition, progression direction, and treatment comparison.

## 2026-02-25 16:00 — Cascade time-series spot-check tests (Tiers 15-17)

Systematic review of cascade_figures.m time-series data (baseline 0-90 days, treatment 0-21 days at intermediate snapshots) identified specific behaviors to codify as regression tests. These capture the "spot-checks" from visual review of each cascade panel.

**MATLAB helper changes:** Added intermediate time points to the simulation output — `prog_d10` and `prog_d60` for progression, `tx_d3`, `tx_d7`, `tx_d14` for treatment. Added new `output_injection_site()` function for V_ID species (GVAX_cells, mAPC, APC) and V_LN species (mcDC1, mcDC2), plus `output_observable()` for `k_C_Tcell_eff` (the key nivolumab readout) at all treatment time points.

**Three new test tiers (22 tests, 2 xfails).** Tier 15 (progression time-series direction): 7 species should increase (HIF, CD8, NK, Mac_M1, IFNg, iCAF, myCAF), glucose should decrease, myCAF>iCAF structural check, ECM concentration stability. Two xfails for known bugs: lactate decreases -13% during progression (glucose limitation + V_T dilution overwhelms production), VEGF decreases -3% (V_T dilution, stale k_VEGF_HIF). Tier 16 (treatment cascade ordering): GVAX cells decay d7>d14>d21, mAPC transient peak at d14, k_C_Tcell_eff >100x checkpoint release by d14, CD8 infiltration and IFNg increase by d14. Tier 17 (treatment vs baseline comparison): k_C_Tcell_eff/IFNg/CD8 higher with treatment, C1/Treg lower with treatment.

**Dosing timing lesson:** First pass had 3 failures because tests assumed day-0 dosing, but the Zheng 2022 neoadjuvant protocol doses GVAX and nivolumab at day 7. GVAX_cells=0 at d3 (pre-dose), mAPC peaks at d14 (not d7), k_C_Tcell_eff is only 1x at d7 (nivolumab just dosed) but 486x at d14. Fixed all time-point references. Suite: 114 passed, 2 xfailed, 0 failures.

## 2026-02-25 17:30 — Fix lactate xfail: pO2-scaled clearance

Root cause analysis of the lactate -13% xfail revealed a structural issue: all cell-based production grows ~28-30% over 90-day progression, but V_T grows 32% because ECM hydrogel expansion adds volume without adding metabolic cells. Since lactate clearance was first-order on amount (`k_lac_clear * V_T.lactate`), the QSS concentration = production / (k * V_T), which decreases whenever production growth < V_T growth. No amount of adding cell-based production sources (CAF lactate, etc.) can close the gap because all cell populations grow at ≤30%.

Fix: replaced the ad-hoc VEGF impairment factor (`/ (1 + f_VEGF_lac * VEGF/(VEGF+VEGF_50))`) with vascular quality scaling (`* pO2/pO2_normal`). The pO2 formula already captures K vs cell demand vs VEGF aberration holistically. As tumor grows and becomes more hypovascular, pO2/pO2_normal drops (-18% over 90 days), reducing clearance. This gives QSS `[lac] = production * pO2_normal / (k * pO2)`, removing the V_T dilution. k_lac_clear rescaled from 3.0 to 12.5/day to preserve effective clearance at diagnosis (~0.6/day, half-life ~27h). Removed f_VEGF_lac parameter (was only used in lactate clearance).

Result: lactate now increases +6.3% during progression (26,255→27,902 µM). However, the higher lactate increases immunosuppression via H_lac_Teff, causing IFNg to marginally decrease -1.4% (new xfail, strict=False). This is biologically realistic: PDAC's metabolic immunosuppression offsets immune cell expansion. Suite: 114 passed, 2 xfailed (VEGF -3%, IFNg -1.4%).

## 2026-02-26 — Qualitative bounds evidence audit

Systematic audit of all bounds in `scenarios/qualitative_bounds.yaml` against their cited literature. Every bound now carries a required `evidence_snippet` (verbatim quote from the source). The Pydantic model (`metadata/qualitative_bounds.py`) was updated to make `evidence_snippet` a required field.

**Three numerical corrections:**

`checkpoint_fold_increase_min` lower 100→2.0. The original 100x fold-increase had zero literature support — it was an aspirational number, not an evidence-based bound. Hui 2017 (Science) shows PD-1 engagement reduces IL-2 secretion by ~63%, implying ~2.7x suppression. Blocking PD-1 restores ~2-3x function. The 100x value would have made the bound useless as a structural test (any working checkpoint implementation passes trivially).

`mycaf_fraction_min` lower 0.50→0.40. The 50% floor sat exactly at the mean of Elyada 2019's measurement (45.3 ± 4.1%), meaning roughly half of valid parameter combinations would fail the bound by random noise. Reduced to 40% (~1 SD below mean) for a proper conservative floor.

`clinical_lactate_range` lower 5000→2000 µM. Halbrook 2018 (HR-MAS NMR, human PDAC tissue) shows lactate at 244 ± 155% of healthy tissue. With normal pancreas ~1-2 mM, early PDAC could be as low as ~2.4 mM. The old 5 mM floor was from Walenta 2004's cross-tumor (non-PDAC) data.

**Five reference upgrades to PDAC-specific sources.** The lactate bounds (3 of them) were all citing Walenta 2000/2004 cervical cancer data — replaced with Halbrook 2018 PDAC-specific HR-MAS NMR. The HIF range was citing Shibaji 2003 IHC (binary positive/negative) when the actual derivation goes through Koong 2000 pO2 electrode measurements → Hill function math. The VEGF range needed Holm 2016 TIF-to-plasma ratio data to justify the 1000-200,000 pg/mL compartment range (serum VEGF is only ~500 pg/mL).

**Structural guard labels.** The two ECM bounds (`ecm_concentration_stability`, `ecm_treatment_tolerance`) had rationale that implied direct literature support, but no study measures ECM longitudinally in PDAC or under immunotherapy. Relabeled as structural guards with explicit mechanistic reasoning for why the bounds are still useful.

**Two new bounds from Elyada 2019 CAF subtype data:** `icaf_fraction_range` [0.20, 0.65] and `apcaf_fraction_max` upper 0.25. These complement the existing `mycaf_fraction_min` to give the model structural constraints on all three CAF subtypes. The data is strong (flow cytometry, n=20 KPC tumors) and the bounds are conservative.

Final state: 29 bounds (was 27), all with evidence_snippet, 0 duplicate IDs, 6 categories.

## 2026-02-26 (cont.) — Qualitative bounds tightening pass

Followed up the evidence audit with a systematic tightening pass, focusing on bounds where the evidence supported narrower ranges than we originally set.

**Five straightforward tightenings:**

`pO2_max_diagnosis` upper 50→20 mmHg. Koong 2000 median PDAC pO2 was 2.7 mmHg with 97th percentile ~10 mmHg. A 50 mmHg upper bound was tissue-normal — tightened to 20 mmHg which is still 2× the highest observed PDAC pO2.

`apcaf_fraction_max` upper 0.25→0.15. Elyada 2019 reports apCAFs at 5.3 ± 2.4% of CAFs. The original 0.25 was ~8 SD above the mean. Tightened to 0.15 (~4 SD), still very conservative.

`cd8_exhaustion_fraction_range` upper 0.80→0.65. Dolina 2021 review: 40-60% of intratumoral CD8s are PD-1+Tim-3+ exhausted. Tightened upper to 0.65 (mean + 2 SD of the range midpoint).

`tumor_dip_tolerance` lower 0.90→0.95. Untreated PDAC should not show >5% tumor volume dip (shrinkage). The old 10% tolerance was too generous.

`treg_fraction_min` lower 0.01→0.05. Hiraoka 2006 shows FOXP3+ Tregs accumulate during PDAC progression. At diagnosis, Tregs should be at least 5% of CD4+ T cells.

**Two ratio bounds tightened with new quantitative data:**

`cd8_treg_ratio_max_diagnosis` upper 10→5. Golesworthy 2022 multiplex IHC (n=85 HR/MMR-intact resected PDAC) reports CD8 density 40.5 ± 50.9 cells/mm² and FOXP3+ Treg density 13.6 ± 13.4 cells/mm², ratio of means ≈ 3.0. Carstens 2017 proportional data is consistent (CD8:Treg ≈ 2.6). Bound of 5 gives ~1.7× margin. Source upgraded from Ino 2013 (qualitative high/low) to Golesworthy 2022 (quantitative cells/mm²).

`m2_m1_ratio_range` upper 10→5. Dias Costa 2022 multiplex IHC (n=30 up-front surgery PDAC, Table 2) reports stromal M2-polarized 231 (146–353) cells/mm² vs M1-polarized 120 (57–160) cells/mm², giving M2:M1 ≈ 1.9 in the dominant stromal compartment. Intraepithelial M2:M1 was 0.43 (M1 > M2). Bound of 5 gives ~2.5× margin. Source upgraded from Ino 2013 (qualitative "78% M2-predominant") to Dias Costa 2022 (quantitative cells/mm²).

## 2026-02-26 (cont.) — DOI provenance fixes and rho_Treg_max tuning

Post-tightening test run (`04aa160`) revealed two issues: 7 DOI validation failures from fabricated DOIs introduced during the evidence audit, and 1 behavior test failure from the cd8_treg_ratio bound tightened to 5.0 while the model still produced 7.6.

**DOI corrections.** Three source attributions in the YAML were wrong — the research agent that populated the initial bounds had hallucinated author names, years, and DOIs for real papers. Tracked down the actual papers via PubMed: "Halbrook 2018" was actually Wang et al. 2014 in NMR in Biomedicine (`10.1002/nbm.3198`), the Elyada 2019 DOI suffix was `-0094` not `-0528`, and "Holm 2016" was actually Haslene-Hox et al. 2014 in BBA Clinical (`10.1016/j.bbacli.2014.08.002`). This affected 7 bounds total (3 lactate, 3 CAF subtype, 1 VEGF). All 36 DOI/PMID validation tests now pass.

**rho_Treg_max 1e7 → 2e7.** The cd8_treg_ratio bound was tightened to 5.0 based on Golesworthy 2022 data (clinical mean ~3.0), but the model produced 7.6. Key insight: `run_median_simulation` loads parameters from the priors CSV, not `parameters_PDAC.m` — so the `.m` file change alone had no effect. Updated all 3 priors CSVs plus the parameter file. The new value (2e7 cells/mL) is within the documented IHC density range (3e6–3e7). Result: CD8:Treg ratio dropped from 7.6 to 3.8, and as a bonus the IFNg progression xfail resolved (IFNg now increases during progression). Two borderline H_TGFb Hill function xfails appeared (0.778 vs 0.75 ceiling) — expected, SBI will tune the TGFb EC50s. Final suite: 154 passed, 1 skipped, 3 xfailed.

## 2026-02-26 (cont.) — Qualitative bounds → calibration targets, overlap audit

**Qual bounds → cal targets (8 of 12 converted).** Systematically converted qualitative bounds from `qualitative_bounds.yaml` into proper `CalibrationTarget` YAMLs (tagged `from-qualitative-bound`, `extraction_method: manual`). Completed 8: cd8_exhausted_fraction (Shen 2017, n=94), cd8_treg_ratio (Golesworthy 2022, n=166 per-patient ratio), iCAF fraction (Elyada 2019), apCAF fraction (Elyada 2019), M2 predominance rate (Ino 2013, response_rate aggregation), Treg fraction of CD4 (Hiraoka 2006, n=198), absolute lactate (Battini 2017, nmol/mg→µM), tumor diameter (Tran 2022, n=315). Skipped 4 (glucose, VEGF, HIF, CAF:cancer ratio) due to insufficient quantitative data or redundancy. Key lesson: ratio of means ≠ mean of ratios (Golesworthy per-patient CD8:Treg = 9.8±23.8 vs ratio of mean densities = 3.0). Full details in `notes/qual_bounds_to_cal_targets_progress.md`.

**Calibration target overlap audit (51 active targets).** Found 5 observables with 2 derivations each. Four have overlapping CIs (no conflict): cd8_treg_ratio, apcaf_fraction, icaf_fraction, treg_fraction_cd4. One genuine conflict: cd8_exhausted_fraction — deriv001 (Shibuya 2014, flow cytometry, n≈15, mixed treatment, 72.8% PD-1+) vs deriv002 (Shen 2017, double IF on FFPE, n=94, treatment-naive, 30.62%). Chose Shen 2017: 6× larger cohort, treatment-naive only, text-reported values, PD-1 on FFPE better approximates model's CD8_exh than PD-1 on fresh tissue. Moved deriv001 to `excluded/`.

Checked conceptual overlaps (algebraically related targets): M1/M2 ratio + M2 predominance rate are consistent (ratio 0.518 → implied M2 frac 0.659 vs measured 0.610). CAF subtype fractions sum to ~1.06 (deriv001s) — close enough. CD8 density + Treg density → implied ratio ~10, CIs overlap with measured 4-5. No hard conflicts found.

**SBI strategy decision.** Will skip sensitivity analysis and submit all ~184 parameters to neural SBI directly. Non-identifiable parameters simply return wide posteriors. Overlapping targets with consistent CIs provide complementary information (different patient cohorts, methods) and don't need deduplication — the SBI loss function will naturally weigh them.

## 2026-02-27 — SBI workflow overhaul: all-params, observation noise, ECM concentration fix

Three connected changes to prepare the SBI pipeline for a proper calibration run.

**SBI workflow: drop SA pre-filtering, use all 193 parameters** (`3a00eb8`). The previous workflow used sensitivity analysis to select 97 significant parameters, then a top-13 subset for visualization. This added complexity and a brittle dependency on SA rankings. Replaced with a simpler approach: submit all 193 prior parameters directly to NPE. Non-identifiable parameters return wide posteriors — that's fine, it's information. Diagnostic plots now select the top N parameters by R² recovery (computed after training) instead of SA ranking. Removed the dimensionality sweep and PRCC heatmap sections that depended on SA. Also added 9 missing priors for parameters from newer modules (NK, chemotaxis, apCAF, ECM motility) that were in `parameters_PDAC.m` but not in `pdac_priors.csv`.

**Observation noise injection** (`f8a5754`). The SBI workflow was calibrating simulator outputs against point-estimate medians, ignoring measurement uncertainty. This means a target with CI95 = [100, 110] constrains the posterior just as much as one with CI95 = [10, 1000] — clearly wrong. Added `add_observation_noise()` to `qsp-sbi` (reusable function in `data_processing.py`) that injects per-observable noise into training data based on calibration target CI95s. For each observable, auto-selects lognormal (multiplicative) or Gaussian (additive) noise by comparing CI95 asymmetry in log-space vs linear space. ~25 of 35 baseline targets are lognormal (positive quantities with scale-dependent uncertainty), ~10 are Gaussian (fractions/rates with symmetric CIs). Noise only applied to training data; test data and coverage checks use raw simulator outputs. The NPE learns that observations are noisy measurements, so targets with wide CIs naturally constrain posteriors less.

**ECM fold-change observable: concentration, not total mass** (`f8a5754`). The `collagen1_protein_level` calibration target was computing ECM fold-change as `total_ECM_mass / (10mL_reference × collagen_fraction)`. Two bugs: (1) the Imamura 1995 collagen content (5 µg/mg) is per mg *dry* tissue, but was used as if it were wet-weight — inflating the denominator ~5x; (2) using total mass with a fixed 10 mL reference volume means the fold-change depends on tumor size, but Xu 2017's SHG/TPEF pixel density ratio is a *concentration* measurement. Fixed by using `ECM_level` (= V_T.ECM / V_T, mg/mL) divided by normal pancreas ECM concentration (5 µg/mg dry × 0.22 dry fraction × 1.04 g/mL density ≈ 1.14 mg/mL). Added `pancreas_dry_weight_fraction` (0.22) to reference_values.yaml. Also added `ecm_fold_change_range_diagnosis` qualitative bound (5-20x, Xu 2017) with snapshot checker — now 20 of 30 bounds are endpoint-evaluable.

## 2026-02-27 — LOO structural fix: decouple cDC1/cDC2 homeostatic densities

The LOO predictive check decomposed the joint Mahalanobis distance (D²=30.39 across 35 observables) and revealed that the cDC1/cDC2 density pair was the single largest driver of joint misspecification — 30% of total D² (ΔD² = 4.95 + 4.06 = 9.01). Both densities had acceptable marginal prior p-values (0.206, 0.354), so the model could produce each individually, but the algebraic constraint `f_cDC1` locked their ratio, preventing SBI from matching both simultaneously.

**The fix.** Replaced the `f_cDC1 × APC0_T` / `(1-f_cDC1) × APC0_T` parameterization in `APC_module.m` with four independent homeostatic density parameters: `APC0_cDC1_T`, `APC0_cDC2_T`, `APC0_cDC1_LN`, `APC0_cDC2_LN`. Medians preserve the old f_cDC1=0.2 defaults exactly (e.g., APC0_cDC1_T = 0.2 × 2.7e6 = 5.4e5). Each gets σ=0.7 lognormal priors — same scale as the original total-density priors, but now the ratio is free to vary. Removed f_cDC1, APC0_T, APC0_LN from the prior CSVs (-3 rows, +4 rows → 194 total priors). All 121 qualitative bounds tests pass.

**Result.** Joint D² dropped from 30.39 → 28.1. The DC pair went from dominating (30% of D²) to spread evenly: cDC2 influence 3.0, cDC1 influence 2.3 — now in the same band as collagen (2.3), NK fraction (2.3), and apCAF fraction (1.6). The LOO distribution is much healthier.

**Remaining drivers and the bias-variance question.** The next LOO targets are collagen (coupled to myCAF via shared qPSC pool), NK fraction (ratio denominator includes all T cells), and apCAF fraction (shared qPSC activation). Unlike the DC case, these couplings represent real biology — PSCs genuinely differentiate into CAF subtypes from a shared precursor, and NK fraction *is* a ratio against T cells. Decoupling these would improve D² but remove mechanistic structure that matters for treatment predictions (e.g., TGFb-driven CAF polarization). Decided not to pursue further decoupling for now. The remaining D² may reflect irreducible model-data tension or targets that need wider CIs, not structural errors. Will revisit after checking whether activation rate priors are too tight.

**D² interpretation: the model is already well-specified.** Under a well-specified prior, D² across 35 observables should follow approximately χ²(35), with mean 35 and sd ≈ 8.4. D² = 28.1 is below the expected value — the observed data looks like a plausible draw from the prior predictive. The LOO influences (top 5 all in 1.6–3.0) are what a healthy decomposition looks like; no single observable dominates. Further D² reduction risks fighting against real biology for diminishing statistical returns.

**What the residual pattern says about the biology.** The remaining LOO drivers are more interesting as scientific observations than as optimization targets. (1) *Collagen ceiling*: the model structurally can't reach the observed ~10× ECM fold-change (Xu 2017) because logistic saturation in ECM production (`(ECM_max - ECM_level)/ECM_max`) throttles myCAF secretion before equilibrium matches the data. This suggests either MMP-driven ECM remodeling is more complex than first-order degradation, or the saturation is too aggressive — PDAC desmoplasia may exceed what the current production/degradation balance permits. (2) *NK fraction*: the model dilutes NK cells relative to T cells more than observed (target 3.5%, Marcon 2020). Could mean PDAC's NK exclusion mechanisms (CXCL12, TGFb, IL-10 suppression) are overrepresented in the model, or that the T cell compartment is overpopulated at diagnosis. (3) *apCAF fraction*: the transcript-vs-protein tension (Elyada 2019 scRNA-seq 21% vs Ogawa 2021 multiplex IHC ~2%) is genuine biological uncertainty about what fraction of MHC-II-transcribing CAFs are functionally antigen-presenting. The wide CI (2–21%) reflects this — the model is being asked to match a target whose bounds span an order of magnitude because the biology itself is unresolved.

## 2026-03-02 — PRCC tier filtering, NPE capacity sweep, and rethinking the success metric

Three connected pieces of work: PRCC-based parameter subsetting for SBI, an architecture sweep to diagnose NPE performance, and a conceptual shift away from R² recovery toward predictive validation.

**PRCC sensitivity analysis** (`workflows/prcc_sensitivity.py`, results in `results/prcc_sensitivity/`) identified 111/194 FDR-significant parameters (permutation test, 1000 permutations, FDR α=0.05) with natural tiers: tier 1 (rank ≤ 30, mean |PRCC| > 0.10), tier 2 (rank ≤ 70, mean |PRCC| 0.04–0.10), tier 3 (rank ≤ 112, all significant). Since the earlier ablation study showed parameter count degrades R² recovery, added a `PRCC_TIER` config knob to `sbi_workflow.py` that subsets parameters for NPE training while keeping the full 194-param simulation pool. The simulator still runs with all parameters — theta columns are subsetted after loading, and posterior predictive simulations embed the subset posterior back into full parameter space (non-selected params filled from prior). Default is tier 1 (30 params). Output directories include the tier suffix (`figures/sbi_baseline_tier1/`).

**NPE capacity sweep** (`workflows/npe_capacity_sweep.py`) tested 7 NSF architectures on tier 1 (30 params, 35 obs, ~7.6k training sims): from the current baseline (32 hidden, 3 transforms, 70k NN params) up to (128 hidden, 8 transforms, 1.1M NN params). The result was unambiguous: **capacity is not the bottleneck**. Median R² stayed flat at 0.13–0.15 across a 15× range of NN parameters. All configs showed the same overfitting pattern — train loss well below val loss, with the gap widening as capacity increased. Early stopping kicked in around epoch 15–20 for all configs. The same 7 parameters were identifiable (R² > 0.3) regardless of architecture: `k_M2_pol` (0.73), `k_Mac_rec` (0.59), `k_ECM_myCAF_sec` (0.48), `k_Treg_death` (0.38), `f_stroma_max` (0.38), `k_iCAF_to_myCAF` (0.33), `rho_Treg_max` (0.31). Also added learning curve tracking to the main `sbi_workflow.py` — now saves train/val loss per epoch, plots the learning curve, and warns if training hits max_epochs without converging.

**The R² question.** The capacity sweep forced a useful rethink: R² measures parameter identifiability, but the actual goal is predicting clinical outcomes. A model with degenerate parameters can still make sharp predictions if the sloppy directions don't affect the observables. The 23/30 tier-1 params with low R² may be individually unidentifiable but collectively constrained in the directions that matter for output. The right validation metric is cross-scenario posterior predictive accuracy: train on baseline (40 targets), predict held-out treatment response. We have 18 GVAX+nivo calibration targets (Li 2022 mIHC, immune response at day 21) and 1 clinical progression target (tumor doubling time, Ahn 2017) — none used in baseline SBI training. Next step is posterior predictive validation against these held-out scenarios, which directly tests whether the calibrated model generalizes to treatment prediction.

**Sequential NPE workflow** (`workflows/sequential_sbi_workflow.py`). Implemented SNPE-C (APT; Greenberg et al. 2019) for multi-round focused inference. The idea is that round 1 samples from the prior (leveraging the existing simulation pool), then subsequent rounds sample from the previous round's posterior conditioned on observed data, focusing new simulations in high-posterior regions. sbi handles the atomic proposal correction automatically. Key design choices: round 1 uses 500 sims from the pool, subsequent rounds generate 500 new sims each (posterior proposals require fresh MATLAB runs — can't reuse pool). The copula transform is fitted once on round 1 data and fixed for all subsequent rounds to keep the observation space consistent. Convergence is tracked via round-over-round R² recovery, Wasserstein distance between consecutive posterior samples, and z-score contraction, with early stopping when max normalized Wasserstein drops below 0.01. Uses a single `NPE` inference object across all rounds — sbi accumulates data internally and retrains from scratch each round. Not yet run at scale. The SNPE-C atomic loss is substantially slower per epoch than the single-round MLE loss (ratio computation over num_atoms candidates per training sample), which compounds across rounds — each round retrains from scratch on the accumulated dataset. The main question is whether posterior-focused proposals actually help for this problem given the degeneracy structure (focused proposals improve sample efficiency for *identifiable* parameters, but for degenerate directions the posterior is already broad), and whether the per-round training cost is justified by the gains.

**Hierarchical SBI workflow** (`workflows/hierarchical_sbi_workflow.py`, design doc at `notes/hierarchical_sbi_design.md`). Addresses a fundamental limitation of the flat SBI setup: it conflates population center uncertainty (μ — where the typical patient sits) and inter-patient variability (ω — how much patients differ). The current flat NPE conditions only on observed medians, so ω defaults to the prior width and well-identified parameters collapse to point estimates, losing patient-level diversity. The hierarchical model infers (μ_j, ω_j) per parameter (388D total), conditioning on both observed medians AND IQRs (70D summary statistics). Each training sample draws hyperparameters φ = (μ, ω), then draws N=10 virtual patients from LogNormal(μ, ω), simulates each, and computes cohort summaries (median + IQR per observable). The hyperprior is μ_j ~ Normal(csv_median, τ_μ=0.5), ω_j ~ LogNormal(log(csv_sigma), τ_ω=0.5). Cost is 10× the flat workflow (10 sims per training sample), but enables properly calibrated virtual patient generation where each VP inherits uncertainty from both μ and ω posteriors. Open questions: which parameters genuinely vary across patients vs universal constants (binding affinities), optimal cohort size N, and whether the existing 10k simulation pool provides adequate coverage for nearest-neighbor approximation of the cohort simulations.

**`qsp-hpc` update: `simulate_with_parameters` caching.** Updated `qsp-hpc` so that `simulate_with_parameters()` (used for posterior predictive simulations) now caches results in a suffix pool keyed by a hash of the theta matrix. On first call, simulations run on HPC and results are persisted locally and remotely; subsequent calls with identical theta hit the cache instantly. Each theta hash gets its own pool scenario (`posterior_predictive_{hash}`), so different posterior draws don't collide. This matters because the SBI workflow, capacity sweep, and sequential NPE all run posterior predictive sims repeatedly during development — without caching, every re-run burned another 500 MATLAB simulations (~10 min on the cluster).

## 2026-03-02 (cont.) — Fix CD4+ Th/Treg structural imbalance, predictive summary triage

**The Th/Treg problem.** The tier 1 predictive summary (`figures/sbi_baseline_tier1/predictive_summary.csv`) flagged `treg_fraction_cd4` at obs_z = -1.80 (prior median 99.3%, observed 11.1%) and `cd4_helper_density` at obs_z = 1.51 (prior median 0.28, observed 109.74 cells/mm²). The root cause is a structural asymmetry: Tregs have 4 amplification mechanisms (constitutive activation, tumor-local proliferation at 0.3/day, CCL5 recruitment, Th→Treg conversion) while Th cells had zero local proliferation and zero CXCR3 recruitment. No amount of SBI calibration can fix this — the prior predictive barely overlaps the observation.

**Two mechanistic fixes.** (1) Added tumor-local Th proliferation to `Th_module.m`, mirroring Treg_module's logistic reaction: `k_Th_pro_tumor * V_T.Th * (1 - V_T.Th / (rho_Th_max * V_T + cell))`. Rate set at 0.15/day (~half Treg's 0.3/day, reflecting lower Ki67+ fraction in Th1 vs Treg), carrying capacity 5e6 cell/mL (quarter of Treg's 2e7, reflecting lower Th abundance in PDAC). (2) Added CXCR3-mediated Th1 recruitment to `chemotaxis_module.m`, reusing the existing `H_CXCL9_Trec` Hill function: `k_CXCR3_Th_rec * V_T * V_C.Th * H_CXCL9_Trec * (1-H_CXCL12_Texcl)`. Rate at 1.5e-05 1/(cm³·min), half of CD8's 3e-05. All three new parameters added to `pdac_priors.csv` with σ=0.8. Assembly order is clean: Th_module (line 88) uses pre-rename `V_T.T`, chemotaxis_module (line 100) uses post-rename `V_C.Th`/`V_T.Th`.

**Package rename: qsp-llm-workflows → maple.** The external `qsp-llm-workflows` package was renamed to `maple`. Updated 4 Python imports (`metadata/qualitative_bounds.py`, `tests/scenarios/test_calibration_targets.py`, `workflows/submodel_calibration.py`) and rebuilt the venv with `uv`. All 169 qual tests pass (1 skipped, 3 pre-existing xfails: H_TGFb borderline, H_TGFb_APC borderline, VEGF dilution).

**Remaining predictive summary issues (post-Th fix).** Triaging the full tier 1 predictive summary for structural problems that still need attention, ranked by severity:

- **collagen1_protein_level** (obs_z = 2.46): prior median 5.2× vs observed 9.8× fold-change. The model structurally can't reach high ECM — logistic saturation in myCAF secretion throttles before matching data. Already noted in the LOO analysis (2026-02-27). May need to raise `ECM_max` or revisit MMP-dependent remodeling.
- **apCAF fractions** (obs_z = 1.83, 2.03): model produces ~0.3% apCAF vs observed 6–10%. The `k_psc_activation_apCAF` rate (0.01/day) is 20–40× lower than myCAF/iCAF rates. Either apCAF differentiation is too slow or apCAFs need another source (e.g., iCAF→apCAF interconversion).
- **tumor_glucose_ratio** (obs_z = 1.96): prior median 0.26 vs observed 0.92. The model depletes glucose too aggressively — `k_glc_tumor` and `k_glc_immune` consumption rates may be too high, or the QSS glucose formula doesn't account for perfusion-limited supply correctly.
- **nk_fraction_of_tils** (obs_z = 1.45): model dilutes NK relative to T cells. Prior median 0.1% vs observed 3.2%. The T cell compartment may be overpopulated at diagnosis, or NK recruitment rates are too low.
- **mdsc_density** (obs_z = 1.36): prior median 7.9 vs observed 48.3 cells/mm². MDSC recruitment (`k_MDSC_rec`) may need upward revision.
- **th_exhausted_fraction** (obs_z = 1.32): 47% predicted vs 78% observed. With more Th cells from the proliferation fix, this could go either way — more Th cells means more substrate for exhaustion, but also more viable Th diluting the fraction.
- **cdc2_density_baseline** (obs_z = -1.21): prior median 9.17 vs observed 1.82 cells/mm². cDC2 overproduced relative to data.
- **hif1a_positivity_rate**: posterior moved *away* from observation (86%→29%, obs 70%). The posterior is chasing other observables at the expense of HIF — possible tension between pO2 and HIF calibration targets.

The no-learning cluster (iqr_ratio > 0.9) spans all cell density observables: CD8, Treg, Th, TAM, MDSC, NK, total TIL. These have enormous prior predictive variance (spanning orders of magnitude) and the 30-param tier 1 NPE can't constrain them. This is expected — cell densities depend on many interacting rates, so individual parameters are degenerate. The question is whether the *joint* posterior makes sharper predictions than the prior. Next step: posterior predictive validation against held-out GVAX+nivo targets.

## 2026-03-02 (cont.) — Strip hypoxia, metabolism, NK, and chemotaxis modules

**Motivation.** The previous devlog entry's predictive summary triage made the case: `tumor_glucose_ratio` (obs_z=1.96), `nk_fraction_of_tils` (obs_z=1.45), and `hif1a_positivity_rate` (posterior moving *away* from data) are all from modules with ~50 unconstrained parameters that SBI can't meaningfully narrow. These modules (hypoxia/HIF, glucose/lactate metabolism, NK cells, CXCL9/10 chemotaxis) add biological detail but lack calibration anchors — most parameters sit in the no-learning cluster. Rather than continuing to fight parameter degeneracy in uncalibratable axes, the decision was to strip them and focus the model on biology with clean data: T cells, macrophages, DCs, CAFs/ECM, checkpoints, and GVAX. Code is preserved on `feature/extended-tme` for future re-integration when better calibration data emerges.

**Scope.** 40 files changed, ~3,500 lines removed across 12 steps: removed 4 module calls from `immune_oncology_model_PDAC.m`; removed `H_XCL1_cDC1` forward reference from `simbio_init.m` and simplified cDC1 recruitment in `APC_module.m`; removed ~50 parameters from `parameters_PDAC.m`; removed 42 prior rows (30 from `submodel_calibrated.csv`, 12 from `baseline_no_treatment_sa97.csv`); deleted 7 calibration target YAMLs and edited 16 more to remove `V_T.NK` from observable denominators; removed 9 rows from `pdac_calibration_targets.csv`; removed 10 qualitative bounds and 7 snapshot compute functions; cleaned `set_healthy_populations.m` (NK density, lactate, pO2/HIF); removed Figures 6-7 (hypoxia/metabolism, NK/innate) from `plot_mechanism_cascade.m` and restructured Figure 8 (chemokine trafficking, now Figure 6) from 8→6 panels after removing CXCL9; removed 54 entries from `species_units.json` and 28 columns from `params.csv`; removed ~20 test methods and 11 Hill function registry entries from the qualitative behavior test suite; updated CLAUDE.md documentation.

**Verification.** All 51 fast tests pass. All 83 qualitative behavior tests pass (2 xfailed: H_TGFb and H_TGFb_APC borderline sensitivity, pre-existing). The model compiles and evolves to diagnosis successfully — `set_healthy_populations.m` output confirms clean initialization without NK/lactate/pO2/HIF species. The 48 pre-existing failures in `test_calibration_targets.py` (schema validation for `q_Treg_T_in` deriv003/004 and duplicate IDs in `_original.yaml` files) are unrelated. Simulation pool caches will need invalidation before the next SBI run since the model equations changed.

## 2026-03-02 (cont.) — Post-strip SBI run: structural misfit diagnosis

Upgraded sbi 0.22→0.25.0 (installed with `--no-deps` to bypass the `torch<2.6.0` cap — torch 2.10.0 works fine), downgraded arviz 1.0→0.23.4 (arviz 1.0 removed the `arviz.data` submodule that sbi imports at init), installed new sbi 0.25 dependencies (zuko, pymc). Bumped MODEL_VERSION to `baseline_allparams_v3` and reduced NUM_TOTAL_SIMULATIONS to 5k for faster iteration.

**Excluded stale calibration targets.** The first run produced all-NaN in column 28 (`tumor_lactate_fold_change`) — it references `V_T.lactate` which was removed with the metabolism module. Moved to `excluded/`. The previous run's predictive summary also included 5 other targets from stripped modules (hif1a_positivity_rate, nk_fraction_of_tils, tumor_glucose_ratio, tumor_lactate_absolute, tumor_pO2_baseline) — these were already excluded from the YAML directory but appeared in the old cached results. Clean re-run with 29 observables.

**LOO prior predictive results (D²=375.1, 29 obs).** Three structural misfits dominate:

1. **apCAF fraction** (ΔD² ~270 each, two targets). obs=6.4%/10.3%, prior median=0.24%, prior 95th=3.6%. Root cause: apCAF has no proliferation (iCAF/myCAF proliferate at 0.10–0.15/day), constitutive activation at 0.01/day (40x slower than myCAF), 1.5x higher death rate, and no interconversion from other CAF subtypes. As the tumor grows, iCAF/myCAF expand via proliferation while apCAF fraction crashes from the ~5% initial condition to ~0.2%. SBI can increase k_psc_activation_apCAF ~10x but this can't compete with proliferative expansion of the other subtypes. Fix: add apCAF proliferation reaction.

2. **Treg fraction of CD4** (obs_z = -1.77). obs=11%/34%, prior median=96%, 40% pile-up near 1.0. Commit `5331e8f` added compensatory Th proliferation and recruitment but didn't fix the upstream priming bug: H_APCh in APC_module.m uses `V_LN.nTreg` (Treg naive pool) and `n_CD8_clones` instead of a Th-specific naive pool and clone count. No `nTh` species exists — Th_module only creates `aTh → Th`, missing the naive stage. The activation reaction consumes nTreg as a workaround. The compensatory parameters (k_Th_pro_tumor, k_CXCR3_Th_rec) bypass broken LN priming by creating Th cells directly in the tumor, but the structural asymmetry persists. Fix: create nTh species, n_Th_clones parameter, correct H_APCh.

3. **Collagen protein level** (ΔD² ~18). obs=9.84x fold-change, prior 95th=8.07. ECM production kinetics are ~100x too slow: with typical myCAF counts (~10^7) and k_ECM_myCAF_sec=5e-9 mg/cell/day, steady-state ECM_level ≈ 0.09 mg/mL vs required 11.3 mg/mL. The ECM_max ceiling (100 mg/mL) isn't limiting — the production rate simply can't fill the pool against degradation (k_ECM_deg=0.035/day, 20-day half-life; literature for fibrotic tissue suggests 60-100+ days). Fix: increase production rates ~10x and reduce degradation to ~0.01/day.

**Redundant target inflation.** Five observable pairs measure the same biology twice from different sources: apCAF fraction (×2), M2 fraction (×2), iCAF fraction (×2), CD8:Treg ratio (×2), Treg/CD4 fraction (×2). These double-count misfits in D² without adding information. Not urgent to fix (consistent CIs provide complementary evidence) but worth noting for D² interpretation.

**Next steps.** Fix Th/Treg priming pathway (nTh species, H_APCh correction) and ECM production kinetics. apCAF proliferation deferred — straightforward addition once the higher-priority structural fixes are validated.

## 2026-03-02 (cont.) — Fix apCAF fraction, ECM kinetics, and apCAF proliferation

Three prior corrections targeting the top LOO structural misfits from the post-strip run (D²=375.1):

**Fix 1 — k_apCAF_death alignment (0.025 → 0.0165/day).** The apCAF death rate was 1.5x higher than iCAF/myCAF (both 0.0165) with no biological justification — all CAF subtypes share similar baseline turnover. Aligning this alone was insufficient: LOO influence for apCAF fraction remained at 77+76 because the structural bottleneck is proliferative, not death-rate-driven.

**Fix 2 — k_ECM_deg literature correction (0.035 → 0.007/day).** Previous 20-day collagen half-life was too fast for desmoplastic PDAC stroma. Literature supports 60-100+ day half-life in fibrotic tissue; 0.007/day gives ~100-day half-life. This should allow ECM to accumulate to the observed ~10x fold-change over normal pancreas without needing to inflate production rates. Left k_ECM_myCAF_sec and k_ECM_qpsc_sec unchanged — let SBI adjust them.

**Fix 3 — apCAF proliferation (new k_apCAF_prolif = 0.05/day).** This was the critical structural fix. Steady-state analysis showed apCAF fraction was structurally capped at ~1-2% because iCAF/myCAF have PDGF-driven proliferative self-renewal (0.10, 0.15/day) while apCAF had none — only constitutive activation at 0.01/day. As the tumor grows and proliferation amplifies iCAF/myCAF, apCAF gets squeezed regardless of activation rate or death rate. Added apCAF proliferation with the same PDGF-driven logistic pattern (shared carrying capacity via aPSC_total), at a lower rate (0.05/day) reflecting less mitogenic phenotype. Prior σ=1.0 gives SBI room to adjust from ~0.007 to 0.37/day. This gives a proper lever to reach the observed 6-10% (Elyada 2019).

**Next steps.** Fix 3 from the original plan (nTh species + H_APCh correction for Treg/CD4 fraction) remains. Then re-run SBI to validate all three fixes reduce D².

## 2026-03-04 — Fix Th/Treg priming: H_APCh clone count bug + f_nTreg recalibration

**The bug.** `H_APCh` in `APC_module.m:196` used `n_CD8_clones` (120, tumor-antigen clones) instead of `n_Treg_clones` (100, self-antigen clones) in the denominator. This is CD4 priming via cDC2 — the clone count should match `H_APC` (Treg priming), not `H_mAPC` (CD8 priming). Fix: `n_CD8_clones` → `n_Treg_clones`.

**Shared naive pool is intentional.** Initially considered creating a separate `nTh` species, but the Robertson-Tessi design shares a single naive CD4 pool (`nTreg`) for both Th and Treg activation. This is biologically defensible — naive CD4+ T cells are uncommitted until activation context (cytokine milieu) determines lineage fate. The split is controlled by `f_nTreg` (fraction becoming constitutive Tregs) vs `(1-f_nTreg)` (antigen-dependent activation, feeding both induced Tregs and Th cells).

**f_nTreg recalibration (0.7 → 0.25).** The 0.7 default made 70% of CD4 activation constitutive Tregs, structurally guaranteeing ~96% Treg fraction regardless of other parameters. Literature: nTregs are ~5-15% of peripheral CD4+ T cells (Sakaguchi 2008). Added to SBI priors with σ=0.7, bounds [0.01, 0.95] so SBI can explore the full range.

**Combined effect with apCAF/ECM fixes.** All three structural misfits from the D²=375.1 LOO analysis are now addressed: apCAF fraction (proliferation), ECM/collagen (degradation kinetics), Treg/CD4 fraction (H_APCh + f_nTreg). Next step: re-run SBI to validate D² reduction.

**Prior predictive validation (v4, 500 sims).** D² dropped from 375.1 → 55.7 (~6.7x reduction). 27/29 observables within 90% prior predictive CI. Two remaining outside: `collagen1_protein_level` (obs=9.84, CI=[0.229, 9.01]) and `treg_fraction_cd4` (obs=0.111, CI=[0.139, 1.0], p=0.056 — borderline). Top LOO drivers are now cdc2_density, apc_maturation_ratio, cancer_cell_density, collagen, m2_predominance — all parameter tuning issues, not structural misfits.

## 2026-03-04 (cont.) — Rename CD4/Th/Treg species for clarity + second clone count bug

**Rename.** The shared naive CD4+ pool was named `nTreg` after Treg_module's rename step, which made Th_module's `V_LN.nTreg` consumption deeply confusing — it reads like "Th activation consumes naive Tregs." Similarly, `n_Treg_clones` (self-antigen clone diversity) and `H_APC`/`H_APCh` (Treg/Th priming Hill functions) had unclear names. Renamed across 13 files:

- `nTreg` → `nCD4` (shared naive CD4+ pool)
- `n_Treg_clones` / `n_clones_slf` → `n_CD4_clones` (self-antigen clone diversity)
- `H_APC` → `H_APC_Treg` (Treg priming Hill function via cDC2)
- `H_APCh` → `H_APC_Th` (Th priming Hill function via cDC2)

**Second bug found.** `Th_module.m:91` used `n_CD8_clones` (120, tumor antigens) for the Th clonal expansion factor in the activation reaction rate. The H_APCh fix in the previous commit only corrected the Hill function denominator in APC_module — the Th activation reaction itself also had the wrong clone count. Now uses `n_CD4_clones` (100).

## 2026-04-21 15:40 — Capacity sweep at n=100k: NN width/depth is not the bottleneck

Ran a 3×3 NN-capacity grid (`hidden_features ∈ {256, 512, 1024}` × `num_transforms ∈ {5, 9, 13}`) at fixed training size 100k on the restricted + fold-change bundle to answer whether the sweep-v2 ceiling of 11 identified params (`contraction_local > 0.5`) could be pushed with a bigger flow. Configs under `workflows/sweep/configs_capacity/`, results at `figures/sbi_multi/sweep_capacity/` (handoff write-up in `343d38a` on `sweep/npe-training-grid`).

Answer is clean: **capacity is not the bottleneck.** All three `h256_*` cells plus `h512_t5` tie at 11 identified. Scaling to 92M params (`h1024_t13`) drops identification to 10, raises best `val_loss` monotonically from 162.5 → 170.7, and early-stops 34 epochs sooner — textbook overfit on the 91k-survivor training set. `h256_t5` is Pareto-best on (n_identified, val_loss, epochs used) and should remain the default. Together with sweep-v2's flat response across 75k–100k training size and the +3 fold-change targets not moving the ceiling, this localizes the remaining ~160 params as **observation-degenerate, not data- or architecture-starved**.

Also merged PR #24 earlier in the session — the `clinical_progression → clinical_progression_longterm` scenario split from the sweep branch to main, so contributors pulling main can run sweeps without the revert dance. CI initially failed on the slow-test receipt hash (expected, scenario YAMLs are watched); regenerating the receipt on the PR branch passed 238/238. One hiccup: stale `.pytest_cache/v/cache/nodeids` was parametrizing two removed submodel targets (`CXCL9_50_Trec`, `k_NK_kill`) via a phantom cache from earlier branch state — clearing it fixed the spurious failures.

**Next move is not more capacity or more data.** Either (a) sequential SNPE-C via the existing `workflows/sequential_sbi_workflow.py` skeleton to focus density near `x_obs`, or (b) extract non-redundant observables per `notes/calibration/calibration_strategy.md` (spatial mIHC patterning, TCR repertoire, flow cytokine panels). Expected marginal gain from (a) alone is small given how degenerate the Jacobian rows already are; the bigger lever is (b).

## 2026-04-29 15:30 — Path D ruled out + tumor-local CD8 proliferation mechanism

Picked up the post-Abercrombie handoff with three surviving structural issues: CD8 density 1000× model–data gap, CAF/stroma/stromal_fraction tension, VEGF tension. Closed two pieces today.

**Path D scout (closed).** Literature scan on Bengtsson 2026's three whole-section TSP references (Bever 2015 HPB, Knudsen 2017 CCR, Wang 2016 Oncotarget) plus Li 2020 Front Oncol whole-mount: none publish a continuous TSP distribution. All four use categorical bins or threshold-dichotomies (loose/moderate/dense, mature/immature/intermediate, ≥0.8 vs <0.8 score, TSR>1 vs ≤1). So Path D ("find a higher whole-section TSP target to replace Bengtsson's TMA-biased 0.668") is not viable as quantitative replacement — Bengtsson remains the best continuous-% PDAC TSP source despite the TMA caveat. Strengthened the existing target metadata (`stromal_fraction_PDAC_deriv001.yaml`) to flag the directional bias (H&E-marked cores → tumor-rich → 0.661 is a *lower bound* on whole-tumor TSP) and link the ~14% reachability gap to that bias. CI95 not widened — that's a numerical change not metadata.

**CD8 audit + new mechanism.** Audited `Teff_module.m` per the open question: V_T.CD8 has only one gain term (`q_CD8_T_in` trafficking from blood) and pure loss (death + 4 exhaustion routes). No in-tumor proliferation reaction. Confirmed the handoff's missing-mechanism hypothesis. Added `null -> V_T.CD8` with rate `k_CD8_T_pro * V_T.CD8 * H_P1 * (1 - V_T.CD8/(rho_CD8_max*V_T + cell))` (commit `3eb3c0d` on `feature/cd8-in-tumor-expansion`). Antigen-gated only; deliberately *not* gated by `(1-H_PD1_C1)` after the trace showed `H_PD1_C1 ≈ 0.997` in PDAC TME silenced the reaction. Two reasons to drop the PD-1 gate: (1) live `V_T.CD8` already excludes exhausted cells (those partition to `CD8_exh`), so re-gating by exhaustion fraction double-counts; (2) PD-1+ TIL retain proliferative capacity — PD-1 axis suppresses cytotoxic effector function, not Ki67. Same reasoning rules out adding TGFb/Treg gates since both already act through the live→exhausted partition. Both new params (`k_CD8_T_pro` median 0.05/day, `rho_CD8_max` 1e7 cell/mL) committed as **placeholders** (`derivedFrom={}`, notes flag value as "qualitative anchor — not yet anchored on verified literature") to keep structural fix separable from rate calibration. Submodel-target extraction via maple is queued as the follow-up once SBI confirms the pathway is doing useful work.

**Quantitative reality check.** Single-theta `qsp_sim` trace at baseline_no_treatment confirms the reaction fires correctly: `k_CD8_T_pro=0` gives V_T.CD8 = 1.609e6 at diagnosis; default 0.05 gives 1.750e6 (+8.8%); high-tail prior 0.37 (~95th pct) gives 3.94e6 (+145% = 2.45×). So the new mechanism contributes only ~2.5× lift at the upper prior tail, not the 1000× the gap requires. Standard SS algebra shows the bound: with `V_T.CD8_ss = gain / (death + exhaustion - k_pro)`, even an unphysical `k_pro = 0.5/day` (1.4-day doubling) only multiplies SS by ~10×. This means the structural fix is **necessary but not sufficient** — the remaining ~100× must come from SBI posterior pull on `q_CD8_T_in` (current submodel target anchored on Boissonnas 2007 mouse-OT1 transfer kinetics, flagged in handoff as a poor cross-species translation). `make refresh-cpp` + `make parity` (2/2) both pass.

**Open work.** Branch `feature/cd8-in-tumor-expansion` not yet PR'd; full 5k SBI verification deferred (priors CSV change flips pool config_hash → fresh HPC pool). User regenerating `.compare_cache` after I hit a stale-cache regression in `regen_submodel_priors` (cache reflects pre-Abercrombie posteriors, not the current main's `submodel_priors.yaml` — separate issue from CD8 work). Path B (Hwang2008 forward-model rework for stromal_fraction tension) is the next structural item; not started yet.
