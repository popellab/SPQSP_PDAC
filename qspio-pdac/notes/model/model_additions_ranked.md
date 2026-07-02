# Ranked Model Additions for the PDAC QSP Model

*Based on qualitative gaps identified in the feedback loop topology analysis (`feedback_loop_analysis_results.md`, Section 9.2) and their impact on the feature-based transition analysis (Idea 28 in `model_interrogation_ideas.md`).*

---

## Ranking Criteria

Each addition is scored on four axes:

- **Topological impact:** Does this close a feedback loop, create a new one, or change the sign/gain of existing loops? Additions that close open loops or connect currently disconnected subsystems rank highest.
- **Clinical relevance for PDAC:** Does this address a known failure mode of PDAC immunotherapy (checkpoint blockade failure, stromal barrier, antigen presentation deficiency)?
- **Connection to transition analysis:** Does this addition create or sharpen qualitative transitions (Idea 28) that the current model cannot exhibit?
- **Feasibility:** How many new species, reactions, and parameters are needed? Can this be implemented as a module without restructuring existing code?

---

## Implementation Audit (Feb 2026)

Code review of the assembled model (module load order in `immune_oncology_model_PDAC.m`) revealed that several mechanisms exist in individual modules but are non-functional after assembly. The macrophage module aggressively rewrites APC dynamics at assembly time, and a stale parameter name from Phase 6 breaks antigen release coupling.

### Bugs and disconnections

| Mechanism | Module where defined | Status in assembled model | Root cause |
|-----------|---------------------|--------------------------|------------|
| Killing-dependent antigen release | `antigen_rate.m:29` | **BROKEN** — T-cell killing contributes zero antigen | `antigen_rate.m` searches for `k_C_CD8` in reaction rates, but Phase 6 reparameterization changed killing to use `k_C_Tcell_eff`. Search finds nothing; `R_T` defaults to `0*cell/day`. |
| DAMP-mediated APC maturation | `APC_module.m:55,92-94` | **DELETED** — species `c`, params `DAMPs`, `k_c`, `c50`, `c0` all removed | `macrophage_module.m:222-255` replaces APC maturation with `k_APC_mat*V_T.APC*H_IL12*(1-H_IL10)`, then deletes the cytokine `c` species and all DAMP-related parameters. |
| `R_Tcell` (Rule 5) | `simbio_init.m:155-157` | **VESTIGIAL** — computed (accumulates CD8 + macrophage killing) but nothing reads it | Was used by the DAMP release reaction (`R_Tcell*DAMPs`), which is deleted by macrophage module. |
| CCL2-driven APC recruitment | Described in changelog (9e22186, Feb 9) | **NOT IN CODE** — `k_APC_recruit_T` and `CCL2_50_APC` don't exist | APC recruitment still uses relaxation dynamics: `k_APC_death*(APC0_T*V_T-V_T.APC)`. Changelog entry may describe planned work or a different branch. |

### PDAC-specific APC maturation problem

The current intratumoral APC maturation formula (set by `macrophage_module.m:228`):

```matlab
k_APC_mat * V_T.APC * H_IL12 * (1 - H_IL10)
```

creates a **chicken-and-egg problem** in PDAC's immunologically cold TME:
- IL-12 comes from mAPCs (`k_IL12_sec*V_T.mAPC`) and M1 macrophages (`k_IL12_Msec*V_T.Mac_M1`)
- But you need IL-12 to mature APCs into mAPCs
- M1→M2 repolarization requires IL-12 or IFNg, both scarce in a cold TME
- Result: in a cold baseline, intratumoral APC maturation is near-zero with no way to bootstrap

The GVAX injection site is unaffected — its maturation (`J_mature_ID`, `gvax_module.m:174`) depends on GM-CSF + antigen, not IL-12. But endogenous intratumoral immune activation cannot self-start.

**Recommended fix**: restore a DAMP/danger signal term using OR logic with IL-12, add TGFb inhibition:

```matlab
% H_DAMP proportional to tumor cell death rate (use R_Tcell, currently vestigial)
% OR logic: either DAMPs or IL-12 can drive maturation; both inhibited by IL-10 and TGFb
k_APC_mat * V_T.APC * (1 - (1-H_DAMP)*(1-H_IL12)) * (1-H_IL10) * (1-H_TGFb_APC)
```

This:
- Restores the bootstrap (tumor death → DAMPs → APC maturation → antigen presentation → T cell priming → more killing)
- Keeps IL-12 positive feedback from the existing activated immune system
- Makes `R_Tcell` functional again (currently vestigial)
- Adds TGFb inhibition of APC maturation, missing but important for PDAC's TGFb-rich stroma (Batlle & Massague 2019)
- Preserves IL-10 inhibition already present

### Other PDAC-specific issues identified

1. **~~No TGFb inhibition of APC maturation.~~** [IMPLEMENTED — commit 9579f11] TGFb inhibition `*(1-H_TGFb_APC)` added to APC maturation formula in `APC_module.m:82`.

2. **~~ECM does not impede APC migration.~~** [IMPLEMENTED] Added `H_ECM_APC_mig` Hill function (ECM_level/(ECM_50_APC_mig+ECM_level)) to both APC migration (`k_APC_mig*V_T.mAPC*(1-H_ECM_APC_mig)`) and APC recruitment. `f_ECM_50_APC_mig=0.12` (12% of ECM_max, higher than T cell 6% — DCs use podosome/MMP-dependent migration).

3. **~~No tumor-derived CXCL12.~~** [IMPLEMENTED] Added `k_CXCL12_sec_cancer` parameter (5e-11 nM·L/(cell·day)) and reaction `null -> V_T.CXCL12` at rate `k_CXCL12_sec_cancer*C_total` in `fibroblast_module_enhanced.m`. Makes T cell exclusion partially tumor-autonomous.

4. **~~APC recruitment uses relaxation dynamics.~~** [IMPLEMENTED] Split relaxation into separate recruitment (`k_APC_death*APC0_T*V_T*(1-H_ECM_APC_mig)`) and death (`k_APC_death*V_T.APC`) reactions in `APC_module.m`. New equilibrium: `APC_eq = APC0_T*V_T*(1-H_ECM_APC_mig)`. Identical to original at zero ECM.

---

## Rank 1: Close the Cancer Immunity Cycle (Gaps 1 + 2) [IMPLEMENTED]

**What's missing.** The most important positive feedback loop in immuno-oncology — tumor killing → antigen release → APC maturation → T cell priming → trafficking → more killing — does not appear as a coherent loop. The model has all the pieces (CD8 killing in `Teff_module.m`, antigen processing in `antigen_module.m`, APC maturation in `APC_module.m`, T cell priming in `TCR_module.m`) but they don't form a closed cycle in the feedback loop analysis. Additionally, C1 (tumor cells) is absent from all biology-only feedback loops — it's a passive target rather than an active participant.

**What to add.**

1. **Killing-dependent antigen release.** [STATUS: **BROKEN** — mechanism exists in `antigen_rate.m` but search string `k_C_CD8` is stale after Phase 6 reparameterization. One-line fix: change `k_C_CD8` → `k_C_Tcell_eff` at `antigen_rate.m:29`.] When CD8 T cells kill C1 cells (producing C_x dead cells), some fraction of intracellular antigen (P1) should be released into the extracellular space. Currently, C_x accumulates and is cleared, but no antigen is liberated. The new reaction:

       C1 death → P1 release (proportional to killing rate × C1 × CD8 interactions)

   This creates the edge: **C1 death → P1 → APC uptake → mAPC → T cell priming → CD8 → C1 killing**, closing the adaptive immunity positive feedback loop.

2. **Tumor cell cytokine/chemokine production.** [STATUS: **PARTIALLY DONE** — TGFb baseline drive exists in `fibroblast_module_enhanced.m`; CCL2 secretion by cancer cells exists in `macrophage_module.m:99` (`k_CCL2_sec*C_total`). **Remaining gap**: CXCL12 is only produced by aPSCs, not tumor cells.] C1 currently produces only c_vas (angiogenic factors). Real PDAC tumor cells also secrete:
   - **TGFb** (immunosuppressive) — creates edge: C1 → TGFb → immunosuppressive cascade
   - **CCL2** (monocyte recruitment) — creates edge: C1 → CCL2 → macrophage/MDSC recruitment
   - **CXCL12** (T cell exclusion) — currently produced only by aPSC; adding C1 as a source strengthens the exclusion mechanism

   These close C1 into the immunosuppressive feedback loops, making tumor burden a driver of suppression rather than just a readout.

3. **DAMP-mediated APC activation.** [STATUS: **DELETED** — `APC_module.m` adds this mechanism via cytokine `c` and `DAMPs` parameter, but `macrophage_module.m:222-255` replaces APC maturation with IL-12-driven formula, then deletes `c`, `DAMPs`, `k_c`, `c50`, `c0`. See Implementation Audit above for recommended fix using OR logic with IL-12.] Tumor cell death (especially necrotic/immunogenic death) releases DAMPs that activate immature APCs independently of antigen processing. A simple implementation: C_x (dead tumor cells) drives APC maturation at a rate proportional to the death rate. This creates a second, faster path through the immunity cycle that bypasses the slow antigen-processing steps.

**Topological impact.** This is the highest-impact addition because it closes *the* central positive feedback loop of cancer immunology. Currently, immunotherapy in the model works by removing brakes (anti-PD-1) or providing an exogenous kick (GVAX). With a closed immunity cycle, immunotherapy can create a *self-sustaining* response — the hallmark of durable clinical benefit.

**Impact on transition analysis (Idea 28).** The immune activation threshold (Transition 3, transcritical analog) becomes sharper and more meaningful. Currently, the model lacks the amplification mechanism that makes immune activation self-sustaining. With the immunity cycle closed, there is a genuine positive feedback loop whose gain determines whether a perturbation (GVAX, checkpoint blockade) triggers a self-amplifying cascade or decays. The threshold for this cascade is the most clinically important qualitative transition in the model.

**Feasibility.** Low complexity. No new species needed — P1, C_x, APC, mAPC all exist. Requires 2-3 new reactions and 3-5 new parameters (antigen release rate per killed cell, DAMP activation rate, C1 cytokine secretion rates). Can be implemented as additions to `cancer_module.m` and `antigen_module.m`.

**Literature.** Chen & Mellman (2013, "Oncology meets immunology: the cancer-immunity cycle"); Galluzzi et al. (2017, immunogenic cell death); Binnewies et al. (2018, PDAC immune evasion mechanisms).

---

## Rank 2: CAF Heterogeneity — iCAF/myCAF Split (Gap 4) [IMPLEMENTED]

**What's missing.** Real PDAC CAFs have at least two functionally distinct subtypes:

- **Inflammatory CAFs (iCAFs):** IL6-producing, located distal to tumor cells, driven by IL1/JAK-STAT signaling. iCAFs promote MDSC recruitment, M2 polarization, and systemic inflammation.
- **Myofibroblastic CAFs (myCAFs):** ECM-producing, located adjacent to tumor cells, driven by TGFb signaling. myCAFs build the physical stromal barrier that excludes T cells.

The current model has a single aPSC state that produces both IL6 and ECM. This conflates two opposing therapeutic targets: iCAFs are harmful (drive immunosuppression) while myCAFs may be protective (physically restrain tumor growth — the Ozdemir et al. 2014 paradox).

**What to add.**

Split `aPSC` into two species: `iCAF` and `myCAF`, both derived from `qPSC` activation.

- **qPSC → iCAF:** Driven by IL1 (or a proxy cytokine; could use IL6 as a self-amplifying loop). iCAFs produce IL6 and CCL2.
- **qPSC → myCAF:** Driven by TGFb. myCAFs produce ECM and CXCL12.
- **iCAF ↔ myCAF interconversion:** TGFb pushes iCAF → myCAF (consistent with Biffi et al. 2019). This creates a competition for the qPSC pool — the balance between iCAF and myCAF depends on the IL1/TGFb ratio in the microenvironment.

**Topological impact.** This splits the current fibroblast loops (which already have 5-6 decade gain variability) into two distinct circuits:

- An **immunosuppressive circuit**: iCAF → IL6 → MDSC → ArgI → T cell suppression
- A **barrier circuit**: myCAF → ECM → T cell exclusion

These have different therapeutic implications. Targeting iCAFs (JAK inhibitors) would reduce immunosuppression without removing the physical barrier. Targeting myCAFs (anti-TGFb) would reduce the barrier but might paradoxically release tumor cells from physical confinement (the Ozdemir paradox). The model currently cannot distinguish these effects.

**Impact on transition analysis.** The desmoplastic irreversibility transition (Transition 6) becomes richer. With iCAF/myCAF split, you can sweep TGFb and observe whether the stroma transitions from iCAF-dominated (immunosuppressive but physically permeable) to myCAF-dominated (physically barrier-forming but less immunosuppressive). This transition has direct implications for stromal targeting strategy selection — the phase diagram of IL1 × TGFb would map which stromal phenotype dominates.

**Feasibility.** Moderate complexity. Adds 1 new species (split aPSC into iCAF + myCAF), modifies the activation reactions, and redistributes existing cytokine production. Requires 4-6 new parameters (interconversion rates, phenotype-specific secretion rates). The existing `fibroblast_module_enhanced.m` provides the scaffold.

**Literature.** Biffi et al. (2019, iCAF vs myCAF in PDAC); Ohlund et al. (2017, CAF heterogeneity); Ozdemir et al. (2014, myofibroblast depletion accelerates PDAC); Elyada et al. (2019, single-cell CAF subtypes in PDAC).

---

## Rank 3: Hypoxia–HIF–VEGF Feedback (Gap 3) [IMPLEMENTED]

**What's missing.** PDAC is among the most hypoxic solid tumors (median pO2 ~2.7 mmHg vs ~50 mmHg in normal pancreas). Hypoxia drives a multi-arm positive feedback loop:

1. **Hypoxia → HIF-1α → VEGF → aberrant vasculature → more hypoxia** (vascular positive feedback)
2. **Hypoxia → HIF-1α → PD-L1 upregulation** (adaptive resistance, independent of IFNg)
3. **Hypoxia → HIF-1α → M2 macrophage polarization** (immunosuppression)
4. **Hypoxia → metabolic shift → lactate accumulation → T cell dysfunction** (connects to Gap 5)

The model already has `c_vas` (VEGF-like angiogenic factor) and `K` (vascular carrying capacity), so there is a hook for the vascular arm. But there is no oxygen tension, no HIF signaling, and VEGF does not feed back onto immune cells.

**What to add.**

1. **Oxygen tension species** (`pO2`): Determined by the balance of vascular supply (`K`) and metabolic demand (C1 + immune cells). Low pO2 = hypoxia.
2. **HIF-1α activity** (or a dimensionless "hypoxia index"): Activated at low pO2. Drives VEGF production, PD-L1 upregulation, and M2 polarization.
3. **VEGF → vascular normalization/aberration:** Connect existing `c_vas` and `K` dynamics to oxygen delivery. High VEGF → high K but aberrant (leaky, poorly functional) → low pO2 delivery per unit K.

**Topological impact.** Creates an entirely new positive feedback loop (hypoxia → VEGF → aberrant vasculature → more hypoxia) and connects it to existing immunosuppressive circuits. HIF-driven PD-L1 upregulation adds a second, IFNg-independent source of checkpoint resistance — this matters because anti-PD-1 would be less effective against constitutive (hypoxia-driven) PD-L1 than against adaptive (IFNg-driven) PD-L1.

**Impact on transition analysis.** The response threshold (Transition 1) becomes dependent on oxygenation. A well-vascularized tumor might be above the immune activation threshold, while a hypoxic tumor of the same size might be below it — because hypoxia simultaneously upregulates PD-L1 and polarizes macrophages toward M2. This creates a new axis in the phase diagram (Transition 8): TGFb × pO2 would reveal whether oxygenation and TGFb are redundant or synergistic immunosuppressive mechanisms.

**Feasibility.** Moderate-high complexity. Adds 2-3 new species (pO2 or hypoxia index, HIF activity, possibly lactate as a bridge to Gap 5), modifies existing VEGF and vascular dynamics, and adds HIF-dependent terms to PD-L1 expression and macrophage polarization. Requires 6-10 new parameters. Could be implemented as a standalone `hypoxia_module.m`.

**Literature.** Koong et al. (2000, PDAC hypoxia measurements); Semenza (2012, HIF and cancer); Noman et al. (2014, HIF-1α → PD-L1); Jain (2014, vascular normalization hypothesis).

---

## Rank 4: NK Cells (Gap 6, partial) [IMPLEMENTED]

**What's missing.** Natural killer cells provide antigen-independent innate killing of tumor cells. In PDAC, NK cell function is suppressed by TGFb, IL-10, and MDSC-derived factors — but NK cells remain an important innate surveillance mechanism and a target of emerging therapies (anti-NKG2A, IL-15 superagonists, NK cell engagers).

**What to add.**

1. **NK cell species** in tumor and peripheral compartments (`V_T.NK`, `V_P.NK`)
2. **NK killing of C1:** Contact-dependent, modulated by NKG2D ligand expression on tumor cells (stress-induced) and inhibited by MHC-I expression.
3. **TGFb/IL-10 suppression of NK function:** Connects NK cells to existing immunosuppressive cytokine network.
4. **NK-derived IFNg:** NK cells produce IFNg upon tumor contact, feeding into existing macrophage polarization circuits.

**Topological impact.** Creates a second, antigen-independent killing loop alongside CD8 T cells. This is important because it provides a killing mechanism that does not require antigen presentation or T cell priming — it can respond to tumor cells that have downregulated MHC-I (a common immune evasion mechanism not currently in the model). NK-derived IFNg also creates a new entry point into the Th self-liberation loop.

**Impact on transition analysis.** NK cells create a "floor" of immune pressure that exists even when the adaptive immune system is suppressed. This could change the character of the immune activation threshold (Transition 3): instead of a sharp on/off switch for adaptive immunity, there would be a basal level of NK-mediated control with a separate threshold for adaptive amplification on top of it. The feature sweep of TGFb might show two thresholds — one where NK killing becomes insufficient and one where adaptive immunity fails — rather than a single transition.

**Feasibility.** Moderate complexity. Adds 2 species, ~5 reactions (trafficking, killing, IFNg production, TGFb/IL-10 suppression, death), ~8 parameters. Can be a standalone `NK_module.m`.

**Literature.** Peng et al. (2021, NK cells in PDAC); Lim et al. (2017, TGFb suppression of NK cells); Andre et al. (2018, anti-NKG2A in cancer).

---

## Rank 5: Metabolic Competition — Glucose/Lactate (Gap 5) [IMPLEMENTED]

**What's missing.** PDAC tumor cells are highly glycolytic (Warburg effect), consuming glucose and producing lactate. Both create immunosuppressive conditions:

- **Glucose depletion:** T cells require glucose for activation, proliferation, and effector function. In glucose-poor environments, T cells shift to a quiescent/anergic state.
- **Lactate accumulation:** Acidifies the microenvironment (pH ~6.5-6.8 in PDAC), directly inhibiting T cell cytotoxicity and IFNg production, and promoting M2 macrophage polarization.

**What to add.**

1. **Glucose species** (`V_T.glucose`): Supplied by vasculature (K-dependent), consumed by C1 (proportional to growth rate and cell number) and immune cells (proportional to activation state).
2. **Lactate species** (`V_T.lactate`): Produced by C1 (proportional to glucose consumption), cleared by vasculature (K-dependent).
3. **Glucose-dependent T cell function:** CD8 and Th effector functions (killing, cytokine production) are multiplied by a Hill function of glucose availability. Below a glucose threshold, T cells become functionally impaired.
4. **Lactate-driven M2 polarization:** Lactate promotes M2 macrophage polarization independently of TGFb/IL-10, adding a metabolic immunosuppression axis.

**Topological impact.** Creates a new positive feedback loop: **C1 growth → glucose consumption → T cell impairment → reduced killing → more C1 growth.** This is a tumor-immune competition loop that operates through resource depletion rather than cytokine signaling. It also connects vascular dynamics (K) to immune function through nutrient delivery, creating a bridge between the angiogenesis subsystem and immune dynamics.

**Impact on transition analysis.** Adds a metabolic dimension to the phase diagram. The immune activation threshold (Transition 3) becomes dependent on tumor size — larger tumors consume more glucose, creating a stronger metabolic barrier to immune activation. This could explain why immunotherapy works better in smaller tumors (lower metabolic competition) and creates a tumor-size-dependent qualitative transition.

**Feasibility.** Moderate-high complexity. Adds 2 species, modifies multiple existing rate laws (T cell killing, macrophage polarization, T cell proliferation), and requires careful parameterization of glucose consumption/production rates. Requires 8-12 new parameters. The vascular dynamics (`K`, `c_vas`) provide the supply-side hook. Could be a `metabolism_module.m`.

**Literature.** Chang et al. (2015, glucose competition in tumors); Brand et al. (2016, lactate and T cell function); Ho et al. (2015, phosphoenolpyruvate and T cell function); Halbrook et al. (2023, PDAC metabolism review).

---

## Rank 6: Dendritic Cell Subtypes — cDC1 vs cDC2 (Gap 6, partial) [IMPLEMENTED]

**What's missing.** The model has a single APC type that handles both antigen uptake and T cell priming. Real dendritic cells have functionally distinct subtypes:

- **cDC1 (BATF3+):** Specialized for cross-presentation of tumor antigens to CD8 T cells. Critical for anti-tumor immunity. Rare in PDAC tumors.
- **cDC2 (IRF4+):** Present antigens to CD4 T helper cells. More abundant but less critical for cytotoxic responses.

In PDAC, cDC1 scarcity is a major barrier to anti-tumor immunity. The tumor actively excludes or suppresses cDC1s through beta-catenin signaling and lipid accumulation.

**What to add.**

Split `APC/mAPC` into `cDC1/cDC2` with distinct functions:
- **cDC1:** Cross-presents to CD8 T cells, produces IL-12, recruited by XCL1 (from NK cells — synergy with Rank 4). Suppressed by IL-10 and TGFb.
- **cDC2:** Presents to CD4 Th cells, produces IL-6 and IL-23. Less affected by immunosuppressive cytokines.

**Topological impact.** Creates a bottleneck in the cancer immunity cycle (Rank 1): even if antigen is released by tumor killing, effective CD8 priming requires cDC1 cross-presentation. If cDC1s are scarce or suppressed, the immunity cycle cannot close. This makes cDC1 abundance a gatekeeping variable for the immune activation threshold.

**Impact on transition analysis.** The immune activation threshold (Transition 3) becomes cDC1-dependent. The feature sweep of cDC1 recruitment rate would reveal whether there is a critical cDC1 density below which the immunity cycle cannot engage — a distinct threshold from the macrophage polarization threshold (M1/M2) or the TGFb threshold.

**Feasibility.** Moderate complexity. Splits existing APC species into two subtypes, modifies T cell priming reactions to route through the appropriate DC subtype. Requires 4-6 new parameters (subtype-specific migration, maturation, and antigen presentation rates). Depends on Rank 1 being implemented first (closed immunity cycle) for full benefit.

**Literature.** Hildner et al. (2008, BATF3 DCs essential for cross-priming); Broz et al. (2014, cDC1 in tumor immunity); Binnewies et al. (2019, cDC1 scarcity in non-responsive tumors); Lin et al. (2020, cDC1 in PDAC).

---

## Rank 6b: IFNg-Driven T Cell Chemotaxis — CXCL9/CXCL10 Loop [IMPLEMENTED]

**What's missing.** T cell recruitment to the tumor is currently driven purely by tumor size (`C_total^2/(C_total^2 + K_C2_recruit)`) and modulated negatively by CXCL12 exclusion. There is no positive, inflammation-dependent signal for T cell trafficking. In reality, the distinction between "immune-hot" and "immune-cold" tumors — the single most important predictor of immunotherapy response — is largely driven by IFNg-induced chemokines:

- **CXCL9 (MIG) and CXCL10 (IP-10):** Produced by tumor cells, macrophages, and stromal cells in response to IFNg (via JAK-STAT1). These are the dominant CXCR3 ligands that recruit activated CD8 T cells and NK cells to the tumor.
- **Positive feedback loop:** T cells → IFNg → CXCL9/10 → CXCR3-mediated T cell recruitment → more T cells → more IFNg. This makes immune infiltration self-reinforcing once it begins.

Without this loop, the model's T cell infiltration is a passive, size-dependent process. The model can distinguish "primed vs unprimed" (via the cancer immunity cycle, Rank 1) but cannot distinguish "infiltrated vs excluded" as separate stable states. Anti-PD-1 response predictions will underestimate the positive feedback where checkpoint blockade → more killing → more IFNg → more CXCL9/10 → more T cell recruitment.

**What to add.**

1. **CXCL9/10 species** (`V_T.CXCL9`, lumped as single species representing total CXCR3 ligands): Produced by tumor cells, M1 macrophages, and cDC1s in response to IFNg. Degraded at first order.
2. **IFNg-dependent CXCL9/10 production:** `k_CXCL9_sec * (C_total + V_T.Mac_M1 + V_T.mcDC1) * H_IFNg_CXCL9` where `H_IFNg_CXCL9 = V_T.IFNg / (IFNg_50_CXCL9 + V_T.IFNg)`.
3. **CXCL9/10-dependent T cell recruitment:** Multiply CD8 (and Th) tumor entry rates by `(1 + H_CXCL9_Trec)` where `H_CXCL9_Trec = V_T.CXCL9 / (CXCL9_50_Trec + V_T.CXCL9)`. This means baseline recruitment (size-dependent) is unchanged, but CXCL9/10 can up to double it. The balance between CXCL9/10 attraction and CXCL12 exclusion determines net infiltration.
4. **NK recruitment boost:** CXCL9/10 also recruit NK cells via CXCR3. Modify NK tumor entry similarly.

**Topological impact.** Creates the trafficking-level positive feedback loop that complements the priming-level loop (Rank 1). Together, these create two nested positive feedback loops:
- **Inner (priming) loop:** killing → antigen → DAMP → APC maturation → T cell priming → more T cells
- **Outer (trafficking) loop:** T cells in tumor → IFNg → CXCL9/10 → more T cell recruitment → more IFNg

The outer loop controls whether primed T cells actually reach the tumor. A tumor with high CXCL12 and low CXCL9/10 can have a fully primed adaptive immune system in the lymph node that never infiltrates — a clinically recognized scenario in PDAC ("immune-excluded" phenotype).

This also creates a natural antagonism between CXCL9/10 (attraction) and CXCL12 (exclusion), producing a chemokine balance that determines the infiltration phenotype. The balance is influenced by:
- IFNg levels (driven by Th, CD8, NK cells) → CXCL9/10
- myCAF density → CXCL12
- TGFb → suppresses IFNg-producing cells → reduces CXCL9/10

**Impact on transition analysis.** The immune activation threshold (Transition 3) splits into two distinct thresholds:
1. **Priming threshold:** Can the immunity cycle generate enough primed T cells? (Already captured by Rank 1)
2. **Infiltration threshold:** Can enough primed T cells overcome CXCL12 exclusion via CXCL9/10 recruitment? (New)

These thresholds can be independently modulated: a tumor with high TGFb might fail at priming, while a tumor with high myCAF/CXCL12 but intact priming might fail at infiltration. This maps to the clinically observed "immune-desert" vs "immune-excluded" phenotypes, which have different therapeutic implications (the former needs priming help like GVAX; the latter needs stromal remodeling like anti-CXCR4).

**Feasibility.** Low complexity. Adds 1 species (CXCL9/10 lumped), modifies existing T cell and NK recruitment reactions at assembly time. Requires 3-4 new parameters (CXCL9/10 secretion rate, degradation rate, EC50 for T cell recruitment, EC50 for IFNg induction). Can be a standalone `chemotaxis_module.m` or added to existing modules.

**Literature.** Mikucki et al. (2015, CXCR3 required for T cell infiltration of tumors, PMID: 26056112); Peng et al. (2015, IFNg/CXCL9 axis in anti-PD-1 response); Dangaj et al. (2019, CXCL9 defines T cell-inflamed tumors, PMID: 30728497); Spranger et al. (2017, T cell exclusion mechanisms in non-inflamed tumors); Feig et al. (2013, CXCL12 vs CXCL9/10 balance in PDAC, PMID: 24264993).

---

## ~~Rank 7: B Cells and Tertiary Lymphoid Structures~~ → Moved to Rank 15

*Renumbered to reflect expanded gap analysis. See Rank 15 below.*

---

## Rank 8: CCL2-Driven APC Recruitment + M1 CCL2 Secretion [IMPLEMENTED]

**What's missing.** APC recruitment currently uses relaxation dynamics (`k_APC_death*(APC0_T*V_T*(1-H_ECM_APC_mig))`) — a homeostatic mechanism where APCs passively fill a niche proportional to tumor volume. Real DC recruitment is chemokine-driven: CCL2 (MCP-1) from tumor cells and M1 macrophages is the dominant monocyte/DC chemoattractant. The model already has CCL2 production by cancer cells (`k_CCL2_sec*C_total` in `macrophage_module.m`) but this CCL2 only recruits macrophages, not APCs. Additionally, M1 macrophages are a major CCL2 source in inflamed tumors but don't produce CCL2 in the model.

**What to add.**

1. **CCL2-dependent APC recruitment.** Replace the relaxation-based APC recruitment with a mechanistic CCL2-driven term:

   ```matlab
   % Recruitment (baseline + CCL2-enhanced, ECM-impeded):
   'k_APC_recruit_T * V_T * (1 + V_T.CCL2/(CCL2_50_APC + V_T.CCL2)) * (1-H_ECM_APC_mig)'
   % Death (unchanged):
   'k_APC_death * V_T.APC'
   ```

   Apply the same pattern to lymph node APCs with `k_APC_recruit_LN`.

2. **M1 macrophage CCL2 secretion.** Add M1 macrophages as a CCL2 source:

   ```matlab
   'null → V_T.CCL2'  at rate  'k_CCL2_Msec * V_T.Mac_M1'
   ```

   This closes an innate amplification loop: immune activation → M1 macrophages → CCL2 → DC recruitment → more antigen presentation → more T cell priming.

**Topological impact.** Makes APC recruitment inflammation-dependent rather than passive. In a cold TME with few M1 macrophages, APC recruitment is low (only cancer-derived CCL2). In an inflamed TME with many M1s, additional CCL2 boosts DC recruitment, creating positive feedback. This strengthens the connection between innate and adaptive immunity.

**Impact on transition analysis.** Mildly sharpens the immune activation threshold (Transition 3) by adding another positive feedback element: more immune activation → more M1s → more CCL2 → more DCs → more priming → more activation. However, the effect is incremental on top of existing feedback loops (CXCL9/10, DAMP-APC maturation, NK-XCL1-cDC1).

**Feasibility.** Very low complexity. No new species needed (CCL2 already exists). 3 new parameters (`k_APC_recruit_T`, `CCL2_50_APC`, `k_CCL2_Msec`). Modifies existing APC recruitment reactions and adds one new reaction. Already spec'd in the changelog.

**Literature.** Nagarsheth et al. (2017, CCL2 in tumor immunity, PMID: 28291677); Fridlender et al. (2011, M1 macrophage CCL2 production); Ruffell et al. (2012, CCL2 and DC recruitment in tumors).

---

## Rank 9: Adenosine Pathway — CD73/CD39/A2AR

**What's missing.** Extracellular adenosine is a major immunosuppressive metabolite in the hypoxic PDAC TME. Dying cells release ATP (a pro-inflammatory "find-me" signal), but ectonucleotidases CD39 and CD73 — expressed by Tregs, tumor cells, CAFs, and MDSCs — sequentially convert ATP → AMP → adenosine. Adenosine signals through A2A receptors on T cells and NK cells, causing:

- Suppression of TCR signaling and cytokine production
- Inhibition of NK cytotoxicity
- Promotion of Treg differentiation and stability
- Inhibition of DC maturation

This creates a perverse "death trap" feedback: immunotherapy-induced tumor killing releases ATP that *should* activate immunity (via DAMPs/P2X7 signaling), but CD73/CD39 convert it to immunosuppressive adenosine instead. The net effect is that more killing can paradoxically increase local immunosuppression.

Critically, HIF-1alpha directly upregulates CD73 expression (Synnestvedt et al. 2002), linking the adenosine axis to the existing hypoxia module. The model already has the hypoxia → HIF pathway and DAMP-dependent APC maturation (which implicitly uses ATP-like danger signals); the adenosine axis would create the counterbalancing suppressive arm.

**What to add.**

1. **Adenosine species** (`V_T.adenosine`): Produced proportionally to tumor cell death rate (reuse `R_Tcell` as ATP source proxy). Degraded at first order (adenosine deaminase).

2. **CD73 activity as a HIF-dependent parameter:** `k_CD73_eff = k_CD73_base * (1 + f_HIF_CD73 * HIF)`. Adenosine production rate = `k_CD73_eff * R_Tcell * adenosine_per_death`. This makes adenosine production both death-dependent and hypoxia-dependent.

3. **A2AR-mediated T cell suppression:** Hill function `H_A2AR = adenosine / (adenosine + A2AR_50)`. Multiply CD8 killing (`k_C_Tcell_eff`), IFNg secretion, and T cell proliferation by `(1 - H_A2AR)`.

4. **A2AR-mediated NK suppression:** Same Hill function applied to NK killing and IFNg secretion.

5. **Optional: Treg stabilization.** Adenosine enhances Treg suppressive function via A2AR — could multiply Treg-mediated exhaustion rate by `(1 + f_A2AR_Treg * H_A2AR)`.

**Topological impact.** Creates a negative feedback loop that directly counteracts the DAMP-driven positive feedback (Rank 1): tumor death → ATP → *either* DAMP-mediated APC activation (pro-immune) *or* adenosine-mediated T cell suppression (anti-immune). The balance between these two arms depends on CD73 expression (HIF-dependent) and adenosine clearance. This is the first mechanism in the model where increased killing can paradoxically weaken the immune response.

This also creates a new connection between hypoxia and immune suppression beyond PD-L1 and M2 polarization: hypoxia → HIF → CD73 → adenosine → T cell/NK suppression. The three HIF-driven immunosuppressive arms (PD-L1, M2 polarization, adenosine) become a multi-pronged barrier that anti-PD-1 alone cannot overcome.

**Impact on transition analysis.** The immune activation threshold (Transition 3) acquires a "futile cycling" regime: immunotherapy increases killing, which increases both DAMPs (pro-immune) and adenosine (anti-immune). Below a critical killing rate, adenosine dominates and the response is self-limiting. Above it, DAMPs overwhelm adenosine and the response becomes self-sustaining. This creates a sharper, more clinically realistic activation threshold. The phase diagram gains a hypoxia × killing rate axis that maps whether tumors can escape the adenosine trap.

**Feasibility.** Moderate complexity. 1 new species (adenosine), 1 algebraic parameter (CD73 activity), ~6-8 new parameters (CD73 base rate, HIF amplification factor, adenosine degradation rate, A2AR EC50, adenosine production per death event, optional Treg amplification). Hooks into existing `R_Tcell` (death rate), `HIF` (hypoxia), and killing/IFNg reactions. Could be a standalone `adenosine_module.m`.

**Literature.** Vijayan et al. (2017, CD73 in cancer immunotherapy, PMID: 28408437); Allard et al. (2020, adenosine pathway in PDAC, PMID: 32205441); Synnestvedt et al. (2002, HIF → CD73, PMID: 12130536); Ohta et al. (2006, A2AR and T cell suppression, PMID: 16489043); Stagg et al. (2010, anti-CD73 enhances anti-tumor immunity, PMID: 20538761).

---

## Rank 10: MHC-I Downregulation and Immune Evasion

**What's missing.** PDAC tumors frequently downregulate MHC class I molecules (40-60% of cases; Pommier et al. 2018), rendering them invisible to CD8 T cells. This is a major immune evasion mechanism that directly undermines checkpoint blockade: anti-PD-1 cannot restore killing of cells that CD8s cannot recognize. However, MHC-I loss simultaneously makes tumor cells *more* vulnerable to NK cell killing (the "missing self" hypothesis — NK cells are activated by the absence of MHC-I inhibitory ligands).

The current model has CD8 killing and NK killing operating in parallel but without this tradeoff axis. CD8 killing efficacy is independent of MHC-I status, and NK killing efficacy depends only on cytokine suppression (TGFb, IL-10) and MDSC inhibition, not on MHC-I ligand balance.

**What to add.**

1. **MHC-I expression level** (`MHC_I_level`): A dimensionless algebraic parameter (0-1) representing the fraction of tumor cells expressing functional MHC-I. Could be a fixed parameter (patient-level heterogeneity) or a slowly evolving species driven by immune selection pressure.

2. **MHC-I dependence of CD8 killing:** Multiply `k_C_Tcell_eff` by `MHC_I_level`. When MHC-I is fully expressed (1.0), no change. When MHC-I is lost (→0), CD8 killing approaches zero regardless of T cell number or checkpoint status.

3. **Missing-self NK activation:** Multiply NK killing by `(1 + f_missing_self * (1 - MHC_I_level))`. When MHC-I is fully expressed, NK killing is unchanged. When MHC-I is lost, NK killing is amplified (up to `1 + f_missing_self`, e.g., 2-3x).

4. **Optional: Immune selection pressure.** If `MHC_I_level` is a dynamic variable:

   ```matlab
   % Immune selection: CD8 killing preferentially removes MHC-I+ cells,
   % shifting population toward MHC-I-low
   d(MHC_I_level)/dt = -k_MHC_I_selection * k_C_Tcell_eff * C_total * MHC_I_level
                       + k_MHC_I_recovery * (MHC_I_baseline - MHC_I_level)
   ```

   This creates immune editing: effective CD8 killing selects for MHC-I-low variants over time, progressively undermining adaptive immunity. This is a slow, irreversible transition.

**Topological impact.** Creates a qualitative tradeoff between CD8 and NK killing that the model currently lacks. With the dynamic version, also creates an immune editing feedback loop: CD8 success → MHC-I selection → CD8 failure → NK compensation. This is the first mechanism that represents tumor evolution/adaptation in the model.

**Impact on transition analysis.** Creates a new qualitative transition (immune editing threshold): below a critical CD8 killing rate, MHC-I is stable and adaptive immunity dominates. Above it, selection pressure drives MHC-I loss, eventually switching the tumor from adaptive-immune-controlled to NK-dependent. This transition is clinically observed (acquired resistance to checkpoint blockade) and therapeutically actionable (NK-based therapies for MHC-I-low tumors).

**Feasibility.** Low complexity for the static version (1 parameter, modify 2 reaction rates). Moderate for the dynamic version (1 new species, ~4 parameters). The static version is sufficient for initial exploration of the CD8/NK tradeoff; the dynamic version adds immune editing but requires careful parameterization of selection timescales.

**Literature.** Pommier et al. (2018, MHC-I loss in PDAC, PMID: 30232225); Garrido et al. (2016, MHC-I defects and immune evasion, PMID: 27485449); Malmberg et al. (2017, NK cell recognition of missing self, PMID: 28432221); Bern et al. (2019, missing-self recognition in cancer, PMID: 30917364).

---

## Rank 11: Gemcitabine Chemotherapy Module

**What's missing.** Gemcitabine + nab-paclitaxel (gem/nabP) is the standard-of-care first-line chemotherapy for metastatic PDAC. The model has a nab-paclitaxel module (`nabpaclitaxel_module.m`) but no gemcitabine. Without it, the model cannot simulate the most common clinical treatment regimen or explore chemo-immunotherapy combinations (gem/nabP + checkpoint blockade, gem/nabP + GVAX), which are the focus of most PDAC clinical trials.

Gemcitabine has both direct cytotoxic effects and immunomodulatory properties:
- **Direct cytotoxicity:** S-phase-specific nucleoside analog. Kills rapidly dividing cells (tumor, also some immune cells).
- **MDSC depletion:** Gemcitabine selectively depletes MDSCs at standard doses (Suzuki et al. 2005), partially relieving immunosuppression.
- **Immunogenic cell death:** At certain doses, gemcitabine-induced cell death can be immunogenic (DAMPs release), potentially synergizing with immunotherapy.
- **Treg depletion:** Moderate Treg depletion at therapeutic doses (Homma et al. 2014).

**What to add.**

1. **Gemcitabine PK** (`V_C.GEM`): Two-compartment PK (similar scaffold to existing `pk_module.m`). Standard dosing: 1000 mg/m^2 IV on days 1, 8, 15 of 28-day cycle.

2. **Tumor cell killing:** `k_GEM_kill * V_T.C1 * H_GEM` where `H_GEM = V_C.GEM / (GEM_50 + V_C.GEM)`. Growth-rate-dependent (only kills dividing cells): could multiply by a growth fraction parameter.

3. **MDSC depletion:** `k_GEM_MDSC * V_T.MDSC * H_GEM`. Selective at therapeutic doses (Suzuki 2005).

4. **Optional immunomodulation:** Gemcitabine-killed cells contribute to `R_Tcell` (antigen release/DAMP signal), creating chemo-immunotherapy synergy with GVAX or checkpoint blockade.

**Topological impact.** Minimal in terms of new feedback loops — chemotherapy is primarily a perturbation (external forcing) rather than a new feedback circuit. However, the MDSC depletion arm creates an indirect positive feedback: gemcitabine → MDSC depletion → reduced Arg-I → improved T cell function → more killing. The immunogenic cell death component bridges chemotherapy to the existing DAMP-APC maturation loop.

**Impact on transition analysis.** Gemcitabine can push the system across the immune activation threshold (Transition 3) by simultaneously debulking tumor (reducing metabolic competition), depleting MDSCs (reducing suppression), and releasing DAMPs (boosting APC maturation). The question of whether chemotherapy opens a window for immunotherapy — and how wide that window is — is one of the most clinically pressing questions in PDAC.

**Feasibility.** Moderate complexity. Follows established PK module patterns in the codebase. ~8-10 new parameters (PK parameters, killing EC50, MDSC depletion rate). The nab-paclitaxel module provides a direct scaffold.

**Literature.** Suzuki et al. (2005, gemcitabine depletes MDSCs, PMID: 16110312); Homma et al. (2014, gemcitabine Treg effects); Nowak et al. (2003, gemcitabine immunomodulation); Von Hoff et al. (2013, gem/nabP phase III in PDAC, PMID: 24131140).

---

## Rank 12: CD40 Agonist Therapy Module

**What's missing.** CD40 agonist antibodies (e.g., selicrelumab, sotigalimab/APX005M) represent a fundamentally different immunotherapy strategy for PDAC: instead of removing brakes (checkpoint blockade) or providing exogenous antigen (GVAX), they directly activate APCs and macrophages, bypassing the T cell priming bottleneck entirely. In the pivotal Beatty et al. (2011) study, CD40 agonism caused rapid tumor regression in PDAC patients through macrophage-mediated stromal depletion — without requiring T cell infiltration.

This is particularly important for PDAC's "cold" TME because CD40 agonism can work in tumors where the cancer immunity cycle is broken at the priming step (low cDC1, poor antigen presentation).

**What to add.**

1. **CD40 agonist PK** (`V_C.aCD40`): Standard antibody PK (existing `pk_module.m` scaffold).

2. **Direct APC/DC activation:** CD40 ligation on cDC1/cDC2 drives maturation independently of DAMPs or IL-12:

   ```matlab
   'V_T.cDC1 → V_T.mcDC1'  at rate  'k_CD40_mat * V_T.cDC1 * H_aCD40'
   'V_T.cDC2 → V_T.mcDC2'  at rate  'k_CD40_mat * V_T.cDC2 * H_aCD40'
   ```

3. **Macrophage tumoricidal activation:** CD40 agonism reprograms M2 macrophages toward a tumoricidal phenotype (distinct from classical M1):

   ```matlab
   'V_T.Mac_M2 → V_T.Mac_M1'  at rate  'k_CD40_repol * V_T.Mac_M2 * H_aCD40'
   ```

4. **Stromal remodeling:** Activated macrophages degrade ECM (Beatty et al. 2011, macrophage-dependent stromal depletion):

   ```matlab
   'V_T.ECM → null'  at rate  'k_ECM_Mac_degrade * V_T.Mac_M1 * H_aCD40'
   ```

**Topological impact.** Creates an alternative entry point into the immunity cycle that bypasses T cell priming entirely. The macrophage-mediated stromal depletion effect also creates a novel connection: CD40 → macrophage activation → ECM degradation → improved T cell infiltration, which could synergize with checkpoint blockade in a two-step strategy (CD40 first to "open" the stroma, then anti-PD-1 to sustain the T cell response).

**Impact on transition analysis.** CD40 agonism could shift the immune activation threshold (Transition 3) by providing a priming-independent kick. The question of whether CD40 alone can sustain a response (vs. requiring follow-up with checkpoint blockade) maps to whether the macrophage-mediated killing arm can generate enough DAMPs/antigen to bootstrap the T cell-dependent immunity cycle.

**Feasibility.** Moderate complexity. Follows established therapy module patterns. ~6-8 new parameters (PK, maturation rate, repolarization rate, ECM degradation rate). The APC maturation and macrophage polarization reactions already exist; this adds a drug-driven parallel pathway.

**Literature.** Beatty et al. (2011, CD40 agonism in PDAC, PMID: 21406402); Morrison et al. (2020, CD40 clinical trials in PDAC, PMID: 32439860); Byrne & Vonderheide (2016, CD40 in cancer immunotherapy, PMID: 26658536); O'Hara et al. (2021, sotigalimab + gem/nabP + nivolumab in PDAC, PMID: 34314688).

---

## Rank 13: IDO/Tryptophan Metabolism

**What's missing.** Indoleamine 2,3-dioxygenase (IDO1) is an enzyme expressed by tumor cells, tolerogenic DCs, MDSCs, and some CAFs that catabolizes tryptophan to kynurenine. Tryptophan depletion inhibits T cell proliferation and effector function (amino acid starvation via GCN2 kinase), while kynurenine directly promotes Treg differentiation via AhR signaling. In PDAC, IDO1 is overexpressed and correlates with poor prognosis (Witkiewicz et al. 2008).

This creates an immunosuppressive axis orthogonal to the existing glucose/lactate metabolic competition (Rank 5): glucose depletion impairs T cell *function* (killing, cytokine production), while tryptophan depletion impairs T cell *proliferation* (cell division arrest).

**What to add.**

1. **Tryptophan species** (`V_T.Trp`): QSS algebraic parameter (same pattern as `glc`). Supply from vasculature (K-dependent), consumed by IDO1 activity (proportional to IDO-expressing cells).

2. **Kynurenine species** (`V_T.Kyn`): Produced by IDO1 activity, degraded at first order.

3. **IDO1 activity:** `IDO_activity = k_IDO_tumor * C_total + k_IDO_MDSC * V_T.MDSC + k_IDO_DC * V_T.mcDC2`. IFNg upregulates IDO1 (adaptive resistance): multiply by `(1 + f_IFNg_IDO * H_IFNg)`.

4. **Tryptophan-dependent T cell proliferation:** Multiply CD8 and Th proliferation rates by `H_Trp = Trp^n / (Trp_50^n + Trp^n)`.

5. **Kynurenine-driven Treg induction:** Enhance Th→Treg conversion by `(1 + f_Kyn_Treg * H_Kyn)`.

**Topological impact.** Creates a parallel metabolic immunosuppressive axis alongside glucose/lactate. Also creates an IFNg-IDO negative feedback: T cell activation → IFNg → IDO upregulation → tryptophan depletion → T cell proliferation arrest. This is an adaptive resistance mechanism — the immune system's own activation triggers its metabolic suppression.

**Impact on transition analysis.** The IFNg-IDO negative feedback could create a "metabolic ceiling" on immune responses: even when the immunity cycle is closed and CXCL9/10 recruitment is active, IDO-mediated tryptophan depletion prevents T cell expansion beyond a certain density. This would manifest as a partial response plateau rather than complete tumor eradication — a common clinical observation in PDAC.

**Feasibility.** Moderate complexity. 1-2 new species (Trp as QSS, Kyn as ODE), ~6-8 new parameters. Same implementation pattern as `metabolism_module.m` (algebraic nutrient + downstream Hill functions). Could be integrated into metabolism_module or a standalone `IDO_module.m`.

**Literature.** Witkiewicz et al. (2008, IDO in PDAC, PMID: 18843041); Munn & Mellor (2016, IDO and immune tolerance, PMID: 27396447); Prendergast et al. (2017, IDO pathway in cancer, PMID: 28303904); Long et al. (2019, epacadostat failure and lessons, PMID: 30753480).

---

## Rank 14: Neutrophils and Neutrophil Extracellular Traps (NETs)

**What's missing.** Tumor-associated neutrophils (TANs) and neutrophil extracellular traps (NETs) play significant roles in PDAC progression:

- **NETs form a physical barrier** that traps T cells and prevents their contact with tumor cells (Teijeira et al. 2020), complementing the ECM barrier from myCAFs.
- **NETs promote metastasis** by trapping circulating tumor cells and creating pre-metastatic niches (Cools-Lartigue et al. 2013).
- **TANs have N1/N2 polarization** (analogous to M1/M2 macrophages): N1 neutrophils are anti-tumoral (produce ROS, phagocytose), while N2 neutrophils are pro-tumoral (produce NETs, secrete MMP-9 for ECM remodeling, suppress T cells via Arg-I). TGFb drives N1→N2 polarization.
- **The neutrophil-to-lymphocyte ratio (NLR)** is one of the strongest prognostic biomarkers in PDAC (>5 predicts poor survival).

**What to add.**

1. **Neutrophil species** (`V_T.Neut`): Recruited by CXCL8/IL-8 (from tumor cells, macrophages) and CXCL2 (from tumor cells). TGFb-dependent N1/N2 polarization could be simplified as a functional suppression similar to NK/macrophage modeling.

2. **NET formation:** `V_T.NET` species produced by N2 neutrophils under inflammatory conditions. NETs contribute to physical barrier (add to ECM-like exclusion of T cells) and degrade over time (DNase clearance).

3. **T cell trapping by NETs:** Hill function `H_NET = V_T.NET / (NET_50 + V_T.NET)` multiplied into T cell trafficking/motility, similar to ECM effects but mechanistically distinct (DNA mesh trapping vs. collagen exclusion).

4. **Arg-I production:** N2 neutrophils produce arginase-I (links to existing MDSC suppression pathway if Arg-I is shared).

**Topological impact.** Adds a parallel physical barrier pathway to ECM/myCAF. The N1/N2 polarization mirrors M1/M2 and is driven by the same signals (TGFb), reinforcing existing immunosuppressive loops. NETs create a novel connection between acute inflammation (neutrophil recruitment) and immune exclusion (T cell trapping) — paradoxically, inflammation-driven neutrophil influx can *worsen* T cell access.

**Impact on transition analysis.** NLR as a composite biomarker would emerge naturally from the model. The NET barrier adds another layer to the infiltration threshold, but largely overlaps mechanistically with existing ECM effects. The most novel contribution is the inflammation → NET → exclusion paradox, which could create a negative feedback where immune activation recruits neutrophils that then impede further T cell infiltration.

**Feasibility.** High complexity. 2-3 new species (Neut, NETs, possibly IL-8), new recruitment/polarization/death reactions, NET formation and clearance dynamics, T cell trapping interactions. ~10-12 new parameters. Substantial overlap with existing macrophage/MDSC suppression mechanisms.

**Literature.** Teijeira et al. (2020, NETs trap T cells, PMID: 32877058); Cools-Lartigue et al. (2013, NETs and metastasis, PMID: 23644512); Fridlender et al. (2009, N1/N2 neutrophil polarization, PMID: 19805653); Jin et al. (2019, neutrophils in PDAC, PMID: 31405505).

---

## Rank 15: B Cells and Tertiary Lymphoid Structures (Gap 6, partial)

*Moved from original Rank 7. Renumbered to reflect expanded list.*

**What's missing.** B cells and organized tertiary lymphoid structures (TLS) are increasingly recognized as positive prognostic factors in PDAC. TLS-positive PDAC tumors have better outcomes across treatment modalities.

- **B cells:** Produce antibodies that can opsonize tumor cells for phagocytosis/ADCC (links to macrophages and potential NK cell addition). Regulatory B cells (Bregs) produce IL-10 and TGFb.
- **TLS:** Organized lymphoid aggregates within the tumor that provide local antigen presentation and T cell priming, bypassing the need for trafficking to draining lymph nodes.

**What to add.**

1. **B cell species** (`V_T.Bcell`, `V_LN.Bcell`): Activated by antigen-presenting cDC2s, produce antibodies.
2. **Antibody species** (`V_T.IgG`): Opsonizes tumor cells, enhancing phagocytosis (existing `phagocytosis_module.m`) and ADCC (via NK cells, Rank 4).
3. **TLS as a local priming compartment:** A simplified additional "lymphoid" compartment within the tumor that allows local T cell priming without trafficking to V_LN. TLS formation driven by CXCL13 (produced by activated B cells and follicular DCs).

**Topological impact.** Creates an alternative, local path through the cancer immunity cycle that does not require lymph node trafficking. This short-circuits the long multi-compartment loop (tumor → LN → peripheral → tumor) with a local loop (tumor killing → antigen → local TLS priming → CD8 expansion → killing). The local loop would be faster but weaker (fewer APCs, less co-stimulation).

**Impact on transition analysis.** TLS presence could create a second immune activation threshold — a lower bar for local immune engagement that does not require the full systemic immunity cycle. Patients with TLS would have a broader basin of attraction for the immune-active state, making them easier to tip with immunotherapy.

**Feasibility.** High complexity. Adds 3-5 new species, a potential new compartment (TLS), and requires interaction with both the APC/DC system and T cell priming. This is the most architecturally complex addition and should be considered only after Ranks 1 and 6 are implemented. Requires 10-15 new parameters.

**Literature.** Helmink et al. (2020, B cells and TLS in immunotherapy response); Cabrita et al. (2020, TLS in cancer immunotherapy); Castino et al. (2016, TLS in PDAC prognosis); Schumacher & Thommen (2022, TLS review).

---

## Summary Table

| Rank | Addition | Status | New Species | New Params | Loops Closed/Created | Key Transition Affected |
|------|----------|--------|-------------|------------|---------------------|------------------------|
| 1 | Cancer immunity cycle | **DONE** | 0 | 3-5 | Closes the central adaptive immunity loop | Transition 3 (immune activation threshold) |
| 2 | iCAF/myCAF/apCAF split | **DONE** | 3 | ~22 | Splits fibroblast loops into immunosuppressive vs barrier circuits | Transition 6 (desmoplastic irreversibility) |
| 3 | Hypoxia–HIF–VEGF | **DONE** | 0 (algebraic) | 8 | Creates vascular positive feedback + HIF-driven immunosuppression | Transition 1 (response threshold), new phase axis |
| 4 | NK cells + XCL1→cDC1 | **DONE** | 3 | 10 | Creates antigen-independent innate killing loop + cDC1 recruitment | Transition 3 (adds basal immune pressure floor) |
| 5 | Glucose/lactate | **DONE** | 1 (+ algebraic) | 9 | Creates metabolic competition loop (tumor vs immune) | Transition 3 (size-dependent immune threshold) |
| 6 | cDC1/cDC2 split | **DONE** | 9 | 6 | Creates cross-presentation bottleneck in immunity cycle | Transition 3 (cDC1-dependent immune activation) |
| 6b | CXCL9/10 T cell chemotaxis | **DONE** | 1 | 6 | Creates trafficking-level positive feedback (IFNg→CXCL9→T cell recruitment) | Transition 3 (infiltration threshold, hot vs cold) |
| 8 | CCL2-driven APC recruitment | **DONE** | 0 | 3 | Makes APC recruitment inflammation-dependent | Transition 3 (mild sharpening) |
| 9 | Adenosine (CD73/CD39/A2AR) | **TODO** | 1 | 6-8 | Creates death trap negative feedback (killing → adenosine → suppression) | Transition 3 (futile cycling threshold) |
| 10 | MHC-I downregulation | **TODO** | 0-1 | 2-4 | Creates CD8/NK tradeoff, optional immune editing | Transition 3 (immune editing threshold) |
| 11 | Gemcitabine | **TODO** | 1 (PK) | 8-10 | MDSC depletion indirect positive feedback | Transition 3 (chemo opens immune window) |
| 12 | CD40 agonism | **TODO** | 1 (PK) | 6-8 | Priming-independent macrophage activation + stromal depletion | Transition 3 (alternative activation pathway) |
| 13 | IDO/tryptophan | **TODO** | 1-2 | 6-8 | IFNg-IDO negative feedback (metabolic ceiling on immune response) | Transition 3 (response plateau) |
| 14 | Neutrophils/NETs | **TODO** | 2-3 | 10-12 | Inflammation → NET → T cell trapping paradox | Transition 3 (infiltration), NLR as biomarker |
| 15 | B cells + TLS | **TODO** | 3-5 | 10-15 | Creates local immunity cycle bypass | Transition 3 (local vs systemic immune activation) |

---

## Recommended Implementation Order (Updated Feb 2026)

Ranks 1–6 are complete. The following prioritizes remaining additions.

**Phase 5 (next, high impact, low complexity):**
- **Rank 6b: CXCL9/10 chemotaxis.** Highest-priority remaining gap. Creates the trafficking-level positive feedback that distinguishes immune-hot from immune-cold tumors. Low complexity (1 species, 3-4 params). Directly enables predictions about checkpoint blockade response.
- **Rank 8: CCL2-driven APC recruitment.** Already spec'd in the changelog. Very low complexity (0 new species, 3 params). Makes APC dynamics mechanistic rather than homeostatic.

**Phase 6 (near-term, moderate complexity, novel biology):**
- **Rank 9: Adenosine pathway.** Connects to existing hypoxia module. Creates the first negative feedback where increased killing can paradoxically increase suppression — a fundamentally different dynamic from anything currently in the model. Multiple clinical trials provide calibration data.
- **Rank 10: MHC-I downregulation.** Low complexity for the static version. Creates the CD8/NK tradeoff that makes the NK module (Rank 4) fully meaningful.

**Phase 7 (medium-term, therapy modules for clinical scenarios):**
- **Rank 11: Gemcitabine.** Essential for modeling standard-of-care treatment. Required before any chemo-immunotherapy combination predictions. Follows established PK module patterns.
- **Rank 12: CD40 agonism.** Emerging therapy with strong PDAC data (Beatty 2011). Particularly valuable for exploring immune activation in cold tumors where the immunity cycle is broken.

**Phase 8 (long-term, high complexity or diminishing returns):**
- **Rank 13: IDO/tryptophan.** Adds a metabolic ceiling on immune responses. Lower priority because the epacadostat trial failure reduced clinical urgency, but mechanistically important for understanding partial response plateaus.
- **Rank 14: Neutrophils/NETs.** High complexity with substantial overlap with existing ECM barrier mechanics. Most novel contribution (inflammation → NET → exclusion paradox) may not justify the parameter burden.
- **Rank 15: B cells/TLS.** Architecturally complex. Most impactful for GVAX response prediction (TLS formation after vaccination) but requires Ranks 6b and 8 first.

---

## Guiding Principle

Each addition should be validated against the feature-based transition analysis (Idea 28) before and after implementation. The question is not just "does the addition make the model more realistic?" but "does it create, sharpen, or move a qualitative transition that matters clinically?" An addition that changes quantitative outputs without affecting the transition landscape is low-priority regardless of biological realism. An addition that creates a new qualitative transition (e.g., the metabolic threshold for immune activation) or sharpens an existing one (e.g., the immunity cycle making the immune activation threshold crisper) is high-priority even if it adds only modest quantitative accuracy.
