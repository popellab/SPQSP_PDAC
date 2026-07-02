---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 24px;
  }
  h1 {
    font-size: 38px;
    color: #1a5276;
  }
  h2 {
    font-size: 32px;
    color: #2c3e50;
  }
  section.title {
    text-align: center;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }
  section.title h1 {
    font-size: 48px;
  }
  section.title h2 {
    font-size: 28px;
    color: #666;
    font-weight: normal;
  }
  section.two-col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto 1fr;
    gap: 0 30px;
  }
  section.two-col h1 {
    grid-column: 1 / -1;
  }
  table {
    font-size: 18px;
  }
  img {
    display: block;
    margin: 0 auto;
  }
  .small {
    font-size: 18px;
  }
  .footnote {
    font-size: 14px;
    color: #888;
    position: absolute;
    bottom: 30px;
  }
  blockquote {
    border-left: 4px solid #2980b9;
    padding-left: 16px;
    font-style: italic;
    color: #555;
  }
---

<!-- _class: title -->

# QSP Model for PDAC Immunotherapy
## Mechanistic Buildout, Calibration Targets & Strategy

Joel Eliason
Lab Meeting -- February 19, 2026

---

# Outline

1. **Model overview** -- what are we modeling and why?
2. **New mechanisms** -- 7 modules added since last update
3. **Cascade figures** -- mechanistic walkthroughs
4. **Calibration targets** -- 50 extracted with full provenance
5. **Initialization strategy** -- evolve-to-diagnosis
6. **Prior cleanup & next steps** -- SA, SBI calibration

---

# Background: PDAC Immunotherapy Challenge

- Pancreatic ductal adenocarcinoma has near-universal resistance to immunotherapy
  - Anti-PD-1 monotherapy: **0% ORR** across multiple trials
- Causes are multifactorial and interconnected:
  - Dense desmoplastic stroma (physical barrier)
  - Immunosuppressive TME (Tregs, M2 TAMs, MDSCs)
  - Metabolic hostility (hypoxia, glucose depletion, lactate)
  - Poor antigen presentation (cDC1 paucity)
  - T cell exclusion (CXCL12 from CAFs)
- **GVAX + nivolumab** is one of few combinations showing signal (Zheng 2022: 2.8x CD8 increase)

---

# Model Architecture

**~470 parameters** across interconnected modules in a SimBiology framework (~174 varied in priors)

| Compartment | Role |
|---|---|
| V_C (central) | Systemic drug PK, circulating immune cells |
| V_P (peripheral) | Drug distribution |
| V_LN (lymph node) | T cell priming, APC-mediated activation |
| V_T (tumor) | Tumor-immune dynamics, TME |
| V_ID (injection) | GVAX vaccine site |

Therapy: GVAX (GM-CSF-secreting tumor vaccine) + nivolumab (anti-PD-1) + cyclophosphamide (Treg depletion)

---

# Implemented Mechanisms (Ranked by Priority)

| Rank | Module | Key Biology |
|---|---|---|
| 1 | Cancer immunity cycle | DAMP-mediated APC activation, killing-dependent antigen release |
| 2 | CAF heterogeneity | iCAF / myCAF / apCAF subtypes with distinct secretomes |
| 3 | Hypoxia-HIF-VEGF | pO2-driven HIF, PD-L1 upregulation, M2 polarization |
| 4 | NK cells | Antigen-independent killing, XCL1-cDC1 innate bridge |
| 5 | Metabolic competition | Glucose depletion, lactate accumulation, immune suppression |
| 6 | DC subtypes | cDC1 (cross-presenting, CD8) vs cDC2 (Th/Treg priming) |
| 6b | CXCR3 chemotaxis | IFNg-driven CXCL9 production, CD8/NK recruitment |
| -- | CCL5/Treg recruitment | iCAF-derived CCL5, Treg trafficking via CCR5 |

---

# Model Overview: Cascade Summary (GVAX + Nivolumab)

![h:500](../figures/cascade_summary_gvax_nivo.png)

<div class="footnote">Baseline (gray dashed) vs GVAX+nivo (blue solid): treatment input, APC activation, T cell infiltration, immune balance, stromal barrier, tumor burden</div>

---

# The PDAC Tumor Microenvironment

Unlike most solid tumors, PDAC is **~85% stroma and only ~15% cancer cells**

```
  Normal Pancreas              PDAC Tumor
  ┌─────────────┐         ┌─────────────────┐
  │ ○ ○ ○ ○ ○ ○ │         │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
  │ ○ ○ ○ ○ ○ ○ │         │ ▓▓▓ ECM ▓▓▓▓▓▓▓ │
  │ ○ acinar  ○ │   ──>   │ ▓▓ + CAFs ▓▓●▓▓ │
  │ ○ ○ ○ ○ ○ ○ │         │ ▓▓▓▓▓▓▓●▓▓▓▓▓▓▓ │
  │ ○ ○ ○ ○ ○ ○ │         │ ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ │
  └─────────────┘         └─────────────────┘
                           ● = cancer cells
                           ▓ = desmoplastic stroma
```

This dense stroma creates multiple overlapping barriers to immune attack.

<!-- The key insight is that PDAC's immunotherapy resistance isn't just one mechanism -- it's a fortress with multiple walls. -->

---

# TME: Layered Immune Evasion

Each barrier reinforces the others -- breaking one alone is insufficient:

```
  Physical Barriers          Cellular Suppression       Metabolic Hostility
  ─────────────────          ────────────────────       ───────────────────
  ECM (5-20x normal)        M2 TAMs >> M1              Hypoxia (pO2 ~2-8 mmHg)
       │                    Tregs (25-35% of CD4+)     Glucose depleted
       ▼                    MDSCs                      Lactate accumulated
  Blocks T cell motility         │                          │
  Blocks APC migration           ▼                          ▼
                            Active suppression         Impairs T cell + NK function
                            of effector cells          Drives M2 polarization
```

> Anti-PD-1 alone gives 0% ORR in PDAC. Checkpoint blockade doesn't address any of these.

<!-- PDAC has the most immunosuppressive TME of any common solid tumor. Each layer of defense is independently sufficient to blunt an immune response. -->

---

# CAF Subtypes: Not All Fibroblasts Are Equal

Quiescent pancreatic stellate cells (qPSC) activate into **three distinct subtypes**:

```
              ┌─────────────────────────────────────┐
              │          qPSC (quiescent)           |
              └─────┬──────────┬──────────┬─────────┘
              TGFb  │    PDGF  │    IL-1  │
                    ▼          ▼          ▼
              ┌──────────┐ ┌────────┐ ┌──────────┐
              │  myCAF   │ │ apCAF  │ │   iCAF   │
              │  (~60%)  │ │ (~5%)  │ │  (~35%)  │
              └────┬─────┘ └───┬────┘ └────┬─────┘
                   │           │           │
            ECM, CXCL12     MHC-II    IL-6, CCL5
            TGFb, VEGF   (tolerogenic) (inflammatory)
                   │           │           │
                   ▼           ▼           ▼
             Physical       Treg        Treg
             barrier      induction   recruitment
```

<!-- myCAFs build the wall, iCAFs recruit Tregs to guard it, apCAFs convert naive T cells into Tregs. -->

---

# CAF Subtypes: Why This Matters for the Model

- **myCAFs** build the physical fortress (ECM) and secrete CXCL12 to exclude T cells
- **iCAFs** recruit Tregs via CCL5/CCR5 chemotaxis
- **apCAFs** directly induce Tregs through MHC-II antigen presentation

Subtypes are **interconvertible** -- not fixed fates:
- TGFb pushes iCAF -> myCAF (more barrier)
- IL-1/TNFa pushes myCAF -> iCAF (more inflammation)

> Targeting one subtype may just shift the balance toward another, not eliminate CAFs.

<!-- This is why we track all three subtypes + qPSC pool rather than lumping them. The dynamics between subtypes matter. Elyada 2019, Ohlund 2017. -->

---

# Metabolic Barriers: The Warburg Effect

PDAC tumors consume glucose and produce lactate at extreme rates:

```
  Normal Tissue                    PDAC Tumor
  ─────────────                    ──────────
  Glucose: ~5 mM       ──>        Glucose: 0.5-2 mM
  Lactate: ~1.5 mM                Lactate: 10-20 mM
  pO2: 40-50 mmHg                 pO2: 2-8 mmHg
  pH: ~7.4                        pH: 6.5-6.8
```

> Cancer cells outcompete immune cells for the same metabolic resources.

<div class="footnote">Battini 2017 (HR-MAS NMR), Koay 2020 (FDG-PET)</div>

<!-- The Warburg effect isn't just a cancer cell adaptation -- it actively creates an environment hostile to immune function. -->

---

# Metabolic Barriers: How This Suppresses Immunity

Two independent mechanisms, both captured in the model:

**Glucose depletion impairs T cells** (Chang 2015):
- GAPDH moonlights as an RNA-binding protein
- When glucose is scarce, GAPDH binds IFNg mRNA instead of metabolizing glucose
- Result: T cells can't produce their key effector cytokine

**Lactate drives immunosuppression** (Colegio 2014, Brand 2016):
- Macrophages take up lactate via MCT transporters -> stabilizes HIF -> M2 polarization
- T cells: intracellular acidification -> impaired NFAT signaling -> reduced killing

```
  ┌────────────┐    glucose↓    ┌─────────────┐    lactate↑   ┌──────────┐
  │   Larger   │ ────────────>  │   T cell    │ <──────────── │    M2    │
  │   tumor    │                │ dysfunction │               │ polariz. │
  └─────┬──────┘                └──────┬──────┘               └──────────┘
        │                              │
        └──── less immune control ◄────┘
```

<!-- This is a positive feedback loop: bigger tumors create worse metabolic environments, which further impair immunity, which allows more tumor growth. -->

---

# NK-cDC1 Innate Bridge: The Loop

A positive feedback loop linking innate and adaptive immunity:

```
         ┌──────────────────────────────────────────┐
         │                                          │
         ▼                                          │
    ┌─────────┐   XCL1    ┌─────────┐   IL-12   ┌───┴─────┐
    │  NK     │ ────────> │  cDC1   │ ────────> │  NK     │
    │  cells  │           │ recruit │           │ activ.  │
    └─────────┘           └────┬────┘           └─────────┘
                               │
                          cross-present
                          antigen to CD8
                               │
                               ▼
                         ┌──────────┐
                         │ Adaptive │
                         │ immunity │
                         └──────────┘
```

> NK cells recruit the very DCs that activate both NK cells AND CD8 T cells.

<!-- Bottcher 2018 first described this loop. Trefny 2024 showed its clinical relevance. -->

---

# NK-cDC1 Bridge: Why It Matters in PDAC

**The problem:** Both halves of this loop are broken in PDAC.

- NK cells: **<0.5%** of TILs (Lim 2019) -- suppressed by TGFb, IL-10, MDSCs
- cDC1: **~100x fewer** than TAMs (Hegde 2020) -- the key antigen-presenting bottleneck

**Why we model it:**
- cDC1 are the **only** DCs that cross-present antigen to CD8 T cells
- Without cDC1, even perfect T cell priming in the lymph node can't translate to tumor killing
- This loop is a potential **therapeutic amplification point** -- small boosts could cascade

> NK-XCL1-cDC1-IL-12 creates a bistable switch. Below a threshold, the loop collapses. Above it, innate and adaptive immunity reinforce each other.

<!-- This is why we added the NK module despite NK cells being rare -- their indirect effect through cDC1 recruitment may be more important than their direct killing. -->

---

# Calibration Targets: Overview

**50 extracted targets** with full provenance (DOI, value snippets, experimental context)

| Category | Count | Examples |
|---|---|---|
| Baseline cell densities | 7 | CD8, CD4, Treg, TAM, cDC1, MDSC, total TIL |
| Tumor & stromal metrics | 5 | Cancer cell density, stromal fraction, collagen, doubling time |
| Immune cell ratios | 11 | CD8/Treg, M1/M2, exhaustion fractions, CAF subtypes |
| CAF subtypes | 4 | iCAF, myCAF, apCAF fractions, PSC activation |
| TME properties | 4 | pO2, HIF, VEGF, glucose/lactate |
| GVAX+nivo treatment (d21) | 18 | Cell percentages, fold-changes, M1/M2 shift |
| Clinical outcomes | 1 | Major pathologic response rate |

<div class="footnote">+ 4 clinical outcome targets in input CSV (ORR, OS endpoints) + 12 deferred cytokine targets</div>

---

# Calibration Targets: Extraction Approach

Targets extracted via **LLM-powered workflow** (`qsp-extract`), then curated interactively with Claude Code:

- **Source paper** with DOI, sample size, experimental context
- **Value snippet** -- exact text from paper documenting the measurement
- **Observable mapping** -- Python code mapping model species to experimental measurement
- **Unit handling** -- Pint-aware conversions (cells/mm^2 -> cells/mL, etc.)
- **Uncertainty** -- Monte Carlo propagation from reported CI/SD to lognormal/beta fits
- **Reference values** -- shared constants database (cell diameters, tissue densities, MW)

Key sources: Jansen 2021 (n=444), Sivagnanam 2021 (n=113), Liudahl 2021 (n=135), Zheng 2022 (n=10), Elyada 2019, Battini 2017

---

# Calibration Targets: Baseline vs Treatment

<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 30px;">
<div>

**Baseline (diagnosis, day 0):**
- Cell densities (IHC, cells/mm^2)
- Ratios & fractions (dimensionless)
- Stromal composition
- Metabolic state (glucose, lactate, pO2)
- Validates TME steady state

</div>
<div>

**GVAX + Nivo (day 21):**
- CD8 fold-increase (2.8x, Zheng 2022)
- Immune cell % in non-lymphoid aggregate areas
- GVAX alone vs GVAX+nivo (paired)
- M1/M2 ratio shift
- Treg fraction reduction
- Validates treatment response dynamics

</div>
</div>

---

# Initialization: Evolve-to-Diagnosis Strategy

**Problem:** Diagnosis-state cell densities can't be directly parameterized -- they emerge from rate parameters.

**Solution:** Three-phase initialization workflow:

| Phase | State | Tumor Size | Duration |
|---|---|---|---|
| 1. Healthy | Microinvasive PDAC | 1e6 cells (~1.7 mm) | t = 0 |
| 2. Evolution | Natural history | Growing | ~365 days |
| 3. Diagnosis | Established TME | **3.2 cm** (median resectable) | Stopping point |

- Homeostatic starting state: M1-skewed macrophages, minimal ECM, mostly quiescent PSCs
- All modules evolve together: immune infiltration, fibrosis, metabolic shift
- Diagnosis state emerges **mechanistically** from rate parameters
- 3.2 cm = median resectable PDAC diameter at diagnosis (fixed, not varied)

---

# Initialization: Why This Matters

**Advantages of evolve-to-diagnosis:**

1. **Decouples ICs from calibration targets** -- diagnosis-state populations are model *outputs*, not inputs
2. **Mechanistic consistency** -- all species reach diagnosis state through the same dynamics
3. **Sensitivity analysis** -- can ask "which parameters drive progression from microinvasive to diagnosis?"
4. **Identifiability** -- fixing tumor diameter at 3.2 cm removes a degree of freedom

---

# Current Work: Prior Cleanup

Ensuring all **174 varied parameters** in `pdac_priors.csv` have sensible distributions:

- Medians match `parameters_PDAC.m` defaults (automated test)
- Ranges span biologically plausible values
- Distributions appropriate (lognormal for rates, beta for fractions)
- New module parameters included with literature-backed ranges
- `initial_tumour_diameter` removed from priors (fixed at 3.2 cm)

**Goal:** Priors ready for Morris screening and SBI calibration

---

# Calibration Strategy: 6-Phase Workflow

| Phase | Task | Status |
|---|---|---|
| 1. Target extraction | Design & extract calibration targets with provenance | **Done** (50 targets) |
| 2. Submodel targets | Isolated-system targets for individual parameters | Planned |
| 3. Sensitivity screening | Morris/Sobol to identify active parameters | **Next** |
| 4. Modular SBI | Decompose into ~8 calibration blocks | Planned |
| 5. Joint refinement | Final SBI on top 30-40 parameters | Planned |
| 6. Validation | Prior/posterior predictive checks, negative controls | Planned |

> ~40-60 of ~470 parameters expected to be identifiable. Morris screening will partition active vs inert before SBI.

---

# Next Steps

1. **Finish prior cleanup** -- ensure all priors biologically grounded
2. **Morris sensitivity analysis** -- identify which parameters matter for each target
3. **Modular SBI calibration** -- calibrate parameter blocks against matched targets
4. **Negative controls** -- anti-PD-1 monotherapy should produce 0% ORR
5. **Remaining mechanisms** (lower priority):
   - CCL2-driven APC recruitment
   - Adenosine pathway (CD73/A2AR)
   - MHC-I downregulation
   - Gemcitabine chemotherapy

---

# Cascade Figures: Treatment Input & APC Activation

![h:500](../figures/cascade_input_apc.png)

<div class="footnote">GVAX vaccine cells -> GM-CSF -> antigen release -> APC recruitment & maturation -> LN migration</div>

---

# Cascade Figures: T Cell Activation & Infiltration

![h:500](../figures/cascade_tcell_activation.png)

<div class="footnote">mcDC1 cross-presentation -> CD8 activation (per-APC rate) -> clonal burst -> tumor infiltration (opposed by CXCL12)</div>

---

# Cascade Figures: Mechanistic Tumor Killing

![h:500](../figures/cascade_tcell_killing.png)

<div class="footnote">T cell density + search volume -> contact-based killing, modulated by ECM motility block & Treg suppression</div>

---

# Cascade Figures: Checkpoint & Immunosuppression

![h:500](../figures/cascade_checkpoint.png)

<div class="footnote">PD-1/PD-L1 checkpoint, nivolumab blockade, macrophage polarization balance, MDSC accumulation</div>

---

# Cascade Figures: CAF Subtypes & Desmoplasia

![h:500](../figures/cascade_caf_subtypes.png)

<div class="footnote">qPSC -> myCAF (ECM/CXCL12) + iCAF (IL-6/CCL5) + apCAF (Treg induction); ECM blocks T cell & APC motility</div>

---

# Cascade Figures: Hypoxia & Metabolic TME

![h:500](../figures/cascade_hypoxia_metabolism.png)

<div class="footnote">pO2 -> HIF -> VEGF feedback; glucose depletion impairs T cell function; lactate drives M2 polarization</div>

---

# Cascade Figures: NK Innate-Adaptive Bridge

![h:500](../figures/cascade_nk_innate.png)

<div class="footnote">NK -> XCL1 -> cDC1 recruitment -> IL-12 -> NK activation (positive feedback loop)</div>

---

# Cascade Figures: Chemokine Trafficking Networks

![h:500](../figures/cascade_chemokine_trafficking.png)

<div class="footnote">CCL5 (Treg recruitment), CXCL9 (CD8/NK recruitment via IFNg), CXCL12 (T cell exclusion), DAMP-APC maturation</div>

---

<!-- _class: title -->

# Questions?

Model code: `pdac-build` repository
Calibration targets: `calibration_targets/` (50 YAMLs)
Cascade figures: `figures/cascade_*.png`
