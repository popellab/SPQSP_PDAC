---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
  :root {
    --color-bg: #fdfdfd;
    --color-fg: #1e1e2e;
    --color-accent: #2563eb;
    --color-muted: #64748b;
    --color-highlight: #dc2626;
    --color-surface: #f1f5f9;
  }
  section {
    background: var(--color-bg);
    color: var(--color-fg);
    font-family: 'Inter', 'Helvetica Neue', sans-serif;
    font-size: 28px;
    padding: 50px 70px;
    line-height: 1.5;
  }
  h1 {
    font-weight: 700;
    font-size: 44px;
    color: var(--color-fg);
    margin-bottom: 0.3em;
    letter-spacing: -0.02em;
  }
  h2 {
    font-weight: 600;
    font-size: 32px;
    color: var(--color-muted);
    margin-top: -0.2em;
  }
  h3 {
    font-weight: 600;
    font-size: 26px;
    color: var(--color-accent);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5em;
  }
  strong {
    font-weight: 600;
    color: var(--color-fg);
  }
  em {
    font-style: normal;
    color: var(--color-highlight);
    font-weight: 600;
  }
  ul, ol {
    padding-left: 1.2em;
  }
  li {
    margin-bottom: 0.4em;
  }
  li::marker {
    color: var(--color-accent);
  }
  blockquote {
    background: var(--color-surface);
    border-left: 4px solid var(--color-accent);
    border-radius: 0 8px 8px 0;
    padding: 0.8em 1.2em;
    margin: 1em 0;
    font-size: 26px;
    color: var(--color-muted);
  }
  blockquote strong {
    color: var(--color-fg);
  }
  table {
    font-size: 22px;
    border-collapse: collapse;
    width: 100%;
    margin: 0.5em 0;
  }
  th {
    background: var(--color-accent);
    color: white;
    font-weight: 600;
    padding: 10px 16px;
    text-align: left;
  }
  td {
    padding: 10px 16px;
    border-bottom: 1px solid #e2e8f0;
  }
  tr:nth-child(even) td {
    background: var(--color-surface);
  }
  code {
    background: var(--color-surface);
    padding: 0.15em 0.4em;
    border-radius: 4px;
    font-size: 0.9em;
  }
  pre {
    background: var(--color-surface);
    padding: 1em;
    border-radius: 8px;
    font-size: 20px;
  }
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 2em;
    align-items: start;
  }
  section.lead {
    display: flex;
    flex-direction: column;
    justify-content: center;
    text-align: center;
    padding: 60px 100px;
  }
  section.lead h1 {
    font-size: 52px;
    line-height: 1.2;
  }
  section.lead h2 {
    font-weight: 300;
    font-size: 28px;
    color: var(--color-muted);
    margin-top: 0.5em;
  }
  section.lead h3 {
    text-transform: none;
    font-size: 22px;
    color: var(--color-muted);
    font-weight: 400;
    letter-spacing: 0;
  }
  .dim {
    color: var(--color-muted);
  }
  .small {
    font-size: 20px;
    color: var(--color-muted);
  }
---

<!-- add figure of model -->
<!-- highlight new model additions -->

<!-- _class: lead -->
<!-- _paginate: false -->

# PDAC QSP Model Update
## Stroma, ECM, and calibration from literature

### Joel Eliason &nbsp;|&nbsp; March 25, 2026

---

# Model Architecture: Tumor & Immune

<div class="cols">
<div>

**Tumor**
- Cancer growth (Gompertzian)
- Antigen release from killing and turnover

**Adaptive Immunity**
- CD8+ T cell priming, trafficking, killing
- CD4+ Th cells + Tregs (CCL5-recruited)
- APC maturation (cDC1/cDC2 subtypes)

</div>
<div>

**Innate Immunity**
- Macrophages (M1/M2 polarization)
- MDSCs

**Cytokine Network**
- IL-1, TGFb, IL-6, IL-2, IFNg
- CXCL12, CCL5, CCL2

</div>
</div>

~194 parameters &nbsp;|&nbsp; 4 treatment scenarios &nbsp;|&nbsp; 87 literature-extracted calibration targets

---

# Model Architecture: Stroma & Treatment

<div class="cols">
<div>

**Fibroblast / CAF Subtypes**
- iCAF (inflammatory, IL-1-driven)
- myCAF (myofibroblastic, TGFb-driven)
- apCAF (antigen-presenting, Treg-inducing)
- IL-1/TGFb-governed interconversion

**ECM Barrier**
- Collagen volume fraction (phi)
- Pore-size-based immune exclusion
- Stiffness-mediated T cell exhaustion

</div>
<div>

**Treatment Modules**
- GVAX + cyclophosphamide
- Anti-PD-1 / anti-PD-L1 / anti-CTLA-4
- Combinable in any configuration

**Treatment Scenarios**
- Baseline (diagnosis snapshot)
- GVAX monotherapy (neoadjuvant, 21d)
- GVAX + nivolumab (neoadjuvant, 21d)
- Clinical progression (90d)

</div>
</div>

---

# CAF Heterogeneity in the PDAC TME

The stroma is not one thing. Three CAF subtypes have distinct — and sometimes opposing — roles:

| Subtype | Driver | Key outputs | Net effect |
|---------|--------|-------------|------------|
| **myCAF** | TGFb | Collagen, CXCL12, TGFb, VEGF | Physical barrier, T cell exclusion |
| **iCAF** | IL-1 | IL-6, CCL5 | Inflammation, MDSC/Treg recruitment |
| **apCAF** | Constitutive | MHC-II antigen presentation | Treg induction via antigen sequestration |

The IL-1/TGFb ratio governs **plasticity** between iCAF and myCAF states (Biffi 2019). Targeting one subtype may shift the balance toward another — the model captures these interconversion dynamics explicitly.

<!-- Speaker notes: emphasize that single-target anti-fibrotic strategies may fail because of subtype switching -->

---

# CAF Subtype Dynamics at Diagnosis

![w:900 center](../figures/cascade_caf_subtypes.png)

<p class="small">Median single-patient trajectories from healthy tissue to diagnosis. Dashed = baseline; solid = with stroma. PSCs differentiate into iCAF/myCAF/apCAF, producing collagen that progressively excludes T cells (panel 7).</p>

---

# ECM as a Physical Barrier

The ECM module replaces generic immunosuppression with measurable physical quantities:

<div class="cols">
<div>

**Collagen volume fraction** (phi) drives three derived quantities:

- **Pore size**: `d_pore ~ phi^(-0.5)` — physical gap between fibers
- **Stiffness**: `E ~ phi^1.8` — tissue mechanical resistance
- **Immune exclusion**: T cells arrest when pore size < cell diameter

</div>
<div>

**Key calibration finding:**

Mechanosensing Hill coefficients are *sub-linear* (n = 0.3–0.6), not the default n = 1.

Stiffness acts as a **gradual rheostat**, not a binary switch. Partial stroma remodeling may yield meaningful immune access.

</div>
</div>

<p class="small">Pore size exclusion from Wolf 2013 (J Cell Biol); stiffness-exhaustion from Saitakis 2017 (eLife); fibroblast mechanosensing from Chang 2025.</p>

---

# GVAX + Nivolumab: Mechanism of Action

![w:900 center](../figures/cascade_summary_gvax_nivo.png)

<p class="small">GVAX drives APC recruitment/maturation (panels 1–2), increasing CD8 infiltration (panel 3) and CD8/Treg ratio (panel 4). Nivolumab releases checkpoint brake. Stromal barrier limits T cell access (panel 5). Net tumor reduction modest at the median — consistent with 35% pathologic response requiring a favorable parameter draw.</p>

---

# What Drives the TME? Sensitivity Analysis

![w:480](../figures/prcc_sensitivity/aggregate_ranking.png) ![w:480](../figures/prcc_sensitivity/prcc_heatmap.png)

<p class="small">PRCC across 31 calibration targets (2,852 simulations). <strong>97/194 parameters</strong> significant at FDR < 0.05. Top drivers: stroma/fibroblast axis, CD8 trafficking chain, macrophage polarization — <em>not</em> killing efficiency.</p>

---

# Calibrating 194 Parameters from Literature
## The MAPLE framework

The core challenge: default wide priors were often centered in the wrong part of parameter space — not just imprecise, but *systematically off*.

| Parameter | Default | Literature | Shift |
|-----------|---------|------------|-------|
| EC50_GMCSF | 0.5 nM | 0.003 nM | *170x lower* |
| k_APC_mature | 0.7/day | 2.73/day | *4x higher* |
| k_TGFb_sec_myCAF | 2e-11 | 1e-9 | *50x higher* |
| k_IL1_sec | 1e-12 | 4.2e-11 | *42x higher* |

**MAPLE** (Model-Aware Parameterization from Literature Evidence) uses structured schemas + LLM-assisted extraction with automated validation to extract quantitative constraints from published data with full provenance. 87 calibration targets extracted across 4 treatment scenarios. Preprint submitted to *CPT:PSP* (March 2026).

---

# Submodel Priors: Before and After

![w:900 center](../figures/prior_comparison/prior_shifts.png)

<p class="small">Prior distributions before (red) and after (blue) submodel calibration. APC0_cDC1_T median shifts from 5.4e5 to 4.1e5. TGFb_50_APC shifts 200x (0.67 → 0.003 nM). f_stroma_max median from 3.0 to 1.49.</p>

---

# Downstream Impact on Simulated Trajectories

![w:850 center](../figures/prior_comparison/spider_comparison.png)

<p class="small">Median trajectories with old priors (red) vs literature-grounded priors (blue). Same random seed — differences solely from 3 updated parameters. The prior is not just narrower — it is centered in a different part of parameter space.</p>

---

# Summary

- Mechanistic PDAC QSP model with **11 modules** spanning tumor, immune, stroma, and treatment
- Explicit **CAF heterogeneity** (iCAF/myCAF/apCAF) with IL-1/TGFb-governed interconversion
- ECM barrier modeled as **physical quantities** (pore size, stiffness) — not generic suppression
- Sensitivity analysis: **stroma and CD8 trafficking** dominate, not killing efficiency
- **87 calibration targets** extracted from literature with full provenance
- 43% of fibroblast parameters now constrained by submodel targets (0% → 43% in 3 months)
- Default priors were *systematically off* (4–170x shifts) — literature-grounded correction is essential

---

# Next Steps

- **Full SBI calibration** — simulation-based inference using all 87 targets across 4 treatment scenarios
- **Virtual patient cohorts** — generate populations matching observed clinical heterogeneity in immune infiltration, stromal density, and treatment response
- **Treatment scenario exploration** — predict response rates, identify resistance mechanisms, and test combination strategies in silico
- **MAPLE preprint** — submitted to *CPT:PSP*, open-source at github.com/popellab/maple
