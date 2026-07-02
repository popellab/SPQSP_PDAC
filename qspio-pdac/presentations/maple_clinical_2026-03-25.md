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

<!-- _class: lead -->
<!-- _paginate: false -->

# MAPLE
## AI-Assisted Calibration of Mechanistic Cancer Models from Published Evidence

### Joel Eliason &nbsp;|&nbsp; March 25, 2026

---

# Why Mechanistic Models?

- GVAX + checkpoint (~35% pathologic response), KRAS neoantigen vaccines, mRNA platforms — combinations are working
- We know some of the correlates: TLS formation, CD8 infiltration, Treg depletion — but no way to **systematically test** how they interact under different treatments
- Too many open questions for trial-and-error: KRAS G12D/G12V alone or with checkpoint? Chemo upfront? What sequence?

---

# What Can Mechanistic Models Do?

- Represent the TME as a system of equations — run it forward to see what happens under different treatments
- Screen combinations, sequences, and doses computationally before committing to a trial
- Problem: these models have **hundreds of biological parameters** scattered across decades of literature

---

# What Is a QSP Model?

![w:700 center](../figures/presentation/qsp_concept_ode.png)

A **quantitative systems pharmacology** model tracks cell types and molecules as differential equations — the same framework as PK/PD, but applied to the full tumor microenvironment.

---

# What Is a "Parameter"?

Each biological rate in the model is governed by a **parameter** — a numerical constant derived from experimental data. Our model has **~194** of them.

![w:900 center](../figures/presentation/parameter_dose_response.png)

<p class="small"><strong>A:</strong> The EC50 parameter controls where a dose-response curve sits — shifting it 170x changes which drug concentrations are effective. <strong>B:</strong> Changing one killing rate parameter produces a completely different tumor trajectory.</p>

---

# Our PDAC Model

<div class="cols">
<div>

![w:420](../figures/presentation/qsp_tumor_compartment.png)

</div>
<div>

**11 modules** &nbsp;|&nbsp; **~194 parameters**

- Cancer growth and antigen release
- *Dendritic cell subtypes (cDC1, cDC2)*
- CD8+ and CD4+ T cells, Tregs
- Macrophages (M1/M2 polarization)
- *3 fibroblast subtypes (iCAF, myCAF, apCAF)*
- *Collagen, tissue stiffness, immune exclusion*
- *IL-1, CCL5, CXCL12, IL-6* + 5 others
- GVAX, anti-PD-1, anti-CTLA-4

</div>
</div>

---

# Three CAF Subtypes, Three Roles

| | **myCAF** | **iCAF** | **apCAF** |
|---|---|---|---|
| **Driver** | TGFb | IL-1 | Constitutive |
| **Output** | Collagen, VEGF | IL-6, CCL5 | MHC-II presentation |
| **Effect** | Physical barrier | Inflammation | Treg induction |

Targeting one subtype shifts the balance toward another (Biffi 2019).

---

# Simulating GVAX + Nivolumab

![w:1000 center](../figures/cascade_summary_gvax_nivo.png)

<p class="small">GVAX activates dendritic cells and boosts CD8 infiltration. Nivolumab blocks PD-1. But the stromal barrier limits T cell access — consistent with the ~35% pathologic response seen clinically.</p>

---

<!-- _class: lead -->

# The model exists.
# But how do you set all of these biological parameters?

---

# Parameters as Distributions, Not Point Estimates

![w:750 center](../figures/presentation/prior_vs_posterior_ec50.png)

A single "best guess" throws away uncertainty. Instead, each parameter is a probability distribution — the center is the best estimate, the width says how sure we are. New data **narrows and shifts** the distribution.

---

# The Calibration Problem

<!-- TODO: add cartoon — wide prior distribution vs narrow literature-informed prior, with arrow showing "default" landing far from truth -->

Most QSP parameters are inherited from other cancer types or carried forward from earlier models. Many are *orders of magnitude wrong* for PDAC:

| Parameter | Inherited / guessed | Literature (PDAC) | Error |
|-----------|---------|------------|-------|
| TGFb secretion rate | 2e-11 (other cancer) | 1e-9 | *50x* |
| IL-1 secretion rate | 1e-12 (placeholder) | 4.2e-11 | *42x* |
| APC maturation rate | 0.7/day (melanoma) | 2.73/day | *4x* |

These aren't edge cases — they're core stromal and immune parameters.

---

<!-- _class: lead -->

# MAPLE
## Model-Aware Parameterization from Literature Evidence

*Structured Schemas for LLM-Modeler Collaboration in QSP Model Calibration*
Eliason & Popel, 2026 (bioRxiv)

---

# How MAPLE Works

```
 Published paper                AI-assisted                  Structured
 (PDF / tables)                 extraction                   target YAML
 ┌──────────────────┐          ┌──────────────────┐         ┌──────────────────┐
 │ "EC50 was 5.5 pM"│          │ value: 5.5       │         │ parameter:       │
 │ Fig 3A           │─────────>│ units: pM        │────────>│   EC50_GMCSF     │
 │ DOI: 10/...      │          │ CI: [2, 10]      │         │ forward_model    │
 │                  │          │ context: TF-1    │         │ error_model      │
 └──────────────────┘          └──────────────────┘         └────────┬─────────┘
                                                                     │
                                                                     ▼
                               Updated                       Automated
                               distribution                  validation
                              ┌──────────────────┐          ┌──────────────────┐
                              │                  │          │ ✓ DOI resolved   │
                              │   ▁▂▅█▅▂▁        │<─────────│ ✓ units match    │
                              │   narrower       │          │ ✓ range plausible│
                              │   + shifted      │          │ ✓ quote verified │
                              └──────────────────┘          └──────────────────┘
```

Each step is **traceable**: every number links back to a DOI, a verbatim quote, and a unit conversion chain.

---

# Translating Between Contexts

Published data rarely matches the model context exactly. MAPLE scores this gap per data source:

<div class="cols">
<div>

![w:500](../figures/presentation/translation_sigma_heatmap.png)

</div>
<div>

Scores combine across axes. More distant sources still contribute — but with wider uncertainty, so they pull less strongly on the final estimate.

</div>
</div>

---

# Two Types of Targets

![w:850 center](../figures/presentation/two_target_types.png)

<div class="cols">
<div>

**40 full-model targets** — CD8 density, Treg fraction, tumor volume, macrophage polarization...

</div>
<div>

**37 submodel targets** — secretion rates, EC50s, proliferation rates, death rates...

</div>
</div>

<p class="small">+ 20 qualitative bounds as structural sanity checks. Total: 124 literature-backed constraints from >60 papers.</p>

---

# Extraction at Scale

| Scenario | Targets | Examples |
|----------|:-------:|---------|
| Diagnosis TME | 23 | CD8 density, Treg fraction, collagen, macrophage polarization |
| GVAX + nivo | 10 | Post-treatment CD8/Treg ratio, APC maturation |
| GVAX mono | 6 | T cell infiltration changes |
| Clinical progression | 1 | Tumor volume trajectory |
| Submodel (in vitro) | 37 | Secretion rates, EC50s, death rates |

Every value traced to a DOI and verbatim quote.

---

# Impact on Parameter Distributions

![w:950 center](../figures/prior_comparison/prior_shifts.png)

<p class="small">Before (red) vs after (blue) literature calibration. These are not small refinements — several core parameters shift 1-2 orders of magnitude.</p>

---

# Impact on Simulated Biology

![w:950 center](../figures/prior_comparison/spider_comparison.png)

<p class="small">Same model, same random seed. Only the parameter values changed. Immune infiltration, stromal density, and treatment response all move substantially.</p>

---

# Model vs Observed Data

![w:1050 center](../figures/presentation/proposal_figure.png)

<p class="small"><strong>A:</strong> Tumor compartment schematic. <strong>B:</strong> Posterior predictive check — simulated fold-change from observed (blue CIs should span zero). <strong>C:</strong> Leave-one-out influence — which targets most constrain the calibration.</p>

---

# Which Parameters Does the Data Constrain?

![w:950 center](../figures/sbi_baseline_sa97/z_score_contraction.png)

<p class="small">Z-score contraction: values < 1 mean the data meaningfully narrowed the parameter beyond its prior. Fibroblast and trafficking parameters show the strongest contraction.</p>

---

# What Drives Treatment Response?

![w:500](../figures/prcc_sensitivity/aggregate_ranking.png) ![w:500](../figures/prcc_sensitivity/prcc_heatmap.png)

<p class="small">Sensitivity analysis: stroma/fibroblast biology and T cell trafficking dominate — <strong>not</strong> killing efficiency.</p>

---

# Where This Goes

<div class="cols">
<div>

**Virtual clinical trials**
Test combination schedules *in silico* before enrolling patients

**Biomarker stratification**
Which baseline features separate responders from non-responders?

</div>
<div>

**Resistance mechanisms**
Why do most patients not respond? What's the bottleneck?

**Combination design**
Use sensitivity analysis to pick the arms worth testing

</div>
</div>

---

<!-- _class: lead -->

# Key Takeaways

Many inherited QSP parameters are **systematically wrong** for PDAC
MAPLE extracted **124 targets from >60 papers** with full provenance
Corrected parameters **change what the model predicts** — not just precision
**Stroma and T cell trafficking** are the dominant axes for treatment response

---

<!-- _class: lead -->

# Thank You

MAPLE preprint: *CPT:PSP* (March 2026)
Open-source: github.com/popellab/maple

Joel Eliason &nbsp;|&nbsp; Popel Lab, Johns Hopkins University

---

<!-- _class: lead -->

# Backup Slides

---

# Example: CAF Interconversion (Biffi 2019)

Biffi 2019 Fig 5H: PSCs cultured in 3 conditions, measured by CellTiter-Glo daily for 5 days.

### Forward model

Simulates each culture condition as a nested model extracted from the full QSP equations:
- **Control** — quiescent growth only (pins baseline proliferation rate)
- **Conditioned media** — iCAF growth driven by IL-1 from tumor organoids (pins iCAF proliferation)
- **CM + TGFb** — iCAFs interconvert to myCAFs via TGFb, with IL-1 driving the reverse. Two cell populations growing and switching simultaneously

Conditions 1 & 2 calibrate the parts → Condition 3 **isolates interconversion**.

---

# Example: CAF Interconversion (cont.)

### Error model

The assay (CellTiter-Glo) measures **total cells** — it cannot distinguish iCAF from myCAF. For conditions 1 & 2 this is a single population; for condition 3 the observable is the **sum of both subtypes**. Each daily timepoint contributes independently with lognormal measurement noise.

### Parameters constrained from this single figure

| QSP model parameters (7) | Nuisance parameters (4) |
|---|---|
| iCAF↔myCAF interconversion rates | Well carrying capacity |
| iCAF & myCAF proliferation rates | Quiescent PSC growth rate |
| TGFb, IL-1, and IL-1R1 half-maximal concentrations | IL-1 concentration in conditioned media |

Nuisance parameters are marginalized out. Weakly constrained parameters are flagged automatically.
