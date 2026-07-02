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

# Stage-1 + Stage-2 Calibration
## A Parameter Audit of the PDAC QSP Model

### Joel Eliason &nbsp;|&nbsp; April 20, 2026 &nbsp;|&nbsp; Biweekly Update

---

# What Is a Mechanistic Model?

![w:650 center](../figures/presentation/qsp_concept_ode.png)

A **quantitative systems pharmacology** (QSP) model represents the tumor microenvironment as a system of **ordinary differential equations** (ODEs) — one equation per cell type or molecule, tracking how each changes over time.

<p class="small">Same math as PK/PD, but applied to the whole TME: cancer cells, T cells, macrophages, fibroblasts, cytokines, drugs. Run it forward → simulated trajectories of every species under any treatment.</p>

---

# What Is an ODE?

<div class="cols">
<div>

An ODE says: *how fast does this quantity change, and what does that rate depend on?*

$$\frac{d[\text{CD8}]}{dt} = \underbrace{k_{\text{in}} \cdot \text{APC}}_{\text{recruitment}} - \underbrace{k_{\text{death}} \cdot \text{CD8}}_{\text{death}}$$

</div>
<div>

- Left side: rate of change of CD8 cells
- Right side: what drives that rate
- Every **k** is a **parameter** — a number we need to set from data
- Multiply this by 200+ species/reactions → the full QSP model

</div>
</div>

<p class="small">The whole model is an interconnected web of these rate laws. Solving them forward from an initial condition gives the simulated trajectory.</p>

---

# What Is a Parameter?

![w:850 center](../figures/presentation/parameter_dose_response.png)

<p class="small"><strong>A:</strong> An EC50 parameter controls where a dose-response curve sits — shifting it 170× changes which drug concentrations are effective. <strong>B:</strong> Changing one killing rate reshapes the whole tumor trajectory. The model has <strong>~217 of these</strong>. Each one needs a value — and a <em>distribution</em>, because we're never perfectly sure.</p>

---

# Where We Are

- Model has **217 free parameters**, ~40 scenario-level observables across 4 treatment scenarios
- No single experiment constrains this — we've built a **staged calibration pipeline**
- Today: (1) what's in, (2) what's identified, (3) what to extract next

<p class="small">Today's focus is the audit — how Stage-1 submodel priors and Stage-2 joint NPE combine, what that tells us about where the model is <em>actually</em> constrained, and the ranked work queue that falls out.</p>

---

# The Two-Stage Pipeline

<div class="cols">
<div>

### Stage 1 — submodel targets
- Per-parameter posteriors from **in vitro / ex vivo** data
- Each target = experiment + ODE submodel + meta-analytic translation sigma
- Output: `submodel_priors.yaml`

</div>
<div>

### Stage 2 — joint NPE
- Neural posterior across **4 scenarios, 38 observables**
- Uses Stage-1 priors, sees full-model non-linearities
- Output: `posterior_samples.csv`

</div>
</div>

The **parameter audit** is what ties them together — comparing pre-prior → S1 → S2 and flagging what each stage did (or didn't) do.

---

# Why Two Stages?

![w:700 center](../figures/presentation/prior_vs_posterior_ec50.png)

<p class="small">Stage 1 narrows individual parameters from targeted experiments. Stage 2 resolves joint constraints that individual experiments can't see. Neither alone is sufficient — together they compose.</p>

---

<!-- _class: lead -->

# Stage 1 — Submodel Targets

---

# Coverage: 117 / 217 Parameters (54%)

<div class="cols">
<div>

**By type (covered):**
- EC50s, secretion rates, death rates, proliferation rates
- Hill coefficients, fractions, biophysical constants

**Partial pooling across groups:**
- CAF_death_rates: 1/4 extracted → pulls other 3
- IL6_EC50s: 2/3 extracted → constrains IL6_50_iCAF

</div>
<div>

**Remaining by type:**

| Type | Count |
|------|:-:|
| rate | 59 |
| EC50 | 43 |
| secretion | 24 |
| death | 16 |
| trafficking | 14 |
| count / fraction | 22 |

</div>
</div>

---

# Stage-1 Impact on Priors

![w:900 center](../figures/prior_comparison/prior_shifts.png)

<p class="small">Red: inherited / guessed priors. Blue: after Stage-1 literature calibration. Several core stromal and immune parameters shift 1–2 orders of magnitude — these are <em>not</em> small refinements.</p>

---

# Not All Targets Work

~40 parameters have submodel targets but stage-1 contraction ≤ 30%.

| Parameter | Contraction | Likely Cause |
|-----------|:-:|---------|
| `TGFb_50_CAF_act` | *-348%* | translation sigma too wide across 4 sources |
| `k_CD8_death` | *-60%* | target measures wrong quantity |
| `TGFb_50_APC` | *-59%* | forward model code mismatch |
| `IL6_50_M2` | *-46%* | prior already tight; target adds noise |
| `k_CAF_encounter` | *-6%* | weakly informative experiment |

<p class="small">A target ≠ a calibration. These land on a <strong>separate action queue</strong>: fix the target or widen the prior, don't extract something new.</p>

---

<!-- _class: lead -->

# Stage 2 — Joint NPE

---

# Joint NPE Setup

- **Backend:** switched `sbi_runner.py` from MATLAB to C++ (`CppSimulator`, HPC array + chained derivation)
- **Budget:** 5000 raw sims → **2467 effective train sims** after joint-scenario NaN filter
- **Scenarios:** baseline diagnosis, GVAX mono (d21), GVAX+nivo (d21), clinical progression
- **Flow:** Neural Spline Flow, **192 hidden × 9 transforms** (~7.6M params)

Simulation pooling accumulates across runs — config hash covers priors, binary SHA, templates, scenario YAMLs.

---

# Capacity vs Information — Flow Sweep

| Config | θ dim | NN params | Best val | **shell ≥50%** | shell <10% |
|--------|:-:|:-:|:-:|:-:|:-:|
| tier 1 (32×3) | 29 | 70k | 26.6 | 3 | 9 |
| tier 3 (64×5) | 116 | 754k | 104.8 | 3 | 75 |
| all (128×7) | 217 | 3.8M | 184.2 | 5 | 134 |
| **all (192×9)** | 217 | **7.6M** | **122.3** | **11** | **51** |
| all (256×11) | 217 | 13.0M | 108.7 | 9 | 83 |

<p class="small">Tier-3 collapse at small flow was <em>capacity</em>, not information. At 192×9 non-informative params correctly marginalize toward zero contraction. Past that, diminishing returns.</p>

---

# Four Numbers Per Parameter

<div class="cols">
<div>

**vs pre** — total calibration effect, pre-prior → S2
**vs S1** — marginal S2 effect on top of submodel priors
**Contraction global** — ID under the prior
**Contraction local-shell** — ID near the observation

</div>
<div>

### Which to trust?
**shell-local.** It averages over test obs within a 20% neighborhood of *x_obs* — guards against single-point flow artifacts where NPE looks sharp at *x_obs* but not nearby.

If local ≫ shell-local: artifact, not identification.

</div>
</div>

---

# Artifact vs Genuine Identification

| Parameter | local contr | shell contr | Verdict |
|-----------|:-:|:-:|---------|
| `k_M2_pol` | 62% | **51%** | *genuine* |
| `f_stroma_max` | 54% | **38%** | *genuine* |
| `k_Th_pro_tumor` | 64% | **55%** | *genuine* |
| `k_TGFb_sec_myCAF` | 79% | 4% | artifact |
| `k_col_myCAF_sec` | 75% | 7% | artifact |
| `d_pore_ref` | 58% | 8% | artifact |

<p class="small">Artifacts cluster where Stage-1 already constrains the parameter — whole-model data lands on one ridge of a larger degeneracy. Lesson: never report local contraction alone.</p>

---

# What Got Identified

![w:900 center](../figures/sbi_baseline_sa97/z_score_contraction.png)

<p class="small">Shell-local contraction by parameter. Fibroblast / stroma and T cell trafficking dominate — consistent with the PRCC sensitivity analysis. Identified core: ~11 parameters at ≥50% shell contraction.</p>

---

# What Didn't — and Why

Information-limited, not capacity-limited:

- **T cell trafficking:** `q_CD8_T_in`, `q_nCD8_LN_in`, `q_nCD8_P_out`
- **IL-2 axis:** `k_IL2_sec`, `N_IL2_CD8`, `k_IL2_cons`
- **T activation / expansion:** `k_T_activation_per_APC`, `n_T_expansion_burst`, `k_CD8_pro`, `k_CD8_death`
- **CAF interconversion:** `k_iCAF_to_myCAF`, `k_myCAF_to_iCAF`, `k_iCAF_prolif`
- **Vascular:** `k_vas_growth`, `k_vas_decay`

These do **not** move with more flow or more sims. Path forward: **new calibration targets**.

---

<!-- _class: lead -->

# Stage 1 × Stage 2 — The Audit

---

# Posterior Predictive Coverage

<div class="cols">
<div>

![w:520](../notes/calibration/parameter_audit_report_files/ppc_coverage.png)

</div>
<div>

**240 / 282** observables covered by 95% CI (**85%**)

- Remaining 42 are inherent — concentrated in treatment-response and ratio targets
- Ready for Stage-2 follow-on (OBED) once PPC file lands
- Full inference DAG is in the audit report — too big for this slide

</div>
</div>

---

# Extraction-Priority Ranking

Score = sigma × PRCC × effective-coverage. **New target** list (top 10):

| # | Parameter | Type | Sigma | #obs | Score |
|:-:|-----------|------|:-:|:-:|:-:|
| 1 | `q_nTreg_P_out` | trafficking | 1.20 | 4 | 0.46 |
| 2 | `k_APC_mat_cDC2` | rate | 0.69 | 2 | 0.45 |
| 3 | `K_C_exh` | rate | 0.97 | 6 | 0.44 |
| 4 | `k_Treg_act` | activation | 1.00 | 5 | 0.44 |
| 5 | `k_P1_up_apCAF` | rate | 1.00 | 5 | 0.43 |
| 6 | `P1_50_apCAF` | EC50 | 1.19 | 7 | 0.43 |
| 7 | `k_Cy_Treg_deplete` | rate | 1.00 | 3 | 0.41 |
| 8 | `N_IL2_CD4` | count | 0.50 | 8 | 0.40 |
| 9 | `k_Th_to_Treg` | rate | 0.69 | 6 | 0.38 |
| 10 | `K_apCAF_Treg` | rate | 1.20 | 2 | 0.38 |

---

# Two Queues, Not One

<div class="cols">
<div>

### Queue A — extract new targets
For under-identified, high-impact params.

Top: `q_nTreg_P_out`, `k_APC_mat_cDC2`, `K_C_exh`, `k_Treg_act`, `P1_50_apCAF`.

</div>
<div>

### Queue B — fix existing targets
For params where Stage-1 inference failed.

Top: `k_CAF_encounter`, `k_CD8_exh_death`, `ArgI_50_Treg`, `k_cell_clear`, `TGFb_50_CAF_act`.

</div>
</div>

<p class="small">Conflating these wastes extraction effort. Queue A needs a new paper; Queue B needs a re-extraction, a forward-model fix, or a prior widening.</p>

---

<!-- _class: lead -->

# What's Next

---

# Targeted Extraction Rounds

1. **Paired pre→post mIHC** (Heumann 2023, Fig 1F/1G, Arms A & B)
   - CD8 densities, M1/M2 ratios by arm
   - Most likely to move the T-cell and macrophage clusters
   - *In flight — Li 2022 Fig 2F batch already merged (PR #14)*

2. **KM survival at landmarks** (6, 12, 18, 24 mo) — Heumann 2023
   - 8 scalar targets; shape info median OS throws away
   - Out-of-sample mechanistic check on `k_C1_growth` + kill/growth balance

Deprioritized: bare median OS/DFS (information-poor at n=14–16).

---

# OBED — The Closing Loop

- **Phase 1 (merged):** `scripts/clinical_endpoints.py` — OS/DFS landmarks, tumor diameter, RECIST 1.1, TTP, VDT
- **Phase 2 (merged, PR #20):** 36 prediction-only YAMLs across 3 long-horizon scenarios
- **Phase 3 (next):** `CppSimulator.simulate_with_parameters(theta)` — PPC at the Stage-2 posterior
- **Phase 4:** one-shot PPC run → `posterior_predictive_clinical.csv`
- **Phase 5:** audit OBED section — attribute endpoint CI95 back to parameters via Spearman

<p class="small">End state: the audit tells us which parameter extractions will most reduce <em>clinical endpoint</em> uncertainty, not just per-parameter contraction.</p>

---

<!-- _class: lead -->

# Takeaways

---

# Takeaways

1. **Staged pipeline works.** S1 + S2 + OBED form a tight, audit-driven loop.
2. **Residual unidentifiability is data-limited**, not capacity-limited. Specific literature targets are queued.
3. **Artifacts ≠ identification.** Report shell-local contraction, not local-at-*x_obs*.
4. **Stage-1 coverage ≠ calibration.** A meaningful fraction of existing submodel targets need re-work, not more siblings.
5. We have a **ranked work queue**, not a parameter soup.

---

<!-- _class: lead -->

# Thank You

### Links
- `notes/calibration/parameter_audit_report.md`
- `notes/calibration/calibration_strategy.md`
- `notes/calibration/sbi_tier_flow_sweep.md`
- `notes/calibration/obed_integration_handoff.md`

Joel Eliason &nbsp;|&nbsp; Popel Lab, Johns Hopkins University

---

<!-- _class: lead -->

# Backup Slides

---

# Why shell-local beats local-at-*x_obs*

- **local-at-*x_obs*:** contraction measured only at the single observation
- **shell-local:** average contraction over `theta_test_local` within a 20% copula-space neighborhood of *x_obs*

Normalizing flows can over-concentrate at any single *x* even when posteriors 5% away are wide — single-point sharpness is not identification.

Shell-local is the SBC-style metric: if the posterior is genuinely tight in a neighborhood, the flow has learned the likelihood. If only tight at one point, it has memorized.

---

# Seed Stability at 192×9

Outstanding verification. At 128×7 all-217, seed-to-seed wobble was **6.2 pp mean |Δ|** across 217 params; identified-core values moved 4–10 pp.

192×9 is 2× bigger flow on the same data — similar or worse seed-wobble is plausible.

**Action:** rerun with SEED=2029, check identified-core stability before pinning any per-parameter contraction values in the audit.

---

# Stage-1 "Weak Target" Root Causes

| Cause | Example | Fix |
|-------|---------|-----|
| Translation sigma too wide (multiple divergent sources) | `TGFb_50_CAF_act` (4 sources, −348%) | Re-weight sources, drop incompatible ones |
| Target measures wrong quantity | `k_CD8_death` (basal vs exhausted) | Replace target |
| Forward-model code bug | `TGFb_50_APC` | Fix code, re-derive |
| Prior already tight | `IL6_50_M2` | Widen prior |
| Experiment is weakly informative | `k_CAF_encounter` | New experiment |

<p class="small">Each row is a distinct action. None of these are "extract more siblings" — that's Queue A.</p>
