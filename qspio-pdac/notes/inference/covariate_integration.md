# Patient Covariate Integration in QSP Modeling

## Motivation

The QSP model captures mechanistic TME dynamics (immune infiltration, fibrosis, tumor growth) but is silent on patient-level factors that strongly predict clinical outcomes. Covariates like performance status, age, CA19-9, and tumor location carry prognostic signal that the mechanistic model cannot recover from first principles.

The key insight: covariates affect outcomes through **two distinct causal paths**, and we need to model both.

## Per-Patient Data Tuple

We design around individual patient-level data. Each patient contributes a tuple:

```
(covariates_i,  x_obs_i,  outcomes_i)

covariates_i:  age=67, ECOG=1, CA19-9=450 U/mL, tumor_location=head, ...
x_obs_i:       CD8_density=180 cells/mm², Treg_frac=0.12, M1_M2_ratio=0.4,
               collagen_fraction=0.35, stromal_fraction=0.62, ...
outcomes_i:    PFS=4.2 mo, OS=7.1 mo, RECIST=SD, discontinued=false, ...
```

- **covariates**: demographic and clinical variables (Table 1 data)
- **x_obs**: TME observables — the same cell densities, fractions, and ratios we already calibrate to, but at patient level rather than cohort summaries
- **outcomes**: clinical endpoints the model should predict

If per-patient data is unavailable for some fields, we fall back to cohort-level distributions.

## Causal Structure

```
                    Path 1 (mechanistic)
covariates ──────────→ θ_QSP ──────────→ simulate ──→ x_obs ──→ outcomes
    |                                                               ↑
    └───────────────────────────────────────────────────────────────┘
                    Path 2 (direct / non-mechanistic)
```

**Path 1: Covariates to QSP parameters to outcomes.** Some covariates modulate the biology the QSP model represents. For example, age affects immune competence (T cell proliferation, exhaustion rates), which changes tumor dynamics and thus PFS/RECIST. This path is mechanistically mediated — covariates shift QSP parameters, the simulator translates that to observables.

**Path 2: Covariates to outcomes directly.** Some covariates affect outcomes through channels the QSP model doesn't represent: frailty, organ function, cachexia, infection risk, treatment tolerance. ECOG performance status predicts OS not because it changes tumor immunology, but because it captures how close a patient is to dying from non-tumor causes. These act as **confounders** relative to the QSP parameters.

Both paths matter for **all** clinical outcomes — the relative contribution varies by endpoint, but it's an empirical question, not something to assert upfront.

**Examples of Path 2 confounding across endpoints:**

- **OS:** Frailty, organ function, comorbidities leading to non-cancer mortality
- **PFS:** Treatment discontinuation due to toxicity/intolerance (not tumor progression), performance-status-driven imaging schedule changes
- **RECIST:** Dose modifications due to tolerability affecting tumor response; assessment timing driven by clinical status
- **Treatment discontinuation:** Directly driven by patient factors (toxicity tolerance, comorbidities, preference)
- **Adverse events / dose modifications:** Body composition, organ function, drug metabolism differences
- **Quality of life / functional status:** Baseline function, comorbidity burden
- **Hospitalization:** Frailty, infection susceptibility, social determinants

Ignoring either path causes the same problem: **QSP parameters absorb the unmodeled covariate effects and become biased.** The parameters are the sponge for any covariate signal that isn't explicitly accounted for.

- **Ignore Path 1:** A trial enrolling older patients has weaker immune responses. Without covariate-to-parameter mappings, this gets absorbed into the posteriors for immune parameters (e.g., lower T cell proliferation rates) — conflating age effects with the true population-level biology.
- **Ignore Path 2:** A trial enrolling frailer patients has worse outcomes. Without a residual covariate effect, the QSP parameters shift to produce worse mechanistic trajectories to match the observed outcomes — even though the real cause is non-mechanistic (treatment intolerance, comorbidity-driven mortality).

## Candidate Covariates for PDAC

| Covariate | Type | Availability | Likely path | Notes |
|-----------|------|-------------|-------------|-------|
| CA19-9 (baseline) | Continuous | High | Both | Tumor burden proxy (Path 1) + liver function indicator (Path 2) |
| ECOG performance status | Ordinal (0-4) | High | Mostly Path 2 | Frailty, organ function, treatment tolerance |
| Age | Continuous | High | Both | Immune senescence (Path 1) + comorbidity burden (Path 2) |
| Tumor location | Categorical | High | Path 1 | Head vs body/tail affects local TME biology |
| Resectability status | Categorical | High | Path 1 | Reflects local invasion, vascular involvement |
| BMI | Continuous | Medium | Both | Cachexia (Path 2) + adipose-driven inflammation (Path 1) |
| Tumor stage (TNM) | Ordinal | High | Path 1 | Metastatic burden |
| NLR | Continuous | Medium | Path 1 | Systemic inflammation; immune parameter proxy |
| Prior therapy lines | Count | Medium | Both | Pretreated biology (Path 1) + cumulative toxicity (Path 2) |

---

## Parameters and Coefficients

There are four distinct sets of quantities in this framework. Being explicit about what is inferred, what is learned, and what is fixed avoids conflating them.

| Symbol | Name | What it is | Dimensionality |
|--------|------|-----------|----------------|
| **θ_pop** | QSP population parameters | The ~100+ mechanistic parameters (proliferation rates, EC50s, etc.) we're already doing SBI on | ~100-200 |
| **β** | Covariate-to-parameter coefficients | Map covariates to shifts in QSP parameters (Component A). E.g., β_age_Tprolif scales T cell proliferation with age | ~5-15 (one per covariate-to-parameter relationship) |
| **γ** | Path 2 outcome model parameters | Parametrize the direct (non-mechanistic) effect of covariates on clinical outcomes. E.g., baseline hazard coefficients, treatment discontinuation rates as a function of ECOG | ~5-20 (depends on outcome model complexity) |
| **φ** | NPE network weights | Neural network parameters of the density estimator | ~10⁵-10⁶ (standard for normalizing flows) |

### Inference targets

All of θ_pop, β, and γ can in principle be inferred jointly by the NPE:

```
p(θ_pop, β, γ | x_obs, covariates, outcomes)
```

This is the most principled approach — β and γ are parameters like any others, and the data updates them from their priors. But there is a spectrum of options for how aggressively to infer each:

| Strategy | β | γ | Pros | Cons |
|----------|---|---|------|------|
| **All fixed** | Literature values | Literature values | Simplest, no additional parameters to identify. Good starting point. | Misspecification if literature values don't match study population. No uncertainty quantification on covariate effects. |
| **β inferred, γ fixed** | Inferred with prior | Literature values | Learns mechanistic covariate effects from data. Path 1 deconfounding is data-driven. | Needs IPD with (covariates, x_obs) to identify β. γ misspecification still possible. |
| **All inferred** | Inferred with prior | Inferred with prior | Most principled. Full uncertainty propagation. Data decides how much covariate signal is Path 1 vs Path 2. | Highest data requirements. β and γ may compete to explain the same variance (identifiability). Largest target dimension for NPE. |

### Identifiability considerations

**β (covariate-to-parameter):** Identifiable when patients spanning the covariate space have different x_obs. E.g., if older patients consistently show lower CD8 density, that pins down β_age_CD8. Degeneracy risk: the same x_obs pattern could be explained by a β shift or a θ_pop shift — you need enough covariate variation to separate them.

**γ (covariate-to-outcome):** Identifiable when patients with similar x_obs but different covariates have different outcomes. E.g., two patients with identical TME profiles but different ECOG scores and different OS — that's pure Path 2 signal. Degeneracy risk: γ and θ_pop compete to explain outcome variation. A patient with bad OS could be explained by aggressive tumor biology (θ_pop) or poor performance status (γ).

**The Bayesian safety net:** With informative priors on β and γ (centered on literature values), degeneracy is handled gracefully. If the data can't distinguish β from θ_pop, the posterior on β stays near the prior and θ_pop absorbs the variance — which is what would happen anyway without covariates. The posteriors widen rather than bias. More data sharpens identification; less data falls back to literature.

**Sample size intuition:** PDAC trial cohorts are typically 30-80 patients. This is likely sufficient to identify a few strong β effects (age, CA19-9) but may struggle with subtle ones. For γ, trial-level survival data (KM curves stratified by covariates) provides additional signal beyond just per-patient tuples. Starting with the "β inferred, γ fixed" strategy and upgrading to "all inferred" as data accumulates is a reasonable path.

### What flows where in the pipeline

```
Training:
  covariates ~ population        ─┐
  θ_pop ~ prior                   │
  β ~ prior (or fixed)            ├→ θ_i = θ_pop × f(covariates; β)
                                  │         ↓
                                  │   x_i = simulator(θ_i)
                                  │         ↓
  γ ~ prior (or fixed)  ─-────────┴→ outcomes_i = g(x_i, covariates; γ)
                                          ↓
                         Train NPE(φ) on: (θ_pop, β, γ) ~ (x_i, covariates, outcomes_i)
                                          ─────────────    ─────────────────────────────
                                             target            conditioning variables

Inference:
  Observe (x_obs, covariates, outcomes) for a patient or cohort
                                          ↓
                         NPE(φ) → p(θ_pop, β, γ | x_obs, covariates, outcomes)
```

When β or γ are fixed (not inferred), they are simply excluded from the NPE target and used as constants during training data generation.

### Key design decisions

- **φ:** Standard NPE training. The conditioning set is larger (x_obs + covariates + outcomes) and the target may include β and γ, so the embedding network may need more capacity.
- **Priors on β and γ:** These should be informative — centered on literature values with spread reflecting uncertainty in the literature. Vague priors on β and γ with small cohorts will produce diffuse, uninformative posteriors.
- **Incremental strategy:** Start with everything fixed, verify the pipeline works, then unlock β inference, then γ. Each step adds parameters and data requirements — validate that each addition actually sharpens posteriors rather than just adding noise.

---

## Model Components

### Component A: Covariate-to-Parameter Relationships (Path 1)

A small number of well-motivated mechanistic relationships between covariates and QSP parameters. These enter the simulator so that covariate-dependent biology is captured.

**Form:**
```
θ_i = θ_pop × f(covariates_i)

e.g.:
  k_Tcell_prolif_i  = k_Tcell_prolif_pop  × (age_i / 60)^β_age_Tprolif
  k_Tcell_exhaust_i = k_Tcell_exhaust_pop × (age_i / 60)^β_age_Texh
```

The β coefficients are additional parameters to estimate (or fix from literature).

**Design principles:**

- Start with a small set of relationships where the biology is well-established (immune senescence with age, tumor burden correlating with CA19-9)
- Each relationship should be motivated by a specific mechanism, not data-mined
- The β coefficients can be estimated jointly with QSP parameters via SBI, or fixed from external data

**Candidate relationships:**

| Covariate | QSP parameter(s) | Mechanism | Evidence quality |
|-----------|-------------------|-----------|-----------------|
| Age | T cell proliferation, exhaustion rates | Thymic involution, immune senescence | Strong (reviewed extensively) |
| Age | APC maturation rates | DC function declines with age | Moderate |
| CA19-9 | Initial tumor burden (C_total_0) | Biomarker-burden correlation | Strong |
| BMI | Inflammatory cytokine baselines | Adipose tissue inflammation | Moderate |
| NLR | Peripheral immune cell ratios | Direct measurement proxy | Strong (tautological) |

### Component B: Joint Covariate-Conditional Inference (SBI)

The NPE conditions on **all available patient observations** — TME observables, covariates, and clinical outcomes — to infer the posterior over the target parameters:

```
p(θ_pop, β, γ | x_obs, covariates, outcomes)
```

(When β and/or γ are fixed, they drop out of the target — see the strategy table in the Parameters section above.)

The goal is not to predict outcomes. The goal is to use outcomes (along with everything else we observe) to **refine the posterior on θ_pop** (and optionally β, γ). Clinical outcomes like OS and PFS are informative observations that constrain the parameters, just like cell densities or M1/M2 ratios.

Covariates help because they explain some outcome variance through Path 2 (non-mechanistic). Without covariates, the NPE would be forced to explain all outcome variation through θ_pop alone — biasing the posterior. With covariates in the conditioning set, the NPE can attribute Path 2 effects to covariates and reserve θ_pop for the mechanistic biology.

**Training data generation** is detailed in the Parameters section pipeline diagram above. The key steps:

1. Sample covariates from population distributions
2. Sample θ_pop, β, γ from priors (or fix β, γ)
3. Apply Component A to get patient-level parameters: θ_i = θ_pop × f(covariates; β)
4. Simulate: x_i = simulator(θ_i)
5. Generate outcomes: outcomes_i = g(x_i, covariates; γ) — includes both mechanistic (from simulation) and non-mechanistic (from γ) contributions
6. Train NPE on the resulting tuples

**Step 5 — generating training outcomes:** This is where Path 2 enters the simulation pipeline. We need a forward model for outcomes that includes both:

- The mechanistic contribution (PFS from tumor trajectory, RECIST from diameter change — already available from QSP simulation)
- The non-mechanistic contribution (covariate-driven treatment discontinuation, non-cancer mortality, etc.), parametrized by γ

This could be as simple as a parametric hazard model for OS that combines QSP-derived tumor burden with covariate-driven baseline hazard.

**Why this works for deconfounding:** When the NPE sees a patient with poor OS, it can attribute that to either (a) aggressive tumor biology (high θ_pop values for growth/immune evasion), or (b) poor performance status (high baseline mortality risk, γ effect). Without covariates in the conditioning set, all of that signal would be forced into θ_pop.

**What can be missing at inference time:** Not all conditioning variables need to be observed for every patient. If outcomes are unavailable (e.g., predicting for a new patient), the NPE can still produce `p(θ_pop, β, γ | x_obs, covariates)` — outcomes just don't contribute to the posterior. Similarly, if some covariates are missing, the NPE marginalizes over them (or they can be imputed from population distributions). The more complete the patient data tuple, the sharper the posterior.

---

## Putting It Together

Component A defines the mechanistic covariate relationships. Component B uses them (along with all observed data) to produce deconfounded posteriors. The full pipeline is shown in the Parameters section diagram above.

**What each component buys you:**

- **A alone:** Covariates affect simulations but the posterior doesn't adjust for patient population
- **A + B (β, γ fixed):** Deconfounded QSP posteriors that use all available patient data — TME observables, covariates, and clinical outcomes — to constrain θ_pop. Covariates absorb Path 2 effects so θ_pop reflects mechanistic biology rather than patient-level confounders.
- **A + B (β, γ inferred):** Same, plus uncertainty quantification on the covariate effects themselves. The data determines how much covariate signal is Path 1 vs Path 2.

## Practical Considerations

**Data requirements:**

- Component A: Literature-derived β coefficients (can start here without IPD)
- Component B: Per-patient tuples (covariates, x_obs, outcomes). At minimum, trial-level covariate distributions (Table 1) and cohort-level observables/outcomes. Per-patient IPD gives the sharpest posteriors.

**Incremental implementation:**

1. Start with Component A — define 2-3 covariate-to-parameter relationships from literature. This requires only changes to the simulation setup, no new inference machinery.
2. Add Component B — extend the NPE to condition on covariates and outcomes. This requires (a) a forward model for Path 2 outcome effects to generate training data, and (b) handling missing conditioning variables at inference time.

**Validation strategy:**

- Component A: Check that covariate-modulated virtual populations match known epidemiological patterns (e.g., older virtual patients should have worse immune infiltration)
- Component B: Compare posteriors with and without covariate/outcome conditioning — are the differences biologically plausible? Do posteriors sharpen when more conditioning data is provided?

## Open Questions

- What IPD do we have access to? Cohort-level data still works but gives less sharp posteriors.
- How sensitive are the QSP posteriors to the choice of covariate-to-parameter relationships (Component A)? Need a robustness check.
- Should covariates enter as raw values or as risk scores (e.g., composite frailty index)?
- How do we handle covariates during training simulation? Sample from target trial demographics, or from a reference population?
- What is the forward model for Path 2 outcome effects (the g function parametrized by γ)? Parametric hazard model? Literature-derived covariate-outcome relationships?
- How does the NPE handle missing conditioning variables at inference time? Masking? Imputation from population distributions?
