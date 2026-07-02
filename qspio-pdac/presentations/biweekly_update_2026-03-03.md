---
marp: true
math: mathjax
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
---

<!-- _class: title -->

# PDAC QSP Model: Biweekly Update

## Joel Eliason | Popel Lab
March 3, 2026

---

# Grant Opportunity: ICI Fund

Working with **Eric Christenson** (oncology) on a proposal for the Fund for Innovation in Cancer Informatics:
- **$200K / 2 years**, 5-page proposal, **due March 13**
- Pairing our QSP platform with Eric's PDAC immunotherapy cohort data

**Aim 1:** Build a reusable, clinically calibrated QSP model of the PDAC TME with predictive utility for **patient stratification** via computational biomarkers and **population outcome prediction**

**Aim 2:** Use the calibrated model to **identify knowledge gaps** in PDAC immunotherapy biology and guide **experimental design** for future studies

---

# Model Simplification

Stripped **4 modules** (~50 unconstrained parameters, ~3,500 lines) to focus on calibratable biology:

| Removed | Reason |
|---------|--------|
| Hypoxia/HIF | HIF posterior moved *away* from data; no anchoring calibration targets |
| Glucose/lactate metabolism | obs_z = 1.96 for glucose ratio; QSS formula poorly constrained |
| NK cells | obs_z = 1.45 for NK fraction; parameters in no-learning cluster |
| CXCL9/10 chemotaxis | Downstream of removed modules; only CXCL12 exclusion axis retained |

**Retained:** T cells, macrophages, DCs (cDC1/cDC2), CAFs (iCAF/myCAF/apCAF), ECM, checkpoints, GVAX

Code preserved on `feature/extended-tme` for re-integration when calibration data emerges.

---

# Bayesian Inference: The Idea

**Goal:** Given observed data, what parameter values are plausible?

$$p(\theta \mid x_{\text{obs}}) = \frac{p(x_{\text{obs}} \mid \theta) \, p(\theta)}{p(x_{\text{obs}})}$$

| Term | Meaning | In our context |
|------|---------|----------------|
| $p(\theta)$ | **Prior** — what we believe before seeing data | Literature ranges for rate constants, cell counts |
| $p(x \mid \theta)$ | **Likelihood** — probability of data given params | Run the QSP model, compare to observations |
| $p(\theta \mid x)$ | **Posterior** — updated belief after seeing data | Calibrated parameter distributions |

**The problem:** For a 200-ODE model with 30+ parameters, the likelihood is **intractable** — we can simulate the model forward, but can't write down $p(x \mid \theta)$ analytically.

---

# Simulation-Based Inference (SBI)

**Key insight:** We can't evaluate the likelihood, but we *can* simulate from the model.

**Classical Bayesian:** Evaluate $p(x \mid \theta)$ analytically or via MCMC
**Simulation-based:** Replace likelihood evaluation with forward simulations

```
  Repeat thousands of times:
    1. Draw θ from prior
    2. Run simulator → get x
    3. Store (θ, x) pair
                                    ┌─────────────────────────┐
  Then train a neural network:      │  Learn p(θ | x) directly│
    Input: observed summary stats   │  from simulated examples│
    Output: posterior over θ        └─────────────────────────┘
```

**Why this matters for QSP:** Each simulation takes ~2 seconds in MATLAB. SBI amortizes the cost — train once on ~5,000 simulations, then get instant posteriors for any new observation.

---

# NPE and Neural Spline Flows

**Neural Posterior Estimation (NPE):** Train a conditional density estimator $q_\phi(\theta \mid x) \approx p(\theta \mid x)$ by minimizing the KL divergence over simulated (θ, x) pairs.

**Neural Spline Flows (NSF):** The architecture used to represent the posterior.

```
  Simple base distribution          Learned invertible transforms
  (standard normal)                 (monotonic rational-quadratic splines)

  z ~ N(0, I)  ──► T₁ ──► T₂ ──► T₃ ──►  θ ~ q(θ | x)
                    │       │       │
                    └───────┴───────┘
                    Conditioned on x
                    (observed data)
```

- Each transform $T_k$ is a **monotonic spline** — flexible but invertible, so we can compute exact densities
- Transforms are **conditioned on x**, so the posterior shape changes with the observation
- 3 transforms, 32 hidden units → ~70K parameters (small by deep learning standards)

---

# SBI Workflow Overview

```
                                            ┌─────────────┐
  pdac_priors.csv ──► Sample θ ──► MATLAB  ──► Observables x
  (30 params)         from prior   QSP sim     (29 targets)
                                            └──────┬──────┘
                                                   │
                         ┌─────────────────────────┘
                         ▼
              ┌─────────────────────┐      ┌──────────────────┐
              │  Neural Posterior   |      │  Posterior       │
              │  Estimation (NPE)   │─────►│  p(θ | x_obs)    │
              │  via NSF flows      │      │  2000 samples    │
              └─────────────────────┘      └──────────────────┘
                         │
              ┌──────────┴──────────┐
              │  Diagnostics        │
              ├─ R² recovery        │
              ├─ Prior vs posterior │
              ├─ Predictive checks  │
              └─────────────────────┘
```

PRCC tier filtering: rank 194 → 30 most influential parameters for NPE training.

---

# Prior vs Posterior

![h:450](../figures/sbi_baseline_tier1/posterior_vs_prior.png)

Gray = prior. Blue = posterior. Sharper peaks = data-informed parameters.

---

# Parameter Recovery (R²)

![h:450](../figures/sbi_baseline_tier1/recovery.png)

7/30 parameters identifiable (R² > 0.3). The other 23 are individually degenerate but may be *jointly* constrained.

---

# Prior vs Posterior Predictive

![h:450](../figures/sbi_baseline_tier1/prior_vs_posterior_predictive.png)

Orange = prior predictive. Blue = posterior predictive. Dashed = observed.

---

# Learning Curve

![h:450](../figures/sbi_baseline_tier1/learning_curve.png)

Validation loss plateaus by epoch ~20. Train-val gap widens with more epochs (mild overfitting). Early stopping prevents degradation.

---

# Major Development Updates

**Structural model fixes:**
- **DC subtype decoupling** — cDC1/cDC2 independently parameterized (was locked ratio)
- **CD4 Th/Treg rebalance** — added Th proliferation + CXCR3 recruitment to fix 99% Treg fraction
- **apCAF proliferation** — added PDGF-driven self-renewal (was structurally capped at ~1%)
- **ECM kinetics** — degradation half-life 20 → 100 days (literature-matched)
- **Hill function audit** — all 28 Hill functions classified and EC50s corrected (IL-12, IL-10, CCL5, NO)
- **HIF/VEGF cascade fix** — corrected VEGF_norm_50 scale mismatch, resolved 7 xfails in one shot

---

# Major Development Updates (cont.)

**Qualitative behavior test suite:** 83 tests (0 failures, 2 xfails) covering:
- Hill function operating points (graded vs switch classification)
- Clinical species ranges (pO2, VEGF, lactate, HIF)
- Cell composition constraints (CAF ratios, M2 dominance, CD8 exhaustion)
- Progression direction (tumor/ECM/Treg increase, pO2 decrease over 90 days)
- Treatment cascade ordering (GVAX decay, mAPC peak, checkpoint release timing)
- Qualitative bounds with literature provenance (20 bounds, all with evidence snippets + DOI/PMID)

**Infrastructure:**
- PRCC sensitivity analysis (194 → 30 tier-1 parameters)
- NPE capacity sweep confirmed bottleneck is parameter degeneracy, not network size
- Observation noise injection (lognormal/Gaussian per-target from CI widths)
- Simulation pooling + posterior predictive caching
- Calibration target extraction pipeline with provenance tracking (29 active targets)

---

# Outstanding Questions

### 1. Validation strategy

We have **held-out scenario data** not used in baseline SBI training:
- 18 GVAX+nivo targets (Li 2022: immune response at day 21)
- 1 clinical progression target (tumor doubling time, Ahn 2017)

**Option A:** Use all data for calibration (maximize posterior sharpness)
**Option B:** Hold out treatment data for validation (test generalization)

With Eric's clinical data, the same question applies: **calibrate on it, or validate against it?**

---

# Outstanding Questions (cont.)

### 2. What is the calibration objective?

| Goal | Metric | Implication |
|------|--------|-------------|
| Parameter identifiability | High R² on parameter recovery | Tells us *which* parameters the data constrains |
| Predictive accuracy | Posterior predictive coverage of held-out data | Tells us if the model *generalizes* |
| Clinical utility | Response rate prediction, biomarker stratification | Tells us if the model is *useful* |

Current R² is modest (median 0.13 across 30 params) but this may not matter if the **joint posterior makes sharp predictions** in clinically relevant directions.

**Proposed approach:** Optimize for predictive accuracy on treatment outcomes, use R² as a diagnostic (not an objective).

---

# Next Steps

1. **Re-run SBI** with stripped model + structural fixes (apCAF, ECM, Th/Treg)
2. **Cross-scenario validation** — train on baseline, predict GVAX+nivo held-out targets
3. **Grant proposal draft** — aims, timeline, data needs for Eric's review
4. **Identify priority endpoints** from Eric's cohort data (pre/post biopsy, matched outcomes)

**Grant deadline: March 13**
