# Misspecification Detection Framework for QSP-SBI Calibration

*Date: 2026-03-07*

## The Double-Dipping Problem

### The Issue

Standard Bayesian workflow uses prior predictive checks to validate priors before calibration. But in practice, we iteratively adjust priors based on prior predictive diagnostics — seeing that Treg/CD4 fraction is 0.97 when clinical data shows 0.34-0.44, then tuning parameters until the prior predictive looks reasonable. This is empirical Bayes: using the data twice.

Betancourt states this directly: *"Allowing comparisons between the prior predictive distribution and any observed data to influence the development of the prior model is a form of empirical Bayes that uses the data twice, once to directly inform the prior model and again to construct the posterior distribution."* This is *"prone to overfitting where we develop a model that is exceptionally good at retrodicting the particular data we observed at the cost of poor predictions"* of new data ([Betancourt, Principled Bayesian Workflow](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)).

Gelman et al. (2020) take a more pragmatic position, acknowledging the practice but arguing that *"the danger of overfitting the data is much more concerning than objections from hypothesis testing and unbiased estimation"* ([Gelman et al., Bayesian Workflow](https://arxiv.org/abs/2011.01808)). Their key distinction: prior predictive checks should compare model assumptions to **domain expertise**, not to specific observed values. Adjusting because "97% Treg fraction is biologically implausible" is domain expertise; adjusting because "the target says 0.44 and I want to match it" is empirical Bayes.

### Consequences for UQ

1. **Posterior credible intervals become too narrow.** The prior is shaped by the data, so the posterior double-counts the data's influence — once through the prior and again through the likelihood. Posteriors become overconfident.
2. **Coverage degrades.** Nominal 90% credible intervals may have actual coverage below 90%. Wehenkel et al. (2024) show that model misspecification in SBI causes NPE posteriors to become overconfident with credible regions that undercover true parameters ([arXiv:2405.08719](https://arxiv.org/abs/2405.08719)).
3. **SBC can't detect it.** Simulation-based calibration checks self-consistency (uniform ranks under the prior-likelihood pair), but if the prior was shaped by data, SBC will report no miscalibration while actual held-out coverage may be poor.

## Proposed Solution: LLM Expert Elicitation

### Core Idea

Elicit priors and hypothetical observational data from a panel of LLM "experts" for each parameter and/or observation. Use this synthetic domain knowledge — explicitly not tied to any real data or publication — to detect and correct model misspecification. Only after the model shows no structural misspecification, calibrate to actual data. This cleanly separates domain knowledge from calibration data, avoiding double-dipping by construction.

### Analogy to Established Methods

This is essentially the SHELF protocol (Structured Expert Judgment) using LLMs as the expert panel. The prior elicitation literature already accepts expert judgment as legitimate; LLMs are a more systematic, reproducible, and scalable version. Benefits:

- **Reproducible**: Same prompts produce same elicitation (unlike human experts)
- **Scalable**: Can elicit for 194 parameters without scheduling expert panels
- **Ensembleable**: Use multiple models (Claude, GPT, Gemini) as a Delphi panel
- **Auditable**: Full prompt/response transcript for each elicitation

### The Leakage Concern

LLM training data includes the same publications that calibration targets come from. An LLM saying "Treg/CD4 fraction should be ~30-50%" is arguably informed by Sivagnanam 2021 and Hiraoka 2006 — the same papers behind our targets. This is indirect data leakage.

**Mitigation**: This is true of human experts too — a pancreatologist would know the same literature. The key difference is that we elicit **broad domain knowledge about plausible ranges** from mechanistic/physiological reasoning, not specific point estimates from specific papers. When prompting the LLM panel, instruct them to reason from mechanism ("What Treg fraction is physiologically plausible in a cold tumor?") rather than citing measurements ("What does the literature report for PDAC Treg/CD4 ratio?").

## D² as a Misspecification Detection Statistic

### Mahalanobis D²

D² = (x_obs - μ_pp)ᵀ Σ_pp⁻¹ (x_obs - μ_pp)

where μ_pp and Σ_pp are the mean and covariance of the prior (or posterior) predictive distribution. Under correct specification with Gaussian prior predictive, D² ~ χ²(k).

### LOO Influence Decomposition

Removing observable i and recomputing D² gives D²_{-i}. The influence of observable i is:

influence_i = D²_full - D²_{-i}

High influence means the model is struggling with that specific observable. This localizes misspecification to specific model subsystems.

### Addressing Potential Criticisms of D²

**"Wide priors mask misspecification"** — True if priors are uninformative. But with the LLM elicitation approach, priors are informed by domain expertise and should be reasonably tight. Additionally, the posterior predictive D² (see below) eliminates this concern entirely by testing after parameter flexibility has been used.

**"Assumes normality"** — In our log-transformed observable space, the prior/posterior predictive is approximately normal. And the self-reference null (see below) removes this assumption entirely.

**"Only tests endpoints"** — This is a target design question, not a D² limitation. Define temporal observables (growth rate slopes, AUC, time-to-peak, fold-changes) and D² covers dynamics automatically.

**"Doesn't explain *why*"** — Correct, but detection and diagnosis are separate tasks. LOO tells you where the problem is; mechanistic reasoning tells you why.

**"Misses correlation structure"** — This means we need better coverage of calibration targets (joint constraints), not a different statistic.

## Prior Predictive vs. Posterior Predictive D²

### Prior Predictive D²

Tests: "Is the observed data plausible under the model with full parameter uncertainty?"

**Limitation**: Parameter flexibility can absorb structural errors. A model with enough free parameters can produce any data from wide enough priors, even if its internal structure is wrong.

### Posterior Predictive D² (After Calibration)

Tests: "Can the model reproduce the data even with optimally chosen parameters?"

This is fundamentally stronger. After calibration, the posterior has concentrated onto the parameter regions that best explain the data. The posterior predictive represents the model's **best attempt**. Any remaining discrepancy is **unambiguously structural** — the model literally cannot satisfy all constraints simultaneously with any parameter combination.

**Why this tests conditional structure**: The posterior is the joint distribution over parameters conditioned on all observables simultaneously. If fitting observable A well forces observable B to be wrong, that appears as residual D² in the posterior predictive. The model's internal wiring (how parameters map to observables) is being directly tested.

**LOO becomes more powerful here**: On the prior predictive, high LOO influence means "this observable is hard to cover under the prior" — could be prior misspecification or structural. On the posterior predictive, high LOO influence means "the model is actively trading off this observable against others" — unambiguous structural tension.

**Conservative by construction**: The posterior was fit to x_LLM, so the posterior predictive *should* cover it. Low D² is expected and uninformative. **High D² is a very strong signal** — the model can't fit data it was explicitly trained on.

**Diagnostic ratio**: D²_posterior / D²_prior tells you what fraction of prior predictive misfit is structural vs. parametric. If this ratio is near 0, most misfit was parametric (fixable by calibration). If near 1, the model is structurally unable to improve on the prior.

## Self-Reference Null Distribution (No Extra Simulations Required)

### The Problem with χ² Reference

Comparing D² to χ²(k) or χ²(k_eff) assumes normality and a specific effective dimensionality — both approximations that may not hold, especially with correlated observables and non-Gaussian tails.

### The Solution: Posterior Predictive Self-Reference

We already have N posterior predictive samples x_1, ..., x_N from simulating θ ~ q(θ|x_LLM). Each x_i is what the data "would look like" if the model were correctly specified. Use them as the null:

1. Compute D²_obs = D²(x_LLM, posterior predictive distribution)
2. For each posterior predictive sample x_i, compute D²_i = D²(x_i, posterior predictive distribution)
3. p-value = fraction of D²_i ≥ D²_obs

If the model is well-specified, x_LLM should look like a typical posterior predictive draw. If D²_obs is in the tail of the {D²_i} distribution, the model is structurally misspecified.

This is Gelman's (1996) posterior predictive p-value with D² as the test statistic. **Zero extra simulations** — just reuse the posterior predictive samples already generated by the SBI workflow.

### LOO Bias Correction

Each x_i was used to estimate μ_pp and Σ_pp, so D²_i is biased low (tested against a distribution it helped define). This makes the test **conservative** — if D²_obs is still in the tail despite a downward-biased null, the signal is strong. For a clean null:

For each x_i, compute μ_{-i} and Σ_{-i} from all x_j where j ≠ i:

D²_i = (x_i - μ_{-i})ᵀ Σ_{-i}⁻¹ (x_i - μ_{-i})

Still no extra simulations — just N recomputations of mean and covariance (cheap matrix algebra on existing samples).

### Per-Observable Null

The same trick works for the LOO influence decomposition. For each x_i, compute the LOO influence of each observable. This gives a **null distribution of influence values per observable**. If observable j's real influence is in the 95th percentile of its null distribution, that specific observable is structurally misspecified — not just "high influence" in an absolute sense, but high relative to what the model expects of itself.

### Why This Handles Over-Parameterization

If excess parameters allow the model to overfit x_LLM, then D²_obs will be low **and** the null {D²_i} will also be low — the p-value stays calibrated because both the test statistic and null shift together. The null adapts to the model's actual flexibility.

## Complete Workflow

```
Phase 1: Structure Validation (no real data)
├── 1a. LLM panel elicits hypothetical x_LLM + priors π_LLM
├── 1b. Prior predictive D² — gross screening, are priors reasonable?
├── 1c. Calibrate (SBI/NPE) against x_LLM → posterior q(θ|x_LLM)
├── 1d. Generate posterior predictive samples x_1, ..., x_N
├── 1e. Posterior predictive D² + self-reference null → structural misspec p-value
├── 1f. Per-observable LOO influence + null → localize structural tensions
├── 1g. Fix model structure, iterate 1b–1f until p-value is acceptable
│
Phase 2: Calibration (first use of real data)
├── 2a. Calibrate to real calibration targets → posterior q(θ|x_real)
├── 2b. Posterior predictive D² on calibration targets (expected to be good)
│
Phase 3: Validation (held-out real data)
├── 3a. Posterior predictive check on held-out treatment targets (e.g., GVAX+nivo)
└── 3b. Coverage assessment on targets never seen during any prior step
```

Phase 1 never touches real data. Phase 2 is the single clean use. Phase 3 is genuine out-of-sample validation.

## Complementary Diagnostics

While D² + LOO + self-reference null forms the core framework, these complement it:

- **D² sensitivity curve**: Compute D² at progressively narrower prior widths. In a well-specified model, D² stays calibrated as priors narrow. In a misspecified model, D² explodes at a critical prior width where structural error can no longer hide behind parameter flexibility. The transition point quantifies how much flexibility compensates for structural error.

- **Conditional D²**: Compute D² on subsets of observables, conditioning on other observables being near targets. If conditioning on immune targets makes stromal targets worse, the model has structural tension between subsystems.

- **Prior-data conflict (Evans & Moshonov 2006)**: Check whether the posterior concentrates in prior-unlikely regions. Complements D² — D² checks observation space, prior-data conflict checks parameter space.

## References

- Gelman, A., Vehtari, A., et al. (2020). Bayesian Workflow. [arXiv:2011.01808](https://arxiv.org/abs/2011.01808)
- Betancourt, M. Towards A Principled Bayesian Workflow. [Case study](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)
- Wehenkel, A., et al. (2024). Addressing Misspecification in Simulation-based Inference through Data-driven Calibration. [arXiv:2405.08719](https://arxiv.org/abs/2405.08719)
- Dalmasso, N., et al. (2023). Variational Inference with Coverage Guarantees in SBI (CANVI). [arXiv:2305.14275](https://arxiv.org/html/2305.14275v3)
- Gelman, A., Meng, X.-L., Stern, H. (1996). Posterior predictive assessment of model fitness via realized discrepancies. *Statistica Sinica*, 6, 733-807.
- Evans, M. & Moshonov, H. (2006). Checking for prior-data conflict. *Bayesian Analysis*, 1(4), 893-914.
