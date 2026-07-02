---
marp: true
math: mathjax
theme: default
paginate: true
size: 16:9
style: |
  section {
    font-size: 22px;
  }
  h1 {
    font-size: 36px;
    color: #1a5276;
  }
  h2 {
    font-size: 30px;
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
    font-size: 19px;
  }
  img {
    display: block;
    margin: 0 auto;
  }
  code {
    font-size: 18px;
  }
---

<!-- _class: title -->

# Simulation-Based Inference

## From ABC to Neural Posterior Estimation

Joel Eliason | Popel Lab
March 2026

---

# Motivation: QSP Model Calibration

Quantitative systems pharmacology (QSP) models simulate drug-disease interactions via systems of ODEs. A typical PDAC immunotherapy model has:

- **~200 ODEs** describing immune cells, tumor, stroma, cytokines, therapies
- **~200 parameters** (rate constants, EC50s, carrying capacities)
- **~30 calibration targets** from literature (cell densities, ratios, clinical outcomes)

Each parameter has a plausible range from literature, but the **joint** values that reproduce observed biology are unknown.

**The calibration problem:** Find the distribution of parameter combinations consistent with observed data, while quantifying what the data does and does not constrain.

---

# Why Not Just Fit Point Estimates?

Point estimation (e.g., nonlinear least squares) returns a single "best" parameter set. This has several problems:

- **Local optima** — nonlinear least squares is susceptible to local minima; the solution depends on initialization, and with 200 parameters the loss landscape is highly non-convex
- **Parameter degeneracy** — many parameter combinations produce similar outputs; a single optimum discards this information
- **No uncertainty quantification** — how confident should predictions be?
- **Virtual patients** — population modeling requires parameter *distributions*, not point values

Bayesian inference provides the full posterior distribution, which naturally captures degeneracy, quantifies uncertainty, and enables virtual patient generation.

---

# Bayes' Theorem

$$p(\theta \mid x_{\text{obs}}) = \frac{p(x_{\text{obs}} \mid \theta) \, p(\theta)}{p(x_{\text{obs}})}$$

| Term | Name | Role |
|------|------|------|
| $p(\theta)$ | **Prior** | Belief about parameters before seeing data |
| $p(x_{\text{obs}} \mid \theta)$ | **Likelihood** | Probability of observed data given parameters |
| $p(\theta \mid x_{\text{obs}})$ | **Posterior** | Updated belief after seeing data |
| $p(x_{\text{obs}})$ | **Evidence** | Normalizing constant (usually cancels) |

All Bayesian inference methods approximate the posterior. They differ in *how*.

---

# The Problem: Intractable Likelihood

Given a simulator (ODE model, agent-based model, etc.) with parameters $\theta$:

$$\theta \xrightarrow{\text{simulator}} x$$

The simulator maps parameters to outputs, but there is no closed-form expression for $p(x \mid \theta)$.

This rules out standard MCMC, which evaluates $p(x \mid \theta)$ at every proposed step.

---

# MCMC (Requires a Likelihood)

```
  Initialize θ₀

  For each step:
    1. Propose θ* near current θ
    2. Evaluate likelihood ratio:  p(x_obs | θ*) / p(x_obs | θ)
    3. Accept/reject proportionally
                                         ▲
                                    Requires p(x|θ)
                                    at every proposal
```

**Strengths:** Asymptotically exact. Well-understood convergence theory.

**Limitation:** Requires the likelihood in closed form. Not applicable to complex simulators.

---

# ABC Rejection: Likelihood-Free Inference

**Approximate Bayesian Computation** replaces likelihood evaluation with simulation:

```
  For i = 1 to N:
    1. Draw θᵢ from prior
    2. Simulate  xᵢ = simulator(θᵢ)
    3. If  d(xᵢ, x_obs) < ε :   keep θᵢ
       else:                      discard θᵢ
```

| Limitation | |
|---|---|
| Threshold ε is a bias-variance tradeoff | Too tight → few samples. Too loose → biased posterior. |
| Scales poorly to high-D | Acceptance rate decays exponentially with summary statistic dimension |

---

# ABC-SMC: Addressing the Efficiency Problem

**ABC Sequential Monte Carlo** (Beaumont 2009, Toni 2009) adaptively shrinks ε through a sequence of intermediate distributions:

```
  Initialize: sample particles {θᵢ} from prior, set ε₁ > ε₂ > ... > ε_T

  For each round t:
    1. Perturb particles from previous round via kernel K(θ* | θᵢ)
    2. Simulate  x* = simulator(θ*)
    3. Accept if  d(x*, x_obs) < εₜ
    4. Reweight particles by importance ratio
```

**Improvements over rejection ABC:**
- Perturbs accepted particles from previous round as proposals (guided search, not blind sampling)
- Adapts ε automatically in some variants

**Remaining limitations:** Still requires choosing summary statistics. Each perturbed particle requires a new simulation — simulations from earlier rounds cannot be reused at tighter ε.

---

# NPE: Use All Simulations

Rejection ABC and ABC-SMC both discard simulations. NPE trains on all of them.

```
  ABC:                              NPE:
  ┌──────────────┐                  ┌──────────────┐
  │ (θ₁, x₁)  ✗  │                  │ (θ₁, x₁)  ✓  │
  │ (θ₂, x₂)  ✗  │                  │ (θ₂, x₂)  ✓  │
  │ (θ₃, x₃)  ✓  │                  │ (θ₃, x₃)  ✓  │
  │ (θ₄, x₄)  ✗  │                  │ (θ₄, x₄)  ✓  │
  │ (θ₅, x₅)  ✗  │                  │ (θ₅, x₅)  ✓  │
  │ 5000 sims    │                  │ 5000 sims    │
  │ ~50 kept     │                  │ all used     │
  └──────┬───────┘                  └──────┬───────┘
         ▼                                 ▼
    Histogram of                   Train a conditional
    accepted θ's                   density estimator
```

---

# NPE: Deriving the Loss

Goal: find $q_\phi(\theta \mid x) \approx p(\theta \mid x)$. A natural measure of approximation quality is the KL divergence, averaged over all possible observations:

$$\mathbb{E}_{p(x)}\left[\text{KL}\left(p(\theta \mid x) \,\|\, q_\phi(\theta \mid x)\right)\right]$$

Expanding:

$$= \mathbb{E}_{p(x)}\!\left[\mathbb{E}_{p(\theta|x)}\!\left[\log p(\theta \mid x) - \log q_\phi(\theta \mid x)\right]\right]$$

$$= \underbrace{\mathbb{E}_{p(\theta, x)}\!\left[\log p(\theta \mid x)\right]}_{\text{does not depend on } \phi} \;-\; \mathbb{E}_{p(\theta, x)}\!\left[\log q_\phi(\theta \mid x)\right]$$

Minimizing over $\phi$ is equivalent to maximizing the second term. Since $(\theta_i, x_i) \sim p(\theta, x)$ are available from the simulator, the Monte Carlo estimate gives the NPE loss:

$$\mathcal{L}(\phi) = \frac{1}{N} \sum_{i=1}^{N} \log q_\phi(\theta_i \mid x_i)$$

---

# NPE: Interpreting the Loss

$$\mathcal{L}(\phi) = \frac{1}{N} \sum_{i=1}^{N} \log q_\phi(\theta_i \mid x_i)$$

The network takes simulator outputs $x$ as input and returns a probability distribution over $\theta$.

For each training pair, the loss measures how much probability the network assigns to the parameters $\theta_i$ that actually produced the outputs $x_i$. Maximizing this across all pairs forces the network to learn the mapping from outputs to the parameters that generated them.

At inference time, conditioning on real observed data $x_{\text{obs}}$ yields the posterior $q_\phi(\theta \mid x_{\text{obs}})$ in a single forward pass.

---

# Normalizing Flows, Not (μ, σ)

NPE does not output a mean and standard deviation. It uses **normalizing flows** — learned invertible transformations:

$$z \sim \mathcal{N}(0, I) \xrightarrow{T_1(x)} \xrightarrow{T_2(x)} \xrightarrow{T_3(x)} \theta \sim q_\phi(\theta \mid x)$$

Each $T_k$ is a monotonic rational-quadratic spline, conditioned on $x$.

The composition can represent arbitrary posterior shapes — skewed, multimodal, correlated, heavy-tailed. The shape is determined by the training data, not by architectural assumptions.

Invertibility provides exact log-probability evaluation.

---

# What the Flow Learns

```
  z ~ N(0,1)             After T₁            After T₂           After T₃

       ╭──╮               ╭─╮                ╭─╮
       │  │               │ │  ╭╮            │ │  ╭╮             Learned
       │  │       →       │ │  ││     →      │ │  ││      →     posterior
       │  │               │ │  ╯│            │ ╰╮ ╯│
       ╰──╯               ╰─╯  ╰╯            ╰──╯ ╰╯

    Symmetric            Shifted           Asymmetric          Skewed +
    Gaussian             + stretched        + 2nd mode         multimodal
```

- Bimodal true posterior → flow learns two modes
- Spike at true value → flow learns a spike
- Unidentifiable parameter → flow reproduces the prior

---

# Interpretability

The normalizing flow is a **density estimator**, analogous to:

| Method | Role |
|--------|------|
| Histogram | Density estimation from binned samples |
| KDE | Density estimation via kernel smoothing |
| Gaussian mixture | Density estimation via EM |
| Normalizing flow | Density estimation via learned transforms |

The relevant question is not whether the estimator is interpretable, but whether it produces **calibrated posteriors**. This is empirically testable.

---

# Approximation Gaps: All Methods Have Them

NPE minimizes $\text{KL}[p(\theta \mid x) \| q_\phi(\theta \mid x)]$. In practice this may not reach zero.

The same is true for every inference method:

| Method | Approximation gap | How to detect |
|--------|-----|-----------|
| MCMC | Chains may not converge | R-hat, ESS |
| ABC | ε > 0, lossy summaries | Decrease ε (expensive) |
| NPE | KL > 0, finite capacity | SBC, capacity sweep |

The question is not "is the gap zero?" but **"can it be detected and measured?"**

---

# Validation: Capacity Sweep

Does the network have sufficient capacity to represent the posterior?

Train NPE across a 15x range of network parameters, holding data constant:

```
  NN params:    70K     140K     280K     560K     1.1M
  Median R²:    0.14    0.13     0.15     0.14     0.13
                ───────────────────────────────────────
                           ← flat →
```

Performance is invariant to capacity. The bottleneck is data and model structure (parameter degeneracy), not network expressivity.

---

# Validation: Simulation-Based Calibration (SBC)

If the posterior is well-calibrated, the true $\theta$ falls inside the $\alpha$-credible region $\alpha$% of the time.

```
  Repeat many times:
    1. Draw θ_true from prior
    2. Simulate x = simulator(θ_true)
    3. Obtain posterior q(θ | x)
    4. Check: is θ_true inside the 90% credible region?

  Well-calibrated:   ~90% yes
  Overconfident:     <90% (intervals too narrow)
  Underconfident:    >90% (intervals too wide)
```

A frequentist coverage check, applicable to any posterior regardless of how it was obtained.

**Note:** Step 3 requires a full posterior for each test case. With MCMC or ABC-SMC (non-amortized), this means running the entire inference procedure hundreds of times — often prohibitively expensive. With NPE (amortized), each posterior is a single forward pass through the network, making SBC practical.

---

# Validation: Parameter Recovery (R²)

Test simulations with known $\theta_{\text{true}}$, held out from training:

$$R^2 = 1 - \frac{\sum_i (\theta_{\text{true},i} - \hat{\theta}_i)^2}{\sum_i (\theta_{\text{true},i} - \bar{\theta})^2}$$

| R² | Interpretation |
|----|---------|
| ~1.0 | Well-identified — posterior concentrates near truth |
| ~0.0 | Unidentifiable — posterior ≈ prior |
| <0 | Pathological (rare) |

Low R² reflects insufficient information in the data, not a method failure.

---

# Validation: Posterior Predictive Checks

Do parameters drawn from the posterior produce simulations consistent with observed data?

```
  1. Draw θ₁, ..., θ_K  from posterior  q(θ | x_obs)
  2. Simulate  x̂ₖ = simulator(θₖ)  for each
  3. Compare distribution of x̂ to x_obs
```

- **Prior predictive:** broad, vague predictions
- **Posterior predictive:** should concentrate around $x_{\text{obs}}$

Method-agnostic. Directly tests whether the posterior is useful for prediction.

---

# Example: Prior vs Posterior Predictive

![h:430](../figures/sbi_baseline_tier1/prior_vs_posterior_predictive.png)

Blue = prior predictive. Orange = posterior predictive. Dashed = observed values.

---

# Example: Parameter Recovery

![h:430](../figures/sbi_baseline_tier1/recovery.png)

R² on held-out test simulations. 7/30 parameters identifiable (R² > 0.3).

---

# Marginal Summaries vs Joint Structure

Taking the marginal median (or mean) of each parameter independently ignores correlations. The resulting vector may have **low joint posterior density**.

| Approach | Preserves joint structure? |
|---|---|
| Marginal median/mean | No — combines coordinates independently |
| MAP (joint mode) | Yes, but atypical in high-D (mode ≠ typical set) |
| Sample full vectors → simulate each | **Yes** — correct approach for predictions |

For diagnostics (R²), marginal summaries are standard. For **predictions**, sample full vectors from the joint posterior and summarize the *outputs*, not the parameters.

---

# Method Comparison

| | MCMC | ABC-SMC | NPE |
|---|------|---------|-----|
| **Requires** | Likelihood $p(x \mid \theta)$ | Simulator + ε + summaries | Simulator only |
| **Sims used** | N/A | Partial (discard per round) | 100% |
| **Arbitrary choices** | Proposal distribution | ε schedule, summary stats, kernel | Architecture, hyperparams |
| **Approx. gap** | Chain convergence | ε > 0, lossy summaries | KL > 0, finite capacity |
| **Gap testable?** | Partially (R-hat) | Decrease ε (expensive) | Yes (SBC, capacity sweep) |
| **Amortized?** | No | No | Yes |
| **SBC feasible?** | Expensive | Expensive | Yes (forward pass) |
| **Scales to high-D?** | Poorly (mixing) | Moderate | Well |

---

# Limitations of NPE

| Limitation | Detail |
|---|---|
| No convergence guarantee | MCMC has ergodicity theorems. NPE convergence is empirical — must be verified via diagnostics. |
| Amortization gap | Posterior is optimized *on average* over $p(x)$. For $x_{\text{obs}}$ in the tails of the prior predictive, quality may degrade. |
| Prior-posterior mismatch | If the prior is far from the true posterior, few training samples cover the high-posterior region. Sequential NPE (SNPE) mitigates this. |
| Upfront simulation budget | Requires ~1,000+ simulations before returning any posterior. MCMC produces samples incrementally. |

---

# When to Use What

**MCMC** — Tractable likelihood available. Asymptotically exact.

**ABC-SMC** — Moderate parameter count, established workflow, no density estimator needed.

**NPE** — Intractable likelihood, many parameters (10+), moderately expensive simulations (seconds to minutes), amortization desired.

---

# Extension: Observation Noise

Calibration targets have measurement uncertainty — a target with CI95 = [100, 110] should constrain the posterior more than one with CI95 = [10, 1000].

Standard NPE trains on raw simulator outputs, treating all targets equally. To incorporate measurement uncertainty, noise is injected into the training data:

$$\tilde{x}_i = x_i + \epsilon_i, \qquad \epsilon_i \sim \text{noise model from CI95}$$

The noise model is selected per-observable based on CI asymmetry:
- **Lognormal** (multiplicative) for positive quantities with scale-dependent uncertainty
- **Gaussian** (additive) for fractions and rates with symmetric CIs

The NPE learns that observations are noisy measurements, so targets with wide CIs naturally constrain the posterior less than precise ones.

---

# Extension: Hierarchical SBI

Standard NPE conflates two sources of uncertainty:
- **Population center** ($\mu$) — where the typical patient sits
- **Inter-patient variability** ($\omega$) — how much patients differ

A flat NPE conditions only on observed medians, so $\omega$ defaults to the prior width.

**Hierarchical NPE** infers $(\mu_j, \omega_j)$ per parameter, conditioning on both observed medians *and* IQRs:

$$\mu_j \sim \mathcal{N}(\text{csv\_median}, \tau_\mu), \quad \omega_j \sim \text{LogNormal}(\log(\text{csv\_}\sigma), \tau_\omega)$$

Each training sample draws hyperparameters $\phi = (\mu, \omega)$, simulates a cohort of $N$ virtual patients from $\text{LogNormal}(\mu, \omega)$, and computes cohort summaries (median + IQR per observable).

Enables properly calibrated virtual patient generation where each VP inherits uncertainty from both $\mu$ and $\omega$ posteriors.

---

# Extension: Updating with New Data

| Scenario | What changes | Approach |
|---|---|---|
| New patients, same endpoints | $x_{\text{obs}}$ values | Plug into trained NPE — instant posterior (amortization) |
| New measurement types | Observation vector dimension | Retrain NPE with augmented summaries + new simulations |
| Sequential updating (dataset 1 → dataset 2) | Prior becomes previous posterior | Sample from posterior, use as prior for next round (SNPE-C) |
| New treatment scenario | Simulator configuration | New simulations required; baseline posterior informs shared parameters |

**Amortization is a key advantage:** Once trained, NPE provides instant posteriors for any new observation vector of the same structure. MCMC and ABC-SMC require a full inference run for each new observation.

Sequential updating (posterior-as-prior) is conceptually standard in Bayesian inference. With NPE, the posterior is a normalizing flow rather than a parametric distribution, so updating requires either retraining with samples from the previous posterior or using SNPE-C's atomic proposal correction.

---

<!-- _class: title -->

# Summary

NPE posteriors are trained on simulated data
and validated with the same tools used for any Bayesian method.

The neural network is a density estimator, not a decision-maker.
The conclusions are the posterior samples and predictions.

**Validate the posterior, not the method.**
