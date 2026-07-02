# Hierarchical SBI Design for PDAC Virtual Patient Generation

## Problem

The current flat SBI setup conflates two uncertainty sources:

1. **Population center uncertainty** (μ): how precisely do we know the typical PDAC patient?
2. **Inter-patient variability** (ω): how much do patients differ from each other?

Conditioning on x_obs = population median with SEM noise constrains μ but leaves ω unconstrained (defaults to prior width). Well-identified parameters collapse to point estimates, losing patient-level diversity in the virtual population.

## Proposed Hierarchical Model

### Parameters

For each model parameter j:

- `μ_j` — population mean (log-space)
- `ω_j` — population standard deviation (log-space)

Together, φ = (μ, ω) are the population-level hyperparameters. Individual patients are draws from this population:

```
θ_i,j ~ LogNormal(μ_j, ω_j)
```

### Three uncertainty sources

| Source | Symbol | Meaning | Constrained by |
|--------|--------|---------|---------------|
| Population center | μ_j | Where the typical patient sits | Observed median (SEM-level precision) |
| Population spread | ω_j | How much patients vary | Observed IQR / CI width |
| Measurement noise | σ_meas | Assay technical variability | Assay specs (~10–20% CV) |

### Training loop

```
For each training sample:
    1. Draw hyperparameters:  μ_j ~ hyperprior_μ,  ω_j ~ hyperprior_ω
    2. Draw N virtual patients:  θ_i,j ~ LogNormal(μ_j, ω_j)   for i = 1..N
    3. Simulate each patient:  x_i = f(θ_i)
    4. Compute cohort summaries:
         x_median_k = median of observable k across patients
         x_iqr_k = IQR of observable k across patients
    5. Training pair:  (φ, x_summary) = ((μ, ω), (x_median, x_iqr))
```

NPE learns: p(μ, ω | x_median, x_iqr)

### Conditioning

Condition on **both** observed medians and observed IQRs from calibration targets:

- Median → constrains μ
- IQR → constrains ω

Currently we only condition on median, so ω is unconstrained.

### Virtual patient generation

```
Draw (μ, ω) ~ posterior
Draw θ_patient ~ LogNormal(μ, ω)
Simulate x_patient = f(θ_patient)
```

Each VP has properly propagated uncertainty from both μ and ω posteriors.

## Hyperprior Design

Current priors CSV already encodes (μ, ω) per parameter. The hyperprior is a prior *on those values*:

- **μ_j ~ Normal(csv_log_median, τ_μ)**: population center is near but not fixed to the CSV median. Width τ_μ controls how much the population center can move during inference.
- **ω_j ~ LogNormal(log(csv_sigma), τ_ω)** or **LogUniform(0.05, 2.0)**: population spread is positive, centered near the CSV sigma but can be learned from data.

The CSV sigma values (dist_param2 column) range from 0.3 to 1.2, reflecting prior beliefs about inter-patient variability per parameter.

## Conditioning Statistics

Currently: 35 observables (medians only) → 35D conditioning vector.

Hierarchical version: 35 medians + 35 IQRs → 70D conditioning vector. The IQR statistics are available — the calibration target YAMLs report Q1 and Q3 (or CI95 derived from IQR).

Alternatively, condition on (median, CI95_lower, CI95_upper) → 105D. But IQR is more robust.

## Inference Dimensionality

Current: 194 parameters (one θ_j per parameter).

Hierarchical: 194 μ_j + 194 ω_j = 388 dimensions. This is larger but the structure is factored — μ and ω for the same parameter are coupled, while (μ_j, ω_j) and (μ_k, ω_k) for different parameters are approximately independent.

## Cost Analysis

Each training sample requires N patient simulations to compute cohort summaries.

| Cohort size N | Sims per training sample | Total for 5k training | IQR precision |
|--------------|--------------------------|----------------------|---------------|
| 5 | 5 | 25k | Very noisy |
| 10 | 10 | 50k | Moderate |
| 20 | 20 | 100k | Reasonable |
| 50 | 50 | 250k | Good |

Current setup uses 10k simulations total. Even at N=10 with 2.5k training samples, that's 25k — manageable.

Note: the NPE can learn through noisy cohort summaries. Small N adds noise to the summary statistics, which the NPE treats as additional observation uncertainty. This is conceptually similar to the current observation noise injection — but grounded in the actual sampling process rather than an ad-hoc noise model.

### Training sample efficiency

Current R² plateaus at ~2.5k sims with 35 observables and 194 params. The flat model is already data-efficient in the small observable space. The hierarchical model doubles the inference dimensionality (388D) but also doubles the conditioning dimensionality (70D), so the ratio stays similar. Training sample count (number of cohort draws) of 2.5–5k should be sufficient, giving total simulation counts of 25–100k depending on N.

## Implementation Notes

### Required changes

1. **Simulator wrapper**: takes (μ, ω), draws N patients, simulates each, returns cohort summaries
2. **Summary statistics**: add IQR computation alongside median for each observable
3. **Hyperprior**: build from priors CSV, adding ω dimension with appropriate hyperprior
4. **Conditioning**: pass (median, IQR) as x instead of just median
5. **VP sampling**: post-inference, draw from LogNormal(μ_posterior, ω_posterior)

### What stays the same

- QSP simulator f(θ) — unchanged
- Observable definitions — unchanged
- Calibration target YAMLs — already have IQR/CI data
- NPE architecture — may need wider hidden layers for 388D
- Copula transform — applied to (median, IQR) jointly

### Simulation pooling

The existing simulation pool (θ, x) pairs from the prior can be reused. For each hyperparameter draw (μ, ω), sample N patient parameters θ_i ~ LogNormal(μ, ω), then find nearest neighbors in the pool to approximate f(θ_i). This avoids re-running MATLAB for every cohort — amortized simulation.

This only works if the pool covers the (μ, ω) hyperprior support well. With 10k existing sims from a broad prior, this should provide reasonable coverage for at least a pilot.

## Open Questions

1. **Which parameters get ω?** Not all parameters represent inter-patient variability. Binding affinities and molecular rate constants may be universal (ω ≈ 0). Only biological state variables (growth rates, immune cell counts, cytokine levels) genuinely vary across patients. Could partition parameters into shared (infer μ only) and variable (infer μ + ω).

2. **Cohort size N vs training samples**: trade-off between precise cohort summaries (large N) and more training samples (small N, more cohort draws). Need to empirically determine the sweet spot.

3. **IQR vs CI95 as conditioning statistic**: IQR is more robust to outliers and matches the source data (many YAMLs report Q1–Q3). CI95 is more sensitive to tails but also noisier from small cohorts.

4. **Correlation structure**: the hierarchical model assumes parameters are independent across patients (θ_i,j independent of θ_i,k given μ, ω). In reality, parameters within a patient are correlated (e.g., high growth rate correlates with high angiogenesis). A full hierarchical model would include a population-level covariance matrix, but that's ~194^2/2 ≈ 19k additional parameters. Probably not feasible — block-diagonal structure or factor model could help.

5. **Can we reuse existing simulation pools?** If we sample (μ, ω), draw θ_i, and then look up nearest-neighbor simulations in the existing pool, we avoid new MATLAB runs. Quality depends on pool density in parameter space.
