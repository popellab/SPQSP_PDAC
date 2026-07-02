#!/usr/bin/env python3
"""
Hierarchical SBI Workflow for PDAC QSP Model

Infers population-level hyperparameters (μ, ω) per parameter — the population
center and spread — conditioned on both observed medians AND observed IQRs.
This properly separates population center uncertainty from inter-patient
variability and enables generating virtual patients with calibrated diversity.

Architecture:
  - Hyperprior: μ_j ~ Normal(csv_log_median_j, τ_μ),
                ω_j ~ LogNormal(log(csv_sigma_j), τ_ω)
  - Patient model: θ_j ~ LogNormal(μ_j, ω_j)  (original space)
  - Cohort summary: x = [median(observables), IQR(observables)]
  - NPE learns: p(μ, log(ω) | x_summary)

Sections:
1. Config + hyperprior construction
2. Generate training data (batch simulation)
3. Compute observed conditioning vector
4. Prior predictive check (structural misspecification)
5. Transform + train NPE
6. Evaluate NPE (φ-space)
7. Posterior from observed data
8. Virtual patient generation
9. Posterior predictive check
10. Save

Usage:
    python workflows/hierarchical_sbi_workflow.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import MultipleIndependent

from qsp_hpc import QSPSimulator
from qsp_inference.inference import (
    get_observed_data,
    sbi_recovery,
    sbi_z_score_contraction,
    sbi_loo_predictive_check,
    sbi_posterior_predictive_coverage,
    save_diagnostics,
)
from qsp_inference.inference.data_processing import _filter_nans
from qsp_inference.inference.gaussian_copula_transform import (
    compute_quantiles_from_array,
    transform_to_normal_from_array,
)


def participation_ratio(X):
    """Participation ratio of covariance eigenvalue spectrum.

    k_eff = (sum lambda_i)^2 / sum(lambda_i^2)

    Equals n if all eigenvalues equal, <n if correlated. Estimates effective
    degrees of freedom for chi^2 reference distribution of Mahalanobis D^2.
    """
    cov = np.cov(X, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    k_eff = float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))
    return k_eff, eigenvalues


# ============================================================================
# 1. CONFIG + HYPERPRIOR CONSTRUCTION
# ============================================================================
print("=" * 80)
print("HIERARCHICAL SBI WORKFLOW: BASELINE SCENARIO — ALL PARAMETERS")
print("=" * 80)

MODEL_VERSION = 'hierarchical_baseline_v1_fActiveTGFb'  # Bumped for f_active_TGFb + averaged polarization
MODEL_DESCRIPTION = 'PDAC hierarchical SBI: infer (μ, ω) per parameter from cohort summaries'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'
SCENARIO = 'baseline_no_treatment'

PRIORS_CSV = 'parameters/pdac_priors.csv'
CALIBRATION_TARGETS_DIR = 'calibration_targets/baseline_no_treatment'
MODEL_STRUCTURE_FILE = 'model_structure.json'

N_COHORT = 10               # patients per cohort
N_TRAINING_COHORTS = 2500    # number of (φ, x_summary) training pairs
TAU_MU = 0.5                # hyperprior slack on μ (log-space)
TAU_OMEGA = 0.5             # hyperprior slack on ω (log-scale of sigma)

MAX_TASKS = 40
SEED = 2025
SPLIT_SEED = 2027

N_PLOT_PARAMS = 20
N_POSTERIOR_SIMS = 500

NPE_HIDDEN_FEATURES = 64    # larger for 388D → 70D
NPE_NUM_TRANSFORMS = 5
NPE_NUM_BINS = 10
NPE_TRAINING_BATCH_SIZE = 128
NPE_MAX_EPOCHS = 500

SAVE_DIR = Path('figures/sbi_hierarchical')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Load priors CSV ---
priors_df = pd.read_csv(PRIORS_CSV)
param_names = priors_df['name'].tolist()
num_params = len(param_names)

csv_mu = priors_df['dist_param1'].values    # log-space centers
csv_sigma = priors_df['dist_param2'].values  # log-space spreads

print(f"\n  Parameters: {num_params}")
print(f"  Hyperparameter space: {2 * num_params}D (μ: {num_params}, ω: {num_params})")

# --- Build hyperprior as torch.distributions ---
# μ_j ~ Normal(csv_mu_j, τ_μ)  — population center in log-space
mu_dists = [
    torch.distributions.Normal(
        torch.tensor([float(csv_mu[j])]),
        torch.tensor([TAU_MU]),
    )
    for j in range(num_params)
]

# ω_j ~ LogNormal(log(csv_sigma_j), τ_ω)  — population spread, positive
omega_dists = [
    torch.distributions.LogNormal(
        torch.tensor([float(np.log(csv_sigma[j]))]),
        torch.tensor([TAU_OMEGA]),
    )
    for j in range(num_params)
]

hyperprior = MultipleIndependent(mu_dists + omega_dists)
print(f"  Hyperprior: {len(mu_dists)} Normal(μ) + {len(omega_dists)} LogNormal(ω)")

# Sanity check: sample and verify ranges
with torch.no_grad():
    hp_check = hyperprior.sample((100,))
    mu_check = hp_check[:, :num_params]
    omega_check = hp_check[:, num_params:]
    print(f"  Hyperprior check (100 samples):")
    print(f"    μ range: [{mu_check.min():.2f}, {mu_check.max():.2f}]")
    print(f"    ω range: [{omega_check.min():.4f}, {omega_check.max():.4f}]")


# ============================================================================
# 2. GENERATE TRAINING DATA (BATCH SIMULATION)
# ============================================================================
print("\n" + "=" * 80)
print("2. GENERATE TRAINING DATA")
print("=" * 80)

# Set up simulator (needed for observable computation)
sim = QSPSimulator(
    priors_csv=PRIORS_CSV,
    calibration_targets=CALIBRATION_TARGETS_DIR,
    model_structure_file=MODEL_STRUCTURE_FILE,
    model_script=MODEL_SCRIPT,
    model_version=MODEL_VERSION,
    model_description=MODEL_DESCRIPTION,
    scenario=SCENARIO,
    project_root=Path('.').resolve(),
    max_tasks=MAX_TASKS,
    verbose=True,
    seed=SEED,
)

target_df = sim._test_stats_df.copy()
observable_names = target_df['test_statistic_id'].tolist()
num_observables = len(observable_names)
print(f"  Observables: {num_observables}")

# --- 2a. Sample hyperparameters and patient parameters ---
print(f"\n  Sampling {N_TRAINING_COHORTS} hyperparameter sets...")
rng = np.random.default_rng(SEED)
torch.manual_seed(SEED)

with torch.no_grad():
    phi_all = hyperprior.sample((N_TRAINING_COHORTS,))  # (2500, 388)

mu_all = phi_all[:, :num_params].numpy()      # (2500, 194) log-space centers
omega_all = phi_all[:, num_params:].numpy()    # (2500, 194) spreads

# For each cohort, draw N_COHORT patients from LogNormal(μ, ω) in original space
# θ_j = exp(Normal(μ_j, ω_j))  ⟹  θ_j ~ LogNormal(μ_j, ω_j)
n_total_patients = N_TRAINING_COHORTS * N_COHORT
print(f"  Drawing {N_COHORT} patients per cohort → {n_total_patients} total parameter vectors")

# Expand μ and ω for all patients: (2500, 10, 194)
mu_expanded = np.repeat(mu_all[:, np.newaxis, :], N_COHORT, axis=1)
omega_expanded = np.repeat(omega_all[:, np.newaxis, :], N_COHORT, axis=1)

# Sample in log-space then exponentiate
log_theta_patients = rng.normal(mu_expanded, omega_expanded)  # (2500, 10, 194)
theta_patients = np.exp(log_theta_patients)                    # original space

# Flatten for batch simulation: (25000, 194)
theta_flat = theta_patients.reshape(-1, num_params)
print(f"  Patient parameters shape: {theta_flat.shape}")

# Quick sanity: check parameter ranges
theta_medians = np.median(theta_flat, axis=0)
print(f"  Patient θ median range: [{theta_medians.min():.2e}, {theta_medians.max():.2e}]")

# --- 2b. Batch simulate all patients ---
print(f"\n  Running {n_total_patients} simulations...")
t0 = time.time()
x_flat = sim.simulate_with_parameters(theta_flat, pool_suffix='hierarchical_training')
elapsed = time.time() - t0
print(f"  Simulations complete in {elapsed:.0f}s")
print(f"  Raw observables shape: {x_flat.shape}")

# Replace inf with NaN
n_inf = np.isinf(x_flat).sum()
if n_inf > 0:
    print(f"  Replacing {n_inf} inf values with NaN")
    x_flat[np.isinf(x_flat)] = np.nan

# --- 2c. Reshape and compute cohort summaries ---
x_cohorts = x_flat.reshape(N_TRAINING_COHORTS, N_COHORT, num_observables)  # (2500, 10, 35)

# For each cohort, check how many patients have all-finite observables
cohort_valid_counts = np.all(np.isfinite(x_cohorts), axis=2).sum(axis=1)  # (2500,)
min_patients_per_cohort = 3  # need at least this many for meaningful summary stats
valid_cohorts_mask = cohort_valid_counts >= min_patients_per_cohort
n_valid_cohorts = valid_cohorts_mask.sum()
print(f"  Cohorts with ≥{min_patients_per_cohort} valid patients: {n_valid_cohorts}/{N_TRAINING_COHORTS}")

# Compute summaries using nanmedian/nanpercentile (robust to individual NaN patients)
x_median = np.nanmedian(x_cohorts, axis=1)                               # (2500, 35)
x_q75 = np.nanpercentile(x_cohorts, 75, axis=1)                          # (2500, 35)
x_q25 = np.nanpercentile(x_cohorts, 25, axis=1)                          # (2500, 35)
x_iqr = x_q75 - x_q25                                                     # (2500, 35)

x_summary = np.hstack([x_median, x_iqr])  # (2500, 70)
print(f"  Summary statistics shape: {x_summary.shape}")

# Filter to valid cohorts (both summary and hyperparams)
phi_np = np.hstack([mu_all, np.log(omega_all)])  # (2500, 388) — log(ω) for NPE

# Filter: valid cohorts AND no NaN in summary
summary_finite = np.all(np.isfinite(x_summary), axis=1)
keep_mask = valid_cohorts_mask & summary_finite
phi_clean = phi_np[keep_mask]
x_summary_clean = x_summary[keep_mask]
print(f"  After filtering: {phi_clean.shape[0]} cohorts with valid summaries")


# ============================================================================
# 3. COMPUTE OBSERVED CONDITIONING VECTOR
# ============================================================================
print("\n" + "=" * 80)
print("3. COMPUTE OBSERVED CONDITIONING VECTOR")
print("=" * 80)

# Save summary CSV for observed data loading
Path('cache').mkdir(exist_ok=True)
summary_csv_path = 'cache/baseline_calibration_targets_summary.csv'
target_df.to_csv(summary_csv_path, index=False)

# Observed medians (already available)
obs_medians = target_df['median'].values  # (35,)

# Derive observed IQRs from CI95 under lognormal assumption
ci95_lower = target_df['ci95_lower'].values
ci95_upper = target_df['ci95_upper'].values

# For each observable, derive IQR from CI95
obs_iqrs = np.zeros(num_observables)
for i in range(num_observables):
    med = obs_medians[i]
    lo = ci95_lower[i]
    hi = ci95_upper[i]

    if med > 0 and lo > 0 and hi > 0:
        # Lognormal assumption: CI95 → σ_log → Q1, Q3
        sigma_log = (np.log(hi) - np.log(lo)) / (2 * 1.96)
        mu_log = np.log(med)
        q1 = np.exp(mu_log - 0.6745 * sigma_log)
        q3 = np.exp(mu_log + 0.6745 * sigma_log)
        obs_iqrs[i] = q3 - q1
    else:
        # Fallback: Gaussian assumption
        sigma = (hi - lo) / (2 * 1.96)
        q1 = med - 0.6745 * sigma
        q3 = med + 0.6745 * sigma
        obs_iqrs[i] = q3 - q1

obs_vector = np.hstack([obs_medians, obs_iqrs])  # (70,)
print(f"  Observed medians: {num_observables}")
print(f"  Observed IQRs: {num_observables}")
print(f"  Observed conditioning vector: {obs_vector.shape[0]}D")
print(f"  IQR range: [{obs_iqrs.min():.3g}, {obs_iqrs.max():.3g}]")


# ============================================================================
# 4. PRIOR PREDICTIVE CHECK (structural misspecification)
# ============================================================================
print("\n" + "=" * 80)
print("4. PRIOR PREDICTIVE CHECK")
print("=" * 80)
print("  LOO decomposition of Mahalanobis D² on 70D cohort summaries —")
print("  identifies which observables the model structurally cannot match...\n")

# Copula transform on cohort summaries for LOO
prior_quantiles = compute_quantiles_from_array(x_summary_clean)
x_summary_copula = transform_to_normal_from_array(x_summary_clean, prior_quantiles)

# Filter any NaN/inf from copula
copula_valid = np.all(np.isfinite(x_summary_copula), axis=1)
x_summary_copula = x_summary_copula[copula_valid]
print(f"  Copula-transformed summaries: {x_summary_copula.shape[0]} valid cohorts")

# Transform observed 70D vector through same copula
obs_copula = transform_to_normal_from_array(obs_vector[np.newaxis, :], prior_quantiles)[0]

# LOO on medians (first 35D of the 70D summary)
# Split into median and IQR halves for separate LOO checks
summary_names_median = [f"med_{n}" for n in observable_names]
summary_names_iqr = [f"iqr_{n}" for n in observable_names]

loo_prior_df, fig_loo_prior = sbi_loo_predictive_check(
    x_summary_copula[:, :num_observables],
    obs_copula[:num_observables],
    observable_names,
)
fig_loo_prior.savefig(SAVE_DIR / 'loo_prior_predictive_medians.png', dpi=150, bbox_inches='tight')
plt.close(fig_loo_prior)

k_eff_prior, _ = participation_ratio(x_summary_copula[:, :num_observables])
d2_prior = loo_prior_df['d_full'].iloc[0]
chi2_p_prior = 1 - sp_stats.chi2.cdf(d2_prior, df=k_eff_prior)
d2_z_prior = (d2_prior - k_eff_prior) / np.sqrt(2 * k_eff_prior)

print(f"  Medians: D² = {d2_prior:.1f}, k_eff = {k_eff_prior:.1f}/{num_observables}, "
      f"χ²(k_eff) p = {chi2_p_prior:.3f} ({d2_z_prior:+.1f}σ)")

print(f"\n  Top 5 structural misspecification drivers (medians):")
for _, row in loo_prior_df.head(5).iterrows():
    print(f"    {row['observable']}: influence = {row['influence']:.1f}")

# LOO on IQRs
loo_prior_iqr_df, fig_loo_prior_iqr = sbi_loo_predictive_check(
    x_summary_copula[:, num_observables:],
    obs_copula[num_observables:],
    observable_names,
)
fig_loo_prior_iqr.savefig(SAVE_DIR / 'loo_prior_predictive_iqrs.png', dpi=150, bbox_inches='tight')
plt.close(fig_loo_prior_iqr)

k_eff_prior_iqr, _ = participation_ratio(x_summary_copula[:, num_observables:])
d2_prior_iqr = loo_prior_iqr_df['d_full'].iloc[0]
chi2_p_prior_iqr = 1 - sp_stats.chi2.cdf(d2_prior_iqr, df=k_eff_prior_iqr)

print(f"\n  IQRs:    D² = {d2_prior_iqr:.1f}, k_eff = {k_eff_prior_iqr:.1f}/{num_observables}, "
      f"χ²(k_eff) p = {chi2_p_prior_iqr:.3f}")

print(f"\n  Top 5 structural misspecification drivers (IQRs):")
for _, row in loo_prior_iqr_df.head(5).iterrows():
    print(f"    {row['observable']}: influence = {row['influence']:.1f}")


# ============================================================================
# 5. TRANSFORM + TRAIN NPE
# ============================================================================
print("\n" + "=" * 80)
print("5. TRANSFORM + TRAIN NPE")
print("=" * 80)

# --- 4a. Train/test split ---
n_total = phi_clean.shape[0]
n_test = max(1, int(n_total * 0.1))
n_train = n_total - n_test

rng_split = np.random.default_rng(SPLIT_SEED)
indices = rng_split.permutation(n_total)
train_idx = indices[:n_train]
test_idx = indices[n_train:]

phi_train_np = phi_clean[train_idx]
phi_test_np = phi_clean[test_idx]
x_train_np = x_summary_clean[train_idx]
x_test_np = x_summary_clean[test_idx]

print(f"  Train: {n_train}, Test: {n_test}")

# --- 4b. Gaussian copula on x_summary (70D) ---
obs_quantiles = compute_quantiles_from_array(x_train_np)
x_train_copula = transform_to_normal_from_array(x_train_np, obs_quantiles)
x_test_copula = transform_to_normal_from_array(x_test_np, obs_quantiles)

# Filter any NaN/inf from copula transform
train_valid = np.all(np.isfinite(x_train_copula), axis=1) & np.all(np.isfinite(phi_train_np), axis=1)
test_valid = np.all(np.isfinite(x_test_copula), axis=1) & np.all(np.isfinite(phi_test_np), axis=1)

phi_train_np = phi_train_np[train_valid]
x_train_copula = x_train_copula[train_valid]
phi_test_np = phi_test_np[test_valid]
x_test_copula = x_test_copula[test_valid]

print(f"  After copula filter — Train: {phi_train_np.shape[0]}, Test: {phi_test_np.shape[0]}")

# --- 4c. Convert to tensors ---
phi_train = torch.tensor(phi_train_np, dtype=torch.float32)
phi_test = torch.tensor(phi_test_np, dtype=torch.float32)
x_train = torch.tensor(x_train_copula, dtype=torch.float32)
x_test = torch.tensor(x_test_copula, dtype=torch.float32)

print(f"  φ (hyperparams): {phi_train.shape[1]}D  (μ: {num_params}, log(ω): {num_params})")
print(f"  x (summary):     {x_train.shape[1]}D  (medians: {num_observables}, IQRs: {num_observables})")

# --- 4d. Build NPE prior for sbi ---
# The NPE prior must match the φ-space: [μ (normal-like), log(ω) (normal-like)]
# Use wide Normal priors in the transformed space
phi_mu_prior = [
    torch.distributions.Normal(
        torch.tensor([float(csv_mu[j])]),
        torch.tensor([TAU_MU * 3]),  # wider than hyperprior for NPE flexibility
    )
    for j in range(num_params)
]
phi_log_omega_prior = [
    torch.distributions.Normal(
        torch.tensor([float(np.log(csv_sigma[j]))]),
        torch.tensor([TAU_OMEGA * 3]),
    )
    for j in range(num_params)
]
npe_prior = MultipleIndependent(phi_mu_prior + phi_log_omega_prior)

# --- 4e. Train NPE ---
density_estimator = posterior_nn(
    model='nsf',
    hidden_features=NPE_HIDDEN_FEATURES,
    num_transforms=NPE_NUM_TRANSFORMS,
    num_bins=NPE_NUM_BINS,
)
print(f"  NSF: {NPE_NUM_TRANSFORMS} transforms, {NPE_HIDDEN_FEATURES} hidden, {NPE_NUM_BINS} bins")

inference = NPE(prior=npe_prior, density_estimator=density_estimator)
inference = inference.append_simulations(phi_train, x_train)
estimator = inference.train(
    training_batch_size=NPE_TRAINING_BATCH_SIZE,
    max_num_epochs=NPE_MAX_EPOCHS,
)
posterior = inference.build_posterior()
print(f"  Training complete")


# ============================================================================
# 6. EVALUATE NPE (φ-space)
# ============================================================================
print("\n" + "=" * 80)
print("6. EVALUATE NPE (φ-SPACE)")
print("=" * 80)

# Hyperparameter names for diagnostics
mu_names = [f"μ_{name}" for name in param_names]
omega_names = [f"ω_{name}" for name in param_names]
all_hp_names = mu_names + omega_names

print(f"  Sampling posterior for {phi_test.shape[0]} test cohorts (200 samples each)...")
with torch.no_grad():
    samples = posterior.posterior_estimator.sample(
        torch.Size([200]), condition=x_test
    )

# --- R² recovery for μ and ω separately ---
# μ recovery
_, _, r2_mu = sbi_recovery(
    samples[:, :, :num_params], phi_test[:, :num_params],
    param_names, plot=False,
)
r2_mu_array = np.array([r2_mu[name] for name in param_names])

# ω recovery (log-ω space)
_, _, r2_omega = sbi_recovery(
    samples[:, :, num_params:], phi_test[:, num_params:],
    param_names, plot=False,
)
r2_omega_array = np.array([r2_omega[name] for name in param_names])

# Combined R² for ranking
r2_all = np.concatenate([r2_mu_array, r2_omega_array])

print(f"\n  μ recovery — median R² = {np.median(r2_mu_array):.3f}")
mu_ranking = np.argsort(r2_mu_array)[::-1]
print(f"  Top 5 μ:")
for i in mu_ranking[:5]:
    print(f"    μ_{param_names[i]}: R² = {r2_mu_array[i]:.3f}")
print(f"  Bottom 5 μ:")
for i in mu_ranking[-5:]:
    print(f"    μ_{param_names[i]}: R² = {r2_mu_array[i]:.3f}")

print(f"\n  ω recovery — median R² = {np.median(r2_omega_array):.3f}")
omega_ranking = np.argsort(r2_omega_array)[::-1]
print(f"  Top 5 ω:")
for i in omega_ranking[:5]:
    print(f"    ω_{param_names[i]}: R² = {r2_omega_array[i]:.3f}")

# Plot top μ parameters
mu_plot_idx = mu_ranking[:N_PLOT_PARAMS].tolist()
fig_mu_recovery, _, _ = sbi_recovery(
    samples[:, :, :num_params], phi_test[:, :num_params],
    param_names, param_indices=mu_plot_idx,
)
fig_mu_recovery.suptitle('μ (population center) recovery', fontsize=14)
fig_mu_recovery.savefig(SAVE_DIR / 'recovery_mu.png', dpi=150, bbox_inches='tight')
plt.close(fig_mu_recovery)

# Plot top ω parameters
omega_plot_idx = omega_ranking[:N_PLOT_PARAMS].tolist()
fig_omega_recovery, _, _ = sbi_recovery(
    samples[:, :, num_params:], phi_test[:, num_params:],
    param_names, param_indices=omega_plot_idx,
)
fig_omega_recovery.suptitle('ω (population spread) recovery', fontsize=14)
fig_omega_recovery.savefig(SAVE_DIR / 'recovery_omega.png', dpi=150, bbox_inches='tight')
plt.close(fig_omega_recovery)

# Z-score contraction for μ
npe_prior_var = torch.var(npe_prior.sample((2000,)), dim=0).numpy()
mu_prior_var = npe_prior_var[:num_params]

_, _, z_scores_mu, contractions_mu = sbi_z_score_contraction(
    samples[:, :, :num_params], phi_test[:, :num_params],
    MultipleIndependent(phi_mu_prior), param_names,
    param_indices=mu_plot_idx, prior_var=mu_prior_var,
)
median_contraction_mu = float(np.median(contractions_mu))
print(f"\n  μ median z-score contraction: {median_contraction_mu:.3f}")


# ============================================================================
# 7. POSTERIOR FROM OBSERVED DATA
# ============================================================================
print("\n" + "=" * 80)
print("7. POSTERIOR FROM OBSERVED DATA")
print("=" * 80)

# Transform observed conditioning vector through copula
obs_transformed = transform_to_normal_from_array(obs_vector[np.newaxis, :], obs_quantiles)[0]
obs_transformed_tensor = torch.tensor(obs_transformed, dtype=torch.float32).unsqueeze(0)

print(f"  Observed vector: {obs_vector.shape[0]}D → copula-transformed")
print(f"  Sampling 2000 posterior draws...")
with torch.no_grad():
    post_phi = posterior.posterior_estimator.sample(
        torch.Size([2000]), condition=obs_transformed_tensor
    ).squeeze(1)  # (2000, 388)

# Split into μ and ω
post_mu = post_phi[:, :num_params].numpy()         # (2000, 194) log-space
post_log_omega = post_phi[:, num_params:].numpy()   # (2000, 194) log of ω
post_omega = np.exp(post_log_omega)                  # (2000, 194) original ω

print(f"  Posterior μ shape: {post_mu.shape}")
print(f"  Posterior ω shape: {post_omega.shape}")

# Summary: compare posterior μ to CSV priors
print(f"\n  Posterior μ summary (top 5 by μ R²):")
for i in mu_ranking[:5]:
    mu_med = np.median(post_mu[:, i])
    mu_5, mu_95 = np.percentile(post_mu[:, i], [5, 95])
    print(f"    μ_{param_names[i]}: median={mu_med:.3f} "
          f"(90%CI=[{mu_5:.3f}, {mu_95:.3f}], prior={csv_mu[i]:.3f})")

print(f"\n  Posterior ω summary (top 5 by ω R²):")
for i in omega_ranking[:5]:
    om_med = np.median(post_omega[:, i])
    om_5, om_95 = np.percentile(post_omega[:, i], [5, 95])
    print(f"    ω_{param_names[i]}: median={om_med:.3f} "
          f"(90%CI=[{om_5:.3f}, {om_95:.3f}], prior_σ={csv_sigma[i]:.3f})")


# ============================================================================
# 8. VIRTUAL PATIENT GENERATION
# ============================================================================
print("\n" + "=" * 80)
print("8. VIRTUAL PATIENT GENERATION")
print("=" * 80)

# Generate VPs by sampling from posterior (μ, ω)
# For each posterior draw, sample one patient: θ = exp(Normal(μ, ω))
print(f"  Generating {N_POSTERIOR_SIMS} virtual patients from posterior...")
rng_vp = np.random.default_rng(SEED + 100)

# Subsample posterior draws
n_post = post_mu.shape[0]
vp_idx = rng_vp.choice(n_post, size=N_POSTERIOR_SIMS, replace=N_POSTERIOR_SIMS > n_post)

vp_mu = post_mu[vp_idx]       # (500, 194)
vp_omega = post_omega[vp_idx]  # (500, 194)

# Sample patient parameters: log-space normal, then exponentiate
vp_log_theta = rng_vp.normal(vp_mu, vp_omega)  # (500, 194)
vp_theta = np.exp(vp_log_theta)                  # original space

print(f"  VP parameters shape: {vp_theta.shape}")
print(f"  VP θ median range: [{np.median(vp_theta, axis=0).min():.2e}, "
      f"{np.median(vp_theta, axis=0).max():.2e}]")


# ============================================================================
# 9. POSTERIOR PREDICTIVE CHECK
# ============================================================================
print("\n" + "=" * 80)
print("9. POSTERIOR PREDICTIVE CHECK")
print("=" * 80)

# --- 8a. Simulate VPs ---
print(f"  Running {N_POSTERIOR_SIMS} VP simulations...")
t0 = time.time()
x_vp = sim.simulate_with_parameters(vp_theta, pool_suffix='hierarchical_posterior_predictive')
elapsed = time.time() - t0
print(f"  Simulations complete in {elapsed:.0f}s")

# Filter valid
valid_mask = np.all(np.isfinite(x_vp), axis=1)
x_vp_clean = x_vp[valid_mask]
print(f"  Valid VP simulations: {valid_mask.sum()}/{N_POSTERIOR_SIMS}")

# --- 8b. Per-observable coverage ---
obs_original = get_observed_data(summary_csv_path, value_column='median')

print("\n  Per-observable posterior predictive coverage:")
coverage_df, fig_coverage = sbi_posterior_predictive_coverage(
    x_vp_clean, obs_original, observable_names, log_scale=True
)
fig_coverage.savefig(SAVE_DIR / 'posterior_predictive_coverage.png', dpi=150, bbox_inches='tight')
plt.close(fig_coverage)

n_in_ci = int(coverage_df['in_90ci'].sum())
print(f"  {n_in_ci}/{num_observables} observables within 90% posterior predictive CI")
for _, row in coverage_df[~coverage_df['in_90ci']].iterrows():
    print(f"    OUTSIDE: {row['observable']}: obs={row['obs_value']:.3g}, "
          f"90%CI=[{row['pp_q05']:.3g}, {row['pp_q95']:.3g}], p={row['p_value']:.4f}")

# --- 8c. Cohort-level check: simulate cohorts from posterior and compare summaries ---
print("\n  Cohort-level posterior predictive check...")
n_check_cohorts = min(100, n_post)
check_idx = rng_vp.choice(n_post, size=n_check_cohorts, replace=False)

cohort_medians_pp = []
cohort_iqrs_pp = []

# Generate cohort-level summaries from posterior
for ci in range(n_check_cohorts):
    mu_i = post_mu[check_idx[ci]]
    omega_i = post_omega[check_idx[ci]]

    # Draw N_COHORT patients
    log_theta_cohort = rng_vp.normal(
        np.repeat(mu_i[np.newaxis, :], N_COHORT, axis=0),
        np.repeat(omega_i[np.newaxis, :], N_COHORT, axis=0),
    )
    theta_cohort = np.exp(log_theta_cohort)

    # Simulate
    x_cohort = sim.simulate_with_parameters(theta_cohort, pool_suffix='hierarchical_cohort_check')

    # Compute summaries
    valid_c = np.all(np.isfinite(x_cohort), axis=1)
    if valid_c.sum() >= min_patients_per_cohort:
        x_valid = x_cohort[valid_c]
        cohort_medians_pp.append(np.median(x_valid, axis=0))
        cohort_iqrs_pp.append(
            np.percentile(x_valid, 75, axis=0) - np.percentile(x_valid, 25, axis=0)
        )

cohort_medians_pp = np.array(cohort_medians_pp)  # (n_check, 35)
cohort_iqrs_pp = np.array(cohort_iqrs_pp)        # (n_check, 35)

print(f"  Posterior cohort summaries: {cohort_medians_pp.shape[0]} valid cohorts")

# Check: what fraction of observed medians fall within 90% CI of posterior cohort medians?
n_median_covered = 0
n_iqr_covered = 0
for j in range(num_observables):
    med_lo, med_hi = np.percentile(cohort_medians_pp[:, j], [5, 95])
    if med_lo <= obs_medians[j] <= med_hi:
        n_median_covered += 1

    iqr_lo, iqr_hi = np.percentile(cohort_iqrs_pp[:, j], [5, 95])
    if iqr_lo <= obs_iqrs[j] <= iqr_hi:
        n_iqr_covered += 1

print(f"  Observed medians within 90% CI of posterior cohort medians: "
      f"{n_median_covered}/{num_observables}")
print(f"  Observed IQRs within 90% CI of posterior cohort IQRs: "
      f"{n_iqr_covered}/{num_observables}")

# --- 8d. LOO decomposition ---
# VP observables are individual-patient 35D (not cohort summaries),
# so compute fresh copula quantiles from the VP observables themselves
vp_quantiles = compute_quantiles_from_array(x_vp_clean)
x_vp_copula = transform_to_normal_from_array(x_vp_clean, vp_quantiles)
valid_copula = np.all(np.isfinite(x_vp_copula), axis=1)
x_vp_copula = x_vp_copula[valid_copula]

# Observed medians in VP copula space
obs_median_transformed = transform_to_normal_from_array(
    obs_medians[np.newaxis, :], vp_quantiles
)[0]

if x_vp_copula.shape[0] >= 10:
    print(f"\n  Posterior predictive LOO ({x_vp_copula.shape[0]} samples)...")
    loo_post_df, fig_loo_post = sbi_loo_predictive_check(
        x_vp_copula, obs_median_transformed, observable_names,
    )
    fig_loo_post.savefig(SAVE_DIR / 'loo_posterior_predictive.png', dpi=150, bbox_inches='tight')
    plt.close(fig_loo_post)

    d2_post = loo_post_df['d_full'].iloc[0]
    print(f"  D² = {d2_post:.1f}")
    print(f"\n  Top 5 posterior misfit drivers:")
    for _, row in loo_post_df.head(5).iterrows():
        print(f"    {row['observable']}: influence = {row['influence']:.1f}")
else:
    print("  Too few valid copula-transformed samples for LOO check")
    loo_post_df = None


# ============================================================================
# 10. SAVE
# ============================================================================
print("\n" + "=" * 80)
print("10. SAVE RESULTS")
print("=" * 80)

# --- 9a. Posterior hyperparameters ---
post_mu_df = pd.DataFrame(post_mu, columns=[f"mu_{n}" for n in param_names])
post_omega_df = pd.DataFrame(post_omega, columns=[f"omega_{n}" for n in param_names])
post_hp_df = pd.concat([post_mu_df, post_omega_df], axis=1)
post_hp_df.to_csv(SAVE_DIR / 'posterior_hyperparameters.csv', index=False)
print(f"  Posterior hyperparameters: {SAVE_DIR}/posterior_hyperparameters.csv")

# --- 9b. Posterior summary ---
post_mu_summary = post_mu_df.describe(percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]).T
post_mu_summary['prior_value'] = csv_mu
post_mu_summary.to_csv(SAVE_DIR / 'posterior_mu_summary.csv')

post_omega_summary = post_omega_df.describe(percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]).T
post_omega_summary['prior_value'] = csv_sigma
post_omega_summary.to_csv(SAVE_DIR / 'posterior_omega_summary.csv')

# --- 9c. VP parameters and observables ---
vp_df = pd.DataFrame(vp_theta[valid_mask], columns=param_names)
vp_df.to_csv(SAVE_DIR / 'virtual_patient_parameters.csv', index=False)

vp_obs_df = pd.DataFrame(x_vp_clean, columns=observable_names)
vp_obs_df.to_csv(SAVE_DIR / 'virtual_patient_observables.csv', index=False)
print(f"  Virtual patients: {vp_df.shape[0]} parameter sets + observables")

# --- 9d. Diagnostics ---
diagnostics_dir = SAVE_DIR / 'diagnostics'

# Combine R² values into single dict for save_diagnostics
r2_combined = {}
for name in param_names:
    r2_combined[f"mu_{name}"] = r2_mu[name]
    r2_combined[f"omega_{name}"] = r2_omega[name]

written = save_diagnostics(
    diagnostics_dir,
    r2_values=r2_combined,
    z_scores=z_scores_mu,
    contractions=contractions_mu,
    param_names=mu_names,  # z-scores are only for μ
    ppc_df=coverage_df,
    training_metadata={
        'n_training_cohorts': N_TRAINING_COHORTS,
        'n_cohort': N_COHORT,
        'n_total_patients': n_total_patients,
        'n_valid_cohorts': int(n_valid_cohorts),
        'n_train': int(phi_train.shape[0]),
        'n_test': int(phi_test.shape[0]),
        'n_params': num_params,
        'n_hyperparams': 2 * num_params,
        'n_observables': num_observables,
        'n_summary_dim': 2 * num_observables,
        'n_posterior_sims': N_POSTERIOR_SIMS,
        'model_version': MODEL_VERSION,
        'scenario': SCENARIO,
        'tau_mu': TAU_MU,
        'tau_omega': TAU_OMEGA,
        'npe_hidden_features': NPE_HIDDEN_FEATURES,
        'npe_num_transforms': NPE_NUM_TRANSFORMS,
        'npe_num_bins': NPE_NUM_BINS,
        'median_r2_mu': float(np.median(r2_mu_array)),
        'median_r2_omega': float(np.median(r2_omega_array)),
        'median_contraction_mu': median_contraction_mu,
    },
)

if loo_post_df is not None:
    loo_post_df.to_csv(diagnostics_dir / 'loo_posterior_predictive.csv', index=False)
coverage_df.to_csv(diagnostics_dir / 'posterior_predictive_coverage.csv', index=False)

# Cohort-level check results
cohort_check_df = pd.DataFrame({
    'observable': observable_names,
    'obs_median': obs_medians,
    'obs_iqr': obs_iqrs,
    'pp_cohort_median_q05': np.percentile(cohort_medians_pp, 5, axis=0),
    'pp_cohort_median_q95': np.percentile(cohort_medians_pp, 95, axis=0),
    'pp_cohort_iqr_q05': np.percentile(cohort_iqrs_pp, 5, axis=0),
    'pp_cohort_iqr_q95': np.percentile(cohort_iqrs_pp, 95, axis=0),
})
cohort_check_df['median_covered'] = (
    (cohort_check_df['pp_cohort_median_q05'] <= cohort_check_df['obs_median']) &
    (cohort_check_df['obs_median'] <= cohort_check_df['pp_cohort_median_q95'])
)
cohort_check_df['iqr_covered'] = (
    (cohort_check_df['pp_cohort_iqr_q05'] <= cohort_check_df['obs_iqr']) &
    (cohort_check_df['obs_iqr'] <= cohort_check_df['pp_cohort_iqr_q95'])
)
cohort_check_df.to_csv(diagnostics_dir / 'cohort_level_check.csv', index=False)

print(f"  Diagnostics: {diagnostics_dir}/")
for key, path in written.items():
    print(f"    {key}")

print("\n" + "=" * 80)
print("HIERARCHICAL SBI WORKFLOW COMPLETE")
print("=" * 80)
