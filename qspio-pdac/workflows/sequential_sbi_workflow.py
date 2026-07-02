#!/usr/bin/env python3
"""
Sequential NPE (SNPE-C) Workflow for PDAC QSP Model

Multi-round SBI using SNPE-C (APT; Greenberg et al., 2019). Each round
samples from the previous round's posterior, simulates, and retrains the NPE
with automatic atomic proposal correction. This focuses simulations in the
high-posterior region, dramatically improving sample efficiency for
high-dimensional parameter spaces.

How it works:
  - Round 1: sample from prior, simulate, train NPE
  - Round R+1: sample from round R's posterior (conditioned on x_obs),
    simulate, append to inference with proposal=posterior, retrain
  - sbi handles proposal correction automatically (atomic loss)

Convergence is tracked via:
  - Round-over-round R² recovery on a fixed test set
  - Posterior sample stability (Wasserstein distance between consecutive rounds)
  - Z-score contraction (posterior narrowing relative to prior)
  - Posterior predictive coverage (final round only)

Based on sbi_workflow.py — same data pipeline (NaN filtering, observation noise,
Gaussian copula transform, NSF architecture).

Usage:
    python workflows/sequential_sbi_workflow.py
"""

import time
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn

from qsp_hpc import QSPSimulator
from qsp_inference.inference import (
    get_observed_data,
    convert_posterior_samples_to_original_space,
    sbi_recovery,
    sbi_z_score_contraction,
    sbi_loo_predictive_check,
    sbi_posterior_predictive_coverage,
    save_diagnostics,
)
from qsp_inference.inference.data_processing import _filter_nans, add_observation_noise
from qsp_inference.inference.gaussian_copula_transform import (
    compute_quantiles_from_array,
    transform_to_normal_from_array,
)
from qsp_inference.priors import load_prior, transform_lognormal_prior_to_normal


def participation_ratio(X):
    """Participation ratio of covariance eigenvalue spectrum.

    k_eff = (sum lambda_i)^2 / sum(lambda_i^2)
    """
    cov = np.cov(X, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = eigenvalues[eigenvalues > 0]
    k_eff = float(np.sum(eigenvalues)**2 / np.sum(eigenvalues**2))
    return k_eff, eigenvalues


# ============================================================================
# 1. CONFIG + SETUP
# ============================================================================
print("=" * 80)
print("SEQUENTIAL NPE (SNPE-C) WORKFLOW: BASELINE SCENARIO")
print("=" * 80)

MODEL_VERSION = 'baseline_allparams_v2'
MODEL_DESCRIPTION = 'PDAC sequential NPE (SNPE-C): multi-round focused inference'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'
SCENARIO = 'baseline_no_treatment'

PRIORS_CSV = 'parameters/pdac_priors.csv'
CALIBRATION_TARGETS_DIR = 'calibration_targets/baseline_no_treatment'
MODEL_STRUCTURE_FILE = 'model_structure.json'

# --- Sequential NPE settings ---
NUM_ROUNDS = 10
N_SIMS_ROUND1 = 500    # large initial round (prior sampling, leverages pool)
N_SIMS_SUBSEQUENT = 500   # smaller focused rounds (posterior proposals, expensive)
TEST_FRACTION = 0.1        # held out from round 1 only

# --- Simulation ---
MAX_TASKS = 40
SEED = 2025
SPLIT_SEED = 2027

# --- NPE architecture ---
NPE_HIDDEN_FEATURES = 50
NPE_NUM_TRANSFORMS = 5
NPE_NUM_BINS = 10
NPE_TRAINING_BATCH_SIZE = 128
NPE_MAX_EPOCHS = 500

# --- Diagnostics ---
N_POSTERIOR_SAMPLES = 2000
N_POSTERIOR_SIMS = 500
N_PLOT_PARAMS = 20

# --- Convergence thresholds ---
CONVERGENCE_WASSERSTEIN_THRESHOLD = 0.01  # stop if max W < this (normalized by prior sigma)
MIN_ROUNDS = 3  # always run at least this many rounds

SAVE_DIR = Path('figures/sbi_sequential')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# --- Load prior ---
prior_lognormal = load_prior(PRIORS_CSV)
prior_normal = transform_lognormal_prior_to_normal(prior_lognormal)
prior_var = torch.var(prior_normal.sample((2000,)), dim=0).numpy()

priors_df = pd.read_csv(PRIORS_CSV)
param_names = priors_df['name'].tolist()
num_params = len(param_names)

# --- Setup simulator ---
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
    cache_sampling_seed=SEED,
)

target_df = sim._test_stats_df.copy()
observable_names = target_df['test_statistic_id'].tolist()
num_observables = len(observable_names)

# Save summary CSV for observed data loading
Path('cache').mkdir(exist_ok=True)
summary_csv_path = 'cache/baseline_calibration_targets_summary.csv'
target_df.to_csv(summary_csv_path, index=False)

# --- Observed data ---
obs_original = get_observed_data(summary_csv_path, value_column='median')
obs_values = np.array([np.squeeze(obs_original[name]) for name in observable_names])

print(f"  Parameters: {num_params}, Observables: {num_observables}")
print(f"  Rounds: {NUM_ROUNDS}, Round 1: {N_SIMS_ROUND1} sims, "
      f"Rounds 2+: {N_SIMS_SUBSEQUENT} sims/round")
print(f"  NSF: {NPE_NUM_TRANSFORMS} transforms, {NPE_HIDDEN_FEATURES} hidden, {NPE_NUM_BINS} bins")


# ============================================================================
# 2. SEQUENTIAL NPE LOOP
# ============================================================================
print("\n" + "=" * 80)
print("2. SEQUENTIAL NPE LOOP")
print("=" * 80)

# Fixed across rounds (set in round 1)
theta_test_t = None       # fixed test set (log-space, tensor)
x_test_t = None           # fixed test set (copula-space, tensor)
obs_quantiles = None      # copula quantiles (from round 1 training data)
obs_transformed = None    # observed data in copula space
obs_tensor = None         # observed data tensor
x_r1_train_copula = None  # round 1 training data in copula space (for prior predictive)

# Convergence tracking
round_metrics = []
prev_post_samples = None
proposal = prior_normal
posterior = None

# Single inference object across all rounds — sbi accumulates data internally
density_estimator_builder = posterior_nn(
    model='nsf',
    hidden_features=NPE_HIDDEN_FEATURES,
    num_transforms=NPE_NUM_TRANSFORMS,
    num_bins=NPE_NUM_BINS,
)
inference = NPE(prior=prior_normal, density_estimator=density_estimator_builder)

for round_idx in range(NUM_ROUNDS):
    t_round_start = time.time()
    print(f"\n{'─' * 60}")
    print(f"  ROUND {round_idx + 1}/{NUM_ROUNDS}")
    print(f"{'─' * 60}")

    # ── 2a. Sample & simulate ──────────────────────────────────────────
    n_sims = N_SIMS_ROUND1 if round_idx == 0 else N_SIMS_SUBSEQUENT

    if round_idx == 0:
        # Round 1: sample from prior via simulator (leverages pool)
        print(f"  Sampling {n_sims} from prior + simulating...")
        t0 = time.time()
        theta_orig, x_raw = sim(n_sims)
        sim_elapsed = time.time() - t0
    else:
        # Round 2+: sample from posterior (proposal), simulate
        print(f"  Sampling {n_sims} from round {round_idx} posterior...")
        with torch.no_grad():
            theta_log_round = proposal.sample(
                (n_sims,), x=obs_tensor.squeeze(0)
            )
        theta_orig = np.exp(theta_log_round.numpy())
        print(f"  Simulating {n_sims} parameter sets...")
        t0 = time.time()
        x_raw = sim.simulate_with_parameters(theta_orig)
        sim_elapsed = time.time() - t0

    print(f"  Simulations complete in {sim_elapsed:.0f}s")

    # Replace inf with NaN
    n_inf = np.isinf(x_raw).sum()
    if n_inf > 0:
        print(f"  Replacing {n_inf} inf values with NaN")
        x_raw[np.isinf(x_raw)] = np.nan

    # ── 2b. Filter NaNs ────────────────────────────────────────────────
    theta_orig_clean, x_clean = _filter_nans(theta_orig, x_raw)
    round_nan_rate = 1 - len(theta_orig_clean) / len(theta_orig)
    print(f"  Valid this round: {len(theta_orig_clean)}/{len(theta_orig)} "
          f"({1 - round_nan_rate:.0%} survived)")

    # ── 2c. Round 1: establish fixed test set + copula ─────────────────
    if round_idx == 0:
        # Split into train/test (test set is fixed for all future rounds)
        n_test = max(1, int(len(theta_orig_clean) * TEST_FRACTION))
        rng_split = np.random.default_rng(SPLIT_SEED)
        indices = rng_split.permutation(len(theta_orig_clean))
        test_idx = indices[:n_test]
        train_idx = indices[n_test:]

        theta_test_orig = theta_orig_clean[test_idx]
        x_test_raw = x_clean[test_idx]
        theta_train_orig_r1 = theta_orig_clean[train_idx]
        x_train_raw_r1 = x_clean[train_idx]

        # Add observation noise to round 1 training data
        x_train_noisy_r1 = add_observation_noise(
            x_train_raw_r1,
            ci95_lower=target_df['ci95_lower'].values,
            ci95_upper=target_df['ci95_upper'].values,
            medians=target_df['median'].values,
            seed=SEED + 1,
            sample_sizes=target_df['sample_size'].values,
        )

        # Fit copula on round 1 training data (FIXED for all rounds)
        obs_quantiles = compute_quantiles_from_array(x_train_noisy_r1, n_quantiles=1000)

        # Transform round 1 training data
        x_train_copula = transform_to_normal_from_array(x_train_noisy_r1, obs_quantiles)
        theta_train_log = np.log(theta_train_orig_r1)

        # Transform test data (fixed forever)
        x_test_copula = transform_to_normal_from_array(x_test_raw, obs_quantiles)
        theta_test_log = np.log(theta_test_orig)

        # Filter NaN/inf from copula transforms
        train_valid = (np.all(np.isfinite(x_train_copula), axis=1) &
                       np.all(np.isfinite(theta_train_log), axis=1))
        test_valid = (np.all(np.isfinite(x_test_copula), axis=1) &
                      np.all(np.isfinite(theta_test_log), axis=1))

        theta_train_log = theta_train_log[train_valid]
        x_train_copula = x_train_copula[train_valid]
        theta_test_log = theta_test_log[test_valid]
        x_test_copula = x_test_copula[test_valid]

        # Fixed test set tensors
        theta_test_t = torch.tensor(theta_test_log, dtype=torch.float32)
        x_test_t = torch.tensor(x_test_copula, dtype=torch.float32)

        # Fixed observed data in copula space
        obs_transformed = transform_to_normal_from_array(
            obs_values[np.newaxis, :], obs_quantiles
        )[0]
        obs_tensor = torch.tensor(obs_transformed, dtype=torch.float32).unsqueeze(0)

        # Store round 1 copula data for prior predictive check later
        x_r1_train_copula = x_train_copula.copy()

        # Append round 1 data to inference (proposal=prior)
        theta_round_t = torch.tensor(theta_train_log, dtype=torch.float32)
        x_round_t = torch.tensor(x_train_copula, dtype=torch.float32)
        inference = inference.append_simulations(
            theta_round_t, x_round_t, proposal=prior_normal
        )

        print(f"  Round 1 setup: {len(theta_train_log)} train, "
              f"{len(theta_test_log)} test (fixed)")

    else:
        # Round 2+: noise + transform with FIXED copula, append with proposal
        x_clean_noisy = add_observation_noise(
            x_clean,
            ci95_lower=target_df['ci95_lower'].values,
            ci95_upper=target_df['ci95_upper'].values,
            medians=target_df['median'].values,
            seed=SEED + round_idx + 1,
            sample_sizes=target_df['sample_size'].values,
        )

        x_copula_round = transform_to_normal_from_array(x_clean_noisy, obs_quantiles)
        theta_log_round_np = np.log(theta_orig_clean)

        # Filter NaN/inf from copula (values outside round 1 quantile range)
        valid = (np.all(np.isfinite(x_copula_round), axis=1) &
                 np.all(np.isfinite(theta_log_round_np), axis=1))
        n_copula_valid = valid.sum()
        if n_copula_valid < len(valid):
            print(f"  Copula filter: {n_copula_valid}/{len(valid)} survived "
                  f"({len(valid) - n_copula_valid} outside round 1 quantile range)")

        theta_log_round_np = theta_log_round_np[valid]
        x_copula_round = x_copula_round[valid]

        # Append with proposal=posterior so sbi applies atomic proposal correction
        theta_round_t = torch.tensor(theta_log_round_np, dtype=torch.float32)
        x_round_t = torch.tensor(x_copula_round, dtype=torch.float32)
        inference = inference.append_simulations(
            theta_round_t, x_round_t, proposal=proposal
        )

        print(f"  Appended {len(theta_log_round_np)} samples (proposal=round {round_idx} posterior)")

    # ── 2d. Train NPE ─────────────────────────────────────────────────
    print("  Training NPE (all accumulated rounds)...")
    estimator = inference.train(
        training_batch_size=NPE_TRAINING_BATCH_SIZE,
        max_num_epochs=NPE_MAX_EPOCHS,
    )
    posterior = inference.build_posterior()
    print(f"  NPE training complete (round {round_idx + 1})")

    # ── 2e. Diagnostics ───────────────────────────────────────────────
    # R² recovery on fixed test set
    with torch.no_grad():
        test_samples = posterior.posterior_estimator.sample(
            torch.Size([200]), condition=x_test_t
        )
    _, _, r2_values = sbi_recovery(test_samples, theta_test_t, param_names, plot=False)
    r2_array = np.array([r2_values[name] for name in param_names])
    r2_ranking = np.argsort(r2_array)[::-1]

    r2_median = float(np.median(r2_array))
    r2_mean = float(np.mean(r2_array))
    n_r2_above_05 = int((r2_array > 0.5).sum())
    n_r2_positive = int((r2_array > 0).sum())

    print(f"  R² recovery: median={r2_median:.3f}, mean={r2_mean:.3f}, "
          f">0.5: {n_r2_above_05}, >0: {n_r2_positive}/{num_params}")
    print("  Top 5 R²:")
    for i in r2_ranking[:5]:
        print(f"    {param_names[i]}: R² = {r2_array[i]:.3f}")

    # Save recovery plot
    plot_indices = r2_ranking[:N_PLOT_PARAMS].tolist()
    fig_recovery, _, _ = sbi_recovery(
        test_samples, theta_test_t, param_names, param_indices=plot_indices
    )
    fig_recovery.suptitle(f'Round {round_idx + 1} Recovery (top {N_PLOT_PARAMS})', fontsize=14)
    fig_recovery.savefig(
        SAVE_DIR / f'recovery_round_{round_idx + 1}.png', dpi=150, bbox_inches='tight'
    )
    plt.close(fig_recovery)

    # Z-score contraction
    fig_zsc, _, z_scores, contractions = sbi_z_score_contraction(
        test_samples, theta_test_t, prior_normal, param_names,
        param_indices=plot_indices, prior_var=prior_var,
    )
    plt.close(fig_zsc)
    median_contraction = float(np.median(contractions))

    # Posterior samples conditioned on observed data
    with torch.no_grad():
        post_samples = posterior.posterior_estimator.sample(
            torch.Size([N_POSTERIOR_SAMPLES]), condition=obs_tensor
        ).squeeze(1)

    # Posterior stability (Wasserstein distance, normalized by prior sigma)
    wasserstein_max = float('nan')
    wasserstein_median = float('nan')
    if prev_post_samples is not None:
        w_distances = []
        for j in range(num_params):
            w = sp_stats.wasserstein_distance(
                prev_post_samples[:, j].numpy(),
                post_samples[:, j].numpy(),
            )
            prior_std = float(np.sqrt(prior_var[j]))
            w_distances.append(w / prior_std if prior_std > 0 else w)
        wasserstein_max = float(np.max(w_distances))
        wasserstein_median = float(np.median(w_distances))
        print(f"  Posterior stability: W_max={wasserstein_max:.4f}, "
              f"W_median={wasserstein_median:.4f} (normalized by prior sigma)")

    prev_post_samples = post_samples.clone()

    # Save per-round R²
    round_r2_df = pd.DataFrame({
        'parameter': param_names,
        'r2': r2_array,
    }).sort_values('r2', ascending=False)
    round_r2_df.to_csv(SAVE_DIR / f'r2_round_{round_idx + 1}.csv', index=False)

    # Track metrics
    round_time = time.time() - t_round_start
    metrics = {
        'round': round_idx + 1,
        'n_sims_round': n_sims,
        'n_valid_round': len(theta_orig_clean),
        'nan_rate_round': round_nan_rate,
        'r2_median': r2_median,
        'r2_mean': r2_mean,
        'n_r2_above_05': n_r2_above_05,
        'n_r2_positive': n_r2_positive,
        'median_contraction': median_contraction,
        'wasserstein_max': wasserstein_max,
        'wasserstein_median': wasserstein_median,
        'sim_time_s': sim_elapsed,
        'round_time_s': round_time,
    }
    round_metrics.append(metrics)

    # ── 2f. Set up proposal for next round ─────────────────────────────
    if round_idx < NUM_ROUNDS - 1:
        # The posterior itself becomes the proposal for the next round.
        # set_default_x makes it a distribution that can be sampled without
        # passing x each time, and tells sbi which observation to condition on.
        proposal = posterior.set_default_x(obs_tensor.squeeze(0))
        print(f"  Proposal for round {round_idx + 2}: posterior conditioned on x_obs")

        # Early stopping
        if (round_idx + 1 >= MIN_ROUNDS
                and not np.isnan(wasserstein_max)
                and wasserstein_max < CONVERGENCE_WASSERSTEIN_THRESHOLD):
            print(f"\n  CONVERGED: W_max={wasserstein_max:.4f} < "
                  f"{CONVERGENCE_WASSERSTEIN_THRESHOLD} (threshold)")
            break


# ============================================================================
# 3. CONVERGENCE SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("3. CONVERGENCE SUMMARY")
print("=" * 80)

metrics_df = pd.DataFrame(round_metrics)
n_rounds_completed = len(metrics_df)
metrics_df.to_csv(SAVE_DIR / 'convergence_metrics.csv', index=False)

print(f"\n  {'Round':<7} {'N_valid':<9} {'R²_med':<9} {'R²>0.5':<8} "
      f"{'Contr.':<8} {'W_max':<9} {'W_med':<9}")
print(f"  {'─'*7} {'─'*9} {'─'*9} {'─'*8} {'─'*8} {'─'*9} {'─'*9}")
for _, row in metrics_df.iterrows():
    w_max_s = f"{row['wasserstein_max']:<9.4f}" if not np.isnan(row['wasserstein_max']) else "   ---  "
    w_med_s = f"{row['wasserstein_median']:<9.4f}" if not np.isnan(row['wasserstein_median']) else "   ---  "
    print(f"  {int(row['round']):<7} {int(row['n_valid_round']):<9} "
          f"{row['r2_median']:<9.3f} {int(row['n_r2_above_05']):<8} "
          f"{row['median_contraction']:<8.3f} {w_max_s} {w_med_s}")

# Convergence plot
fig_conv, axes = plt.subplots(2, 2, figsize=(12, 8))
fig_conv.suptitle(f'Sequential NPE Convergence ({n_rounds_completed} rounds)', fontsize=14)

rounds = metrics_df['round'].values

ax = axes[0, 0]
ax.plot(rounds, metrics_df['r2_median'], 'o-', label='median R²')
ax.plot(rounds, metrics_df['r2_mean'], 's--', label='mean R²', alpha=0.7)
ax.set_xlabel('Round')
ax.set_ylabel('R²')
ax.set_title('Parameter Recovery')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.plot(rounds, metrics_df['n_r2_above_05'], 'o-', color='green')
ax.set_xlabel('Round')
ax.set_ylabel('Count')
ax.set_title('Parameters with R² > 0.5')
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
ax.plot(rounds, metrics_df['median_contraction'], 'o-', color='purple')
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Round')
ax.set_ylabel('Contraction')
ax.set_title('Median Z-score Contraction')
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
w_max = metrics_df['wasserstein_max'].values
w_med = metrics_df['wasserstein_median'].values
valid_w = ~np.isnan(w_max)
if valid_w.sum() > 0:
    ax.plot(rounds[valid_w], w_max[valid_w], 'o-', label='max W')
    ax.plot(rounds[valid_w], w_med[valid_w], 's--', label='median W', alpha=0.7)
    ax.axhline(y=CONVERGENCE_WASSERSTEIN_THRESHOLD, color='red', linestyle=':', alpha=0.5,
               label=f'threshold ({CONVERGENCE_WASSERSTEIN_THRESHOLD})')
    ax.legend()
ax.set_xlabel('Round')
ax.set_ylabel('Wasserstein (normalized by prior sigma)')
ax.set_title('Posterior Stability')
ax.grid(True, alpha=0.3)

fig_conv.tight_layout()
fig_conv.savefig(SAVE_DIR / 'convergence.png', dpi=150, bbox_inches='tight')
plt.close(fig_conv)


# ============================================================================
# 4. PRIOR PREDICTIVE CHECK
# ============================================================================
print("\n" + "=" * 80)
print("4. PRIOR PREDICTIVE CHECK")
print("=" * 80)

# Use round 1 training data (prior samples) for prior predictive LOO
k_eff_prior, _ = participation_ratio(x_r1_train_copula)

loo_prior_df, fig_loo_prior = sbi_loo_predictive_check(
    x_r1_train_copula, obs_transformed, observable_names,
)
fig_loo_prior.savefig(SAVE_DIR / 'loo_prior_predictive.png', dpi=150, bbox_inches='tight')
plt.close(fig_loo_prior)

d2_prior = loo_prior_df['d_full'].iloc[0]
chi2_p_prior = 1 - sp_stats.chi2.cdf(d2_prior, df=k_eff_prior)
d2_z_prior = (d2_prior - k_eff_prior) / np.sqrt(2 * k_eff_prior)

print(f"  D² = {d2_prior:.1f}, k_eff = {k_eff_prior:.1f}/{num_observables}, "
      f"chi2(k_eff) p = {chi2_p_prior:.3f} ({d2_z_prior:+.1f} sigma)")


# ============================================================================
# 5. POSTERIOR FROM OBSERVED DATA (final round)
# ============================================================================
print("\n" + "=" * 80)
print("5. POSTERIOR FROM OBSERVED DATA")
print("=" * 80)

# post_samples already computed in last round of the loop
post_samples_dict = convert_posterior_samples_to_original_space(post_samples, param_names)
print(f"  Posterior samples: {post_samples.shape}")

post_df_summary = pd.DataFrame(post_samples_dict).describe(percentiles=[0.05, 0.5, 0.95]).T
print("  Posterior summary (top 5 by R²):")
for i in r2_ranking[:5]:
    name = param_names[i]
    row = post_df_summary.loc[name]
    print(f"    {name}: median={row['50%']:.3g}, 90%CI=[{row['5%']:.3g}, {row['95%']:.3g}]")


# ============================================================================
# 6. POSTERIOR PREDICTIVE CHECK (actual simulations)
# ============================================================================
print("\n" + "=" * 80)
print("6. POSTERIOR PREDICTIVE CHECK (ACTUAL SIMULATIONS)")
print("=" * 80)

rng = np.random.default_rng(SEED)
idx = rng.choice(post_samples.shape[0], size=N_POSTERIOR_SIMS, replace=False)
theta_posterior = post_samples[idx].numpy()
theta_posterior_orig = np.exp(theta_posterior)

print(f"  Running {N_POSTERIOR_SIMS} posterior predictive simulations...")
t0 = time.time()
x_posterior_pred = sim.simulate_with_parameters(theta_posterior_orig)
elapsed = time.time() - t0
print(f"  Simulations complete in {elapsed:.0f}s")

valid_mask = np.all(np.isfinite(x_posterior_pred), axis=1)
x_posterior_pred_clean = x_posterior_pred[valid_mask]
print(f"  Valid posterior predictive sims: {valid_mask.sum()}/{N_POSTERIOR_SIMS}")

print("\n  Per-observable posterior predictive coverage:")
coverage_df, fig_coverage = sbi_posterior_predictive_coverage(
    x_posterior_pred_clean, obs_original, observable_names, log_scale=True
)
fig_coverage.savefig(SAVE_DIR / 'posterior_predictive_coverage.png', dpi=150, bbox_inches='tight')
plt.close(fig_coverage)

n_in_ci = int(coverage_df['in_90ci'].sum())
print(f"  {n_in_ci}/{num_observables} observables within 90% posterior predictive CI")
for _, row in coverage_df[~coverage_df['in_90ci']].iterrows():
    print(f"    OUTSIDE: {row['observable']}: obs={row['obs_value']:.3g}, "
          f"90%CI=[{row['pp_q05']:.3g}, {row['pp_q95']:.3g}], p={row['p_value']:.4f}")

# Posterior predictive LOO
x_pp_copula = transform_to_normal_from_array(x_posterior_pred_clean, obs_quantiles)
valid_copula = np.all(np.isfinite(x_pp_copula), axis=1)
x_pp_copula = x_pp_copula[valid_copula]
print(f"  Copula-transformed posterior predictive: {x_pp_copula.shape[0]} valid samples")

loo_post_df, fig_loo_post = sbi_loo_predictive_check(
    x_pp_copula, obs_transformed, observable_names,
)
fig_loo_post.savefig(SAVE_DIR / 'loo_posterior_predictive.png', dpi=150, bbox_inches='tight')
plt.close(fig_loo_post)

k_eff_post, _ = participation_ratio(x_pp_copula)
d2_post = loo_post_df['d_full'].iloc[0]
chi2_p_post = 1 - sp_stats.chi2.cdf(d2_post, df=k_eff_post)

print(f"\n  {'Distribution':<35s} {'k_eff':>6s} {'D²':>7s} {'chi² p':>8s}")
print(f"  {'-'*35} {'-'*6} {'-'*7} {'-'*8}")
print(f"  {'Prior predictive':<35s} {k_eff_prior:6.1f} {d2_prior:7.1f} {chi2_p_prior:8.3f}")
print(f"  {'Posterior predictive (actual sims)':<35s} {k_eff_post:6.1f} {d2_post:7.1f} {chi2_p_post:8.3f}")

print("\n  Top 5 posterior misfit drivers:")
for _, row in loo_post_df.head(5).iterrows():
    print(f"    {row['observable']}: influence = {row['influence']:.1f}")


# ============================================================================
# 7. SAVE
# ============================================================================
print("\n" + "=" * 80)
print("7. SAVE RESULTS")
print("=" * 80)

posterior_df = pd.DataFrame(post_samples_dict)
posterior_df.to_csv(SAVE_DIR / 'posterior_samples.csv', index=False)
print(f"  Posterior samples: {SAVE_DIR}/posterior_samples.csv")

post_summary = posterior_df.describe(percentiles=[0.025, 0.25, 0.5, 0.75, 0.975]).T
post_summary.to_csv(SAVE_DIR / 'posterior_summary.csv')

diagnostics_dir = SAVE_DIR / 'diagnostics'
written = save_diagnostics(
    diagnostics_dir,
    r2_values=r2_values,
    z_scores=z_scores,
    contractions=contractions,
    param_names=param_names,
    loo_df=loo_prior_df,
    ppc_df=coverage_df,
    training_metadata={
        'n_rounds': n_rounds_completed,
        'n_sims_round1': N_SIMS_ROUND1,
        'n_sims_subsequent': N_SIMS_SUBSEQUENT,
        'n_test': int(theta_test_t.shape[0]),
        'n_params': num_params,
        'n_observables': num_observables,
        'n_posterior_sims': N_POSTERIOR_SIMS,
        'model_version': MODEL_VERSION,
        'scenario': SCENARIO,
        'npe_hidden_features': NPE_HIDDEN_FEATURES,
        'npe_num_transforms': NPE_NUM_TRANSFORMS,
        'npe_num_bins': NPE_NUM_BINS,
        'final_r2_median': r2_median,
        'final_r2_mean': r2_mean,
        'final_n_r2_above_05': n_r2_above_05,
        'final_wasserstein_max': wasserstein_max,
        'converged': (not np.isnan(wasserstein_max)
                      and wasserstein_max < CONVERGENCE_WASSERSTEIN_THRESHOLD),
    },
)

loo_post_df.to_csv(diagnostics_dir / 'loo_posterior_predictive.csv', index=False)
coverage_df.to_csv(diagnostics_dir / 'posterior_predictive_coverage.csv', index=False)

print(f"  Diagnostics: {diagnostics_dir}/")
for key, path in written.items():
    print(f"    {key}")

print("\n" + "=" * 80)
print(f"SEQUENTIAL NPE WORKFLOW COMPLETE ({n_rounds_completed} rounds)")
print("=" * 80)
