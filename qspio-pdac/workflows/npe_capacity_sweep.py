#!/usr/bin/env python3
"""
NPE Capacity Sweep — NSF architecture search for PRCC tier 1 parameters.

Sweeps hidden_features and num_transforms while tracking training/validation
learning curves to diagnose underfitting vs overfitting.

Uses the same simulation pool as sbi_workflow.py (no new MATLAB runs).

Usage:
    python workflows/npe_capacity_sweep.py
"""

import time
from pathlib import Path
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sbi.inference import NPE
from sbi.neural_nets import posterior_nn
from sbi.utils import MultipleIndependent

from qsp_hpc import QSPSimulator
from qsp_inference.inference import (
    get_observed_data,
    sbi_recovery,
    sbi_z_score_contraction,
)
from qsp_inference.inference.data_processing import _filter_nans, _split_data, _transform_data, add_observation_noise
from qsp_inference.inference.gaussian_copula_transform import transform_to_normal_from_array
from qsp_inference.priors import load_prior, transform_lognormal_prior_to_normal

# ============================================================================
# CONFIG
# ============================================================================

MODEL_VERSION = 'baseline_allparams_v2'
MODEL_DESCRIPTION = 'PDAC baseline SBI: all params, calibration targets from YAMLs'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'
SCENARIO = 'baseline_no_treatment'

PRIORS_CSV = 'parameters/pdac_priors.csv'
CALIBRATION_TARGETS_DIR = 'calibration_targets/baseline_no_treatment'
MODEL_STRUCTURE_FILE = 'model_structure.json'

# PRCC tier 1 filtering
PRCC_RANKING_CSV = 'results/prcc_sensitivity/aggregate_parameter_ranking.csv'
PRCC_TIER_CUTOFF = 30

NUM_TOTAL_SIMULATIONS = 1e4
MAX_TASKS = 40
SEED = 2025
CACHE_SAMPLING_SEED = 2025
SPLIT_SEED = 2027

SAVE_DIR = Path('figures/npe_capacity_sweep')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Sweep grid
SWEEP_CONFIGS = [
    # (hidden_features, num_transforms, num_bins, label)
    (32,  3,  10, '32h_3t'),    # current baseline
    (64,  3,  10, '64h_3t'),
    (128, 3,  10, '128h_3t'),
    (64,  5,  10, '64h_5t'),
    (128, 5,  10, '128h_5t'),
    (64,  8,  10, '64h_8t'),
    (128, 8,  10, '128h_8t'),
]

NPE_TRAINING_BATCH_SIZE = 128
NPE_MAX_EPOCHS = 500

# ============================================================================
# 1. LOAD & PROCESS SIMULATIONS (shared across sweep)
# ============================================================================
print("=" * 80)
print("NPE CAPACITY SWEEP — PRCC TIER 1")
print("=" * 80)

print("\n1. Loading simulations...")
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
    verbose=False,
    seed=SEED,
    cache_sampling_seed=CACHE_SAMPLING_SEED,
)

theta_raw, x_raw = sim(NUM_TOTAL_SIMULATIONS)

target_df = sim._test_stats_df.copy()
observable_names = target_df['test_statistic_id'].tolist()
num_observables = len(observable_names)

Path('cache').mkdir(exist_ok=True)
summary_csv_path = 'cache/baseline_calibration_targets_summary.csv'
target_df.to_csv(summary_csv_path, index=False)

# Full param names
priors_df = pd.read_csv(PRIORS_CSV)
all_param_names = priors_df['name'].tolist()

# PRCC tier 1 subset
ranking_df = pd.read_csv(PRCC_RANKING_CSV)
selected_params = set(ranking_df[ranking_df['rank_prcc'] <= PRCC_TIER_CUTOFF]['parameter'].tolist())
param_indices = [i for i, n in enumerate(all_param_names) if n in selected_params]
param_names = [all_param_names[i] for i in param_indices]
num_params = len(param_names)
theta_raw = theta_raw[:, param_indices]
print(f"  PRCC tier 1: {num_params} parameters")
print(f"  Observables: {num_observables}")

# Replace inf, filter NaN
x_raw[np.isinf(x_raw)] = np.nan
theta_clean, x_clean = _filter_nans(theta_raw, x_raw)
n_survived = len(theta_clean)
print(f"  Valid sims: {n_survived}/{len(theta_raw)}")

# Split
theta_train_np, theta_test_np, x_train_np, x_test_np = _split_data(
    theta_clean, x_clean, test_fraction=0.1, seed=SPLIT_SEED
)
x_train_raw = x_train_np.copy()

# Add observation noise
x_train_np = add_observation_noise(
    x_train_np,
    ci95_lower=target_df['ci95_lower'].values,
    ci95_upper=target_df['ci95_upper'].values,
    medians=target_df['median'].values,
    seed=SEED + 1,
    sample_sizes=target_df['sample_size'].values,
)

# Transform
theta_train, x_train, theta_test, x_test, obs_quantiles = _transform_data(
    theta_train_np, x_train_np, theta_test_np, x_test_np
)
print(f"  Train: {theta_train.shape[0]}, Test: {theta_test.shape[0]}")

# Build prior (subset)
prior_lognormal_full = load_prior(PRIORS_CSV)
prior_lognormal = MultipleIndependent([prior_lognormal_full.dists[i] for i in param_indices])
prior_normal = transform_lognormal_prior_to_normal(prior_lognormal)
prior_var = torch.var(prior_normal.sample((2000,)), dim=0).numpy()

# ============================================================================
# 2. SWEEP
# ============================================================================
print("\n" + "=" * 80)
print("2. CAPACITY SWEEP")
print("=" * 80)

results = []

for hidden, transforms, bins, label in SWEEP_CONFIGS:
    print(f"\n{'─' * 60}")
    print(f"  Config: {label} (hidden={hidden}, transforms={transforms}, bins={bins})")
    print(f"{'─' * 60}")
    t0 = time.time()

    torch.manual_seed(SEED)

    density_estimator = posterior_nn(
        model='nsf',
        hidden_features=hidden,
        num_transforms=transforms,
        num_bins=bins,
    )

    inference = NPE(prior=prior_normal, density_estimator=density_estimator)
    inference = inference.append_simulations(theta_train, x_train)
    estimator = inference.train(
        training_batch_size=NPE_TRAINING_BATCH_SIZE,
        max_num_epochs=NPE_MAX_EPOCHS,
        show_train_summary=True,
    )
    posterior = inference.build_posterior()

    # Extract learning curves
    train_losses = inference._summary['training_loss']
    val_losses = inference._summary['validation_loss']
    n_epochs = len(train_losses)
    best_val = inference._summary['best_validation_loss'][-1]

    print(f"  Epochs: {n_epochs}, best val loss: {best_val:.4f}")

    # R² recovery
    with torch.no_grad():
        samples = posterior.posterior_estimator.sample(
            torch.Size([200]), condition=x_test
        )

    _, _, r2_values = sbi_recovery(samples, theta_test, param_names, plot=False)
    r2_array = np.array([r2_values[name] for name in param_names])
    r2_ranking = np.argsort(r2_array)[::-1]

    # Z-score contraction
    _, _, z_scores, contractions = sbi_z_score_contraction(
        samples, theta_test, prior_normal, param_names,
        param_indices=r2_ranking[:20].tolist(), prior_var=prior_var
    )
    median_contraction = float(np.median(contractions))

    # Count trainable parameters
    n_nn_params = sum(p.numel() for p in estimator.parameters())

    elapsed = time.time() - t0

    row = {
        'config': label,
        'hidden_features': hidden,
        'num_transforms': transforms,
        'num_bins': bins,
        'n_nn_params': n_nn_params,
        'n_epochs': n_epochs,
        'best_val_loss': float(best_val),
        'final_train_loss': float(train_losses[-1]),
        'r2_median': float(np.median(r2_array)),
        'r2_mean': float(np.mean(r2_array)),
        'n_r2_above_0.5': int((r2_array > 0.5).sum()),
        'n_r2_above_0.3': int((r2_array > 0.3).sum()),
        'n_r2_above_0.1': int((r2_array > 0.1).sum()),
        'median_contraction': median_contraction,
        'elapsed_s': elapsed,
    }
    results.append(row)

    print(f"  NN params: {n_nn_params:,}")
    print(f"  R² median={row['r2_median']:.4f}, R²>0.5={row['n_r2_above_0.5']}, "
          f"R²>0.3={row['n_r2_above_0.3']}, R²>0.1={row['n_r2_above_0.1']}")
    print(f"  Contraction: {median_contraction:.3f}, elapsed: {elapsed:.0f}s")

    # Save per-config R²
    r2_df = pd.DataFrame({'parameter': param_names, 'r2': r2_array})
    r2_df = r2_df.sort_values('r2', ascending=False)
    r2_df.to_csv(SAVE_DIR / f'recovery_r2_{label}.csv', index=False)

    # Save learning curve
    lc_df = pd.DataFrame({
        'epoch': range(1, n_epochs + 1),
        'train_loss': train_losses,
        'val_loss': val_losses,
    })
    lc_df.to_csv(SAVE_DIR / f'learning_curve_{label}.csv', index=False)

# ============================================================================
# 3. SUMMARY & PLOTS
# ============================================================================
print("\n" + "=" * 80)
print("3. RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df.to_csv(SAVE_DIR / 'sweep_summary.csv', index=False)

print(f"\n{'Config':<12} {'NN params':>10} {'Epochs':>6} {'Val loss':>9} "
      f"{'R² med':>7} {'R²>.5':>5} {'R²>.3':>5} {'R²>.1':>5} {'Contr':>6} {'Time':>5}")
print("─" * 85)
for _, r in results_df.iterrows():
    print(f"{r['config']:<12} {r['n_nn_params']:>10,} {r['n_epochs']:>6} {r['best_val_loss']:>9.4f} "
          f"{r['r2_median']:>7.4f} {r['n_r2_above_0.5']:>5.0f} {r['n_r2_above_0.3']:>5.0f} "
          f"{r['n_r2_above_0.1']:>5.0f} {r['median_contraction']:>6.3f} {r['elapsed_s']:>5.0f}s")

# --- Learning curves plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for _, (hidden, transforms, bins, label) in enumerate(SWEEP_CONFIGS):
    lc = pd.read_csv(SAVE_DIR / f'learning_curve_{label}.csv')
    axes[0].plot(lc['epoch'], lc['train_loss'], label=label, alpha=0.8)
    axes[1].plot(lc['epoch'], lc['val_loss'], label=label, alpha=0.8)

axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Training loss')
axes[0].set_title('Training loss')
axes[0].legend(fontsize=8)

axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Validation loss')
axes[1].set_title('Validation loss')
axes[1].legend(fontsize=8)

fig.suptitle(f'NPE Capacity Sweep — {num_params} params, {num_observables} obs', fontsize=13)
fig.tight_layout()
fig.savefig(SAVE_DIR / 'learning_curves.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# --- R² vs capacity plot ---
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(results_df['n_nn_params'], results_df['r2_median'], 'o-', markersize=8, label='median R²')
ax.fill_between(
    results_df['n_nn_params'],
    results_df['r2_mean'] - 0.05,  # rough spread indicator
    results_df['r2_mean'] + 0.05,
    alpha=0.15,
)
for _, r in results_df.iterrows():
    ax.annotate(r['config'], (r['n_nn_params'], r['r2_median']),
                textcoords='offset points', xytext=(5, 5), fontsize=7)
ax.set_xlabel('NN parameters')
ax.set_ylabel('Median R²')
ax.set_title(f'R² recovery vs NSF capacity ({num_params} params)')
ax.set_xscale('log')
fig.tight_layout()
fig.savefig(SAVE_DIR / 'r2_vs_capacity.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# --- Train/val gap plot (overfitting diagnostic) ---
fig, ax = plt.subplots(figsize=(8, 5))
gap = results_df['final_train_loss'] - results_df['best_val_loss']
ax.bar(results_df['config'], gap)
ax.set_ylabel('Train loss − Best val loss')
ax.set_title('Overfitting gap (negative = overfitting, near zero = good fit)')
ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
plt.xticks(rotation=30, ha='right')
fig.tight_layout()
fig.savefig(SAVE_DIR / 'overfitting_gap.png', dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"\nPlots saved to {SAVE_DIR}/")
print("  learning_curves.png — train/val loss over epochs")
print("  r2_vs_capacity.png — R² recovery vs NN size")
print("  overfitting_gap.png — underfitting/overfitting diagnostic")

print("\n" + "=" * 80)
print("CAPACITY SWEEP COMPLETE")
print("=" * 80)
