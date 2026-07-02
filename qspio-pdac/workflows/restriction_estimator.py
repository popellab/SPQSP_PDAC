#!/usr/bin/env python3
"""
Prior Restriction Estimator for PDAC QSP Model

Trains a classifier to distinguish parameter combinations that produce valid
simulations (tumor grows to diagnosis threshold) from those that don't
(evolve_to_diagnosis returns NaN). The resulting RestrictedPrior eliminates
wasted simulations in both Morris SA and SBI workflows.

Uses sbi.utils.RestrictionEstimator (Deistler et al. 2022) which trains a
binary classifier on (theta, valid/invalid) labels and produces a RestrictedPrior
that uses rejection sampling to only propose valid parameter sets.

Usage:
    python workflows/restriction_estimator.py

Output:
    cache/restriction_estimator/
        restriction_estimator.pkl  - Full RestrictionEstimator (for iterative refinement)
        restricted_prior.pkl       - RestrictedPrior object (torch Distribution)
        metadata.json              - Config, NaN rates, training stats
        diagnostics.png            - 2D projections of valid vs invalid samples

"""

import hashlib
import json
import pickle
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from sbi.utils import RestrictionEstimator

from qsp_hpc import QSPSimulator
from qsp_inference.priors import load_prior

# ============================================================================
# CONFIGURATION
# ============================================================================

PRIORS_CSV = 'parameters/pdac_priors.csv'
CALIBRATION_TARGETS_DIR = 'calibration_targets/baseline_no_treatment'
MODEL_STRUCTURE_FILE = 'model_structure.json'
SCENARIO = 'baseline_no_treatment'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'
MODEL_VERSION = 'restriction_estimator_v4'
MODEL_DESCRIPTION = 'Restriction estimator: learn valid/invalid boundary in prior space'

N_INITIAL_SIMS = 5000        # Initial simulation round (~28 samples/dim for 177 params)
N_RESTRICTION_ROUNDS = 3     # Iterative refinement rounds (sample from restricted, retrain)
N_SIMS_PER_ROUND = 1000      # Simulations per refinement round
ALLOWED_FALSE_NEGATIVES = 0.05  # Allow 5% false negatives for meaningful rejection
MAX_TASKS = 40               # Parallel SLURM array tasks
SEED = 2025

SAVE_DIR = Path('cache/restriction_estimator')
FIG_DIR = Path('figures/restriction_estimator')

# ============================================================================
# 1. Load priors and set up simulator
# ============================================================================
print("=" * 80)
print("PRIOR RESTRICTION ESTIMATOR")
print("=" * 80)

print(f"\n1. Loading priors from {PRIORS_CSV}...")
prior = load_prior(PRIORS_CSV)
priors_df = pd.read_csv(PRIORS_CSV)
param_names = priors_df['name'].tolist()
n_params = len(param_names)
print(f"   {n_params} parameters loaded")
print(f"   Prior type: {type(prior).__name__}")

print(f"\n   Setting up QSPSimulator for scenario: {SCENARIO}")
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
)

# ============================================================================
# 2. Initial simulation round
# ============================================================================
print(f"\n2. Running initial simulation round ({N_INITIAL_SIMS} simulations)...")
t0 = time.time()
theta_np, x_np = sim(N_INITIAL_SIMS)
t_sim = time.time() - t0
print(f"   Completed in {t_sim:.1f}s")
print(f"   theta shape: {theta_np.shape}")
print(f"   x shape: {x_np.shape}")

# Replace inf with NaN (matches surrogate_sensitivity.py)
n_inf = np.isinf(x_np).sum()
if n_inf > 0:
    print(f"   Replacing {n_inf} inf values with NaN")
    x_np[np.isinf(x_np)] = np.nan

# Report NaN rate
nan_mask = np.isnan(x_np).any(axis=1)
nan_rate = nan_mask.mean()
n_valid = (~nan_mask).sum()
n_invalid = nan_mask.sum()
print(f"\n   Initial NaN rate: {nan_rate:.1%} ({n_invalid} invalid, {n_valid} valid)")

# --- Sanity check: verify simulations are biologically plausible ---
target_names = sim._test_stats_df['test_statistic_id'].tolist()
SANITY_CHECKS = {
    'tumor_volume_resection': (0.5, 500.0, 'cm³'),
    'cancer_cell_density_baseline': (10, 50000, 'cell/mm²'),
}
for check_name, (lo, hi, unit_label) in SANITY_CHECKS.items():
    if check_name in target_names:
        j = target_names.index(check_name)
        vals = x_np[:, j]
        med = np.nanmedian(vals)
        if not (lo <= med <= hi):
            print(f"\n   SANITY CHECK FAILED: {check_name}")
            print(f"   Simulated median = {med:.4g} {unit_label} "
                  f"(expected {lo}–{hi})")
            print(f"   This may indicate evolve_to_diagnosis did not reach "
                  f"the target tumor diameter.")
            print(f"   Check that initial_tumour_diameter is accessible "
                  f"to evolve_to_diagnosis.")
            raise SystemExit(1)
        else:
            print(f"   Sanity check OK: {check_name} median = {med:.4g} {unit_label}")

# Convert to torch tensors
theta_tensor = torch.tensor(theta_np, dtype=torch.float32)
x_tensor = torch.tensor(x_np, dtype=torch.float32)

# ============================================================================
# 3. Train RestrictionEstimator
# ============================================================================
print(f"\n3. Training RestrictionEstimator...")
print(f"   decision_criterion='nan' (labels rows with any NaN/Inf as invalid)")

restriction_estimator = RestrictionEstimator(
    prior=prior,
    hidden_features=512,   # Default 100 too small for 177 input dims
    num_blocks=4,          # Default 2; more capacity for complex boundary
)
restriction_estimator.append_simulations(theta_tensor, x_tensor)

t0 = time.time()
classifier = restriction_estimator.train(
    stop_after_epochs=50,           # Default 20; let it train longer
    loss_importance_weights=True,   # Reweigh loss to balance valid/invalid
)
t_train = time.time() - t0
print(f"   Training completed in {t_train:.1f}s")

# Get labels to report accuracy
theta_all, _, labels = restriction_estimator.get_simulations()
print(f"   Labels: {labels.sum().item()} valid, {(~labels.bool()).sum().item()} invalid")

# Build restricted prior
restricted_prior = restriction_estimator.restrict_prior(
    allowed_false_negatives=ALLOWED_FALSE_NEGATIVES,
)
print(f"   RestrictedPrior type: {type(restricted_prior).__name__}")

# --- Classifier diagnostics ---
arfn = restricted_prior._accept_reject_fn
classifier_nn = arfn._classifier
classifier_nn.eval()
print(f"   Classifier threshold: {arfn._classifier_thr:.4f}")

valid_val = arfn._validation_theta[arfn._validation_label.bool()]
invalid_val = arfn._validation_theta[~arfn._validation_label.bool()]
if len(valid_val) > 0:
    probs_valid = F.softmax(classifier_nn(valid_val), dim=1)[:, 1]
    print(f"   Valid:   P(valid) min={probs_valid.min():.3f} med={probs_valid.median():.3f} max={probs_valid.max():.3f}")
if len(invalid_val) > 0:
    probs_invalid = F.softmax(classifier_nn(invalid_val), dim=1)[:, 1]
    print(f"   Invalid: P(valid) min={probs_invalid.min():.3f} med={probs_invalid.median():.3f} max={probs_invalid.max():.3f}")

# Classification accuracy at 0.5 threshold
with torch.no_grad():
    all_probs = F.softmax(classifier_nn(theta_all), dim=1)[:, 1]
preds_05 = (all_probs > 0.5).long()
accuracy = (preds_05 == labels).float().mean()
print(f"   ResNet accuracy at 0.5 threshold: {accuracy:.1%}")

# Sweep allowed_false_negatives to find useful operating point
print(f"\n   Sweeping allowed_false_negatives:")
for afn in [0.05, 0.10, 0.15, 0.20, 0.30]:
    rp = restriction_estimator.restrict_prior(allowed_false_negatives=afn)
    rp.sample((200,), save_acceptance_rate=True, print_rejected_frac=False)
    ar = rp.acceptance_rate.item()
    print(f"     afn={afn:.2f} -> acceptance={ar:.3f}, rejection={1-ar:.1%}")

# Test sampling from restricted prior
print(f"\n   Testing restricted prior sampling (100 samples)...")
test_samples = restricted_prior.sample((100,), print_rejected_frac=True)
print(f"   Sample shape: {test_samples.shape}")

# ---- Deep diagnostics: understand why classification is hard ----
labels_np = labels.numpy()
theta_np_all = theta_all.numpy()

# Permutation importance: which params does the ResNet use?
print(f"\n   ResNet permutation importance (top 20):")
with torch.no_grad():
    base_preds = (F.softmax(classifier_nn(theta_all), dim=1)[:, 1] > 0.5).long()
    base_acc = (base_preds == labels).float().mean().item()
importances = {}
for j, name in enumerate(param_names):
    theta_perm = theta_all.clone()
    theta_perm[:, j] = theta_perm[torch.randperm(len(theta_perm)), j]
    with torch.no_grad():
        perm_preds = (F.softmax(classifier_nn(theta_perm), dim=1)[:, 1] > 0.5).long()
        perm_acc = (perm_preds == labels).float().mean().item()
    importances[name] = base_acc - perm_acc
sorted_imp = sorted(importances.items(), key=lambda x: -x[1])
for name, imp in sorted_imp[:20]:
    print(f"     {name:<40s} {imp:+.4f}")

# Univariate AUC: can any single param predict validity?
print(f"\n   Univariate AUC (top 20):")
aucs = {}
for j, name in enumerate(param_names):
    try:
        aucs[name] = roc_auc_score(labels_np, theta_np_all[:, j])
    except Exception:
        aucs[name] = 0.5
sorted_auc = sorted(aucs.items(), key=lambda x: -abs(x[1] - 0.5))
for name, auc in sorted_auc[:20]:
    direction = 'higher=valid' if auc > 0.5 else 'lower=valid'
    print(f"     {name:<40s} AUC={auc:.3f} ({direction})")

# Growth vs death ratio
if 'k_C1_growth' in param_names and 'k_C1_death' in param_names:
    ig = param_names.index('k_C1_growth')
    id_ = param_names.index('k_C1_death')
    ratio = theta_np_all[:, ig] / (theta_np_all[:, id_] + 1e-30)
    ratio_auc = roc_auc_score(labels_np, ratio)
    valid_ratio = ratio[labels_np == 1]
    invalid_ratio = ratio[labels_np == 0]
    print(f"\n   k_C1_growth / k_C1_death ratio:")
    print(f"     AUC={ratio_auc:.3f}")
    print(f"     Valid median:   {np.median(valid_ratio):.2f}")
    print(f"     Invalid median: {np.median(invalid_ratio):.2f}")

# Random forest baseline: is the problem learnable with a better model?
print(f"\n   Random forest 5-fold CV accuracy (log10 space):")
rf = RandomForestClassifier(n_estimators=200, max_features='sqrt', random_state=42)
scores = cross_val_score(rf, np.log10(theta_np_all), labels_np, cv=5, scoring='accuracy')
print(f"     {scores.mean():.3f} +/- {scores.std():.3f}")

rf.fit(np.log10(theta_np_all), labels_np)
rf_imp = sorted(zip(param_names, rf.feature_importances_), key=lambda x: -x[1])
print(f"\n   RF feature importance (top 20):")
for name, imp in rf_imp[:20]:
    print(f"     {name:<40s} {imp:.4f}")

# ============================================================================
# 4. Iterative refinement rounds
# ============================================================================
if N_RESTRICTION_ROUNDS > 0:
    print(f"\n4. Iterative refinement ({N_RESTRICTION_ROUNDS} rounds, "
          f"{N_SIMS_PER_ROUND} sims/round)...")

    for round_idx in range(N_RESTRICTION_ROUNDS):
        print(f"\n   --- Round {round_idx + 1}/{N_RESTRICTION_ROUNDS} ---")

        # Sample from current restricted prior
        print(f"   Sampling {N_SIMS_PER_ROUND} from restricted prior...")
        new_theta = restricted_prior.sample(
            (N_SIMS_PER_ROUND,), print_rejected_frac=False
        )

        # Simulate with these parameters
        print(f"   Simulating...")
        new_theta_np = new_theta.numpy()
        new_x_np = sim.simulate_with_parameters(new_theta_np, pool_suffix='restriction_round')
        new_x_tensor = torch.tensor(new_x_np, dtype=torch.float32)

        # Report NaN rate for this round
        round_nan_mask = np.isnan(new_x_np).any(axis=1)
        round_nan_rate = round_nan_mask.mean()
        print(f"   Round NaN rate: {round_nan_rate:.1%} "
              f"({round_nan_mask.sum()} invalid, {(~round_nan_mask).sum()} valid)")

        # Append and retrain
        restriction_estimator.append_simulations(new_theta, new_x_tensor)
        classifier = restriction_estimator.train(
            stop_after_epochs=50,
            loss_importance_weights=True,
        )
        restricted_prior = restriction_estimator.restrict_prior(
            allowed_false_negatives=ALLOWED_FALSE_NEGATIVES,
        )

        # Accumulate for diagnostics
        theta_np = np.vstack([theta_np, new_theta_np])
        x_np = np.vstack([x_np, new_x_np])

    print(f"\n   Refinement complete. Total simulations: {theta_np.shape[0]}")
else:
    print(f"\n4. Skipping iterative refinement (N_RESTRICTION_ROUNDS=0)")

# ============================================================================
# 5. Diagnostic reporting
# ============================================================================
print(f"\n5. Diagnostics...")

# Final NaN rate
nan_mask_all = np.isnan(x_np).any(axis=1)
final_nan_rate = nan_mask_all.mean()
n_total = len(nan_mask_all)
print(f"   Total simulations: {n_total}")
print(f"   Overall NaN rate (unrestricted prior): {final_nan_rate:.1%}")

# Estimate restricted prior acceptance rate
print(f"   Estimating restricted prior acceptance rate...")
check_samples = restricted_prior.sample(
    (500,), save_acceptance_rate=True, print_rejected_frac=True
)
acceptance_rate = restricted_prior.acceptance_rate
if acceptance_rate is not None:
    print(f"   Acceptance rate: {acceptance_rate.item():.3f}")
else:
    acceptance_rate = torch.tensor(float('nan'))

# Simulate a small batch from restricted prior to estimate effective NaN rate
print(f"   Estimating effective NaN rate from restricted prior...")
check_theta_np = check_samples[:100].numpy()
check_x_np = sim.simulate_with_parameters(check_theta_np, pool_suffix='restriction_check')
effective_nan_rate = np.isnan(check_x_np).any(axis=1).mean()
print(f"   Effective NaN rate (restricted prior): {effective_nan_rate:.1%}")

# ============================================================================
# 6. Diagnostic figure: 2D projections of valid vs invalid
# ============================================================================
print(f"\n6. Generating diagnostic figures in {FIG_DIR}/...")
FIG_DIR.mkdir(parents=True, exist_ok=True)

valid_mask = ~nan_mask_all
theta_valid = theta_np[valid_mask]
theta_invalid = theta_np[~valid_mask]

# Pick top pairs: use first 6 parameters (these are typically most impactful)
n_show = min(6, n_params)
fig, axes = plt.subplots(n_show - 1, n_show - 1, figsize=(16, 16))

for i in range(n_show - 1):
    for j in range(n_show - 1):
        ax = axes[i, j] if n_show > 2 else axes
        if j <= i:
            # Plot valid (blue) and invalid (red) in log10-space
            if len(theta_invalid) > 0:
                ax.scatter(
                    np.log10(theta_invalid[:, j]),
                    np.log10(theta_invalid[:, i + 1]),
                    s=3, alpha=0.3, c='red', label='Invalid' if (i == 0 and j == 0) else None,
                    rasterized=True,
                )
            ax.scatter(
                np.log10(theta_valid[:, j]),
                np.log10(theta_valid[:, i + 1]),
                s=3, alpha=0.3, c='steelblue', label='Valid' if (i == 0 and j == 0) else None,
                rasterized=True,
            )
            if i == n_show - 2:
                ax.set_xlabel(param_names[j], fontsize=7, rotation=45, ha='right')
            if j == 0:
                ax.set_ylabel(param_names[i + 1], fontsize=7)
            ax.tick_params(labelsize=5)
        else:
            ax.set_visible(False)

# Add legend to first visible axis
if n_show > 2:
    axes[0, 0].legend(fontsize=8, markerscale=3)
fig.suptitle(
    f'Valid vs Invalid Samples (log10 space)\n'
    f'NaN rate: {final_nan_rate:.1%} unrestricted, {effective_nan_rate:.1%} restricted',
    fontsize=13,
)
plt.tight_layout()
fig.savefig(FIG_DIR / 'diagnostics.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"   Saved: {FIG_DIR / 'diagnostics.png'}")

# ============================================================================
# 7. Save artifacts
# ============================================================================
print(f"\n7. Saving artifacts to {SAVE_DIR}/...")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Compute config hash for staleness detection
priors_hash = hashlib.md5(Path(PRIORS_CSV).read_bytes()).hexdigest()[:12]

metadata = {
    'priors_csv': PRIORS_CSV,
    'priors_hash': priors_hash,
    'scenario': SCENARIO,
    'model_version': MODEL_VERSION,
    'n_initial_sims': N_INITIAL_SIMS,
    'n_restriction_rounds': N_RESTRICTION_ROUNDS,
    'n_sims_per_round': N_SIMS_PER_ROUND,
    'n_total_sims': int(n_total),
    'seed': SEED,
    'nan_rate_unrestricted': float(final_nan_rate),
    'nan_rate_restricted': float(effective_nan_rate),
    'acceptance_rate': float(acceptance_rate.item()),
    'n_params': n_params,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
}

# Save metadata
with open(SAVE_DIR / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   Saved: metadata.json")

# Save RestrictionEstimator (for iterative refinement / retraining)
with open(SAVE_DIR / 'restriction_estimator.pkl', 'wb') as f:
    pickle.dump(restriction_estimator, f)
print(f"   Saved: restriction_estimator.pkl")

# Save RestrictedPrior (the Distribution object for downstream use)
with open(SAVE_DIR / 'restricted_prior.pkl', 'wb') as f:
    pickle.dump(restricted_prior, f)
print(f"   Saved: restricted_prior.pkl")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("RESTRICTION ESTIMATOR COMPLETE")
print("=" * 80)
print(f"\n  Parameters:              {n_params}")
print(f"  Total simulations:       {n_total}")
print(f"  NaN rate (unrestricted): {final_nan_rate:.1%}")
print(f"  NaN rate (restricted):   {effective_nan_rate:.1%}")
print(f"  Acceptance rate:         {acceptance_rate.item():.3f}")
print(f"  Artifacts:               {SAVE_DIR}/")
print(f"  Figures:                 {FIG_DIR}/")
print(f"\n  Usage in Morris SA:")
print(f"    Set RESTRICTION_ESTIMATOR_DIR = '{SAVE_DIR}'")
print(f"    in workflows/morris_sensitivity.py")
print(f"\n  Usage in SBI:")
print(f"    restricted_prior = pickle.load(open('{SAVE_DIR}/restricted_prior.pkl', 'rb'))")
print(f"    # Use as drop-in replacement for load_prior() — it's a Distribution subclass")
