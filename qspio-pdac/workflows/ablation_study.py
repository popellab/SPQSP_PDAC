#!/usr/bin/env python3
"""
Ablation Study: Noise vs Parameter Count

Disentangles what caused the diagnostic differences between the SA97 run
(97 params, 30 obs, no noise) and the allparams run (193 params, 35 obs, noise).

Uses the same simulation pool (allparams_v1) throughout, varying only:
  - Observation noise: on / off
  - Parameter subset: all 193 / SA97-overlap 96

All 4 configs share the same simulations and 35 observables,
so differences are purely from noise and param count.

Usage:
    python workflows/ablation_study.py
"""

import sys
import time
from pathlib import Path
import numpy as np
import torch
import pandas as pd

from sbi.inference import NPE

from qsp_hpc import QSPSimulator
from qsp_inference.inference import (
    get_observed_data,
    sbi_recovery,
    sbi_mmd_misspecification,
    sbi_prior_predictive_pvalues,
    sbi_posterior_predictive_check,
    sbi_boundary_piling,
    sbi_posterior_correlations,
    save_diagnostics,
)
from qsp_inference.inference.data_processing import _filter_nans, _split_data, _transform_data, add_observation_noise
from qsp_inference.inference.gaussian_copula_transform import transform_to_normal_from_array
from qsp_inference.priors import load_prior, transform_lognormal_prior_to_normal

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_VERSION = 'baseline_allparams_v1_fActiveTGFb'  # Bumped for f_active_TGFb + averaged polarization
MODEL_DESCRIPTION = 'PDAC baseline SBI: all params, calibration targets from YAMLs'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'
SCENARIO = 'baseline_no_treatment'

PRIORS_CSV = 'scenarios/pdac_priors_bounded.csv'  # matches existing pool hash
CALIBRATION_TARGETS_DIR = 'calibration_targets/baseline_no_treatment'
MODEL_STRUCTURE_FILE = 'model_structure.json'

NUM_TOTAL_SIMULATIONS = 2500
MAX_TASKS = 40
SEED = 2025
CACHE_SAMPLING_SEED = 2025
SPLIT_SEED = 2027

SAVE_DIR = Path('figures/ablation_study')
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# SA97 parameter names (from previous run's recovery file)
SA97_RECOVERY = Path('figures/sbi_baseline_sa97/diagnostics/recovery_r2.csv')

# ============================================================================
# 1. Load Simulations (shared across all ablations)
# ============================================================================
print("=" * 80)
print("ABLATION STUDY: NOISE vs PARAMETER COUNT")
print("=" * 80)

print("\n1. Loading simulations from pool...")
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

# Save summary CSV for observed data loading
Path('cache').mkdir(exist_ok=True)
summary_csv_path = 'cache/baseline_calibration_targets_summary.csv'
target_df.to_csv(summary_csv_path, index=False)

priors_df = pd.read_csv(PRIORS_CSV)
all_param_names = priors_df['name'].tolist()

# Replace inf with NaN
x_raw[np.isinf(x_raw)] = np.nan

# Filter NaN rows
theta_clean, x_clean = _filter_nans(theta_raw, x_raw)
print(f"  {theta_clean.shape[0]} valid sims from {theta_raw.shape[0]} raw")

# Split train/test (fixed seed — same split for all configs)
theta_train_np, theta_test_np, x_train_np, x_test_np = _split_data(
    theta_clean, x_clean, test_fraction=0.1, seed=SPLIT_SEED
)
x_train_raw = x_train_np.copy()

# Load observed data
obs_data = get_observed_data(summary_csv_path, value_column='median')
obs_values = np.array([np.squeeze(obs_data[name]) for name in observable_names])

# SA97 param subset
sa97_df = pd.read_csv(SA97_RECOVERY)
sa97_param_names = [n for n in sa97_df['parameter'].tolist() if n in all_param_names]
sa97_param_indices = [all_param_names.index(n) for n in sa97_param_names]
print(f"  SA97 overlap: {len(sa97_param_indices)} params")

# ============================================================================
# 2. Define Ablation Configs
# ============================================================================

configs = {
    'allparams_noise': {
        'label': '193 params + noise',
        'param_indices': list(range(len(all_param_names))),
        'param_names': all_param_names,
        'noise': True,
    },
    'allparams_no_noise': {
        'label': '193 params, no noise',
        'param_indices': list(range(len(all_param_names))),
        'param_names': all_param_names,
        'noise': False,
    },
    'sa96_noise': {
        'label': '96 params + noise',
        'param_indices': sa97_param_indices,
        'param_names': sa97_param_names,
        'noise': True,
    },
    'sa96_no_noise': {
        'label': '96 params, no noise',
        'param_indices': sa97_param_indices,
        'param_names': sa97_param_names,
        'noise': False,
    },
}

# ============================================================================
# 3. Run Ablations
# ============================================================================

results = []

for config_name, cfg in configs.items():
    print(f"\n{'=' * 80}")
    print(f"  ABLATION: {cfg['label']}")
    print(f"{'=' * 80}")
    t0 = time.time()

    pidx = cfg['param_indices']
    pnames = cfg['param_names']
    n_params = len(pnames)

    # Subset theta to selected params
    theta_train_sub = theta_train_np[:, pidx]
    theta_test_sub = theta_test_np[:, pidx]

    # Optionally add noise
    if cfg['noise']:
        x_train_noisy = add_observation_noise(
            x_train_raw.copy(),
            ci95_lower=target_df['ci95_lower'].values,
            ci95_upper=target_df['ci95_upper'].values,
            medians=target_df['median'].values,
            seed=SEED + 1,
        )
    else:
        x_train_noisy = x_train_raw.copy()

    # Transform
    theta_train, x_train, theta_test, x_test, obs_quantiles = _transform_data(
        theta_train_sub, x_train_noisy, theta_test_sub, x_test_np
    )

    # Build prior for this param subset
    from sbi.utils import BoxUniform
    if torch.is_tensor(theta_train):
        t = theta_train
    else:
        t = torch.tensor(theta_train, dtype=torch.float32)
    low = t.min(dim=0).values - 0.5 * t.std(dim=0)
    high = t.max(dim=0).values + 0.5 * t.std(dim=0)
    prior = BoxUniform(low=low, high=high)

    # Train NPE
    print(f"  Training NPE ({n_params} params, {len(observable_names)} obs)...")
    torch.manual_seed(SEED)
    inference = NPE(prior=prior)
    inference = inference.append_simulations(theta_train, x_train)
    estimator = inference.train()
    posterior = inference.build_posterior()

    # Sample posteriors for test observations
    with torch.no_grad():
        samples = posterior.posterior_estimator.sample(
            torch.Size([200]), condition=x_test
        )

    # R² recovery
    _, _, r2_values = sbi_recovery(samples, theta_test, pnames, plot=False)
    r2_array = np.array([r2_values[n] for n in pnames])

    # MMD misspecification
    obs_transformed = transform_to_normal_from_array(obs_values[np.newaxis, :], obs_quantiles)[0]
    obs_tensor = torch.tensor(obs_transformed, dtype=torch.float32).unsqueeze(0)
    mmd_results = sbi_mmd_misspecification(obs_tensor, x_train, inference)

    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt

    # Prior predictive p-values
    prior_pval_df, fig_pp = sbi_prior_predictive_pvalues(
        x_train_raw, obs_data, observable_names
    )
    plt.close(fig_pp)

    # Posterior predictive check
    with torch.no_grad():
        post_samples_obs = posterior.posterior_estimator.sample(
            torch.Size([2000]), condition=obs_tensor
        ).squeeze(1)

    from qsp_inference.inference import convert_posterior_samples_to_original_space
    post_dict = convert_posterior_samples_to_original_space(post_samples_obs, pnames)

    ppc_df, fig_ppc = sbi_posterior_predictive_check(
        theta_train, x_train_raw, post_samples_obs, obs_data,
        observable_names, k=10,
    )
    plt.close(fig_ppc)

    # Posterior correlations
    _, corr_strong, fig_corr = sbi_posterior_correlations(
        post_dict, pnames, threshold=0.5,
    )
    plt.close(fig_corr)

    # Boundary piling
    priors_sub = priors_df[priors_df['name'].isin(pnames)].copy()
    piling_df = sbi_boundary_piling(post_dict, priors_sub)

    elapsed = time.time() - t0

    # Collect summary
    n_prior_fail = int((prior_pval_df['p_value'] < 0.05).sum())
    n_ppc_in = int(ppc_df['obs_in_90ci'].sum())
    n_piling = int(piling_df['flagged'].sum())

    row = {
        'config': config_name,
        'label': cfg['label'],
        'n_params': n_params,
        'noise': cfg['noise'],
        'r2_median': float(np.median(r2_array)),
        'r2_mean': float(np.mean(r2_array)),
        'n_r2_above_0.5': int((r2_array > 0.5).sum()),
        'n_r2_above_0.1': int((r2_array > 0.1).sum()),
        'mmd_p_value': float(mmd_results['p_value']),
        'n_prior_pval_below_0.05': n_prior_fail,
        'n_ppc_in_90ci': n_ppc_in,
        'n_ppc_total': len(ppc_df),
        'n_strong_correlations': len(corr_strong),
        'n_boundary_piling': n_piling,
        'elapsed_s': elapsed,
    }
    results.append(row)

    print(f"  R² median={row['r2_median']:.4f}, R²>0.5={row['n_r2_above_0.5']}, "
          f"MMD p={row['mmd_p_value']:.3f}, prior_fail={n_prior_fail}, "
          f"PPC={n_ppc_in}/{len(ppc_df)}, corr={len(corr_strong)}, "
          f"piling={n_piling} ({elapsed:.0f}s)")

    # Save per-config R² for detailed comparison
    r2_df = pd.DataFrame({'parameter': pnames, 'r2': r2_array})
    r2_df = r2_df.sort_values('r2', ascending=False)
    r2_df.to_csv(SAVE_DIR / f'recovery_r2_{config_name}.csv', index=False)

# ============================================================================
# 4. Summary Table
# ============================================================================
print("\n" + "=" * 80)
print("ABLATION RESULTS SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results)
results_df.to_csv(SAVE_DIR / 'ablation_summary.csv', index=False)

# Print formatted table
print(f"\n{'Config':<25} {'Params':>6} {'Noise':>5} {'R² med':>7} {'R²>.5':>5} "
      f"{'R²>.1':>5} {'MMD p':>7} {'PrFail':>6} {'PPC':>7} {'Corr':>4} {'Pile':>4}")
print("-" * 95)
for _, r in results_df.iterrows():
    print(f"{r['label']:<25} {r['n_params']:>6} {'yes' if r['noise'] else 'no':>5} "
          f"{r['r2_median']:>7.4f} {r['n_r2_above_0.5']:>5} {r['n_r2_above_0.1']:>5} "
          f"{r['mmd_p_value']:>7.3f} {r['n_prior_pval_below_0.05']:>6} "
          f"{r['n_ppc_in_90ci']:>3}/{r['n_ppc_total']:<3} {r['n_strong_correlations']:>4} "
          f"{r['n_boundary_piling']:>4}")

# Effect decomposition
print("\n" + "=" * 80)
print("EFFECT DECOMPOSITION")
print("=" * 80)

def get_row(name):
    return results_df[results_df['config'] == name].iloc[0]

metrics = ['r2_median', 'n_r2_above_0.5', 'mmd_p_value', 'n_prior_pval_below_0.05',
           'n_ppc_in_90ci', 'n_strong_correlations', 'n_boundary_piling']

a = get_row('allparams_noise')
b = get_row('allparams_no_noise')
c = get_row('sa96_noise')
d = get_row('sa96_no_noise')

print(f"\nNoise effect (averaged over param counts):")
for m in metrics:
    noise_effect = ((a[m] - b[m]) + (c[m] - d[m])) / 2
    print(f"  {m}: {noise_effect:+.4f}")

print(f"\nParam count effect (averaged over noise settings):")
for m in metrics:
    param_effect = ((a[m] - c[m]) + (b[m] - d[m])) / 2
    print(f"  {m}: {param_effect:+.4f}")

print(f"\nInteraction (noise × param count):")
for m in metrics:
    interaction = (a[m] - b[m]) - (c[m] - d[m])
    print(f"  {m}: {interaction:+.4f}")

print(f"\n  Saved to {SAVE_DIR}/ablation_summary.csv")
print(f"  Per-config R² files in {SAVE_DIR}/")
print("\n" + "=" * 80)
print("ABLATION STUDY COMPLETE")
print("=" * 80)
