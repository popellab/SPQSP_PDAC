#!/usr/bin/env python3
"""
Multi-Scenario SBI Workflow for PDAC QSP Model

This script implements simulation-based inference (SBI) for multi-scenario therapy evaluation.
The workflow performs joint parameter inference across multiple treatment protocols (control,
GVAX, GVAX+anti-PD1) by learning p(θ | x_control, x_gvax, x_combo).

Key features:
- Shared parameters across scenarios (same patient, different treatments)
- Joint inference on concatenated observables from all scenarios
- Per-scenario posterior predictive checks
- Full diagnostic suite (recovery, calibration, active subspace)

Usage:
    python workflows/multi_scenario_sbi_workflow.py
"""

from sbi.inference.posteriors.posterior_parameters import VIPosteriorParameters
import sys
import time
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sbi.inference import NPE
from sbi.inference import NRE

from qsp_hpc import QSPSimulator
from qsp_inference.data import plot_processed_data_normality
from qsp_inference.inference import (
    get_observed_data,
    generate_prior_predictive_checks,
    generate_posterior_predictive_checks,
    generate_posterior_predictive_simulations,
    plot_ppc_histograms,
    plot_posterior_predictive_spaghetti,
    plot_posterior_marginals,
    plot_posterior_pairs,
    plot_posterior_vs_prior_marginals,
    processed_multi_scenario_simulator,
    prepare_observed_data,
    convert_posterior_samples_to_original_space,
    sbi_recovery,
    sbi_z_score_contraction,
    sbi_calibration_ecdf
)
from qsp_inference.priors import load_prior, transform_lognormal_prior_to_normal

# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Model Configuration ---
MODEL_VERSION = 'pdac_multi_scenario_v4'  # Descriptive version name (bump when model changes)
MODEL_DESCRIPTION = 'PDAC multi-scenario: 119 params, scenario-specific observables (control, GVAX, GVAX+anti-PD1)'
PRIORS_CSV = 'parameters/pdac_priors.csv'

# Scenario-specific test statistics CSVs
# Each scenario can have different observables, time points, or metrics
# CSVs are now in scenarios/test_statistics/ for version control
TEST_STATS_CSVS = {
    'baseline_no_treatment': 'test_statistics_summary.csv',
    # 'gvax_nivo_neoadjuvant_zheng2022': 'scenarios/test_statistics/gvax_nivo_neoadjuvant_zheng2022.csv'
}

# Observables to exclude from training (e.g., zero values cause log transform issues)
EXCLUDE_OBSERVABLES = ['il2_concentration_baseline']

# Model structure file (species metadata with units for Pint)
MODEL_STRUCTURE_FILE = 'model_structure.json'

MODEL_SCRIPT = 'immune_oncology_model_PDAC'

# --- Scenario Configuration ---
# Scenarios derived from TEST_STATS_CSVS keys
SCENARIOS = list(TEST_STATS_CSVS.keys())
NUM_TOTAL_SIMULATIONS = 1e4  # Total simulations (same params across all scenarios)
MAX_TASKS = 40  # Maximum number of parallel SLURM array tasks
LOCAL_MATLAB_PATH = '/Applications/MATLAB_R2024b.app/bin/matlab'
# Seeds for reproducibility
SIMULATION_SEED = int(time.time()) % (2**31)  # Unique seed per run
CACHE_SAMPLING_SEED = 2025  # Fixed seed for reproducible cache sampling
SPLIT_SEED = 2027  # Fixed seed for reproducible train/val/test splits

# Training configuration
BATCH_SIZE = 32
EPOCHS = 500

# --- Plot Toggles ---
PLOT_NORMALITY_CHECKS = False  # Set to True to generate normality QQ plots
PLOT_PRIOR_PPC = False  # Set to True to generate prior predictive checks

print(f"🎲 Simulation seed: {SIMULATION_SEED} (unique per run)")
print(f"🎲 Cache sampling seed: {CACHE_SAMPLING_SEED} (fixed for cache sampling)")
print(f"🎲 Split seed: {SPLIT_SEED} (fixed for train/test splits)")
print(f"📊 Scenarios: {SCENARIOS}")

# Load parameter names
priors_df = pd.read_csv(PRIORS_CSV)
param_names = priors_df['name'].tolist()
num_params = len(param_names)

# Load observable names per scenario (each scenario can have different observables)
# Filter out excluded observables and save filtered CSVs for simulators
observable_names_per_scenario = {}
num_observables_per_scenario = {}
FILTERED_TEST_STATS_CSVS = {}
for scenario in SCENARIOS:
    test_stats_df = pd.read_csv(TEST_STATS_CSVS[scenario])
    # Filter out excluded observables
    if EXCLUDE_OBSERVABLES:
        test_stats_df = test_stats_df[~test_stats_df['test_statistic_id'].isin(EXCLUDE_OBSERVABLES)]
    observable_names_per_scenario[scenario] = test_stats_df['test_statistic_id'].tolist()
    num_observables_per_scenario[scenario] = len(observable_names_per_scenario[scenario])
    # Save filtered CSV for simulator to use
    filtered_csv_path = f'cache/{scenario}_test_stats_filtered.csv'
    test_stats_df.to_csv(filtered_csv_path, index=False)
    FILTERED_TEST_STATS_CSVS[scenario] = filtered_csv_path

if EXCLUDE_OBSERVABLES:
    print(f"⚠️  Excluded observables: {EXCLUDE_OBSERVABLES}")

# Calculate total observables (sum across all scenarios)
num_observables_total = sum(num_observables_per_scenario.values())

print(f"✓ Loaded {num_params} parameters")
print(f"✓ Loaded observables per scenario:")
for scenario in SCENARIOS:
    print(f"   - {scenario}: {num_observables_per_scenario[scenario]} observables")
print(f"✓ Total observables for joint inference: {num_observables_total}")

# ============================================================================
# 0. Create Scenario-Specific Simulators
# ============================================================================
print("\n" + "="*80)
print("CREATING SCENARIO-SPECIFIC SIMULATORS")
print("="*80)

# Create simulator for each scenario (with scenario-specific test stats)
simulators = {}
for scenario in SCENARIOS:
    print(f"\n📊 Creating simulator for scenario: {scenario}")
    print(f"   Test stats CSV: {FILTERED_TEST_STATS_CSVS[scenario]}")
    simulators[scenario] = QSPSimulator(
        test_stats_csv=FILTERED_TEST_STATS_CSVS[scenario],  # Filtered test stats (excludes IL-2, etc.)
        model_structure_file=MODEL_STRUCTURE_FILE,  # For Pint unit handling
        priors_csv=PRIORS_CSV,
        model_script=MODEL_SCRIPT,
        model_version=MODEL_VERSION,
        model_description=MODEL_DESCRIPTION,
        scenario=scenario,
        verbose=True,
        project_root=Path('.'),
        max_tasks=MAX_TASKS,
        seed=SIMULATION_SEED,
        cache_sampling_seed=CACHE_SAMPLING_SEED
    )
    print(f"   ✓ {scenario}: {simulators[scenario]}")

# ============================================================================
# 1. Generate and Process Multi-Scenario Data with Joint NaN Filtering
# ============================================================================
# This uses joint NaN filtering to ensure ALL scenarios have the same valid samples.
# This is critical for multi-scenario SBI where we need the same parameter sets
# evaluated under each scenario.

multi_data = processed_multi_scenario_simulator(
    simulators=simulators,
    scenarios=SCENARIOS,
    num_simulations=NUM_TOTAL_SIMULATIONS,
    test_fraction=0.001,
    split_seed=SPLIT_SEED
)

# Extract results from multi-scenario processor
theta_train = multi_data['theta_train']
theta_test = multi_data['theta_test']
scenario_data = multi_data['scenario_data']
x_train_multi = multi_data['x_train_multi']
x_test_multi = multi_data['x_test_multi']

# Store quantiles for each scenario (needed for PPCs later)
obs_quantiles_multi = {s: scenario_data[s]['obs_quantiles'] for s in SCENARIOS}

# ============================================================================
# 3. Check Normality of Processed Data (Optional)
# ============================================================================
if PLOT_NORMALITY_CHECKS:
    print("\n" + "="*80)
    print("CHECKING NORMALITY OF PROCESSED DATA")
    print("="*80)

    # Check normality for each scenario separately
    cumulative_obs_count = 0
    for scenario in SCENARIOS:
        print(f"\n📊 Checking normality for scenario: {scenario}")

        # Extract observables for this scenario
        n_obs = num_observables_per_scenario[scenario]
        start_idx = cumulative_obs_count
        end_idx = start_idx + n_obs
        x_train_scenario = x_train_multi[:, start_idx:end_idx]

        # Create scenario-specific observable names for display
        obs_names_scenario = [f"{obs}_{scenario}" for obs in observable_names_per_scenario[scenario]]

        plot_processed_data_normality(
            theta_train,
            x_train_scenario,
            param_names,
            obs_names_scenario,
            max_params=12,
            param_indices=None,  # Check all params
            obs_indices=None     # Check all observables
        )

        cumulative_obs_count += n_obs
else:
    print("\n⏭️  Skipping normality checks (set PLOT_NORMALITY_CHECKS=True to enable)")

# ============================================================================
# 3b. Diagnostic: Check if observed data is within training data range
# ============================================================================
print("\n" + "="*80)
print("OBSERVED DATA vs TRAINING DATA COVERAGE")
print("="*80)
print("\nChecking if observed values fall within the range of training simulations...")
print("⚠️  Values outside training range may cause posterior sampling issues.\n")

coverage_issues = []
for scenario in SCENARIOS:
    obs_data = get_observed_data(FILTERED_TEST_STATS_CSVS[scenario], value_column='median')
    obs_names = observable_names_per_scenario[scenario]
    # Use RAW (untransformed) training data for comparison with observed values
    x_train_raw = scenario_data[scenario]['x_train_raw']

    print(f"📊 {scenario}:")
    for i, name in enumerate(obs_names):
        if name in obs_data:
            obs_val_raw = obs_data[name]
            obs_val = float(obs_val_raw.item()) if hasattr(obs_val_raw, 'item') else float(obs_val_raw)
            train_min = float(x_train_raw[:, i].min().item()) if hasattr(x_train_raw[:, i].min(), 'item') else float(x_train_raw[:, i].min())
            train_max = float(x_train_raw[:, i].max().item()) if hasattr(x_train_raw[:, i].max(), 'item') else float(x_train_raw[:, i].max())
            train_mean = float(x_train_raw[:, i].mean().item()) if hasattr(x_train_raw[:, i].mean(), 'item') else float(x_train_raw[:, i].mean())
            train_std = float(x_train_raw[:, i].std().item()) if hasattr(x_train_raw[:, i].std(), 'item') else float(x_train_raw[:, i].std())

            # Check if observed is within range
            in_range = train_min <= obs_val <= train_max
            # Check how many std devs from mean
            z_score = (obs_val - train_mean) / train_std if train_std > 0 else 0

            status = "✓" if in_range else "❌ OUT OF RANGE"
            if not in_range:
                coverage_issues.append((scenario, name, obs_val, train_min, train_max))

            print(f"   {name}: obs={obs_val:.3f}, range=[{train_min:.3f}, {train_max:.3f}], z={z_score:.1f} {status}")

if coverage_issues:
    print(f"\n⚠️  WARNING: {len(coverage_issues)} observables have values outside training range!")
    print("   This will cause posterior sampling to fail with rejection sampling.")
    print("   Consider: (1) expanding priors, (2) using MCMC sampling, or (3) checking observed data.")
else:
    print("\n✓ All observed values are within training data range.")

# ============================================================================
# 4. Set up SBI inference
# ============================================================================
print("\n" + "="*80)
print("SETTING UP SBI INFERENCE")
print("="*80)

# Load and transform prior
print("\n📊 Loading prior distributions...")
prior_lognormal = load_prior(PRIORS_CSV)
print(f"   Original prior (LogNormal space): {prior_lognormal}")

print("\n🔄 Transforming prior from LogNormal to Normal (for log-space)...")
prior_normal = transform_lognormal_prior_to_normal(prior_lognormal)
print(f"   Transformed prior (Normal space): {prior_normal}")

print("\n✓ Multi-scenario data ready for SBI:")
print(f"   - theta_train: {theta_train.shape}")
print(f"   - x_train_multi: {x_train_multi.shape} (concatenated scenarios)")
print(f"   - theta_test: {theta_test.shape}")
print(f"   - x_test_multi: {x_test_multi.shape}")

# ============================================================================
# 5. Prior Predictive Checks (Per Scenario) - Optional
# ============================================================================
if PLOT_PRIOR_PPC:
    print("\n" + "="*80)
    print("PRIOR PREDICTIVE CHECKS")
    print("="*80)

    # Generate prior PPCs for each scenario separately
    prior_ppc_results = {}
    for scenario in SCENARIOS:
        print(f"\n📊 Generating prior predictive checks for {scenario}...")

        # Load observed data for this scenario (using scenario-specific test stats)
        obs_original = get_observed_data(FILTERED_TEST_STATS_CSVS[scenario], value_column='median')

        # Generate PPCs using scenario-specific simulator
        prior_ppc_results[scenario] = generate_prior_predictive_checks(
            observed_data=obs_original,
            qsp_simulator=simulators[scenario],
            n_samples=200
        )

        # Plot prior PPC histograms
        print(f"📊 Plotting prior PPC for {scenario}...")
        fig_prior, axes_prior = plot_ppc_histograms(
            prior_ppc_results[scenario],
            figsize=(16, 12),
            max_cols=3,
            bins=30,
            log_transform='auto',
            title=f'Prior Predictive Checks - {scenario.upper()}'
        )
        fig_prior.savefig(f'figures/prior_ppc_{scenario}.png', dpi=150, bbox_inches='tight')
        print(f"   ✓ Saved to figures/prior_ppc_{scenario}.png")
        plt.close(fig_prior)

    print("\n✓ Prior predictive checks complete for all scenarios")
else:
    print("\n⏭️  Skipping prior predictive checks (set PLOT_PRIOR_PPC=True to enable)")


# ============================================================================
# 6. Train Neural Posterior Estimator
# ============================================================================
print("\n" + "="*80)
print("TRAINING NEURAL POSTERIOR ESTIMATOR")
print("="*80)

print("\n🔮 Training NPE on multi-scenario data...")
print(f"   Input: θ ({num_params}D) → x_multi ({num_observables_total}D)")
print(f"   Training samples: {theta_train.shape[0]}")

inference = NPE(prior=prior_normal)
inference = NRE(prior = prior_normal)
inference = inference.append_simulations(theta_train, x_train_multi)
estimator = inference.train()

from sbi.inference.posteriors import MCMCPosteriorParameters, VIPosteriorParameters

params = MCMCPosteriorParameters(
    method="slice_np",
    thin=1,
    warmup_steps=100,
    num_chains=1,
    init_strategy="sir",
    init_strategy_parameters={"num_candidate_samples": 100},
    num_workers=1
    # mp_context="fork"
)

params = VIPosteriorParameters()
posterior = inference.build_posterior(posterior_parameters=params)
# posterior = inference.build_posterior()


print(f"✓ Training complete")
print(f"   Posterior: {posterior}")


# ============================================================================
# 7. Sample from Posterior and Run Diagnostics
# ============================================================================
print("\n" + "="*80)
print("POSTERIOR SAMPLING AND DIAGNOSTICS")
print("="*80)

print(f"\n🔮 Sampling from posterior (100 samples per test observation)...")

# Filter out test observations with values outside training range
# (extreme values cause rejection sampling to fail)
train_min = x_train_multi.min().item()
train_max = x_train_multi.max().item()
valid_mask = (x_test_multi <= train_max).all(dim=1) & (x_test_multi >= train_min).all(dim=1)
x_test_valid = x_test_multi[valid_mask]
theta_test_valid = theta_test[valid_mask]
print(f"   Using {valid_mask.sum()}/{len(valid_mask)} valid test observations (filtered out-of-range)")

samples_list = Parallel(n_jobs=-1, prefer="threads")(
    delayed(posterior.sample)((100,), x=x_o) for x_o in x_test_valid
)

samples = torch.stack(samples_list, dim=1)  # Shape: (100, n_obs, n_params)
print(f"   Samples shape: {samples.shape}")

# Select subset of parameters to plot
param_indices = range(min(8, num_params))  # First 8 parameters

# Recovery plot
print(f"\n📊 Creating recovery plot...")
fig_recovery, axes_recovery = sbi_recovery(
    samples, theta_test_valid, param_names, param_indices=param_indices
)
fig_recovery.savefig('figures/multi_scenario_recovery.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved to figures/multi_scenario_recovery.png")
plt.close(fig_recovery)

# Z-score contraction plot
print(f"\n📊 Creating z-score contraction plot...")
fig_zscore, axes_zscore, z_scores, contractions = sbi_z_score_contraction(
    samples, theta_test_valid, prior_normal, param_names, param_indices=param_indices
)
fig_zscore.savefig('figures/multi_scenario_z_score_contraction.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved to figures/multi_scenario_z_score_contraction.png")
plt.close(fig_zscore)

# Calibration ECDF plot
print(f"\n📊 Creating calibration ECDF plot...")
fig_calib, axes_calib, ranks = sbi_calibration_ecdf(
    samples, theta_test_valid, param_names, difference=True, param_indices=param_indices
)
fig_calib.savefig('figures/multi_scenario_calibration_ecdf.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved to figures/multi_scenario_calibration_ecdf.png")
plt.close(fig_calib)

print("\n✓ Diagnostics complete")

# ============================================================================
# 8. Load Observed Data and Sample from Posterior
# ============================================================================
print("\n" + "="*80)
print("LOADING OBSERVED DATA AND SAMPLING POSTERIOR")
print("="*80)

# Prepare observed data for each scenario and concatenate
print(f"\n📊 Preparing observed data for all scenarios...")
obs_transformed_list = []
obs_original_dict = {}

for scenario in SCENARIOS:
    obs_orig, obs_trans = prepare_observed_data(
        test_stats_csv=FILTERED_TEST_STATS_CSVS[scenario],  # Filtered test stats
        observable_names=observable_names_per_scenario[scenario],  # Scenario-specific observable names
        obs_quantiles=obs_quantiles_multi[scenario]
    )
    obs_transformed_list.append(obs_trans)
    obs_original_dict[scenario] = obs_orig

# Concatenate transformed observables for all scenarios
obs_transformed_multi = torch.cat(obs_transformed_list, dim=0)
print(f"   ✓ Multi-scenario observed data: {obs_transformed_multi.shape}")

# Sample from posterior conditioned on multi-scenario observations
print(f"\n🔮 Sampling from posterior conditioned on all scenarios...")
post_samples = posterior.sample((1000,), x=obs_transformed_multi)
print(f"   ✓ Sampled {post_samples.shape[0]} posterior samples")

# Convert to original space
post_samples_dict = convert_posterior_samples_to_original_space(post_samples, param_names)

# Plot posterior marginals
print("\n📊 Plotting posterior marginal distributions...")
fig_post, axes_post = plot_posterior_marginals(
    post_samples_dict,
    log_transform=True,
    priors_csv=PRIORS_CSV,
    credible_intervals=[0.95],
    figsize=(16, 12),
    param_indices=param_indices
)
fig_post.savefig('figures/multi_scenario_posterior_marginals.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved to figures/multi_scenario_posterior_marginals.png")
plt.close(fig_post)

# Plot posterior vs prior comparison
print("\n📊 Plotting posterior vs prior comparison...")
fig_vs_prior, axes_vs_prior = plot_posterior_vs_prior_marginals(
    post_samples_dict,
    priors_csv=PRIORS_CSV,
    param_names=param_names,
    n_prior_samples=5000,
    log_transform=True,
    figsize=(16, 12),
    param_indices=param_indices
)
fig_vs_prior.savefig('figures/multi_scenario_posterior_vs_prior.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved to figures/multi_scenario_posterior_vs_prior.png")
plt.close(fig_vs_prior)

# Plot posterior pairs
print("\n📊 Creating pairs plot for selected parameters...")
selected_params = param_names[:6]  # First 6 parameters
fig_pairs, axes_pairs = plot_posterior_pairs(
    post_samples_dict,
    param_names_subset=selected_params,
    all_param_names=param_names,
    priors_csv=PRIORS_CSV,
    log_transform=True,
    credible_intervals=[0.95],
    contour_levels=[0.68, 0.95]
)
fig_pairs.savefig('figures/multi_scenario_posterior_pairs.png', dpi=150, bbox_inches='tight')
print(f"   ✓ Saved to figures/multi_scenario_posterior_pairs.png")
plt.close(fig_pairs)

# ============================================================================
# 9. Posterior Predictive Checks (Per Scenario)
# ============================================================================
print("\n" + "="*80)
print("POSTERIOR PREDICTIVE CHECKS")
print("="*80)

# Generate PPCs for each scenario separately
n_samples_for_analysis = 200
ppc_samples = {k: v[:n_samples_for_analysis] for k, v in post_samples_dict.items()}
ppc_results_all = {}

for scenario in SCENARIOS:
    print(f"\n🔮 Generating posterior predictive checks for {scenario} ({n_samples_for_analysis} samples)...")

    ppc_results_all[scenario] = generate_posterior_predictive_checks(
        posterior_samples=ppc_samples,
        observed_data=obs_original_dict[scenario],
        qsp_simulator=simulators[scenario]
    )

    # Plot PPC histograms
    print(f"📊 Plotting posterior PPC for {scenario}...")
    fig_ppc, axes_ppc = plot_ppc_histograms(
        ppc_results_all[scenario],
        figsize=(16, 12),
        max_cols=3,
        bins=30,
        log_transform='auto',
        title=f'Posterior Predictive Checks - {scenario.upper()}'
    )
    fig_ppc.savefig(f'figures/posterior_ppc_{scenario}.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to figures/posterior_ppc_{scenario}.png")
    plt.close(fig_ppc)

print("\n✓ Posterior predictive checks complete for all scenarios")

# ============================================================================
# 10. Posterior Predictive Simulations (Per Scenario)
# ============================================================================
print("\n" + "="*80)
print("POSTERIOR PREDICTIVE SIMULATIONS")
print("="*80)

# Species to plot
species_of_interest = [
    'V_T_C1',      # Tumor cancer cells
    'V_T_CD8',     # CD8+ T cells in tumor
    'V_T_Treg',    # Regulatory T cells in tumor
    'V_T_MDSC',    # MDSCs in tumor
    'V_T_Mac_M1',  # M1 macrophages in tumor
    'V_T_Mac_M2',  # M2 macrophages in tumor
]

# Generate spaghetti plots for each scenario
n_full_sims = 50
pp_samples = {k: v[:n_full_sims] for k, v in post_samples_dict.items()}

for scenario in SCENARIOS:
    print(f"\n🔮 Generating posterior predictive simulations for {scenario} ({n_full_sims} samples)...")

    pp_sim_results = generate_posterior_predictive_simulations(
        posterior_samples=pp_samples,
        observed_data=obs_original_dict[scenario],
        qsp_simulator=simulators[scenario],
        species_of_interest=species_of_interest
    )

    # Plot spaghetti plots
    print(f"📊 Plotting posterior predictive trajectories for {scenario}...")
    fig_spaghetti, axes_spaghetti = plot_posterior_predictive_spaghetti(
        pp_sim_results,
        species_to_plot=species_of_interest,
        figsize=(16, 10),
        max_cols=2,
        alpha=0.2,
        show_median=True,
        show_credible=True,
        credible_level=0.95
    )
    fig_spaghetti.suptitle(f'Posterior Predictive Simulations - {scenario.upper()}', fontsize=16, y=1.002)
    fig_spaghetti.savefig(f'figures/posterior_spaghetti_{scenario}.png', dpi=150, bbox_inches='tight')
    print(f"   ✓ Saved to figures/posterior_spaghetti_{scenario}.png")
    plt.close(fig_spaghetti)

print("\n✓ Posterior predictive simulations complete for all scenarios")
