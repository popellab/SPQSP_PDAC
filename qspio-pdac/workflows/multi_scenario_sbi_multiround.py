#!/usr/bin/env python3
"""
Multi-Round Multi-Scenario SBI Workflow for PDAC QSP Model

This script implements multi-round simulation-based inference (SBI) using
a function-based approach. Each round can be executed manually, allowing
you to inspect results and decide whether to continue.

Usage:
    # Run interactively in IPython/Jupyter:
    %run workflows/multi_scenario_sbi_multiround.py

    # Then call functions step by step:
    theta1, x1 = run_round1_simulations(n_sims=3000)
    posterior1, inference = train_round(theta1, x1, round_num=1)
    plot_diagnostics(posterior1, round_num=1)

    # If satisfied, continue to round 2:
    theta2, x2 = run_posterior_simulations(posterior1, n_sims=2000)
    posterior2, inference = train_round(theta2, x2, round_num=2, inference=inference)
    plot_diagnostics(posterior2, round_num=2)

Reference:
    https://sbi.readthedocs.io/en/latest/advanced_tutorials/02_multiround_inference.html
"""

import sys
import time
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt

from sbi.inference import NPE

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
    prepare_observed_data,
    convert_posterior_samples_to_original_space,
    sbi_recovery,
    sbi_z_score_contraction,
    sbi_calibration_ecdf
)
from qsp_inference.priors import load_prior, transform_lognormal_prior_to_normal
from qsp_inference.inference import compute_quantiles_from_array, transform_to_normal_from_array

# ============================================================================
# CONFIGURATION
# ============================================================================

MODEL_VERSION = 'pdac_multi_scenario_multiround_v2_fActiveTGFb'  # Bumped for f_active_TGFb + averaged polarization
MODEL_DESCRIPTION = 'PDAC multi-scenario multi-round: iterative posterior refinement'
PRIORS_CSV = 'scenarios/pdac_priors_tuned.csv'

# Using calibration_targets_derived.csv which contains:
#   - test_statistic_id: observable name
#   - median: observed value from literature (used as x_o)
#   - model_output_code: Python code to compute observable from simulation
TEST_STATS_CSVS = {
    'baseline_no_treatment': 'metadata-storage/calibration_targets_derived.csv',
}

MODEL_STRUCTURE_FILE = 'model_structure.json'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'
SCENARIOS = list(TEST_STATS_CSVS.keys())

# Simulation settings
MAX_TASKS = 40
LOCAL_MATLAB_PATH = '/Applications/MATLAB_R2024b.app/bin/matlab'
SIMULATION_SEED = 2026

# Execution mode: 'local' or 'hpc'
# - 'local': Run MATLAB simulations on local machine (good for testing, small runs)
# - 'hpc': Use HPC cluster via qsp_hpc (requires SSH connection to cluster)
EXECUTION_MODE = 'hpc'

# Training settings
BATCH_SIZE = 64
EPOCHS = 500

# ============================================================================
# SETUP (runs on import)
# ============================================================================

print("="*80)
print("MULTI-ROUND SBI WORKFLOW - FUNCTION-BASED")
print("="*80)

# Load parameter names
priors_df = pd.read_csv(PRIORS_CSV)
param_names = priors_df['name'].tolist()
num_params = len(param_names)

# Load observable names per scenario
observable_names_per_scenario = {}
num_observables_per_scenario = {}
for scenario in SCENARIOS:
    test_stats_df = pd.read_csv(TEST_STATS_CSVS[scenario])
    observable_names_per_scenario[scenario] = test_stats_df['test_statistic_id'].tolist()
    num_observables_per_scenario[scenario] = len(observable_names_per_scenario[scenario])

num_observables_total = sum(num_observables_per_scenario.values())

print(f"\n✓ Loaded {num_params} parameters")
print(f"✓ Loaded {num_observables_total} observables")
for scenario in SCENARIOS:
    print(f"   - {scenario}: {num_observables_per_scenario[scenario]} observables")

# Create simulators
print("\n📊 Creating simulators...")
simulators = {}
for scenario in SCENARIOS:
    simulators[scenario] = QSPSimulator(
        test_stats_csv=TEST_STATS_CSVS[scenario],
        model_structure_file=MODEL_STRUCTURE_FILE,
        priors_csv=PRIORS_CSV,
        model_script=MODEL_SCRIPT,
        model_version=MODEL_VERSION,
        model_description=MODEL_DESCRIPTION,
        scenario=scenario,
        verbose=True,
        project_root=Path('.'),
        max_tasks=MAX_TASKS,
        seed=SIMULATION_SEED,
        cache_sampling_seed=2025
    )
print(f"   ✓ Created {len(simulators)} simulator(s)")

# Load priors
print("\n📊 Loading priors...")
prior_lognormal = load_prior(PRIORS_CSV)
prior_normal = transform_lognormal_prior_to_normal(prior_lognormal)
print(f"   ✓ Prior (log-space): {num_params} parameters")

# Load observed data (calibration targets)
print("\n📊 Loading observed data (calibration targets)...")
obs_raw_dict = {}
for scenario in SCENARIOS:
    test_stats_df = pd.read_csv(TEST_STATS_CSVS[scenario])
    obs_raw_dict[scenario] = {}
    for _, row in test_stats_df.iterrows():
        obs_raw_dict[scenario][row['test_statistic_id']] = row['median']
    print(f"   ✓ {scenario}: {len(obs_raw_dict[scenario])} calibration targets")

# Global state for transforms (fitted on round 1)
obs_quantiles = None
x_o_transformed = None

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def run_round1_simulations(n_sims: int = 3000, execution_mode: str = None) -> tuple:
    """
    Run round 1 simulations sampling from the prior.

    Args:
        n_sims: Number of simulations to run
        execution_mode: 'local' or 'hpc' (default: use EXECUTION_MODE config)

    Returns:
        (theta, x_raw): Parameters (log-space) and raw observables
    """
    global obs_quantiles, x_o_transformed

    mode = execution_mode or EXECUTION_MODE

    print(f"\n{'='*60}")
    print(f"ROUND 1: SIMULATIONS FROM PRIOR")
    print(f"{'='*60}")
    print(f"Running {n_sims} simulations ({mode.upper()} mode)...")

    x_per_scenario = []
    theta_out = None

    for scenario in SCENARIOS:
        print(f"\n   📊 Simulating {scenario}...")

        if mode == 'local':
            # Local MATLAB execution
            params, test_stats = simulators[scenario].run_local_simulation(
                n_sims=n_sims,
                seed=SIMULATION_SEED,
                matlab_path=LOCAL_MATLAB_PATH
            )
        elif mode == 'hpc':
            # HPC execution (uses pool/cluster)
            params, test_stats = simulators[scenario](n_sims)
        else:
            raise ValueError(f"Unknown execution mode: {mode}. Use 'local' or 'hpc'.")

        # Log test statistics summary (passing/failing table)
        simulators[scenario].log_test_statistics_summary(test_stats)

        x_per_scenario.append(test_stats)
        if theta_out is None:
            theta_out = params

    # Convert to tensors
    theta_log = torch.from_numpy(np.log(theta_out)).float()
    x_raw = torch.from_numpy(np.hstack(x_per_scenario)).float()

    # Filter NaN simulations
    nan_mask = torch.isnan(x_raw).any(dim=1)
    n_valid = (~nan_mask).sum().item()
    print(f"\n✓ Valid simulations: {n_valid}/{n_sims} ({100*n_valid/n_sims:.1f}%)")

    theta_valid = theta_log[~nan_mask]
    x_valid = x_raw[~nan_mask]

    # Fit quantile transforms on round 1 data
    print(f"\n📊 Fitting quantile transforms on {n_valid} samples...")
    obs_quantiles = compute_quantiles_from_array(x_valid.numpy(), n_quantiles=1000)

    # Transform observed data
    x_obs_list = []
    for scenario in SCENARIOS:
        for name in observable_names_per_scenario[scenario]:
            x_obs_list.append(obs_raw_dict[scenario][name])
    x_obs_array = np.array([x_obs_list])
    x_o_transformed = torch.from_numpy(
        transform_to_normal_from_array(x_obs_array, obs_quantiles)[0]
    ).float()
    print(f"   ✓ Observed data transformed: {x_o_transformed.shape}")

    return theta_valid, x_valid


def run_posterior_simulations(posterior, n_sims: int = 2000) -> tuple:
    """
    Run simulations sampling from the posterior.

    Args:
        posterior: Trained posterior from previous round
        n_sims: Number of simulations to run

    Returns:
        (theta, x_raw): Parameters (log-space) and raw observables
    """
    import json
    from qsp_hpc.simulation.batch_runner import run_batch_worker
    from qsp_hpc.batch.derive_test_stats_worker import (
        build_test_stat_registry,
        compute_test_statistics_batch,
    )

    print(f"\n{'='*60}")
    print(f"SIMULATIONS FROM POSTERIOR")
    print(f"{'='*60}")

    # Sample from posterior
    print(f"🎲 Sampling {n_sims} parameters from posterior...")
    theta_log = posterior.sample((n_sims,), x=x_o_transformed)
    theta_original = torch.exp(theta_log).numpy()
    print(f"   ✓ Sampled {theta_log.shape}")

    # Run simulations for each scenario
    x_per_scenario = []
    for scenario in SCENARIOS:
        print(f"\n   📊 Simulating {scenario}...")
        sim = simulators[scenario]

        # Run MATLAB
        parquet_file = run_batch_worker(
            params=theta_original,
            param_names=param_names,
            model_script=sim.model_script or "default_model",
            project_root=sim.project_root,
            seed=SIMULATION_SEED + 1000,  # Different seed for posterior sims
            dosing=sim.dosing,
            sim_config=sim.sim_config,
            matlab_path=LOCAL_MATLAB_PATH,
            verbose=sim.verbose,
        )

        # Derive test statistics
        species_df = pd.read_parquet(parquet_file)
        test_stats_df = pd.read_csv(TEST_STATS_CSVS[scenario])
        test_stat_registry = build_test_stat_registry(test_stats_df)

        if sim.model_structure_file is not None and sim.model_structure_file.exists():
            with open(sim.model_structure_file, "r") as f:
                data = json.load(f)
            species_units = {s["name"]: s["units"] for s in data["species"]}
        else:
            species_units = {}

        test_stats = compute_test_statistics_batch(
            species_df, test_stats_df, test_stat_registry, species_units
        )

        # Log test statistics summary (passing/failing table)
        sim.log_test_statistics_summary(test_stats)

        x_per_scenario.append(test_stats)

    # Convert to tensors
    x_raw = torch.from_numpy(np.hstack(x_per_scenario)).float()

    # Filter NaN
    nan_mask = torch.isnan(x_raw).any(dim=1)
    n_valid = (~nan_mask).sum().item()
    print(f"\n✓ Valid simulations: {n_valid}/{n_sims} ({100*n_valid/n_sims:.1f}%)")

    theta_valid = theta_log[~nan_mask]
    x_valid = x_raw[~nan_mask]

    return theta_valid, x_valid


def train_round(theta, x, round_num: int, inference=None, proposal=None):
    """
    Train NPE on simulation data.

    Args:
        theta: Parameters (log-space), shape (n_sims, n_params)
        x: Raw observables, shape (n_sims, n_obs)
        round_num: Round number (1, 2, 3, ...)
        inference: Existing NPE inference object (None for round 1)
        proposal: Proposal distribution used to generate theta
                  (prior for round 1, posterior for round 2+)

    Returns:
        (posterior, inference): Trained posterior and inference object
    """
    print(f"\n{'='*60}")
    print(f"TRAINING ROUND {round_num}")
    print(f"{'='*60}")

    # Transform observables
    print(f"📊 Transforming {x.shape[0]} samples...")
    x_transformed = torch.from_numpy(
        transform_to_normal_from_array(x.numpy(), obs_quantiles)
    ).float()

    # Initialize or reuse inference
    if inference is None:
        print(f"   Initializing new NPE...")
        inference = NPE(prior=prior_normal)
        proposal = prior_normal

    if proposal is None:
        proposal = prior_normal

    print(f"\n🧠 Training NPE...")
    print(f"   θ: {theta.shape}, x: {x_transformed.shape}")
    print(f"   Batch size: {BATCH_SIZE}, Max epochs: {EPOCHS}")

    density_estimator = inference.append_simulations(
        theta, x_transformed, proposal=proposal
    ).train(
        training_batch_size=BATCH_SIZE,
        max_num_epochs=EPOCHS
    )

    posterior = inference.build_posterior(density_estimator)

    # Update proposal for next round
    posterior_with_default_x = posterior.set_default_x(x_o_transformed)

    print(f"\n✓ Round {round_num} training complete")

    return posterior_with_default_x, inference


def plot_diagnostics(posterior, round_num: int, n_samples: int = 1000):
    """
    Plot diagnostic figures for a trained posterior.

    Args:
        posterior: Trained posterior
        round_num: Round number for labeling
        n_samples: Number of posterior samples for plots
    """
    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS - ROUND {round_num}")
    print(f"{'='*60}")

    # Sample from posterior
    print(f"\n🔮 Sampling {n_samples} from posterior...")
    samples = posterior.sample((n_samples,), x=x_o_transformed)
    samples_dict = convert_posterior_samples_to_original_space(samples, param_names)

    param_indices = range(min(12, num_params))

    # 1. Posterior marginals
    print(f"\n📊 Plotting posterior marginals...")
    fig1, _ = plot_posterior_marginals(
        samples_dict,
        log_transform=True,
        priors_csv=PRIORS_CSV,
        credible_intervals=[0.95],
        figsize=(16, 12),
        param_indices=param_indices
    )
    fig1.suptitle(f'Posterior Marginals - Round {round_num}', fontsize=14, y=1.01)
    fig1.savefig(f'figures/multiround_r{round_num}_posterior_marginals.png', dpi=150, bbox_inches='tight')
    plt.close(fig1)
    print(f"   ✓ figures/multiround_r{round_num}_posterior_marginals.png")

    # 2. Posterior vs Prior
    print(f"\n📊 Plotting posterior vs prior...")
    fig2, _ = plot_posterior_vs_prior_marginals(
        samples_dict,
        priors_csv=PRIORS_CSV,
        param_names=param_names,
        n_prior_samples=5000,
        log_transform=True,
        figsize=(16, 12),
        param_indices=param_indices
    )
    fig2.suptitle(f'Round {round_num}: Posterior (blue) vs Prior (gray)', fontsize=14, y=1.01)
    fig2.savefig(f'figures/multiround_r{round_num}_posterior_vs_prior.png', dpi=150, bbox_inches='tight')
    plt.close(fig2)
    print(f"   ✓ figures/multiround_r{round_num}_posterior_vs_prior.png")

    # 3. Pairs plot (first 6 params)
    print(f"\n📊 Plotting pairs...")
    selected_params = param_names[:6]
    fig3, _ = plot_posterior_pairs(
        samples_dict,
        param_names_subset=selected_params,
        all_param_names=param_names,
        priors_csv=PRIORS_CSV,
        log_transform=True,
        credible_intervals=[0.95],
        contour_levels=[0.68, 0.95]
    )
    fig3.suptitle(f'Posterior Pairs - Round {round_num}', fontsize=14, y=1.01)
    fig3.savefig(f'figures/multiround_r{round_num}_posterior_pairs.png', dpi=150, bbox_inches='tight')
    plt.close(fig3)
    print(f"   ✓ figures/multiround_r{round_num}_posterior_pairs.png")

    print(f"\n✓ Diagnostics complete for round {round_num}")

    return samples_dict


def run_ppc(posterior, round_num: int, n_samples: int = 200):
    """
    Run posterior predictive checks.

    Args:
        posterior: Trained posterior
        round_num: Round number for labeling
        n_samples: Number of samples for PPC
    """
    print(f"\n{'='*60}")
    print(f"POSTERIOR PREDICTIVE CHECKS - ROUND {round_num}")
    print(f"{'='*60}")

    # Sample from posterior
    samples = posterior.sample((n_samples,), x=x_o_transformed)
    samples_dict = convert_posterior_samples_to_original_space(samples, param_names)
    ppc_samples = {k: v[:n_samples] for k, v in samples_dict.items()}

    for scenario in SCENARIOS:
        print(f"\n🔮 Running PPC for {scenario}...")

        ppc_results = generate_posterior_predictive_checks(
            posterior_samples=ppc_samples,
            observed_data=obs_raw_dict[scenario],
            qsp_simulator=simulators[scenario]
        )

        fig, _ = plot_ppc_histograms(
            ppc_results,
            figsize=(16, 12),
            max_cols=3,
            bins=30,
            log_transform='auto',
            title=f'Posterior Predictive Checks - Round {round_num} - {scenario.upper()}'
        )
        fig.savefig(f'figures/multiround_r{round_num}_ppc_{scenario}.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✓ figures/multiround_r{round_num}_ppc_{scenario}.png")



# ROUND 1: Sample from prior
N_SIMS_R1 = 500
theta1, x1 = run_round1_simulations(n_sims=N_SIMS_R1)
# posterior1, inference = train_round(theta1, x1, round_num=1)
# plot_diagnostics(posterior1, round_num=1)

# Review figures/multiround_r1_posterior_vs_prior.png
# If the posterior is informative, continue...

# ROUND 2: Sample from posterior
# theta2, x2 = run_posterior_simulations(posterior1, n_sims=N_SIMS_R2)
# posterior2, inference = train_round(theta2, x2, round_num=2,
#                                     inference=inference, proposal=posterior1)
# plot_diagnostics(posterior2, round_num=2)
