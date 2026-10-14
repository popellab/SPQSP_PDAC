#!/usr/bin/env python3
"""
Local simulation smoke test for PDAC QSP model.

Runs a small number of simulations locally (via MATLAB) and verifies that
test statistics are computed correctly. Use this to validate the pipeline
before running larger workflows (restriction estimator, Morris SA, SBI).

Usage:
    .venv/bin/python scripts/test_local_simulation.py
"""

from pathlib import Path

import numpy as np

from qsp_hpc import QSPSimulator

# ============================================================================
# Configuration
# ============================================================================

PRIORS_CSV = 'parameters/pdac_priors.csv'
CALIBRATION_TARGETS_DIR = 'calibration_targets/baseline_no_treatment'
MODEL_STRUCTURE_FILE = 'model_structure.json'
SCENARIO = 'baseline_no_treatment'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'
MATLAB_PATH = '/Applications/MATLAB_R2024b.app/bin/matlab'

N_SIMS = 3
SEED = 42

# ============================================================================
# 1. Create simulator
# ============================================================================
print("=" * 80)
print("LOCAL SIMULATION SMOKE TEST")
print("=" * 80)

project_root = Path('.').resolve()
print(f"\nProject root: {project_root}")
print(f"Priors CSV:   {PRIORS_CSV}")
print(f"Cal targets:  {CALIBRATION_TARGETS_DIR}")
print(f"Scenario:     {SCENARIO}")
print(f"N sims:       {N_SIMS}")
print(f"Seed:         {SEED}")

sim = QSPSimulator(
    priors_csv=PRIORS_CSV,
    calibration_targets=CALIBRATION_TARGETS_DIR,
    model_structure_file=MODEL_STRUCTURE_FILE,
    model_script=MODEL_SCRIPT,
    model_version='local_test',
    model_description='Local smoke test',
    scenario=SCENARIO,
    project_root=project_root,
    verbose=True,
)

# Inspect what test stats will be computed
test_stats_df = sim._test_stats_df
target_names = test_stats_df['test_statistic_id'].tolist()
print(f"\nTest statistics ({len(target_names)}):")
for i, name in enumerate(target_names):
    print(f"  [{i:2d}] {name}")

# ============================================================================
# 2. Run local simulations
# ============================================================================
print(f"\n{'=' * 80}")
print(f"RUNNING {N_SIMS} LOCAL SIMULATIONS")
print("=" * 80)

params, test_stats = sim.run_local_simulation(
    n_sims=N_SIMS,
    seed=SEED,
    matlab_path=MATLAB_PATH,
)

print(f"\nParams shape:     {params.shape}")
print(f"Test stats shape: {test_stats.shape}")

# ============================================================================
# 3. Analyze results
# ============================================================================
print(f"\n{'=' * 80}")
print("RESULTS ANALYSIS")
print("=" * 80)

# Per-simulation NaN check
for i in range(N_SIMS):
    row = test_stats[i]
    n_nan = np.isnan(row).sum()
    n_inf = np.isinf(row).sum()
    n_valid = (~np.isnan(row) & ~np.isinf(row)).sum()
    status = "VALID" if n_nan == 0 and n_inf == 0 else "HAS NaN/Inf"
    print(f"\n  Sim {i}: {status} ({n_valid} valid, {n_nan} NaN, {n_inf} Inf)")

    if n_nan == 0:
        # Print all test stat values for valid sims
        for j, name in enumerate(target_names):
            print(f"    {name:45s} = {row[j]:.6g}")

# Summary statistics across sims
print(f"\n{'=' * 80}")
print("SUMMARY")
print("=" * 80)

nan_per_sim = np.isnan(test_stats).any(axis=1)
print(f"\n  Total sims:   {N_SIMS}")
print(f"  Valid sims:   {(~nan_per_sim).sum()}")
print(f"  Invalid sims: {nan_per_sim.sum()} (contain NaN)")

if (~nan_per_sim).any():
    valid_stats = test_stats[~nan_per_sim]
    print(f"\n  Test statistic ranges (across valid sims):")
    for j, name in enumerate(target_names):
        col = valid_stats[:, j]
        if len(col) > 1:
            print(f"    {name:45s}  min={col.min():.4g}  max={col.max():.4g}  "
                  f"mean={col.mean():.4g}")
        else:
            print(f"    {name:45s}  value={col[0]:.4g}")

# Parameter ranges
print(f"\n  Parameter ranges (first 10):")
param_names_list = sim.param_names[:10]
for j, name in enumerate(param_names_list):
    col = params[:, j]
    print(f"    {name:45s}  [{col.min():.4g}, {col.max():.4g}]")

print(f"\nDone.")
