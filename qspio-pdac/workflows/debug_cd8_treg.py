"""Debug script to diagnose CD8/Treg ratio issues."""
import time
from pathlib import Path
import pandas as pd
import numpy as np

from qsp_hpc import QSPSimulator

# Configuration
NUM_SIMS = 5  # Small number for debugging
PRIORS_CSV = 'scenarios/pdac_priors_tuned.csv'
TEST_STATS_CSV = 'metadata-storage/calibration_targets_derived.csv'
MODEL_STRUCTURE_FILE = 'model_structure.json'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'
LOCAL_MATLAB_PATH = '/Applications/MATLAB_R2024b.app/bin/matlab'

print("=" * 60)
print("CD8/Treg Ratio Diagnostic")
print("=" * 60)

# Create simulator
print("\n📊 Creating simulator...")
simulator = QSPSimulator(
    test_stats_csv=TEST_STATS_CSV,
    model_structure_file=MODEL_STRUCTURE_FILE,
    priors_csv=PRIORS_CSV,
    model_script=MODEL_SCRIPT,
    model_version='debug_cd8_treg',
    model_description='Debug CD8/Treg ratio',
    scenario='baseline_no_treatment',
    verbose=True,
    project_root=Path('.'),
    max_tasks=1,
    seed=42,
    cache_sampling_seed=42
)

# Run simulation
print(f"\n🔬 Running {NUM_SIMS} simulations...")
params, test_stats = simulator.run_local_simulation(
    seed=42,
    matlab_path=LOCAL_MATLAB_PATH,
    n_sims=NUM_SIMS
)

# Load test stat names
test_stats_df = pd.read_csv(TEST_STATS_CSV)
obs_names = test_stats_df['test_statistic_id'].tolist()

print(f"\n✓ Simulation complete")
print(f"  params shape: {params.shape}")
print(f"  test_stats shape: {test_stats.shape}")

# Find indices for key observables
key_obs = ['cd8_density_baseline', 'treg_density_baseline', 'cd8_treg_ratio',
           'cd8_exhausted_fraction', 'cancer_cell_density_baseline']

print("\n" + "=" * 60)
print("TEST STATISTIC VALUES")
print("=" * 60)

for obs_name in key_obs:
    if obs_name in obs_names:
        idx = obs_names.index(obs_name)
        values = test_stats[:, idx]
        observed = test_stats_df[test_stats_df['test_statistic_id'] == obs_name]['median'].values[0]
        print(f"\n{obs_name}:")
        print(f"  Observed: {observed:.4g}")
        print(f"  Model values: {values}")
        print(f"  Model median: {np.nanmedian(values):.4g}")
        print(f"  Ratio (model/obs): {np.nanmedian(values)/observed:.2f}x")

# Check consistency
print("\n" + "=" * 60)
print("CONSISTENCY CHECK")
print("=" * 60)

cd8_dens_idx = obs_names.index('cd8_density_baseline')
treg_dens_idx = obs_names.index('treg_density_baseline')
ratio_idx = obs_names.index('cd8_treg_ratio')

cd8_dens = test_stats[:, cd8_dens_idx]
treg_dens = test_stats[:, treg_dens_idx]
ratio_reported = test_stats[:, ratio_idx]

# Note: density uses only V_T.CD8, ratio uses V_T.CD8 + V_T.CD8_exh
# So they won't match exactly
print(f"\ncd8_density / treg_density = {np.nanmedian(cd8_dens / treg_dens):.4g}")
print(f"cd8_treg_ratio (reported) = {np.nanmedian(ratio_reported):.4g}")
print("\nNote: density uses V_T.CD8 only, ratio uses V_T.CD8 + V_T.CD8_exh")

# Load priors to check key parameters
print("\n" + "=" * 60)
print("KEY PARAMETER VALUES (from priors)")
print("=" * 60)

priors_df = pd.read_csv(PRIORS_CSV)
key_params = ['q_CD8_T_in', 'q_Treg_T_in', 'k_CCR5_Treg_rec', 'k_C_CD8_exh', 'k_Treg_death']

for param in key_params:
    if param in priors_df['name'].values:
        row = priors_df[priors_df['name'] == param].iloc[0]
        print(f"  {param}: median={row['median']:.4g}, bounds=[{row['lower_bound']:.4g}, {row['upper_bound']:.4g}]")

print("\nDone.")
