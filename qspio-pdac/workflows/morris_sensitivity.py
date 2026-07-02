#!/usr/bin/env python3
"""
Morris Sensitivity Analysis for PDAC QSP Model

Performs Morris (Elementary Effects) screening to identify which of the 175 QSP
model parameters most influence calibration target outputs. This is a pre-SBI
screening step to enable dimensionality reduction.

Morris method outputs per parameter per output:
- mu_star: Mean absolute elementary effect (importance measure)
- sigma: Std of elementary effects (interaction/nonlinearity indicator)
- When sigma >> mu_star, the parameter is involved in interactions or nonlinear effects

Usage:
    python workflows/morris_sensitivity.py

Output:
    figures/morris_sensitivity/   - Bar plots, scatter plots, heatmap
    results/morris_sensitivity/   - CSV rankings and raw results
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from SALib.sample import morris as morris_sampler
from SALib.analyze import morris as morris_analyzer

from qsp_hpc import QSPSimulator

# ============================================================================
# CONFIGURATION
# ============================================================================

PRIORS_CSV = 'parameters/pdac_priors.csv'
CALIBRATION_TARGETS_DIR = 'calibration_targets/baseline_no_treatment'
MODEL_STRUCTURE_FILE = 'model_structure.json'
SCENARIO = 'baseline_no_treatment'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'
MODEL_VERSION = 'morris_screening_v2_fActiveTGFb'  # Bumped for f_active_TGFb + averaged polarization
MODEL_DESCRIPTION = 'Morris screening: 175 params, 31 calibration targets, baseline scenario'

N_TRAJECTORIES = 4        # How many to simulate THIS run (increase progressively: 2 → 5 → 20 → 50)
N_TRAJECTORIES_MAX = 50   # Pre-generate this many trajectories; DOE is stable across runs
NUM_LEVELS = 4            # p=4 levels in the Morris grid
MAX_TASKS = 40            # Parallel SLURM array tasks
SEED = 2025
CACHE_DIR = Path('morris-cache')

# Restriction estimator: set to path to pre-filter DOE using trained classifier
# Run workflows/restriction_estimator.py first to generate these artifacts
RESTRICTION_ESTIMATOR_DIR = None  # e.g., Path('cache/restriction_estimator')
OVERSAMPLE_FACTOR = 3             # Generate 3x trajectories, filter down to N_TRAJECTORIES

# Visualization
TOP_N_PARAMS = 20         # Show top-N params in bar plots
FIG_DIR = Path('figures/morris_sensitivity')
RESULTS_DIR = Path('results/morris_sensitivity')

# ============================================================================
# 1. Load priors and build SALib problem definition
# ============================================================================
print("=" * 80)
print("MORRIS SENSITIVITY ANALYSIS FOR PDAC QSP MODEL")
print("=" * 80)

print(f"\n1. Loading priors from {PRIORS_CSV}...")
priors_df = pd.read_csv(PRIORS_CSV)
param_names = priors_df['name'].tolist()
n_params = len(param_names)
print(f"   {n_params} parameters loaded")

# All parameters in pdac_priors.csv are lognormal — sample in log10-space
# Bounds: [log10(lower_bound), log10(upper_bound)]
lower_bounds = priors_df['lower_bound'].values
upper_bounds = priors_df['upper_bound'].values
distributions = priors_df['distribution'].values

# Verify all distributions are lognormal (expected for this model)
non_lognormal = [n for n, d in zip(param_names, distributions) if d != 'lognormal']
if non_lognormal:
    print(f"   WARNING: Non-lognormal parameters found: {non_lognormal}")
    print(f"   These will still be sampled in log10-space (assuming positive-valued)")

# Build bounds in log10-space
log10_bounds = []
for i in range(n_params):
    lb = np.log10(lower_bounds[i])
    ub = np.log10(upper_bounds[i])
    log10_bounds.append([lb, ub])

problem = {
    'num_vars': n_params,
    'names': param_names,
    'bounds': log10_bounds,
}
print(f"   SALib problem: {n_params} variables, bounds in log10-space")

# ============================================================================
# 2. Generate Morris DOE matrix (with optional classifier pre-filtering)
# ============================================================================

if RESTRICTION_ESTIMATOR_DIR is not None:
    re_dir = Path(RESTRICTION_ESTIMATOR_DIR)
    n_generate = N_TRAJECTORIES * OVERSAMPLE_FACTOR
    print(f"\n2. Generating Morris DOE with classifier pre-filtering...")
    print(f"   Oversampling: {n_generate} trajectories (x{OVERSAMPLE_FACTOR}), target {N_TRAJECTORIES}")

    # Load restriction estimator artifacts
    print(f"   Loading classifier from {re_dir}/...")
    with open(re_dir / 'restricted_prior.pkl', 'rb') as f:
        restricted_prior = pickle.load(f)
    with open(re_dir / 'metadata.json', 'r') as f:
        re_metadata = json.load(f)
    print(f"   Restriction estimator: NaN rate {re_metadata['nan_rate_unrestricted']:.1%} -> "
          f"{re_metadata['nan_rate_restricted']:.1%}")

    # Extract the accept/reject function from the RestrictedPrior
    accept_reject_fn = restricted_prior._accept_reject_fn

    # Generate oversampled DOE
    X_log10_full = morris_sampler.sample(problem, N=n_generate, num_levels=NUM_LEVELS, seed=SEED)
    traj_size = n_params + 1
    n_generated_trajs = X_log10_full.shape[0] // traj_size
    print(f"   Generated {n_generated_trajs} trajectories ({X_log10_full.shape[0]} points)")

    # Predict validity for all DOE points using the classifier
    theta_full = 10.0 ** X_log10_full  # original space
    theta_tensor = torch.tensor(theta_full, dtype=torch.float32)

    with torch.no_grad():
        predictions = accept_reject_fn(theta_tensor)  # True = valid

    # Filter: keep only trajectories where ALL points are predicted valid
    kept_trajs = []
    for t in range(n_generated_trajs):
        start = t * traj_size
        end = start + traj_size
        if predictions[start:end].all():
            kept_trajs.append(t)

    print(f"   Classifier prediction: {predictions.sum().item()}/{len(predictions)} points valid")
    print(f"   Complete valid trajectories: {len(kept_trajs)}/{n_generated_trajs}")

    if len(kept_trajs) < N_TRAJECTORIES:
        print(f"   WARNING: Only {len(kept_trajs)} valid trajectories (need {N_TRAJECTORIES})")
        print(f"   Using all {len(kept_trajs)} valid trajectories. Consider increasing OVERSAMPLE_FACTOR.")
        selected_trajs = kept_trajs
    else:
        selected_trajs = kept_trajs[:N_TRAJECTORIES]
        print(f"   Selected first {N_TRAJECTORIES} valid trajectories")

    # Rebuild DOE from selected trajectories
    selected_rows = []
    for t in selected_trajs:
        start = t * traj_size
        end = start + traj_size
        selected_rows.extend(range(start, end))

    X_log10 = X_log10_full[selected_rows]
    theta = theta_full[selected_rows]
    n_sims = X_log10.shape[0]
    print(f"   Pre-filtered DOE: {n_sims} simulations ({len(selected_trajs)} trajectories)")

else:
    # Generate the full DOE matrix (N_TRAJECTORIES_MAX) and cache it.
    # On subsequent runs, load from cache so the first k trajectories are always
    # the same — this enables progressive simulation (2 → 5 → 20 → 50).
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    doe_file = CACHE_DIR / 'doe_matrix.npy'
    doe_meta_file = CACHE_DIR / 'doe_metadata.json'

    if doe_file.exists():
        X_log10_full = np.load(doe_file)
        with open(doe_meta_file, 'r') as f:
            doe_meta = json.load(f)
        n_max_cached = doe_meta['n_trajectories_max']
        print(f"\n2. Loaded cached DOE matrix ({n_max_cached} trajectories from {doe_file})")

        if N_TRAJECTORIES_MAX > n_max_cached:
            print(f"   WARNING: N_TRAJECTORIES_MAX={N_TRAJECTORIES_MAX} > cached {n_max_cached}")
            print(f"   Regenerating DOE with {N_TRAJECTORIES_MAX} trajectories...")
            X_log10_full = morris_sampler.sample(
                problem, N=N_TRAJECTORIES_MAX, num_levels=NUM_LEVELS, seed=SEED
            )
            np.save(doe_file, X_log10_full)
            doe_meta = {'n_trajectories_max': N_TRAJECTORIES_MAX, 'num_levels': NUM_LEVELS, 'seed': SEED}
            with open(doe_meta_file, 'w') as f:
                json.dump(doe_meta, f)
    else:
        print(f"\n2. Generating Morris DOE (r_max={N_TRAJECTORIES_MAX}, p={NUM_LEVELS})...")
        X_log10_full = morris_sampler.sample(
            problem, N=N_TRAJECTORIES_MAX, num_levels=NUM_LEVELS, seed=SEED
        )
        np.save(doe_file, X_log10_full)
        doe_meta = {'n_trajectories_max': N_TRAJECTORIES_MAX, 'num_levels': NUM_LEVELS, 'seed': SEED}
        with open(doe_meta_file, 'w') as f:
            json.dump(doe_meta, f)
        print(f"   Saved DOE matrix to {doe_file}")

    # Take the first N_TRAJECTORIES trajectories for this run
    traj_size = n_params + 1
    n_rows = N_TRAJECTORIES * traj_size
    X_log10 = X_log10_full[:n_rows]
    n_sims = X_log10.shape[0]
    print(f"   Using first {N_TRAJECTORIES}/{N_TRAJECTORIES_MAX} trajectories ({n_sims} sims)")

    # Transform from log10-space to original parameter space
    theta = 10.0 ** X_log10

print(f"   Theta range [{theta.min():.2e}, {theta.max():.2e}]")

# ============================================================================
# 3. Run simulations via QSPSimulator (with trajectory-level caching)
# ============================================================================
print(f"\n3. Setting up QSPSimulator...")
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

# Check cache for previously simulated trajectories
results_file = CACHE_DIR / 'results.npz'
traj_size = n_params + 1

if results_file.exists():
    cached = np.load(results_file)
    Y_cached = cached['Y']
    n_cached_rows = Y_cached.shape[0]
    n_cached_trajs = n_cached_rows // traj_size
    print(f"   Found cached results: {n_cached_trajs} trajectories ({n_cached_rows} sims)")
else:
    Y_cached = None
    n_cached_trajs = 0
    print(f"   No cached results found")

if N_TRAJECTORIES <= n_cached_trajs:
    # All requested trajectories already simulated
    Y = Y_cached[:n_rows]
    print(f"   All {N_TRAJECTORIES} trajectories already cached — no new simulations needed")
else:
    # Simulate only the new trajectories
    n_new_start = n_cached_trajs * traj_size
    theta_new = theta[n_new_start:]
    n_new_sims = theta_new.shape[0]
    n_new_trajs = N_TRAJECTORIES - n_cached_trajs
    print(f"   Simulating {n_new_trajs} new trajectories ({n_new_sims} sims)...")

    Y_new = sim.simulate_with_parameters(theta_new, pool_suffix='morris_doe')

    # Merge with cached results
    if Y_cached is not None:
        Y = np.vstack([Y_cached, Y_new])
    else:
        Y = Y_new

    # Save merged results
    np.savez(results_file, Y=Y)
    print(f"   Saved {N_TRAJECTORIES} trajectories to cache ({Y.shape[0]} sims)")

print(f"   Output matrix shape: {Y.shape}")

# Get target names from the simulator's internal test statistics
target_names = sim._test_stats_df['test_statistic_id'].tolist()
print(f"   {len(target_names)} calibration targets")

# ============================================================================
# 4. Per-target NaN filtering
# ============================================================================
print(f"\n4. Per-target NaN filtering...")

# Instead of dropping entire trajectories when ANY target has NaN, we filter
# independently per target. Each target gets its own set of clean trajectories.
# This avoids the combinatorial NaN amplification where a 0.4% per-sim NaN rate
# causes ~50% trajectory loss with 176 sims/trajectory.

traj_size = n_params + 1
n_trajectories_total = n_sims // traj_size
MIN_TRAJECTORIES = 5  # Minimum for reliable Morris analysis

# For each target, find which trajectories are fully clean
target_clean_trajs = {}  # target_name -> list of clean trajectory indices
for j, target_name in enumerate(target_names):
    col = Y[:, j]
    clean = []
    for t in range(n_trajectories_total):
        start = t * traj_size
        end = start + traj_size
        if not np.any(np.isnan(col[start:end])):
            clean.append(t)
    target_clean_trajs[target_name] = clean

# Report per-target NaN rates
print(f"   {'Target':<45s} {'Clean trajs':>12s} {'NaN rate':>10s}")
print(f"   {'-'*45} {'-'*12} {'-'*10}")
kept_target_names = []
skipped_targets = []
for j, target_name in enumerate(target_names):
    n_clean = len(target_clean_trajs[target_name])
    nan_rate = 1.0 - n_clean / n_trajectories_total if n_trajectories_total > 0 else 1.0
    status = ""
    if n_clean < MIN_TRAJECTORIES:
        status = "  SKIPPED"
        skipped_targets.append((target_name, n_clean))
    else:
        kept_target_names.append(target_name)
    print(f"   {target_name:<45s} {n_clean:>5d}/{n_trajectories_total:<5d} {nan_rate:>9.1%}{status}")

print(f"\n   Targets with enough data: {len(kept_target_names)}/{len(target_names)}")
if skipped_targets:
    print(f"   Skipped ({len(skipped_targets)} targets, <{MIN_TRAJECTORIES} clean trajectories):")
    for name, n in skipped_targets:
        print(f"      - {name} ({n} clean trajectories)")

if len(kept_target_names) == 0:
    print("\n   ERROR: No targets have enough clean trajectories for Morris analysis.")
    print("   Consider increasing N_TRAJECTORIES or checking model stability.")
    sys.exit(1)

# ============================================================================
# 5. Morris analysis per output (each target uses its own clean trajectories)
# ============================================================================
print(f"\n5. Running Morris analysis for {len(kept_target_names)} targets...")

results = {}
target_n_trajs = {}  # track effective trajectory count per target
for target_name in kept_target_names:
    j = target_names.index(target_name)
    clean_trajs = target_clean_trajs[target_name]

    # Rebuild X and Y using only this target's clean trajectories
    good_rows = []
    for t in clean_trajs:
        start = t * traj_size
        end = start + traj_size
        good_rows.extend(range(start, end))

    X_target = X_log10[good_rows]
    Y_target = Y[good_rows, j]

    Si = morris_analyzer.analyze(
        problem,
        X_target,
        Y_target,
        conf_level=0.95,
        num_levels=NUM_LEVELS,
    )
    results[target_name] = Si
    target_n_trajs[target_name] = len(clean_trajs)

print(f"   Done. Analyzed {len(results)} targets.")
traj_counts = list(target_n_trajs.values())
print(f"   Effective trajectories: min={min(traj_counts)}, max={max(traj_counts)}, "
      f"median={np.median(traj_counts):.0f}")

# ============================================================================
# 6. Export tabular results
# ============================================================================
print(f"\n6. Exporting results to {RESULTS_DIR}/...")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Per-target ranking CSVs
for target_name, Si in results.items():
    df = pd.DataFrame({
        'parameter': param_names,
        'mu_star': Si['mu_star'],
        'mu_star_conf': Si['mu_star_conf'],
        'sigma': Si['sigma'],
        'mu': Si['mu'],
    })
    df = df.sort_values('mu_star', ascending=False)
    safe_name = target_name.replace('/', '_').replace(' ', '_')
    # Include effective trajectory count in the filename for traceability
    n_eff = target_n_trajs[target_name]
    df.attrs['n_trajectories'] = n_eff
    df.to_csv(RESULTS_DIR / f'morris_{safe_name}.csv', index=False)

# Summary heatmap data: (params x targets) of mu_star
mu_star_matrix = np.zeros((n_params, len(kept_target_names)))
for j, target_name in enumerate(kept_target_names):
    mu_star_matrix[:, j] = results[target_name]['mu_star']

mu_star_df = pd.DataFrame(
    mu_star_matrix,
    index=param_names,
    columns=kept_target_names,
)
mu_star_df.to_csv(RESULTS_DIR / 'mu_star_matrix.csv')

# Aggregate ranking: mean mu_star across all targets
mean_mu_star = mu_star_matrix.mean(axis=1)
aggregate_df = pd.DataFrame({
    'parameter': param_names,
    'mean_mu_star': mean_mu_star,
    'max_mu_star': mu_star_matrix.max(axis=1),
    'n_targets_top20': [(mu_star_matrix[i] >= np.sort(mu_star_matrix[:, j])[-min(TOP_N_PARAMS, n_params)]
                          ).sum() if False else 0 for i, _ in enumerate(param_names)],
})

# Compute n_targets_top20 properly: for each param, count how many targets it's in the top 20
n_targets_influential = np.zeros(n_params, dtype=int)
for j in range(len(kept_target_names)):
    col = mu_star_matrix[:, j]
    threshold = np.sort(col)[-min(TOP_N_PARAMS, n_params)]
    n_targets_influential += (col >= threshold).astype(int)

aggregate_df['n_targets_top20'] = n_targets_influential
aggregate_df = aggregate_df.sort_values('mean_mu_star', ascending=False)
aggregate_df.to_csv(RESULTS_DIR / 'aggregate_parameter_ranking.csv', index=False)

print(f"   Saved {len(results)} per-target CSVs")
print(f"   Saved mu_star_matrix.csv ({mu_star_matrix.shape})")
print(f"   Saved aggregate_parameter_ranking.csv")

# Print top 20 overall
print(f"\n   Top {TOP_N_PARAMS} parameters by mean mu_star across all targets:")
for i, row in aggregate_df.head(TOP_N_PARAMS).iterrows():
    print(f"      {row['parameter']:40s}  mean_mu*={row['mean_mu_star']:.4e}  "
          f"max_mu*={row['max_mu_star']:.4e}  influential_for={row['n_targets_top20']} targets")

# ============================================================================
# 7. Visualizations
# ============================================================================
print(f"\n7. Generating figures in {FIG_DIR}/...")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- 7a. Per-output horizontal bar plots (top-N params by mu_star) ---
print("   7a. Per-output bar plots...")
for target_name, Si in results.items():
    mu_star = Si['mu_star']
    mu_star_conf = Si['mu_star_conf']

    # Sort by mu_star descending, take top N
    order = np.argsort(mu_star)[::-1][:TOP_N_PARAMS]

    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(order))
    bars = ax.barh(
        y_pos,
        mu_star[order],
        xerr=mu_star_conf[order],
        align='center',
        color='steelblue',
        edgecolor='white',
        capsize=3,
    )
    ax.set_yticks(y_pos)
    ax.set_yticklabels([param_names[i] for i in order], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel('$\\mu^*$ (Mean Absolute Elementary Effect)')
    n_eff = target_n_trajs[target_name]
    ax.set_title(f'Morris Screening: {target_name} (r={n_eff})', fontsize=11)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    safe_name = target_name.replace('/', '_').replace(' ', '_')
    fig.savefig(FIG_DIR / f'bar_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- 7b. Morris scatter plots (mu_star vs sigma) ---
print("   7b. Morris scatter plots (mu* vs sigma)...")
for target_name, Si in results.items():
    mu_star = Si['mu_star']
    sigma = Si['sigma']

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(mu_star, sigma, s=20, alpha=0.6, c='steelblue', edgecolors='white', linewidth=0.5)

    # Label top-10 by mu_star
    top10 = np.argsort(mu_star)[::-1][:10]
    for idx in top10:
        ax.annotate(
            param_names[idx],
            (mu_star[idx], sigma[idx]),
            fontsize=6,
            alpha=0.8,
            textcoords='offset points',
            xytext=(5, 3),
        )

    # Diagonal line: sigma = mu_star (interaction boundary)
    max_val = max(mu_star.max(), sigma.max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.3, label='$\\sigma = \\mu^*$')
    ax.set_xlabel('$\\mu^*$ (Mean Absolute Elementary Effect)')
    ax.set_ylabel('$\\sigma$ (Std of Elementary Effects)')
    n_eff = target_n_trajs[target_name]
    ax.set_title(f'Morris Plot: {target_name} (r={n_eff})', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    safe_name = target_name.replace('/', '_').replace(' ', '_')
    fig.savefig(FIG_DIR / f'scatter_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- 7c. Summary heatmap (params x outputs colored by mu_star) ---
print("   7c. Summary heatmap...")

# Filter to params that are in top-20 for at least one output
influential_mask = n_targets_influential > 0
influential_params = [p for p, m in zip(param_names, influential_mask) if m]
influential_indices = [i for i, m in enumerate(influential_mask) if m]

if len(influential_params) > 0:
    heatmap_data = mu_star_matrix[influential_indices, :]

    # Normalize each column (target) to [0, 1] for better visual comparison
    col_max = heatmap_data.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1  # avoid division by zero
    heatmap_norm = heatmap_data / col_max

    # Truncate target names for readability
    short_targets = [t[:30] for t in kept_target_names]

    fig, ax = plt.subplots(figsize=(max(14, len(kept_target_names) * 0.4),
                                     max(8, len(influential_params) * 0.3)))
    im = ax.imshow(heatmap_norm, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xticks(range(len(short_targets)))
    ax.set_xticklabels(short_targets, rotation=90, fontsize=6, ha='right')
    ax.set_yticks(range(len(influential_params)))
    ax.set_yticklabels(influential_params, fontsize=7)
    ax.set_title('Morris $\\mu^*$ Heatmap (column-normalized)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Normalized $\\mu^*$', shrink=0.8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'heatmap_mu_star.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"      {len(influential_params)} influential params shown in heatmap")
else:
    print("      No influential parameters found (all mu_star ~ 0)")

# --- 7d. Aggregate bar plot: top parameters across all targets ---
print("   7d. Aggregate ranking bar plot...")
top_agg = aggregate_df.head(TOP_N_PARAMS)

fig, ax = plt.subplots(figsize=(10, 8))
y_pos = np.arange(len(top_agg))
ax.barh(y_pos, top_agg['mean_mu_star'].values, color='steelblue', edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_agg['parameter'].values, fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('Mean $\\mu^*$ across all calibration targets')
ax.set_title(f'Top {TOP_N_PARAMS} Most Influential Parameters (Aggregated)', fontsize=12)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / 'aggregate_ranking.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 80)
print("MORRIS SENSITIVITY ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n  Parameters:          {n_params}")
if RESTRICTION_ESTIMATOR_DIR is not None:
    print(f"  Pre-filtering:       classifier from {RESTRICTION_ESTIMATOR_DIR}")
print(f"  Trajectories:        {N_TRAJECTORIES} generated, "
      f"{min(traj_counts)}-{max(traj_counts)} effective (per-target NaN filtering)")
print(f"  Total simulations:   {n_sims}")
print(f"  Targets analyzed:    {len(kept_target_names)}/{len(target_names)}")
if skipped_targets:
    print(f"  Targets skipped:     {len(skipped_targets)} (<{MIN_TRAJECTORIES} clean trajectories)")
print(f"  Figures:             {FIG_DIR}/")
print(f"  Results:             {RESULTS_DIR}/")
print(f"\n  Key files:")
print(f"    aggregate_parameter_ranking.csv  - Overall parameter importance")
print(f"    mu_star_matrix.csv               - Full (params x targets) matrix")
print(f"    heatmap_mu_star.png              - Visual overview")
print(f"    aggregate_ranking.png            - Top parameters bar chart")
