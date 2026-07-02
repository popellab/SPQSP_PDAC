#!/usr/bin/env python3
"""
Surrogate-Based Sensitivity Analysis for PDAC QSP Model

Screens 175 parameters using two complementary methods:
  1. Spearman correlation + PRCC (Partial Rank Correlation Coefficients)
  2. Random Forest permutation importance

Unlike Morris SA, these methods operate on independent samples — NaN rows are
simply dropped per-target, avoiding the combinatorial NaN amplification that
made Morris impractical (50% trajectory loss). Reuses ~2,000 prior-sampled
simulations already cached in the simulation pool from the restriction
estimator run.

References:
  - Marino et al. (2008) "A methodology for performing global uncertainty and
    sensitivity analysis in systems biology" J Theor Biol 254(1):178-196
  - Breiman (2001) Random Forests. Machine Learning 45:5-32

Usage:
    python workflows/surrogate_sensitivity.py

Output:
    results/surrogate_sensitivity/  - CSV rankings, PRCC/RF matrices
    figures/surrogate_sensitivity/  - Bar plots, heatmaps, scatter plots
"""

import json
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from qsp_hpc import QSPSimulator

# ============================================================================
# CONFIGURATION
# ============================================================================

PRIORS_CSV = 'parameters/pdac_priors.csv'

CALIBRATION_TARGETS_DIR = 'calibration_targets/baseline_no_treatment'
MODEL_STRUCTURE_FILE = 'model_structure.json'
SCENARIO = 'baseline_no_treatment'
MODEL_SCRIPT = 'immune_oncology_model_PDAC'

# Reuse the restriction estimator's cached simulation pool.
# This must match the MODEL_VERSION in restriction_estimator.py.
MODEL_VERSION = 'baseline_allparams_v2'
MODEL_DESCRIPTION = 'Surrogate sensitivity: reuse restriction estimator simulations'

N_SIMULATIONS = 5e3
OVERSAMPLE_FACTOR = 2         # Generate 2x sims to compensate for ~43% NaN rate
MIN_SAMPLES_PER_TARGET = 10   # Skip targets with fewer valid samples
MAX_TASKS = 40
SEED = 2025

# Visualization / export
TOP_N_PARAMS = 30              # Show top-N in bar plots
TOP_N_AGGREGATE = 40           # Show top-N in aggregate ranking
RF_N_ESTIMATORS = 100
RF_PERM_REPEATS = 5
PRCC_ALPHA = 0.05              # Significance threshold for PRCC

FIG_DIR = Path('figures/surrogate_sensitivity')
RESULTS_DIR = Path('results/surrogate_sensitivity')

# ============================================================================
# 1. Load priors and set up simulator
# ============================================================================
print("=" * 80)
print("SURROGATE-BASED SENSITIVITY ANALYSIS")
print("=" * 80)

print(f"\n1. Loading priors from {PRIORS_CSV}...")
priors_df = pd.read_csv(PRIORS_CSV)
param_names = priors_df['name'].tolist()
n_params = len(param_names)
print(f"   {n_params} parameters loaded")

print(f"\n   Setting up QSPSimulator (scenario: {SCENARIO})...")
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
# 2. Get simulations (from cached pool), with oversampling to handle NaN rows
# ============================================================================
n_request = int(N_SIMULATIONS * OVERSAMPLE_FACTOR)
print(f"\n2. Loading {n_request} simulations ({N_SIMULATIONS} x {OVERSAMPLE_FACTOR} oversample)...")
t0 = time.time()
theta_raw, x_raw = sim(n_request)
t_load = time.time() - t0
print(f"   Loaded in {t_load:.1f}s")
print(f"   Raw: theta {theta_raw.shape}, x {x_raw.shape}")

target_names = sim._test_stats_df['test_statistic_id'].tolist()
n_targets = len(target_names)
print(f"   {n_targets} calibration targets")

# Replace inf with NaN before filtering
n_inf = np.isinf(x_raw).sum()
if n_inf > 0:
    print(f"   Replacing {n_inf} inf values with NaN")
    x_raw[np.isinf(x_raw)] = np.nan

# Drop rows where ALL targets are NaN (completely failed simulations)
all_nan_mask = np.isnan(x_raw).all(axis=1)
any_nan_mask = np.isnan(x_raw).any(axis=1)
print(f"   All-NaN rows (failed sims): {all_nan_mask.sum()}/{len(all_nan_mask)} "
      f"({all_nan_mask.mean():.1%})")
print(f"   Any-NaN rows:               {any_nan_mask.sum()}/{len(any_nan_mask)} "
      f"({any_nan_mask.mean():.1%})")

# Keep rows that have at least one valid target
valid_rows = ~all_nan_mask
theta = theta_raw[valid_rows]
x = x_raw[valid_rows]
print(f"   Valid rows after filtering: {len(theta)}")

# --- Sanity check: verify simulations are biologically plausible ---
# Catches initialization bugs (e.g., evolve_to_diagnosis not reaching target diameter)
SANITY_CHECKS = {
    'tumor_volume_resection': (0.5, 500.0, 'cm³'),    # expect ~12 cm³ median
    'cancer_cell_density_baseline': (10, 50000, 'cell/mm²'),  # expect ~1000 median
}
for check_name, (lo, hi, unit_label) in SANITY_CHECKS.items():
    if check_name in target_names:
        j = target_names.index(check_name)
        vals = x[:, j]
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

# Log10-transform all parameters (all are lognormal)
theta_log = np.log10(theta)

# ============================================================================
# 3. Per-target NaN filtering
# ============================================================================
print(f"\n3. Per-target NaN filtering (min {MIN_SAMPLES_PER_TARGET} valid samples)...")

overall_nan_rate = np.isnan(x).any(axis=1).mean()
print(f"   Overall row NaN rate: {overall_nan_rate:.1%}")

kept_targets = []
skipped_targets = []
target_valid_counts = {}

print(f"\n   {'Target':<50s} {'Valid':>6s} {'NaN%':>6s}")
print(f"   {'-'*50} {'-'*6} {'-'*6}")
for j, name in enumerate(target_names):
    valid = ~np.isnan(x[:, j])
    n_valid = valid.sum()
    nan_pct = 1.0 - n_valid / len(valid)
    target_valid_counts[name] = n_valid
    if n_valid >= MIN_SAMPLES_PER_TARGET:
        kept_targets.append(name)
        status = ""
    else:
        skipped_targets.append((name, n_valid))
        status = " SKIP"
    print(f"   {name:<50s} {n_valid:>6d} {nan_pct:>5.1%}{status}")

print(f"\n   Targets with enough data: {len(kept_targets)}/{n_targets}")
if skipped_targets:
    print(f"   Skipped: {[s[0] for s in skipped_targets]}")

if len(kept_targets) == 0:
    print("\n   ERROR: No targets have enough valid samples. Aborting.")
    raise SystemExit(1)

# ============================================================================
# 4. Tier 1 — Spearman correlation + PRCC
# ============================================================================
print(f"\n4. Computing Spearman correlations and PRCC...")
t0 = time.time()

spearman_matrix = np.full((n_params, len(kept_targets)), np.nan)
prcc_matrix = np.full((n_params, len(kept_targets)), np.nan)
prcc_pvalue_matrix = np.full((n_params, len(kept_targets)), np.nan)


def _compute_prcc_and_spearman(tj, target_name):
    """
    Compute Spearman correlations and PRCC for a single target.

    PRCC via rank correlation matrix inversion (Marino et al. 2008):
    Given the (p+1)x(p+1) rank correlation matrix R of [X, y],
    PRCC(k, y) = -P[k,p] / sqrt(P[k,k] * P[p,p])  where P = R^{-1}.
    Replaces O(p) lstsq regressions with one matrix inversion.
    """
    j = target_names.index(target_name)
    valid = ~np.isnan(x[:, j])
    X_valid = theta_log[valid]
    y_valid = x[valid, j]
    n_samples, n_vars = X_valid.shape

    # Rank-transform
    X_ranked = np.apply_along_axis(stats.rankdata, 0, X_valid)
    y_ranked = stats.rankdata(y_valid)

    # --- Spearman (vectorized) ---
    data = np.column_stack([X_ranked, y_ranked])
    R = np.corrcoef(data, rowvar=False)  # (p+1) x (p+1)
    spearman_vals = R[:-1, -1]  # correlations of each param with y

    # --- PRCC via matrix inversion ---
    try:
        P = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        P = np.linalg.pinv(R)

    diag = P.diagonal()
    denom = np.sqrt(diag[:-1] * diag[-1])
    denom = np.where(denom == 0, np.nan, denom)
    prcc_vals = -P[:-1, -1] / denom
    prcc_vals = np.clip(prcc_vals, -1, 1)

    # p-values: t-test with df = n - p (Marino et al. 2008, eq. 8)
    df = n_samples - n_vars
    t_stat = prcc_vals * np.sqrt(df) / np.sqrt(1 - prcc_vals**2 + 1e-15)
    pvals = 2 * stats.t.sf(np.abs(t_stat), max(df, 1))

    return tj, target_name, spearman_vals, prcc_vals, pvals, int(valid.sum())


results = Parallel(n_jobs=-1)(
    delayed(_compute_prcc_and_spearman)(tj, name)
    for tj, name in enumerate(kept_targets)
)
for tj, target_name, spearman_vals, prcc_vals, pvals, n_valid in results:
    spearman_matrix[:, tj] = spearman_vals
    prcc_matrix[:, tj] = prcc_vals
    prcc_pvalue_matrix[:, tj] = pvals
    n_sig = (pvals < PRCC_ALPHA).sum()
    print(f"   {target_name}: {n_sig}/{n_params} significant PRCCs "
          f"(n={n_valid})")

t_prcc = time.time() - t0
print(f"\n   Tier 1 completed in {t_prcc:.1f}s")

# ============================================================================
# 5. Tier 2 — Random Forest permutation importance
# ============================================================================
print(f"\n5. Computing Random Forest permutation importance...")
t0 = time.time()

rf_importance_matrix = np.full((n_params, len(kept_targets)), np.nan)
rf_r2_scores = {}


def _fit_one_target(tj, target_name):
    """Fit RF and compute permutation importance for a single target."""
    j = target_names.index(target_name)
    valid = ~np.isnan(x[:, j])
    X_valid = theta_log[valid]
    y_valid = x[valid, j]

    # Log-transform positive targets for better RF fitting
    y_fit = np.log10(y_valid) if np.all(y_valid > 0) else y_valid

    rf = RandomForestRegressor(
        n_estimators=RF_N_ESTIMATORS,
        max_features='sqrt',
        random_state=SEED,
        n_jobs=1,
    )
    rf.fit(X_valid, y_fit)
    r2 = rf.score(X_valid, y_fit)

    # Permutation importance (more reliable than impurity-based for correlated features)
    perm_result = permutation_importance(
        rf, X_valid, y_fit,
        n_repeats=RF_PERM_REPEATS,
        random_state=SEED,
        n_jobs=1,
    )
    return tj, target_name, r2, perm_result.importances_mean


results = Parallel(n_jobs=-1, verbose=10)(
    delayed(_fit_one_target)(tj, name) for tj, name in enumerate(kept_targets)
)
for tj, target_name, r2, imp in results:
    rf_importance_matrix[:, tj] = imp
    rf_r2_scores[target_name] = r2
    print(f"   {target_name}: R²={r2:.3f}, "
          f"top param={param_names[np.argmax(imp)]}")

t_rf = time.time() - t0
print(f"\n   Tier 2 completed in {t_rf:.1f}s")

# ============================================================================
# 6. Aggregate rankings
# ============================================================================
print(f"\n6. Aggregating rankings...")

# Primary: mean |PRCC| across targets (QSP standard metric)
abs_prcc = np.abs(prcc_matrix)
mean_abs_prcc = np.nanmean(abs_prcc, axis=1)
max_abs_prcc = np.nanmax(abs_prcc, axis=1)

# Count significant PRCCs per parameter
n_significant = np.sum(prcc_pvalue_matrix < PRCC_ALPHA, axis=1)

# Secondary: mean RF permutation importance across targets
mean_rf_importance = np.nanmean(rf_importance_matrix, axis=1)
max_rf_importance = np.nanmax(rf_importance_matrix, axis=1)

# Build aggregate DataFrame
aggregate_df = pd.DataFrame({
    'parameter': param_names,
    'mean_abs_prcc': mean_abs_prcc,
    'max_abs_prcc': max_abs_prcc,
    'n_significant_prcc': n_significant,
    'mean_rf_importance': mean_rf_importance,
    'max_rf_importance': max_rf_importance,
})

# Rank by each method
aggregate_df['rank_prcc'] = aggregate_df['mean_abs_prcc'].rank(ascending=False).astype(int)
aggregate_df['rank_rf'] = aggregate_df['mean_rf_importance'].rank(ascending=False).astype(int)

# Cross-method agreement: Spearman rank correlation between methods
rank_agreement, rank_agreement_p = stats.spearmanr(
    aggregate_df['mean_abs_prcc'],
    aggregate_df['mean_rf_importance'],
)
print(f"   Cross-method rank agreement: rho={rank_agreement:.3f} (p={rank_agreement_p:.2e})")

# Identify robust parameters (top-N in BOTH methods)
for top_n in [20, 30, 40]:
    top_prcc = set(aggregate_df.nsmallest(top_n, 'rank_prcc')['parameter'])
    top_rf = set(aggregate_df.nsmallest(top_n, 'rank_rf')['parameter'])
    overlap = top_prcc & top_rf
    print(f"   Top-{top_n} overlap: {len(overlap)} params in both methods")

# Sort by mean |PRCC| (primary metric)
aggregate_df = aggregate_df.sort_values('mean_abs_prcc', ascending=False)

# Print top parameters
print(f"\n   Top {TOP_N_AGGREGATE} parameters by mean |PRCC|:")
print(f"   {'Param':<40s} {'|PRCC|':>8s} {'RF imp':>10s} {'#sig':>5s} {'Rank_P':>7s} {'Rank_RF':>7s}")
print(f"   {'-'*40} {'-'*8} {'-'*10} {'-'*5} {'-'*7} {'-'*7}")
for _, row in aggregate_df.head(TOP_N_AGGREGATE).iterrows():
    print(f"   {row['parameter']:<40s} {row['mean_abs_prcc']:>8.4f} "
          f"{row['mean_rf_importance']:>10.4f} {row['n_significant_prcc']:>5.0f} "
          f"{row['rank_prcc']:>7d} {row['rank_rf']:>7d}")

# ============================================================================
# 6b. Data-driven cutoff analysis
# ============================================================================
print("\n6b. Data-driven cutoff analysis...")

# --- Cumulative importance (scree-style) ---
# Fraction of total signal captured by the top-K parameters
sorted_prcc = np.sort(mean_abs_prcc)[::-1]
sorted_rf = np.sort(np.clip(mean_rf_importance, 0, None))[::-1]
cum_prcc = np.cumsum(sorted_prcc) / sorted_prcc.sum()
cum_rf = np.cumsum(sorted_rf) / sorted_rf.sum()

for frac in [0.80, 0.90, 0.95]:
    k_prcc = int(np.searchsorted(cum_prcc, frac)) + 1
    k_rf = int(np.searchsorted(cum_rf, frac)) + 1
    print(f"   {frac:.0%} cumulative importance: PRCC top-{k_prcc}, RF top-{k_rf}")

# --- Permutation null test for parameter significance ---
# For each parameter, test whether its mean |PRCC| across targets exceeds what
# you'd see by chance if that parameter had no relationship with any output.
# Shuffle each parameter's values independently, recompute mean |PRCC|, repeat
# to build a null distribution.  Apply Benjamini-Hochberg FDR correction.
PERM_REPEATS = 1000
FDR_ALPHA = 0.05

n_total = theta.shape[0]

print(f"\n   Permutation null test ({PERM_REPEATS} permutations, "
      f"FDR alpha={FDR_ALPHA})...")


def _compute_mean_abs_prcc(X_log_in, x_in):
    """Compute mean |PRCC| per parameter across all kept targets."""
    n_p = X_log_in.shape[1]
    abs_prcc_accum = np.zeros(n_p)
    n_counted = 0
    for tname in kept_targets:
        j = target_names.index(tname)
        valid = np.isfinite(x_in[:, j])
        if valid.sum() < MIN_SAMPLES_PER_TARGET:
            continue
        data = np.column_stack([
            np.apply_along_axis(stats.rankdata, 0, X_log_in[valid]),
            stats.rankdata(x_in[valid, j]),
        ])
        R = np.corrcoef(data, rowvar=False)
        try:
            P = np.linalg.inv(R)
        except np.linalg.LinAlgError:
            P = np.linalg.pinv(R)
        diag = P.diagonal()
        denom = np.sqrt(np.abs(diag[:-1] * diag[-1]))
        denom = np.where(denom == 0, np.nan, denom)
        prcc_vals = np.clip(np.abs(-P[:-1, -1] / denom), 0, 1)
        abs_prcc_accum += np.nan_to_num(prcc_vals)
        n_counted += 1
    return abs_prcc_accum / max(n_counted, 1)


# Observed mean |PRCC| (already computed above, but recompute for consistency)
observed_mean_abs_prcc = _compute_mean_abs_prcc(theta_log, x)

# Build null distribution: for each permutation, shuffle ONE parameter at a time
# is too expensive (177 * 100 full PRCC computations).  Instead, shuffle ALL
# sample indices (breaking theta-x correspondence) and compute null mean |PRCC|.
# This gives a single null distribution that all parameters are compared against.
null_mean_abs_prcc = np.zeros((PERM_REPEATS, n_params))

for perm in range(PERM_REPEATS):
    if (perm + 1) % 20 == 0:
        print(f"      Permutation {perm + 1}/{PERM_REPEATS}...")
    rng = np.random.default_rng(perm * 7919 + n_total)
    # Shuffle row order of x to break the theta-x relationship
    x_shuffled = x[rng.permutation(n_total)]
    null_mean_abs_prcc[perm] = _compute_mean_abs_prcc(theta_log, x_shuffled)

# p-value: fraction of null permutations where null >= observed
perm_pvalues = np.zeros(n_params)
for j in range(n_params):
    perm_pvalues[j] = (np.sum(null_mean_abs_prcc[:, j] >= observed_mean_abs_prcc[j]) + 1) / (PERM_REPEATS + 1)

# FDR correction (Benjamini-Hochberg)
sorted_idx = np.argsort(perm_pvalues)
pvalues_corrected = np.zeros(n_params)
m = n_params
cummin = 1.0
for i in range(m - 1, -1, -1):
    j = sorted_idx[i]
    corrected = perm_pvalues[j] * m / (i + 1)
    cummin = min(cummin, corrected)
    pvalues_corrected[j] = min(cummin, 1.0)
reject = pvalues_corrected < FDR_ALPHA

# Add to aggregate dataframe
aggregate_df['perm_pvalue'] = [perm_pvalues[param_names.index(p)]
                               for p in aggregate_df['parameter']]
aggregate_df['perm_pvalue_fdr'] = [pvalues_corrected[param_names.index(p)]
                                   for p in aggregate_df['parameter']]
aggregate_df['significant'] = [reject[param_names.index(p)]
                               for p in aggregate_df['parameter']]

# Recommended set: FDR-significant parameters
union_rec = sorted(aggregate_df[aggregate_df['significant']]['parameter'].tolist())
aggregate_df['in_recommended_set'] = aggregate_df['parameter'].isin(union_rec)

n_sig = reject.sum()
print(f"\n   {n_sig}/{n_params} parameters significant at FDR < {FDR_ALPHA}")

print(f"\n   {'Parameter':<40s} {'|PRCC|':>8s} {'p-val':>7s} {'FDR':>7s} {'Sig':>5s}")
print(f"   {'-'*40} {'-'*8} {'-'*7} {'-'*7} {'-'*5}")
for _, row in aggregate_df.head(50).iterrows():
    sig = "*" if row['significant'] else ""
    print(f"   {row['parameter']:<40s} {row['mean_abs_prcc']:>8.4f} "
          f"{row['perm_pvalue']:>7.3f} {row['perm_pvalue_fdr']:>7.3f} {sig:>5s}")

print(f"\n   Recommended set: {len(union_rec)} parameters (FDR < {FDR_ALPHA})")

# ============================================================================
# 7. Export CSVs
# ============================================================================
print(f"\n7. Exporting results to {RESULTS_DIR}/...")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Primary output: aggregate ranking
aggregate_df.to_csv(RESULTS_DIR / 'aggregate_parameter_ranking.csv', index=False)

# Full matrices
prcc_df = pd.DataFrame(prcc_matrix, index=param_names, columns=kept_targets)
prcc_df.to_csv(RESULTS_DIR / 'prcc_matrix.csv')

prcc_pval_df = pd.DataFrame(prcc_pvalue_matrix, index=param_names, columns=kept_targets)
prcc_pval_df.to_csv(RESULTS_DIR / 'prcc_pvalue_matrix.csv')

spearman_df = pd.DataFrame(spearman_matrix, index=param_names, columns=kept_targets)
spearman_df.to_csv(RESULTS_DIR / 'spearman_matrix.csv')

rf_imp_df = pd.DataFrame(rf_importance_matrix, index=param_names, columns=kept_targets)
rf_imp_df.to_csv(RESULTS_DIR / 'rf_importance_matrix.csv')

# Per-target PRCC ranking CSVs
per_target_dir = RESULTS_DIR / 'per_target'
per_target_dir.mkdir(exist_ok=True)
for tj, target_name in enumerate(kept_targets):
    target_df = pd.DataFrame({
        'parameter': param_names,
        'prcc': prcc_matrix[:, tj],
        'prcc_pvalue': prcc_pvalue_matrix[:, tj],
        'spearman': spearman_matrix[:, tj],
        'rf_importance': rf_importance_matrix[:, tj],
    })
    target_df['abs_prcc'] = np.abs(target_df['prcc'])
    target_df = target_df.sort_values('abs_prcc', ascending=False)
    safe_name = target_name.replace('/', '_').replace(' ', '_')
    target_df.to_csv(per_target_dir / f'prcc_{safe_name}.csv', index=False)

# Cross-method agreement JSON
top_20_prcc = set(aggregate_df.head(20)['parameter'])
top_20_rf = set(aggregate_df.nsmallest(20, 'rank_rf')['parameter'])
top_30_prcc = set(aggregate_df.head(30)['parameter'])
top_30_rf = set(aggregate_df.nsmallest(30, 'rank_rf')['parameter'])

agreement = {
    'rank_correlation': float(rank_agreement),
    'rank_correlation_pvalue': float(rank_agreement_p),
    'top_20_overlap': sorted(top_20_prcc & top_20_rf),
    'top_20_overlap_count': len(top_20_prcc & top_20_rf),
    'top_20_prcc_only': sorted(top_20_prcc - top_20_rf),
    'top_20_rf_only': sorted(top_20_rf - top_20_prcc),
    'top_30_overlap': sorted(top_30_prcc & top_30_rf),
    'top_30_overlap_count': len(top_30_prcc & top_30_rf),
    'robust_parameters': sorted(top_20_prcc & top_20_rf),
    'nonlinear_only_parameters': sorted(top_20_rf - top_20_prcc),
    'recommended_parameters': union_rec,
    'recommended_parameters_count': len(union_rec),
    'cumulative_90pct_prcc': int(np.searchsorted(cum_prcc, 0.90)) + 1,
    'cumulative_90pct_rf': int(np.searchsorted(cum_rf, 0.90)) + 1,
    'permutation_test': {
        'n_permutations': PERM_REPEATS,
        'fdr_alpha': FDR_ALPHA,
        'n_significant': int(n_sig),
        'significant_parameters': union_rec,
    },
    'rf_r2_per_target': {k: float(v) for k, v in rf_r2_scores.items()},
    'n_simulations': int(theta.shape[0]),
    'n_valid_per_target': {k: int(v) for k, v in target_valid_counts.items()
                           if k in kept_targets},
}

# Save recommended parameter set as a standalone CSV for downstream use
recommended_df = aggregate_df[aggregate_df['in_recommended_set']].copy()
recommended_df.to_csv(RESULTS_DIR / 'recommended_parameters.csv', index=False)
print(f"   Saved recommended_parameters.csv ({len(recommended_df)} params)")
with open(RESULTS_DIR / 'cross_method_agreement.json', 'w') as f:
    json.dump(agreement, f, indent=2)

print(f"   Saved aggregate_parameter_ranking.csv")
print(f"   Saved prcc_matrix.csv ({prcc_matrix.shape})")
print(f"   Saved prcc_pvalue_matrix.csv")
print(f"   Saved spearman_matrix.csv")
print(f"   Saved rf_importance_matrix.csv")
print(f"   Saved {len(kept_targets)} per-target CSVs in {per_target_dir}/")
print(f"   Saved cross_method_agreement.json")

# ============================================================================
# 8. Figures
# ============================================================================
print(f"\n8. Generating figures in {FIG_DIR}/...")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- 8a0. Cumulative importance + inclusion frequency ---
print("   8a0. Cumulative importance + inclusion frequency...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ks = np.arange(1, n_params + 1)
ax1.plot(ks, cum_prcc, 'steelblue', linewidth=2, label='PRCC')
ax1.plot(ks, cum_rf, 'darkorange', linewidth=2, label='RF')
for frac in [0.80, 0.90, 0.95]:
    ax1.axhline(frac, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    k_p = int(np.searchsorted(cum_prcc, frac)) + 1
    k_r = int(np.searchsorted(cum_rf, frac)) + 1
    ax1.annotate(f'{frac:.0%}: P={k_p}, RF={k_r}',
                 xy=(max(k_p, k_r), frac), fontsize=7, alpha=0.7,
                 xytext=(5, 3), textcoords='offset points')
ax1.set_xlabel('Number of parameters (ranked)')
ax1.set_ylabel('Cumulative fraction of total importance')
ax1.set_title('Cumulative Importance (Scree-Style)')
ax1.legend(fontsize=8)
ax1.set_xlim(1, min(100, n_params))
ax1.grid(alpha=0.3)

# Per-parameter significance (permutation null) bar chart
top_50 = aggregate_df.head(50)
fdr_vals = top_50['perm_pvalue_fdr'].values
names_50 = top_50['parameter'].values
y_pos = np.arange(len(names_50))
neg_log_fdr = -np.log10(np.clip(fdr_vals, 1e-10, 1))
colors = ['steelblue' if s else 'lightgray'
          for s in top_50['significant'].values]
ax2.barh(y_pos, neg_log_fdr, color=colors, edgecolor='white', height=0.7)
ax2.axvline(-np.log10(FDR_ALPHA), color='red', linestyle='--', alpha=0.6,
            label=f'FDR = {FDR_ALPHA}')
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names_50, fontsize=6)
ax2.invert_yaxis()
ax2.set_xlabel(f'-log10(FDR-corrected p-value)')
ax2.set_title(f'Permutation Significance ({PERM_REPEATS} perms)\n'
              f'Blue = significant (FDR < {FDR_ALPHA}), '
              f'{n_sig}/{n_params} params')
ax2.legend(fontsize=8)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / 'cutoff_analysis.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# --- 8a. Per-target PRCC bar plots (top-30 by |PRCC|) ---
print("   8a. Per-target PRCC bar plots...")
per_target_fig_dir = FIG_DIR / 'per_target'
per_target_fig_dir.mkdir(exist_ok=True)

for tj, target_name in enumerate(kept_targets):
    prcc_vals = prcc_matrix[:, tj]
    order = np.argsort(np.abs(prcc_vals))[::-1][:TOP_N_PARAMS]

    fig, ax = plt.subplots(figsize=(10, 9))
    y_pos = np.arange(len(order))
    colors = ['steelblue' if prcc_vals[i] >= 0 else 'indianred' for i in order]
    ax.barh(y_pos, prcc_vals[order], color=colors, edgecolor='white', height=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([param_names[i] for i in order], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel('PRCC')
    ax.axvline(0, color='black', linewidth=0.5)

    # Mark significance
    for yi, idx in enumerate(order):
        if prcc_pvalue_matrix[idx, tj] < PRCC_ALPHA:
            ax.text(prcc_vals[idx], yi, ' *', va='center', fontsize=8, fontweight='bold')

    n_valid = target_valid_counts[target_name]
    ax.set_title(f'PRCC: {target_name} (n={n_valid}, * p<{PRCC_ALPHA})', fontsize=10)
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()

    safe_name = target_name.replace('/', '_').replace(' ', '_')
    fig.savefig(per_target_fig_dir / f'prcc_{safe_name}.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

# --- 8b. PRCC heatmap (influential params x targets) ---
print("   8b. PRCC heatmap...")

# Select parameters that are significant for at least one target
sig_any = np.any(prcc_pvalue_matrix < PRCC_ALPHA, axis=1)
sig_indices = np.where(sig_any)[0]
# Sort by mean |PRCC| descending
sig_indices = sig_indices[np.argsort(mean_abs_prcc[sig_indices])[::-1]]

if len(sig_indices) > 0:
    # Cap at 60 params for readability
    sig_indices = sig_indices[:60]
    heatmap_data = prcc_matrix[sig_indices, :]
    heatmap_params = [param_names[i] for i in sig_indices]
    short_targets = [t[:35] for t in kept_targets]

    fig, ax = plt.subplots(figsize=(max(14, len(kept_targets) * 0.5),
                                     max(10, len(heatmap_params) * 0.25)))
    vmax = np.nanmax(np.abs(heatmap_data)) * 0.95
    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdBu_r',
                   vmin=-vmax, vmax=vmax, interpolation='nearest')
    ax.set_xticks(range(len(short_targets)))
    ax.set_xticklabels(short_targets, rotation=90, fontsize=6, ha='right')
    ax.set_yticks(range(len(heatmap_params)))
    ax.set_yticklabels(heatmap_params, fontsize=6)
    ax.set_title('PRCC Heatmap (params significant for >= 1 target)', fontsize=12)
    plt.colorbar(im, ax=ax, label='PRCC', shrink=0.8)
    plt.tight_layout()
    fig.savefig(FIG_DIR / 'prcc_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"      {len(heatmap_params)} params shown in heatmap")
else:
    print("      No significant PRCCs found — skipping heatmap")

# --- 8c. RF importance heatmap ---
print("   8c. RF importance heatmap...")

# Use same params as PRCC heatmap (or top-40 by RF if no sig PRCCs)
if len(sig_indices) > 0:
    rf_heatmap_indices = sig_indices
else:
    rf_heatmap_indices = np.argsort(mean_rf_importance)[::-1][:40]

rf_heatmap_data = rf_importance_matrix[rf_heatmap_indices, :]
rf_heatmap_params = [param_names[i] for i in rf_heatmap_indices]

fig, ax = plt.subplots(figsize=(max(14, len(kept_targets) * 0.5),
                                 max(10, len(rf_heatmap_params) * 0.25)))
im = ax.imshow(rf_heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
ax.set_xticks(range(len(kept_targets)))
ax.set_xticklabels([t[:35] for t in kept_targets], rotation=90, fontsize=6, ha='right')
ax.set_yticks(range(len(rf_heatmap_params)))
ax.set_yticklabels(rf_heatmap_params, fontsize=6)
ax.set_title('RF Permutation Importance Heatmap', fontsize=12)
plt.colorbar(im, ax=ax, label='Permutation Importance', shrink=0.8)
plt.tight_layout()
fig.savefig(FIG_DIR / 'rf_importance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# --- 8d. Aggregate ranking bar plot ---
print("   8d. Aggregate ranking bar plot...")
top_agg = aggregate_df.head(TOP_N_AGGREGATE)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))

# Left: mean |PRCC|
y_pos = np.arange(len(top_agg))
ax1.barh(y_pos, top_agg['mean_abs_prcc'].values, color='steelblue', edgecolor='white')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(top_agg['parameter'].values, fontsize=7)
ax1.invert_yaxis()
ax1.set_xlabel('Mean |PRCC| across targets')
ax1.set_title(f'Top {TOP_N_AGGREGATE} by Mean |PRCC|', fontsize=11)
ax1.grid(axis='x', alpha=0.3)

# Right: mean RF importance
# Re-sort by RF for this panel
top_rf = aggregate_df.nsmallest(TOP_N_AGGREGATE, 'rank_rf')
y_pos2 = np.arange(len(top_rf))
ax2.barh(y_pos2, top_rf['mean_rf_importance'].values, color='darkorange', edgecolor='white')
ax2.set_yticks(y_pos2)
ax2.set_yticklabels(top_rf['parameter'].values, fontsize=7)
ax2.invert_yaxis()
ax2.set_xlabel('Mean RF Permutation Importance')
ax2.set_title(f'Top {TOP_N_AGGREGATE} by RF Importance', fontsize=11)
ax2.grid(axis='x', alpha=0.3)

plt.tight_layout()
fig.savefig(FIG_DIR / 'aggregate_ranking.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# --- 8e. Cross-method agreement scatter ---
print("   8e. Cross-method agreement scatter...")

fig, ax = plt.subplots(figsize=(9, 9))
ax.scatter(
    aggregate_df['mean_abs_prcc'],
    aggregate_df['mean_rf_importance'],
    s=20, alpha=0.5, c='gray', edgecolors='white', linewidth=0.5,
)

# Highlight robust params (top-20 in both)
robust_set = top_20_prcc & top_20_rf
for _, row in aggregate_df.iterrows():
    if row['parameter'] in robust_set:
        ax.scatter(
            row['mean_abs_prcc'], row['mean_rf_importance'],
            s=60, c='steelblue', edgecolors='black', linewidth=0.8, zorder=5,
        )
        ax.annotate(
            row['parameter'], (row['mean_abs_prcc'], row['mean_rf_importance']),
            fontsize=5, alpha=0.9, textcoords='offset points', xytext=(4, 4),
        )

# Highlight nonlinear-only (top-20 RF, not top-20 PRCC)
nonlinear_set = top_20_rf - top_20_prcc
for _, row in aggregate_df.iterrows():
    if row['parameter'] in nonlinear_set:
        ax.scatter(
            row['mean_abs_prcc'], row['mean_rf_importance'],
            s=60, c='darkorange', edgecolors='black', linewidth=0.8, zorder=5,
            marker='D',
        )
        ax.annotate(
            row['parameter'], (row['mean_abs_prcc'], row['mean_rf_importance']),
            fontsize=5, alpha=0.9, textcoords='offset points', xytext=(4, 4),
        )

ax.set_xlabel('Mean |PRCC|')
ax.set_ylabel('Mean RF Permutation Importance')
ax.set_title(f'Cross-Method Agreement (rho={rank_agreement:.3f})\n'
             f'Blue = robust (top-20 both), Orange = nonlinear-only (top-20 RF only)',
             fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / 'cross_method_agreement.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# 9. Summary
# ============================================================================
print("\n" + "=" * 80)
print("SURROGATE SENSITIVITY ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n  Parameters:              {n_params}")
print(f"  Simulations:             {theta.shape[0]} (from cached pool)")
print(f"  Targets analyzed:        {len(kept_targets)}/{n_targets}")
if skipped_targets:
    print(f"  Targets skipped:         {len(skipped_targets)} (<{MIN_SAMPLES_PER_TARGET} valid)")
print(f"  Overall NaN rate:        {overall_nan_rate:.1%}")
print(f"  PRCC time:               {t_prcc:.1f}s")
print(f"  RF time:                 {t_rf:.1f}s")
print(f"  Rank agreement (rho):    {rank_agreement:.3f}")
print(f"  Top-20 overlap:          {len(top_20_prcc & top_20_rf)} params")
print(f"  Top-30 overlap:          {len(top_30_prcc & top_30_rf)} params")
print(f"  Figures:                 {FIG_DIR}/")
print(f"  Results:                 {RESULTS_DIR}/")
print(f"\n  Key files:")
print(f"    aggregate_parameter_ranking.csv  - Per-param summary (primary output)")
print(f"    prcc_matrix.csv                  - Signed PRCC (175 x {len(kept_targets)})")
print(f"    rf_importance_matrix.csv         - RF importance (175 x {len(kept_targets)})")
print(f"    cross_method_agreement.json      - Top-N overlap stats")
print(f"    cross_method_agreement.png       - PRCC vs RF scatter")
