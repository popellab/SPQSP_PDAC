#!/usr/bin/env python3
"""
PRCC Sensitivity Analysis for PDAC QSP Model

Screens parameters using Partial Rank Correlation Coefficients (PRCC) with
permutation-based significance testing and FDR correction. Lightweight
alternative to surrogate_sensitivity.py (no Random Forest).

Reuses cached simulations from the simulation pool.

References:
  - Marino et al. (2008) "A methodology for performing global uncertainty and
    sensitivity analysis in systems biology" J Theor Biol 254(1):178-196

Usage:
    python workflows/prcc_sensitivity.py

Output:
    results/prcc_sensitivity/  - CSV rankings, PRCC matrices
    figures/prcc_sensitivity/  - Bar plots, heatmaps
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
MODEL_VERSION = 'baseline_allparams_v3'
MODEL_DESCRIPTION = 'PRCC sensitivity: reuse restriction estimator simulations'

N_SIMULATIONS = 5e3
OVERSAMPLE_FACTOR = 2         # Generate 2x sims to compensate for NaN rate
MIN_SAMPLES_PER_TARGET = 10   # Skip targets with fewer valid samples
MAX_TASKS = 40
SEED = 2025

# Visualization / export
TOP_N_PARAMS = 30              # Show top-N in bar plots
TOP_N_AGGREGATE = 40           # Show top-N in aggregate ranking
PRCC_ALPHA = 0.05              # Significance threshold for PRCC

FIG_DIR = Path('figures/prcc_sensitivity')
RESULTS_DIR = Path('results/prcc_sensitivity')

# ============================================================================
# 1. Load priors and set up simulator
# ============================================================================
print("=" * 80)
print("PRCC SENSITIVITY ANALYSIS")
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
SANITY_CHECKS = {
    'tumor_volume_resection': (0.5, 500.0, 'cm³'),
    'cancer_cell_density_baseline': (10, 50000, 'cell/mm²'),
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
# 4. Spearman correlation + PRCC
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
print(f"\n   PRCC completed in {t_prcc:.1f}s")

# ============================================================================
# 5. Aggregate rankings
# ============================================================================
print(f"\n5. Aggregating rankings...")

abs_prcc = np.abs(prcc_matrix)
mean_abs_prcc = np.nanmean(abs_prcc, axis=1)
max_abs_prcc = np.nanmax(abs_prcc, axis=1)

# Count significant PRCCs per parameter
n_significant = np.sum(prcc_pvalue_matrix < PRCC_ALPHA, axis=1)

aggregate_df = pd.DataFrame({
    'parameter': param_names,
    'mean_abs_prcc': mean_abs_prcc,
    'max_abs_prcc': max_abs_prcc,
    'n_significant_prcc': n_significant,
})

aggregate_df['rank_prcc'] = aggregate_df['mean_abs_prcc'].rank(ascending=False).astype(int)

# Sort by mean |PRCC|
aggregate_df = aggregate_df.sort_values('mean_abs_prcc', ascending=False)

print(f"\n   Top {TOP_N_AGGREGATE} parameters by mean |PRCC|:")
print(f"   {'Param':<40s} {'|PRCC|':>8s} {'max|PRCC|':>10s} {'#sig':>5s} {'Rank':>6s}")
print(f"   {'-'*40} {'-'*8} {'-'*10} {'-'*5} {'-'*6}")
for _, row in aggregate_df.head(TOP_N_AGGREGATE).iterrows():
    print(f"   {row['parameter']:<40s} {row['mean_abs_prcc']:>8.4f} "
          f"{row['max_abs_prcc']:>10.4f} {row['n_significant_prcc']:>5.0f} "
          f"{row['rank_prcc']:>6d}")

# ============================================================================
# 5b. Data-driven cutoff analysis (permutation null test)
# ============================================================================
print("\n5b. Permutation null test for parameter significance...")

# --- Cumulative importance (scree-style) ---
sorted_prcc = np.sort(mean_abs_prcc)[::-1]
cum_prcc = np.cumsum(sorted_prcc) / sorted_prcc.sum()

for frac in [0.80, 0.90, 0.95]:
    k_prcc = int(np.searchsorted(cum_prcc, frac)) + 1
    print(f"   {frac:.0%} cumulative importance: top-{k_prcc} parameters")

# --- Permutation null test ---
PERM_REPEATS = 1000
FDR_ALPHA = 0.05

n_total = theta.shape[0]

print(f"\n   Permutation null test ({PERM_REPEATS} permutations, "
      f"FDR alpha={FDR_ALPHA})...")


def _compute_mean_abs_prcc(X_log_in, x_in, kept_target_indices):
    """Compute mean |PRCC| per parameter across all kept targets.

    Parameters
    ----------
    X_log_in : array (n_samples, n_params) — log10 parameter values
    x_in : array (n_samples, n_targets) — simulation outputs
    kept_target_indices : list[int] — column indices into x_in for kept targets
    """
    n_p = X_log_in.shape[1]
    abs_prcc_accum = np.zeros(n_p)
    n_counted = 0
    for j in kept_target_indices:
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


# Pre-compute kept target column indices (avoid repeated str lookups in workers)
_kept_target_indices = [target_names.index(t) for t in kept_targets]

observed_mean_abs_prcc = _compute_mean_abs_prcc(theta_log, x, _kept_target_indices)


def _one_permutation(perm):
    """Shuffle x rows and compute null mean |PRCC| for one permutation."""
    rng = np.random.default_rng(perm * 7919 + n_total)
    x_shuffled = x[rng.permutation(n_total)]
    return _compute_mean_abs_prcc(theta_log, x_shuffled, _kept_target_indices)


null_mean_abs_prcc = np.array(
    Parallel(n_jobs=-1, verbose=10)(
        delayed(_one_permutation)(perm) for perm in range(PERM_REPEATS)
    )
)

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
recommended = sorted(aggregate_df[aggregate_df['significant']]['parameter'].tolist())
aggregate_df['in_recommended_set'] = aggregate_df['parameter'].isin(recommended)

n_sig = reject.sum()
print(f"\n   {n_sig}/{n_params} parameters significant at FDR < {FDR_ALPHA}")

print(f"\n   {'Parameter':<40s} {'|PRCC|':>8s} {'p-val':>7s} {'FDR':>7s} {'Sig':>5s}")
print(f"   {'-'*40} {'-'*8} {'-'*7} {'-'*7} {'-'*5}")
for _, row in aggregate_df.head(50).iterrows():
    sig = "*" if row['significant'] else ""
    print(f"   {row['parameter']:<40s} {row['mean_abs_prcc']:>8.4f} "
          f"{row['perm_pvalue']:>7.3f} {row['perm_pvalue_fdr']:>7.3f} {sig:>5s}")

print(f"\n   Recommended set: {len(recommended)} parameters (FDR < {FDR_ALPHA})")

# ============================================================================
# 6. Export CSVs
# ============================================================================
print(f"\n6. Exporting results to {RESULTS_DIR}/...")
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

# Per-target PRCC ranking CSVs
per_target_dir = RESULTS_DIR / 'per_target'
per_target_dir.mkdir(exist_ok=True)
for tj, target_name in enumerate(kept_targets):
    target_df = pd.DataFrame({
        'parameter': param_names,
        'prcc': prcc_matrix[:, tj],
        'prcc_pvalue': prcc_pvalue_matrix[:, tj],
        'spearman': spearman_matrix[:, tj],
    })
    target_df['abs_prcc'] = np.abs(target_df['prcc'])
    target_df = target_df.sort_values('abs_prcc', ascending=False)
    safe_name = target_name.replace('/', '_').replace(' ', '_')
    target_df.to_csv(per_target_dir / f'prcc_{safe_name}.csv', index=False)

# Recommended parameter set
recommended_df = aggregate_df[aggregate_df['in_recommended_set']].copy()
recommended_df.to_csv(RESULTS_DIR / 'recommended_parameters.csv', index=False)

# Summary JSON
summary = {
    'n_params': n_params,
    'n_simulations': int(theta.shape[0]),
    'n_targets_analyzed': len(kept_targets),
    'n_targets_skipped': len(skipped_targets),
    'overall_nan_rate': float(overall_nan_rate),
    'cumulative_90pct_prcc': int(np.searchsorted(cum_prcc, 0.90)) + 1,
    'permutation_test': {
        'n_permutations': PERM_REPEATS,
        'fdr_alpha': FDR_ALPHA,
        'n_significant': int(n_sig),
        'significant_parameters': recommended,
    },
    'n_valid_per_target': {k: int(v) for k, v in target_valid_counts.items()
                           if k in kept_targets},
}
with open(RESULTS_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print(f"   Saved aggregate_parameter_ranking.csv")
print(f"   Saved recommended_parameters.csv ({len(recommended_df)} params)")
print(f"   Saved prcc_matrix.csv ({prcc_matrix.shape})")
print(f"   Saved prcc_pvalue_matrix.csv")
print(f"   Saved spearman_matrix.csv")
print(f"   Saved {len(kept_targets)} per-target CSVs in {per_target_dir}/")
print(f"   Saved summary.json")

# ============================================================================
# 7. Figures
# ============================================================================
print(f"\n7. Generating figures in {FIG_DIR}/...")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- 7a. Cutoff analysis: cumulative importance + permutation significance ---
print("   7a. Cutoff analysis...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

ks = np.arange(1, n_params + 1)
ax1.plot(ks, cum_prcc, 'steelblue', linewidth=2)
for frac in [0.80, 0.90, 0.95]:
    ax1.axhline(frac, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    k_p = int(np.searchsorted(cum_prcc, frac)) + 1
    ax1.annotate(f'{frac:.0%}: top-{k_p}',
                 xy=(k_p, frac), fontsize=8, alpha=0.7,
                 xytext=(5, 3), textcoords='offset points')
ax1.set_xlabel('Number of parameters (ranked)')
ax1.set_ylabel('Cumulative fraction of total |PRCC|')
ax1.set_title('Cumulative Importance (Scree-Style)')
ax1.set_xlim(1, min(100, n_params))
ax1.grid(alpha=0.3)

# Per-parameter significance bar chart
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

# --- 7b. Per-target PRCC bar plots (top-30 by |PRCC|) ---
print("   7b. Per-target PRCC bar plots...")
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

# --- 7c. PRCC heatmap (influential params x targets) ---
print("   7c. PRCC heatmap...")

sig_any = np.any(prcc_pvalue_matrix < PRCC_ALPHA, axis=1)
sig_indices = np.where(sig_any)[0]
sig_indices = sig_indices[np.argsort(mean_abs_prcc[sig_indices])[::-1]]

if len(sig_indices) > 0:
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

# --- 7d. Aggregate ranking bar plot ---
print("   7d. Aggregate ranking bar plot...")
top_agg = aggregate_df.head(TOP_N_AGGREGATE)

fig, ax = plt.subplots(figsize=(10, 10))
y_pos = np.arange(len(top_agg))
bar_colors = ['steelblue' if s else 'lightsteelblue'
              for s in top_agg['in_recommended_set'].values]
ax.barh(y_pos, top_agg['mean_abs_prcc'].values, color=bar_colors, edgecolor='white')
ax.set_yticks(y_pos)
ax.set_yticklabels(top_agg['parameter'].values, fontsize=7)
ax.invert_yaxis()
ax.set_xlabel('Mean |PRCC| across targets')
ax.set_title(f'Top {TOP_N_AGGREGATE} by Mean |PRCC|\n'
             f'Dark = FDR-significant, Light = not significant', fontsize=11)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
fig.savefig(FIG_DIR / 'aggregate_ranking.png', dpi=150, bbox_inches='tight')
plt.close(fig)

# ============================================================================
# 8. Summary
# ============================================================================
print("\n" + "=" * 80)
print("PRCC SENSITIVITY ANALYSIS COMPLETE")
print("=" * 80)
print(f"\n  Parameters:              {n_params}")
print(f"  Simulations:             {theta.shape[0]} (from cached pool)")
print(f"  Targets analyzed:        {len(kept_targets)}/{n_targets}")
if skipped_targets:
    print(f"  Targets skipped:         {len(skipped_targets)} (<{MIN_SAMPLES_PER_TARGET} valid)")
print(f"  Overall NaN rate:        {overall_nan_rate:.1%}")
print(f"  PRCC time:               {t_prcc:.1f}s")
print(f"  FDR-significant params:  {n_sig}")
print(f"  Recommended set:         {len(recommended)} parameters")
print(f"  Figures:                 {FIG_DIR}/")
print(f"  Results:                 {RESULTS_DIR}/")
print(f"\n  Key files:")
print(f"    aggregate_parameter_ranking.csv  - Per-param summary (primary output)")
print(f"    recommended_parameters.csv       - FDR-significant subset")
print(f"    prcc_matrix.csv                  - Signed PRCC ({n_params} x {len(kept_targets)})")
print(f"    prcc_heatmap.png                 - Visual summary")
