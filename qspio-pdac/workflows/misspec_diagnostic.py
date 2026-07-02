#!/usr/bin/env python3
"""
Model Misspecification Diagnostic — CMA-ES Best-Fit + Jacobian Analysis

Quick diagnostic to check whether the QSP model can structurally hit calibration
targets before committing to expensive SBI runs. Uses CMA-ES (population-based,
parallelizable) to find the best-fit parameter vector, then computes a finite-
difference Jacobian at the optimum to identify:
  - Which targets the model cannot hit (structural misspecification)
  - Which targets compete for the same parameters (conflict diagnosis)
  - Which parameters are identifiable vs sloppy

Per-generation diagnostics are written incrementally so you can monitor progress
or kill the script early and still have usable results.

Sections:
1. Config + setup
2. CMA-ES optimization (log-space, batched via simulate_with_parameters)
3. Per-observable residual report
4. Jacobian at optimum (finite-difference)
5. Target conflict + identifiability analysis
6. Save report + figures

Usage:
    python workflows/misspec_diagnostic.py
"""

import time
from pathlib import Path

import cma
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from qsp_hpc import QSPSimulator


if __name__ == "__main__":
    # ============================================================================
    # 1. CONFIG + SETUP
    # ============================================================================

    MODEL_VERSION = 'baseline_allparams_v6'
    MODEL_DESCRIPTION = 'PDAC baseline misspec diagnostic'
    MODEL_SCRIPT = 'immune_oncology_model_PDAC'
    SCENARIO = 'baseline_no_treatment'

    PRIORS_CSV = 'parameters/pdac_priors.csv'
    CALIBRATION_TARGETS_DIR = 'calibration_targets/baseline_no_treatment'
    MODEL_STRUCTURE_FILE = 'model_structure.json'

    # Execution mode
    LOCAL_MODE = True        # True = run MATLAB locally, False = use HPC
    N_LOCAL_WORKERS = 2      # parallel MATLAB processes (local mode)
    MATLAB_PATH = 'matlab'   # path to MATLAB executable (local mode)

    # CMA-ES settings
    POPSIZE = 20            # candidates per generation (parallelized)
    MAX_GENERATIONS = 15    # total generations
    SIGMA0 = 0.5           # initial step size in log-space (0.5 = ~1.6x factor)
    SEED = 2025
    MAX_TASKS = 40          # HPC tasks (HPC mode only)

    # Jacobian settings
    JACOBIAN_STEP = 0.05    # finite-diff step in log-space (~5% perturbation)

    # Use PRCC tier to reduce dimensionality (None = all params)
    PRCC_RANKING_CSV = 'results/prcc_sensitivity/aggregate_parameter_ranking.csv'
    PRCC_TIER = 1           # 1=top30, 2=top70, 3=all-significant, None=all
    TIER_CUTOFFS = {1: 30, 2: 70, 3: 112}

    RUN_JACOBIAN = True

    tier_suffix = f'_tier{PRCC_TIER}' if PRCC_TIER is not None else ''
    SAVE_DIR = Path(f'figures/misspec_diagnostic{tier_suffix}')
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    CMAES_DIR = SAVE_DIR / 'cmaes_generations'
    CMAES_DIR.mkdir(parents=True, exist_ok=True)

    np.random.seed(SEED)

    print("=" * 80)
    tier_label = (
        f"PRCC TIER {PRCC_TIER} ({TIER_CUTOFFS[PRCC_TIER]} PARAMS)"
        if PRCC_TIER is not None else "ALL PARAMETERS"
    )
    mode_label = f"LOCAL ({N_LOCAL_WORKERS} workers)" if LOCAL_MODE else "HPC"
    print(f"MISSPECIFICATION DIAGNOSTIC — {tier_label} — {mode_label}")
    print("=" * 80)

    # --- Set up simulator ---
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
        seed=SEED,
        cache_sampling_seed=SEED,
        local_only=LOCAL_MODE,
        matlab_path=MATLAB_PATH,
        n_local_workers=N_LOCAL_WORKERS,
    )

    # --- Load targets ---
    target_df = sim._test_stats_df.copy()
    observable_names = target_df['test_statistic_id'].tolist()
    target_medians = target_df['median'].values
    target_ci_lower = target_df['ci95_lower'].values
    target_ci_upper = target_df['ci95_upper'].values
    num_observables = len(observable_names)

    # Compute per-observable scale for residuals: half-width of CI95 in log-space
    # For targets near zero or with zero lower bound, fall back to target median
    target_log_scale = np.zeros(num_observables)
    for i in range(num_observables):
        lo, hi, med = target_ci_lower[i], target_ci_upper[i], target_medians[i]
        if lo > 0 and hi > 0 and med > 0:
            target_log_scale[i] = (np.log(hi) - np.log(lo)) / (2 * 1.96)
        elif med > 0:
            # Fallback: use CV=1 as a generous scale
            target_log_scale[i] = 1.0
        else:
            target_log_scale[i] = 1.0

    print(f"\n  Observables: {num_observables}")
    print(f"  Target log-scales (SD): min={target_log_scale.min():.2f}, "
          f"median={np.median(target_log_scale):.2f}, max={target_log_scale.max():.2f}")

    # --- Load priors and set up parameter space ---
    priors_df = pd.read_csv(PRIORS_CSV)
    all_param_names = priors_df['name'].tolist()
    all_log_medians = priors_df['dist_param1'].values  # mu in log-space
    all_log_sigmas = priors_df['dist_param2'].values    # sigma in log-space
    num_params_full = len(all_param_names)

    # PRCC tier filtering
    if PRCC_TIER is not None:
        ranking_df = pd.read_csv(PRCC_RANKING_CSV)
        n_keep = TIER_CUTOFFS[PRCC_TIER]
        selected_params = set(
            ranking_df[ranking_df['rank_prcc'] <= n_keep]['parameter'].tolist()
        )
        param_indices = [i for i, n in enumerate(all_param_names) if n in selected_params]
        param_names = [all_param_names[i] for i in param_indices]
        log_medians = all_log_medians[param_indices]
        log_sigmas = all_log_sigmas[param_indices]
        num_params = len(param_names)
        print(f"  PRCC tier {PRCC_TIER}: optimizing {num_params}/{num_params_full} parameters")
    else:
        param_indices = list(range(num_params_full))
        param_names = all_param_names
        log_medians = all_log_medians
        log_sigmas = all_log_sigmas
        num_params = num_params_full


    def log_theta_to_full_original(log_theta_subset):
        """Convert log-space subset theta to full original-space theta matrix.

        Non-optimized parameters are fixed at their prior medians.
        """
        if log_theta_subset.ndim == 1:
            log_theta_subset = log_theta_subset[np.newaxis, :]

        n = log_theta_subset.shape[0]
        log_theta_full = np.tile(all_log_medians, (n, 1))
        for j_sub, j_full in enumerate(param_indices):
            log_theta_full[:, j_full] = log_theta_subset[:, j_sub]
        return np.exp(log_theta_full)


    def compute_residuals(x_sim, return_per_obs=False):
        """Weighted residuals: (log(sim) - log(target)) / scale.

        Returns sum of squares (scalar) or per-observable array.
        """
        # Clip to avoid log(0)
        x_safe = np.clip(x_sim, 1e-30, None)
        t_safe = np.clip(target_medians, 1e-30, None)

        log_resid = (np.log(x_safe) - np.log(t_safe)) / target_log_scale
        if return_per_obs:
            return log_resid
        return np.sum(log_resid**2, axis=-1)


    # ============================================================================
    # 2. CMA-ES OPTIMIZATION
    # ============================================================================
    print("\n" + "=" * 80)
    print("2. CMA-ES OPTIMIZATION")
    print("=" * 80)
    print(f"  Population: {POPSIZE}, Max generations: {MAX_GENERATIONS}")
    print(f"  Sigma0: {SIGMA0} (log-space), Seed: {SEED}")

    # Bounds in log-space: prior median +/- 3*sigma
    log_lower = log_medians - 3 * log_sigmas
    log_upper = log_medians + 3 * log_sigmas

    opts = cma.CMAOptions()
    opts.set('popsize', POPSIZE)
    opts.set('maxiter', MAX_GENERATIONS)
    opts.set('seed', SEED)
    opts.set('bounds', [log_lower.tolist(), log_upper.tolist()])
    opts.set('verbose', -1)  # suppress CMA-ES stdout; we print our own

    es = cma.CMAEvolutionStrategy(log_medians.tolist(), SIGMA0, opts)

    t0 = time.time()
    gen = 0
    history = []
    global_best_loss = np.inf
    global_best_log_theta = log_medians.copy()
    global_best_x = None

    while not es.stop():
        gen += 1
        candidates_log = np.array(es.ask())  # (POPSIZE, num_params) in log-space

        # Evaluate: convert to original space, simulate, compute loss
        theta_orig = log_theta_to_full_original(candidates_log)
        x_sim = sim.simulate_with_parameters(theta_orig, pool_suffix=f'misspec_gen{gen}')

        # Handle NaN sims: assign worst-case fitness
        losses = compute_residuals(x_sim)
        nan_mask = ~np.isfinite(losses)
        if nan_mask.any():
            losses[nan_mask] = 1e10
            print(f"  Gen {gen}: {nan_mask.sum()}/{POPSIZE} NaN sims (penalized)")

        es.tell(candidates_log.tolist(), losses.tolist())

        best_this_gen = np.min(losses)
        median_loss = np.median(losses[~nan_mask]) if (~nan_mask).any() else float('nan')

        # Track global best
        if best_this_gen < global_best_loss:
            best_idx = np.argmin(losses)
            global_best_loss = best_this_gen
            global_best_log_theta = candidates_log[best_idx].copy()
            global_best_x = x_sim[best_idx].copy()

        history.append({
            'generation': gen,
            'best_loss': best_this_gen,
            'global_best_loss': global_best_loss,
            'median_loss': median_loss,
            'n_nan': int(nan_mask.sum()),
        })

        print(f"  Gen {gen:3d}: best={best_this_gen:.2f}, global={global_best_loss:.2f}, "
              f"median={median_loss:.2f}, NaN={int(nan_mask.sum())}/{POPSIZE}")

        # --- Write per-generation diagnostics ---
        # 1. Running convergence history
        hist_df = pd.DataFrame(history)
        hist_df.to_csv(CMAES_DIR / 'convergence_history.csv', index=False)

        # 2. This generation's population (log-space + losses)
        gen_df = pd.DataFrame(candidates_log, columns=param_names)
        gen_df['loss'] = losses
        gen_df['nan'] = nan_mask
        gen_df.to_csv(CMAES_DIR / f'gen_{gen:03d}_population.csv', index=False)

        # 3. Running best-fit residual report (updated each gen)
        best_resid = compute_residuals(
            global_best_x[np.newaxis, :], return_per_obs=True
        ).flatten()
        best_report = pd.DataFrame({
            'observable': observable_names,
            'target_median': target_medians,
            'ci95_lower': target_ci_lower,
            'ci95_upper': target_ci_upper,
            'best_fit': global_best_x,
            'z_score': best_resid,
            'abs_z': np.abs(best_resid),
            'in_ci95': [
                target_ci_lower[i] <= global_best_x[i] <= target_ci_upper[i]
                for i in range(num_observables)
            ],
        }).sort_values('abs_z', ascending=False)
        best_report.to_csv(CMAES_DIR / 'running_best_residuals.csv', index=False)

        # 4. Running best-fit params
        pd.DataFrame({
            'parameter': param_names,
            'log_value': global_best_log_theta,
            'original_value': np.exp(global_best_log_theta),
            'prior_log_median': log_medians,
            'z_from_prior': (global_best_log_theta - log_medians) / log_sigmas,
        }).to_csv(CMAES_DIR / 'running_best_params.csv', index=False)

        # 5. Convergence plot (overwritten each gen)
        fig_conv_running, ax_conv_running = plt.subplots(figsize=(8, 4))
        ax_conv_running.plot(hist_df['generation'], hist_df['best_loss'], 'o-',
                             label='Gen best', alpha=0.7)
        ax_conv_running.plot(hist_df['generation'], hist_df['global_best_loss'], 's-',
                             label='Global best', linewidth=2)
        ax_conv_running.plot(hist_df['generation'], hist_df['median_loss'], 'x--',
                             alpha=0.5, label='Gen median')
        ax_conv_running.set_xlabel('Generation')
        ax_conv_running.set_ylabel('Weighted sum of squared log-residuals')
        ax_conv_running.set_title(f'CMA-ES convergence (gen {gen})')
        ax_conv_running.legend()
        fig_conv_running.tight_layout()
        fig_conv_running.savefig(CMAES_DIR / 'convergence.png', dpi=150, bbox_inches='tight')
        plt.close(fig_conv_running)

    elapsed_opt = time.time() - t0
    total_sims = gen * POPSIZE
    print(f"\n  Optimization complete: {gen} generations, {total_sims} sims, {elapsed_opt:.0f}s")

    # Use tracked global best (CMA-ES result.xbest may differ due to boundary handling)
    best_log_theta = global_best_log_theta
    best_loss = global_best_loss
    print(f"  Best loss (sum of weighted squared log-residuals): {best_loss:.2f}")

    # ============================================================================
    # 3. PER-OBSERVABLE RESIDUAL REPORT
    # ============================================================================
    print("\n" + "=" * 80)
    print("3. PER-OBSERVABLE RESIDUAL REPORT")
    print("=" * 80)

    # Use global best from CMA-ES (already simulated, no extra sim needed)
    x_best = global_best_x[np.newaxis, :]
    residuals = compute_residuals(x_best, return_per_obs=True).flatten()

    report_rows = []
    for i in range(num_observables):
        obs_name = observable_names[i]
        sim_val = float(x_best[0, i])
        tgt_val = float(target_medians[i])
        z = float(residuals[i])
        in_ci = target_ci_lower[i] <= sim_val <= target_ci_upper[i]
        report_rows.append({
            'observable': obs_name,
            'target_median': tgt_val,
            'ci95_lower': float(target_ci_lower[i]),
            'ci95_upper': float(target_ci_upper[i]),
            'best_fit': sim_val,
            'z_score': z,
            'abs_z': abs(z),
            'in_ci95': in_ci,
        })

    report_df = pd.DataFrame(report_rows).sort_values('abs_z', ascending=False)
    report_df.to_csv(SAVE_DIR / 'residual_report.csv', index=False)

    print(f"\n  {'Observable':<45s} {'Target':>10s} {'Best-fit':>10s} {'z':>7s} {'In CI?':>6s}")
    print("  " + "-" * 82)
    for _, row in report_df.iterrows():
        flag = 'YES' if row['in_ci95'] else ' NO'
        print(f"  {row['observable']:<45s} {row['target_median']:10.3g} {row['best_fit']:10.3g} "
              f"{row['z_score']:+7.2f} {flag:>6s}")

    n_in_ci = int(report_df['in_ci95'].sum())
    n_outside = num_observables - n_in_ci
    print(f"\n  Summary: {n_in_ci}/{num_observables} in CI95, {n_outside} outside")

    # --- Residual bar chart ---
    fig_resid, ax_resid = plt.subplots(figsize=(10, max(6, num_observables * 0.35)))
    colors = ['#2ecc71' if row['in_ci95'] else '#e74c3c' for _, row in report_df.iterrows()]
    ax_resid.barh(range(num_observables), report_df['z_score'].values, color=colors)
    ax_resid.set_yticks(range(num_observables))
    ax_resid.set_yticklabels(report_df['observable'].values, fontsize=8)
    ax_resid.set_xlabel('Weighted log-residual (z-score)')
    ax_resid.set_title(f'Best-fit residuals ({n_in_ci}/{num_observables} in CI95)')
    ax_resid.axvline(0, color='black', linewidth=0.5)
    ax_resid.axvline(-2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_resid.axvline(2, color='gray', linestyle='--', linewidth=0.5, alpha=0.5)
    ax_resid.invert_yaxis()
    fig_resid.tight_layout()
    fig_resid.savefig(SAVE_DIR / 'residual_barchart.png', dpi=150, bbox_inches='tight')
    plt.close(fig_resid)

    # ============================================================================
    # 4. JACOBIAN AT OPTIMUM (finite-difference)
    # ============================================================================
    if RUN_JACOBIAN:
        print("\n" + "=" * 80)
        print("4. JACOBIAN AT OPTIMUM")
        print("=" * 80)
        print(f"  Step size: {JACOBIAN_STEP} in log-space ({num_params} params, "
              f"{2 * num_params} sims)")

        JAC_DIR = SAVE_DIR / 'jacobian'
        JAC_DIR.mkdir(parents=True, exist_ok=True)

        t0_jac = time.time()

        # Build perturbed parameter matrix: [+h for param 0, -h for param 0, +h for param 1, ...]
        perturbed_log = np.tile(best_log_theta, (2 * num_params, 1))
        for j in range(num_params):
            perturbed_log[2 * j, j] += JACOBIAN_STEP
            perturbed_log[2 * j + 1, j] -= JACOBIAN_STEP

        theta_perturbed_orig = log_theta_to_full_original(perturbed_log)
        x_perturbed = sim.simulate_with_parameters(
            theta_perturbed_orig, pool_suffix='misspec_jacobian'
        )

        # Central finite difference: d(log x) / d(log theta)
        # Jacobian[i, j] = (log x_i(theta+h_j) - log x_i(theta-h_j)) / (2h)
        x_perturbed_safe = np.clip(x_perturbed, 1e-30, None)
        log_x_perturbed = np.log(x_perturbed_safe)

        jacobian = np.zeros((num_observables, num_params))
        for j in range(num_params):
            jacobian[:, j] = (
                (log_x_perturbed[2 * j] - log_x_perturbed[2 * j + 1]) / (2 * JACOBIAN_STEP)
            )

        elapsed_jac = time.time() - t0_jac
        print(f"  Jacobian computed in {elapsed_jac:.0f}s")

        # --- Save Jacobian immediately ---
        jac_df = pd.DataFrame(jacobian, index=observable_names, columns=param_names)
        jac_df.to_csv(JAC_DIR / 'jacobian.csv')
        jac_df.to_csv(SAVE_DIR / 'jacobian.csv')

        # Per-observable: max absolute sensitivity across params
        max_sens = np.max(np.abs(jacobian), axis=1)
        dead_targets = [observable_names[i] for i in range(num_observables) if max_sens[i] < 0.01]
        if dead_targets:
            print(f"\n  WARNING: {len(dead_targets)} targets with near-zero sensitivity to all params:")
            for name in dead_targets:
                print(f"    {name}")
        else:
            print("  All targets have nonzero sensitivity to at least one parameter.")

        # Save per-observable sensitivity summary
        sens_df = pd.DataFrame({
            'observable': observable_names,
            'max_abs_sensitivity': max_sens,
            'top_param': [param_names[np.argmax(np.abs(jacobian[i]))] for i in range(num_observables)],
            'top_param_elasticity': [float(jacobian[i, np.argmax(np.abs(jacobian[i]))]) for i in range(num_observables)],
            'dead': max_sens < 0.01,
        }).sort_values('max_abs_sensitivity', ascending=True)
        sens_df.to_csv(JAC_DIR / 'sensitivity_summary.csv', index=False)

        # ============================================================================
        # 5. TARGET CONFLICT + IDENTIFIABILITY ANALYSIS
        # ============================================================================
        print("\n" + "=" * 80)
        print("5. TARGET CONFLICT + IDENTIFIABILITY ANALYSIS")
        print("=" * 80)

        # --- 5a. Target conflict: which targets want the same param to move opposite ways? ---
        # For each pair of targets that are both outside CI, check if their gradient
        # directions conflict (negative cosine similarity of Jacobian rows, weighted by residual)
        outside_mask = ~report_df.set_index('observable').loc[observable_names, 'in_ci95'].values
        outside_idx = np.where(outside_mask)[0]

        if len(outside_idx) >= 2:
            print(f"\n  Checking conflicts among {len(outside_idx)} targets outside CI95...")
            # Normalized Jacobian rows (direction each target wants params to move)
            # Sign: if residual > 0, target is too high, want to decrease -> negate gradient
            desired_direction = np.zeros_like(jacobian)
            for i in range(num_observables):
                if residuals[i] > 0:  # sim > target, want decrease
                    desired_direction[i] = -jacobian[i]
                else:  # sim < target, want increase
                    desired_direction[i] = jacobian[i]

            # Normalize rows
            norms = np.linalg.norm(desired_direction, axis=1, keepdims=True)
            norms = np.clip(norms, 1e-10, None)
            desired_normed = desired_direction / norms

            # Cosine similarity between outside-CI target pairs
            conflict_pairs = []
            for ii, i in enumerate(outside_idx):
                for jj in range(ii + 1, len(outside_idx)):
                    j = outside_idx[jj]
                    cos_sim = float(desired_normed[i] @ desired_normed[j])
                    if cos_sim < -0.3:  # meaningful conflict
                        conflict_pairs.append({
                            'target_a': observable_names[i],
                            'target_b': observable_names[j],
                            'cosine_similarity': cos_sim,
                            'z_a': float(residuals[i]),
                            'z_b': float(residuals[j]),
                        })

            if conflict_pairs:
                conflict_df = pd.DataFrame(conflict_pairs).sort_values('cosine_similarity')
                conflict_df.to_csv(JAC_DIR / 'target_conflicts.csv', index=False)
                conflict_df.to_csv(SAVE_DIR / 'target_conflicts.csv', index=False)
                print(f"\n  Found {len(conflict_pairs)} conflicting target pairs:")
                for _, row in conflict_df.head(10).iterrows():
                    print(f"    {row['target_a']} vs {row['target_b']}: "
                          f"cos={row['cosine_similarity']:.2f}")
            else:
                print("  No strong conflicts detected among outside-CI targets.")
        else:
            print("  Fewer than 2 targets outside CI95 — no conflict analysis needed.")

        # --- 5b. Parameter identifiability (SVD of Jacobian) ---
        U, S, Vt = np.linalg.svd(jacobian / target_log_scale[:, np.newaxis], full_matrices=False)
        n_effective = float(np.sum(S)**2 / np.sum(S**2))  # participation ratio
        print(f"\n  Jacobian SVD: {num_params} params, {num_observables} observables")
        print(f"  Effective rank (participation ratio): {n_effective:.1f}/{min(num_params, num_observables)}")
        print(f"  Top 5 singular values: {', '.join(f'{s:.2f}' for s in S[:5])}")
        if len(S) > 5:
            print(f"  Bottom 5 singular values: {', '.join(f'{s:.3g}' for s in S[-5:])}")

        svd_df = pd.DataFrame({
            'index': range(len(S)),
            'singular_value': S,
            'cumulative_variance_frac': np.cumsum(S**2) / np.sum(S**2),
        })
        svd_df.to_csv(JAC_DIR / 'svd_spectrum.csv', index=False)

        # Sloppy params: columns of V with small singular values
        sloppy_threshold = S[0] * 0.01  # 1% of largest
        n_sloppy = int(np.sum(S < sloppy_threshold))

        if n_sloppy > 0:
            # Sloppy directions are the last n_sloppy rows of Vt
            sloppy_directions = Vt[-n_sloppy:]
            # Which params contribute most to sloppy directions?
            sloppy_loading = np.sum(sloppy_directions**2, axis=0)
            sloppy_ranking = np.argsort(sloppy_loading)[::-1]
            print(f"\n  {n_sloppy} sloppy directions (SV < {sloppy_threshold:.3g}):")
            print(f"  Top 5 params in sloppy subspace:")
            for idx in sloppy_ranking[:5]:
                print(f"    {param_names[idx]}: loading = {sloppy_loading[idx]:.3f}")

            sloppy_df = pd.DataFrame({
                'parameter': param_names,
                'sloppy_loading': sloppy_loading,
            }).sort_values('sloppy_loading', ascending=False)
            sloppy_df.to_csv(JAC_DIR / 'sloppy_params.csv', index=False)

        # --- Jacobian heatmap ---
        fig_jac, ax_jac = plt.subplots(
            figsize=(max(10, num_params * 0.4), max(6, num_observables * 0.35))
        )
        jac_display = jacobian.copy()
        vmax = np.percentile(np.abs(jac_display), 95)
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
        im = ax_jac.imshow(jac_display, aspect='auto', cmap='RdBu_r', norm=norm)
        ax_jac.set_xticks(range(num_params))
        ax_jac.set_xticklabels(param_names, rotation=90, fontsize=6)
        ax_jac.set_yticks(range(num_observables))
        ax_jac.set_yticklabels(observable_names, fontsize=7)
        ax_jac.set_title('Jacobian: d(log observable) / d(log parameter)')
        fig_jac.colorbar(im, ax=ax_jac, shrink=0.8, label='Elasticity')
        fig_jac.tight_layout()
        fig_jac.savefig(SAVE_DIR / 'jacobian_heatmap.png', dpi=150, bbox_inches='tight')
        plt.close(fig_jac)

    # ============================================================================
    # 6. SAVE SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("6. SUMMARY")
    print("=" * 80)

    # Save best-fit parameters
    best_params_df = pd.DataFrame({
        'parameter': param_names,
        'log_value': best_log_theta,
        'original_value': np.exp(best_log_theta),
        'prior_log_median': log_medians,
        'prior_log_sigma': log_sigmas,
        'z_from_prior': (best_log_theta - log_medians) / log_sigmas,
    })
    best_params_df.to_csv(SAVE_DIR / 'best_fit_params.csv', index=False)

    total_sims_all = total_sims + (2 * num_params if RUN_JACOBIAN else 0)
    print(f"  Total simulations: {total_sims_all}")
    print(f"  Best-fit loss: {best_loss:.2f}")
    print(f"  Targets in CI95: {n_in_ci}/{num_observables}")
    print(f"  Outputs saved to: {SAVE_DIR}/")
    print(f"    residual_report.csv — per-observable breakdown")
    print(f"    best_fit_params.csv — optimized parameter values")
    if RUN_JACOBIAN:
        print(f"    jacobian.csv — full Jacobian matrix")
        print(f"    target_conflicts.csv — conflicting target pairs")
    print("=" * 80)
