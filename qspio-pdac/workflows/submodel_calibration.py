#!/usr/bin/env python3
"""
Submodel Calibration → Updated Priors Pipeline

Orchestrates the pipeline from curated submodel target YAMLs through Bayesian
MCMC inference to an updated priors CSV for full-model SBI calibration:

    submodel YAMLs → JointInferenceBuilder → joint_calibration.jl →
    Julia MCMC → inference_results.json + posterior_samples.json →
    updated priors CSV (scenarios/priors/submodel_calibrated.csv)

Each submodel target constrains one QSP parameter using isolated experimental
data (e.g., in vitro kinetics, clinical imaging). The joint MCMC shares
parameters that appear across multiple targets, producing narrower posteriors
than the wide defaults in pdac_priors.csv.

Usage:
    python workflows/submodel_calibration.py                # full pipeline
    python workflows/submodel_calibration.py --generate     # Julia script only
    python workflows/submodel_calibration.py --from-results # parse existing results
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ============================================================================
# CONFIGURATION
# ============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Input paths ---
SUBMODEL_TARGETS_DIR = PROJECT_ROOT / 'calibration_targets' / 'submodel_targets'
MODEL_STRUCTURE = PROJECT_ROOT / 'model_structure.json'
REFERENCE_VALUES = PROJECT_ROOT / 'scenarios' / 'calibration_targets' / 'reference_values.yaml'
BASE_PRIORS_CSV = PROJECT_ROOT / 'scenarios' / 'pdac_priors.csv'

# --- Output paths ---
OUTPUT_PRIORS_CSV = PROJECT_ROOT / 'scenarios' / 'priors' / 'submodel_calibrated.csv'
JULIA_SCRIPT = PROJECT_ROOT / 'cache' / 'submodel_joint_calibration.jl'
JULIA_RESULTS_DIR = PROJECT_ROOT / 'cache' / 'submodel_calibration'

# --- Parameter name mapping ---
# Submodel targets use generic names; pdac_priors.csv may use different names.
# Simple 1-to-1 renames:
PARAM_NAME_MAP = {
    'k_apsc_death': 'k_apCAF_death',
}

# Parameters to exclude from prior updates (e.g., 1-to-many mappings that
# can't be applied blindly). These are still calibrated in the Julia script
# but their posteriors are not propagated to the priors CSV.
EXCLUDED_PARAMS = {
    'k_psc_activation',  # maps to myCAF/iCAF/apCAF variants — needs manual review
}

# --- MCMC convergence thresholds ---
RHAT_THRESHOLD = 1.05
ESS_BULK_THRESHOLD = 400


# ============================================================================
# 1. GENERATE JULIA SCRIPT
# ============================================================================

def generate_julia_script(target_dir: Path, output_path: Path) -> str:
    """Generate a joint Julia MCMC script from all submodel target YAMLs.

    Returns the Julia code as a string and writes it to output_path.
    """
    # Import JointInferenceBuilder from maple
    maple_path = PROJECT_ROOT.parent / 'maple' / 'src'
    if maple_path.exists() and str(maple_path) not in sys.path:
        sys.path.insert(0, str(maple_path))

    try:
        from maple.core.calibration.julia_translator import (
            JointInferenceBuilder,
        )
    except ImportError:
        print("ERROR: Cannot import JointInferenceBuilder.")
        print(f"  Searched: {maple_path}")
        print("  Install maple or ensure ../maple exists.")
        sys.exit(1)

    # Discover all YAML targets
    yaml_paths = sorted(target_dir.glob('*.yaml'))
    if not yaml_paths:
        print(f"ERROR: No YAML files found in {target_dir}")
        sys.exit(1)
    print(f"  Found {len(yaml_paths)} submodel target YAMLs")

    # Build joint inference script
    ref_path = str(REFERENCE_VALUES) if REFERENCE_VALUES.exists() else None
    builder = JointInferenceBuilder.from_model_structure_file(
        str(MODEL_STRUCTURE),
        reference_values_path=ref_path,
    )
    julia_code = builder.build_from_files(
        [str(p) for p in yaml_paths],
        force_fixed_sigma=False,
    )

    # Write to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(julia_code)
    print(f"  Generated Julia script: {output_path}")
    print(f"  Script size: {len(julia_code):,} characters")

    return julia_code


# ============================================================================
# 2. RUN JULIA
# ============================================================================

def run_julia(script_path: Path, results_dir: Path) -> bool:
    """Run the Julia MCMC script. Returns True if successful."""
    # Check Julia availability
    try:
        version = subprocess.run(
            ['julia', '--version'],
            capture_output=True, text=True, timeout=10,
        )
        if version.returncode != 0:
            raise FileNotFoundError
        print(f"  Julia found: {version.stdout.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("\n  WARNING: Julia not found on PATH.")
        print("  To run the MCMC calibration:")
        print(f"    1. Install Julia from https://julialang.org/downloads/")
        print(f"    2. Install packages: julia -e 'using Pkg; Pkg.add([\"Turing\", \"DifferentialEquations\", \"JSON\", \"StatsPlots\"])'")
        print(f"    3. Run: julia --threads=auto {script_path}")
        print(f"    4. Then re-run: python {__file__} --from-results")
        return False

    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Running Julia MCMC (this may take several minutes)...")
    print(f"  Working directory: {results_dir}")

    result = subprocess.run(
        ['julia', '--threads=auto', str(script_path)],
        cwd=str(results_dir),
        timeout=3600,  # 1 hour max
    )

    if result.returncode != 0:
        print(f"\n  ERROR: Julia script failed with return code {result.returncode}")
        return False

    print("  Julia MCMC completed successfully.")
    return True


# ============================================================================
# 3. PARSE RESULTS
# ============================================================================

def load_results(results_dir: Path) -> tuple[dict, dict]:
    """Load inference_results.json and posterior_samples.json.

    Returns (results_dict, samples_dict).
    """
    results_path = results_dir / 'inference_results.json'
    samples_path = results_dir / 'posterior_samples.json'

    if not results_path.exists():
        print(f"  ERROR: {results_path} not found.")
        print(f"  Run Julia first: julia --threads=auto {JULIA_SCRIPT}")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    samples = {}
    if samples_path.exists():
        with open(samples_path) as f:
            samples = json.load(f)
    else:
        print(f"  WARNING: {samples_path} not found. Using summary stats only.")

    print(f"  Loaded results for {len(results.get('parameters', {}))} parameters")
    if samples:
        first_key = next(iter(samples))
        print(f"  Posterior samples: {len(samples[first_key])} per parameter")

    return results, samples


# ============================================================================
# 4. CONVERT POSTERIORS TO PRIORS
# ============================================================================

def fit_lognormal(samples: list[float]) -> tuple[float, float]:
    """Fit a lognormal distribution to posterior samples.

    Returns (mu, sigma) on the log scale.
    """
    arr = np.array(samples, dtype=float)
    arr = arr[arr > 0]  # lognormal requires positive values
    if len(arr) == 0:
        raise ValueError("No positive samples to fit")
    log_samples = np.log(arr)
    mu = float(np.mean(log_samples))
    sigma = float(np.std(log_samples, ddof=1))
    return mu, sigma


def posteriors_to_priors(
    results: dict,
    samples: dict,
    base_priors_df: pd.DataFrame,
) -> pd.DataFrame:
    """Convert posterior distributions to an updated priors CSV.

    For each calibrated parameter:
    - Fit lognormal to posterior samples
    - Validate MCMC convergence (rhat, ESS)
    - Update the corresponding row(s) in the priors CSV

    Parameters not calibrated retain their original priors.
    """
    updated_df = base_priors_df.copy()
    params_info = results.get('parameters', {})

    calibrated = []
    skipped = []
    excluded = []

    for param_name, info in params_info.items():
        # Check exclusion list
        if param_name in EXCLUDED_PARAMS:
            excluded.append((param_name, 'in EXCLUDED_PARAMS'))
            continue

        # Check convergence
        rhat = info.get('rhat', float('inf'))
        ess_bulk = info.get('ess_bulk', 0)
        if rhat > RHAT_THRESHOLD:
            skipped.append((param_name, f'rhat={rhat:.3f} > {RHAT_THRESHOLD}'))
            continue
        if ess_bulk < ESS_BULK_THRESHOLD:
            skipped.append((param_name, f'ess_bulk={ess_bulk:.0f} < {ESS_BULK_THRESHOLD}'))
            continue

        # Get posterior samples (prefer raw samples, fall back to summary stats)
        if param_name in samples and len(samples[param_name]) > 0:
            mu, sigma = fit_lognormal(samples[param_name])
        else:
            # Fall back: use summary statistics to approximate lognormal params
            median_val = info.get('median', info.get('mean'))
            std_val = info.get('std', 0)
            if median_val is None or median_val <= 0:
                skipped.append((param_name, 'non-positive median'))
                continue
            mu = np.log(median_val)
            # Approximate sigma from CV: sigma_log ≈ sqrt(log(1 + (std/mean)^2))
            mean_val = info.get('mean', median_val)
            if mean_val > 0 and std_val > 0:
                cv = std_val / mean_val
                sigma = float(np.sqrt(np.log(1 + cv**2)))
            else:
                sigma = 0.5  # conservative default

        # Map parameter name to priors CSV name(s)
        csv_name = PARAM_NAME_MAP.get(param_name, param_name)

        # Find matching rows in the priors CSV
        mask = updated_df['name'] == csv_name
        if not mask.any():
            skipped.append((param_name, f'not found in priors CSV as "{csv_name}"'))
            continue

        # Compute derived values
        median_val = float(np.exp(mu))
        lower = float(np.exp(mu - 2 * sigma))
        upper = float(np.exp(mu + 2 * sigma))

        # Update the row
        updated_df.loc[mask, 'median'] = median_val
        updated_df.loc[mask, 'distribution'] = 'lognormal'
        updated_df.loc[mask, 'dist_param1'] = mu
        updated_df.loc[mask, 'dist_param2'] = sigma
        updated_df.loc[mask, 'lower_bound'] = lower
        updated_df.loc[mask, 'upper_bound'] = upper

        calibrated.append({
            'submodel_param': param_name,
            'csv_param': csv_name,
            'mu': mu,
            'sigma': sigma,
            'median': median_val,
            'lower': lower,
            'upper': upper,
            'rhat': rhat,
            'ess_bulk': ess_bulk,
        })

    return updated_df, calibrated, skipped, excluded


# ============================================================================
# 5. REPORTING
# ============================================================================

def print_comparison(calibrated: list[dict], base_priors_df: pd.DataFrame):
    """Print a comparison table of old vs new priors."""
    if not calibrated:
        print("\n  No parameters were updated.")
        return

    print(f"\n  {'Parameter':<28s} {'Old σ':>7s} {'New σ':>7s} {'Old median':>12s} "
          f"{'New median':>12s} {'Narrowed':>9s} {'rhat':>6s} {'ESS':>6s}")
    print("  " + "-" * 100)

    for entry in calibrated:
        csv_name = entry['csv_param']
        old_row = base_priors_df[base_priors_df['name'] == csv_name]
        if old_row.empty:
            continue
        old_sigma = float(old_row['dist_param2'].iloc[0])
        old_median = float(old_row['median'].iloc[0])
        new_sigma = entry['sigma']
        new_median = entry['median']
        narrowed = f"{(1 - new_sigma/old_sigma)*100:.0f}%" if old_sigma > 0 else "N/A"

        print(f"  {entry['submodel_param']:<28s} {old_sigma:>7.3f} {new_sigma:>7.3f} "
              f"{old_median:>12.4g} {new_median:>12.4g} {narrowed:>9s} "
              f"{entry['rhat']:>6.3f} {entry['ess_bulk']:>6.0f}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Submodel calibration → updated priors pipeline',
    )
    parser.add_argument(
        '--generate', action='store_true',
        help='Only generate the Julia script (do not run Julia or update priors)',
    )
    parser.add_argument(
        '--from-results', action='store_true',
        help='Skip Julia generation/execution, parse existing results and update priors',
    )
    parser.add_argument(
        '--results-dir', type=Path, default=JULIA_RESULTS_DIR,
        help=f'Directory containing Julia results (default: {JULIA_RESULTS_DIR})',
    )
    parser.add_argument(
        '--output', type=Path, default=OUTPUT_PRIORS_CSV,
        help=f'Output priors CSV path (default: {OUTPUT_PRIORS_CSV})',
    )
    args = parser.parse_args()

    print("=" * 80)
    print("SUBMODEL CALIBRATION PIPELINE")
    print("=" * 80)

    # --- Step 1: Generate Julia script ---
    if not args.from_results:
        print("\n" + "=" * 80)
        print("1. GENERATING JOINT JULIA MCMC SCRIPT")
        print("=" * 80)
        generate_julia_script(SUBMODEL_TARGETS_DIR, JULIA_SCRIPT)

        if args.generate:
            print("\n  --generate flag set. Stopping after script generation.")
            print(f"  To run: julia --threads=auto {JULIA_SCRIPT}")
            return

        # --- Step 2: Run Julia ---
        print("\n" + "=" * 80)
        print("2. RUNNING JULIA MCMC")
        print("=" * 80)
        success = run_julia(JULIA_SCRIPT, args.results_dir)
        if not success:
            print("\n  Julia execution skipped or failed.")
            print(f"  After running Julia manually, re-run with: python {__file__} --from-results")
            return

    # --- Step 3: Parse results ---
    print("\n" + "=" * 80)
    print("3. PARSING MCMC RESULTS")
    print("=" * 80)
    results, samples = load_results(args.results_dir)

    # --- Step 4: Convert posteriors to priors ---
    print("\n" + "=" * 80)
    print("4. CONVERTING POSTERIORS TO UPDATED PRIORS")
    print("=" * 80)

    base_priors_df = pd.read_csv(BASE_PRIORS_CSV)
    print(f"  Base priors: {len(base_priors_df)} parameters from {BASE_PRIORS_CSV.name}")

    updated_df, calibrated, skipped, excluded = posteriors_to_priors(
        results, samples, base_priors_df,
    )

    if excluded:
        print(f"\n  Excluded ({len(excluded)}):")
        for name, reason in excluded:
            print(f"    {name}: {reason}")

    if skipped:
        print(f"\n  Skipped ({len(skipped)}):")
        for name, reason in skipped:
            print(f"    {name}: {reason}")

    # --- Step 5: Write updated CSV ---
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    updated_df.to_csv(output_path, index=False)
    print(f"\n  Wrote updated priors to {output_path}")
    print(f"  Updated: {len(calibrated)} parameters")
    print(f"  Unchanged: {len(base_priors_df) - len(calibrated)} parameters")

    # --- Step 6: Report ---
    print("\n" + "=" * 80)
    print("5. COMPARISON: OLD vs NEW PRIORS")
    print("=" * 80)
    print_comparison(calibrated, base_priors_df)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\n  Updated priors CSV: {output_path}")
    print(f"  To use in SBI workflow, set FULL_PRIORS_CSV to this file.")


if __name__ == '__main__':
    main()
