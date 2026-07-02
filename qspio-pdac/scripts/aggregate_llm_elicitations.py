#!/usr/bin/env python3
"""
Aggregate multiple LLM expert elicitations into a consensus target set.

For each observable, samples 10,000 draws from each expert's stated distribution
(lognormal for positive quantities, logit-normal for fractions), pools them into
a mixture, and takes the median as the consensus estimate with 2.5th/97.5th
percentiles as the consensus CI95.

Usage:
    python scripts/aggregate_llm_elicitations.py

Reads:  notes/inference/llm_elicitation/llm_elicited_targets_v*.csv
Writes: notes/inference/llm_elicitation/llm_elicited_consensus.csv
"""

import glob
import numpy as np
import pandas as pd
from scipy.special import logit, expit

N_SAMPLES_PER_EXPERT = 10_000

# Mapping from elicited (generic) names → model observable names
NAME_REMAP = {
    'cd8_exhausted_fraction': 'cd8_exhausted_fraction_shen2017',
    'cd8_treg_ratio_alt': 'cd8_treg_ratio_golesworthy2022',
    'treg_fraction_cd4_alt': 'treg_fraction_cd4_hiraoka2006',
    'apcaf_fraction_of_caf_alt': 'apcaf_fraction_of_caf_elyada2019',
}


def fit_lognormal(median, ci95_lo, ci95_hi):
    """Fit lognormal params from median and CI95."""
    mu = np.log(median)
    # Average the two sigma estimates from lo and hi
    sigma_lo = (mu - np.log(ci95_lo)) / 1.96
    sigma_hi = (np.log(ci95_hi) - mu) / 1.96
    sigma = (sigma_lo + sigma_hi) / 2
    return mu, max(sigma, 0.01)


def fit_logit_normal(median, ci95_lo, ci95_hi):
    """Fit logit-normal params from median and CI95 for (0,1) bounded quantities."""
    # Clamp to avoid logit(0) or logit(1)
    eps = 1e-6
    median = np.clip(median, eps, 1 - eps)
    ci95_lo = np.clip(ci95_lo, eps, 1 - eps)
    ci95_hi = np.clip(ci95_hi, eps, 1 - eps)

    mu = logit(median)
    sigma_lo = (mu - logit(ci95_lo)) / 1.96
    sigma_hi = (logit(ci95_hi) - mu) / 1.96
    sigma = (sigma_lo + sigma_hi) / 2
    return mu, max(sigma, 0.01)


def sample_expert(row, n=N_SAMPLES_PER_EXPERT):
    """Sample from a single expert's stated distribution."""
    median = float(row['estimate'])
    lo = float(row['ci95_lo'])
    hi = float(row['ci95_hi'])
    units = str(row.get('units', ''))

    is_fraction = (0 < median < 1 and 0 <= lo < 1 and 0 < hi <= 1
                   and 'dimensionless' in units.lower())

    if is_fraction:
        mu, sigma = fit_logit_normal(median, lo, hi)
        samples = expit(np.random.normal(mu, sigma, n))
    else:
        mu, sigma = fit_lognormal(median, lo, hi)
        samples = np.random.lognormal(mu, sigma, n)

    return samples


def main():
    # Baseline elicitations
    baseline_files = sorted(glob.glob('notes/inference/llm_elicitation/llm_elicited_targets_v*.csv'))
    import os
    original = 'notes/inference/llm_elicitation/llm_elicited_targets.csv'
    if os.path.exists(original) and original not in baseline_files:
        baseline_files.insert(0, original)

    # Treatment elicitations
    treatment_files = sorted(glob.glob('notes/inference/llm_elicitation/llm_elicited_treatment_v*.csv'))

    csv_files = baseline_files + treatment_files

    print(f"Found {len(baseline_files)} baseline + {len(treatment_files)} treatment elicitation files:")
    for f in csv_files:
        print(f"  {f}")

    if len(csv_files) < 2:
        print("\nWARNING: Only 1 elicitation found. Generate more with:")
        print('  for i in $(seq 1 10); do')
        print('    claude --allowedTools "" --print "$(cat notes/inference/llm_elicitation/llm_elicitation_prompt.md)" \\')
        print('      > notes/inference/llm_elicitation/llm_elicited_targets_v${i}.csv')
        print('  done')

    # Load all expert DataFrames
    expert_dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df['observable'] = df['observable'].replace(NAME_REMAP)
        expert_dfs.append(df)
        print(f"  Loaded {len(df)} observables from {f}")

    # Get union of all observable names
    all_obs = set()
    for df in expert_dfs:
        all_obs.update(df['observable'].tolist())
    all_obs = sorted(all_obs)

    # For each observable, pool samples from all experts
    np.random.seed(42)
    records = []
    for obs_name in all_obs:
        pooled = []
        n_experts = 0
        for df in expert_dfs:
            rows = df[df['observable'] == obs_name]
            if len(rows) == 0:
                continue
            row = rows.iloc[0]
            samples = sample_expert(row)
            pooled.append(samples)
            n_experts += 1

        if n_experts == 0:
            continue

        pooled = np.concatenate(pooled)
        consensus_median = float(np.median(pooled))
        consensus_lo = float(np.percentile(pooled, 2.5))
        consensus_hi = float(np.percentile(pooled, 97.5))

        # Get units from first expert that has this observable
        units = 'dimensionless'
        for df in expert_dfs:
            rows = df[df['observable'] == obs_name]
            if len(rows) > 0:
                units = str(rows.iloc[0].get('units', 'dimensionless'))
                break

        records.append({
            'observable': obs_name,
            'estimate': consensus_median,
            'ci95_lo': consensus_lo,
            'ci95_hi': consensus_hi,
            'units': units,
            'n_experts': n_experts,
            'reasoning': f'Consensus from {n_experts} LLM expert elicitations (mixture of {n_experts * N_SAMPLES_PER_EXPERT:,} samples)',
        })

    consensus_df = pd.DataFrame(records)
    out_path = 'notes/inference/llm_elicitation/llm_elicited_consensus.csv'
    consensus_df.to_csv(out_path, index=False)
    print(f"\nWrote {len(consensus_df)} consensus targets to {out_path}")

    # Print comparison table
    print(f"\n{'Observable':<45s} {'Consensus':>10s} {'CI95':>20s} {'N experts':>10s}")
    print("-" * 90)
    for _, row in consensus_df.iterrows():
        print(f"  {row['observable']:<43s} {row['estimate']:10.3g} "
              f"[{row['ci95_lo']:.3g}, {row['ci95_hi']:.3g}]{'':<5s} {row['n_experts']:>5d}")


if __name__ == '__main__':
    main()