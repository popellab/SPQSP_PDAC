#!/usr/bin/env python
"""Quick experiment: can a classifier trained on (theta, is_valid) pairs from
the existing 100k pool predict which fresh theta draws will survive the joint
NaN filter? If yes, the effective survival rate at inference time can be raised
above the current ~54% by rejecting low-p(valid) samples before simulating.

Nothing is trained into the SBI pipeline; this is just a yield-diagnosis script.
"""
from pathlib import Path
import glob

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
POOLS = {
    'baseline_no_treatment':        'cache/sbi_simulations/multi_scenario_v4_all_scenarios_9cfdff18_baseline_no_treatment/test_stats/*/test_stats.parquet',
    'clinical_progression':         'cache/sbi_simulations/multi_scenario_v4_all_scenarios_03246360_clinical_progression/test_stats/*/test_stats.parquet',
    'gvax_neoadjuvant_zheng2022':   'cache/sbi_simulations/multi_scenario_v4_all_scenarios_4fb91adf_gvax_neoadjuvant_zheng2022/test_stats/*/test_stats.parquet',
    'gvax_nivo_neoadjuvant_zheng2022':'cache/sbi_simulations/multi_scenario_v4_all_scenarios_4dae3a3e_gvax_nivo_neoadjuvant_zheng2022/test_stats/*/test_stats.parquet',
}


def load_pool_and_mask():
    theta = None
    valid = None
    for scen, pattern in POOLS.items():
        paths = glob.glob(str(ROOT / pattern))
        df = pd.read_parquet(paths[0]).sort_values('sample_index').reset_index(drop=True)
        param_cols = [c for c in df.columns if c.startswith('param:')]
        x_cols = [c for c in df.columns if c != 'sample_index' and not c.startswith('param:')]
        if theta is None:
            theta = df[param_cols].to_numpy()
            names = [c[len('param:'):] for c in param_cols]
        v = np.all(np.isfinite(df[x_cols].to_numpy()), axis=1)
        valid = v if valid is None else (valid & v)
    return theta, valid, names


def main():
    theta, valid, names = load_pool_and_mask()
    print(f"Pool: {theta.shape[0]} theta × {theta.shape[1]} params")
    print(f"Joint survival: {valid.sum()}/{len(valid)} ({valid.mean():.1%})")

    # Classifier on log(theta) — priors are log-normal so log-space is natural.
    X = np.log(theta)
    y = valid.astype(np.int32)

    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
    clf = HistGradientBoostingClassifier(
        max_iter=400, max_depth=5, learning_rate=0.05, random_state=0,
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, p)
    print(f"\nClassifier AUC on held-out 20k: {auc:.4f}")

    # Rejection-rate / yield curve: at threshold τ, accept if p(valid) >= τ.
    # Yield = accepted | valid = true_positive_rate at that threshold.
    print("\nAt threshold τ:")
    print(f"  {'τ':>5s} | {'accept_frac':>12s} | {'survival_on_accepted':>22s} | {'yield_per_sim':>13s}")
    print(f"  {'-'*5} | {'-'*12} | {'-'*22} | {'-'*13}")
    for tau in (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
        accept = p >= tau
        accept_frac = accept.mean()
        if accept_frac == 0:
            continue
        surv = yte[accept].mean()
        yield_per_sim = accept_frac * surv  # usable sims per prior draw
        print(f"  {tau:>5.2f} | {accept_frac:>12.3f} | {surv:>22.3f} | {yield_per_sim:>13.3f}")

    # Economic question: how many prior draws needed to get 50k valid sims?
    # Baseline: 50000 / 0.54 ≈ 92.6k draws.
    print("\nPrior draws needed for 50,000 usable sims:")
    baseline = 50000 / yte.mean()
    print(f"  baseline (accept all):              {baseline:>7.0f}  (survival {yte.mean():.1%})")
    for tau in (0.5, 0.6, 0.7, 0.8, 0.9):
        accept = p >= tau
        if accept.sum() == 0:
            continue
        accept_frac = accept.mean()
        surv = yte[accept].mean()
        yield_per_sim = accept_frac * surv
        # Need n_draws such that n_draws * accept_frac * surv = 50000, but only
        # the accepted ones are simulated; so n_draws * accept_frac sims.
        sims_needed = 50000 / surv
        draws_needed = sims_needed / accept_frac
        savings = 1 - sims_needed / baseline
        print(f"  τ={tau:.2f}: {draws_needed:>7.0f} prior draws, {sims_needed:>7.0f} sims ({savings:+.1%} fewer sims)")

    # Top-20 features via permutation importance (HGB has no feature_importances_)
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(clf, Xte[:5000], yte[:5000], n_repeats=3,
                                 random_state=0, n_jobs=-1)
    ranking = np.argsort(imp.importances_mean)[::-1]
    print("\nTop 20 params by permutation importance (AUC drop):")
    for i in ranking[:20]:
        print(f"  {names[i]:<30s}  {imp.importances_mean[i]:.4f}")


if __name__ == "__main__":
    main()
