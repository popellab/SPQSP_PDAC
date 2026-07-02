#!/usr/bin/env python
"""Diagnose how a restriction-classifier threshold reshapes the prior.

For each requested threshold τ, computes:
  - n_accepted, accept_frac, observed simulator survival
  - per-parameter KS distance (prior marginal vs accepted marginal)
  - per-parameter log-median shift (accepted - prior, in log space)
  - per-parameter width_ratio (accepted std / prior std, log space)

Use this to pick a threshold that balances HPC sim yield vs prior-coverage loss.
Rule of thumb (from current 50k baseline pool):
  - τ ≤ 0.7: marginals essentially intact (max KS ≤ ~0.15), classifier filters
    via multivariate constraints (joint upper-tails of rate constants that blow
    up the integrator), not by clipping single-param tails.
  - τ ≥ 0.9: real per-parameter squeezing (KS up to 0.36, log-medians shift
    by half a decade) — starts dropping biologically interpretable prior mass.

Usage:
    .venv/bin/python -m workflows.restrict_threshold_geometry \\
        --pool cache/sbi_simulations/multi_scenario_v5_qual_migration_f6cdac33_baseline_no_treatment \\
        --thresholds 0.5 0.7 0.9 \\
        --top 15
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier

from qsp_inference.inference import train_restriction_classifier

ROOT = Path(__file__).resolve().parent.parent


def load_pool(pool_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    paths = glob.glob(str(pool_dir / "test_stats" / "*" / "test_stats.parquet"))
    if not paths:
        raise FileNotFoundError(f"no test_stats.parquet under {pool_dir}")
    df = pd.read_parquet(paths[0]).sort_values("sample_index").reset_index(drop=True)
    param_cols = [c for c in df.columns if c.startswith("param:")]
    x_cols = [c for c in df.columns if c != "sample_index" and not c.startswith("param:")]
    theta = df[param_cols].to_numpy()
    valid = np.all(np.isfinite(df[x_cols].to_numpy()), axis=1)
    names = [c[len("param:"):] for c in param_cols]
    return theta, valid, names


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pool", type=Path, required=True, help="Pool directory containing test_stats/<hash>/test_stats.parquet")
    p.add_argument("--priors-csv", type=Path, default=ROOT / "parameters" / "pdac_priors.csv")
    p.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.7, 0.9])
    p.add_argument("--top", type=int, default=15, help="Top-K most-filtered params per threshold")
    p.add_argument("--max-iter", type=int, default=1000)
    p.add_argument("--max-depth", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=0.03)
    p.add_argument("--l2", type=float, default=1.0)
    p.add_argument("--output-csv", type=Path, default=None, help="Optional CSV dump of full per-param table")
    args = p.parse_args()

    theta, valid, names = load_pool(args.pool)
    log_theta = np.log(theta)
    print(f"Pool: {theta.shape[0]} × {theta.shape[1]} | unfiltered survival {valid.mean():.1%}")

    priors_df = pd.read_csv(args.priors_csv).set_index("name") if args.priors_csv.exists() else None

    clf = train_restriction_classifier(
        theta=theta, valid_mask=valid, feature_order=names,
        input_transform="log", default_threshold=0.5, cv_folds=0,
        model=HistGradientBoostingClassifier(
            max_iter=args.max_iter, max_depth=args.max_depth,
            learning_rate=args.learning_rate, l2_regularization=args.l2, random_state=0,
        ),
    )
    print(f"Classifier in-sample AUC = {clf.in_sample_auc:.4f} "
          f"(HistGBM max_iter={args.max_iter} depth={args.max_depth} l2={args.l2})")

    scores = clf.model.predict_proba(log_theta)[:, 1]
    prior_med = np.median(log_theta, axis=0)
    prior_std = log_theta.std(axis=0)

    rows = []
    for thr in args.thresholds:
        mask = scores >= thr
        n = int(mask.sum())
        if n < 50:
            print(f"\nτ={thr}: only {n} accepted, skipping")
            continue
        acc_med = np.median(log_theta[mask, :], axis=0)
        acc_std = log_theta[mask, :].std(axis=0)
        log_shift = acc_med - prior_med
        width_ratio = np.where(prior_std > 0, acc_std / prior_std, 1.0)

        ks_vals = np.array([ks_2samp(log_theta[:, j], log_theta[mask, j])[0] for j in range(theta.shape[1])])

        for j, name in enumerate(names):
            rows.append({
                "param": name, "thr": thr, "ks": ks_vals[j],
                "log_med_shift": log_shift[j], "width_ratio": width_ratio[j],
            })

        print(f"\n=== τ={thr}: {n}/{len(mask)} ({n/len(mask):.1%}) accepted, "
              f"survival={valid[mask].mean():.1%}, max-KS={ks_vals.max():.3f} ===")
        order = np.argsort(-ks_vals)[:args.top]
        sub = pd.DataFrame({
            "param": [names[j] for j in order],
            "ks": ks_vals[order].round(3),
            "log_med_shift": log_shift[order].round(3),
            "width_ratio": width_ratio[order].round(3),
        })
        if priors_df is not None:
            sub["prior_median"] = [priors_df.loc[p, "median"] if p in priors_df.index else np.nan for p in sub["param"]]
            sub["prior_dp2"] = [priors_df.loc[p, "dist_param2"] if p in priors_df.index else np.nan for p in sub["param"]]
        print(sub.to_string(index=False))

    if args.output_csv:
        pd.DataFrame(rows).to_csv(args.output_csv, index=False)
        print(f"\nFull per-param table → {args.output_csv}")


if __name__ == "__main__":
    main()
