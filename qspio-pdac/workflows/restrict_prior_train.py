#!/usr/bin/env python
"""Train and persist the restriction classifier used to reject evolve-to-diagnosis
failures before simulation.

Loads (theta, valid_mask) from the 4 scenario test_stats parquets, fits the
classifier via qsp_inference.inference.train_restriction_classifier, saves to
``cache/restriction_classifier/``.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from qsp_inference.inference import train_restriction_classifier

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "cache" / "restriction_classifier"

POOLS = {
    "baseline_no_treatment": "cache/sbi_simulations/multi_scenario_v5_qual_migration_3a0063dd_baseline_no_treatment/test_stats/*/test_stats.parquet",
}


def load_theta_and_valid_mask():
    theta = None
    valid = None
    names = None
    for scen, pattern in POOLS.items():
        paths = glob.glob(str(ROOT / pattern))
        df = pd.read_parquet(paths[0]).sort_values("sample_index").reset_index(drop=True)
        param_cols = [c for c in df.columns if c.startswith("param:")]
        x_cols = [c for c in df.columns if c != "sample_index" and not c.startswith("param:")]
        if theta is None:
            theta = df[param_cols].to_numpy()
            names = [c[len("param:"):] for c in param_cols]
        v = np.all(np.isfinite(df[x_cols].to_numpy()), axis=1)
        valid = v if valid is None else (valid & v)
    return theta, valid, names


def main() -> None:
    theta, valid, names = load_theta_and_valid_mask()
    print(f"Pool: {theta.shape[0]} theta × {theta.shape[1]} params")
    print(f"Joint survival: {valid.sum()}/{len(valid)} ({valid.mean():.1%})")

    # deeper + l2 config selected via workflows/restrict_threshold_geometry.py
    # sweep on the f6cdac33 50k pool: +0.014 CV AUC vs default, smallest
    # overfit gap of the high-capacity configs.
    model = HistGradientBoostingClassifier(
        max_iter=1000, max_depth=8, learning_rate=0.03,
        l2_regularization=1.0, random_state=0,
    )
    clf = train_restriction_classifier(
        theta=theta, valid_mask=valid, feature_order=names,
        input_transform="log", default_threshold=0.5, cv_folds=5, model=model,
    )
    print(f"5-fold CV AUC: {clf.cv_auc:.4f}")
    print(f"In-sample AUC: {clf.in_sample_auc:.4f}")
    print(f"Threshold curve: {clf.threshold_curve}")

    clf.save(OUT_DIR)
    print(f"\nSaved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
