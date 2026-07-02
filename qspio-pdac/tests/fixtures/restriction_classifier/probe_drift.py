"""Empirical drift check on the cached restriction classifier (gh #57 sub-task).

Loads the 177-feature HistGBM classifier checked in alongside this script
and asks two questions:

1. **Drift impact** — does it still produce high-precision acceptance on the
   current priors? Projects current theta to the classifier's feature order
   (drops new params, fills any retired params with their old prior median),
   scores N samples, and compares predicted-valid to actual evolve-success
   from a real baseline_no_treatment batch. Recall must stay >= 0.95 — see
   ``README.md::DRIFT WATCH``.

2. **Low-d structure** — permutation importance on a held-out subsample to
   see whether the decision collapses to a handful of dominant features
   (which would suggest we could cast the classifier as simple rules).

Usage:
    .venv/bin/python tests/fixtures/restriction_classifier/probe_drift.py [--n 200]
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from qsp_hpc.cpp.batch_runner import CppBatchRunner
from qsp_inference.priors import load_prior

REPO_ROOT = Path(__file__).resolve().parents[3]
CLF_DIR = Path(__file__).resolve().parent
SCENARIO = "baseline_no_treatment"


def load_classifier():
    clf = pickle.load(open(CLF_DIR / "classifier.pkl", "rb"))
    meta = json.load(open(CLF_DIR / "metadata.json"))
    return clf, meta


def project_theta(
    theta_full: np.ndarray,
    full_names: list[str],
    clf_features: list[str],
    fill_value_by_name: dict[str, float],
) -> np.ndarray:
    """Reorder/drop columns of ``theta_full`` to match ``clf_features``.

    Live-only params are dropped (not in clf_features). Classifier-only
    params are filled with the supplied constant (their pre-retirement
    prior median).
    """
    name_to_col = {n: i for i, n in enumerate(full_names)}
    cols = []
    for feat in clf_features:
        if feat in name_to_col:
            cols.append(("live", name_to_col[feat]))
        else:
            cols.append(("fill", fill_value_by_name[feat]))
    out = np.empty((theta_full.shape[0], len(clf_features)), dtype=np.float64)
    for j, (kind, val) in enumerate(cols):
        if kind == "live":
            out[:, j] = theta_full[:, val]
        else:
            out[:, j] = val
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument(
        "--ifng50-fill",
        type=float,
        default=2.07,
        help="value to impute for the retired IFNg_50 (linear-space). "
        "Default 2.07 is a literature plausible IFNg EC50 in nM; the "
        "classifier saw log-transformed values, so this fill enters as "
        "log(2.07) ≈ 0.73.",
    )
    args = ap.parse_args()

    print(f"loading classifier from {CLF_DIR}")
    clf, meta = load_classifier()
    clf_features = meta["feature_order"]
    print(
        f"  classifier: {meta['model_class']}, n_features={meta['n_features']}, "
        f"cv_auc={meta['cv_auc']:.4f}, default_threshold={meta['default_threshold']}"
    )

    print(f"\nsampling {args.n} from CURRENT live priors (seed={args.seed})")
    priors_csv = REPO_ROOT / "parameters/pdac_priors.csv"
    full_names = pd.read_csv(priors_csv)["name"].tolist()
    prior = load_prior(priors_csv)
    torch.manual_seed(args.seed)
    theta_full = prior.sample((args.n,)).numpy().astype(np.float64)
    print(f"  theta_full: {theta_full.shape}, {len(full_names)} live params")

    fill_by_name = {"IFNg_50": args.ifng50_fill}
    theta_clf = project_theta(theta_full, full_names, clf_features, fill_by_name)
    print(f"  theta_clf:  {theta_clf.shape} (projected to classifier feature order)")

    # Classifier predictions
    log_theta = np.log(theta_clf)  # input_transform=log per metadata
    proba = clf.predict_proba(log_theta)[:, 1]
    threshold = meta["default_threshold"]
    pred_valid = proba >= threshold
    print(
        f"\nclassifier @ threshold={threshold}: "
        f"{int(pred_valid.sum())}/{args.n} ({pred_valid.mean()*100:.1f}%) predicted valid"
    )

    # Run baseline_no_treatment to get ground-truth evolve success
    print(f"\nrunning ground-truth {SCENARIO} on all {args.n} draws ...")
    cfg_res = REPO_ROOT / "resources/cpp"
    runner = CppBatchRunner(
        binary_path=str(REPO_ROOT / "cpp/sim/build/qsp_sim"),
        template_path=str(cfg_res / "param_all.xml"),
        subtree="QSP",
        scenario_yaml=REPO_ROOT / "scenarios" / f"{SCENARIO}.yaml",
        drug_metadata_yaml=str(cfg_res / "drug_metadata.yaml"),
        healthy_state_yaml=str(cfg_res / "healthy_state.yaml"),
    )
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "out.parquet"
        runner.run(
            theta_matrix=theta_full,
            param_names=full_names,
            t_end_days=1.0,
            dt_days=1.0,
            output_path=out_path,
            scenario=SCENARIO,
            seed=0,
            workdir=Path(td) / "work",
        )
        df = pq.read_table(str(out_path)).to_pandas()
    elapsed = time.perf_counter() - t0
    actual_valid = (df["status"].to_numpy() == 0)
    print(f"  done in {elapsed:.1f}s ({elapsed/args.n*1000:.0f} ms/draw)")
    print(
        f"  actual evolve_success: {int(actual_valid.sum())}/{args.n} "
        f"({actual_valid.mean()*100:.1f}%)"
    )

    # Confusion matrix
    tp = int((pred_valid & actual_valid).sum())
    fp = int((pred_valid & ~actual_valid).sum())
    fn = int((~pred_valid & actual_valid).sum())
    tn = int((~pred_valid & ~actual_valid).sum())
    print("\nconfusion matrix (rows=predicted, cols=actual):")
    print(f"               actual_valid={int(actual_valid.sum())}  actual_invalid={int((~actual_valid).sum())}")
    print(f"  pred_valid    {tp:5d}            {fp:5d}")
    print(f"  pred_invalid  {fn:5d}            {tn:5d}")
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
        print(f"\n  precision (P[actual_valid | pred_valid]) = {precision:.3f}")
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
        print(f"  recall    (P[pred_valid  | actual_valid]) = {recall:.3f}")

    # Effective speedup: if we use the classifier as a pre-filter, we waste
    # only (1 - precision) of the simulator runs vs (1 - actual_valid_rate)
    # without filtering.
    actual_rate = actual_valid.mean()
    if actual_rate > 0 and (tp + fp) > 0:
        effective_yield_per_sim = precision  # if we sim only predicted-valid
        baseline_yield_per_sim = actual_rate  # if we sim all prior draws
        print(
            f"\n  yield/sim: classifier-filtered = {effective_yield_per_sim:.3f}, "
            f"baseline = {baseline_yield_per_sim:.3f}, "
            f"speedup = {effective_yield_per_sim/baseline_yield_per_sim:.2f}x"
        )

    # Permutation importance — coarse, on log-transformed theta
    print("\npermutation importance (top 15 features, n_repeats=3):")
    rng = np.random.default_rng(0)
    base_score = (clf.predict_proba(log_theta)[:, 1] >= threshold).mean()
    importances = np.zeros(len(clf_features))
    for j in range(len(clf_features)):
        deltas = []
        for _ in range(3):
            shuffled = log_theta.copy()
            rng.shuffle(shuffled[:, j])
            score = (clf.predict_proba(shuffled)[:, 1] >= threshold).mean()
            deltas.append(abs(base_score - score))
        importances[j] = np.mean(deltas)
    order = np.argsort(importances)[::-1][:15]
    cum = 0.0
    total = importances.sum()
    for rank, j in enumerate(order):
        cum += importances[j]
        share = importances[j] / total if total > 0 else 0
        print(
            f"  {rank+1:2d}. {clf_features[j]:30s}  "
            f"Δacceptance={importances[j]:.4f}  share={share*100:.1f}%  cum={cum/total*100:.1f}%"
        )
    nonzero = int((importances > 1e-4).sum())
    print(f"\n  features with non-negligible importance (>1e-4): {nonzero}/{len(clf_features)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
