"""Train vpop neural network surrogate from pool results.

Reads pool_results.npz (from run_vpop_pool.py), trains one surrogate arm
per scenario, and saves the trained arms for downstream fitting.

Usage
-----
    .venv/bin/python -I scripts/train_vpop_surrogate.py --pool-dir results/2026-10-16_vpop_pool_30k
    .venv/bin/python -I scripts/train_vpop_surrogate.py --pool-dir results/vpop_pool
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vpop.surrogate.train import train_arm, save_arm, score_arm

from vpop_readouts import SP, N_SPECIES, SCENARIOS, READOUTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


REPO = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description="Train vpop surrogate from pool results")
    parser.add_argument("--pool-dir", type=str,
                        default=str(REPO / "results" / "vpop_pool"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--hidden", type=str, default="512,512,256",
                        help="Comma-separated hidden layer sizes")
    args = parser.parse_args()

    hidden = tuple(int(h) for h in args.hidden.split(","))

    pool_dir = Path(args.pool_dir)
    results_path = pool_dir / "pool_results.npz"
    if not results_path.exists():
        log.error("No pool_results.npz found at %s", results_path)
        log.error("Run scripts/run_vpop_pool.py first.")
        return 1

    log.info("Loading pool results from %s", results_path)
    data = np.load(results_path, allow_pickle=True)
    log_theta = data["log_theta"]          # (N, P)
    status = data["status"]                # (N,)
    param_names = tuple(str(s) for s in data["param_names"])
    scenario_names = tuple(str(s) for s in data["scenario_names"])

    # Load species arrays dynamically from whatever scenarios are in the npz
    species_arrays = {}
    for s_name in scenario_names:
        key = f"species_{s_name}"
        if key in data:
            species_arrays[s_name] = data[key]
        else:
            log.warning("Missing species array for scenario %s", s_name)

    N, P = log_theta.shape
    n_ok = (status == 0).sum()
    log.info("Pool: %d total, %d admissible (%.1f%%)", N, n_ok, 100 * n_ok / max(N, 1))
    log.info("Parameters: %d, Species: %d, Scenarios: %d",
             P, species_arrays[scenario_names[0]].shape[1], len(scenario_names))
    log.info("Hidden layers: %s", hidden)

    if n_ok < 50:
        log.error("Too few admissible samples (%d). Need at least 50 for training.", n_ok)
        return 1

    sp_names = tuple(sp.name for sp in SP)

    arms_dir = pool_dir / "arms"
    arms_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    for s_idx, s_name in enumerate(scenario_names):
        if s_name not in species_arrays:
            log.warning("Skipping %s (no species array)", s_name)
            continue

        log.info("=" * 60)
        log.info("Training arm: %s (%d/%d)", s_name, s_idx + 1, len(scenario_names))
        log.info("=" * 60)

        Y = species_arrays[s_name].copy()  # (N, Q)

        # Trajectory columns that are all-NaN for this scenario (not filled by
        # the pool) would make train_arm drop every row.  Fill them with 1.0
        # (neutral in log space) so the net sees a finite target; the readouts
        # function never reads these columns for this scenario.
        ok_mask = status == 0
        for col in range(Y.shape[1]):
            if np.all(np.isnan(Y[ok_mask, col])):
                Y[:, col] = 1.0
                log.debug("  filled all-NaN column %d (%s) with 1.0",
                          col, sp_names[col])

        arm = train_arm(
            X=log_theta,
            Y=Y,
            status=status,
            param_names=param_names,
            target_names=sp_names,
            reject_codes=(2,),
            seed=args.seed + s_idx,
            max_epochs=args.epochs,
            patience=args.patience,
            hidden=hidden,
            verbose=True,
        )

        # Report key metrics
        sc = arm.scores
        log.info("Arm %s training complete:", s_name)
        log.info("  Epochs: species=%d, status=%d",
                 sc.get("epochs", 0), sc.get("epochs_status", 0))
        log.info("  Training rows: %d total, %d with species",
                 sc.get("n_train", 0), sc.get("n_train_species", 0))

        sp_test = sc.get("species_test", {})
        if sp_test:
            r2 = sp_test.get("r2")
            shrink = sp_test.get("shrinkage")
            if r2 is not None:
                finite_r2 = r2[np.isfinite(r2)]
                log.info("  Test R² (log): median=%.3f, p10=%.3f, p90=%.3f",
                         np.median(finite_r2), np.percentile(finite_r2, 10),
                         np.percentile(finite_r2, 90))
            if shrink is not None:
                finite_s = shrink[np.isfinite(shrink)]
                log.info("  Test shrinkage: median=%.3f, p10=%.3f, p90=%.3f",
                         np.median(finite_s), np.percentile(finite_s, 10),
                         np.percentile(finite_s, 90))

        st_test = sc.get("status_test", {})
        if st_test:
            log.info("  Status accuracy=%.3f, balanced=%.3f",
                     st_test.get("accuracy", 0), st_test.get("balanced_accuracy", 0) or 0)

        arm_path = save_arm(arms_dir / s_name, arm)
        log.info("  Saved to %s", arm_path)

        summary[s_name] = {
            "epochs": sc.get("epochs", 0),
            "r2_median": float(np.nanmedian(sp_test.get("r2", [np.nan]))),
            "shrinkage_median": float(np.nanmedian(sp_test.get("shrinkage", [np.nan]))),
            "status_balanced_acc": st_test.get("balanced_accuracy", None),
        }

    # Write summary JSON
    summary_path = arms_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("Training summary saved to %s", summary_path)

    log.info("=" * 60)
    log.info("All %d arms trained and saved to %s", len(summary), arms_dir)
    for s_name, s in summary.items():
        log.info("  %-25s  R²=%.3f  shrink=%.3f  status_bal_acc=%.3f  epochs=%d",
                 s_name, s["r2_median"], s["shrinkage_median"],
                 s["status_balanced_acc"] or 0, s["epochs"])
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
