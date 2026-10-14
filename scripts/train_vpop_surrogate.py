"""Train vpop neural network surrogate from pool results.

Reads pool_results.npz (from run_vpop_pool.py), trains one surrogate arm
per scenario, and saves the trained arms for downstream fitting.

Usage
-----
    .venv/bin/python -I scripts/train_vpop_surrogate.py
    .venv/bin/python -I scripts/train_vpop_surrogate.py --pool-dir results/vpop_pool
"""
from __future__ import annotations

import argparse
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
    args = parser.parse_args()

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

    species_arrays = {
        "baseline": data["species_baseline"],       # (N, Q)
        "gvax_d21": data["species_gvax"],           # (N, Q)
        "gvax_nivo_d21": data["species_gvax_nivo"], # (N, Q)
    }

    N = log_theta.shape[0]
    n_ok = (status == 0).sum()
    log.info("Pool: %d total, %d admissible (%.1f%%)", N, n_ok, 100 * n_ok / max(N, 1))

    if n_ok < 50:
        log.error("Too few admissible samples (%d). Need at least 50 for training.", n_ok)
        return 1

    # Species column names for targets
    sp_names = tuple(sp.name for sp in SP)

    # Train one arm per scenario
    arms_dir = pool_dir / "arms"
    arms_dir.mkdir(parents=True, exist_ok=True)

    for s_idx, s_name in enumerate(scenario_names):
        log.info("=" * 60)
        log.info("Training arm: %s", s_name)
        log.info("=" * 60)

        Y = species_arrays[s_name]  # (N, Q)

        # Status 2 = EVOLVE_TOO_FAST (patient ineligible, not solver failure)
        # Status -6 = CVODE failure (solver failure)
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
            verbose=True,
        )

        # Score the arm
        scores = score_arm(arm, log_theta, Y, status)
        log.info("Arm %s scores:", s_name)
        for k, v in scores.items():
            if isinstance(v, (int, float)):
                log.info("  %s: %.4f", k, v)
            elif isinstance(v, np.ndarray) and v.ndim == 0:
                log.info("  %s: %.4f", k, float(v))

        # Save the arm
        arm_path = save_arm(arms_dir / s_name, arm)
        log.info("Arm saved to %s", arm_path)

    log.info("=" * 60)
    log.info("All arms trained and saved to %s", arms_dir)
    log.info("Next: wire FunctionSurrogate with readouts function for fitting")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
