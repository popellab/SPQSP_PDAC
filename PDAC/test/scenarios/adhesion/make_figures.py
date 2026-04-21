#!/usr/bin/env python3
"""
Representative figure for the adhesion slope-sweep test.

Reads:
  PDAC/test/scenarios/adhesion/outputs/slope_sweep.csv
    columns: k, expected_p, observed_p, n_focals, err

Writes:
  slope_sweep.png — observed vs expected p_move across the k sweep,
                    with binomial 2σ error bars.

Test design (see test_adhesion.cu header): CANCER_STEM focal + k CANCER_PROG
decoy neighbours, M[stem][prog] = 0.15. Under the adhesion formula
  p_move = max(0, 1 - sum_j M[i][j] * n_j)
the expected move rate at each k is 1 - 0.15*k, clamped at 0.

Run from repo root:
  python3 PDAC/test/scenarios/adhesion/make_figures.py
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

M_COEFF = 0.15
TOL = 0.10


def load(out_dir: Path) -> pd.DataFrame:
    csv = out_dir / "slope_sweep.csv"
    if not csv.exists():
        sys.exit(f"[adhesion] missing {csv} — run ./build/bin/pdac_test adhesion first")
    return pd.read_csv(csv)


def plot_slope_sweep(df: pd.DataFrame, out_path: Path):
    k = df["k"].to_numpy()
    obs = df["observed_p"].to_numpy()
    n = df["n_focals"].to_numpy().astype(float)
    sigma = np.sqrt(np.clip(obs * (1 - obs), 0, None) / np.maximum(n, 1))

    k_dense = np.linspace(0, k.max(), 200)
    expected_dense = np.clip(1.0 - M_COEFF * k_dense, 0, None)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(k_dense, expected_dense, lw=2.0, color="tab:blue",
            label=f"analytic: max(0, 1 - {M_COEFF}·k)")
    ax.errorbar(k, obs, yerr=2 * sigma, fmt="o", color="tab:red",
                markersize=7, capsize=4, lw=1.5,
                label=f"observed (n={int(n[0])} per k, 2σ)")
    ax.fill_between(k_dense,
                    np.clip(expected_dense - TOL, 0, 1),
                    np.clip(expected_dense + TOL, 0, 1),
                    color="tab:blue", alpha=0.12, label=f"±{TOL} pass band")

    ax.set_xlabel("k  (same-type neighbours)")
    ax.set_ylabel("p_move  (move rate at step 1)")
    ax.set_title("Adhesion slope sweep — STEM focal + PROG decoys  "
                 f"M[stem][prog]={M_COEFF}")
    ax.set_ylim(-0.05, 1.1)
    ax.set_xticks(k)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir / "outputs"
    df = load(out_dir)
    plot_slope_sweep(df, out_dir / "slope_sweep.png")
    print(f"[adhesion] wrote {out_dir / 'slope_sweep.png'}")

    err = (df["observed_p"] - df["expected_p"]).abs()
    print(f"[adhesion] max |err| = {err.max():.4f} "
          f"(tolerance {TOL}); mean = {err.mean():.4f}")


if __name__ == "__main__":
    main()
