#!/usr/bin/env python3
"""
Figures for the vascular_o2_saturation test.

Setup: single PHALANX vessel at center of 21³ grid, no tissue uptake,
background decay λ=1e-5/s only. Steady state is mass-balance-limited:
   C_ss = C_blood / (1 + λ·N·V/KvLv)
This is NOT simple saturation at C_blood — every grid voxel decays, and the
single vessel must supply the entire grid's decay loss.

Reads:
  outputs/time_series.csv  (step, vessel_conc, min_conc, max_conc, mean_conc)
  outputs/xline.csv        (step, x, conc) — snapshot steps only
  outputs/params.csv       (key, value)

Writes:
  outputs/saturation_time_series.png   vessel/min/max/mean vs step + asymptote
  outputs/saturation_xline.png         x-line profile at snapshot steps
  outputs/saturation_mass_balance.png  observed vs analytic mass-balance ratio

Pass criteria (per the test header):
  (P1) No overshoot: max_conc ≤ C_blood at every step.
  (P2) Vessel approach monotonically non-decreasing (no oscillation).
  (P3) (max − min) / mean < 1e-3 at final step (near-uniform field).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load(out_dir: Path):
    ts = pd.read_csv(out_dir / "time_series.csv")
    xl = pd.read_csv(out_dir / "xline.csv")
    pr = pd.read_csv(out_dir / "params.csv")
    params = {}
    for _, row in pr.iterrows():
        try:
            params[row["key"]] = float(row["value"])
        except ValueError:
            params[row["key"]] = row["value"]
    return ts, xl, params


def plot_time_series(ts: pd.DataFrame, params: dict, out_path: Path):
    c_blood = params["PARAM_VAS_O2_CONC"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax1.plot(ts["step"], ts["vessel_conc"], lw=1.8, marker="o", ms=3,
             color="tab:blue", label="vessel voxel")
    ax1.plot(ts["step"], ts["mean_conc"], lw=1.5, ls="--",
             color="tab:cyan", label="grid mean")
    ax1.axhline(c_blood, color="tab:red", ls=":", lw=1.2,
                label=f"C_blood = {c_blood:g}")
    final_vessel = ts["vessel_conc"].iloc[-1]
    ax1.axhline(final_vessel, color="tab:gray", ls=":", lw=1.0,
                label=f"observed asymptote = {final_vessel:.3e}")
    ax1.set_ylabel("O₂ [mM]")
    ax1.set_title("Vessel approach to mass-balance steady state "
                  "(C_blood >> C_ss due to whole-grid decay)")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)

    # Zoom into vessel vs min to check uniformity
    ax2.plot(ts["step"], ts["vessel_conc"] - ts["min_conc"],
             lw=1.8, marker="o", ms=3, color="tab:purple",
             label="vessel − min (uniformity proxy)")
    ax2.plot(ts["step"], ts["max_conc"] - ts["min_conc"],
             lw=1.2, ls="--", color="tab:orange",
             label="max − min")
    ax2.set_xlabel("ABM step")
    ax2.set_ylabel("Δ O₂ [mM]")
    ax2.set_title("Grid uniformity — small Δ means diffusion well-mixed the grid")
    ax2.legend(loc="upper right", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_xline(xl: pd.DataFrame, params: dict, out_path: Path):
    cx = int(params["vessel_x"])
    c_blood = params["PARAM_VAS_O2_CONC"]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    snap_steps = sorted(xl["step"].unique())
    cmap = plt.get_cmap("viridis")
    for i, step in enumerate(snap_steps):
        g = xl[xl["step"] == step].sort_values("x")
        r = g["x"].values - cx
        ax.plot(r, g["conc"].values, lw=1.5, marker="o", ms=3,
                color=cmap(i / max(1, len(snap_steps) - 1)),
                label=f"step {step}")
    ax.axhline(c_blood, color="tab:red", ls=":", lw=1.0, label="C_blood")
    ax.set_xlabel(r"$x - x_{\mathrm{vessel}}$ (voxels)")
    ax.set_ylabel("O₂ [mM]")
    ax.set_title("x-line cross-section at snapshot steps (near-uniform at SS)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mass_balance(ts: pd.DataFrame, params: dict, out_path: Path):
    """Reverse-compute the effective KvLv implied by the observed steady state
    and compare to the test's physical expectation (mass balance only)."""
    c_blood = params["PARAM_VAS_O2_CONC"]
    lam = params["PARAM_O2_DECAY_RATE"]
    gs = int(params["grid_size"])
    vox_cm = params["PARAM_VOXEL_SIZE_CM"]
    N = gs ** 3
    V_vox = vox_cm ** 3               # cm³ per voxel
    c_ss = float(ts["vessel_conc"].iloc[-1])

    # From C_ss = C_blood / (1 + λ·N·V/KvLv):
    ratio = c_blood / c_ss - 1.0
    kvlv_implied = lam * N * V_vox / max(ratio, 1e-30)

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ax.plot(ts["step"], ts["vessel_conc"] / c_blood, lw=1.8, marker="o", ms=3,
            color="tab:blue", label="vessel / C_blood")
    ax.axhline(1.0, color="tab:red", ls=":", lw=1.0, label="saturation (1.0)")
    ax.axhline(c_ss / c_blood, color="tab:gray", ls="--", lw=1.0,
               label=f"observed asymptote = {c_ss/c_blood:.4f}")
    ax.set_xlabel("ABM step")
    ax.set_ylabel("vessel / C_blood")
    ax.set_title(
        f"Fraction of C_blood reached — mass balance dominates\n"
        f"λ·N·V/KvLv ≈ {ratio:.2f}, implied KvLv ≈ {kvlv_implied:.2e} cm³/s"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_pass_criteria(ts: pd.DataFrame, params: dict):
    c_blood = params["PARAM_VAS_O2_CONC"]
    ok = True
    print()
    print("=" * 68)
    print("Pass criteria")
    print("=" * 68)

    # P1: No overshoot
    max_over = float(ts["max_conc"].max()) - c_blood
    p1 = max_over <= 1e-9 * c_blood
    print(f"  [P1] max_conc ≤ C_blood: max_conc={ts['max_conc'].max():.4e}, "
          f"C_blood={c_blood:.4e} → {'PASS' if p1 else 'FAIL'}")
    ok &= p1

    # P2: Monotone non-decreasing vessel conc
    diffs = np.diff(ts["vessel_conc"].values)
    p2 = bool(np.all(diffs >= -1e-9))
    print(f"  [P2] vessel_conc monotone ↑: min Δ={diffs.min():+.3e} "
          f"→ {'PASS' if p2 else 'FAIL'}")
    ok &= p2

    # P3: Near-uniform at final step
    final = ts.iloc[-1]
    spread = (final["max_conc"] - final["min_conc"]) / final["mean_conc"]
    p3 = spread < 1e-3
    print(f"  [P3] (max−min)/mean < 1e-3 at final: {spread:.3e} "
          f"→ {'PASS' if p3 else 'FAIL'}")
    ok &= p3

    print("=" * 68)
    print(f"  OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 68)
    return ok


def main():
    out_dir = (Path(sys.argv[1]) if len(sys.argv) > 1
               else Path(__file__).parent / "outputs")
    if not out_dir.exists():
        print(f"[error] output dir not found: {out_dir}", file=sys.stderr)
        print("Run the test first: ./build/bin/pdac_test vascular_o2_saturation",
              file=sys.stderr)
        sys.exit(1)

    ts, xl, params = load(out_dir)
    print(f"[info] {len(ts)} time-series rows, {len(xl)} x-line rows, "
          f"{len(params)} params")

    plot_time_series(ts, params, out_dir / "saturation_time_series.png")
    plot_xline(xl, params, out_dir / "saturation_xline.png")
    plot_mass_balance(ts, params, out_dir / "saturation_mass_balance.png")

    ok = evaluate_pass_criteria(ts, params)
    print(f"\n[done] 3 figures written to {out_dir}/")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
