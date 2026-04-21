#!/usr/bin/env python3
"""
Figures for the vascular_sprout tests.

Two sub-scenarios share this script:

  A. vascular_tip_chemotaxis — 100 TIPs on a slab at x=20 under a pinned
     +x VEGF gradient. No state transitions, no division. Each ABM step
     is one TIP move (vascular_move runs once per step). Measures empirical
     chemotaxis index:   CI = <dx / |displacement|>   over all moves.
     Expected: CI ≈ PARAM_CHEMO_CI_VAS_TIP (0.27).

  B. vascular_sprout_growth — single PHALANX at x=5 under a saturating
     pinned VEGF gradient. Full state_transition + movement + division.
     Tracks network growth: n_tip / n_phalanx / max_x / mean_x_phalanx
     per step. Exposes integrated sprouting + stalk-trail growth rate.

Pass criteria (evaluated below):
  A1. Empirical CI within 5% of PARAM_CHEMO_CI_VAS_TIP.
  A2. n_still < 5% of total moves — TIPs consistently step each turn.
  B1. max_x strictly increases at some point (TIP migration happened).
  B2. Final n_phalanx >> 1 (stalk trail grew) AND monotone non-decreasing.
  B3. First TIP spawns within the first 5 steps (source PHALANX sprouts).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================================
# Common loaders
# ============================================================================

def load_params(path: Path) -> dict:
    pr = pd.read_csv(path)
    out = {}
    for _, row in pr.iterrows():
        try:
            out[row["key"]] = float(row["value"])
        except ValueError:
            out[row["key"]] = row["value"]
    return out


# ============================================================================
# Scenario A: TIP chemotaxis CI
# ============================================================================

def plot_tip_chemotaxis(run_dir: Path) -> bool:
    traj = pd.read_csv(run_dir / "trajectories.csv")
    ci_ts = pd.read_csv(run_dir / "ci_running.csv")
    params = load_params(run_dir / "params.csv")
    target_ci = params["PARAM_CHEMO_CI_VAS_TIP"]

    # ---- CI running plot ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4))

    ax1.plot(ci_ts["step"], ci_ts["ci_running"], lw=1.8, color="tab:blue",
             label="empirical CI (running)")
    ax1.axhline(target_ci, color="tab:red", ls="--", lw=1.2,
                label=f"target = {target_ci:.3f}")
    ax1.axhspan(target_ci * 0.95, target_ci * 1.05,
                color="tab:red", alpha=0.12, label="±5% band")
    ax1.set_xlabel("ABM step")
    ax1.set_ylabel(r"$\langle \cos\theta \rangle = \langle dx/|d| \rangle$")
    ax1.set_title("TIP chemotaxis CI under pinned +x VEGF gradient")
    ax1.legend(loc="lower right", fontsize=9)
    ax1.grid(alpha=0.3)

    # ---- Trajectory <x> over time ----
    mean_x = traj.groupby("step")["x"].mean()
    ax2.plot(mean_x.index, mean_x.values, lw=1.8, color="tab:green",
             label="mean TIP x")
    ax2.axhline(20, color="tab:gray", ls=":", lw=1.0, label="initial x=20")
    ax2.set_xlabel("ABM step")
    ax2.set_ylabel("x (voxels)")
    ax2.set_title("Mean TIP position drifts +x")
    ax2.legend(loc="lower right", fontsize=9)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "tip_chemotaxis.png", dpi=150)
    plt.close(fig)

    # ---- Pass criteria ----
    final_ci = ci_ts["ci_running"].iloc[-1]
    n_moves = int(ci_ts["n_moves"].iloc[-1])
    n_still = int(ci_ts["n_still"].iloc[-1])
    total = n_moves + n_still

    rel_err = abs(final_ci - target_ci) / target_ci
    still_frac = n_still / max(1, total)

    print()
    print("=" * 68)
    print(f"Pass criteria — vascular_tip_chemotaxis")
    print("=" * 68)
    a1 = rel_err <= 0.05
    print(f"  [A1] CI within 5% of target:  "
          f"empirical={final_ci:.4f}  target={target_ci:.4f}  "
          f"rel_err={rel_err:.2%} → {'PASS' if a1 else 'FAIL'}")
    a2 = still_frac < 0.05
    print(f"  [A2] Still fraction < 5%:  still={n_still}/{total} "
          f"({still_frac:.2%}) → {'PASS' if a2 else 'FAIL'}")
    ok = a1 and a2
    print("=" * 68)
    print(f"  OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# Scenario B: Sprout growth
# ============================================================================

def plot_sprout_growth(run_dir: Path) -> bool:
    ts = pd.read_csv(run_dir / "time_series.csv")
    agents = pd.read_csv(run_dir / "agents.csv")
    params = load_params(run_dir / "params.csv")

    # ---- Counts + front + mean-x ----
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))

    ax = axes[0, 0]
    ax.plot(ts["step"], ts["n_tip"], lw=1.8, marker="o", ms=3,
            color="tab:orange", label="TIP")
    ax.plot(ts["step"], ts["n_phalanx"], lw=1.8, marker="s", ms=3,
            color="tab:blue", label="PHALANX")
    ax.plot(ts["step"], ts["n_total"], lw=1.5, ls="--",
            color="tab:gray", label="total")
    ax.set_xlabel("ABM step")
    ax.set_ylabel("n cells")
    ax.set_title("Vessel population — stalk trail from TIP division")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ts["step"], ts["max_x"], lw=1.8, marker="o", ms=3,
            color="tab:red", label="max_x (TIP frontier)")
    ax.plot(ts["step"], ts["mean_x_phalanx"], lw=1.5,
            color="tab:blue", label="mean_x_phalanx")
    ax.axhline(params["source_x"], color="tab:gray", ls=":", lw=1.0,
               label=f"source x={int(params['source_x'])}")
    ax.set_xlabel("ABM step")
    ax.set_ylabel("x (voxels)")
    ax.set_title("Frontier migration + mean stalk position")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # ---- Trajectory scatter: final agent positions colored by state ----
    ax = axes[1, 0]
    final_step = int(agents["step"].max())
    final = agents[agents["step"] == final_step]
    tips = final[final["state"] == 0]     # VAS_TIP
    phals = final[final["state"] == 2]    # VAS_PHALANX
    ax.scatter(phals["x"], phals["y"], s=12, c="tab:blue",
               label=f"PHALANX (n={len(phals)})", alpha=0.6)
    ax.scatter(tips["x"], tips["y"], s=30, c="tab:orange", marker="*",
               label=f"TIP (n={len(tips)})")
    ax.scatter([params["source_x"]], [params["source_y"]],
               s=80, c="tab:red", marker="x", label="source")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("y (voxels)")
    ax.set_title(f"Final network (step {final_step}) — xy projection")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    # ---- Frontier over time: spatial x-histogram vs step ----
    ax = axes[1, 1]
    step_bins = sorted(agents["step"].unique())[::max(1, len(agents["step"].unique()) // 6)]
    cmap = plt.get_cmap("viridis")
    for i, s in enumerate(step_bins):
        sub = agents[agents["step"] == s]
        ax.hist(sub["x"], bins=range(0, 52, 2), histtype="step",
                lw=1.5, color=cmap(i / max(1, len(step_bins) - 1)),
                label=f"step {s}")
    ax.set_xlabel("x (voxels)")
    ax.set_ylabel("count")
    ax.set_title("x-distribution evolution")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(run_dir / "sprout_growth.png", dpi=150)
    plt.close(fig)

    # ---- Pass criteria ----
    print()
    print("=" * 68)
    print(f"Pass criteria — vascular_sprout_growth")
    print("=" * 68)
    b1 = ts["max_x"].iloc[-1] > ts["max_x"].iloc[0]
    print(f"  [B1] max_x increased: {ts['max_x'].iloc[0]} → "
          f"{ts['max_x'].iloc[-1]} → {'PASS' if b1 else 'FAIL'}")

    phal_diffs = np.diff(ts["n_phalanx"].values)
    b2a = bool(np.all(phal_diffs >= 0))
    b2b = int(ts["n_phalanx"].iloc[-1]) > 5
    b2 = b2a and b2b
    print(f"  [B2] n_phalanx monotone ↑ and grew: "
          f"final={int(ts['n_phalanx'].iloc[-1])}, "
          f"min Δ={phal_diffs.min():+d} → {'PASS' if b2 else 'FAIL'}")

    first_tip_step = ts[ts["n_tip"] >= 1]["step"]
    if len(first_tip_step) > 0:
        fts = int(first_tip_step.iloc[0])
        b3 = fts <= 5
        print(f"  [B3] First TIP within 5 steps: at step {fts} "
              f"→ {'PASS' if b3 else 'FAIL'}")
    else:
        b3 = False
        print(f"  [B3] First TIP within 5 steps: NEVER SPAWNED → FAIL")

    ok = b1 and b2 and b3
    print("=" * 68)
    print(f"  OVERALL: {'PASS' if ok else 'FAIL'}")
    return ok


# ============================================================================
# Main
# ============================================================================

def main():
    base = (Path(sys.argv[1]) if len(sys.argv) > 1
            else Path(__file__).parent / "outputs")
    if not base.exists():
        print(f"[error] output dir not found: {base}", file=sys.stderr)
        sys.exit(1)

    all_ok = True

    chemo_dir = base / "vascular_tip_chemotaxis"
    if (chemo_dir / "trajectories.csv").exists():
        print(f"[info] plotting tip_chemotaxis from {chemo_dir}")
        ok = plot_tip_chemotaxis(chemo_dir)
        all_ok &= ok
    else:
        print(f"[skip] {chemo_dir} not populated — run "
              f"`./build/bin/pdac_test vascular_tip_chemotaxis` first")

    growth_dir = base / "vascular_sprout_growth"
    if (growth_dir / "time_series.csv").exists():
        print(f"[info] plotting sprout_growth from {growth_dir}")
        ok = plot_sprout_growth(growth_dir)
        all_ok &= ok
    else:
        print(f"[skip] {growth_dir} not populated — run "
              f"`./build/bin/pdac_test vascular_sprout_growth` first")

    print(f"\n[done] figures written under {base}/")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
