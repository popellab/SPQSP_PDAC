#!/usr/bin/env python3
"""
Representative figures for the chemotaxis_cdf test.

Reads:
  PDAC/test/scenarios/chemotaxis_cdf/outputs/trajectories.csv  (step,id,x,y,z per agent per step)
  PDAC/test/scenarios/chemotaxis_cdf/outputs/ci_running.csv    (step,n_moves,n_still,ci_running)

Writes five PNGs into the same directory:
  ci_convergence.png    — running CI vs step with expected-CI reference line
  x_trajectories.png    — x-position vs step for all agents
  cohort_mean_x.png     — mean x-position of the cohort vs step (linear drift)
  per_step_dx_hist.png  — histogram of per-step +x displacements (bias visible)
  slice_snapshots.png   — CCL5 z=49 slice with agent positions at 5 timepoints

Run from repo root:
  python3 PDAC/test/scenarios/chemotaxis_cdf/make_figures.py
Or with a custom output dir:
  python3 .../make_figures.py <path_to_outputs_dir>
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXPECTED_CI = 0.17  # PARAM_CHEMO_CI_TCELL_EFF


def load(out_dir: Path):
    traj = pd.read_csv(out_dir / "trajectories.csv")
    ci = pd.read_csv(out_dir / "ci_running.csv")
    return traj, ci


def plot_ci_convergence(ci: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ci["step"], ci["ci_running"], lw=1.5, color="tab:blue", label="empirical CI")
    ax.axhline(EXPECTED_CI, color="tab:red", ls="--", lw=1.2,
               label=f"expected CI = {EXPECTED_CI}")
    ax.axhline(EXPECTED_CI - 0.03, color="tab:red", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(EXPECTED_CI + 0.03, color="tab:red", ls=":", lw=0.8, alpha=0.6,
               label="±0.03 tolerance")
    ax.set_xlabel("ABM step")
    ax.set_ylabel(r"running $\langle\cos\theta_{\nabla C}\rangle$")
    ax.set_title("Chemotaxis CDF — running CI convergence")
    ax.set_ylim(-0.05, 0.25)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_x_trajectories(traj: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    # Plot each agent's x(t) as a thin line.
    for aid, g in traj.groupby("id"):
        ax.plot(g["step"], g["x"], lw=0.4, alpha=0.25, color="tab:blue")
    ax.set_xlabel("ABM step")
    ax.set_ylabel("x voxel")
    ax.set_title("Chemotaxis CDF — individual agent x-trajectories")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_cohort_mean_x(traj: pd.DataFrame, out_path: Path):
    mean_x = traj.groupby("step")["x"].mean()
    std_x = traj.groupby("step")["x"].std()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(mean_x.index, mean_x.values, color="tab:blue", lw=1.5, label="mean x")
    ax.fill_between(mean_x.index, mean_x - std_x, mean_x + std_x,
                    color="tab:blue", alpha=0.2, label="±1 std")

    # Linear fit over the whole run; slope ≈ CI × E[step_length].
    steps = mean_x.index.values.astype(float)
    slope, intercept = np.polyfit(steps, mean_x.values, 1)
    ax.plot(steps, slope * steps + intercept, color="tab:red", ls="--", lw=1.2,
            label=f"linear fit: slope = {slope:.3f} vox/step")

    ax.set_xlabel("ABM step")
    ax.set_ylabel("cohort mean x (voxel)")
    ax.set_title("Chemotaxis CDF — mean x drift")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_step_dx_hist(traj: pd.DataFrame, out_path: Path):
    # Per-agent, per-step dx = x(t+1) - x(t).
    traj_sorted = traj.sort_values(["id", "step"])
    dx = traj_sorted.groupby("id")["x"].diff().dropna().values
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bins = np.arange(-1.5, 1.6, 1.0)
    counts, edges, _ = ax.hist(dx, bins=bins, color="tab:blue", alpha=0.8,
                               edgecolor="black", rwidth=0.9)
    mean_dx = dx.mean()
    ax.axvline(mean_dx, color="tab:red", lw=1.5,
               label=f"mean dx = {mean_dx:+.3f}")
    ax.axvline(0.0, color="black", lw=0.8, alpha=0.5)
    ax.set_xlabel("per-step dx (voxel)")
    ax.set_ylabel("count")
    ax.set_title("Chemotaxis CDF — per-step +x displacement distribution")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_slice_snapshots(traj: pd.DataFrame, out_dir: Path, out_path: Path,
                         snapshot_steps=(0, 50, 100, 150, 199),
                         z_slice=49):
    """One background heatmap per panel (CCL5 on z=slice), agents overlaid.
    Agents projected onto (x,y) regardless of z, since the gradient is constant
    along y,z in this test."""
    slice_path = out_dir / "ccl5_slice_z49.csv"
    if not slice_path.exists():
        print(f"[warn] {slice_path} missing — skipping slice_snapshots figure",
              file=sys.stderr)
        return
    slice_df = pd.read_csv(slice_path)
    conc_2d = slice_df.pivot(index="y", columns="x", values="conc").values
    gs_x, gs_y = conc_2d.shape[1], conc_2d.shape[0]

    n = len(snapshot_steps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharey=True)
    vmin, vmax = conc_2d.min(), conc_2d.max()
    im = None
    for ax, step in zip(axes, snapshot_steps):
        im = ax.imshow(conc_2d, origin="lower", aspect="equal", cmap="viridis",
                       vmin=vmin, vmax=vmax, extent=[0, gs_x, 0, gs_y])
        snap = traj[traj["step"] == step]
        ax.scatter(snap["x"] + 0.5, snap["y"] + 0.5, s=6, c="white",
                   edgecolors="black", linewidths=0.3, alpha=0.9)
        ax.set_title(f"step {step}  (n={len(snap)})")
        ax.set_xlabel("x voxel")
        # Gradient direction arrow in the upper-right corner.
        ax.annotate("", xy=(gs_x - 8, gs_y - 10), xytext=(gs_x - 28, gs_y - 10),
                    arrowprops=dict(arrowstyle="->", color="red", lw=1.8))
        ax.text(gs_x - 18, gs_y - 6, r"$\nabla C$", color="red",
                fontsize=10, ha="center", fontweight="bold")
    axes[0].set_ylabel(f"y voxel  (z = {z_slice} slice)")
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8,
                 label="CCL5 concentration")
    fig.suptitle("Chemotaxis CDF — agent drift along CCL5 gradient", y=0.98)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    if len(sys.argv) > 1:
        out_dir = Path(sys.argv[1])
    else:
        out_dir = Path(__file__).parent / "outputs"

    if not out_dir.exists():
        print(f"[error] output dir not found: {out_dir}", file=sys.stderr)
        print("Run the test first: ./build/bin/pdac_test chemotaxis_cdf", file=sys.stderr)
        sys.exit(1)

    traj, ci = load(out_dir)
    print(f"[info] {len(traj)} trajectory rows, {ci['step'].max()+1} steps, "
          f"{traj['id'].nunique()} agents")

    plot_ci_convergence(ci, out_dir / "ci_convergence.png")
    plot_x_trajectories(traj, out_dir / "x_trajectories.png")
    plot_cohort_mean_x(traj, out_dir / "cohort_mean_x.png")
    plot_per_step_dx_hist(traj, out_dir / "per_step_dx_hist.png")
    plot_slice_snapshots(traj, out_dir, out_dir / "slice_snapshots.png")
    print(f"[done] figures written to {out_dir}/")


if __name__ == "__main__":
    main()
