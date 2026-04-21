#!/usr/bin/env python3
"""
Representative figures for the persistence test.

Reads:
  PDAC/test/scenarios/persistence/outputs/trajectories.csv      (step,id,x,y,z)
  PDAC/test/scenarios/persistence/outputs/persist_running.csv   (step,n_pairs,n_exact_match,mean_cos)
  PDAC/test/scenarios/persistence/outputs/per_cell_autocorr.csv (id,n_pairs,mean_cos)

Writes five PNGs:
  persist_convergence.png     — running <cos(θ)> vs step, expected reference line
  per_cell_autocorr_hist.png  — per-cell mean cos distribution
  cos_distribution.png        — full distribution of pair cos values (spike at 1 + bulk ~0)
  trajectories_2d.png         — xy projection of sampled single-cell paths (visible runs)
  position_snapshots.png      — cohort positions at 5 timepoints (isotropic spread)

Run from repo root:
  python3 PDAC/test/scenarios/persistence/make_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

EXPECTED = 0.4  # PARAM_PERSIST_TCELL_EFF
TOL = 0.03


def load(out_dir: Path):
    traj = pd.read_csv(out_dir / "trajectories.csv")
    running = pd.read_csv(out_dir / "persist_running.csv")
    per_cell = pd.read_csv(out_dir / "per_cell_autocorr.csv")
    return traj, running, per_cell


def plot_persist_convergence(running: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(running["step"], running["mean_cos"], lw=1.5, color="tab:blue",
            label=r"empirical $\langle\cos\theta\rangle$")
    ax.axhline(EXPECTED, color="tab:red", ls="--", lw=1.2,
               label=f"expected = {EXPECTED}")
    ax.axhline(EXPECTED - TOL, color="tab:red", ls=":", lw=0.8, alpha=0.6)
    ax.axhline(EXPECTED + TOL, color="tab:red", ls=":", lw=0.8, alpha=0.6,
               label=f"±{TOL} tolerance")
    ax.set_xlabel("ABM step")
    ax.set_ylabel(r"running $\langle\cos\theta_{t,t+1}\rangle$")
    ax.set_title("Persistence — running lag-1 autocorrelation")
    ax.set_ylim(0.0, 0.7)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_per_cell_autocorr_hist(per_cell: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(7, 4.2))
    vals = per_cell["mean_cos"].values
    ax.hist(vals, bins=25, color="tab:blue", alpha=0.8,
            edgecolor="black", rwidth=0.95)
    mean = vals.mean()
    ax.axvline(EXPECTED, color="tab:red", ls="--", lw=1.5,
               label=f"expected = {EXPECTED}")
    ax.axvline(mean, color="tab:green", ls="-", lw=1.2,
               label=f"population mean = {mean:.3f}")
    ax.set_xlabel(r"per-cell mean $\cos\theta_{t,t+1}$")
    ax.set_ylabel("cells")
    ax.set_title(f"Persistence — per-cell autocorrelation (n={len(vals)})")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _compute_pair_cos(traj: pd.DataFrame) -> np.ndarray:
    """Consecutive-displacement cos(θ) values across all cells."""
    traj_sorted = traj.sort_values(["id", "step"])
    vals = []
    for _, g in traj_sorted.groupby("id"):
        pos = g[["x", "y", "z"]].values
        d = np.diff(pos, axis=0)                       # per-step displacement
        # Drop zero displacements; break chain around stills.
        mag = np.sqrt((d * d).sum(axis=1))
        # Pair only where both d_t and d_{t+1} are non-zero.
        for i in range(len(d) - 1):
            if mag[i] > 0 and mag[i + 1] > 0:
                vals.append((d[i] * d[i + 1]).sum() / (mag[i] * mag[i + 1]))
    return np.array(vals)


def plot_cos_distribution(traj: pd.DataFrame, out_path: Path):
    cos_vals = _compute_pair_cos(traj)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    # On the 26-Moore lattice, cos takes a small number of discrete values.
    # 30 bins across [-1, 1] captures the spike at 1 clearly.
    ax.hist(cos_vals, bins=np.linspace(-1.02, 1.02, 52),
            color="tab:blue", alpha=0.8, edgecolor="black", rwidth=0.95)
    mean = cos_vals.mean()
    ax.axvline(mean, color="tab:red", lw=1.5,
               label=f"mean = {mean:+.3f}")
    ax.axvline(EXPECTED, color="tab:green", ls="--", lw=1.2,
               label=f"expected = {EXPECTED}")
    ax.set_xlabel(r"pair $\cos\theta_{t,t+1}$")
    ax.set_ylabel("count")
    ax.set_title(f"Persistence — distribution of pair cos values (n={len(cos_vals)})")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_trajectories_2d(traj: pd.DataFrame, out_path: Path, n_sample=10):
    rng = np.random.default_rng(0)
    ids = traj["id"].unique()
    sample_ids = rng.choice(ids, size=min(n_sample, len(ids)), replace=False)

    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    for i, aid in enumerate(sample_ids):
        g = traj[traj["id"] == aid].sort_values("step")
        ax.plot(g["x"], g["y"], lw=0.9, color=cmap(i % 10),
                alpha=0.85, label=f"id {aid}")
        ax.scatter(g["x"].iloc[0], g["y"].iloc[0], s=30, color=cmap(i % 10),
                   marker="o", edgecolors="black", linewidths=0.5, zorder=3)
        ax.scatter(g["x"].iloc[-1], g["y"].iloc[-1], s=50, color=cmap(i % 10),
                   marker="*", edgecolors="black", linewidths=0.5, zorder=3)
    ax.set_xlabel("x voxel")
    ax.set_ylabel("y voxel")
    ax.set_aspect("equal")
    ax.set_title(f"Persistence — single-cell xy paths ({len(sample_ids)} random cells)\n"
                 "○ = start, ★ = end (correlated runs are visible as straight segments)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_position_snapshots(traj: pd.DataFrame, out_path: Path,
                            snapshot_steps=(0, 50, 100, 150, 199)):
    n = len(snapshot_steps)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), sharex=True, sharey=True)
    # Determine common xy extents from the union of all snapshot positions.
    snaps = [traj[traj["step"] == s] for s in snapshot_steps]
    all_xy = np.vstack([snap[["x", "y"]].values for snap in snaps])
    pad = 3
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    for ax, step, snap in zip(axes, snapshot_steps, snaps):
        ax.scatter(snap["x"], snap["y"], s=8, c="tab:blue",
                   edgecolors="black", linewidths=0.2, alpha=0.85)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        ax.set_title(f"step {step}  (n={len(snap)})")
        ax.set_xlabel("x voxel")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("y voxel (projected)")
    fig.suptitle("Persistence — cohort spread is isotropic (no net drift)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "outputs"
    if not out_dir.exists():
        print(f"[error] output dir not found: {out_dir}", file=sys.stderr)
        print("Run the test first: ./build/bin/pdac_test persistence", file=sys.stderr)
        sys.exit(1)

    traj, running, per_cell = load(out_dir)
    print(f"[info] {len(traj)} trajectory rows, {running['step'].max()+1} steps, "
          f"{traj['id'].nunique()} agents, {len(per_cell)} per-cell records")

    plot_persist_convergence(running, out_dir / "persist_convergence.png")
    plot_per_cell_autocorr_hist(per_cell, out_dir / "per_cell_autocorr_hist.png")
    plot_cos_distribution(traj, out_dir / "cos_distribution.png")
    plot_trajectories_2d(traj, out_dir / "trajectories_2d.png")
    plot_position_snapshots(traj, out_dir / "position_snapshots.png")
    print(f"[done] 5 figures written to {out_dir}/")


if __name__ == "__main__":
    main()
