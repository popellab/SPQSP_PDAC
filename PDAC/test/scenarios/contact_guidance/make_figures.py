#!/usr/bin/env python3
"""
Representative figures for the contact-guidance test.

Reads per-scenario outputs (aligned and, if present, iso):
  outputs/trajectories.csv      (step,id,x,y,z)
  outputs/per_step_stats.csv    (step,n,mean_dx,mean_dy,mean_dz,mean_dx2,mean_dy2,mean_dz2)

Writes up to five PNGs into the aligned scenario's outputs/:
  moments_vs_step.png        — running <Δx>,<Δy>,<Δz> with expected lines
  variances_vs_step.png      — running <Δx²>,<Δy²>,<Δz²> with expected lines
  displacement_hist.png      — distribution of per-step Δx (asymmetric vs symmetric)
  trajectories_2d.png        — xy-projected single-cell paths (drift vs diffusion)
  position_snapshots.png     — cohort xy positions at 4 timepoints

Pulls the iso sibling from ../contact_guidance_iso/outputs when available.

Run from repo root:
  python3 PDAC/test/scenarios/contact_guidance/make_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Analytic expectations (see test_contact_guidance.cu header for derivation).
EXP_ALIGNED = {
    "mdx":  0.265,   # sign-flip Markov steady state, |mode|=0.548
    "mdy":  0.0,
    "mdz":  0.0,
    "mdx2": 0.781,
    "mdy2": 0.648,
    "mdz2": 0.648,
}
EXP_ISO = {
    "mdx": 0.0, "mdy": 0.0, "mdz": 0.0,
    "mdx2": 18.0 / 26.0, "mdy2": 18.0 / 26.0, "mdz2": 18.0 / 26.0,
}


def load_scenario(out_dir: Path):
    if not out_dir.exists():
        return None
    try:
        traj = pd.read_csv(out_dir / "trajectories.csv")
        stats = pd.read_csv(out_dir / "per_step_stats.csv")
    except FileNotFoundError:
        return None
    return traj, stats


def _running_mean(col: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Cumulative <col> weighted by per-step sample counts."""
    num = np.cumsum(col * counts)
    den = np.cumsum(counts)
    den = np.where(den == 0, 1, den)
    return num / den


def plot_moments_vs_step(aligned_stats, iso_stats, exp_a, exp_i, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, stats, exp, label in (
        (axes[0], aligned_stats, exp_a, "Aligned (orient = +x)"),
        (axes[1], iso_stats,     exp_i, "Iso (orient = 0)"),
    ):
        if stats is None:
            ax.set_title(label + " — no data")
            ax.axis("off")
            continue
        step = stats["step"].values
        n = stats["n"].values
        rdx = _running_mean(stats["mean_dx"].values, n)
        rdy = _running_mean(stats["mean_dy"].values, n)
        rdz = _running_mean(stats["mean_dz"].values, n)
        ax.plot(step, rdx, color="tab:red",  lw=1.5, label=r"running $\langle\Delta x\rangle$")
        ax.plot(step, rdy, color="tab:green", lw=1.2, alpha=0.85, label=r"running $\langle\Delta y\rangle$")
        ax.plot(step, rdz, color="tab:blue",  lw=1.2, alpha=0.85, label=r"running $\langle\Delta z\rangle$")
        ax.axhline(exp["mdx"], color="tab:red",  ls="--", lw=1.0, alpha=0.7,
                   label=f"exp $\\Delta x$ = {exp['mdx']:.3f}")
        ax.axhline(0, color="0.5", ls=":", lw=0.8)
        ax.set_xlabel("ABM step")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    axes[0].set_ylabel("running mean per-step displacement [voxels]")
    fig.suptitle("Contact guidance — first moments converge to analytic", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_variances_vs_step(aligned_stats, iso_stats, exp_a, exp_i, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, stats, exp, label in (
        (axes[0], aligned_stats, exp_a, "Aligned — expect anisotropic"),
        (axes[1], iso_stats,     exp_i, "Iso — expect isotropic"),
    ):
        if stats is None:
            ax.set_title(label + " — no data")
            ax.axis("off")
            continue
        step = stats["step"].values
        n = stats["n"].values
        rx2 = _running_mean(stats["mean_dx2"].values, n)
        ry2 = _running_mean(stats["mean_dy2"].values, n)
        rz2 = _running_mean(stats["mean_dz2"].values, n)
        ax.plot(step, rx2, color="tab:red",  lw=1.5, label=r"$\langle\Delta x^2\rangle$")
        ax.plot(step, ry2, color="tab:green", lw=1.2, label=r"$\langle\Delta y^2\rangle$")
        ax.plot(step, rz2, color="tab:blue",  lw=1.2, label=r"$\langle\Delta z^2\rangle$")
        ax.axhline(exp["mdx2"], color="tab:red",   ls="--", lw=0.9, alpha=0.6)
        ax.axhline(exp["mdy2"], color="tab:green", ls="--", lw=0.9, alpha=0.6)
        ax.set_xlabel("ABM step")
        ax.set_title(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right", fontsize=8)
    axes[0].set_ylabel("running mean squared displacement [voxels²]")
    fig.suptitle("Contact guidance — second moments (parallel > perpendicular for aligned)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _displacements(traj: pd.DataFrame) -> np.ndarray:
    """Per-step (Δx, Δy, Δz) from consecutive trajectory rows per cell."""
    traj_sorted = traj.sort_values(["id", "step"])
    deltas = []
    for _, g in traj_sorted.groupby("id"):
        pos = g[["x", "y", "z"]].values
        deltas.append(np.diff(pos, axis=0))
    return np.vstack(deltas) if deltas else np.empty((0, 3))


def plot_displacement_hist(aligned_traj, iso_traj, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    bins = np.arange(-1.5, 2.0) - 0.5  # centred at {-1, 0, 1}
    for ax, traj, label in (
        (axes[0], aligned_traj, "Aligned"),
        (axes[1], iso_traj,     "Iso"),
    ):
        if traj is None:
            ax.set_title(label + " — no data")
            ax.axis("off")
            continue
        d = _displacements(traj)
        if d.size == 0:
            ax.set_title(label + " — no displacements")
            ax.axis("off")
            continue
        width = 0.28
        centres = np.array([-1, 0, 1], dtype=float)
        for i, (axis, colour) in enumerate(
                (("Δx", "tab:red"), ("Δy", "tab:green"), ("Δz", "tab:blue"))):
            counts = np.array([(d[:, i] == v).sum() for v in centres])
            frac = counts / counts.sum()
            ax.bar(centres + (i - 1) * width, frac, width=width,
                   color=colour, alpha=0.85, edgecolor="black", label=axis)
        ax.set_xticks(centres)
        ax.set_xlabel("per-step displacement [voxels]")
        ax.set_title(f"{label} (n={len(d)})")
        ax.grid(alpha=0.3, axis="y")
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("fraction of per-step moves")
    fig.suptitle("Contact guidance — Δx asymmetry on aligned vs. symmetric on iso", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_trajectories_2d(aligned_traj, iso_traj, out_path: Path, n_sample=12):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    rng = np.random.default_rng(0)
    for ax, traj, label, arrow in (
        (axes[0], aligned_traj, "Aligned — +x drift", True),
        (axes[1], iso_traj,     "Iso — diffusive",    False),
    ):
        if traj is None:
            ax.set_title(label + " — no data"); ax.axis("off"); continue
        ids = traj["id"].unique()
        sample_ids = rng.choice(ids, size=min(n_sample, len(ids)), replace=False)
        cmap = plt.get_cmap("tab10")
        for i, aid in enumerate(sample_ids):
            g = traj[traj["id"] == aid].sort_values("step")
            ax.plot(g["x"], g["y"], lw=0.9, color=cmap(i % 10), alpha=0.85)
            ax.scatter(g["x"].iloc[0],  g["y"].iloc[0],  s=30,
                       color=cmap(i % 10), marker="o",
                       edgecolors="black", linewidths=0.4, zorder=3)
            ax.scatter(g["x"].iloc[-1], g["y"].iloc[-1], s=55,
                       color=cmap(i % 10), marker="*",
                       edgecolors="black", linewidths=0.4, zorder=3)
        if arrow:
            ax.annotate("", xy=(0.95, 0.92), xytext=(0.70, 0.92),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
            ax.text(0.70, 0.95, "fiber axis", transform=ax.transAxes, fontsize=9)
        ax.set_xlabel("x voxel"); ax.set_ylabel("y voxel")
        ax.set_title(label + f"  ({len(sample_ids)} cells)  ○ start  ★ end")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
    fig.suptitle("Contact guidance — single-cell trajectories (xy projection)", y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_position_snapshots(aligned_traj, iso_traj, out_path: Path,
                            snapshot_steps=(0, 10, 20, 29)):
    rows = []
    if aligned_traj is not None:
        rows.append(("Aligned", aligned_traj))
    if iso_traj is not None:
        rows.append(("Iso", iso_traj))
    if not rows:
        return

    n_cols = len(snapshot_steps)
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(2.6 * n_cols, 2.8 * n_rows),
                             sharex=True, sharey=True)
    if n_rows == 1:
        axes = np.array([axes])

    # Common extent from union of all snapshots.
    all_xy = []
    for _, traj in rows:
        for s in snapshot_steps:
            snap = traj[traj["step"] == s]
            if len(snap):
                all_xy.append(snap[["x", "y"]].values)
    all_xy = np.vstack(all_xy)
    pad = 3
    xlim = (all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ylim = (all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    for r, (label, traj) in enumerate(rows):
        for c, step in enumerate(snapshot_steps):
            ax = axes[r, c]
            snap = traj[traj["step"] == step]
            ax.scatter(snap["x"], snap["y"], s=10, c="tab:blue",
                       edgecolors="black", linewidths=0.2, alpha=0.85)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.set_aspect("equal")
            if r == 0:
                ax.set_title(f"step {step}  (n={len(snap)})")
            if r == n_rows - 1:
                ax.set_xlabel("x voxel")
            if c == 0:
                ax.set_ylabel(f"{label}\ny voxel")
            ax.grid(alpha=0.3)

    fig.suptitle("Contact guidance — cohort xy positions (aligned translates +x, iso spreads isotropically)",
                 y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    aligned_dir = (Path(sys.argv[1]) if len(sys.argv) > 1
                   else Path(__file__).parent / "outputs")
    iso_dir = aligned_dir.parent.parent / "contact_guidance_iso" / "outputs"

    aligned = load_scenario(aligned_dir)
    iso     = load_scenario(iso_dir)

    if aligned is None:
        print(f"[error] aligned outputs not found: {aligned_dir}", file=sys.stderr)
        print("Run: ./build/bin/pdac_test contact_guidance", file=sys.stderr)
        sys.exit(1)
    if iso is None:
        print(f"[warn] iso outputs not found at {iso_dir}; iso panels will be empty.",
              file=sys.stderr)

    aligned_traj, aligned_stats = aligned
    iso_traj,     iso_stats     = iso if iso is not None else (None, None)

    print(f"[info] aligned: {len(aligned_traj)} traj rows, "
          f"{aligned_stats['step'].max()+1 if len(aligned_stats) else 0} step rows")
    if iso is not None:
        print(f"[info] iso:     {len(iso_traj)} traj rows, "
              f"{iso_stats['step'].max()+1 if len(iso_stats) else 0} step rows")

    plot_moments_vs_step(  aligned_stats, iso_stats, EXP_ALIGNED, EXP_ISO,
                           aligned_dir / "moments_vs_step.png")
    plot_variances_vs_step(aligned_stats, iso_stats, EXP_ALIGNED, EXP_ISO,
                           aligned_dir / "variances_vs_step.png")
    plot_displacement_hist(aligned_traj, iso_traj,
                           aligned_dir / "displacement_hist.png")
    plot_trajectories_2d(  aligned_traj, iso_traj,
                           aligned_dir / "trajectories_2d.png")
    plot_position_snapshots(aligned_traj, iso_traj,
                            aligned_dir / "position_snapshots.png")

    print(f"[done] 5 figures written to {aligned_dir}/")


if __name__ == "__main__":
    main()
