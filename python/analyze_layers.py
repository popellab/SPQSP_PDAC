"""
analyze_layers.py — Per-layer timing analysis for SPQSP PDAC simulation.

Reads outputs/layer_timing.csv (long format: step, layer, ms) and produces:
  1. Stacked bar chart of mean time per phase across all steps
  2. Per-step stacked area chart showing time breakdown over the run
  3. GPU memory usage over time
  4. Box plots showing variability of each phase
  5. Summary table printed to stdout

Usage:
    cd /home/chase/SPQSP/SPQSP_PDAC-main
    python python/analyze_layers.py [--csv outputs/layer_timing.csv] [--out python/outputs/layers]
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ─── CLI ─────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="outputs/layer_timing.csv")
parser.add_argument("--out", default="python/outputs/layers")
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

# ─── Load data ────────────────────────────────────────────────────────────────
df = pd.read_csv(args.csv)
df.columns = ["step", "layer", "ms"]

# ─── Phase order and colours ─────────────────────────────────────────────────
# These match the checkpoint names recorded by timing_after_* host functions.
# pde_wall_ms is the wall-clock cross-check; pde_solve_ms is the internal timer.
# We prefer pde_solve_ms for accuracy and drop pde_wall_ms from stacked plots.
PHASE_ORDER = [
    "recruit",
    "broadcast_scan",
    "state_sources",
    "pde_solve_ms",     # from internal g_last_pde_ms timer (most accurate)
    "gradients",
    "ecm",
    "movement",
    "division",
    "qsp_solve_ms",     # from internal g_last_qsp_ms timer
]

PHASE_LABELS = {
    "recruit":        "Recruitment",
    "broadcast_scan": "Broadcast + Scan",
    "state_sources":  "State + Sources",
    "pde_solve_ms":   "PDE Solve",
    "gradients":      "PDE Gradients",
    "ecm":            "ECM Update",
    "movement":       "Movement",
    "division":       "Division",
    "qsp_solve_ms":   "QSP Solve",
}

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
]

# ─── Pivot to wide format ─────────────────────────────────────────────────────
wide = df.pivot_table(index="step", columns="layer", values="ms", aggfunc="first")
wide.columns.name = None

# Fill any missing phases with NaN
for p in PHASE_ORDER:
    if p not in wide.columns:
        wide[p] = np.nan

# Keep only the phases we care about (in order)
phases = [p for p in PHASE_ORDER if p in wide.columns]
data = wide[phases].copy().clip(lower=0)

# "Other ABM" = total_ms - sum(all named phases) — captures FLAMEGPU2 framework overhead
if "total_ms" in wide.columns:
    named_sum = data.sum(axis=1)
    other = (wide["total_ms"] - named_sum).clip(lower=0)
    data["other_overhead"] = other
    phases_plot = phases + ["other_overhead"]
    labels_plot = [PHASE_LABELS.get(p, p) for p in phases] + ["Other/Overhead"]
    colors_plot = COLORS[:len(phases)] + ["#aaaaaa"]
else:
    phases_plot = phases
    labels_plot = [PHASE_LABELS.get(p, p) for p in phases]
    colors_plot = COLORS[:len(phases)]

steps = data.index.values

# ─── 1. Stacked bar: mean time per phase ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
means = data[phases_plot].mean()
bottom = 0
for p, col, lbl in zip(phases_plot, colors_plot, labels_plot):
    val = means.get(p, 0)
    if np.isnan(val):
        val = 0
    ax.bar("Mean", val, bottom=bottom, color=col, label=lbl, edgecolor="white", linewidth=0.5)
    bottom += val

ax.set_ylabel("Wall time (ms)")
ax.set_title("Mean per-step time by phase (50³ grid, 50 steps)")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_xlim(-0.5, 0.5)
plt.tight_layout()
out_path = os.path.join(args.out, "mean_phase_timing.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")

# ─── 2. Stacked area chart: time breakdown over steps ────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))
vals = np.nan_to_num(data[phases_plot].values.T, nan=0.0)
ax.stackplot(steps, vals, labels=labels_plot, colors=colors_plot, alpha=0.85)
ax.set_xlabel("ABM Step")
ax.set_ylabel("Wall time (ms)")
ax.set_title("Per-step layer timing breakdown")
ax.legend(loc="upper right", fontsize=8, ncol=2)
ax.set_xlim(steps[0], steps[-1])
plt.tight_layout()
out_path = os.path.join(args.out, "layer_timing_area.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")

# ─── 3. GPU memory over time ──────────────────────────────────────────────────
if "gpu_mem_mb" in wide.columns:
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(wide.index, wide["gpu_mem_mb"], color="#3cb44b", linewidth=1.5)
    ax.set_xlabel("ABM Step")
    ax.set_ylabel("GPU memory used (MB)")
    ax.set_title("GPU memory usage over simulation")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(args.out, "gpu_memory.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")

# ─── 4. Box plots: per-phase variability ─────────────────────────────────────
named_phases = [p for p in phases_plot if p != "other_overhead"]
named_labels = [PHASE_LABELS.get(p, p) for p in named_phases]
named_colors = colors_plot[:len(named_phases)]

fig, ax = plt.subplots(figsize=(12, 5))
box_data = [data[p].dropna().values for p in named_phases]
bp = ax.boxplot(box_data, patch_artist=True, showfliers=True,
                flierprops=dict(marker=".", markersize=3, alpha=0.5))
for patch, col in zip(bp["boxes"], named_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.7)

ax.set_xticks(range(1, len(named_labels) + 1))
ax.set_xticklabels(named_labels, rotation=30, ha="right", fontsize=9)
ax.set_ylabel("Wall time (ms)")
ax.set_title("Per-phase timing distribution across steps")
ax.grid(True, axis="y", alpha=0.3)
plt.tight_layout()
out_path = os.path.join(args.out, "phase_boxplot.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")

# ─── 5. Summary table ─────────────────────────────────────────────────────────
print("\n=== Per-Phase Timing Summary (ms) ===")
print(f"{'Phase':<22} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'% Total':>8}")
print("-" * 72)

total_mean = wide["total_ms"].mean() if "total_ms" in wide.columns else data[phases_plot].sum(axis=1).mean()

for p, lbl in zip(phases_plot, labels_plot):
    if p not in data.columns:
        continue
    col = data[p].dropna()
    if len(col) == 0:
        continue
    pct = 100 * col.mean() / total_mean if total_mean > 0 else 0
    print(f"{lbl:<22} {col.mean():>8.2f} {col.median():>8.2f} {col.std():>8.2f} "
          f"{col.min():>8.2f} {col.max():>8.2f} {pct:>7.1f}%")

print(f"{'TOTAL':<22} {total_mean:>8.2f}")
print()

# ─── 6. Horizontal bar: mean % breakdown (optimisation view) ─────────────────
fig, ax = plt.subplots(figsize=(10, 4))
pcts = []
lbls = []
cols = []
for p, lbl, col in zip(phases_plot, labels_plot, colors_plot):
    if p not in data.columns:
        continue
    m = data[p].mean()
    if np.isnan(m) or m <= 0:
        continue
    pcts.append(100 * m / total_mean)
    lbls.append(lbl)
    cols.append(col)

y = range(len(lbls))
bars = ax.barh(list(y), pcts, color=cols, edgecolor="white")
ax.set_yticks(list(y))
ax.set_yticklabels(lbls, fontsize=9)
ax.set_xlabel("% of total step time")
ax.set_title("Time fraction per phase (optimisation target view)")
ax.invert_yaxis()
for bar, pct in zip(bars, pcts):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%", va="center", fontsize=8)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
out_path = os.path.join(args.out, "phase_fractions.png")
fig.savefig(out_path, dpi=150)
plt.close(fig)
print(f"Saved: {out_path}")
