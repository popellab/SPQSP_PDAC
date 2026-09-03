"""Diagnostic figures for the 2D healthy-tissue SA placement test."""
import sys
sys.path.insert(0, "/home/chase/SPQSP/SPQSP_PDAC-main")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PDAC.tools.placement_io import read_placement
from PDAC.tools.seeding_sa import renormalize_targets
from PDAC.tools.funcn_core import build_kde_field, funcn_frac

OUT = "results/2026-09-03_healthy_2d_sa"
PLC = f"{OUT}/healthy_200.plc"
NPZ = f"{OUT}/healthy_200.npz"

table, vvas, ecm, meta = read_placement(PLC)
gx, gy, gz = meta["shape"]

# Load actual SA positions
sa = np.load(NPZ)
pos_fib_2d = sa["pos_fib"]
pos_vas_2d = sa["pos_vas"]

cancer = table[table[:, 0] == 0]
fibs = table[table[:, 0] == 5]
fib_pos = fibs[:, 1:4]

target = renormalize_targets("data/healthy_imc/funcn_target_mean.csv")

# ── Fig 1: Overview (fibs, Vvas, fib KDE) ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

ax = axes[0]
ax.scatter(pos_vas_2d[:, 0], pos_vas_2d[:, 1], s=0.2, alpha=0.2, c="tab:orange", label="Vasculature")
ax.scatter(pos_fib_2d[:, 0], pos_fib_2d[:, 1], s=0.2, alpha=0.2, c="tab:blue", label="Fibroblast")
ax.scatter(cancer[:, 1], cancer[:, 2], s=50, c="red", marker="*", label="Cancer stem", zorder=5)
ax.set_xlim(0, gx)
ax.set_ylim(0, gy)
ax.set_aspect("equal")
ax.set_title(f"Cell positions ({len(pos_fib_2d)} fib + {len(pos_vas_2d)} vas)")
ax.legend(loc="upper right", markerscale=3)
ax.set_xlabel("x (voxels)")
ax.set_ylabel("y (voxels)")

ax = axes[1]
im = ax.imshow(vvas[:, :, 0].T, origin="lower", cmap="inferno", vmin=0, vmax=1)
ax.set_title("Vvas field (from vas KDE)")
ax.set_xlabel("x (voxels)")
ax.set_ylabel("y (voxels)")
plt.colorbar(im, ax=ax, shrink=0.8, label="Vvas")

ax = axes[2]
fib_kde = build_kde_field(fib_pos, (gx, gy, gz), sigma_vox=1.5)
im = ax.imshow(fib_kde[:, :, 0].T, origin="lower", cmap="Greens")
ax.set_title("Fibroblast KDE (sigma=30um)")
ax.set_xlabel("x (voxels)")
ax.set_ylabel("y (voxels)")
plt.colorbar(im, ax=ax, shrink=0.8, label="Density")

plt.tight_layout()
plt.savefig(f"{OUT}/placement_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved placement_overview.png")

# ── Fig 2: FunCN fraction distributions (using actual SA positions) ────
vas_pos_3d = np.column_stack([pos_vas_2d, np.zeros(len(pos_vas_2d), dtype=np.int32)])
fib_pos_3d = np.column_stack([pos_fib_2d, np.zeros(len(pos_fib_2d), dtype=np.int32)])

cats = ["vas", "fib"]
pos_by_cat = {"vas": vas_pos_3d, "fib": fib_pos_3d}
frac = funcn_frac(pos_by_cat, cats, (gx, gy, gz), sigma_vox=1.5)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
pairs = [("vas", "vas"), ("vas", "fib"), ("fib", "vas"), ("fib", "fib")]
titles = ["vas x vas", "vas x fib", "fib x vas", "fib x fib"]
target_vals = [target[0, 0], target[0, 1], target[1, 0], target[1, 1]]

for ax, (t, s), title, tgt in zip(axes.flat, pairs, titles, target_vals):
    vals = frac[(t, s)]
    if len(vals) > 0:
        ax.hist(vals, bins=50, density=True, alpha=0.7, color="steelblue", edgecolor="white")
        mn = np.mean(vals)
        ax.axvline(mn, color="red", lw=2, label=f"mean={mn:.3f}")
        ax.axvline(tgt, color="green", lw=2, ls="--", label=f"target={tgt:.3f}")
    ax.set_title(title)
    ax.set_xlabel("Neighbourhood fraction")
    ax.set_ylabel("Density")
    ax.legend()

plt.suptitle("FunCN neighbourhood-fraction distributions (SA placement, actual positions)", fontsize=14)
plt.tight_layout()
plt.savefig(f"{OUT}/funcn_frac_distributions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved funcn_frac_distributions.png")

# ── Fig 3: Center detail ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
cx, cy = gx // 2, gy // 2
w = 50

ax = axes[0]
m_f = ((pos_fib_2d[:, 0] >= cx - w) & (pos_fib_2d[:, 0] < cx + w) &
       (pos_fib_2d[:, 1] >= cy - w) & (pos_fib_2d[:, 1] < cy + w))
m_v = ((pos_vas_2d[:, 0] >= cx - w) & (pos_vas_2d[:, 0] < cx + w) &
       (pos_vas_2d[:, 1] >= cy - w) & (pos_vas_2d[:, 1] < cy + w))
ax.scatter(pos_vas_2d[m_v, 0], pos_vas_2d[m_v, 1], s=4, alpha=0.6, c="tab:orange", label="Vas")
ax.scatter(pos_fib_2d[m_f, 0], pos_fib_2d[m_f, 1], s=4, alpha=0.6, c="tab:blue", label="Fib")
ax.scatter(cancer[:, 1], cancer[:, 2], s=100, c="red", marker="*", label="Cancer", zorder=5)
ax.set_xlim(cx - w, cx + w)
ax.set_ylim(cy - w, cy + w)
ax.set_aspect("equal")
ax.set_title("Center 100x100 detail: agents")
ax.legend()

ax = axes[1]
im = ax.imshow(vvas[cx-w:cx+w, cy-w:cy+w, 0].T, origin="lower",
               extent=[cx-w, cx+w, cy-w, cy+w], cmap="inferno", vmin=0, vmax=1)
ax.set_title("Center 100x100 detail: Vvas")
plt.colorbar(im, ax=ax, shrink=0.8, label="Vvas")
ax.set_aspect("equal")

plt.tight_layout()
plt.savefig(f"{OUT}/center_detail.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved center_detail.png")

# ── Fig 4: Vvas histogram ──────────────────────────────────────────────
vvas_2d = vvas[:, :, 0]
fig, ax = plt.subplots(figsize=(8, 5))
nz = vvas_2d[vvas_2d > 0].ravel()
ax.hist(nz, bins=80, density=True, alpha=0.7, color="darkorange", edgecolor="white")
ax.axvline(np.median(nz), color="red", lw=2, label=f"median={np.median(nz):.3f}")
ax.axvline(np.mean(nz), color="blue", lw=2, ls="--", label=f"mean={np.mean(nz):.3f}")
ax.set_xlabel("Vvas (non-zero voxels)")
ax.set_ylabel("Density")
ax.set_title(f"Vvas field distribution (coverage={100*(vvas_2d > 0).mean():.1f}%)")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/vvas_histogram.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved vvas_histogram.png")

# ── Summary table ──────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"Grid: {gx}x{gy}x{gz}")
print(f"Agents: {meta['n_agents']} ({len(pos_fib_2d)} fib + 1 cancer)")
print(f"Vas seeds: {len(pos_vas_2d)} (field only)")
for (t, s), title, tgt in zip(pairs, titles, target_vals):
    vals = frac[(t, s)]
    mn = np.mean(vals) if len(vals) > 0 else float("nan")
    print(f"  {title}: mean={mn:.4f}  target={tgt:.4f}  delta={mn-tgt:+.4f}")
print("\nAll figures saved.")
