"""
visualize.py — Figures for IMC spatial analysis results.

Requires density_summary.csv, septa_distances.csv, nn_distances.csv,
ripley_L.csv from spatial_analysis.py.

Produces:
  figures/density_histograms.png   — per-type density histograms + lognormal fit
  figures/ripley_clustering.png    — Ripley L(r)-r curves (same-type clustering)
  figures/nn_heatmap.png           — cross-type nearest-neighbor distance heatmap
  figures/septa_distances.png      — distance-to-stroma distributions per type

Usage:
    python visualize.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy import stats

BASE = Path(__file__).parent
SPATIAL = BASE / "spatial"
FIG_DIR = BASE / "spatial" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Cell types to show (ordered)
TYPES = ["cd4_t", "cd8_t", "b_cell", "macrophage", "mdsc", "fibroblast", "endothelial"]

LABELS = {
    "cd4_t":       "CD4⁺ T",
    "cd8_t":       "CD8⁺ T",
    "b_cell":      "B cell",
    "macrophage":  "Macrophage",
    "mdsc":        "MDSC",
    "fibroblast":  "Fibroblast",
    "endothelial": "Endothelial",
}

COLORS = {
    "cd4_t":       "#3498db",
    "cd8_t":       "#e74c3c",
    "b_cell":      "#9b59b6",
    "macrophage":  "#e67e22",
    "mdsc":        "#1abc9c",
    "fibroblast":  "#f39c12",
    "endothelial": "#2ecc71",
}


# ── 1. Density histograms with lognormal fit ───────────────────────────────────

def plot_density_histograms(density_df: pd.DataFrame):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, stype in enumerate(TYPES):
        ax = axes[i]
        col = f"density_{stype}"
        vals = density_df[col].dropna().values
        nonzero = vals[vals > 0]
        zero_frac = (vals == 0).mean()

        color = COLORS[stype]

        # Histogram of non-zero values
        if len(nonzero) > 5:
            ax.hist(nonzero, bins=30, density=True, alpha=0.55,
                    color=color, edgecolor="white", linewidth=0.4)

            # Lognormal fit
            shape, loc, scale = stats.lognorm.fit(nonzero, floc=0)
            x = np.linspace(nonzero.min(), np.percentile(nonzero, 98), 300)
            pdf = stats.lognorm.pdf(x, shape, loc, scale)
            ax.plot(x, pdf, color="black", linewidth=1.8, label="Lognormal fit")

            mu = np.log(scale)          # lognormal mu
            sigma = shape               # lognormal sigma
            median_fit = np.exp(mu)
            mean_fit = np.exp(mu + sigma**2 / 2)

            # Annotate
            ax.axvline(np.median(nonzero), color=color, linestyle="--",
                       linewidth=1.2, alpha=0.8, label=f"Median {np.median(nonzero):.0f}")
            info = (f"n={len(vals)} ROIs\n"
                    f"zero: {zero_frac*100:.0f}%\n"
                    f"median: {np.median(nonzero):.0f}\n"
                    f"mean: {nonzero.mean():.0f}\n"
                    f"σ_ln: {sigma:.2f}")
            ax.text(0.97, 0.97, info, transform=ax.transAxes,
                    fontsize=7.5, va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
        else:
            ax.text(0.5, 0.5, "insufficient data", transform=ax.transAxes,
                    ha="center", va="center", color="gray")

        ax.set_title(LABELS[stype], fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Density (cells/mm²)", fontsize=9)
        ax.set_ylabel("Probability density", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_visible(False)
    fig.suptitle("Cell type densities across 299 ROIs (non-zero values, lognormal fit)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "density_histograms.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── 2. Ripley's L curves — same-type clustering ────────────────────────────────

def plot_ripley(ripley_df: pd.DataFrame):
    """
    L(r) - r > 0: more clustered than random at radius r.
    L(r) - r = 0: complete spatial randomness (CSR).
    L(r) - r < 0: more dispersed than random.
    Peak radius = characteristic clustering scale.
    """
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, stype in enumerate(TYPES):
        ax = axes[i]
        sub = ripley_df[ripley_df["subtype"] == stype]
        if sub.empty:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(LABELS[stype])
            continue

        color = COLORS[stype]
        # Per-ROI curves (thin, transparent)
        for roi_key, grp in sub.groupby(["donor", "region", "roi"]):
            ax.plot(grp["r_um"], grp["L_minus_r"], color=color,
                    alpha=0.12, linewidth=0.6)

        # Mean ± 95% CI
        mean_L = sub.groupby("r_um")["L_minus_r"].mean()
        ci95   = sub.groupby("r_um")["L_minus_r"].apply(
            lambda x: 1.96 * x.std() / np.sqrt(len(x)))
        r_vals = mean_L.index.values

        ax.fill_between(r_vals, mean_L - ci95, mean_L + ci95,
                        alpha=0.25, color=color)
        ax.plot(r_vals, mean_L, color=color, linewidth=2.2, label="Mean")
        ax.axhline(0, color="black", linestyle="--", linewidth=1.0, alpha=0.6,
                   label="CSR")

        # Mark peak
        if mean_L.max() > 0:
            peak_r = mean_L.idxmax()
            ax.axvline(peak_r, color=color, linestyle=":", linewidth=1.2, alpha=0.8)
            ax.text(peak_r + 3, mean_L.max() * 0.9,
                    f"peak\n{peak_r:.0f}µm", fontsize=7.5, color=color)

        ax.set_title(LABELS[stype], fontsize=11, fontweight="bold", color=color)
        ax.set_xlabel("Radius r (µm)", fontsize=9)
        ax.set_ylabel("L(r) − r (µm)", fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_visible(False)
    fig.suptitle(
        "Ripley's L(r)−r: same-type spatial clustering\n"
        "(>0 = more clustered than random; peak = characteristic aggregation scale)",
        fontsize=12, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = FIG_DIR / "ripley_clustering.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── 3. Cross-type nearest-neighbor distance heatmap ───────────────────────────

def plot_nn_heatmap(nn_df: pd.DataFrame):
    """
    Median nearest-neighbor distance (µm) from type A to nearest type B cell.
    Short distance = types tend to co-localize.
    """
    types = TYPES
    n = len(types)
    matrix = np.full((n, n), np.nan)

    for i, a in enumerate(types):
        for j, b in enumerate(types):
            col = f"nn_{a}_to_{b}"
            if col in nn_df.columns:
                matrix[i, j] = nn_df[col].median()

    fig, ax = plt.subplots(figsize=(9, 7.5))
    vmax = np.nanpercentile(matrix, 90)
    im = ax.imshow(matrix, cmap="YlOrRd_r", aspect="auto",
                   vmin=0, vmax=vmax)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            if not np.isnan(matrix[i, j]):
                txt = f"{matrix[i, j]:.0f}"
                brightness = matrix[i, j] / vmax
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=8.5, fontweight="bold",
                        color="white" if brightness < 0.5 else "black")

    tick_labels = [LABELS[t] for t in types]
    ax.set_xticks(range(n)); ax.set_xticklabels(tick_labels, rotation=35, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(tick_labels)
    ax.set_xlabel("Target cell type (nearest neighbour)", fontsize=10)
    ax.set_ylabel("Source cell type", fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Median NN distance (µm)", fontsize=9)

    ax.set_title(
        "Cross-type nearest-neighbour distances\n"
        "(median µm — lower = more co-localised)",
        fontsize=12, fontweight="bold"
    )
    plt.tight_layout()
    out = FIG_DIR / "nn_heatmap.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── 4. Septa (stroma boundary) distance distributions ─────────────────────────

def plot_septa_distances(septa_df: pd.DataFrame):
    """
    Distribution of distance (µm) from each cell type to the nearest
    collagen-rich stromal boundary (septa). Fibroblasts expected nearest.
    """
    septa_only = septa_df[septa_df["has_septa"]].copy()
    n_rois = len(septa_only)

    # Collect p25/p50/p75 per type per ROI
    data = {}
    for stype in TYPES:
        col50 = f"septa_p50_{stype}"
        col25 = f"septa_p25_{stype}"
        col75 = f"septa_p75_{stype}"
        if col50 in septa_only.columns:
            data[stype] = {
                "p25": septa_only[col25].dropna().values if col25 in septa_only.columns else None,
                "p50": septa_only[col50].dropna().values,
                "p75": septa_only[col75].dropna().values if col75 in septa_only.columns else None,
            }

    if not data:
        print("No septa distance data found.")
        return

    # ── Panel A: violin of median distance per type ────────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    types_present = [t for t in TYPES if t in data and len(data[t]["p50"]) > 5]
    medians_per_type = [data[t]["p50"] for t in types_present]
    colors_list = [COLORS[t] for t in types_present]
    xlabels = [LABELS[t] for t in types_present]

    parts = ax1.violinplot(medians_per_type, positions=range(len(types_present)),
                           showmedians=True, showextrema=False)
    for pc, col in zip(parts["bodies"], colors_list):
        pc.set_facecolor(col)
        pc.set_alpha(0.7)
    parts["cmedians"].set_color("black")
    parts["cmedians"].set_linewidth(1.8)

    # Overlay individual ROI medians
    for j, (vals, col) in enumerate(zip(medians_per_type, colors_list)):
        ax1.scatter(np.full(len(vals), j) + np.random.uniform(-0.08, 0.08, len(vals)),
                    vals, color=col, s=10, alpha=0.5, zorder=3)

    ax1.set_xticks(range(len(types_present)))
    ax1.set_xticklabels(xlabels, rotation=30, ha="right")
    ax1.set_ylabel("Median distance to nearest septa (µm)")
    ax1.set_title(f"Distance to stromal boundaries\n(n={n_rois} ROIs with septa data)",
                  fontsize=11, fontweight="bold")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Highlight fibroblast
    if "fibroblast" in types_present:
        fi = types_present.index("fibroblast")
        ax1.axvspan(fi - 0.45, fi + 0.45, alpha=0.08, color=COLORS["fibroblast"])
        ax1.text(fi, ax1.get_ylim()[1] * 0.97, "↓ stroma", ha="center",
                 fontsize=8, color=COLORS["fibroblast"], fontweight="bold")

    # ── Panel B: cumulative distributions for key types ────────────────────────
    highlight = ["fibroblast", "macrophage", "cd8_t", "cd4_t", "endothelial"]
    for stype in highlight:
        if stype not in data or len(data[stype]["p50"]) < 5:
            continue
        vals = np.sort(data[stype]["p50"])
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax2.plot(vals, cdf, color=COLORS[stype], linewidth=2.0,
                 label=f"{LABELS[stype]} (n={len(vals)})")

    ax2.set_xlabel("Median distance to nearest septa (µm)")
    ax2.set_ylabel("Cumulative fraction of ROIs")
    ax2.set_title("CDF of septa distance — key types", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.set_xlim(left=0)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    plt.suptitle("Proximity to stromal boundaries (septa)\n"
                 "Lower distance → cells tend to localise near collagen-rich stroma",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = FIG_DIR / "septa_distances.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Print summary table
    print("\n=== Median distance to septa (µm), across ROIs ===")
    print(f"  {'Type':15s}  {'Median':>8}  {'Mean':>8}  {'n ROIs':>7}")
    for stype in TYPES:
        if stype in data and len(data[stype]["p50"]) > 0:
            v = data[stype]["p50"]
            print(f"  {LABELS[stype]:15s}  {np.median(v):8.1f}  {np.mean(v):8.1f}  {len(v):7d}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    density_df = pd.read_csv(SPATIAL / "density_summary.csv")
    septa_df   = pd.read_csv(SPATIAL / "septa_distances.csv")
    nn_df      = pd.read_csv(SPATIAL / "nn_distances.csv")
    ripley_df  = pd.read_csv(SPATIAL / "ripley_L.csv")

    print(f"Loaded {len(density_df)} ROIs")
    print("Generating figures...\n")

    plot_density_histograms(density_df)
    plot_ripley(ripley_df)
    plot_nn_heatmap(nn_df)
    plot_septa_distances(septa_df)

    print(f"\nAll figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
