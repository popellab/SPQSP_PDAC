"""
qc_annotation.py — Quality control for IMC cell type annotations.

Checks:
  1. GMM threshold placement — are thresholds splitting sensible populations?
  2. Marker expression per subtype — do annotated cells express expected markers?
  3. Cross-marker contamination — e.g. are "macrophages" also CD3+?
  4. Annotation rate — fraction unknown/other_immune per ROI; flag outliers
  5. Per-donor consistency — cell type fraction distributions across donors
  6. Panel-stratified comparison — CD11b vs CD163 panel: MDSC rate consistency

Usage:
    python qc_annotation.py

Outputs in analysis/qc/:
  gmm_thresholds.png        — expression distributions + GMM threshold lines
  marker_expression.png     — median marker intensity per annotated subtype (heatmap)
  contamination.png         — cross-marker positivity rates per subtype
  annotation_rates.png      — per-ROI unknown/other_immune fraction; outlier ROIs
  donor_consistency.png     — per-donor cell type fraction distributions
  panel_comparison.png      — CD11b vs CD163 panel MDSC/macrophage rate comparison
  qc_summary.csv            — per-ROI QC flags
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy import stats

BASE    = Path(__file__).parent
IMC_DIR = BASE.parent / "pancdb_imc"
QC_DIR  = BASE / "qc"
QC_DIR.mkdir(exist_ok=True)

GATING_MARKERS = ["CD3", "CD4", "CD8", "CD68", "CD20", "CD11b", "CD14"]

MARKER_SUFFIXES = {
    "CD45":  ["CD45"],
    "CD3":   ["CD3"],
    "CD4":   ["CD4"],
    "CD8":   ["CD8", "CD8a"],
    "CD20":  ["CD20"],
    "CD68":  ["CD68"],
    "CD11b": ["CD11b"],
    "CD14":  ["CD14"],
    "Foxp3": ["Foxp3"],
    "Collagen": ["Collagen", "CollagenI", "CollagenType1", "CollagenTypeI"],
    "CD31":  ["CD31"],
    "pan-Keratin": ["pan-Keratin", "Pan-keratin", "Keratins"],
}

SUBTYPES = ["cd4_t", "cd8_t", "b_cell", "macrophage", "mdsc",
            "other_immune", "fibroblast", "endothelial"]

LABELS = {
    "cd4_t": "CD4⁺ T", "cd8_t": "CD8⁺ T", "b_cell": "B cell",
    "macrophage": "Macrophage", "mdsc": "MDSC",
    "other_immune": "Other immune", "fibroblast": "Fibroblast",
    "endothelial": "Endothelial",
}

# Expected high/low markers per subtype — for contamination QC
# (marker, should_be_positive)
EXPECTED = {
    "cd4_t":       [("CD45", True),  ("CD3", True),  ("CD4", True),
                    ("CD8", False),  ("CD68", False), ("CD20", False)],
    "cd8_t":       [("CD45", True),  ("CD3", True),  ("CD8", True),
                    ("CD4", False),  ("CD68", False), ("CD20", False)],
    "b_cell":      [("CD45", True),  ("CD20", True), ("CD3", False),
                    ("CD68", False)],
    "macrophage":  [("CD45", True),  ("CD68", True), ("CD3", False),
                    ("CD20", False)],
    "mdsc":        [("CD45", True),  ("CD14", True), ("CD68", False)],
    "fibroblast":  [("CD45", False), ("CD31", False), ("Collagen", True)],
    "endothelial": [("CD45", False), ("CD31", True),  ("Collagen", False)],
}


def find_var(var_names, suffixes):
    for s in suffixes:
        for v in var_names:
            if v.endswith("_" + s):
                return v
    return None


def get_expr(adata, marker):
    var = find_var(list(adata.var_names), MARKER_SUFFIXES.get(marker, [marker]))
    if var is None:
        return None
    idx = list(adata.var_names).index(var)
    return np.asarray(adata.X[:, idx]).ravel()


# ── Load all data ──────────────────────────────────────────────────────────────

def load_all(max_rois=None):
    paths = sorted(IMC_DIR.rglob("cells.h5ad"))
    if max_rois:
        paths = paths[:max_rois]

    adatas = []
    for p in paths:
        adata = ad.read_h5ad(p)
        if "cell_subtype" not in adata.obs.columns:
            continue
        adatas.append(adata)
    print(f"Loaded {len(adatas)} ROIs")
    return adatas


# ── QC 1: GMM threshold plots ──────────────────────────────────────────────────

def plot_gmm_thresholds(adatas, thresholds):
    """Show expression histogram of each gating marker with GMM threshold."""
    n_markers = len(GATING_MARKERS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.ravel()

    for i, marker in enumerate(GATING_MARKERS):
        ax = axes[i]
        vals_list = []
        for adata in adatas:
            immune = adata.obs["cell_type"] == "immune"
            expr = get_expr(adata[immune], marker)
            if expr is not None:
                vals_list.append(expr)

        if not vals_list:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            continue

        vals = np.concatenate(vals_list)
        thr  = thresholds.get(marker)

        # Histogram
        ax.hist(vals, bins=80, density=True, alpha=0.5, color="#888888",
                label="All immune cells")

        # GMM fit
        gmm = GaussianMixture(n_components=2, random_state=42)
        gmm.fit(vals.reshape(-1, 1))
        order = np.argsort(gmm.means_.ravel())
        x = np.linspace(vals.min(), np.percentile(vals, 99.5), 400)
        for k, col, lbl in zip(order, ["#3498db", "#e74c3c"],
                                ["Negative", "Positive"]):
            mu  = gmm.means_.ravel()[k]
            sig = np.sqrt(gmm.covariances_.ravel()[k])
            w   = gmm.weights_[k]
            ax.plot(x, w * stats.norm.pdf(x, mu, sig),
                    color=col, linewidth=1.8, label=f"{lbl} (µ={mu:.2f})")

        # Threshold
        if thr is not None:
            ax.axvline(thr, color="black", linewidth=2.0, linestyle="--",
                       label=f"Threshold={thr:.3f}")
            # Positive fraction
            pos_frac = (vals >= thr).mean()
            ax.text(0.97, 0.97, f"pos: {pos_frac*100:.1f}%",
                    transform=ax.transAxes, fontsize=9, va="top", ha="right",
                    color="#e74c3c", fontweight="bold")

        ax.set_title(marker, fontsize=11, fontweight="bold")
        ax.set_xlabel("Arcsinh expression", fontsize=9)
        ax.set_ylabel("Density", fontsize=9)
        ax.legend(fontsize=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[-1].set_visible(False)
    fig.suptitle("GMM gating thresholds (pooled immune cells, n=299 ROIs)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = QC_DIR / "gmm_thresholds.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── QC 2: Marker expression heatmap per subtype ────────────────────────────────

def plot_marker_expression(adatas):
    """Median arcsinh expression of key markers per annotated subtype."""
    show_markers = ["CD45", "CD3", "CD4", "CD8", "CD20", "CD68",
                    "CD11b", "CD14", "Foxp3", "CD31", "Collagen", "pan-Keratin"]

    # Collect per-cell expression + subtype across all ROIs
    # Build a DataFrame by sampling (cap at 500 cells per subtype for speed)
    rng = np.random.default_rng(42)
    rows = {m: {s: [] for s in SUBTYPES} for m in show_markers}

    for adata in adatas:
        for stype in SUBTYPES:
            mask = (adata.obs["cell_subtype"] == stype).values
            if mask.sum() < 2:
                continue
            idx = np.where(mask)[0]
            if len(idx) > 100:
                idx = rng.choice(idx, 100, replace=False)
            for marker in show_markers:
                expr = get_expr(adata[idx], marker)
                if expr is not None:
                    rows[marker][stype].extend(expr.tolist())

    # Build median matrix
    matrix = np.full((len(show_markers), len(SUBTYPES)), np.nan)
    for i, marker in enumerate(show_markers):
        for j, stype in enumerate(SUBTYPES):
            v = rows[marker][stype]
            if len(v) >= 5:
                matrix[i, j] = np.median(v)

    # Z-score per marker row for visibility
    matrix_z = np.full_like(matrix, np.nan)
    for i in range(len(show_markers)):
        row = matrix[i]
        valid = row[~np.isnan(row)]
        if len(valid) > 1:
            mu, sig = valid.mean(), valid.std()
            if sig > 0:
                matrix_z[i] = (row - mu) / sig

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7),
                                    gridspec_kw={"width_ratios": [1, 1]})

    xlabels = [LABELS.get(s, s) for s in SUBTYPES]
    ylabels = show_markers

    for ax, data, title, cmap, vrange in [
        (ax1, matrix,   "Median arcsinh expression", "YlOrRd", (None, None)),
        (ax2, matrix_z, "Z-scored per marker",        "RdBu_r", (-2.5, 2.5)),
    ]:
        vmin, vmax = vrange
        im = ax.imshow(data, cmap=cmap, aspect="auto",
                       vmin=vmin or np.nanpercentile(data, 2),
                       vmax=vmax or np.nanpercentile(data, 98))
        ax.set_xticks(range(len(SUBTYPES)))
        ax.set_xticklabels(xlabels, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(show_markers)))
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)

        # Annotate raw values
        for i in range(len(show_markers)):
            for j in range(len(SUBTYPES)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=6.5, color="black")

    fig.suptitle("Marker expression per annotated cell subtype\n"
                 "(QC: each subtype should be high for expected markers, low for others)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = QC_DIR / "marker_expression.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── QC 3: Cross-marker contamination ──────────────────────────────────────────

def plot_contamination(adatas, thresholds):
    """
    For each subtype, show what % of cells are positive for markers they
    should NOT be positive for. High values = annotation errors.
    """
    rng = np.random.default_rng(42)

    # Pairs to check: (subtype, marker, should_be_positive)
    check_pairs = []
    for stype, checks in EXPECTED.items():
        for marker, expected_pos in checks:
            check_pairs.append((stype, marker, expected_pos))

    results = []
    for stype, marker, expected_pos in check_pairs:
        thr = thresholds.get(marker)
        if thr is None:
            continue
        vals = []
        for adata in adatas:
            mask = (adata.obs["cell_subtype"] == stype).values
            if mask.sum() < 2:
                continue
            idx = np.where(mask)[0]
            if len(idx) > 200:
                idx = rng.choice(idx, 200, replace=False)
            expr = get_expr(adata[idx], marker)
            if expr is not None:
                vals.extend(expr.tolist())
        if not vals:
            continue
        vals = np.array(vals)
        pos_frac = (vals >= thr).mean()
        results.append({
            "subtype": stype,
            "marker": marker,
            "expected_positive": expected_pos,
            "positive_fraction": pos_frac,
            "n_cells": len(vals),
        })

    df = pd.DataFrame(results)
    df["is_error"] = df.apply(
        lambda r: r["positive_fraction"] < 0.3 if r["expected_positive"]
                  else r["positive_fraction"] > 0.2, axis=1
    )

    # Separate expected-positive (should be high) and expected-negative (should be low)
    pos_df = df[df["expected_positive"]].copy()
    neg_df = df[~df["expected_positive"]].copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    def bar_panel(ax, data, title, threshold_line, err_color="#e74c3c", ok_color="#2ecc71"):
        if data.empty:
            return
        labels = [f"{LABELS.get(r.subtype, r.subtype)}\n{r.marker}" for _, r in data.iterrows()]
        vals   = data["positive_fraction"].values
        colors = [err_color if e else ok_color for e in data["is_error"].values]
        bars = ax.bar(range(len(vals)), vals * 100, color=colors, alpha=0.8, edgecolor="white")
        ax.axhline(threshold_line * 100, color="black", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"Threshold {threshold_line*100:.0f}%")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% cells positive for marker")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 100)
        # Annotate bars
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val*100:.0f}%", ha="center", va="bottom", fontsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    bar_panel(ax1, pos_df, "Expected POSITIVE markers\n(should be >30%)", 0.3)
    bar_panel(ax2, neg_df, "Expected NEGATIVE markers\n(should be <20%)",  0.2)

    fig.suptitle("Cross-marker contamination QC\n"
                 "Green = pass, Red = potential annotation error",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = QC_DIR / "contamination.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Print failures
    failures = df[df["is_error"]]
    if not failures.empty:
        print("\n  ⚠ Contamination QC failures:")
        for _, r in failures.iterrows():
            direction = "too low" if r["expected_positive"] else "too high"
            print(f"    {LABELS.get(r.subtype, r.subtype)} × {r.marker}: "
                  f"{r.positive_fraction*100:.1f}% pos ({direction})")
    else:
        print("  ✓ No contamination failures")

    return df


# ── QC 4: Annotation rate per ROI ─────────────────────────────────────────────

def plot_annotation_rates(adatas):
    """
    Per ROI: fraction of cells that are 'unknown' (pipeline) or 'other_immune'.
    Outlier ROIs (>50% unresolved) flagged.
    """
    records = []
    for adata in adatas:
        total = adata.n_obs
        subtypes = adata.obs["cell_subtype"]
        broad    = adata.obs["cell_type"]

        unknown_frac      = (broad == "unknown").sum() / total
        other_immune_frac = (subtypes == "other_immune").sum() / total
        unresolved_frac   = unknown_frac + other_immune_frac

        records.append({
            "donor":          adata.obs["donor"].iloc[0],
            "region":         adata.obs["region"].iloc[0],
            "roi":            adata.obs["roi"].iloc[0],
            "n_cells":        total,
            "unknown_frac":   unknown_frac,
            "other_immune_frac": other_immune_frac,
            "unresolved_frac": unresolved_frac,
        })

    df = pd.DataFrame(records).sort_values("unresolved_frac")
    outliers = df[df["unresolved_frac"] > 0.5]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram of unresolved fractions
    ax = axes[0]
    ax.hist(df["unresolved_frac"] * 100, bins=30, color="#3498db", alpha=0.7,
            edgecolor="white")
    ax.axvline(50, color="#e74c3c", linestyle="--", linewidth=1.5,
               label="50% threshold")
    ax.set_xlabel("% cells unresolved (unknown + other_immune)")
    ax.set_ylabel("Number of ROIs")
    ax.set_title(f"Annotation resolution per ROI\n"
                 f"(n={len(df)}, {len(outliers)} ROIs >50% unresolved)",
                 fontsize=11)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Stacked bar: breakdown by unknown vs other_immune, sorted
    ax = axes[1]
    ax.bar(range(len(df)), df["unknown_frac"] * 100,
           label="pipeline 'unknown'", color="#e74c3c", alpha=0.7)
    ax.bar(range(len(df)), df["other_immune_frac"] * 100,
           bottom=df["unknown_frac"] * 100,
           label="other_immune (gated)", color="#f39c12", alpha=0.7)
    ax.axhline(50, color="black", linestyle="--", linewidth=1.0)
    ax.set_xlabel("ROIs (sorted by unresolved fraction)")
    ax.set_ylabel("% unresolved cells")
    ax.set_title("Unresolved cells per ROI (stacked)", fontsize=11)
    ax.legend(fontsize=9)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.suptitle("Annotation completeness QC", fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = QC_DIR / "annotation_rates.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # Save QC CSV
    df.to_csv(QC_DIR / "qc_summary.csv", index=False)
    print(f"Saved {QC_DIR / 'qc_summary.csv'}")

    if not outliers.empty:
        print(f"\n  ⚠ {len(outliers)} ROIs with >50% unresolved:")
        for _, r in outliers.iterrows():
            print(f"    {r.donor}/{r.region}/{r.roi}: "
                  f"{r.unresolved_frac*100:.0f}% unresolved "
                  f"({r.n_cells} cells)")

    return df


# ── QC 5: Per-donor consistency ────────────────────────────────────────────────

def plot_donor_consistency(adatas):
    """
    Per-donor: cell type fraction distributions. Donors with unusual
    profiles are potential batch effects or annotation failures.
    """
    records = []
    for adata in adatas:
        total    = adata.n_obs
        subtypes = adata.obs["cell_subtype"]
        donor    = adata.obs["donor"].iloc[0]
        for stype in SUBTYPES:
            records.append({
                "donor": donor,
                "subtype": stype,
                "fraction": (subtypes == stype).sum() / total,
            })

    df = pd.DataFrame(records)
    donors = sorted(df["donor"].unique())

    # Focus on ABM-relevant types
    plot_types = ["macrophage", "cd8_t", "cd4_t", "fibroblast", "endothelial", "mdsc"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.ravel()

    for i, stype in enumerate(plot_types):
        ax = axes[i]
        sub = df[df["subtype"] == stype]
        per_donor = sub.groupby("donor")["fraction"].mean().sort_values()

        colors = []
        for d, val in per_donor.items():
            # Flag donors >3 IQR from median
            q1, q3 = per_donor.quantile(0.25), per_donor.quantile(0.75)
            iqr = q3 - q1
            colors.append("#e74c3c" if (val < q1 - 3*iqr or val > q3 + 3*iqr)
                          else "#3498db")

        ax.barh(range(len(per_donor)), per_donor.values * 100,
                color=colors, alpha=0.75)
        ax.axvline(per_donor.median() * 100, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.7)
        ax.set_yticks(range(len(per_donor)))
        ax.set_yticklabels(per_donor.index, fontsize=6.5)
        ax.set_xlabel("Mean fraction (%)")
        ax.set_title(LABELS.get(stype, stype), fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Flag outliers in title
        n_outliers = sum(1 for c in colors if c == "#e74c3c")
        if n_outliers:
            ax.set_title(f"{LABELS.get(stype, stype)} ⚠ {n_outliers} outlier donors",
                         fontsize=10, fontweight="bold", color="#e74c3c")

    fig.suptitle("Per-donor mean cell type fractions\n"
                 "(red = >3 IQR from median; potential batch effect or annotation issue)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = QC_DIR / "donor_consistency.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── QC 6: Broad cell type annotation QC ───────────────────────────────────────

# Expected defining marker + markers that should be low for each broad type
BROAD_TYPES = ["immune", "fibroblast", "endothelial", "epithelial",
               "beta cell", "alpha cell", "acinar", "unknown"]

BROAD_EXPECTED = {
    "immune":      [("CD45", True),  ("CD31", False), ("Collagen", False), ("pan-Keratin", False)],
    "fibroblast":  [("Collagen", True),  ("CD45", False), ("CD31", False), ("pan-Keratin", False)],
    "endothelial": [("CD31", True),      ("CD45", False), ("Collagen", False)],
    "epithelial":  [("pan-Keratin", True), ("CD45", False), ("CD31", False)],
    "beta cell":   [("CD45", False), ("pan-Keratin", False)],
    "alpha cell":  [("CD45", False), ("pan-Keratin", False)],
    "acinar":      [("CD45", False), ("CD31", False)],
    "unknown":     [],   # no expectations — just show its expression profile
}

# Pipeline marker → annotation threshold (Otsu-like, use median as proxy)
BROAD_MARKER_THRESHOLDS = {
    "Collagen":    0.3,
    "CD31":        0.2,
    "pan-Keratin": 0.3,
    "CD45":        0.2,
}


def plot_broad_type_qc(adatas):
    """
    Two panels:
    A) Median expression heatmap for all broad cell types × key structural markers
    B) Contamination: % cells in each broad type positive for markers they shouldn't express
       + expression of defining marker in the correct type (sanity check)
    """
    rng = np.random.default_rng(42)
    show_markers = ["CD45", "CD31", "Collagen", "pan-Keratin",
                    "Nd145_C-peptide", "Sm147_Glucagon", "Ho165_CA2",
                    "CD3", "CD4", "CD8", "CD68", "CD20"]
    show_markers_labels = ["CD45", "CD31", "Collagen", "pan-Keratin",
                           "C-peptide", "Glucagon", "CA2 (acinar)",
                           "CD3", "CD4", "CD8", "CD68", "CD20"]

    # ── Panel A: expression heatmap ────────────────────────────────────────────
    matrix = np.full((len(show_markers), len(BROAD_TYPES)), np.nan)

    for j, btype in enumerate(BROAD_TYPES):
        cell_exprs = {i: [] for i in range(len(show_markers))}
        for adata in adatas:
            mask = (adata.obs["cell_type"] == btype).values
            if mask.sum() < 2:
                continue
            idx = np.where(mask)[0]
            if len(idx) > 80:
                idx = rng.choice(idx, 80, replace=False)
            for i, marker in enumerate(show_markers):
                # Allow direct var_name match (for isotope-prefixed like Nd145_C-peptide)
                if marker in adata.var_names:
                    midx = list(adata.var_names).index(marker)
                    cell_exprs[i].extend(np.asarray(adata.X[idx, midx]).ravel().tolist())
                else:
                    expr = get_expr(adata[idx], marker)
                    if expr is not None:
                        cell_exprs[i].extend(expr.tolist())
        for i, vals in cell_exprs.items():
            if len(vals) >= 5:
                matrix[i, j] = np.median(vals)

    # Z-score per row
    matrix_z = np.full_like(matrix, np.nan)
    for i in range(len(show_markers)):
        row = matrix[i]
        valid = row[~np.isnan(row)]
        if len(valid) > 1 and valid.std() > 0:
            matrix_z[i] = (row - valid.mean()) / valid.std()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, data, title, cmap, vrange in [
        (axes[0], matrix,   "Median arcsinh expression", "YlOrRd", (None, None)),
        (axes[1], matrix_z, "Z-scored per marker",        "RdBu_r", (-2.5, 2.5)),
    ]:
        im = ax.imshow(data, cmap=cmap, aspect="auto",
                       vmin=vrange[0] or np.nanpercentile(data, 2),
                       vmax=vrange[1] or np.nanpercentile(data, 98))
        ax.set_xticks(range(len(BROAD_TYPES)))
        ax.set_xticklabels(BROAD_TYPES, rotation=35, ha="right", fontsize=9)
        ax.set_yticks(range(len(show_markers)))
        ax.set_yticklabels(show_markers_labels, fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
        for i in range(len(show_markers)):
            for j in range(len(BROAD_TYPES)):
                if not np.isnan(matrix[i, j]):
                    ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                            fontsize=6, color="black")

    fig.suptitle("Broad cell type annotation QC — marker expression heatmap\n"
                 "(pipeline annotations: each type should be high for its defining marker)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = QC_DIR / "broad_type_expression.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    # ── Panel B: contamination for structural markers ──────────────────────────
    results = []
    for btype, checks in BROAD_EXPECTED.items():
        for marker, expected_pos in checks:
            thr = BROAD_MARKER_THRESHOLDS.get(marker, 0.25)
            vals = []
            for adata in adatas:
                mask = (adata.obs["cell_type"] == btype).values
                if mask.sum() < 2:
                    continue
                idx = np.where(mask)[0]
                if len(idx) > 200:
                    idx = rng.choice(idx, 200, replace=False)
                expr = get_expr(adata[idx], marker)
                if expr is not None:
                    vals.extend(expr.tolist())
            if not vals:
                continue
            vals = np.array(vals)
            pos_frac = (vals >= thr).mean()
            is_error = (pos_frac < 0.25 if expected_pos else pos_frac > 0.25)
            results.append({
                "broad_type": btype, "marker": marker,
                "expected_positive": expected_pos,
                "positive_fraction": pos_frac,
                "is_error": is_error,
            })

    df = pd.DataFrame(results)
    pos_df = df[df["expected_positive"]]
    neg_df = df[~df["expected_positive"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for ax, data, title, thr_line in [
        (ax1, pos_df, "Defining markers: should be HIGH (>25%)", 0.25),
        (ax2, neg_df, "Exclusion markers: should be LOW (<25%)",  0.25),
    ]:
        if data.empty:
            continue
        labels = [f"{r.broad_type}\n× {r.marker}" for _, r in data.iterrows()]
        vals   = data["positive_fraction"].values
        colors = ["#e74c3c" if e else "#2ecc71" for e in data["is_error"].values]
        bars = ax.bar(range(len(vals)), vals * 100, color=colors, alpha=0.8,
                      edgecolor="white")
        ax.axhline(thr_line * 100, color="black", linestyle="--",
                   linewidth=1.2, alpha=0.7, label=f"{thr_line*100:.0f}% threshold")
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("% cells positive for marker")
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 105)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                    f"{val*100:.0f}%", ha="center", va="bottom", fontsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Broad cell type contamination QC\n"
                 "Green = pass, Red = potential pipeline annotation error",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out = QC_DIR / "broad_type_contamination.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")

    failures = df[df["is_error"]]
    if not failures.empty:
        print(f"\n  ⚠ Broad type QC failures:")
        for _, r in failures.iterrows():
            direction = "too low" if r["expected_positive"] else "too high"
            print(f"    {r.broad_type} × {r.marker}: "
                  f"{r.positive_fraction*100:.1f}% pos ({direction})")
    else:
        print("  ✓ No broad type QC failures")


# ── QC 7: Panel comparison ─────────────────────────────────────────────────────

def plot_panel_comparison(adatas):
    """
    Compare cell type fractions between CD11b-panel and CD163-panel donors.
    If MDSC rates are wildly different, the CD14-only gating is unreliable.
    """
    records = []
    for adata in adatas:
        var_names = list(adata.var_names)
        has_cd11b  = any(v.endswith("_CD11b") for v in var_names)
        has_cd163s = any(v.endswith("_CD163") for v in var_names)
        panel = ("CD11b" if has_cd11b and not has_cd163s else
                 "CD163" if has_cd163s and not has_cd11b else "both")
        total = adata.n_obs
        subtypes = adata.obs["cell_subtype"]
        for stype in ["mdsc", "macrophage", "cd8_t", "cd4_t"]:
            records.append({
                "panel": panel,
                "subtype": stype,
                "fraction": (subtypes == stype).sum() / total,
            })

    df = pd.DataFrame(records)
    panels = [p for p in ["CD11b", "CD163", "both"] if p in df["panel"].values]
    plot_types = ["mdsc", "macrophage", "cd8_t", "cd4_t"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5))
    panel_colors = {"CD11b": "#3498db", "CD163": "#e74c3c", "both": "#2ecc71"}

    for i, stype in enumerate(plot_types):
        ax = axes[i]
        sub = df[df["subtype"] == stype]
        data_by_panel = [sub[sub["panel"] == p]["fraction"].values * 100
                         for p in panels]
        bp = ax.boxplot(data_by_panel, patch_artist=True, labels=panels,
                        medianprops={"color": "black", "linewidth": 1.5})
        for patch, panel in zip(bp["boxes"], panels):
            patch.set_facecolor(panel_colors.get(panel, "gray"))
            patch.set_alpha(0.7)

        # Mann-Whitney U test between CD11b and CD163 if both present
        if "CD11b" in panels and "CD163" in panels:
            a = sub[sub["panel"] == "CD11b"]["fraction"].values
            b = sub[sub["panel"] == "CD163"]["fraction"].values
            if len(a) > 3 and len(b) > 3:
                from scipy.stats import mannwhitneyu
                _, pval = mannwhitneyu(a, b, alternative="two-sided")
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                ymax = ax.get_ylim()[1]
                ax.text(0.5, 0.97, f"p={pval:.3f} {sig}", transform=ax.transAxes,
                        ha="center", va="top", fontsize=9,
                        color="#e74c3c" if pval < 0.05 else "gray")

        ax.set_ylabel("Cell fraction (%)")
        ax.set_title(LABELS.get(stype, stype), fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Panel sizes
    panel_counts = df.groupby("panel")["fraction"].count() // len(plot_types)
    size_str = " | ".join(f"{p}: {n} ROIs" for p, n in panel_counts.items())
    fig.suptitle(f"Panel comparison: CD11b vs CD163 donors\n({size_str})\n"
                 "Key check: MDSC rate should be comparable between panels",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = QC_DIR / "panel_comparison.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    thr_path = BASE / "gating_thresholds.json"
    if not thr_path.exists():
        print("ERROR: gating_thresholds.json not found. Run immune_subtype.py first.")
        return
    with open(thr_path) as f:
        thresholds = json.load(f)

    print("Loading all ROIs...")
    adatas = load_all()

    print("\n[1/7] GMM threshold plots...")
    plot_gmm_thresholds(adatas, thresholds)

    print("\n[2/7] Broad cell type annotation QC...")
    plot_broad_type_qc(adatas)

    print("\n[3/7] Immune subtype marker expression heatmap...")
    plot_marker_expression(adatas)

    print("\n[4/7] Cross-marker contamination (subtypes)...")
    plot_contamination(adatas, thresholds)

    print("\n[5/7] Annotation rates per ROI...")
    qc_df = plot_annotation_rates(adatas)

    print("\n[6/7] Per-donor consistency...")
    plot_donor_consistency(adatas)

    print("\n[7/7] Panel comparison (CD11b vs CD163)...")
    plot_panel_comparison(adatas)

    print(f"\nAll QC figures saved to {QC_DIR}")

    # Overall summary
    unresolved = qc_df["unresolved_frac"].mean()
    print(f"\n=== QC summary ===")
    print(f"  Mean unresolved fraction: {unresolved*100:.1f}%")
    print(f"  ROIs >50% unresolved: {(qc_df['unresolved_frac'] > 0.5).sum()}")


if __name__ == "__main__":
    main()
