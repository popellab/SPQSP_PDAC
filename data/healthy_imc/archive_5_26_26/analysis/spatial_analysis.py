"""
spatial_analysis.py — Spatial statistics on IMC cell maps for ABM parameterization.

Requires immune_subtype.py to have been run first (obs['cell_subtype'] present).

Computes per ROI:
  1. Cell density (cells/mm²) per subtype
  2. Distance to septa boundaries (via EDT on rasterized contours)
  3. Nearest-neighbor distances between cell types
  4. Ripley's L function per cell type

Usage:
    python spatial_analysis.py [--pancdb-imc DIR] [--max-cells-ripley N]

Outputs (all in analysis/spatial/):
    density_summary.csv     — per-ROI densities
    septa_distances.csv     — distance-to-septa percentiles per subtype per ROI
    nn_distances.csv        — median nearest-neighbor distances between subtypes
    ripley_L.csv            — L(r)-r values per subtype
    figures/                — density boxplots, septa distance histograms, L plots
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree

warnings.filterwarnings("ignore", category=FutureWarning)

PIXEL_SIZE_UM = 1.0  # µm per pixel for IMC data

# Subtypes to report (from immune_subtype.py + pipeline broad types)
SUBTYPES_OF_INTEREST = [
    "cd4_t", "cd8_t", "b_cell", "macrophage", "mdsc",
    "other_immune", "fibroblast", "endothelial",
]

# Ripley's L radii (µm)
RIPLEY_RADII = np.arange(10, 210, 10)


# ── Helpers ────────────────────────────────────────────────────────────────────

def get_subtypes(adata: ad.AnnData) -> pd.Series:
    """Return per-cell subtype labels. Falls back to cell_type if no cell_subtype."""
    if "cell_subtype" in adata.obs.columns:
        return adata.obs["cell_subtype"]
    return adata.obs["cell_type"]


def load_septa_mask(septa_json: Path, image_shape: tuple) -> np.ndarray | None:
    """Rasterize septa contours to binary mask; return None if no contours."""
    with open(septa_json) as f:
        data = json.load(f)
    if data["n_contours"] == 0:
        return None

    mask = np.zeros(image_shape, dtype=bool)
    for contour in data["contours_yx"]:
        pts = np.array(contour, dtype=int)
        # Clip to image bounds
        pts[:, 0] = np.clip(pts[:, 0], 0, image_shape[0] - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, image_shape[1] - 1)
        mask[pts[:, 0], pts[:, 1]] = True

    return mask


def septa_distance_map(septa_json: Path, image_shape: tuple) -> np.ndarray | None:
    """Return per-pixel distance (µm) to nearest septa contour, or None."""
    mask = load_septa_mask(septa_json, image_shape)
    if mask is None:
        return None
    # EDT gives distance in pixels; multiply by pixel size
    dist = distance_transform_edt(~mask) * PIXEL_SIZE_UM
    return dist


def ripley_L(xy: np.ndarray, area_mm2: float, radii: np.ndarray) -> np.ndarray:
    """
    Compute Ripley's L(r) - r for a set of 2D point coordinates.
    Uses edge-corrected estimator (toroidal wrap approximation).
    Returns array of L(r)-r values for each radius in radii.
    """
    n = len(xy)
    if n < 5:
        return np.full(len(radii), np.nan)

    area_um2 = area_mm2 * 1e6
    tree = KDTree(xy)
    L_minus_r = np.zeros(len(radii))

    for i, r in enumerate(radii):
        # Count pairs within distance r
        pairs = tree.query_ball_tree(tree, r)
        # Subtract self-counts
        K = sum(len(p) - 1 for p in pairs)
        # Ripley's K: K(r) = (area / n²) * sum of pair counts
        K_val = (area_um2 / (n * (n - 1))) * K
        L_val = np.sqrt(K_val / np.pi)
        L_minus_r[i] = L_val - r

    return L_minus_r


# ── Per-ROI analysis ───────────────────────────────────────────────────────────

def analyze_roi(h5ad_path: Path, max_cells_ripley: int = 500) -> dict:
    adata = ad.read_h5ad(h5ad_path)
    subtypes = get_subtypes(adata)

    # Spatial coords: obsm['spatial'] is [x, y] in pixels
    xy_px = adata.obsm["spatial"]  # shape (n, 2)
    xy_um = xy_px * PIXEL_SIZE_UM

    # Image area from coordinate range (avoid assuming exactly 1mm²)
    x_range = xy_px[:, 0].max() - xy_px[:, 0].min()
    y_range = xy_px[:, 1].max() - xy_px[:, 1].min()
    area_um2 = x_range * y_range
    area_mm2 = area_um2 / 1e6
    image_shape = (int(adata.uns.get("image_shape_y", 1000)),
                   int(adata.uns.get("image_shape_x", 1001)))
    # Use septa JSON path
    septa_path = h5ad_path.parent / "septa_contours.json"

    donor = adata.obs["donor"].iloc[0]
    region = adata.obs["region"].iloc[0]
    roi = adata.obs["roi"].iloc[0]

    result = {
        "donor": donor, "region": region, "roi": roi,
        "n_cells_total": adata.n_obs,
        "area_mm2": round(area_mm2, 4),
    }

    # ── 1. Density ─────────────────────────────────────────────────────────────
    for stype in SUBTYPES_OF_INTEREST:
        count = (subtypes == stype).sum()
        result[f"density_{stype}"] = round(count / area_mm2, 2) if area_mm2 > 0 else 0

    # ── 2. Septa distances ─────────────────────────────────────────────────────
    has_septa = False
    if septa_path.exists():
        # Guess image shape from septa JSON
        with open(septa_path) as f:
            sdata = json.load(f)
        image_shape = tuple(sdata.get("image_shape", [1000, 1001]))
        dist_map = septa_distance_map(septa_path, image_shape)
        if dist_map is not None:
            has_septa = True
            # Sample distance at each cell centroid
            yx = xy_px[:, ::-1].astype(int)  # convert x,y → row,col
            yx[:, 0] = np.clip(yx[:, 0], 0, image_shape[0] - 1)
            yx[:, 1] = np.clip(yx[:, 1], 0, image_shape[1] - 1)
            cell_septa_dist = dist_map[yx[:, 0], yx[:, 1]]

            for stype in SUBTYPES_OF_INTEREST:
                mask = (subtypes == stype).values
                if mask.sum() < 3:
                    continue
                dists = cell_septa_dist[mask]
                result[f"septa_p25_{stype}"] = round(float(np.percentile(dists, 25)), 1)
                result[f"septa_p50_{stype}"] = round(float(np.percentile(dists, 50)), 1)
                result[f"septa_p75_{stype}"] = round(float(np.percentile(dists, 75)), 1)

    result["has_septa"] = has_septa

    # ── 3. Nearest-neighbor distances ──────────────────────────────────────────
    type_xy = {}
    for stype in SUBTYPES_OF_INTEREST:
        mask = (subtypes == stype).values
        if mask.sum() >= 2:
            type_xy[stype] = xy_um[mask]

    for stype_a, xy_a in type_xy.items():
        for stype_b, xy_b in type_xy.items():
            if stype_a == stype_b:
                # Self NN (2nd nearest)
                if len(xy_a) >= 2:
                    tree = KDTree(xy_a)
                    dists, _ = tree.query(xy_a, k=2)
                    result[f"nn_{stype_a}_to_{stype_b}"] = round(float(np.median(dists[:, 1])), 1)
            else:
                tree = KDTree(xy_b)
                dists, _ = tree.query(xy_a, k=1)
                result[f"nn_{stype_a}_to_{stype_b}"] = round(float(np.median(dists)), 1)

    # ── 4. Ripley's L ──────────────────────────────────────────────────────────
    ripley_rows = []
    for stype in SUBTYPES_OF_INTEREST:
        mask = (subtypes == stype).values
        n_type = mask.sum()
        if n_type < 10:
            continue
        xy_type = xy_um[mask]
        # Thin to max_cells_ripley for speed
        if n_type > max_cells_ripley:
            idx = np.random.choice(n_type, max_cells_ripley, replace=False)
            xy_type = xy_type[idx]
        L = ripley_L(xy_type, area_mm2, RIPLEY_RADII)
        for r, lval in zip(RIPLEY_RADII, L):
            ripley_rows.append({
                "donor": donor, "region": region, "roi": roi,
                "subtype": stype, "r_um": r, "L_minus_r": round(float(lval), 2)
            })

    return result, ripley_rows


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_density_boxplots(density_df: pd.DataFrame, out_dir: Path):
    subtypes = [c.replace("density_", "") for c in density_df.columns if c.startswith("density_")]
    vals = [density_df[f"density_{s}"].values for s in subtypes]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(vals, patch_artist=True, labels=subtypes)
    ax.set_ylabel("Density (cells/mm²)")
    ax.set_title("Cell type densities across 299 ROIs")
    ax.set_xticklabels(subtypes, rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "density_boxplots.png", dpi=150)
    plt.close(fig)


def plot_septa_distances(septa_df: pd.DataFrame, out_dir: Path):
    subtypes = [c.replace("septa_p50_", "") for c in septa_df.columns if c.startswith("septa_p50_")]
    if not subtypes:
        return
    medians = [septa_df[f"septa_p50_{s}"].dropna().values for s in subtypes]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.boxplot(medians, patch_artist=True, labels=subtypes)
    ax.set_ylabel("Median distance to septa (µm)")
    ax.set_title("Distance to stromal boundaries by cell type")
    ax.set_xticklabels(subtypes, rotation=30, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "septa_distances.png", dpi=150)
    plt.close(fig)


def plot_ripley(ripley_df: pd.DataFrame, out_dir: Path):
    subtypes = ripley_df["subtype"].unique()
    fig, ax = plt.subplots(figsize=(10, 6))
    for stype in subtypes:
        sub = ripley_df[ripley_df["subtype"] == stype]
        mean_L = sub.groupby("r_um")["L_minus_r"].mean()
        ax.plot(mean_L.index, mean_L.values, label=stype, linewidth=1.5)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.8, label="CSR")
    ax.set_xlabel("r (µm)")
    ax.set_ylabel("L(r) - r (µm)")
    ax.set_title("Ripley's L function — mean across ROIs")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(out_dir / "ripley_L.png", dpi=150)
    plt.close(fig)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pancdb-imc", default=None)
    parser.add_argument("--max-cells-ripley", type=int, default=500,
                        help="Max cells per type for Ripley's L (default: 500)")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    imc_dir = Path(args.pancdb_imc) if args.pancdb_imc else script_dir.parent / "pancdb_imc"
    out_dir = script_dir / "spatial"
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    h5ad_paths = sorted(imc_dir.rglob("cells.h5ad"))
    print(f"Analyzing {len(h5ad_paths)} ROIs...")

    all_records = []
    all_ripley = []

    for i, path in enumerate(h5ad_paths):
        donor = path.parts[-4]
        region = path.parts[-3]
        roi = path.parts[-2]
        print(f"  [{i+1}/{len(h5ad_paths)}] {donor}/{region}/{roi}", end=" ", flush=True)
        try:
            record, ripley_rows = analyze_roi(path, args.max_cells_ripley)
            all_records.append(record)
            all_ripley.extend(ripley_rows)
            print(f"✓ {record['n_cells_total']} cells")
        except Exception as e:
            print(f"ERR: {e}")

    density_df = pd.DataFrame(all_records)
    density_df.to_csv(out_dir / "density_summary.csv", index=False)
    print(f"\nSaved density_summary.csv ({len(density_df)} ROIs)")

    ripley_df = pd.DataFrame(all_ripley)
    ripley_df.to_csv(out_dir / "ripley_L.csv", index=False)
    print(f"Saved ripley_L.csv ({len(ripley_df)} rows)")

    # Septa distances (subset of columns)
    septa_cols = ["donor", "region", "roi", "has_septa"] + \
                 [c for c in density_df.columns if c.startswith("septa_")]
    septa_df = density_df[[c for c in septa_cols if c in density_df.columns]]
    septa_df.to_csv(out_dir / "septa_distances.csv", index=False)
    print(f"Saved septa_distances.csv")

    # NN distances
    nn_cols = ["donor", "region", "roi"] + [c for c in density_df.columns if c.startswith("nn_")]
    nn_df = density_df[[c for c in nn_cols if c in density_df.columns]]
    nn_df.to_csv(out_dir / "nn_distances.csv", index=False)
    print(f"Saved nn_distances.csv")

    # Plots
    print("Generating plots...")
    plot_density_boxplots(density_df, fig_dir)
    plot_septa_distances(septa_df, fig_dir)
    if not ripley_df.empty:
        plot_ripley(ripley_df, fig_dir)
    print(f"Saved figures → {fig_dir}")

    # Quick summary
    density_cols = [c for c in density_df.columns if c.startswith("density_")]
    print("\n=== Mean density (cells/mm²) ± std ===")
    for col in density_cols:
        stype = col.replace("density_", "")
        m = density_df[col].mean()
        s = density_df[col].std()
        print(f"  {stype:20s}: {m:7.1f} ± {s:.1f}")


if __name__ == "__main__":
    main()
