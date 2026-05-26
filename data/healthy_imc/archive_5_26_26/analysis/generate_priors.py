"""
generate_priors.py — Generate ABM initialization priors from filtered IMC data.

Pipeline:
  1. Filter out ROIs with >50% unresolved (from qc_summary.csv)
  2. Add DC candidate sub-gate to other_immune (HLA-DR-high/CD68-low; provisional only)
  3. For each cell type fit density distributions (lognormal)
  4. Compute septa vs lobule placement fractions (cells within SEPTA_THRESHOLD_UM = in septa voxel)
  5. Fit NN-distance distributions (same-type and cross-type)
  6. Pull Ripley clustering scale from ripley_L.csv
  7. Apply Abercrombie 2D→3D conversion
  8. Write abm_priors.json + abm_priors_summary.csv

ABM context:
  - Domain: 50³ voxels × 20µm = 1mm³, 125,000 voxels
  - Voxels labelled lobule or septa
  - septa_fraction = fraction of cells expected in septa voxels
  - clustering_radius_um = Ripley L peak (characteristic aggregation scale)
  - nn_to_X_um = median distance to nearest cell of type X (for relative placement)

Usage:
    python generate_priors.py [--max-unresolved 0.5] [--septa-threshold 20]
                              [--section-thickness 5]
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats
from scipy.spatial import KDTree
from sklearn.mixture import GaussianMixture

BASE       = Path(__file__).parent
IMC_DIR    = BASE.parent / "pancdb_imc"
SPATIAL    = BASE / "spatial"
QC_DIR     = BASE / "qc"

# ABM constants
ABM_VOXEL_UM   = 20.0
ABM_GRID_SIDE  = 50
ABM_DOMAIN_MM3 = (ABM_VOXEL_UM * ABM_GRID_SIDE / 1000) ** 3
ABM_N_VOXELS   = ABM_GRID_SIDE ** 3
SECTION_THICKNESS_UM = 5.0

CELL_DIAMETERS_UM = {
    "cd4_t": 8.0, "cd8_t": 8.0, "b_cell": 8.0, "dc_candidate": 9.0,
    "macrophage": 15.0, "mdsc": 10.0, "other_immune": 9.0,
    "fibroblast": 15.0, "endothelial": 10.0,
}

PIXEL_SIZE_UM = 1.0

LABELS = {
    "cd4_t": "CD4⁺ T", "cd8_t": "CD8⁺ T", "b_cell": "B cell",
    "dc_candidate": "DC (candidate)", "macrophage": "Macrophage",
    "mdsc": "MDSC", "other_immune": "Other immune",
    "fibroblast": "Fibroblast", "endothelial": "Endothelial",
}

ABM_AGENT = {
    "cd4_t": "TReg", "cd8_t": "TCell", "b_cell": "B_cell(future)",
    "dc_candidate": "DC(future)", "macrophage": "Macrophage",
    "mdsc": "MDSC", "other_immune": "(unassigned)",
    "fibroblast": "Fibroblast", "endothelial": "VascularCell",
}

MARKER_SUFFIXES = {
    "CD3":   ["CD3"], "CD4":   ["CD4"], "CD8":   ["CD8","CD8a"],
    "CD20":  ["CD20"], "CD68":  ["CD68"], "CD11b": ["CD11b"],
    "CD14":  ["CD14"], "HLA-DR": ["HLA-DR"],
}


def find_var(var_names, suffixes):
    for s in suffixes:
        for v in var_names:
            if v.endswith("_" + s):
                return v
    return None


def get_expr(adata, marker, idx_subset=None):
    var = find_var(list(adata.var_names), MARKER_SUFFIXES.get(marker, [marker]))
    if var is None:
        return None
    col = list(adata.var_names).index(var)
    X = adata.X if idx_subset is None else adata.X[idx_subset]
    return np.asarray(X[:, col]).ravel()


# ── 1. Load thresholds + build good-ROI list ──────────────────────────────────

def load_good_rois(max_unresolved: float) -> set[str]:
    qc_path = QC_DIR / "qc_summary.csv"
    if not qc_path.exists():
        print("  qc_summary.csv not found — using all ROIs")
        return None
    df = pd.read_csv(qc_path)
    good = df[df["unresolved_frac"] <= max_unresolved]
    keys = set(f"{r.donor}|{r.region}|{r.roi}" for _, r in good.iterrows())
    dropped = len(df) - len(good)
    print(f"  Keeping {len(good)}/{len(df)} ROIs (dropped {dropped} with >{max_unresolved*100:.0f}% unresolved)")
    return keys


def load_thresholds() -> dict:
    p = BASE / "gating_thresholds.json"
    with open(p) as f:
        return json.load(f)


# ── 2. DC candidate gate ──────────────────────────────────────────────────────

def add_dc_gate(adata: ad.AnnData, thresholds: dict,
                hladr_percentile: float = 80) -> ad.AnnData:
    """
    Within other_immune: flag HLA-DR-high / CD68-low / CD3-low cells as dc_candidate.
    Uses a per-ROI HLA-DR percentile threshold (no global GMM — insufficient bimodality).
    NOTE: provisional label only; no CD11c in panel.
    """
    if "cell_subtype" not in adata.obs.columns:
        return adata

    subtypes = adata.obs["cell_subtype"].copy()
    other_mask = (subtypes == "other_immune").values

    if other_mask.sum() < 10:
        return adata

    hladr_expr = get_expr(adata, "HLA-DR")
    cd68_expr  = get_expr(adata, "CD68")
    cd3_expr   = get_expr(adata, "CD3")

    if hladr_expr is None:
        return adata

    # Per-ROI HLA-DR threshold: top percentile of all immune cells
    immune_mask = (adata.obs["cell_type"] == "immune").values
    if immune_mask.sum() < 10:
        return adata
    hladr_thr = np.percentile(hladr_expr[immune_mask], hladr_percentile)

    cd68_thr = thresholds.get("CD68", 0.454)
    cd3_thr  = thresholds.get("CD3",  0.385)

    dc_mask = (
        other_mask &
        (hladr_expr >= hladr_thr) &
        ((cd68_expr is None) | (cd68_expr < cd68_thr)) &
        ((cd3_expr is None)  | (cd3_expr  < cd3_thr))
    )
    subtypes = subtypes.astype(str)
    subtypes.iloc[np.where(dc_mask)[0]] = "dc_candidate"
    adata.obs["cell_subtype"] = subtypes.values
    return adata


# ── 3. Per-ROI data extraction ────────────────────────────────────────────────

SUBTYPES = ["cd4_t", "cd8_t", "b_cell", "dc_candidate",
            "macrophage", "mdsc", "fibroblast", "endothelial"]


def process_roi(adata: ad.AnnData, septa_threshold_um: float) -> dict | None:
    """Extract density, septa fractions, NN distances for one ROI."""
    import json as _json
    from scipy.ndimage import distance_transform_edt

    subtypes  = adata.obs["cell_subtype"]
    xy_px     = adata.obsm["spatial"]           # [x, y] pixels
    xy_um     = xy_px * PIXEL_SIZE_UM

    x_range = xy_px[:, 0].max() - xy_px[:, 0].min()
    y_range = xy_px[:, 1].max() - xy_px[:, 1].min()
    area_mm2 = (x_range * y_range) / 1e6
    if area_mm2 < 0.1:
        return None

    result = {"area_mm2": area_mm2}

    # ── Densities ──────────────────────────────────────────────────────────────
    for st in SUBTYPES:
        result[f"density_{st}"] = (subtypes == st).sum() / area_mm2

    # ── Septa distance (lobule/septa fraction) ─────────────────────────────────
    septa_path = adata.uns.get("septa_path", None)
    # Derive from h5ad path via obs metadata
    donor  = adata.obs["donor"].iloc[0]
    region = adata.obs["region"].iloc[0]
    roi    = adata.obs["roi"].iloc[0]
    sp = IMC_DIR / donor / region / roi / "septa_contours.json"

    septa_dist = None
    if sp.exists():
        with open(sp) as f:
            sdata = _json.load(f)
        if sdata["n_contours"] > 0:
            image_shape = tuple(sdata["image_shape"])
            mask = np.zeros(image_shape, dtype=bool)
            for contour in sdata["contours_yx"]:
                pts = np.clip(np.array(contour, dtype=int), 0,
                              [image_shape[0]-1, image_shape[1]-1])
                mask[pts[:, 0], pts[:, 1]] = True
            dist_map = distance_transform_edt(~mask) * PIXEL_SIZE_UM
            yx = xy_px[:, ::-1].astype(int)
            yx[:, 0] = np.clip(yx[:, 0], 0, image_shape[0]-1)
            yx[:, 1] = np.clip(yx[:, 1], 0, image_shape[1]-1)
            septa_dist = dist_map[yx[:, 0], yx[:, 1]]

    for st in SUBTYPES:
        mask_st = (subtypes == st).values
        if mask_st.sum() < 2:
            result[f"septa_frac_{st}"] = np.nan
            result[f"septa_dist_median_{st}"] = np.nan
            continue
        if septa_dist is not None:
            d = septa_dist[mask_st]
            result[f"septa_frac_{st}"]        = (d <= septa_threshold_um).mean()
            result[f"septa_dist_median_{st}"] = float(np.median(d))
        else:
            result[f"septa_frac_{st}"]        = np.nan
            result[f"septa_dist_median_{st}"] = np.nan

    # ── NN distances ───────────────────────────────────────────────────────────
    type_xy = {}
    for st in SUBTYPES:
        mask_st = (subtypes == st).values
        if mask_st.sum() >= 2:
            type_xy[st] = xy_um[mask_st]

    for st_a, xy_a in type_xy.items():
        for st_b, xy_b in type_xy.items():
            col = f"nn_{st_a}_to_{st_b}"
            if st_a == st_b:
                if len(xy_a) >= 2:
                    tree = KDTree(xy_a)
                    d, _ = tree.query(xy_a, k=2)
                    result[col] = float(np.median(d[:, 1]))
                else:
                    result[col] = np.nan
            else:
                tree = KDTree(xy_b)
                d, _ = tree.query(xy_a, k=1)
                result[col] = float(np.median(d))

    return result


# ── 4. Distribution fitting ───────────────────────────────────────────────────

def fit_lognormal(values: np.ndarray) -> dict | None:
    """Fit lognormal to non-zero values. Returns dict with mu, sigma, mean, std."""
    nonzero = values[np.isfinite(values) & (values > 0)]
    if len(nonzero) < 10:
        return None
    shape, loc, scale = stats.lognorm.fit(nonzero, floc=0)
    mu    = float(np.log(scale))
    sigma = float(shape)
    return {
        "distribution": "lognormal",
        "mu": round(mu, 4),
        "sigma": round(sigma, 4),
        "mean": round(float(np.exp(mu + sigma**2 / 2)), 2),
        "median": round(float(np.exp(mu)), 2),
        "std": round(float(np.sqrt((np.exp(sigma**2)-1)*np.exp(2*mu+sigma**2))), 2),
        "zero_fraction": round(float((values == 0).mean()), 3),
        "n_rois": int(len(values)),
        "n_nonzero": int(len(nonzero)),
    }


def abercrombie_3d(density_2d: float, t_um: float, d_um: float) -> float:
    return density_2d / ((t_um + d_um) / 1000.0)


def ripley_peak(ripley_df: pd.DataFrame, subtype: str) -> float | None:
    sub = ripley_df[ripley_df["subtype"] == subtype]
    if sub.empty:
        return None
    mean_L = sub.groupby("r_um")["L_minus_r"].mean()
    if mean_L.max() <= 0:
        return None
    return float(mean_L.idxmax())


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-unresolved",   type=float, default=0.5)
    parser.add_argument("--septa-threshold",  type=float, default=20.0,
                        help="Distance (µm) within which a cell is 'septa-associated'")
    parser.add_argument("--section-thickness", type=float, default=SECTION_THICKNESS_UM)
    args = parser.parse_args()

    print("=== ABM Prior Generation ===\n")

    good_rois  = load_good_rois(args.max_unresolved)
    thresholds = load_thresholds()

    ripley_path = SPATIAL / "ripley_L.csv"
    ripley_df   = pd.read_csv(ripley_path) if ripley_path.exists() else pd.DataFrame()

    # ── Load + filter + DC-gate all ROIs ──────────────────────────────────────
    h5ad_paths = sorted(IMC_DIR.rglob("cells.h5ad"))
    all_records = []
    n_kept = n_dropped = n_dc_added = 0

    print("Loading ROIs and extracting per-ROI statistics...")
    for path in h5ad_paths:
        adata = ad.read_h5ad(path)
        if "cell_subtype" not in adata.obs.columns:
            n_dropped += 1
            continue
        key = f"{adata.obs['donor'].iloc[0]}|{adata.obs['region'].iloc[0]}|{adata.obs['roi'].iloc[0]}"
        if good_rois is not None and key not in good_rois:
            n_dropped += 1
            continue

        # Add DC gate
        n_before = (adata.obs["cell_subtype"] == "dc_candidate").sum()
        adata = add_dc_gate(adata, thresholds)
        n_dc_added += (adata.obs["cell_subtype"] == "dc_candidate").sum() - n_before

        rec = process_roi(adata, args.septa_threshold)
        if rec:
            rec["donor"]  = adata.obs["donor"].iloc[0]
            rec["region"] = adata.obs["region"].iloc[0]
            rec["roi"]    = adata.obs["roi"].iloc[0]
            all_records.append(rec)
            n_kept += 1

    print(f"  Kept {n_kept} ROIs, dropped {n_dropped}")
    print(f"  DC candidates added: {n_dc_added:,} cells across {n_kept} ROIs\n")

    df = pd.DataFrame(all_records)
    df.to_csv(BASE / "filtered_roi_data.csv", index=False)

    # ── Build priors ──────────────────────────────────────────────────────────
    priors = {}
    t = args.section_thickness

    for st in SUBTYPES:
        d_um   = CELL_DIAMETERS_UM.get(st, 10.0)
        label  = LABELS.get(st, st)
        agent  = ABM_AGENT.get(st, "?")

        # Density 2D
        d2d_col = f"density_{st}"
        d2d_vals = df[d2d_col].dropna().values if d2d_col in df else np.array([])
        d2d_fit = fit_lognormal(d2d_vals)

        # Density 3D (Abercrombie)
        if d2d_fit:
            d3d_vals = np.array([abercrombie_3d(v, t, d_um) for v in d2d_vals if v > 0])
            d3d_fit  = fit_lognormal(d3d_vals)
            cells_per_domain = float(np.exp(d3d_fit["mu"] + d3d_fit["sigma"]**2/2)) if d3d_fit else 0.0
        else:
            d3d_fit, cells_per_domain = None, 0.0

        # Septa fraction
        sf_col  = f"septa_frac_{st}"
        sf_vals = df[sf_col].dropna().values if sf_col in df else np.array([])
        septa_frac_mean = float(np.nanmean(sf_vals)) if len(sf_vals) >= 5 else None
        septa_frac_std  = float(np.nanstd(sf_vals))  if len(sf_vals) >= 5 else None

        # Septa distance distribution
        sd_col   = f"septa_dist_median_{st}"
        sd_vals  = df[sd_col].dropna().values if sd_col in df else np.array([])
        sd_fit   = fit_lognormal(sd_vals) if len(sd_vals) >= 10 else None

        # NN distances to self and other types
        nn_priors = {}
        for st_b in SUBTYPES:
            nn_col = f"nn_{st}_to_{st_b}"
            if nn_col not in df:
                continue
            nn_vals = df[nn_col].dropna().values
            nn_fit  = fit_lognormal(nn_vals) if len(nn_vals) >= 10 else None
            if nn_fit:
                nn_priors[st_b] = {"median_um": nn_fit["median"], **nn_fit}

        # Ripley clustering scale
        clust_r = ripley_peak(ripley_df, st) if not ripley_df.empty else None

        priors[st] = {
            "label":          label,
            "abm_agent":      agent,
            "cell_diameter_um": d_um,
            "density_2d":     d2d_fit,
            "density_3d":     d3d_fit,
            "cells_per_abm_domain_mean": round(cells_per_domain, 1),
            "septa_fraction": {
                "mean": round(septa_frac_mean, 3) if septa_frac_mean is not None else None,
                "std":  round(septa_frac_std, 3)  if septa_frac_std  is not None else None,
                "note": f"fraction of cells within {args.septa_threshold}µm of septa contour",
            },
            "septa_distance_um": sd_fit,
            "clustering_radius_um": clust_r,
            "nn_distances_um": nn_priors,
            "dc_note": ("Provisional gate: HLA-DR-high/CD68-low/CD3-low within other_immune. "
                        "No CD11c in panel — cannot confirm DC identity.") if st == "dc_candidate" else None,
        }

        # Console summary
        d2d_str  = f"{d2d_fit['median']:.0f}/mm²"  if d2d_fit  else "n/a"
        d3d_str  = f"{d3d_fit['median']:.0f}/mm³"  if d3d_fit  else "n/a"
        sf_str   = f"{septa_frac_mean*100:.0f}%"   if septa_frac_mean is not None else "n/a"
        cr_str   = f"{clust_r:.0f}µm"              if clust_r  else "n/a"
        nn_self  = nn_priors.get(st, {}).get("median_um", float("nan"))
        print(f"  {label:20s}: 2D={d2d_str:12s} 3D={d3d_str:12s} "
              f"septa={sf_str:5s} clust={cr_str:7s} nn_self={nn_self:.0f}µm" if not np.isnan(nn_self)
              else f"  {label:20s}: 2D={d2d_str:12s} 3D={d3d_str:12s} septa={sf_str:5s} clust={cr_str}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_json = BASE / "abm_priors.json"
    with open(out_json, "w") as f:
        json.dump(priors, f, indent=2, default=lambda x: None if np.isnan(x) else x)
    print(f"\nSaved → {out_json}")

    # Summary CSV
    rows = []
    for st, p in priors.items():
        rows.append({
            "subtype":              st,
            "abm_agent":            p["abm_agent"],
            "density_2d_median":    p["density_2d"]["median"]  if p["density_2d"]  else None,
            "density_2d_sigma_ln":  p["density_2d"]["sigma"]   if p["density_2d"]  else None,
            "density_3d_median":    p["density_3d"]["median"]  if p["density_3d"]  else None,
            "density_3d_sigma_ln":  p["density_3d"]["sigma"]   if p["density_3d"]  else None,
            "cells_per_domain":     p["cells_per_abm_domain_mean"],
            "septa_fraction_mean":  p["septa_fraction"]["mean"],
            "septa_fraction_std":   p["septa_fraction"]["std"],
            "clustering_radius_um": p["clustering_radius_um"],
            "nn_self_median_um":    p["nn_distances_um"].get(st, {}).get("median_um"),
            "n_rois":               p["density_2d"]["n_rois"] if p["density_2d"] else 0,
        })
    pd.DataFrame(rows).to_csv(BASE / "abm_priors_summary.csv", index=False)
    print(f"Saved → {BASE / 'abm_priors_summary.csv'}")

    # Total cell budget
    total = sum(p["cells_per_abm_domain_mean"] for p in priors.values())
    print(f"\nTotal ABM domain cell estimate: {total:.0f} cells in 1mm³")


if __name__ == "__main__":
    main()
