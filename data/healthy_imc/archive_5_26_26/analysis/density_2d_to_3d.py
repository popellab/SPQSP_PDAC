"""
density_2d_to_3d.py — Convert IMC 2D densities to 3D ABM initialization parameters.

Reads density_summary.csv and ripley_L.csv from spatial_analysis.py.
Applies Abercrombie stereological correction:

    N_3D (cells/mm³) = N_2D (cells/mm²) / (t + d)

where t = section thickness (µm), d = mean cell diameter (µm).

Produces abm_params.json with per-cell-type 3D densities and ABM voxel counts
for the SPQSP_PDAC domain (50³ voxels × 20µm = 1mm³, 125,000 voxels).

Usage:
    python density_2d_to_3d.py [--section-thickness UM] [--spatial-dir DIR]

Output:
    analysis/abm_params.json
    analysis/abm_params_summary.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── ABM domain constants ───────────────────────────────────────────────────────
ABM_VOXEL_SIZE_UM = 20.0      # µm per voxel edge
ABM_GRID_SIDE = 50            # voxels per side
ABM_DOMAIN_MM3 = (ABM_VOXEL_SIZE_UM * ABM_GRID_SIDE / 1000) ** 3  # = 1.0 mm³
ABM_N_VOXELS = ABM_GRID_SIDE ** 3  # = 125,000

# ── Cell diameter estimates (µm) — literature values ──────────────────────────
# Used for Abercrombie correction: d = mean nuclear/cell diameter
# Sources: lymphocytes ~8µm, macrophages ~15µm, fibroblasts ~15µm, endothelial ~10µm
CELL_DIAMETERS_UM = {
    "cd4_t":        8.0,
    "cd8_t":        8.0,
    "b_cell":       8.0,
    "macrophage":  15.0,
    "mdsc":        10.0,
    "other_immune": 9.0,
    "fibroblast":  15.0,
    "endothelial": 10.0,
}

# ABM agent type mapping
ABM_AGENT = {
    "cd4_t":        "TReg",
    "cd8_t":        "TCell",
    "b_cell":       "B_cell (future)",
    "macrophage":   "Macrophage",
    "mdsc":         "MDSC",
    "other_immune": "(unassigned)",
    "fibroblast":   "Fibroblast",
    "endothelial":  "VascularCell",
}


def abercrombie(density_2d: float, section_thickness_um: float,
                cell_diameter_um: float) -> float:
    """
    Abercrombie correction: N_3D = N_2D / (t + d) * 1000
    where density_2d is in cells/mm², result in cells/mm³.
    t and d must be in µm; factor 1000 converts µm to mm denominator.
    """
    denom_mm = (section_thickness_um + cell_diameter_um) / 1000.0
    return density_2d / denom_mm if denom_mm > 0 else 0.0


def ripley_correlation_length(ripley_df: pd.DataFrame, subtype: str) -> float | None:
    """
    Return the radius at which L(r)-r peaks for the given subtype.
    This is the characteristic clustering scale.
    Returns None if insufficient data.
    """
    sub = ripley_df[ripley_df["subtype"] == subtype]
    if sub.empty:
        return None
    mean_L = sub.groupby("r_um")["L_minus_r"].mean()
    if mean_L.max() <= 0:
        return None
    return float(mean_L.idxmax())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--section-thickness", type=float, default=5.0,
                        help="FFPE section thickness in µm (default: 5.0)")
    parser.add_argument("--spatial-dir", default=None,
                        help="Path to spatial/ directory from spatial_analysis.py")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    spatial_dir = Path(args.spatial_dir) if args.spatial_dir else script_dir / "spatial"

    density_path = spatial_dir / "density_summary.csv"
    ripley_path = spatial_dir / "ripley_L.csv"

    if not density_path.exists():
        print(f"ERROR: {density_path} not found. Run spatial_analysis.py first.")
        return

    density_df = pd.read_csv(density_path)
    ripley_df = pd.read_csv(ripley_path) if ripley_path.exists() else pd.DataFrame()

    t = args.section_thickness
    print(f"Section thickness: {t} µm")
    print(f"ABM domain: {ABM_GRID_SIDE}³ × {ABM_VOXEL_SIZE_UM}µm = {ABM_DOMAIN_MM3:.1f} mm³, {ABM_N_VOXELS:,} voxels\n")

    params = {}
    rows = []

    subtypes = [c.replace("density_", "") for c in density_df.columns
                if c.startswith("density_")]

    for stype in subtypes:
        col = f"density_{stype}"
        vals_2d = density_df[col].dropna().values
        if len(vals_2d) == 0:
            continue

        d = CELL_DIAMETERS_UM.get(stype, 10.0)
        density_3d_vals = [abercrombie(v, t, d) for v in vals_2d]

        mean_2d = float(np.mean(vals_2d))
        std_2d  = float(np.std(vals_2d))
        mean_3d = float(np.mean(density_3d_vals))
        std_3d  = float(np.std(density_3d_vals))

        cells_per_domain = mean_3d * ABM_DOMAIN_MM3
        cells_per_voxel = mean_3d / ABM_N_VOXELS * ABM_DOMAIN_MM3

        corr_length = ripley_correlation_length(ripley_df, stype) if not ripley_df.empty else None

        params[stype] = {
            "abm_agent":                ABM_AGENT.get(stype, "unknown"),
            "cell_diameter_um":         d,
            "density_2d_mean":          round(mean_2d, 1),
            "density_2d_std":           round(std_2d, 1),
            "density_3d_mean":          round(mean_3d, 1),
            "density_3d_std":           round(std_3d, 1),
            "cells_per_abm_domain":     round(cells_per_domain, 1),
            "cells_per_voxel_mean":     round(cells_per_voxel, 4),
            "spatial_correlation_um":   corr_length,
            "n_rois":                   len(vals_2d),
        }

        rows.append({
            "subtype": stype,
            "abm_agent": ABM_AGENT.get(stype, "unknown"),
            "density_2d_mean": round(mean_2d, 1),
            "density_2d_std": round(std_2d, 1),
            "density_3d_mean": round(mean_3d, 1),
            "density_3d_std": round(std_3d, 1),
            "cells_per_abm_domain": round(cells_per_domain, 1),
            "spatial_correlation_um": corr_length,
            "n_rois": len(vals_2d),
        })

        corr_str = f"{corr_length:.0f}µm" if corr_length else "n/a"
        print(f"{stype:20s} ({ABM_AGENT.get(stype,'?'):20s}): "
              f"2D={mean_2d:.0f}±{std_2d:.0f}/mm²  "
              f"3D={mean_3d:.0f}±{std_3d:.0f}/mm³  "
              f"ABM={cells_per_domain:.0f} cells  "
              f"corr={corr_str}")

    # Save outputs
    out_json = script_dir / "abm_params.json"
    with open(out_json, "w") as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved → {out_json}")

    out_csv = script_dir / "abm_params_summary.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    # Print total cell budget
    total = sum(p["cells_per_abm_domain"] for p in params.values())
    print(f"\nTotal estimated cells in 1mm³ ABM domain: {total:.0f}")
    print(f"(at {ABM_N_VOXELS:,} voxels, avg occupancy = {total/ABM_N_VOXELS:.4f} cells/voxel)")


if __name__ == "__main__":
    main()
