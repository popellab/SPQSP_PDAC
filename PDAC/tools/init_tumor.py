"""Build a 2D tumor-microenvironment placement file for the PDAC ABM.

Three-phase placement:
  1. Cancer cells in a compact disc (radius from cell count)
  2. Immune + activated stroma via SA to match per-patient IMC phi targets
  3. Background quiescent fibroblasts fill remaining voxels + Vvas/ECM fields

Two count modes:
  --qsp-csv   Cell counts from QSP V_T species (primary). Grid size and
              n_cancer auto-computed from QSP tumor volume (spherical
              cross-section). All species scaled by cancer_2d/cancer_3d.
  --sample    Cell counts from IMC composition (fallback, for testing
              without QSP; requires --n-cancer, --grid-x, --grid-y).

Phi targets always come from --sample (IMC spatial organization).

Usage::

    # Production: QSP provides counts + geometry, IMC provides phi targets
    python -m PDAC.tools.init_tumor \\
        --qsp-csv run/qsp_presim.csv --qsp-step -1 \\
        --sample 100-DFC-003 \\
        --seed 42 --n-steps 5000000 \\
        --output tumor.plc

    # With explicit overrides (grid/n_cancer override auto-computed values)
    python -m PDAC.tools.init_tumor \\
        --qsp-csv run/qsp_presim.csv \\
        --sample 100-DFC-003 --n-cancer 500 \\
        --grid-x 707 --grid-y 707 \\
        --output tumor.plc

    # Testing: IMC provides both counts and spatial organization
    python -m PDAC.tools.init_tumor \\
        --sample 100-DFC-003 --n-cancer 500 \\
        --grid-x 707 --grid-y 707 \\
        --output tumor.plc
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# Categories in the SA optimization (order matters for matrix indexing)
SA_CATEGORIES: list[str] = [
    "cancer_prog", "cancer_sen",
    "tcell_eff",
    "mac_M1", "mac_M2",
    "treg_th", "treg_treg",
    "mdsc",
    "fib_myCAF", "fib_iCAF", "fib_apCAF",
]

CANCER_CATS = frozenset({"cancer_prog", "cancer_sen"})

# Categories that exist in the ABM but are not optimized by SA
BACKGROUND_CAT = "fib_quiescent"
FIELD_CATS = {"vas_phalanx"}

# IMC categories with no ABM agent — skip in count computation
NO_AGENT_CATS = {"bcell_naive", "bcell_act", "dc_immature_any", "dc_mature_any"}

# QSP V_T species → SA category. Only tumor-infiltrating cell species.
# qPSC is EXCLUDED: it represents activation dynamics in the QSP, not the
# resident fibroblast pool. Background fib_quiescent comes from healthy tissue
# placement (init_2d_healthy), same as vasculature → Vvas field.
QSP_CELL_SPECIES: dict[str, str] = {
    "C1": "cancer_prog",
    "C2": "cancer_prog",
    "C_x": "cancer_sen",
    "CD8": "tcell_eff",
    "CD8_exh": "tcell_eff",
    "Th": "treg_th",
    "Th_exh": "treg_th",
    "Treg": "treg_treg",
    "Tfh": "treg_th",
    "MDSC": "mdsc",
    "Mac_M1": "mac_M1",
    "Mac_M2": "mac_M2",
    "iCAF": "fib_iCAF",
    "myCAF": "fib_myCAF",
    "apCAF": "fib_apCAF",
}

# QSP species that require sub-state splitting (the panel can't resolve these,
# but the QSP tracks them separately). Used for placement_io substate_frac.
QSP_SUBSTATE: dict[str, list[tuple[str, int]]] = {
    "tcell_eff": [("CD8", 0), ("CD8_exh", 2)],
}


def counts_from_qsp(
    qsp_csv: str,
    n_disc: int,
    step: int = -1,
    min_cells: int = 5,
) -> tuple[dict[str, int], dict[str, dict[int, float]], dict[str, float]]:
    """Derive per-category cell counts from QSP V_T species.

    The 2D cross-section has *n_disc* total cells (cancer + immune +
    activated stroma).  Each category gets its 3D fraction of that disc,
    preserving QSP composition exactly.

    Parameters
    ----------
    qsp_csv : path to QSP output CSV (columns: step, V_T.C1, V_T.CD8, ...)
    n_disc : total cells in the 2D cross-section (from grid_from_qsp)
    step : QSP step to read (-1 = last row)
    min_cells : minimum cells per non-cancer category (0 → category dropped)

    Returns
    -------
    counts : {category: n_cells} for SA_CATEGORIES
    substate_frac : {category: {state: fraction}} for placement_io
    qsp_raw : {species: raw_3d_count} for diagnostics
    """
    df = pd.read_csv(qsp_csv)
    row = df.iloc[step]

    qsp_raw: dict[str, float] = {}
    cat_totals_3d: dict[str, float] = {}
    for sp, cat in QSP_CELL_SPECIES.items():
        col = f"V_T.{sp}"
        val = float(row.get(col, 0.0))
        qsp_raw[sp] = val
        cat_totals_3d[cat] = cat_totals_3d.get(cat, 0.0) + val

    n_total_3d = sum(qsp_raw.values())
    if n_total_3d < 1:
        raise ValueError(f"QSP has no tumor cells at step {step}")

    scale = n_disc / n_total_3d
    counts: dict[str, int] = {}
    for cat in SA_CATEGORIES:
        raw = cat_totals_3d.get(cat, 0.0)
        n = max(0, int(round(raw * scale)))
        if cat not in CANCER_CATS and raw > 0:
            n = max(min_cells, n)
        counts[cat] = n

    # Compute sub-state fractions from QSP species
    substate_frac: dict[str, dict[int, float]] = {}
    for cat, splits in QSP_SUBSTATE.items():
        total = sum(qsp_raw.get(sp, 0.0) for sp, _ in splits)
        if total > 0:
            substate_frac[cat] = {
                state: qsp_raw.get(sp, 0.0) / total
                for sp, state in splits
            }

    return counts, substate_frac, qsp_raw


def load_onset_json(path: str) -> dict[str, float]:
    """Read a ``qsp_onset/1`` JSON and return species cell counts."""
    with open(path) as f:
        data = json.load(f)
    if data.get("schema") != "qsp_onset/1":
        raise ValueError(f"Unknown onset schema: {data.get('schema')}")
    return {
        "cells": {k: float(v) for k, v in data["cells"].items()},
        "V_T_m3": float(data["V_T_m3"]),
        "t_warmup_s": float(data["t_warmup_s"]),
    }


def grid_from_onset(
    onset: dict,
    margin_vox: int = 50,
    volume_fraction: float = 1.0,
    voxel_um: float = 20.0,
    max_grid: int | None = None,
) -> dict:
    """Compute grid + counts from QSP onset JSON.

    The onset already represents the placed state (e.g. 0.95× target).
    *volume_fraction* can further scale the placed disc (default 1.0 =
    place exactly what the onset says).  Grid is sized for the FULL disc
    (1.0×) so the ABM has room to grow.

    When *max_grid* is set and the natural grid exceeds it, the tumor is
    scaled down into the smaller domain: the effective radius shrinks so
    that ``2*r_eff + 2*margin == max_grid``, and all cell counts scale
    by the area ratio ``(r_eff / r_3d)²``.  Voxel size stays unchanged,
    preserving spatial metrics and PDE kinetics.
    """
    cells = onset["cells"]

    cat_totals_3d: dict[str, float] = {}
    species_3d: dict[str, float] = {}
    for sp, cat in QSP_CELL_SPECIES.items():
        val = cells.get(sp, 0.0)
        species_3d[sp] = val
        cat_totals_3d[cat] = cat_totals_3d.get(cat, 0.0) + val

    n_total_3d = sum(species_3d.values())
    cancer_3d = sum(cat_totals_3d.get(c, 0.0) for c in CANCER_CATS)
    if n_total_3d < 1:
        raise ValueError("Onset has no mapped tumor cells")

    r_3d = (3.0 * n_total_3d / (4.0 * np.pi)) ** (1.0 / 3.0)

    # Natural (full-size) grid
    grid_side_natural = int(np.ceil(2.0 * r_3d)) + 2 * margin_vox
    if grid_side_natural % 2 == 0:
        grid_side_natural += 1

    # Domain scaling: shrink tumor into smaller grid if needed
    domain_scale = 1.0
    r_eff = r_3d
    if max_grid is not None and grid_side_natural > max_grid:
        grid_side = max_grid if max_grid % 2 == 1 else max_grid - 1
        r_eff = (grid_side - 2 * margin_vox) / 2.0
        domain_scale = (r_eff / r_3d) ** 2  # area ratio
    else:
        grid_side = grid_side_natural

    n_disc_full = int(round(np.pi * r_eff ** 2))
    n_disc = max(1, int(round(volume_fraction * n_disc_full)))

    # Per-category 2D counts (scaled by area ratio)
    scale = n_disc / n_total_3d
    cat_counts_2d: dict[str, int] = {}
    for cat in SA_CATEGORIES:
        cat_counts_2d[cat] = max(0, int(round(cat_totals_3d.get(cat, 0.0) * scale)))
    n_cancer_2d = sum(cat_counts_2d.get(c, 0) for c in CANCER_CATS)

    # Sub-state fractions (ratios preserved regardless of scale)
    substate_frac: dict[str, dict[int, float]] = {}
    for cat, splits in QSP_SUBSTATE.items():
        total = sum(species_3d.get(sp, 0.0) for sp, _ in splits)
        if total > 0:
            substate_frac[cat] = {
                state: species_3d.get(sp, 0.0) / total
                for sp, state in splits
            }

    return {
        "grid_x": grid_side,
        "grid_y": grid_side,
        "n_disc": n_disc,
        "n_disc_full": n_disc_full,
        "n_cancer_2d": n_cancer_2d,
        "n_total_3d": n_total_3d,
        "cancer_3d": cancer_3d,
        "cancer_fraction": cancer_3d / n_total_3d,
        "cat_counts_2d": cat_counts_2d,
        "cat_totals_3d": cat_totals_3d,
        "species_3d": species_3d,
        "substate_frac": substate_frac,
        "r_3d_vox": r_3d,
        "r_eff_vox": r_eff,
        "domain_scale": domain_scale,
        "volume_fraction": volume_fraction,
        "margin_vox": margin_vox,
        "domain_mm": grid_side * voxel_um / 1000.0,
        "tumor_diam_mm": 2.0 * r_eff * voxel_um / 1000.0,
        "grid_natural": grid_side_natural,
        "V_T_m3": onset["V_T_m3"],
        "t_warmup_s": onset["t_warmup_s"],
    }


def counts_from_onset(
    onset: dict,
    n_disc: int,
    min_cells: int = 5,
) -> tuple[dict[str, int], dict[str, dict[int, float]], dict[str, float]]:
    """Derive per-category counts from onset JSON, same interface as counts_from_qsp."""
    cells = onset["cells"]

    qsp_raw: dict[str, float] = {}
    cat_totals_3d: dict[str, float] = {}
    for sp, cat in QSP_CELL_SPECIES.items():
        val = cells.get(sp, 0.0)
        qsp_raw[sp] = val
        cat_totals_3d[cat] = cat_totals_3d.get(cat, 0.0) + val

    n_total_3d = sum(qsp_raw.values())
    if n_total_3d < 1:
        raise ValueError("Onset has no mapped tumor cells")

    scale = n_disc / n_total_3d
    counts: dict[str, int] = {}
    for cat in SA_CATEGORIES:
        raw = cat_totals_3d.get(cat, 0.0)
        n = max(0, int(round(raw * scale)))
        if cat not in CANCER_CATS and raw > 0:
            n = max(min_cells, n)
        counts[cat] = n

    substate_frac: dict[str, dict[int, float]] = {}
    for cat, splits in QSP_SUBSTATE.items():
        total = sum(qsp_raw.get(sp, 0.0) for sp, _ in splits)
        if total > 0:
            substate_frac[cat] = {
                state: qsp_raw.get(sp, 0.0) / total
                for sp, state in splits
            }

    return counts, substate_frac, qsp_raw


def grid_from_qsp(
    qsp_csv: str,
    step: int = -1,
    margin_vox: int = 50,
    volume_fraction: float = 0.95,
    voxel_um: float = 20.0,
) -> dict:
    """Compute grid size from QSP tumor volume (all cell types).

    The tumor is a mixed sphere of cancer + immune + activated stroma.
    A midplane cross-section through that sphere gives the 2D disc,
    and each cell type's 2D count is its 3D fraction of that disc.

    Grid sized for full (1.0x) target volume + *margin_vox*.
    All placed counts scaled by *volume_fraction* (< 1 = growth room).

    Returns
    -------
    dict with grid_x, grid_y, n_disc, per-category 2D counts, etc.
    """
    df = pd.read_csv(qsp_csv)
    row = df.iloc[step]

    # Total 3D cellularity = all mapped V_T species
    species_3d: dict[str, float] = {}
    cat_totals_3d: dict[str, float] = {}
    for sp, cat in QSP_CELL_SPECIES.items():
        val = float(row.get(f"V_T.{sp}", 0.0))
        species_3d[sp] = val
        cat_totals_3d[cat] = cat_totals_3d.get(cat, 0.0) + val

    n_total_3d = sum(species_3d.values())
    cancer_3d = sum(cat_totals_3d.get(c, 0.0) for c in CANCER_CATS)
    if n_total_3d < 1:
        raise ValueError(f"QSP has no tumor cells at step {step}")
    if cancer_3d < 1:
        raise ValueError(f"QSP has no cancer cells at step {step}")

    # Sphere from TOTAL cellularity → midplane disc
    r_3d = (3.0 * n_total_3d / (4.0 * np.pi)) ** (1.0 / 3.0)
    n_disc_full = int(round(np.pi * r_3d ** 2))
    n_disc = max(1, int(round(volume_fraction * n_disc_full)))

    # Grid: full-target diameter + margin
    grid_side = int(np.ceil(2.0 * r_3d)) + 2 * margin_vox
    if grid_side % 2 == 0:
        grid_side += 1

    # Per-category 2D counts (preserves QSP composition exactly)
    scale = n_disc / n_total_3d
    cat_counts_2d: dict[str, int] = {}
    for cat in SA_CATEGORIES:
        cat_counts_2d[cat] = max(0, int(round(cat_totals_3d.get(cat, 0.0) * scale)))
    n_cancer_2d = sum(cat_counts_2d.get(c, 0) for c in CANCER_CATS)

    domain_mm = grid_side * voxel_um / 1000.0
    tumor_diam_mm = 2.0 * r_3d * voxel_um / 1000.0
    cancer_frac = cancer_3d / n_total_3d

    return {
        "grid_x": grid_side,
        "grid_y": grid_side,
        "n_disc": n_disc,
        "n_disc_full": n_disc_full,
        "n_cancer_2d": n_cancer_2d,
        "n_total_3d": n_total_3d,
        "cancer_3d": cancer_3d,
        "cancer_fraction": cancer_frac,
        "cat_counts_2d": cat_counts_2d,
        "cat_totals_3d": cat_totals_3d,
        "species_3d": species_3d,
        "r_3d_vox": r_3d,
        "volume_fraction": volume_fraction,
        "margin_vox": margin_vox,
        "domain_mm": domain_mm,
        "tumor_diam_mm": tumor_diam_mm,
    }


def load_patient_phi(
    sample: str,
    data_dir: Path,
    norm: str = "placed_only",
) -> tuple[np.ndarray, dict[str, int]]:
    """Load per-patient phi matrix from funcn_target_summary.csv.

    Returns (phi_matrix, imc_counts) indexed by SA_CATEGORIES.
    """
    summary_csv = data_dir / "funcn_target_summary.csv"
    df = pd.read_csv(summary_csv)
    df = df[(df["sample"] == sample) & (df["norm"] == norm)]
    if df.empty:
        available = sorted(pd.read_csv(summary_csv)["sample"].unique())
        raise ValueError(
            f"Sample '{sample}' not found with norm='{norm}'. "
            f"Available: {available[:5]}..."
        )

    n = len(SA_CATEGORIES)
    phi = np.full((n, n), np.nan, dtype=np.float64)
    imc_counts: dict[str, int] = {}

    for _, row in df.iterrows():
        tgt, src = row["target"], row["source"]
        if tgt in SA_CATEGORIES and src in SA_CATEGORIES:
            ti = SA_CATEGORIES.index(tgt)
            si = SA_CATEGORIES.index(src)
            phi[ti, si] = row["phi"]
        if tgt in SA_CATEGORIES and tgt not in imc_counts:
            imc_counts[tgt] = int(row["n_target_cells"])

    return phi, imc_counts


def load_patient_composition(
    sample: str,
    data_dir: Path,
) -> dict[str, float]:
    """Load per-patient composition ratios from composition_per_sample.csv."""
    comp_csv = data_dir / "composition_per_sample.csv"
    df = pd.read_csv(comp_csv)
    row = df[df["sample"] == sample]
    if row.empty:
        raise ValueError(f"Sample '{sample}' not in composition_per_sample.csv")
    row = row.iloc[0]
    ratios = {}
    for col in df.columns:
        if col.startswith("ratio__"):
            cat = col[len("ratio__"):]
            ratios[cat] = float(row[col])
    return ratios


def compute_cell_counts(
    n_cancer: int,
    ratios: dict[str, float],
    min_cells: int = 5,
) -> dict[str, int]:
    """Derive per-category cell counts from cancer count + IMC ratios.

    Cancer categories are filled first (proportional to their ratio within
    cancer). Non-cancer categories are scaled so the total matches the IMC
    composition.
    """
    cancer_ratio = sum(ratios.get(c, 0) for c in CANCER_CATS)
    if cancer_ratio < 1e-6:
        raise ValueError("No cancer in composition ratios")

    total_placed = n_cancer / cancer_ratio
    counts: dict[str, int] = {}

    for cat in SA_CATEGORIES:
        r = ratios.get(cat, 0.0)
        if cat in CANCER_CATS:
            counts[cat] = max(1, int(round(n_cancer * r / cancer_ratio)))
        else:
            counts[cat] = max(0, int(round(total_placed * r)))

    actual_cancer = sum(counts[c] for c in CANCER_CATS)
    if actual_cancer != n_cancer:
        counts["cancer_prog"] += n_cancer - actual_cancer

    for cat in SA_CATEGORIES:
        r = ratios.get(cat, 0.0)
        if cat not in CANCER_CATS and r > 1e-4 and counts[cat] > 0:
            counts[cat] = max(min_cells, counts[cat])

    return counts


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="2D tumor microenvironment SA placement for PDAC ABM"
    )
    p.add_argument("--sample", type=str, required=True,
                   help="PASS01 sample ID for phi targets (e.g. 100-DFC-003)")
    p.add_argument("--n-cancer", type=int, default=None,
                   help="Cancer cells in 2D slice (for IMC-only mode).")
    src = p.add_mutually_exclusive_group()
    src.add_argument("--onset-json", type=str, default=None,
                     help="QSP onset JSON from --dump-qsp-onset (preferred).")
    src.add_argument("--qsp-csv", type=str, default=None,
                     help="QSP output CSV for cell counts (V_T species).")
    p.add_argument("--qsp-step", type=int, default=-1,
                   help="QSP CSV step to read (-1 = last row, default: -1)")
    p.add_argument("--grid-x", type=int, default=None,
                   help="Grid width. Auto-computed from QSP if omitted.")
    p.add_argument("--grid-y", type=int, default=None,
                   help="Grid height. Auto-computed from QSP if omitted.")
    p.add_argument("--margin", type=int, default=50,
                   help="Voxels of healthy tissue border around tumor "
                        "(for auto-grid, default 50 = 1 mm)")
    p.add_argument("--max-grid", type=int, default=None,
                   help="Cap grid side at this size and scale the tumor "
                        "down proportionally (voxel size unchanged)")
    p.add_argument("--volume-fraction", type=float, default=0.95,
                   help="Fraction of QSP target volume to place (default "
                        "0.95 = 5%% growth room for ABM)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=5_000_000)
    p.add_argument("--log-interval", type=int, default=100_000)
    p.add_argument("--sigma-vox", type=float, default=2.5,
                   help="FunCN kernel sigma in voxels (default 2.5 = 50 µm)")
    p.add_argument("--voxel-um", type=float, default=20.0)
    p.add_argument("--output", type=str, required=True,
                   help="Output .plc file path")
    p.add_argument("--healthy-plc", type=str, default=None,
                   help="Existing healthy tissue .plc (for Vvas field). "
                        "If omitted, background fib fills uniformly.")
    p.add_argument("--fib-density", type=float, default=784.0,
                   help="Background fibroblast density [cells/mm²] (default 784)")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Path to data/PASS01/derived/stage5/")
    p.add_argument("--norm", type=str, default="placed_only",
                   choices=["placed_only", "with_background"])
    p.add_argument("--tumor-margin", type=int, default=30,
                   help="Voxels beyond cancer disc for immune placement (default 30)")
    args = p.parse_args(argv)

    voxel_um = args.voxel_um

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = (Path(__file__).resolve().parents[2]
                    / "data" / "PASS01" / "derived" / "stage5")

    # ── Auto-compute grid & cell counts from QSP onset or CSV ──
    geo = None
    qsp_source = args.onset_json or args.qsp_csv

    if args.onset_json:
        onset = load_onset_json(args.onset_json)
        geo = grid_from_onset(
            onset,
            margin_vox=args.margin,
            volume_fraction=args.volume_fraction,
            voxel_um=voxel_um,
            max_grid=args.max_grid,
        )
    elif args.qsp_csv:
        geo = grid_from_qsp(
            args.qsp_csv,
            step=args.qsp_step,
            margin_vox=args.margin,
            volume_fraction=args.volume_fraction,
            voxel_um=voxel_um,
        )

    if geo is not None:
        if args.grid_x is None:
            args.grid_x = geo["grid_x"]
        if args.grid_y is None:
            args.grid_y = geo["grid_y"]

        src_label = args.onset_json or f"QSP CSV step {args.qsp_step}"
        print(f"Tumor geometry ({src_label}):")
        print(f"  Total cells (3D):      {geo['n_total_3d']:,.0f}")
        print(f"  Cancer cells (3D):     {geo['cancer_3d']:,.0f}  "
              f"({geo['cancer_fraction']:.1%} of tumor)")
        print(f"  Sphere radius:         {geo['r_3d_vox']:.1f} vox "
              f"({geo['r_3d_vox'] * voxel_um / 1000:.2f} mm)")
        if geo.get("domain_scale", 1.0) < 1.0:
            print(f"  Domain scaling:        {geo['domain_scale']:.4f}x area "
                  f"(natural grid {geo['grid_natural']}, capped at {geo['grid_x']})")
            print(f"  Effective radius:      {geo['r_eff_vox']:.1f} vox "
                  f"({geo['r_eff_vox'] * voxel_um / 1000:.2f} mm)")
        print(f"  Midplane disc (1.0x):  {geo['n_disc_full']:,d} cells")
        print(f"  Placed disc ({geo['volume_fraction']:.0%}):  "
              f"{geo['n_disc']:,d} cells  "
              f"(cancer={geo['n_cancer_2d']:,d})")
        print(f"  Auto grid:             {args.grid_x} x {args.grid_y}  "
              f"({geo['domain_mm']:.2f} mm, margin={args.margin} vox)")
        if "V_T_m3" in geo:
            print(f"  QSP V_T:               {geo['V_T_m3']:.3e} m³  "
                  f"(warmup {geo['t_warmup_s']/86400:.0f} days)")
    else:
        if args.grid_x is None or args.grid_y is None or args.n_cancer is None:
            p.error("--grid-x, --grid-y, --n-cancer are required "
                    "when --onset-json / --qsp-csv is not provided")

    gx, gy = args.grid_x, args.grid_y

    # ── Load patient phi targets (IMC spatial organization) ──
    print(f"\nLoading IMC phi targets for {args.sample}...")
    phi_matrix, imc_counts = load_patient_phi(args.sample, data_dir, args.norm)

    # ── Compute cell counts ──
    substate_frac: dict[str, dict[int, float]] = {}
    if args.onset_json:
        n_disc = geo["n_disc"]
        print(f"\nDeriving cell counts from onset ({n_disc:,d} cells in disc)")
        counts, substate_frac, qsp_raw = counts_from_onset(
            onset, n_disc
        )
    elif args.qsp_csv:
        n_disc = geo["n_disc"]
        print(f"\nDeriving cell counts from QSP cross-section ({n_disc:,d} cells in disc)")
        counts, substate_frac, qsp_raw = counts_from_qsp(
            args.qsp_csv, n_disc, step=args.qsp_step
        )
    else:
        print(f"\nDeriving cell counts from IMC composition (no QSP source)")
        ratios = load_patient_composition(args.sample, data_dir)
        counts = compute_cell_counts(args.n_cancer, ratios)
        qsp_raw = {}

    if qsp_source and qsp_raw:
        n_total_3d = sum(qsp_raw.values())
        scale = geo["n_disc"] / n_total_3d if n_total_3d > 0 else 0
        print(f"\n  V_T species → 2D cross-section:")
        for sp in sorted(QSP_CELL_SPECIES):
            cat = QSP_CELL_SPECIES[sp]
            raw = qsp_raw.get(sp, 0)
            print(f"    {sp:12s} → {cat:15s}  ({raw:>12.0f} 3D → {raw*scale:>8.1f} 2D)")
        if substate_frac:
            print(f"\n  Sub-state splits:")
            for cat, fracs in substate_frac.items():
                print(f"    {cat}: {fracs}")

    n_total_sa = sum(counts.values())
    n_cancer = sum(counts[c] for c in CANCER_CATS)
    n_immune = n_total_sa - n_cancer

    print(f"\nPlacement cell counts:")
    for cat in SA_CATEGORIES:
        print(f"  {cat:20s}  {counts[cat]:>6d}")
    print(f"  {'TOTAL':20s}  {n_total_sa:>6d}  (cancer={n_cancer}, immune/stroma={n_immune})")

    # ── Phase 1: Place cancer disc ──
    print(f"\n── Phase 1: Cancer disc ──")
    from .seeding_tumor import place_cancer_disc, phi_to_frac, sa_place_tumor

    rng = np.random.default_rng(args.seed)
    cx, cy = gx // 2, gy // 2
    cancer_pos = place_cancer_disc(n_cancer, (cx, cy), gx, gy, rng)

    # Split into cancer_prog and cancer_sen based on counts
    n_prog = counts["cancer_prog"]
    n_sen = counts["cancer_sen"]
    perm = rng.permutation(len(cancer_pos))
    prog_idx = perm[:n_prog]
    sen_idx = perm[n_prog:n_prog + n_sen]

    fixed_positions: dict[str, np.ndarray] = {}
    if n_prog > 0:
        fixed_positions["cancer_prog"] = cancer_pos[prog_idx]
    if n_sen > 0:
        fixed_positions["cancer_sen"] = cancer_pos[sen_idx]

    disc_radius = np.sqrt(n_cancer / np.pi)
    print(f"  Cancer disc at ({cx}, {cy}), radius={disc_radius:.1f} voxels")
    print(f"  Placed {n_prog} prog + {n_sen} sen = {len(cancer_pos)} cancer cells")

    # ── Phase 2: Phi targets ──
    print(f"\n── Phase 2: Phi targets (direct, no frac conversion) ──")
    n_valid = int(np.sum(~np.isnan(phi_matrix)))
    n_total_pairs = len(SA_CATEGORIES) ** 2
    print(f"  Valid phi pairs: {n_valid}/{n_total_pairs}")

    key_pairs = [
        ("tcell_eff", "cancer_prog"), ("mac_M1", "cancer_prog"),
        ("mac_M2", "cancer_prog"), ("fib_myCAF", "cancer_prog"),
    ]
    print(f"\n  Key phi targets:")
    for tgt, src in key_pairs:
        if tgt in SA_CATEGORIES and src in SA_CATEGORIES:
            ti = SA_CATEGORIES.index(tgt)
            si = SA_CATEGORIES.index(src)
            phi_val = phi_matrix[ti, si]
            direction = "cluster" if phi_val > 1 else "avoid" if phi_val < 1 else "random"
            print(f"    {tgt:15s} × {src:15s}:  phi={phi_val:.3f}  ({direction})")

    # Also compute frac target for diagnostics (saved in npz)
    count_arr = np.array([counts.get(c, 0) for c in SA_CATEGORIES], dtype=np.float64)
    frac_target = phi_to_frac(phi_matrix, count_arr)

    # ── Phase 3: SA placement (phi-based energy) ──
    print(f"\n── Phase 3: Simulated annealing ({args.n_steps:,d} steps, phi energy) ──")
    pos_by_cat, sa_info = sa_place_tumor(
        counts=counts,
        shape=(gx, gy),
        target_frac=phi_matrix,
        categories=SA_CATEGORIES,
        fixed_cats=CANCER_CATS,
        fixed_positions=fixed_positions,
        sigma_vox=args.sigma_vox,
        n_steps=args.n_steps,
        seed=args.seed,
        log_interval=args.log_interval,
        target_phi=phi_matrix,
    )
    print(f"\nSA finished in {sa_info['elapsed_s']:.0f}s")
    print(f"  Final energy: {sa_info['energy_final']:.2e}")
    print(f"  Acceptance rate: {sa_info['acceptance_rate']:.3f}")
    print(f"  Movable cells: {sa_info['n_movable']}")

    # ── Phase 4: Validate with funcn_phi ──
    print(f"\n── Phase 4: Validation ──")
    from .funcn_core import funcn_phi as compute_phi

    pos_3d = {}
    for cat in SA_CATEGORIES:
        p = pos_by_cat.get(cat, np.empty((0, 2), np.int32))
        if len(p) > 0:
            pos_3d[cat] = np.column_stack([p, np.zeros(len(p), dtype=np.int32)])
        else:
            pos_3d[cat] = np.empty((0, 3), np.int64)

    cats_present = [c for c in SA_CATEGORIES if len(pos_3d.get(c, [])) > 0]
    if len(cats_present) >= 2:
        val_phi = compute_phi(pos_3d, cats_present, (gx, gy, 1),
                              sigma_vox=args.sigma_vox, min_targets=3)
        print(f"\n  Achieved phi (vs IMC target):")
        for tgt, src in key_pairs:
            if tgt in cats_present and src in cats_present:
                ti = cats_present.index(tgt)
                si = cats_present.index(src)
                achieved = val_phi[ti, si]
                orig_ti = SA_CATEGORIES.index(tgt)
                orig_si = SA_CATEGORIES.index(src)
                target_phi = phi_matrix[orig_ti, orig_si]
                status = "✓" if abs(achieved - target_phi) < 0.5 * abs(target_phi) else "~"
                print(f"    {status} {tgt:15s} × {src:15s}:  "
                      f"achieved={achieved:.3f}  target={target_phi:.3f}")

    # ── Phase 5: Background fill ──
    print(f"\n── Phase 5: Background fill ──")
    occupied = set()
    all_positions = []
    all_categories = []

    for cat in SA_CATEGORIES:
        p = pos_by_cat.get(cat, np.empty((0, 2), np.int32))
        for row in p:
            occupied.add((int(row[0]), int(row[1])))
            all_positions.append([int(row[0]), int(row[1]), 0])
            all_categories.append(cat)

    area_mm2 = gx * gy * (voxel_um / 1000) ** 2
    n_fib_bg = int(round(args.fib_density * area_mm2)) - n_total_sa
    n_fib_bg = max(0, n_fib_bg)

    n_free = gx * gy - len(occupied)
    if n_fib_bg > n_free:
        n_fib_bg = n_free

    if n_fib_bg > 0:
        # Sample random positions, rejecting occupied ones
        bg_positions = []
        batch_size = n_fib_bg * 2
        while len(bg_positions) < n_fib_bg:
            candidates = rng.integers(0, [gx, gy], size=(batch_size, 2))
            for cx, cy in candidates:
                key = (int(cx), int(cy))
                if key not in occupied:
                    bg_positions.append([int(cx), int(cy), 0])
                    occupied.add(key)
                    if len(bg_positions) >= n_fib_bg:
                        break
            batch_size = min(batch_size * 2, n_free)
        all_positions.extend(bg_positions)
        all_categories.extend([BACKGROUND_CAT] * len(bg_positions))

    print(f"  Background fibroblasts: {n_fib_bg}")
    print(f"  Total agents: {len(all_positions)}")

    # ── Phase 6: Build Vvas and ECM fields ──
    print(f"\n── Phase 6: Fields ──")
    shape_3d = (gx, gy, 1)
    vvas = None
    ecm = None

    if args.healthy_plc:
        from .placement_io import read_placement
        _, vvas_from_plc, _, plc_meta = read_placement(args.healthy_plc)
        if vvas_from_plc is not None:
            plc_shape = plc_meta["shape"]
            if plc_shape == shape_3d:
                vvas = vvas_from_plc
                print(f"  Vvas loaded from {args.healthy_plc}")
            else:
                print(f"  WARNING: healthy .plc shape {plc_shape} != {shape_3d}, skipping Vvas")

    if vvas is None:
        print("  No Vvas field (provide --healthy-plc for perfusion)")

    # ECM from myCAF positions
    mycaf_pos = pos_by_cat.get("fib_myCAF", np.empty((0, 2), np.int32))
    if len(mycaf_pos) > 0:
        from .seeding_ecm import seed_ecm
        all_pos_arr = np.array(all_positions, dtype=np.int32)
        all_cat_arr = np.array(all_categories)
        ecm, ecm_info = seed_ecm(all_pos_arr, all_cat_arr, shape_3d)
        print(f"  ECM: p90={ecm_info['p90']:.1f}, max={ecm_info['max']:.1f}, "
              f"above_collapse={ecm_info['frac_above_collapse']:.1%}")
    else:
        print("  No myCAF → no ECM field")

    # ── Phase 7: Write .plc ──
    print(f"\n── Phase 7: Write placement file ──")
    from .placement_io import build_agent_table, write_placement

    pos_arr = np.array(all_positions, dtype=np.int32)
    cat_arr = np.array(all_categories)

    table, table_info = build_agent_table(
        pos_arr, cat_arr, shape_3d,
        substate_frac=substate_frac if substate_frac else None,
        seed=args.seed,
    )
    print(f"  Agent table: {table_info['n_agents']} agents")
    print(f"  Per category: {table_info['per_category']}")
    if table_info.get("substates"):
        print(f"  Sub-states: {table_info['substates']}")

    result = write_placement(args.output, table, shape_3d, vvas=vvas, ecm=ecm)
    print(f"\nWrote {result['path']} ({result['bytes']:,d} bytes)")
    print(f"  has_vvas={result['has_vvas']}, has_ecm={result['has_ecm']}")

    # ── Save diagnostics ──
    npz_path = str(Path(args.output).with_suffix(".npz"))
    diag = {
        "sample": args.sample,
        "n_cancer": n_cancer,
        "grid": np.array([gx, gy]),
        "phi_target": phi_matrix,
        "frac_target": frac_target,
        "categories": np.array(SA_CATEGORIES),
        "counts": np.array([counts.get(c, 0) for c in SA_CATEGORIES]),
        "energy_final": sa_info["energy_final"],
        "elapsed_s": sa_info["elapsed_s"],
    }
    for cat in SA_CATEGORIES:
        p = pos_by_cat.get(cat, np.empty((0, 2), np.int32))
        diag[f"pos_{cat}"] = p
    np.savez(npz_path, **diag)
    print(f"Saved diagnostics to {npz_path}")


if __name__ == "__main__":
    main()
