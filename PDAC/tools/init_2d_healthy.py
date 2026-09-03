"""Build a 2D healthy-tissue placement file for the PDAC ABM.

Reads healthy pancreatic IMC data, runs simulated annealing to match spatial
clustering targets, constructs the Vvas field from vessel positions, and writes
a ``.plc`` file that ``initializeFromPlacement`` can load.

Usage::

    python -m PDAC.tools.init_2d_healthy \\
        --grid-x 825 --grid-y 825 --seed 42 \\
        --n-steps 5000000 --output healthy_825.plc
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="2D healthy-tissue SA placement for PDAC ABM"
    )
    p.add_argument("--grid-x", type=int, required=True)
    p.add_argument("--grid-y", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-steps", type=int, default=5_000_000)
    p.add_argument("--log-interval", type=int, default=100_000)
    p.add_argument("--sigma-vox", type=float, default=1.5,
                   help="FunCN kernel sigma in voxels (default 1.5 = 30 µm)")
    p.add_argument("--voxel-um", type=float, default=20.0)
    p.add_argument("--output", type=str, required=True,
                   help="Output .plc file path")
    p.add_argument("--data-dir", type=str, default=None,
                   help="Path to data/healthy_imc/ (auto-detected if omitted)")
    args = p.parse_args(argv)

    gx, gy = args.grid_x, args.grid_y
    voxel_um = args.voxel_um

    # ── Locate data directory ──
    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        data_dir = Path(__file__).resolve().parents[2] / "data" / "healthy_imc"
    density_csv = data_dir / "density_per_roi.csv"
    funcn_csv = data_dir / "funcn_target_mean.csv"
    for f in (density_csv, funcn_csv):
        if not f.exists():
            print(f"ERROR: required file not found: {f}", file=sys.stderr)
            sys.exit(1)

    # ── Cell counts from IMC areal densities ──
    df = pd.read_csv(density_csv)
    med_fib = float(df["density_Fibroblast"].median())
    med_vas = float(df["density_Endothelial"].median())
    area_mm2 = gx * gy * (voxel_um / 1000) ** 2
    n_fib = int(round(med_fib * area_mm2))
    n_vas = int(round(med_vas * area_mm2))
    print(f"Domain: {gx}×{gy}×1 at {voxel_um} µm = {area_mm2:.1f} mm²")
    print(f"Fibroblast density: {med_fib:.0f}/mm² → {n_fib} cells")
    print(f"Endothelial density: {med_vas:.0f}/mm² → {n_vas} cells")
    print(f"Total placed: {n_fib + n_vas}")

    # ── FunCN targets ──
    from .seeding_sa import renormalize_targets, sa_place, validate_placement

    target = renormalize_targets(str(funcn_csv))
    print(f"\nRenormalized FunCN targets (0=vas, 1=fib):")
    print(f"  vas×vas={target[0,0]:.4f}  vas×fib={target[0,1]:.4f}")
    print(f"  fib×vas={target[1,0]:.4f}  fib×fib={target[1,1]:.4f}")

    # ── Simulated annealing ──
    print(f"\nRunning SA ({args.n_steps:,d} steps, sigma_vox={args.sigma_vox})...")
    pos_fib, pos_vas, info = sa_place(
        n_fib, n_vas, (gx, gy), target,
        sigma_vox=args.sigma_vox,
        n_steps=args.n_steps,
        seed=args.seed,
        log_interval=args.log_interval,
    )
    print(f"\nSA finished in {info['elapsed_s']:.0f}s")
    print(f"  Final energy: {info['energy_final']:.2e}")
    print(f"  Acceptance rate: {info['acceptance_rate']:.3f}")
    mf = info["mean_frac"]
    print(f"  Achieved fracs:")
    print(f"    vas×vas={mf[0][0]:.4f} (target {target[0,0]:.4f})")
    print(f"    vas×fib={mf[0][1]:.4f} (target {target[0,1]:.4f})")
    print(f"    fib×vas={mf[1][0]:.4f} (target {target[1,0]:.4f})")
    print(f"    fib×fib={mf[1][1]:.4f} (target {target[1,1]:.4f})")

    # ── Validate with independent FFT recompute ──
    print("\nValidating with independent FFT funcn_frac...")
    val = validate_placement(pos_fib, pos_vas, (gx, gy), args.sigma_vox)
    print(f"  FFT vas×vas={val[0,0]:.4f}  vas×fib={val[0,1]:.4f}")
    print(f"  FFT fib×vas={val[1,0]:.4f}  fib×fib={val[1,1]:.4f}")
    max_diff = float(np.max(np.abs(val - np.array(mf))))
    print(f"  Max |SA - FFT|: {max_diff:.6f}")

    # ── Save raw SA positions for diagnostics ──
    npz_path = str(Path(args.output).with_suffix(".npz"))
    np.savez(npz_path, pos_fib=pos_fib, pos_vas=pos_vas,
             target_frac=target, achieved_frac=np.array(mf))
    print(f"\nSaved raw positions to {npz_path}")

    # ── Build Vvas field ──
    print("\nBuilding Vvas field...")
    from .seeding_vvas import seed_vvas

    shape_3d = (gx, gy, 1)
    vas_pos_3d = np.column_stack(
        [pos_vas, np.zeros(len(pos_vas), dtype=np.int32)]
    )
    vas_cat = np.full(len(pos_vas), "vas_phalanx")
    vvas, vvas_info = seed_vvas(
        vas_pos_3d, vas_cat, shape_3d, normalize="unit"
    )
    print(f"  Vvas coverage (>1% max): {vvas_info['coverage_1pct_of_max']:.1%}")
    print(f"  Vvas zero fraction: {vvas_info['zero_frac']:.1%}")
    print(f"  Vvas max: {vvas_info['max']:.4f}")

    # ── Build agent table ──
    from .placement_io import build_agent_table, write_placement

    fib_pos_3d = np.column_stack(
        [pos_fib, np.zeros(len(pos_fib), dtype=np.int32)]
    )
    fib_cat = np.full(len(pos_fib), "fib_quiescent")

    cancer_pos = np.array([[gx // 2, gy // 2, 0]], dtype=np.int32)
    cancer_cat = np.array(["cancer_stem"])

    all_pos = np.vstack([fib_pos_3d, cancer_pos])
    all_cat = np.concatenate([fib_cat, cancer_cat])

    table, table_info = build_agent_table(all_pos, all_cat, shape_3d)
    print(f"\nAgent table: {table_info['n_agents']} agents")
    print(f"  Per category: {table_info['per_category']}")

    # ── Write .plc ──
    result = write_placement(args.output, table, shape_3d, vvas=vvas)
    print(f"\nWrote {result['path']} ({result['bytes']:,d} bytes)")
    print(f"  has_vvas={result['has_vvas']}, has_ecm={result['has_ecm']}")


if __name__ == "__main__":
    main()
