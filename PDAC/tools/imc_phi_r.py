"""Compute FunCN phi at variable Gaussian radii for all IMC patient ROIs.

Uses the SAME Gaussian-KDE formula as the ABM phi diagnostic:
  - Each source of type B deposits an unnormalized Gaussian (peak 1.0) on a voxel grid
  - Each target of type A samples the KDE at its voxel, edge-corrected by sumK_local
  - phi(sigma) = mean(KDE_B(x_i) / sumK_local(x_i)) / rho_B

Edge correction: sumK_local(v) = SUM_{u in domain} K(v - u) ensures that targets
near domain boundaries get the same expected value as interior targets under uniform B.

Coordinates are discretized to 20 um voxels to match the ABM grid.

Usage:
    python -m PDAC.tools.imc_phi_r                        # default paths
    python -m PDAC.tools.imc_phi_r --data-dir <dir> -o <out.csv>
    python -m PDAC.tools.imc_phi_r --sigmas 1.0 2.5 5.0   # custom sigma list
"""
import argparse, pathlib, time
import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

RIP_NAMES = [
    "cancer", "cd8", "treg", "th", "m1", "m2",
    "icaf", "mycaf", "apcaf", "mdsc", "bcell", "dc",
]
N_RIP = len(RIP_NAMES)

IMC_TO_RIP = {
    ("cancer", "proliferating"): 0,
    ("cancer", "quiescent"): 0,
    ("cd8_t", "effector"): 1,
    ("cd4_t", "treg"): 2,
    ("cd4_t", "th"): 3,
    ("macrophage", "m1"): 4,
    ("macrophage", "m2"): 5,
    ("fibroblast", "iCAF"): 6,
    ("fibroblast", "myCAF"): 7,
    ("fibroblast", "activated_CAF"): 7,
    ("fibroblast", "apCAF"): 8,
    ("mdsc", "m_mdsc"): 9,
    ("mdsc", "pmn_mdsc"): 9,
    ("mdsc", "mdsc_other"): 9,
    ("b_cell", "naive"): 10,
    ("b_cell", "activated"): 10,
    ("dendritic", "mature"): 11,
    ("dendritic", "immature"): 11,
}

VOXEL_SIZE = 20.0

PHI_PAIRS = [
    (0, 1, "cd8_c1"),
    (4, 0, "m1_c1"),
    (1, 2, "cd8_treg"),
    (5, 0, "m2_c1"),
    (6, 0, "icaf_c1"),
    (9, 1, "mdsc_cd8"),
    (10, 1, "bcell_cd8"),
    (7, 0, "mycaf_c1"),
]

DEFAULT_SIGMAS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 5.0, 7.5, 10.0, 15.0, 20.0, 25.0]


def make_gaussian_kernel_3d(sigma, radius):
    """Unnormalized 3D isotropic Gaussian kernel (peak 1.0), matching ABM phi_kernel."""
    r = int(radius)
    coords = np.arange(-r, r + 1)
    dz, dy, dx = np.meshgrid(coords, coords, coords, indexing="ij")
    d2 = dx**2 + dy**2 + dz**2
    K = np.exp(-d2 / (2.0 * sigma**2)).astype(np.float64)
    return K


def make_gaussian_kernel_2d(sigma, radius):
    """Unnormalized 2D isotropic Gaussian kernel (peak 1.0)."""
    r = int(radius)
    coords = np.arange(-r, r + 1)
    dy, dx = np.meshgrid(coords, coords, indexing="ij")
    d2 = dx**2 + dy**2
    K = np.exp(-d2 / (2.0 * sigma**2)).astype(np.float64)
    return K


def _fft_conv_2d(data, kernel):
    """Zero-padded 2D convolution via FFT (O(N log N), fast for large kernels)."""
    return fftconvolve(data, kernel, mode="same")


def compute_sumk_2d(gx, gy, kernel):
    """Precompute sumK_local: kernel mass inside domain at each voxel position."""
    ones = np.ones((gy, gx), dtype=np.float64)
    return _fft_conv_2d(ones, kernel)


def compute_phi_pair_2d(count_grid_target, count_grid_source, kernel, sumk, nvox):
    """Compute phi for one (target, source) pair on a 2D grid.

    phi = mean_over_targets( KDE_source(x_i) / sumK(x_i) ) / rho_source
    """
    n_target = count_grid_target.sum()
    n_source = count_grid_source.sum()
    if n_target < 1 or n_source < 1:
        return np.nan, False

    kde_field = _fft_conv_2d(count_grid_source.astype(np.float64), kernel)

    safe_sumk = np.where(sumk > 0, sumk, 1.0)
    ratio_field = kde_field / safe_sumk
    ratio_field[sumk <= 0] = 0.0

    weighted_sum = (ratio_field * count_grid_target).sum()
    mean_local = weighted_sum / n_target

    rho_source = n_source / nvox
    phi = mean_local / rho_source
    return phi, True


def process_one_roi(df, sigmas, voxel_size=VOXEL_SIZE):
    """Compute phi at each sigma for all pairs in one ROI."""
    keys = list(zip(df["cell_type"].astype(str), df["subtype"].astype(str)))
    rip_idx = np.array([IMC_TO_RIP.get(k, -1) for k in keys])
    keep = rip_idx >= 0
    rip_arr = rip_idx[keep]
    x_um = df["x"].values[keep]
    y_um = df["y"].values[keep]

    x_vox = (x_um - x_um.min()) / voxel_size
    y_vox = (y_um - y_um.min()) / voxel_size

    gx = int(np.ceil(x_vox.max())) + 1
    gy = int(np.ceil(y_vox.max())) + 1
    if gx < 3 or gy < 3:
        return None

    nvox = gx * gy

    count_grids = {}
    counts = np.zeros(N_RIP, dtype=int)
    for t in range(N_RIP):
        mask = rip_arr == t
        counts[t] = mask.sum()
        if counts[t] > 0:
            grid = np.zeros((gy, gx), dtype=np.int32)
            xi = np.clip(np.floor(x_vox[mask]).astype(int), 0, gx - 1)
            yi = np.clip(np.floor(y_vox[mask]).astype(int), 0, gy - 1)
            np.add.at(grid, (yi, xi), 1)
            count_grids[t] = grid

    results = {}
    for t in range(N_RIP):
        results[f"n_{RIP_NAMES[t]}"] = int(counts[t])
    results["domain_vox_x"] = gx
    results["domain_vox_y"] = gy

    for sigma in sigmas:
        radius = int(np.ceil(3.2 * sigma))
        if radius < 1:
            radius = 1
        kernel = make_gaussian_kernel_2d(sigma, radius)
        sumk = compute_sumk_2d(gx, gy, kernel)

        for tgt_idx, src_idx, pair_name in PHI_PAIRS:
            col = f"phi_{pair_name}_s{sigma:.1f}"
            if tgt_idx not in count_grids or src_idx not in count_grids:
                results[col] = np.nan
                results[col.replace("phi_", "valid_")] = 0
                continue
            phi, valid = compute_phi_pair_2d(
                count_grids[tgt_idx], count_grids[src_idx],
                kernel, sumk, nvox)
            results[col] = phi
            results[col.replace("phi_", "valid_")] = 1 if valid else 0

    return results


def main():
    proj_root = pathlib.Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Compute IMC phi(r) at variable sigma")
    parser.add_argument("--data-dir", type=pathlib.Path,
                        default=proj_root / "data" / "PASS01" / "derived" / "stage3")
    parser.add_argument("-o", "--output", type=pathlib.Path,
                        default=proj_root / "data" / "PASS01" / "derived" / "imc_phi_r.csv")
    parser.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS,
                        help="Gaussian sigma values in voxel units")
    args = parser.parse_args()

    parquets = sorted(args.data_dir.glob("*.parquet"))
    print(f"Found {len(parquets)} patient ROIs in {args.data_dir}")
    print(f"Sigma values (voxels): {args.sigmas}")
    print(f"Sigma values (um): {[s * VOXEL_SIZE for s in args.sigmas]}")

    rows = []
    t0 = time.time()
    for i, pq in enumerate(parquets):
        sample = pq.stem
        print(f"  [{i+1}/{len(parquets)}] {sample} ...", end=" ", flush=True)
        t1 = time.time()
        df = pd.read_parquet(pq, columns=["x", "y", "cell_type", "subtype"])
        result = process_one_roi(df, args.sigmas)
        if result is None:
            print("SKIP (too small)")
            continue
        result["sample"] = sample
        rows.append(result)
        dt = time.time() - t1
        print(f"{dt:.1f}s")

    out_df = pd.DataFrame(rows)

    phi_cols = sorted([c for c in out_df.columns if c.startswith("phi_")])
    valid_cols = sorted([c for c in out_df.columns if c.startswith("valid_")])
    n_cols = [f"n_{n}" for n in RIP_NAMES]
    meta_cols = ["domain_vox_x", "domain_vox_y"]
    col_order = (["sample"] + phi_cols + valid_cols
                 + [c for c in n_cols if c in out_df.columns] + meta_cols)
    col_order = [c for c in col_order if c in out_df.columns]
    out_df = out_df[col_order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    total = time.time() - t0
    print(f"\nWrote {args.output}  ({len(out_df)} ROIs, {len(out_df.columns)} cols, {total:.0f}s)")


if __name__ == "__main__":
    main()
