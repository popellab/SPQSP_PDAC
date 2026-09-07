"""Compute Ripley's L(r)-r for all IMC patient ROIs.

Produces a CSV with the same column structure as the ABM's ripley_*.csv,
enabling direct comparison of spatial clustering at every integer radius.

Both ABM and IMC use the 2D Ripley formula: L(r) = sqrt(K/pi).
Coordinates are converted to voxel units (÷ 20 µm) so radii match ABM.

Usage:
    python -m PDAC.tools.imc_ripley                        # default paths
    python -m PDAC.tools.imc_ripley --data-dir <dir> -o <out.csv>
"""
import argparse, pathlib, time
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# ── Ripley type enum (must match PDAC::RipleyType in diagnostics/ripley.cuh) ──
RIP_NAMES = [
    "cancer", "cd8", "treg", "th", "m1", "m2",
    "icaf", "mycaf", "apcaf", "mdsc", "bcell", "dc",
]
N_RIP = len(RIP_NAMES)

# ── IMC (cell_type, subtype) → Ripley index ──
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

VOXEL_SIZE = 20.0  # µm per voxel


def ripley_lr_cross(pts_a, pts_b, radii, domain_size, is_auto=False):
    """L(r)-r for point sets A, B in 2D rectangle [0,Lx]×[0,Ly].

    Uses translation edge correction. Vectorized over radii via searchsorted.
    """
    na, nb = len(pts_a), len(pts_b)
    if na < 1 or nb < 1:
        return np.full(len(radii), np.nan)

    Lx, Ly = domain_size
    A = Lx * Ly
    r_max = float(radii[-1])
    n_radii = len(radii)
    radii_f = radii.astype(np.float64)

    tree_b = cKDTree(pts_b)
    K_sum = np.zeros(n_radii)

    all_pairs = tree_b.query_ball_point(pts_a, r_max)

    for i in range(na):
        idxs = all_pairs[i]
        if len(idxs) == 0:
            continue
        b_sel = pts_b[idxs]
        pair_dists = np.sqrt(np.sum((b_sel - pts_a[i])**2, axis=1))

        if is_auto:
            mask = pair_dists > 0
            pair_dists = pair_dists[mask]
            b_sel = b_sel[mask]
            if len(pair_dists) == 0:
                continue

        dx = np.abs(b_sel[:, 0] - pts_a[i, 0])
        dy = np.abs(b_sel[:, 1] - pts_a[i, 1])
        weights = A / (np.maximum(Lx - dx, 1.0) * np.maximum(Ly - dy, 1.0))

        sort_idx = np.argsort(pair_dists)
        pair_dists_sorted = pair_dists[sort_idx]
        weights_sorted = weights[sort_idx]
        cum_weights = np.cumsum(weights_sorted)

        cutoffs = np.searchsorted(pair_dists_sorted, radii_f, side='right')
        valid = cutoffs > 0
        K_sum[valid] += cum_weights[cutoffs[valid] - 1]

    denom = na * (nb - 1) if is_auto else na * nb
    if denom <= 0:
        return np.full(n_radii, np.nan)
    K = A * K_sum / denom

    Lr = np.sqrt(K / np.pi) - radii_f
    return Lr


def process_one_roi(df, voxel_size=VOXEL_SIZE, max_radii=50):
    """Compute Ripley L(r)-r for all 78 unique subtype pairs in one ROI."""
    keys = list(zip(df["cell_type"].astype(str), df["subtype"].astype(str)))
    rip_idx = np.array([IMC_TO_RIP.get(k, -1) for k in keys])
    keep = rip_idx >= 0
    rip_arr = rip_idx[keep]
    x = (df["x"].values[keep] - df["x"].values[keep].min()) / voxel_size
    y = (df["y"].values[keep] - df["y"].values[keep].min()) / voxel_size

    Lx = x.max()
    Ly = y.max()
    if Lx < 2 or Ly < 2:
        return None

    n_radii = min(int(min(Lx, Ly) / 2), max_radii)
    if n_radii < 1:
        return None
    radii = np.arange(1, n_radii + 1)

    pts = {}
    counts = np.zeros(N_RIP, dtype=int)
    for t in range(N_RIP):
        mask = rip_arr == t
        counts[t] = mask.sum()
        if counts[t] > 0:
            pts[t] = np.column_stack([x[mask], y[mask]])

    domain = (Lx, Ly)

    results = {}
    for a in range(N_RIP):
        for b in range(a, N_RIP):
            if counts[a] < 1 or counts[b] < 1:
                Lr = np.full(n_radii, np.nan)
            else:
                Lr = ripley_lr_cross(pts[a], pts[b], radii, domain, is_auto=(a == b))
            for ri in range(n_radii):
                col = f"L_{RIP_NAMES[a]}_{RIP_NAMES[b]}_r{radii[ri]}"
                results[col] = Lr[ri]

    for t in range(N_RIP):
        results[f"n_{RIP_NAMES[t]}"] = int(counts[t])

    results["n_radii"] = n_radii
    results["domain_vox_x"] = float(Lx)
    results["domain_vox_y"] = float(Ly)
    return results


def main():
    proj_root = pathlib.Path(__file__).resolve().parents[2]

    parser = argparse.ArgumentParser(description="Compute IMC Ripley L(r)-r")
    parser.add_argument("--data-dir", type=pathlib.Path,
                        default=proj_root / "data" / "PASS01" / "derived" / "stage3")
    parser.add_argument("-o", "--output", type=pathlib.Path,
                        default=proj_root / "data" / "PASS01" / "derived" / "imc_ripley.csv")
    args = parser.parse_args()

    parquets = sorted(args.data_dir.glob("*.parquet"))
    print(f"Found {len(parquets)} patient ROIs in {args.data_dir}")

    rows = []
    t0 = time.time()
    for i, pq in enumerate(parquets):
        sample = pq.stem
        print(f"  [{i+1}/{len(parquets)}] {sample} ...", end=" ", flush=True)
        t1 = time.time()
        df = pd.read_parquet(pq, columns=["x", "y", "cell_type", "subtype"])
        result = process_one_roi(df)
        if result is None:
            print("SKIP (too small)")
            continue
        result["sample"] = sample
        rows.append(result)
        dt = time.time() - t1
        nc = result.get("n_cancer", 0)
        print(f"n_radii={result['n_radii']}, n_cancer={nc}, {dt:.1f}s")

    out_df = pd.DataFrame(rows)

    l_cols = sorted([c for c in out_df.columns if c.startswith("L_")])
    n_cols = [f"n_{n}" for n in RIP_NAMES]
    meta_cols = ["n_radii", "domain_vox_x", "domain_vox_y"]
    col_order = ["sample"] + l_cols + [c for c in n_cols if c in out_df.columns] + meta_cols
    col_order = [c for c in col_order if c in out_df.columns]
    out_df = out_df[col_order]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    total = time.time() - t0
    print(f"\nWrote {args.output}  ({len(out_df)} ROIs, {len(out_df.columns)} cols, {total:.0f}s)")


if __name__ == "__main__":
    main()
