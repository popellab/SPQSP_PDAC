"""Spatial heterogeneity via Shannon entropy over cell categories.

Two scopes:
  shannon_global(df, category=...)        → single float (whole snapshot)
  shannon_binned(df, bin_size_voxels, ...) → 3D float array + bin metadata

Both return entropy in bits (log base 2) by default; pass base=np.e for nats.
A bin with zero cells returns NaN in the spatial map.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import io as _io


def _counts_to_entropy(counts: np.ndarray, base: float) -> float:
    """H = -Σ p_i log_b(p_i), 0·log0 = 0."""
    total = counts.sum()
    if total <= 0:
        return np.nan
    p = counts / total
    nz = p > 0
    return float(-(p[nz] * (np.log(p[nz]) / np.log(base))).sum())


def shannon_global(
    df: pd.DataFrame,
    category: str = "type_state",
    base: float = 2.0,
) -> float:
    """Shannon entropy of the category distribution across the whole snapshot."""
    df_cat = _io.add_category(df, mode=category)
    categories = _io.category_list(mode=category)
    counts = df_cat["category"].value_counts().reindex(categories, fill_value=0).to_numpy()
    return _counts_to_entropy(counts, base)


def shannon_binned(
    df: pd.DataFrame,
    bin_size_voxels: int,
    grid_size: int | tuple[int, int, int] | None = None,
    category: str = "type_state",
    base: float = 2.0,
    min_cells_per_bin: int = 1,
) -> dict:
    """Per-bin Shannon entropy over a cubic bin grid.

    Parameters
    ----------
    bin_size_voxels : side length of each cubic bin in voxel units (>= 1).
    grid_size : int (cube) or (gx, gy, gz); inferred from positions if None.
    min_cells_per_bin : bins with < this many cells return NaN.

    Returns
    -------
    dict with keys:
      'entropy'         : (bx, by, bz) float array (NaN where insufficient data).
      'counts'          : (bx, by, bz) int array — total cells per bin.
      'bin_size_voxels' : echoed param.
      'grid_shape'      : inferred/used (gx, gy, gz).
    """
    if bin_size_voxels < 1:
        raise ValueError(f"bin_size_voxels must be >= 1, got {bin_size_voxels}")

    df_cat = _io.add_category(df, mode=category)
    categories = _io.category_list(mode=category)
    cat_to_idx = {c: i for i, c in enumerate(categories)}

    if grid_size is None:
        gx = int(df_cat["x"].max()) + 1
        gy = int(df_cat["y"].max()) + 1
        gz = int(df_cat["z"].max()) + 1
    elif isinstance(grid_size, int):
        gx = gy = gz = int(grid_size)
    else:
        gx, gy, gz = (int(v) for v in grid_size)

    bx = int(np.ceil(gx / bin_size_voxels))
    by = int(np.ceil(gy / bin_size_voxels))
    bz = int(np.ceil(gz / bin_size_voxels))

    xs = df_cat["x"].to_numpy(dtype=np.int64) // bin_size_voxels
    ys = df_cat["y"].to_numpy(dtype=np.int64) // bin_size_voxels
    zs = df_cat["z"].to_numpy(dtype=np.int64) // bin_size_voxels
    cidx = np.array([cat_to_idx[c] for c in df_cat["category"].to_numpy()], dtype=np.int64)

    # 4D count histogram: (bx, by, bz, ncat)
    counts_4d = np.zeros((bx, by, bz, len(categories)), dtype=np.int64)
    np.add.at(counts_4d, (xs, ys, zs, cidx), 1)

    totals = counts_4d.sum(axis=3)
    entropy = np.full((bx, by, bz), np.nan, dtype=np.float64)

    valid = totals >= min_cells_per_bin
    if np.any(valid):
        # Vectorized entropy: H = -Σ p log_b p
        p = counts_4d[valid].astype(np.float64) / totals[valid][:, None]
        with np.errstate(divide="ignore", invalid="ignore"):
            logp = np.log(p, out=np.zeros_like(p), where=p > 0)
        entropy[valid] = -(p * logp / np.log(base)).sum(axis=1)

    return {
        "entropy": entropy,
        "counts": totals,
        "bin_size_voxels": bin_size_voxels,
        "grid_shape": (gx, gy, gz),
    }
