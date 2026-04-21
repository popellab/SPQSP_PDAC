"""Functional Cell Neighborhood (FunCN) score — full 3D kernel implementation.

For each target cell i and each source category S, compute:

    raw  : FunCN(i, S) = sum_{j in S, j != i}  K_sigma(x_i - x_j)
    nw   : FunCN(i, S) = raw(i, S) / sum_{j != i} K_sigma(x_i - x_j)   [Nadaraya-Watson]

with a 3D-normalized isotropic Gaussian kernel:

    K_sigma(r) = 1 / ((2 pi)^(3/2) sigma^3) * exp(-||r||^2 / (2 sigma^2))

Distance is in microns (voxel_um * voxel_index difference). Agents sit on
discrete voxel centers; self-contribution K(0) is subtracted from both numerator
(when i itself is in S) and denominator to recover the leave-one-out interpretation.

The kernel is evaluated by painting per-category density fields onto the grid
and convolving with a separable Gaussian via scipy.ndimage — O(N_cells + grid^3)
per category, vs O(N^2) for pairwise summation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from . import io as _io


def _paint_density(positions: np.ndarray, grid_shape: tuple[int, int, int]) -> np.ndarray:
    """Accumulate per-voxel counts. positions: (N, 3) int voxel coords."""
    field = np.zeros(grid_shape, dtype=np.float64)
    if len(positions) == 0:
        return field
    xs, ys, zs = positions[:, 0], positions[:, 1], positions[:, 2]
    # Clamp to grid (should be unnecessary; safety for off-by-one)
    xs = np.clip(xs, 0, grid_shape[0] - 1)
    ys = np.clip(ys, 0, grid_shape[1] - 1)
    zs = np.clip(zs, 0, grid_shape[2] - 1)
    np.add.at(field, (xs, ys, zs), 1.0)
    return field


def funcn_per_cell(
    df: pd.DataFrame,
    sigma_um: float = 10.0,
    voxel_um: float = 20.0,
    grid_size: int | tuple[int, int, int] | None = None,
    category: str = "type_state",
    normalize: str = "nw",
    truncate: float = 4.0,
) -> pd.DataFrame:
    """Per-cell FunCN score matrix (N_cells rows × N_categories columns).

    Parameters
    ----------
    df : DataFrame with columns [type, id, x, y, z, state, ...] (output of io.load_abm_lz4).
    sigma_um : Gaussian kernel bandwidth in microns (default 10 per manuscript).
    voxel_um : voxel size in microns (default 20).
    grid_size : int (cube) or (gx, gy, gz). Inferred from positions if None.
    category : 'type' or 'type_state' (see io.add_category).
    normalize : 'nw'   → Nadaraya-Watson ratio (fraction of source density),
                'raw'  → raw kernel-weighted count,
                'none' → kernel-weighted with prefactor 1/((2π)^(3/2) σ³) dropped.
    truncate : scipy.ndimage.gaussian_filter truncate parameter (std-devs).

    Returns
    -------
    DataFrame indexed by agent row order, columns = sorted category labels.
    Each row = one target cell; value = FunCN(cell, source_category).
    """
    if normalize not in {"nw", "raw", "none"}:
        raise ValueError(f"normalize must be 'nw'|'raw'|'none', got {normalize!r}")

    df_cat = _io.add_category(df, mode=category)
    categories = _io.category_list(mode=category)

    if grid_size is None:
        gx = int(df_cat["x"].max()) + 1
        gy = int(df_cat["y"].max()) + 1
        gz = int(df_cat["z"].max()) + 1
    elif isinstance(grid_size, int):
        gx = gy = gz = int(grid_size)
    else:
        gx, gy, gz = (int(v) for v in grid_size)
    grid_shape = (gx, gy, gz)

    # Sigma in voxel units for scipy (separable Gaussian along each axis)
    sigma_vox = sigma_um / voxel_um

    # Measure scipy's actual kernel peak K(0) by convolving a unit delta. This
    # matches scipy.ndimage.gaussian_filter's discrete-sum normalization (which
    # differs from the continuous Gaussian integral 1/((2π)^(3/2) σ³) for small σ).
    _probe_r = int(np.ceil(truncate * sigma_vox)) + 1
    _probe_side = 2 * _probe_r + 1
    _probe = np.zeros((_probe_side,) * 3, dtype=np.float64)
    _probe[_probe_r, _probe_r, _probe_r] = 1.0
    _probe_conv = gaussian_filter(_probe, sigma=sigma_vox, mode="constant",
                                   cval=0.0, truncate=truncate)
    k0_vox = float(_probe_conv[_probe_r, _probe_r, _probe_r])  # in voxel units

    xs = np.clip(df_cat["x"].to_numpy(dtype=np.int64), 0, gx - 1)
    ys = np.clip(df_cat["y"].to_numpy(dtype=np.int64), 0, gy - 1)
    zs = np.clip(df_cat["z"].to_numpy(dtype=np.int64), 0, gz - 1)
    cat_arr = df_cat["category"].to_numpy()

    # Voxel volume in cubic microns — needed to convert density-field convolution
    # (which sums unnormalized kernel values over integer-grid voxel centers)
    # into the continuous-space integral.
    vox_vol_um3 = voxel_um ** 3

    # Rescale convolved field to requested units. scipy returns a discrete-sum
    # normalized Gaussian in voxel units. For 'raw' and 'nw' we want a kernel
    # whose values represent the continuous 3D Gaussian in micron-space — i.e.,
    # K_um(r) = K_vox(r/voxel_um) / vox_vol_um3. 'nw' divides out the scale, so
    # only 'raw' actually depends on the absolute normalization.
    scale = 1.0 if normalize == "none" else (1.0 / vox_vol_um3)
    k0 = k0_vox * scale  # kernel peak in whichever units we're outputting

    # Per-category density fields, convolved with 3D Gaussian
    convolved: dict[str, np.ndarray] = {}
    for cat in categories:
        mask = cat_arr == cat
        if not np.any(mask):
            convolved[cat] = np.zeros(grid_shape, dtype=np.float64)
            continue
        pos = np.stack([xs[mask], ys[mask], zs[mask]], axis=1)
        density = _paint_density(pos, grid_shape)
        conv = gaussian_filter(density, sigma=sigma_vox, mode="constant",
                                cval=0.0, truncate=truncate)
        convolved[cat] = conv * scale

    # Also compute total-cell convolution for N-W denominator
    if normalize == "nw":
        pos_all = np.stack([xs, ys, zs], axis=1)
        dens_all = _paint_density(pos_all, grid_shape)
        conv_all = gaussian_filter(dens_all, sigma=sigma_vox, mode="constant",
                                    cval=0.0, truncate=truncate)
        conv_all = conv_all * scale

    # Sample per cell and build output matrix
    out = np.empty((len(df_cat), len(categories)), dtype=np.float64)
    for col_idx, cat in enumerate(categories):
        sampled = convolved[cat][xs, ys, zs]
        # Leave-one-out: subtract self contribution where cell_i is in category cat
        self_in_cat = (cat_arr == cat).astype(np.float64)
        sampled = sampled - k0 * self_in_cat
        # Numerical floor (self-subtract can produce tiny negatives)
        sampled = np.maximum(sampled, 0.0)
        if normalize == "nw":
            denom = conv_all[xs, ys, zs] - k0  # subtract self (every cell is in total)
            # Cells with no other neighbors within kernel support → N-W undefined.
            # Use a small fraction of K(0) as the threshold: below this the cell is
            # effectively isolated and any ratio is numerical noise.
            valid = denom > (k0 * 1e-3)
            result = np.full_like(sampled, np.nan, dtype=np.float64)
            result[valid] = sampled[valid] / denom[valid]
            sampled = result
        out[:, col_idx] = sampled

    return pd.DataFrame(out, columns=categories, index=df_cat.index)


def funcn_by_target(
    per_cell: pd.DataFrame,
    df: pd.DataFrame,
    category: str = "type_state",
    agg: str = "mean",
) -> pd.DataFrame:
    """Aggregate per-cell FunCN scores by target category.

    Returns (N_target_categories × N_source_categories) DataFrame.
    Rows = target category, columns = source category, values = agg(scores).
    """
    df_cat = _io.add_category(df, mode=category)
    joined = per_cell.copy()
    joined["_target"] = df_cat["category"].values
    grouped = joined.groupby("_target")
    if agg == "mean":
        return grouped.mean()
    if agg == "median":
        return grouped.median()
    raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")
