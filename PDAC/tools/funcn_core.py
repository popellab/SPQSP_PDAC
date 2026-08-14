"""Pure FunCN kernel + KDE functions (no import side effects).

Extracted from ``python/funcn_analysis.py`` so they can be imported without
triggering that script's analysis/plotting body, and generalized from the
cubic 50³ grid to an arbitrary anisotropic ``(gx, gy, gz)`` shape (needed for
the flat 100×100×5 section domain).

FunCN(A, B) = mean over target cells a∈A of the Gaussian-kernel density of
source cells b∈B evaluated at a:
    K_σ(r) = 1/(2πσ²)·exp(-r²/2σ²)          (normalization kept 2D per manuscript)
    FunCN(A,B) = mean_{a∈A} Σ_{b∈B} K_σ(a-b)

σ is a PARAMETER, not a constant. The module default is σ = 10 µm (σ_vox 0.5), which
is what the existing ABM-side observable was built with — but note that σ_vox = 0.5 is
the coupling-contract §4.4 artifact regime, where the kernel barely leaves the target's
own voxel and the statistic measures same-voxel co-occupancy rather than a spatial
average. The IMC neighbourhood pipeline (``imc_pipeline/stage4_neighborhoods.py``)
therefore passes σ = 30 µm (σ_vox 1.5, per-voxel-offset weights
1.00/0.80/0.41/0.135/0.03), the locked geometry constant. Callers should pass ``sigma_vox``
explicitly rather than inherit the default.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

SIGMA_UM = 10.0
VOXEL_UM = 20.0
SIGMA_VOX = SIGMA_UM / VOXEL_UM                      # 0.5 voxels
CUTOFF_VOX = max(2, int(np.ceil(3 * SIGMA_VOX)))     # 2 voxels


def cutoff_for(sigma_vox: float) -> int:
    """3σ kernel radius in voxels (≥2, so the kernel always spans a neighbourhood)."""
    return max(2, int(np.ceil(3 * sigma_vox)))


def k0_for(sigma_vox: float) -> float:
    """Kernel value at r=0 — a cell's self-weight, removed by ``exclude_self``."""
    return 1.0 / (2.0 * np.pi * sigma_vox ** 2)


def _as_shape(shape: int | tuple[int, int, int]) -> tuple[int, int, int]:
    return (shape, shape, shape) if isinstance(shape, int) else tuple(shape)  # type: ignore[return-value]


def build_gaussian_kernel(sigma_vox: float = SIGMA_VOX, cutoff_vox: int | None = None) -> np.ndarray:
    """3D Gaussian kernel, normalized 1/(2πσ²)·exp(-r²/2σ²). Grid-independent."""
    R = cutoff_for(sigma_vox) if cutoff_vox is None else cutoff_vox
    ax = np.arange(-R, R + 1, dtype=np.float64)
    kx, ky, kz = np.meshgrid(ax, ax, ax, indexing="ij")
    kernel = (1.0 / (2.0 * np.pi * sigma_vox ** 2)) * np.exp(
        -(kx ** 2 + ky ** 2 + kz ** 2) / (2.0 * sigma_vox ** 2)
    )
    return kernel.astype(np.float32)


_KERNEL = build_gaussian_kernel()


@lru_cache(maxsize=8)
def _kernel_for(sigma_vox: float) -> np.ndarray:
    return build_gaussian_kernel(sigma_vox)


@lru_cache(maxsize=8)
def _kernel_fft(sigma_vox: float, s: tuple[int, ...], flat: bool) -> np.ndarray:
    """Cached FFT of the kernel for one padded grid shape.

    ``flat`` selects the gz==1 path, which convolves with the kernel's kz=0 plane.
    That plane IS the 2D Gaussian (the 3D normalization is already 2D), and it is
    exactly what the 3D path selects when it crops a single z-slice — same estimator,
    without paying for the (2R+1) z-planes that a 36 mm-wide core makes expensive.
    """
    k = _kernel_for(sigma_vox)
    return np.fft.fftn(k[:, :, k.shape[2] // 2] if flat else k, s=s)


def build_kde_field(positions: np.ndarray, shape: int | tuple[int, int, int],
                    sigma_vox: float = SIGMA_VOX) -> np.ndarray:
    """3D source-density field over an anisotropic grid via FFT convolution.

    ``positions``: (N, 3) integer voxel coords. ``shape``: (gx, gy, gz) or int.
    Returns a (gx, gy, gz) float32 array; voxel value = Σ_b K_σ(voxel - b).
    """
    gx, gy, gz = _as_shape(shape)
    field = np.zeros((gx, gy, gz), dtype=np.float32)
    if len(positions) == 0:
        return field

    pos = positions.astype(np.int64)
    pos[:, 0] = np.clip(pos[:, 0], 0, gx - 1)
    pos[:, 1] = np.clip(pos[:, 1], 0, gy - 1)
    pos[:, 2] = np.clip(pos[:, 2], 0, gz - 1)
    np.add.at(field, (pos[:, 0], pos[:, 1], pos[:, 2]), 1.0)

    R = cutoff_for(sigma_vox)
    pad = 2 * R
    flat = gz == 1
    s = (gx + pad, gy + pad) if flat else (gx + pad, gy + pad, gz + pad)
    src = field[:, :, 0] if flat else field
    conv = np.real(np.fft.ifftn(np.fft.fftn(src, s=s) * _kernel_fft(sigma_vox, s, flat)))
    # Density is non-negative; clip tiny negative FFT roundoff (parity with IMC nbhd_*).
    np.maximum(conv, 0.0, out=conv)
    if flat:
        return conv[R:R + gx, R:R + gy, None].astype(np.float32)
    return conv[R:R + gx, R:R + gy, R:R + gz].astype(np.float32)


def funcn_distribution(target_pos: np.ndarray, source_kde: np.ndarray) -> np.ndarray:
    """Per-target-cell FunCN score (1D array, length N_target)."""
    if len(target_pos) == 0:
        return np.array([], dtype=np.float32)
    gx, gy, gz = source_kde.shape
    p = target_pos.astype(np.int64)
    px = np.clip(p[:, 0], 0, gx - 1)
    py = np.clip(p[:, 1], 0, gy - 1)
    pz = np.clip(p[:, 2], 0, gz - 1)
    return source_kde[px, py, pz]


def funcn_stats(target_pos: np.ndarray, source_kde: np.ndarray) -> tuple[float, float]:
    """(mean, median) of the FunCN distribution, or (nan, nan) if no targets."""
    dist = funcn_distribution(target_pos, source_kde)
    if len(dist) == 0:
        return np.nan, np.nan
    return float(np.mean(dist)), float(np.median(dist))


# Quantile grid for the distributional observable (maps the within-core shape of
# each target→source FunCN distribution, replacing the single mean). See
# observables.py. p10..p90 captures body + both tails (the dense-niche signal).
FUNCN_QUANTILES: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75, 0.90)

# Minimum target-cell count below which a pair's quantiles are unreliable → NaN.
FUNCN_MIN_TARGETS: int = 10


def funcn_quantiles(target_pos: np.ndarray, source_kde: np.ndarray,
                    qs: tuple[float, ...] = FUNCN_QUANTILES,
                    min_targets: int = FUNCN_MIN_TARGETS) -> np.ndarray:
    """Quantiles of the per-target-cell FunCN distribution.

    Returns an array of length ``len(qs)``; all-NaN when the target category has
    fewer than ``min_targets`` cells (quantiles of a tiny sample are noise).
    """
    dist = funcn_distribution(target_pos, source_kde)
    if len(dist) < min_targets:
        return np.full(len(qs), np.nan, dtype=np.float64)
    return np.quantile(dist.astype(np.float64), qs)


# ── Normalized neighbourhood-FRACTION FunCN (the 0-1, scale-free metric) ─────
# The raw density above is a fixed-amplitude weighted COUNT that over-packs (a
# dense/multi-per-voxel neighbourhood can exceed 1). The normalized metric asks,
# per target cell: of the kernel-weighted cells in the ~3σ neighbourhood, what
# FRACTION are of each source class? Self-contribution K0 is removed from the
# target's own class and the denominator, so the fractions sum to 1 and lie in
# [0, 1] regardless of packing density (see the PM-031 patient-match findings:
# raw-density EMD 0.86 vs normalized 0.09-0.12).
K0 = 1.0 / (2.0 * np.pi * SIGMA_VOX ** 2)   # kernel value at r=0 (a cell's self-weight)


def funcn_frac(pos_by_cat: dict[str, np.ndarray], categories: list[str],
               shape: int | tuple[int, int, int],
               exclude_self: bool = True,
               sigma_vox: float = SIGMA_VOX) -> dict[tuple[str, str], np.ndarray]:
    """{(target, source): 1D per-target-cell neighbourhood FRACTION} over ``categories``.

    For a target cell at voxel v: frac_S = (Σ_{S cells} K) / (Σ_{all cells} K).
    Isolated target cells (denom≈0 after self-exclusion) are dropped, so each
    returned array holds the fractions of the NON-isolated target cells only
    (NaN-free); the len of any (t, s) array = that target's non-isolated count.
    """
    k0 = k0_for(sigma_vox)
    fields = {c: build_kde_field(pos_by_cat.get(c, np.empty((0, 3), np.int64)), shape,
                                 sigma_vox)
              for c in categories}
    total = sum(fields.values())
    gx, gy, gz = total.shape
    out: dict[tuple[str, str], np.ndarray] = {}
    for t in categories:
        tp = pos_by_cat.get(t, np.empty((0, 3), np.int64))
        if len(tp) == 0:
            for s in categories:
                out[(t, s)] = np.array([], dtype=np.float64)
            continue
        p = tp.astype(np.int64)
        px = np.clip(p[:, 0], 0, gx - 1); py = np.clip(p[:, 1], 0, gy - 1)
        pz = np.clip(p[:, 2], 0, gz - 1)
        denom = total[px, py, pz].astype(np.float64)
        if exclude_self:
            denom = denom - k0
        keep = denom > 1e-6
        for s in categories:
            num = fields[s][px, py, pz].astype(np.float64)
            if exclude_self and s == t:
                num = num - k0
            frac = np.full(len(tp), np.nan)
            frac[keep] = np.clip(num[keep] / denom[keep], 0.0, 1.0)
            out[(t, s)] = frac[keep]
    return out


def funcn_frac_quantiles(pos_by_cat: dict[str, np.ndarray], categories: list[str],
                         shape: int | tuple[int, int, int],
                         qs: tuple[float, ...] = FUNCN_QUANTILES,
                         min_targets: int = FUNCN_MIN_TARGETS) -> np.ndarray:
    """(n_cat, n_cat, len(qs)) quantiles of the normalized-fraction FunCN.

    Row (target) is all-NaN when it has < ``min_targets`` non-isolated target cells.
    """
    frac = funcn_frac(pos_by_cat, categories, shape)
    n, nq = len(categories), len(qs)
    out = np.full((n, n, nq), np.nan, dtype=np.float64)
    for i, t in enumerate(categories):
        for j, s in enumerate(categories):
            dist = frac[(t, s)]
            if len(dist) >= min_targets:
                out[i, j] = np.quantile(dist, qs)
    return out
