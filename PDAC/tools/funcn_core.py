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
Unified σ = 50 µm (σ_vox = 2.5) across IMC targets, SA placement, and ABM runtime.
The default SIGMA_VOX is 2.5 voxels at 20 µm/vox = 50 µm.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

SIGMA_UM = 50.0
VOXEL_UM = 20.0
SIGMA_VOX = SIGMA_UM / VOXEL_UM                      # 2.5 voxels
CUTOFF_VOX = max(2, int(np.ceil(3 * SIGMA_VOX)))     # 8 voxels

CANCER_CATS = frozenset({"cancer_sen", "cancer_prog", "cancer_stem"})
CANCER_WINDOW_MARGIN = 10  # voxels of padding beyond cancer extent


def cutoff_for(sigma_vox: float) -> int:
    """3σ kernel radius in voxels (≥2, so the kernel always spans a neighbourhood)."""
    return max(2, int(np.ceil(3 * sigma_vox)))


def k0_for(sigma_vox: float) -> float:
    """Kernel value at r=0 — a cell's self-weight, removed by ``exclude_self``."""
    return 1.0 / (2.0 * np.pi * sigma_vox ** 2)


def _as_shape(shape: int | tuple[int, int, int]) -> tuple[int, int, int]:
    return (shape, shape, shape) if isinstance(shape, int) else tuple(shape)  # type: ignore[return-value]


def cancer_window(
    pos_by_cat: dict[str, np.ndarray],
    shape: int | tuple[int, int, int],
    margin: int = CANCER_WINDOW_MARGIN,
    cancer_cats: frozenset[str] = CANCER_CATS,
) -> tuple[dict[str, np.ndarray], tuple[int, int, int]]:
    """Crop all positions to a bounding box around cancer cells + margin.

    Returns (cropped_pos_by_cat, window_shape) where positions are translated
    to window-relative coordinates. Cells outside the window are dropped.
    If no cancer cells exist, returns the original data unchanged.
    """
    gx, gy, gz = _as_shape(shape)
    cancer_pos = [pos_by_cat[c] for c in cancer_cats
                  if c in pos_by_cat and len(pos_by_cat[c]) > 0]
    if not cancer_pos:
        return pos_by_cat, (gx, gy, gz)
    all_cancer = np.concatenate(cancer_pos, axis=0)

    x0 = max(0, int(all_cancer[:, 0].min()) - margin)
    x1 = min(gx - 1, int(all_cancer[:, 0].max()) + margin)
    y0 = max(0, int(all_cancer[:, 1].min()) - margin)
    y1 = min(gy - 1, int(all_cancer[:, 1].max()) + margin)
    z0 = max(0, int(all_cancer[:, 2].min()) - margin)
    z1 = min(gz - 1, int(all_cancer[:, 2].max()) + margin)

    win_shape = (x1 - x0 + 1, y1 - y0 + 1, z1 - z0 + 1)
    out: dict[str, np.ndarray] = {}
    for cat, pos in pos_by_cat.items():
        if len(pos) == 0:
            out[cat] = pos
            continue
        mask = ((pos[:, 0] >= x0) & (pos[:, 0] <= x1) &
                (pos[:, 1] >= y0) & (pos[:, 1] <= y1) &
                (pos[:, 2] >= z0) & (pos[:, 2] <= z1))
        cropped = pos[mask].copy()
        cropped[:, 0] -= x0
        cropped[:, 1] -= y0
        cropped[:, 2] -= z0
        out[cat] = cropped
    return out, win_shape


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


# ── Permutation-based phi (the co-location metric) ───────────────────────────
#
# phi(A, B) = T_obs / E_perm[T]
#
# where T = mean_{a in A} KDE_B(a) is the average kernel density of source
# cells B evaluated at target cells A, and E_perm is the expectation under
# random label permutation (positions fixed, type labels shuffled without
# replacement, group sizes preserved).
#
# CLOSED-FORM DERIVATION:
#
# Under permutation, any cell i gets label A with probability n_A/N, and for
# i != j, the joint probability P(i in A, j in B) = n_A * n_B / (N * (N-1))
# (sampling without replacement from N labels of which n_A are "A" and n_B
# are "B").
#
# The observed statistic (for A != B, so i in A and j in B implies i != j):
#
#   T_obs = (1/n_A) * sum_{i in A} sum_{j in B} K(x_i - x_j)
#         = (1/n_A) * sum_{i in A} KDE_B(x_i)
#
# Its expectation under label permutation:
#
#   E_perm[T] = (1/n_A) * sum_{i != j} K(x_i - x_j) * P(i in A, j in B)
#             = (1/n_A) * sum_{i != j} K(x_i - x_j) * n_A * n_B / (N*(N-1))
#             = n_B / (N*(N-1)) * sum_{i != j} K(x_i - x_j)
#
# Now sum_{i != j} K(x_i - x_j) = sum_i KDE_ALL(x_i) - N * K(0) = S - N*k0,
# where S = sum_i KDE_ALL(x_i) and k0 = K(0) = 1/(2*pi*sigma^2).
#
# Define W = S/N - k0 (the mean self-excluded total density). Then:
#
#   E_perm[T] = n_B * W / (N - 1)                                 [A != B]
#
# For the SELF-PAIR (A = B), the observed T self-excludes (j != i):
#
#   T_obs = (1/n_A) * sum_{i in A} (KDE_A(x_i) - k0)
#
# and P(i in A, j in A | i != j) = n_A*(n_A-1) / (N*(N-1)), so:
#
#   E_perm[T] = (n_A - 1) * W / (N - 1)                           [A = B]
#
# phi = T_obs / E_perm[T], with phi = 1 for random labeling, > 1 for
# co-location, < 1 for dispersal.
#
# KEY PROPERTIES:
# - Controls for non-uniform spatial density (positions fixed under permutation)
# - Kernel normalization cancels (any constant factor in K divides out)
# - No edge correction needed (permutation null inherits the same boundary)
# - W is a single scalar computed once, making this O(N + n_cat^2 * max_n_A)


def funcn_phi(pos_by_cat: dict[str, np.ndarray], categories: list[str],
              shape: int | tuple[int, int, int],
              sigma_vox: float = SIGMA_VOX,
              min_targets: int = FUNCN_MIN_TARGETS) -> np.ndarray:
    """(n_cat, n_cat) permutation-based phi matrix.

    phi(target, source) = T_obs / E_perm[T], where E_perm is the closed-form
    expectation under label permutation without replacement (see derivation above).
    phi = 1 for random labeling, > 1 co-located, < 1 dispersed.
    """
    gx, gy, gz = _as_shape(shape)
    n = len(categories)
    k0 = k0_for(sigma_vox)

    fields = {}
    all_positions = []
    counts = {}
    for c in categories:
        p = pos_by_cat.get(c, np.empty((0, 3), np.int64))
        fields[c] = build_kde_field(p, shape, sigma_vox)
        if len(p) > 0:
            all_positions.append(p)
        counts[c] = len(p)

    kde_all = sum(fields.values())
    N = sum(counts.values())
    if N < 2:
        return np.full((n, n), np.nan, dtype=np.float64)

    all_pos = np.concatenate(all_positions, axis=0).astype(np.int64)
    apx = np.clip(all_pos[:, 0], 0, gx - 1)
    apy = np.clip(all_pos[:, 1], 0, gy - 1)
    apz = np.clip(all_pos[:, 2], 0, gz - 1)
    S = float(np.sum(kde_all[apx, apy, apz].astype(np.float64)))
    W = S / N - k0
    if W < 1e-12:
        return np.full((n, n), np.nan, dtype=np.float64)

    phi = np.full((n, n), np.nan, dtype=np.float64)
    for ti, t in enumerate(categories):
        tp = pos_by_cat.get(t, np.empty((0, 3), np.int64))
        n_A = counts[t]
        if n_A < min_targets:
            continue
        p = tp.astype(np.int64)
        px = np.clip(p[:, 0], 0, gx - 1)
        py = np.clip(p[:, 1], 0, gy - 1)
        pz = np.clip(p[:, 2], 0, gz - 1)
        for si, s in enumerate(categories):
            n_B = counts[s]
            if n_B < 1:
                continue
            kde_at_targets = fields[s][px, py, pz].astype(np.float64)
            if t == s:
                numerator = float(np.mean(kde_at_targets - k0))
                denominator = (n_A - 1) * W / (N - 1)
            else:
                numerator = float(np.mean(kde_at_targets))
                denominator = n_B * W / (N - 1)
            if denominator < 1e-12:
                continue
            phi[ti, si] = numerator / denominator
    return phi


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
