"""Multi-category SA placement for tumor microenvironment initialization.

Extends ``seeding_sa.py`` from 2 categories (vas, fib) to ~12 categories.
Cancer cells form a compact disc (fixed during SA); immune and activated stroma
cells move via SA to match per-patient IMC phi targets.

Energy function (phi-based)::

    E = Σ_{T,S ∈ cats} w[T,S] * (phi_achieved[T,S] - phi_target[T,S])²

where ``phi_achieved`` is the permutation-null co-location metric:
    phi(A,B) = T_obs / E_perm[T]
    T_obs = (1/n_A) Σ_{a∈A} KDE_B(a)    [self-excluded for A=B]
    E_perm = n_B · W / (N-1)              [or (n_A-1)·W/(N-1) for A=B]
    W = (1/N) Σ_i KDE_ALL(x_i) - k0      [mean self-excluded total density]

The SA tracks raw KDE sums (sum_kde[T,S] = Σ_{i of type T} KDE_S(x_i))
and computes phi from those and the global W. This avoids the local-vs-global
normalization mismatch that occurs when optimizing neighbourhood fractions:
with a spatially concentrated cancer disc, locally-normalized fracs can
satisfy avoidance targets while the globally-normalized phi shows clustering.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .funcn_core import build_kde_field, cutoff_for, funcn_frac, funcn_phi, k0_for


# ---------------------------------------------------------------------------
# Kernel helpers (duplicated from seeding_sa for self-containment)
# ---------------------------------------------------------------------------

def _build_kernel_2d(sigma_vox: float) -> np.ndarray:
    R = cutoff_for(sigma_vox)
    ax = np.arange(-R, R + 1, dtype=np.float64)
    kx, ky = np.meshgrid(ax, ax, indexing="ij")
    return (1.0 / (2 * np.pi * sigma_vox ** 2)) * np.exp(
        -(kx ** 2 + ky ** 2) / (2 * sigma_vox ** 2)
    )


def _patch_update(field: np.ndarray, cx: int, cy: int,
                  kernel: np.ndarray, R: int, gx: int, gy: int,
                  sign: float = 1.0) -> None:
    x0 = max(0, cx - R)
    x1 = min(gx, cx + R + 1)
    y0 = max(0, cy - R)
    y1 = min(gy, cy + R + 1)
    kx0 = x0 - (cx - R)
    kx1 = kernel.shape[0] - ((cx + R + 1) - x1)
    ky0 = y0 - (cy - R)
    ky1 = kernel.shape[1] - ((cy + R + 1) - y1)
    field[x0:x1, y0:y1] += sign * kernel[kx0:kx1, ky0:ky1]


# ---------------------------------------------------------------------------
# Phi → frac target conversion (kept for diagnostics/backwards compat)
# ---------------------------------------------------------------------------

def phi_to_frac(phi_matrix: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Convert a phi matrix to a funcn_frac target matrix.

    Under uniform total density, the relationship is:

        frac[A,B] ≈ phi[A,B] * n_B / (N - 1)          for A ≠ B
        frac[A,A] ≈ phi[A,A] * (n_A - 1) / (N - 1)    for A = A

    NOTE: This conversion is approximate and breaks when density is
    non-uniform (e.g. cancer disc). The SA should target phi directly.
    """
    n = len(counts)
    N = int(counts.sum())
    frac = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if np.isnan(phi_matrix[i, j]):
                continue
            if i == j:
                frac[i, j] = phi_matrix[i, j] * (counts[i] - 1) / (N - 1)
            else:
                frac[i, j] = phi_matrix[i, j] * counts[j] / (N - 1)
    return frac


# ---------------------------------------------------------------------------
# Bulk KDE helpers (N-category generalization)
# ---------------------------------------------------------------------------

def _rebuild_fields(kde: list[np.ndarray], pos: np.ndarray,
                    cell_type: np.ndarray, n_cats: int,
                    gx: int, gy: int, sigma_vox: float) -> None:
    for t in range(n_cats):
        m = cell_type == t
        pts = np.column_stack([pos[m], np.zeros(int(m.sum()), dtype=np.int32)])
        kde[t][:] = build_kde_field(pts, (gx, gy, 1), sigma_vox)[:, :, 0].astype(
            np.float64
        )


def _compute_kde_vec(
    kde: list[np.ndarray],
    px: np.ndarray,
    py: np.ndarray,
    n_cats: int,
) -> np.ndarray:
    """Per-cell raw KDE values for N categories. Returns (n_cells, n_cats)."""
    n = len(px)
    if n == 0:
        return np.zeros((0, n_cats), dtype=np.float64)
    return np.column_stack([kde[t][px, py] for t in range(n_cats)]).astype(np.float64)


def _compute_fracs_vec(
    kde: list[np.ndarray],
    px: np.ndarray,
    py: np.ndarray,
    types: np.ndarray,
    n_cats: int,
    k0: float,
) -> np.ndarray:
    """Per-cell neighbourhood fraction for N categories (kept for diagnostics)."""
    n = len(px)
    frac = np.zeros((n, n_cats), dtype=np.float64)
    if n == 0:
        return frac
    vals = np.column_stack([kde[t][px, py] for t in range(n_cats)])
    self_k0 = np.zeros((n, n_cats), dtype=np.float64)
    for t in range(n_cats):
        self_k0[types == t, t] = k0
    denom = vals.sum(axis=1) - k0
    valid = denom > 1e-6
    denom_safe = np.where(valid, denom, 1.0)[:, None]
    frac = np.where(
        valid[:, None],
        np.clip((vals - self_k0) / denom_safe, 0, 1),
        0.0,
    )
    return frac


# ---------------------------------------------------------------------------
# Cancer disc placement
# ---------------------------------------------------------------------------

def place_cancer_disc(
    n_cancer: int,
    center: tuple[int, int],
    gx: int, gy: int,
    rng: np.random.Generator,
    prog_frac: float = 0.9,
) -> tuple[np.ndarray, np.ndarray]:
    """Place cancer cells in a compact 2D disc.

    Returns (positions (N,2) int32, type_indices (N,) int32) where type indices
    correspond to the cancer categories in the category list.
    """
    cx, cy = center
    radius = np.sqrt(n_cancer / np.pi)

    pos_list = []
    r_int = int(np.ceil(radius)) + 1
    for dx in range(-r_int, r_int + 1):
        for dy in range(-r_int, r_int + 1):
            if dx * dx + dy * dy <= radius * radius:
                px = cx + dx
                py = cy + dy
                if 0 <= px < gx and 0 <= py < gy:
                    pos_list.append((px, py))

    if len(pos_list) < n_cancer:
        extra_r = r_int + 1
        while len(pos_list) < n_cancer:
            extra_r += 1
            pos_list = []
            for dx in range(-extra_r, extra_r + 1):
                for dy in range(-extra_r, extra_r + 1):
                    if dx * dx + dy * dy <= extra_r * extra_r:
                        px = cx + dx
                        py = cy + dy
                        if 0 <= px < gx and 0 <= py < gy:
                            pos_list.append((px, py))

    if len(pos_list) > n_cancer:
        dists = [(dx - cx) ** 2 + (dy - cy) ** 2 for dx, dy in pos_list]
        order = np.argsort(dists)[:n_cancer]
        pos_list = [pos_list[i] for i in order]

    return np.array(pos_list, dtype=np.int32)


# ---------------------------------------------------------------------------
# Phi-based energy (the correct global-normalization metric)
# ---------------------------------------------------------------------------

def _phi_from_sum_kde(
    sum_kde: np.ndarray,
    n_type: np.ndarray,
    N: int,
    k0: float,
) -> np.ndarray:
    """Compute phi matrix from raw KDE sums.

    sum_kde[t, s] = Σ_{i of type t} KDE_s(x_i)
    """
    n_cats = len(n_type)
    S = sum_kde.sum()
    W = S / N - k0
    if W < 1e-12:
        return np.full((n_cats, n_cats), np.nan)

    mean_kde = sum_kde / np.maximum(n_type[:, None], 1.0)

    # T_obs: subtract k0 on diagonal (self-exclusion)
    T_obs = mean_kde.copy()
    for t in range(n_cats):
        T_obs[t, t] -= k0

    # E_perm: n_s * W / (N-1), except diagonal: (n_t-1) * W / (N-1)
    W_Nm1 = W / (N - 1)
    E_perm = np.tile(n_type, (n_cats, 1)).astype(np.float64) * W_Nm1
    for t in range(n_cats):
        E_perm[t, t] = (n_type[t] - 1) * W_Nm1

    safe = np.maximum(E_perm, 1e-12)
    return T_obs / safe


# ---------------------------------------------------------------------------
# Main SA
# ---------------------------------------------------------------------------

def sa_place_tumor(
    counts: dict[str, int],
    shape: tuple[int, int],
    target_frac: np.ndarray,
    categories: list[str],
    fixed_cats: frozenset[str],
    fixed_positions: dict[str, np.ndarray],
    sigma_vox: float = 2.5,
    n_steps: int = 5_000_000,
    seed: int = 0,
    log_interval: int = 100_000,
    recompute_interval: int = 100_000,
    weight_matrix: np.ndarray | None = None,
    target_phi: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Multi-category SA for tumor microenvironment placement.

    Parameters
    ----------
    counts : {category: n_cells} for all categories in the SA
    shape : (gx, gy) 2D grid dimensions
    target_frac : (n_cat, n_cat) target funcn_frac matrix (NaN = ignore pair).
        Ignored when ``target_phi`` is provided.
    categories : ordered list of category names matching target rows/cols
    fixed_cats : categories whose positions are fixed (e.g. cancer)
    fixed_positions : {category: (N,2) int32 positions} for fixed categories
    sigma_vox : FunCN kernel width in voxels
    n_steps : total SA steps
    seed : RNG seed
    weight_matrix : (n_cat, n_cat) weights for each pair (default: sqrt(n_T*n_S))
    target_phi : (n_cat, n_cat) phi targets. When provided, the SA uses
        phi-based energy (globally-normalized density ratios) instead of
        frac-based energy (locally-normalized fractions). This is the correct
        metric when cancer cells form a concentrated disc.

    Returns
    -------
    pos_by_cat : {category: (N,2) int32 positions}
    info : convergence diagnostics
    """
    gx, gy = shape
    n_cats = len(categories)
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    use_phi = target_phi is not None

    n_per_cat = np.array([counts.get(c, 0) for c in categories], dtype=np.int64)
    N = int(n_per_cat.sum())
    if N == 0:
        return {c: np.empty((0, 2), np.int32) for c in categories}, {"energy_final": 0.0}

    k0 = k0_for(sigma_vox)
    R = cutoff_for(sigma_vox)
    kernel = _build_kernel_2d(sigma_vox)
    rng = np.random.default_rng(seed)

    # Select target and build mask
    if use_phi:
        target = np.asarray(target_phi, dtype=np.float64)
    else:
        target = np.asarray(target_frac, dtype=np.float64)

    if weight_matrix is not None:
        weights = np.asarray(weight_matrix, dtype=np.float64)
    else:
        weights = np.ones((n_cats, n_cats), dtype=np.float64)
        for i in range(n_cats):
            for j in range(n_cats):
                if not np.isnan(target[i, j]):
                    weights[i, j] = np.sqrt(max(1, n_per_cat[i]) * max(1, n_per_cat[j]))
    mask_valid = ~np.isnan(target)
    weights[~mask_valid] = 0.0
    target_safe = np.where(mask_valid, target, 0.0)

    cell_type = np.empty(N, dtype=np.int32)
    pos = np.empty((N, 2), dtype=np.int32)
    is_fixed = np.zeros(N, dtype=bool)

    # Compute initial placement center from fixed positions
    all_fixed = [fp for c, fp in fixed_positions.items() if c in fixed_cats and len(fp) > 0]
    if all_fixed:
        center = np.concatenate(all_fixed).mean(axis=0)
        init_cx, init_cy = int(center[0]), int(center[1])
        fixed_all = np.concatenate(all_fixed)
        max_dist = np.sqrt(np.max((fixed_all[:, 0] - init_cx) ** 2 +
                                   (fixed_all[:, 1] - init_cy) ** 2))
        init_radius = max(int(2 * max_dist), 20)
    else:
        init_cx, init_cy = gx // 2, gy // 2
        init_radius = max(gx, gy) // 4

    offset = 0
    for ci, cat in enumerate(categories):
        nc = int(n_per_cat[ci])
        if nc == 0:
            continue
        cell_type[offset:offset + nc] = ci
        if cat in fixed_cats and cat in fixed_positions:
            fp = fixed_positions[cat][:nc]
            pos[offset:offset + len(fp)] = fp
            if len(fp) < nc:
                extra = rng.integers(0, [gx, gy], size=(nc - len(fp), 2)).astype(np.int32)
                pos[offset + len(fp):offset + nc] = extra
            is_fixed[offset:offset + nc] = True
        else:
            angles = rng.uniform(0, 2 * np.pi, nc)
            radii = init_radius * np.sqrt(rng.uniform(0, 1, nc))
            px = np.clip(init_cx + (radii * np.cos(angles)).astype(np.int32), 0, gx - 1)
            py = np.clip(init_cy + (radii * np.sin(angles)).astype(np.int32), 0, gy - 1)
            pos[offset:offset + nc, 0] = px
            pos[offset:offset + nc, 1] = py
        offset += nc

    movable_indices = np.where(~is_fixed)[0]
    n_movable = len(movable_indices)
    if n_movable == 0:
        pos_by_cat = {}
        for ci, cat in enumerate(categories):
            m = cell_type == ci
            pos_by_cat[cat] = pos[m].copy()
        return pos_by_cat, {"energy_final": 0.0, "n_movable": 0}

    kde = [np.zeros((gx, gy), dtype=np.float64) for _ in range(n_cats)]
    _rebuild_fields(kde, pos, cell_type, n_cats, gx, gy, sigma_vox)

    cell_at: dict[tuple[int, int], list[int]] = {}
    for i in range(N):
        key = (int(pos[i, 0]), int(pos[i, 1]))
        if key not in cell_at:
            cell_at[key] = []
        cell_at[key].append(i)

    n_type = np.maximum(n_per_cat.astype(np.float64), 1.0)

    if use_phi:
        # Phi-based energy: track raw KDE sums
        stored_kde = _compute_kde_vec(kde, pos[:, 0], pos[:, 1], n_cats)
        sum_kde = np.zeros((n_cats, n_cats), dtype=np.float64)
        for t in range(n_cats):
            m = cell_type == t
            if m.any():
                sum_kde[t] = stored_kde[m].sum(axis=0)

        def _energy():
            phi_now = _phi_from_sum_kde(sum_kde, n_type, N, k0)
            diff = np.where(mask_valid, np.nan_to_num(phi_now) - target_safe, 0.0)
            return float(np.sum(weights * diff * diff))
    else:
        # Frac-based energy (legacy): track normalized fracs
        stored_frac = _compute_fracs_vec(kde, pos[:, 0], pos[:, 1], cell_type, n_cats, k0)
        sum_frac = np.zeros((n_cats, n_cats), dtype=np.float64)
        for t in range(n_cats):
            m = cell_type == t
            if m.any():
                sum_frac[t] = stored_frac[m].sum(axis=0)

        def _energy():
            mean_frac = sum_frac / n_type[:, None]
            return float(np.sum(weights * (mean_frac - target_safe) ** 2))

    E = _energy()

    offsets_list = [
        (dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1)
    ]

    n_cal = min(1000, n_steps // 10)
    max_radius = max(gx, gy) // 4
    min_radius = R
    T0 = 1.0
    alpha = 1.0
    T = 1e30

    n_accept = 0
    cal_dEs: list[float] = []
    energy_log: list[dict] = []
    t_start = time.time()

    for step in range(n_steps):
        j = int(movable_indices[rng.integers(0, n_movable)])
        t_j = int(cell_type[j])
        ox = int(pos[j, 0])
        oy = int(pos[j, 1])

        if step < n_cal:
            radius = max_radius
        else:
            frac_T = min(1.0, T / T0) if T0 > 0 else 0.0
            radius = max(
                min_radius,
                int(min_radius + (max_radius - min_radius) * frac_T),
            )

        dx = int(rng.integers(-radius, radius + 1))
        dy = int(rng.integers(-radius, radius + 1))
        nx = max(0, min(gx - 1, ox + dx))
        ny = max(0, min(gy - 1, oy + dy))

        if nx == ox and ny == oy:
            if step >= n_cal:
                T *= alpha
            continue

        affected_set: set[int] = set()
        for ddx, ddy in offsets_list:
            for bx, by in ((ox, oy), (nx, ny)):
                cx = bx + ddx
                cy = by + ddy
                if 0 <= cx < gx and 0 <= cy < gy:
                    lst = cell_at.get((cx, cy))
                    if lst:
                        affected_set.update(lst)

        if not affected_set:
            if step >= n_cal:
                T *= alpha
            continue

        affected = np.fromiter(affected_set, dtype=np.int64)
        aff_types = cell_type[affected]

        if use_phi:
            old_kde_aff = stored_kde[affected].copy()
        else:
            old_frac_aff = stored_frac[affected].copy()

        _patch_update(kde[t_j], ox, oy, kernel, R, gx, gy, -1.0)
        _patch_update(kde[t_j], nx, ny, kernel, R, gx, gy, +1.0)

        pos[j, 0] = nx
        pos[j, 1] = ny

        if use_phi:
            new_kde_aff = _compute_kde_vec(
                kde, pos[affected, 0], pos[affected, 1], n_cats
            )
            delta_kde = new_kde_aff - old_kde_aff
            delta_sum = np.zeros((n_cats, n_cats), dtype=np.float64)
            for t in range(n_cats):
                m = aff_types == t
                if m.any():
                    delta_sum[t] = delta_kde[m].sum(axis=0)

            new_sum_kde = sum_kde + delta_sum
            phi_now = _phi_from_sum_kde(new_sum_kde, n_type, N, k0)
            diff = np.where(mask_valid, np.nan_to_num(phi_now) - target_safe, 0.0)
            new_E = float(np.sum(weights * diff * diff))
        else:
            new_frac_aff = _compute_fracs_vec(
                kde, pos[affected, 0], pos[affected, 1], aff_types, n_cats, k0
            )
            delta_frac = new_frac_aff - old_frac_aff
            delta_sum = np.zeros((n_cats, n_cats), dtype=np.float64)
            for t in range(n_cats):
                m = aff_types == t
                if m.any():
                    delta_sum[t] = delta_frac[m].sum(axis=0)

            new_sum = sum_frac + delta_sum
            new_mean = new_sum / n_type[:, None]
            new_E = float(np.sum(weights * (new_mean - target_safe) ** 2))

        dE = new_E - E

        if step < n_cal:
            cal_dEs.append(abs(dE))
            accept = True
        else:
            accept = dE < 0 or rng.random() < np.exp(
                min(0.0, -dE / max(T, 1e-30))
            )

        if accept:
            n_accept += 1
            E = new_E
            if use_phi:
                sum_kde[:] = new_sum_kde
                stored_kde[affected] = new_kde_aff
            else:
                sum_frac[:] = new_sum
                stored_frac[affected] = new_frac_aff
            cell_at[(ox, oy)].remove(j)
            if not cell_at[(ox, oy)]:
                del cell_at[(ox, oy)]
            if (nx, ny) not in cell_at:
                cell_at[(nx, ny)] = []
            cell_at[(nx, ny)].append(j)
        else:
            _patch_update(kde[t_j], nx, ny, kernel, R, gx, gy, -1.0)
            _patch_update(kde[t_j], ox, oy, kernel, R, gx, gy, +1.0)
            pos[j, 0] = ox
            pos[j, 1] = oy

        if step == n_cal - 1 and cal_dEs:
            med = float(np.median(cal_dEs))
            T0 = max(5.0 * med, 1e-20)
            T_final = 0.001 * T0
            remaining = n_steps - n_cal
            alpha = (T_final / T0) ** (1.0 / max(remaining, 1))
            T = T0
            n_accept = 0

        if step >= n_cal:
            T *= alpha

        if (step + 1) % recompute_interval == 0 and step >= n_cal:
            _rebuild_fields(kde, pos, cell_type, n_cats, gx, gy, sigma_vox)
            if use_phi:
                stored_kde = _compute_kde_vec(kde, pos[:, 0], pos[:, 1], n_cats)
                for t in range(n_cats):
                    m = cell_type == t
                    if m.any():
                        sum_kde[t] = stored_kde[m].sum(axis=0)
                    else:
                        sum_kde[t] = 0.0
            else:
                stored_frac = _compute_fracs_vec(
                    kde, pos[:, 0], pos[:, 1], cell_type, n_cats, k0
                )
                for t in range(n_cats):
                    m = cell_type == t
                    if m.any():
                        sum_frac[t] = stored_frac[m].sum(axis=0)
                    else:
                        sum_frac[t] = 0.0
            E = _energy()
            cell_at = {}
            for i in range(N):
                key = (int(pos[i, 0]), int(pos[i, 1]))
                if key not in cell_at:
                    cell_at[key] = []
                cell_at[key].append(i)

        if (step + 1) % log_interval == 0:
            sa_done = max(1, step + 1 - n_cal)
            acc = n_accept / sa_done if step >= n_cal else 1.0
            elapsed = time.time() - t_start
            log_entry = {
                "step": step + 1,
                "energy": E,
                "temperature": T,
                "acceptance_rate": acc,
                "elapsed_s": elapsed,
            }
            energy_log.append(log_entry)
            if use_phi:
                phi_now = _phi_from_sum_kde(sum_kde, n_type, N, k0)
                key_pairs_idx = [(3, 0), (4, 0), (2, 0)]  # M1×C1, M2×C1, CD8×C1
                phi_str = "  ".join(
                    f"{phi_now[t,s]:.2f}" for t, s in key_pairs_idx
                    if t < n_cats and s < n_cats
                )
                print(
                    f"  step {step + 1:>8d}  E={E:.2e}  T={T:.2e}  "
                    f"accept={acc:.3f}  phi[M1,M2,CD8×C1]=[{phi_str}]  [{elapsed:.0f}s]"
                )
            else:
                print(
                    f"  step {step + 1:>8d}  E={E:.2e}  T={T:.2e}  "
                    f"accept={acc:.3f}  [{elapsed:.0f}s]"
                )

    pos_by_cat: dict[str, np.ndarray] = {}
    for ci, cat in enumerate(categories):
        m = cell_type == ci
        pos_by_cat[cat] = pos[m].copy()

    info: dict[str, Any] = {
        "energy_final": E,
        "acceptance_rate": n_accept / max(1, n_steps - n_cal),
        "energy_log": energy_log,
        "n_total": N,
        "n_movable": n_movable,
        "counts": {c: int(n_per_cat[i]) for i, c in enumerate(categories)},
        "shape": shape,
        "sigma_vox": sigma_vox,
        "n_steps": n_steps,
        "elapsed_s": time.time() - t_start,
        "energy_mode": "phi" if use_phi else "frac",
    }

    if use_phi:
        phi_final = _phi_from_sum_kde(sum_kde, n_type, N, k0)
        info["phi_achieved"] = phi_final.tolist()
        info["target_phi"] = target_safe.tolist()
    else:
        mf = sum_frac / n_type[:, None]
        info["mean_frac"] = mf.tolist()
        info["target_frac"] = target_safe.tolist()

    return pos_by_cat, info
