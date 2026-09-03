"""Simulated annealing placement for 2D healthy-tissue initialization.

Places fibroblasts and vasculature seeds on a flat (nz=1) grid so their spatial
clustering matches healthy pancreatic tissue FunCN neighbourhood-fraction targets
from IMC data.

Energy function::

    E = Σ_{T,S ∈ {vas,fib}} (mean_frac_sim[T,S] - target[T,S])²

where mean_frac_sim is the mean self-excluded neighbourhood fraction (same metric
as ``funcn_frac`` in ``funcn_core.py``).  Target values are the 2×2 submatrix of
the 13-category healthy IMC FunCN, row-renormalized so the two placed types sum
to 1.

The inner loop maintains KDE fields and per-cell fracs incrementally: each move
updates a small kernel patch (~121 voxels) and recomputes fracs only for the
~100 cells near the move.  Periodic FFT rebuilds correct float drift.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from .funcn_core import build_kde_field, cutoff_for, funcn_frac, k0_for


# ---------------------------------------------------------------------------
# Kernel helpers
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
# Bulk KDE / frac helpers
# ---------------------------------------------------------------------------

def _rebuild_fields(kde: list[np.ndarray], pos: np.ndarray,
                    cell_type: np.ndarray, gx: int, gy: int,
                    sigma_vox: float) -> None:
    for t in range(2):
        m = cell_type == t
        pts = np.column_stack([pos[m], np.zeros(int(m.sum()), dtype=np.int32)])
        kde[t][:] = build_kde_field(pts, (gx, gy, 1), sigma_vox)[:, :, 0].astype(
            np.float64
        )


def _compute_fracs_vec(
    kde: list[np.ndarray],
    px: np.ndarray,
    py: np.ndarray,
    types: np.ndarray,
    k0: float,
) -> np.ndarray:
    n = len(px)
    frac = np.zeros((n, 2), dtype=np.float64)
    if n == 0:
        return frac
    k0v = kde[0][px, py]
    k1v = kde[1][px, py]
    denom = k0v + k1v - k0
    valid = denom > 1e-6
    denom_safe = np.where(valid, denom, 1.0)
    frac[:, 0] = np.where(
        valid,
        np.clip((k0v - np.where(types == 0, k0, 0.0)) / denom_safe, 0, 1),
        0.0,
    )
    frac[:, 1] = np.where(
        valid,
        np.clip((k1v - np.where(types == 1, k0, 0.0)) / denom_safe, 0, 1),
        0.0,
    )
    return frac


# ---------------------------------------------------------------------------
# Target extraction
# ---------------------------------------------------------------------------

def renormalize_targets(funcn_csv: str) -> np.ndarray:
    """Extract the (Endothelial, Fibroblast) 2×2 from the 13-category FunCN
    and row-normalize so placed types sum to 1.

    Returns (2, 2) array — index 0 = vas, 1 = fib.
    """
    import pandas as pd

    df = pd.read_csv(funcn_csv, index_col=0)
    sub = df.loc[
        ["Endothelial", "Fibroblast"], ["Endothelial", "Fibroblast"]
    ].values.astype(np.float64)
    return sub / sub.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------------------
# Validation (independent FFT recompute)
# ---------------------------------------------------------------------------

def validate_placement(
    pos_fib: np.ndarray,
    pos_vas: np.ndarray,
    shape: tuple[int, int],
    sigma_vox: float = 1.5,
) -> np.ndarray:
    """Compute mean funcn_frac from scratch for placed positions.

    Returns (2, 2) array — index 0 = vas, 1 = fib.
    """
    gx, gy = shape
    cats = ["vas", "fib"]
    pos_by_cat = {
        "vas": np.column_stack(
            [pos_vas, np.zeros(len(pos_vas), dtype=np.int32)]
        ),
        "fib": np.column_stack(
            [pos_fib, np.zeros(len(pos_fib), dtype=np.int32)]
        ),
    }
    frac = funcn_frac(pos_by_cat, cats, (gx, gy, 1), sigma_vox=sigma_vox)
    result = np.zeros((2, 2))
    for ti, t in enumerate(cats):
        for si, s in enumerate(cats):
            vals = frac[(t, s)]
            result[ti, si] = float(np.mean(vals)) if len(vals) > 0 else 0.0
    return result


# ---------------------------------------------------------------------------
# Main SA
# ---------------------------------------------------------------------------

def sa_place(
    n_fib: int,
    n_vas: int,
    shape: tuple[int, int],
    target_frac: np.ndarray,
    sigma_vox: float = 1.5,
    n_steps: int = 5_000_000,
    seed: int = 0,
    log_interval: int = 100_000,
    recompute_interval: int = 100_000,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Simulated annealing to place fib + vas cells matching FunCN targets.

    Parameters
    ----------
    n_fib, n_vas : cell counts
    shape : (gx, gy) 2D grid dimensions
    target_frac : (2, 2) neighbourhood-fraction targets, indices 0=vas 1=fib
    sigma_vox : FunCN kernel width in voxels (default 1.5 = 30 µm)
    n_steps : total SA steps (including calibration)
    seed : RNG seed

    Returns
    -------
    pos_fib : (n_fib, 2) int32 voxel coordinates
    pos_vas : (n_vas, 2) int32 voxel coordinates
    info : convergence diagnostics
    """
    gx, gy = shape
    N = n_fib + n_vas
    if N == 0:
        return (np.empty((0, 2), np.int32), np.empty((0, 2), np.int32),
                {"energy_final": 0.0})

    k0 = k0_for(sigma_vox)
    R = cutoff_for(sigma_vox)
    kernel = _build_kernel_2d(sigma_vox)
    rng = np.random.default_rng(seed)
    target = np.asarray(target_frac, dtype=np.float64)

    cell_type = np.zeros(N, dtype=np.int32)
    cell_type[n_vas:] = 1
    n_type = np.array([float(n_vas), float(n_fib)])

    pos = np.column_stack(
        [rng.integers(0, gx, N), rng.integers(0, gy, N)]
    ).astype(np.int32)

    # KDE fields via FFT
    kde: list[np.ndarray] = [
        np.zeros((gx, gy), dtype=np.float64) for _ in range(2)
    ]
    _rebuild_fields(kde, pos, cell_type, gx, gy, sigma_vox)

    # Spatial index: (x, y) → [cell indices]
    cell_at: dict[tuple[int, int], list[int]] = {}
    for i in range(N):
        key = (int(pos[i, 0]), int(pos[i, 1]))
        if key not in cell_at:
            cell_at[key] = []
        cell_at[key].append(i)

    # Per-cell stored fracs + running sums
    stored_frac = _compute_fracs_vec(
        kde, pos[:, 0], pos[:, 1], cell_type, k0
    )
    sum_frac = np.zeros((2, 2), dtype=np.float64)
    for t in range(2):
        sum_frac[t] = stored_frac[cell_type == t].sum(axis=0)

    def _energy():
        return float(np.sum((sum_frac / n_type[:, None] - target) ** 2))

    E = _energy()

    offsets = [
        (dx, dy) for dx in range(-R, R + 1) for dy in range(-R, R + 1)
    ]

    # ── Calibration: first n_cal steps at T=∞ to measure |ΔE| ──
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
        j = int(rng.integers(0, N))
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

        # ── Find affected cells ──
        affected_set: set[int] = set()
        for ddx, ddy in offsets:
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

        # ── Old fracs ──
        old_frac_aff = stored_frac[affected].copy()

        # ── Update field ──
        _patch_update(kde[t_j], ox, oy, kernel, R, gx, gy, -1.0)
        _patch_update(kde[t_j], nx, ny, kernel, R, gx, gy, +1.0)

        # ── Move cell j ──
        pos[j, 0] = nx
        pos[j, 1] = ny

        # ── New fracs for affected cells ──
        aff_types = cell_type[affected]
        new_frac_aff = _compute_fracs_vec(
            kde, pos[affected, 0], pos[affected, 1], aff_types, k0
        )

        # ── ΔE via delta sums ──
        delta_frac = new_frac_aff - old_frac_aff
        delta_sum = np.zeros((2, 2), dtype=np.float64)
        for t in range(2):
            m = aff_types == t
            if m.any():
                delta_sum[t] = delta_frac[m].sum(axis=0)

        new_sum = sum_frac + delta_sum
        new_E = float(
            np.sum((new_sum / n_type[:, None] - target) ** 2)
        )
        dE = new_E - E

        # ── Accept / reject ──
        if step < n_cal:
            cal_dEs.append(abs(dE))
            accept = True
        else:
            accept = dE < 0 or rng.random() < np.exp(
                min(0.0, -dE / max(T, 1e-30))
            )

        if accept:
            n_accept += 1
            sum_frac[:] = new_sum
            E = new_E
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

        # ── End of calibration → set cooling schedule ──
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

        # ── Periodic FFT recompute ──
        if (step + 1) % recompute_interval == 0 and step >= n_cal:
            _rebuild_fields(kde, pos, cell_type, gx, gy, sigma_vox)
            stored_frac = _compute_fracs_vec(
                kde, pos[:, 0], pos[:, 1], cell_type, k0
            )
            for t in range(2):
                sum_frac[t] = stored_frac[cell_type == t].sum(axis=0)
            E = _energy()
            cell_at = {}
            for i in range(N):
                key = (int(pos[i, 0]), int(pos[i, 1]))
                if key not in cell_at:
                    cell_at[key] = []
                cell_at[key].append(i)

        # ── Log ──
        if (step + 1) % log_interval == 0:
            sa_done = max(1, step + 1 - n_cal)
            acc = n_accept / sa_done if step >= n_cal else 1.0
            elapsed = time.time() - t_start
            mf = sum_frac / n_type[:, None]
            energy_log.append(
                {
                    "step": step + 1,
                    "energy": E,
                    "temperature": T,
                    "acceptance_rate": acc,
                    "elapsed_s": elapsed,
                }
            )
            print(
                f"  step {step + 1:>8d}  E={E:.2e}  T={T:.2e}  "
                f"accept={acc:.3f}  "
                f"frac=[{mf[0,0]:.3f},{mf[0,1]:.3f},{mf[1,0]:.3f},{mf[1,1]:.3f}]  "
                f"[{elapsed:.0f}s]"
            )

    # ── Return ──
    pos_vas = pos[:n_vas].copy()
    pos_fib = pos[n_vas:].copy()
    mf = sum_frac / n_type[:, None]

    return pos_fib, pos_vas, {
        "energy_final": E,
        "mean_frac": mf.tolist(),
        "target_frac": target.tolist(),
        "acceptance_rate": n_accept / max(1, n_steps - n_cal),
        "energy_log": energy_log,
        "n_fib": n_fib,
        "n_vas": n_vas,
        "shape": shape,
        "sigma_vox": sigma_vox,
        "n_steps": n_steps,
        "elapsed_s": time.time() - t_start,
    }
