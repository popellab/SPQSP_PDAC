"""Seed the perfusion field ``V_vas(t=0)`` from placed vessel cells (contract §6.4, C6.5).

D-4 dropped vascular AGENTS but kept perfusion as a FIELD. So ``vas_phalanx`` stays a
placement category — the Potts sampler reproduces its measured arrangement (self-clustering
φ = 6.2) — and at handoff those positions are rasterized and smoothed into ``V_vas``, after
which **no vascular agent is ever instantiated**. Placement supplies `V_vas(t=0)`; the
corrected growth law supplies `∂V_vas/∂t`.

Why seed from the measurement rather than infer it
--------------------------------------------------
An earlier proposal was to derive the field from a smoothed local immune fraction, on the
reasoning that immune cells extravasate at vessels. Measured on PASS01, that does not hold:

  * local vessel fraction vs local immune fraction: **r = +0.008** (nothing). vs stroma
    +0.154, vs cancer **−0.250**.
  * median distance to the nearest CD31+ cell: immune 112.8 µm, stroma 117.0 µm, cancer
    155.4 µm — immune is only 4% closer to vessels than stroma is. The discriminating
    signal is that *cancer avoids vessels*, not that immune marks them.
  * with compositional closure properly broken, local vessel density is only **R² = 0.076**
    predictable from immune+stroma+cancer, and immune enters NEGATIVELY once the other two
    are controlled for.

The direct measurement is far stronger than a 7.6%-R² inference from it, and there is
plenty of it: 58,416 endothelial cells across 95 samples, median 371/sample, median
55 CD31+ cells/mm² — consistent with PDAC MVD of ~20-50 microvessels/mm² at 1-3 CD31+
cells per vessel cross-section.

Rasterize THEN smooth — never raw occupancy
-------------------------------------------
At realistic density a 2 mm × 2 mm × 0.24 mm domain holds ~1,850 vessel cells over 120,000
voxels, so **only ~1.5% of voxels contain one**. Raw per-voxel occupancy is far too sparse
and grainy to drive recruitment; smoothed at σ ≈ 60-100 µm it becomes a continuous field
with ~5-10× contrast between vessel-rich and vessel-poor regions. ``sigma_um`` is exposed
because the right coverage is not obvious a priori and between-sample vessel density spans
147× (2.4 to 356 cells/mm²).
"""
from __future__ import annotations

import numpy as np

from .funcn_core import build_kde_field

VOXEL_UM = 20.0
VESSEL_CAT = "vas_phalanx"

# Endothelial cell volume, and the multiplier from endothelial-cell volume to total vessel
# volume (a capillary is mostly lumen; wall/total ≈ 0.3-0.5). Used only to put the field on a
# physical scale — see ``seed_vvas`` on why the scale matters.
CELL_VOLUME_UM3 = 1600.0
LUMEN_FACTOR = 2.5


def vessel_volume_fraction(n_vessel_cells: int, shape: tuple[int, int, int],
                           voxel_um: float = VOXEL_UM,
                           cell_volume_um3: float = CELL_VOLUME_UM3,
                           lumen_factor: float = LUMEN_FACTOR) -> float:
    """Domain-mean vascular volume fraction implied by the placed vessel cells."""
    vol = np.prod(shape) * voxel_um ** 3
    return float(n_vessel_cells * cell_volume_um3 * lumen_factor / vol)


def seed_vvas(pos: np.ndarray, cat: np.ndarray, shape: tuple[int, int, int],
              sigma_um: float = 60.0, voxel_um: float = VOXEL_UM,
              vessel_cat: str = VESSEL_CAT, normalize: str = "physical",
              cell_volume_um3: float = CELL_VOLUME_UM3,
              lumen_factor: float = LUMEN_FACTOR,
              floor_frac: float = 0.05) -> tuple[np.ndarray, dict]:
    """Placed configuration → ``V_vas(t=0)`` field + diagnostics.

    ``floor_frac`` — voxels below this fraction of the field maximum are set to EXACTLY zero.
    This is not cosmetic. A Gaussian KDE is strictly positive everywhere, and the ``V_vas``
    growth law is logistic in ``V`` (``pde_integration.cu``, 2026-08-08), so ``V = 0`` is a
    fixed point but any positive value — however tiny — eventually grows to the equilibrium.
    Without a floor the seeded geometry survives only as a slow transient; with one, genuinely
    avascular tissue stays avascular for the whole run, which is what makes tumour nests
    unperfused and gives ``φ_chemo`` something to vary over. At σ=80 µm, ``floor_frac`` = 0.05
    zeroes ~47% of the domain. Set to 0 to disable.

    ``normalize``:
      * ``"physical"`` (default) — rescale so the field MEAN equals the vascular volume
        fraction implied by the placed vessel count. This is the scale ``V_vas`` is
        *supposed* to be on: the Krogh coefficient in ``source_o2_from_vvas_kernel`` uses
        ``Lv = voxel_volume/(π·Rc²)``, the length of an Rc-radius cylinder filling the whole
        voxel, so ``V_vas = 1`` means a 100%-vessel voxel. Real tissue is ~1-5%.
        ⚠️ The current `Hill(VEGFA)` formulation ranges over [0,1) and therefore runs the
        field ~20× too high on this scale. Switching to a physically-scaled seed will cut
        O2 delivery and recruitment flux by that factor unless ``KvLv_base`` and
        ``PARAM_ENTRY_ADHESION_SCALE`` are re-derived at the same time. Do not change one
        without the other.
      * ``"unit"`` — rescale to max 1. Preserves the current code's numeric range so the
        geometry fix can be tested in isolation from the scale recalibration.

    Returns ``(field, info)``; ``info`` carries coverage and contrast so the smoothing width
    can be chosen against a stated target rather than by eye.
    """
    if normalize not in ("physical", "unit"):
        raise ValueError(f"unknown normalize={normalize!r}")
    m = np.asarray(cat) == vessel_cat
    n_vessel = int(m.sum())
    gx, gy, gz = shape
    if n_vessel == 0:
        return np.zeros(shape, dtype=np.float32), {"n_vessel_cells": 0, "coverage": 0.0}

    sigma_vox = sigma_um / voxel_um
    raw = build_kde_field(np.asarray(pos)[m], shape, sigma_vox)

    # Floor BEFORE normalizing, so the physical rescaling puts the surviving (perfused) tissue
    # on the right scale rather than spreading the vessel budget over voxels we then discard.
    if floor_frac > 0 and raw.max() > 0:
        raw = np.where(raw < floor_frac * raw.max(), 0.0, raw)

    mean_raw = float(raw.mean())
    if normalize == "physical":
        target = vessel_volume_fraction(n_vessel, shape, voxel_um,
                                        cell_volume_um3, lumen_factor)
        field = raw * (target / mean_raw) if mean_raw > 0 else raw
    else:
        mx = float(raw.max())
        field = raw / mx if mx > 0 else raw

    nz = field[field > 0]
    info = {
        "n_vessel_cells": n_vessel,
        "sigma_um": sigma_um,
        "floor_frac": floor_frac,
        # The avascular fraction — the whole point of the floor. Under the logistic growth law
        # these voxels stay at V_vas = 0 permanently.
        "zero_frac": float((field <= 0).mean()),
        "mean": float(field.mean()),
        "max": float(field.max()),
        # Coverage: the fraction of voxels receiving a non-trivial share of the field.
        # Raw occupancy would put this at ~1.5%; the point of smoothing is to raise it.
        "coverage_1pct_of_max": float((field > 0.01 * field.max()).mean()),
        "coverage_10pct_of_max": float((field > 0.10 * field.max()).mean()),
        # Contrast: p90/p50 over occupied voxels. Too low and the field is uniform (φ_chemo
        # → 1, the smooth-field failure mode §6.2 point 5 warns about); too high and it is
        # effectively the raw point set again.
        "contrast_p90_p50": float(np.quantile(nz, 0.9) / max(np.quantile(nz, 0.5), 1e-30))
        if len(nz) else np.nan,
        "volume_fraction": vessel_volume_fraction(n_vessel, shape, voxel_um,
                                                  cell_volume_um3, lumen_factor),
    }
    return field.astype(np.float32), info
