"""Seed the ECM density field ``ECM(t=0)`` from placed myCAFs (§C6.2 "ECM field state").

Why this exists
---------------
Placement replaces the presim, and the presim was the only thing that ever built the ECM
field: both procedural ECM initializers (``initialize_ecm_to_saturation``,
``initialize_ecm_floor_uniform``) are dead code — they are called from ``set_internal_params``
at ``main.cu:954`` but ``d_ecm_density`` is not allocated until ``initialize_pde_solver`` at
``:1020``, and both early-return on a null pointer. So a placement run with the presim skipped
starts at ECM = 0 and stays there.

That is not a cosmetic gap. Measured on a 21³ placement run with ``--presim-steps 0``, ECM
reached only 136 nmol/mL after 8 steps — far below the 700 nmol/mL vessel-collapse threshold —
so desmoplastic compression never fired, ``V_vas`` climbed 0.0074 → 0.0213 unopposed, and the
O2 coefficient of variation collapsed from 0.28 to 0.057. The perfusion heterogeneity that
placement exists to create was being erased within a few steps by the absence of ECM.

Built from myCAF alone, deliberately
------------------------------------
``fib_build_density_field`` (``agents/fibroblast.cuh``) opens with
``if (cs != FIB_MYCAF) return`` — **only myCAF deposits ECM** in this model. So the seed is
built from placed ``fib_myCAF`` only, and iCAF/apCAF/quiescent contribute nothing. That is the
model's own rule rather than an invented per-state weighting, which keeps the seed consistent
with what the running deposition law would produce from the same cells.

The smoothing kernel matches the agent kernel too: a 3D Gaussian with variance
``PARAM_FIB_ECM_VARIANCE`` (2.0 voxels²) truncated at ``PARAM_FIB_ECM_RADIUS`` (3 voxels), so
seeded and grown ECM have the same spatial texture.

Scale
-----
The absolute level is set by ``target_p90`` rather than derived, because the deposition law's
own fixed point is the pathology we are trying to avoid: with ``PARAM_ECM_MMP_DEGRADE_RATE``
zeroed, ECM's only sink is a 0.007/day passive decay, so given enough steps the field goes to
its cap everywhere (measured 985/1000 at cv 0.0053 — spatially uniform, and every voxel above
the collapse threshold, which makes the occlusion term structureless). Anchoring the p90 near
but below the cap keeps dense myCAF stroma compressive while leaving cancer nests — which have
no myCAF and therefore no ECM — genuinely uncompressed.
"""
from __future__ import annotations

import numpy as np

from .funcn_core import build_kde_field

VOXEL_UM = 20.0
MYCAF_CAT = "fib_myCAF"

# Mirrors of the agent-side constants — keep in sync with the XML.
ECM_VARIANCE_VOX = 2.0     # PARAM_FIB_ECM_VARIANCE  [voxels²]
ECM_DENSITY_CAP = 1000.0   # PARAM_ECM_DENSITY_CAP   [nmol/mL]
COLLAPSE_THRESHOLD = 700.0  # PARAM_VAS_COLLAPSE_THRESHOLD
RECOVERY_THRESHOLD = 600.0  # PARAM_VAS_RECOVERY_THRESHOLD


def seed_ecm(pos: np.ndarray, cat: np.ndarray, shape: tuple[int, int, int],
             target_p90: float = 900.0, mycaf_cat: str = MYCAF_CAT,
             variance_vox: float = ECM_VARIANCE_VOX,
             density_cap: float = ECM_DENSITY_CAP) -> tuple[np.ndarray, dict]:
    """Placed configuration → ``ECM(t=0)`` [nmol/mL] + diagnostics.

    ``target_p90`` — the field is rescaled so its 90th percentile lands here. Default 900 sits
    above the 700 collapse threshold, so the top decile of stroma actively occludes vessels,
    and below the 1000 cap, so deposition still has headroom (the law's ``(1 - density/cap)``
    factor would otherwise be zero at t=0).

    Returned diagnostics carry the fractions above the collapse threshold and below the
    recovery threshold, because those two numbers — not the mean — decide whether the
    compression term has any spatial structure at all.
    """
    m = np.asarray(cat) == mycaf_cat
    n_mycaf = int(m.sum())
    if n_mycaf == 0:
        return np.zeros(shape, dtype=np.float32), {
            "n_mycaf": 0, "mean": 0.0, "frac_above_collapse": 0.0,
            "frac_below_recovery": 1.0,
        }

    sigma_vox = float(np.sqrt(variance_vox))
    raw = build_kde_field(np.asarray(pos)[m], shape, sigma_vox)

    p90 = float(np.quantile(raw, 0.90))
    if p90 > 0:
        field = raw * (target_p90 / p90)
    else:
        # Degenerate case: >90% of the domain has zero myCAF signal. Fall back to the max so
        # the field is still on a sane scale rather than being blown up by a near-zero divisor.
        mx = float(raw.max())
        field = raw * (target_p90 / mx) if mx > 0 else raw
    np.clip(field, 0.0, density_cap, out=field)

    info = {
        "n_mycaf": n_mycaf,
        "sigma_vox": sigma_vox,
        "target_p90": target_p90,
        "mean": float(field.mean()),
        "p50": float(np.quantile(field, 0.5)),
        "p90": float(np.quantile(field, 0.9)),
        "max": float(field.max()),
        # The two numbers that decide whether occlusion has structure. At the end of a long
        # presim these read 1.0 and 0.0 — every voxel collapsing, none recovering, no contrast.
        "frac_above_collapse": float((field > COLLAPSE_THRESHOLD).mean()),
        "frac_below_recovery": float((field < RECOVERY_THRESHOLD).mean()),
        "frac_zero": float((field <= 0).mean()),
    }
    return field.astype(np.float32), info
