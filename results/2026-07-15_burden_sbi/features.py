#!/usr/bin/env python
"""
Feature extraction for the amortized burden NPE.

The bank stores rel-SLD and rel-CA19-9 on a canonical 76-pt grid (0:14:1050).
A patient is observed as rel-SLD at sld_days and rel-CA19-9 at ca_days over a
1st-line window W. To amortize, we map any trajectory -> a FIXED feature vector.

v1 (this file) = fixed-grid features: interpolate rel-SLD and rel-log-CA19-9 onto
a fixed output grid OUT_GRID; concatenate -> x (2*len(OUT_GRID)). Valid when the
patient window covers OUT_GRID (full-window patients, e.g. PM-036). For shorter
windows we will add window-context / masking in v2.

Both the bank and a real patient go through the SAME feature function, so the NPE
sees comparable x at train and inference.
"""
import numpy as np

# monthly-ish fixed grid over the 1st-line horizon; extend/trim in v2 per window.
OUT_GRID = np.arange(0.0, 1008.0+1e-6, 56.0)   # 0,56,...,1008 -> 19 pts
NF = 2*len(OUT_GRID)

# v1.1 feature clipping: tame eradication-sim pathologies (C_total->0 gives
# SLD_rel~3e-4 and log(CA_rel)~-69) that wreck NPE z-scoring, WITHOUT touching
# real data (real SLD_rel in ~[0.1,2], real log(CA_rel) in ~[-7,+5]).
SLD_CLIP   = (0.02, 5.0)      # rel diameter
LOGCA_CLIP = (-12.0, 6.0)     # log rel CA19-9  (e^-12=6e-6 .. e^6=400x baseline)

def _feat(sld, cal):
    sld = np.clip(sld, *SLD_CLIP)
    cal = np.clip(cal, *LOGCA_CLIP)
    return np.concatenate([sld, cal]).astype(np.float32)

def features_from_grid(sld_rel76, ca_rel76, bank_grid):
    """Bank trajectory (on 76-pt bank_grid) -> feature vector on OUT_GRID."""
    sld = np.interp(OUT_GRID, bank_grid, sld_rel76)
    cal = np.log(np.clip(np.interp(OUT_GRID, bank_grid, ca_rel76), 1e-30, None))
    return _feat(sld, cal)

def features_from_patient(sld_days, sld_rel, ca_days, ca_rel_):
    """Patient observations -> feature vector on OUT_GRID (linear interp/extrap-hold)."""
    sld = np.interp(OUT_GRID, sld_days, sld_rel)                 # holds ends beyond range
    cal = np.log(np.clip(np.interp(OUT_GRID, ca_days, ca_rel_), 1e-30, None))
    return _feat(sld, cal)

# ---------------------------------------------------------------------------
# v2: window-relative features (amortize over 1st-line window length W).
# Grid is 0->W (19 pts) so features are always "observed" (no extrapolation
# past a patient's window); W is appended (normalized) so the NPE knows the
# absolute horizon. Same fn for bank (full traj) and patient (their obs).
# ---------------------------------------------------------------------------
NG2   = 19
W_MAX = 1050.0
NF2   = 2*NG2 + 1

def _grid_rel(W):
    return np.linspace(0.0, float(W), NG2)

def _feat2(sld, cal, W):
    sld = np.clip(sld, *SLD_CLIP)
    cal = np.clip(cal, *LOGCA_CLIP)
    return np.concatenate([sld, cal, [W/W_MAX]]).astype(np.float32)

def features_v2_from_grid(sld_rel76, ca_rel76, bank_grid, W):
    g = _grid_rel(W)
    sld = np.interp(g, bank_grid, sld_rel76)
    cal = np.log(np.clip(np.interp(g, bank_grid, ca_rel76), 1e-30, None))
    return _feat2(sld, cal, W)

def features_v2_from_patient(sld_days, sld_rel, ca_days, ca_rel_, W):
    g = _grid_rel(W)
    sld = np.interp(g, sld_days, sld_rel)
    cal = np.log(np.clip(np.interp(g, ca_days, ca_rel_), 1e-30, None))
    return _feat2(sld, cal, W)

def add_noise_v2(X, sigma_sld=0.10, sigma_logca=0.45, rng=None):
    """Noise on the 2*NG2 trajectory features; leave the W-context feature clean."""
    rng = rng or np.random.default_rng(0)
    Xn = X.copy()
    Xn[:, :NG2]        += rng.normal(0, sigma_sld,   size=(X.shape[0], NG2))
    Xn[:, NG2:2*NG2]   += rng.normal(0, sigma_logca, size=(X.shape[0], NG2))
    return Xn.astype(np.float32)

def add_noise(X, sigma_sld=0.10, sigma_logca=0.45, rng=None):
    """Per-channel observation noise (matches the emcee posterior sigma scales:
    SLD ~10% relative, CA19-9 ~0.45 on log). Teaches the NPE that x is noisy."""
    rng = rng or np.random.default_rng(0)
    n = len(OUT_GRID)
    Xn = X.copy()
    Xn[:, :n]  += rng.normal(0, sigma_sld,   size=(X.shape[0], n))
    Xn[:, n:]  += rng.normal(0, sigma_logca, size=(X.shape[0], n))
    return Xn.astype(np.float32)
