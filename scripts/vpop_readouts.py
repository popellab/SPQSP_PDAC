"""vpop readouts function for the coupled QSP PDAC model.

Maps (S, N, Q) species → (N, M_r) readouts on log/logit scale,
covering all calibration targets except predictions and submodel targets.

Species layout
--------------
Each scenario produces (N, Q) raw positive species. The Q columns are
defined by the ``SPECIES`` enum below and must be emitted in that order
by the species function (ODE solver or trained surrogate).  V_T (tumor
volume) and phi_collagen are assignment-rule quantities; the species
function must compute and append them.  Columns 41+ are trajectory-
derived scalar features pre-computed during pool generation.

Scenarios
---------
Six scenarios (three point-in-time, three longterm trajectory):

    0  baseline                diagnosis snapshot (t=0)
    1  gvax_d21                GVAX neoadjuvant, readout at day 21
    2  gvax_nivo_d21           GVAX + nivolumab neoadjuvant, readout at day 30
    3  untreated_longterm      no treatment, 1095 days (trajectory features)
    4  gvax_longterm           GVAX neoadjuvant, 1095 days (trajectory features)
    5  gvax_nivo_longterm      GVAX + nivo neoadjuvant, 1095 days (trajectory features)

Scenarios 1–2 also carry trajectory features (excursion mass-center
timing, checkpoint release fold) extracted from the full CSV output.

Auxiliary readout parameters (log_R)
------------------------------------
Four nuisance parameters that bridge measurement to model space,
sampled during the fit alongside phi.  Indexed as:

    0  f_serum_to_tumor_TGFb    serum:tumor bridge for TGF-beta
    1  f_serum_to_tumor_IL12    serum:tumor bridge for IL-12
    2  f_serum_to_tumor_ArgI    serum:tumor bridge for Arginase I
    3  active_fraction_IL1      active:total IL-1 fraction

All are on log scale in log_R, so the raw positive value is exp(log_R[i]).

Usage
-----
    from vpop.model import FunctionSurrogate
    from scripts.vpop_readouts import (
        SCENARIOS, READOUTS, LOGIT_READOUT, N_SPECIES, readouts,
        LOG_R_0, SIGMA_R,
    )

    surrogate = FunctionSurrogate(
        scenarios=SCENARIOS, readouts=READOUTS, n_species=N_SPECIES,
        g_fn=species_fn, h_fn=readouts, logit_readout=LOGIT_READOUT,
    )
"""
from __future__ import annotations

from enum import IntEnum

import jax.numpy as jnp
import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# Species column indices  (Q columns per scenario)
# ═══════════════════════════════════════════════════════════════════════
class SP(IntEnum):
    """Column indices into the (N, Q) species block."""
    C1       = 0
    C_x      = 1
    CD8      = 2
    CD8_exh  = 3
    CD8_pex  = 4
    Th       = 5
    Th_exh   = 6
    nCD4     = 7
    Treg     = 8
    Mac_M1   = 9
    Mac_M2   = 10
    MDSC     = 11
    cDC1     = 12
    cDC2     = 13
    mcDC1    = 14
    mcDC2    = 15
    iCAF     = 16
    myCAF    = 17
    apCAF    = 18
    qPSC     = 19
    nCD8     = 20
    nB       = 21
    aB       = 22
    Tfh      = 23
    Plasma   = 24
    nBreg    = 25
    aBreg    = 26
    collagen = 27   # mg
    IFNg     = 28   # nM
    TGFb     = 29   # nM
    IL10     = 30   # nM
    IL12     = 31   # nM
    CCL2     = 32   # nM
    ArgI     = 33   # nM
    NO       = 34   # nM
    IL6      = 35   # nM
    IL1      = 36   # nM
    VEGF     = 37   # pg/mL
    V_T      = 38   # tumor volume, mL  (assignment rule)
    phi_col  = 39   # collagen volume fraction (assignment rule)
    rho_col  = 40   # collagen packing density, g/mL (parameter)
    # ── trajectory-derived scalar features (pre-computed in pool) ─────
    traj_C1_fold_d90   = 41   # C1(d90)/C1(d0)
    traj_exh_fold_d90  = 42   # exhaustion fraction fold at d90
    traj_ecm_fold_d90  = 43   # ECM concentration fold at d90
    traj_diam_d365     = 44   # tumor diameter (cm) at d365
    traj_diam_d730     = 45   # tumor diameter (cm) at d730
    traj_diam_fold_d90 = 46   # diam(d90)/diam(d0)
    traj_TTP           = 47   # time to progression (days)
    traj_doubling_time = 48   # tumor volume doubling time (days)
    traj_mapc_cd8_lag  = 49   # mAPC→CD8 excursion mass-center lag (days)
    traj_tumor_cd8_lag = 50   # tumor-nadir → CD8-peak lag (days)
    traj_ckpt_fold     = 51   # k_C_Tcell_eff fold at target day


N_SPECIES = len(SP)

# Map from model_structure.json species index to SP column.
# model_structure.json V_T species are indices 19-60.
MODEL_IDX_TO_SP = {
    23: SP.C1, 19: SP.C_x, 27: SP.CD8, 20: SP.CD8_exh, 21: SP.CD8_pex,
    36: SP.Th, 22: SP.Th_exh, 37: SP.nCD4, 26: SP.Treg,
    44: SP.Mac_M1, 45: SP.Mac_M2, 40: SP.MDSC,
    29: SP.cDC1, 30: SP.cDC2, 31: SP.mcDC1, 32: SP.mcDC2,
    49: SP.iCAF, 50: SP.myCAF, 51: SP.apCAF, 48: SP.qPSC,
    28: SP.nCD8, 57: SP.nB, 58: SP.aB, 56: SP.Tfh, 59: SP.Plasma,
    60: SP.nBreg, 61: SP.aBreg,
    52: SP.collagen,
    38: SP.IFNg, 39: SP.TGFb, 47: SP.IL10, 46: SP.IL12,
    43: SP.CCL2, 42: SP.ArgI, 41: SP.NO, 54: SP.IL6, 55: SP.IL1,
    25: SP.VEGF,
}


# ═══════════════════════════════════════════════════════════════════════
# Scenarios
# ═══════════════════════════════════════════════════════════════════════
SCENARIOS = (
    "baseline",              # 0
    "gvax_d21",              # 1
    "gvax_nivo_d21",         # 2
    "untreated_longterm",    # 3
    "gvax_longterm",         # 4
    "gvax_nivo_longterm",    # 5
)
_BL, _GV, _GN = 0, 1, 2
_UL, _GL, _GNL = 3, 4, 5


# ═══════════════════════════════════════════════════════════════════════
# Physical constants
# ═══════════════════════════════════════════════════════════════════════
_THICK  = 4.0       # FFPE section thickness, µm
_D_T    = 7.0       # T cell / Treg diameter, µm
_D_DC   = 13.0      # dendritic cell diameter, µm
_D_MDSC = 12.0      # MDSC diameter, µm
_D_MAC  = 20.0      # macrophage diameter, µm
_D_B    = 8.0       # B cell diameter, µm

_MW_TGFB = 25_000.0   # TGF-β1 homodimer, g/mol
_MW_IL12 = 70_000.0   # IL-12p70, g/mol
_MW_ARGI = 34_735.0   # Arginase I monomer, g/mol

_ML_PER_MG_PROT = 0.01  # VEGF: mL per mg total protein (tissue)
_VOL_CANCER     = 2.572e-9   # cancer cell volume, mL/cell
_VE_T           = 0.37       # intracellular volume fraction

_NORMAL_COL_DRY   = 5.0    # µg/mg dry tissue
_DRY_FRAC         = 0.22   # pancreas dry weight fraction
_TISSUE_DENSITY   = 1.04   # g/cm³ = g/mL


# ═══════════════════════════════════════════════════════════════════════
# Auxiliary readout parameters (log_R)
# ═══════════════════════════════════════════════════════════════════════
N_AUX = 4
_AUX_TGFB = 0
_AUX_IL12 = 1
_AUX_ARGI = 2
_AUX_IL1  = 3

LOG_R_0 = jnp.zeros(N_AUX)
SIGMA_R = jnp.array([0.7, 0.7, 0.7, 0.5])


# ═══════════════════════════════════════════════════════════════════════
# Readout definitions
# ═══════════════════════════════════════════════════════════════════════

READOUTS = (
    # ── baseline fractions (logit channel) ────────────────────────────
    "cancer_fraction",                 #  0
    "cd8_fraction",                    #  1
    "cd4_fraction",                    #  2
    "macrophage_fraction",             #  3
    "fibroblast_fraction",             #  4
    "dendritic_fraction",              #  5
    "mdsc_fraction",                   #  6
    "bcell_fraction",                  #  7
    "cd8_exhausted_fraction",          #  8
    "th_exhausted_fraction",           #  9
    "treg_fraction_of_tcells",         # 10
    "treg_fraction_cd4",               # 11
    "stromal_fraction",                # 12
    "apc_maturation_ratio",            # 13
    "icaf_fraction_caf",               # 14
    "mycaf_fraction_caf",              # 15
    "apcaf_fraction_caf",              # 16
    # ── baseline densities (log channel) ──────────────────────────────
    "cd8_density",                     # 17
    "cd4_helper_density",              # 18
    "treg_density",                    # 19
    "tam_density",                     # 20
    "mdsc_density",                    # 21
    "cdc1_density",                    # 22
    "cdc2_density",                    # 23
    "total_til_density",               # 24
    "bcell_density",                   # 25
    # ── baseline ratios (log channel) ─────────────────────────────────
    "cd8_treg_ratio",                  # 26
    "m1_m2_ratio",                     # 27
    "collagen_fold",                   # 28
    # ── baseline cytokine concentrations (log channel) ────────────────
    "tgfb_concentration",              # 29
    "il12_concentration",              # 30
    "arginase_concentration",          # 31
    "il10_concentration",              # 32
    "il1_concentration",               # 33
    "il6_concentration",               # 34
    "ccl2_concentration",              # 35
    "vegf_concentration",              # 36
    # ── GVAX d21 targets ──────────────────────────────────────────────
    "cd4_pct_gvax_d21",                # 37
    "cd8_pct_gvax_d21",                # 38
    "cd8exh_pct_gvax_d21",             # 39
    "m1tam_pct_gvax_d21",              # 40
    "m2tam_pct_gvax_d21",              # 41
    "treg_pct_gvax_d21",               # 42
    "cd8_fc_gvax_d21",                 # 43
    "cd8gzmb_fc_gvax_d21",             # 44
    "tumor_fold_d21_gvax",             # 45
    # ── GVAX+nivo d21 targets ─────────────────────────────────────────
    "cd4_pct_gvax_nivo_d21",           # 46
    "cd8_pct_gvax_nivo_d21",           # 47
    "cd8exh_pct_gvax_nivo_d21",        # 48
    "m1tam_pct_gvax_nivo_d21",         # 49
    "m2tam_pct_gvax_nivo_d21",         # 50
    "treg_pct_gvax_nivo_d21",          # 51
    "cd8_fc_gvax_nivo_d21",            # 52
    "cd8gzmb_fc_gvax_nivo_d21",        # 53
    "treg_fraction_immune_gvax_nivo",  # 54
    "m1_m2_ratio_gvax_nivo_d21",       # 55
    "tumor_fold_d21_gvax_nivo",        # 56
    # ── trajectory-derived readouts (log channel) ─────────────────────
    "tumor_no_regression_d90",              # 57
    "exhaustion_worsens_d90",               # 58
    "ecm_stability_d90",                    # 59
    "tumor_diameter_d365",                  # 60
    "tumor_diameter_d730",                  # 61
    "tumor_diam_fold_d90_untreated",        # 62
    "time_to_progression",                  # 63
    "tumor_doubling_time",                  # 64
    "mapc_cd8_peak_lag_gvax",               # 65
    "tumor_nadir_cd8_lag_gvax",             # 66
    "tumor_diam_fold_d90_gvax",             # 67
    "mapc_cd8_peak_lag_gvax_nivo",          # 68
    "tumor_nadir_cd8_lag_gvax_nivo",        # 69
    "tumor_diam_fold_d90_gvax_nivo",        # 70
    "checkpoint_release_fold",              # 71
    "tumor_diam_fold_d90_gvax_vs_untreated",# 72
)

M_R = len(READOUTS)


# ── logit/log channel mask ────────────────────────────────────────────
# True → readout is on (0, 1), returned as log-odds (logit channel).
# False → readout is positive, returned as log.

_logit_indices = {
    0, 1, 2, 3, 4, 5, 6, 7,           # baseline PASS-01 fractions
    8, 9, 10, 11, 12, 13, 14, 15, 16,  # baseline sub-fractions
    37, 38, 39, 40, 41, 42,            # gvax pct
    46, 47, 48, 49, 50, 51,            # gvax_nivo pct
    54,                                 # treg fraction of immune
}

LOGIT_READOUT = np.array([i in _logit_indices for i in range(M_R)])


# ═══════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════

def _pass01_denom(s):
    """20-species denominator matching PASS-01 FunCN fraction targets.

    Excludes C_x, CD8_pex, qPSC, Plasma, Tfh relative to the full
    nucleated_total_T assignment rule.
    """
    return (s[:, SP.C1]
          + s[:, SP.Mac_M1] + s[:, SP.Mac_M2]
          + s[:, SP.iCAF] + s[:, SP.myCAF] + s[:, SP.apCAF]
          + s[:, SP.Th] + s[:, SP.Th_exh] + s[:, SP.Treg]
          + s[:, SP.CD8] + s[:, SP.CD8_exh]
          + s[:, SP.nB] + s[:, SP.aB]
          + s[:, SP.nBreg] + s[:, SP.aBreg]
          + s[:, SP.cDC1] + s[:, SP.cDC2]
          + s[:, SP.mcDC1] + s[:, SP.mcDC2]
          + s[:, SP.MDSC]
          + s[:, SP.nCD8] + s[:, SP.nCD4])


def _nucleated_total(s):
    """Full nucleated_total_T (25 species) matching the QSP assignment rule."""
    return (s[:, SP.C1] + s[:, SP.C_x]
          + s[:, SP.CD8] + s[:, SP.CD8_exh] + s[:, SP.CD8_pex] + s[:, SP.nCD8]
          + s[:, SP.nCD4] + s[:, SP.Th] + s[:, SP.Th_exh] + s[:, SP.Treg]
          + s[:, SP.Mac_M1] + s[:, SP.Mac_M2]
          + s[:, SP.MDSC]
          + s[:, SP.cDC1] + s[:, SP.cDC2] + s[:, SP.mcDC1] + s[:, SP.mcDC2]
          + s[:, SP.nB] + s[:, SP.aB] + s[:, SP.Plasma]
          + s[:, SP.nBreg] + s[:, SP.aBreg]
          + s[:, SP.Tfh]
          + s[:, SP.qPSC] + s[:, SP.iCAF] + s[:, SP.myCAF] + s[:, SP.apCAF])


def _density(cells, v_t, d_cell):
    """3D→2D density projection: cells/mm² of FFPE section."""
    return cells / v_t * (_THICK + d_cell) * 1e-6


def _logit(p):
    """Stable log-odds: log(p / (1 - p)), clamped to avoid NaN."""
    p = jnp.clip(p, 1e-12, 1.0 - 1e-12)
    return jnp.log(p) - jnp.log1p(-p)


def _caf_total(s):
    return s[:, SP.iCAF] + s[:, SP.myCAF] + s[:, SP.apCAF]


def _immune_total(s):
    """All leukocytes (Tcell + MAC + MDSC + DC), excluding cancer and CAFs."""
    return (s[:, SP.CD8] + s[:, SP.CD8_exh] + s[:, SP.CD8_pex] + s[:, SP.nCD8]
          + s[:, SP.nCD4] + s[:, SP.Th] + s[:, SP.Th_exh] + s[:, SP.Treg]
          + s[:, SP.Mac_M1] + s[:, SP.Mac_M2]
          + s[:, SP.MDSC]
          + s[:, SP.cDC1] + s[:, SP.cDC2] + s[:, SP.mcDC1] + s[:, SP.mcDC2])


# ═══════════════════════════════════════════════════════════════════════
# Readouts function
# ═══════════════════════════════════════════════════════════════════════

def readouts(y, log_R=None):
    """``(S, N, Q) → (N, M_r)`` readouts on log/logit scale.

    Parameters
    ----------
    y : tuple of jnp.ndarray
        ``(y[0], ..., y[5])`` with shapes ``(N, Q)`` each.
        Scenarios: 0=baseline, 1=gvax_d21, 2=gvax_nivo_d21,
        3=untreated_longterm, 4=gvax_longterm, 5=gvax_nivo_longterm.
    log_R : jnp.ndarray or None
        ``(4,)`` auxiliary readout parameters on log scale.
        If None, defaults to zeros (unit scaling).
    """
    bl = y[0]   # baseline  (N, Q)
    gv = y[1]   # gvax d21  (N, Q)
    gn = y[2]   # gvax+nivo d21  (N, Q)
    ul = y[3]   # untreated longterm  (N, Q)
    gl = y[4]   # gvax longterm  (N, Q)
    gnl = y[5]  # gvax+nivo longterm  (N, Q)

    if log_R is None:
        log_R = jnp.zeros(N_AUX)

    f_tgfb = jnp.exp(log_R[_AUX_TGFB])
    f_il12 = jnp.exp(log_R[_AUX_IL12])
    f_argi = jnp.exp(log_R[_AUX_ARGI])
    f_il1  = jnp.exp(log_R[_AUX_IL1])

    # ── baseline derived quantities ───────────────────────────────────
    denom_p = _pass01_denom(bl)        # PASS-01 20-species denominator
    v_t     = bl[:, SP.V_T]            # tumor volume, mL

    # ── baseline fractions (logit) ────────────────────────────────────
    cancer_frac = bl[:, SP.C1] / denom_p
    cd8_frac    = (bl[:, SP.CD8] + bl[:, SP.CD8_exh] + bl[:, SP.nCD8]) / denom_p
    cd4_frac    = (bl[:, SP.Th] + bl[:, SP.Th_exh] + bl[:, SP.Treg] + bl[:, SP.nCD4]) / denom_p
    mac_frac    = (bl[:, SP.Mac_M1] + bl[:, SP.Mac_M2]) / denom_p
    fib_frac    = (bl[:, SP.iCAF] + bl[:, SP.myCAF] + bl[:, SP.apCAF]) / denom_p
    dc_frac     = (bl[:, SP.cDC1] + bl[:, SP.cDC2] + bl[:, SP.mcDC1] + bl[:, SP.mcDC2]) / denom_p
    mdsc_frac   = (bl[:, SP.MDSC] + bl[:, SP.nCD8] + bl[:, SP.nCD4]) / denom_p
    bcell_frac  = (bl[:, SP.nB] + bl[:, SP.aB] + bl[:, SP.nBreg] + bl[:, SP.aBreg]) / denom_p

    cd8_total_bl = bl[:, SP.CD8] + bl[:, SP.CD8_exh]
    th_total_bl  = bl[:, SP.Th] + bl[:, SP.Th_exh]

    cd8exh_frac = bl[:, SP.CD8_exh] / (cd8_total_bl + 1e-30)
    thexh_frac  = bl[:, SP.Th_exh] / (th_total_bl + 1e-30)

    tcell_total_bl = (cd8_total_bl + bl[:, SP.CD8_pex] + bl[:, SP.nCD8]
                    + bl[:, SP.Th] + bl[:, SP.Th_exh] + bl[:, SP.Treg])
    treg_of_tcell = bl[:, SP.Treg] / (tcell_total_bl + 1e-30)

    cd4_with_treg = bl[:, SP.Treg] + bl[:, SP.Th] + bl[:, SP.Th_exh]
    treg_of_cd4   = bl[:, SP.Treg] / (cd4_with_treg + 1e-30)

    cancer_epi_vol = bl[:, SP.C1] * _VOL_CANCER / _VE_T
    stromal_frac   = 1.0 - cancer_epi_vol / (v_t + 1e-30)

    dc_total = (bl[:, SP.cDC1] + bl[:, SP.cDC2]
              + bl[:, SP.mcDC1] + bl[:, SP.mcDC2])
    mdc_total = bl[:, SP.mcDC1] + bl[:, SP.mcDC2]
    apc_mat   = mdc_total / (dc_total + 1e-30)

    caf_t = _caf_total(bl)
    icaf_of_caf  = bl[:, SP.iCAF]  / (caf_t + 1e-30)
    mycaf_of_caf = bl[:, SP.myCAF] / (caf_t + 1e-30)
    apcaf_of_caf = bl[:, SP.apCAF] / (caf_t + 1e-30)

    # ── baseline densities (log) ──────────────────────────────────────
    cd8_dens    = _density(cd8_total_bl, v_t, _D_T)
    cd4h_dens   = _density(th_total_bl, v_t, _D_T)
    treg_dens   = _density(bl[:, SP.Treg], v_t, _D_T)
    tam_dens    = _density(bl[:, SP.Mac_M1] + bl[:, SP.Mac_M2], v_t, _D_MAC)
    mdsc_dens   = _density(bl[:, SP.MDSC], v_t, _D_MDSC)
    cdc1_dens   = _density(bl[:, SP.cDC1] + bl[:, SP.mcDC1], v_t, _D_DC)
    cdc2_dens   = _density(bl[:, SP.cDC2] + bl[:, SP.mcDC2], v_t, _D_DC)
    til_total   = cd8_total_bl + th_total_bl + bl[:, SP.Treg]
    til_dens    = _density(til_total, v_t, _D_T)

    bcell_total_bl = bl[:, SP.nB] + bl[:, SP.aB] + bl[:, SP.nBreg] + bl[:, SP.aBreg]
    bcell_dens  = _density(bcell_total_bl, v_t, _D_B)

    # ── baseline ratios (log) ─────────────────────────────────────────
    cd8_treg = cd8_total_bl / (bl[:, SP.Treg] + 1e-30)
    m1_m2_bl = bl[:, SP.Mac_M1] / (bl[:, SP.Mac_M2] + 1e-30)

    # collagen fold vs normal pancreas
    phi_col    = bl[:, SP.phi_col]
    rho_col    = bl[:, SP.rho_col]
    normal_conc = _NORMAL_COL_DRY * 1e-3 * _DRY_FRAC * _TISSUE_DENSITY
    phi_normal  = normal_conc / (rho_col + 1e-30)
    col_fold    = phi_col / (phi_normal + 1e-30)

    # ── baseline cytokine concentrations (log) ────────────────────────
    tgfb_conc = bl[:, SP.TGFb] * f_tgfb * _MW_TGFB * 0.001
    il12_conc = bl[:, SP.IL12] * f_il12 * _MW_IL12
    argi_conc = bl[:, SP.ArgI] * f_argi * _MW_ARGI * 0.001
    il10_conc = bl[:, SP.IL10]
    il1_conc  = bl[:, SP.IL1] / (f_il1 + 1e-30)
    il6_conc  = bl[:, SP.IL6]
    ccl2_conc = bl[:, SP.CCL2]
    vegf_conc = bl[:, SP.VEGF] * _ML_PER_MG_PROT

    # ── GVAX d21 targets ──────────────────────────────────────────────
    nt_gv = _nucleated_total(gv)

    cd4_pct_gv    = (gv[:, SP.Th] + gv[:, SP.Th_exh] + gv[:, SP.Treg]) / nt_gv
    cd8_pct_gv    = (gv[:, SP.CD8] + gv[:, SP.CD8_exh]) / nt_gv
    cd8exh_pct_gv = gv[:, SP.CD8_exh] / nt_gv
    m1_pct_gv     = gv[:, SP.Mac_M1] / nt_gv
    m2_pct_gv     = gv[:, SP.Mac_M2] / nt_gv
    treg_pct_gv   = gv[:, SP.Treg] / nt_gv

    cd8_total_gv = gv[:, SP.CD8] + gv[:, SP.CD8_exh]
    cd8_fc_gv     = cd8_total_gv / (cd8_total_bl + 1e-30)
    cd8gzmb_fc_gv = gv[:, SP.CD8] / (bl[:, SP.CD8] + 1e-30)
    tumor_fold_gv = gv[:, SP.C1] / (bl[:, SP.C1] + 1e-30)

    # ── GVAX+nivo d21 targets ─────────────────────────────────────────
    nt_gn = _nucleated_total(gn)

    cd4_pct_gn    = (gn[:, SP.Th] + gn[:, SP.Th_exh] + gn[:, SP.Treg]) / nt_gn
    cd8_pct_gn    = (gn[:, SP.CD8] + gn[:, SP.CD8_exh]) / nt_gn
    cd8exh_pct_gn = gn[:, SP.CD8_exh] / nt_gn
    m1_pct_gn     = gn[:, SP.Mac_M1] / nt_gn
    m2_pct_gn     = gn[:, SP.Mac_M2] / nt_gn
    treg_pct_gn   = gn[:, SP.Treg] / nt_gn

    cd8_total_gn = gn[:, SP.CD8] + gn[:, SP.CD8_exh]
    cd8_fc_gn     = cd8_total_gn / (cd8_total_bl + 1e-30)
    cd8gzmb_fc_gn = gn[:, SP.CD8] / (bl[:, SP.CD8] + 1e-30)

    immune_gn = _immune_total(gn)
    treg_frac_imm_gn = gn[:, SP.Treg] / (immune_gn + 1e-30)

    m1_m2_gn = gn[:, SP.Mac_M1] / (gn[:, SP.Mac_M2] + 1e-30)
    tumor_fold_gn = gn[:, SP.C1] / (bl[:, SP.C1] + 1e-30)

    # ── trajectory-derived readouts (all log channel) ─────────────────
    traj_no_regr    = ul[:, SP.traj_C1_fold_d90]
    traj_exh_wors   = ul[:, SP.traj_exh_fold_d90]
    traj_ecm_stab   = ul[:, SP.traj_ecm_fold_d90]
    traj_diam_365   = ul[:, SP.traj_diam_d365]
    traj_diam_730   = ul[:, SP.traj_diam_d730]
    traj_fold_d90_u = ul[:, SP.traj_diam_fold_d90]
    traj_ttp        = ul[:, SP.traj_TTP]
    traj_dt         = ul[:, SP.traj_doubling_time]

    traj_mapc_lag_gv  = gv[:, SP.traj_mapc_cd8_lag]
    traj_tum_lag_gv   = gv[:, SP.traj_tumor_cd8_lag]
    traj_fold_d90_gv  = gl[:, SP.traj_diam_fold_d90]

    traj_mapc_lag_gn  = gn[:, SP.traj_mapc_cd8_lag]
    traj_tum_lag_gn   = gn[:, SP.traj_tumor_cd8_lag]
    traj_fold_d90_gn  = gnl[:, SP.traj_diam_fold_d90]
    traj_ckpt         = gn[:, SP.traj_ckpt_fold]

    traj_gvax_vs_untr = traj_fold_d90_gv / (traj_fold_d90_u + 1e-30)

    # ── assemble output ───────────────────────────────────────────────
    # Each readout goes through its channel: logit for (0,1) readouts,
    # log for positive readouts.
    vals = [
        # baseline fractions → logit
        _logit(cancer_frac),            #  0
        _logit(cd8_frac),               #  1
        _logit(cd4_frac),               #  2
        _logit(mac_frac),               #  3
        _logit(fib_frac),               #  4
        _logit(dc_frac),                #  5
        _logit(mdsc_frac),              #  6
        _logit(bcell_frac),             #  7
        _logit(cd8exh_frac),            #  8
        _logit(thexh_frac),             #  9
        _logit(treg_of_tcell),          # 10
        _logit(treg_of_cd4),            # 11
        _logit(stromal_frac),           # 12
        _logit(apc_mat),                # 13
        _logit(icaf_of_caf),            # 14
        _logit(mycaf_of_caf),           # 15
        _logit(apcaf_of_caf),           # 16
        # baseline densities → log
        jnp.log(cd8_dens),              # 17
        jnp.log(cd4h_dens),             # 18
        jnp.log(treg_dens),             # 19
        jnp.log(tam_dens),              # 20
        jnp.log(mdsc_dens),             # 21
        jnp.log(cdc1_dens),             # 22
        jnp.log(cdc2_dens),             # 23
        jnp.log(til_dens),              # 24
        jnp.log(bcell_dens),            # 25
        # baseline ratios → log
        jnp.log(cd8_treg),              # 26
        jnp.log(m1_m2_bl),              # 27
        jnp.log(col_fold),              # 28
        # baseline cytokines → log
        jnp.log(tgfb_conc),             # 29
        jnp.log(il12_conc),             # 30
        jnp.log(argi_conc),             # 31
        jnp.log(il10_conc),             # 32
        jnp.log(il1_conc),              # 33
        jnp.log(il6_conc),              # 34
        jnp.log(ccl2_conc),             # 35
        jnp.log(vegf_conc),             # 36
        # gvax d21 → logit for pct, log for folds
        _logit(cd4_pct_gv),             # 37
        _logit(cd8_pct_gv),             # 38
        _logit(cd8exh_pct_gv),          # 39
        _logit(m1_pct_gv),              # 40
        _logit(m2_pct_gv),              # 41
        _logit(treg_pct_gv),            # 42
        jnp.log(cd8_fc_gv),             # 43
        jnp.log(cd8gzmb_fc_gv),         # 44
        jnp.log(tumor_fold_gv),         # 45
        # gvax+nivo d21 → logit for pct, log for folds/ratios
        _logit(cd4_pct_gn),             # 46
        _logit(cd8_pct_gn),             # 47
        _logit(cd8exh_pct_gn),          # 48
        _logit(m1_pct_gn),              # 49
        _logit(m2_pct_gn),              # 50
        _logit(treg_pct_gn),            # 51
        jnp.log(cd8_fc_gn),             # 52
        jnp.log(cd8gzmb_fc_gn),         # 53
        _logit(treg_frac_imm_gn),       # 54
        jnp.log(m1_m2_gn),              # 55
        jnp.log(tumor_fold_gn),         # 56
        # trajectory readouts → log
        jnp.log(traj_no_regr),          # 57
        jnp.log(traj_exh_wors),         # 58
        jnp.log(traj_ecm_stab),         # 59
        jnp.log(traj_diam_365),         # 60
        jnp.log(traj_diam_730),         # 61
        jnp.log(traj_fold_d90_u),       # 62
        jnp.log(traj_ttp),             # 63
        jnp.log(traj_dt),              # 64
        jnp.log(traj_mapc_lag_gv),      # 65
        jnp.log(traj_tum_lag_gv),       # 66
        jnp.log(traj_fold_d90_gv),      # 67
        jnp.log(traj_mapc_lag_gn),      # 68
        jnp.log(traj_tum_lag_gn),       # 69
        jnp.log(traj_fold_d90_gn),      # 70
        jnp.log(traj_ckpt),            # 71
        jnp.log(traj_gvax_vs_untr),     # 72
    ]
    return jnp.stack(vals, axis=1)


# ═══════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print(f"Scenarios:  {len(SCENARIOS)}")
    print(f"Species/sc: {N_SPECIES}")
    print(f"Readouts:   {M_R}")
    print(f"Logit:      {int(LOGIT_READOUT.sum())}")
    print(f"Log:        {M_R - int(LOGIT_READOUT.sum())}")
    print(f"Aux (log_R):{N_AUX}")

    assert len(READOUTS) == M_R
    assert LOGIT_READOUT.shape == (M_R,)

    N = 16
    rng = np.random.default_rng(42)
    y = tuple(jnp.array(rng.uniform(0.1, 10.0, (N, N_SPECIES))) for _ in range(len(SCENARIOS)))
    log_R = jnp.zeros(N_AUX)
    out = readouts(y, log_R)
    print(f"Output:     {out.shape}  (expected ({N}, {M_R}))")
    assert out.shape == (N, M_R)

    n_finite = int(jnp.isfinite(out).sum())
    print(f"Finite:     {n_finite}/{N * M_R}")

    for i, name in enumerate(READOUTS):
        ch = "logit" if LOGIT_READOUT[i] else "  log"
        med = float(jnp.median(out[:, i]))
        print(f"  [{ch}] {i:2d} {name:40s}  median={med:+.3f}")

    print("\nSelf-test passed.")
