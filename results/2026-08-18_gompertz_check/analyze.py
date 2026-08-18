#!/usr/bin/env python3
"""Gompertz check: cancer growth with QSP-derived rates + local CC + natural death.

Reproduces the v6 stabilization analysis with the three cancer fixes:
  1. PROG_GROWTH_RATE derived from QSP k_C1_growth (0.010/day)
  2. Local Gompertz carrying capacity (K_local=20)
  3. Natural death for stem/progenitor (k_C1_death=0.003/day)

Usage:
    cd results/2026-08-18_gompertz_check
    ~/miniforge3/envs/panc-mech/bin/python3 analyze.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
SEED = 42
DT_H = 6.0

COMPOSITION = {
    "cancer_total":    ["agentCount.cancer.stem", "agentCount.cancer.prog", "agentCount.cancer.sen"],
    "CD8_active":      ["agentCount.CD8.effector", "agentCount.CD8.cytotoxic", "agentCount.CD8.suppressed"],
    "treg_th":         ["agentCount.Th.default", "agentCount.Tfh.default"],
    "treg_treg":       ["agentCount.Treg.default"],
    "B_lineage":       ["agentCount.BCell.naive", "agentCount.BCell.activated", "agentCount.BCell.plasma"],
    "mac_M1":          ["agentCount.MAC.M1"],
    "mac_M2":          ["agentCount.MAC.M2"],
    "mdsc":            ["agentCount.MDSC.default"],
    "fib_total":       ["agentCount.FIB.quiescent", "agentCount.FIB.myCAF", "agentCount.FIB.iCAF",
                        "agentCount.FIB.FRC", "agentCount.FIB.apCAF"],
    "fib_apCAF":       ["agentCount.FIB.apCAF"],
    "fib_quiescent":   ["agentCount.FIB.quiescent"],
    "CAF_activated":   ["agentCount.FIB.myCAF", "agentCount.FIB.iCAF"],
    "dc_immature_any": ["agentCount.DC.cDC1_immature", "agentCount.DC.cDC2_immature"],
    "dc_mature_any":   ["agentCount.DC.cDC1_mature", "agentCount.DC.cDC2_mature"],
}

FIT_CATEGORIES = [
    "cancer_total", "fib_apCAF", "CAF_activated", "fib_quiescent",
    "CD8_active", "treg_th", "treg_treg", "B_lineage",
    "mac_M1", "mac_M2", "mdsc", "dc_immature_any", "dc_mature_any",
]

LABELS = {
    "cancer_total": "Cancer", "fib_apCAF": "apCAF", "CAF_activated": "CAF(act)",
    "fib_quiescent": "Quiesc fib", "CD8_active": "CD8", "treg_th": "Th",
    "treg_treg": "Treg", "B_lineage": "B cell", "mac_M1": "M1 mac",
    "mac_M2": "M2 mac", "mdsc": "MDSC", "dc_immature_any": "iDC",
    "dc_mature_any": "mDC", "fib_total": "Fib(total)",
}

COLORS = {
    "cancer_total": "#d62728", "fib_apCAF": "#8c564b", "CAF_activated": "#e377c2",
    "fib_quiescent": "#bcbd22", "CD8_active": "#1f77b4", "treg_th": "#ff7f0e",
    "treg_treg": "#2ca02c", "B_lineage": "#9467bd", "mac_M1": "#17becf",
    "mac_M2": "#7f7f7f", "mdsc": "#aec7e8", "dc_immature_any": "#ffbb78",
    "dc_mature_any": "#98df8a",
}


def abm_composition(stats_row):
    total = (
        stats_row.get("agentCount.cancer.stem", 0) +
        stats_row.get("agentCount.cancer.prog", 0) +
        stats_row.get("agentCount.cancer.sen", 0) +
        stats_row.get("agentCount.CD8.effector", 0) +
        stats_row.get("agentCount.CD8.cytotoxic", 0) +
        stats_row.get("agentCount.CD8.suppressed", 0) +
        stats_row.get("agentCount.CD8.naive", 0) +
        stats_row.get("agentCount.Th.default", 0) +
        stats_row.get("agentCount.Treg.default", 0) +
        stats_row.get("agentCount.Tfh.default", 0) +
        stats_row.get("agentCount.TCD4.naive", 0) +
        stats_row.get("agentCount.MDSC.default", 0) +
        stats_row.get("agentCount.MAC.M1", 0) +
        stats_row.get("agentCount.MAC.M2", 0) +
        stats_row.get("agentCount.FIB.quiescent", 0) +
        stats_row.get("agentCount.FIB.myCAF", 0) +
        stats_row.get("agentCount.FIB.iCAF", 0) +
        stats_row.get("agentCount.FIB.FRC", 0) +
        stats_row.get("agentCount.FIB.apCAF", 0) +
        stats_row.get("agentCount.BCell.naive", 0) +
        stats_row.get("agentCount.BCell.activated", 0) +
        stats_row.get("agentCount.BCell.plasma", 0) +
        stats_row.get("agentCount.DC.cDC1_immature", 0) +
        stats_row.get("agentCount.DC.cDC1_mature", 0) +
        stats_row.get("agentCount.DC.cDC2_immature", 0) +
        stats_row.get("agentCount.DC.cDC2_mature", 0)
    )
    if total == 0:
        total = 1
    return {
        "cancer_total": (
            stats_row.get("agentCount.cancer.stem", 0) +
            stats_row.get("agentCount.cancer.prog", 0) +
            stats_row.get("agentCount.cancer.sen", 0)
        ) / total,
        "CAF_activated": (
            stats_row.get("agentCount.FIB.myCAF", 0) +
            stats_row.get("agentCount.FIB.iCAF", 0)
        ) / total,
        "fib_apCAF": stats_row.get("agentCount.FIB.apCAF", 0) / total,
        "fib_quiescent": stats_row.get("agentCount.FIB.quiescent", 0) / total,
        "CD8_active": (
            stats_row.get("agentCount.CD8.effector", 0) +
            stats_row.get("agentCount.CD8.cytotoxic", 0) +
            stats_row.get("agentCount.CD8.suppressed", 0) +
            stats_row.get("agentCount.CD8.naive", 0)
        ) / total,
        "treg_th": stats_row.get("agentCount.Th.default", 0) / total,
        "treg_treg": stats_row.get("agentCount.Treg.default", 0) / total,
        "B_lineage": (
            stats_row.get("agentCount.BCell.naive", 0) +
            stats_row.get("agentCount.BCell.activated", 0) +
            stats_row.get("agentCount.BCell.plasma", 0)
        ) / total,
        "mac_M1": stats_row.get("agentCount.MAC.M1", 0) / total,
        "mac_M2": stats_row.get("agentCount.MAC.M2", 0) / total,
        "mdsc": stats_row.get("agentCount.MDSC.default", 0) / total,
        "dc_immature_any": (
            stats_row.get("agentCount.DC.cDC1_immature", 0) +
            stats_row.get("agentCount.DC.cDC2_immature", 0)
        ) / total,
        "dc_mature_any": (
            stats_row.get("agentCount.DC.cDC1_mature", 0) +
            stats_row.get("agentCount.DC.cDC2_mature", 0)
        ) / total,
    }


def qsp_composition(qsp_row):
    counts = {
        "cancer_total": (qsp_row["V_T.C1"] + qsp_row.get("V_T.C2", 0) +
                         qsp_row.get("V_T.C_x", 0)),
        "CAF_activated": qsp_row.get("V_T.iCAF", 0) + qsp_row.get("V_T.myCAF", 0),
        "fib_apCAF": qsp_row.get("V_T.apCAF", 0),
        "fib_quiescent": qsp_row.get("V_T.qPSC", 0),
        "CD8_active": qsp_row.get("V_T.CD8", 0) + qsp_row.get("V_T.CD8_exh", 0),
        "treg_th": qsp_row.get("V_T.Th", 0) + qsp_row.get("V_T.Th_exh", 0),
        "treg_treg": qsp_row.get("V_T.Treg", 0),
        "B_lineage": (qsp_row.get("V_T.B", 0) + qsp_row.get("V_T.Plasma", 0)),
        "mac_M1": qsp_row.get("V_T.Mac_M1", 0),
        "mac_M2": qsp_row.get("V_T.Mac_M2", 0),
        "mdsc": qsp_row.get("V_T.MDSC", 0),
        "dc_immature_any": (qsp_row.get("V_T.cDC1", 0) + qsp_row.get("V_T.cDC2", 0)),
        "dc_mature_any": (qsp_row.get("V_T.mcDC1", 0) + qsp_row.get("V_T.mcDC2", 0)),
    }
    total = sum(counts.values())
    if total == 0:
        total = 1
    return {k: v / total for k, v in counts.items()}


def main():
    stats = pd.read_csv(OUT / f"stats_{SEED}.csv")
    phi = pd.read_csv(OUT / f"phi_{SEED}.csv")
    qsp = pd.read_csv(OUT / f"qsp_{SEED}.csv")

    days = stats["step"] * DT_H / 24.0
    phi_days = phi["step"] * DT_H / 24.0

    # Load QSP target from v6 presim
    report_file = HERE.parent / "2026-08-17_presim_v6" / "presim_posterior_report_v6.json"
    if report_file.exists():
        with open(report_file) as f:
            qsp_target = json.load(f)["target_composition"]
    else:
        qsp_target = {}

    # ── Composition time series ──
    counts = {}
    for cat, cols in COMPOSITION.items():
        present = [c for c in cols if c in stats.columns]
        counts[cat] = stats[present].sum(axis=1).to_numpy(dtype=float)

    total = sum(counts[c] for c in FIT_CATEGORIES)
    fracs = {c: counts[c] / np.clip(total, 1.0, None) for c in FIT_CATEGORIES}

    REF = 1

    # ── Stability table ──
    print(f"{'Category':<18s} {'Init(s1)':>10s} {'Final':>10s} {'QSP':>10s} "
          f"{'drift(dec)':>10s} {'n_init':>8s} {'n_final':>8s}")
    print("-" * 80)

    results = {}
    for cat in FIT_CATEGORIES:
        f0 = fracs[cat][REF]
        ff = fracs[cat][-1]
        qt = qsp_target.get(cat, 0.0)
        drift = np.log10(max(ff, 1e-10) / max(f0, 1e-10))
        n0 = int(counts[cat][REF])
        nf = int(counts[cat][-1])
        flag = " ** DRIFT" if abs(drift) > 0.3 else " * drift" if abs(drift) > 0.15 else ""
        print(f"  {LABELS.get(cat,cat):<16s} {f0:10.5f} {ff:10.5f} {qt:10.5f} "
              f"{drift:+10.3f} {n0:>8d} {nf:>8d}{flag}")
        results[cat] = {"frac_init": f0, "frac_final": ff, "qsp_target": qt,
                        "drift_dec": drift, "n_init": n0, "n_final": nf}

    total_init = int(total[REF])
    total_final = int(total[-1])
    print(f"\n  Total cells: {total_init} -> {total_final} "
          f"({100*(total_final - total_init)/total_init:+.1f}%)")

    # ── QSP scaled counts (volume ratio) ──
    VOXEL_CM = 20e-4  # 20 µm in cm
    gx, gy, gz = 100, 100, 12
    V_abm_cm3 = gx * gy * gz * VOXEL_CM**3
    V_qsp_cm3 = 17.2  # mL (onset V_T ≈ 17.2 mL)
    vol_ratio = V_abm_cm3 / V_qsp_cm3

    qsp_days = qsp["step"] * DT_H / 24.0

    QSP_MAP = {
        "cancer_total": ["V_T.C1", "V_T.C2", "V_T.C_x"],
        "CD8_active":   ["V_T.CD8", "V_T.CD8_exh"],
        "treg_th":      ["V_T.Th", "V_T.Th_exh"],
        "treg_treg":    ["V_T.Treg"],
        "B_lineage":    ["V_T.B", "V_T.Plasma"],
        "mac_M1":       ["V_T.Mac_M1"],
        "mac_M2":       ["V_T.Mac_M2"],
        "mdsc":         ["V_T.MDSC"],
        "fib_apCAF":    ["V_T.apCAF"],
        "CAF_activated": ["V_T.iCAF", "V_T.myCAF"],
        "fib_quiescent": ["V_T.qPSC"],
        "dc_immature_any": ["V_T.cDC1", "V_T.cDC2"],
        "dc_mature_any":   ["V_T.mcDC1", "V_T.mcDC2"],
    }

    qsp_counts = {}
    for cat, cols in QSP_MAP.items():
        present = [c for c in cols if c in qsp.columns]
        qsp_counts[cat] = qsp[present].sum(axis=1).to_numpy(dtype=float) * vol_ratio

    # ── FIGURE 1: Cell counts (ABM solid, QSP dashed) — 3×3 grid ──
    fig, axes = plt.subplots(3, 3, figsize=(16, 13))
    fig.suptitle("Gompertz check: cancer growth caps (v6 median, 50 days)\n"
                 "PROG_RATE=QSP 0.010/day, local CC K=30, no natural death\n"
                 f"QSP scaled by V_ABM/V_QSP = {vol_ratio:.2e}",
                 fontsize=11, y=1.01)

    def _plot_pair(ax, cat, title=None, logy=False, sub_cols=None):
        ax.plot(days, counts[cat], color=COLORS[cat], linewidth=2, label="ABM")
        ax.plot(qsp_days, qsp_counts[cat], color=COLORS[cat],
                linewidth=2, linestyle="--", alpha=0.7, label="QSP")
        if sub_cols:
            for col, lbl in sub_cols:
                if col in stats.columns:
                    ax.plot(days, stats[col], linewidth=1, alpha=0.5, label=lbl)
        ax.set_title(title or LABELS.get(cat, cat), fontsize=10)
        ax.set_xlabel("Day", fontsize=8)
        ax.set_ylabel("Cell count", fontsize=8)
        if logy:
            ax.set_yscale("log")
        ax.legend(fontsize=7)

    # Row 0: Cancer, Macrophages (M1 + M2), Fibroblasts
    _plot_pair(axes[0, 0], "cancer_total", "Cancer cells",
               sub_cols=[("agentCount.cancer.prog", "prog"),
                         ("agentCount.cancer.sen", "sen"),
                         ("agentCount.cancer.stem", "stem")])

    ax = axes[0, 1]
    ax.plot(days, counts["mac_M1"], color=COLORS["mac_M1"], linewidth=2, label="M1 ABM")
    ax.plot(qsp_days, qsp_counts["mac_M1"], color=COLORS["mac_M1"],
            linewidth=2, linestyle="--", alpha=0.7, label="M1 QSP")
    ax.plot(days, counts["mac_M2"], color=COLORS["mac_M2"], linewidth=2, label="M2 ABM")
    ax.plot(qsp_days, qsp_counts["mac_M2"], color=COLORS["mac_M2"],
            linewidth=2, linestyle="--", alpha=0.7, label="M2 QSP")
    ax.set_title("Macrophages (M1 & M2)", fontsize=10)
    ax.set_xlabel("Day", fontsize=8)
    ax.set_ylabel("Cell count", fontsize=8)
    ax.legend(fontsize=7)

    ax = axes[0, 2]
    for cat in ["fib_apCAF", "CAF_activated", "fib_quiescent"]:
        ax.plot(days, counts[cat], color=COLORS[cat], linewidth=1.5, label=LABELS[cat])
        ax.plot(qsp_days, qsp_counts[cat], color=COLORS[cat],
                linewidth=1.5, linestyle="--", alpha=0.5)
    ax.set_title("Fibroblasts", fontsize=10)
    ax.set_xlabel("Day", fontsize=8)
    ax.set_ylabel("Cell count", fontsize=8)
    ax.legend(fontsize=7)

    # Row 1: T cells (CD8, Th, Treg separate), B cells, MDSC
    _plot_pair(axes[1, 0], "CD8_active", "CD8 T cells")

    ax = axes[1, 1]
    ax.plot(days, counts["treg_th"], color=COLORS["treg_th"], linewidth=2, label="Th ABM")
    ax.plot(qsp_days, qsp_counts["treg_th"], color=COLORS["treg_th"],
            linewidth=2, linestyle="--", alpha=0.7, label="Th QSP")
    ax.plot(days, counts["treg_treg"], color=COLORS["treg_treg"], linewidth=2, label="Treg ABM")
    ax.plot(qsp_days, qsp_counts["treg_treg"], color=COLORS["treg_treg"],
            linewidth=2, linestyle="--", alpha=0.7, label="Treg QSP")
    ax.set_title("Th & Treg", fontsize=10)
    ax.set_xlabel("Day", fontsize=8)
    ax.set_ylabel("Cell count", fontsize=8)
    ax.legend(fontsize=7)

    _plot_pair(axes[1, 2], "B_lineage", "B cells")

    # Row 2: MDSC, DCs, total immune summary
    _plot_pair(axes[2, 0], "mdsc", "MDSC")

    ax = axes[2, 1]
    for cat in ["dc_immature_any", "dc_mature_any"]:
        ax.plot(days, counts[cat], color=COLORS[cat], linewidth=1.5, label=LABELS[cat] + " ABM")
        ax.plot(qsp_days, qsp_counts[cat], color=COLORS[cat],
                linewidth=1.5, linestyle="--", alpha=0.5, label=LABELS[cat] + " QSP")
    ax.set_title("Dendritic cells", fontsize=10)
    ax.set_xlabel("Day", fontsize=8)
    ax.set_ylabel("Cell count", fontsize=8)
    ax.legend(fontsize=7)

    # Summary panel: all immune on log scale
    ax = axes[2, 2]
    for cat in ["CD8_active", "treg_th", "treg_treg", "B_lineage",
                "mac_M1", "mac_M2", "mdsc"]:
        ax.plot(days, counts[cat], color=COLORS[cat], linewidth=1.5, label=LABELS[cat])
        ax.plot(qsp_days, qsp_counts[cat], color=COLORS[cat],
                linewidth=1, linestyle="--", alpha=0.4)
    ax.set_title("All immune (log, solid=ABM, dash=QSP)", fontsize=10)
    ax.set_xlabel("Day", fontsize=8)
    ax.set_ylabel("Cell count", fontsize=8)
    ax.set_yscale("log")
    ax.legend(fontsize=6, ncol=2)

    plt.tight_layout()
    fig.savefig(HERE / "cell_counts.png", dpi=150, bbox_inches="tight")
    print(f"\nSaved cell_counts.png")

    # ── FIGURE 2: Phi drift ──
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("FunCN $\\varphi$ closure scores (Gompertz check, v6 median)",
                 fontsize=12, y=1.01)

    phi_defs = [
        ("phi_cd8_c1", "valid_cd8_c1", "n_cd8", "$\\varphi_1$: CD8$\\times$C1", "#2B6777"),
        ("phi_m1_c1", "valid_m1_c1", "n_m1", "$\\varphi_2$: M1$\\times$C1", "#B85545"),
        ("phi_cd8_treg", "valid_cd8_treg", "n_treg", "$\\varphi_3$: CD8$\\times$Treg", "#3D8B5C"),
        ("phi_m2_c1", "valid_m2_c1", "n_m2", "$\\varphi_4$: M2$\\times$C1", "#7f7f7f"),
        ("phi_icaf_c1", "valid_icaf_c1", "n_icaf", "$\\varphi_5$: iCAF$\\times$C1", "#8c564b"),
        ("phi_mdsc_cd8", "valid_mdsc_cd8", "n_mdsc", "$\\varphi_6$: MDSC$\\times$CD8", "#aec7e8"),
    ]

    for ax, (pcol, vcol, ncol, title, color) in zip(axes.flat, phi_defs):
        if vcol in phi.columns:
            valid = phi[vcol].astype(bool)
            if valid.any():
                ax.plot(phi_days[valid], phi[pcol][valid], color=color, linewidth=1.5)
                ax.axhline(phi.loc[valid, pcol].iloc[0], color=color, linestyle="--",
                           alpha=0.5, linewidth=1)
        ax.axhline(1.0, color="#AAAAAA", linestyle=":", linewidth=0.8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Day")
        ax.set_ylabel("$\\varphi$")
        if ncol in phi.columns:
            ax2 = ax.twinx()
            ax2.fill_between(phi_days, phi[ncol], alpha=0.08, color="#B85C38")
            ax2.set_ylabel(f"n({ncol.replace('n_','')})", color="#B85C38", fontsize=7)
            ax2.tick_params(axis="y", labelcolor="#B85C38", labelsize=6)

    plt.tight_layout()
    fig.savefig(HERE / "phi_drift.png", dpi=150, bbox_inches="tight")
    print(f"Saved phi_drift.png")

    # ── FIGURE 3: ABM vs QSP composition ──
    abm_ts = {}
    for _, row in stats.iterrows():
        comp = abm_composition(row)
        for k, v in comp.items():
            abm_ts.setdefault(k, []).append(v)

    qsp_ts = {}
    for _, row in qsp.iterrows():
        comp = qsp_composition(row)
        for k, v in comp.items():
            qsp_ts.setdefault(k, []).append(v)

    abm_days_c = stats["step"] * DT_H / 24.0
    qsp_days_c = qsp["step"] * DT_H / 24.0

    CATS = [
        "cancer_total", "CAF_activated", "fib_apCAF", "fib_quiescent",
        "CD8_active", "treg_th", "treg_treg", "B_lineage",
        "mac_M1", "mac_M2", "mdsc", "dc_immature_any", "dc_mature_any",
    ]

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    axes_flat = axes.flatten()

    for i, cat in enumerate(CATS):
        ax = axes_flat[i]
        abm_y = np.array(abm_ts[cat])
        qsp_y = np.array(qsp_ts[cat])
        ax.plot(abm_days_c, abm_y, color="#2B6777", linewidth=1.5, label="ABM")
        ax.plot(qsp_days_c, qsp_y, color="#B85545", linewidth=1.5, linestyle="--", label="QSP")
        ax.set_title(LABELS[cat], fontsize=9)
        ax.set_xlabel("Day", fontsize=7)
        ax.set_ylabel("Fraction", fontsize=7)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=7, loc="best")

    for j in range(len(CATS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle("ABM vs QSP composition (Gompertz check, v6 median, 50 days)\n"
                 "Cancer growth: QSP-derived rate + Gompertz + natural death",
                 fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(HERE / "abundance_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved abundance_comparison.png")

    # ── FIGURE 4: Cancer balance ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("Cancer proliferation/death balance (Gompertz check)", fontsize=12, y=1.04)

    ax = axes[0]
    p_stem = stats.get("prolif.cancer.stem", pd.Series(0, index=stats.index))
    p_prog = stats.get("prolif.cancer.prog", pd.Series(0, index=stats.index))
    d_stem = stats.get("death.cancer.stem", pd.Series(0, index=stats.index))
    d_prog = stats.get("death.cancer.prog", pd.Series(0, index=stats.index))
    d_sen  = stats.get("death.cancer.sen", pd.Series(0, index=stats.index))
    ax.plot(days, p_stem + p_prog, label="Prolif (stem+prog)", color="#2ca02c")
    ax.plot(days, d_stem + d_prog + d_sen, label="Death (all)", color="#d62728")
    ax.plot(days, (p_stem + p_prog) - (d_stem + d_prog + d_sen), label="Net", color="k", linestyle="--")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_title("Cancer birth/death per step")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cells/step")
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(days, d_stem, label="Stem death", color="#ff7f0e")
    ax.plot(days, d_prog, label="Prog death", color="#d62728")
    ax.plot(days, d_sen, label="Sen death", color="#9467bd")
    ax.set_title("Cancer deaths by type")
    ax.set_xlabel("Day")
    ax.set_ylabel("Deaths/step")
    ax.legend(fontsize=7)

    ax = axes[2]
    cancer = counts["cancer_total"]
    if len(cancer) > 1 and cancer[0] > 0:
        growth_rate = np.gradient(np.log(np.clip(cancer, 1, None))) / (DT_H / 24.0)
        ax.plot(days, growth_rate, color="#d62728", linewidth=1.5)
    ax.axhline(0.010, color="gray", linestyle="--", alpha=0.5, label="QSP k_C1_growth=0.010")
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_title("Cancer effective growth rate")
    ax.set_xlabel("Day")
    ax.set_ylabel("d(ln N)/dt [1/day]")
    ax.legend(fontsize=7)

    plt.tight_layout()
    fig.savefig(HERE / "cancer_balance.png", dpi=150, bbox_inches="tight")
    print(f"Saved cancer_balance.png")

    # ── FIGURE 5: Composition drift lollipop ──
    fig, ax = plt.subplots(figsize=(8, 5))
    cats = [c for c in FIT_CATEGORIES if c != "cancer_total"]
    y_pos = np.arange(len(cats))
    drifts = [results[c]["drift_dec"] for c in cats]
    colors_d = ["#d62728" if abs(d) > 0.3 else "#ff7f0e" if abs(d) > 0.15 else "#2ca02c"
                for d in drifts]
    ax.barh(y_pos, drifts, color=colors_d, height=0.6, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([LABELS.get(c, c) for c in cats], fontsize=8)
    ax.set_xlabel("Drift (decades)")
    ax.set_title("Composition drift (day 0 -> day 50, Gompertz check)")
    ax.axvline(0, color="k", linewidth=0.5)
    ax.axvline(-0.3, color="r", linestyle="--", alpha=0.3, linewidth=0.8)
    ax.axvline(0.3, color="r", linestyle="--", alpha=0.3, linewidth=0.8)
    plt.tight_layout()
    fig.savefig(HERE / "composition_drift.png", dpi=150, bbox_inches="tight")
    print(f"Saved composition_drift.png")

    # ── Save results ──
    with open(HERE / "results.json", "w") as f:
        json.dump({
            "seed": SEED,
            "n_steps": int(stats["step"].iloc[-1]),
            "dt_hours": DT_H,
            "total_days": float(days.iloc[-1]),
            "total_cells_init": total_init,
            "total_cells_final": total_final,
            "composition": results,
        }, f, indent=2)
    print(f"Saved results.json")

    # ── Summary ──
    print("\n=== Summary ===")
    print(f"Cancer: {int(counts['cancer_total'][REF])} -> {int(counts['cancer_total'][-1])} "
          f"({100*(counts['cancer_total'][-1] - counts['cancer_total'][REF])/max(counts['cancer_total'][REF],1):+.1f}%)")
    print(f"Total:  {total_init} -> {total_final} ({100*(total_final-total_init)/total_init:+.1f}%)")


if __name__ == "__main__":
    main()
