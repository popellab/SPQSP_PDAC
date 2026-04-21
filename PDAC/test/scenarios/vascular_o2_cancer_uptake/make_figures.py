#!/usr/bin/env python3
"""
Figures for the vascular_o2_cancer_uptake substep-convergence sweep.

Three test variants are run at PARAM_MOLECULAR_STEPS ∈ {36, 360, 3600}:
- sub36   (dt_sub=600s, k·dt=60)   — production default, operator-splitting artifact
- sub360  (dt_sub=60s,  k·dt=6)    — partial gradient
- sub3600 (dt_sub=6s,   k·dt=0.6)  — converged K₀ profile

Per-substep exact-ODE uptake applies C → C·exp(-k·dt_sub). At k=0.1/s and
dt_sub=600s, exp(-60) ≈ 0 — tissue is drained to zero each substep before
LOD diffuses the vessel source. Higher substep counts bring the split closer
to the continuous-time solution D∇²C - k·C + S·δ = 0, whose radial form is
K₀(r/L) with L=√(D/k_eff).

Reads (per variant):
  outputs/<variant>/time_series.csv  (step, vessel, min, max, mean)
  outputs/<variant>/radial.csv       (step, r_voxel, n_samples, mean_conc)
  outputs/<variant>/xline.csv        (step, x, conc)
  outputs/<variant>/params.csv       (key, value)

Writes:
  outputs/radial_convergence.png    radial profiles @ final step, all 3 variants + K₀
  outputs/xline_convergence.png     x-line cross-sections @ final step
  outputs/radial_ratio_bar.png      C(r=1)/C(r=10) ratio per variant, pass/fail
  outputs/timeseries_vessel.png     vessel/min/max vs step per variant

Pass criteria (evaluated at the final ABM step):
  (P1) Radial ratio C(r=1)/C(r=10) increases monotonically with substep count.
  (P2) |fit_err| at r∈{3,5,7} against K₀(r/L) decreases monotonically with
       substep count — profile shape converges toward analytic K₀.
       (Absolute convergence requires k·dt_sub << 1, i.e., ≳36k substeps —
       prohibitive for an iterative test. Demonstrating convergence direction
       is sufficient to validate the solver.)
  (P3) sub36 radial ratio < 1.5 (documents the operator-splitting artifact).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import k0  # modified Bessel K_0

VARIANTS = ["vascular_o2_uptake_sub36",
            "vascular_o2_uptake_sub360",
            "vascular_o2_uptake_sub3600"]
VARIANT_LABELS = {
    "vascular_o2_uptake_sub36":   "36 substeps (k·dt=60)",
    "vascular_o2_uptake_sub360":  "360 substeps (k·dt=6)",
    "vascular_o2_uptake_sub3600": "3600 substeps (k·dt=0.6)",
}
VARIANT_COLORS = {
    "vascular_o2_uptake_sub36":   "tab:red",
    "vascular_o2_uptake_sub360":  "tab:orange",
    "vascular_o2_uptake_sub3600": "tab:green",
}


def load_variant(out_root: Path, variant: str):
    d = out_root / variant
    ts = pd.read_csv(d / "time_series.csv")
    rd = pd.read_csv(d / "radial.csv")
    xl = pd.read_csv(d / "xline.csv")
    pr = pd.read_csv(d / "params.csv")
    params = {}
    for _, row in pr.iterrows():
        try:
            params[row["key"]] = float(row["value"])
        except ValueError:
            params[row["key"]] = row["value"]  # strings (variant name)
    return ts, rd, xl, params


def diffusion_length_voxels(params: dict) -> float:
    # Effective first-order sink: cancer uptake + background decay.
    # L = sqrt(D / k_eff), convert to voxels via voxel size.
    D_cm2_s = params["PARAM_O2_DIFFUSIVITY"]          # cm²/s
    k_cancer = params["PARAM_O2_UPTAKE"]              # 1/s
    k_bg = params["PARAM_O2_DECAY_RATE"]              # 1/s
    k_eff = k_cancer + k_bg
    vox_cm = params["PARAM_VOXEL_SIZE_CM"]            # cm/voxel
    L_cm = np.sqrt(D_cm2_s / k_eff)
    return L_cm / vox_cm


def plot_radial_convergence(variant_data, out_path: Path):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.8))

    # Use each variant's final-step radial profile
    for variant in VARIANTS:
        ts, rd, xl, params = variant_data[variant]
        final_step = int(rd["step"].max())
        g = rd[rd["step"] == final_step].sort_values("r_voxel")
        r = g["r_voxel"].values.astype(float)
        c = g["mean_conc"].values
        color = VARIANT_COLORS[variant]
        axL.plot(r, c, lw=1.8, marker="o", ms=3.5, color=color,
                 label=VARIANT_LABELS[variant])
        axR.semilogy(r, np.maximum(c, 1e-12), lw=1.8, marker="o", ms=3.5,
                     color=color, label=VARIANT_LABELS[variant])

    # Analytic K_0(r/L) overlay, normalized to sub3600 at r=1.
    _, rd_hi, _, params_hi = variant_data["vascular_o2_uptake_sub3600"]
    final = int(rd_hi["step"].max())
    g_hi = rd_hi[rd_hi["step"] == final].sort_values("r_voxel")
    L = diffusion_length_voxels(params_hi)
    r_hi = g_hi["r_voxel"].values.astype(float)
    c_hi = g_hi["mean_conc"].values
    # Anchor at r=1 (avoid singularity at r=0; K_0 diverges there).
    if np.any(r_hi == 1):
        anchor_c = float(c_hi[r_hi == 1][0])
        anchor_K = float(k0(1.0 / L))
        amp = anchor_c / anchor_K
        rr = np.linspace(0.5, r_hi.max(), 200)
        analytic = amp * k0(rr / L)
        for ax in (axL, axR):
            ax.plot(rr, analytic, lw=1.2, ls="--", color="black",
                    label=rf"$A\cdot K_0(r/L)$, $L$={L:.2f} vox")

    for ax, title in ((axL, "linear"), (axR, "log")):
        ax.set_xlabel("radial distance r (voxels)")
        ax.set_ylabel("mean O₂ [mM]")
        ax.set_title(f"Radial profile at final step ({title} scale)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("Vascular O₂ + cancer uptake — substep convergence toward K₀",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_xline_convergence(variant_data, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for variant in VARIANTS:
        ts, rd, xl, params = variant_data[variant]
        final_step = int(xl["step"].max())
        g = xl[xl["step"] == final_step].sort_values("x")
        cx = int(params["vessel_x"])
        ax.plot(g["x"].values - cx, g["conc"].values, lw=1.8, marker="o", ms=3,
                color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
    ax.set_xlabel(r"$x - x_{\mathrm{vessel}}$ (voxels)")
    ax.set_ylabel("O₂ [mM]")
    ax.set_title("x-line cross-section through vessel column (final step)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_radial_ratio_bar(variant_data, out_path: Path):
    """Bar chart: C(r=1)/C(r=10) per variant. Monotone increase = P1."""
    names = []
    ratios = []
    colors = []
    for variant in VARIANTS:
        _, rd, _, _ = variant_data[variant]
        final = int(rd["step"].max())
        g = rd[rd["step"] == final]
        try:
            c1 = float(g[g["r_voxel"] == 1]["mean_conc"].iloc[0])
            c10 = float(g[g["r_voxel"] == 10]["mean_conc"].iloc[0])
            ratios.append(c1 / c10 if c10 > 0 else np.nan)
        except Exception:
            ratios.append(np.nan)
        names.append(VARIANT_LABELS[variant])
        colors.append(VARIANT_COLORS[variant])

    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(names, ratios, color=colors, alpha=0.85, edgecolor="black")
    for b, r in zip(bars, ratios):
        if np.isfinite(r):
            ax.text(b.get_x() + b.get_width() / 2, r + 0.05, f"{r:.2f}×",
                    ha="center", va="bottom", fontsize=9)
    ax.axhline(1.5, color="tab:gray", ls=":", lw=1.0,
               label="P3 threshold (sub36 must be < 1.5)")
    ax.set_ylabel(r"radial ratio $C(r=1)/C(r=10)$")
    ax.set_title("Radial ratio at final step — higher = sharper gradient")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=10, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return ratios


def plot_timeseries_vessel(variant_data, out_path: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for variant in VARIANTS:
        ts, _, _, _ = variant_data[variant]
        ax1.plot(ts["step"], ts["vessel_conc"], lw=1.8, marker="o", ms=3,
                 color=VARIANT_COLORS[variant], label=VARIANT_LABELS[variant])
        ax2.plot(ts["step"], ts["min_conc"], lw=1.5, ls="-", marker="v", ms=3,
                 color=VARIANT_COLORS[variant], alpha=0.9,
                 label=f"{VARIANT_LABELS[variant]} (min)")
    ax1.set_ylabel("vessel-voxel O₂ [mM]")
    ax1.set_title("Vessel voxel concentration over time")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel("ABM step")
    ax2.set_ylabel("grid minimum O₂ [mM]")
    ax2.set_title("Grid-min O₂ (sub36 artifact: fully drained each substep)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def evaluate_pass_criteria(variant_data, ratios):
    """Print pass/fail for P1, P2, P3. Returns overall ok flag."""
    ok = True
    print()
    print("=" * 68)
    print("Pass criteria")
    print("=" * 68)

    # P1: monotonic convergence of radial ratio
    mono = all(
        np.isfinite(ratios[i]) and np.isfinite(ratios[i + 1])
        and ratios[i + 1] >= ratios[i] - 1e-6
        for i in range(len(ratios) - 1)
    )
    p1_sym = "PASS" if mono else "FAIL"
    print(f"  [P1] Radial ratio monotone w/ substep count: {ratios} → {p1_sym}")
    ok &= mono

    # P2: per-variant K_0(r/L) fit error at r ∈ {3,5,7} decreases monotonically
    # with substep count (convergence direction). Amplitude anchored at r=1
    # per-variant so only shape is compared.
    test_r = [3, 5, 7]
    mean_errs = []
    for variant in VARIANTS:
        _, rd, _, params = variant_data[variant]
        final = int(rd["step"].max())
        g = rd[rd["step"] == final].set_index("r_voxel")
        L = diffusion_length_voxels(params)
        try:
            c1 = float(g.loc[1, "mean_conc"])
            amp = c1 / float(k0(1.0 / L))
            errs = []
            for r in test_r:
                c_obs = float(g.loc[r, "mean_conc"])
                c_ana = amp * float(k0(r / L))
                rel = abs(c_obs - c_ana) / max(c_ana, 1e-30)
                errs.append(rel)
            mean_err = float(np.mean(errs))
            mean_errs.append(mean_err)
            print(f"       {variant}: L={L:.2f} vox, mean |rel err| @ r∈{{3,5,7}}"
                  f" = {mean_err*100:.1f}%")
        except KeyError:
            mean_errs.append(float("inf"))
            print(f"       {variant}: radial bins 3,5,7 not all present")
    p2 = all(
        np.isfinite(mean_errs[i]) and np.isfinite(mean_errs[i + 1])
        and mean_errs[i + 1] <= mean_errs[i] + 1e-6
        for i in range(len(mean_errs) - 1)
    )
    p2_sym = "PASS" if p2 else "FAIL"
    print(f"  [P2] K_0 fit error monotone ↓ with substeps: {p2_sym}")
    ok &= p2

    # P3: sub36 radial ratio below 1.5 (documents artifact)
    r_sub36 = ratios[0]
    p3 = np.isfinite(r_sub36) and r_sub36 < 1.5
    p3_sym = "PASS" if p3 else "FAIL"
    print(f"  [P3] sub36 ratio < 1.5 (artifact documented): "
          f"{r_sub36:.2f} → {p3_sym}")
    ok &= p3

    print("=" * 68)
    print(f"  OVERALL: {'PASS' if ok else 'FAIL'}")
    print("=" * 68)
    return ok


def main():
    out_root = (Path(sys.argv[1]) if len(sys.argv) > 1
                else Path(__file__).parent / "outputs")
    if not out_root.exists():
        print(f"[error] output dir not found: {out_root}", file=sys.stderr)
        print("Run the tests first:", file=sys.stderr)
        for v in VARIANTS:
            print(f"  ./build/bin/pdac_test {v}", file=sys.stderr)
        sys.exit(1)

    missing = [v for v in VARIANTS if not (out_root / v).exists()]
    if missing:
        print(f"[error] missing variant output dirs: {missing}", file=sys.stderr)
        sys.exit(1)

    variant_data = {v: load_variant(out_root, v) for v in VARIANTS}
    plot_radial_convergence(variant_data, out_root / "radial_convergence.png")
    plot_xline_convergence(variant_data,  out_root / "xline_convergence.png")
    ratios = plot_radial_ratio_bar(variant_data, out_root / "radial_ratio_bar.png")
    plot_timeseries_vessel(variant_data,  out_root / "timeseries_vessel.png")

    ok = evaluate_pass_criteria(variant_data, ratios)
    print(f"\n[done] 4 figures written to {out_root}/")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
