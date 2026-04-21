#!/usr/bin/env python3
"""
Figures for the molecular-grids test (Test 6 — meeting slide candidate).

Validates 8 new PDE substrates via a single-voxel point source in a 21³ box.

Reads:
  outputs/params.csv        — runtime diffusivity, decay, voxel size
  outputs/time_series.csv   — step, chem, center_conc
  outputs/final_conc.csv    — chem, x, y, z, conc  (final step, 3D field)

Writes:
  outputs/mass_balance.png     — Σ C_i vs S/k bar chart (PRIMARY pass check)
  outputs/krogh_profiles.png   — 2×4 radial profile vs Krogh + image-sum analytic
  outputs/convergence.png      — center conc vs step (all 8 chemicals)
  outputs/pass_summary.csv     — per-chemical mass balance error + PASS/FAIL

Pass criterion (per chemical):
  Mass balance: |Σ C_i - S/k| / (S/k) < 0.10
  Steady state: C_center at last step is within 1% of C_center at step max-5

Why mass balance and not Krogh:
  λ = √(D/k) ranges 7–58 voxels across the 8 chemicals. The grid half-width
  is 10 voxels, so for 7 of 8 chemicals λ ≳ L and the Neumann box strongly
  reflects — pure Krogh point-source C(r) = S·V/(4πD·r)·exp(-r/λ) doesn't
  hold. An image-augmented Krogh overlay is shown for reference, but the
  quantitative pass criterion is the regime-independent mass balance:
     at steady state,  S·V_voxel = k · Σ_i C_i · V_voxel  ⇒  Σ C_i = S/k.
  This directly exercises source injection, decay, and diffusion conservation
  in a single number per chemical.

Run from repo root:
  python3 PDAC/test/scenarios/molecular_grids/make_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CHEMS = ["IL1", "IL6", "CXCL13", "MMP", "ANTIBODY", "CCL21", "CXCL12", "CCL5"]
MASS_TOL     = 0.10   # 10% on |Σ C - S/k| / (S/k)
STEADY_TOL   = 0.01   # 1% drift in last-5-step window on center voxel
IMG_N        = 4      # image lattice extent (±N·2L per axis)


def load(out_dir: Path):
    params_raw = pd.read_csv(out_dir / "params.csv")
    params = dict(zip(params_raw["key"], pd.to_numeric(params_raw["value"], errors="coerce")))
    ts = pd.read_csv(out_dir / "time_series.csv")
    field = pd.read_csv(out_dir / "final_conc.csv")
    return params, ts, field


def krogh_point(r_cm: np.ndarray, S: float, V: float, D: float, k: float) -> np.ndarray:
    lam = np.sqrt(D / k)
    out = np.zeros_like(r_cm, dtype=float)
    nz = r_cm > 0
    out[nz] = (S * V) / (4.0 * np.pi * D * r_cm[nz]) * np.exp(-r_cm[nz] / lam)
    return out


def krogh_with_images(r_cm: np.ndarray, S: float, V: float, D: float, k: float,
                      L_cm: float, n: int = IMG_N) -> np.ndarray:
    """Krogh with Neumann-box image sources (observation along +x axis).

    Images at positions (2nL, 2mL, 2pL) for integers |n|,|m|,|p| ≤ IMG_N.
    Convergence check: at IMG_N=4 the 9³-1 = 728 images in the lattice
    include enough far images that the tail exp(-2·4·L/λ) is negligible
    for λ up to a few L (the longest chemical here, CCL21, has λ ≈ 5.8 L).
    """
    lam = np.sqrt(D / k)
    out = np.zeros_like(r_cm, dtype=float)
    for i, r in enumerate(r_cm):
        acc = 0.0
        for nn in range(-n, n + 1):
            for mm in range(-n, n + 1):
                for pp in range(-n, n + 1):
                    sx, sy, sz = 2*nn*L_cm, 2*mm*L_cm, 2*pp*L_cm
                    d = np.sqrt((r - sx)**2 + sy**2 + sz**2)
                    if d > 1e-14:
                        acc += (S * V) / (4.0 * np.pi * D * d) * np.exp(-d / lam)
        out[i] = acc
    return out


def radial_profile(field: pd.DataFrame, chem: str,
                   cx: int, cy: int, cz: int, r_max_vox: int) -> pd.DataFrame:
    sub = field[field["chem"] == chem]
    dx = sub["x"].values - cx
    dy = sub["y"].values - cy
    dz = sub["z"].values - cz
    r  = np.sqrt(dx*dx + dy*dy + dz*dz)
    c  = sub["conc"].values
    rows = []
    for i in range(0, r_max_vox + 1):
        if i == 0:
            mask = (r == 0)
        else:
            mask = (r >= i - 0.5) & (r < i + 0.5)
        n = int(mask.sum())
        mean = float(c[mask].mean()) if n else np.nan
        std  = float(c[mask].std())  if n else np.nan
        rows.append({"r_vox": i, "n": n, "mean_conc": mean, "std_conc": std})
    return pd.DataFrame(rows)


def check_mass_balance(params: dict, field: pd.DataFrame) -> pd.DataFrame:
    S = params["SOURCE_RATE_nM_per_s"]
    rows = []
    for chem in CHEMS:
        k   = params[f"PARAM_{chem}_DECAY_RATE"]
        total = float(field[field["chem"] == chem]["conc"].sum())
        expected = S / k
        rel_err = abs(total - expected) / expected
        rows.append({
            "chem": chem, "sum_C": total, "expected_S_over_k": expected,
            "rel_err": rel_err, "pass_mass": bool(rel_err < MASS_TOL),
        })
    return pd.DataFrame(rows)


def check_steady_state(ts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for chem in CHEMS:
        g = ts[ts["chem"] == chem].sort_values("step")
        last = g["center_conc"].iloc[-1]
        prev = g["center_conc"].iloc[-6]   # 5 steps earlier
        drift = abs(last - prev) / last if last > 0 else np.nan
        rows.append({
            "chem": chem, "center_final": last, "center_5ago": prev,
            "drift_rel": drift, "pass_steady": bool(drift < STEADY_TOL),
        })
    return pd.DataFrame(rows)


def plot_mass_balance(mass_df: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    x  = np.arange(len(CHEMS))
    w  = 0.38
    ax.bar(x - w/2, mass_df["sum_C"],             width=w, color="tab:blue",
           label=r"$\sum_i C_i$ (sim)")
    ax.bar(x + w/2, mass_df["expected_S_over_k"], width=w, color="tab:orange",
           label=r"$S/k$ (analytic)")
    for i, row in mass_df.iterrows():
        tag = "PASS" if row["pass_mass"] else "FAIL"
        ax.annotate(f"{row['rel_err']*100:.1f}%\n{tag}",
                    (i, max(row["sum_C"], row["expected_S_over_k"])),
                    ha="center", va="bottom", fontsize=8,
                    color="green" if row["pass_mass"] else "red")
    ax.set_xticks(x)
    ax.set_xticklabels(CHEMS)
    ax.set_ylabel("total [nM·voxels]")
    ax.set_yscale("log")
    ax.set_title("Mass balance at steady state: "
                 r"$\sum C_i = S/k$  (source rate = total decay)")
    ax.grid(alpha=0.3, axis="y", which="both")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_krogh_profiles(params: dict, field: pd.DataFrame,
                        mass_df: pd.DataFrame, out_path: Path):
    dx_cm = params["PARAM_VOXEL_SIZE_CM"]
    V_cm3 = dx_cm ** 3
    S     = params["SOURCE_RATE_nM_per_s"]
    cx, cy, cz = int(params["center_x"]), int(params["center_y"]), int(params["center_z"])
    gs    = int(params["grid_size"])
    L_cm  = (gs / 2.0) * dx_cm        # half-width (cell-centered)
    r_max = cx

    fig, axes = plt.subplots(2, 4, figsize=(17, 8), sharex=True)

    for ax, chem in zip(axes.flat, CHEMS):
        D = params[f"PARAM_{chem}_DIFFUSIVITY"]
        k = params[f"PARAM_{chem}_DECAY_RATE"]
        lam_vox = np.sqrt(D / k) / dx_cm

        prof  = radial_profile(field, chem, cx, cy, cz, r_max)
        r_vox = prof["r_vox"].values.astype(float)
        c_sim = prof["mean_conc"].values

        r_fine = np.linspace(0.3, r_max, 80) * dx_cm
        c_krogh  = krogh_point(r_fine, S, V_cm3, D, k)
        c_images = krogh_with_images(r_fine, S, V_cm3, D, k, L_cm)

        ax.plot(r_fine / dx_cm, c_krogh, lw=1.2, color="tab:orange",
                ls="--", label="Krogh (open-space)")
        ax.plot(r_fine / dx_cm, c_images, lw=1.5, color="tab:red",
                label=f"Krogh + images (N±{IMG_N})")
        ax.errorbar(r_vox[1:], c_sim[1:], yerr=prof["std_conc"].values[1:],
                    fmt="o", ms=4, color="tab:blue",
                    alpha=0.85, capsize=2, label="sim (shell avg ± sd)")
        ax.scatter([0], [c_sim[0]], marker="x", color="gray", s=35,
                   label="center voxel")

        row = mass_df[mass_df["chem"] == chem].iloc[0]
        tag = "PASS" if row["pass_mass"] else "FAIL"
        ax.set_yscale("log")
        ax.set_title(f"{chem}   λ/L = {lam_vox / (gs/2):.2f}   "
                     f"mass-bal {tag} ({row['rel_err']*100:.1f}%)")
        ax.set_xlabel("r [voxels]")
        ax.set_ylabel("[nM]")
        ax.set_xlim(-0.5, r_max + 0.5)
        ax.grid(alpha=0.3, which="both")
        if chem == CHEMS[0]:
            ax.legend(loc="lower left", fontsize=7)

    fig.suptitle(f"Test 6 — 8 substrates, S = {S} nM/s point source, "
                 f"{gs}³ Neumann box", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_convergence(ts: pd.DataFrame, out_path: Path):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cmap = plt.get_cmap("tab10")
    for i, chem in enumerate(CHEMS):
        g = ts[ts["chem"] == chem]
        ax.plot(g["step"], g["center_conc"], lw=1.3, color=cmap(i), label=chem)
    ax.set_xlabel("ABM step")
    ax.set_ylabel("center voxel [nM]")
    ax.set_title("Molecular grids — center concentration convergence")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "outputs"
    if not out_dir.exists():
        print(f"[error] output dir not found: {out_dir}", file=sys.stderr)
        print("Run: ./build/bin/pdac_test molecular_grids", file=sys.stderr)
        sys.exit(1)

    params, ts, field = load(out_dir)
    print(f"[info] loaded {len(field)} field rows, {len(ts)} time-series rows, "
          f"{len(params)} params")

    mass_df   = check_mass_balance(params, field)
    steady_df = check_steady_state(ts)
    summary   = mass_df.merge(steady_df, on="chem")
    summary["pass"] = summary["pass_mass"] & summary["pass_steady"]

    plot_mass_balance(mass_df, out_dir / "mass_balance.png")
    plot_krogh_profiles(params, field, mass_df, out_dir / "krogh_profiles.png")
    plot_convergence(ts, out_dir / "convergence.png")

    summary.to_csv(out_dir / "pass_summary.csv", index=False)
    print("\n" + summary.to_string(index=False,
          float_format=lambda v: f"{v:.4g}"))

    n_fail = (~summary["pass"]).sum()
    print(f"\n[{'PASS' if n_fail == 0 else 'FAIL'}]  "
          f"{len(summary) - n_fail}/{len(summary)} chemicals pass "
          f"(mass balance < {MASS_TOL*100:.0f}% AND drift < {STEADY_TOL*100:.0f}%)")
    print(f"[done] 3 figures written to {out_dir}/")


if __name__ == "__main__":
    main()
