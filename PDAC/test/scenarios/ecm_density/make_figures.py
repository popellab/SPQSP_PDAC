#!/usr/bin/env python3
"""
Representative figures for the ecm_density test.

Reads:
  PDAC/test/scenarios/ecm_density/outputs/ecm_time_series.csv   (step, center_density, center_crosslink, max_density, mmp_value)
  PDAC/test/scenarios/ecm_density/outputs/ecm_xline.csv         (step, x, density, crosslink) — snapshot steps only
  PDAC/test/scenarios/ecm_density/outputs/ecm_params.csv        (key, value) — runtime parameter values

Writes four PNGs:
  crosslink_curve.png    — center_crosslink(t) vs analytic 1-(1-k_lox·dt)^n
  density_radial.png     — density profile along x at snapshot steps, with Gaussian-shape overlay
  mmp_degradation.png    — center_density(t) zoomed around MMP injection, analytic two-regime fit
  timeseries_overview.png — full 300-step trajectory: center density + crosslink + MMP gate

Run from repo root or the scenario dir:
  python3 PDAC/test/scenarios/ecm_density/make_figures.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load(out_dir: Path):
    ts = pd.read_csv(out_dir / "ecm_time_series.csv")
    xl = pd.read_csv(out_dir / "ecm_xline.csv")
    pr = pd.read_csv(out_dir / "ecm_params.csv")
    params = {row["key"]: float(row["value"]) for _, row in pr.iterrows()}
    return ts, xl, params


def plot_crosslink_curve(ts: pd.DataFrame, params: dict, out_path: Path):
    k_lox = params["PARAM_ECM_CROSSLINK_RATE"]
    dt = params["PARAM_SEC_PER_SLICE"]
    per_step = k_lox * dt  # forward-Euler increment coefficient
    # Discrete analytic: c(n) = 1 - (1 - k·dt)^n  from c_{n+1} = c_n + k·dt·(1 - c_n).
    # The kernel applies ONE update per ABM step. The callback dumps crosslink
    # AFTER the step completes, so empirical row `step=N` corresponds to n=N+1
    # analytic updates.
    n = ts["step"].values.astype(float) + 1.0
    c_analytic = 1.0 - (1.0 - per_step) ** n

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(ts["step"], ts["center_crosslink"], lw=1.8, color="tab:blue",
            label="empirical (center voxel)")
    ax.plot(n, c_analytic, lw=1.2, ls="--", color="tab:red",
            label=rf"analytic: $1-(1-k_{{\mathrm{{lox}}}}\Delta t)^n$,"
                  rf" $k_{{\mathrm{{lox}}}}\Delta t$={per_step:.4f}")
    ax.set_xlabel("ABM step")
    ax.set_ylabel("center voxel crosslink")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("ECM — crosslink accumulation at center (fib > 0 → accrues every step)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    # Per-step empirical vs analytic error
    err = np.abs(ts["center_crosslink"].values - c_analytic)
    print(f"[crosslink] max |err| = {err.max():.2e}, mean |err| = {err.mean():.2e}")


def plot_density_radial(xl: pd.DataFrame, params: dict, out_path: Path):
    cx = int(params["center_x"])
    variance = params["PARAM_FIB_ECM_VARIANCE"]
    gs = int(params["grid_size"])
    # Pick pre-MMP snapshot steps that show the build-up shape.
    pre_mmp_steps = [s for s in (0, 10, 50, 100, 149) if s in set(xl["step"].unique())]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    cmap = plt.get_cmap("viridis")
    for i, step in enumerate(pre_mmp_steps):
        g = xl[xl["step"] == step].sort_values("x")
        r = g["x"].values - cx
        ax.plot(r, g["density"].values, lw=1.5, marker="o", ms=3,
                color=cmap(i / max(1, len(pre_mmp_steps) - 1)),
                label=f"step {step}")
    # Overlay analytic Gaussian shape (σ² = PARAM_FIB_ECM_VARIANCE, peak normalized to
    # empirical step-149 center density to illustrate shape, not magnitude).
    peak_ref_step = pre_mmp_steps[-1] if pre_mmp_steps else 0
    peak_empirical = xl[(xl["step"] == peak_ref_step) & (xl["x"] == cx)]["density"].values
    if len(peak_empirical):
        peak = float(peak_empirical[0])
        rr = np.linspace(-(gs // 2), gs // 2, 201)
        gauss_shape = peak * np.exp(-rr * rr / (2.0 * variance))
        ax.plot(rr, gauss_shape, lw=1.0, ls=":", color="black",
                label=rf"Gaussian, $\sigma^2$={variance}, peak={peak:.1f}")
    ax.set_xlabel(r"$x - x_{\mathrm{center}}$ (voxels)")
    ax.set_ylabel("ECM density [nmol/mL]")
    ax.set_title("ECM — x-line density profile (pre-MMP deposition phase)")
    ax.set_xlim(-8, 8)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mmp_degradation(ts: pd.DataFrame, params: dict, out_path: Path):
    inject_step = int(params["MMP_INJECT_STEP"])
    mmp_val = params["MMP_INJECT_VALUE"]
    k_mmp = params["PARAM_ECM_MMP_DEGRADE_RATE"]
    alpha = params["PARAM_ECM_CROSSLINK_RESISTANCE"]
    dt = params["PARAM_SEC_PER_SLICE"]

    # One-step-post-injection: verify instantaneous degradation rate
    # ρ(n+1) - ρ(n) ≈ -k_mmp · MMP · ρ(n) · dt / (1 + α·c) + deposition
    # At the MMP kick-in step, we have a clean before/after pair:
    pre  = ts[ts["step"] == inject_step].iloc[0]
    post = ts[ts["step"] == inject_step + 1].iloc[0]
    c_pre = pre["center_crosslink"]
    rho_pre = pre["center_density"]
    rho_post = post["center_density"]
    delta_obs = rho_post - rho_pre
    degrade_pred = k_mmp * mmp_val * rho_pre * dt / (1.0 + alpha * c_pre)

    # To estimate deposition at step inject_step (just before MMP), look at the
    # previous no-MMP step.
    prev = ts[ts["step"] == inject_step - 1].iloc[0]
    depo_pre_approx = rho_pre - prev["center_density"]  # positive: net growth w/o MMP
    delta_pred = depo_pre_approx - degrade_pred

    fig, ax = plt.subplots(figsize=(8, 4.6))
    post_mmp = ts[ts["step"] >= inject_step - 5].copy()
    ax.plot(post_mmp["step"], post_mmp["center_density"], lw=1.8, color="tab:blue",
            label="center density (empirical)")
    ax.axvline(inject_step + 0.5, color="tab:red", ls="--", lw=1.0,
               label=f"MMP injected (step {inject_step}→{inject_step+1})")

    # Annotate the one-step observation
    ax.annotate(
        rf"$\Delta\rho$ observed = {delta_obs:+.2f}"
        + "\n"
        + rf"$\Delta\rho$ predicted = deposition({depo_pre_approx:+.2f}) − "
        + rf"$k_{{\mathrm{{mmp}}}}\,M\,\rho\,\Delta t/(1+\alpha c)$ = "
        + rf"{depo_pre_approx:+.2f} − {degrade_pred:+.2f} = {delta_pred:+.2f}",
        xy=(inject_step + 1, rho_post),
        xytext=(inject_step + 30, rho_pre - 40),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=0.8),
    )

    ax.set_xlabel("ABM step")
    ax.set_ylabel("center density [nmol/mL]")
    ax.set_title(
        "ECM — MMP degradation, center voxel\n"
        rf"(MMP={mmp_val:g}, $k_{{\mathrm{{mmp}}}}$={k_mmp:g}, $\alpha$={alpha:g})"
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    err = abs(delta_obs - delta_pred)
    print(f"[mmp kick-in] observed Δρ = {delta_obs:+.3f}, predicted = {delta_pred:+.3f}, "
          f"|err| = {err:.3f}")


def plot_timeseries_overview(ts: pd.DataFrame, params: dict, out_path: Path):
    inject_step = int(params["MMP_INJECT_STEP"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 6.5), sharex=True)

    ax1.plot(ts["step"], ts["center_density"], lw=1.5, color="tab:blue",
             label="center density")
    ax1.plot(ts["step"], ts["max_density"], lw=0.8, ls=":", color="tab:cyan",
             label="max density (any voxel)")
    ax1.axvline(inject_step + 0.5, color="tab:red", ls="--", lw=1.0,
                label=f"MMP injected (step {inject_step+1})")
    ax1.set_ylabel("ECM density [nmol/mL]")
    ax1.set_title("ECM density + crosslink over full run")
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(alpha=0.3)

    ax2b = ax2.twinx()
    ax2.plot(ts["step"], ts["center_crosslink"], lw=1.5, color="tab:purple",
             label="crosslink (center)")
    ax2b.plot(ts["step"], ts["mmp_value"], lw=1.2, color="tab:orange",
              label="MMP value")
    ax2.axvline(inject_step + 0.5, color="tab:red", ls="--", lw=1.0)
    ax2.set_xlabel("ABM step")
    ax2.set_ylabel("crosslink", color="tab:purple")
    ax2b.set_ylabel(r"MMP [$\mu$M]", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:purple")
    ax2b.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0.0, 1.05)
    ax2.grid(alpha=0.3)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "outputs"
    if not out_dir.exists():
        print(f"[error] output dir not found: {out_dir}", file=sys.stderr)
        print("Run the test first: ./build/bin/pdac_test ecm_density", file=sys.stderr)
        sys.exit(1)

    ts, xl, params = load(out_dir)
    print(f"[info] {len(ts)} time-series rows, {len(xl)} x-line rows, "
          f"{len(params)} params")

    plot_crosslink_curve(ts, params, out_dir / "crosslink_curve.png")
    plot_density_radial(xl, params, out_dir / "density_radial.png")
    plot_mmp_degradation(ts, params, out_dir / "mmp_degradation.png")
    plot_timeseries_overview(ts, params, out_dir / "timeseries_overview.png")
    print(f"[done] 4 figures written to {out_dir}/")


if __name__ == "__main__":
    main()
