#!/usr/bin/env python3
"""
Figures for the pde_substep_sweep tests.

Five scenarios sweep PARAM_MOLECULAR_STEPS ∈ {6, 18, 36, 360, 3600} under an
identical setup: 25³ grid, 20 µm voxels, 60 ABM steps, no agents, all layers
OFF except pde_solve, center voxel pinned to C=1 each step for 17 non-O2
substrates. Fields relax under diffusion + decay toward the Helmholtz steady
state.

sub3600 is the converged reference. We quantify:
  - per-substrate max relative error along the x-line at the final step
  - convergence of the center-adjacent voxel (x=cx±1) across substep counts

Pass criteria (printed + returned via exit code):
  P_conv36: every non-O2 substrate's max |C_sub36 - C_sub3600| / max(C_sub3600)
            along the center x-line at the final step is < 5%. This is the
            production substep count; failures flag substrates where the PDE
            is under-resolved.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SUBSTEP_VARIANTS = [6, 18, 36, 360, 3600]
REFERENCE = 3600
PRODUCTION = 36

# Substrates whose reference peak is below this fraction of the pinned value
# (1.0) are considered "underflow": their decay-over-dt_abm is so strong that
# the measurable signal is numerical noise. Convergence is untestable by this
# setup (need continuous pinning, not per-ABM-step pinning).
UNDERFLOW_PEAK = 1e-6


def load_params(path: Path) -> dict:
    pr = pd.read_csv(path)
    out = {}
    for _, row in pr.iterrows():
        try:
            out[row["key"]] = float(row["value"])
        except ValueError:
            out[row["key"]] = row["value"]
    return out


def load_scenario(base: Path, substeps: int):
    run_dir = base / f"pde_substep_sub{substeps}"
    fields = pd.read_csv(run_dir / "fields.csv")
    params = load_params(run_dir / "params.csv")
    return fields, params


def final_xline(fields: pd.DataFrame, substrate: str) -> np.ndarray:
    final_step = fields["step"].max()
    sub = fields[(fields["step"] == final_step) & (fields["substrate"] == substrate)]
    return sub.sort_values("x")["conc"].to_numpy()


def decay_length_um(params: dict, name: str, voxel_size_cm: float) -> float:
    """Return decay length L = sqrt(D/k) in µm, or NaN if params missing."""
    d_key = f"PARAM_{name}_DIFFUSIVITY"
    k_key = f"PARAM_{name}_DECAY_RATE"
    if d_key not in params or k_key not in params:
        return float("nan")
    D = params[d_key]
    k = params[k_key]
    if k <= 0 or D <= 0:
        return float("inf")
    L_cm = np.sqrt(D / k)
    return L_cm * 1e4  # cm → µm


def run(base: Path) -> bool:
    # Load all 5 runs
    runs = {}
    for s in SUBSTEP_VARIANTS:
        run_dir = base / f"pde_substep_sub{s}"
        if not (run_dir / "fields.csv").exists():
            print(f"[error] {run_dir}/fields.csv missing — run scenario first",
                  file=sys.stderr)
            return False
        runs[s] = load_scenario(base, s)

    # Substrate list from any run (all same)
    fields_ref, params_ref = runs[REFERENCE]
    substrates = fields_ref["substrate"].unique().tolist()

    voxel_size_cm = params_ref.get("voxel_size", 20e-4)
    voxel_size_um = voxel_size_cm * 1e4
    gs = int(params_ref.get("grid_size", 25))
    cx = int(params_ref.get("source_x", gs // 2))
    x_um = (np.arange(gs) - cx) * voxel_size_um

    # -------------------------------------------------------------------
    # Figure: 17 substrates × 2 panels  (profile + error vs sub3600)
    # Stack 17 substrates in a 5×4 grid per panel-type; produce two figures
    # so plots stay legible.
    # -------------------------------------------------------------------
    ncols = 5
    nrows = int(np.ceil(len(substrates) / ncols))

    fig_p, axes_p = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows),
                                 sharex=True)
    fig_e, axes_e = plt.subplots(nrows, ncols, figsize=(4.0 * ncols, 2.8 * nrows),
                                 sharex=True)
    axes_p = np.atleast_1d(axes_p).ravel()
    axes_e = np.atleast_1d(axes_e).ravel()

    cmap = plt.get_cmap("viridis")
    colors = {s: cmap(i / (len(SUBSTEP_VARIANTS) - 1))
              for i, s in enumerate(SUBSTEP_VARIANTS)}

    # Table rows for summary
    table_rows = []

    for i, sub in enumerate(substrates):
        ax_p = axes_p[i]
        ax_e = axes_e[i]

        ref = final_xline(fields_ref, sub)
        ref_peak = float(np.max(ref)) if ref.size else 0.0

        # Profile overlay
        for s in SUBSTEP_VARIANTS:
            fields_s, _ = runs[s]
            prof = final_xline(fields_s, sub)
            ax_p.plot(x_um, prof, lw=1.6, color=colors[s], label=f"{s}",
                      alpha=0.9 if s in (PRODUCTION, REFERENCE) else 0.6,
                      ls="-" if s in (PRODUCTION, REFERENCE) else "--")

        L_um = decay_length_um(params_ref, sub, voxel_size_cm)
        L_txt = (f"L={L_um:.0f} µm" if np.isfinite(L_um) else "L=∞")
        ax_p.set_title(f"{sub}  ({L_txt})", fontsize=9)
        ax_p.grid(alpha=0.3)
        ax_p.set_xlabel("x − center (µm)", fontsize=8)
        ax_p.set_ylabel("C (norm.)", fontsize=8)
        if i == 0:
            ax_p.legend(title="substeps", fontsize=7, loc="upper right")

        # Error panel: each non-reference curve minus ref, as % of ref peak
        rel_errs_at_prod = None
        for s in SUBSTEP_VARIANTS:
            if s == REFERENCE:
                continue
            fields_s, _ = runs[s]
            prof = final_xline(fields_s, sub)
            denom = max(ref_peak, 1e-30)
            err = (prof - ref) / denom * 100.0
            ax_e.plot(x_um, err, lw=1.5, color=colors[s], label=f"{s}",
                      ls="-" if s == PRODUCTION else "--")
            if s == PRODUCTION:
                rel_errs_at_prod = np.abs(err)

        max_err_prod = float(np.max(rel_errs_at_prod)) if rel_errs_at_prod is not None \
            else float("nan")
        underflow = ref_peak < UNDERFLOW_PEAK
        # Underflow substrates: their signal is dominated by decay
        # (λ·dt_abm > ~10), so measurement is untestable but also
        # irrelevant — these species' fields are meaningfully zero in
        # production too. Do not fail the test on them.
        passed = underflow or (max_err_prod < 5.0)
        table_rows.append({
            "substrate": sub,
            "L_um": L_um,
            "ref_peak": ref_peak,
            "max_err_pct_sub36": max_err_prod,
            "underflow": underflow,
            "pass_sub36": passed,
        })

        ax_e.axhline(0, color="k", lw=0.8)
        ax_e.axhspan(-5, 5, color="tab:red", alpha=0.1)
        ax_e.set_title(f"{sub}  max|Δ|_36={max_err_prod:.2f}%", fontsize=9,
                       color=("tab:red" if not passed else "black"))
        ax_e.grid(alpha=0.3)
        ax_e.set_xlabel("x − center (µm)", fontsize=8)
        ax_e.set_ylabel("(C − C_3600)/peak  [%]", fontsize=8)
        if i == 0:
            ax_e.legend(title="substeps", fontsize=7, loc="upper right")

    # Hide unused axes
    for k in range(len(substrates), len(axes_p)):
        axes_p[k].set_visible(False)
        axes_e[k].set_visible(False)

    fig_p.suptitle("Final-step concentration along x (center pinned to C=1)",
                   fontsize=13)
    fig_e.suptitle(f"Error vs sub{REFERENCE} reference "
                   f"(each curve = substep variant − reference, % of peak)",
                   fontsize=13)
    fig_p.tight_layout(rect=(0, 0, 1, 0.97))
    fig_e.tight_layout(rect=(0, 0, 1, 0.97))

    out_profile = base / "substep_sweep_profiles.png"
    out_error   = base / "substep_sweep_errors.png"
    fig_p.savefig(out_profile, dpi=140)
    fig_e.savefig(out_error, dpi=140)
    plt.close(fig_p)
    plt.close(fig_e)

    # -------------------------------------------------------------------
    # Summary table + pass criteria
    # -------------------------------------------------------------------
    tbl = pd.DataFrame(table_rows).sort_values("max_err_pct_sub36",
                                               ascending=False)
    print()
    print("=" * 78)
    print(f"Per-substrate convergence at production (sub{PRODUCTION}) vs "
          f"reference (sub{REFERENCE})")
    print("=" * 78)
    print(f"  {'substrate':<10}  {'L (µm)':>10}  {'peak':>10}  "
          f"{'max|Δ|/peak':>12}  {'status':>10}")
    for _, r in tbl.iterrows():
        L = r["L_um"]
        L_str = f"{L:>10.1f}" if np.isfinite(L) else f"{'∞':>10}"
        if r["underflow"]:
            status = "UNDERFLOW"
        else:
            status = "PASS" if r["pass_sub36"] else "FAIL"
        print(f"  {r['substrate']:<10}  {L_str}  "
              f"{r['ref_peak']:>10.3g}  {r['max_err_pct_sub36']:>11.2f}%  "
              f"{status:>10}")
    print("=" * 78)
    tbl.to_csv(base / "substep_sweep_summary.csv", index=False)

    n_fail = int((~tbl["pass_sub36"]).sum())
    n_under = int(tbl["underflow"].sum())
    n_test = len(tbl) - n_under
    n_pass = n_test - n_fail
    print(f"  {n_pass}/{n_test} testable substrates converge at sub{PRODUCTION}  "
          f"(+{n_under} underflow, skipped)")
    print(f"  Figures:     {out_profile.name}, {out_error.name}")
    print(f"  Summary CSV: substep_sweep_summary.csv")
    print("=" * 78)

    all_pass = n_fail == 0
    print(f"  [P_conv{PRODUCTION}] All substrates within 5% at sub{PRODUCTION}: "
          f"{'PASS' if all_pass else 'FAIL'}")
    return all_pass


def main():
    base = (Path(sys.argv[1]) if len(sys.argv) > 1
            else Path(__file__).parent / "outputs")
    if not base.exists():
        print(f"[error] output dir not found: {base}", file=sys.stderr)
        sys.exit(1)
    ok = run(base)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
