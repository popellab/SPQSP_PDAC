"""Run 5 local preliminary presims with varied parameters.

Each parameter set is run coupled (qsp_abm) and uncoupled (abm_only) = 10 total.
Domain: 1000×1000×1 voxels at 20µm = 2cm × 2cm.

Usage::

    python -m PDAC.tools.run_local_tests --phase placement   # SA placement only
    python -m PDAC.tools.run_local_tests --phase sim          # presim only (needs placement)
    python -m PDAC.tools.run_local_tests --phase all          # both
    python -m PDAC.tools.run_local_tests --phase observables  # FunCN phi + abundance extraction

Placement files are reused across coupled/uncoupled runs (same seed → same layout).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNS_DIR = REPO / "runs" / "prelim_matching"
BASE_XML = REPO / "PDAC" / "sim" / "resource" / "param_all_test.xml"
SIM_BIN = REPO / "PDAC" / "sim" / "build" / "bin" / "pdac"

GRID_X, GRID_Y = 1000, 1000

# ── Parameter sets ──
# Split into three categories:
#   QSP_PARAMS   — elements found by tag name under <Param><QSP>
#   ABM_PARAMS   — elements found by name= attribute anywhere in XML
#   INT_PARAMS   — subset of ABM_PARAMS that must be written as integers
#
# Selection rationale (user directive Sep 24):
#   SAMPLE: patient-specific, high-uncertainty, movement/ECM blocking
#   FIXED:  PDE diffusivities, PDE decay rates

QSP_PARAMS = {
    "k_C1_growth",         # cancer QSP growth rate
    "k_vas_growth",        # vascular growth
    "k_vas_decay",         # vascular decay
    "k_iCAF_prolif",       # iCAF proliferation
    "k_myCAF_prolif",      # myCAF proliferation
    "k_col_myCAF_sec",     # collagen secretion (ECM deposition driver)
    "k_Mac_death",         # macrophage turnover
    "k_M2_pol",            # M2 polarization rate
    "k_IL10_sec",          # IL-10 secretion (immunosuppression)
}

ABM_PARAMS = {
    "PARAM_PROG_GROWTH_RATE",       # ABM cancer division rate
    "PARAM_SEED_CANCER_SEN_FRAC",   # proliferative index at seeding
    "PARAM_VAS_RC",                 # vascular carrying capacity
    # Movement / ECM blocking (user: "high uncertainty, should be calibrated")
    "PARAM_ECM_POROSITY_TCELL",     # T cell ECM exclusion threshold
    "PARAM_ECM_POROSITY_MAC",       # macrophage ECM exclusion threshold
    "PARAM_FIB_HIF_ECM_BOOST",     # hypoxic ECM boost (fibroblast collagen under hypoxia)
    "PARAM_CANCER_MOVE_STEPS",      # cancer cell speed (voxel-moves per ABM step)
    "PARAM_FIB_MOVE_STEPS",         # fibroblast speed
}

INT_PARAMS = {"PARAM_CANCER_MOVE_STEPS", "PARAM_FIB_MOVE_STEPS"}

RUN_PARAMS: list[dict[str, float]] = [
    {  # Run 1 — slow growth, permissive ECM (immune-accessible)
        "k_C1_growth": 0.004,
        "PARAM_PROG_GROWTH_RATE": 0.08,
        "PARAM_SEED_CANCER_SEN_FRAC": 0.3,
        "k_vas_growth": 0.08,
        "k_vas_decay": 0.002,
        "PARAM_VAS_RC": 2e-4,
        "k_iCAF_prolif": 0.03,
        "k_myCAF_prolif": 0.04,
        "k_col_myCAF_sec": 1.5e-8,
        "k_Mac_death": 0.02,
        "k_M2_pol": 0.03,
        "k_IL10_sec": 8e-13,
        "PARAM_ECM_POROSITY_TCELL": 0.25,   # low threshold = immune can enter dense ECM
        "PARAM_ECM_POROSITY_MAC": 0.20,
        "PARAM_FIB_HIF_ECM_BOOST": 1.5,
        "PARAM_CANCER_MOVE_STEPS": 3,
        "PARAM_FIB_MOVE_STEPS": 10,
    },
    {  # Run 2 — below-median growth, moderate ECM barrier
        "k_C1_growth": 0.007,
        "PARAM_PROG_GROWTH_RATE": 0.12,
        "PARAM_SEED_CANCER_SEN_FRAC": 0.5,
        "k_vas_growth": 0.15,
        "k_vas_decay": 0.004,
        "PARAM_VAS_RC": 4e-4,
        "k_iCAF_prolif": 0.06,
        "k_myCAF_prolif": 0.08,
        "k_col_myCAF_sec": 3e-8,
        "k_Mac_death": 0.035,
        "k_M2_pol": 0.06,
        "k_IL10_sec": 2e-12,
        "PARAM_ECM_POROSITY_TCELL": 0.35,
        "PARAM_ECM_POROSITY_MAC": 0.25,
        "PARAM_FIB_HIF_ECM_BOOST": 1.8,
        "PARAM_CANCER_MOVE_STEPS": 2,
        "PARAM_FIB_MOVE_STEPS": 8,
    },
    {  # Run 3 — median baseline (default priors)
        "k_C1_growth": 0.01,
        "PARAM_PROG_GROWTH_RATE": 0.15,
        "PARAM_SEED_CANCER_SEN_FRAC": 0.5,
        "k_vas_growth": 0.15,
        "k_vas_decay": 0.004,
        "PARAM_VAS_RC": 4e-4,
        "k_iCAF_prolif": 0.1,
        "k_myCAF_prolif": 0.15,
        "k_col_myCAF_sec": 5e-8,
        "k_Mac_death": 0.05,
        "k_M2_pol": 0.1,
        "k_IL10_sec": 3e-12,
        "PARAM_ECM_POROSITY_TCELL": 0.4,    # XML defaults
        "PARAM_ECM_POROSITY_MAC": 0.3,
        "PARAM_FIB_HIF_ECM_BOOST": 2.17,
        "PARAM_CANCER_MOVE_STEPS": 2,
        "PARAM_FIB_MOVE_STEPS": 8,
    },
    {  # Run 4 — above median, strong desmoplastic barrier
        "k_C1_growth": 0.015,
        "PARAM_PROG_GROWTH_RATE": 0.19,
        "PARAM_SEED_CANCER_SEN_FRAC": 0.7,
        "k_vas_growth": 0.25,
        "k_vas_decay": 0.008,
        "PARAM_VAS_RC": 6e-4,
        "k_iCAF_prolif": 0.2,
        "k_myCAF_prolif": 0.25,
        "k_col_myCAF_sec": 1e-7,
        "k_Mac_death": 0.08,
        "k_M2_pol": 0.2,
        "k_IL10_sec": 8e-12,
        "PARAM_ECM_POROSITY_TCELL": 0.5,    # high threshold = immune excluded
        "PARAM_ECM_POROSITY_MAC": 0.4,
        "PARAM_FIB_HIF_ECM_BOOST": 2.5,
        "PARAM_CANCER_MOVE_STEPS": 2,
        "PARAM_FIB_MOVE_STEPS": 6,
    },
    {  # Run 5 — fast growth, dense stroma, max ECM blocking
        "k_C1_growth": 0.025,
        "PARAM_PROG_GROWTH_RATE": 0.25,
        "PARAM_SEED_CANCER_SEN_FRAC": 0.4,
        "k_vas_growth": 0.4,
        "k_vas_decay": 0.012,
        "PARAM_VAS_RC": 8e-4,
        "k_iCAF_prolif": 0.35,
        "k_myCAF_prolif": 0.35,
        "k_col_myCAF_sec": 1.8e-7,
        "k_Mac_death": 0.14,
        "k_M2_pol": 0.35,
        "k_IL10_sec": 3e-11,
        "PARAM_ECM_POROSITY_TCELL": 0.55,
        "PARAM_ECM_POROSITY_MAC": 0.45,
        "PARAM_FIB_HIF_ECM_BOOST": 3.0,
        "PARAM_CANCER_MOVE_STEPS": 1,
        "PARAM_FIB_MOVE_STEPS": 5,
    },
]


def patch_xml(base_xml: Path, out_xml: Path, overrides: dict[str, float]) -> None:
    """Copy base XML, override specified QSP and ABM parameters."""
    tree = ET.parse(base_xml)
    root = tree.getroot()

    for name, val in overrides.items():
        found = False
        if name in QSP_PARAMS:
            for elem in root.iter(name):
                elem.text = f"{val:.6g}"
                found = True
                break
        if name in ABM_PARAMS:
            fmt = f"{int(val)}" if name in INT_PARAMS else f"{val:.6g}"
            for elem in root.iter():
                if elem.get("name") == name:
                    elem.text = fmt
                    found = True
                    break
        if not found:
            print(f"  WARNING: parameter '{name}' not found in XML")

    tree.write(out_xml, xml_declaration=True, encoding="unicode")


def run_placement(run_dir: Path, seed: int) -> Path:
    plc_path = run_dir / f"healthy_{GRID_X}.plc"
    if plc_path.exists():
        print(f"  Placement exists, skipping: {plc_path}")
        return plc_path

    cmd = [
        sys.executable, "-m", "PDAC.tools.init_2d_healthy",
        "--grid-x", str(GRID_X),
        "--grid-y", str(GRID_Y),
        "--seed", str(seed),
        "--n-steps", "5000000",
        "--output", str(plc_path),
    ]
    print(f"  Running SA placement (seed={seed})...")
    result = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True,
                            timeout=1800)
    if result.returncode != 0:
        print(f"  FAILED:\n{result.stderr[-1000:]}")
        sys.exit(1)
    print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
    return plc_path


def run_presim(run_dir: Path, plc_path: Path, xml_path: Path,
               mode: str, seed: int) -> Path:
    out_dir = run_dir / mode / "outputs"
    if (out_dir / "timing.csv").exists():
        print(f"  {mode} outputs exist, skipping")
        return out_dir

    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(SIM_BIN),
        "--healthy-2d", str(plc_path),
        "-gx", str(GRID_X), "-gy", str(GRID_Y),
        "--presim-mode", "qsp_abm" if mode == "coupled" else "abm_only",
        "--presim-steps", "500",
        "-G", "3",
        "-oi", "50",
        "-s", "0",
        "-p", str(xml_path),
        "--seed", str(seed),
        "--output-root", str(out_dir),
    ]

    print(f"  Running presim ({mode}, seed={seed})...")
    print(f"  CMD: {' '.join(cmd)}")

    result = subprocess.run(cmd, cwd=str(REPO / "PDAC" / "sim"),
                            capture_output=True, text=True,
                            timeout=7200)
    if result.returncode != 0:
        print(f"  FAILED (rc={result.returncode}):")
        print(result.stderr[-1000:])
    else:
        print(f"  Completed. Output in {out_dir}")
        tail = result.stdout.split("\n")[-20:]
        print("\n".join(tail))

    for stream, suffix in [(result.stdout, "stdout"), (result.stderr, "stderr")]:
        if stream:
            with open(run_dir / f"{mode}_{suffix}.log", "w") as f:
                f.write(stream)

    return out_dir


def extract_observables(run_dir: Path, mode: str) -> None:
    """Compute FunCN phi and cell abundances from the final presim ABM snapshot."""
    import numpy as np
    import pandas as pd
    from PDAC.tools.io import load_abm_lz4, add_category
    from PDAC.tools.funcn_core import funcn_phi
    from PDAC.tools.imc_mapping import ABM_TO_MATCHED, MATCHED_CATEGORIES

    out_dir = run_dir / mode / "outputs"
    abm_dir = out_dir / "abm"
    if not abm_dir.exists():
        print(f"  {mode}: no ABM output directory, skipping observables")
        return

    presim_snaps = sorted(abm_dir.glob("agents_presim_*.abm.lz4"))
    if not presim_snaps:
        print(f"  {mode}: no presim ABM snapshots found")
        return

    final = presim_snaps[-1]
    print(f"  {mode}: extracting from {final.name}")

    df = load_abm_lz4(str(final))
    if len(df) == 0:
        print(f"  {mode}: empty snapshot")
        return

    df = add_category(df, mode="type_state")

    # Map ABM categories → matched categories (drop fibs marked None)
    df["matched"] = df["category"].map(ABM_TO_MATCHED)
    df_matched = df.dropna(subset=["matched"])

    # Cell abundances (matched categories)
    counts = df_matched["matched"].value_counts()
    total = len(df_matched)
    print(f"  {mode} final counts (total={total}, raw={len(df)}):")
    for ct in MATCHED_CATEGORIES:
        n = counts.get(ct, 0)
        if n > 0:
            print(f"    {ct}: {n} ({100*n/total:.1f}%)")

    counts_path = run_dir / f"{mode}_counts.csv"
    pd.DataFrame({"category": MATCHED_CATEGORIES,
                   "count": [counts.get(c, 0) for c in MATCHED_CATEGORIES],
                   "fraction": [counts.get(c, 0) / max(total, 1) for c in MATCHED_CATEGORIES],
                   }).to_csv(counts_path, index=False)

    # Build pos_by_cat for FunCN
    pos_by_cat: dict[str, np.ndarray] = {}
    for cat in MATCHED_CATEGORIES:
        mask = df_matched["matched"] == cat
        if mask.any():
            pos_by_cat[cat] = df_matched.loc[mask, ["x", "y", "z"]].values

    shape = (GRID_X, GRID_Y, 1)

    try:
        phi = funcn_phi(pos_by_cat, MATCHED_CATEGORIES, shape=shape)
        phi_df = pd.DataFrame(phi, index=MATCHED_CATEGORIES, columns=MATCHED_CATEGORIES)
        phi_path = run_dir / f"{mode}_phi.csv"
        phi_df.to_csv(phi_path)
        print(f"  {mode}: FunCN phi ({phi.shape}) saved to {phi_path}")
        nvalid = np.isfinite(phi).sum()
        print(f"    {nvalid}/{phi.size} finite entries, "
              f"range [{np.nanmin(phi):.3f}, {np.nanmax(phi):.3f}]")
    except Exception as e:
        import traceback
        print(f"  {mode}: FunCN failed: {e}")
        traceback.print_exc()


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Run 5 local prelim presims (coupled + uncoupled)")
    p.add_argument("--phase", choices=["placement", "sim", "all", "observables"],
                   default="all")
    p.add_argument("--runs", type=str, default="1,2,3,4,5",
                   help="Comma-separated run numbers (default: 1,2,3,4,5)")
    args = p.parse_args(argv)

    run_ids = [int(x) for x in args.runs.split(",")]

    if not SIM_BIN.exists() and args.phase in ("sim", "all"):
        print(f"ERROR: simulator not found at {SIM_BIN}")
        print("Run: cd PDAC/sim && ./build.sh")
        sys.exit(1)

    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    for i in run_ids:
        params = RUN_PARAMS[i - 1]
        run_dir = RUNS_DIR / f"run_{i}"
        run_dir.mkdir(exist_ok=True)
        seed = 42 + i * 7

        print(f"\n{'='*60}")
        print(f"Run {i} (seed={seed})")
        print(f"{'='*60}")

        # Patch XML
        xml_path = run_dir / "params.xml"
        if not xml_path.exists():
            patch_xml(BASE_XML, xml_path, params)
            print(f"  Wrote {xml_path}")
            for k, v in params.items():
                fmt = f"{int(v)}" if k in INT_PARAMS else f"{v:.4g}"
                print(f"    {k} = {fmt}")

        # Placement
        if args.phase in ("placement", "all"):
            plc_path = run_placement(run_dir, seed)
        else:
            plc_path = run_dir / f"healthy_{GRID_X}.plc"

        # Simulation
        if args.phase in ("sim", "all"):
            if not plc_path.exists():
                print(f"  ERROR: placement file not found: {plc_path}")
                print("  Run with --phase placement first")
                continue
            for mode in ("coupled", "uncoupled"):
                run_presim(run_dir, plc_path, xml_path, mode, seed)

        # Observables
        if args.phase in ("observables",):
            for mode in ("coupled", "uncoupled"):
                extract_observables(run_dir, mode)
        elif args.phase == "all":
            print(f"  Run --phase observables after sims complete")


if __name__ == "__main__":
    main()
