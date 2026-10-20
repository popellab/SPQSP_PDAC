"""End-to-end patient pipeline: QSP onset → placement → coupled ABM.

1. QSP-only warmup (tiny grid, --presim-steps 0) → qsp_warmup.bin + qsp_onset.json
2. Compute grid size from onset volume
3. Healthy tissue placement (background + Vvas)
4. Tumor placement (cancer disc + SA immune/stroma)
5. Coupled ABM+QSP presim: --placement + --qsp-state, volume-stopped
   at the XML target (initial_tumour_diameter → π/6 × D³)

Usage::

    python -m PDAC.tools.run_patient \\
        --sample 100-DFC-003 \\
        --param-xml PDAC/sim/resource/param_all_test.xml \\
        --seed 42

All intermediate files (onset JSON, healthy .plc, tumor .plc) are placed
under the output directory.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
ABM_BIN = REPO / "PDAC" / "sim" / "build" / "bin" / "pdac"
PYTHON = sys.executable


def _read_target_volume_from_xml(xml_path: str) -> float:
    """Extract initial_tumour_diameter from the param XML → π/6 × D³ in m³."""
    import xml.etree.ElementTree as ET
    tree = ET.parse(xml_path)
    elem = tree.find(".//initial_tumour_diameter")
    if elem is None or elem.text is None:
        raise ValueError(f"No <initial_tumour_diameter> in {xml_path}")
    D_cm = float(elem.text.strip())
    D_m = D_cm * 1e-2
    return np.pi / 6.0 * D_m ** 3


def run(cmd: list[str], label: str, **kw) -> subprocess.CompletedProcess:
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  CMD: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        print(f"FATAL: {label} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    return result


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="End-to-end patient pipeline: QSP → placement → ABM"
    )
    p.add_argument("--sample", type=str, required=True,
                   help="PASS01 sample ID (e.g. 100-DFC-003)")
    p.add_argument("--param-xml", type=str, required=True,
                   help="ABM parameter XML file (contains initial_tumour_diameter)")
    p.add_argument("--main-steps", type=int, default=0,
                   help="Post-target main-sim steps (treatment/observation, default 0)")
    p.add_argument("--sa-steps", type=int, default=5_000_000,
                   help="SA placement steps (default 5M)")
    p.add_argument("--healthy-sa-steps", type=int, default=5_000_000,
                   help="Healthy tissue SA steps (default 5M)")
    p.add_argument("--margin", type=int, default=50,
                   help="Grid margin around tumor in voxels (default 50)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: results/<date>_patient_<sample>)")
    p.add_argument("--grid-output", type=int, default=3,
                   help="ABM grid output level (default 3 = ABM+PDE)")
    p.add_argument("--out-interval", type=int, default=1,
                   help="ABM output interval (default 1 = every step)")
    p.add_argument("--skip-onset", action="store_true",
                   help="Skip QSP onset (reuse existing onset.json)")
    p.add_argument("--skip-placement", action="store_true",
                   help="Skip placement (reuse existing .plc files)")
    args = p.parse_args(argv)

    param_xml = str(Path(args.param_xml).resolve())

    # ── Derive target volume from XML (informational) ──
    target_vol = _read_target_volume_from_xml(param_xml)

    # ── Output directory ──
    if args.output_dir:
        out = Path(args.output_dir)
    else:
        out = REPO / "results" / f"{date.today().isoformat()}_patient_{args.sample}"
    out.mkdir(parents=True, exist_ok=True)

    tumor_plc = out / "tumor.plc"
    healthy_plc = out / "healthy.plc"

    print(f"Patient pipeline: {args.sample}")
    print(f"  XML target vol:  {target_vol:.2e} m³ (from initial_tumour_diameter)")
    print(f"  Main-sim steps:  {args.main_steps}")
    print(f"  Output:          {out}")

    # ═══════════════════════════════════════════════════════════
    # Phase 1: QSP onset warmup
    #   Run binary at tiny grid with -s 0 (no main-sim).
    #   Phase A (QSP warmup) writes qsp_warmup.bin + qsp_onset.json,
    #   then the volume-stopped presim runs on the tiny grid (fast, ~2s).
    #   We only care about the QSP warmup files, not the ABM output.
    # ═══════════════════════════════════════════════════════════
    warmup_dir = out / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    qsp_state_path = warmup_dir / "qsp_warmup.bin"
    onset_path = warmup_dir / "qsp_onset.json"

    if not args.skip_onset:
        run([
            str(ABM_BIN),
            "-p", param_xml,
            "-gx", "11", "-gy", "11", "-gz", "1",
            "-s", "0",
            "-G", "0",
            "--output-root", str(warmup_dir),
        ], "Phase 1: QSP onset warmup")
    else:
        print(f"\nSkipping onset (reusing {onset_path})")

    if not onset_path.exists():
        print(f"FATAL: onset file not found: {onset_path}")
        sys.exit(1)

    with open(onset_path) as f:
        onset_data = json.load(f)
    print(f"\nOnset: V_T = {onset_data['V_T_m3']:.3e} m³, "
          f"warmup = {onset_data['t_warmup_s']/86400:.0f} days")

    # ═══════════════════════════════════════════════════════════
    # Phase 2: Compute grid size from onset
    # ═══════════════════════════════════════════════════════════
    from .init_tumor import load_onset_json, grid_from_onset
    onset = load_onset_json(str(onset_path))
    geo = grid_from_onset(onset, margin_vox=args.margin, volume_fraction=1.0)
    gx, gy = geo["grid_x"], geo["grid_y"]

    print(f"\nAuto grid: {gx} x {gy} ({geo['domain_mm']:.2f} mm)")
    print(f"  Disc: {geo['n_disc']:,d} cells "
          f"(cancer={geo['n_cancer_2d']:,d}, {geo['cancer_fraction']:.1%})")

    # ═══════════════════════════════════════════════════════════
    # Phase 3: Healthy tissue placement (background + Vvas)
    # ═══════════════════════════════════════════════════════════
    if not args.skip_placement:
        run([
            PYTHON, "-m", "PDAC.tools.init_2d_healthy",
            "--grid-x", str(gx),
            "--grid-y", str(gy),
            "--seed", str(args.seed),
            "--n-steps", str(args.healthy_sa_steps),
            "--output", str(healthy_plc),
        ], "Phase 3: Healthy tissue placement")

    # ═══════════════════════════════════════════════════════════
    # Phase 4: Tumor placement (cancer disc + SA immune/stroma)
    # ═══════════════════════════════════════════════════════════
    if not args.skip_placement:
        tumor_cmd = [
            PYTHON, "-m", "PDAC.tools.init_tumor",
            "--onset-json", str(onset_path),
            "--sample", args.sample,
            "--volume-fraction", "1.0",
            "--margin", str(args.margin),
            "--seed", str(args.seed),
            "--n-steps", str(args.sa_steps),
            "--output", str(tumor_plc),
        ]
        if healthy_plc.exists():
            tumor_cmd.extend(["--healthy-plc", str(healthy_plc)])
        run(tumor_cmd, "Phase 4: Tumor placement")
    else:
        print(f"\nSkipping placement (reusing {tumor_plc})")

    if not tumor_plc.exists():
        print(f"FATAL: tumor placement not found: {tumor_plc}")
        sys.exit(1)

    # ═══════════════════════════════════════════════════════════
    # Phase 5: Coupled ABM+QSP
    #   --placement + --qsp-state → load agents + QSP from files
    #   presim (default)          → volume-stopped at XML target
    #   -s N                      → post-target main-sim steps
    # ═══════════════════════════════════════════════════════════
    run([
        str(ABM_BIN),
        "-p", param_xml,
        "-gx", str(gx),
        "-gy", str(gy),
        "-gz", "1",
        "--placement", str(tumor_plc),
        "--qsp-state", str(qsp_state_path),
        "-s", str(args.main_steps),
        "-G", str(args.grid_output),
        "-oi", str(args.out_interval),
        "--seed", str(args.seed),
        "--output-root", str(out),
    ], "Phase 5: Coupled ABM+QSP (volume-stopped presim)")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete: {out}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
