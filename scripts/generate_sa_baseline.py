"""Generate SA baseline directory from the posterior median.

Outputs a directory containing:
  - param_all_median.xml   (QSP params set to posterior median, ABM params unchanged)
  - tumor.plc              (placement file from init_tumor)
  - qsp_warmup.bin         (QSP ODE state at onset)
  - placement_phi.npz      (phi values at placement)

Usage::

    python -I scripts/generate_sa_baseline.py \
        --fit-dir results/2026-10-16_vpop_pool_30k/fit \
        --output-dir results/sa_baseline

    # Quick test (skip placement SA, just disc fill)
    python -I scripts/generate_sa_baseline.py \
        --fit-dir results/2026-10-16_vpop_pool_30k/fit \
        --output-dir /tmp/sa_test --sa-steps 0
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ABM_BIN = REPO / "PDAC" / "sim" / "build" / "bin" / "pdac"


def _find_xml_params(tree: ET.ElementTree) -> dict[str, ET.Element]:
    out: dict[str, ET.Element] = {}
    for elem in tree.iter():
        if len(elem) == 0 and elem.text is not None:
            out[elem.tag] = elem
    return out


def _write_modified_xml(
    tree: ET.ElementTree,
    leaf_map: dict[str, ET.Element],
    param_names: list[str],
    theta_row: np.ndarray,
    out_path: Path,
) -> list[str]:
    missing = []
    for name, val in zip(param_names, theta_row):
        if name in leaf_map:
            leaf_map[name].text = f"{val:.15g}"
        else:
            missing.append(name)
    tree.write(str(out_path), xml_declaration=True, encoding="utf-8")
    return missing


def main():
    p = argparse.ArgumentParser(description="Generate SA baseline from posterior median")
    p.add_argument("--fit-dir", type=str, required=True,
                   help="Dir containing idata.nc")
    p.add_argument("--template-xml", type=str,
                   default=str(REPO / "PDAC" / "sim" / "resource" / "param_all_test.xml"),
                   help="Template XML (default: PDAC/sim/resource/param_all_test.xml)")
    p.add_argument("--output-dir", type=str, required=True,
                   help="SA baseline output directory")
    p.add_argument("--domain-factor", type=float, default=2.0,
                   help="Domain side = factor × tumor diameter (default 2.0)")
    p.add_argument("--sa-steps", type=int, default=5_000_000,
                   help="SA placement steps (0 = skip SA, just disc fill)")
    p.add_argument("--abm-steps", type=int, default=0,
                   help="Post-placement ABM steps for presim verification (default 0)")
    p.add_argument("--presim-fraction", type=float, default=0.95,
                   help="QSP presim stop fraction (default 0.95)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--sample", type=str, default="100-MSK-001",
                   help="PASS-01 sample for phi targets (default: 100-MSK-001, closest to cohort median)")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Extract posterior median ──────────────────────────────────────
    print("=" * 60)
    print("  Step 1: Extract posterior median")
    print("=" * 60)

    import arviz as az
    idata_path = Path(args.fit_dir) / "idata.nc"
    idata = az.from_netcdf(str(idata_path))
    post = idata.posterior

    centre = "mu_pop" if "mu_pop" in post.data_vars else "mu"
    mu = post[centre].values  # (chains, draws, P)
    mu_flat = mu.reshape(-1, mu.shape[-1])
    mu_median = np.median(mu_flat, axis=0)
    theta_median = np.exp(mu_median)

    param_names = [str(v) for v in post.coords["param"].values]
    print(f"  Posterior: {mu.shape[0]} chains × {mu.shape[1]} draws, {len(param_names)} params")
    print(f"  Centre var: {centre}")
    print(f"  theta_median range: [{theta_median.min():.4e}, {theta_median.max():.4e}]")

    # ── 2. Write param_all_median.xml ────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 2: Write param_all_median.xml")
    print("=" * 60)

    tree = ET.parse(args.template_xml)
    leaf_map = _find_xml_params(tree)

    # Inject any QSP params missing from the ABM template
    root = tree.getroot()
    injected = []
    for name, val in zip(param_names, theta_median):
        if name not in leaf_map:
            elem = ET.SubElement(root, name)
            elem.text = f"{val:.15g}"
            leaf_map[name] = elem
            injected.append(name)
    if injected:
        print(f"  Injected {len(injected)} QSP params into template: {injected[:5]}...")

    # Override presim fraction
    if "presimAbmStartFrac" in leaf_map:
        leaf_map["presimAbmStartFrac"].text = f"{args.presim_fraction:.2f}"
        print(f"  Presim fraction set to {args.presim_fraction}")

    xml_out = out / "param_all_median.xml"
    missing = _write_modified_xml(tree, leaf_map, param_names, theta_median, xml_out)
    matched = len(param_names) - len(missing)
    print(f"  Template: {args.template_xml}")
    print(f"  Matched: {matched}/{len(param_names)} QSP params")
    if missing:
        print(f"  WARNING: {len(missing)} missing: {missing[:10]}")
    print(f"  Written: {xml_out}")

    # ── 3. QSP warmup ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 3: QSP warmup (tiny grid, -s 0)")
    print("=" * 60)

    warmup_dir = out / "warmup"
    warmup_dir.mkdir(parents=True, exist_ok=True)
    qsp_state = warmup_dir / "qsp_warmup.bin"
    onset_json = warmup_dir / "qsp_onset.json"

    warmup_cmd = [
        str(ABM_BIN),
        "-p", str(xml_out),
        "-gx", "11", "-gy", "11", "-gz", "1",
        "-s", "0",
        "-G", "0",
        "--output-root", str(warmup_dir),
    ]
    print(f"  CMD: {' '.join(warmup_cmd)}")
    proc = subprocess.run(warmup_cmd, capture_output=True, text=True, timeout=300)
    if proc.returncode != 0:
        print(f"  FATAL: warmup failed (exit {proc.returncode})")
        print(proc.stderr[-2000:] if proc.stderr else "(no stderr)")
        sys.exit(1)

    if not onset_json.exists():
        print(f"  FATAL: {onset_json} not found")
        print("  stdout (last 2000 chars):")
        print(proc.stdout[-2000:] if proc.stdout else "(empty)")
        sys.exit(1)

    with open(onset_json) as f:
        onset = json.load(f)
    print(f"  V_T = {onset['V_T_m3']:.3e} m³")
    print(f"  warmup = {onset['t_warmup_s']/86400:.0f} days")
    print(f"  qsp_warmup.bin: {qsp_state.exists()}")

    # Copy warmup products to baseline root
    shutil.copy2(qsp_state, out / "qsp_warmup.bin")

    # ── 4. Compute grid from onset (domain = factor × diameter) ──────────
    print("\n" + "=" * 60)
    print("  Step 4: Compute grid dimensions")
    print("=" * 60)

    sys.path.insert(0, str(REPO / "PDAC" / "tools"))
    from init_tumor import load_onset_json, grid_from_onset

    onset_data = load_onset_json(str(onset_json))

    # First call with default margin to get r_3d_vox
    geo_probe = grid_from_onset(onset_data, margin_vox=0, volume_fraction=1.0)
    r_vox = geo_probe["r_3d_vox"]

    # domain = factor × diameter → margin_vox = (factor-1) × radius
    # grid_from_onset: grid_side = ceil(2*r) + 2*margin
    # For factor=2: margin = r → grid = ceil(2r) + 2r = ~4r = 2*diameter
    margin_vox = int(np.ceil((args.domain_factor - 1.0) * r_vox))

    geo = grid_from_onset(onset_data, margin_vox=margin_vox, volume_fraction=1.0)
    gx, gy = geo["grid_x"], geo["grid_y"]

    print(f"  Tumor radius: {r_vox:.1f} voxels")
    print(f"  Domain factor: {args.domain_factor}× → margin = {margin_vox} vox")
    print(f"  Grid: {gx} × {gy} × 1")
    print(f"  Domain: {geo['domain_mm']:.2f} mm")
    print(f"  Disc cells: {geo['n_disc']:,d} (cancer={geo['n_cancer_2d']:,d})")

    # ── 5. Generate placement ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  Step 5: Generate placement")
    print("=" * 60)

    tumor_plc = out / "tumor.plc"
    placement_cmd = [
        sys.executable, "-m", "PDAC.tools.init_tumor",
        "--onset-json", str(onset_json),
        "--sample", args.sample,
        "--volume-fraction", "1.0",
        "--margin", str(margin_vox),
        "--seed", str(args.seed),
        "--n-steps", str(args.sa_steps),
        "--output", str(tumor_plc),
    ]
    env = {**__import__("os").environ, "PYTHONPATH": str(REPO)}
    print(f"  CMD: {' '.join(str(c) for c in placement_cmd)}")
    proc = subprocess.run(placement_cmd, capture_output=True, text=True, timeout=3600,
                          env=env, cwd=str(REPO))
    if proc.returncode != 0:
        print(f"  FATAL: placement failed (exit {proc.returncode})")
        print(proc.stderr[-2000:] if proc.stderr else "(no stderr)")
        sys.exit(1)
    print(f"  {tumor_plc}: {tumor_plc.stat().st_size:,d} bytes")

    # ── 6. Compute placement phi from diagnostics NPZ ──────────────────
    print("\n" + "=" * 60)
    print("  Step 6: Compute placement phi")
    print("=" * 60)

    from init_tumor import SA_CATEGORIES
    from funcn_core import funcn_phi as compute_phi

    # init_tumor saves pos_{category} arrays in the diagnostics NPZ
    diag_npz = Path(str(tumor_plc).replace(".plc", ".npz"))
    if not diag_npz.exists():
        print(f"  FATAL: diagnostics NPZ not found: {diag_npz}")
        sys.exit(1)

    diag = np.load(str(diag_npz), allow_pickle=True)

    pos_3d: dict[str, np.ndarray] = {}
    for cat in SA_CATEGORIES:
        key = f"pos_{cat}"
        if key in diag:
            p = diag[key]
            if len(p) > 0:
                if p.shape[1] == 2:
                    pos_3d[cat] = np.column_stack([p, np.zeros(len(p), dtype=np.int32)])
                else:
                    pos_3d[cat] = p
            else:
                pos_3d[cat] = np.empty((0, 3), dtype=np.int64)
        else:
            pos_3d[cat] = np.empty((0, 3), dtype=np.int64)

    cats_present = [c for c in SA_CATEGORIES if len(pos_3d.get(c, [])) > 0]
    print(f"  Categories present: {cats_present}")
    for c in cats_present:
        print(f"    {c}: {len(pos_3d[c])} cells")

    phi_mat = compute_phi(pos_3d, cats_present, (gx, gy, 1), sigma_vox=2.5)
    phi_path = out / "placement_phi.npz"
    np.savez_compressed(str(phi_path), phi=phi_mat, categories=np.array(cats_present))
    print(f"  Phi matrix: {phi_mat.shape}")
    print(f"  Saved: {phi_path}")

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SA Baseline Complete")
    print("=" * 60)
    print(f"  Directory: {out}")
    print(f"  Files:")
    for f in sorted(out.iterdir()):
        if f.is_file():
            print(f"    {f.name}: {f.stat().st_size:,d} bytes")
    print(f"\n  Grid: {gx} × {gy} × 1")
    print(f"  QSP params: {matched} set to posterior median")
    print(f"  Ready for: python -I scripts/run_abm_sa.py generate --baseline-dir {out}")


if __name__ == "__main__":
    main()
