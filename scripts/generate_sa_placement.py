"""Generate the baseline placement file for ABM sensitivity analysis.

Posterior median → QSP warmup to onset → 2D placement with SA phi matching.

Usage::

    python -I scripts/generate_sa_placement.py \
        --fit-dir results/2026-10-16_vpop_pool_30k/fit \
        --sample 100-DFC-003 \
        --output-dir results/sa_baseline \
        --seed 42
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import arviz as az
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
ABM_BIN = REPO / "PDAC" / "sim" / "build" / "bin" / "pdac"
PYTHON = sys.executable
QSP_XML_TEMPLATE = REPO / "qspio-pdac" / "resources" / "cpp" / "param_all.xml"


def _find_xml_params(tree: ET.ElementTree) -> dict[str, ET.Element]:
    out: dict[str, ET.Element] = {}
    for elem in tree.iter():
        if len(elem) == 0 and elem.text is not None:
            out[elem.tag] = elem
    return out


def _write_modified_xml(
    template_tree: ET.ElementTree,
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
    template_tree.write(str(out_path), xml_declaration=True, encoding="utf-8")
    return missing


def _run(cmd: list[str], label: str, **kw) -> subprocess.CompletedProcess:
    log.info(f"{label}: {' '.join(str(c) for c in cmd[:6])}...")
    result = subprocess.run(cmd, **kw)
    if result.returncode != 0:
        log.error(f"{label} failed (exit {result.returncode})")
        sys.exit(result.returncode)
    return result


def posterior_median(idata: az.InferenceData, prior_names: list[str]) -> np.ndarray:
    """Extract the median of each parameter across all chains+draws."""
    post = idata.posterior
    theta = np.empty(len(prior_names))
    for i, name in enumerate(prior_names):
        if name in post:
            theta[i] = float(np.median(post[name].values))
        else:
            raise KeyError(f"Parameter {name!r} not in posterior")
    return theta


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fit-dir", required=True,
                   help="Directory with idata.nc from NUTS fit")
    p.add_argument("--sample", default="100-DFC-003",
                   help="PASS01 sample ID for phi targets (default: 100-DFC-003)")
    p.add_argument("--onset-fraction", type=float, default=0.95,
                   help="Fraction of target volume for QSP onset (default 0.95)")
    p.add_argument("--margin", type=int, default=50,
                   help="Grid margin in voxels (default 50)")
    p.add_argument("--max-grid", type=int, default=None,
                   help="Cap grid side and scale tumor down proportionally "
                        "(e.g. 1001 for ~20mm domain)")
    p.add_argument("--sa-steps", type=int, default=5_000_000,
                   help="SA placement optimization steps (default 5M)")
    p.add_argument("--healthy-sa-steps", type=int, default=5_000_000,
                   help="Healthy tissue SA steps (default 5M)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", default="results/sa_baseline",
                   help="Output directory")
    p.add_argument("--param-xml", default=None,
                   help="ABM param XML (default: qspio-pdac template)")
    args = p.parse_args(argv)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. Load posterior and compute median ──
    fit_dir = Path(args.fit_dir)
    idata_path = fit_dir / "idata.nc"
    log.info(f"Loading posterior from {idata_path}")
    idata = az.from_netcdf(str(idata_path))

    prior = np.load(REPO / "qspio-pdac" / "parameters" / "vpop_prior.npz",
                    allow_pickle=True)
    prior_names = list(prior["names"])
    log.info(f"Posterior: {len(prior_names)} parameters")

    theta_median = posterior_median(idata, prior_names)
    log.info(f"Posterior median computed (first 3: "
             f"{prior_names[0]}={theta_median[0]:.4g}, "
             f"{prior_names[1]}={theta_median[1]:.4g}, "
             f"{prior_names[2]}={theta_median[2]:.4g})")

    # Save the median theta for reproducibility
    np.savez(out / "posterior_median.npz",
             names=prior_names, theta=theta_median)
    log.info(f"Saved posterior median to {out / 'posterior_median.npz'}")

    # ── 2. Write modified XML with posterior median QSP params ──
    xml_template = Path(args.param_xml) if args.param_xml else QSP_XML_TEMPLATE
    tree = ET.parse(str(xml_template))
    leaf_map = _find_xml_params(tree)
    xml_out = out / "param_all_median.xml"
    missing = _write_modified_xml(tree, leaf_map, prior_names, theta_median, xml_out)
    if missing:
        log.warning(f"  {len(missing)} params not in XML: {missing[:5]}...")
    log.info(f"Wrote median-parameter XML to {xml_out}")

    # ── 3. Read target volume from XML ──
    elem = tree.find(".//initial_tumour_diameter")
    if elem is None or elem.text is None:
        raise ValueError("No <initial_tumour_diameter> in XML")
    D_cm = float(elem.text.strip())
    D_m = D_cm * 1e-2
    target_vol = np.pi / 6.0 * D_m ** 3
    onset_vol = target_vol * args.onset_fraction
    log.info(f"Target volume: {target_vol:.3e} m³ (D={D_cm:.3f} cm)")
    log.info(f"Onset volume:  {onset_vol:.3e} m³ ({args.onset_fraction:.0%})")

    # ── 4. QSP warmup to onset ──
    onset_path = out / "onset.json"
    if not ABM_BIN.exists():
        log.error(f"ABM binary not found: {ABM_BIN}")
        log.error("Build first: cd PDAC/sim && ./build.sh")
        sys.exit(1)

    _run([
        str(ABM_BIN),
        "-p", str(xml_out),
        "-g", "11",
        "--qsp-onset-volume", str(onset_vol),
        "--dump-qsp-onset", str(onset_path),
    ], "QSP onset warmup")

    with open(onset_path) as f:
        onset_data = json.load(f)
    log.info(f"Onset: V_T = {onset_data['V_T_m3']:.3e} m³, "
             f"warmup = {onset_data['t_warmup_s'] / 86400:.0f} days")

    # Save onset metadata
    with open(out / "onset_meta.json", "w") as f:
        json.dump({
            "onset_volume_m3": onset_data["V_T_m3"],
            "target_volume_m3": target_vol,
            "onset_fraction": args.onset_fraction,
            "warmup_days": onset_data["t_warmup_s"] / 86400,
            "initial_tumour_diameter_cm": D_cm,
        }, f, indent=2)

    # ── 5. Compute grid size from onset ──
    sys.path.insert(0, str(REPO))
    from PDAC.tools.init_tumor import load_onset_json, grid_from_onset
    onset = load_onset_json(str(onset_path))
    geo = grid_from_onset(onset, margin_vox=args.margin, volume_fraction=1.0,
                          max_grid=args.max_grid)
    gx, gy = geo["grid_x"], geo["grid_y"]
    log.info(f"Grid: {gx} x {gy} ({geo['domain_mm']:.2f} mm)")
    if geo.get("domain_scale", 1.0) < 1.0:
        log.info(f"Domain scaling: {geo['domain_scale']:.4f}x area "
                 f"(natural {geo['grid_natural']}, capped at {gx})")
    log.info(f"Disc: {geo['n_disc']:,d} cells "
             f"(cancer={geo['n_cancer_2d']:,d}, {geo['cancer_fraction']:.1%})")

    # ── 6. Healthy tissue placement ──
    healthy_plc = out / "healthy.plc"
    _run([
        PYTHON, "-I", "-m", "PDAC.tools.init_2d_healthy",
        "--grid-x", str(gx),
        "--grid-y", str(gy),
        "--seed", str(args.seed),
        "--n-steps", str(args.healthy_sa_steps),
        "--output", str(healthy_plc),
    ], "Healthy tissue placement")

    # ── 7. Tumor placement (cancer disc + SA immune/stroma) ──
    tumor_plc = out / "tumor.plc"
    tumor_cmd = [
        PYTHON, "-I", "-m", "PDAC.tools.init_tumor",
        "--onset-json", str(onset_path),
        "--sample", args.sample,
        "--volume-fraction", "1.0",
        "--margin", str(args.margin),
        "--seed", str(args.seed),
        "--n-steps", str(args.sa_steps),
        "--output", str(tumor_plc),
    ]
    if args.max_grid is not None:
        tumor_cmd.extend(["--max-grid", str(args.max_grid)])
    if healthy_plc.exists():
        tumor_cmd.extend(["--healthy-plc", str(healthy_plc)])
    _run(tumor_cmd, "Tumor placement")

    if not tumor_plc.exists():
        log.error(f"Placement file not found: {tumor_plc}")
        sys.exit(1)

    # ── 8. Compute placement phi baseline ──
    log.info("Computing placement phi baseline...")
    from PDAC.tools.placement_io import read_placement, CATEGORY_TO_AGENT
    from PDAC.tools.funcn_core import funcn_phi
    from PDAC.tools.init_tumor import SA_CATEGORIES

    table, vvas, ecm, meta = read_placement(str(tumor_plc))
    shape = meta["shape"]  # (gx, gy, gz)

    # Reverse map: (type_id, cell_state) → category
    agent_to_cat = {}
    for cat, (tid, state, _sub) in CATEGORY_TO_AGENT.items():
        agent_to_cat[(tid, state)] = cat

    # Build pos_by_cat for funcn_phi: {category: (N, 3) int positions}
    pos_by_cat: dict[str, list] = {c: [] for c in SA_CATEGORIES}
    for row in table:
        tid, x, y, z, state, sub = row
        cat = agent_to_cat.get((tid, state))
        if cat and cat in pos_by_cat:
            pos_by_cat[cat].append([x, y, z])

    pos_arrays = {}
    for cat in SA_CATEGORIES:
        if pos_by_cat[cat]:
            pos_arrays[cat] = np.array(pos_by_cat[cat], dtype=np.int64)
        else:
            pos_arrays[cat] = np.empty((0, 3), dtype=np.int64)

    sigma_vox = 2.5  # 50 µm at 20 µm/vox
    categories = [c for c in SA_CATEGORIES if len(pos_arrays[c]) > 0]
    phi_matrix = funcn_phi(pos_arrays, categories, shape, sigma_vox=sigma_vox)

    # Save phi baseline
    np.savez(out / "placement_phi.npz",
             phi=phi_matrix,
             categories=np.array(categories),
             grid_shape=np.array(shape),
             sigma_vox=sigma_vox)
    log.info(f"Placement phi saved: {phi_matrix.shape} matrix for {len(categories)} categories")

    # Pretty-print diagonal (self-clustering)
    for i, cat in enumerate(categories):
        log.info(f"  phi({cat},{cat}) = {phi_matrix[i, i]:.3f}")

    # ── Summary ──
    log.info("=" * 60)
    log.info("SA BASELINE GENERATION COMPLETE")
    log.info(f"  Posterior median:   {out / 'posterior_median.npz'}")
    log.info(f"  Median-param XML:   {xml_out}")
    log.info(f"  Onset state:        {onset_path}")
    log.info(f"  Healthy placement:  {healthy_plc}")
    log.info(f"  Tumor placement:    {tumor_plc}")
    log.info(f"  Placement phi:      {out / 'placement_phi.npz'}")
    log.info(f"  Grid: {gx} x {gy}, {geo['n_disc']:,d} cells")


if __name__ == "__main__":
    main()
