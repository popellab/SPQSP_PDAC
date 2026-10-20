"""Generate vpop training pool: draw theta, run qsp_sim, collect species.

Usage
-----
    .venv/bin/python -I scripts/run_vpop_pool.py --n-design 2000 --workers 8
    .venv/bin/python -I scripts/run_vpop_pool.py --n-design 50 --workers 4  # quick test
"""
from __future__ import annotations

import argparse
import csv
import io
import logging
import shutil
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vpop.pool import training_pool, write_pool
from vpop.prior import read_prior

from vpop_readouts import SP, N_SPECIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── paths ────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parent.parent
QSPIO = REPO / "qspio-pdac"
QSP_SIM = QSPIO / "cpp" / "sim" / "build" / "qsp_sim"
PARAM_XML = QSPIO / "resources" / "cpp" / "param_all.xml"
HEALTHY_YAML = QSPIO / "resources" / "cpp" / "healthy_state.yaml"
DRUG_META = QSPIO / "resources" / "cpp" / "drug_metadata.yaml"
PRIOR_NPZ = REPO / "parameters" / "vpop_prior.npz"

SCENARIO_YAMLS = {
    "baseline": QSPIO / "scenarios" / "baseline_no_treatment.yaml",
    "gvax_d21": QSPIO / "scenarios" / "gvax_neoadjuvant_zheng2022.yaml",
    "gvax_nivo_d21": QSPIO / "scenarios" / "gvax_nivo_neoadjuvant_zheng2022.yaml",
    "untreated_longterm": QSPIO / "scenarios" / "clinical_progression_longterm.yaml",
    "gvax_longterm": QSPIO / "scenarios" / "gvax_neoadjuvant_longterm.yaml",
    "gvax_nivo_longterm": QSPIO / "scenarios" / "gvax_nivo_neoadjuvant_longterm.yaml",
}

# CSV column name → SP enum mapping (from the qsp_sim CSV header)
CSV_COL_TO_SP = {
    "V_T.C1": SP.C1, "V_T.C_x": SP.C_x, "V_T.CD8": SP.CD8,
    "V_T.CD8_exh": SP.CD8_exh, "V_T.CD8_pex": SP.CD8_pex,
    "V_T.Th": SP.Th, "V_T.Th_exh": SP.Th_exh, "V_T.nCD4": SP.nCD4,
    "V_T.Treg": SP.Treg, "V_T.Mac_M1": SP.Mac_M1, "V_T.Mac_M2": SP.Mac_M2,
    "V_T.MDSC": SP.MDSC, "V_T.cDC1": SP.cDC1, "V_T.cDC2": SP.cDC2,
    "V_T.mcDC1": SP.mcDC1, "V_T.mcDC2": SP.mcDC2,
    "V_T.iCAF": SP.iCAF, "V_T.myCAF": SP.myCAF, "V_T.apCAF": SP.apCAF,
    "V_T.qPSC": SP.qPSC, "V_T.nCD8": SP.nCD8,
    "V_T.nB": SP.nB, "V_T.aB": SP.aB, "V_T.Tfh": SP.Tfh,
    "V_T.Plasma": SP.Plasma,
    "V_T.nBreg": SP.nBreg, "V_T.aBreg": SP.aBreg,
    "V_T.collagen": SP.collagen,
    "V_T.IFNg": SP.IFNg, "V_T.TGFb": SP.TGFb, "V_T.IL10": SP.IL10,
    "V_T.IL12": SP.IL12, "V_T.CCL2": SP.CCL2, "V_T.ArgI": SP.ArgI,
    "V_T.NO": SP.NO, "V_T.IL6": SP.IL6, "V_T.IL1": SP.IL1,
    "V_T.VEGF": SP.VEGF,
}
# Assignment rules in the CSV
CSV_RULES = {"V_T": SP.V_T, "phi_collagen": SP.phi_col}


def _find_xml_params(tree: ET.ElementTree) -> dict[str, ET.Element]:
    """Map every leaf tag → Element in the XML tree."""
    out: dict[str, ET.Element] = {}
    for elem in tree.iter():
        if len(elem) == 0 and elem.text is not None:
            out[elem.tag] = elem
    return out


def _write_modified_xml(
    template_tree: ET.ElementTree,
    leaf_map: dict[str, ET.Element],
    param_names: tuple[str, ...],
    theta_row: np.ndarray,
    out_path: Path,
) -> list[str]:
    """Write a modified param_all.xml with theta values. Returns names of missing params."""
    missing = []
    for name, val in zip(param_names, theta_row):
        if name in leaf_map:
            leaf_map[name].text = f"{val:.15g}"
        else:
            missing.append(name)
    template_tree.write(str(out_path), xml_declaration=True, encoding="utf-8")
    return missing


def _parse_csv_last_row(csv_text: str) -> dict[str, float]:
    """Parse qsp_sim CSV output and return the LAST time-row as {col: value}."""
    reader = csv.DictReader(io.StringIO(csv_text))
    row = None
    for row in reader:
        pass
    if row is None:
        return {}
    return {k: float(v) for k, v in row.items() if v}


def _extract_species(row: dict[str, float]) -> np.ndarray:
    """Extract Q-length species vector from a parsed CSV row."""
    sp = np.full(N_SPECIES, np.nan)
    for col, idx in CSV_COL_TO_SP.items():
        if col in row:
            sp[idx] = row[col]
    for col, idx in CSV_RULES.items():
        if col in row:
            sp[idx] = row[col]
    # rho_col is a parameter, not a species — fill from XML if present
    if "rho_collagen" in row:
        sp[SP.rho_col] = row["rho_collagen"]
    return sp


def _parse_csv_trajectory(csv_text: str) -> tuple[np.ndarray, list[str], list[dict[str, float]]]:
    """Parse full trajectory CSV. Returns (time_array, header, list_of_row_dicts)."""
    reader = csv.DictReader(io.StringIO(csv_text))
    header = reader.fieldnames or []
    rows = []
    times = []
    for row in reader:
        if not row.get("Time"):
            continue
        times.append(float(row["Time"]))
        rows.append({k: float(v) for k, v in row.items() if v})
    return np.array(times), header, rows


def _interp_at(times: np.ndarray, values: np.ndarray, t: float) -> float:
    """Linear interpolation of values at time t."""
    if len(times) == 0:
        return np.nan
    if t <= times[0]:
        return float(values[0])
    if t >= times[-1]:
        return float(values[-1])
    return float(np.interp(t, times, values))


def _tumor_diameter_cm(v_t_mL: np.ndarray) -> np.ndarray:
    """Tumor volume (mL = cm³) → diameter (cm), assuming sphere."""
    v_cm3 = np.clip(v_t_mL, 0.0, None)  # mL = cm³
    return 2.0 * np.cbrt(3.0 * v_cm3 / (4.0 * np.pi))


def _extract_untreated_longterm_features(
    times: np.ndarray, rows: list[dict[str, float]],
) -> dict[int, float]:
    """Compute trajectory features from untreated longterm run."""
    out: dict[int, float] = {}
    n = len(times)
    if n < 2:
        return out
    t_days = times / 86400.0  # seconds → days

    c1 = np.array([r.get("V_T.C1", np.nan) for r in rows])
    cd8 = np.array([r.get("V_T.CD8", np.nan) for r in rows])
    cd8_exh = np.array([r.get("V_T.CD8_exh", np.nan) for r in rows])
    collagen = np.array([r.get("V_T.collagen", np.nan) for r in rows])
    v_t = np.array([r.get("V_T", np.nan) for r in rows])

    # C1 fold at d90
    c1_0 = c1[0] if c1[0] > 0 else np.nan
    out[SP.traj_C1_fold_d90] = _interp_at(t_days, c1, 90.0) / c1_0 if np.isfinite(c1_0) else np.nan

    # Exhaustion fraction fold at d90
    total_cd8 = cd8 + cd8_exh
    exh_frac = np.where(total_cd8 > 0, cd8_exh / total_cd8, np.nan)
    ef0 = exh_frac[0] if np.isfinite(exh_frac[0]) and exh_frac[0] > 0 else np.nan
    out[SP.traj_exh_fold_d90] = _interp_at(t_days, exh_frac, 90.0) / ef0 if np.isfinite(ef0) else np.nan

    # ECM concentration fold at d90
    ecm_conc = np.where(v_t > 0, collagen / v_t, np.nan)
    ec0 = ecm_conc[0] if np.isfinite(ecm_conc[0]) and ecm_conc[0] > 0 else np.nan
    out[SP.traj_ecm_fold_d90] = _interp_at(t_days, ecm_conc, 90.0) / ec0 if np.isfinite(ec0) else np.nan

    # Tumor diameters
    diam = _tumor_diameter_cm(v_t)
    out[SP.traj_diam_d365] = _interp_at(t_days, diam, 365.0)
    out[SP.traj_diam_d730] = _interp_at(t_days, diam, 730.0)

    # Diameter fold at d90
    d0 = diam[0] if diam[0] > 0 else np.nan
    out[SP.traj_diam_fold_d90] = _interp_at(t_days, diam, 90.0) / d0 if np.isfinite(d0) else np.nan

    # Time to progression (RECIST: 20% diameter increase from nadir)
    if np.isfinite(d0) and d0 > 0:
        nadir = d0
        ttp = np.nan
        for i in range(1, n):
            if diam[i] < nadir:
                nadir = diam[i]
            if diam[i] >= nadir * 1.2:
                ttp = t_days[i]
                break
        if np.isnan(ttp):
            ttp = t_days[-1]  # no progression within window
        out[SP.traj_TTP] = ttp
    else:
        out[SP.traj_TTP] = np.nan

    # Doubling time (exponential fit from d0 to first doubling)
    if np.isfinite(c1_0) and c1_0 > 0:
        target_2x = c1_0 * 2.0
        t_double = np.nan
        for i in range(1, n):
            if c1[i] >= target_2x:
                # linear interpolation for exact crossing
                frac = (target_2x - c1[i-1]) / (c1[i] - c1[i-1] + 1e-30)
                t_double = t_days[i-1] + frac * (t_days[i] - t_days[i-1])
                break
        if np.isnan(t_double):
            # estimate from exponential fit if never reached 2x
            if c1[-1] > c1_0:
                t_double = t_days[-1] * np.log(2) / np.log(c1[-1] / c1_0)
            else:
                t_double = t_days[-1]  # no growth
        out[SP.traj_doubling_time] = max(t_double, 1e-3)
    else:
        out[SP.traj_doubling_time] = np.nan

    return out


def _excursion_mass_center(times: np.ndarray, signal: np.ndarray, rising: bool) -> float:
    """Compute ε-relaxed excursion mass-center time for a signal.

    If rising=True, computes mass-center of the rising excursion (signal - signal[0]).
    If rising=False, computes mass-center of the falling excursion (signal[0] - signal).
    Returns time coordinate of mass center (days).
    """
    eps = 1e-12
    ref = signal[0]
    if not np.isfinite(ref):
        return np.nan
    if rising:
        excursion = np.maximum(signal - ref, 0.0)
    else:
        excursion = np.maximum(ref - signal, 0.0)
    excursion = np.maximum(excursion, eps)
    total = np.trapezoid(excursion, times)
    if total <= 0:
        return times[-1] / 2.0
    return float(np.trapezoid(excursion * times, times) / total)


def _extract_d21_trajectory_features(
    times: np.ndarray, rows: list[dict[str, float]], include_ckpt: bool = False,
) -> dict[int, float]:
    """Extract excursion mass-center features from d21 (or d30) trajectory."""
    out: dict[int, float] = {}
    n = len(times)
    if n < 2:
        return out
    t_days = times / 86400.0

    # mAPC = mcDC1 + mcDC2 (rising)
    mapc = np.array([r.get("V_T.mcDC1", 0) + r.get("V_T.mcDC2", 0) for r in rows])
    # CD8_total = CD8 + CD8_exh (rising)
    cd8_total = np.array([r.get("V_T.CD8", 0) + r.get("V_T.CD8_exh", 0) for r in rows])
    # C1 (falling for tumor nadir)
    c1 = np.array([r.get("V_T.C1", np.nan) for r in rows])

    t_mapc = _excursion_mass_center(t_days, mapc, rising=True)
    t_cd8 = _excursion_mass_center(t_days, cd8_total, rising=True)
    t_c1_fall = _excursion_mass_center(t_days, c1, rising=False)

    eps_out = 0.01
    out[SP.traj_mapc_cd8_lag] = max(t_cd8 - t_mapc, eps_out)
    out[SP.traj_tumor_cd8_lag] = max(t_c1_fall - t_cd8, eps_out)

    if include_ckpt:
        k_eff = np.array([r.get("k_C_Tcell_eff", np.nan) for r in rows])
        k0 = k_eff[0] if np.isfinite(k_eff[0]) and k_eff[0] > 0 else np.nan
        if np.isfinite(k0):
            out[SP.traj_ckpt_fold] = max(_interp_at(t_days, k_eff, 14.0) / k0, 1e-6)
        else:
            out[SP.traj_ckpt_fold] = np.nan

    return out


def _extract_longterm_treatment_features(
    times: np.ndarray, rows: list[dict[str, float]],
) -> dict[int, float]:
    """Extract diameter fold at d90 from longterm treatment trajectory."""
    out: dict[int, float] = {}
    if len(times) < 2:
        return out
    t_days = times / 86400.0
    v_t = np.array([r.get("V_T", np.nan) for r in rows])
    diam = _tumor_diameter_cm(v_t)
    d0 = diam[0] if diam[0] > 0 else np.nan
    out[SP.traj_diam_fold_d90] = _interp_at(t_days, diam, 90.0) / d0 if np.isfinite(d0) else np.nan
    return out


def _run_one_scenario(
    qsp_sim: str,
    xml_path: str,
    healthy_yaml: str,
    scenario_yaml: str | None,
    drug_meta: str,
    t_end_days: int,
    csv_out: str,
    use_evolve_cache: bool = False,
    state_path: str | None = None,
) -> tuple[int, str, str]:
    """Run qsp_sim for one scenario. Returns (exit_code, stdout, stderr)."""
    cmd = [qsp_sim, "--param", xml_path]
    if use_evolve_cache and state_path:
        cmd += ["--initial-state", state_path]
    else:
        cmd += ["--evolve-to-diagnosis", healthy_yaml]
    if scenario_yaml:
        cmd += ["--scenario", scenario_yaml, "--drug-metadata", drug_meta]
    cmd += ["--t-end-days", str(t_end_days), "--csv-out", csv_out]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    return proc.returncode, proc.stdout, proc.stderr


def _run_sample(args: tuple) -> tuple[int, int, np.ndarray | None]:
    """Run all scenarios for one theta sample.

    Returns (sample_idx, status, species_array_or_None).
    species_array shape = (S, Q) where S=6 scenarios.
    """
    (
        sample_idx, theta_row, param_names, tmpdir_base,
        qsp_sim, param_xml_template, healthy_yaml, drug_meta,
        scenario_configs,
    ) = args

    workdir = Path(tmpdir_base) / f"sample_{sample_idx}"
    workdir.mkdir(parents=True, exist_ok=True)

    # Parse the template XML fresh for this worker
    tree = ET.parse(param_xml_template)
    leaf_map = _find_xml_params(tree)

    xml_path = workdir / "param.xml"
    _write_modified_xml(tree, leaf_map, param_names, theta_row, xml_path)

    species_all = np.full((len(scenario_configs), N_SPECIES), np.nan)

    # Phase 1: evolve to diagnosis (dump state for reuse)
    state_path = workdir / "evolve_state.qsth"
    dump_cmd = [
        str(qsp_sim), "--param", str(xml_path),
        "--evolve-to-diagnosis", str(healthy_yaml),
        "--dump-state", str(state_path),
    ]
    proc = subprocess.run(dump_cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        status = proc.returncode
        shutil.rmtree(workdir, ignore_errors=True)
        return sample_idx, status, None

    # Phase 2: run each scenario from the cached evolve state
    for s_idx, (s_name, s_cfg) in enumerate(scenario_configs):
        csv_out = workdir / f"{s_name}.csv"
        needs_trajectory = s_cfg.get("trajectory", False)
        timeout_s = s_cfg.get("timeout", 600)

        cmd = [
            str(qsp_sim), "--param", str(xml_path),
            "--initial-state", str(state_path),
        ]
        if s_cfg["scenario_yaml"]:
            cmd += [
                "--scenario", str(s_cfg["scenario_yaml"]),
                "--drug-metadata", str(drug_meta),
            ]
        cmd += [
            "--t-end-days", str(s_cfg["t_end_days"]),
            "--csv-out", str(csv_out),
        ]
        if needs_trajectory:
            cadence = s_cfg.get("cadence_hours", 4.0)
            cmd += ["--min-cadence-hours", str(cadence)]

        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        if proc.returncode != 0:
            shutil.rmtree(workdir, ignore_errors=True)
            return sample_idx, proc.returncode, None

        csv_text = csv_out.read_text()

        if needs_trajectory:
            times, header, traj_rows = _parse_csv_trajectory(csv_text)
            # Endpoint species from last row
            if traj_rows:
                species_all[s_idx] = _extract_species(traj_rows[-1])

            # Extract trajectory features by scenario type
            feat_fn = s_cfg.get("feature_fn")
            if feat_fn and len(traj_rows) >= 2:
                feat_kwargs = s_cfg.get("feature_kwargs", {})
                features = feat_fn(times, traj_rows, **feat_kwargs)
                for col_idx, val in features.items():
                    species_all[s_idx, col_idx] = val
        else:
            row = _parse_csv_last_row(csv_text)
            species_all[s_idx] = _extract_species(row)

    # Fill rho_col from the XML (it's a parameter, constant across scenarios)
    if "rho_collagen" in leaf_map:
        rho_val = float(leaf_map["rho_collagen"].text)
        for s_idx in range(len(scenario_configs)):
            species_all[s_idx, SP.rho_col] = rho_val

    shutil.rmtree(workdir, ignore_errors=True)
    return sample_idx, 0, species_all


def main():
    parser = argparse.ArgumentParser(description="Generate vpop training pool")
    parser.add_argument("--n-design", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prior", type=str, default=str(PRIOR_NPZ))
    parser.add_argument("-o", "--output-dir", type=str,
                        default=str(REPO / "results" / "vpop_pool"))
    parser.add_argument("--checkpoint-interval", type=int, default=2000,
                        help="Save checkpoint every N completed samples")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prior and draw training pool
    log.info("Loading prior from %s", args.prior)
    prior = read_prior(args.prior)
    log.info("Prior: %d parameters", prior.n_params)

    # Draw in batches to avoid logit-bound overflow (wide omega on f_nTreg)
    BATCH_SIZE = 100
    theta_chunks = []
    param_names = None
    drawn = 0
    seed_offset = 0
    while drawn < args.n_design:
        n_this = min(BATCH_SIZE, args.n_design - drawn)
        seed_b = args.seed + seed_offset
        try:
            design = training_pool(prior, n_design=n_this, seed=seed_b)
            theta_chunks.append(design.theta)
            if param_names is None:
                param_names = design.names
            drawn += n_this
            log.info("Batch drawn: %d/%d total (seed=%d)", drawn, args.n_design, seed_b)
        except RuntimeError as e:
            if "left its bound" in str(e):
                log.warning("Seed %d caused bound overflow, trying next seed", seed_b)
            else:
                raise
        seed_offset += 1
    theta = np.concatenate(theta_chunks, axis=0)
    log.info("Total theta shape: %s", theta.shape)

    # Write the pool CSV (theta in natural units for read_pool compatibility)
    pool_csv = out_dir / "pool" / "theta.csv"
    pool_csv.parent.mkdir(parents=True, exist_ok=True)
    header = "sample_index," + ",".join(param_names)
    rows = np.column_stack([np.arange(theta.shape[0]), theta])
    np.savetxt(str(pool_csv), rows, delimiter=",", header=header, comments="", fmt="%.15g")
    log.info("Pool written to %s", pool_csv)

    # Scenario configs (order must match vpop_readouts.SCENARIOS)
    scenario_configs = [
        ("baseline", {
            "scenario_yaml": None,  # baseline = just evolve, t=1d
            "t_end_days": 1,
        }),
        ("gvax_d21", {
            "scenario_yaml": str(SCENARIO_YAMLS["gvax_d21"]),
            "t_end_days": 21,
            "trajectory": True,
            "cadence_hours": 2.0,
            "feature_fn": _extract_d21_trajectory_features,
            "feature_kwargs": {"include_ckpt": False},
        }),
        ("gvax_nivo_d21", {
            "scenario_yaml": str(SCENARIO_YAMLS["gvax_nivo_d21"]),
            "t_end_days": 30,  # 30d for solver stability
            "trajectory": True,
            "cadence_hours": 2.0,
            "feature_fn": _extract_d21_trajectory_features,
            "feature_kwargs": {"include_ckpt": True},
        }),
        ("untreated_longterm", {
            "scenario_yaml": str(SCENARIO_YAMLS["untreated_longterm"]),
            "t_end_days": 1095,
            "timeout": 1200,
            "trajectory": True,
            "cadence_hours": 24.0,
            "feature_fn": _extract_untreated_longterm_features,
            "feature_kwargs": {},
        }),
        ("gvax_longterm", {
            "scenario_yaml": str(SCENARIO_YAMLS["gvax_longterm"]),
            "t_end_days": 1095,
            "timeout": 1200,
            "trajectory": True,
            "cadence_hours": 24.0,
            "feature_fn": _extract_longterm_treatment_features,
            "feature_kwargs": {},
        }),
        ("gvax_nivo_longterm", {
            "scenario_yaml": str(SCENARIO_YAMLS["gvax_nivo_longterm"]),
            "t_end_days": 1095,
            "timeout": 1200,
            "trajectory": True,
            "cadence_hours": 24.0,
            "feature_fn": _extract_longterm_treatment_features,
            "feature_kwargs": {},
        }),
    ]

    N = theta.shape[0]
    log.info("Running %d samples × %d scenarios with %d workers",
             N, len(scenario_configs), args.workers)

    # Prepare work items
    tmpdir = Path(tempfile.mkdtemp(prefix="vpop_pool_"))
    log.info("Temp directory: %s", tmpdir)

    work_items = [
        (
            i, theta[i], param_names, str(tmpdir),
            str(QSP_SIM), str(PARAM_XML), str(HEALTHY_YAML), str(DRUG_META),
            scenario_configs,
        )
        for i in range(N)
    ]

    # Run in parallel
    S = len(scenario_configs)
    species = np.full((S, N, N_SPECIES), np.nan)
    status = np.full(N, -1, dtype=np.int32)
    t0 = time.time()
    done = 0
    n_ok = 0

    def _save_results(path, label=""):
        scenario_name_list = [cfg[0] for cfg in scenario_configs]
        save_dict = {
            "log_theta": np.log(theta),
            "status": status,
            "param_names": np.array(param_names),
            "scenario_names": np.array(scenario_name_list),
            "species_names": np.array([sp.name for sp in SP]),
        }
        for s_idx, s_name in enumerate(scenario_name_list):
            save_dict[f"species_{s_name}"] = species[s_idx]
        np.savez_compressed(path, **save_dict)
        if label:
            log.info("Checkpoint (%s): saved to %s", label, path)

    last_ckpt = 0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_sample, w): w[0] for w in work_items}
        for fut in as_completed(futures):
            sample_idx, st, sp_arr = fut.result()
            status[sample_idx] = st
            if st == 0 and sp_arr is not None:
                for s_idx in range(S):
                    species[s_idx, sample_idx] = sp_arr[s_idx]
                n_ok += 1
            done += 1
            if done % 50 == 0 or done == N:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                log.info(
                    "Progress: %d/%d done (%.0f/min), %d ok, %d rejected, "
                    "%.0fs elapsed",
                    done, N, rate * 60, n_ok, done - n_ok, elapsed,
                )
            if done - last_ckpt >= args.checkpoint_interval and done < N:
                _save_results(out_dir / "pool_results.npz",
                              label=f"{done}/{N}")
                last_ckpt = done

    shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed_total = time.time() - t0
    log.info("Finished: %d/%d admissible (%.1f%%), %.1f min total",
             n_ok, N, 100 * n_ok / max(N, 1), elapsed_total / 60)

    # Save final results
    results_path = out_dir / "pool_results.npz"
    _save_results(results_path)

    reject_counts = {}
    for s in status:
        reject_counts[int(s)] = reject_counts.get(int(s), 0) + 1
    log.info("Status distribution: %s", reject_counts)
    log.info("Results saved to %s", results_path)


if __name__ == "__main__":
    main()
