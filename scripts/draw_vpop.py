"""Draw virtual populations from a fitted posterior, run qsp_sim, save trajectories.

Workflow
-------
1. Load idata.nc (fitted posterior)
2. Draw virtual patients via vpop.population.virtual_populations
3. Convert log_theta → natural-unit theta
4. Run qsp_sim for each patient × scenario, saving full CSV trajectories
5. Collect endpoint species + trajectory features into pool_results.npz

Usage
-----
    .venv/bin/python -I scripts/draw_vpop.py \
        --fit-dir results/2026-10-16_vpop_pool_30k/fit \
        --n-patients 200 --draws 50 --workers 8

    # Quick test
    .venv/bin/python -I scripts/draw_vpop.py \
        --fit-dir results/2026-10-16_vpop_pool_30k/fit \
        --n-patients 5 --draws 2 --workers 4
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

from vpop_readouts import SP, N_SPECIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

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
CSV_RULES = {"V_T": SP.V_T, "phi_collagen": SP.phi_col}

# ── reuse helpers from run_vpop_pool ────────────────────────────────────
from run_vpop_pool import (
    _find_xml_params,
    _write_modified_xml,
    _parse_csv_last_row,
    _extract_species,
    _parse_csv_trajectory,
    _extract_untreated_longterm_features,
    _extract_d21_trajectory_features,
    _extract_longterm_treatment_features,
    _tumor_diameter_cm,
)


def _scenario_configs():
    """Scenario configurations matching run_vpop_pool."""
    return [
        ("baseline", {
            "scenario_yaml": None,
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
            "t_end_days": 30,
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


def _run_patient(args: tuple) -> tuple[int, int, np.ndarray | None, dict | None]:
    """Run all scenarios for one virtual patient.

    Returns (patient_idx, status, species_array, trajectories_dict).
    species_array shape = (S, Q).
    trajectories_dict = {scenario_name: csv_text} for scenarios with trajectory=True.
    """
    (
        patient_idx, theta_row, param_names, tmpdir_base,
        qsp_sim, param_xml_template, healthy_yaml, drug_meta,
        scenario_configs, save_trajectories,
    ) = args

    workdir = Path(tmpdir_base) / f"patient_{patient_idx}"
    workdir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(param_xml_template)
    leaf_map = _find_xml_params(tree)

    xml_path = workdir / "param.xml"
    _write_modified_xml(tree, leaf_map, param_names, theta_row, xml_path)

    species_all = np.full((len(scenario_configs), N_SPECIES), np.nan)
    trajectories = {}

    # Evolve to diagnosis
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
        return patient_idx, status, None, None

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
            return patient_idx, proc.returncode, None, None

        csv_text = csv_out.read_text()

        if needs_trajectory:
            times, header, traj_rows = _parse_csv_trajectory(csv_text)
            if traj_rows:
                species_all[s_idx] = _extract_species(traj_rows[-1])

            feat_fn = s_cfg.get("feature_fn")
            if feat_fn and len(traj_rows) >= 2:
                feat_kwargs = s_cfg.get("feature_kwargs", {})
                features = feat_fn(times, traj_rows, **feat_kwargs)
                for col_idx, val in features.items():
                    species_all[s_idx, col_idx] = val

            if save_trajectories:
                trajectories[s_name] = csv_text
        else:
            row = _parse_csv_last_row(csv_text)
            species_all[s_idx] = _extract_species(row)
            if save_trajectories:
                trajectories[s_name] = csv_text

    # Fill rho_col from the XML
    if "rho_collagen" in leaf_map:
        rho_val = float(leaf_map["rho_collagen"].text)
        for s_idx in range(len(scenario_configs)):
            species_all[s_idx, SP.rho_col] = rho_val

    shutil.rmtree(workdir, ignore_errors=True)
    return patient_idx, 0, species_all, trajectories


def main():
    parser = argparse.ArgumentParser(
        description="Draw virtual populations and run observable simulations")
    parser.add_argument("--fit-dir", type=str, required=True,
                        help="Directory containing idata.nc from the fit")
    parser.add_argument("--n-patients", type=int, default=200,
                        help="Number of virtual patients per population draw")
    parser.add_argument("--draws", type=int, default=None,
                        help="Number of posterior draws to use (None=all)")
    parser.add_argument("--draw-index", type=int, default=0,
                        help="Single posterior draw index to simulate (default: 0 = MAP-like)")
    parser.add_argument("--gate", type=str, default="none",
                        choices=["none", "reject", "weight"],
                        help="Eligibility gate for virtual population (default: none; "
                             "qsp_sim naturally rejects ineligible patients)")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prior", type=str, default=str(PRIOR_NPZ))
    parser.add_argument("-o", "--output-dir", type=str, default=None,
                        help="Output directory (default: <fit-dir>/../vpop)")
    parser.add_argument("--save-trajectories", action="store_true",
                        help="Save full CSV trajectories per patient per scenario")
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    args = parser.parse_args()

    fit_dir = Path(args.fit_dir)
    idata_path = fit_dir / "idata.nc"
    if not idata_path.exists():
        log.error("No idata.nc found at %s", idata_path)
        return 1

    out_dir = Path(args.output_dir) if args.output_dir else fit_dir.parent / "vpop"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Draw virtual populations ──────────────────────────────────────
    log.info("Loading posterior from %s", idata_path)
    import arviz as az
    idata = az.from_netcdf(str(idata_path))

    post = idata.posterior
    centre = "mu_pop" if "mu_pop" in post.data_vars else "mu"
    n_chains = post.dims["chain"]
    n_draws_per_chain = post.dims["draw"]
    total_draws = n_chains * n_draws_per_chain
    param_names = tuple(str(v) for v in post.coords["param"].values)
    P = len(param_names)
    log.info("Posterior: %d chains × %d draws = %d total, %d params",
             n_chains, n_draws_per_chain, total_draws, P)

    from vpop.population import virtual_populations, save_populations

    # Build a surrogate for the eligibility gate if needed
    surrogate = None
    if args.gate != "none":
        pool_dir = fit_dir.parent
        arms_dir = pool_dir / "arms"
        if arms_dir.exists():
            log.info("Loading surrogate for eligibility gate from %s", arms_dir)
            from vpop.surrogate.train import load_arm
            from vpop.model.mechanism import Surrogate
            scenario_names = [cfg[0] for cfg in _scenario_configs()]
            arms = {}
            for s_name in scenario_names:
                arm_path = arms_dir / s_name
                if arm_path.exists():
                    arms[s_name] = load_arm(arm_path)
            if arms:
                surrogate = Surrogate.from_arms(arms, n_params=P)
                log.info("Surrogate loaded with %d arms", len(arms))
        if surrogate is None:
            log.warning("No surrogate available; falling back to gate='none'")
            args.gate = "none"

    log.info("Drawing virtual populations: %d patients, draws=%s, gate=%s",
             args.n_patients, args.draws or "all", args.gate)

    vp = virtual_populations(
        idata,
        n_patients=args.n_patients,
        surrogate=surrogate,
        gate=args.gate,
        draws=args.draws,
        seed=args.seed,
    )

    save_populations(out_dir / "vpop.npz", vp)
    log.info("Virtual populations saved: %s", vp.log_theta.shape)
    log.info("  Gate: %s | Caveat: %s", vp.gate, vp.caveat[:80])

    # ── 2. Select which population(s) to simulate ────────────────────────
    # For observable simulations, pick a single posterior draw
    draw_idx = min(args.draw_index, vp.n_populations - 1)
    log.info("Simulating population from posterior draw %d / %d", draw_idx, vp.n_populations)

    log_theta_pop = vp.log_theta[draw_idx]  # (N, P)
    theta_pop = np.exp(log_theta_pop)       # natural units

    # Apply keep mask if rejection gating was used
    if vp.keep is not None:
        kept = vp.keep[draw_idx]
        log_theta_pop = log_theta_pop[kept]
        theta_pop = theta_pop[kept]
        log.info("  After rejection gate: %d / %d patients kept",
                 kept.sum(), len(kept))
    N = theta_pop.shape[0]
    log.info("  Running %d patients × 6 scenarios", N)

    # ── 3. Run qsp_sim for each patient ─────────────────────────────────
    scenario_configs = _scenario_configs()
    S = len(scenario_configs)

    traj_dir = out_dir / "trajectories"
    if args.save_trajectories:
        traj_dir.mkdir(parents=True, exist_ok=True)

    tmpdir = Path(tempfile.mkdtemp(prefix="vpop_draw_"))
    log.info("Temp directory: %s", tmpdir)

    work_items = [
        (
            i, theta_pop[i], param_names, str(tmpdir),
            str(QSP_SIM), str(PARAM_XML), str(HEALTHY_YAML), str(DRUG_META),
            scenario_configs, args.save_trajectories,
        )
        for i in range(N)
    ]

    species = np.full((S, N, N_SPECIES), np.nan)
    status = np.full(N, -1, dtype=np.int32)
    t0 = time.time()
    done = 0
    n_ok = 0

    def _save_checkpoint(label=""):
        scenario_name_list = [cfg[0] for cfg in scenario_configs]
        save_dict = {
            "log_theta": log_theta_pop,
            "status": status,
            "param_names": np.array(param_names),
            "scenario_names": np.array(scenario_name_list),
            "species_names": np.array([sp.name for sp in SP]),
            "draw_index": np.array(draw_idx),
            "mu": vp.mu[draw_idx],
            "omega": vp.omega[draw_idx],
        }
        for s_idx, s_name in enumerate(scenario_name_list):
            save_dict[f"species_{s_name}"] = species[s_idx]
        np.savez_compressed(out_dir / "vpop_sim_results.npz", **save_dict)
        if label:
            log.info("Checkpoint (%s): saved", label)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_patient, w): w[0] for w in work_items}
        for fut in as_completed(futures):
            patient_idx, st, sp_arr, traj_dict = fut.result()
            status[patient_idx] = st
            if st == 0 and sp_arr is not None:
                for s_idx in range(S):
                    species[s_idx, patient_idx] = sp_arr[s_idx]
                n_ok += 1

                if args.save_trajectories and traj_dict:
                    for s_name, csv_text in traj_dict.items():
                        p_dir = traj_dir / f"patient_{patient_idx:04d}"
                        p_dir.mkdir(parents=True, exist_ok=True)
                        (p_dir / f"{s_name}.csv").write_text(csv_text)
            done += 1
            if done % 20 == 0 or done == N:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                log.info(
                    "Progress: %d/%d done (%.1f/min), %d ok, %d failed, "
                    "%.0fs elapsed",
                    done, N, rate * 60, n_ok, done - n_ok, elapsed,
                )
            if done % args.checkpoint_interval == 0 and done < N:
                _save_checkpoint(f"{done}/{N}")

    shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed_total = time.time() - t0
    log.info("Finished: %d/%d admissible (%.1f%%), %.1f min total",
             n_ok, N, 100 * n_ok / max(N, 1), elapsed_total / 60)

    _save_checkpoint()

    reject_counts = {}
    for s in status:
        reject_counts[int(s)] = reject_counts.get(int(s), 0) + 1
    log.info("Status distribution: %s", reject_counts)
    log.info("Results saved to %s", out_dir)

    # Summary
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("  Posterior draw: %d", draw_idx)
    log.info("  Patients drawn: %d", N)
    log.info("  Admissible (qsp_sim pass): %d (%.1f%%)",
             n_ok, 100 * n_ok / max(N, 1))
    if args.save_trajectories:
        log.info("  Trajectories saved to: %s", traj_dir)
    log.info("  Species results: %s/vpop_sim_results.npz", out_dir)
    log.info("  Virtual populations: %s/vpop.npz", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
