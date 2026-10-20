"""Generate virtual patient cohorts from NUTS posterior and run QSP simulations.

Loads the posterior from idata.nc, draws patient-level parameters via the
population model (apply_margins with correlated normals), writes modified XMLs,
and runs qsp_sim for each scenario.

Usage
-----
    # Quick test: 2 posterior draws × 3 patients each
    .venv/bin/python -I scripts/generate_vpop_cohort.py --n-posterior 2 --n-patients 3 --workers 2

    # Production: 50 posterior draws × 100 patients each
    .venv/bin/python -I scripts/generate_vpop_cohort.py --n-posterior 50 --n-patients 100 --workers 8
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
from scipy.special import log_expit

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vpop_readouts import SP, N_SPECIES, SCENARIOS

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

SCENARIO_CONFIGS = [
    ("baseline", {
        "scenario_yaml": None,
        "t_end_days": 1,
    }),
    ("gvax_d21", {
        "scenario_yaml": str(QSPIO / "scenarios" / "gvax_neoadjuvant_zheng2022.yaml"),
        "t_end_days": 21,
    }),
    ("gvax_nivo_d21", {
        "scenario_yaml": str(QSPIO / "scenarios" / "gvax_nivo_neoadjuvant_zheng2022.yaml"),
        "t_end_days": 30,
    }),
]

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


def draw_posterior_samples(idata_path, n_posterior, seed):
    """Draw n_posterior (mu_pop, omega_pop) samples from the posterior."""
    import arviz as az

    idata = az.from_netcdf(str(idata_path))
    post = idata.posterior

    mu_pop = post["mu_pop"].values    # (chains, draws, P)
    omega_pop = post["omega_pop"].values

    n_chains, n_draws, P = mu_pop.shape
    total = n_chains * n_draws

    mu_flat = mu_pop.reshape(total, P)
    omega_flat = omega_pop.reshape(total, P)

    rng = np.random.default_rng(seed)
    idx = rng.choice(total, size=n_posterior, replace=n_posterior > total)

    L_R = idata.constant_data["L_R"].values
    logit = idata.constant_data["logit_margin"].values

    return mu_flat[idx], omega_flat[idx], L_R, logit


def draw_patients(mu, omega, L_R, logit, n_patients, seed):
    """Draw n_patients parameter vectors for one posterior (mu, omega) draw.

    Returns theta in natural scale, shape (n_patients, P).
    """
    P = len(mu)
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_patients, P))
    zL = z @ L_R.T

    raw = mu[None, :] + zL * omega[None, :]
    log_theta = np.where(logit[None, :], log_expit(raw), raw)
    return np.exp(log_theta)


def _find_xml_params(tree):
    out = {}
    for elem in tree.iter():
        if len(elem) == 0 and elem.text is not None:
            out[elem.tag] = elem
    return out


def _write_modified_xml(template_tree, leaf_map, param_names, theta_row, out_path):
    missing = []
    for name, val in zip(param_names, theta_row):
        if name in leaf_map:
            leaf_map[name].text = f"{val:.15g}"
        else:
            missing.append(name)
    template_tree.write(str(out_path), xml_declaration=True, encoding="utf-8")
    return missing


def _parse_csv_last_row(csv_text):
    reader = csv.DictReader(io.StringIO(csv_text))
    row = None
    for row in reader:
        pass
    if row is None:
        return {}
    return {k: float(v) for k, v in row.items() if v}


def _extract_species(row):
    sp = np.full(N_SPECIES, np.nan)
    for col, idx in CSV_COL_TO_SP.items():
        if col in row:
            sp[idx] = row[col]
    for col, idx in CSV_RULES.items():
        if col in row:
            sp[idx] = row[col]
    if "rho_collagen" in row:
        sp[SP.rho_col] = row["rho_collagen"]
    return sp


def _run_patient(args):
    """Run all scenarios for one patient. Returns (post_idx, pat_idx, status, species_or_None)."""
    (
        post_idx, pat_idx, theta_row, param_names, tmpdir_base,
        qsp_sim, param_xml_template, healthy_yaml, drug_meta,
        scenario_configs,
    ) = args

    workdir = Path(tmpdir_base) / f"p{post_idx}_v{pat_idx}"
    workdir.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(param_xml_template)
    leaf_map = _find_xml_params(tree)

    xml_path = workdir / "param.xml"
    _write_modified_xml(tree, leaf_map, param_names, theta_row, xml_path)

    species_all = np.full((len(scenario_configs), N_SPECIES), np.nan)

    state_path = workdir / "evolve_state.qsth"
    dump_cmd = [
        str(qsp_sim), "--param", str(xml_path),
        "--evolve-to-diagnosis", str(healthy_yaml),
        "--dump-state", str(state_path),
    ]
    proc = subprocess.run(dump_cmd, capture_output=True, text=True, timeout=600)
    if proc.returncode != 0:
        shutil.rmtree(workdir, ignore_errors=True)
        return post_idx, pat_idx, proc.returncode, None

    for s_idx, (s_name, s_cfg) in enumerate(scenario_configs):
        csv_out = workdir / f"{s_name}.csv"
        cmd = [
            str(qsp_sim), "--param", str(xml_path),
            "--initial-state", str(state_path),
        ]
        if s_cfg["scenario_yaml"]:
            cmd += [
                "--scenario", str(s_cfg["scenario_yaml"]),
                "--drug-metadata", str(drug_meta),
            ]
        cmd += ["--t-end-days", str(s_cfg["t_end_days"]), "--csv-out", str(csv_out)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            shutil.rmtree(workdir, ignore_errors=True)
            return post_idx, pat_idx, proc.returncode, None

        csv_text = csv_out.read_text()
        row = _parse_csv_last_row(csv_text)
        species_all[s_idx] = _extract_species(row)

    if "rho_collagen" in leaf_map:
        rho_val = float(leaf_map["rho_collagen"].text)
        for s_idx in range(len(scenario_configs)):
            species_all[s_idx, SP.rho_col] = rho_val

    shutil.rmtree(workdir, ignore_errors=True)
    return post_idx, pat_idx, 0, species_all


def main():
    parser = argparse.ArgumentParser(
        description="Generate virtual patient cohorts from NUTS posterior")
    parser.add_argument("--idata", type=str,
                        default=str(REPO / "results" / "vpop_pool" / "fit" / "idata.nc"))
    parser.add_argument("--prior", type=str,
                        default=str(REPO / "parameters" / "vpop_prior.npz"))
    parser.add_argument("--n-posterior", type=int, default=50,
                        help="Number of posterior (mu, omega) draws")
    parser.add_argument("--n-patients", type=int, default=100,
                        help="Number of virtual patients per posterior draw")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output-dir", type=str,
                        default=str(REPO / "results" / "vpop_cohorts"))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load prior for param names
    pi = np.load(args.prior, allow_pickle=False)
    param_names = tuple(str(n) for n in pi["names"])
    P = len(param_names)
    log.info("Prior: %d parameters", P)

    # Draw posterior samples
    log.info("Drawing %d posterior samples from %s", args.n_posterior, args.idata)
    mu_draws, omega_draws, L_R, logit = draw_posterior_samples(
        args.idata, args.n_posterior, seed=args.seed)
    log.info("Posterior draws: mu=%s, omega=%s", mu_draws.shape, omega_draws.shape)

    # Generate patient-level thetas for each posterior draw
    log.info("Generating %d patients per posterior draw (%d total)",
             args.n_patients, args.n_posterior * args.n_patients)
    all_thetas = []
    post_indices = []
    pat_indices = []
    for i in range(args.n_posterior):
        theta_i = draw_patients(
            mu_draws[i], omega_draws[i], L_R, logit,
            args.n_patients, seed=args.seed + i + 1)
        all_thetas.append(theta_i)
        for j in range(args.n_patients):
            post_indices.append(i)
            pat_indices.append(j)

    all_thetas = np.concatenate(all_thetas, axis=0)  # (N_total, P)
    N_total = len(all_thetas)
    log.info("Total patients to simulate: %d", N_total)

    # Save the theta draws before simulation
    np.savez_compressed(
        out_dir / "cohort_thetas.npz",
        theta=all_thetas,
        log_theta=np.log(all_thetas),
        param_names=np.array(param_names),
        post_idx=np.array(post_indices),
        pat_idx=np.array(pat_indices),
        mu_draws=mu_draws,
        omega_draws=omega_draws,
        n_posterior=args.n_posterior,
        n_patients=args.n_patients,
    )
    log.info("Saved theta draws to %s/cohort_thetas.npz", out_dir)

    # Run simulations
    tmpdir = Path(tempfile.mkdtemp(prefix="vpop_cohort_"))
    log.info("Temp directory: %s", tmpdir)

    work_items = [
        (
            post_indices[k], pat_indices[k], all_thetas[k],
            param_names, str(tmpdir),
            str(QSP_SIM), str(PARAM_XML), str(HEALTHY_YAML), str(DRUG_META),
            SCENARIO_CONFIGS,
        )
        for k in range(N_total)
    ]

    S = len(SCENARIO_CONFIGS)
    species = np.full((S, N_total, N_SPECIES), np.nan)
    status = np.full(N_total, -1, dtype=np.int32)
    t0 = time.time()
    done = 0
    n_ok = 0

    log.info("Running %d patients × %d scenarios with %d workers",
             N_total, S, args.workers)

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for k, w in enumerate(work_items):
            futures[pool.submit(_run_patient, w)] = k

        for fut in as_completed(futures):
            k = futures[fut]
            post_idx, pat_idx, st, sp_arr = fut.result()
            status[k] = st
            if st == 0 and sp_arr is not None:
                for s_idx in range(S):
                    species[s_idx, k] = sp_arr[s_idx]
                n_ok += 1
            done += 1
            if done % max(1, N_total // 20) == 0 or done == N_total:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                log.info(
                    "Progress: %d/%d done (%.0f/min), %d ok, %d rejected, "
                    "%.0fs elapsed",
                    done, N_total, rate * 60, n_ok, done - n_ok, elapsed,
                )

    shutil.rmtree(tmpdir, ignore_errors=True)

    elapsed_total = time.time() - t0
    log.info("Finished: %d/%d admissible (%.1f%%), %.1f min total",
             n_ok, N_total, 100 * n_ok / max(N_total, 1), elapsed_total / 60)

    # Save results
    np.savez_compressed(
        out_dir / "cohort_results.npz",
        theta=all_thetas,
        log_theta=np.log(all_thetas),
        species_baseline=species[0],
        species_gvax=species[1],
        species_gvax_nivo=species[2],
        status=status,
        param_names=np.array(param_names),
        scenario_names=np.array([c[0] for c in SCENARIO_CONFIGS]),
        species_names=np.array([sp.name for sp in SP]),
        post_idx=np.array(post_indices),
        pat_idx=np.array(pat_indices),
        mu_draws=mu_draws,
        omega_draws=omega_draws,
        n_posterior=args.n_posterior,
        n_patients=args.n_patients,
    )

    # Summary
    reject_counts = {}
    for s in status:
        reject_counts[int(s)] = reject_counts.get(int(s), 0) + 1
    log.info("Status distribution: %s", reject_counts)

    # Per-posterior-draw admissibility
    post_arr = np.array(post_indices)
    for i in range(args.n_posterior):
        mask = post_arr == i
        n_i = mask.sum()
        ok_i = (status[mask] == 0).sum()
        log.info("  Posterior draw %d: %d/%d admissible (%.1f%%)",
                 i, ok_i, n_i, 100 * ok_i / max(n_i, 1))

    log.info("Results saved to %s/cohort_results.npz", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
