"""Morris sensitivity analysis for ABM spatial architecture (phi).

Generates the DOE matrix, runs ABM simulations from a shared placement file,
and computes delta-phi (phi_end - phi_placement) as the observable.

Each shard loads a pre-warmed QSP state + placement file, runs coupled
QSP+ABM presim until volume reaches 100% target, then measures phi.

Baseline directory must contain:
  - param_all_median.xml   — XML at posterior median
  - tumor.plc              — placement file from init_tumor.py
  - qsp_warmup.bin         — QSP ODE state at onset (from warmup run)
  - placement_phi.npz      — phi matrix of the placement (baseline)

Usage::

    # 1. Generate the Morris DOE and problem definition
    python -I scripts/run_abm_sa.py generate \
        --baseline-dir results/sa_baseline \
        --output-dir results/sa_morris \
        --trajectories 10 --levels 4 --replicates 3 --seed 42

    # 2. Run one shard (for SLURM array or local testing)
    python -I scripts/run_abm_sa.py run-shard \
        --sa-dir results/sa_morris \
        --shard-id 0

    # 3. Consolidate all shards
    python -I scripts/run_abm_sa.py consolidate \
        --sa-dir results/sa_morris
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ABM_BIN = REPO / "PDAC" / "sim" / "build" / "bin" / "pdac"
PYTHON = sys.executable


def _load_priors(csv_path: Path) -> list[dict]:
    """Load calibrate-type parameters from the ABM priors CSV."""
    params = []
    with open(csv_path) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 9:
                continue
            name = row[0].strip()
            vpop_type = row[8].strip()
            if vpop_type != "calibrate":
                continue
            median = row[1].strip()
            dist = row[3].strip()
            p1 = row[4].strip()
            p2 = row[5].strip()
            lb = row[6].strip()
            ub = row[7].strip()

            if not median:
                continue

            entry = {
                "name": name,
                "median": float(median),
                "distribution": dist,
            }
            if dist == "lognormal" and p1 and p2:
                mu = float(p1)
                sigma = float(p2)
                entry["bounds"] = [
                    np.exp(mu - 3 * sigma),
                    np.exp(mu + 3 * sigma),
                ]
            elif dist == "uniform" and lb and ub:
                entry["bounds"] = [float(lb), float(ub)]
            elif dist == "uniform" and p1 and p2:
                entry["bounds"] = [float(p1), float(p2)]
            else:
                med = float(median)
                entry["bounds"] = [med * 0.1, med * 10.0]

            params.append(entry)
    return params


def cmd_generate(args):
    """Generate Morris DOE matrix and problem definition."""
    from SALib.sample import morris as morris_sample

    baseline = Path(args.baseline_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    priors_csv = REPO / "PDAC" / "sim" / "resource" / "pdac_abm_priors.csv"
    params = _load_priors(priors_csv)
    log.info(f"Loaded {len(params)} calibrate parameters from priors CSV")

    problem = {
        "num_vars": len(params),
        "names": [p["name"] for p in params],
        "bounds": [p["bounds"] for p in params],
    }

    thetas = morris_sample.sample(
        problem,
        N=args.trajectories,
        num_levels=args.levels,
        seed=args.seed,
    )
    n_sets = thetas.shape[0]
    log.info(f"Morris DOE: {n_sets} parameter sets "
             f"({args.trajectories} trajectories × {len(params)+1} evaluations)")

    n_total = n_sets * args.replicates
    log.info(f"With {args.replicates} replicates: {n_total} total runs")

    np.save(out / "thetas.npy", thetas)
    with open(out / "theta_names.txt", "w") as f:
        for p in params:
            f.write(p["name"] + "\n")

    with open(out / "problem.json", "w") as f:
        json.dump(problem, f, indent=2)

    shutil.copy2(baseline / "placement_phi.npz", out / "placement_phi.npz")
    shutil.copy2(baseline / "tumor.plc", out / "baseline.plc")
    shutil.copy2(baseline / "param_all_median.xml", out / "param_all_median.xml")
    shutil.copy2(baseline / "qsp_warmup.bin", out / "qsp_warmup.bin")

    meta = {
        "n_params": len(params),
        "n_parameter_sets": n_sets,
        "n_replicates": args.replicates,
        "n_total_runs": n_total,
        "trajectories": args.trajectories,
        "levels": args.levels,
        "seed": args.seed,
        "baseline_dir": str(baseline),
        "medians": {p["name"]: p["median"] for p in params},
    }
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    (out / "shards").mkdir(exist_ok=True)

    log.info(f"DOE saved to {out}")
    log.info(f"  thetas.npy: {thetas.shape}")
    log.info(f"  Total shards to run: {n_total}")
    log.info(f"  Shard IDs: 0..{n_total-1}")
    log.info(f"  (set_id = shard_id // {args.replicates}, "
             f"rep = shard_id % {args.replicates})")


def cmd_run_shard(args):
    """Run one shard: one parameter set × one replicate."""
    sa_dir = Path(args.sa_dir)

    with open(sa_dir / "meta.json") as f:
        meta = json.load(f)
    n_reps = meta["n_replicates"]
    n_sets = meta["n_parameter_sets"]

    set_id = args.shard_id // n_reps
    rep_id = args.shard_id % n_reps

    if set_id >= n_sets:
        log.error(f"shard_id {args.shard_id} out of range "
                  f"(max {n_sets * n_reps - 1})")
        sys.exit(1)

    thetas = np.load(sa_dir / "thetas.npy")
    theta_row = thetas[set_id]

    with open(sa_dir / "theta_names.txt") as f:
        param_names = [line.strip() for line in f if line.strip()]

    shard_path = sa_dir / "shards" / f"shard_{args.shard_id:06d}.npz"
    if shard_path.exists() and not args.force:
        log.info(f"Shard {args.shard_id} already exists, skipping")
        return

    log.info(f"Shard {args.shard_id}: set={set_id}, rep={rep_id}")

    # Load baseline XML and modify with this theta
    tree = ET.parse(str(sa_dir / "param_all_median.xml"))
    leaf_map = {}
    for elem in tree.iter():
        param_name = elem.get("name")
        if param_name and elem.text is not None:
            leaf_map[param_name] = elem

    matched = 0
    for name, val in zip(param_names, theta_row):
        if name in leaf_map:
            elem = leaf_map[name]
            if elem.get("type") == "int":
                elem.text = str(int(round(val)))
            else:
                elem.text = f"{val:.15g}"
            matched += 1

    log.info(f"  Matched {matched}/{len(param_names)} params in XML")

    seed = args.base_seed + args.shard_id

    # Read grid size from placement metadata
    from PDAC.tools.placement_io import read_placement
    _, _, _, plc_meta = read_placement(str(sa_dir / "baseline.plc"))
    gx, gy, gz = plc_meta["shape"]

    with tempfile.TemporaryDirectory(prefix=f"sa_shard_{args.shard_id}_") as tmpdir:
        tmp = Path(tmpdir)
        xml_path = tmp / "param_all.xml"
        tree.write(str(xml_path), xml_declaration=True, encoding="utf-8")

        plc_path = sa_dir / "baseline.plc"
        qsp_path = sa_dir / "qsp_warmup.bin"
        out_dir = tmp / "sa_out"

        # Load QSP state + placement → volume-stopped presim → measure phi
        # -s 0: no main-sim steps, we measure phi at the presim endpoint
        cmd = [
            str(ABM_BIN),
            "-p", str(xml_path),
            "-gx", str(gx),
            "-gy", str(gy),
            "-gz", str(gz),
            "--placement", str(plc_path),
            "--qsp-state", str(qsp_path),
            "-s", "0",
            "--seed", str(seed),
            "-G", "1",
            "-oi", "1000000",
            "--output-root", str(out_dir),
        ]
        log.info(f"  Running ABM: placement + volume-stopped presim, grid {gx}x{gy}x{gz}")
        result = subprocess.run(cmd, capture_output=True, text=True,
                                timeout=args.timeout)
        if result.returncode != 0:
            log.error(f"  ABM failed (exit {result.returncode})")
            log.error(f"  stderr: {result.stderr[-500:]}")
            np.savez(shard_path,
                     shard_id=args.shard_id,
                     set_id=set_id,
                     rep_id=rep_id,
                     status=-1,
                     error=result.stderr[-500:])
            return

        # Post-presim state is agents_step_000000.abm.lz4 (day-0 of main sim).
        # With -s 0, no further main-sim steps run — this is the endpoint.
        abm_dir = out_dir / "abm"
        final_snap = abm_dir / "agents_step_000000.abm.lz4"
        if not final_snap.exists():
            # Fall back to last presim snapshot
            presim_files = sorted(abm_dir.glob("agents_presim_*.abm.lz4"))
            if presim_files:
                final_snap = presim_files[-1]
            else:
                log.error("  No ABM output files found")
                np.savez(shard_path,
                         shard_id=args.shard_id,
                         set_id=set_id,
                         rep_id=rep_id,
                         status=-2,
                         error="No output files")
                return
        log.info(f"  Reading final state: {final_snap.name}")

        # Compute phi from final ABM state
        from PDAC.tools import io as abm_io
        from PDAC.tools.funcn_core import funcn_phi
        from PDAC.tools.init_tumor import SA_CATEGORIES
        from PDAC.tools.placement_io import CATEGORY_TO_AGENT

        df = abm_io.load_abm_lz4(str(final_snap))
        df = abm_io.add_category(df, mode="type_state")

        # Map ABM category names (from io._STATE_NAMES) → SA category names
        # io uses e.g. "cancer_prog", "tcell_eff", "mac_M1", "fib_myCAF"
        # which match CATEGORY_TO_AGENT keys exactly
        pos_by_cat: dict[str, list] = {c: [] for c in SA_CATEGORIES}
        for _, row in df.iterrows():
            cat = row["category"]
            if cat in pos_by_cat:
                pos_by_cat[cat].append([int(row["x"]), int(row["y"]), int(row["z"])])

        pos_arrays = {}
        for cat in SA_CATEGORIES:
            if pos_by_cat[cat]:
                pos_arrays[cat] = np.array(pos_by_cat[cat], dtype=np.int64)
            else:
                pos_arrays[cat] = np.empty((0, 3), dtype=np.int64)

        shape = (gx, gy, gz)
        sigma_vox = 2.5
        categories = [c for c in SA_CATEGORIES if len(pos_arrays[c]) > 0]
        phi_end = funcn_phi(pos_arrays, categories, shape, sigma_vox=sigma_vox)

        # Load placement phi baseline
        baseline_data = np.load(sa_dir / "placement_phi.npz")
        phi_init = baseline_data["phi"]
        init_cats = list(baseline_data["categories"])

        # Compute delta-phi (align categories to SA_CATEGORIES ordering)
        n_cat = len(SA_CATEGORIES)
        delta_phi = np.full((n_cat, n_cat), np.nan)
        for i, ci in enumerate(SA_CATEGORIES):
            for j, cj in enumerate(SA_CATEGORIES):
                end_val = np.nan
                if ci in categories and cj in categories:
                    ei = categories.index(ci)
                    ej = categories.index(cj)
                    end_val = phi_end[ei, ej]
                init_val = np.nan
                if ci in init_cats and cj in init_cats:
                    ii = init_cats.index(ci)
                    ij = init_cats.index(cj)
                    init_val = phi_init[ii, ij]
                if not np.isnan(end_val) and not np.isnan(init_val):
                    delta_phi[i, j] = end_val - init_val

        pop_counts = {cat: len(pos_arrays.get(cat, [])) for cat in SA_CATEGORIES}

        np.savez(shard_path,
                 shard_id=args.shard_id,
                 set_id=set_id,
                 rep_id=rep_id,
                 status=0,
                 delta_phi=delta_phi,
                 phi_end=phi_end,
                 categories_end=np.array(categories),
                 pop_counts=np.array([pop_counts.get(c, 0) for c in SA_CATEGORIES]),
                 sa_categories=np.array(SA_CATEGORIES))

        log.info(f"  Shard saved: {shard_path}")
        log.info(f"  Population: {sum(pop_counts.values())} total cells")


def cmd_consolidate(args):
    """Consolidate shards into a single observable matrix and run Morris analysis."""
    sa_dir = Path(args.sa_dir)

    with open(sa_dir / "meta.json") as f:
        meta = json.load(f)

    n_sets = meta["n_parameter_sets"]
    n_reps = meta["n_replicates"]
    n_total = n_sets * n_reps

    from PDAC.tools.init_tumor import SA_CATEGORIES
    n_cat = len(SA_CATEGORIES)
    n_obs = n_cat * n_cat

    all_delta_phi = np.full((n_total, n_obs), np.nan)
    all_pop = np.full((n_total, n_cat), np.nan)
    status = np.full(n_total, -99, dtype=int)

    shard_dir = sa_dir / "shards"
    found = 0
    for sid in range(n_total):
        shard_path = shard_dir / f"shard_{sid:06d}.npz"
        if not shard_path.exists():
            continue
        data = np.load(shard_path, allow_pickle=True)
        status[sid] = int(data["status"])
        if status[sid] == 0:
            dp = data["delta_phi"]
            all_delta_phi[sid] = dp.ravel()
            all_pop[sid] = data["pop_counts"]
        found += 1

    log.info(f"Found {found}/{n_total} shards ({np.sum(status == 0)} successful)")

    avg_delta_phi = np.full((n_sets, n_obs), np.nan)
    avg_pop = np.full((n_sets, n_cat), np.nan)
    for s in range(n_sets):
        rep_indices = [s * n_reps + r for r in range(n_reps)]
        valid = [i for i in rep_indices if status[i] == 0]
        if valid:
            avg_delta_phi[s] = np.nanmean(all_delta_phi[valid], axis=0)
            avg_pop[s] = np.nanmean(all_pop[valid], axis=0)

    obs_names = []
    for ci in SA_CATEGORIES:
        for cj in SA_CATEGORIES:
            obs_names.append(f"dphi_{ci}__{cj}")

    np.savez(sa_dir / "observables.npz",
             delta_phi_all=all_delta_phi,
             delta_phi_avg=avg_delta_phi,
             pop_all=all_pop,
             pop_avg=avg_pop,
             obs_names=np.array(obs_names),
             sa_categories=np.array(SA_CATEGORIES),
             status=status)

    with open(sa_dir / "obs_names.txt", "w") as f:
        for name in obs_names:
            f.write(name + "\n")

    log.info(f"Consolidated observables saved to {sa_dir / 'observables.npz'}")
    log.info(f"  {n_sets} parameter sets × {n_obs} observables")

    # Run Morris analysis
    from SALib.analyze import morris as morris_analyze
    with open(sa_dir / "problem.json") as f:
        problem = json.load(f)

    thetas = np.load(sa_dir / "thetas.npy")

    results_dir = sa_dir / "results"
    results_dir.mkdir(exist_ok=True)

    all_results = []
    for oi, obs_name in enumerate(obs_names):
        Y = avg_delta_phi[:, oi]
        valid = ~np.isnan(Y)
        if valid.sum() < len(thetas):
            continue
        try:
            si = morris_analyze.analyze(problem, thetas, Y)
            for pi, pname in enumerate(problem["names"]):
                all_results.append({
                    "observable": obs_name,
                    "parameter": pname,
                    "mu_star": si["mu_star"][pi],
                    "sigma": si["sigma"][pi],
                    "mu": si["mu"][pi],
                    "mu_star_conf": si["mu_star_conf"][pi],
                })
        except Exception as e:
            log.warning(f"  Morris failed for {obs_name}: {e}")

    if all_results:
        with open(results_dir / "morris_results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["observable", "parameter",
                                               "mu_star", "sigma", "mu",
                                               "mu_star_conf"])
            w.writeheader()
            w.writerows(all_results)
        log.info(f"Morris results: {len(all_results)} rows → "
                 f"{results_dir / 'morris_results.csv'}")
    else:
        log.warning("No Morris results computed (insufficient valid shards)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate Morris DOE matrix")
    gen.add_argument("--baseline-dir", required=True)
    gen.add_argument("--output-dir", required=True)
    gen.add_argument("--trajectories", type=int, default=10)
    gen.add_argument("--levels", type=int, default=4)
    gen.add_argument("--replicates", type=int, default=3)
    gen.add_argument("--seed", type=int, default=42)

    rs = sub.add_parser("run-shard", help="Run one SA shard")
    rs.add_argument("--sa-dir", required=True)
    rs.add_argument("--shard-id", type=int, required=True)
    rs.add_argument("--base-seed", type=int, default=1000)
    rs.add_argument("--timeout", type=int, default=7200)
    rs.add_argument("--force", action="store_true")

    con = sub.add_parser("consolidate", help="Consolidate shards and run Morris")
    con.add_argument("--sa-dir", required=True)

    args = p.parse_args()
    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "run-shard":
        cmd_run_shard(args)
    elif args.command == "consolidate":
        cmd_consolidate(args)


if __name__ == "__main__":
    main()
