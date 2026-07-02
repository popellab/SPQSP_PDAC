"""Small 2-scenario parallel run_hpc() smoke test — validates PR #49.

Fires clinical_progression + gvax_nivo_neoadjuvant_zheng2022 as concurrent
run_hpc() calls with a tiny 800-sim pool each. After both complete,
asserts each scenario's pool dir got its own chunks (the exact failure
mode observed 2026-04-20 with the parallel fan-out was array jobs
submitted for scenario A writing their chunks into scenario B's pool).

Run:
    .venv/bin/python workflows/smoke_parallel_hpc_fanout.py

Expected wall time: ~3-5 min (warm evolve cache from the earlier
baseline_no_treatment run makes scenarios 2..N cheap).
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# CppSimulator and HPCJobManager come from qsp-hpc-tools.
from qsp_hpc.batch.hpc_job_manager import HPCJobManager
from qsp_hpc.simulation.cpp_simulator import CppSimulator

# Config mirrors workflows/sbi_runner.py defaults.
PRIORS_CSV = "parameters/pdac_priors.csv"
MODEL_STRUCTURE_FILE = "model_structure.json"
SUBMODEL_PRIORS_YAML = "notes/calibration/submodel_priors.yaml"
SCENARIOS_DIR = "scenarios"
MODEL_VERSION = "multi_scenario_v4_all_scenarios"

_SPQSP_ROOT = Path("/Users/joeleliason/Projects/SPQSP_PDAC")
CPP_BINARY_PATH = str(_SPQSP_ROOT / "PDAC/qsp/sim/build/qsp_sim")
CPP_TEMPLATE_XML = str(_SPQSP_ROOT / "PDAC/sim/resource/param_all.xml")
CPP_SUBTREE = "QSP"
CPP_DRUG_METADATA_YAML = str(_SPQSP_ROOT / "PDAC/sim/resource/drug_metadata.yaml")
CPP_HEALTHY_STATE_YAML = str(_SPQSP_ROOT / "PDAC/sim/resource/healthy_state.yaml")
CPP_T_END_DAYS = 21.0  # both scenarios here use a treatment window
CPP_DT_DAYS = 1.0

SEED = 2025
N_TOTAL = 800  # 2 array tasks × 400 sims — smallest shape that exercises the
# 'populate evolve cache' → 'reuse across scenarios' flow while staying cheap.

RESTRICTION_CLASSIFIER_DIR = "cache/restriction_classifier"
RESTRICTION_THRESHOLD = 0.5

# Two scenarios that had empty pool dirs after the 2026-04-20 cleanup.
# baseline_no_treatment runs first (sequentially) because its dir is clean
# and populates the shared evolve_cache that the other two reuse.
FAN_OUT_SCENARIOS = {
    "clinical_progression": {
        "calibration_targets": "calibration_targets/clinical_progression",
        "t_end_days": 90.0,
    },
    "gvax_nivo_neoadjuvant_zheng2022": {
        "calibration_targets": "calibration_targets/gvax_nivo_neoadjuvant",
        "t_end_days": 21.0,
    },
}


def build_sim(scen_name: str, scen_cfg: dict, job_manager: HPCJobManager) -> CppSimulator:
    scenario_yaml = f"{SCENARIOS_DIR}/{scen_name}.yaml"
    return CppSimulator(
        priors_csv=PRIORS_CSV,
        binary_path=CPP_BINARY_PATH,
        template_xml=CPP_TEMPLATE_XML,
        subtree=CPP_SUBTREE,
        model_version=MODEL_VERSION,
        scenario=scen_name,
        submodel_priors_yaml=SUBMODEL_PRIORS_YAML,
        scenario_yaml=scenario_yaml,
        drug_metadata_yaml=CPP_DRUG_METADATA_YAML,
        healthy_state_yaml=CPP_HEALTHY_STATE_YAML,
        calibration_targets=scen_cfg["calibration_targets"],
        model_structure_file=MODEL_STRUCTURE_FILE,
        job_manager=job_manager,
        t_end_days=scen_cfg.get("t_end_days", CPP_T_END_DAYS),
        dt_days=CPP_DT_DAYS,
        seed=SEED,
        theta_pool_size=max(N_TOTAL, 100_000),
        restriction_classifier_dir=RESTRICTION_CLASSIFIER_DIR,
        restriction_threshold=RESTRICTION_THRESHOLD,
        verbose=True,
    )


def main() -> None:
    t0 = time.time()
    job_manager = HPCJobManager(verbose=True)
    # Use the same --cpp-branch as sbi_runner. cpp-sweep-binary-io is the
    # current branch tag for the restricted-pool work.
    job_manager.config.cpp_branch = "cpp-sweep-binary-io"
    job_manager.config.jobs_per_chunk = 400

    print(f"Building {len(FAN_OUT_SCENARIOS)} CppSimulators...")
    sims = {
        name: build_sim(name, cfg, job_manager)
        for name, cfg in FAN_OUT_SCENARIOS.items()
    }

    # Fire both scenarios concurrently. No sequential first-scenario warm-up
    # here — the baseline_no_treatment pool's evolve cache is already warm on
    # HPC from an earlier run, so both these scenarios should hit cache.
    print(f"\nLaunching parallel fan-out (max_workers={len(sims)})...")
    results: dict[str, tuple] = {}
    with ThreadPoolExecutor(max_workers=len(sims)) as executor:
        futures = {executor.submit(sim.run_hpc, N_TOTAL): name for name, sim in sims.items()}
        for fut in as_completed(futures):
            name = futures[fut]
            try:
                results[name] = fut.result()
                theta, x = results[name]
                print(f"  ✓ {name}: theta={theta.shape}, x={x.shape}")
            except Exception as e:
                print(f"  ✗ {name}: {type(e).__name__}: {e}")
                raise

    dt = time.time() - t0
    print(f"\n✓ Parallel fan-out complete in {dt:.1f}s")
    print("\nIf both scenarios reported non-trivial theta/x shapes and no")
    print("'git pull failed' errors appeared in the logs, the pool-scoped")
    print("upload paths + flock are working as intended.")


if __name__ == "__main__":
    main()
