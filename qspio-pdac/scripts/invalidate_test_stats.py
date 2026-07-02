"""Invalidate cached derived test statistics (HPC + local) for a rerun.

Use when the derivation worker code changed but calibration target YAMLs
did not (so test_stats_hash is identical and the workflow would otherwise
skip re-derivation).

- Deletes `test_stats/<hash>/` on HPC for each scenario (raw parquets are kept).
- Deletes local batch `.mat` files (raw sims stay on HPC; next run re-downloads
  freshly-derived observables).
"""

from __future__ import annotations

import shutil
from pathlib import Path

from qsp_hpc.constants import HASH_PREFIX_LENGTH
from qsp_hpc.simulation.qsp_simulator import QSPSimulator

SCENARIOS = [
    ("baseline_no_treatment", "calibration_targets/baseline_no_treatment"),
    ("clinical_progression", "calibration_targets/clinical_progression"),
    ("gvax_neoadjuvant_zheng2022", "calibration_targets/gvax_neoadjuvant"),
    ("gvax_nivo_neoadjuvant_zheng2022", "calibration_targets/gvax_nivo_neoadjuvant"),
]

COMMON = dict(
    priors_csv="parameters/pdac_priors.csv",
    submodel_priors_yaml="notes/calibration/submodel_priors.yaml",
    model_structure_file="model_structure.json",
    model_script="immune_oncology_model_PDAC",
    model_version="multi_scenario_v4_all_scenarios",
    max_tasks=1,
    seed=2025,
    cache_sampling_seed=2025,
)


def invalidate(scenario: str, cal_dir: str, dry_run: bool) -> None:
    sim = QSPSimulator(scenario=scenario, calibration_targets=cal_dir, **COMMON)
    test_stats_hash = sim._compute_test_stats_hash()
    priors_hash = sim._compute_priors_hash()
    hpc_pool_id = f"{sim.model_version}_{priors_hash[:HASH_PREFIX_LENGTH]}_{scenario}"
    hpc_pool_path = f"{sim.job_manager.config.simulation_pool_path}/{hpc_pool_id}"
    hpc_stats_dir = f"{hpc_pool_path}/test_stats/{test_stats_hash}"

    local_pool = Path(sim.pool.pool_dir)
    mats = sorted(local_pool.glob("batch_*.mat"))
    manifest = local_pool / "manifest.json"

    print(f"\n=== {scenario} ===")
    print(f"  HPC test_stats: {hpc_stats_dir}")
    print(f"  Local pool:    {local_pool}  ({len(mats)} batches)")

    if dry_run:
        status, out = sim.job_manager.transport.exec(f'ls -la "{hpc_stats_dir}" 2>&1 | head -3')
        print(f"  [dry-run] HPC ls: {out.strip()[:200]}")
        return

    status, out = sim.job_manager.transport.exec(f'rm -rf "{hpc_stats_dir}" && echo OK')
    print(f"  rm HPC: status={status}, out={out.strip()}")
    status2, check = sim.job_manager.transport.exec(f'test -d "{hpc_stats_dir}" && echo EXISTS || echo GONE')
    print(f"  verify: {check.strip()}")
    for m in mats:
        m.unlink()
    if manifest.exists():
        manifest.unlink()
    if local_pool.exists() and not any(local_pool.iterdir()):
        local_pool.rmdir()
    print("  deleted local batches + manifest")


if __name__ == "__main__":
    import sys
    dry_run = "--execute" not in sys.argv
    if dry_run:
        print("DRY RUN — pass --execute to actually delete")
    for name, cal in SCENARIOS:
        invalidate(name, cal, dry_run=dry_run)
    if dry_run:
        print("\n[dry-run] Re-run with --execute to invalidate.")
    else:
        print("\nDone. Next sbi_workflow run will re-derive on HPC with updated library.")
