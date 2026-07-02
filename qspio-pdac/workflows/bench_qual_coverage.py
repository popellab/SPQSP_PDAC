"""Cost recon for the qual-bounds coverage rewrite (gh #57).

Times batched CppBatchRunner runs for the three qual-test scenarios at
N prior draws, with evolve_cache_root wired in so evolve-to-diagnosis
runs once per (theta, healthy-state-relevant-params) tuple instead of
once per scenario. Reports cold + warm wall-clock and evolve cache hit
rates so we can decide whether N=100 / N=500 is feasible for a coverage
test suite.

Usage:
    .venv/bin/python workflows/bench_qual_coverage.py [--n 50] [--clear-cache]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import yaml

from qsp_hpc.cpp.batch_runner import CppBatchRunner
from qsp_inference.priors import load_prior

REPO_ROOT = Path(__file__).resolve().parent.parent

SCENARIOS = (
    "baseline_no_treatment",
    "clinical_progression",
    "gvax_nivo_neoadjuvant_zheng2022",
)


def run_cfg() -> dict:
    res = REPO_ROOT / "resources/cpp"
    return {
        "binary_path": str(REPO_ROOT / "cpp/sim/build/qsp_sim"),
        "template_path": str(res / "param_all.xml"),
        "drug_metadata_yaml": str(res / "drug_metadata.yaml"),
        "healthy_state_yaml": str(res / "healthy_state.yaml"),
    }


def scenario_yaml(name: str) -> Path:
    return REPO_ROOT / "scenarios" / f"{name}.yaml"


def scenario_stop_time(name: str) -> float:
    with open(scenario_yaml(name)) as f:
        cfg = yaml.safe_load(f)
    return float(cfg["sim_config"]["stop_time"])


def sample_prior_thetas(n: int, seed: int = 2025) -> tuple[np.ndarray, list[str]]:
    csv = REPO_ROOT / "parameters/pdac_priors.csv"
    prior = load_prior(csv)
    torch.manual_seed(seed)
    samples = prior.sample((n,)).numpy().astype(np.float64)
    import pandas as pd
    names = pd.read_csv(csv)["name"].tolist()
    if samples.shape[1] != len(names):
        raise RuntimeError(
            f"prior sample width {samples.shape[1]} != priors CSV width {len(names)}"
        )
    return samples, names


def time_scenario(
    scenario: str,
    theta: np.ndarray,
    names: list[str],
    evolve_cache_root: Path,
    workdir: Path,
) -> tuple[float, int, int]:
    cfg = run_cfg()
    runner = CppBatchRunner(
        binary_path=cfg["binary_path"],
        template_path=cfg["template_path"],
        subtree="QSP",
        scenario_yaml=scenario_yaml(scenario),
        drug_metadata_yaml=cfg["drug_metadata_yaml"],
        healthy_state_yaml=cfg["healthy_state_yaml"],
        evolve_cache_root=str(evolve_cache_root),
    )
    t_end = scenario_stop_time(scenario)
    out_path = workdir / f"{scenario}.parquet"
    t0 = time.perf_counter()
    result = runner.run(
        theta_matrix=theta,
        param_names=names,
        t_end_days=t_end,
        dt_days=1.0,
        output_path=out_path,
        scenario=scenario,
        seed=0,
        workdir=workdir / f"{scenario}_work",
    )
    elapsed = time.perf_counter() - t0
    table = pq.read_table(str(out_path))
    df = table.to_pandas()
    n_ok = int((df["status"] == 0).sum()) if "status" in df.columns else len(df)
    return elapsed, n_ok, len(df)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2025)
    ap.add_argument(
        "--evolve-cache-root",
        type=Path,
        default=REPO_ROOT / "cache/qual_coverage_bench/evolve",
    )
    ap.add_argument(
        "--clear-cache",
        action="store_true",
        help="rm -rf the evolve cache before running (forces cold)",
    )
    args = ap.parse_args()

    binary = Path(run_cfg()["binary_path"])
    if not binary.exists():
        raise SystemExit(f"qsp_sim not built at {binary}; run `make refresh-cpp`")

    if args.clear_cache and args.evolve_cache_root.exists():
        print(f"clearing {args.evolve_cache_root}")
        shutil.rmtree(args.evolve_cache_root)
    args.evolve_cache_root.mkdir(parents=True, exist_ok=True)

    print(f"sampling {args.n} prior thetas (seed={args.seed})")
    theta, names = sample_prior_thetas(args.n, seed=args.seed)
    print(f"  theta: {theta.shape}, {len(names)} params")

    print()
    print(f"=== COLD pass (evolve cache: {'cleared' if args.clear_cache else 'as-is'}) ===")
    cold_times: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as td:
        for scen in SCENARIOS:
            elapsed, n_ok, n_total = time_scenario(
                scen, theta, names, args.evolve_cache_root, Path(td)
            )
            cold_times[scen] = elapsed
            per = elapsed / args.n
            print(
                f"  {scen:42s}  {elapsed:7.2f}s   "
                f"{per*1000:6.1f} ms/draw   ok={n_ok}/{n_total}"
            )

    print()
    print(f"=== WARM pass (evolve cache populated) ===")
    warm_times: dict[str, float] = {}
    with tempfile.TemporaryDirectory() as td:
        for scen in SCENARIOS:
            elapsed, n_ok, n_total = time_scenario(
                scen, theta, names, args.evolve_cache_root, Path(td)
            )
            warm_times[scen] = elapsed
            per = elapsed / args.n
            print(
                f"  {scen:42s}  {elapsed:7.2f}s   "
                f"{per*1000:6.1f} ms/draw   ok={n_ok}/{n_total}"
            )

    print()
    print("=== summary ===")
    cold_total = sum(cold_times.values())
    warm_total = sum(warm_times.values())
    print(f"  cold total: {cold_total:7.2f}s  ({cold_total/args.n*1000:6.1f} ms/draw)")
    print(f"  warm total: {warm_total:7.2f}s  ({warm_total/args.n*1000:6.1f} ms/draw)")
    speedup = cold_total / warm_total if warm_total > 0 else float("inf")
    print(f"  warm speedup: {speedup:.2f}x")
    print()
    print(f"  evolve cache contents: {sum(1 for _ in args.evolve_cache_root.rglob('*')):,} entries")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
