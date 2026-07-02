#!/usr/bin/env python3
"""Generate sweep configs + a single SLURM array sbatch, and optionally submit.

Reads a YAML sweep spec, expands the Cartesian product of axes, writes one
RunConfig JSON per cell plus one `sweep.sbatch` array script that covers all
cells (indexed by $SLURM_ARRAY_TASK_ID via config_list.txt).

Spec format:

    sweep_name: seed_flow_nsims_27cell
    output_dir: sweep_runs                    # parent dir (relative to project root)
    fixed:                                    # applied to every cell
      prcc_tier: null
      run_posterior_predictive: false
    axes:                                     # Cartesian product
      - name: seed
        values: [2025, 2029, 2033]
      - name: flow
        values:
          - {npe_hidden_features: 128, npe_num_transforms: 7}
          - {npe_hidden_features: 192, npe_num_transforms: 9}
          - {npe_hidden_features: 256, npe_num_transforms: 11}
      - name: num_total_simulations
        values: [2000, 3500, 5000]
    sbatch:                                   # optional; overrides defaults
      partition: shared
      time: "03:00:00"
      cpus_per_task: 4
      mem: "32G"

Usage:
    python workflows/sweep_submit.py sweep_specs/seed_flow_nsims.yaml
    python workflows/sweep_submit.py sweep_specs/... --dry-run
    python workflows/sweep_submit.py sweep_specs/... --no-submit   # write files only
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import shutil
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from workflows.sbi_runner import RunConfig  # noqa: E402


# Default Rockfish-friendly sbatch parameters. Override via `sbatch:` in spec.
DEFAULT_SBATCH = {
    "partition": "shared",
    "time": "03:00:00",
    "cpus_per_task": 4,
    "mem": "32G",
}


ARRAY_SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={sweep_name}
#SBATCH --partition={partition}
#SBATCH --time={time}
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --array=0-{n_last}{concurrency_suffix}
#SBATCH --output={log_dir}/%A_%a.out
#SBATCH --error={log_dir}/%A_%a.err

set -euo pipefail

cd {project_root}
source {venv_activate}

# Pull latest qsp-hpc-tools and qsp-inference from main BEFORE any Python imports,
# so sweep cells always run against merged-to-main code. Must happen outside
# Python: once Python has imported a module, later pip upgrades don't affect it.
if [[ "${{SKIP_SWEEP_UPGRADE:-0}}" != "1" ]]; then
    echo "[sbatch] Upgrading qsp-hpc-tools + qsp-inference from main..."
    uv pip install --quiet --upgrade \\
        "qsp-hpc-tools @ git+ssh://git@github.com/popellab/qsp-hpc-tools.git@main" \\
        "qsp-inference @ git+ssh://git@github.com/popellab/qsp-inference.git@main" || \\
        echo "[sbatch] WARNING: dep upgrade failed; continuing with current venv"
fi

# Pick the config for this array index from the ordered list.
CONFIG=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" {config_list})
echo "[array $SLURM_ARRAY_TASK_ID] config=$CONFIG"

python -m workflows.sweep_run_one "$CONFIG"
"""


def _cell_hash(overrides: dict) -> str:
    """Deterministic 8-char hash of a cell's overrides (excluding save_dir)."""
    canon = {k: v for k, v in overrides.items() if k != "save_dir"}
    payload = json.dumps(canon, sort_keys=True, default=str).encode()
    return hashlib.md5(payload).hexdigest()[:8]


def _validate_runconfig_keys(overrides: dict, axis_name: str) -> None:
    valid = set(asdict(RunConfig()).keys())
    bad = [k for k in overrides if k not in valid]
    if bad:
        raise ValueError(f"Axis {axis_name!r} introduces unknown RunConfig keys: {bad}")


def expand_cells(spec: dict) -> list[dict]:
    """Cartesian product of axes, merged with fixed overrides."""
    fixed = dict(spec.get("fixed", {}) or {})
    _validate_runconfig_keys(fixed, "fixed")
    axes = spec.get("axes", []) or []

    # Normalize axis values: scalar → {axis.name: value}; dict → merge as-is.
    expanded: list[list[dict]] = []
    for ax in axes:
        name = ax["name"]
        entries = []
        for val in ax["values"]:
            if isinstance(val, dict):
                _validate_runconfig_keys(val, name)
                entries.append(dict(val))
            else:
                _validate_runconfig_keys({name: val}, name)
                entries.append({name: val})
        expanded.append(entries)

    cells = []
    for combo in itertools.product(*expanded):
        merged = dict(fixed)
        for entry in combo:
            merged.update(entry)
        cells.append(merged)
    return cells


def write_cell_config(cell: dict, sweep_dir: Path) -> tuple[str, Path]:
    job_hash = _cell_hash(cell)
    cfg = dict(cell)
    save_dir = sweep_dir / "runs" / job_hash
    cfg["save_dir"] = str(save_dir)
    cfg_path = sweep_dir / "configs" / f"{job_hash}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, default=str))
    return job_hash, cfg_path


def write_array_sbatch(
    sweep_dir: Path,
    sweep_name: str,
    config_list_path: Path,
    n_cells: int,
    sbatch_params: dict,
    project_root: Path,
    venv_activate: str,
    max_concurrent: int | None,
) -> Path:
    concurrency_suffix = f"%{max_concurrent}" if max_concurrent else ""
    sbatch_path = sweep_dir / "sweep.sbatch"
    sbatch_path.write_text(
        ARRAY_SBATCH_TEMPLATE.format(
            sweep_name=sweep_name,
            n_last=n_cells - 1,
            concurrency_suffix=concurrency_suffix,
            log_dir=sweep_dir / "logs",
            project_root=project_root,
            venv_activate=venv_activate,
            config_list=config_list_path,
            **sbatch_params,
        )
    )
    return sbatch_path


def submit_array(sbatch_path: Path) -> str:
    """Submit array sbatch; return the job id."""
    result = subprocess.run(
        ["sbatch", str(sbatch_path)],
        capture_output=True,
        text=True,
        check=True,
    )
    # "Submitted batch job 12345"
    return result.stdout.strip().split()[-1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit an SBI sweep to SLURM")
    parser.add_argument("spec_yaml", type=Path)
    parser.add_argument(
        "--project-root",
        default=None,
        help="Project root on the cluster (defaults to local repo root).",
    )
    parser.add_argument(
        "--venv-activate",
        default=None,
        help="Path to `source`-able venv activate script on the cluster. "
        "Defaults to <project_root>/.venv/bin/activate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write nothing; just print the cell plan.",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Write configs + sbatch scripts but do not call `sbatch`.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete sweep_dir if it already exists.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=None,
        help="Cap simultaneous array tasks (SLURM --array=0-N%%K). Default: no cap.",
    )
    args = parser.parse_args()

    spec = yaml.safe_load(args.spec_yaml.read_text())
    sweep_name = spec["sweep_name"]
    output_root = Path(spec.get("output_dir", "sweep_runs"))
    sweep_dir = output_root / sweep_name

    sbatch_params = dict(DEFAULT_SBATCH)
    sbatch_params.update(spec.get("sbatch", {}) or {})

    cells = expand_cells(spec)
    print(f"Sweep '{sweep_name}': {len(cells)} cells")

    if args.dry_run:
        for cell in cells:
            print(f"  {_cell_hash(cell)}: {cell}")
        return

    if sweep_dir.exists():
        if args.overwrite:
            shutil.rmtree(sweep_dir)
        else:
            print(f"ERROR: {sweep_dir} exists. Pass --overwrite or pick a new sweep_name.")
            sys.exit(1)

    for sub in ("configs", "logs", "runs"):
        (sweep_dir / sub).mkdir(parents=True)

    project_root = Path(args.project_root).resolve() if args.project_root else _ROOT
    venv_activate = args.venv_activate or str(project_root / ".venv" / "bin" / "activate")

    # Write one config per cell; record ordered list for array indexing.
    manifest: list[dict[str, Any]] = []
    config_list_path = sweep_dir / "config_list.txt"
    with open(config_list_path, "w") as f:
        for array_idx, cell in enumerate(cells):
            job_hash, cfg_path = write_cell_config(cell, sweep_dir)
            f.write(f"{cfg_path}\n")
            manifest.append(
                {
                    "array_idx": array_idx,
                    "job_hash": job_hash,
                    "overrides": cell,
                    "config_json": str(cfg_path),
                }
            )

    sbatch_path = write_array_sbatch(
        sweep_dir=sweep_dir,
        sweep_name=sweep_name,
        config_list_path=config_list_path,
        n_cells=len(cells),
        sbatch_params=sbatch_params,
        project_root=project_root,
        venv_activate=venv_activate,
        max_concurrent=args.max_concurrent,
    )
    print(f"Wrote array sbatch: {sbatch_path}")

    array_job_id: str | None = None
    if not args.no_submit:
        try:
            array_job_id = submit_array(sbatch_path)
            print(f"Submitted array job {array_job_id} ({len(cells)} tasks)")
        except subprocess.CalledProcessError as exc:
            print(f"SUBMIT FAILED: {exc.stderr.strip()}")
        except FileNotFoundError:
            print("sbatch not found on PATH — write only; submit on the cluster with:")
            print(f"  sbatch {sbatch_path}")

    manifest_path = sweep_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "sweep_name": sweep_name,
                "n_cells": len(cells),
                "spec": spec,
                "project_root": str(project_root),
                "venv_activate": venv_activate,
                "array_job_id": array_job_id,
                "sbatch_path": str(sbatch_path),
                "config_list": str(config_list_path),
                "cells": manifest,
            },
            indent=2,
            default=str,
        )
    )
    print(f"Wrote manifest: {manifest_path}")

    if args.no_submit:
        print(f"\nTo submit on the cluster:")
        print(f"  sbatch {sbatch_path}")


if __name__ == "__main__":
    main()