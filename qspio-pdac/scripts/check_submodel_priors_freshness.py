#!/usr/bin/env python3
"""Check whether `notes/calibration/submodel_priors.yaml` is fresh.

Reads the per-component fingerprint manifest stamped into the YAML
header by ``regen_submodel_priors.py`` / ``parameter_audit.py``, then
recomputes a live manifest from the current tree. Reports any
component whose inputs (target YAML bytes, prior-CSV rows, submodel
config slice, reference values, qsp-inference version) have drifted
since the prior was last regenerated.

Exit codes:
    0 — fresh (or freshness header absent and ``--allow-missing`` set)
    1 — stale, missing components, or version drift detected
    2 — submodel_priors.yaml itself is missing

Usage:
    .venv/bin/python scripts/check_submodel_priors_freshness.py
        Default paths: submodel_priors.yaml at notes/calibration/,
        submodel targets at calibration_targets/submodel_targets/,
        priors CSV at parameters/pdac_priors.csv.

    .venv/bin/python scripts/check_submodel_priors_freshness.py \\
        --submodel-priors-yaml path/to/other.yaml

    .venv/bin/python scripts/check_submodel_priors_freshness.py --quiet
        Print only the one-line headline; suppress the per-component diffs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from qsp_inference.submodel.freshness import check_submodel_priors_freshness


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check submodel_priors.yaml freshness against current tree.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root (default: cwd).",
    )
    parser.add_argument(
        "--submodel-priors-yaml",
        type=Path,
        default=None,
        help="Path to submodel_priors.yaml (default: <root>/notes/calibration/submodel_priors.yaml).",
    )
    parser.add_argument(
        "--submodel-dir",
        type=Path,
        default=None,
        help="Submodel target directory (default: <root>/calibration_targets/submodel_targets).",
    )
    parser.add_argument(
        "--priors-csv",
        type=Path,
        default=None,
        help="Priors CSV path (default: <root>/parameters/pdac_priors.csv).",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Treat a missing freshness header as OK (legacy submodel_priors.yaml "
            "from before fingerprinting was added). Default: error out so the "
            "user knows to regenerate."
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Print only the one-line headline; suppress per-component diffs.",
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    yaml_path = args.submodel_priors_yaml or root / "notes" / "calibration" / "submodel_priors.yaml"
    submodel_dir = args.submodel_dir or root / "calibration_targets" / "submodel_targets"
    priors_csv = args.priors_csv or root / "parameters" / "pdac_priors.csv"

    if not yaml_path.exists():
        print(f"ERROR: {yaml_path} does not exist.")
        return 2

    report = check_submodel_priors_freshness(yaml_path, submodel_dir, priors_csv)

    # Distinguish "fresh because no fingerprints exist" from "genuinely fresh"
    import yaml as _yaml

    with open(yaml_path) as f:
        stored = _yaml.safe_load(f) or {}
    has_header = bool(stored.get("metadata", {}).get("freshness"))

    if not has_header:
        if args.allow_missing:
            print(f"{yaml_path}: no freshness header (pre-fingerprint vintage); --allow-missing set, OK.")
            return 0
        print(
            f"{yaml_path}: no freshness header. Regenerate with "
            f"`python scripts/regen_submodel_priors.py --rebuild-all` to add fingerprints, "
            f"or pass --allow-missing to ignore."
        )
        return 1

    if report.is_fresh():
        print(f"{yaml_path}: FRESH (all components match current tree).")
        return 0

    if args.quiet:
        n_stale = len(report.stale)
        n_miss_yaml = len(report.missing_in_yaml)
        n_miss_disk = len(report.missing_on_disk)
        ver = "version drift" if report.version_drift else ""
        bits = [s for s in (
            f"{n_stale} stale" if n_stale else "",
            f"{n_miss_yaml} new components on disk" if n_miss_yaml else "",
            f"{n_miss_disk} components missing on disk" if n_miss_disk else "",
            ver,
        ) if s]
        print(f"{yaml_path}: STALE — {'; '.join(bits)}.")
    else:
        print(report.format())
        print()
        print(
            "To regenerate: `python scripts/regen_submodel_priors.py "
            "--invalidate <PARAM> [<PARAM> ...]` for the affected parameters, "
            "or `--rebuild-all` for a clean slate."
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
