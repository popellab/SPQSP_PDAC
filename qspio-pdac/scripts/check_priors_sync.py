"""Diff parameters_PDAC.m against pdac_priors.csv.

Fails (exit 1) if any name mismatch or duplicate is detected. Warns (exit 0)
on params present in parameters_PDAC.m but not in priors — not every
parameter must be sampled by SBI, but an unnoticed omission of a newly-added
SBI parameter is exactly the class of error this script exists to catch.

Usage: .venv/bin/python scripts/check_priors_sync.py
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PARAMS_M = REPO / "parameters" / "parameters_PDAC.m"
PRIORS_CSV = REPO / "parameters" / "pdac_priors.csv"
MODULES_DIR = REPO / "core" / "modules"
MAIN_MODEL = REPO / "scripts" / "immune_oncology_model_PDAC.m"

# `params.<name>.Value = ...` in parameters_PDAC.m
PARAM_RE = re.compile(r"^\s*params\.([A-Za-z_][A-Za-z0-9_]*)\.Value\s*=", re.M)
# `addparameter(model,'<name>',...)` inside a module.
ADDPARAM_RE = re.compile(
    r"addparameter\s*\(\s*model\s*,\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
)
# `model = <modulename>(model, ...)` call in immune_oncology_model_PDAC.m.
MODULE_CALL_RE = re.compile(
    r"^\s*model\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M
)


def params_from_matlab(path: Path) -> set[str]:
    return set(PARAM_RE.findall(path.read_text()))


def active_modules(main_model: Path) -> set[str]:
    """Return the set of module-function names called from the main model script."""
    return set(MODULE_CALL_RE.findall(main_model.read_text()))


def params_from_active_modules(modules_dir: Path, active: set[str]) -> set[str]:
    """Only scan modules that are actually invoked — a defined-but-never-called
    module (e.g., chemotaxis_module.m) contributes no live parameters."""
    names: set[str] = set()
    for m in modules_dir.glob("*.m"):
        if m.stem not in active:
            continue
        names.update(ADDPARAM_RE.findall(m.read_text()))
    return names


def params_from_priors(path: Path) -> list[str]:
    with path.open() as f:
        return [row["name"] for row in csv.DictReader(f)]


def main() -> int:
    m_params = params_from_matlab(PARAMS_M)
    active = active_modules(MAIN_MODEL)
    mod_params = params_from_active_modules(MODULES_DIR, active)
    live = m_params | mod_params  # "live" = declared in parameters_PDAC.m OR
                                  # addparameter'd by an actively-called module
    csv_names = params_from_priors(PRIORS_CSV)
    csv_params = set(csv_names)

    problems: list[str] = []
    warnings: list[str] = []

    dups = [n for n in csv_names if csv_names.count(n) > 1]
    if dups:
        problems.append(f"Duplicate rows in {PRIORS_CSV.name}: {sorted(set(dups))}")

    in_csv_not_live = csv_params - live
    if in_csv_not_live:
        problems.append(
            f"In {PRIORS_CSV.name} but not declared anywhere in the model: "
            f"{sorted(in_csv_not_live)}"
        )

    in_m_not_csv = m_params - csv_params
    if in_m_not_csv:
        warnings.append(
            f"In {PARAMS_M.name} but not in {PRIORS_CSV.name} "
            f"(fine if not SBI-sampled): {sorted(in_m_not_csv)[:20]}"
            + (f" ... (+{len(in_m_not_csv)-20} more)" if len(in_m_not_csv) > 20 else "")
        )

    print(
        f"parameters_PDAC.m: {len(m_params)} params   "
        f"addparameter in {len(active)} active modules: {len(mod_params)} params   "
        f"pdac_priors.csv: {len(csv_params)} params"
    )
    for w in warnings:
        print(f"  warn: {w}")
    for p in problems:
        print(f"  FAIL: {p}")

    if problems:
        print(
            "\nSync check FAILED. If a prior row refers to a renamed/deleted "
            "parameter, remove it from pdac_priors.csv; otherwise add the "
            "missing definition to parameters_PDAC.m or a module addparameter call."
        )
        return 1
    print("Sync OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
