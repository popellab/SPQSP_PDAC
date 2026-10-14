"""Verify every submodel target YAML references parameters that exist in the model.

Parses ``calibration_targets/submodel_targets/*.yaml``, reads each target's
``parameters:`` field, and fails when any named parameter is absent from
``parameters_PDAC.m``. YAMLs flagged ``deprecated: true`` are skipped (they
are retained for provenance but not used for calibration).

Also checks ``submodel_config.yaml`` groups for stale member references.

Exit: 0 on clean, 1 on any stale reference.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PARAMS_M = REPO / "parameters" / "parameters_PDAC.m"
MODULES_DIR = REPO / "core" / "modules"
MAIN_MODEL = REPO / "scripts" / "immune_oncology_model_PDAC.m"
TARGETS_DIR = REPO / "calibration_targets" / "submodel_targets"
CONFIG_YAML = TARGETS_DIR / "submodel_config.yaml"

PARAM_RE = re.compile(r"^\s*params\.([A-Za-z_][A-Za-z0-9_]*)\.Value\s*=", re.M)
ADDPARAM_RE = re.compile(
    r"addparameter\s*\(\s*model\s*,\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]"
)
MODULE_CALL_RE = re.compile(r"^\s*model\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.M)

SKIP_FILES = {"submodel_priors.yaml", "submodel_config.yaml", "reference_values.yaml"}
SKIP_DIRS = {"excluded", "retired", "to-review"}


def model_params() -> set[str]:
    """Live model params = declared in parameters_PDAC.m OR addparameter'd
    by an actively-called module. Same rule as check_priors_sync.py."""
    static = set(PARAM_RE.findall(PARAMS_M.read_text()))
    active = set(MODULE_CALL_RE.findall(MAIN_MODEL.read_text()))
    dynamic: set[str] = set()
    for m in MODULES_DIR.glob("*.m"):
        if m.stem in active:
            dynamic.update(ADDPARAM_RE.findall(m.read_text()))
    return static | dynamic


def scan_targets(valid: set[str]) -> tuple[list[str], list[str]]:
    """Return (failures, skipped_files)."""
    failures: list[str] = []
    skipped: list[str] = []
    # rglob so we see the excluded/ subdirectory, then filter it out here —
    # this way we still report on it in the skipped count.
    for yml in sorted(TARGETS_DIR.rglob("*.yaml")):
        if yml.name in SKIP_FILES:
            continue
        if any(d in yml.parts for d in SKIP_DIRS):
            skipped.append(str(yml.relative_to(TARGETS_DIR)))
            continue
        try:
            data = yaml.safe_load(yml.read_text()) or {}
        except yaml.YAMLError as exc:
            failures.append(f"{yml.name}: YAML parse error: {exc}")
            continue
        # Schema: `parameters:` can be a string or list of strings.
        params = data.get("parameters", [])
        if isinstance(params, str):
            params = [params]
        missing = [p for p in params if p not in valid]
        if missing:
            failures.append(
                f"{yml.name}: references missing param(s) {missing} "
                f"(add to parameters_PDAC.m or mark YAML deprecated: true)"
            )
    return failures, skipped


def scan_config(valid: set[str]) -> list[str]:
    if not CONFIG_YAML.exists():
        return []
    data = yaml.safe_load(CONFIG_YAML.read_text()) or {}
    failures: list[str] = []
    for group in data.get("groups", []):
        gid = group.get("group_id", "<unnamed>")
        for member in group.get("members", []):
            name = member.get("name")
            if name and name not in valid:
                failures.append(
                    f"submodel_config.yaml group '{gid}' references missing "
                    f"param '{name}' (remove from members list)"
                )
    return failures


def main() -> int:
    valid = model_params()
    tgt_fails, dep_skipped = scan_targets(valid)
    cfg_fails = scan_config(valid)

    total = sum(1 for p in TARGETS_DIR.rglob("*.yaml") if p.name not in SKIP_FILES)
    print(f"Scanned {total} submodel target YAMLs + submodel_config.yaml.")
    print(f"  {len(dep_skipped)} in SKIP_DIRS (excluded/retired/to-review) — not validated.")
    for f in tgt_fails + cfg_fails:
        print(f"  FAIL: {f}")

    if tgt_fails or cfg_fails:
        return 1
    print("Submodel sync OK.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
