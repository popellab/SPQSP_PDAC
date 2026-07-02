"""Migrate calibration_targets/*.yaml `observable.code` from the legacy
4-arg `compute_observable(time, species_dict, constants, ureg)` Pint API
to the 3-arg `compute_observable(time, species_dict, constants)` raw-float
API introduced in qsp-hpc-tools PR `feature/pintless-derive`.

species_dict / constants now carry raw floats in their declared units:
  - cells:        `cell`
  - V_T:          `milliliter`
  - cytokines:    `nanomolarity` (≡ nM) — except V_T.VEGF which is `picogram/milliliter`
  - V_T.collagen: `milligram`
  - V_T.ECM:      `milligram`
  - time:         `day`
  - constants:    raw float in YAML-declared `units`

Strategy (for non-prediction targets):
  1. Mechanical AST/regex strip of all `.to(...)`, `.magnitude`, `* ureg.X`,
     `* ureg('X')` constructs.
  2. Drop ureg from signature.
  3. Apply per-file scale factor on the final return statement when the
     return-side `.to(<target>)` was a non-trivial unit conversion.

Prediction targets (calibration_targets/predictions/) are NOT touched here —
the generator (`scripts/generate_prediction_yamls.py` on top of
`scripts/clinical_endpoints.py`) is the source of truth and emits pintless
code directly. Re-run that generator instead.

This is a one-shot migration script kept for reproducibility.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Iterable

import yaml

REPO = Path(__file__).resolve().parent.parent
TARGETS_ROOT = REPO / "calibration_targets"

# ----------------------------------------------------------------------
# Per-file return-scale overrides (applied AFTER mechanical strip).
# Key: relative path under calibration_targets/. Value: scale-factor to
# multiply the final returned expression by, OR a callable taking the
# stripped code and returning the migrated code.
# ----------------------------------------------------------------------
RETURN_SCALE: dict[str, float] = {
    # Density: (cells/mL) * µm  →  cell/mm² requires ×1e-6
    "baseline_no_treatment/cd8_density_baseline_PDAC_deriv001.yaml": 1e-6,
    "baseline_no_treatment/cd8_density_baseline_golesworthy2022.yaml": 1e-6,
    "baseline_no_treatment/cd8_density_baseline_liu2015.yaml": 1e-6,
    "baseline_no_treatment/cd4_helper_density_PDAC_deriv001.yaml": 1e-6,
    "baseline_no_treatment/cdc1_density_PDAC_deriv001.yaml": 1e-6,
    "baseline_no_treatment/cdc2_density_PDAC_deriv001.yaml": 1e-6,
    "baseline_no_treatment/mdsc_density_PDAC_deriv001.yaml": 1e-6,
    "baseline_no_treatment/tam_density_PDAC_deriv001.yaml": 1e-6,
    "baseline_no_treatment/total_til_density_PDAC_deriv001.yaml": 1e-6,
    "baseline_no_treatment/treg_density_PDAC_deriv001.yaml": 1e-6,
    # Note: cd8 + cd4 + treg + tam + cdc1 + cdc2 + mdsc + total_til = 8
    # Plus cd8_density 3 variants = 11 density files (matches inventory).
    # Cytokine nM × MW(g/mol) → ng/mL: ×1e-3
    "baseline_no_treatment/arginase_concentration_tumor_PDAC_deriv001.yaml": 1e-3,
    "baseline_no_treatment/tgfb_concentration_tumor_PDAC_deriv001.yaml": 1e-3,
    # Cytokine nM × MW(g/mol) → pg/mL: ×1.0 (no scaling)
    # IL12 already pg/mL natively after raw multiplication
    # nM × MW = pg/mL, no factor
    # CCL2: nM × MW(g/mol) → pg/mL, then /(mg/mL) → pg/mg, no factor
    # VEGF: pg/mL × (mL/mg) → pg/mg, no factor
}

# ----------------------------------------------------------------------
# Hand-migrated prediction templates (keyed by md5(code-without-last-line))
# ----------------------------------------------------------------------
PRED_TEMPLATES: dict[str, str] = {}


def _shared_pred_helpers() -> str:
    """Common helper functions used across prediction templates.

    Differences from the original Pint version:
      - signature: drops `ureg`
      - `_get_magnitude(qty, target_unit)` becomes a no-op asarray cast,
        with explicit per-call scaling embedded in callers
      - `_compartment_volume_uL` does mL→µL (×1000) explicitly
      - removed `pint.Quantity` / `pint.UnitRegistry` annotations
    """
    return """    import math
    from typing import Mapping

    import numpy as np

    N_C_LETHAL_DEFAULT = 100000000000.0
    PROGRESSION_INCREASE_DEFAULT = 0.2
    RESPONSE_DECREASE_DEFAULT = 0.3
    CR_DIAMETER_CM_DEFAULT = 0.1

    def _as_float_array(value) -> np.ndarray:
        \"\"\"Coerce a raw-float scalar / array into a 1-D numpy array.\"\"\"
        return np.asarray(value, dtype=float)

    def _tumor_diameter_cm(tumor_volume_uL: np.ndarray) -> np.ndarray:
        v_cm3 = np.clip(tumor_volume_uL, 0.0, None) * 1e-3
        return 2.0 * np.cbrt(3.0 * v_cm3 / (4.0 * math.pi))

    def _compartment_volume_uL(species_dict: Mapping[str, np.ndarray]) -> np.ndarray | None:
        # V_T canonical unit is milliliter; convert to microliter (×1000).
        vol = species_dict.get("V_T")
        if vol is None:
            return None
        return _as_float_array(vol) * 1000.0

    def _diameter_from_species(
        species_dict: Mapping[str, np.ndarray],
        cells_per_cm3: float = 1.0e9,
    ) -> np.ndarray:
        v_uL = _compartment_volume_uL(species_dict)
        if v_uL is not None:
            return _tumor_diameter_cm(v_uL)
        c1 = _as_float_array(species_dict["V_T.C1"])
        v_cm3 = np.clip(c1, 0.0, None) / cells_per_cm3
        return 2.0 * np.cbrt(3.0 * v_cm3 / (4.0 * math.pi))

    def _interp_at(time_days: np.ndarray, values: np.ndarray, landmark_days: float) -> float:
        if time_days.size == 0:
            return float("nan")
        if landmark_days < time_days[0] or landmark_days > time_days[-1]:
            return float("nan")
        return float(np.interp(landmark_days, time_days, values))

"""


# Template body for tumor_diameter_at_day variants (6 files)
PRED_TEMPLATES["31f9b40a"] = (
    "def compute_observable(time, species_dict, constants):\n"
    + _shared_pred_helpers()
    + """    def tumor_diameter_at_day(
        time,
        species_dict,
        *,
        landmark_days: float,
        n_c_lethal: float = N_C_LETHAL_DEFAULT,
    ) -> np.ndarray:
        t = _as_float_array(time)
        diameter = _diameter_from_species(species_dict)
        if t.size == 0 or landmark_days > float(t[-1]):
            value = float("nan")
        else:
            c1 = _as_float_array(species_dict["V_T.C1"])
            mask = t <= landmark_days
            dead_before = bool(mask.any() and c1[mask].max() >= n_c_lethal)
            value = float("nan") if dead_before else _interp_at(t, diameter, landmark_days)
        return np.full_like(t, value)

"""
)

# Template body for RECIST classification (12 files)
PRED_TEMPLATES["794e22cc"] = (
    "def compute_observable(time, species_dict, constants):\n"
    + _shared_pred_helpers()
    + """    def recist_at_day(
        time,
        species_dict,
        *,
        baseline_day: float,
        landmark_day: float,
        progression_increase: float = PROGRESSION_INCREASE_DEFAULT,
        response_decrease: float = RESPONSE_DECREASE_DEFAULT,
        cr_diameter_cm: float = CR_DIAMETER_CM_DEFAULT,
        n_c_lethal: float = N_C_LETHAL_DEFAULT,
    ) -> dict:
        t = _as_float_array(time)
        diameter = _diameter_from_species(species_dict)
        out = {"PD": np.full_like(t, 0.0), "SD": np.full_like(t, 0.0),
               "PR": np.full_like(t, 0.0), "CR": np.full_like(t, 0.0)}
        if t.size == 0 or landmark_day > float(t[-1]):
            return out
        c1 = _as_float_array(species_dict["V_T.C1"])
        mask = t <= landmark_day
        dead_before = bool(mask.any() and c1[mask].max() >= n_c_lethal)
        if dead_before:
            out["PD"] = np.full_like(t, 1.0)
            return out
        d_base = _interp_at(t, diameter, baseline_day)
        d_land = _interp_at(t, diameter, landmark_day)
        if not np.isfinite(d_base) or not np.isfinite(d_land) or d_base <= 0:
            return out
        rel = (d_land - d_base) / d_base
        if d_land <= cr_diameter_cm:
            out["CR"] = np.full_like(t, 1.0)
        elif rel <= -response_decrease:
            out["PR"] = np.full_like(t, 1.0)
        elif rel >= progression_increase:
            out["PD"] = np.full_like(t, 1.0)
        else:
            out["SD"] = np.full_like(t, 1.0)
        return out

    _result = recist_at_day(
        time, species_dict, baseline_day=0.0, landmark_day=90.0,
    )
"""
)

# Template body for time_to_progression (3 files)
PRED_TEMPLATES["776e3534"] = (
    "def compute_observable(time, species_dict, constants):\n"
    + _shared_pred_helpers()
    + """    def time_to_progression(
        time,
        species_dict,
        *,
        baseline_day: float = 0.0,
        progression_increase: float = PROGRESSION_INCREASE_DEFAULT,
        min_absolute_increase_cm: float = 0.5,
        censoring_day: float = 1095.0,
        n_c_lethal: float = N_C_LETHAL_DEFAULT,
    ) -> np.ndarray:
        t = _as_float_array(time)
        if t.size == 0:
            return np.full_like(t, float("nan"))
        diameter = _diameter_from_species(species_dict)
        d_base = _interp_at(t, diameter, baseline_day)
        if not np.isfinite(d_base) or d_base <= 0:
            return np.full_like(t, float("nan"))
        threshold = max(d_base * (1.0 + progression_increase),
                        d_base + min_absolute_increase_cm)
        c1 = _as_float_array(species_dict["V_T.C1"])
        # progression: diameter exceeds threshold
        prog_mask = (t <= censoring_day) & (diameter >= threshold)
        # death also counts as event
        death_mask = (t <= censoring_day) & (c1 >= n_c_lethal)
        event_mask = prog_mask | death_mask
        if event_mask.any():
            ttp = float(t[event_mask][0])
        else:
            ttp = float(min(censoring_day, t[-1]))
        return np.full_like(t, ttp)

"""
)

# Template body for OS landmark (9 files)
PRED_TEMPLATES["e14fb46a"] = (
    "def compute_observable(time, species_dict, constants):\n"
    + _shared_pred_helpers()
    + """    def os_at_landmark(
        time,
        species_dict,
        *,
        landmark_days: float,
        n_c_lethal: float = N_C_LETHAL_DEFAULT,
    ) -> np.ndarray:
        t = _as_float_array(time)
        if t.size == 0:
            return np.full_like(t, float("nan"))
        c1 = _as_float_array(species_dict["V_T.C1"])
        if landmark_days > float(t[-1]):
            return np.full_like(t, 0.0)
        mask = t <= landmark_days
        alive = not bool(mask.any() and c1[mask].max() >= n_c_lethal)
        return np.full_like(t, 1.0 if alive else 0.0)

"""
)

# Template body for DFS landmark (6 files)
PRED_TEMPLATES["e66836d0"] = (
    "def compute_observable(time, species_dict, constants):\n"
    + _shared_pred_helpers()
    + """    def dfs_at_landmark(
        time,
        species_dict,
        *,
        landmark_days: float,
        n_c_lethal: float = N_C_LETHAL_DEFAULT,
        progression_increase: float = PROGRESSION_INCREASE_DEFAULT,
        min_absolute_increase_cm: float = 0.5,
    ) -> np.ndarray:
        t = _as_float_array(time)
        if t.size == 0 or landmark_days > float(t[-1]):
            return np.full_like(t, 0.0)
        diameter = _diameter_from_species(species_dict)
        d_base = _interp_at(t, diameter, 0.0)
        if not np.isfinite(d_base) or d_base <= 0:
            return np.full_like(t, 0.0)
        threshold = max(d_base * (1.0 + progression_increase),
                        d_base + min_absolute_increase_cm)
        c1 = _as_float_array(species_dict["V_T.C1"])
        mask = t <= landmark_days
        prog_before = bool(mask.any() and (diameter[mask] >= threshold).any())
        dead_before = bool(mask.any() and c1[mask].max() >= n_c_lethal)
        return np.full_like(t, 0.0 if (prog_before or dead_before) else 1.0)

"""
)


# ----------------------------------------------------------------------
# Mechanical migration
# ----------------------------------------------------------------------
def mechanical_strip(code: str) -> str:
    """Apply uniform regex strips that are safe across all observables."""
    # 1. signature
    code = re.sub(
        r"def compute_observable\(time, species_dict, constants, ureg\)",
        "def compute_observable(time, species_dict, constants)",
        code,
    )

    # 2. `.to('<unit>').magnitude` and `.to("<unit>").magnitude`
    code = re.sub(r"""\.to\(\s*['"][^'"]+['"]\s*\)\.magnitude""", "", code)

    # 3. plain `.to('<unit>')`
    code = re.sub(r"""\.to\(\s*['"][^'"]+['"]\s*\)""", "", code)

    # 4. plain `.magnitude`
    code = re.sub(r"\.magnitude\b", "", code)

    # 5. `* ureg.<attr>`
    code = re.sub(r"\s*\*\s*ureg\.\w+\b", "", code)

    # 6. `* ureg('<unit>')` / `* ureg("<unit>")`
    code = re.sub(r"""\s*\*\s*ureg\(\s*['"][^'"]+['"]\s*\)""", "", code)

    return code


def apply_return_scale(code: str, scale: float) -> str:
    """Wrap the return expression in `(... ) * scale` on the last `return` stmt
    that is at function-body indent (i.e., 4 leading spaces).

    We deliberately match only `^    return ` (top-level inside the function)
    to avoid touching nested helper returns.
    """
    if scale == 1.0:
        return code
    # Find all top-level returns; replace the LAST one.
    pattern = re.compile(r"^(    return )(.+)$", flags=re.MULTILINE)
    matches = list(pattern.finditer(code))
    if not matches:
        raise RuntimeError("apply_return_scale: no top-level return found")
    last = matches[-1]
    expr = last.group(2).rstrip()
    new = f"{last.group(1)}({expr}) * {scale!r}"
    return code[: last.start()] + new + code[last.end() :]


def migrate_one(yaml_path: Path) -> tuple[str, str]:
    """Return (new_code, action) where action ∈ {'mechanical', 'template:<hash>',
    'noop'}."""
    import hashlib

    rel = yaml_path.resolve().relative_to(TARGETS_ROOT).as_posix()
    data = yaml.safe_load(yaml_path.read_text())
    code = data["observable"]["code"]
    if "compute_observable(time, species_dict, constants, ureg)" not in code:
        return code, "noop"

    # Prediction templates: hash the body except the last return
    lines = code.rstrip().split("\n")
    body_hash = hashlib.md5("\n".join(lines[:-1]).encode()).hexdigest()[:8]
    if body_hash in PRED_TEMPLATES:
        # Replace whole code with template body + last return migrated
        last_return = lines[-1].strip()
        # mechanical strip on the last return only (drop ureg arg, .to, etc)
        # The last_return looks like:
        #   return tumor_diameter_at_day(time, species_dict, ureg, ...)
        last_return = re.sub(r",\s*ureg\b", "", last_return)
        # 794e22cc has no last-return per call (it inlines _result['PD'] etc)
        new_code = PRED_TEMPLATES[body_hash]
        if body_hash == "794e22cc":
            # last_return was e.g. "return _result['PD']"; append it
            new_code = new_code + f"    {last_return}\n"
        elif body_hash == "776e3534":
            # last_return was e.g. "return time_to_progression(time, species_dict, ureg, ...)"
            new_code = new_code + f"    {last_return}\n"
        else:
            new_code = new_code + f"    {last_return}\n"
        return new_code, f"template:{body_hash}"

    # Generic: mechanical strip
    new_code = mechanical_strip(code)
    scale = RETURN_SCALE.get(rel, 1.0)
    if scale != 1.0:
        new_code = apply_return_scale(new_code, scale)
    return new_code, "mechanical"


def write_yaml_with_code(yaml_path: Path, new_code: str) -> None:
    """Replace observable.code in YAML preserving everything else.

    yaml.dump round-trips destroy the literal block style and reorder keys,
    so do a byte-level replacement of the code block instead.
    """
    text = yaml_path.read_text()
    # observable.code can be `code: |` or `code: |-` or `code: "..."` (quoted)
    # We locate the existing code region and replace.
    data = yaml.safe_load(text)
    old_code = data["observable"]["code"]
    if old_code not in text:
        # Quoted-string form (escaped newlines). Re-dump observable block.
        # Round-trip via yaml dumping just observable.code as block scalar.
        # Find: pattern `code: ".*?"\n` (single-line) or `code: |...` block.
        # Fallback: regex match
        # For our 114 files, only a handful use single-line escaped form.
        # Rewrite by dumping a fresh block scalar.
        marker_re = re.compile(
            r"^(\s*)code:\s*(?P<form>[\"'|>])(?P<rest>(?:.|\n)*?)(?=\n\s*units:\s)",
            flags=re.MULTILINE,
        )
        m = marker_re.search(text)
        if not m:
            raise RuntimeError(
                f"{yaml_path}: could not locate observable.code block"
            )
        indent = m.group(1)
        body_indent = indent + "  "
        block = f"{indent}code: |\n"
        for line in new_code.rstrip("\n").split("\n"):
            block += f"{body_indent}{line}\n" if line.strip() else "\n"
        text = text[: m.start()] + block + text[m.end() :]
    else:
        # Block scalar form: replace in-place. Need to re-indent.
        # Find original indent of the code block.
        # Pattern: search for `code: |` line, then compute indent of next line.
        idx = text.index(old_code)
        # walk back to start of line containing 'code:' before `idx`
        line_start = text.rfind("\n", 0, idx) + 1
        # The block scalar lines are the `old_code` rendered with leading spaces
        # at body_indent. We can detect body_indent from the first old_code line
        # in the file: the block scalar form means the YAML loader stripped the
        # body_indent. So we must re-indent new_code by the same amount.
        # Find the literal indent of the first content line.
        # Simpler: locate `code: |\n` then read indent of following non-blank line.
        code_line_re = re.compile(r"^(\s*)code:\s*\|[+-]?\s*\n", flags=re.MULTILINE)
        m2 = code_line_re.search(text)
        if not m2:
            raise RuntimeError(f"{yaml_path}: no `code: |` block found")
        prefix_indent = m2.group(1)
        body_indent = prefix_indent + "  "
        # Construct the new block-scalar body
        new_block_lines = []
        for line in new_code.rstrip("\n").split("\n"):
            if line.strip():
                new_block_lines.append(body_indent + line)
            else:
                new_block_lines.append("")
        new_block = "\n".join(new_block_lines)
        # Replace old_code with new_block in text. old_code was extracted by
        # the YAML loader with body_indent stripped, so it doesn't appear
        # verbatim in text. Use regex from end of `code: |\n` line up to the
        # next sibling key at prefix_indent depth.
        block_start = m2.end()
        # find first subsequent line that does NOT start with body_indent
        # and is not blank → that's the boundary.
        rest = text[block_start:]
        rest_lines = rest.split("\n")
        end_offset = 0
        consumed = 0
        for line in rest_lines:
            if line.strip() == "":
                consumed += len(line) + 1
                continue
            if line.startswith(body_indent):
                consumed += len(line) + 1
                continue
            break
        end_offset = block_start + consumed
        text = text[:block_start] + new_block + "\n" + text[end_offset:]
    yaml_path.write_text(text)


def main(argv: list[str]) -> int:
    if len(argv) > 1 and argv[1] == "--dry-run":
        dry_run = True
        files_arg = argv[2:]
    else:
        dry_run = False
        files_arg = argv[1:]

    if files_arg:
        files = [Path(p) for p in files_arg]
    else:
        # Default: all 114 files
        list_file = Path("/tmp/pintless_yamls.txt")
        if list_file.exists():
            files = [Path(p) for p in list_file.read_text().strip().split("\n")]
        else:
            files = sorted(TARGETS_ROOT.rglob("*.yaml"))

    actions: dict[str, int] = {}
    failures: list[tuple[Path, str]] = []
    for f in files:
        try:
            new_code, action = migrate_one(f)
            actions[action] = actions.get(action, 0) + 1
            if action == "noop":
                continue
            if not dry_run:
                write_yaml_with_code(f, new_code)
        except Exception as e:
            failures.append((f, str(e)))

    print(f"Action summary: {actions}")
    if failures:
        print(f"\nFAILURES ({len(failures)}):")
        for f, err in failures:
            print(f"  {f}: {err}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
