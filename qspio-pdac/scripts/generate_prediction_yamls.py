#!/usr/bin/env python3
"""Generate ``calibration_targets/predictions/<scenario>/*.yaml`` from clinical_endpoints.py.

Each prediction YAML inlines the endpoint function + its private helpers into
a self-contained ``compute_observable`` code block (no sys.path gymnastics
required at simulation time). Re-run whenever ``clinical_endpoints.py`` changes
to keep the YAMLs in sync — they are the single source of truth that the
simulator executes.

Layout
------
``calibration_targets/predictions/<scenario>/<endpoint_id>.yaml``, one file per
(scenario, endpoint). The endpoint manifest is the cross-product of
:data:`SCENARIOS` (each with its long-horizon ``stop_time``) and
:data:`ENDPOINT_TEMPLATES` (each an endpoint function + kwargs + optional
``extract_key`` for dict-returning functions like RECIST).

Usage::

    python scripts/generate_prediction_yamls.py

Validates each emitted YAML against :class:`scripts.prediction_target.PredictionTarget`
before writing; any schema failure aborts the run with a clear error.
"""

from __future__ import annotations

import inspect
import sys
import textwrap
from pathlib import Path

import yaml
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
OUT_DIR = ROOT / "calibration_targets" / "predictions"

sys.path.insert(0, str(SCRIPTS))

import clinical_endpoints as ce  # noqa: E402
from prediction_target import PredictionTarget  # noqa: E402


# ---------------------------------------------------------------------------
# YAML: literal-block scalar for multi-line strings (clean diffs).
# ---------------------------------------------------------------------------


def _str_representer(dumper, data):
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


yaml.add_representer(str, _str_representer)


# ---------------------------------------------------------------------------
# Scenarios and helpers
# ---------------------------------------------------------------------------


SCENARIOS: list[dict] = [
    {
        "name": "clinical_progression_longterm",
        "stop_time": 1095.0,
        "context": "untreated natural-history cohort",
    },
    {
        # Long-horizon variant of gvax_neoadjuvant_zheng2022 — the short
        # scenario (stop_time=21) stays reserved for day-21 calibration
        # observables where existing qualitative-behavior tests live.
        "name": "gvax_neoadjuvant_longterm",
        "stop_time": 1095.0,
        "context": "neoadjuvant GVAX alone (no surgery/adjuvant modeled)",
    },
    {
        "name": "gvax_nivo_neoadjuvant_longterm",
        "stop_time": 1095.0,
        "context": "neoadjuvant GVAX + nivolumab (no surgery/adjuvant modeled)",
    },
]


_COMMON_HELPERS = (ce._get_magnitude,)
_DIAMETER_HELPERS = (
    ce._get_magnitude,
    ce._tumor_diameter_cm,
    ce._compartment_volume_uL,
    ce._diameter_from_species,
)
_INTERP_HELPERS = _DIAMETER_HELPERS + (ce._interp_at,)
_TTP_HELPERS = _DIAMETER_HELPERS + (ce._first_crossing,)


# ---------------------------------------------------------------------------
# Endpoint templates — one entry per (endpoint, landmark/category) tuple.
# ---------------------------------------------------------------------------
# Each template expands into one YAML per scenario. ``fn`` is the endpoint
# function in clinical_endpoints.py; ``helpers`` are the private helpers it
# uses (inlined into the code block). ``kwargs`` are substituted as literals
# into the function call. ``extract_key`` (optional) is for dict-returning
# endpoints (RECIST) — the generated wrapper returns the specified key only.


ENDPOINT_TEMPLATES: list[dict] = [
    # OS landmarks
    {
        "id": "os_at_12mo",
        "fn": ce.os_at_landmark,
        "helpers": _COMMON_HELPERS,
        "kwargs": {"landmark_days": 365.0, "n_c_lethal": ce.N_C_LETHAL_DEFAULT},
        "units": "dimensionless",
        "species": ["V_T.C1"],
        "index_values": [365.0],
        "description": (
            "Overall survival at 12 months (alive=1 / dead=0). Death defined as "
            f"V_T.C1 ≥ {ce.N_C_LETHAL_DEFAULT:.0e} cells at any t ≤ 365 days."
        ),
        "rationale": (
            "12-month OS is the standard landmark across adjuvant PDAC trials "
            "(Heumann 2023, Zheng 2021)."
        ),
    },
    {
        "id": "os_at_24mo",
        "fn": ce.os_at_landmark,
        "helpers": _COMMON_HELPERS,
        "kwargs": {"landmark_days": 730.0, "n_c_lethal": ce.N_C_LETHAL_DEFAULT},
        "units": "dimensionless",
        "species": ["V_T.C1"],
        "index_values": [730.0],
        "description": "Overall survival at 24 months. Same death threshold as os_at_12mo.",
        "rationale": "24-month OS captures late-event separation across PDAC treatment arms.",
    },
    {
        "id": "os_at_36mo",
        "fn": ce.os_at_landmark,
        "helpers": _COMMON_HELPERS,
        "kwargs": {"landmark_days": 1095.0, "n_c_lethal": ce.N_C_LETHAL_DEFAULT},
        "units": "dimensionless",
        "species": ["V_T.C1"],
        "index_values": [1095.0],
        "description": "Overall survival at 36 months (long-term outcome).",
        "rationale": (
            "3-year OS is the published durable-benefit landmark for PDAC "
            "immunotherapy (Zheng 2021 OS>2y subset)."
        ),
    },
    # DFS landmarks
    {
        "id": "dfs_at_12mo",
        "fn": ce.dfs_at_landmark,
        "helpers": _DIAMETER_HELPERS,
        "kwargs": {
            "landmark_days": 365.0,
            "n_c_lethal": ce.N_C_LETHAL_DEFAULT,
            "progression_increase": ce.PROGRESSION_INCREASE_DEFAULT,
            "min_absolute_increase_cm": 0.5,
        },
        "units": "dimensionless",
        "species": ["V_T.C1", "V_T"],
        "index_values": [365.0],
        "description": (
            "Disease-free survival at 12 months. DFS=1 when patient is alive AND "
            "has not progressed from nadir per RECIST 1.1 (≥20% diameter increase "
            "AND ≥5 mm absolute increase)."
        ),
        "rationale": "12-month DFS is a co-primary endpoint alongside OS in neoadjuvant PDAC trials.",
    },
    {
        "id": "dfs_at_24mo",
        "fn": ce.dfs_at_landmark,
        "helpers": _DIAMETER_HELPERS,
        "kwargs": {
            "landmark_days": 730.0,
            "n_c_lethal": ce.N_C_LETHAL_DEFAULT,
            "progression_increase": ce.PROGRESSION_INCREASE_DEFAULT,
            "min_absolute_increase_cm": 0.5,
        },
        "units": "dimensionless",
        "species": ["V_T.C1", "V_T"],
        "index_values": [730.0],
        "description": "Disease-free survival at 24 months. Same progression rules as dfs_at_12mo.",
        "rationale": "2-year DFS captures durable-response fraction and approximates 'cure' in resectable PDAC.",
    },
    # Tumor diameter landmarks — NaN-censored when the patient has already
    # crossed the lethal-burden threshold before the landmark.
    {
        "id": "tumor_diameter_day_365",
        "fn": ce.tumor_diameter_at_day,
        "helpers": _INTERP_HELPERS,
        "kwargs": {"landmark_days": 365.0, "n_c_lethal": ce.N_C_LETHAL_DEFAULT},
        "units": "centimeter",
        "species": ["V_T.C1", "V_T"],
        "index_values": [365.0],
        "description": (
            "Tumor diameter (cm) at day 365, linearly interpolated from the "
            "simulator trajectory. Sphere-equivalent from V_T volume when "
            "available, else cell-count-derived. NaN when V_T.C1 ≥ "
            f"{ce.N_C_LETHAL_DEFAULT:.0e} cells before day 365 (patient dead; "
            "diameter not measurable)."
        ),
        "rationale": "Continuous diameter landmark supports downstream response-binning and feeds tumor-burden OBED.",
    },
    {
        "id": "tumor_diameter_day_730",
        "fn": ce.tumor_diameter_at_day,
        "helpers": _INTERP_HELPERS,
        "kwargs": {"landmark_days": 730.0, "n_c_lethal": ce.N_C_LETHAL_DEFAULT},
        "units": "centimeter",
        "species": ["V_T.C1", "V_T"],
        "index_values": [730.0],
        "description": (
            "Tumor diameter (cm) at day 730, linearly interpolated. NaN when "
            f"V_T.C1 ≥ {ce.N_C_LETHAL_DEFAULT:.0e} cells before day 730."
        ),
        "rationale": "Late-horizon diameter captures long-term growth/regression dynamics.",
    },
    # Time to progression
    {
        "id": "time_to_progression",
        "fn": ce.time_to_progression,
        "helpers": _TTP_HELPERS,
        "kwargs": {
            "baseline_day": 0.0,
            "progression_increase": ce.PROGRESSION_INCREASE_DEFAULT,
            "min_absolute_increase_cm": 0.5,
            "censoring_day": 1095.0,
        },
        "units": "day",
        "species": ["V_T.C1", "V_T"],
        "index_values": [1095.0],
        "description": (
            "Time (days) from baseline to RECIST 1.1 PD; censored at 1095 days "
            "when no progression observed."
        ),
        "rationale": "Continuous PFS analog; essential for Kaplan-Meier-style OBED attribution.",
    },
    # RECIST at day 90 — four one-hot categories
    {
        "id": "recist_cr_at_day_90",
        "fn": ce.recist_category_at_day,
        "helpers": _INTERP_HELPERS,
        "kwargs": {
            "landmark_days": 90.0,
            "baseline_day": 0.0,
            "progression_increase": ce.PROGRESSION_INCREASE_DEFAULT,
            "response_decrease": ce.RESPONSE_DECREASE_DEFAULT,
            "cr_diameter_cm": ce.CR_DIAMETER_CM_DEFAULT,
            "min_absolute_increase_cm": 0.5,
        },
        "extract_key": "CR",
        "units": "dimensionless",
        "species": ["V_T.C1", "V_T"],
        "index_values": [90.0],
        "description": "RECIST 1.1 Complete Response indicator at day 90 (1.0 if CR, else 0.0).",
        "rationale": "CR fraction at day-90 landmark — most meaningful for treatment scenarios; near-zero for untreated natural history.",
    },
    {
        "id": "recist_pr_at_day_90",
        "fn": ce.recist_category_at_day,
        "helpers": _INTERP_HELPERS,
        "kwargs": {
            "landmark_days": 90.0,
            "baseline_day": 0.0,
            "progression_increase": ce.PROGRESSION_INCREASE_DEFAULT,
            "response_decrease": ce.RESPONSE_DECREASE_DEFAULT,
            "cr_diameter_cm": ce.CR_DIAMETER_CM_DEFAULT,
            "min_absolute_increase_cm": 0.5,
        },
        "extract_key": "PR",
        "units": "dimensionless",
        "species": ["V_T.C1", "V_T"],
        "index_values": [90.0],
        "description": "RECIST 1.1 Partial Response indicator at day 90 (≥30% diameter decrease from baseline).",
        "rationale": "PR fraction is the primary objective response rate metric; most informative for treatment arms.",
    },
    {
        "id": "recist_sd_at_day_90",
        "fn": ce.recist_category_at_day,
        "helpers": _INTERP_HELPERS,
        "kwargs": {
            "landmark_days": 90.0,
            "baseline_day": 0.0,
            "progression_increase": ce.PROGRESSION_INCREASE_DEFAULT,
            "response_decrease": ce.RESPONSE_DECREASE_DEFAULT,
            "cr_diameter_cm": ce.CR_DIAMETER_CM_DEFAULT,
            "min_absolute_increase_cm": 0.5,
        },
        "extract_key": "SD",
        "units": "dimensionless",
        "species": ["V_T.C1", "V_T"],
        "index_values": [90.0],
        "description": "RECIST 1.1 Stable Disease indicator at day 90.",
        "rationale": "SD fraction surfaces the low-signal middle ground; included for completeness of the RECIST one-hot.",
    },
    {
        "id": "recist_pd_at_day_90",
        "fn": ce.recist_category_at_day,
        "helpers": _INTERP_HELPERS,
        "kwargs": {
            "landmark_days": 90.0,
            "baseline_day": 0.0,
            "progression_increase": ce.PROGRESSION_INCREASE_DEFAULT,
            "response_decrease": ce.RESPONSE_DECREASE_DEFAULT,
            "cr_diameter_cm": ce.CR_DIAMETER_CM_DEFAULT,
            "min_absolute_increase_cm": 0.5,
        },
        "extract_key": "PD",
        "units": "dimensionless",
        "species": ["V_T.C1", "V_T"],
        "index_values": [90.0],
        "description": "RECIST 1.1 Progressive Disease indicator at day 90 (≥20% increase from nadir AND ≥5 mm absolute).",
        "rationale": "PD fraction at day 90 — dominant outcome for natural-history cohort; informative for treatment failure.",
    },
]


# ---------------------------------------------------------------------------
# Codegen
# ---------------------------------------------------------------------------


def _python_literal(v) -> str:
    if isinstance(v, (float, int, str)):
        return repr(v)
    if v is None:
        return "None"
    raise TypeError(f"Unsupported kwarg type: {type(v)}")


def _render_code(tpl: dict) -> str:
    """Build the ``compute_observable`` source string for a single endpoint template.

    Layout::

        def compute_observable(time, species_dict, constants, ureg):
            import math, numpy as np, pint
            from typing import Mapping
            # ... DEFAULT_* constants inlined ...
            # ... helper functions inlined ...
            # ... endpoint function inlined ...
            _result = <fn>(time, species_dict, ureg, **literal_kwargs)
            return _result[<key>]   # only when extract_key is set
            return _result          # otherwise
    """
    lines: list[str] = [
        "def compute_observable(time, species_dict, constants):",
        "    import math",
        "    from typing import Mapping",
        "",
        "    import numpy as np",
        "",
        f"    N_C_LETHAL_DEFAULT = {ce.N_C_LETHAL_DEFAULT!r}",
        f"    PROGRESSION_INCREASE_DEFAULT = {ce.PROGRESSION_INCREASE_DEFAULT!r}",
        f"    RESPONSE_DECREASE_DEFAULT = {ce.RESPONSE_DECREASE_DEFAULT!r}",
        f"    CR_DIAMETER_CM_DEFAULT = {ce.CR_DIAMETER_CM_DEFAULT!r}",
        "",
    ]

    seen: set[str] = set()
    for helper in tpl["helpers"]:
        if helper.__name__ in seen:
            continue
        seen.add(helper.__name__)
        src = textwrap.dedent(inspect.getsource(helper)).rstrip()
        lines.append(textwrap.indent(src, "    "))
        lines.append("")

    fn = tpl["fn"]
    fn_src = textwrap.dedent(inspect.getsource(fn)).rstrip()
    lines.append(textwrap.indent(fn_src, "    "))
    lines.append("")

    kwargs_literal = ", ".join(f"{k}={_python_literal(v)}" for k, v in tpl["kwargs"].items())
    extract_key = tpl.get("extract_key")
    if extract_key is not None:
        lines.append(f"    _result = {fn.__name__}(time, species_dict, {kwargs_literal})")
        lines.append(f"    return _result[{extract_key!r}]")
    else:
        lines.append(f"    return {fn.__name__}(time, species_dict, {kwargs_literal})")

    return "\n".join(lines) + "\n"


def _yaml_for(tpl: dict, scenario: dict) -> dict:
    """Construct the YAML-serializable dict for a single (scenario, endpoint)."""
    code = _render_code(tpl)
    scenario_context = scenario.get("context")
    rationale = tpl["rationale"]
    if scenario_context:
        rationale = f"{rationale} Scenario: {scenario_context}."
    return {
        "prediction_target_id": f"{tpl['id']}",
        "description": tpl["description"],
        "observable": {
            "code": code,
            "units": tpl["units"],
            "species": list(tpl["species"]),
            "constants": [],
        },
        "scenario": scenario["name"],
        "index_values": list(tpl["index_values"]),
        "index_unit": "day",
        "index_type": "time",
        "rationale": rationale,
        "tags": ["ai-generated", "codegen"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Track existing files so we can prune stale YAMLs from prior runs.
    existing = {p for p in OUT_DIR.rglob("*.yaml")}
    written: set[Path] = set()

    for scenario in SCENARIOS:
        scen_dir = OUT_DIR / scenario["name"]
        scen_dir.mkdir(parents=True, exist_ok=True)

        for tpl in ENDPOINT_TEMPLATES:
            doc = _yaml_for(tpl, scenario)
            try:
                PredictionTarget.model_validate(doc)
            except ValidationError as exc:
                print(
                    f"[fail] {scenario['name']}/{tpl['id']} failed validation:\n{exc}",
                    file=sys.stderr,
                )
                return 2

            path = scen_dir / f"{tpl['id']}.yaml"
            with path.open("w") as f:
                yaml.dump(
                    doc,
                    f,
                    sort_keys=False,
                    default_flow_style=False,
                    allow_unicode=True,
                    width=100,
                )
            written.add(path)
            try:
                shown = path.relative_to(ROOT)
            except ValueError:
                shown = path
            print(f"[write] {shown}")

    stale = existing - written
    for path in sorted(stale):
        path.unlink()
        try:
            shown = path.relative_to(ROOT)
        except ValueError:
            shown = path
        print(f"[prune] {shown} (no longer in manifest)")

    # Clean up any empty directories left over from pruned scenarios.
    for sub in sorted(OUT_DIR.rglob("*"), reverse=True):
        if sub.is_dir() and not any(sub.iterdir()):
            sub.rmdir()

    try:
        out_shown = OUT_DIR.relative_to(ROOT)
    except ValueError:
        out_shown = OUT_DIR
    print(f"[done]  {len(written)} prediction YAMLs across {len(SCENARIOS)} scenarios in {out_shown}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
