# ruff: noqa
# pyrefly: ignore-all-errors
# pyright: reportGeneralIssues=false
#!/usr/bin/env python3
"""
Staged calibration / submodel target extraction pipeline.

Pipeline mode is selected by the top-level TARGET_KIND constant ("submodel"
or "cal"). The stages run identically for both kinds; only the schema /
prompt / promotion destination differ at stage 3 onward.

Workflow overview (each stage caches results — re-running skips completed work):

  Stage 1  — Lit search:    Web search for papers matching each target
  Stage 1b — DOIs:          Combine DOIs into a single file for Zotero import
  Stage 1c — PDF fetch:     Interactive: fetch from Zotero, open in browser if missing
  Stage 2  — Assess:        Read papers (text-extracted, not binary PDFs) and decide
                            which have usable data, what needs digitization
  Stage 2b — Plan review:   Single LLM call reviews ALL plans together for consistency,
                            can swap plans or request lit search reruns
  ── Human touchpoint ──    Digitize figures flagged in digitization_summary.md
                            (place CSVs in work/<target>/digitized/<source_tag>/)
  Stage 3  — Extract:       Assemble target YAMLs (SubmodelTarget for submodel kind,
                            CalibrationTarget for cal kind) from papers + digitized data.
                            Snippet-in-PDF validation runs inline (retries on failure).
                            Targets missing digitizations proceed anyway (warning only).
  Stage 3b — Deriv review:  Single LLM call reviews ALL derivations for cross-target
                            consistency, flags issues
  Stage 3c — Validate:      Unit validation, code execution (MCMC prior derivation
                            for submodel kind, distribution_code + observable.code for
                            cal kind), snippet validation.
                            Promotion targets:
                              submodel kind -> calibration_targets/submodel_targets/
                              cal kind      -> calibration_targets/<scenario>/

Side effects to be aware of:
  - apply_plan_swaps():         Modifies assessment.json (replaces primary with alt plan)
  - apply_lit_search_reruns():  Deletes cached lit_search + assessment, appends RERUN
                                notes to targets.csv
  - summarize_digitizations():  Writes digitization_summary.md, optionally opens PDFs
  - collect_missing_pdfs():     Interactive — copies DOIs to clipboard, opens browser

Usage:
    python scripts/staged_extraction.py
"""

import csv
import json
import sys
from dotenv import load_dotenv                                                                                                                                        
load_dotenv(override=True) 
from pathlib import Path
from functools import partial

import logfire

logfire.configure()
logfire.instrument_pydantic_ai()

from maple.extraction import (
    apply_lit_search_reruns,
    apply_plan_swaps,
    collect_missing_pdfs,
    make_agents,
    report_digitization_preflight,
    run_assess,
    run_complete,
    run_derivation_review,
    run_lit_search,
    run_plan_review,
    run_stage,
    run_validate_all,
    summarize_digitizations,
    write_assessment_report,
    write_dois_md,
)


# ===========================================================================
# Config — edit these before running
# ===========================================================================

# Pipeline mode:
#   "submodel" — assemble SubmodelTarget YAMLs (in vitro / isolated-system data,
#                ODE forward model + error model, output to
#                calibration_targets/submodel_targets/). Reads notes/targets.csv.
#   "cal"      — assemble CalibrationTarget YAMLs (in vivo / cohort observables
#                of the full QSP model, e.g. PDAC tumor TGFb concentration,
#                output to calibration_targets/<scenario>/). Reads
#                notes/cytokine_targets.csv (or any CSV with the cal-mode
#                columns: target_id, parameters, cancer_type, scenario,
#                observable_description, model_species, model_indication,
#                model_compartment, model_system, model_treatment_history,
#                model_stage_burden, relevant_compartments, notes).
#                Wired via target_kind='cal' through make_agents / run_complete /
#                run_validate_all (maple PRs #29 + #30 + #31).
TARGET_KIND = "cal"
TARGETS_CSV = (
    Path("notes/targets.csv") if TARGET_KIND == "submodel"
    else Path("notes/cytokine_targets.csv")
)
WORK_DIR = (
    Path("work/staged_extraction") if TARGET_KIND == "submodel"
    else Path("work/staged_extraction_cal")
)
MODEL_STRUCTURE = Path("model_structure.json") # QSP model structure (for unit validation)
MODEL_CONTEXT = Path("model_context.txt")      # Model description for LLM prompts
ZOTERO_STORAGE = Path("~/Zotero/storage").expanduser()
PRIORS_CSV = Path("parameters/pdac_priors.csv")
MODEL = "gpt-5.5"
MAX_RETRIES = 7
TARGET_RANGE = (0, None)  # (start, end) indices from targets CSV — set end=None for all

if TARGET_KIND not in ("submodel", "cal"):
    raise ValueError(f"TARGET_KIND must be 'submodel' or 'cal', got {TARGET_KIND!r}")


# ===========================================================================
# 0. Setup — load targets, create agents
# ===========================================================================

if not Path.exists(TARGETS_CSV):
    print(f"Error: {TARGETS_CSV} not found", file=sys.stderr)
    sys.exit(1)

WORK_DIR.mkdir(parents=True, exist_ok=True)

model_context = MODEL_CONTEXT.read_text().strip()

all_targets = []
with open(TARGETS_CSV, encoding="utf-8") as f:
    for row in csv.DictReader(f):
        all_targets.append(row)

start, end = TARGET_RANGE
batch = all_targets[start:end]

SCENARIOS_DIR = Path("scenarios")

targets = []
for i, row in enumerate(batch, start=start):
    # Pass all CSV columns through so cal-mode picks up model_species,
    # model_indication, model_compartment, model_system, model_treatment_history,
    # model_stage_burden, relevant_compartments, scenario, observable_description.
    # Submodel-mode ignores the extras.
    t = {**row, "idx": i}
    t["dir"] = WORK_DIR / t["target_id"]
    t["dir"].mkdir(parents=True, exist_ok=True)
    # Resolve scenario YAML for cal-mode targets (maple PR #36 reads this from
    # the target dict rather than assuming a directory layout). Submodel mode
    # ignores the field.
    if TARGET_KIND == "cal" and t.get("scenario"):
        scen_path = SCENARIOS_DIR / f"{t['scenario']}.yaml"
        if scen_path.exists():
            t["scenario_yaml"] = scen_path.read_text()
        else:
            print(f"  [{i}] WARNING: scenario '{t['scenario']}' has no YAML at {scen_path}")
    targets.append(t)
    print(f"  [{i}] {t['target_id']} ({t['parameters']})")
print()

lit_search_agent, assess_agent, plan_review_agent, complete_agent, derivation_review_agent = make_agents(MODEL, MAX_RETRIES, target_kind=TARGET_KIND)


# ===========================================================================
# Stage 1: Literature search
# Each target gets a web search for papers with quantitative data.
# Results cached in work/<target>/lit_search_results.json
# ===========================================================================

print("=" * 60)
print(f"Stage 1: Literature Search ({len(targets)} targets)")
print("=" * 60)

_AUXILIARY_CONFIG_PATH = Path("auxiliary_config.yaml")
_aux_kwargs = {"auxiliary_config_path": _AUXILIARY_CONFIG_PATH if _AUXILIARY_CONFIG_PATH.exists() else None}
_lit_search = partial(run_lit_search, lit_search_agent=lit_search_agent, model_context=model_context, model_structure_path=MODEL_STRUCTURE, target_kind=TARGET_KIND, **_aux_kwargs)
await run_stage(_lit_search, targets, "lit search")

# Re-read from disk (as_completed returns in arbitrary order)
lit_results = []
for t in targets:
    with open(t["dir"] / "lit_search_results.json") as f:
        lit_results.append(json.load(f))

# Stage 1b: Combine all DOIs into one file
print("\n" + "=" * 60)
print("Stage 1b: DOIs for Zotero")
print("=" * 60)

write_dois_md(targets, lit_results, WORK_DIR / "dois.md")

# Stage 1c: Fetch PDFs from Zotero storage, with browser fallback (interactive)
# Side effects: copies DOIs to clipboard, opens Europe PMC URLs in browser
collect_missing_pdfs(targets, lit_results, ZOTERO_STORAGE)


# ===========================================================================
# Stage 2: Paper assessment
# Reads papers as extracted text (not binary PDFs — avoids context overflow).
# Decides: which papers are usable, what data they have, what needs digitization.
# Results cached in work/<target>/assessment.json
# ===========================================================================

print("\n" + "=" * 60)
print(f"Stage 2: Assess Papers ({len(targets)} targets)")
print("=" * 60)

_assess = partial(run_assess, assess_agent=assess_agent, model_context=model_context, model_structure_path=MODEL_STRUCTURE, priors_csv=PRIORS_CSV, target_kind=TARGET_KIND, **_aux_kwargs)
await run_stage(_assess, targets, "assess")

# Write human-readable assessment reports + digitization README templates
for t in targets:
    ap = t["dir"] / "assessment.json"
    if ap.exists():
        with open(ap) as f:
            write_assessment_report(t, json.load(f))


# ===========================================================================
# Stage 2b: Plan review
# Single LLM call reviews ALL extraction plans together for cross-target
# consistency. Can recommend: proceed, swap to alternative, rerun lit search.
# Result cached in work/plan_review.json
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 2b: Plan Review")
print("=" * 60)

plan_review_result = await run_plan_review(targets, WORK_DIR, plan_review_agent, TARGETS_CSV, target_kind=TARGET_KIND, **_aux_kwargs)

# Apply plan review side effects (explicitly separated from the review itself):
#   apply_plan_swaps:         Modifies assessment.json for swapped targets
#   apply_lit_search_reruns:  Deletes cached lit/assessment, appends RERUN notes to targets.csv
if plan_review_result:
    apply_plan_swaps(plan_review_result.reviews, WORK_DIR)
    apply_lit_search_reruns(plan_review_result.reviews, WORK_DIR, TARGETS_CSV)


# ===========================================================================
# Digitization summary + pre-flight
# After this point, you should digitize any required figures before proceeding.
# Place CSVs in work/<target>/digitized/<source_tag>/<fig_id>.csv
# Re-run summarize_digitizations() to check progress.
# ===========================================================================

# Writes digitization_summary.md with pending/done status, optionally opens PDFs
summarize_digitizations(WORK_DIR, target_ids=[t["target_id"] for t in targets])

# Reports which targets have pending digitizations (informational, does not block)
report_digitization_preflight(targets)


# ===========================================================================
# Stage 3: Assemble target YAMLs (kind selected by TARGET_KIND above).
#   - submodel: maple emits SubmodelTarget (output_type) using the
#     submodel_target_prompt + schema (forward_model + error_model + MCMC).
#   - cal:      maple emits CalibrationTarget using the
#     calibration_target_prompt + schema (observable.code + empirical_data
#     with distribution_code; no forward_model / no MCMC).
# Both kinds: binary PDFs + digitized CSVs are forwarded to the agent.
# Inline snippet-in-PDF validation catches hallucinated quotes (retries
# automatically). Targets with missing digitizations proceed with a warning.
# Results cached in work/<target>/<target_id>_<cancer_type>_deriv001.yaml.
# ===========================================================================

print("=" * 60)
_kind_label = "SubmodelTarget" if TARGET_KIND == "submodel" else "CalibrationTarget"
print(f"Stage 3: Assemble {_kind_label} YAMLs ({len(targets)} targets)")
print("=" * 60)

_complete = partial(
    run_complete,
    complete_agent=complete_agent,
    model_context=model_context,
    model_structure_path=MODEL_STRUCTURE,
    priors_csv=PRIORS_CSV,
    work_dir=WORK_DIR,
    target_kind=TARGET_KIND,
    **_aux_kwargs,
)
await run_stage(_complete, targets, "complete")


# ===========================================================================
# Stage 3b: Derivation review
# Single LLM call reviews ALL completed derivations for scientific soundness
# and cross-target consistency (e.g., contradictory assumptions, IL-10 MW mismatch).
# Result cached in work/derivation_review.json + derivation_review.md
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 3b: Derivation Review")
print("=" * 60)

await run_derivation_review(targets, WORK_DIR, derivation_review_agent)


# ===========================================================================
# Stage 3c: Validate
# Sequential (validation runs forward-model code with mock inputs and, for
# submodel kind, MCMC prior derivation — both CPU-bound). For each target:
#   1. Unit validation (parameter / observable units match QSP model)
#   2. submodel: MCMC prior derivation (numpyro fit, reports contraction).
#      cal:      distribution_code execution + observable.code execution with
#                mock species data (Pint dimensional check, range check,
#                computed median/CI95 vs declared empirical_data tolerance).
#   3. Snippet validation (checks value_snippets against PDF text)
# Passing targets auto-promote:
#   - submodel kind: -> calibration_targets/submodel_targets/
#   - cal kind:      -> calibration_targets/<scenario>/  (scenario from
#                       t['scenario'] or t['model_treatment_history'])
# Already-promoted targets are skipped.
# ===========================================================================

print("\n" + "=" * 60)
print(f"Stage 3c: Validate {_kind_label}s")
print("=" * 60)

run_validate_all(targets, model_structure_path=MODEL_STRUCTURE, priors_csv=PRIORS_CSV, target_kind=TARGET_KIND)


# ===========================================================================
# Stage 3c.aux: Auxiliary-parameter cross-target validation
#
# Independent of distribution_code / observable.code validation — surfaces
# inconsistencies in observable.auxiliary_parameters declarations across the
# *full* calibration_targets/ tree (not just newly-extracted ones), so an
# LLM-authored target that picks the wrong group name or units gets caught
# before it's promoted into a workflow.
#
# Validates:
#   - Every group referenced by any cal target is declared in
#     auxiliary_config.yaml.
#   - Same auxiliary name → same group + same units across all references
#     (the consistency rules enforced by qsp_inference.auxiliary.discovery).
#
# Skipped silently when auxiliary_config.yaml is missing, qsp_inference is
# missing the aux module, or no cal target declares aux parameters — avoids
# bleeding stage failures into runs that don't use the bridging mechanism.
# ===========================================================================

print("\n" + "=" * 60)
print("Stage 3c.aux: Cross-target auxiliary-parameter validation")
print("=" * 60)

_AUX_CONFIG_PATH = Path("auxiliary_config.yaml")
_CAL_TARGETS_ROOT = Path("calibration_targets")

if not _CAL_TARGETS_ROOT.exists():
    print(f"  Skipped: {_CAL_TARGETS_ROOT} does not exist.")
else:
    try:
        from qsp_inference.auxiliary import (
            discover_auxiliary_members as _discover_aux,
            load_auxiliary_config as _load_aux_cfg,
        )
    except ImportError as e:
        print(
            "  Skipped: qsp_inference.auxiliary is not importable "
            f"({e}). Upgrade qsp-inference to enable cross-target validation."
        )
    else:
        _aux_cfg = _load_aux_cfg(_AUX_CONFIG_PATH)
        try:
            _aux_reg = _discover_aux([_CAL_TARGETS_ROOT], _aux_cfg)
        except ValueError as e:
            print("  FAIL — auxiliary cross-target validation:")
            for _line in str(e).splitlines():
                print(f"    {_line}")
            print(
                "  Fix the offending YAMLs (or auxiliary_config.yaml) and rerun. "
                "Newly-extracted YAMLs may still have landed in calibration_targets/, "
                "but downstream sbi_runner will refuse to consume them until the "
                "cross-target consistency is restored."
            )
        else:
            if _aux_reg.is_empty:
                print(
                    "  OK — no auxiliary parameters declared on any cal target "
                    f"under {_CAL_TARGETS_ROOT}. (Bridging mechanism unused.)"
                )
            else:
                _n_grp = len(_aux_reg.group_names)
                _n_mem = len(_aux_reg.member_names)
                print(
                    f"  OK — {_n_mem} auxiliary member(s) across {_n_grp} group(s) "
                    "declared consistently."
                )
