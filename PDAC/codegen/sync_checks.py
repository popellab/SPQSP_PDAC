"""Configuration / codegen / param-XML sync checks.

Single source of truth for the "is everything in sync?" guards used by both
the pytest harness (PDAC/sim/tests/test_ode_vs_matlab.py) and the standalone
CLI (PDAC/codegen/check_sync.py). Each check returns (ok, message).

Paths are passed in so callers can override (tests, CI) without env vars.
"""
from __future__ import annotations

import filecmp
import os
import re
from pathlib import Path
from typing import Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# qspio-pdac is the MATLAB QSP model (source of truth for ODE, SBML, params).
QSPIO = Path(os.environ.get("QSPIO_DIR", REPO_ROOT / "qspio-pdac"))
DEFAULT_SBML = QSPIO / "PDAC_model.sbml"
DEFAULT_MATLAB_MODEL = QSPIO / "scripts" / "immune_oncology_model_PDAC.m"
DEFAULT_ODE_CPP = REPO_ROOT / "PDAC" / "qsp" / "ode" / "ODE_system.cpp"
DEFAULT_ODE_CPP_SRC = QSPIO / "cpp" / "qsp" / "ode" / "ODE_system.cpp"
DEFAULT_PARAM_SNIPPET = REPO_ROOT / "PDAC" / "qsp" / "ode" / "qsp_params_xml_snippet.xml"
DEFAULT_PARAM_XML = REPO_ROOT / "PDAC" / "sim" / "resource" / "param_all.xml"
DEFAULT_DUMP_BIN = REPO_ROOT / "PDAC" / "qsp" / "sim" / "build" / "qsp_sim"
DEFAULT_PRIORS_CSV = REPO_ROOT / "parameters" / "pdac_priors.csv"

ODE_FILES = ("ODE_system.cpp", "ODE_system.h", "QSPParam.cpp", "QSPParam.h", "QSP_enum.h")

Result = Tuple[bool, str]


def check_sbml_newer_than_matlab(
    sbml: Path = DEFAULT_SBML,
    matlab_script: Path = DEFAULT_MATLAB_MODEL,
) -> Result:
    """SBML must be re-exported after edits to the live MATLAB model script."""
    if not matlab_script.exists():
        return True, f"skip: MATLAB model script not found ({matlab_script})"
    if not sbml.exists():
        return False, (
            f"SBML missing: {sbml}\n"
            f"  Re-export: cd qspio-pdac && make export-sbml"
        )
    drift = matlab_script.stat().st_mtime - sbml.stat().st_mtime
    if drift > 0:
        return False, (
            f"{sbml.name} is {drift:.0f}s older than {matlab_script.name}.\n"
            f"  Re-export: cd qspio-pdac && make export-sbml"
        )
    return True, "SBML is up to date with MATLAB model script"


def check_codegen_newer_than_sbml(
    ode_cpp: Path = DEFAULT_ODE_CPP,
    sbml: Path = DEFAULT_SBML,
) -> Result:
    """ODE_system.cpp must be regenerated after SBML changes."""
    if not sbml.exists():
        return True, f"skip: SBML not found ({sbml})"
    if not ode_cpp.exists():
        return False, (
            f"Generated ODE missing: {ode_cpp}\n"
            f"  Regenerate: cd qspio-pdac && make refresh-cpp"
        )
    drift = sbml.stat().st_mtime - ode_cpp.stat().st_mtime
    if drift > 0:
        return False, (
            f"{ode_cpp.name} is {drift:.0f}s older than the SBML.\n"
            f"  Regenerate: cd qspio-pdac && make refresh-cpp"
        )
    return True, "Generated ODE is up to date with SBML"


def check_ode_files_match(
    abm_ode_dir: Path = DEFAULT_ODE_CPP.parent,
    qspio_ode_dir: Path = DEFAULT_ODE_CPP_SRC.parent,
) -> Result:
    """PDAC/qsp/ode/ must be identical to qspio-pdac/cpp/qsp/ode/."""
    if not qspio_ode_dir.exists():
        return True, f"skip: qspio-pdac ODE dir not found ({qspio_ode_dir})"
    mismatched = []
    for name in ODE_FILES:
        abm_f = abm_ode_dir / name
        src_f = qspio_ode_dir / name
        if not src_f.exists():
            continue
        if not abm_f.exists():
            mismatched.append(f"{name} (missing in PDAC/)")
        elif not filecmp.cmp(abm_f, src_f, shallow=False):
            mismatched.append(name)
    if mismatched:
        return False, (
            f"ODE files out of sync: {', '.join(mismatched)}\n"
            f"  Fix: make sync-qsp"
        )
    return True, f"all {len(ODE_FILES)} ODE files match qspio-pdac"


def check_binary_newer_than_codegen(
    dump_bin: Path = DEFAULT_DUMP_BIN,
    ode_cpp: Path = DEFAULT_ODE_CPP,
) -> Result:
    """The qsp_sim binary must be rebuilt after codegen runs."""
    if not dump_bin.exists():
        return True, f"skip: qsp_sim not built ({dump_bin})"
    if not ode_cpp.exists():
        return True, "skip: no codegen output to compare against"
    drift = ode_cpp.stat().st_mtime - dump_bin.stat().st_mtime
    if drift > 0:
        return False, (
            f"{dump_bin.name} is {drift:.0f}s older than {ode_cpp.name}.\n"
            f"  Rebuild: cmake --build {dump_bin.parent} --target qsp_sim"
        )
    return True, "qsp_sim is up to date"


def check_param_xml_contains_snippet(
    snippet: Path = DEFAULT_PARAM_SNIPPET,
    param_xml: Path = DEFAULT_PARAM_XML,
) -> Result:
    """Every leaf tag in qsp_params_xml_snippet.xml must appear in param_all.xml."""
    if not snippet.exists():
        return True, f"skip: snippet not found ({snippet})"
    if not param_xml.exists():
        return False, f"param_all.xml missing: {param_xml}"
    snippet_names = set(re.findall(r"<([A-Za-z_][\w]*)>", snippet.read_text()))
    xml_names = set(re.findall(r"<([A-Za-z_][\w]*)>", param_xml.read_text()))
    missing = sorted(snippet_names - xml_names)
    if missing:
        return False, (
            f"{len(missing)} name(s) in {snippet.name} missing from {param_xml.name}:\n"
            f"  {missing[:20]}\n"
            f"  Fix: make sync-qsp"
        )
    return True, f"{param_xml.name} contains all snippet names"


def check_priors_csv_names_in_param_xml(
    priors_csv: Path = DEFAULT_PRIORS_CSV,
    param_xml: Path = DEFAULT_PARAM_XML,
) -> Result:
    """Every parameter name in pdac_priors.csv must appear in param_all.xml."""
    if not priors_csv.exists():
        return True, f"skip: priors CSV not found ({priors_csv})"
    if not param_xml.exists():
        return False, f"param_all.xml missing: {param_xml}"
    import csv
    with open(priors_csv) as f:
        prior_names = {row["name"] for row in csv.DictReader(f)}
    xml_names = set(re.findall(r"<([A-Za-z_][\w]*)>", param_xml.read_text()))
    missing = sorted(prior_names - xml_names)
    if missing:
        return False, (
            f"{len(missing)} prior(s) in {priors_csv.name} missing from "
            f"{param_xml.name}:\n"
            f"  {missing[:20]}{'...' if len(missing) > 20 else ''}\n"
            f"  Fix: ensure qspio-pdac codegen is current, then make sync-qsp"
        )
    return True, f"all {len(prior_names)} prior names present in {param_xml.name}"


ALL_CHECKS = [
    check_sbml_newer_than_matlab,
    check_codegen_newer_than_sbml,
    check_ode_files_match,
    check_binary_newer_than_codegen,
    check_param_xml_contains_snippet,
    check_priors_csv_names_in_param_xml,
]


def run_all() -> list[Tuple[str, bool, str]]:
    """Run every check and return (name, ok, message) triples."""
    return [(fn.__name__, *fn()) for fn in ALL_CHECKS]
