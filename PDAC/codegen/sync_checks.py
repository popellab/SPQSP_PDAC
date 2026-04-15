"""Configuration / codegen / param-XML sync checks.

Single source of truth for the "is everything in sync?" guards used by both
the pytest harness (PDAC/sim/tests/test_ode_vs_matlab.py) and the standalone
CLI (PDAC/codegen/check_sync.py). Each check returns (ok, message).

Paths are passed in so callers can override (tests, CI) without env vars.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import os

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# SBML is checked in here as the contract between MATLAB and C++.
# Re-export from MATLAB → copy file → commit.
DEFAULT_SBML = REPO_ROOT / "PDAC" / "qsp" / "PDAC_model.sbml"

# The live MATLAB model script lives outside this repo. Override the
# directory with PDAC_BUILD_DIR; defaults to a sibling pdac-build/.
PDAC_BUILD_DIR = Path(os.environ.get("PDAC_BUILD_DIR", REPO_ROOT.parent / "pdac-build"))
DEFAULT_MATLAB_MODEL = PDAC_BUILD_DIR / "scripts" / "immune_oncology_model_PDAC.m"
DEFAULT_ODE_CPP = REPO_ROOT / "PDAC" / "qsp" / "ode" / "ODE_system.cpp"
DEFAULT_PARAM_SNIPPET = REPO_ROOT / "PDAC" / "qsp" / "ode" / "qsp_params_xml_snippet.xml"
DEFAULT_PARAM_XML = REPO_ROOT / "PDAC" / "sim" / "resource" / "param_all.xml"
DEFAULT_DUMP_BIN = REPO_ROOT / "PDAC" / "qsp" / "sim" / "build" / "qsp_sim"

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
            f"  Re-export in pdac-build: matlab -batch \"run('scripts/export_sbml.m')\""
        )
    drift = matlab_script.stat().st_mtime - sbml.stat().st_mtime
    if drift > 0:
        return False, (
            f"{sbml.name} is {drift:.0f}s older than {matlab_script.name}.\n"
            f"  Re-export: matlab -batch \"run('scripts/export_sbml.m')\""
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
            f"  Run: python PDAC/codegen/qsp_codegen.py {sbml}"
        )
    drift = sbml.stat().st_mtime - ode_cpp.stat().st_mtime
    if drift > 0:
        return False, (
            f"{ode_cpp.name} is {drift:.0f}s older than the SBML.\n"
            f"  Regenerate: python PDAC/codegen/qsp_codegen.py {sbml}"
        )
    return True, "Generated ODE is up to date with SBML"


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
            f"  Refresh: python PDAC/codegen/refresh_param_xml.py"
        )
    return True, f"{param_xml.name} contains all snippet names"


ALL_CHECKS = [
    check_sbml_newer_than_matlab,
    check_codegen_newer_than_sbml,
    check_binary_newer_than_codegen,
    check_param_xml_contains_snippet,
]


def run_all() -> list[Tuple[str, bool, str]]:
    """Run every check and return (name, ok, message) triples."""
    return [(fn.__name__, *fn()) for fn in ALL_CHECKS]