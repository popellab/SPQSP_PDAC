"""Tests for prior distribution consistency.

These tests verify that the prior CSV files have internally consistent values
and that prior medians match the default values in parameters_PDAC.m.

Usage:
    pytest tests/scenarios/test_prior_consistency.py
"""

import re

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent.parent
PRIORS_FILES = [
    REPO_ROOT / "parameters" / "pdac_priors.csv",
]
PARAMS_FILE = REPO_ROOT / "parameters" / "parameters_PDAC.m"
SBML_FILE = REPO_ROOT / "PDAC_model.sbml"


def parse_parameters_pdac_values(filepath: Path) -> dict[str, float]:
    """Parse params.<name>.Value = <number> from parameters_PDAC.m.

    Returns dict mapping parameter name to its default numeric value.
    Skips parameters with non-numeric values (e.g. [] or expressions).
    """
    text = filepath.read_text()
    pattern = re.compile(r"params\.(\w+)\.Value\s*=\s*([^;%\n]+)")
    params = {}
    for m in pattern.finditer(text):
        name = m.group(1)
        val_str = m.group(2).strip()
        try:
            params[name] = float(val_str)
        except ValueError:
            continue
    return params


def parse_sbml_parameter_names(filepath: Path) -> set[str]:
    """Return the set of SBML <parameter name="..."> names (both top-level
    and reaction-local). SimBiology writes opaque ``id`` attributes
    (e.g. ``mw93db...``) and puts the human-readable name on ``name``,
    so we key on ``name``. These names are what flow into qsp_codegen.py
    and become the tag set of ``param_all.xml``; any prior row whose name
    isn't here triggers a ParamNotFoundError on the C++ worker.

    The SBML is parsed as XML so we don't rely on libsbml being available.
    """
    import xml.etree.ElementTree as ET
    tree = ET.parse(filepath)
    root = tree.getroot()
    names: set[str] = set()
    for elem in root.iter():
        tag = elem.tag.rpartition("}")[2]  # strip namespace
        if tag == "parameter":
            n = elem.get("name") or elem.get("id")
            if n:
                names.add(n)
    return names


def parse_parameters_pdac_units(filepath: Path) -> dict[str, str]:
    """Parse params.<name>.Units = '<units>' from parameters_PDAC.m.

    Returns dict mapping parameter name to its unit string.
    Handles both single-quoted and double-quoted unit strings.
    For duplicate definitions, the last one wins (matches MATLAB behavior).
    """
    text = filepath.read_text()
    pattern = re.compile(r"params\.(\w+)\.Units\s*=\s*['\"]([^'\"]+)['\"]")
    params = {}
    for m in pattern.finditer(text):
        params[m.group(1)] = m.group(2)
    return params


class TestPriorConsistency:
    """Test that prior CSV files have internally consistent values."""

    @pytest.fixture(scope="class", params=PRIORS_FILES, ids=lambda p: p.name)
    def priors_df(self, request) -> pd.DataFrame:
        """Load each priors CSV file."""
        priors_file = request.param
        if not priors_file.exists():
            pytest.skip(f"Priors file not found: {priors_file}")
        return pd.read_csv(priors_file)

    def test_lognormal_median_matches_exp_mu(self, priors_df: pd.DataFrame):
        """For lognormal distributions, median should equal exp(mu).

        The median of a lognormal distribution is exp(mu).
        """
        lognormal_priors = priors_df[priors_df["distribution"] == "lognormal"].copy()

        if lognormal_priors.empty:
            pytest.skip("No lognormal priors found")

        lognormal_priors["computed_median"] = np.exp(lognormal_priors["dist_param1"])

        lognormal_priors["rel_error"] = np.abs(
            lognormal_priors["median"] - lognormal_priors["computed_median"]
        ) / lognormal_priors["median"]

        # Allow 1% relative error for floating point
        tolerance = 0.01
        mismatches = lognormal_priors[lognormal_priors["rel_error"] >= tolerance]

        if not mismatches.empty:
            details = mismatches[[
                "name", "median", "dist_param1", "computed_median", "rel_error"
            ]].copy()
            details.columns = ["name", "median", "mu", "exp(mu)", "rel_error"]

            pytest.fail(
                f"\n{len(mismatches)} lognormal prior(s) have median that doesn't "
                f"match exp(mu):\n\n"
                f"{details.to_string(index=False)}\n\n"
                f"The mu (dist_param1) value should be ln(median)."
            )

    def test_prior_medians_match_parameters_pdac(self, priors_df: pd.DataFrame):
        """Prior CSV medians should match default values in parameters_PDAC.m.

        This catches drift between the priors CSV and the MATLAB parameter file
        (e.g. median=3.2 in CSV vs Value=3 in .m file).
        """
        if not PARAMS_FILE.exists():
            pytest.skip(f"Parameters file not found: {PARAMS_FILE}")

        matlab_params = parse_parameters_pdac_values(PARAMS_FILE)

        mismatches = []
        for _, row in priors_df.iterrows():
            name = row["name"]
            if name not in matlab_params:
                continue
            csv_median = row["median"]
            matlab_val = matlab_params[name]
            if matlab_val == 0:
                continue
            rel_error = abs(csv_median - matlab_val) / abs(matlab_val)
            if rel_error >= 0.01:
                mismatches.append({
                    "name": name,
                    "csv_median": csv_median,
                    "matlab_value": matlab_val,
                    "rel_error": f"{rel_error:.2%}",
                })

        if mismatches:
            details = pd.DataFrame(mismatches)
            pytest.fail(
                f"\n{len(mismatches)} prior median(s) don't match "
                f"parameters_PDAC.m defaults (>1% relative error):\n\n"
                f"{details.to_string(index=False)}\n\n"
                f"Update either the CSV median or the .m Value to keep them in sync."
            )

    def test_prior_names_in_sbml(self, priors_df: pd.DataFrame):
        """Every prior row must correspond to an SBML parameter id.

        The SBML export is what flows into qsp_codegen.py → param_all.xml →
        the C++ worker. Prior rows whose names aren't in the SBML come from
        MATLAB-side parameters whose module has been removed (but the decl
        left in parameters_PDAC.m is dead code and therefore not exported).

        Checking against SBML (not parameters_PDAC.m) is stricter and
        matches what actually propagates to the simulator.
        """
        if not SBML_FILE.exists():
            pytest.skip(f"SBML not found: {SBML_FILE}")

        sbml_names = parse_sbml_parameter_names(SBML_FILE)
        orphans = sorted(set(priors_df["name"]) - sbml_names)
        if orphans:
            pytest.fail(
                f"\n{len(orphans)} prior row(s) not present as <parameter id> "
                f"in {SBML_FILE.name}:\n\n"
                f"  {orphans[:20]}{'...' if len(orphans) > 20 else ''}\n\n"
                f"These priors reference MATLAB parameters whose module has "
                f"been removed from the SimBiology model (the decl in "
                f"parameters_PDAC.m is dead code, not exported). Either "
                f"re-add the module + re-export SBML, or drop the rows "
                f"from pdac_priors.csv."
            )

    def test_prior_units_match_parameters_pdac(self, priors_df: pd.DataFrame):
        """Prior CSV units should match units in parameters_PDAC.m.

        Mismatched units (e.g. 'nanomole/cell/day' vs 'nanomolarity*liter/(cell*day)')
        cause silent value errors when priors override model defaults.
        """
        if not PARAMS_FILE.exists():
            pytest.skip(f"Parameters file not found: {PARAMS_FILE}")

        matlab_units = parse_parameters_pdac_units(PARAMS_FILE)

        mismatches = []
        for _, row in priors_df.iterrows():
            name = row["name"]
            if name not in matlab_units:
                continue
            csv_units = row["units"]
            matlab_unit = matlab_units[name]
            if csv_units != matlab_unit:
                mismatches.append({
                    "name": name,
                    "csv_units": csv_units,
                    "matlab_units": matlab_unit,
                })

        if mismatches:
            details = pd.DataFrame(mismatches)
            pytest.fail(
                f"\n{len(mismatches)} prior unit(s) don't match "
                f"parameters_PDAC.m units:\n\n"
                f"{details.to_string(index=False)}\n\n"
                f"Update either the CSV units or the .m Units to keep them in sync."
            )
