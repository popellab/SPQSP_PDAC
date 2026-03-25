"""Shared fixtures for converter tests."""

import sys
from pathlib import Path

import pytest

# Add parent directory to path so we can import converter modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    parser.addoption("--run-matlab", action="store_true", default=False,
                     help="Run MATLAB-dependent validation tests (Tier 3)")

SBML_PATH = Path(__file__).resolve().parent.parent.parent.parent.parent / "pdac-build" / "PDAC_model.sbml"


@pytest.fixture(scope="session")
def sbml_path():
    if not SBML_PATH.exists():
        pytest.skip(f"SBML file not found: {SBML_PATH}")
    return SBML_PATH


@pytest.fixture(scope="session")
def parsed_model(sbml_path):
    from sbml_parser import parse_sbml
    return parse_sbml(sbml_path)


@pytest.fixture(scope="session")
def model_with_units(parsed_model):
    from unit_converter import compute_unit_conversions
    compute_unit_conversions(parsed_model)
    return parsed_model


@pytest.fixture(scope="session")
def model_with_sorting(model_with_units):
    from dependency_sorter import sort_assignment_rules, sort_initial_assignments
    model_with_units.assignment_rule_order = sort_assignment_rules(model_with_units)
    model_with_units.initial_assignment_order = sort_initial_assignments(model_with_units)
    return model_with_units


@pytest.fixture(scope="session")
def mapper(model_with_sorting):
    from name_mapper import NameMapper
    return NameMapper(model_with_sorting)
