#!/usr/bin/env python3
"""Convert an SBML model to C++ ODE code for SPQSP_PDAC.

Usage:
    # From local SBML file:
    python convert_sbml.py --sbml ../pdac-build/PDAC_model.sbml

    # Fetch latest from GitHub:
    python convert_sbml.py --fetch

    # Custom output directory:
    python convert_sbml.py --sbml model.sbml --output PDAC/qsp/ode/
"""

import argparse
import sys
import urllib.request
from pathlib import Path

from sbml_parser import parse_sbml
from name_mapper import NameMapper
from unit_converter import compute_unit_conversions
from dependency_sorter import sort_assignment_rules, sort_initial_assignments
from generators.qsp_enum_gen import generate_qsp_enum_h
from generators.ode_system_h_gen import generate_ode_system_h
from generators.ode_system_cpp_gen import generate_ode_system_cpp
from generators.qsp_param_h_gen import generate_qsp_param_h
from generators.qsp_param_cpp_gen import generate_qsp_param_cpp

GITHUB_SBML_URL = (
    "https://raw.githubusercontent.com/jeliason/qspio-pdac/main/PDAC_model.sbml"
)

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent.parent / "PDAC" / "qsp" / "ode"
NAMESPACE = "CancerVCT"
CLASS_NAME = "ODE_system"


def fetch_sbml(url: str, dest: Path) -> Path:
    """Download SBML file from GitHub."""
    print(f"Fetching SBML from {url}...")
    urllib.request.urlretrieve(url, dest)
    print(f"Saved to {dest}")
    return dest


def main():
    parser = argparse.ArgumentParser(description="SBML to C++ ODE converter")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sbml", type=Path, help="Path to local SBML file")
    group.add_argument(
        "--fetch", action="store_true", help="Fetch latest SBML from GitHub"
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, help="Output directory for C++ files"
    )
    parser.add_argument(
        "--no-unit-conversion", action="store_true",
        help="Skip SI unit conversion (keep model units: cells, nM, mL, 1/day)"
    )
    args = parser.parse_args()

    # Get SBML file
    if args.fetch:
        sbml_path = args.output / "PDAC_model.sbml"
        fetch_sbml(GITHUB_SBML_URL, sbml_path)
    else:
        sbml_path = args.sbml
        if not sbml_path.exists():
            print(f"Error: SBML file not found: {sbml_path}", file=sys.stderr)
            sys.exit(1)

    # Parse SBML
    print(f"Parsing {sbml_path}...")
    model = parse_sbml(sbml_path)

    # Build name mappings
    mapper = NameMapper(model)

    # Compute unit conversions (or skip for model-unit mode)
    if not args.no_unit_conversion:
        compute_unit_conversions(model)

    # Sort assignment rules and initial assignments by dependencies
    model.assignment_rule_order = sort_assignment_rules(model)
    model.initial_assignment_order = sort_initial_assignments(model)

    # Generate all output files
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "QSP_enum.h": generate_qsp_enum_h(model, mapper, NAMESPACE),
        "ODE_system.h": generate_ode_system_h(NAMESPACE, CLASS_NAME),
        "ODE_system.cpp": generate_ode_system_cpp(model, mapper, NAMESPACE, CLASS_NAME),
        "QSPParam.h": generate_qsp_param_h(model, mapper, NAMESPACE),
        "QSPParam.cpp": generate_qsp_param_cpp(model, mapper, NAMESPACE),
    }

    for filename, content in files.items():
        path = output_dir / filename
        path.write_text(content)
        print(f"  Generated {path}")

    # Generate default parameter XML
    from generators.param_xml_gen import generate_param_xml
    xml_content = generate_param_xml(model, mapper)
    xml_path = output_dir.parent.parent / "sim" / "resource" / "param_all.xml"
    xml_path.parent.mkdir(parents=True, exist_ok=True)
    xml_path.write_text(xml_content)
    print(f"  Generated {xml_path}")

    print("Done.")


if __name__ == "__main__":
    main()
