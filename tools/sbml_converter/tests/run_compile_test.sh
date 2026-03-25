#!/bin/bash
# Run the Tier 2 compile test: generate ODE code, build, and run.
#
# Usage: ./run_compile_test.sh [path/to/PDAC_model.sbml]
#
# Requires: cmake, C++ compiler, Boost (serialization), python-libsbml

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERTER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$CONVERTER_DIR/../.." && pwd)"
ODE_OUTPUT_DIR="$REPO_ROOT/PDAC/qsp/ode"
BUILD_DIR="$SCRIPT_DIR/ode_compile_test/build"

SBML_PATH="${1:-$REPO_ROOT/../pdac-build/PDAC_model.sbml}"

if [ ! -f "$SBML_PATH" ]; then
    echo "SBML file not found: $SBML_PATH"
    exit 1
fi

echo "=== Step 1: Generate C++ from SBML ==="
cd "$CONVERTER_DIR"
source .venv/bin/activate 2>/dev/null || true
python convert_sbml.py --sbml "$SBML_PATH" --output "$ODE_OUTPUT_DIR"

echo ""
echo "=== Step 2: Build ==="
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
cmake .. -DPARAM_XML_PATH="$REPO_ROOT/PDAC/sim/resource/param_all.xml"
cmake --build . -j$(nproc 2>/dev/null || sysctl -n hw.ncpu)

echo ""
echo "=== Step 3: Run test ==="
ctest --output-on-failure
