#!/bin/bash
# Run Tier 3 MATLAB vs C++ validation and stamp the result.
#
# Usage: ./run_validation.sh
#
# Requires: MATLAB, numpy, built dump_trajectories binary

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONVERTER_DIR="$(dirname "$SCRIPT_DIR")"
REPO_ROOT="$(cd "$CONVERTER_DIR/../.." && pwd)"

cd "$CONVERTER_DIR"
source .venv/bin/activate 2>/dev/null || true

echo "=== Running Tier 3: MATLAB vs C++ validation ==="
python -m pytest tests/test_runtime_validation.py -v --run-matlab

# If we get here, tests passed — write stamp
SHA=$(git -C "$REPO_ROOT" rev-parse HEAD)
DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
echo "$SHA $DATE PASS" > "$SCRIPT_DIR/validation_stamp.txt"
echo ""
echo "Validation passed. Stamp written: $SHA"