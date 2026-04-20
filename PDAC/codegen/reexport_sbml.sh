#!/bin/bash
# Re-export PDAC_model.sbml from the live MATLAB project into this repo.
#
# Reads the MATLAB project location from PDAC_BUILD_DIR (defaults to a
# sibling pdac-build/). Runs MATLAB, then copies the fresh SBML next to
# the codegen so CMake will pick it up on the next build.
#
# Usage: ./PDAC/codegen/reexport_sbml.sh
set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PDAC_BUILD_DIR="${PDAC_BUILD_DIR:-${REPO_ROOT}/../pdac-build}"
DEST="${REPO_ROOT}/PDAC/qsp/PDAC_model.sbml"

if [[ ! -d "${PDAC_BUILD_DIR}" ]]; then
    echo "ERROR: PDAC_BUILD_DIR not found: ${PDAC_BUILD_DIR}" >&2
    echo "Set PDAC_BUILD_DIR to your pdac-build checkout." >&2
    exit 1
fi

echo "Re-exporting SBML from ${PDAC_BUILD_DIR}..."
(cd "${PDAC_BUILD_DIR}" && matlab -batch "run('scripts/export_sbml.m')")
cp "${PDAC_BUILD_DIR}/PDAC_model.sbml" "${DEST}"
echo "Updated: ${DEST}"
echo "Next: rebuild (cmake will regenerate ODE C++ and refresh param_all.xml)."
