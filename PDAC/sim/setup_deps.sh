#!/bin/bash
# setup_deps.sh — Fetch all build dependencies into external/
#
# Run once on a node with internet access (login node).
# After this, builds work offline (GPU nodes without internet).
#
# Usage:
#   ./setup_deps.sh              # fetch all deps
#   ./setup_deps.sh --status     # check what's already fetched
#   ./setup_deps.sh --clean      # remove external/ and start fresh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_DIR="${SCRIPT_DIR}/external"

# Dependency versions (keep in sync with CMakeLists.txt)
FLAMEGPU_TAG="v2.0.0-rc.4"
SUNDIALS_TAG="v4.1.0"
BOOST_VERSION="1.82.0"
BOOST_SHA256="b62bd839ea6c28265af9a1f68393eda37fab3611425d3b28882d8e424535ec9d"

# --------------------------------------------------------------------------

status() {
    echo "=== Dependency Status (${EXT_DIR}) ==="
    for dep in flamegpu2 sundials boost; do
        if [[ -f "${EXT_DIR}/${dep}/CMakeLists.txt" ]]; then
            echo "  ${dep}: OK"
        else
            echo "  ${dep}: MISSING"
        fi
    done
}

if [[ "${1}" == "--status" ]]; then
    status
    exit 0
fi

if [[ "${1}" == "--clean" ]]; then
    echo "Removing ${EXT_DIR}..."
    rm -rf "${EXT_DIR}"
    echo "Done."
    exit 0
fi

# Check for internet
if ! curl -s --connect-timeout 5 https://github.com -o /dev/null 2>/dev/null; then
    echo "ERROR: No internet access. Run this on a login node."
    exit 1
fi

mkdir -p "${EXT_DIR}"

# --- FLAME GPU 2 ---
if [[ -f "${EXT_DIR}/flamegpu2/CMakeLists.txt" ]]; then
    echo "FLAMEGPU2: already present, skipping."
else
    echo "Fetching FLAMEGPU2 ${FLAMEGPU_TAG}..."
    git clone --branch "${FLAMEGPU_TAG}" --depth 1 \
        https://github.com/FLAMEGPU/FLAMEGPU2.git "${EXT_DIR}/flamegpu2"
    cd "${EXT_DIR}/flamegpu2"
    git submodule update --init --recursive
    cd "${SCRIPT_DIR}"
    echo "FLAMEGPU2: done."
fi

# --- SUNDIALS ---
if [[ -f "${EXT_DIR}/sundials/CMakeLists.txt" ]]; then
    echo "SUNDIALS: already present, skipping."
else
    echo "Fetching SUNDIALS ${SUNDIALS_TAG}..."
    git clone --branch "${SUNDIALS_TAG}" --depth 1 \
        https://github.com/LLNL/sundials.git "${EXT_DIR}/sundials"
    echo "SUNDIALS: done."
fi

# --- Boost ---
if [[ -f "${EXT_DIR}/boost/CMakeLists.txt" ]]; then
    echo "Boost: already present, skipping."
else
    echo "Fetching Boost ${BOOST_VERSION}..."
    BOOST_TAR="boost-${BOOST_VERSION}.tar.gz"
    curl -L -o "${EXT_DIR}/${BOOST_TAR}" \
        "https://github.com/boostorg/boost/releases/download/boost-${BOOST_VERSION}/${BOOST_TAR}"

    # Verify checksum
    ACTUAL_SHA=$(sha256sum "${EXT_DIR}/${BOOST_TAR}" | awk '{print $1}')
    if [[ "${ACTUAL_SHA}" != "${BOOST_SHA256}" ]]; then
        echo "ERROR: Boost checksum mismatch!"
        echo "  Expected: ${BOOST_SHA256}"
        echo "  Got:      ${ACTUAL_SHA}"
        rm -f "${EXT_DIR}/${BOOST_TAR}"
        exit 1
    fi

    tar xzf "${EXT_DIR}/${BOOST_TAR}" -C "${EXT_DIR}"
    mv "${EXT_DIR}/boost-${BOOST_VERSION}" "${EXT_DIR}/boost"
    rm -f "${EXT_DIR}/${BOOST_TAR}"
    echo "Boost: done."
fi

echo ""
status
echo ""
echo "Dependencies ready. You can now build (even on offline GPU nodes):"
echo "  sbatch submit.sh"
