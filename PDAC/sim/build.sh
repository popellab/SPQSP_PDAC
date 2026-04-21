#!/bin/bash
# Build script for SPQSP PDAC GPU simulation
# All dependencies (FLAME GPU 2, SUNDIALS, Boost) are auto-fetched if not found on system.
# Usage: ./build.sh [options]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Default options
BUILD_TYPE="Release"
CUDA_ARCH=""
FLAMEGPU_PATH=""
JOBS=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --cuda-arch)
            CUDA_ARCH="$2"
            shift 2
            ;;
        --flamegpu)
            FLAMEGPU_PATH="$2"
            shift 2
            ;;
        -j|--jobs)
            JOBS="$2"
            shift 2
            ;;
        --clean)
            echo "Cleaning build directory..."
            rm -rf "${BUILD_DIR}"
            exit 0
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug             Build in debug mode"
            echo "  --cuda-arch ARCH    Set CUDA architecture (e.g., 80 for A100, 90 for H100)"
            echo "  --flamegpu PATH     Path to local FLAMEGPU2 source"
            echo "  -j, --jobs N        Number of parallel build jobs (default: nproc)"
            echo "  --clean             Remove build directory"
            echo "  --help              Show this help"
            echo ""
            echo "Environment variables (optional, for using system-installed libraries):"
            echo "  SUNDIALS_DIR        Path to SUNDIALS installation"
            echo "  BOOST_ROOT          Path to Boost installation"
            echo ""
            echo "All dependencies are auto-fetched if not found on the system."
            echo ""
            echo "Examples:"
            echo "  $0                                  # Build (auto-fetch deps if needed)"
            echo "  $0 --cuda-arch 80                   # Build for A100"
            echo "  $0 --cuda-arch 90 -j 16             # Build for H100, 16 jobs"
            echo "  SUNDIALS_DIR=/opt/sundials $0       # Use system SUNDIALS"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Cap parallelism by RAM — each nvcc TU with FLAMEGPU/Boost/SUNDIALS peaks
# at ~3–5 GB. Default to min(nproc, RAM_GB / 4) to avoid OOM (WSL crash).
if [[ -z "${JOBS}" ]]; then
    RAM_GB=$(awk '/MemTotal/ {printf "%d", $2/1024/1024}' /proc/meminfo)
    RAM_JOBS=$(( RAM_GB / 4 ))
    (( RAM_JOBS < 1 )) && RAM_JOBS=1
    NPROC_JOBS=$(nproc)
    JOBS=$(( RAM_JOBS < NPROC_JOBS ? RAM_JOBS : NPROC_JOBS ))
    echo "  auto -j ${JOBS} (RAM=${RAM_GB}GB, nproc=${NPROC_JOBS})"
fi

# ============================================================================
# Prerequisite checks
# ============================================================================

check_command() {
    if ! command -v "$1" &>/dev/null; then
        echo "ERROR: '$1' not found. $2"
        return 1
    fi
}

echo "=== Checking prerequisites ==="
check_command cmake "Install cmake >= 3.18" || exit 1
check_command nvcc "Install CUDA Toolkit (https://developer.nvidia.com/cuda-toolkit)" || exit 1
check_command g++ "Install a C++17 compiler (g++ 7+)" || exit 1
check_command git "Install git" || exit 1

# Check cmake version
CMAKE_VER=$(cmake --version | head -1 | grep -oP '\d+\.\d+')
CMAKE_MAJOR=$(echo "$CMAKE_VER" | cut -d. -f1)
CMAKE_MINOR=$(echo "$CMAKE_VER" | cut -d. -f2)
if [[ "$CMAKE_MAJOR" -lt 3 ]] || { [[ "$CMAKE_MAJOR" -eq 3 ]] && [[ "$CMAKE_MINOR" -lt 18 ]]; }; then
    echo "ERROR: cmake >= 3.18 required (found $CMAKE_VER)"
    exit 1
fi

echo "  cmake   $(cmake --version | head -1 | grep -oP '[\d.]+')"
echo "  nvcc    $(nvcc --version | grep release | grep -oP '[\d.]+')"
echo "  g++     $(g++ -dumpversion)"

# ============================================================================
# Build
# ============================================================================

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Use ccache if available
if command -v ccache &>/dev/null; then
    export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
    export CMAKE_CXX_COMPILER_LAUNCHER=ccache
    export CMAKE_C_COMPILER_LAUNCHER=ccache
    echo "  ccache  enabled"
fi

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)

[[ -n "${CUDA_ARCH}" ]] && CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}")
[[ -n "${FLAMEGPU_PATH}" ]] && CMAKE_ARGS+=(-DFLAMEGPU_ROOT="${FLAMEGPU_PATH}")
[[ -n "${SUNDIALS_DIR}" ]] && CMAKE_ARGS+=(-DSUNDIALS_DIR="${SUNDIALS_DIR}")
[[ -n "${BOOST_ROOT}" ]] && CMAKE_ARGS+=(-DBOOST_ROOT="${BOOST_ROOT}")

echo ""
echo "=== Configuring SPQSP PDAC ==="
echo "  Build type:  ${BUILD_TYPE}"
echo "  Build dir:   ${BUILD_DIR}"
[[ -n "${CUDA_ARCH}" ]] && echo "  CUDA arch:   ${CUDA_ARCH}"
[[ -n "${SUNDIALS_DIR}" ]] && echo "  SUNDIALS:    ${SUNDIALS_DIR}"
[[ -n "${BOOST_ROOT}" ]] && echo "  Boost:       ${BOOST_ROOT}"
echo ""
echo "  Dependencies not found on system will be fetched automatically."
echo ""

cmake "${SCRIPT_DIR}" "${CMAKE_ARGS[@]}"

# Patch FLAMEGPU Curve hash table limit (512 → 1024) for SPQSP's large variable count
CURVE_HDR="${BUILD_DIR}/_deps/flamegpu-src/include/flamegpu/runtime/detail/curve/Curve.cuh"
if [[ -f "${CURVE_HDR}" ]]; then
    sed -i 's/MAX_VARIABLES = 512;/MAX_VARIABLES = 1024;/' "${CURVE_HDR}"
fi

echo ""
echo "=== Building (${JOBS} jobs) ==="
time cmake --build . --parallel "${JOBS}"

echo ""
echo "=== Build Complete ==="
echo "Executable: ${BUILD_DIR}/bin/pdac"
echo ""
echo "Run with: ${BUILD_DIR}/bin/pdac --help"
