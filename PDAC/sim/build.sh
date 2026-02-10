#!/bin/bash
# Build script for TNBC ABM GPU simulation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

# Default options
BUILD_TYPE="Release"
CUDA_ARCH=""
FLAMEGPU_PATH=""

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
        --clean)
            echo "Cleaning build directory..."
            rm -rf "${BUILD_DIR}"
            exit 0
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --debug           Build in debug mode"
            echo "  --cuda-arch ARCH  Set CUDA architecture (e.g., 75, 80, 86)"
            echo "  --flamegpu PATH   Path to local FLAMEGPU2 source"
            echo "  --clean           Remove build directory"
            echo "  --help            Show this help"
            echo ""
            echo "Example:"
            echo "  $0 --cuda-arch 86 --flamegpu ~/FLAMEGPU2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure ccache for CUDA compilation
export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
export CMAKE_CXX_COMPILER_LAUNCHER=ccache
export CMAKE_C_COMPILER_LAUNCHER=ccache

# Configure CMake
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}"
)

if [[ -n "${CUDA_ARCH}" ]]; then
    CMAKE_ARGS+=(-DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}")
fi

if [[ -n "${FLAMEGPU_PATH}" ]]; then
    CMAKE_ARGS+=(-DFLAMEGPU_ROOT="${FLAMEGPU_PATH}")
fi

echo "=== Configuring TNBC ABM GPU ==="
echo "Build type: ${BUILD_TYPE}"
echo "Build directory: ${BUILD_DIR}"
if [[ -n "${CUDA_ARCH}" ]]; then
    echo "CUDA architecture: ${CUDA_ARCH}"
fi
if [[ -n "${FLAMEGPU_PATH}" ]]; then
    echo "FLAMEGPU2 path: ${FLAMEGPU_PATH}"
fi
echo ""

cmake "${SCRIPT_DIR}" "${CMAKE_ARGS[@]}"

echo ""
echo "=== Building ==="
echo "Using $(nproc) parallel jobs"
time cmake --build . --parallel $(nproc)

echo ""
echo "=== Build Complete ==="
echo "Executable: ${BUILD_DIR}/bin/pdac"
echo ""
echo "Run with: ${BUILD_DIR}/bin/pdac --help"
