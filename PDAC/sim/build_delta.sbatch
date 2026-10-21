#!/bin/bash
#SBATCH --job-name=pdac_build
#SBATCH --partition=gpuA100x4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32g
#SBATCH --time=01:30:00
#SBATCH --output=build_%j.out
#SBATCH --error=build_%j.err

# ============================================================================
# Build PDAC on Delta A100.
#
# Usage (from repo root or PDAC/sim/):
#   sbatch --account=bgre-delta-gpu PDAC/sim/build_delta.sbatch
#   sbatch --account=bgre-delta-gpu build_delta.sbatch
# ============================================================================

set -e

# SLURM copies scripts to a spool dir, so BASH_SOURCE won't work.
# Use SLURM_SUBMIT_DIR (where sbatch was called from).
if [[ -f "${SLURM_SUBMIT_DIR}/build.sh" ]]; then
    SIM_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/PDAC/sim/build.sh" ]]; then
    SIM_DIR="${SLURM_SUBMIT_DIR}/PDAC/sim"
else
    echo "ERROR: Can't find build.sh. Submit from PDAC/sim/ or repo root."
    exit 1
fi
REPO_DIR="$(cd "${SIM_DIR}/../.." && pwd)"
cd "${SIM_DIR}"

echo "=== Build started: $(date) ==="
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"
echo "Repo dir: ${REPO_DIR}"

# Delta default env has gcc 13+ but CUDA libs (cufft etc.) need explicit module
# Default cudatoolkit conflicts with cuda module — swap it
module unload cudatoolkit 2>/dev/null || true
module load cuda/12.9
module load cmake/3.31.8 2>/dev/null || module load cmake 2>/dev/null || true

# HPC SDK splits math libs (cufft, cublas, etc.) from the CUDA runtime
CUDA_MATH_LIBS="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/REDIST/math_libs/12.9/targets/x86_64-linux/lib"
if [[ -d "${CUDA_MATH_LIBS}" ]]; then
    export LIBRARY_PATH="${CUDA_MATH_LIBS}:${LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${CUDA_MATH_LIBS}:${LD_LIBRARY_PATH}"
    echo "CUDA math libs: ${CUDA_MATH_LIBS}"
fi

echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "cmake: $(cmake --version 2>/dev/null | head -1 || echo 'not found')"
echo "gcc: $(gcc --version 2>/dev/null | head -1 || echo 'not found')"

# ── Pre-stage Boost (CMake FetchContent tar extraction fails on Lustre) ──
BOOST_STAGE="${REPO_DIR}/PDAC/sim/external"
BOOST_VER="1.82.0"
BOOST_DIR="${BOOST_STAGE}/boost-${BOOST_VER}"
if [[ ! -f "${BOOST_DIR}/CMakeLists.txt" ]]; then
    echo ""
    echo "=== Pre-staging Boost ${BOOST_VER} ==="
    mkdir -p "${BOOST_STAGE}"
    BOOST_TAR="${BOOST_STAGE}/boost-${BOOST_VER}.tar.gz"
    if [[ ! -f "${BOOST_TAR}" ]]; then
        echo "  Downloading..."
        curl -L -o "${BOOST_TAR}" \
            "https://github.com/boostorg/boost/releases/download/boost-${BOOST_VER}/boost-${BOOST_VER}.tar.gz"
    fi
    echo "  Extracting with system tar (avoids CMake timestamp bug)..."
    cd "${BOOST_STAGE}"
    tar xzf "${BOOST_TAR}"
    cd "${SIM_DIR}"
    echo "  Boost ready: ${BOOST_DIR}"
fi
export BOOST_ROOT="${BOOST_DIR}"

# First-time build fetches FLAMEGPU2 + SUNDIALS from GitHub (~8-15 min)
# Subsequent incremental builds are ~2 min
./build.sh --cuda-arch 80 -j 4

echo "=== Build finished: $(date) ==="
