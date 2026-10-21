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

# Delta default gcc-toolset-14 (gcc 14) is too new for FLAMEGPU2 v2.0.0-rc.4.
# Load an older gcc before CUDA so nvcc picks it up as the host compiler.
module load gcc/11.4.0 2>/dev/null || module load gcc/12.3.0 2>/dev/null || \
    echo "WARNING: could not load older gcc module; build may fail with gcc 14"

# Default cudatoolkit conflicts with cuda module — swap it
module unload cudatoolkit 2>/dev/null || true
module load cuda/12.9
module load cmake/3.31.8 2>/dev/null || module load cmake 2>/dev/null || true

# HPC SDK splits math libs (cufft, cublas, etc.) from the CUDA runtime
CUDA_MATH_LIBS="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/REDIST/math_libs/12.9/targets/x86_64-linux/lib"
CUDA_MATH_INC="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/REDIST/math_libs/12.9/targets/x86_64-linux/include"
if [[ -d "${CUDA_MATH_LIBS}" ]]; then
    export LIBRARY_PATH="${CUDA_MATH_LIBS}:${LIBRARY_PATH}"
    export LD_LIBRARY_PATH="${CUDA_MATH_LIBS}:${LD_LIBRARY_PATH}"
    echo "CUDA math libs: ${CUDA_MATH_LIBS}"
fi
if [[ -d "${CUDA_MATH_INC}" ]]; then
    export CPATH="${CUDA_MATH_INC}:${CPATH}"
fi

echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "cmake: $(cmake --version 2>/dev/null | head -1 || echo 'not found')"
echo "gcc: $(gcc --version 2>/dev/null | head -1 || echo 'not found')"

# ── Pre-stage dependencies (CMake FetchContent fails on Lustre filesystem) ──
EXT="${REPO_DIR}/PDAC/sim/external"
mkdir -p "${EXT}"

# Boost (tar download)
BOOST_VER="1.82.0"
BOOST_DIR="${EXT}/boost-${BOOST_VER}"
if [[ ! -f "${BOOST_DIR}/CMakeLists.txt" ]]; then
    echo ""
    echo "=== Pre-staging Boost ${BOOST_VER} ==="
    BOOST_TAR="${EXT}/boost-${BOOST_VER}.tar.gz"
    [[ -f "${BOOST_TAR}" ]] || curl -L -o "${BOOST_TAR}" \
        "https://github.com/boostorg/boost/releases/download/boost-${BOOST_VER}/boost-${BOOST_VER}.tar.gz"
    cd "${EXT}" && tar xzf "${BOOST_TAR}" && cd "${SIM_DIR}"
    echo "  Boost ready: ${BOOST_DIR}"
fi
export BOOST_ROOT="${BOOST_DIR}"

# SUNDIALS (git clone)
SUNDIALS_DIR="${EXT}/sundials"
if [[ ! -f "${SUNDIALS_DIR}/CMakeLists.txt" ]]; then
    echo ""
    echo "=== Pre-staging SUNDIALS v7.6.0 ==="
    git clone --depth 1 --branch v7.6.0 https://github.com/LLNL/sundials.git "${SUNDIALS_DIR}"
    echo "  SUNDIALS ready: ${SUNDIALS_DIR}"
fi
export SUNDIALS_DIR="${SUNDIALS_DIR}"

# FLAMEGPU2 (git clone)
FLAMEGPU_DIR="${EXT}/flamegpu2"
if [[ ! -f "${FLAMEGPU_DIR}/CMakeLists.txt" ]]; then
    echo ""
    echo "=== Pre-staging FLAMEGPU2 v2.0.0-rc.4 ==="
    git clone --depth 1 --branch v2.0.0-rc.4 https://github.com/FLAMEGPU/FLAMEGPU2.git "${FLAMEGPU_DIR}"
    echo "  FLAMEGPU ready: ${FLAMEGPU_DIR}"
fi

# yaml-cpp (git clone)
YAMLCPP_DIR="${EXT}/yaml-cpp"
if [[ ! -f "${YAMLCPP_DIR}/CMakeLists.txt" ]]; then
    echo ""
    echo "=== Pre-staging yaml-cpp v0.8.0 ==="
    git clone --depth 1 --branch 0.8.0 https://github.com/jbeder/yaml-cpp.git "${YAMLCPP_DIR}"
    echo "  yaml-cpp ready: ${YAMLCPP_DIR}"
fi

# Tell CMake to use pre-staged sources instead of fetching
# FETCHCONTENT_SOURCE_DIR_<name> bypasses FetchContent download entirely
export CMAKE_ARGS_EXTRA="\
  -DFETCHCONTENT_SOURCE_DIR_SUNDIALS=${SUNDIALS_DIR} \
  -DFETCHCONTENT_SOURCE_DIR_YAML-CPP=${YAMLCPP_DIR} \
  -DFETCHCONTENT_SOURCE_DIR_BOOST=${BOOST_DIR} \
  -DCMAKE_LIBRARY_PATH=${CUDA_MATH_LIBS}"

# Clean stale build cache (CMake config changes require fresh configure)
rm -rf "${SIM_DIR}/build"

# Build with pre-staged deps (no network needed after first run)
./build.sh --cuda-arch 80 --flamegpu "${FLAMEGPU_DIR}" -j 4

echo "=== Build finished: $(date) ==="
