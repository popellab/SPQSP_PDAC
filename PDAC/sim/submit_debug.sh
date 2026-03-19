#!/bin/bash
#SBATCH --job-name=pdac
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=pdac_%j.out
#SBATCH --error=pdac_%j.err

# ============================================================================
# SPQSP PDAC — SLURM submission script for Anvil
#
# Source lives on /anvil/projects/, outputs go to /anvil/scratch/.
# Automatically builds if no binary exists (CUDA is only available on GPU nodes).
#
# Usage:
#   sbatch submit.sh                          # defaults: 500 steps, 50^3 grid
#   sbatch submit.sh -s 1000 -g 101           # custom run
#   SCRATCH_DIR=/custom/path sbatch submit.sh  # override scratch location
# ============================================================================

# --- Configuration ---
# BASH_SOURCE resolves to SLURM's spool copy, not the original file.
# Use SLURM_SUBMIT_DIR (the directory where sbatch was invoked) instead.
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
PDAC_BIN="${PROJECT_DIR}/build/bin/pdac"

# Scratch directory for outputs
SCRATCH_BASE="${SCRATCH_DIR:-/anvil/scratch/${USER}}"
RUN_DIR="${SCRATCH_BASE}/pdac_runs/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"

# Default simulation arguments (overridden by anything passed after sbatch submit.sh)
DEFAULT_ARGS="-s 50 -g 50 -oa 1 -op 1"

# --- Load modules ---
# Adjust these to match your cluster's module names
module purge
module load modtree/gpu
module load gcc/11.2.0 cuda/12.8.0

# Use local cmake (Anvil system cmake is too old for FLAME GPU 2)
LOCAL_CMAKE="${PROJECT_DIR}/external/cmake-3.28.3-linux-x86_64/bin"
if [[ -x "${LOCAL_CMAKE}/cmake" ]]; then
    export PATH="${LOCAL_CMAKE}:${PATH}"
fi

# Redirect nvcc temp files to scratch (shared /tmp may be too small)
export TMPDIR="${SCRATCH_BASE}/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

echo "  nvcc: $(nvcc --version 2>/dev/null | grep release)"
echo "  cmake: $(cmake --version | head -1)"
echo "  gcc: $(gcc --version | head -1)"
echo "  TMPDIR: ${TMPDIR}"

echo "================================================"
echo "PDAC Job: ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  nvcc:       $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "  cmake:      $(cmake --version 2>/dev/null | head -1 || echo 'not found')"
echo "================================================"

# --- Build if needed (CUDA only available on GPU nodes) ---
if [[ ! -x "${PDAC_BIN}" ]]; then
    echo ""
    echo "=== Binary not found — building on GPU node ==="
    cd "${PROJECT_DIR}"
    rm -rf build  # clear stale CMake cache
    EXT_DIR="${PROJECT_DIR}/external"
    echo "=== External deps check ==="
    echo "  FLAMEGPU: $(ls ${EXT_DIR}/flamegpu2/CMakeLists.txt 2>/dev/null && echo OK || echo MISSING)"
    echo "  SUNDIALS: $(ls ${EXT_DIR}/sundials/CMakeLists.txt 2>/dev/null && echo OK || echo MISSING)"
    echo "  BOOST:    $(ls ${EXT_DIR}/boost/CMakeLists.txt 2>/dev/null && echo OK || echo MISSING)"
    ls "${EXT_DIR}/" 2>/dev/null
    SUNDIALS_DIR="${EXT_DIR}/sundials" \
    BOOST_ROOT="${EXT_DIR}/boost" \
    ./build.sh --cuda-arch 80 --flamegpu "${EXT_DIR}/flamegpu2"
    echo ""
fi

# --- Setup scratch output directory ---
mkdir -p "${RUN_DIR}"
echo "================================================"
echo "  Binary:      ${PDAC_BIN}"
echo "  Working dir: ${RUN_DIR}"
echo "  Args:        ${@:-${DEFAULT_ARGS}}"
echo "================================================"

# --- Run from scratch directory ---
cd "${RUN_DIR}"

# Copy param file for reproducibility
cp "${PROJECT_DIR}/resource/param_all_test.xml" "${RUN_DIR}/param_snapshot.xml"

# Run simulation
${PDAC_BIN} ${@:-${DEFAULT_ARGS}}

echo ""
echo "================================================"
echo "Run complete. Outputs in: ${RUN_DIR}/outputs/"
echo "================================================"
