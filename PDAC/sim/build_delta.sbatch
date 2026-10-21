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
# Submit from PDAC/sim/, or pass the path explicitly.
if [[ -f "${SLURM_SUBMIT_DIR}/build.sh" ]]; then
    SIM_DIR="${SLURM_SUBMIT_DIR}"
elif [[ -f "${SLURM_SUBMIT_DIR}/PDAC/sim/build.sh" ]]; then
    SIM_DIR="${SLURM_SUBMIT_DIR}/PDAC/sim"
else
    echo "ERROR: Can't find build.sh. Submit from PDAC/sim/ or repo root."
    exit 1
fi
cd "${SIM_DIR}"

echo "=== Build started: $(date) ==="
echo "Node: $(hostname)"
echo "Working dir: $(pwd)"

# Delta default env has gcc 13+ but CUDA libs (cufft etc.) need explicit module
# Default cudatoolkit conflicts with cuda module — swap it
module unload cudatoolkit 2>/dev/null || true
module load cuda/12.9
module load cmake/3.31.8 2>/dev/null || module load cmake 2>/dev/null || true

echo "CUDA: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "cmake: $(cmake --version 2>/dev/null | head -1 || echo 'not found')"
echo "gcc: $(gcc --version 2>/dev/null | head -1 || echo 'not found')"

# First-time build fetches FLAMEGPU2, Boost, SUNDIALS from GitHub (~8-15 min)
# Subsequent incremental builds are ~2 min
./build.sh --cuda-arch 80 -j 4

echo "=== Build finished: $(date) ==="
