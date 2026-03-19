#!/bin/bash
#SBATCH --job-name=pdac
#SBATCH --partition=gpu-debug
#SBATCH --gres=gpu:1
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
module purge 2>/dev/null
module load modtree/gpu 2>/dev/null
module load cmake gcc cuda boost 2>/dev/null

echo "================================================"
echo "PDAC Job: ${SLURM_JOB_ID}"
echo "  Node:       $(hostname)"
echo "  GPU:        $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "  nvcc:       $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo "================================================"

# --- Build if needed (CUDA only available on GPU nodes) ---
if [[ ! -x "${PDAC_BIN}" ]]; then
    echo ""
    echo "=== Binary not found — building on GPU node ==="
    cd "${PROJECT_DIR}"
    ./build.sh --cuda-arch 80
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
