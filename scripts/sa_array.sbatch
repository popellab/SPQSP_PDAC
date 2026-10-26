#!/bin/bash
#SBATCH --job-name=pdac_sa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=sa_logs/sa_%A_%a.out
#SBATCH --error=sa_logs/sa_%A_%a.err

# ============================================================================
# SLURM array job for ABM Morris sensitivity analysis.
#
# Each array task runs one shard (= one parameter set × one replicate).
# The SA directory (--sa-dir) must already contain the DOE files from
# `run_abm_sa.py generate`.
#
# Usage:
#   export SA_DIR=/path/to/sa_morris
#   sbatch --array=0-5639 scripts/sa_array.sbatch
#
# Or via sa_submit.sh which computes the array range automatically.
# ============================================================================

set -e

# SLURM copies scripts to a spool dir, so BASH_SOURCE won't work.
if [[ -n "${SLURM_SUBMIT_DIR}" ]]; then
    if [[ -d "${SLURM_SUBMIT_DIR}/scripts" ]]; then
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    elif [[ -d "${SLURM_SUBMIT_DIR}/../scripts" ]]; then
        REPO_DIR="$(cd "${SLURM_SUBMIT_DIR}/.." && pwd)"
    else
        REPO_DIR="${SLURM_SUBMIT_DIR}"
    fi
else
    SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    REPO_DIR="$(dirname "${SCRIPTS_DIR}")"
fi

# ── Cluster environment (mirrors submit.sh) ──
detect_cluster() {
    local host
    host=$(hostname -f 2>/dev/null || hostname)
    case "${host}" in
        *delta*|*dt-login*|*dt-gpu*|*ncsa*)  echo "delta" ;;
        *anvil*|*bell*)                       echo "anvil" ;;
        *)                                    echo "unknown" ;;
    esac
}

CLUSTER="${CLUSTER:-$(detect_cluster)}"
case "${CLUSTER}" in
    delta)
        module unload cudatoolkit 2>/dev/null || true
        module load cuda/12.9
        _MATH="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/REDIST/math_libs/12.9/targets/x86_64-linux/lib"
        [[ -d "${_MATH}" ]] && export LD_LIBRARY_PATH="${_MATH}:${LD_LIBRARY_PATH}"
        module load cmake/3.31.8 2>/dev/null || true
        ;;
    anvil)
        module purge
        module load modtree/gpu
        module load gcc/11.2.0 cuda/12.8.0
        ;;
esac

# ── Required env vars ──
SA_DIR="${SA_DIR:?Set SA_DIR to the Morris SA directory}"
BASE_SEED="${BASE_SEED:-1000}"
TIMEOUT="${TIMEOUT:-7200}"

SHARD_ID="${SLURM_ARRAY_TASK_ID:?Must run as SLURM array job}"

echo "================================================"
echo "SA Shard: ${SHARD_ID}"
echo "  SA dir:    ${SA_DIR}"
echo "  Mode:      placement + qsp-state → volume-stopped presim"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo N/A)"
echo "================================================"

# ── Activate Python environment ──
if [[ -d "${REPO_DIR}/.venv" ]]; then
    source "${REPO_DIR}/.venv/bin/activate"
elif [[ -n "${CONDA_DEFAULT_ENV}" ]]; then
    : # already in conda
else
    echo "WARNING: No .venv found; using system Python"
fi

cd "${REPO_DIR}"

python -I scripts/run_abm_sa.py run-shard \
    --sa-dir "${SA_DIR}" \
    --shard-id "${SHARD_ID}" \
    --base-seed "${BASE_SEED}" \
    --timeout "${TIMEOUT}"

echo "Shard ${SHARD_ID} complete: $(date)"
