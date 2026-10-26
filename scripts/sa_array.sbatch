#!/bin/bash
#SBATCH --job-name=pdac_sa
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=sa_logs/sa_%A_%a.out
#SBATCH --error=sa_logs/sa_%A_%a.err

# ============================================================================
# SLURM array job for ABM Morris sensitivity analysis.
#
# Each array task runs SHARDS_PER_JOB sequential shards to amortize queue
# wait time and CUDA warmup.  Completed shards are auto-skipped by
# run_abm_sa.py (checks for existing .npz).
#
# Environment variables (set by sa_submit.sh or export before sbatch):
#   SA_DIR            — path to Morris SA directory (required)
#   SHARDS_PER_JOB    — shards per array task (default: 10)
#   N_TOTAL           — total shard count (required, for bounds check)
#   BASE_SEED         — RNG seed offset (default: 1000)
#   TIMEOUT           — per-shard timeout in seconds (default: 600)
#
# Usage:
#   export SA_DIR=/path/to/sa_morris N_TOTAL=5640 SHARDS_PER_JOB=10
#   sbatch --array=0-563 scripts/sa_array.sbatch
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
N_TOTAL="${N_TOTAL:?Set N_TOTAL to total shard count}"
SHARDS_PER_JOB="${SHARDS_PER_JOB:-10}"
BASE_SEED="${BASE_SEED:-1000}"
TIMEOUT="${TIMEOUT:-600}"

TASK_ID="${SLURM_ARRAY_TASK_ID:?Must run as SLURM array job}"
FIRST_SHARD=$((TASK_ID * SHARDS_PER_JOB))
LAST_SHARD=$(( (TASK_ID + 1) * SHARDS_PER_JOB - 1 ))
if (( LAST_SHARD >= N_TOTAL )); then
    LAST_SHARD=$((N_TOTAL - 1))
fi

echo "================================================"
echo "SA Batch: task ${TASK_ID}, shards ${FIRST_SHARD}-${LAST_SHARD}"
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

FAILED=0
for SHARD_ID in $(seq "${FIRST_SHARD}" "${LAST_SHARD}"); do
    echo ""
    echo "--- Shard ${SHARD_ID} ($(date)) ---"
    python -I scripts/run_abm_sa.py run-shard \
        --sa-dir "${SA_DIR}" \
        --shard-id "${SHARD_ID}" \
        --base-seed "${BASE_SEED}" \
        --timeout "${TIMEOUT}" || {
            echo "WARNING: Shard ${SHARD_ID} failed (exit $?), continuing"
            FAILED=$((FAILED + 1))
        }
done

echo ""
echo "Batch ${TASK_ID} complete: shards ${FIRST_SHARD}-${LAST_SHARD}, ${FAILED} failures ($(date))"
