#!/bin/bash
# ============================================================================
# Orchestrate ABM Morris SA on cluster:
#   1. Generate DOE (if not already done)
#   2. Submit SLURM array job
#   3. Submit consolidation job (dependency: afterok)
#
# Usage:
#   # First run (generates DOE + submits):
#   ./scripts/sa_submit.sh \
#       --baseline-dir results/sa_baseline \
#       --sa-dir results/sa_morris \
#       --trajectories 10 --replicates 3
#
#   # Resume (skips DOE, submits remaining shards):
#   ./scripts/sa_submit.sh --sa-dir results/sa_morris
#
#   # Consolidate only (after all shards complete):
#   ./scripts/sa_submit.sh --sa-dir results/sa_morris --consolidate-only
# ============================================================================

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPTS_DIR}")"

# ── Defaults ──
BASELINE_DIR=""
SA_DIR=""
TRAJECTORIES=10
LEVELS=4
REPLICATES=3
SEED=42
PARTITION="gpuA100x4"
ACCOUNT=""
CONSOLIDATE_ONLY=false
MAX_CONCURRENT=200

# ── Parse args ──
while [[ $# -gt 0 ]]; do
    case "$1" in
        --baseline-dir)  BASELINE_DIR="$2"; shift 2 ;;
        --sa-dir)        SA_DIR="$2"; shift 2 ;;
        --trajectories)  TRAJECTORIES="$2"; shift 2 ;;
        --replicates)    REPLICATES="$2"; shift 2 ;;
        --levels)        LEVELS="$2"; shift 2 ;;
        --seed)          SEED="$2"; shift 2 ;;
        --partition)     PARTITION="$2"; shift 2 ;;
        --account)       ACCOUNT="$2"; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT="$2"; shift 2 ;;
        --consolidate-only) CONSOLIDATE_ONLY=true; shift ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "${SA_DIR}" ]]; then
    echo "ERROR: --sa-dir is required"
    exit 1
fi

# Load account from cluster.conf if not given
if [[ -z "${ACCOUNT}" && -f "${REPO_DIR}/PDAC/sim/cluster.conf" ]]; then
    source "${REPO_DIR}/PDAC/sim/cluster.conf"
fi
if [[ -z "${ACCOUNT}" ]]; then
    echo "ERROR: --account or cluster.conf ACCOUNT required"
    exit 1
fi

SA_DIR="$(cd "${REPO_DIR}" && realpath "${SA_DIR}")"
mkdir -p "${SA_DIR}"

cd "${REPO_DIR}"

# Activate python env
if [[ -d ".venv" ]]; then
    source .venv/bin/activate
fi

# ── Step 1: Generate DOE (if needed) ──
if [[ "${CONSOLIDATE_ONLY}" == "false" && ! -f "${SA_DIR}/meta.json" ]]; then
    if [[ -z "${BASELINE_DIR}" ]]; then
        echo "ERROR: --baseline-dir required for first-time DOE generation"
        exit 1
    fi
    BASELINE_DIR="$(realpath "${BASELINE_DIR}")"
    echo "=== Generating Morris DOE ==="
    python -I scripts/run_abm_sa.py generate \
        --baseline-dir "${BASELINE_DIR}" \
        --output-dir "${SA_DIR}" \
        --trajectories "${TRAJECTORIES}" \
        --levels "${LEVELS}" \
        --replicates "${REPLICATES}" \
        --seed "${SEED}"
    echo ""
fi

# ── Read meta ──
N_TOTAL=$(python -I -c "import json; m=json.load(open('${SA_DIR}/meta.json')); print(m['n_total_runs'])")
echo "Total shards: ${N_TOTAL}"

if [[ "${CONSOLIDATE_ONLY}" == "true" ]]; then
    echo "=== Consolidating ==="
    python -I scripts/run_abm_sa.py consolidate --sa-dir "${SA_DIR}"
    echo "=== Running analysis ==="
    python -I scripts/analyze_abm_sa.py --sa-dir "${SA_DIR}"
    echo "Done."
    exit 0
fi

# ── Step 2: Submit array job ──
ARRAY_MAX=$((N_TOTAL - 1))

mkdir -p sa_logs

echo "=== Submitting SA array job ==="
echo "  Array: 0-${ARRAY_MAX}"
echo "  Partition: ${PARTITION}"
echo "  Account: ${ACCOUNT}"
echo "  Mode: placement + qsp-state → volume-stopped presim"
echo "  Max concurrent: ${MAX_CONCURRENT}"

ARRAY_JOB_ID=$(sbatch \
    --parsable \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --array="0-${ARRAY_MAX}%${MAX_CONCURRENT}" \
    --export="SA_DIR=${SA_DIR}" \
    scripts/sa_array.sbatch)

echo "  Array job submitted: ${ARRAY_JOB_ID}"

# ── Step 3: Submit consolidation job (runs after array completes) ──
CONSOL_JOB_ID=$(sbatch \
    --parsable \
    --account="${ACCOUNT}" \
    --partition=cpu \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=16G --time=00:30:00 \
    --job-name=pdac_sa_consol \
    --output=sa_logs/consolidate_%j.out \
    --error=sa_logs/consolidate_%j.err \
    --wrap="cd ${REPO_DIR} && source .venv/bin/activate 2>/dev/null; \
            python -I scripts/run_abm_sa.py consolidate --sa-dir ${SA_DIR} && \
            python -I scripts/analyze_abm_sa.py --sa-dir ${SA_DIR}")

echo "  Consolidation job submitted: ${CONSOL_JOB_ID} (depends on ${ARRAY_JOB_ID})"

echo ""
echo "=== Submitted ==="
echo "  Array:         ${ARRAY_JOB_ID} (${N_TOTAL} shards, 0-${ARRAY_MAX})"
echo "  Consolidation: ${CONSOL_JOB_ID}"
echo "  Monitor: squeue -u \$USER | grep pdac_sa"
echo "  Results: ${SA_DIR}/results/"
