#!/bin/bash
#SBATCH --job-name=pdac_sa_timing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=sa_logs/timing_%j.out
#SBATCH --error=sa_logs/timing_%j.err

# ============================================================================
# Single-sample ABM timing test on Delta A100.
#
# Runs one ABM at the posterior median (no parameter perturbation) from the
# placement file for a few steps to measure wall-time per step at full
# 3077×3077 grid resolution.
#
# Usage:
#   sbatch --account=bgre-delta-gpu --partition=gpuA100x4 \
#       scripts/sa_timing_test.sbatch
# ============================================================================

set -e

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPTS_DIR}")"
BASELINE_DIR="${REPO_DIR}/results/sa_baseline"
ABM_BIN="${REPO_DIR}/PDAC/sim/build/bin/pdac"

# ── Cluster environment ──
module load cuda/12.9
module load cmake/3.31.8 2>/dev/null || true

echo "================================================"
echo "PDAC SA Timing Test"
echo "  Node:      $(hostname)"
echo "  GPU:       $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo N/A)"
echo "  Binary:    ${ABM_BIN}"
echo "  Baseline:  ${BASELINE_DIR}"
echo "================================================"

# ── Check binary ──
if [[ ! -x "${ABM_BIN}" ]]; then
    echo "ERROR: ABM binary not found at ${ABM_BIN}"
    echo "  Run build first: sbatch PDAC/sim/build_delta.sbatch"
    exit 1
fi

# ── Check baseline files ──
for f in param_all_median.xml tumor.plc qsp_warmup.bin; do
    if [[ ! -f "${BASELINE_DIR}/${f}" ]]; then
        echo "ERROR: ${f} not found in ${BASELINE_DIR}"
        exit 1
    fi
done

# ── Setup run directory ──
RUN_DIR="/tmp/${USER}/sa_timing_${SLURM_JOB_ID}"
mkdir -p "${RUN_DIR}"
cd "${RUN_DIR}"

echo ""
echo "=== Running 10 ABM steps at full resolution ==="
echo "  Run dir: ${RUN_DIR}"
echo "  Grid: 3077 × 3077 × 1"
echo "  Steps: 10"
echo ""

# Run ABM from placement file with timing output
# -s 10: only 10 steps (enough to measure per-step time)
# -G 0: no grid output (we just want timing)
# --placement: load from placement file instead of initialization
${ABM_BIN} \
    -p "${BASELINE_DIR}/param_all_median.xml" \
    -gx 3077 -gy 3077 -gz 1 \
    -s 10 \
    -G 0 \
    --placement "${BASELINE_DIR}/tumor.plc" \
    --qsp-state "${BASELINE_DIR}/qsp_warmup.bin" \
    --output-root "${RUN_DIR}"

echo ""
echo "=== Timing results ==="

# Print timing CSV if it exists
if [[ -f "${RUN_DIR}/outputs/timing.csv" ]]; then
    echo "Per-step timing (seconds):"
    cat "${RUN_DIR}/outputs/timing.csv"
    echo ""
    # Compute average step time
    python3 -c "
import csv
with open('${RUN_DIR}/outputs/timing.csv') as f:
    rows = list(csv.DictReader(f))
if rows:
    totals = [float(r.get('total', r.get('step_time', 0))) for r in rows if float(r.get('total', r.get('step_time', 0))) > 0]
    if totals:
        print(f'  Steps measured: {len(totals)}')
        print(f'  Mean step time: {sum(totals)/len(totals):.2f} s')
        print(f'  Min step time:  {min(totals):.2f} s')
        print(f'  Max step time:  {max(totals):.2f} s')
        print(f'  Est. 50 steps:  {50*sum(totals)/len(totals)/60:.1f} min')
        print(f'  Est. 200 steps: {200*sum(totals)/len(totals)/60:.1f} min')
" 2>/dev/null || echo "(could not parse timing)"
else
    echo "WARNING: timing.csv not found"
    echo "Check stdout above for any timing output."
fi

echo ""
echo "=== GPU memory usage ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

echo ""
echo "=== Done: $(date) ==="
