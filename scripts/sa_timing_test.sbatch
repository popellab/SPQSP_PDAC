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

# Don't set -e: the ABM may crash (that's what we're diagnosing), and we
# still want the post-crash diagnostics (GPU mem, timing partial) to run.

# SLURM requires the output directory to exist BEFORE the job starts.
# Create it from the submit dir (this line runs on the compute node).
mkdir -p "${SLURM_SUBMIT_DIR}/sa_logs" 2>/dev/null || true

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
BASELINE_DIR="${REPO_DIR}/results/sa_baseline"
ABM_BIN="${REPO_DIR}/PDAC/sim/build/bin/pdac"

# ── Cluster environment ──
module unload cudatoolkit 2>/dev/null || true
module load cuda/12.9
_MATH="/opt/nvidia/hpc_sdk/Linux_x86_64/26.5/REDIST/math_libs/12.9/targets/x86_64-linux/lib"
[[ -d "${_MATH}" ]] && export LD_LIBRARY_PATH="${_MATH}:${LD_LIBRARY_PATH}"
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

# Debug env vars (uncomment to diagnose CUDA crashes):
# export CUDA_LAUNCH_BLOCKING=1    # Synchronous kernel launches
# export PDAC_CUDA_LAYER_CHECK=1   # Per-layer sync + error check

STEPS=10

echo ""
echo "=== Running ${STEPS} ABM steps at full resolution ==="
echo "  Run dir: ${RUN_DIR}"
echo "  Grid: 3077 × 3077 × 1"
echo "  Steps: ${STEPS} (presim + main)"
echo ""

# Optional: run under compute-sanitizer for exact memcheck (VERY slow — ~100x).
# Set USE_SANITIZER=1 when submitting:  USE_SANITIZER=1 sbatch ...
ABM_CMD="${ABM_BIN}"
if [[ "${USE_SANITIZER:-0}" == "1" ]]; then
    ABM_CMD="compute-sanitizer --tool memcheck --show-backtrace yes ${ABM_BIN}"
    echo "  *** compute-sanitizer enabled (will be ~100x slower) ***"
fi

# Run ABM from placement file with timing output
# -G 0: no grid output (we just want timing)
# --placement: load from placement file instead of initialization
${ABM_CMD} \
    -p "${BASELINE_DIR}/param_all_median.xml" \
    -gx 3077 -gy 3077 -gz 1 \
    -s "${STEPS}" \
    -G 0 \
    --placement "${BASELINE_DIR}/tumor.plc" \
    --qsp-state "${BASELINE_DIR}/qsp_warmup.bin" \
    --output-root "${RUN_DIR}"
ABM_EXIT=$?
echo "ABM exit code: ${ABM_EXIT}"

echo ""
echo "=== Timing results ==="

# Find the timing CSV (named timing_<seed>.csv or layer_timing.csv at output root)
TIMING_CSV=$(find "${RUN_DIR}" -maxdepth 1 -name 'timing_*.csv' -o -name 'timing.csv' | head -1)
LAYER_CSV=$(find "${RUN_DIR}" -maxdepth 1 -name 'layer_timing.csv' | head -1)

if [[ -n "${TIMING_CSV}" ]]; then
    echo "Per-step timing: ${TIMING_CSV}"
    head -5 "${TIMING_CSV}"
    echo "..."
    # Compute average step time
    python3 -c "
import csv, sys
with open('${TIMING_CSV}') as f:
    rows = list(csv.DictReader(f))
if rows:
    # Try common column names for total step time
    for key in ('total', 'step_time', 'wall_s'):
        totals = [float(r[key]) for r in rows if key in r and float(r[key]) > 0]
        if totals: break
    if totals:
        print(f'  Steps measured: {len(totals)}')
        print(f'  Mean step time: {sum(totals)/len(totals):.2f} s')
        print(f'  Min step time:  {min(totals):.2f} s')
        print(f'  Max step time:  {max(totals):.2f} s')
        print(f'  Est. 50 steps:  {50*sum(totals)/len(totals)/60:.1f} min')
        print(f'  Est. 200 steps: {200*sum(totals)/len(totals)/60:.1f} min')
    else:
        print(f'  (columns: {list(rows[0].keys())})')
" 2>/dev/null || echo "(could not parse timing)"
else
    echo "WARNING: timing CSV not found in ${RUN_DIR}"
    ls -la "${RUN_DIR}"/*.csv 2>/dev/null || echo "  (no CSV files)"
fi

if [[ -n "${LAYER_CSV}" ]]; then
    echo ""
    echo "Layer timing: ${LAYER_CSV}"
    head -3 "${LAYER_CSV}"
    echo "..."
fi

echo ""
echo "=== GPU memory usage ==="
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null || true

echo ""
echo "=== Done: $(date) ==="
