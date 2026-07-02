#!/bin/bash
# NPE training-settings sweep driver.
#
# Runs each grid cell as:
#   1. stage  — sbi_runner --stage-artifacts-only (populates sweep/stage/$tag/artifacts)
#   2. train  — run_delta_training.sh $tag (SLURM on Delta, background)
#   3. analyze — sbi_runner --pretrained-dir delta_out/$tag --sample-on-delta
#
# Steps 2 and 3 run in background per cell. Delta's SLURM queue handles
# concurrency; ControlMaster keeps SSH warm (one Duo auth). On completion,
# invokes collate_sweep.py to merge per-cell local_calibration.csv's.
#
# Usage:  workflows/sweep/run_sweep.sh [--dry-run]
#   Env vars:
#     CONFIGS_DIR     (default: workflows/sweep/configs)
#     SWEEP_ROOT      (default: figures/sbi_multi/sweep)
#     NUM_TOTAL_SIMULATIONS  (default: 100000)
#     CPP_BRANCH      (default: cpp-sweep-binary-io)
#     TOOLS_BRANCH    (default: cpp-backend-v1)
#     CPP_JOBS_PER_CHUNK     (default: 400)
set -euo pipefail

CONFIGS_DIR="${CONFIGS_DIR:-workflows/sweep/configs}"
SWEEP_ROOT="${SWEEP_ROOT:-figures/sbi_multi/sweep}"
NUM_TOTAL_SIMULATIONS="${NUM_TOTAL_SIMULATIONS:-100000}"
CPP_BRANCH="${CPP_BRANCH:-cpp-sweep-binary-io}"
TOOLS_BRANCH="${TOOLS_BRANCH:-cpp-backend-v1}"
CPP_JOBS_PER_CHUNK="${CPP_JOBS_PER_CHUNK:-400}"
# Restriction classifier (rejection-sample low-validity thetas before sim).
# Leave RESTRICT_DIR empty to run against the unrestricted pool.
RESTRICT_DIR="${RESTRICT_DIR:-cache/restriction_classifier}"
RESTRICT_THRESHOLD="${RESTRICT_THRESHOLD:-0.5}"
# Max concurrent ssh multiplexed sessions. OpenSSH server's MaxSessions
# default is ~10; each run_delta_*.sh makes several ssh calls (upload,
# sbatch, poll, download), so keep this well below 10.
MAX_PARALLEL="${MAX_PARALLEL:-6}"
# Optional: restrict to a space-separated subset of tags (e.g. to re-run
# only cells that failed previously). Empty = all tags in _tags.txt.
ONLY_TAGS="${ONLY_TAGS:-}"

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

PY="${PY:-.venv/bin/python}"
if [ ! -x "$PY" ]; then
  echo "Python not found at $PY — set PY=/path/to/python (e.g. sibling-worktree .venv)." >&2
  exit 1
fi
TAGS_FILE="$CONFIGS_DIR/_tags.txt"
if [ ! -f "$TAGS_FILE" ]; then
  echo "Missing $TAGS_FILE — run workflows/sweep/generate_configs.py first." >&2
  exit 1
fi

TAGS=()
while IFS= read -r line; do
  [ -n "$line" ] && TAGS+=("$line")
done < "$TAGS_FILE"
if [ -n "$ONLY_TAGS" ]; then
  # Filter to the ONLY_TAGS subset (preserve original order).
  FILTERED=()
  for tag in "${TAGS[@]}"; do
    for keep in $ONLY_TAGS; do
      if [ "$tag" = "$keep" ]; then FILTERED+=("$tag"); break; fi
    done
  done
  TAGS=("${FILTERED[@]}")
  echo "[sweep] ONLY_TAGS filter: ${#TAGS[@]} cells kept"
fi
echo "[sweep] ${#TAGS[@]} cells:" "${TAGS[@]}"

# Shared staging root per cell. sbi_runner writes
#   $SWEEP_ROOT/stage/$tag/artifacts/{theta_train.pt, x_train.pt, npe_config.json, ...}
# which we then feed to run_delta_training.sh via ARTIFACTS env var.
mkdir -p "$SWEEP_ROOT/stage" "$SWEEP_ROOT/delta_out" "$SWEEP_ROOT/runs"

extract () {
  # extract <config.json> <key> — minimal jq-free JSON field reader
  "$PY" -c "import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])" "$1" "$2"
}

# ==== 1. STAGE ARTIFACTS (serial — step 2 shares cache, safe & fast) ====
echo ""
echo "[sweep] staging artifacts for ${#TAGS[@]} cells..."
for tag in "${TAGS[@]}"; do
  cfg="$CONFIGS_DIR/$tag.json"
  hf=$(extract "$cfg" hidden_features)
  nt=$(extract "$cfg" num_transforms)
  sub=$(extract "$cfg" n_train_subsample)
  stage_dir="$SWEEP_ROOT/stage/$tag"
  # Build args as an array so the shell doesn't split/reparse a multi-line
  # string (which was silently dropping all but the first argv element).
  args=(
    -u workflows/sbi_runner.py
    --cpp-jobs-per-chunk "$CPP_JOBS_PER_CHUNK"
    --num-total-simulations "$NUM_TOTAL_SIMULATIONS"
    --cpp-branch "$CPP_BRANCH" --tools-branch "$TOOLS_BRANCH"
    --hidden-features "$hf" --num-transforms "$nt"
    --n-train-subsample "$sub"
    --cache-processed-tensors --stage-artifacts-only
    --save-dir "$stage_dir"
  )
  if [ -n "$RESTRICT_DIR" ]; then
    args+=(--restriction-classifier-dir "$RESTRICT_DIR" --restriction-threshold "$RESTRICT_THRESHOLD")
  fi
  echo "[sweep][stage] $tag  (hf=$hf nt=$nt sub=$sub)"
  if [ "$DRY_RUN" = 1 ]; then echo "  $PY ${args[*]}"; continue; fi
  "$PY" "${args[@]}"

  # Override batch_size / max_epochs / num_bins in the staged npe_config.json
  # with the grid-level settings (sbi_runner's defaults are the single-run
  # values). We already wrote hidden_features / num_transforms via CLI flags.
  "$PY" -c "
import json, pathlib, sys
cfg_path = pathlib.Path('$cfg')
stage_cfg = pathlib.Path('$stage_dir/artifacts/npe_config.json')
grid = json.loads(cfg_path.read_text())
current = json.loads(stage_cfg.read_text())
for k in ('num_bins','batch_size','max_epochs'):
    current[k] = grid[k]
stage_cfg.write_text(json.dumps(current))
print(f'  Staged {stage_cfg}: {current}')
"
done

if [ "$DRY_RUN" = 1 ]; then
  echo "[sweep] dry run — stopping before training."
  exit 0
fi

# ==== 2. TRAIN (parallel on Delta) ====
echo ""
echo "[sweep] kicking off Delta training jobs (up to $MAX_PARALLEL parallel)..."
declare -a TRAIN_PIDS=()
mkdir -p "$SWEEP_ROOT/logs"
for tag in "${TAGS[@]}"; do
  # Semaphore: block until fewer than MAX_PARALLEL children are running.
  while [ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 2; done
  stage_dir="$SWEEP_ROOT/stage/$tag"
  delta_out="$SWEEP_ROOT/delta_out/$tag"
  log="$SWEEP_ROOT/logs/train_$tag.log"
  echo "[sweep][train] $tag -> log=$log"
  ARTIFACTS="$stage_dir/artifacts" LOCAL_OUT="$delta_out" \
    workflows/delta/run_delta_training.sh "$tag" >"$log" 2>&1 &
  TRAIN_PIDS+=($!)
done
echo "[sweep] waiting for ${#TRAIN_PIDS[@]} training jobs..."
FAIL_TRAIN=0
for pid in "${TRAIN_PIDS[@]}"; do
  if ! wait "$pid"; then FAIL_TRAIN=$((FAIL_TRAIN+1)); fi
done
if [ "$FAIL_TRAIN" -gt 0 ]; then
  echo "[sweep] WARNING: $FAIL_TRAIN training jobs failed — check $SWEEP_ROOT/logs/" >&2
fi

# ==== 3. ANALYZE (parallel on Delta + local) ====
echo ""
echo "[sweep] kicking off analysis jobs (up to $MAX_PARALLEL parallel)..."
declare -a ANALYZE_PIDS=()
for tag in "${TAGS[@]}"; do
  while [ "$(jobs -r -p | wc -l)" -ge "$MAX_PARALLEL" ]; do sleep 2; done
  cfg="$CONFIGS_DIR/$tag.json"
  hf=$(extract "$cfg" hidden_features)
  nt=$(extract "$cfg" num_transforms)
  sub=$(extract "$cfg" n_train_subsample)
  run_dir="$SWEEP_ROOT/runs/$tag"
  delta_out="$SWEEP_ROOT/delta_out/$tag"
  log="$SWEEP_ROOT/logs/analyze_$tag.log"
  if [ ! -f "$delta_out/density_estimator.pt" ]; then
    echo "[sweep][analyze] $tag  SKIP (no density_estimator.pt — training failed)"
    continue
  fi
  echo "[sweep][analyze] $tag -> log=$log"
  analyze_args=(
    -u workflows/sbi_runner.py
    --cpp-jobs-per-chunk "$CPP_JOBS_PER_CHUNK"
    --num-total-simulations "$NUM_TOTAL_SIMULATIONS"
    --cpp-branch "$CPP_BRANCH" --tools-branch "$TOOLS_BRANCH"
    --hidden-features "$hf" --num-transforms "$nt"
    --n-train-subsample "$sub"
    --cache-processed-tensors
    --pretrained-dir "$delta_out" --sample-on-delta
    --save-dir "$run_dir"
  )
  if [ -n "$RESTRICT_DIR" ]; then
    analyze_args+=(--restriction-classifier-dir "$RESTRICT_DIR" --restriction-threshold "$RESTRICT_THRESHOLD")
  fi
  ( "$PY" "${analyze_args[@]}" ) >"$log" 2>&1 &
  ANALYZE_PIDS+=($!)
done
echo "[sweep] waiting for ${#ANALYZE_PIDS[@]} analysis jobs..."
FAIL_ANALYZE=0
for pid in "${ANALYZE_PIDS[@]}"; do
  if ! wait "$pid"; then FAIL_ANALYZE=$((FAIL_ANALYZE+1)); fi
done
if [ "$FAIL_ANALYZE" -gt 0 ]; then
  echo "[sweep] WARNING: $FAIL_ANALYZE analysis jobs failed — check $SWEEP_ROOT/logs/" >&2
fi

# ==== 4. COLLATE ====
echo ""
echo "[sweep] collating results..."
$PY workflows/sweep/collate_sweep.py --sweep-root "$SWEEP_ROOT" --configs-dir "$CONFIGS_DIR"
echo "[sweep] done. Outputs in $SWEEP_ROOT/"
