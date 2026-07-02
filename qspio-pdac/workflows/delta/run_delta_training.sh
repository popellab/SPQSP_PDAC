#!/bin/bash
# Upload training artifacts to Delta, sbatch GPU training job, poll, download results.
#
# Usage:  workflows/delta/run_delta_training.sh [tag]
# Expects cwd = pdac-build repo root.
set -euo pipefail

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
REMOTE="delta"
# ARTIFACTS / LOCAL_OUT default to the single-run layout; sweep driver can
# override them per-cell (via env vars) to avoid races.
ARTIFACTS="${ARTIFACTS:-figures/sbi_multi/artifacts}"
LOCAL_OUT="${LOCAL_OUT:-figures/sbi_multi/delta_out/${TAG}}"
REMOTE_BASE="sbi_runs/${TAG}"

for f in theta_train.pt x_train.pt npe_config.json; do
  [ -f "$ARTIFACTS/$f" ] || { echo "missing: $ARTIFACTS/$f" >&2; exit 1; }
done

echo "[delta] uploading -> $REMOTE:$REMOTE_BASE"
ssh "$REMOTE" "mkdir -p $REMOTE_BASE"
scp -q "$ARTIFACTS/theta_train.pt" "$ARTIFACTS/x_train.pt" "$ARTIFACTS/npe_config.json" \
    workflows/delta/train_npe_delta.py workflows/delta/train_npe.sbatch \
    "$REMOTE:$REMOTE_BASE/"

echo "[delta] sbatch"
JID=$(ssh "$REMOTE" "cd ~/$REMOTE_BASE && RUN_DIR=\$PWD sbatch --parsable train_npe.sbatch")
echo "[delta] job id: $JID"

echo "[delta] polling (Ctrl-C to detach; job keeps running)"
while :; do
  state=$(ssh "$REMOTE" "squeue -h -j $JID -o '%T' 2>/dev/null" || true)
  state="${state// /}"
  [ -z "$state" ] && { echo "[$(date +%H:%M:%S)] job left queue"; break; }
  echo "[$(date +%H:%M:%S)] $state"
  sleep 20
done

mkdir -p "$LOCAL_OUT"
echo "[delta] downloading -> $LOCAL_OUT"
scp -q "$REMOTE:$REMOTE_BASE/out/density_estimator.pt" \
       "$REMOTE:$REMOTE_BASE/out/learning_curve.csv" \
       "$REMOTE:$REMOTE_BASE/out/metrics.json" \
       "$REMOTE:$REMOTE_BASE/sbi_train-${JID}.out" \
       "$LOCAL_OUT/" || echo "[delta] some files missing (check sbi_train-${JID}.out on delta)"

if [ -f "$LOCAL_OUT/metrics.json" ]; then
  echo "---"; cat "$LOCAL_OUT/metrics.json"
fi
