#!/bin/bash
# Upload per-round training tensors + manifest to Delta, sbatch the sequential
# SNPE-C trainer, poll, download results.
#
# Usage:  workflows/delta/run_delta_sequential_training.sh [tag]
#
# Mirrors run_delta_training.sh's contract: ARTIFACTS / LOCAL_OUT can be
# overridden in env so concurrent rounds (or independent runs) don't race
# on shared paths. ARTIFACTS must contain manifest.json + theta_rK.pt /
# x_rK.pt for every round listed in the manifest.
set -euo pipefail

TAG="${1:-$(date +%Y%m%d_%H%M%S)}"
REMOTE="delta"
ARTIFACTS="${ARTIFACTS:-figures/sbi_sequential/staging}"
LOCAL_OUT="${LOCAL_OUT:-figures/sbi_sequential/delta_out/${TAG}}"
REMOTE_BASE="sbi_runs/${TAG}"

[ -f "$ARTIFACTS/manifest.json" ] || { echo "missing: $ARTIFACTS/manifest.json" >&2; exit 1; }
[ -f "$ARTIFACTS/npe_config.json" ] || { echo "missing: $ARTIFACTS/npe_config.json" >&2; exit 1; }

# Verify every round's tensors are present locally before any SSH.
ROUND_FILES=$(/usr/bin/python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
for rd in m["rounds"]:
    print(rd["theta_file"]); print(rd["x_file"])
' "$ARTIFACTS/manifest.json")
for f in $ROUND_FILES; do
  [ -f "$ARTIFACTS/$f" ] || { echo "missing round tensor: $ARTIFACTS/$f" >&2; exit 1; }
done

echo "[delta] uploading -> $REMOTE:$REMOTE_BASE"
ssh "$REMOTE" "mkdir -p $REMOTE_BASE"

# Upload manifest, npe config, all round tensors, and the trainer.
SCP_ARGS=("$ARTIFACTS/manifest.json" "$ARTIFACTS/npe_config.json")
for f in $ROUND_FILES; do
  SCP_ARGS+=("$ARTIFACTS/$f")
done

# Warm-start mode: upload the pretrained estimator (and its metrics.json,
# if present) so the trainer can load it as initial weights instead of
# training round 1 from scratch. The trainer figures out warm-start from
# the manifest's "warm_start_from" key — no extra CLI flag needed.
WARM_FILE=$(/usr/bin/python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))
print(m.get("warm_start_from") or "")
' "$ARTIFACTS/manifest.json")
if [ -n "$WARM_FILE" ] && [ -f "$ARTIFACTS/$WARM_FILE" ]; then
  SCP_ARGS+=("$ARTIFACTS/$WARM_FILE")
  if [ -f "$ARTIFACTS/metrics.json" ]; then
    SCP_ARGS+=("$ARTIFACTS/metrics.json")
  fi
fi

scp -q "${SCP_ARGS[@]}" \
    workflows/delta/train_npe_sequential.py \
    workflows/delta/train_npe_sequential.sbatch \
    "$REMOTE:$REMOTE_BASE/"

echo "[delta] sbatch"
JID=$(ssh "$REMOTE" "cd ~/$REMOTE_BASE && RUN_DIR=\$PWD sbatch --parsable train_npe_sequential.sbatch")
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
# Density estimator + per-round learning curves + summary metrics. Per-round
# estimator snapshots ride along too (density_estimator_rK.pt) for any
# round-by-round diagnostic later.
scp -q "$REMOTE:$REMOTE_BASE/out/density_estimator.pt" \
       "$REMOTE:$REMOTE_BASE/out/metrics.json" \
       "$REMOTE:$REMOTE_BASE/sbi_train_seq-${JID}.out" \
       "$LOCAL_OUT/" || echo "[delta] some files missing (check sbi_train_seq-${JID}.out)"
scp -q "$REMOTE:$REMOTE_BASE/out/learning_curve_r*.csv" "$LOCAL_OUT/" || true
scp -q "$REMOTE:$REMOTE_BASE/out/density_estimator_r*.pt" "$LOCAL_OUT/" || true

if [ -f "$LOCAL_OUT/metrics.json" ]; then
  echo "---"; cat "$LOCAL_OUT/metrics.json"
fi
