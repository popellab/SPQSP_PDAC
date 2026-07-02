#!/bin/bash
# Upload a trained estimator + sampling jobs to Delta, sbatch on GPU, poll, download.
#
# Usage:  workflows/delta/run_delta_sampling.sh <stage_dir> [tag]
#
# stage_dir MUST contain:
#   density_estimator.pt
#   jobs.json                   (manifest)
#   <x files referenced by jobs.json>
#
# Downloads into <stage_dir>/out/ (one {name}_samples.pt per job entry).
set -euo pipefail

STAGE_DIR="${1:-}"
[ -n "$STAGE_DIR" ] && [ -d "$STAGE_DIR" ] || { echo "usage: $0 <stage_dir> [tag]" >&2; exit 1; }
TAG="${2:-$(date +%Y%m%d_%H%M%S)}"
REMOTE="delta"
REMOTE_BASE="sbi_runs/sample_${TAG}"

for f in density_estimator.pt jobs.json; do
  [ -f "$STAGE_DIR/$f" ] || { echo "missing: $STAGE_DIR/$f" >&2; exit 1; }
done

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[delta-sample] uploading -> $REMOTE:$REMOTE_BASE"
ssh "$REMOTE" "mkdir -p $REMOTE_BASE"
# Upload estimator, manifest, and every .pt referenced in the staging dir (x files)
scp -q "$STAGE_DIR"/*.pt "$STAGE_DIR/jobs.json" \
    "$HERE/sample_npe_delta.py" "$HERE/sample_npe.sbatch" \
    "$REMOTE:$REMOTE_BASE/"

echo "[delta-sample] sbatch"
JID=$(ssh "$REMOTE" "cd ~/$REMOTE_BASE && RUN_DIR=\$PWD sbatch --parsable sample_npe.sbatch")
echo "[delta-sample] job id: $JID"

echo "[delta-sample] polling (Ctrl-C to detach; job keeps running)"
while :; do
  state=$(ssh "$REMOTE" "squeue -h -j $JID -o '%T' 2>/dev/null" || true)
  state="${state// /}"
  [ -z "$state" ] && { echo "[$(date +%H:%M:%S)] job left queue"; break; }
  echo "[$(date +%H:%M:%S)] $state"
  sleep 15
done

OUT_LOCAL="$STAGE_DIR/out"
mkdir -p "$OUT_LOCAL"
echo "[delta-sample] downloading -> $OUT_LOCAL"
rsync -a "$REMOTE:$REMOTE_BASE/out/" "$OUT_LOCAL/"
scp -q "$REMOTE:$REMOTE_BASE/sample-${JID}.out" "$OUT_LOCAL/" 2>/dev/null || true

if ls "$OUT_LOCAL"/*_samples.pt >/dev/null 2>&1; then
  echo "[delta-sample] done. Artifacts:"
  ls -la "$OUT_LOCAL"/*_samples.pt
else
  echo "[delta-sample] no sample files downloaded — check $OUT_LOCAL/sample-${JID}.out" >&2
  exit 2
fi
