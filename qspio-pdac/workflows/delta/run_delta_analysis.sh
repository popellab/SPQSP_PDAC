#!/bin/bash
# Run full NPE analysis (steps 5/5b/6) on Delta GPU; return only summaries + plots.
#
# Usage:  workflows/delta/run_delta_analysis.sh <stage_dir> [tag]
#
# stage_dir must contain an "input/" subdir with:
#   density_estimator.pt, theta_test.pt, x_test.pt, obs_transformed_tensor.pt,
#   obs_transformed.npy, param_names.json, pdac_priors.csv, prior_var.npy,
#   config.json
#
# Downloads small artifacts (CSVs/PNGs/JSON) into <stage_dir>/output/.
set -euo pipefail

STAGE_DIR="${1:-}"
[ -n "$STAGE_DIR" ] && [ -d "$STAGE_DIR/input" ] || {
  echo "usage: $0 <stage_dir> [tag]" >&2
  echo "  (stage_dir/input/ must contain the bundled inputs)" >&2
  exit 1
}
TAG="${2:-$(date +%Y%m%d_%H%M%S)}"
REMOTE="delta"
REMOTE_BASE="sbi_runs/analyze_${TAG}"

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "[delta-analyze] uploading -> $REMOTE:$REMOTE_BASE"
ssh "$REMOTE" "mkdir -p $REMOTE_BASE/input $REMOTE_BASE/output"
rsync -az "$STAGE_DIR/input/" "$REMOTE:$REMOTE_BASE/input/"
scp -q "$HERE/analyze_npe_delta.py" "$HERE/analyze_npe.sbatch" "$REMOTE:$REMOTE_BASE/"

echo "[delta-analyze] sbatch"
JID=$(ssh "$REMOTE" "cd ~/$REMOTE_BASE && RUN_DIR=\$PWD sbatch --parsable analyze_npe.sbatch")
echo "[delta-analyze] job id: $JID"

echo "[delta-analyze] polling (Ctrl-C to detach; job keeps running)"
while :; do
  state=$(ssh "$REMOTE" "squeue -h -j $JID -o '%T' 2>/dev/null" || true)
  state="${state// /}"
  [ -z "$state" ] && { echo "[$(date +%H:%M:%S)] job left queue"; break; }
  echo "[$(date +%H:%M:%S)] $state"
  sleep 20
done

OUT_LOCAL="$STAGE_DIR/output"
mkdir -p "$OUT_LOCAL"
echo "[delta-analyze] downloading -> $OUT_LOCAL"
rsync -a "$REMOTE:$REMOTE_BASE/output/" "$OUT_LOCAL/"
scp -q "$REMOTE:$REMOTE_BASE/analyze-${JID}.out" "$OUT_LOCAL/" 2>/dev/null || true

if [ -f "$OUT_LOCAL/run_metrics.json" ]; then
  echo "[delta-analyze] done. Outputs:"
  ls -la "$OUT_LOCAL"
else
  echo "[delta-analyze] no run_metrics.json — check $OUT_LOCAL/analyze-${JID}.out" >&2
  exit 2
fi
