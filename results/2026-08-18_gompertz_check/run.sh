#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
BIN="$DIR/../../PDAC/sim/build/bin/pdac"
V6="$DIR/../2026-08-17_abm_stabilization_v6"

"$BIN" \
  --placement "$V6/seed.plc" \
  --qsp-onset-volume 1.7177e-05 \
  -p "$DIR/param_fitted.xml" \
  -gx 100 -gy 100 -gz 12 \
  -s 200 \
  --main-mode qsp_abm \
  -G 0 \
  -oi 1 \
  --seed 42 \
  --output-root "$DIR/outputs"
