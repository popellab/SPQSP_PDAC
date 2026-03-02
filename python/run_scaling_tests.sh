#!/bin/bash
# Scaling study: grid size vs. timing and GPU memory

BINARY="/home/chase/SPQSP/SPQSP_PDAC-main/PDAC/sim/build/bin/pdac"
OUTDIR="/home/chase/SPQSP/SPQSP_PDAC-main/python/outputs/scaling"
mkdir -p "$OUTDIR"

# Grid sizes to test (cubic), step counts
GRIDS=(32 64 96 128 160 192 256 320)
WARMUP_STEPS=3       # discard: CUDA JIT compilation happens here
TIMING_STEPS=20      # steps to time after warmup

# Ensure binary exists
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    echo "Please build with: cd PDAC/sim && ./build.sh"
    exit 1
fi

echo "Starting scaling tests..."
echo "Binary: $BINARY"
echo "Output dir: $OUTDIR"
echo ""

for g in "${GRIDS[@]}"; do
    echo "=== Testing ${g}³ grid (${TIMING_STEPS} steps + ${WARMUP_STEPS} warmup) ==="
    outdir="${OUTDIR}/grid_${g}"
    mkdir -p "$outdir"

    # Run with timing output, no agent/PDE CSV (I/O-free timing)
    # Use QSP-seeded initialization (-i 1) for consistency with calibration work
    $BINARY -g $g -s $((WARMUP_STEPS + TIMING_STEPS)) \
        -oa 0 -op 0 -i 1 \
        > "${outdir}/stdout.txt" 2>&1

    # Capture GPU memory lines
    grep "\[MEM\]" "${outdir}/stdout.txt" > "${outdir}/memory.txt" 2>&1 || true

    # Copy timing CSVs (produced by binary)
    if [ -f "outputs/timing.csv" ]; then
        cp outputs/timing.csv "${outdir}/timing.csv"
    fi
    if [ -f "outputs/init_timing.csv" ]; then
        cp outputs/init_timing.csv "${outdir}/init_timing.csv"
    fi

    # Print summary
    if [ -f "${outdir}/memory.txt" ]; then
        echo "  Memory:"
        tail -1 "${outdir}/memory.txt"
    fi
    if [ -f "${outdir}/timing.csv" ]; then
        echo "  Timing rows: $(wc -l < ${outdir}/timing.csv)"
    fi
    echo "  Output in ${outdir}"
    echo ""
done

echo "All tests complete."
echo "Analysis script: cd python && python analyze_scaling.py"
