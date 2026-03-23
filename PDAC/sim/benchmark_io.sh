#!/bin/bash
# ============================================================================
# benchmark_io.sh — A/B comparison of simulation I/O performance
#
# Modes:
#   ./benchmark_io.sh                  Run current binary only (3 configs)
#   ./benchmark_io.sh --ab [REF]       Build & run both old (REF) and new (working tree)
#
# The --ab mode:
#   1. Creates a git worktree at the REF commit (default: HEAD, i.e., last commit)
#   2. Builds the old binary in that worktree
#   3. Builds the new binary in the current working tree
#   4. Runs both binaries through identical test configs
#   5. Compares results side-by-side
#
# Test configurations per binary:
#   - no_io:          ABM+PDE output disabled (pure compute baseline)
#   - io_every_step:  Full output every step (worst case I/O)
#   - io_interval_N:  Full output every N steps (typical use)
#
# Usage:
#   ./benchmark_io.sh [options]
#
# Options:
#   --ab [REF]        A/B mode: compare REF (default HEAD) vs working tree
#   --binary PATH     Skip build, use this binary (single-binary mode only)
#   --param-file PATH Path to parameter XML
#   --steps N         Simulation steps (default: 50)
#   --interval N      Output interval for interval test (default: 5)
#   --seed N          Random seed (default: 12345)
#   --skip-build      In --ab mode, skip building (use existing binaries)
#   --cuda-arch N     CUDA architecture for builds (e.g., 80 for A100)
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PARAM_FILE="${SCRIPT_DIR}/resource/param_all_test.xml"
STEPS=50
INTERVAL=5
SEED=12345
REPS=1
AB_MODE=false
AB_REF="HEAD"
SKIP_BUILD=false
CUDA_ARCH=""
BINARY=""
SUNDIALS_DIR=""
BOOST_ROOT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --ab)
            AB_MODE=true
            # Next arg is optional ref (not starting with --)
            if [[ ${2:-} && ! ${2:-} == --* ]]; then
                AB_REF="$2"; shift
            fi
            shift ;;
        --binary)      BINARY="$2";     shift 2 ;;
        --param-file)  PARAM_FILE="$2"; shift 2 ;;
        --steps)       STEPS="$2";      shift 2 ;;
        --interval)    INTERVAL="$2";   shift 2 ;;
        --seed)        SEED="$2";       shift 2 ;;
        --reps)        REPS="$2";       shift 2 ;;
        --skip-build)  SKIP_BUILD=true; shift ;;
        --cuda-arch)   CUDA_ARCH="$2";  shift 2 ;;
        --sundials-dir) SUNDIALS_DIR="$2"; shift 2 ;;
        --boost-root)  BOOST_ROOT="$2";  shift 2 ;;
        --help|-h)
            sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

BENCH_DIR="${SCRIPT_DIR}/benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================================
# Helper functions
# ============================================================================

build_binary() {
    local src_dir="$1"
    local build_dir="$2"
    local label="$3"

    echo "  Building ${label}..."
    mkdir -p "$build_dir"

    # Reuse pre-built FLAMEGPU2 cache if available (from Docker image)
    if [[ -n "${FLAMEGPU_CACHE:-}" && -d "$FLAMEGPU_CACHE" && ! -d "${build_dir}/_deps" ]]; then
        echo "  Copying pre-built FLAMEGPU2 cache into ${build_dir}/_deps ..."
        cp -a "$FLAMEGPU_CACHE" "${build_dir}/_deps"
    fi

    local cmake_args=(-DCMAKE_BUILD_TYPE=Release)
    if [[ -n "$CUDA_ARCH" ]]; then
        cmake_args+=(-DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH")
    fi
    if [[ -n "$SUNDIALS_DIR" ]]; then
        cmake_args+=(-DSUNDIALS_DIR="$SUNDIALS_DIR")
    fi
    if [[ -n "$BOOST_ROOT" ]]; then
        cmake_args+=(-DBOOST_ROOT="$BOOST_ROOT")
    fi

    cmake -S "$src_dir" -B "$build_dir" "${cmake_args[@]}" 2>&1 | tee "${build_dir}/cmake_config.log"
    cmake --build "$build_dir" --parallel "$(nproc)" 2>&1 | tee "${build_dir}/cmake_build.log"

    if [[ ! -x "${build_dir}/bin/pdac" ]]; then
        echo "  ERROR: Build failed for ${label}. Check ${build_dir}/cmake_build.log"
        exit 1
    fi
    echo "  Built: ${build_dir}/bin/pdac"
}

run_config() {
    local binary="$1"
    local label="$2"
    local grid_out="$3"
    local interval="$4"
    local run_dir="$5"

    echo "  [${label}] grid_out=${grid_out} interval=${interval}"

    rm -rf "$run_dir"
    mkdir -p "$run_dir"

    pushd "$run_dir" > /dev/null

    local t_start t_end wall_ms
    t_start=$(date +%s%N)

    "$binary" \
        -p "$PARAM_FILE" \
        -s "$STEPS" \
        --seed "$SEED" \
        -G "$grid_out" \
        -oi "$interval" \
        > "${run_dir}/stdout.log" 2>&1

    t_end=$(date +%s%N)
    wall_ms=$(( (t_end - t_start) / 1000000 ))

    echo "$wall_ms" > "${run_dir}/wall_time_ms.txt"
    echo "    Wall time: ${wall_ms} ms"

    # Clean up bulk output data (keep timing CSVs and logs)
    rm -rf "${run_dir}/outputs/abm" "${run_dir}/outputs/pde"

    popd > /dev/null
}

# Run the 3 test configs for a given binary, with repetitions
run_suite() {
    local binary="$1"
    local suite_dir="$2"
    local suite_label="$3"

    echo ""
    echo "--- Running suite: ${suite_label} (${REPS} rep(s)) ---"
    echo "  Binary: ${binary}"

    for r in $(seq 1 "$REPS"); do
        local suffix=""
        [[ "$REPS" -gt 1 ]] && suffix="_r${r}"

        run_config "$binary" "no_io${suffix}"          0 1          "${suite_dir}/no_io${suffix}"
        run_config "$binary" "io_every_step${suffix}"  3 1          "${suite_dir}/io_every_step${suffix}"
        run_config "$binary" "io_interval_${INTERVAL}${suffix}" 3 "$INTERVAL" "${suite_dir}/io_interval_${INTERVAL}${suffix}"
    done
}

# ============================================================================
# Main
# ============================================================================

echo "============================================================"
echo "  I/O Benchmark Suite"
echo "============================================================"
echo "  Mode:       $(if $AB_MODE; then echo "A/B comparison (${AB_REF} vs working tree)"; else echo "single binary"; fi)"
echo "  Steps:      $STEPS"
echo "  Interval:   $INTERVAL"
echo "  Seed:       $SEED"
echo "  Params:     $PARAM_FILE"
if [[ -n "$CUDA_ARCH" ]]; then
    echo "  CUDA arch:  $CUDA_ARCH"
fi
echo "============================================================"

if $AB_MODE; then
    # ================================================================
    # A/B MODE: build old commit in worktree, build new from working tree
    # ================================================================
    RESULT_DIR="${BENCH_DIR}/ab_${TIMESTAMP}"
    mkdir -p "$RESULT_DIR"

    # Resolve the ref to a short hash for labeling
    OLD_HASH=$(git -C "$REPO_ROOT" rev-parse --short "$AB_REF")
    echo ""
    echo "Old version: ${AB_REF} (${OLD_HASH})"
    echo "New version: working tree (uncommitted changes)"

    WORKTREE_DIR="${BENCH_DIR}/.worktree_${OLD_HASH}"
    OLD_BINARY="${WORKTREE_DIR}/PDAC/sim/build/bin/pdac"
    NEW_BINARY="${SCRIPT_DIR}/build/bin/pdac"

    if ! $SKIP_BUILD; then
        # Create worktree for old version
        echo ""
        echo "=== Setting up worktree for ${AB_REF} ==="
        if [[ -d "$WORKTREE_DIR" ]]; then
            echo "  Removing existing worktree..."
            git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || rm -rf "$WORKTREE_DIR"
        fi
        git -C "$REPO_ROOT" worktree add "$WORKTREE_DIR" "$AB_REF" --detach
        echo "  Worktree at: $WORKTREE_DIR"

        # Build old version
        echo ""
        echo "=== Building OLD version (${OLD_HASH}) ==="
        build_binary "${WORKTREE_DIR}/PDAC/sim" "${WORKTREE_DIR}/PDAC/sim/build" "old (${OLD_HASH})"

        # Build new version (current working tree)
        echo ""
        echo "=== Building NEW version (working tree) ==="
        build_binary "${SCRIPT_DIR}" "${SCRIPT_DIR}/build" "new (working tree)"
    else
        echo ""
        echo "Skipping builds (--skip-build). Using existing binaries."
    fi

    # Verify both binaries exist
    for bin_path in "$OLD_BINARY" "$NEW_BINARY"; do
        if [[ ! -x "$bin_path" ]]; then
            echo "ERROR: Binary not found: $bin_path"
            echo "Run without --skip-build to build first."
            exit 1
        fi
    done

    # Run suites
    run_suite "$OLD_BINARY" "${RESULT_DIR}/old_${OLD_HASH}" "OLD (${OLD_HASH})"
    run_suite "$NEW_BINARY" "${RESULT_DIR}/new_working"      "NEW (working tree)"

    # Clean up worktree (keep build artifacts for re-runs with --skip-build)
    # Uncomment to auto-clean:
    # git -C "$REPO_ROOT" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true

    echo ""
    echo "============================================================"
    echo "  All runs complete. Analyzing..."
    echo "============================================================"

    # Run analysis across all subdirectories
    python3 "${SCRIPT_DIR}/analyze_benchmark.py" "$RESULT_DIR" "$STEPS"

    echo ""
    echo "Results saved to: ${RESULT_DIR}/"
    echo "  benchmark_report.txt   — human-readable comparison"
    echo "  benchmark_summary.csv  — machine-readable for plotting"
    echo "============================================================"

else
    # ================================================================
    # SINGLE BINARY MODE
    # ================================================================
    RESULT_DIR="${BENCH_DIR}/run_${TIMESTAMP}"

    if [[ -n "$BINARY" ]]; then
        if [[ ! -x "$BINARY" ]]; then
            echo "ERROR: Binary not found: $BINARY"
            exit 1
        fi
    else
        BINARY="${SCRIPT_DIR}/build/bin/pdac"
        if [[ ! -x "$BINARY" ]]; then
            echo "ERROR: Binary not found at $BINARY"
            echo "Build first with: ./build.sh"
            exit 1
        fi
    fi

    run_suite "$BINARY" "$RESULT_DIR" "current"

    echo ""
    echo "============================================================"
    echo "  All runs complete. Analyzing..."
    echo "============================================================"

    python3 "${SCRIPT_DIR}/analyze_benchmark.py" "$RESULT_DIR" "$STEPS"

    echo ""
    echo "Results saved to: ${RESULT_DIR}/"
    echo "============================================================"
fi