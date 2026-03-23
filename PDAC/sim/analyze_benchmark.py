#!/usr/bin/env python3
"""
analyze_benchmark.py — Parse timing CSVs from benchmark runs and compare.

Usage:
    python3 analyze_benchmark.py <benchmark_dir> [total_steps]
    python3 analyze_benchmark.py --single <outputs_dir>

Handles two directory layouts:

  Single-binary mode (flat):
    <benchmark_dir>/no_io/outputs/timing.csv
    <benchmark_dir>/io_every_step/outputs/timing.csv

  A/B mode (nested suites):
    <benchmark_dir>/old_abc1234/no_io/outputs/timing.csv
    <benchmark_dir>/old_abc1234/io_every_step/outputs/timing.csv
    <benchmark_dir>/new_working/no_io/outputs/timing.csv
    <benchmark_dir>/new_working/io_every_step/outputs/timing.csv
"""

import csv
import math
import re
import sys
from pathlib import Path
from collections import defaultdict

NUM_SUBSTRATES = 10  # O2, IFN, IL2, IL10, TGFB, CCL2, ARGI, NO, IL12, VEGFA


# ============================================================================
# Data readers
# ============================================================================

def read_timing_csv(path):
    rows = []
    if not path.exists():
        return rows
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def read_layer_timing_csv(path):
    layers = defaultdict(list)
    if not path.exists():
        return layers
    with open(path) as f:
        for row in csv.DictReader(f):
            layers[row["layer"]].append(float(row["ms"]))
    return layers


def read_init_timing_csv(path):
    phases = {}
    if not path.exists():
        return phases
    with open(path) as f:
        for row in csv.DictReader(f):
            phases[row["phase"]] = float(row["ms"])
    return phases


def read_wall_time(path):
    if not path.exists():
        return None
    return int(path.read_text().strip())


def read_grid_size(run_dir):
    """Parse grid dimensions from stdout.log (e.g. 'Grid: 50x50x50')."""
    log = Path(run_dir) / "stdout.log"
    if not log.exists():
        return None
    m = re.search(r"Grid:\s*(\d+)x(\d+)x(\d+)", log.read_text())
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


# ============================================================================
# Summarize one run directory
# ============================================================================

def summarize_run(run_dir, label=None, warmup=5):
    run_dir = Path(run_dir)
    outputs = run_dir / "outputs"

    timing_path = find_timing_csv(outputs)
    timing_all = read_timing_csv(timing_path) if timing_path else []
    layers = read_layer_timing_csv(outputs / "layer_timing.csv")
    init = read_init_timing_csv(outputs / "init_timing.csv")
    wall_ms = read_wall_time(run_dir / "wall_time_ms.txt")

    if not timing_all:
        return None

    # Skip warm-up steps for more stable averages
    timing = timing_all[warmup:] if len(timing_all) > warmup else timing_all
    n = len(timing)
    avg = lambda key: sum(r[key] for r in timing) / n

    abm_size = pde_size = 0
    for subdir, accum in [("abm", "abm"), ("pde", "pde")]:
        d = outputs / subdir
        if d.exists():
            total = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
            if subdir == "abm":
                abm_size = total
            else:
                pde_size = total

    layer_avgs = {name: sum(v[warmup:])/len(v[warmup:]) for name, v in layers.items() if len(v) > warmup}
    if not layer_avgs:
        layer_avgs = {name: sum(v)/len(v) for name, v in layers.items() if v}

    grid = read_grid_size(run_dir)

    return {
        "name": label or run_dir.name,
        "n_steps": n,
        "wall_ms": wall_ms,
        "avg_total_ms": avg("total_ms"),
        "avg_pde_ms": avg("pde_ms"),
        "avg_qsp_ms": avg("qsp_ms"),
        "avg_abm_ms": avg("abm_ms"),
        "sum_total_ms": sum(r["total_ms"] for r in timing),
        "min_step_ms": min(r["total_ms"] for r in timing),
        "max_step_ms": max(r["total_ms"] for r in timing),
        "layer_avgs": layer_avgs,
        "init_phases": init,
        "abm_output_mb": abm_size / (1024 * 1024),
        "pde_output_mb": pde_size / (1024 * 1024),
        "grid_size": grid,
    }


# ============================================================================
# Directory discovery
# ============================================================================

def find_timing_csv(outputs_dir):
    """Find timing CSV (timing.csv or timing_*.csv) in an outputs directory."""
    outputs_dir = Path(outputs_dir)
    # Try exact name first, then glob for seed-suffixed variants
    if (outputs_dir / "timing.csv").exists():
        return outputs_dir / "timing.csv"
    matches = sorted(outputs_dir.glob("timing_*.csv"))
    return matches[0] if matches else None


def find_run_dirs(bench_dir):
    """Find all directories containing outputs/timing*.csv, up to 2 levels deep."""
    bench_dir = Path(bench_dir)
    runs = []
    for d in sorted(bench_dir.iterdir()):
        if not d.is_dir():
            continue
        if find_timing_csv(d / "outputs"):
            runs.append(d)
        else:
            # Check one level deeper (A/B suite layout)
            for dd in sorted(d.iterdir()):
                if dd.is_dir() and find_timing_csv(dd / "outputs"):
                    runs.append(dd)
    return runs


def detect_suites(run_dirs):
    """Group run dirs into suites. Returns dict {suite_label: [run_dirs]}.
    If all runs are at the same level (flat), returns one suite.
    If runs are in suite subdirs (A/B), returns multiple suites."""
    # Check if runs share the same parent → flat layout
    parents = set(d.parent for d in run_dirs)
    if len(parents) == 1:
        return {"current": run_dirs}

    # Grouped by parent directory name
    suites = defaultdict(list)
    for d in run_dirs:
        suites[d.parent.name].append(d)
    return dict(suites)


# ============================================================================
# Repetition aggregation
# ============================================================================

def aggregate_reps(summaries):
    """Merge summaries with _rN suffixes into averaged entries with stddev."""
    groups = defaultdict(list)
    for s in summaries:
        # Strip _r1, _r2, etc. suffix to get base config name
        base = re.sub(r'_r\d+$', '', s['name'])
        groups[base].append(s)

    aggregated = []
    for base, reps in groups.items():
        if len(reps) == 1:
            aggregated.append(reps[0])
            continue
        n = len(reps)
        avg = lambda key: sum(r[key] for r in reps) / n
        std = lambda key: math.sqrt(sum((r[key] - avg(key))**2 for r in reps) / n)
        agg = {
            "name": base,
            "n_steps": reps[0]["n_steps"],
            "n_reps": n,
            "wall_ms": avg("wall_ms") if reps[0]["wall_ms"] else None,
            "avg_total_ms": avg("avg_total_ms"),
            "avg_total_std": std("avg_total_ms"),
            "avg_pde_ms": avg("avg_pde_ms"),
            "avg_qsp_ms": avg("avg_qsp_ms"),
            "avg_abm_ms": avg("avg_abm_ms"),
            "sum_total_ms": avg("sum_total_ms"),
            "min_step_ms": min(r["min_step_ms"] for r in reps),
            "max_step_ms": max(r["max_step_ms"] for r in reps),
            "abm_output_mb": avg("abm_output_mb"),
            "pde_output_mb": avg("pde_output_mb"),
            "grid_size": reps[0]["grid_size"],
            "layer_avgs": {},
            "init_phases": {},
        }
        # Merge layer averages
        all_layers = set()
        for r in reps:
            all_layers.update(r.get("layer_avgs", {}).keys())
        for layer in all_layers:
            vals = [r["layer_avgs"][layer] for r in reps if layer in r.get("layer_avgs", {})]
            if vals:
                agg["layer_avgs"][layer] = sum(vals) / len(vals)
        aggregated.append(agg)
    return aggregated


# ============================================================================
# Report formatting
# ============================================================================

def format_suite_table(suite_label, summaries):
    """Format a table for one suite."""
    lines = []
    n_reps = summaries[0].get("n_reps", 1) if summaries else 1
    reps_str = f" ({n_reps} reps)" if n_reps > 1 else ""
    lines.append(f"  Suite: {suite_label}{reps_str}")
    lines.append("")
    header = f"  {'Config':<25} {'Wall(s)':>8} {'Avg/step':>10} {'±std':>8} {'Min':>8} {'Max':>8} {'PDE':>8} {'QSP':>8} {'ABM*':>8}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for s in summaries:
        wall_s = f"{s['wall_ms']/1000:.1f}" if s["wall_ms"] else "N/A"
        std_s = f"{s.get('avg_total_std', 0):.1f}" if s.get('avg_total_std') else ""
        lines.append(
            f"  {s['name']:<25} {wall_s:>8} {s['avg_total_ms']:>9.1f}ms "
            f"{std_s:>7} {s['min_step_ms']:>7.1f} {s['max_step_ms']:>7.1f} "
            f"{s['avg_pde_ms']:>7.1f} {s['avg_qsp_ms']:>7.1f} {s['avg_abm_ms']:>7.1f}"
        )

    # I/O overhead within this suite
    no_io = next((s for s in summaries if "no_io" in s["name"]), None)
    if no_io:
        lines.append("")
        lines.append(f"  I/O overhead (vs no_io baseline {no_io['avg_total_ms']:.1f} ms/step):")
        for s in summaries:
            if "no_io" in s["name"]:
                continue
            overhead = s["avg_total_ms"] - no_io["avg_total_ms"]
            pct = (overhead / no_io["avg_total_ms"]) * 100 if no_io["avg_total_ms"] > 0 else 0
            lines.append(
                f"    {s['name']:<23} +{overhead:>6.1f} ms/step  "
                f"({pct:>5.1f}% of compute)  "
                f"[ABM: {s['abm_output_mb']:.1f} MB, PDE: {s['pde_output_mb']:.1f} MB]"
            )

    return lines


def format_ab_comparison(suites_data):
    """Compare matching configs across two suites (old vs new)."""
    lines = []
    suite_names = list(suites_data.keys())
    if len(suite_names) != 2:
        return lines

    old_name = next((n for n in suite_names if "old" in n.lower()), suite_names[0])
    new_name = next((n for n in suite_names if "new" in n.lower()), suite_names[1])

    old_runs = {s["name"]: s for s in suites_data[old_name]}
    new_runs = {s["name"]: s for s in suites_data[new_name]}

    common_configs = sorted(set(old_runs.keys()) & set(new_runs.keys()),
                            key=lambda x: (0 if "no_io" in x else 1, x))

    if not common_configs:
        return lines

    lines.append("-" * 72)
    lines.append("  HEAD-TO-HEAD COMPARISON (OLD vs NEW)")
    lines.append("-" * 72)
    lines.append("")

    header = f"  {'Config':<25} {'Old (ms)':>10} {'New (ms)':>10} {'Delta':>10} {'Speedup':>10}"
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))

    for config in common_configs:
        old = old_runs[config]
        new = new_runs[config]
        delta = new["avg_total_ms"] - old["avg_total_ms"]
        speedup = old["avg_total_ms"] / new["avg_total_ms"] if new["avg_total_ms"] > 0 else float("inf")
        sign = "+" if delta > 0 else ""
        lines.append(
            f"  {config:<25} {old['avg_total_ms']:>9.1f} {new['avg_total_ms']:>9.1f} "
            f"{sign}{delta:>9.1f} {speedup:>9.2f}x"
        )

    lines.append("")
    lines.append("  (Speedup > 1.0 means NEW is faster; < 1.0 means OLD was faster)")

    # Focused I/O overhead comparison
    old_noio = old_runs.get("no_io")
    new_noio = new_runs.get("no_io")
    old_io = old_runs.get("io_every_step")
    new_io = new_runs.get("io_every_step")

    if old_noio and new_noio and old_io and new_io:
        old_overhead = old_io["avg_total_ms"] - old_noio["avg_total_ms"]
        new_overhead = new_io["avg_total_ms"] - new_noio["avg_total_ms"]
        reduction = old_overhead - new_overhead
        reduction_pct = (reduction / old_overhead) * 100 if old_overhead > 0 else 0

        lines.append("")
        lines.append("  I/O overhead (every-step output):")
        lines.append(f"    Old: {old_overhead:.1f} ms/step")
        lines.append(f"    New: {new_overhead:.1f} ms/step")
        lines.append(f"    Reduction: {reduction:.1f} ms/step ({reduction_pct:.0f}%)")

    # Memory cost of pinned double buffers (new optimization)
    grid = new_io["grid_size"] if new_io else None
    if grid is None:
        # Try any run
        for s in list(new_runs.values()) + list(old_runs.values()):
            if s.get("grid_size"):
                grid = s["grid_size"]
                break
    if grid:
        gx, gy, gz = grid
        total_voxels = gx * gy * gz
        pinned_mb = 2 * NUM_SUBSTRATES * total_voxels * 4 / (1024 * 1024)
        lines.append("")
        lines.append(f"  Memory cost of async I/O optimization:")
        lines.append(f"    Grid: {gx}x{gy}x{gz} ({total_voxels:,} voxels)")
        lines.append(f"    Pinned host buffers: 2 x {NUM_SUBSTRATES} substrates x {total_voxels:,} x 4B = {pinned_mb:.1f} MB")
        lines.append(f"    (host RAM only — no additional GPU memory)")

    return lines


def format_report(suites_data, warmup=5):
    lines = []
    lines.append("=" * 72)
    lines.append("  I/O BENCHMARK REPORT")
    lines.append(f"  (first {warmup} warm-up steps excluded from averages)")
    lines.append("=" * 72)
    lines.append("")

    is_ab = len(suites_data) > 1

    for suite_name, summaries in suites_data.items():
        # Aggregate repetitions (e.g. no_io_r1, no_io_r2 → no_io with stddev)
        summaries = aggregate_reps(summaries)
        suites_data[suite_name] = summaries
        # Sort: no_io first
        summaries.sort(key=lambda s: (0 if "no_io" in s["name"] else 1, s["name"]))
        lines.extend(format_suite_table(suite_name, summaries))
        lines.append("")

    lines.append("  * ABM = total - PDE - QSP (includes I/O overhead from step functions)")
    lines.append("")

    # A/B head-to-head
    if is_ab:
        lines.extend(format_ab_comparison(suites_data))
        lines.append("")

    # Layer-level detail from io_every_step (pick the first suite that has one)
    for suite_name, summaries in suites_data.items():
        io_run = next((s for s in summaries if "every" in s["name"]), None)
        if io_run and io_run["layer_avgs"]:
            lines.append("-" * 72)
            label = f"PER-LAYER AVERAGE TIMING ({suite_name} / io_every_step)"
            lines.append(f"  {label}")
            lines.append("-" * 72)
            sorted_layers = sorted(io_run["layer_avgs"].items(), key=lambda x: -x[1])
            for name, avg in sorted_layers:
                lines.append(f"    {name:<30} {avg:>8.2f} ms")
            lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


# ============================================================================
# Single-run analysis
# ============================================================================

def analyze_single(outputs_dir):
    outputs_dir = Path(outputs_dir)
    run_dir = outputs_dir.parent if outputs_dir.name == "outputs" else outputs_dir

    s = summarize_run(run_dir)
    if not s:
        print(f"ERROR: No timing.csv found in {run_dir}/outputs/")
        return

    lines = []
    lines.append("=" * 60)
    lines.append("  SINGLE RUN ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"  Steps:     {s['n_steps']}")
    lines.append(f"  Wall time: {s['wall_ms']/1000:.1f}s" if s["wall_ms"] else "  Wall time: N/A")
    lines.append(f"  Avg/step:  {s['avg_total_ms']:.1f} ms")
    lines.append(f"    PDE:     {s['avg_pde_ms']:.1f} ms ({s['avg_pde_ms']/s['avg_total_ms']*100:.0f}%)")
    lines.append(f"    QSP:     {s['avg_qsp_ms']:.1f} ms ({s['avg_qsp_ms']/s['avg_total_ms']*100:.0f}%)")
    lines.append(f"    ABM+I/O: {s['avg_abm_ms']:.1f} ms ({s['avg_abm_ms']/s['avg_total_ms']*100:.0f}%)")
    lines.append(f"  Step range: {s['min_step_ms']:.1f} - {s['max_step_ms']:.1f} ms")
    lines.append(f"  ABM output: {s['abm_output_mb']:.1f} MB")
    lines.append(f"  PDE output: {s['pde_output_mb']:.1f} MB")
    lines.append("")

    if s["layer_avgs"]:
        lines.append("  Per-layer averages:")
        for name, avg in sorted(s["layer_avgs"].items(), key=lambda x: -x[1]):
            lines.append(f"    {name:<30} {avg:>8.2f} ms")

    lines.append("=" * 60)
    print("\n".join(lines))


# ============================================================================
# CSV export
# ============================================================================

def write_summary_csv(path, suites_data):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "suite", "config", "wall_s", "avg_step_ms", "avg_pde_ms",
            "avg_qsp_ms", "avg_abm_ms", "min_step_ms", "max_step_ms",
            "abm_output_mb", "pde_output_mb"
        ])
        for suite_name, summaries in suites_data.items():
            for s in summaries:
                writer.writerow([
                    suite_name,
                    s["name"],
                    f"{s['wall_ms']/1000:.1f}" if s["wall_ms"] else "",
                    f"{s['avg_total_ms']:.2f}",
                    f"{s['avg_pde_ms']:.2f}",
                    f"{s['avg_qsp_ms']:.2f}",
                    f"{s['avg_abm_ms']:.2f}",
                    f"{s['min_step_ms']:.2f}",
                    f"{s['max_step_ms']:.2f}",
                    f"{s['abm_output_mb']:.2f}",
                    f"{s['pde_output_mb']:.2f}",
                ])


# ============================================================================
# Main
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 analyze_benchmark.py <benchmark_dir> [total_steps]")
        print("  python3 analyze_benchmark.py --single <outputs_dir>")
        sys.exit(1)

    if sys.argv[1] == "--single":
        analyze_single(sys.argv[2])
        return

    bench_dir = Path(sys.argv[1])
    warmup = 5
    # Check for --warmup flag
    remaining = sys.argv[2:]
    i = 0
    while i < len(remaining):
        if remaining[i] == "--warmup" and i + 1 < len(remaining):
            warmup = int(remaining[i + 1])
            i += 2
        else:
            i += 1

    run_dirs = find_run_dirs(bench_dir)

    if not run_dirs:
        print(f"ERROR: No benchmark runs found in {bench_dir}")
        print("Each run directory should contain outputs/timing.csv")
        sys.exit(1)

    suites = detect_suites(run_dirs)

    suites_data = {}
    for suite_name, dirs in suites.items():
        summaries = []
        for d in dirs:
            s = summarize_run(d, warmup=warmup)
            if s:
                summaries.append(s)
        if summaries:
            suites_data[suite_name] = summaries

    report = format_report(suites_data, warmup=warmup)
    print(report)

    # Save outputs
    report_path = bench_dir / "benchmark_report.txt"
    report_path.write_text(report)

    csv_path = bench_dir / "benchmark_summary.csv"
    write_summary_csv(csv_path, suites_data)


if __name__ == "__main__":
    main()