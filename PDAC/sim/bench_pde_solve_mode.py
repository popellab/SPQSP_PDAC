#!/usr/bin/env python3
"""Benchmark + A/B-validate the PDE solve modes (transient LOD vs quasi-steady-state).

For each mode in {0=transient, 1=steady} the script:
  1. Writes a temp param XML with <pde_solve_mode> set to that mode.
  2. Runs ./build/bin/pdac with a FIXED seed/grid/steps and a per-mode --output-root.
  3. Parses timing.csv (per-step pde/total wall time) and stats_<seed>.csv (agent
     composition) from each run.

It then reports:
  * Timing: mean per-step pde_ms and total_ms, and the speedup (transient/steady).
  * Equivalence: final-step agent-count relative differences between the two modes
    — the downstream ABM observables that actually matter (fields are inaccurate in
    transient mode by construction, so we compare what the ABM consumes, not C itself).

Usage:
  python3 bench_pde_solve_mode.py [--grid 40] [--steps 60] [--seed 12345]
                                  [--bin build/bin/pdac] [--xml resource/param_all_test.xml]
                                  [--extra "--tcells 50 --macs 5"]
"""
from __future__ import annotations
import argparse, csv, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def set_solve_mode(src_xml: Path, mode: int, dst_xml: Path) -> None:
    txt = src_xml.read_text()
    new, n = re.subn(
        r'(<pde_solve_mode\b[^>]*>)\s*\d+\s*(</pde_solve_mode>)',
        rf'\g<1>{mode}\g<2>', txt)
    if n != 1:
        sys.exit(f"ERROR: expected exactly 1 <pde_solve_mode> node, found {n} in {src_xml}")
    dst_xml.write_text(new)


def run_mode(binary: Path, xml: Path, mode: int, grid: int, steps: int,
             seed: int, out_root: Path, extra: list[str]) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-p", str(xml), "-g", str(grid), "-s", str(steps),
           "--seed", str(seed), "--output-root", str(out_root), *extra]
    print(f"\n[mode {mode}] $ {' '.join(cmd)}")
    t = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if t.returncode != 0:
        print(t.stdout[-3000:]); print(t.stderr[-3000:])
        sys.exit(f"ERROR: mode {mode} run failed (exit {t.returncode})")


def read_timing(out_root: Path, seed: int) -> dict:
    p = out_root / f"timing_{seed}.csv"
    if not p.exists():
        sys.exit(f"ERROR: missing {p}")
    pde, tot = [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            pde.append(float(row["pde_ms"])); tot.append(float(row["total_ms"]))
    # drop step 0 (warmup/JIT) from the mean
    pde, tot = pde[1:] or pde, tot[1:] or tot
    return {"pde_ms": sum(pde) / len(pde), "total_ms": sum(tot) / len(tot), "n": len(pde)}


def read_final_stats(out_root: Path, seed: int) -> dict:
    p = out_root / f"stats_{seed}.csv"
    if not p.exists():
        sys.exit(f"ERROR: missing {p}")
    with open(p) as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    return {k: float(v) for k, v in last.items()
            if k.startswith("agentCount") and v not in ("", None)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=40)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--bin", default="build/bin/pdac")
    ap.add_argument("--xml", default="resource/param_all_test.xml")
    ap.add_argument("--extra", default="", help="extra CLI args passed verbatim to pdac")
    ap.add_argument("--keep", action="store_true", help="keep per-mode output dirs")
    ap.add_argument("--modes", default="0,1",
                    help="comma-separated solve modes to compare (0=transient,1=CG,2=FFT)")
    args = ap.parse_args()
    modes = [int(m) for m in args.modes.split(",")]
    base_mode = modes[0]

    binary = (HERE / args.bin).resolve()
    base_xml = (HERE / args.xml).resolve()
    if not binary.exists():
        sys.exit(f"ERROR: binary not found: {binary} (build first)")

    label = {0: "transient(0)", 1: "CG(1)", 2: "FFT(2)"}
    tmpdir = Path(tempfile.mkdtemp(prefix="pde_bench_"))
    results = {}
    try:
        for mode in modes:
            xml = tmpdir / f"param_mode{mode}.xml"
            set_solve_mode(base_xml, mode, xml)
            out_root = tmpdir / f"out_mode{mode}"
            run_mode(binary, xml, mode, args.grid, args.steps, args.seed,
                     out_root, args.extra.split())
            results[mode] = {"timing": read_timing(out_root, args.seed),
                             "stats": read_final_stats(out_root, args.seed)}

        tb = results[base_mode]["timing"]
        print("\n" + "=" * 64)
        print(f"PDE solve-mode benchmark  (grid={args.grid}^3, steps={args.steps}, seed={args.seed})")
        print("=" * 64)
        hdr = f"{'metric':<12}" + "".join(f"{label.get(m, m):>14}" for m in modes) + f"{'speedup(base/last)':>20}"
        print(hdr)
        for key in ("pde_ms", "total_ms"):
            cells = "".join(f"{results[m]['timing'][key]:>14.3f}" for m in modes)
            last = results[modes[-1]]["timing"][key]
            sp = tb[key] / last if last else float("nan")
            print(f"{key:<12}{cells}{sp:>19.2f}x")

        # equivalence: base mode vs each other mode
        sb = results[base_mode]["stats"]
        for m in modes[1:]:
            sm = results[m]["stats"]
            keys = sorted(set(sb) | set(sm))
            worst = 0.0
            print(f"\nComposition: {label.get(base_mode)} vs {label.get(m)} (counts >5 only flagged):")
            for k in keys:
                a, b = sb.get(k, 0.0), sm.get(k, 0.0)
                rel = abs(a - b) / max(abs(a), abs(b), 1.0)
                if max(a, b) > 5:
                    worst = max(worst, rel)
                flag = "  <-- check" if rel > 0.15 and max(a, b) > 5 else ""
                print(f"  {k:<40}{a:>9.0f}{b:>9.0f}{rel:>8.1%}{flag}")
            print(f"  worst diff on counts >5: {worst:.1%}")
    finally:
        if not args.keep:
            shutil.rmtree(tmpdir, ignore_errors=True)
        else:
            print(f"\nkept: {tmpdir}")


if __name__ == "__main__":
    main()
