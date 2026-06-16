#!/usr/bin/env python3
"""Sweep PARAM_MOLECULAR_STEPS (transient LOD substeps) and measure speed vs accuracy.

Hypothesis: at dt_abm=6h the backward-Euler LOD solve already reaches the quasi-steady
field in very few substeps, so the default 36 is largely redundant. This sweeps the
substep count, runs a fixed seed/grid, and reports per-step pde_ms plus the final-step
composition drift relative to the 36-substep baseline.

Usage: python3 bench_substeps.py [--grid 30] [--steps 40] [--seed 12345]
                                 [--list 36,12,6,4,2,1]
"""
from __future__ import annotations
import argparse, csv, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def set_substeps(src_xml: Path, n: int, dst_xml: Path) -> None:
    txt = src_xml.read_text()
    new, k = re.subn(r'(<stepPerSlice\b[^>]*>)\s*\d+\s*(</stepPerSlice>)',
                     rf'\g<1>{n}\g<2>', txt)
    if k != 1:
        sys.exit(f"ERROR: expected 1 <stepPerSlice>, found {k}")
    dst_xml.write_text(new)


def run(binary, xml, grid, steps, seed, out_root, extra):
    out_root.mkdir(parents=True, exist_ok=True)
    cmd = [str(binary), "-p", str(xml), "-g", str(grid), "-s", str(steps),
           "--seed", str(seed), "--output-root", str(out_root), *extra]
    t = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    if t.returncode != 0:
        print(t.stdout[-2000:]); print(t.stderr[-2000:])
        sys.exit(f"run failed (substeps run, exit {t.returncode})")


def mean_pde(out_root, seed):
    pde = []
    with open(out_root / f"timing_{seed}.csv") as f:
        for row in csv.DictReader(f):
            pde.append(float(row["pde_ms"]))
    pde = pde[1:] or pde
    return sum(pde) / len(pde)


def final_stats(out_root, seed):
    with open(out_root / f"stats_{seed}.csv") as f:
        rows = list(csv.DictReader(f))
    return {k: float(v) for k, v in rows[-1].items()
            if k.startswith("agentCount") and v not in ("", None)}


def worst_diff(base, other, floor=5):
    worst = 0.0
    keys = sorted(set(base) | set(other))
    for k in keys:
        a, b = base.get(k, 0.0), other.get(k, 0.0)
        if max(a, b) <= floor:   # ignore small (noise-dominated) populations
            continue
        worst = max(worst, abs(a - b) / max(abs(a), abs(b), 1.0))
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=30)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--bin", default="build/bin/pdac")
    ap.add_argument("--xml", default="resource/param_all_test.xml")
    ap.add_argument("--extra", default="")
    ap.add_argument("--list", default="36,12,6,4,2,1")
    args = ap.parse_args()

    binary = (HERE / args.bin).resolve()
    base_xml = (HERE / args.xml).resolve()
    subs = [int(x) for x in args.list.split(",")]
    tmp = Path(tempfile.mkdtemp(prefix="substep_bench_"))
    res = {}
    try:
        for n in subs:
            xml = tmp / f"p{n}.xml"; set_substeps(base_xml, n, xml)
            out = tmp / f"out{n}"
            run(binary, xml, args.grid, args.steps, args.seed, out, args.extra.split())
            res[n] = (mean_pde(out, args.seed), final_stats(out, args.seed))
        base_stats = res[subs[0]][1]
        base_pde = res[subs[0]][0]
        print(f"\nSubstep sweep (grid={args.grid}^3, steps={args.steps}, seed={args.seed}); "
              f"baseline = {subs[0]} substeps")
        print(f"{'substeps':>9}{'pde_ms':>12}{'speedup':>10}{'worst Δ(>5)':>14}{'worst Δ(>50)':>14}")
        for n in subs:
            pde, st = res[n]
            sp = base_pde / pde if pde else float('nan')
            wd5 = worst_diff(base_stats, st, 5)
            wd50 = worst_diff(base_stats, st, 50)
            print(f"{n:>9}{pde:>12.3f}{sp:>9.2f}x{wd5:>13.1%}{wd50:>13.1%}")

        # Per-population detail on the BULK populations (baseline count > 20)
        bulk = sorted(k for k in base_stats if base_stats[k] > 20)
        print("\nBulk populations (baseline >20) — absolute counts per substep setting:")
        hdr = "  " + f"{'population':<32}" + "".join(f"{n:>9}" for n in subs)
        print(hdr)
        for k in bulk:
            row = "  " + f"{k:<32}" + "".join(f"{res[n][1].get(k,0):>9.0f}" for n in subs)
            print(row)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
