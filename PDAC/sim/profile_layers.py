#!/usr/bin/env python3
"""Aggregate layer_timing.csv into a per-phase profile (mean ms + % of step).

Reports overall mean and a "late-run" mean (last `--tail` steps, the expensive
large-agent-count regime). Usage: profile_layers.py <output_root> [--tail 20]
"""
from __future__ import annotations
import csv, sys, argparse
from collections import defaultdict
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--tail", type=int, default=20)
ap.add_argument("--seed", default=None, help="seed (auto-detect if omitted)")
args = ap.parse_args()

root = Path(args.root)
cands = list(root.glob("layer_timing.csv")) + list(root.glob("layer_timing*.csv"))
if not cands:
    sys.exit(f"no layer_timing.csv under {root}")
path = cands[0]

# label -> {step -> value}
data: dict[str, dict[int, float]] = defaultdict(dict)
with open(path) as f:
    for row in csv.reader(f):
        if len(row) != 3:
            continue
        step, label, val = row
        try:
            data[label][int(step)] = float(val)
        except ValueError:
            continue

steps = sorted(data.get("total_ms", {}).keys())
if not steps:
    sys.exit("no total_ms rows")
tail_steps = set(steps[-args.tail:])

# Phase labels in execution order forming a partition step_start→after-division,
# plus qsp/io (post-division). pde_solve_ms is excluded (duplicate of pde_wall).
order = ["recruit", "occupancy", "movement", "bcast_out", "broadcast_scan",
         "state_sources", "pde_wall", "gradients", "ecm", "division", "qsp_solve_ms"]
io = sorted(l for l in data if l.startswith("io_"))
phases = [l for l in order if l in data] + io
exclude = ("total_ms", "gpu_mem_mb", "pde_solve_ms")
other = [l for l in data if l not in phases and l not in exclude]
phases += sorted(other)

def mean(label, stepset):
    vals = [v for s, v in data[label].items() if s in stepset]
    return sum(vals) / len(vals) if vals else 0.0

allset = set(steps)
tot_all = mean("total_ms", allset)
tot_tail = mean("total_ms", tail_steps)

print(f"\nProfile: {path}   ({len(steps)} steps; tail = last {args.tail})")
print(f"{'phase':<18}{'all ms':>10}{'all %':>8}{'tail ms':>10}{'tail %':>8}")
print("-" * 54)
rows = []
for p in phases:
    ma, mt = mean(p, allset), mean(p, tail_steps)
    rows.append((p, ma, mt))
# sort by tail ms desc (the expensive-regime ranking)
for p, ma, mt in sorted(rows, key=lambda r: -r[2]):
    pa = 100 * ma / tot_all if tot_all else 0
    pt = 100 * mt / tot_tail if tot_tail else 0
    print(f"{p:<18}{ma:>10.2f}{pa:>7.1f}%{mt:>10.2f}{pt:>7.1f}%")
print("-" * 54)
print(f"{'TOTAL_ms':<18}{tot_all:>10.2f}{'':>8}{tot_tail:>10.2f}")
gm = mean("gpu_mem_mb", tail_steps)
if gm:
    print(f"{'gpu_mem_mb (tail)':<18}{'':>10}{'':>8}{gm:>10.0f}")
# coverage check
cov_tail = sum(mt for _, _, mt in rows)
print(f"\nphases sum (tail) = {cov_tail:.1f} ms vs total {tot_tail:.1f} ms "
      f"({100*cov_tail/tot_tail:.0f}% accounted)")
