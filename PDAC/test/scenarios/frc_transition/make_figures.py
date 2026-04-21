#!/usr/bin/env python3
# Test 10 — FRC Transition figures + pass/fail summary.
#
# Three-panel dwell-counter trace + final state bar.
# Reads outputs_pos/, outputs_treg/, outputs_sub/ under the scenario dir.

from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent

RUNS = [
    ("pos",  "Positive (10 B + 10 T_EFF)",  "LTβ⁺ cluster → FRC"),
    ("treg", "Treg control (20 TCD4_TREG)", "LTβ⁻ cluster → no transition"),
    ("sub",  "Sub-threshold (3 B + 3 T_EFF)", "n_ltb = 6 < threshold 8"),
]

FIB_ICAF = 2
FIB_FRC  = 3

# ── Load runs ──────────────────────────────────────────────────────────────
runs = {}
for key, _, _ in RUNS:
    traj_p   = HERE / f"outputs_{key}" / "trajectory.csv"
    params_p = HERE / f"outputs_{key}" / "params.csv"
    if not traj_p.exists():
        print(f"[FAIL] missing {traj_p}", file=sys.stderr)
        sys.exit(1)
    rows = list(csv.DictReader(traj_p.open()))
    step    = np.array([int(r["step"])                 for r in rows])
    state   = np.array([int(r["fib_state"])            for r in rows])
    dwell   = np.array([int(r["frc_dwell_counter"])    for r in rows])
    n_ltb   = np.array([int(r["neighbor_ltb_count"])   for r in rows])
    params = {}
    for row in csv.DictReader(params_p.open()):
        params[row["key"]] = float(row["value"])
    runs[key] = dict(step=step, state=state, dwell=dwell, n_ltb=n_ltb,
                     params=params)

threshold   = runs["pos"]["params"]["PARAM_FIB_FRC_LYMPH_THRESHOLD"]
dwell_steps = runs["pos"]["params"]["PARAM_FIB_FRC_DWELL_STEPS"]

# ── Pass criteria ──────────────────────────────────────────────────────────
checks = []

# Run 1 (positive): state flips to FIB_FRC, and dwell counter reached dwell_steps
# at some step before the flip. Since counter resets to 0 on flip, check that
# dwell hit dwell_steps - 1 (the step *before* flip) at some point.
pos = runs["pos"]
flipped = (pos["state"] == FIB_FRC).any()
# The flip occurs the step AFTER dwell reaches dwell_steps (inclusive). Observed
# counter sequence: 1, 2, 3 (pre-flip), then 0 (post-flip).
max_dwell_pos = int(pos["dwell"].max())
checks.append((
    "pos_transitions_to_frc",
    "Positive: fib state reaches FIB_FRC (3) within run",
    bool(flipped),
    f"states observed: {sorted(set(pos['state'].tolist()))}",
))
checks.append((
    "pos_counter_increments",
    f"Positive: counter reached {int(dwell_steps)-1} pre-flip",
    max_dwell_pos >= int(dwell_steps) - 1,
    f"max dwell before flip = {max_dwell_pos}",
))
checks.append((
    "pos_n_ltb_correct",
    "Positive: neighbor_ltb_count == 20 (all 20 LTβ⁺ cells counted)",
    pos["n_ltb"][0] == 20,
    f"n_ltb at step 0 = {pos['n_ltb'][0]}",
))

# Run 2 (Treg): state stays ICAF, counter never non-zero, n_ltb == 0 always.
tr = runs["treg"]
checks.append((
    "treg_no_transition",
    "Treg: state stays FIB_ICAF (2) entire run",
    bool((tr["state"] == FIB_ICAF).all()),
    f"final state = {int(tr['state'][-1])}",
))
checks.append((
    "treg_counter_zero",
    "Treg: counter never leaves 0",
    bool((tr["dwell"] == 0).all()),
    f"max dwell = {int(tr['dwell'].max())}",
))
checks.append((
    "treg_excluded_from_ltb",
    "Treg: neighbor_ltb_count == 0 (Treg state correctly excluded from LTβ count)",
    bool((tr["n_ltb"] == 0).all()),
    f"max n_ltb = {int(tr['n_ltb'].max())}",
))

# Run 3 (sub-threshold): state stays ICAF, counter stays 0 (resets each step
# because n_ltb < threshold), n_ltb == 6.
sb = runs["sub"]
checks.append((
    "sub_no_transition",
    "Sub-threshold: state stays FIB_ICAF entire run",
    bool((sb["state"] == FIB_ICAF).all()),
    f"final state = {int(sb['state'][-1])}",
))
checks.append((
    "sub_counter_zero",
    "Sub-threshold: counter stays 0 (resets each step when n_ltb < threshold)",
    bool((sb["dwell"] == 0).all()),
    f"max dwell = {int(sb['dwell'].max())}",
))
checks.append((
    "sub_n_ltb_six",
    f"Sub-threshold: n_ltb == 6 < threshold ({int(threshold)})",
    bool((sb["n_ltb"] == 6).all()),
    f"observed n_ltb = {int(sb['n_ltb'][0])}",
))

# Write summary
summary_path = HERE / "pass_summary.csv"
with summary_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["check", "description", "pass", "detail"])
    for name, desc, ok, detail in checks:
        w.writerow([name, desc, "PASS" if ok else "FAIL", detail])

all_pass = all(c[2] for c in checks)

# ── Figure: 3 panels, dwell counter + shaded state ─────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)

for ax, (key, label, subtitle) in zip(axes, RUNS):
    r = runs[key]
    # Shade FIB_FRC region green, FIB_ICAF white
    # Paint a state stripe at y=5.3 as visual indicator
    ax2 = ax.twinx()
    state_colors = np.where(r["state"] == FIB_FRC, 1, 0)
    ax.bar(r["step"], r["dwell"], width=0.8, color="#4575b4",
           label="frc_dwell_counter")
    ax.axhline(dwell_steps, color="#d62728", ls="--", lw=1.2,
               label=f"dwell_steps = {int(dwell_steps)}")
    # Shade where state == FIB_FRC
    for s, st in zip(r["step"], r["state"]):
        if st == FIB_FRC:
            ax.axvspan(s - 0.5, s + 0.5, color="#b4f77e", alpha=0.25, zorder=-1)

    ax.plot(r["step"], r["n_ltb"], marker="o", lw=1.2, color="#ff8c00",
            label="n_ltb neighbors", markersize=3)
    ax.axhline(threshold, color="#ff8c00", ls=":", lw=1.0, alpha=0.6,
               label=f"ltb threshold = {int(threshold)}")
    ax.set_xlabel("ABM step")
    ax.set_ylabel("counter / n_ltb")
    ax.set_title(f"{label}\n{subtitle}", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(-0.5, 22)
    ax2.set_yticks([])

status = "PASS" if all_pass else "FAIL"
fig.suptitle(f"Test 10 — FRC transition  [{status}]  "
             f"(green shade = FIB_FRC)", fontsize=12)
fig.tight_layout()
fig.savefig(HERE / "frc_transition.png", dpi=120)
plt.close(fig)

# ── Console summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Test 10 — FRC transition  :  {status}")
print("=" * 60)
for name, desc, ok, detail in checks:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:<28} {desc}")
    print(f"           {detail}")
print(f"\n  Figure:  {HERE/'frc_transition.png'}")
print(f"  Summary: {summary_path}")

sys.exit(0 if all_pass else 1)
