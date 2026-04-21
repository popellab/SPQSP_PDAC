#!/usr/bin/env python3
# Test 8 — DC Priming Cascade figures + pass/fail summary.
#
# Reads outputs/state_counts.csv + outputs/per_cell.csv + outputs/params.csv
# and produces priming_cascade.png + pass_summary.csv.

from pathlib import Path
import csv
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path(__file__).parent / "outputs"

# ── Load params ────────────────────────────────────────────────────────────
params = {}
with (OUT_DIR / "params.csv").open() as f:
    for row in csv.DictReader(f):
        params[row["key"]] = float(row["value"])

sec_per_slice = params["PARAM_SEC_PER_SLICE"]
k_cd8         = params["PARAM_DC_PRIME_K_CD8"]
n_sites       = params["PARAM_DC_N_SITES"]
param_cell    = params["PARAM_CELL"]
div_burst     = int(params["PARAM_PRIME_DIV_BURST"])
N_PAIRS       = int(params["N_PAIRS"])

# Analytic priming probability per step (each T has 1 cDC1 neighbor,
# neighbor_all_count also = 1 since the lattice has only the paired DC).
# H = n_sites·1 / (n_sites·1 + 1 + PARAM_CELL)
H_analytic = n_sites / (n_sites + 1 + param_cell)
p_per_step = 1.0 - math.exp(-k_cd8 * H_analytic * sec_per_slice)

# ── Load state counts ──────────────────────────────────────────────────────
rows = []
with (OUT_DIR / "state_counts.csv").open() as f:
    for row in csv.DictReader(f):
        rows.append(row)

step       = np.array([int(r["step"]) for r in rows])
n_naive    = np.array([int(r["n_naive"]) for r in rows])
n_cyt      = np.array([int(r["n_cyt"])   for r in rows])
n_eff      = np.array([int(r["n_eff"])   for r in rows])
n_dc       = np.array([int(r["n_dc"])    for r in rows])
evt_prime  = np.array([int(r["evt_prime_cd8"]) for r in rows])

# ── Load per-cell step-0 data for divide_limit check ───────────────────────
primed_div_limits = []
with (OUT_DIR / "per_cell.csv").open() as f:
    for row in csv.DictReader(f):
        if int(row["step"]) == 0 and int(row["cell_state"]) == 1:  # T_CELL_CYT
            primed_div_limits.append(int(row["divide_limit"]))

# ── Pass criteria (all at step 0) ──────────────────────────────────────────
idx0 = np.where(step == 0)[0][0]
checks = []

checks.append((
    "priming_fires",
    "step 0: n_naive ≤ 2 (most primed on contact)",
    n_naive[idx0] <= 2,
    f"got n_naive={n_naive[idx0]}  (p_per_step_analytic={p_per_step:.4f})",
))

checks.append((
    "no_spurious_deaths",
    "step 0: n_naive + n_cyt == 25 (no T cell deaths)",
    n_naive[idx0] + n_cyt[idx0] == N_PAIRS,
    f"got n_naive+n_cyt={n_naive[idx0] + n_cyt[idx0]}",
))

checks.append((
    "event_matches_transitions",
    "step 0: EVT_PRIME_CD8 == n_cyt",
    int(evt_prime[idx0]) == int(n_cyt[idx0]),
    f"got evt={evt_prime[idx0]} n_cyt={n_cyt[idx0]}",
))

checks.append((
    "dc_stable",
    "step 0: n_dc == 25 (no DC deaths)",
    n_dc[idx0] == N_PAIRS,
    f"got n_dc={n_dc[idx0]}",
))

# divide_limit on primed cells should equal PARAM_PRIME_DIV_BURST (starting from 0).
ok_burst = (len(primed_div_limits) > 0
            and all(dl == div_burst for dl in primed_div_limits))
checks.append((
    "divide_burst_applied",
    f"primed T cells have divide_limit == PARAM_PRIME_DIV_BURST ({div_burst})",
    ok_burst,
    f"n_primed={len(primed_div_limits)}, "
    f"limits={set(primed_div_limits) if primed_div_limits else 'none'}",
))

# ── Write pass_summary.csv ─────────────────────────────────────────────────
with (OUT_DIR / "pass_summary.csv").open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["check", "description", "pass", "detail"])
    for name, desc, ok, detail in checks:
        w.writerow([name, desc, "PASS" if ok else "FAIL", detail])

all_pass = all(c[2] for c in checks)

# ── Figure: stacked T cell states + event counter ──────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

ax = axes[0]
ax.stackplot(
    step,
    n_naive, n_cyt, n_eff,
    labels=["naive", "cytotoxic", "effector"],
    colors=["#cbd5e8", "#66c2a5", "#8da0cb"],
    alpha=0.85,
)
ax.set_xlabel("ABM step")
ax.set_ylabel("# T cells")
ax.set_ylim(0, N_PAIRS * 1.05)
ax.axhline(N_PAIRS, color="k", ls=":", lw=0.6, alpha=0.5)
ax.legend(loc="upper right", fontsize=9)
ax.set_title(f"T cell state  (p_per_step analytic = {p_per_step:.4f})")

ax = axes[1]
ax.plot(step, evt_prime, marker="o", lw=2, color="#d62728", label="EVT_PRIME_CD8")
ax.plot(step, n_cyt, marker="s", lw=1, ls="--", color="#66c2a5",
        label="n_cyt (should match)")
ax.set_xlabel("ABM step")
ax.set_ylabel("count")
ax.set_ylim(0, N_PAIRS * 1.1)
ax.legend(loc="lower right", fontsize=9)
ax.set_title("Priming events vs primed population")

status = "PASS" if all_pass else "FAIL"
fig.suptitle(f"Test 8 — DC priming cascade  [{status}]", fontsize=12)
fig.tight_layout()
fig.savefig(OUT_DIR / "priming_cascade.png", dpi=120)
plt.close(fig)

# ── Console summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Test 8 — DC priming cascade  :  {status}")
print("=" * 60)
for name, desc, ok, detail in checks:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:<28} {desc}")
    print(f"           {detail}")
print(f"\n  Figure:  {OUT_DIR/'priming_cascade.png'}")
print(f"  Summary: {OUT_DIR/'pass_summary.csv'}")

sys.exit(0 if all_pass else 1)
