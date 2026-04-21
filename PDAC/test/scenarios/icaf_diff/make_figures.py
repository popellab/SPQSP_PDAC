#!/usr/bin/env python3
# Test 9 — iCAF / myCAF Differentiation figures + pass/fail summary.
#
# Reads three sibling output dirs (outputs_il1/, outputs_tgfb/, outputs_both/)
# from each sub-scenario and produces icaf_diff.png + pass_summary.csv.

from pathlib import Path
import csv
import sys

import numpy as np
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RUNS = [
    ("il1",  "IL1 high, TGFB=0",    "iCAF dominant"),
    ("tgfb", "TGFB high, IL1=0",    "myCAF dominant"),
    ("both", "Both high",           "TGFB suppression + detailed balance"),
]

N_TOTAL = 100

# ── Load all three runs ────────────────────────────────────────────────────
runs = {}
for key, _, _ in RUNS:
    path = HERE / f"outputs_{key}" / "counts.csv"
    if not path.exists():
        print(f"[FAIL] missing {path}", file=sys.stderr)
        sys.exit(1)
    rows = list(csv.DictReader(path.open()))
    step = np.array([int(r["step"]) for r in rows])
    n_q  = np.array([int(r["n_quiescent"]) for r in rows])
    n_my = np.array([int(r["n_mycaf"])     for r in rows])
    n_i  = np.array([int(r["n_icaf"])      for r in rows])
    runs[key] = dict(step=step, q=n_q, my=n_my, i=n_i)

# ── Pass criteria ──────────────────────────────────────────────────────────
checks = []

# Run 1: IL1 only → ≥90% iCAF at final step
final_il1 = runs["il1"]
frac_icaf_il1 = final_il1["i"][-1] / N_TOTAL
checks.append((
    "il1_drives_icaf",
    "IL1 run: final n_iCAF ≥ 90 (TGFB absent → iCAF path wins)",
    frac_icaf_il1 >= 0.90,
    f"iCAF={final_il1['i'][-1]}, myCAF={final_il1['my'][-1]}, q={final_il1['q'][-1]}",
))

# Run 2: TGFB only → ≥90% myCAF at final step
final_tg = runs["tgfb"]
frac_mycaf_tg = final_tg["my"][-1] / N_TOTAL
checks.append((
    "tgfb_drives_mycaf",
    "TGFB run: final n_myCAF ≥ 90 (IL1 absent → myCAF path wins)",
    frac_mycaf_tg >= 0.90,
    f"myCAF={final_tg['my'][-1]}, iCAF={final_tg['i'][-1]}, q={final_tg['q'][-1]}",
))

# Run 3: Both → myCAF > iCAF (TGFB suppresses quiescent→iCAF; interconversion
# gives detailed-balance ratio ≈ 2:1 myCAF:iCAF from rate ratio 0.025/0.0125).
# Accept any myCAF-dominant mixed outcome with both populations present.
final_b = runs["both"]
m_b, i_b = final_b["my"][-1], final_b["i"][-1]
checks.append((
    "both_mycaf_dominant_mixed",
    "Both run: myCAF > iCAF and both > 0 (TGFB supp + detailed balance)",
    (m_b > i_b) and (m_b > 0) and (i_b > 0),
    f"myCAF={m_b}, iCAF={i_b}, ratio myCAF:iCAF={m_b/max(i_b,1):.2f}",
))

# Run 3 detailed-balance check: with k_i2m=0.025, k_m2i=0.0125 at saturation,
# expected f_iCAF = k_m2i/(k_m2i+k_i2m) = 1/3. Allow wide tolerance.
exp_frac_icaf_both = 0.0125 / (0.0125 + 0.025)  # = 1/3
obs_frac_icaf_both = i_b / N_TOTAL
within_range = abs(obs_frac_icaf_both - exp_frac_icaf_both) < 0.15
checks.append((
    "both_detailed_balance",
    f"Both run: iCAF fraction within ±0.15 of {exp_frac_icaf_both:.2f} "
    f"(k_m2i/(k_m2i+k_i2m))",
    within_range,
    f"observed f_iCAF={obs_frac_icaf_both:.2f}, expected≈{exp_frac_icaf_both:.2f}",
))

# No quiescent cells left after 100 steps in any run (activation must fire).
for key, label, _ in RUNS:
    q_final = runs[key]["q"][-1]
    checks.append((
        f"{key}_no_quiescent_left",
        f"{label}: n_quiescent == 0 at final step (activation fired)",
        q_final == 0,
        f"q={q_final}",
    ))

# ── Write pass_summary.csv ─────────────────────────────────────────────────
summary_path = HERE / "pass_summary.csv"
with summary_path.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["check", "description", "pass", "detail"])
    for name, desc, ok, detail in checks:
        w.writerow([name, desc, "PASS" if ok else "FAIL", detail])

all_pass = all(c[2] for c in checks)

# ── Figure: three trajectory panels + final-state bars ─────────────────────
fig, axes = plt.subplots(1, 4, figsize=(15, 4.2))

colors = {"q": "#cbd5e8", "my": "#e78ac3", "i": "#66c2a5"}

for ax, (key, label, subtitle) in zip(axes[:3], RUNS):
    r = runs[key]
    ax.stackplot(
        r["step"], r["q"], r["my"], r["i"],
        labels=["quiescent", "myCAF", "iCAF"],
        colors=[colors["q"], colors["my"], colors["i"]],
        alpha=0.85,
    )
    ax.set_xlabel("ABM step")
    ax.set_ylabel("# fibroblasts")
    ax.set_ylim(0, N_TOTAL * 1.05)
    ax.axhline(N_TOTAL, color="k", ls=":", lw=0.6, alpha=0.5)
    ax.legend(loc="center right", fontsize=8)
    ax.set_title(f"{label}\n{subtitle}", fontsize=10)

# Final-state bar chart
ax = axes[3]
x = np.arange(3)
w = 0.25
my_vals = [runs[k]["my"][-1] for k, _, _ in RUNS]
ic_vals = [runs[k]["i"][-1]  for k, _, _ in RUNS]
q_vals  = [runs[k]["q"][-1]  for k, _, _ in RUNS]
ax.bar(x - w, q_vals,  w, color=colors["q"], label="quiescent")
ax.bar(x,      my_vals, w, color=colors["my"], label="myCAF")
ax.bar(x + w,  ic_vals, w, color=colors["i"], label="iCAF")
ax.set_xticks(x)
ax.set_xticklabels([k for k, _, _ in RUNS])
ax.set_ylabel("# fibroblasts (final step)")
ax.set_ylim(0, N_TOTAL * 1.05)
ax.legend(fontsize=8)
ax.set_title(f"Final-state counts (step {RUN_STEPS if False else runs['il1']['step'][-1]})")

status = "PASS" if all_pass else "FAIL"
fig.suptitle(f"Test 9 — iCAF / myCAF differentiation  [{status}]", fontsize=12)
fig.tight_layout()
fig.savefig(HERE / "icaf_diff.png", dpi=120)
plt.close(fig)

# ── Console summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"Test 9 — iCAF / myCAF differentiation  :  {status}")
print("=" * 60)
for name, desc, ok, detail in checks:
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name:<30} {desc}")
    print(f"           {detail}")
print(f"\n  Figure:  {HERE/'icaf_diff.png'}")
print(f"  Summary: {summary_path}")

sys.exit(0 if all_pass else 1)
