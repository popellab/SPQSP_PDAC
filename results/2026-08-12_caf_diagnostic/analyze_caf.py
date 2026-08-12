"""CAF sub-state diagnostic: QSP target vs ABM trajectory."""
import sys, pathlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

audit_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/class_audit_42.csv"
out_dir = pathlib.Path(sys.argv[2]) if len(sys.argv) > 2 else pathlib.Path(".")

df = pd.read_csv(audit_path)

df["day"] = df["step"] * 0.25

main = df.copy()

substates = [
    ("qPSC",  "tgt.qPSC",  "abm.qPSC",  "#2196F3"),
    ("iCAF",  "tgt.iCAF",  "abm.iCAF",  "#FF9800"),
    ("myCAF", "tgt.myCAF", "abm.myCAF", "#E91E63"),
    ("apCAF", "tgt.apCAF", "abm.apCAF", "#9C27B0"),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
axes = axes.ravel()

for ax, (name, tgt_col, abm_col, color) in zip(axes, substates):
    ax.plot(main["day"], main[tgt_col], "--", color="gray", lw=1.5, label="QSP target (f·V_T)")
    ax.plot(main["day"], main[abm_col], "-", color=color, lw=2, label="ABM count")
    ax.set_ylabel("Cell count")
    ax.set_title(name)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-2].set_xlabel("Day")
axes[-1].set_xlabel("Day")
fig.suptitle("Fibroblast sub-state tracking: QSP target vs ABM", fontsize=14, y=0.98)
fig.tight_layout(rect=[0, 0, 1, 0.95])

out_path = out_dir / "caf_substate_tracking.png"
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")

# Summary statistics at final step
print("\n=== Final step summary ===")
last = df.iloc[-1]
total_tgt = last["tgt.iCAF"] + last["tgt.myCAF"] + last["tgt.apCAF"]
total_abm = last["abm.iCAF"] + last["abm.myCAF"] + last["abm.apCAF"]
qpsc_tgt = last["tgt.qPSC"]
qpsc_abm = last["abm.qPSC"]
print(f"Step {int(last['step'])}, day {last['step']*0.25:.1f}")
print(f"{'Sub-state':<10} {'QSP tgt':>10} {'ABM':>10} {'Ratio':>10} {'QSP %':>10} {'ABM %':>10}")
for name, tgt_col, abm_col, _ in substates:
    t, a = last[tgt_col], last[abm_col]
    ratio = a / t if t > 0 else float("inf")
    t_pct = 100 * t / (total_tgt + qpsc_tgt) if (total_tgt + qpsc_tgt) > 0 else 0
    a_pct = 100 * a / (total_abm + qpsc_abm) if (total_abm + qpsc_abm) > 0 else 0
    print(f"{name:<10} {t:10.1f} {a:10.0f} {ratio:10.2f} {t_pct:9.1f}% {a_pct:9.1f}%")
print(f"{'CAF TOTAL':<10} {total_tgt:10.1f} {total_abm:10.0f} {total_abm/total_tgt:.2f}")
print(f"{'ALL FIB':<10} {total_tgt+qpsc_tgt:10.1f} {total_abm+qpsc_abm:10.0f}")

# Composition trajectory plot
fig2, ax2 = plt.subplots(figsize=(10, 5))
bottom = np.zeros(len(main))
colors = ["#2196F3", "#FF9800", "#E91E63", "#9C27B0"]
for (name, _, abm_col, color) in substates:
    vals = main[abm_col].values
    ax2.fill_between(main["day"].values, bottom, bottom + vals, alpha=0.7, color=color, label=name)
    bottom += vals

bottom_tgt = np.zeros(len(main))
for (name, tgt_col, _, color) in substates:
    vals = main[tgt_col].values
    ax2.plot(main["day"].values, bottom_tgt + vals, "--", color=color, lw=1, alpha=0.5)
    bottom_tgt += vals
ax2.plot(main["day"].values, bottom_tgt, "k--", lw=1.5, label="QSP total", alpha=0.5)

ax2.set_xlabel("Day")
ax2.set_ylabel("Cell count")
ax2.set_title("Fibroblast composition: ABM (filled) vs QSP target (dashed)")
ax2.legend(fontsize=8, loc="upper right")
ax2.grid(True, alpha=0.3)

out_path2 = out_dir / "caf_composition_stack.png"
fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path2}")
