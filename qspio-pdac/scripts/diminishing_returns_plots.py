#!/usr/bin/env python3
"""Diminishing returns analysis for submodel target extraction.

Two plots:
1. Cumulative impact curve — impact score vs. number of parameters extracted
2. Contraction efficiency — contraction % vs. extraction order (by target count)
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qsp_inference.audit.report import (
    AuditConfig,
    _score_params,
    load_compare_results,
    load_parameter_groups,
    load_prcc,
    load_priors,
    load_submodel_targets,
)

ROOT = Path(__file__).resolve().parent.parent
config = AuditConfig(project_root=ROOT)

# Load data
priors = load_priors(config.priors_csv)
targets = load_submodel_targets(config.submodel_dir)
groups = load_parameter_groups(config.param_groups)
prcc = load_prcc(config.prcc_csv)
compare = load_compare_results(config.compare_cache, priors=priors)

scored = _score_params(priors, targets, groups, prcc, compare, cross_model=None)

# ── Plot 1: Cumulative impact curve ──────────────────────────────────────────

# Only uncovered parameters (no existing targets)
uncovered = [s for s in scored if s["n_targets"] == 0]
scores = [s["score"] for s in uncovered]
cumulative = np.cumsum(scores)
total = cumulative[-1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cumulative) + 1), cumulative, "k-", linewidth=1.5)
ax.fill_between(range(1, len(cumulative) + 1), cumulative, alpha=0.1, color="steelblue")

# Mark elbow candidates
for n in [5, 10, 20]:
    if n <= len(cumulative):
        pct = cumulative[n - 1] / total * 100
        ax.axvline(n, color="gray", linestyle="--", alpha=0.5)
        ax.annotate(
            f"{n} params → {pct:.0f}%",
            (n, cumulative[n - 1]),
            textcoords="offset points",
            xytext=(8, -5),
            fontsize=9,
        )

ax.set_xlabel("Number of additional parameters extracted")
ax.set_ylabel("Cumulative impact score")
ax.set_title("Diminishing returns: cumulative impact of uncovered parameters")
ax.set_xlim(1, len(cumulative))
fig.tight_layout()
fig.savefig(ROOT / "figures" / "diminishing_returns_cumulative.png", dpi=150)
print(f"Saved cumulative impact plot")

# ── Plot 2: Contraction efficiency ───────────────────────────────────────────

# All parameters that have posteriors
covered = []
for s in scored:
    name = s["name"]
    if name in compare.get("parameters", {}):
        joint = compare["parameters"][name].get("joint") or {}
        contraction = joint.get("contraction")
        if contraction is not None:
            covered.append({
                "name": name,
                "contraction": contraction,
                "n_targets": s["n_targets"],
                "type": s["type"],
                "prcc_rank": s["prcc_rank"],
            })

# Sort by number of targets (proxy for extraction effort)
covered.sort(key=lambda x: x["n_targets"], reverse=True)

fig2, ax2 = plt.subplots(figsize=(10, 5))

contractions = [c["contraction"] * 100 for c in covered]
names = [c["name"] for c in covered]
n_tgts = [c["n_targets"] for c in covered]
colors = ["#d62728" if c < 0 else "#2ca02c" if c >= 50 else "#1f77b4" for c in contractions]

bars = ax2.bar(range(len(contractions)), contractions, color=colors, width=0.8, edgecolor="none")
ax2.axhline(0, color="black", linewidth=0.5)
ax2.axhline(50, color="gray", linestyle="--", alpha=0.4, label="50% contraction")

# Label the worst offenders (negative contraction)
for i, c in enumerate(covered):
    if c["contraction"] < -0.3:
        ax2.annotate(
            c["name"].replace("_", "\n", 1),
            (i, contractions[i]),
            fontsize=6,
            ha="center",
            va="top",
            rotation=0,
        )

# Add target count as secondary info
for i, nt in enumerate(n_tgts):
    if nt > 0:
        ax2.text(i, max(contractions[i], 0) + 2, str(nt), ha="center", va="bottom", fontsize=6, color="gray")

ax2.set_ylabel("Contraction (%)")
ax2.set_title("Posterior contraction by parameter (sorted by target count, labeled)")
ax2.set_xlabel("Parameters (sorted by # targets, descending)")
ax2.legend(loc="upper right")

# Suppress x ticks (too many names)
ax2.set_xticks([])
ax2.set_xlim(-0.5, len(contractions) - 0.5)

fig2.tight_layout()
fig2.savefig(ROOT / "figures" / "diminishing_returns_contraction.png", dpi=150)
print(f"Saved contraction efficiency plot")

# ── Summary stats ────────────────────────────────────────────────────────────

print(f"\n{'─'*60}")
print(f"Uncovered parameters: {len(uncovered)}")
print(f"Top 5 capture {cumulative[4]/total*100:.0f}% of remaining impact")
print(f"Top 10 capture {cumulative[9]/total*100:.0f}% of remaining impact")
print(f"Top 20 capture {cumulative[19]/total*100:.0f}% of remaining impact")

neg = [c for c in covered if c["contraction"] < 0]
print(f"\nParameters with negative contraction: {len(neg)}")
for c in sorted(neg, key=lambda x: x["contraction"]):
    print(f"  {c['name']}: {c['contraction']*100:.0f}% ({c['n_targets']} targets)")

plt.show()
