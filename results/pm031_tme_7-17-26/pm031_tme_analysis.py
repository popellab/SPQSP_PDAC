#!/usr/bin/env python3
# %% [markdown]
# # PM-031 presim-TME coverage sweep — abundance-fraction analysis
#
# **Question (coverage, not inference):** across a deliberately broad 322-d sweep of
# ABM/QSP parameters (patient identity fixed to PM-031), does the ABM's grown
# pre-treatment TME **ever** reproduce PM-031's measured immune abundances — or do
# whole cell classes simply never appear?
#
# The real patient is cold and myeloid-dominant: M2 macrophages are the largest immune
# population (~0.10), CD8 T-cell states sit at ~0.001. The sweep's job is to show, per
# category, where the simulations landed and whether the patient value is reachable.
#
# FunCN spatial organization is deliberately **out of scope here** — first we establish
# whether the cells appear at all.
#
# Run as a script (`python pm031_tme_analysis.py`) or open as a notebook via jupytext.
# Regenerate the input first: `python build_composition.py --snaps <pull-dir>`.

# %%
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO / "abm-inference" / "src"))
from abm_inference.observables import compute_observables, MATCHED_CATEGORIES  # noqa: E402

IMMUNE = {"mac_M1", "mac_M2", "mdsc", "treg_th", "treg_treg", "treg_tfh",
          "tcell_eff", "tcell_cyt", "tcell_sup", "bcell_naive", "bcell_act",
          "dc_mature_any", "dc_immature_any", "vas_phalanx"}
PM031_CSV = _REPO / "data/PASS01/neighborhoods/100-PM-031_cells.csv"

# %% [markdown]
# ## Load: ABM sweep compositions + PM-031's measured composition

# %%
d = np.load(_HERE / "composition.npz", allow_pickle=True)
cats = list(d["categories"])
comp_stop = d["comp_stop"]          # (N, 19) presim=stop = grown pre-treatment TME
comp_zero = d["comp_zero"]          # (N, 19) presim=0 = seeding
n_stop = d["n_stop"]                # matched mid-z cell count per snapshot
n_theta = len(d["theta_idx"])

# Judge coverage on PROPERLY-GROWN TMEs only. ~2% of runs are near-extinct (<500 cells),
# where a handful of cells makes a single category 100% of the slice -- these produce
# spurious max~1.0 "coverage" for immune categories. Require >=2000 mid-z cells (95% of
# runs clear this; the grown median is ~10k). Verdicts use p95/p99, never max.
MIN_CELLS = 2000
valid = ~np.isnan(comp_stop).all(axis=1)
grown = valid & (n_stop >= MIN_CELLS)
S = np.nan_to_num(comp_stop[grown])           # (Ng, 19)
print(f"{n_theta} theta cached; {valid.sum()} valid; {grown.sum()} properly grown "
      f"(>={MIN_CELLS} mid-z cells) -> used for coverage")

# PM-031 measured (the target line on every histogram)
patient = compute_observables(PM031_CSV)[:19]
assert len(patient) == len(cats)

# %% [markdown]
# ## The headline number: does immune EVER appear?

# %%
imm_ix = [cats.index(c) for c in IMMUNE]
imm_frac = S[:, imm_ix].sum(axis=1)                     # immune+vascular share per TME
print(f"grown presim=stop TMEs: {len(S)}")
print(f"  median total immune+vascular fraction:  {np.median(imm_frac):.4f}  "
      f"(PM-031 measured: {patient[imm_ix].sum():.4f})")
print(f"  TMEs with <1% immune+vascular:          {100*(imm_frac<0.01).mean():.1f}%")
# per-immune-category median across grown TMEs -- the 'typical TME' immune profile
print("  per-category MEDIAN across grown TMEs (the typical grown TME is immune-empty):")
for c in ["mac_M2", "dc_mature_any", "treg_th", "mdsc", "tcell_eff"]:
    print(f"      {c:16s} median {np.median(S[:, cats.index(c)]):.5f}   "
          f"patient {patient[cats.index(c)]:.4f}")

# %% [markdown]
# ## Per-category coverage table — verdict vs p95/p99 (NOT max; max is degenerate)
#
# `covered` = the sim p95 reaches the patient value. `tail-only` = only p99 does (the
# model gets there in <5% of runs). `MISS` = even p99 falls short — the model
# essentially cannot produce this patient's abundance. `STRUCTURAL-0` = category the
# rebuilt ABM can never emit (vascular agents were dropped; O2 is now a field).

# %%
STRUCTURAL_ZERO = {"vas_phalanx"}     # vascular agents removed in the recruitment rebuild
verdicts = {}
print(f"{'category':18s} {'patient':>8s} {'med':>8s} {'p95':>8s} {'p99':>8s}  verdict")
for i, c in enumerate(cats):
    col = S[:, i]
    pv = patient[i]
    med, p95, p99 = np.median(col), np.percentile(col, 95), np.percentile(col, 99)
    if c in STRUCTURAL_ZERO:
        v = "STRUCTURAL-0"
    elif p95 >= pv:
        v = "covered"
    elif p99 >= pv:
        v = "tail-only"
    else:
        v = "MISS"
    verdicts[c] = v
    star = "*" if c in IMMUNE else " "
    print(f"{star}{c:17s} {pv:8.4f} {med:8.4f} {p95:8.4f} {p99:8.4f}  {v}")

misses = [c for c, v in verdicts.items() if v == "MISS"]
print(f"\nMISS (model cannot reach patient abundance): {misses}")
print(f"STRUCTURAL-0 (uncoverable by construction): "
      f"{[c for c, v in verdicts.items() if v == 'STRUCTURAL-0']}")

# %% [markdown]
# ## Histogram grid — one panel per category, patient value marked
#
# Red line = PM-031 measured. If the red line sits to the right of the whole ABM
# distribution, the model never reaches that patient's abundance — that is a miss to
# diagnose. Immune/vascular categories are titled in red.

# %%
ncol = 4
nrow = int(np.ceil(len(cats) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
axes = axes.ravel()
for i, c in enumerate(cats):
    ax = axes[i]
    col = np.nan_to_num(S[:, i])
    pv = patient[i]
    # log-friendly: many fractions are near 0. Use a symlog-ish clip for the hist.
    ax.hist(col, bins=40, color="#4C78A8", alpha=0.85)
    ax.axvline(pv, color="#E45756", lw=2)
    ax.set_title(c, color="#E45756" if c in IMMUNE else "black", fontsize=10)
    ax.text(0.97, 0.9, f"patient {pv:.3f}", transform=ax.transAxes, ha="right",
            va="top", fontsize=7, color="#E45756")
    v = verdicts[c]
    if v in ("MISS", "STRUCTURAL-0"):
        ax.text(0.97, 0.72, v, transform=ax.transAxes, ha="right",
                va="top", fontsize=7, color="#E45756", weight="bold")
    ax.tick_params(labelsize=7)
for j in range(len(cats), len(axes)):
    axes[j].axis("off")
fig.suptitle(f"PM-031 sweep: ABM presim=stop abundance vs measured (n={len(S)} TMEs)",
             fontsize=13, y=1.002)
fig.tight_layout()
fig.savefig(_HERE / "abundance_histograms_7-17-26.png", dpi=130, bbox_inches="tight")
print("wrote abundance_histograms_7-17-26.png")

# %% [markdown]
# ## Immune-only zoom (log x) — the categories that matter for the diagnosis
#
# Immune fractions are tiny; a log x-axis shows whether the ABM produces *any* mass
# near the patient value or stays pinned at zero.

# %%
imm_cats = [c for c in cats if c in IMMUNE]
ncol = 4
nrow = int(np.ceil(len(imm_cats) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 2.6 * nrow))
axes = axes.ravel()
for i, c in enumerate(imm_cats):
    ax = axes[i]
    col = np.nan_to_num(S[:, cats.index(c)])
    pv = patient[cats.index(c)]
    pos = col[col > 0]
    frac_zero = 100 * (col == 0).mean()
    if len(pos):
        ax.hist(np.log10(pos), bins=30, color="#54A24B", alpha=0.85)
    if pv > 0:
        ax.axvline(np.log10(pv), color="#E45756", lw=2)
    ax.set_title(f"{c}  ({frac_zero:.0f}% of sims = 0)", fontsize=9,
                 color="#E45756" if col.max() < pv else "black")
    ax.set_xlabel("log10 fraction (nonzero sims)", fontsize=7)
    ax.tick_params(labelsize=7)
for j in range(len(imm_cats), len(axes)):
    axes[j].axis("off")
fig.suptitle("Immune/vascular abundance (log): does the ABM produce ANY mass near the "
             "patient value?", fontsize=12, y=1.003)
fig.tight_layout()
fig.savefig(_HERE / "abundance_immune_log_7-17-26.png", dpi=130, bbox_inches="tight")
print("wrote abundance_immune_log_7-17-26.png")
