"""IMC ↔ ABM category vocabulary for the FunCN observable.

Lifted from ``python/sweep/notebooks/match_imc_patients.ipynb`` (the
authoritative source) so the SBI observable and the PASS01 cleanup share one
definition. Two sides must agree on a single matched category set:

- **IMC side** (``TARGET_MAP`` / ``SOURCE_MAP``): map each cell's
  ``(cell_type, subtype)`` to an ABM target category, and each precomputed
  ``nbhd_*`` density column to ABM source categories.
- **ABM side** (``ABM_TO_MATCHED``): map each ``PDAC.tools.io.add_category``
  ``type_state`` label to the same matched category, applying the locked
  "collapse unmapped subtype → nearest IMC class" rule.

``MATCHED_CATEGORIES`` is **frozen** here (not recomputed per dataset as the
notebook does) so the ABM and IMC observable vectors always share one shape.
The 19 categories are the notebook's ``kept_targets`` (count>0 across PASS01).

PASS01 cleanup must emit ``cell_type``/``subtype`` values using exactly the
keys of ``TARGET_MAP`` — any renamed subtype silently drops cells.

LABEL SET v2 (Aug 2026, stage-3 retyping)
-----------------------------------------
``imc_pipeline/stage3_type.py`` re-derived the labels from the expert-validated
HALO ``<marker>+`` flags. Three consequences land here:

* **MATCHED_CATEGORIES 19 → 17.** Dropped ``tcell_cyt``/``tcell_sup`` (no
  granzyme/perforin in the panel) and ``treg_tfh`` (no CXCR5); added
  ``fib_apCAF`` (CAF & ``MHC2mem+``, the phi_4 coupling site).
* **``tcell_eff`` now means ALL CD8 and ``treg_th`` means all non-Treg CD4.**
  The names are kept for stability, but post-v2 the panel cannot subtype CD8 at
  all, so both sides must compare CD8-total to CD8-total: the ABM's cyt/sup
  states LUMP into ``tcell_eff``, and ``treg_tfh`` lumps into ``treg_th``.
  Lumping, not dropping — dropping would delete ABM cells that the IMC side
  really did measure, just without a subtype.
* **Seeding policy: DROP, do not invent** (see ``SEED_DROP``). An IMC label with
  no ABM equivalent places no agent, and no unclassified cell is ever recruited
  to fill an ABM state the data did not measure. Missing states are left to the
  ABM's own recruitment/transition dynamics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# ── IMC (cell_type, subtype) → ABM target category ──────────────────────────
TARGET_MAP: dict[tuple[str, str], str] = {
    ("cancer", "proliferating"): "cancer_prog",
    ("cancer", "quiescent"): "cancer_sen",
    ("fibroblast", "myCAF"): "fib_myCAF",
    ("fibroblast", "iCAF"): "fib_iCAF",
    ("fibroblast", "apCAF"): "fib_apCAF",
    # activated_CAF (aSMA+ & FAP+) has no 1:1 ABM state. Collapsed to its nearest
    # class, the same Decision-6 convention ABM_TO_MATCHED uses in the other
    # direction (fib_FRC → fib_quiescent), and the merge python/sweep/
    # funcn_subtype_grid.py already applies to nbhd_fib_activated_CAF. This is a
    # LABELLED subtype finding its nearest state — not an unclassified cell being
    # recruited to fill a gap, which SEED_DROP forbids.
    ("fibroblast", "activated_CAF"): "fib_myCAF",
    ("fibroblast", "fibroblast_other"): "fib_quiescent",
    ("macrophage", "m1"): "mac_M1",
    ("macrophage", "m2"): "mac_M2",
    ("mdsc", "m_mdsc"): "mdsc",
    ("mdsc", "pmn_mdsc"): "mdsc",
    ("mdsc", "mdsc_other"): "mdsc",
    ("cd4_t", "th"): "treg_th",
    ("cd4_t", "treg"): "treg_treg",
    # v2: the panel cannot subtype CD8 (no granzyme/perforin), so stage 3 emits
    # every CD8 cell as "effector" and this key carries CD8-total.
    ("cd8_t", "effector"): "tcell_eff",
    ("b_cell", "naive"): "bcell_naive",
    ("b_cell", "activated"): "bcell_act",
    ("endothelial", "phalanx"): "vas_phalanx",
    # DCs collapsed — IMC panel can't distinguish cDC1/cDC2.
    ("dendritic", "mature"): "dc_mature_any",
    ("dendritic", "immature"): "dc_immature_any",
}

# ── IMC label → placed nothing, and WHY (seeding policy: drop, never invent) ─
# Every v2 (cell_type, subtype) is either in TARGET_MAP or listed here; a label in
# neither is a bug, so seeding code must assert the union covers the data rather
# than let an unrecognised label vanish. Dropped cells stay in the density/FunCN
# DENOMINATOR as real cells (they are real) — they simply place no ABM agent.
SEED_DROP: dict[tuple[str, str], str] = {
    ("unclassified", "none"): "segmentation failure, type unrecoverable (~24% of cells)",
    ("artifact", "none"): "HALO artifact object, not a cell",
    ("other_immune", "none"): "CD45+ but no lineage flag; no ABM state to place it in",
    ("t_other", "none"): "CD3+ but CD4/CD8 ambiguous; ABM has no such T state",
    ("nk", "none"): "ABM has no NK agent type",
    ("neutrophil", "none"): "ABM has no neutrophil agent type",
}

# ── IMC nbhd_* column → ABM source category(ies) (summed if >1) ─────────────
# ⚠️ The shipped ``nbhd_*`` columns were computed from the v1 labels and are STALE
# under v2 — stage 4 must regenerate them before this map is used. The v1-only
# columns (nbhd_cd8_cytotoxic / _suppressed, nbhd_cd4_tfh) are folded into their
# v2 parents rather than dropped, so cell mass is preserved if a v1 file is read.
SOURCE_MAP: dict[str, list[str]] = {
    "nbhd_cancer_proliferating": ["cancer_prog"],
    "nbhd_cancer_quiescent": ["cancer_sen"],
    "nbhd_bcells_naive": ["bcell_naive"],
    "nbhd_bcells_activated": ["bcell_act"],
    "nbhd_cd4_th": ["treg_th"],
    "nbhd_cd4_treg": ["treg_treg"],
    "nbhd_cd4_tfh": ["treg_th"],            # v1 only; tfh pruned into th
    "nbhd_cd8_effector": ["tcell_eff"],
    "nbhd_cd8_cytotoxic": ["tcell_eff"],    # v1 only; CD8 not subtypeable
    "nbhd_cd8_suppressed": ["tcell_eff"],   # v1 only; CD8 not subtypeable
    "nbhd_mdsc_pmn": ["mdsc"],
    "nbhd_mdsc_m": ["mdsc"],
    "nbhd_mdsc_other": ["mdsc"],
    "nbhd_mac_m1": ["mac_M1"],
    "nbhd_mac_m2": ["mac_M2"],
    "nbhd_vas_phalanx": ["vas_phalanx"],
    "nbhd_fib_myCAF": ["fib_myCAF"],
    "nbhd_fib_iCAF": ["fib_iCAF"],
    "nbhd_fib_apCAF": ["fib_apCAF"],
    "nbhd_fib_activated_CAF": ["fib_myCAF"],
    "nbhd_fib_other": ["fib_quiescent"],
    "nbhd_dc_mature": ["dc_mature_any"],
    "nbhd_dc_immature": ["dc_immature_any"],
    # patient-only sources without a clean ABM equivalent are dropped:
    #   nbhd_nk, nbhd_neut
}

# ── Aggregate matched categories → underlying fine-grained ABM labels ───────
# When computing the ABM FunCN, an aggregate category expands to these and the
# sources are SUMMED / targets are averaged (matches the patient-side sum).
SOURCE_AGG: dict[str, list[str]] = {
    "dc_mature_any": ["dc_cDC1_mature", "dc_cDC2_mature"],
    "dc_immature_any": ["dc_cDC1_immature", "dc_cDC2_immature"],
}
TARGET_AGG: dict[str, list[str]] = dict(SOURCE_AGG)

# ── Frozen matched category set — v2: 17 (was 19) ───────────────────────────
# -tcell_cyt -tcell_sup (no granzyme/perforin in panel) -treg_tfh (no CXCR5)
# +fib_apCAF (CAF & MHC2mem+). ``tcell_eff`` now carries CD8-total and
# ``treg_th`` all non-Treg CD4 — see the LABEL SET v2 note in the module docstring.
MATCHED_CATEGORIES: list[str] = [
    "bcell_act", "bcell_naive",
    "cancer_prog", "cancer_sen",
    "dc_immature_any", "dc_mature_any",
    "fib_apCAF", "fib_iCAF", "fib_myCAF", "fib_quiescent",
    "mac_M1", "mac_M2",
    "mdsc",
    "tcell_eff",
    "treg_th", "treg_treg",
    "vas_phalanx",
]

# ── ABM io.add_category(type_state) label → matched category (None = drop) ──
# Decision-6 collapses (unmapped ABM subtype → nearest IMC class):
#   cancer_stem → cancer_prog (stem is proliferating-like)
#   vas_hev / vas_tip / vas_collapsed → vas_phalanx (only vascular cat kept)
#   bcell_plasma → bcell_act (nearest activated B state)
#   fib_FRC → fib_quiescent
# Subtypes present in TARGET_MAP but dropped from the set (low IMC counts) are
# dropped on the ABM side too for symmetry: tcell_naive, treg_tcd4_naive.
# v2 LUMPS (the panel cannot subtype these, so compare total-to-total):
#   tcell_cyt, tcell_sup → tcell_eff ;  treg_tfh → treg_th
ABM_TO_MATCHED: dict[str, str | None] = {
    "cancer_stem": "cancer_prog",
    "cancer_prog": "cancer_prog",
    "cancer_sen": "cancer_sen",
    "tcell_eff": "tcell_eff",
    "tcell_cyt": "tcell_eff",
    "tcell_sup": "tcell_eff",
    "tcell_naive": None,
    "treg_th": "treg_th",
    "treg_treg": "treg_treg",
    "treg_tfh": "treg_th",
    "treg_tcd4_naive": None,
    "mdsc": "mdsc",
    "mac_M1": "mac_M1",
    "mac_M2": "mac_M2",
    "fib_quiescent": "fib_quiescent",
    "fib_myCAF": "fib_myCAF",
    "fib_iCAF": "fib_iCAF",
    "fib_apCAF": "fib_apCAF",
    "fib_FRC": "fib_quiescent",
    "vas_tip": "vas_phalanx",
    "vas_phalanx": "vas_phalanx",
    "vas_collapsed": "vas_phalanx",
    "vas_hev": "vas_phalanx",
    "bcell_naive": "bcell_naive",
    "bcell_act": "bcell_act",
    "bcell_plasma": "bcell_act",
    "dc_cDC1_immature": "dc_immature_any",
    "dc_cDC2_immature": "dc_immature_any",
    "dc_cDC1_mature": "dc_mature_any",
    "dc_cDC2_mature": "dc_mature_any",
}


# ── 15-category COMMON taxonomy (production pre-treatment observable) ───────
# IHC-anchored vocabulary shared by the ABM, QSP, and IHC composition blocks so
# all three views are directly comparable (see docs/production_observable_spec.md).
# QSP-comparability lumps vs MATCHED_CATEGORIES: tcell_eff → CD8_active,
# bcell_naive+bcell_act → B_lineage; vas_phalanx dropped (no QSP equivalent,
# structural coverage-0).
# v2: tcell_sup and treg_tfh removed (the IMC panel cannot call either, so
# keeping them would compare an ABM/QSP number against a structural zero);
# fib_apCAF added, which also resolves the old ⚠️ apCAF→fib_myCAF default below.
COMMON_CATEGORIES: list[str] = [
    "cancer_prog", "cancer_sen",
    "fib_apCAF", "fib_iCAF", "fib_myCAF", "fib_quiescent",
    "CD8_active",
    "treg_th", "treg_treg",
    "B_lineage",
    "mac_M1", "mac_M2",
    "mdsc",
    "dc_immature_any", "dc_mature_any",
]

# MATCHED (17) → COMMON (15); None drops the category (vas).
MATCHED_TO_COMMON: dict[str, str | None] = {c: c for c in MATCHED_CATEGORIES}
MATCHED_TO_COMMON.update({
    "tcell_eff": "CD8_active",
    "bcell_naive": "B_lineage", "bcell_act": "B_lineage",
    "vas_phalanx": None,
})

# QSP V_T.<species> → COMMON category. Two ⚠️ rows remain (C_x→cancer_sen,
# Th_exh→treg_th) as the spec's low-stakes defaults — confirm biologically. The
# old ⚠️ apCAF→fib_myCAF default is RESOLVED: the IMC can now call apCAF directly
# (CAF & MHC2mem+), so the QSP species maps to its own category.
# Systemic-only QSP pools (naive B/T) have no V_T column, so they never appear here.
QSP_TO_COMMON: dict[str, str] = {
    "C1": "cancer_prog", "C2": "cancer_prog", "C_x": "cancer_sen",      # ⚠️ C_x
    "qPSC": "fib_quiescent", "iCAF": "fib_iCAF",
    "myCAF": "fib_myCAF", "apCAF": "fib_apCAF",
    # v2: the panel cannot separate exhausted from active CD8, so CD8_exh lumps
    # into CD8_active rather than into a category the IMC scores as structurally 0.
    "CD8": "CD8_active", "CD8_exh": "CD8_active",
    "Th": "treg_th", "Th_exh": "treg_th",                               # ⚠️ Th_exh
    "Treg": "treg_treg", "Tfh": "treg_th",
    "B": "B_lineage", "Plasma": "B_lineage",
    "Mac_M1": "mac_M1", "Mac_M2": "mac_M2", "MDSC": "mdsc",
    "cDC1": "dc_immature_any", "cDC2": "dc_immature_any",
    "mcDC1": "dc_mature_any", "mcDC2": "dc_mature_any",
}


# QSP V_T.<species> → MATCHED category, for §C6.1 placement seeding (counts come from
# f·X_QSP, arrangement from IMC). This is QSP_TO_COMMON refined in exactly two places,
# because placement needs the finer MATCHED vocabulary that CATEGORY_TO_AGENT keys on:
#
#   CD8_active  → tcell_eff   (rename only; MATCHED has no exhausted CD8 category, so the
#                              placed pool is CD8 + CD8_exh and the EFF/SUPP split is applied
#                              downstream from ``QSP_SUBSTATE_SPLIT`` — the panel cannot call
#                              it, but the QSP can, so it is derived, never invented)
#   B_lineage   → bcell_naive / bcell_act, splitting on the QSP's own two species. Plasma
#                 goes to the bcell_act CATEGORY (that is where ABM_TO_MATCHED puts plasma
#                 when collapsing the other way) but is then placed in the ABM's PLASMA
#                 state via QSP_SUBSTATE_SPLIT — not the ACTIVATED state. Inverting
#                 ABM_TO_MATCHED naively would seed a QSP plasma cell as an activated B
#                 cell, a different cell with different behaviour, and the class audit
#                 shows it immediately as abm.Plasma=0 against a positive tgt.Plasma.
#                 The QSP has no activated-B species in the tumour at all, so the whole
#                 bcell_act pool is plasma; the IMC's activated-B call informs arrangement,
#                 while the QSP owns the total, exactly as C6.1 specifies.
#
# vas_phalanx has NO QSP species and is deliberately absent: ``seeding.counts_from_qsp``
# falls back to the data ratio for categories the QSP does not supply, which is the correct
# behaviour for a compartment the ODE never modelled (and D-4 makes it a field, not agents).
QSP_TO_MATCHED: dict[str, str] = {
    **{k: v for k, v in QSP_TO_COMMON.items() if v not in ("CD8_active", "B_lineage")},
    "CD8": "tcell_eff", "CD8_exh": "tcell_eff",
    "B": "bcell_naive", "Plasma": "bcell_act",
}

# Sub-states the IMC panel cannot resolve but the QSP tracks separately. Placement puts the
# whole MATCHED pool in the FIRST state unless a split is supplied; supplying one from the
# onset dump keeps the seeded ABM consistent with the QSP species the class audit compares it
# against (otherwise the ABM starts with 0 exhausted CD8 against a positive V_T_CD8_exh).
# Values are (matched category, [(ABM cell_state, QSP species)]) with states from common.cuh.
QSP_SUBSTATE_SPLIT: dict[str, list[tuple[int, str]]] = {
    "tcell_eff": [(0, "CD8"), (2, "CD8_exh")],   # T_CELL_EFF, T_CELL_SUPP
    "bcell_act": [(2, "Plasma")],                # BCELL_PLASMA (see B_lineage note above)
}


def expand_cat(cat: str, agg: dict[str, list[str]]) -> list[str]:
    """Expand an aggregate matched category to its underlying ABM labels.

    Non-aggregate categories return ``[cat]`` unchanged.
    """
    return agg.get(cat, [cat])


def build_patient_matrix(
    df: pd.DataFrame, categories: list[str] = MATCHED_CATEGORIES
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (target×source FunCN matrix, per-target cell counts) for one IMC ROI.

    For each ABM target category, average each ABM source column over patient
    cells mapping to that target. Source values come from summing the
    corresponding ``nbhd_*`` columns per ``SOURCE_MAP``. Lifted verbatim from
    ``match_imc_patients.ipynb``.
    """
    keys = list(zip(df["cell_type"].astype(str), df["subtype"].astype(str)))
    df = df.assign(_target=[TARGET_MAP.get(k, np.nan) for k in keys])
    df = df.dropna(subset=["_target"]).reset_index(drop=True)

    src_cols = {c: np.zeros(len(df), dtype=float) for c in categories}
    for nbhd_col, abm_srcs in SOURCE_MAP.items():
        if nbhd_col not in df.columns:
            continue
        v = df[nbhd_col].to_numpy(dtype=float)
        for s in abm_srcs:
            if s in src_cols:
                src_cols[s] += v

    src_df = pd.DataFrame(src_cols)
    src_df["_target"] = df["_target"].values
    mat = src_df.groupby("_target").mean().reindex(index=categories, columns=categories)
    counts = src_df.groupby("_target").size().reindex(categories).fillna(0).astype(int)
    return mat, counts
