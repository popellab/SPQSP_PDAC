# CD8/Treg structural fix — proposal

**Date:** 2026-04-26
**Source:** Re-stratification of 1000-draw composite probe with two new
qualitative bounds (`cd8_density_min_diagnosis` ≥ 5 cells/mm²,
`treg_density_max_diagnosis` ≤ 300 cells/mm²) added to
`tests/qualitative_bounds.yaml`.

## TL;DR

The CD8/Treg pathology is **fixable with a coordinated 5–6 parameter
prior re-design**. Stacking the helping directions on these priors
lifts joint T-cell-axis pass rate (CD8 density + Treg density bounds
together) from **2.7% → 71%** in the existing probe distribution.
The fix is **not** a single knob — every individually-tightened prior
moves things by ≤ +5%; the gains are multiplicative across the priming
chain and Treg dynamics.

The audit's pull on `TGFb_50_APC` (4× tighter EC50 than CSV) is one of
the load-bearing wrong moves. Reverting it is part of the fix.

## What the new bounds say

| metric | composite probe | literature |
|---|---:|---:|
| CD8 density (cells/mm²) median | 0.05 | 139 (Jansen 2021) |
| % draws with CD8 < 5 cells/mm² | 91% | (catastrophic absence) |
| Treg density (cells/mm²) median | 466 | 14 (Liudahl 2021) |
| % draws with Treg > 300 cells/mm² | 61% | (>2× CI95 upper) |

The previously-existing `cd8_treg_ratio_max_diagnosis` bound (lit ~3,
bound 0.3-15) was being **passed by 5.9%** of draws, but only because
both pools were 100-1000× below literature absolute densities. The
new absolute density bounds make this hidden failure mode visible.

## The fix — coordinated prior shifts

Drivers identified by Spearman correlation against the new bounds.
"Direction" = which way the prior should shift to improve pass.

| Param | Direction | CSV current | Proposed | Rationale |
|---|:---:|---:|---|---|
| `f_active_TGFb` | **lower** | lognormal(med=0.10, σ=1.5) | lognormal(med=0.03, σ=0.5) | Active TGFb fraction in vivo is 1–10% (latent TGFb dominates); CSV σ=1.5 spans 3 OOM. PRCC −0.40 on apc_maturation; #2 driver for CD8 density. |
| `TGFb_50_APC` | **higher (revert audit)** | CSV 0.07 nM, audit→0.018 nM | restore to 0.07 nM, σ=0.5 | Audit pulled it 4× too sensitive based on one IL-12 IC50 in vitro derivation. The in-vitro→in-vivo translation is suspect for an assay that ignores stromal TGFb sequestration. |
| `N_div_costim` | **higher** | lognormal(med=3.0, σ=0.5) | shift median to 5.0, σ=0.3 | Top driver for CD8 density (PRCC +0.27). T cell divisions per priming event; passing-draw median is 5 vs failing-draw 3. |
| `n_T_expansion_burst` | **higher** | lognormal(med=800, σ=0.7) | shift median to 1100, σ=0.4 | T cell expansion burst, PRCC +0.23; passing-draw median 1130 vs failing 764. |
| `k_Treg_death` | **higher** | lognormal(med=0.10, σ=1.2), audit→0.055 | restore to ≥0.10, σ=0.6 | Audit pulled Treg death down (slower turnover); reverting raises Treg clearance. |
| `k_Treg_pro_tumor` | **lower** | lognormal(med=0.20, σ=0.7), audit→0.133 | tighten σ→0.3, keep median ≤0.10 | Direct Treg in-tumor proliferation. PRCC −0.46 (the strongest single driver for Treg density). |

### Empirical lift from stacking these shifts

Each shift adds ~3–10% on the joint T-cell pass rate. Stacked greedily
(restrict to favorable half on each successive parameter):

| # helpers added | n_kept (of 441) | CD8 density pass | Treg density pass | T-axis joint |
|---:|---:|---:|---:|---:|
| 0 (baseline) | 441 | 8.8% | 38.8% | **2.7%** |
| 1 (`f_active_TGFb` low) | 221 | 14.5% | 38.9% | 5.4% |
| 2 (+ `k_Treg_death` high) | 107 | 18.7% | 44.9% | 9.3% |
| 3 (+ `TGFb_50_APC` high) | 50 | 30.0% | 44.0% | 18.0% |
| 4 (+ `k_Treg_pro_tumor` low) | 21 | 42.9% | 71.4% | 33.3% |
| 5 (+ `n_T_expansion_burst` high) | 11 | 63.6% | 72.7% | 54.5% |
| 6 (+ `N_div_costim` high) | 7 | 85.7% | 85.7% | **71.4%** |

This is a *conditional* analysis — picking the favorable half of the
existing prior for each param. To realize the fix in deployment, those
shifts need to be **baked into the prior CSV** (and the audit needs
to respect them rather than overriding for `TGFb_50_APC` and
`k_Treg_death`).

### Why stacking works (and single fixes don't)

The CD8/Treg pathology is **multiplicative across two coupled chains**:

1. **Priming chain (CD8 side)**: APC maturation → naive CD8 priming
   → expansion burst → infiltration. Each link can be the bottleneck.
   Shifting any one link gives +3–5%; shifting the chain gives 10×+.
2. **Treg dynamics (Treg side)**: in-flow + in-tumor proliferation +
   death + apCAF/iCAF priming. Same multiplicative structure.

`k_APC_mat_cDC1` and several other priming params show **opposite-sign
PRCCs** for CD8 density vs Treg density. That's because mature DC
expansion drives CD8 priming **and** kicks off the IFNg → M1 → reduced
IL-10/TGFb negative feedback that tames Treg priming. **Fixing the
priming chain helps both bounds simultaneously** — a coupled benefit
no single-parameter analysis would catch.

## What this fix does NOT solve

Even with all 6 stacked T-cell-axis helpers, the **all-18-bound joint
pass remains 0%**. Other failure modes persist:

- `m2_m1_ratio_range` (~83% fail) — needs macrophage-axis fix
  (`k_M1_pol` higher); the existing audit pulled `k_M2_pol` up 5× but
  M1 pol stayed close to CSV. PRCC analysis says `k_M1_pol` is the
  dominant lever (+0.36).
- `apc_maturation_min_diagnosis` (~83% fail) — partially solved by
  the TGFb fixes above, but other contributors remain.
- `ecm_fold_change_range_diagnosis` (~31% fail) and
  `icaf_fraction_range` (~48% fail) — CAF-axis fixes.
- `apcaf_fraction_max` — driven by `k_PSC_to_apCAF` audit artifact
  (separate note).

## Recommended deployment

### Tier 1 (immediate, prior-only fixes)

1. **Edit `parameters/pdac_priors.csv`**: tighten σ_log on
   `f_active_TGFb`, `k_Treg_pro_tumor` from 1.5/0.7 toward 0.5/0.3;
   verify `N_div_costim`, `n_T_expansion_burst` priors against
   literature values (currently TBD — need to look up).
2. **Edit `submodel_config.yaml`** to add a `component_overrides`
   entry for `TGFb_50_APC_PDAC_deriv001` — revert to NUTS with a
   tighter base prior centered at the CSV value (0.07 nM), so the
   submodel's 4× pull doesn't clobber the structural fix.
3. **Same for `k_Treg_death`** — the audit pulled it from CSV 0.10
   to 0.055; tighten the source target's prior or move to NUTS.

### Tier 2 (validation)

4. Re-run the 1000-draw probe with the updated priors. Expect joint
   T-axis pass rate to lift from ~3% to ~30–50% (extrapolating from
   the conditional analysis: realized prior shifts deliver less than
   the maximum-conditioning lift, but should still be substantial).
5. If the M2/M1 axis remains broken, repeat the analysis for
   macrophage drivers and stack those fixes too. Same approach;
   different parameters.

### Tier 3 (structural, if Tier 1+2 insufficient)

6. **Add explicit Treg carrying capacity in V_T**. The model currently
   has Treg expanding to 6.4×10⁸ cells in V_T (well above any
   physiological density). If the carrying capacity term doesn't
   exist or is too high, that's a structural fix.
7. **Audit Th→Treg conversion gain**. With saturating TGFb (which is
   structurally near-inevitable in PDAC), the Th→Treg branching
   becomes near-deterministic. May need a competing factor or an
   IL-2 → Treg-resistance gain for the Th lineage.

## Diagnosis — am I sure?

The diagnosis was **partially wrong** in the previous note (`cd8_treg_gap_diagnosis.md`):

- I claimed the issue was "Treg overproduction not CD8 trafficking." This was based on comparing **passing-draw Treg vs failing-draw Treg** (4000× swing), but both populations are non-realistic in absolute terms.
- Against literature, CD8 absolute density is the bigger failure (3–4 OOM gap, 91% of draws); Treg is also too high (33× median gap, 73% of draws) but the gap is smaller in OOM terms.
- The right framing: **both pools are broken; the CD8 collapse is dominantly upstream-priming-driven; the Treg excess is dominantly Treg-dynamics-driven; both are individually and jointly fixable through coordinated prior shifts.**

The diagnosis note will be updated to reflect this.

## Files

- New bounds added: `tests/qualitative_bounds.yaml` (24 bounds total, +2 new)
- New observable functions: `tests/qualitative_bounds.py` (`_compute_cd8_density`, `_compute_treg_density`)
- Probe data (unchanged, just re-evaluated): `results/prior_diagnostic/composite/probe_data.npz`
- Coverage cache (species values): `cache/qual_coverage/runs/cd7e20abe4226139.npz`
- Companion notes:
  - `notes/calibration/structural_treg_vortex.md`
  - `notes/calibration/cd8_treg_gap_diagnosis.md` (will be revised)
  - `notes/calibration/k_PSC_to_apCAF_audit_overfit.md`
  - `notes/calibration/stage1_value_analysis.md`
