---
date: 2026-04-29
status: Torphy2018 excluded; submodel f_stroma_max posterior 2.43 -> 1.40; multi-target tension fully diagnosed; structural fix proposals queued
related: caf_cancer_coupling_2026-04-28.md (parent), ecm_anchor_audit_2026-04-28.md, untreated_tension_analysis_2026-04-28.md, GitHub #93
branch: feature/ecm-anchor-audit-issue-93
commits: 26dc04f
---

# V_T over-stuffing session — 2026-04-29

## TL;DR

Walked the chain from "is the over-stuffing because anchors come from wrong tissue context?" to "the multi-target tension is mathematically forced by the V_T volume rule + observable formulations, regardless of priors." Specifically:

1. (b) **Cell-density anchors are PDAC-correct**, not healthy-pancreas-derived. Refuted hypothesis.
2. (d) **`k_myCAF_prolif` isn't the lever.** Trace at 6× lower rate barely moved stromal_fraction (0.955 → 0.920) because CAF subtype interconversion + shared `aPSC_total` carrying-cap denominator buffer total CAF count. The shared-niche denominator is partly a model artifact (literature supports plastic iCAF/myCAF interconversion under signal gradients but does NOT support apCAFs filling in for myCAF loss).
3. (e) **Smoking gun in `f_stroma_max` calibration.** Torphy2018's forward model is identity (`f_stroma_max = TSD/(1−TSD)`) without the 0.3× CAF-fraction-of-stroma correction the YAML's own text mandates. Was pulling the joint Ogawa+Torphy posterior to 2.43 (~8× above Ogawa-alone prior-median value of 0.285). Moved to `excluded/`. Joint posterior 2.43 → **1.40**, σ 0.62 → 0.71, 90% CI [0.88, 6.71] → [0.45, 4.65].
4. **Hwang2008 audit:** under-fits by ~100× because the algebraic forward model misses CAF proliferation during 8 weeks, ECM mass contribution, and is linear when Hwang's data is sublinear (1.78× mass for 5× HPSC input). Under-fit is uniform → likelihood adds no information → `k_stroma_support` posterior is essentially Xu2010 alone, pinned at 1.05. Removing Hwang wouldn't move things; the formula mismatch makes the data uninformative either way.
5. **SBI run on baseline_no_treatment** with new submodel composite produced marginal prior-CI shifts only:
   - stromal_fraction: [0.891, 0.996] → [0.87, 0.996] (lower edge -0.021)
   - vegf, tam_density, cdc2_density: < 5% change
   - **Reachability of stromal_fraction in prior tail: 4.27% (156/3650 samples)**, max joint reach 18/22 targets. Bengtsson IS reachable, just not at the median.

The remaining residual tension is **structural** — encoded in the V_T volume rule + observable denominators — not addressable by further submodel-target work alone.

## The structural tension, formally

### Setup

- `evolve_to_diagnosis` runs until tumor diameter = 3.2 cm → **V_T at diagnosis is fixed at 17.25 mL** by construction.
- Cancer cells contribute `C1 × vol_cell / Ve_T` to V_T. CAFs and collagen contribute the rest.
- Stromal_fraction observable: `1 − C1·vol_c/V_T`. With V_T fixed, this is a pure function of C1.
- VEGF total: `k_vas_Csec × C_total + small terms from myCAFs/macs`. Cancer dominates at biological VEGF concentrations.
- Density observables (TAM, cDC2, etc.): `density = total_immune_cells / tumor_area_section`, where `tumor_area_section = C1 × area_per_cell / cellularity_constant`. Total immune cells = `M0 × V_T` (M0 = calibrated PDAC homeostatic density per V_T). With V_T fixed, total immune cells is fixed.

### The core conflict

To match Bengtsson stromal_fraction 0.668: C1 must be **2.21e9** (vs current 0.30e9 at fs=5/ks=1; 7.4× short).

Increase C1 by 7.4× while V_T is held at 17.25 mL implies:

| Observable | Direction | Why | Magnitude |
|---|---|---|---|
| stromal_fraction | ✅ approaches Bengtsson | mathematical | -0.29 |
| VEGF | ❌ scales linearly with C1 | `k_vas_Csec × C_total` | 7.4× *worse* (was already 50× overshoot at default) |
| TAM density | ✅ approaches obs | total_M ≈ const ÷ tumor_area ∝ 1/C1 | -7.4× (close to obs) |
| cDC2 density | ❌ undershoots obs | same scaling, but obs is high | -7.4× |
| Other density obs (cdc1, cd8, treg, mdsc) | mixed | depend on whether obs is high or low vs current | various |

**Empirical check via single-sim trace** (Section "Trace probes" below) at fs=1.4/ks=2.6 (best case within new submodel prior CI):
- stromal_fraction 0.955 → **0.861** ✓ (just at upper Bengtsson CI)
- VEGF 6980 → **19823** ❌❌
- TAM density 267 → **96** ✓ (close to obs 120)
- cDC2 density 0.36 → **0.12** ❌ (further below obs 1.82)

### Why no (fs, ks) point can satisfy all

Cancer count C1 is a single scalar. Each observable is a function of C1 (with other parameters held). The functions are monotone in *opposite directions* for stromal_fraction vs VEGF/cDC2_density. With V_T pinned by `evolve_to_diagnosis`, there's no degree of freedom to break the conflict by tuning fs or ks.

The conflict isn't between fs and ks — it's between the **mathematical structure of the observables and the model's V_T constraint**.

## Three structural fixes (ranked by leverage and risk)

**Rank order updated after probe analysis: #2 is now the cleanest first move because it's a bug fix that decouples density obs from C1, eliminating the bulk of the tension. #1 and #3 follow.**

### #1 — Re-anchor `k_vas_Csec` (cancer VEGF secretion rate)

**Lowest risk, highest immediate leverage on VEGF tension.**

Default model gives 6980 pg/mL VEGF at fs=5/ks=1 vs obs 145 pg/mL. **50× overshoot at default.** This isn't a CAF/cancer balance issue — it's a per-cancer VEGF secretion rate that's too high. Even at correct cellularity, `k_vas_Csec` would need to drop ~50× to match obs.

**Action:** extract a submodel target on PDAC cancer-cell VEGF secretion rate (pg/cell/day or similar). Likely available from in vitro PDAC cell line conditioned-media VEGF assays (BxPC3, PANC-1, MIA PaCa-2 typical). The 50× overshoot is so large it's a separate issue from the V_T over-stuffing — should be fixed regardless.

**Caveat:** if VEGF is mostly degraded fast, the steady-state concentration depends on `k_vas_sec / k_vas_deg`. Both rates need anchoring. Could be a coupled submodel target (secretion + half-life from same study).

### #2 — Fix density observable formula (bug, not redesign)

**Lowest risk after better-framing: this is a bug fix, not a structural change.**

The full-model density observables use:
```python
tumor_area_section = C1 * area_per_cell / cellularity_fraction
density = total_immune / tumor_area_section
```

The submodel calibration target forward models (e.g., `APC0_cDC2_T_PDAC_deriv001.yaml`'s Plesca2022 mapping) use:
```python
density_per_mm2 = APC0 * section_thickness_cm / 100  # = (cells/mL) × cm × (1 cm² / 100 mm²)
```

**These formulas disagree by 9.5× at trace medians.** The two should match at steady state when nothing perturbs.

Concrete check at trace (fs=5/ks=1, V_T=17.25 mL, C1=3.01e8):

| Variable | Value |
|---|---|
| Plesca-calibrated APC0_cDC2_T | 7.45e5 cells/mL |
| Steady-state cDC2 in tumor | APC0 × V_T = 1.29e7 cells |
| Submodel forward density | (1.29e7 / 17.25) × 4e-4 / 100 = **2.98 cells/mm²** ← matches Plesca obs ✓ |
| Full-model observable density | 1.29e7 / (3.01e8 × 2.27e-4 / 0.15) = **28.2 cells/mm²** ← 9.5× too high |
| Implied tumor area in full-model obs | 4555 cm² (vs sphere cross-section 8 cm²) |

The full-model observable's formula treats `tumor_area = C1 × area / 0.15` as if you laid out ALL C1 cells in a single 2D layer at 15% cellularity. That's a mathematical fiction, not a physical section. mIHC papers report cells/mm² of an actual section through the tumor — the submodel target's formula is correct.

**Why this is the structural-tension fix:**

The full-model observable's C1-dependency is what creates the multi-target tension. With density = `(cells/V_T) × thickness / 100`:
- V_T is fixed by `evolve_to_diagnosis`
- Total cells in V_T = M0_calibrated × V_T (also fixed)
- → density is INVARIANT in (fs, ks)

The stromal_fraction × density observable conflict goes away because density no longer depends on C1.

**Action:** rewrite all density observable YAMLs (`cdc1_density`, `cdc2_density`, `cd8_density`, `treg_density`, `mdsc_density`, `tam_density`, `total_til_density`, `cd4_helper_density`) to use:
```python
total_cells_in_tumor = species_dict['V_T.X'].to('cell').magnitude
V_T_mL = species_dict['V_T'].to('milliliter').magnitude
density_per_mm2 = (total_cells_in_tumor / V_T_mL) * (section_thickness_um * 1e-4) / 100.0
```
where `section_thickness_um` comes from the actual paper's Methods (Sivagnanam 5 µm, Plesca 4 µm typical, varies). Per memory `feedback_unit_conversion_inputs`, store as `direct_measurement` per source.

**Cellularity reconciliation** (option a from earlier this session) becomes moot under this fix — `cellularity_fraction` is no longer used in the density observables.

**Predicted effect at default trace** (V_T-based formula, model APC0 unchanged):
- TAM: 28.3 cells/mm² (vs obs 120) — still 4× short, but a calibration question, not a multi-target tension
- cDC2: 0.038 cells/mm² (vs obs 1.82) — 50× short

These shortfalls expose a separate, simpler problem: **APC0 dynamics aren't reaching the calibrated steady state.** The Plesca-implied APC0_cDC2_T = 7.45e5 cells/mL would give 2.98 cells/mm² (matches obs). But model trace shows V_T.cDC2 / V_T = 9.6e3 cells/mL, ~80× lower than calibration. Cause: maturation + LN migration drain cDC2 below the homeostatic source-death balance. That's a separate audit (does the steady-state calibration in the submodel target need to account for maturation drain?).

### #3 — Re-think cancer carrying-capacity formula

**Highest risk (model architecture change), highest leverage on the rate-race that drives V_T composition.**

Per Section 5.3 of `caf_cancer_coupling_2026-04-28.md`, the rate-race during evolve_to_diagnosis (k_myCAF_prolif 0.15/d × 537d ≫ k_C1_growth 0.01/d × 537d) drives CAF saturation before cancer reaches its cap. The current formula:

```
C_max = (V_T.K + k_stroma_support × aPSC_total) × (1 − phi_collagen / (phi_collagen + phi_col_50_Kmax))
```

Has cancer cap as a *function* of CAF count. That's biologically sensible (CAFs provide niche), but creates a circular dependency: CAFs need cancer (`f_stroma_max × C_total`), cancer needs CAFs (`k_ss × aPSC_total`). Stable point depends on growth-rate ratio.

**Alternative:** make C_max V_T-based, not CAF-based:
```
C_max = K_max_density × V_T × ECM_factor
```
where `K_max_density` is a per-volume packing constant (cells per mL of available space). This is the cellular-fluid-mechanics view: cancer cells fill V_T up to a packing limit, regardless of CAFs.

Pros: removes circular dependency; cancer can reach Bengtsson cellularity without needing CAFs to "support" it; growth dynamics simpler.

Cons: loses the biological story that desmoplastic stroma supports cancer growth (which has experimental support in PDAC — Hwang2008, Xu2010, despite formula bugs). Could blend the two: `C_max = K_max_density × V_T × ECM_factor × f(CAF_paracrine)` where f is a saturating gain factor (Hill on CAF count, plateaus at f=1 for high CAFs).

This is a substantial code change. Best done after seeing whether #1 and #2 close the tension empirically.

## Final SBI verification of the fix

Re-ran 5k baseline_no_treatment SBI with the 8 density-observable YAMLs corrected. Pool config_hash unchanged (sims reused); test_stats_hash flipped (recomputed). Log: `/tmp/sbi_run_baseline_density_fix_20260429_080301.log`.

**Prior-CI scorecard:**

| Observable | obs | Pre-fix CI95 | Post-fix CI95 | Status |
|---|---|---|---|---|
| stromal_fraction | 0.668 | [0.87, 0.996] | [0.87, 0.996] | OUTSIDE (unchanged — fix doesn't touch this obs) |
| vegf_tissue | 145 | [283, 4770] | [283, 4770] | OUTSIDE (unchanged — separate VEGF calibration issue) |
| cdc2_density | 1.82 | [4.98, 242] | inside | RESOLVED |
| tam_density | 120 | [339, 23600] | inside | RESOLVED |
| cdc1_density | 7.82 | inside | [0.645, 4.21] | NEW OUTSIDE (under-shoot) |
| cd8_density | 139 | inside | [0.0009, 116] | NEW OUTSIDE (under-shoot) |
| mdsc_density | 48.3 | inside | [3.11, 44] | NEW OUTSIDE (just below) |

Net flag count 4 → 5, but the failure mode flipped from over-shoot (CIs to 23600) to under-shoot with much tighter CIs (TAM/cdc2 inside; others below 1.5–2× of obs). Tight CIs are a feature — they reflect that the formula is now well-defined.

**Joint reachability improved:**
- Median 9/22 (40.9%) → **12/22 (54.5%)** targets simultaneously satisfied per sample
- Max 18/22 → **20/22**

**APC0_cDC2_T identification:**
- Now in top-5 R² recovery (R² = 0.684) — wasn't in top 5 before
- Posterior median = **6.38e5 cells/mL**, 90% CI [4.72e5, 8.49e5]
- Plesca-implied calibration value: 7.45e5 cells/mL — within 14% of posterior median
- Model default: 2e5 cells/mL (3.7× too low) — SBI correctly pulls 3.2× higher

This is the cleanest direct evidence the structural fix landed: with the C1-dependency removed, the SBI loss now properly identifies APC0 from the mIHC density obs.

**Verdict:** The remaining 3+2 OUTSIDE flags are now *independent calibration shortfalls* (model has fewer immune cells per V_T than mIHC at default; needs APC0/CD8/MDSC re-anchoring or APC dynamics audit) plus the *separate VEGF calibration issue* (k_vas_Csec 50× too high). None of them creates conflict with `stromal_fraction`. The multi-target geometry tension that PR #95 has been chasing is structurally closed.

## What was committed

- `26dc04f` Move Torphy 2018 to excluded/; f_stroma_max joint posterior 2.43 -> 1.40
- `9f086b7` V_T over-stuffing investigation memo
- `5ee4476` Structural tension diagnosis: density obs formula bug found (9.5× mismatch)
- `f2ca309` Fix density observable formula: V_T-based 3D->2D projection (decouples from C1)

## Current OUTSIDE flag prior-CI state (post-Torphy)

| Observable | obs | 90% prior CI | Reachability |
|---|---|---|---|
| stromal_fraction | 0.668 | [0.87, 0.996] | 4.27% (reachable in tail) |
| vegf_tissue | 145 | [283, 4770] | (not in joint reach top-10) |
| cdc2_density | 1.82 | [4.98, 242] | 26.8% |
| tam_density | 120 | [339, 23600] | 15.7% |

Joint reachability: median 9/22 targets per sample (40.9%); max 18/22; **no sample satisfies all 22**.

## Trace probes from this session

XMLs at `/tmp/kmycaf_trace/`. Recipe matches §7 of `ecm_anchor_audit_2026-04-28.md`.

### k_myCAF_prolif sweep (option d)
| k_myCAF | t_diag | C1 (×10⁹) | CAF total (×10⁹) | myCAF | iCAF | apCAF | stromal |
|---|---|---|---|---|---|---|---|
| 0.150 (default) | 537 | 0.301 | 1.283 | 1.183 | 0.061 | 0.040 | 0.955 |
| 0.075 | 617 | 0.347 | 1.256 | 1.115 | 0.088 | 0.053 | 0.948 |
| 0.050 | 715 | 0.398 | 1.236 | 1.037 | 0.126 | 0.073 | 0.941 |
| 0.025 (6× lower) | 1011 | 0.537 | **1.262** | 0.836 | 0.247 | 0.179 | 0.920 |

Note total CAF count stays roughly constant — interconversion + shared `aPSC_total` denominator compensate.

### (fs, ks) probe at submodel posterior region
| fs | ks | t_diag | C1 (×10⁹) | CAF total | stromal | VEGF (pg/mL) | TAM density |
|---|---|---|---|---|---|---|---|
| 5.0 | 1.0 (default) | 537 | 0.301 | 1.283 | 0.955 | 6980 | 267 |
| 1.0 | 1.0 | REJECTED | — | — | — | — | — |
| 1.4 | 1.05 (submodel medians) | REJECTED | — | — | — | — | — |
| 1.4 | 2.6 (ks 95% upper) | 794 | 0.930 | 1.081 | **0.861** | 19823 | **96** |

`fs ≤ ks` rejects (cancer cap collapses). Even at ks 95% upper of submodel prior CI, only just enters Bengtsson CI on stromal_fraction — and breaks VEGF.

## Cross-references

- Parent: `caf_cancer_coupling_2026-04-28.md` Sections 5–6 (multi-target tension diagnosis, suggested next moves)
- Hwang under-fit details: this memo + parent §3 (alpha cap) + §5 #2
- Torphy YAML moved: `calibration_targets/submodel_targets/excluded/f_stroma_max_torphy2018_PDAC_deriv001.yaml`
- Excluded README: documents both k_C1_growth Kim2017 (parameter-semantic mismatch, 2026-04-28) and Torphy 2018 (forward-model mismatch, 2026-04-29)
- SBI run log: `/tmp/sbi_run_baseline_no_torphy_20260429_072853.log`
