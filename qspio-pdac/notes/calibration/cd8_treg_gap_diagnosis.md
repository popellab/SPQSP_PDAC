# CD8/Treg gap diagnosis — composite-prior 1000-draw probe

**Date:** 2026-04-26
**Source:** `results/prior_diagnostic/composite/probe_data.npz` +
`cache/qual_coverage/runs/cd7e20abe4226139.npz`
**Bound:** `cd8_treg_ratio_max_diagnosis` (literature mean ≈ 3, bound ≤ 15;
26/441 pass under composite prior, 94% fail).

> **CORRECTION (2026-04-26, later same day):** the "Treg overproduction
> not CD8 trafficking" framing below was based on comparing
> passing-draw vs failing-draw Treg medians (4000× swing) — but **both
> populations are non-realistic in absolute terms**. Against literature,
> CD8 is the bigger failure (3–4 OOM gap, 91% of draws below CI95
> lower); Treg is also too high (~33× median gap, 73% above CI95
> upper) but smaller in OOM. The corrected analysis with new absolute
> density bounds, the actual structural drivers, and a proposed fix
> are in `cd8_treg_fix_proposal.md`. Treat this note as historical.

## TL;DR — it's a Treg overproduction problem, not CD8 trafficking

| species (V_T median) | passing draws (n=26) | failing draws (n=415) | fail/pass |
|---|---:|---:|---:|
| **CD8** | 2.86×10⁵ | 3.08×10⁴ | **0.11×** (CD8 is ~10× lower in failing) |
| **Treg** | 1.59×10⁵ | 6.38×10⁸ | **4015×** (Treg is 4000× higher in failing) |
| Th | 1.02×10⁴ | 4.29×10⁷ | 4200× |
| T_total | 6.57×10⁵ | 7.52×10⁸ | 1145× |
| Mac_M2 | 6.47×10⁸ | 8.02×10⁸ | 1.24× |
| Mac_M1 | 7.78×10⁷ | 9.15×10⁷ | 1.18× |
| TGFb | 22.7 nM | 16.5 nM | 0.73× |

**The 5-OOM CD8/Treg gap is dominated by the Treg side**: passing
draws have Treg ≈ 1.6×10⁵, failing draws have Treg ≈ 6.4×10⁸. CD8
moves only ~1 OOM. Total T cells are 1000× higher in failing draws,
mostly Treg (85% Treg in failing vs 24% Treg in passing). This is
**not** "CD8 fails to traffic to V_T" — it's "T cell pool blows up
and is dominated by Treg expansion."

## Top parameter shifts (passing vs failing, log10-median, p<0.01 Mann-Whitney)

| param | shift_log10 | direction | mechanism |
|---|---:|---|---|
| **`p1_50`** | **−0.35** | failing draws have lower P1 epitope EC50 | higher antigen sensitivity → more T cell activation overall |
| `E_50_fib` | +0.28 | passing draws have higher stiffness EC50 | less mechanotransduction → less iCAF/myCAF activation |
| `k_Th_pro_tumor` | −0.22 | failing → faster Th proliferation in tumor | more Th substrate for Th→Treg |
| `k_Treg_death` | +0.21 | passing → faster Treg death | direct Treg pool reduction |
| `k_CAF_encounter` | −0.21 | failing → more T-CAF encounters | apCAF→Treg priming amplified |
| `k_iCAF_to_myCAF` | +0.21 | passing → more iCAF→myCAF | iCAF (Treg-promoting via CCL5) drained |
| `k_APC_mig` | +0.20 | passing → more APC migration to LN | more priming substrate; restores CD8 priming |
| `q_Treg_T_in` | −0.20 | failing → more Treg in-flow from V_C | direct Treg recruitment |
| `k_col_myCAF_sec` | −0.18 | failing → more collagen secretion | denser ECM → more T cell trapping |
| `k_Treg_pro_tumor` | −0.18 | failing → faster Treg proliferation | direct Treg pool expansion |
| `k_CD8_death` | −0.18 | failing → faster CD8 death | counterintuitive — likely correlation, not causation |
| `k_IL1_sec` | −0.17 | failing → more IL-1 | IL-1 → iCAF → CCL5 → Treg recruit |
| `k_mAPC_death` | −0.16 | failing → faster mAPC death | priming substrate loss |

CSV-prior-vs-pass-vs-fail quick reference:

| param | CSV | passing median | failing median | pass/CSV | fail/CSV |
|---|---:|---:|---:|---:|---:|
| `p1_50` | 2.6×10⁻⁵ | 1.5×10⁻³ | 3.4×10⁻³ | 57× | 128× |
| `k_Th_pro_tumor` | 0.15 | 0.099 | 0.163 | 0.66× | 1.09× |
| `k_Treg_death` | 0.10 | 0.090 | 0.056 | 0.90× | 0.56× |
| `q_Treg_T_in` | 5.8×10⁻⁵ | 6.7×10⁻⁵ | 1.05×10⁻⁴ | 1.16× | 1.82× |
| `k_Treg_pro_tumor` | 0.20 | 0.088 | 0.132 | 0.44× | 0.66× |
| `k_IL1_sec` | 1×10⁻¹² | 2.8×10⁻¹¹ | 4.1×10⁻¹¹ | 28× | 41× |

Notes on outliers:
- `p1_50` and `k_IL1_sec` are 1.5-σ-wide CSV priors (lognormal sigma=1.5),
  so the live prior allows huge fold-changes; even passing draws sit
  ~50× above CSV median.
- `q_Treg_T_in`, `k_Th_pro_tumor`, `k_Treg_pro_tumor`, `k_Treg_death`
  are tight CSV priors (sigma 0.5–1.0); meaningful fold-shifts.

## Mechanism

The cascade has **multiple coupled drivers**, not a single bottleneck:

1. **Antigen sensitivity (`p1_50`)**: lower EC50 → both CD8 priming
   and CD4 priming amplified. With cDC2-mediated CD4 priming feeding
   the Th-Treg branch, and TGFb-tone pushing CD4 toward Treg, more
   antigen sensitivity means more Treg.
2. **Direct Treg recruitment (`q_Treg_T_in`)**: failing draws have
   ~80% more Treg V_C→V_T migration. CSV default is already 5.8×10⁻⁵.
3. **Tregfactory pool dynamics**: failing draws have slower Treg death
   (0.056 vs CSV 0.10) AND faster Treg in-tumor proliferation (0.13).
   Both submodel-derived shifts (audit's k_Treg_death −0.49σ vs CSV).
4. **Th pool feedstock (`k_Th_pro_tumor`)**: failing draws have
   ~9% faster Th proliferation. This is the indirect Treg pump via
   Th→Treg conversion.
5. **apCAF priming amplifier (`k_CAF_encounter`)**: failing draws
   have ~63% more T-CAF encounter. Higher k_PSC_to_apCAF (audit
   artifact, separate note) creates more apCAFs; this parameter
   determines how often CD4s see them.
6. **iCAF retention (`k_iCAF_to_myCAF`)**: failing draws keep iCAFs
   alive (slower interconversion to myCAF). iCAFs secrete CCL5 which
   recruits Tregs.

## Why this isn't CD8 trafficking

CD8 in failing draws is 3×10⁴ — about 1800 cells/mL of V_T. That's
genuinely low (literature ≥ 10⁶/mL), but it's only 1 OOM below
passing-draw median (2.9×10⁵ ≈ 17000/mL — also below literature).
Even passing draws don't have literature-level CD8 density; they
just don't have the catastrophic Treg expansion.

CD8 trafficking parameters did **not** show up in the top shift list.
`q_CD8_T_in`, `q_CD8_LN_in`, `k_CD8_pro_tumor`, etc. all had Mann-
Whitney p > 0.01 and small magnitudes. The data-grounded `CXCL12_50_Texcl`
PRCC (-0.15) from the original report reflects a partial CD8-exclusion
effect, but it's a minor contributor.

## Recommended fixes

1. **Re-examine `p1_50` prior**. CSV is lognormal(median=2.65×10⁻⁵,
   σ_log=1.5). σ=1.5 spans ~3 OOM — likely the loosest prior in the
   set. Can the prior be narrowed using a published epitope EC50?
   This would directly cap how much T cell activation the model can
   produce.
2. **Audit `q_Treg_T_in` separately from `q_T_in`**. The Treg-specific
   in-flow rate is 1.8× CSV in failing draws. If there's no Treg-
   specific chemoattraction submodel, this parameter should probably
   be tied to `q_T_in` or the chemokine-mediated trafficking module
   rather than treated as a free Treg-specific knob.
3. **Add `treg_fraction_max` qualitative bound**. Currently
   `treg_fraction_min` exists (0.05 floor) but has no upper. Adding
   ≤ 0.4 turns the silent failures (Treg = 85% of T cells) into
   loud ones in the test suite.
4. **Investigate Th→Treg conversion gain**. `k_Th_pro_tumor` shifts
   say Th proliferation differs between pass and fail; but Th itself
   isn't the failure direction. Check whether the Th→Treg rate
   constant has an audit pull from a submodel target — if so, the
   constraint may be too aggressive.
5. **Coupled re-prior on Treg dynamics**. `k_Treg_death`,
   `k_Treg_pro_tumor`, `q_Treg_T_in` all show 1.5–2× shifts in
   failing draws. The audit individually moved them in the
   "Treg-friendly" direction. A joint reanalysis (treating Treg
   pool size as the calibration target rather than each rate
   independently) might converge faster.

## Files

- Probe data: `results/prior_diagnostic/composite/probe_data.npz`
- Coverage cache: `cache/qual_coverage/runs/cd7e20abe4226139.npz`
- Companion notes:
  - `notes/calibration/structural_treg_vortex.md` (high-level co-failure)
  - `notes/calibration/k_PSC_to_apCAF_audit_overfit.md` (apCAF artifact)
  - `notes/qual_bounds_state_summary.md` (posterior-median snapshot)
