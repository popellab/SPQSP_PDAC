---
date: 2026-04-28
status: 2 submodel targets extracted + joint inference + 2 SBI runs (pre-CAF, post-CAF, post-CAF + submodel-sync); structural gap partially closed but stromal_fraction still OUTSIDE
related: ecm_anchor_audit_2026-04-28.md (parent audit), untreated_tension_analysis_2026-04-28.md, GitHub #93
branch: feature/ecm-anchor-audit-issue-93 (stacked on feature/option-a-collagen-damper-fix)
---

# CAF–cancer coupling re-anchoring (f_stroma_max, k_stroma_support) — 2026-04-28

## TL;DR

The ECM anchor audit (`ecm_anchor_audit_2026-04-28.md`) re-anchored Ve_T, rho_collagen, and phi_col_ref via 3 submodel targets, but the post-anchor SBI run still showed `stromal_fraction` OUTSIDE Bengtsson CI95 (model 0.96 vs obs 0.668). A diagnostic parameter sweep identified `f_stroma_max` (the CAF:cancer count-ratio carrying-capacity cap) as the dominant driver — *not* the V_T volume rule.

Two new submodel targets shifted `f_stroma_max` 5.0 → 2.43 (joint posterior median, n_eff > 6000, rhat=1.00). The third SBI run (`--with-submodel-sync`) was the cleanest of the investigation: **OUTSIDE flag count dropped 4 → 2** (cdc1_density and apc_maturation came inside their CIs for the first time), and the stromal × VEGF tension hit its lowest value (−0.267). `stromal_fraction` and `vegf_tissue_concentration` remain OUTSIDE — the structural gap is partially but not fully closed by parameter re-anchoring alone.

## 1. Trigger — diagnostic parameter sweep

After the ECM anchor work landed, an SBI run on the patched code still showed:

- `stromal_fraction`: 90% CI [0.82, 0.99] — OUTSIDE obs 0.668
- `cdc1_density`: still way below obs 7.82 cells/mm²
- Tension hub `stromal × VEGF`: φ = −0.29 (was −0.40 pre-audit; partial improvement)

The audit's "Option B" recommendation was a *hard packing-budget constraint* at the V_T volume rule. Before pursuing that, a single qsp_sim trace at prior medians revealed the actual problem: at diagnosis, **the model has 4× more myCAF cells than cancer cells** (V_T.myCAF = 1.18 × 10⁹ vs V_T.C1 = 3.01 × 10⁸). The Bengtsson observable `cancer_cellularity = C1 × vol_cell / V_T = 0.045` (4.5%) doesn't match obs 0.33 because most of V_T is occupied by CAFs, not cancer.

A 9-cell sweep over `(f_stroma_max, k_stroma_support)` confirmed:

| f_stroma | k_stroma | t_diag (d) | C1 (×10⁹) | myCAF (×10⁹) | phi_col | stromal_frac |
|---|---|---|---|---|---|---|
| **5.0** | 1.0 | 537 | 0.30 | 1.18 | 0.659 | **0.955** ← current default |
| 5.0 | 5.0 | 299 | 0.42 | 1.44 | 0.599 | 0.938 |
| 3.0 | 5.0 | 366 | 0.63 | 1.23 | 0.595 | 0.907 |
| 2.0 | 5.0 | 443 | 0.85 | 1.05 | 0.586 | 0.873 |
| 1.0 | 5.0 | 675 | 1.38 | 0.72 | 0.557 | 0.795 |
| **0.5** | **5.0** | 1559 | **2.01** | 0.41 | 0.510 | **0.699** ← inside Bengtsson CI95 |

`fs=0.5, ks=5` reproduces stromal_fraction 0.699 (Bengtsson CI95 [0.42, 0.86]). `fs ≤ 1.0 × ks = 1.0` rejects (cancer cap collapses, tumor never reaches 3.2 cm). Lower `fs` *requires* higher `ks` to keep cancer growth viable — the two parameters are coupled levers.

Implications for the audit's options:
- **Option A** (already merged: `(1−phi)²` damper + bare-volume cells): partial — moved tension from −0.40 to −0.29.
- **Option B** (hard packing constraint at ODE level): unlikely to fix this. The 5:1 ratio is set by `f_stroma_max` carrying-cap design, not by volume-rule arithmetic. Hard budget would clip both species equally.
- **Re-anchor `f_stroma_max` + `k_stroma_support`**: high-confidence lever. Cellularity ratio is hard-coded by these two params.

The current `f_stroma_max = 5` had `derivedFrom = {}` and a soft .Notes citation: "PDAC stroma 50–80% by volume; typical CAF:cancer ratio 1–5:1." The likely root cause: confusing CAF *area* fraction (50–80%) with *count* ratio. Standard PDAC nuclear-density estimates (cancer ~5–10k cells/mm² × CAF ~1.5–3k cells/mm²) imply count ratio ≈ 0.3–1.0 — exactly matching the diagnostic sweep optimum.

## 2. Submodel target extraction

Two parallel literature subagents found:

### `f_stroma_max_ogawa2021_PDAC_deriv001.yaml`

- **Source:** Ogawa et al. 2021, *Clin Cancer Res* 27(1):107–119, DOI `10.1158/1078-0432.CCR-20-2298`
- **Data:** n=215 treatment-naive resected PDACs, whole-slide multiplex IHC (ACTA2 + FAP + collagen via van Gieson), pixel-level area fractions reported as proportions of desmoplastic elements:
  - ACTA2-dominant fibroblasts: mean 33.4%, IQR 24.2–41.1%
  - FAP-dominant fibroblasts: mean 13.0%, IQR 8.2–17.4%
  - Collagen: mean 53.4%
- **Forward model:** algebraic identity that converts a count ratio (`f_stroma_max`) to area ratio via cancer/CAF nuclear density nuisance parameters (`cancer_density_per_mm²` ~ 7000, `caf_density_per_mm²` ~ 2000), with collagen/cancer area ratio anchored from same Ogawa data.

### `k_stroma_support_hwang2008_PDAC_deriv001.yaml`

- **Source:** Hwang et al. 2008, *Cancer Research* 68(3):918–926, DOI `10.1158/0008-5472.CAN-07-5714`
- **Data:** Orthotopic BxPC3 + HPSC co-injection in nude mice, 8-week pancreas weights:
  - 1:0 ratio: 0.22 ± 0.01 g (basal pancreas, 0% tumor formation)
  - 1:1 ratio (0.5e6 BxPC3 + 0.5e6 HPSC): 0.46 ± 0.18 g (57% formation)
  - 1:5 ratio (0.5e6 BxPC3 + 2.5e6 HPSC): 0.82 ± 0.22 g (100% formation)
- **Forward model:** algebraic identity treating cancer cells as having reached carrying capacity at 8 weeks, with `alpha_endpoint_mass` nuisance for saturation efficiency. The `k_stroma_support × HPSC` term scales cancer C_max linearly.

Both targets validated cleanly (schema + snippets, lognormal-fit posteriors, no divergences).

## 3. Joint posterior

`scripts/parameter_audit.py --invalidate f_stroma_max k_stroma_support` ran joint MCMC over the new + pre-existing targets:

- f_stroma_max group: `Torphy2018` (existing) + `Ogawa2021` (new) → 3 params, 2 likelihood terms
- k_stroma_support group: `Xu2010` (existing) + `Hwang2008` (new) → 3 params, 4 likelihood terms

Results (rhat=1.00, n_eff > 5900 for all params, no divergences):

| Param | Old default | Joint posterior median | 90% CI | Contraction |
|---|---|---|---|---|
| `f_stroma_max` | 5.0 | **2.43** | [0.57, 5.53] | 11% |
| `k_stroma_support` | 1.0 | **1.05** | [0.15, 2.62] | 0% (Xu2010 was already pinning at 1.0) |

`f_stroma_max` halved but the 90% CI is wide (0.57–5.53), reflecting the structural identifiability concern: the area-data signal needs nuisance density priors to convert to count ratio, and the conversion is degenerate with `f_stroma_max` itself. `k_stroma_support` essentially didn't move — the pre-existing Xu 2010 target had it pinned at 1.0; Hwang's data wanted higher but couldn't override.

Forward-model self-consistency on Hwang prediction at posterior median: `(0.5e6 + 1.05 × 0.5e6) × 1.05 × 2.572e-9 × 0.75 ≈ 0.001 g` — barely above basal vs Hwang's observed 0.46 g. The forward model is under-fitting Hwang heavily; the `alpha_endpoint_mass` nuisance (lognormal mu=0, sigma=0.7) caps how aggressively the saturation-reach factor can compensate. Either the Hwang forward model needs richer dynamics (e.g., custom_ode tracking 8-week growth curve) or the alpha prior needs widening. **Open methodological item.**

## 4. SBI runs (multi-target full-model)

### Pre-CAF (post-ECM only) — pdac-build commit `83f2097`

Run command: `.venv/bin/python -m workflows.sbi_runner --scenarios baseline_no_treatment --num-total-simulations 5000 --skip-submodel-sync --cpp-branch feature/ecm-anchor-audit-issue-93`

| Observable | obs | 90% CI | Status |
|---|---|---|---|
| stromal_fraction | 0.668 | [0.82, 0.99] | OUTSIDE |
| cdc1_density | 7.82 | [9e-4, 2.41] | OUTSIDE |
| vegf_tissue | 145 | [179, 8060] | OUTSIDE |
| apc_maturation | 0.014 | [0.022, 0.74] | OUTSIDE |

Top tension `stromal × VEGF`: φ = −0.293 (was −0.40 pre-audit).

### Post-CAF, CSV-only — pdac-build commit `851f25a`

Run command: same with updated CSV priors (f_stroma_max median 2.43, k_stroma_support median 1.05).

| Observable | obs | 90% CI | Pre-CAF | Direction |
|---|---|---|---|---|
| stromal_fraction | 0.668 | **[0.76, 0.99]** | [0.82, 0.99] | toward Bengtsson — better, still OUTSIDE |
| cdc1_density | 7.82 | [7e-4, 1.63] | [9e-4, 2.41] | further from obs — worse |
| vegf_tissue | 145 | [252, 1.06e4] | [179, 8060] | further from obs — worse |
| apc_maturation | 0.014 | [0.022, 0.74] | unchanged | unchanged |

Top tension `stromal × VEGF`: φ = −0.305 (basically unchanged).

**SBI posterior medians (CSV-only run):**

| Param | Submodel posterior | Full-model SBI posterior | Direction |
|---|---|---|---|
| Ve_T | 0.37 | 0.33 | similar |
| phi_col_ref | 0.046 | 0.050 | similar |
| rho_collagen | 0.103 | 0.130 | similar |
| **f_stroma_max** | 2.43 | **3.05** | **SBI wants HIGHER, not lower** |
| **k_stroma_support** | 1.05 | **1.51** | SBI wants higher |

This is the surprise: SBI's full-model joint likelihood pulls `f_stroma_max` *up* from 2.43 to 3.05, not toward the diagnostic-sweep optimum of 0.5–1.0. Other observables in the multi-target loss outweigh Bengtsson stromal_fraction. The diagnostic-sweep result reflects what a *single-observable* fit on stromal_fraction would do; the multi-observable SBI doesn't see stromal_fraction as the binding constraint.

### Post-CAF, with submodel-sync — pdac-build commit `851f25a`

Run command: same with `--with-submodel-sync` (composite priors override CSV → tighter f_stroma_max σ 0.62, k_stroma_support σ 0.70).

| Observable | obs | 90% CI | Post-CAF CSV | Direction |
|---|---|---|---|---|
| stromal_fraction | 0.668 | [0.80, 0.99] | [0.76, 0.99] | OUTSIDE (slightly worse than CSV) |
| cdc1_density | 7.82 | within CI | OUTSIDE | **resolved** |
| vegf_tissue | 145 | [287, 9620] | [252, 1.06e4] | still OUTSIDE |
| apc_maturation | 0.014 | within CI | OUTSIDE | **resolved** |

Top tension `stromal × VEGF`: φ = **−0.267** (best of all three runs; pre-audit was −0.40, post-ECM −0.29, post-CAF CSV −0.30).

**SBI posterior medians (with-sync run):**

| Param | Submodel posterior | SBI CSV-only | SBI with-sync |
|---|---|---|---|
| Ve_T | 0.37 | 0.33 | 0.40 |
| **phi_col_ref** | 0.046 | 0.050 | **0.139** ← SBI rejects the tight Grizzi-derived prior, prefers near the original 0.12 |
| rho_collagen | 0.103 | 0.130 | 0.120 |
| f_stroma_max | 2.43 | 3.05 | 2.54 |
| k_stroma_support | 1.05 | 1.51 | 1.48 |

The composite-prior SBI is the cleanest run of the three:

- **OUTSIDE flag count: 4 → 2** (cdc1_density and apc_maturation came inside the CI for the first time in this whole investigation)
- **Lowest stromal × VEGF tension** (−0.267)

The cost: `stromal_fraction` lower CI rose slightly (0.76 → 0.80, away from Bengtsson's 0.668) and `phi_col_ref` posterior median bounced back to 0.14 — implying SBI rejects the tight Grizzi-derived prior at 0.046 and prefers something close to the original 0.12 once given the multi-target constraints. Either Grizzi's 2.23% Sirius-red anchor is too low for the QSP convention, or other observables in the SBI loss demand a higher phi_col_ref (e.g., d_pore Hill terms saturating differently, stiffness power-law constraints).

## 5. Verdict on the structural gap

The CAF anchor extraction was the right move (data-derived posteriors replaced unanchored defaults), but it's necessary-not-sufficient. The remaining issues:

1. **Multiple competing observables in the joint SBI likelihood.** Bengtsson stromal_fraction prefers low f_stroma_max; other observables (cdc1_density, vegf_tissue, immune-fraction targets) prefer higher f_stroma_max. The model can't satisfy all at once. The diagnostic parameter sweep was a single-observable optimum that the multi-target SBI overrides.
2. **Hwang forward model under-fits.** The QSP linear-in-CAF carrying-cap formula doesn't capture the sublinear scaling Hwang shows. `k_stroma_support` posterior is pinned by Xu2010, not by Hwang. Open item: revise Hwang forward model to a custom_ode or accept that the 1.05 posterior median reflects Xu2010 alone.
3. **Cancer:CAF ratio at diagnosis is dynamic, not algebraic.** Per `evolve_to_diagnosis` semantics, baseline cellularity is the OUTCOME of growth-rate races during evolution, not directly set by density priors. To pull cellularity higher requires either (a) increasing cancer prolif rates relative to CAF prolif rates, or (b) constraining the ratio explicitly via a calibration target weighted heavily.

## 6. Suggested next moves

In rough order of leverage:

1. **Compare CSV-only vs `--with-submodel-sync` runs** (in progress). If submodel-sync significantly improves the calibration scorecard, the submodel posteriors are valuable and should be retained as default. If not, they don't matter for SBI inference and the choice of CSV vs submodel-derived defaults is purely a design preference.
2. **Revise Hwang 2008 forward model** to a custom_ode that tracks 8-week growth dynamics with cancer-CAF logistic interaction. Expected to widen `k_stroma_support` posterior to capture the sublinear scaling Hwang observes.
3. **Add a paired-density calibration target** — i.e., a `CalibrationTarget` (full-model observable) that directly measures CAF area fraction in PDAC tissue at diagnosis (using Ogawa or similar). This would put more weight on the cellularity ratio in the SBI loss without requiring the submodel framework's awkward count-ratio conversion.
4. **Revisit cancer cell growth dynamics** — the current k_C1_growth = 0.01/day is much smaller than CAF prolif rates (k_myCAF_prolif = 0.15/day), which is what drives the CAF-dominance at diagnosis. A submodel target on cancer growth from PDAC GEMM data could anchor this.
5. **Hard packing-budget constraint** (audit's Option B) is now lower priority — the diagnostic showed cellularity is set by growth-rate dynamics + carrying caps, not by V_T accounting. Hard packing constraint would prevent runaway but not fix the rate-race outcome.

## 7. Cross-references

- Parent audit: `notes/calibration/ecm_anchor_audit_2026-04-28.md`
- Tension diagnosis: `notes/calibration/untreated_tension_analysis_2026-04-28.md`
- Phi attractor diagnosis: `notes/calibration/collagen_attractor_overshoot.md`
- GitHub issue: #93
- Branch: `feature/ecm-anchor-audit-issue-93` (commits `83f2097`, `e45e690`, `6a5470d`, `851f25a`)
- Submodel target YAMLs:
  - `calibration_targets/submodel_targets/f_stroma_max_ogawa2021_PDAC_deriv001.yaml`
  - `calibration_targets/submodel_targets/k_stroma_support_hwang2008_PDAC_deriv001.yaml`
- Existing complementary targets (already in main):
  - `calibration_targets/submodel_targets/f_stroma_max_PDAC_deriv001.yaml` (Torphy 2018)
  - `calibration_targets/submodel_targets/k_stroma_support_PDAC_deriv001.yaml` (Xu 2010)
- SBI run logs:
  - Pre-CAF: `/tmp/sbi_run_ecm_baseline.log`
  - Post-CAF CSV-only: `/tmp/sbi_run_caf_baseline_retry.log`
  - Post-CAF +submodel-sync: `/tmp/sbi_run_caf_baseline_with_sync.log` (pending)
