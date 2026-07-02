---
date: 2026-04-27
status: investigation note; phi_cross_50 σ widened 1.2 → 2.0 (median unchanged); literature anchoring deferred
related: prior_audit_2026-04-27.md (§6.4); cdc1_density / apc_maturation cal targets
---

# Collagen attractor overshoot at PDAC baseline

## TL;DR

At prior medians, `phi_collagen` reaches a stable attractor at **~0.58** during `evolve_to_diagnosis` and stays there indefinitely. This drives `H_pore_APC = 0.75` (75% of cDC recruitment blocked), which fully explains the 6× cDC1 density gap vs Plesca 2022 cal target (model 1.26 cells/mm² vs obs 7.82 cells/mm²) — and accounts for most of the modest apc_maturation_ratio overshoot too.

The attractor is set by three compounding factors in the collagen rate equation:
1. TGFb saturates the secretion modulator (TGFb=0.79 nM ≫ TGFb_50_col_sec=0.07 nM).
2. The MMP-M2 degradation path is gated off by `f_nascent = phi_cross_50/(phi_cross_50+phi)` — at `phi_cross_50=0.03` the gate closes essentially immediately as collagen accumulates above 3% volume fraction.
3. Constitutive `k_col_deg = 0.007/day` (~99-day half-life) is much slower than secretion.

**Action taken in this commit (and only this):** widened `phi_cross_50` σ from 1.2 → 2.0 (median unchanged at 0.03). New CI95 ≈ [6e-4, 1.5]. Per `feedback_no_prior_tuning`, shifting the median requires an independent biological anchor; the original 0.03 was a deliberate trade-off targeting Sivakumar 2008's 60-100d in vivo collagen t½ (per `parameters_PDAC.m:1840` comment). The widened σ gives SBI room to navigate via the cDC1 / maturation observables now in the loop.

**Deferred:** principled re-anchoring of `phi_cross_50` (and the partner `k_col_deg`) via maple submodel-target extraction targeting MMP-13 / TAM activity on mature crosslinked collagen and PDAC stromal collagen turnover separately from osteoid. That is the real fix; this widening is the conservative-but-honest interim move.

## Diagnostic trace

Single sim at prior-median XML (`resources/cpp/param_all.xml`), `baseline_no_treatment` scenario with 1095-day extension. Evolve hits diagnosis at t=908 d, V_T = 17.18 mL, V_T.collagen = 248 mg.

### Hill values at PDAC baseline (post-evolve)

```
H_DAMP      = 0.997   (saturated by tumor cell death)
H_IL12      = 0.348
H_IL10      = 0.463
H_TGFb_APC  = 0.918
H_pore_APC  = 0.750   ← the load-bearing one
phi_collagen = 0.577
d_pore       = 1.60 µm   (vs d_crit_APC = 2.1 µm → exclusion regime)
```

`H_pore_APC = 1 / (1 + (d_pore/d_crit_APC)^n_pore)` with n_pore=4 → sharp transition near d_crit. At phi=0.58 we're solidly in the "DC-excluded" regime. Sensitivity table:

| `phi_collagen` | `d_pore` (µm) | `H_pore_APC` | (1−H_pore_APC) recruitment fraction |
|---|---|---|---|
| 0.12 (normal pancreas reference) | 3.50 | 0.11 | 0.89 |
| 0.20 | 2.71 | 0.26 | 0.74 |
| 0.30 | 2.21 | 0.45 | 0.55 |
| 0.50 | 1.71 | 0.69 | 0.31 |
| **0.58 (model PDAC baseline)** | **1.60** | **0.75** | **0.25** |
| 0.70 | 1.45 | 0.82 | 0.18 |

If phi were 0.20-0.30, recruitment would be 2-3× higher and cDC1 density would close most of the gap with the 7.82 cells/mm² target.

### Trajectory: phi_collagen is at attractor, not still climbing

```
day    V_T.collagen(mg)    V_T(mL)    phi_collagen    H_pore_APC
  0          248             17.18       0.577           0.750
 30          280             19.35       0.578           0.750
180          496             34.22       0.579           0.751
365          962             66.24       0.581           0.752
```

Collagen and V_T grow in lockstep (~4× over 365 days), keeping phi pinned. phi is at quasi-steady state, not in a transient runaway. Whatever sets the attractor is a stable feature of the rate equation at these prior medians.

### Rate balance at t=0 (post-evolve, prior medians)

Reaction (`fibroblast_module_enhanced.m:544-551`):
- Secretion: `k_col_myCAF_sec × (1 − phi_collagen) × V_T.myCAF × (f_col_basal + (1−f_col_basal) × TGFb/(TGFb_50_col_sec + TGFb))`
- Degradation: `(k_col_deg + k_MMP_M2 × V_T.Mac_M2 / V_T × f_nascent) × V_T.collagen`
- where `f_nascent = phi_cross_50 / (phi_cross_50 + phi_collagen)`

At t=0 (V_T.myCAF=1.9e9, V_T.Mac_M2=1.6e8, V_T.TGFb=0.79 nM, phi=0.577):

```
TGFb modulator (f_basal + (1−f_basal)·TGFb/(TGFb_50+TGFb)) = 0.96   (saturated)
f_nascent = 0.03 / (0.03 + 0.577) = 0.0494                           (gate ~5% open)

Secretion:        3.86 mg/day   (TGFb-saturated, partial damping from (1-phi)=0.42)
Constitutive deg: 1.74 mg/day   (k_col_deg=0.007/day on 248 mg)
MMP deg:          1.13 mg/day   (M2 path gated to ~5% of capacity)
Net dCol/dt:     +1.00 mg/day   (small positive; tracks V_T expansion)
```

Quasi-equilibrium (ignoring the f_nascent term and V_T expansion): solving secretion = constitutive deg gives `phi_eq ≈ 0.75`. The MMP-M2 path is what holds the attractor below 0.75 down to 0.58, but only weakly because `f_nascent` is small.

## Why `phi_cross_50` is the load-bearing prior

`f_nascent = phi_cross_50 / (phi_cross_50 + phi_collagen)` represents the fraction of collagen that's still "nascent" (un-crosslinked, MMP-susceptible) as a function of total phi. The biological intent: as collagen accumulates and ages, LOX cross-links protect it from MMP cleavage.

The current prior median `phi_cross_50 = 0.03` puts the half-saturation at phi=0.03 — i.e., **MMP susceptibility falls below 50% as soon as collagen volume fraction exceeds 3%**. Normal pancreas reference is `phi_col_ref = 0.12`. So under the current model, MMPs are essentially shut off in ALL physiological collagen regimes (normal AND fibrotic), regardless of M2 macrophage abundance.

**The original 0.03 was a deliberate trade-off**, not an unanchored guess: per the comment in `parameters_PDAC.m:1840`, the modeler picked it to make the f_nascent shield tight enough to reproduce a 60-100d in vivo collagen half-life target (the partner anchor for `k_col_deg = 0.007/day`). The system is internally consistent against that target. The question is whether the *target* (Sivakumar 2008's 60-100d) is the right one for PDAC stromal collagen.

There is anecdotal/secondary evidence in CLAUDE.md and `parameters_PDAC.m` that M2 macrophages are effective remodelers of mature fibrotic collagen in PDAC ("Madsen 2013, LaRue 2022" cited in the M2-MMP comment, and "Apte 2004" cited for normal pancreas phi=0.10-0.15). **These references are propagated from existing project documentation; this note has not independently verified them.** Resolving the prior properly requires (a) verifying the source claims and (b) running maple submodel-target extraction to anchor `phi_cross_50` and `k_col_deg` against verifiable PDAC-stromal-collagen data, not osteoid.

### Why σ-widening is the honest interim move

Per `feedback_no_prior_tuning`: only widen, don't shift the median, until you have an independent biological anchor. The empirical trace is a strong signal that *something* is off in the {`k_col_deg`, `phi_cross_50`, secretion saturation} chain, but back-deriving a new median from "the value that makes the cDC1 cal target fit" is exactly the prior tuning the rule prohibits — even if I dress it up as forward-model fixing.

What was done: `phi_cross_50` σ widened 1.2 → 2.0 (CI95 ≈ [6e-4, 1.5]). Median unchanged. SBI now has the cDC1 density and apc_maturation_ratio observables, which are sensitive to phi_collagen via H_pore_APC and the maturation Hill suppression. Calibration can pull `phi_cross_50` upward if the data warrants — and we'll learn what value the data prefers.

What was *not* done: median shift. That's deferred to a literature anchoring sprint.

## Other candidate fixes (lower leverage, listed for completeness)

In rough order of empirical leverage on the attractor:

2. **`k_col_deg` (constitutive degradation rate)** — currently 0.007/day (Sivakumar 2008 anchor, 99-day half-life). Sivakumar studied osteoid (bone matrix), not stromal collagen. PDAC stromal collagen turnover anchors are scarce, but if the true half-life were 30-50 days (k ≈ 0.014-0.023), constitutive deg at baseline would jump from 1.74 to 3.5-5.7 mg/day, attractor drops to ~0.40-0.45. Worth a separate literature pull. (Audit currently flags ✓ OK.)

3. **`k_col_myCAF_sec` (secretion rate)** — current prior 5e-9 mg/cell/day; audit §6.4 already flags as 2-20× LOW vs Bachem 1992 (1e-8 to 1e-7 mg/cell/day) and recommends shifting to 5e-8. **Increasing the prior would make the attractor worse, not better.** This conflict is a flag for the audit recommendation: applying §6.4's recommended 10× bump to k_col_myCAF_sec without simultaneously raising degradation would push phi to >0.8.

4. **`TGFb_50_col_sec`** — currently 0.07 nM, well below baseline TGFb of 0.79 nM. Even raising to 1 nM would only drop the secretion modulator from 0.96 to ~0.72 (~25% reduction). Modest leverage.

5. **`(1−phi_collagen)` saturation in secretion** — current form is linear in (1−phi). Steeper forms like `(1 − phi/phi_max)^2` would damp secretion harder near saturation. Structural change to the model, not a prior change. Defer unless the EC50 fixes don't suffice.

6. **`rho_collagen = 0.025 g/mL`** — packing density. Audit §6.4 flags ✓ OK with van der Rest 1991 anchor for fibrillar packing. The `phi_collagen` formula `V_T.collagen/(V_T·rho)` then interprets phi=0.58 as "58% of tumor volume is collagen at fibrillar packing density." That's biologically suspect (cells need volume) but it's the model's accounting convention; raising rho would lower phi but break the anchor with the cited biology, and would also affect H_stiff scaling. Don't touch without a separate audit pass.

## Cellularity-fraction sanity check on the cal target

The cal target observable converts cancer cell count to tumor area via `tumor_area = C1 × area_per_cell / cellularity_fraction` with `cellularity_fraction = 0.15` (Plesca-cohort assumption). If true PDAC cellularity is lower (some PDAC subtypes have <10%), tumor_area would be larger, observed density lower. Setting cellularity to 0.10 would scale the cal target target from 7.82 down to ~5.2 cells/mm² — meaningful but doesn't close the 6× gap.

The cellularity assumption is in the cal target YAML (`calibration_targets/baseline_no_treatment/cdc1_density_baseline_PDAC_deriv001.yaml`) sourced from `pdac_cellularity_fraction` reference DB. Worth confirming that 0.15 actually matches Plesca 2022's cohort histology, but it's a secondary concern — the model-side gap dominates.

## Action items

**Done in this commit:**
1. **Widen `phi_cross_50` σ from 1.2 → 2.0** in `pdac_priors.csv`. Median unchanged at 0.03. New CI95 ≈ [6e-4, 1.5].
2. **Update `prior_audit_2026-04-27.md` §6.4** to record the σ widening, mark `k_col_deg` for anchor review, and document the BLOCK on the Bachem 1992 `k_col_myCAF_sec` bump until `phi_cross_50` is resolved.

**Deferred (real fix):**
3. **Verify the cited references** — LaRue 2022, Madsen 2013, Apte 2004, Sivakumar 2008 — all propagated from CLAUDE.md / existing audit / `parameters_PDAC.m` comments without independent verification in this session. Per `feedback_no_fabricated_citations`, do this before any of them is used as load-bearing for a parameter shift.
4. **Maple extraction sprint** — submodel targets for (a) PDAC stromal collagen turnover rate (separate from osteoid), (b) MMP-13 / TAM activity on mature crosslinked collagen, (c) PDAC stromal collagen volume fraction at diagnosis (cross-check Apte 2004). With those in hand, `phi_cross_50` and `k_col_deg` can be re-anchored together.
5. **Block the recommended `k_col_myCAF_sec` 10× bump** (audit §6.4) until phi_cross_50 is resolved. Applying secretion bump alone makes the attractor catastrophically worse.
6. **Re-run this trace after the next SBI run** with the widened σ to see whether calibration pulls `phi_cross_50` upward and how far. That posterior is the empirical signal of where the median actually wants to sit, which guides where to focus the literature anchoring effort.

**Reproducibility one-liner** (re-trace after any related change):
```
qsp_sim --param resources/cpp/param_all.xml \
        --evolve-to-diagnosis resources/cpp/healthy_state.yaml \
        --scenario scenarios/clinical_progression.yaml \
        --t-end-days 1095 --dt-days 30 \
        --csv-out /tmp/dc_diag/long.csv
```
Inspect `phi_collagen`, `H_pore_APC`, `f_nascent` at the post-evolve state. Target after literature anchoring: phi_collagen ≤ 0.30 at PDAC baseline.

## Reproducibility

```bash
cd /Users/joeleliason/Projects/pdac-build-prior-diag
./cpp/sim/build/qsp_sim \
  --param resources/cpp/param_all.xml \
  --evolve-to-diagnosis resources/cpp/healthy_state.yaml \
  --scenario scenarios/baseline_no_treatment.yaml \
  --drug-metadata resources/cpp/drug_metadata.yaml \
  --csv-out /tmp/dc_diag/species.csv \
  --t-end-days 1.0 --dt-days 0.5
```

Inspect `H_pore_APC`, `phi_collagen`, `H_TGFb_APC`, `H_IL10`, `f_nascent` at t=0 (post-evolve).
