# Cluster C structural diagnostic — sim-count sweep (5k → 20k τ=1.0 baseline)

**Date:** 2026-04-30
**Branch:** `feature/cluster-c-apcaf-tetrahedron`
**Pool:** `multi_scenario_v5_qual_migration_ed91a7b2_baseline_no_treatment` (fresh hash from qsp-codegen byte-determinism rebuild — original 5k pool `c8b73f69` orphaned)
**Logs:**
- 10k: `/tmp/sbi_run_cluster_c_10k_tau10_20260430_111316.log`
- 20k: `/tmp/sbi_run_cluster_c_20k_tau10_20260430_113306.log`

## TL;DR

Sweeping NPE training data from 5k → 10k → 20k sims revealed that **most of the "Cluster C tetrahedron" was an NPE under-training artifact**, not a structural problem:

- At 5k, 6 misspecified targets with strong mutual conflicts (`cdc1×m1_m2` cos=-0.69 top).
- At 10k, the apCAF hub appears: 7 misspecified targets, `apcaf×cdc1` cos=-0.79 — the handoff's framing.
- **At 20k, only 3 marginal misfits remain — *and none of them have competing-target conflicts*.** The conflict tetrahedron disappears.

The handoff-proposed structural fixes (second apCAF source, σ-widen `APC0_cDC1_T`) are **no longer motivated** — both targets self-resolve with sufficient training data.

The single genuinely structural item is **`stromal_fraction`, top driver `Ve_T`** — exactly where the single-θ trace pointed. This is the only Cluster C structural fix to scope.

## Sim-count sweep summary

| | 5k | 10k | 20k |
|---|---|---|---|
| Best val loss | 140.19 | 129.96 | **121.49** |
| Jacobian effective rank | 3.0/22 | 3.7/22 | **5.3/22** |
| D²_post / D²_prior | (n/a) | 1.53 | **1.44** |
| Misspecified targets | 6 | 7 | **3** |
| Conflicting target pairs | 1 (cdc1×m1_m2) | 5+ (apCAF hub) | **0** |

Doubling sims 5k→10k *increased* visible misfits (NPE finally resolves apCAF). Doubling 10k→20k *collapses* the conflict structure (NPE disambiguates).

## 20k posterior-predictive misspec table (canonical)

```
3 targets outside 90% posterior predictive CI:

  cdc1_density_baseline  MARGINAL  z=-1.93  top driver: n_CD8_clones    no conflicts
  m1_m2_ratio            MARGINAL  z=-1.16  top driver: k_PSC_to_apCAF  no conflicts
  stromal_fraction       p=0.0000  z=+1.59  top driver: Ve_T            no conflicts
```

`stromal_fraction` is the dominant misfit (`p=0.0000`); the other two are marginal.

Top elasticities (|d log obs / d log θ|) at 20k all driven by `k_PSC_to_apCAF`:

| observable | elasticity | top driver |
|---|---|---|
| `treg_density_baseline` | 23.28 | `k_PSC_to_apCAF` |
| `cd4_helper_density` | 20.61 | `k_PSC_to_apCAF` |
| `cd8_density_baseline` | 15.29 | `k_PSC_to_apCAF` |
| `total_til_density` | 12.91 | `k_PSC_to_apCAF` |
| `treg_fraction_baseline` | 10.37 | `k_PSC_to_apCAF` |

`k_PSC_to_apCAF` is the dominant routing lever, but at 20k the NPE has learned to position it correctly — the elasticity dominance no longer translates to misfits.

## 10k diagnostic snapshot (kept for reference)

At 10k τ=1.0 the picture was:

```
Posterior misfits (z-scores):
  cd8_density_baseline   z=-6.57  ← extreme
  apcaf_fraction_of_caf  z=-3.30
  cdc1_density_baseline  z=-1.99
  stromal_fraction       z=+1.68  (p=0.0000)
  m1_m2_ratio            z=-1.37
  vegf_tissue_concentration z=+1.15
  tam_density            z=+0.94

Top conflicting pairs (Jacobian cosines):
  apcaf_fraction_of_caf × cdc1_density_baseline    cos=-0.79  (top hub)
  cdc1_density_baseline × m1_m2_ratio              cos=-0.52
  apcaf_fraction_of_caf × cd8_density_baseline     cos=-0.45
  apcaf_fraction_of_caf × vegf_tissue_concentration cos=-0.41
  apcaf_fraction_of_caf × stromal_fraction         cos=-0.37
  apcaf_fraction_of_caf × tam_density              cos=-0.31
```

At 10k it looked like the handoff's apCAF-hub tetrahedron was real. The 20k run shows it was a sample-count-induced over-attribution: NPE tied apCAF and cdc1 together because it couldn't separately resolve their underlying drivers (`k_PSC_to_apCAF` and `n_CD8_clones` respectively) on 10k samples.

## Posterior predictive misspec table (10k)

```
OUTSIDE: apcaf_fraction_of_caf:    obs=0.0643, 90%CI=[0.0022, 0.0633], p=0.10
OUTSIDE: cd8_density_baseline:     obs=139,    90%CI=[0.0012, 39.9],  p=0.06
OUTSIDE: cdc1_density_baseline:    obs=7.82,   90%CI=[0.537, 4.73],   p=0.03
OUTSIDE: m1_m2_ratio:              obs=0.518,  90%CI=[0.0667, 0.243], p=0.01
OUTSIDE: stromal_fraction:         obs=0.668,  90%CI=[0.799, 0.96],   p=0.00
OUTSIDE: tam_density:              obs=120,    90%CI=[133, 534],      p=0.05
OUTSIDE: vegf_tissue_concentration: obs=145,    90%CI=[147, 1400],     p=0.10
```

z-scores:

| target | z | direction |
|---|---|---|
| `cd8_density_baseline` | -6.57 | sim too low (extreme) |
| `apcaf_fraction_of_caf` | -3.30 | sim too low |
| `cdc1_density_baseline` | -1.99 | sim too low |
| `stromal_fraction` | +1.68 | sim too high (p=0.0000 — biggest miss) |
| `m1_m2_ratio` | -1.37 | sim too low |
| `vegf_tissue_concentration` | +1.15 | sim too high |
| `tam_density` | +0.94 | sim too high |

## Posterior Jacobian — top conflicting pairs

```
apcaf_fraction_of_caf  × cdc1_density_baseline      cos=-0.79  ← #1 hub
cdc1_density_baseline  × m1_m2_ratio                cos=-0.52
apcaf_fraction_of_caf  × cd8_density_baseline       cos=-0.45
apcaf_fraction_of_caf  × vegf_tissue_concentration  cos=-0.41
apcaf_fraction_of_caf  × stromal_fraction           cos=-0.37
apcaf_fraction_of_caf  × tam_density                cos=-0.31
```

apCAF participates in 5 of the top 6 conflicting pairs — it is the central tension hub.

Top elasticities (|d log obs / d log θ|):

| observable | elasticity | top driver |
|---|---|---|
| `apcaf_fraction_of_caf` | 17.22 | `k_PSC_to_apCAF` |
| `collagen1_protein_level` | 14.73 | `k_PSC_to_apCAF` |
| `cd8_treg_ratio_golesworthy2022` | 8.10 | `k_PSC_to_apCAF` |
| `mycaf_fraction_of_caf` | 6.87 | `k_PSC_to_apCAF` |
| `treg_density_baseline` | 5.65 | `k_Treg_pro_tumor` |

`k_PSC_to_apCAF` dominates 4 of the top 5 most-sensitive observables — the routing-bottleneck pattern is real. SVD effective rank: 3.7/22 (vs 3.0/22 at 5k — slightly more directions resolved).

## What changed 5k → 10k

| | 5k τ=1.0 | 10k τ=1.0 |
|---|---|---|
| Top conflicting pair | `cdc1×m1_m2` (cos=-0.69) | `apcaf×cdc1` (cos=-0.79) |
| `apcaf_fraction_of_caf` flagged | no (posterior pinned at ~0.051, sd=0.0009) | **yes** (z=-3.30, hub) |
| `cd4_helper_density` | flagged z=-5.68 | **resolved** |
| `treg_fraction_baseline` | flagged | resolved |
| `cd8_density_baseline` | marginal | **z=-6.57**, extreme miss |
| `stromal_fraction` | z=+1.70 | z=+1.68, p=0.0000 |
| Best val loss | 140.19 | 129.96 |

## Single-θ trace context (from earlier session, posterior median of 5k run)

| target | sim @ post med | obs (CI95) | verdict |
|---|---|---|---|
| `cdc1_density_baseline` | 12.18 | 7.82 [2.13, 29.22] | within CI |
| `m1_m2_ratio` | 0.224 | 0.518 [0.071, 3.72] | within CI |
| `apcaf_fraction_of_caf` | 0.0078 | 0.0643 [0.02, 0.21] | 8× below CI |
| `stromal_fraction` | 0.907 | 0.668 [0.424, 0.863] | above CI |

`k_PSC_to_apCAF` sweep showed apCAF target reachable at 0.5/day (~10× posterior median, prior 97.7%-ile). V_T volume composition: ~11.2 mL with only ~3.7 mL accounted for by cancer + tracked stromal cells + collagen → ~7.5 mL "structural" interstitial volume. Target cellularity 0.332 requires V_T ≈ 3.1 mL OR 3.6× more cancer cells per V_T.

## Revised structural-fix priority (post-20k)

1. **`stromal_fraction` — V_T volume rule fix via `Ve_T`.** The 20k posterior Jacobian directly identifies `Ve_T` (tumor extracellular volume) as the top elasticity driver. The single-θ trace's "7.5 mL of unaccounted interstitial volume in V_T at posterior median" maps onto this exact parameter. **This is the only Cluster C structural fix to scope — see `cluster_c_ve_t_fix_*.md` for the working plan.**

2. **`cdc1_density_baseline` MARGINAL — top driver `n_CD8_clones`.** Surprising at 20k: the misfit is now driven by a CD8-clonality parameter, not by APC0_cDC1_T. Likely indirect (CD8 abundance → APC turnover / cross-presentation feedback). Marginal (z=-1.93); worth understanding before any fix. May further resolve at higher sim count.

3. **`m1_m2_ratio` MARGINAL — top driver `k_PSC_to_apCAF`.** Marginal residual (z=-1.16) with no conflicts. Probably just below the noise threshold; defer until #1 lands and rerun.

## Demoted from the priority list

- **Second apCAF source** (handoff "Fix #1"). `apcaf_fraction_of_caf` resolved at 20k — the apCAF hub at 10k was an NPE under-training artifact.
- **σ-widen `APC0_cDC1_T`** (handoff "Fix #3"). cDC1 misfit driver is `n_CD8_clones`, not `APC0_cDC1_T`. The σ-widen would chase a phantom misfit. (Already demoted at 5k trace stage; 20k confirms.)
- **`cd4_helper_density` Th-pro / regression follow-up** (handoff secondary thread). Resolved at 10k, stayed resolved at 20k.
- **`cd8_density_baseline` extreme miss** (10k z=-6.57). Resolved at 20k. Was an NPE under-training artifact.

## Methodology note

This run sequence is a textbook illustration of `feedback_audit_vs_npe_quality.md` — distinguishing NPE-quality misfits from structural misfits requires running the SBI past its under-training regime. At 5k or 10k it would have been easy to commit to a multi-month "second apCAF source" structural project that turns out to be unnecessary at 20k. The marginal cost of doubling sims is far cheaper than the cost of an unnecessary mechanism addition.

For future structural debugging on this model, **20k τ=1.0 should be the floor for declaring a misfit "structural"** — anything flagged at lower sim counts should be re-examined at 20k before committing to a structural fix.

## Stromal_fraction observable formula fix (intra-session, no re-sim)

Bengtsson 2026's CA19-9 segmentation uses Gaussian-blur + particle-size filtering, which captures cancer **epithelial regions** (cell bodies + tight intercellular spaces in nests), not bare cancer cell volume. The model's V_T volume rule allocates `vol_cancer/Ve_T` of compartment volume per cell. The correct stromal_fraction observable is therefore:

```python
stromal_fraction = 1 - (C1 * vol_cancer / Ve_T) / V_T   # cancer epithelium / V_T
```

The previous formulation `1 - C1*vol_cancer/V_T` (bare cancer volume) systematically allocated each cancer cell's interstitial share to "stroma", inflating the observable. At posterior median, the bare-volume form gave 0.91 vs target 0.668; the cancer-epithelium form gives 0.83 (within CI95 [0.424, 0.863]).

Updated `calibration_targets/baseline_no_treatment/stromal_fraction_PDAC_deriv001.yaml` with the corrected formula. Test stats hash flipped (838f9fe1 → caaa6ade) — reused existing 20k pool, only re-derived test stats. Effect on posterior PC: `stromal_fraction` z=+1.59 → z=-0.19, dropped from misspec list. Marginal cdc1/m1_m2 unchanged.

## Classifier drift bug + retrain (intra-session)

After the stromal-fix run, 58.6% of sims still NaN'd. Diagnosis:

- `tests/fixtures/restriction_classifier/` was trained pre-PR-111 (Cluster B Th-exh fix), 190 features.
- Current prior has 193 features (3 new: `k_Th_basal_exh`, `k_TGFb_Th_exh`, `TGFb_50_Th_exh`).
- Projection (`accept_named` → `project`) handles the missing features correctly — drops them on classifier input.
- BUT the simulator's NaN profile changed post-PR-111. The classifier still scores the new theta at high probabilities (median 0.90 at threshold 0.5), but those "valid" predictions no longer match the post-PR-111 simulator's behavior — drift between trained and live model.
- Survival 41% (= unfiltered baseline 44.8%) confirmed the classifier was effectively bypassed.

Retrained against the current 20k ed91a7b2 pool (which has live NaN labels for the post-PR-111 model):

```
5-fold CV AUC = 0.9851  (was 0.950 — improved with current model dynamics)
baseline_survival = 0.414  (= unfiltered NaN rate of current model)
```

Geometric threshold sweep (via `workflows/restrict_threshold_geometry.py`):

| τ | accept | survival | yield | max-KS |
|---|---|---|---|---|
| 0.5 | 0.413 | 0.993 | 0.410 | 0.203 |
| **0.7** | **0.407** | **0.997** | **0.406** | **0.205** |
| 0.85 | 0.390 | 0.999 | 0.390 | 0.212 |
| 0.9 | 0.352 | 0.999 | 0.352 | 0.231 |

τ=0.7 is the geometric optimum: highest threshold that keeps max-KS pinned at the joint-constraint floor (0.205), saturates survival at 99.7%, only adds 0.6% prior oversampling vs τ=0.5. Updated `sbi_runner.py` default 0.5 → 0.7.

Verification run (20k fresh on new pool 2529e2fa, τ=1.0): NaN rate **1.8%** (was 58.6%), val loss **116.65** (was 121.83), effective rank **7.0/22** (was 5.3).

## Final state — 20k τ=0.5 (default inference) on the post-fix branch

```
2 targets outside 90% posterior predictive CI (no longer coupled):

  cdc1_density_baseline   z=-1.46   p=0.0000   top driver: APC0_cDC1_T   no conflicts
  m1_m2_ratio             z=-0.84   p=0.0839   top driver: k_M2_pol       no conflicts

Best val loss:  114.17       (down from 116.65 at τ=1.0)
Effective rank: 5.7/22
D²_post / D²_prior: 1.237   (down from 1.485 at τ=1.0)
NaN rate:       1.8%
```

## τ=1.0 vs τ=0.5 verdict on the marginals

| target | τ=1.0 z | τ=0.5 z | reading |
|---|---|---|---|
| `cdc1_density_baseline` | −1.74 | **−1.46** | persists, mildly improved |
| `m1_m2_ratio` | −1.07 | **−0.84** | persists, mildly improved |
| `cdc1 × m1_m2` cosine | −0.47 | **none** | conflict was τ=1.0-only |
| `cdc1` top driver | n_CD8_clones | **APC0_cDC1_T** | driver attribution shifts |

**Both marginals survive cohort-RE absorption — they're structural residuals, not diagnostic-mode noise.** But:

1. They're independent at τ=0.5 (no `cdc1×m1_m2` conflict). The τ=1.0 conflict was wide-cohort-RE artifact.
2. z magnitudes shrink slightly; both stay above the visibility threshold.
3. Driver attribution flips for cdc1: τ=1.0 said `n_CD8_clones`, τ=0.5 says `APC0_cDC1_T`. The τ=0.5 reading is more credible (tighter likelihood, less cohort-noise injection) and brings the handoff's σ-widen-`APC0_cDC1_T` candidate back into play, this time with proper diagnostic basis.

Both are mild enough to not block; standalone follow-ups recommended:

- **`APC0_cDC1_T` σ-widen audit** — re-examine Plesca 2022 reported spread (0.3-30 cells/mm²) to justify σ ≥ 1.0 prior.
- **`k_M2_pol` audit** — macrophage M0→M2 polarization rate; trace to see whether prior median is too low.
