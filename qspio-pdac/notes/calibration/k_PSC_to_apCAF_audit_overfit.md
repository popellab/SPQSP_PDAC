# k_PSC_to_apCAF audit overfit — diagnosis

**Date:** 2026-04-26
**Context:** Composite-prior diagnostic flagged `k_PSC_to_apCAF` with the
tightest support of all 113 audit-emitted parameters
(invgamma shape=1395, σ_yaml/σ_csv = 0.027, audit pinned 5.5× higher than
CSV median). Question: is the data really saying this, or is it a
parameterizer artifact?

**Conclusion:** Parameterizer artifact — narrow bridge in an over-merged
NPE component, not data-grounded precision. The posterior **does not
satisfy either of its own target observables**.

## Evidence

### What error propagation says the data implies

The submodel target (`k_PSC_to_apCAF_PDAC_deriv001.yaml`, Huang2022)
has two algebraic observables in a 3-parameter system:

- `f_apCAF = k_PSC_to_apCAF / (k_PSC_to_apCAF + k_PSC_to_iCAF + k_PSC_to_myCAF)`
  — bootstrap mean ≈ 0.413, SE-of-mean ≈ 0.146/√3 = 0.084 (n=3 KPfC tumors)
- `k_PSC_total = k_PSC_to_apCAF + k_PSC_to_iCAF + k_PSC_to_myCAF`
  — bootstrap mean ≈ 0.143/day, SE-of-mean ≈ 0.0092 (n=3)

Therefore the data implies:

- `k_PSC_to_apCAF = f_apCAF × k_PSC_total ≈ 0.413 × 0.143 = 0.0590/day`
- Delta-method propagation:
  `Var ≈ E[k_total]² Var(f_ap) + E[f_ap]² Var(k_total) ≈ 1.59e-4`
  `SD ≈ 0.0126` → **CV ≈ 21%**

### What the audit's NPE component actually produced

From `calibration_targets/submodel_targets/.compare_cache/comp_660d69460d846e37.json`:

| Param | Posterior median | Posterior CV |
|---|---|---|
| k_PSC_to_apCAF | 0.0512 | **1.7%** |
| k_myCAF_prolif | 0.815 | 1.3% |
| k_PSC_to_myCAF | 0.112 | 13% |
| k_PSC_to_iCAF | 0.024 | 61% |
| TGFb_50_CAF_act | 0.0026 | 86% |

CV is ~12× tighter than error-propagation predicts.

### The posterior fails both target observables

PPC table from the same cache:

| Observable | Observed (CI95) | Posterior median (CI95) | Covered? |
|---|---|---|---|
| `apCAF_fraction_kpfc_flow` | 0.412 (0.246–0.578) | **0.270 (0.243–0.296)** | **No** |
| `mesothelial_lineage_activation_rate` | 0.143 (0.125–0.161) | **0.190 (0.174–0.208)** | **No** |

The posterior internally consistent: `0.270 × 0.190 = 0.0513 ≈ 0.0512`.
But it disagrees with both targets — by ~50% on the ratio and ~33% on
the rate.

### Why is the posterior off?

The NPE component merges 5 parameters and 26 observables across **4
targets** (Huang2022 + bachem1992 + bachem1993 + biffi2019). Per
`submodel_config.yaml`, the merge is intentional via cascade-cut on
`k_PSC_to_myCAF` ("k_PSC_to_myCAF stays free and naturally merges the
activation targets with k_PSC_to_apCAF into one component").

The joint NPE finds a compromise that minimizes loss across all 26
observables but doesn't satisfy any individual target. SBC reports
k_PSC_to_apCAF as `underconfident` (KS p = 2.87e-7) — the marginal is
*wider* than the truth-of-the-NPE-model would warrant, but it's still
~12× tighter than the data-of-the-target. Both can be true: the NPE has
collapsed onto a narrow "bridge" in parameter space that is internally
self-consistent but data-mismatched.

7/26 observables in this component are not covered:
- both Huang2022 k_PSC_to_apCAF observables
- bachem1993 ASMA fractions at days 10 and 16
- bachem1993 cell count at day 6
- bachem1992 tgfb_asma at days 11 and 16

### Implication for composite prior

The composite prior pulls k_PSC_to_apCAF up to 0.0512/day (5× the CSV
median) with σ_log ≈ 0.017 — a near-deterministic constraint that
**does not match its own target data**. Higher activation into apCAFs
strengthens the apCAF→Treg loop, deepening the structural Treg-
dominated TME failure observed in the prior-predictive bounds
diagnostic.

## Recommended fixes (in priority order)

1. **Split this component on the `k_PSC_to_apCAF` target.** Force
   k_PSC_to_apCAF + k_PSC_to_iCAF + k_PSC_to_myCAF into a 3-param
   component scoped to *just* the Huang2022 target, with the bachem/
   biffi targets in a separate component. This requires either an
   override in `submodel_config.yaml` or a structural change to which
   targets share parameters.
2. **Switch this component to NUTS** (the inference is small enough —
   3 free params per target with algebraic forwards). NUTS will respect
   the data-implied broader posterior; NPE collapse won't occur.
3. **Investigate the other tight-CV bridge param `k_myCAF_prolif`
   (CV=1.3%)** — same pathology likely.

## Action items

- [ ] Open issue/PR against this component's configuration in
  `calibration_targets/submodel_targets/submodel_config.yaml` proposing
  a `component_overrides` entry (NUTS) for the Huang2022 target.
- [ ] Re-run audit with the override; confirm k_PSC_to_apCAF posterior
  matches `~0.059, CV ≈ 21%` as data warrants.
- [ ] Re-run composite-prior probe (`workflows/prior_predictive_bounds_
  diagnostic.py --prior composite`) with the corrected
  `submodel_priors.yaml`; check whether Treg-dominated failure mode
  softens.

## Files

- Target: `calibration_targets/submodel_targets/k_PSC_to_apCAF_PDAC_deriv001.yaml`
- Audit cache: `calibration_targets/submodel_targets/.compare_cache/comp_660d69460d846e37.json`
- Audit YAML: `notes/calibration/submodel_priors.yaml` (entry `k_PSC_to_apCAF`)
- Audit report: `notes/calibration/parameter_audit_report.md` (PPC and SBC tables)
- Submodel config: `calibration_targets/submodel_targets/submodel_config.yaml`
- Quantile diff that surfaced this:
  `/tmp/composite_prior_diff/log_space_diff.csv`
