# Structural Treg-CD8-M2-APC vortex under composite prior

**Date:** 2026-04-26
**Source:** 1000-draw composite-prior probe
(`results/prior_diagnostic/composite/{report,probe_data,summary}.{md,npz,json}`),
`tests/qualitative_bounds.yaml`, audit cache files in
`calibration_targets/submodel_targets/.compare_cache/`.

## TL;DR

Under the live composite prior (Stage-1-calibrated, the prior NPE
actually sees), the prior-predictive distribution at diagnosis collapses
into an over-immunosuppressed cascade: **0/441 valid sims pass all 16
snapshot bounds**. The dominant failures cluster in a tight
co-failure block (Treg dominance ⇄ M2 dominance ⇄ APC suppression ⇄
CD8 collapse). PRCC drivers and audit-shift analysis show the cascade
emerges from legitimately-calibrated submodel rates that, when
combined in the full QSP, push the immune compartment past a
mechanistic tipping point. **This is a model-coupling issue, not a
parameterizer artifact (with one exception, k_PSC_to_apCAF).**

This complements `notes/qual_bounds_state_summary.md` (2026-04-22,
posterior-median snapshot — same failure mode at a single point) by
showing the cascade is robust across the prior-predictive sample, not
just at the mode.

## Failure block

From the 1000-draw composite probe (16 snapshot bounds, 441 valid
sims). Shown: % failed and PRCC top driver per bound.

| Bound | Fail % | Top PRCC driver | sign |
|---|---:|---|---|
| `cd8_treg_ratio_max_diagnosis` | 94% | `k_Treg_pro_tumor` | -0.23 |
| `treg_fraction_min` | 86% | `k_Treg_death` | +0.29 |
| `m2_m1_ratio_range` | 83% | `k_M1_pol` | +0.36 |
| `apc_maturation_min_diagnosis` | 83% | `f_active_TGFb` | -0.40 |
| `treg_fraction_max_diagnosis` | 79% | `k_Treg_death` | +0.37 |
| `icaf_fraction_range` | 48% | `k_iCAF_to_myCAF` | -0.25 |
| `ecm_fold_change_range_diagnosis` | 31% | `k_MMP_M2` | -0.26 |
| `mycaf_fraction_min` | 18% | `k_iCAF_to_myCAF` | +0.29 |

The top-5 binding bounds co-fail almost in lockstep (counts of
co-failure pairs):

| count | A | B |
|---:|---|---|
| 372 | cd8_treg_ratio | treg_fraction_min |
| 347 | cd8_treg_ratio | m2_m1_ratio |
| 347 | treg_fraction_max | treg_fraction_min |
| 342 | apc_maturation | cd8_treg_ratio |
| 327 | m2_m1_ratio | treg_fraction_min |
| 315 | apc_maturation | treg_fraction_min |
| 311 | apc_maturation | m2_m1_ratio |
| 305 | m2_m1_ratio | treg_fraction_max |

Failure fingerprint (median values across failing draws):

| Species | Failing-draw median |
|---|---:|
| Treg (V_T) | ~6.4×10⁸ cells |
| CD8 (V_T) | ~3.1×10⁴ cells |
| Mac_M2 (V_T) | ~8.3×10⁸ cells |
| Mac_M1 (V_T) | ~8.0×10⁷ cells |
| cDC1 (V_LN, immature) | ~6.8×10⁶ |
| mcDC1 (V_LN, mature) | ~459 |

**CD8/Treg ratio ≈ 5e-5** (literature ~3, bound ≤ 15).
**M2/M1 ≈ 10** (literature ~2, bound 1–5).
**Mature/immature DC ≈ 1e-4** (bound ≥ 0.001).

## Why the cascade emerges — audit-shift analysis

How the composite prior shifts the binding-driver priors away from CSV
defaults (in CSV-σ_log units):

| Param | CSV median | Audit median | Shift (σ) | Audit data-grounded? |
|---|---:|---:|---:|---|
| `k_M2_pol` | 0.10 | 0.233 | **+1.69** | yes — PPC covered (DeSimoni2023 CD163/iNOS = 19) |
| `k_PSC_to_apCAF` | 0.01 | 0.055 | **+1.70** | **NO — PPC fails both target obs** (see `k_PSC_to_apCAF_audit_overfit.md`) |
| `TGFb_50_APC` | 0.07 nM | 0.018 nM | **−1.46** | yes — PPC covered (own target's IL-12 IC50) |
| `k_Mac_death` | 0.02 | 0.028 | +0.49 | yes (group-pooled, partial constraint) |
| `k_Treg_death` | 0.10 | 0.055 | −0.49 | yes |
| `k_Treg_pro_tumor` | 0.20 | 0.133 | −0.58 | yes |
| `k_apCAF_prolif` | 0.05 | 0.049 | −0.02 | (negligible shift) |
| `k_M1_pol` | 0.045 | 0.036 | −0.22 | yes (same M2pol component) |

### Mechanism

- **M2 polarization +1.7σ**: macrophages polarize to M2 ~5× faster
  than the CSV default. With CSV M1pol ≈ M2pol, the macrophage pool
  is roughly balanced. Under the audit, M2pol/M1pol ≈ 7×, and
  whichever branch wins amplifies via downstream cytokines (IL-10
  promotes more M2; IFNg from CD8 promotes M1). Once M2 wins
  numerically, IL-10/TGFb suppress CD8 → less IFNg → fewer M1 →
  positive feedback. This is mechanistically correct given the
  DeSimoni2023 IHC ratio, but the QSP pushes far past the empirical
  ~2:1 ratio because the loop gain is too high.
- **TGFb_50_APC −1.5σ**: APCs become ~3× more sensitive to TGFb
  suppression. When myCAFs/M2s pump out TGFb (which they already do
  abundantly given the desmoplastic priors — see TGFb-Hill-saturation
  xfails in `qual_bounds_state_summary.md` Part 3), the lower EC50
  shuts down DC maturation entirely (mcDC1 ≈ 459 vs cDC1 ≈ 7×10⁶).
  Without mature APCs, CD8 priming in V_LN collapses and Treg
  priming via cDC2 wins by default.
- **k_Treg_death −0.5σ + k_Treg_pro_tumor −0.6σ**: Treg dynamics are
  partially compensating (slower proliferation, slower death), so net
  effect on Treg pool is small — it's the *upstream* environment
  (apCAF abundance, TGFb tone, no CD8/IFNg counterforce) that allows
  Tregs to expand unchecked.
- **k_PSC_to_apCAF +1.7σ** (parameterizer artifact, see separate note):
  apCAFs proliferate ~5× more than CSV. apCAFs sequester antigen and
  prime Tregs via MHC-II → adds another Treg pump on top of the
  legitimate immunosuppressive cascade. Removing this shift wouldn't
  fix the cascade alone, but it deepens it.

## What this means

The Treg-vortex is an emergent property of the QSP coupling — three
data-grounded submodel calibrations (M2 polarization, APC TGFb
sensitivity, Treg dynamics) are individually correct, but their joint
effect at the population level produces an immune-cold TME more
extreme than literature observes. This is the gap that single-target
audit cannot fix:

- Each submodel's PPC says "I match my data."
- The joint qualitative-bounds checker says "but together you produce
  CD8/Treg = 5e-5, which violates literature by 5 orders of
  magnitude."

**The cascade is mechanistically real but the model's loop gains are
too high.** Plausible structural levers (in priority order, by # of
binding bounds touched and PRCC magnitude):

1. **CD8 chemoattraction / trafficking.** The probe shows
   `CXCL12_50_Texcl` (-0.15), `k_vas_decay` (-0.17), `k_vas_Csec`
   (-0.15), and `k_Mac_rec` (-0.15) all PRCC-correlate with the
   CD8/Treg ratio failure. CD8 V_T = 3×10⁴ on a ~17 mL tumor → 1800
   CD8/mL, ~3 orders of magnitude below human PDAC IHC. Plausible
   that something throttles CD8 entry into V_T; investigate the
   trafficking module against the failure draws.
2. **APC-TGFb feedback gain.** `f_active_TGFb` PRCC = -0.40 on
   apc_maturation (the strongest single-prior driver in the probe).
   `f_active_TGFb` is at CSV default (lognormal(-2.30, 1.5),
   median 0.10), not in audit. With TGFb saturating its Hills (per
   `qual_bounds_state_summary.md` Part 3), even a modest reduction
   in `f_active_TGFb` (or addition of a TGFb-buffer mechanism)
   could rescue APC maturation. But this pushes against the
   submodel-calibrated TGFb_50_APC = 0.018 nM, so it's a coupled
   constraint.
3. **Macrophage loop gain.** `k_M1_pol` is the biggest single PRCC
   driver for m2_m1_ratio (+0.36). Audit pulled M1pol *down* by
   -0.22σ while M2pol went up +1.69σ — net ~2σ swing of the M2/M1
   balance. The DeSimoni2023 target underpinning this is one IHC
   ratio (CD163/iNOS = 19) at one timepoint; if there's a more
   direct M1pol target available, it could rebalance.
4. **k_PSC_to_apCAF parameterizer fix.** Per
   `k_PSC_to_apCAF_audit_overfit.md`, splitting that target from its
   over-merged 5-param NPE component (or moving to NUTS) should
   restore CV ≈ 21% and pull median from 0.055 back toward 0.059
   (still an upward shift from CSV 0.01 but not 5×). Modest direct
   effect on cascade but addresses the audit-trustworthiness story.

The 5+ order-of-magnitude CD8/Treg ratio gap and the 1e-4 mature/
immature DC ratio together suggest CD8 trafficking/priming is the
load-bearing gap. The hypothesis to test next: are CD8s reaching V_T
at all, or being killed at the gate?

## Next steps

- [ ] **Trafficking probe.** Inspect CD8 trafficking flux in failing
  draws — is k_CD8_recruit being throttled, is the
  classifier-rejection-of-LN-priming happening upstream, or is V_T
  CD8 being killed (`k_Treg_CD8_exh`, `k_CD8_exh_death`) faster than
  it arrives? `probe_data.npz` has the full theta×species matrix.
- [ ] **APC-TGFb sensitivity test.** Hold TGFb_50_APC at CSV
  default (0.07 nM) and re-run a 200-draw probe — does
  apc_maturation_min recover? If yes, the TGFb_50_APC submodel target
  is over-pulling.
- [ ] **CD8 priming in V_LN audit.** Failing-draw fingerprints show
  cDC1 = 6.8e6 (immature, abundant) and mcDC1 = 459 (mature, sparse).
  Check whether the GVAX/checkpoint pathways even have substrate to
  work with at this maturation level — if not, every treatment
  scenario starts with a broken priming chain.
- [ ] **Add `cd8_treg_ratio_min` and `treg_fraction_max` qualitative
  bounds.** Per `qual_bounds_state_summary.md` Part 5 — turn the
  current "trivial passes at extreme values" (0 < x ≤ 15 trivially
  passes upper bound when x is 1e-5) into proper hard failures.
- [ ] **Restriction-classifier retraining on composite-prior
  samples.** With `--no-restriction` the probe accepts 441/1000
  draws. A composite-trained classifier could filter for the subset
  that survives sim and provide better signal-to-noise.

## Files

- 1000-draw probe report:
  `results/prior_diagnostic/composite/report.md`
- Probe data (raw matrices):
  `results/prior_diagnostic/composite/probe_data.npz`
- Audit YAML (composite source): `notes/calibration/submodel_priors.yaml`
- CSV priors (raw): `parameters/pdac_priors.csv`
- M2pol component cache:
  `calibration_targets/submodel_targets/.compare_cache/comp_05669aae261db4eb.json`
- TGFb_50_APC component cache:
  `calibration_targets/submodel_targets/.compare_cache/comp_8870c1059b4c69cf.json`
- k_PSC_to_apCAF analysis: `notes/calibration/k_PSC_to_apCAF_audit_overfit.md`
- Existing qual-bounds posterior-median snapshot:
  `notes/qual_bounds_state_summary.md`
