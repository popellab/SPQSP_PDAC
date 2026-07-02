# Is Stage 1 (submodel-target calibration) helping us?

**Date:** 2026-04-26
**Question raised by:** user, while reviewing the structural-Treg
findings under the composite prior.
**Short answer:** mixed. Stage 1 is *individually* useful (81% of
submodel observables are PPC-covered) but the *joint composite*
prior is currently **net-negative for downstream qualitative-bound
coverage**, and the failure mode is not a Stage-2-fixable artifact.

## What the data says

### Per-component (submodel level): mostly working

From the audit (`notes/calibration/parameter_audit_report.md`):

- **82 components** total (78 NUTS, 4 NPE).
- **225/277 observables PPC-covered (81%)** at the submodel level.
- **53/82 components are clean** (0 divergences, full PPC, passing
  SBC). The remaining 29 components have at least one issue.
- **113/184 (61%)** of QSP parameters have any audit signal; the
  rest stay at CSV defaults in the composite prior.

So at the submodel-target level, the calibration **is matching the
in vitro / ex vivo data it's being trained on**. This is real
information.

### Joint composite (all submodels combined): pathological

From the 1000-draw composite-prior probe
(`results/prior_diagnostic/composite/`):

| metric | base CSV prior | composite prior |
|---|---:|---:|
| joint-pass rate (16 snapshot bounds) | **2/872 = 0.2%** | **0/441 = 0%** |
| `cd8_treg_ratio_max_diagnosis` fail | binding | 94% fail |
| `treg_fraction_min` fail | binding | 86% fail |
| `m2_m1_ratio_range` fail | binding | 83% fail |
| `apc_maturation_min_diagnosis` fail | binding | 83% fail |

The composite prior is **strictly worse** on this metric than the
raw CSV prior. Not by a little — base CSV had 2 sims clear all
bounds; composite had zero across more samples.

The cascade is documented in:
- `notes/calibration/structural_treg_vortex.md` (Treg/CD8/M2/APC vortex)
- `notes/calibration/cd8_treg_gap_diagnosis.md` (Treg overproduction)
- `notes/calibration/k_PSC_to_apCAF_audit_overfit.md` (parameterizer artifact)

### Why "individually correct" → "jointly pathological"

Three distinct failure modes are present in the Stage-1 outputs:

**1. Bridge-collapse parameterizer artifacts.** k_PSC_to_apCAF is
the documented case: a 5-param NPE component (apCAF target merged
with bachem/biffi targets via shared k_PSC_to_myCAF) finds a tight
"bridge" that internally consistent but **fails both target
observables** (post 0.27 vs obs 0.41; post 0.19 vs obs 0.14). The
audit's SBC explicitly flags 5 CAF/PSC parameters as biased or
underconfident. **In these cases Stage 1 is pushing a wrong answer.**

**2. Jointly-pathological data-grounded calibrations.** k_M2_pol +1.69σ
and TGFb_50_APC −1.46σ are *both* PPC-covered against their own
targets (DeSimoni2023 IHC and an algebraic IL-12 IC50 derivation,
respectively). Each looks clean. But together — high M2 polarization
*plus* high APC sensitivity to TGFb — they push the macrophage-DC
axis past a tipping point that no single submodel can detect. The
qual-bounds checker says CD163/iNOS = 11.6 (lit ~2), mature/immature
DC ≈ 1e-4 (lit ≥ 1e-3). This is a **model-coupling pathology**
that no individual submodel could have caught.

**3. In-vitro → in-vivo translation drift.** Many submodel targets
are mouse-model or in vitro experiments. The translation_sigma
mechanism widens the likelihood, but doesn't undo the systematic
bias if mouse-model PSC kinetics simply differ from human in vivo.
This shows up as data-grounded shifts that don't match the human
literature qual bounds.

The three modes shade into each other and aren't easy to separate
without a leave-one-out test.

## Comparison: where would Stage 2 NPE training start without Stage 1?

We don't have an apples-to-apples comparison of (Stage-2-from-CSV)
vs (Stage-2-from-composite) on TME observables. The current
deployment uses composite. This is the **central missing experiment**
for answering the user's question definitively.

What we do know:
- Base CSV is also bad (2/872 joint-pass) but better than composite.
- Base CSV has 1.5σ_log priors on many key params (`p1_50`, `f_active_TGFb`,
  `k_IL1_sec`); Stage 2 NPE has 113 free params to identify from a
  few hundred TME observables. Without Stage 1 prior shrinkage,
  Stage 2 has ~1000× more prior volume to search.
- Stage 2 training takes ~10⁴ simulations and is compute-bound. If
  Stage 1 shrinks the search by 10× per parameter and is correct
  for ~70% of params, the net is still likely positive — but the
  pathologies it introduces in the other ~30% can be hard for Stage
  2 to climb out of, especially when they're correlated via the
  copula.

## Recommended path forward (in priority order)

### Tier 1 — diagnostic tests we can run cheaply

1. **Leave-one-out submodel test on qual-bound coverage.** For each
   of the ~15 submodel-target groups (e.g., apCAF activation, M2pol,
   TGFb_EC50_immune, etc.), build a "composite-minus-this-group"
   prior and run a 200-draw probe. Rank groups by Δ(joint-pass).
   Groups with negative Δ are net-helpers; groups with positive Δ
   are net-hurters. Total cost: ~15 × 200 sims = 3000 sims, much
   cheaper than a Stage 2 retrain. (`workflows/prior_predictive_bounds_diagnostic.py`
   already supports `--submodel-priors-yaml` override; needs a
   ablation-driver wrapper.)

2. **Marginal-only composite (drop copula correlations).** Replace
   the gaussian copula in the composite prior with independent
   marginals (just use audit-emitted univariate distributions). If
   joint-pass improves materially, the bridge-collapse correlations
   are part of the disease, not the cure.

3. **Drop submodel targets with PPC failures.** ~7/26 PPC observables
   in the apCAF/CAF component miss; bridge components with documented
   SBC failures (k_PSC_to_apCAF, k_iCAF_prolif, k_myCAF_prolif,
   k_myCAF_to_iCAF, TGFb_50_CAF_act) are the parameterizer-artifact
   tier. Refit the composite without them; rerun probe.

### Tier 2 — structural fixes if Tier 1 confirms

4. **Per-component NUTS override for over-merged NPE components.**
   Per the k_PSC_to_apCAF note, splitting on the apCAF target alone
   should restore data-consistent posteriors for that component.
   78/82 components already use NUTS; the remaining 4 NPE components
   are the ones to scrutinize.

5. **In vitro → in vivo translation audit.** Check whether the
   translation_sigma is sized correctly for each source. The
   per_source_translation_sigma.md should already document this; if
   any sources are getting < 0.5σ_log translation widening for
   in-vitro-to-PDAC-TME, that's likely too tight.

6. **Joint Stage-1 with qual-bound regularization.** Add the literature
   qual-bound thresholds as soft constraints in the Stage-1 inference
   itself, so the joint posterior can't enter regions that violate
   structural literature. This is a substantial change but addresses
   the failure mode directly.

### Tier 3 — the big question

7. **Run Stage 2 from base CSV vs composite, head-to-head.** Train
   two NPE models on the same scenarios with the same simulation
   pool, one starting from CSV and one from composite. Compare:
   - PPC coverage on the 282 TME calibration target observables.
   - Predictive uncertainty on clinical endpoints (the OBED upstream).
   - Joint-pass on qualitative bounds at the posterior median.

   This is the only way to answer "does Stage 1 actually help Stage 2"
   directly. It's a 2× compute cost but should be tractable on HPC.

## Bottom line

Stage 1 is not "useless" — it's matching its data 81% of the time
and shrinking 113/184 priors. But the **default is currently
introducing more downstream pathology than it removes**, and we have
empirical evidence (base CSV beats composite on joint qual-bound
pass) that the user's question has a real answer to find.

The right mental model is probably: **Stage 1 is a useful but
unaudited filter**. The submodel calibrations need a downstream
sanity check — exactly the qual-bound diagnostic — before being
folded into a composite that Stage 2 trains on. Right now we don't
have that loop; we're calibrating Stage 1 against in vitro data,
exporting to YAML, and training Stage 2 on it without re-checking
that the composite is structurally plausible.

The Tier 1 diagnostics are the cheapest way to make this loop real.
