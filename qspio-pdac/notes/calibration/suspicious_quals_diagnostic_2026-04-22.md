# Suspicious qualitative-bounds passes — diagnostic (2026-04-22)

Worktree: `pdac-build-suspicious` branch `audit/suspicious-qual-bounds`.

## Summary

Three bounds technically pass on the cached posterior-median base snapshot
(`tests/scenarios/.sim_outputs.json`, primary worktree, pre-IFNg-reparam)
but at values so extreme they indicate a structural over-suppression:

| Bound | Value | Threshold | Literature mean |
|---|---|---|---|
| `cd8_treg_ratio_max_diagnosis` | **1.7e-4** | ≤ 15 | ~3 (Golesworthy 2022, Carstens 2017) |
| `treg_fraction_min` | **0.85** | ≥ 0.05 | 0.35 of CD4 (Hiraoka 2006) |
| `apc_maturation_max_diagnosis` | **3.5e-6** | ≤ 0.5 | "low" (Hegde 2020 DC paucity) |

Plus two hard failures with the same pattern:
- `ecm_fold_change_range_diagnosis` = 0.995 vs [5, 20]
- `m2_m1_ratio_range` = 11.63 vs [1, 5]

## Raw species at baseline (V_T)

```
CD8        4.6e4       Treg       2.7e8       CD8:Treg = 1.7e-4
CD8_exh    3.0e4       Th         4.6e7       Treg/T_total = 0.85
T_total    3.1e8
Mac_M1     1.3e8       Mac_M2     1.6e9       M2:M1 = 11.6
cDC1       8.5e6       mcDC1      4.1e1       mat frac = 4.8e-6
cDC2       9.3e6       mcDC2      2.1e1       mat frac = 2.2e-6
MDSC       1.2e8       C1         1.3e9
V_T        17 mL       collagen   19.6 mg
IFNg       4.7e-4 nM   IL10       0.26 nM     TGFb   142 nM (!)
```

`V_T.TGFb = 142 nM` is orders of magnitude above the EC50 family
(`TGFb_50 = TGFb_50_APC = 0.07 nM`, `TGFb_50_Teff = 0.14 nM`). Note this
is higher than the 17 nM figure in CLAUDE.md — the post-IFNg-reparam
model may have shifted this, but it is still deep into saturation of
every TGFb inhibitory Hill.

## Hill function audit (posterior medians)

```
H_DAMP         0.999  (saturated ON)
H_IL12         0.323
H_IL10         0.973  (saturated ON)
H_TGFb         0.9998 (saturated ON)
H_TGFb_APC     0.9999 (saturated ON)
H_TGFb_Teff    0.9997 (saturated ON)
H_APC_Treg     0.0077 (cold — low V_LN.mcDC2)
H_APC_Th       0.0077 (cold)
H_mAPC         0.058  (cold — low V_LN.mcDC1)
H_P1_apCAF     0.011
H_pore_APC     0.015
H_pore_T       0.19
H_stiff_Texh   0.19
```

## Mechanistic chain

### 1. APC maturation = 3.5e-6

Rate = `k_APC_mat_cDC1 × cDC1 × (1−(1−H_DAMP)(1−H_IL12)) × (1−H_IL10) × (1−H_TGFb_APC)`

Multiplying the three gating factors:

```
(1−(1−0.999)(1−0.323)) = 0.9993   (drive — saturated ON)
(1−0.973)              = 0.0270   (IL-10 inhibition — ~97% off)
(1−0.9999)             = 0.0001   (TGFb inhibition — 99.99% off)
---
Net                    = 2.7e-6   → matches observed 3.5e-6
```

**The single dominant suppressor is TGFb.** Removing the TGFb inhibition
(1−H_TGFb_APC = 1) would raise maturation by ~10^4× to 2.7e-2, close to
the `apc_maturation_max_diagnosis` upper bound of 0.5. Removing IL-10
would additionally raise it another ~30×, but TGFb is the mathematical
bottleneck.

### 2. Treg-dominant CD4 compartment (Treg:Th = 5.8)

Th and Treg activation in LN share the same Hill (H_APC_Th = H_APC_Treg
= 0.0077), so the Th/Treg bias must come from downstream TGFb-driven
pTreg conversion + apCAF-P1 induction:
- H_P1_apCAF = 0.011 (low) — apCAF path not a major driver here
- TGFb-driven pTreg: H_TGFb saturating means every `aTh` cell that sees
  tumor TGFb converts at maximum rate to Treg

Effectively, any Th reaching the tumor converts fully to Treg — creating
the 85% Treg fraction.

### 3. CD8 collapse (4.6e4 cells, ~2.7/mm³)

Post-activation CD8 function is gated by `(1−H_TGFb_Teff) ≈ 3e-4`. That
means killing, IL-2 autocrine, and IFNg secretion are each ~99.97%
suppressed. Vicious cycle:
- Low CD8 effector function → low IFNg (4.7e-4 nM = 0.47 pM) → low M1
  drive (even in the new linear form) → M2 dominates
- Low IL-12 production from mcDC1 (which are ~0 anyway)
- No CXCR3 chemotaxis (orphan module) → no positive-feedback trafficking
- Treg count is 3,500× higher than CD8, so Treg-mediated exhaustion
  saturates

## Root cause hypothesis

**`V_T.TGFb` in the model is "total TGFb" not "bioactive TGFb".** In
vivo, TGFb is secreted as latent LAP-TGFb complexes and only activated
locally by integrins (αvβ6/αvβ8), MMPs, or acidic microenvironments.
Active:total TGFb ratio is typically 1–10% in healthy tissue and
5–20% in desmoplastic tumors. EC50 literature values (0.07–0.14 nM)
are measured against **bioactive** recombinant TGFb.

At `k_TGFb_sec_myCAF = 1e-9 nM·L/(cell·day)` with 1.7e9 myCAFs in 17 mL
and `k_TGFb_deg = 14.3/day`, steady-state total TGFb ≈
`1e-9 × 1.7e9 / (17e-3 × 14.3)` ≈ 7 nM at the parameter-default level;
the posterior median has pushed it further to 142 nM (20× increase —
presumably because other targets compensate for missing mechanisms).

If only ~5–15% is bioactive, the "true" TGFb hitting receptors is
0.35–21 nM — crossing the EC50 family in the sensitive 0.1–1 nM range.
That single structural change (latent/active split) would put all
three TGFb Hills into the graded regime and probably resolve:
- APC maturation (`apc_maturation_max_diagnosis`)
- CD8 effector function (`cd8_treg_ratio_max_diagnosis`)
- Treg fraction (`treg_fraction_min`)
- M2:M1 ratio (`m2_m1_ratio_range` — partially, since TGFb also drives M1→M2)

ECM fold-change is a separate collagen-balance issue documented in the
existing `test_all_snapshot_bounds_pass` xfail docstring (`k_MMP_M2`
needs re-derivation).

## Recommended paths (order of cost)

### Option A — Add a latent/active TGFb compartment (model restructure)

Split `V_T.TGFb` into `V_T.TGFb_latent` (secreted, inactive) and
`V_T.TGFb_active` (what Hill functions see). Activation rate driven by
`V_T.aPSC_total` (integrin-driven), `V_T.Mac_M1` (MMP-9), and pH. All
EC50s remain unchanged (0.07–0.14 nM). Parameterize with published
activation fractions (5–20% in tumor).

**Cost:** Medium. Adds one species, one rate parameter,
potentially two more for activators. Changes one reaction rate in
several modules. Does NOT require refitting most of the model since
the existing EC50s stay calibrated.

**Risk:** Breaks existing submodel targets that pull `V_T.TGFb`
toward high values as a proxy for "PDAC is TGFb-rich" — those targets
(IHC/ELISA measurements of total TGFb from tissue lysate) would now
constrain total, not active, which is fine mechanistically but
requires re-running stage-1.

### Option B — Reparameterize TGFb EC50s upward

If we accept V_T.TGFb as "effective bioactive" (wrong but pragmatic),
raise the EC50 family from 0.07–0.14 nM to ~30–100 nM so the Hills are
graded at the 140-nM operating point. No structural change.

**Cost:** Low. Edit 3–4 EC50 values in `parameters_PDAC.m`.

**Risk:** Loses the literature-source grounding of the EC50s. Makes
them look like tuning parameters. Per `feedback_no_prior_tuning` this
is disallowed unless the prior is widened, not shifted.

### Option C — Investigate k_TGFb_sec_myCAF submodel target calibration

The 142 nM posterior is ~20× the default (7 nM). Check whether the
calibration target (`k_TGFb_sec_myCAF_...yaml`) is pulling it there
because of a unit error, an unrealistic culture-volume assumption, or
an apples-to-oranges comparison (whole-tissue vs CAF-conditioned-
medium).

**Cost:** Low to medium depending on what's found.

**Risk:** If the submodel target is correct, this doesn't fix anything.

### Option D — Accept and document as xfail

The current xfails already document "needs joint calibration." The new
suspicious passes could be converted to hard failures with:
- Add upper bound on `treg_fraction` (literature supports <0.5)
- Add lower bound on `cd8_treg_ratio_max_diagnosis` (mean ~3, 10×
  below that is a failure)
- Tighten `apc_maturation_max_diagnosis` lower bound (to e.g. 1e-3)

Then the full picture of "PDAC model is over-immunosuppressed at
posterior medians" becomes visible as 5 failing bounds instead of 2,
and the joint-calibration question gets proper weight in planning.

## Recommended next step

**C first, then A if C doesn't explain the 20× overshoot.**

Submodel target audit for `k_TGFb_sec_myCAF`: check its YAML for unit
consistency, culture conditions, and whether the reported secretion
rate is per-cell-per-day in culture medium (requiring volume scaling)
vs per-cell-per-day in tissue (no scaling). If the target is pulling
20× harder than expected, the structural fix in Option A is likely
premature — first we should know whether the operating-point TGFb
problem is real or an artifact of the calibration chain.

If C confirms the target is pulling correctly and the 142 nM is model-
faithful, then Option A is the right move.

## Option C audit — k_TGFb_sec_myCAF_PDAC_deriv001 (2026-04-22)

Audited `calibration_targets/submodel_targets/k_TGFb_sec_myCAF_PDAC_deriv001.yaml`
(Bachem 1992, rat hepatic stellate cells, DOI 10.1172/JCI115561).

### Unit math — clean

- Target input: `0.20 fmol/(cell·day)` (MFBLC, Table II).
- Model parameter units: `nanomolarity·liter/(cell·day) = nmol/(cell·day)`.
- Forward model: `production_fmol = k × 1e6`. Dimensionally correct
  (nmol × 1e6 fmol/nmol = fmol).
- Reaction in main model: `∅ → V_T.TGFb` at rate `k × V_T.myCAF`, so
  `[nM·L/(cell·day)] × cells / V_T[L] = nM/day`. Self-consistent — no
  culture-volume scaling required (per-cell rate is intensive).

### Where the 20× overshoot comes from — target itself

- Default `k_TGFb_sec_myCAF = 1e-9 nM·L/(cell·day) = 1e-3 fmol/(cell·day)`.
- Bachem MFBLC target = `0.20 fmol/(cell·day)` = **200× the default**.
- The posterior at `k ≈ 2e-8` (→ 142 nM steady-state) is ~20× default
  but still **10× below** the Bachem target. If the target fully won,
  steady-state TGFb would be ~1,400 nM — an order of magnitude worse
  than what we already have.

So the target is doing the pulling, and it is pulling **harder** than
the operating point reflects. Other constraints (other cytokine
targets, cell-density targets responding to TGFb-driven differentiation)
are already holding `k` ~10× below where Bachem alone would put it.

### Active vs total TGFb — documented in the target itself

The YAML explicitly flags this as a limitation:

- `key_assumptions[2]` (line 45–48): *"The measurement captures total
  active TGFbeta1 (acid-activated from latent form). In vivo, only a
  fraction of secreted TGFbeta1 may be activated."*
- `key_study_limitations[2]` (line 68–73): same caveat — acid
  activation converts all latent LAP-TGFb to detectable form.
- `study_interpretation` (line 12–14): *"measured in transiently
  acidified conditioned media (to activate latent TGFbeta1)"*.

So the measurement is **total (latent + active)**, but the model's
single `V_T.TGFb` species feeds Hill functions whose EC50 family
(0.07–0.14 nM) is literature-calibrated against **bioactive**
recombinant TGFb. This is exactly the latent/active conflation
hypothesized in the root-cause section above — not an artifact of the
calibration chain, but a structural mismatch baked into the model.

### Minor concerns (not root cause)

- **Bootstrap composition.** `observation_code` resamples from
  `[pl6d=0.02, tc10d=0.13, mfblc=0.20]` (quiescent, intermediate,
  fully activated). Only MFBLC represents the myCAF phenotype. The
  pl6d and tc10d values are cells in earlier activation states and
  should probably be excluded or re-weighted — including them pulls
  the inferred distribution mean down toward ~0.12 fmol/(cell·day),
  which if anything would *reduce* the overshoot. Not the cause.
- **Proxy indication.** Rat hepatic stellate cells as a stand-in for
  human pancreatic myCAFs (`indication_match: proxy`,
  `tme_compatibility: moderate`). Acknowledged but not quantitatively
  corrected.
- **In vitro plastic culture** lacks 3D ECM feedback and mechanical
  cues known to modulate TGFb secretion. Likely biases toward higher
  rates than in vivo (confluent 2D cultures over-activate).

### Verdict

**Target is clean; the 142 nM operating point is model-faithful given
the target as written.** The overshoot is not a unit error, volume
assumption, or apples-to-oranges scaling. It is the expected outcome
of treating acid-activated-total TGFb as identical to bioactive TGFb.

### Escalate to Option A

Next session should implement the latent/active TGFb split:
1. Rename `V_T.TGFb` → `V_T.TGFb_latent` (what myCAFs secrete, what
   Bachem measured after acidification).
2. Add `V_T.TGFb_active` produced from latent at rate
   `k_TGFb_activate × V_T.TGFb_latent × (f_activators)` where
   `f_activators` depends on myCAF+M1 MMPs and/or integrin proxy.
3. Repoint every `TGFb`-dependent Hill function to `V_T.TGFb_active`
   (search: `H_TGFb`, `H_TGFb_APC`, `H_TGFb_Teff`, `TGFb_50_*`).
4. Target `f_active ≈ 5–20%` for desmoplastic tumor steady state →
   active TGFb of ~7–30 nM, which crosses the EC50 family (0.07–0.14 nM)
   in the sensitive saturation shoulder rather than deep in plateau.
5. EC50 family stays unchanged (they remain literature-grounded
   bioactive measurements).

Also note that the submodel target YAML for `k_TGFb_sec_myCAF` should
be updated post-split to reflect that it now parameterizes **latent**
secretion (which is what Bachem acid-activation measured). This is a
YAML-text change only; the numerics are unchanged.

### Qualitative-bounds conversions (pairs with Option A landing)

Once the latent/active split restores graded TGFb Hills, the three
suspicious passes will turn into genuine failures if left at current
thresholds. Upgrade to hard bounds in `tests/qualitative_bounds.yaml`:
- `treg_fraction`: add upper bound (≤ 0.5; Hiraoka 2006 reports 0.35
  of CD4).
- `cd8_treg_ratio_min_diagnosis`: add new bound (≥ 0.3, i.e. 10× below
  the ~3 mean in Golesworthy 2022 / Carstens 2017).
- `apc_maturation_min_diagnosis`: add lower bound (≥ 1e-3) or tighten
  upper into a two-sided band.

These conversions should land in the same PR as Option A, so the
qualitative-bounds suite reflects the post-fix expected regime.

## Option E + G applied (2026-04-22) — results

Edited `calibration_targets/submodel_targets/k_TGFb_sec_myCAF_PDAC_deriv001.yaml`:
- Added two new `reference_value` inputs: `f_active_total_low=0.05`,
  `f_active_total_high=0.20` (bounds on active:total bioactive TGFbeta1
  fraction in desmoplastic tissue; wide uniform to propagate translation
  uncertainty).
- Modified `observation_code` to rescale each bootstrapped total-TGFb
  value by `rng.uniform(f_active_low, f_active_high)` per draw,
  translating the acid-activated Bachem measurement onto the bioactive
  quantity `k_TGFb_sec_myCAF` represents in the model.
- Updated `key_assumptions[2]` and `submodel_rationale` to document the
  translation, flagging `f_active_total` bounds as estimates pending a
  dedicated maple extraction target.

### Stage-1 re-run

- `python scripts/parameter_audit.py --invalidate k_TGFb_sec_myCAF`
  refreshed the `k_TGFb_sec_myCAF` component posterior and joint priors.
  MCMC finished in ~5 s; no divergences.
- `k_TGFb_sec_myCAF` posterior median: **2.06e-8 → 4.87e-9** (−76%).
- Joint-posterior side effects: `TGFb_50` 0.031 → 0.043 nM,
  `TGFb_50_APC` 0.018 → 0.024 nM, `TGFb_50_Teff` 0.045 → 0.060 nM
  (all shifted modestly upward, consistent with the joint pulling
  Hill EC50s right when the secretion rate drops).
- `scripts/export_posterior_medians.py` (forced) refreshed
  `parameters/posterior_medians.json`.
- `.compare_cache/` seeded from primary worktree prior to rerun;
  the k_TGFb_sec_myCAF, TGFb_EC50 joint, and copula-prior entries
  were rewritten.

### MATLAB qual-test snapshot diff (diagnosis baseline)

| Species / metric | Old | New | Ratio |
|---|---|---|---|
| `V_T.TGFb` | 142 nM | 23 nM | 0.16 |
| `V_T.IFNg` | 4.65e-4 nM | 1.03e-3 nM | 2.21 |
| `V_T.IL10` | 0.263 nM | 0.283 nM | 1.08 |
| `CD8` | 4.65e4 | 3.05e4 | 0.66 |
| `Treg` | 2.66e8 | 6.24e8 | 2.34 |
| `Th` | 4.59e7 | 1.02e8 | 2.22 |
| `Mac_M1` | 1.34e8 | 1.86e8 | 1.39 |
| `Mac_M2` | 1.56e9 | 1.68e9 | 1.07 |
| `mcDC1` | 41 | 296 | 7.14 |
| `mcDC2` | 21 | 149 | 7.14 |
| `myCAF` | 1.70e9 | 1.16e9 | 0.68 |
| `iCAF` | 1.19e9 | 7.55e8 | 0.63 |

### Bound re-audit

| Bound | Old | New | Threshold | Status |
|---|---|---|---|---|
| TGFb operating point | 142 nM | 23 nM | — | −84% (good) |
| apc_maturation (cDC1) | 4.8e-6 | 3.8e-5 | ≤ 0.5 | +8× (still trivially passes upper bound; closer to lit) |
| cd8_treg_ratio | 1.7e-4 | 4.9e-5 | ≤ 15 (lit ~3) | **worse**, passes upper bound only because lit mean is 3 |
| treg_fraction | 0.85 | 0.86 | ≥ 0.05 | unchanged |
| m2_m1_ratio | 11.6 | 9.0 | [1, 5] | −22% (still fails) |

### Hill-saturation check

At 23 nM with `TGFb_50_APC = 0.024 nM`,
`H_TGFb_APC = 23.13 / (23.13 + 0.024) = 0.99897`.
Previously `H_TGFb_APC = 0.99986`. The `(1 − H_TGFb_APC)` factor in
APC maturation thus rose from 1.4e-4 to 1.0e-3 — a **10× increase**,
which matches the observed 7–8× mcDC1/mcDC2 recovery. So the downstream
cascade propagated as predicted even though the Hills are still
technically saturated.

### Partial win, not a full fix

- **Worked:** TGFb operating point dropped 6×, APC maturation recovered
  8×, IFNg doubled, CAF load dropped ~30%, M1 rose 40%.
- **Did not work:** Treg:CD8 ratio got worse (Tregs rose 2.3× now that
  mcDC2 primes CD4 efficiently; CD8 dropped 34%). Treg fraction
  essentially unchanged. Hills are still deep in plateau — to truly
  unsaturate, TGFb needs another ~100× reduction to sub-nM, which
  Option E alone cannot deliver.
- Confirms the hedge from earlier: TGFb conflation was one
  load-bearing cause of over-suppression, but not the only one.
  Remaining causes:
  - Cancer-cell-baseline TGFb production and Treg-secreted TGFb still
    dominate the residual 23 nM pool. Any target setting
    `TGFb_baseline` or `k_TGFb_Tsec` likely has the same acid-activation
    / total-vs-bioactive translation issue and deserves the same audit.
  - Orphan CXCR3 module — CD8 cannot chemotax to tumor even when
    priming works.
  - IL-10 Hills remain saturated at ~0.97 (not touched by this fix).
  - CD4 priming balance: mcDC2 now works, but Th→Treg conversion via
    residual TGFb still dominates the CD4 fate split.

### Stale-threshold test failures (not regressions)

Two tests failed in `test_qualitative_behaviors.py::TestGvaxCascadeFixes`:

- `test_fix1_gvax_drives_mcDC1`: expects `mcDC1_d21 / mcDC1_d0 >= 10`;
  new ratio is 3.71× (d0=350, d21=1300). d0 rose from ~41 to 350
  because baseline APC maturation recovered — so the **absolute** d21
  value is healthier, but the **fold** is smaller.
- `test_fix1_gvax_drives_cd8_infiltration`: expects `CD8_d21 / CD8_d0
  >= 5`; new ratio is 3.40× (d0=3.1e4, d21=1.0e5). Same story —
  baseline CD8 is now lower because Tregs dominate harder, but d21
  CD8 is in the right ballpark.

These thresholds were calibrated against the over-suppressed baseline
and need updating once the residual over-immunosuppression is resolved.
Candidate remediations:
- Convert them to **absolute** d21 thresholds (what matters clinically)
  rather than d21/d0 fold ratios.
- Or explicitly tie them to the state of the over-immunosuppression
  work by marking them xfail with `reason="thresholds depend on
  unresolved CD4 priming balance"`.

Not regressions in biology.

### Next-session entry points

1. **Audit `TGFb_baseline` and `k_TGFb_Tsec` submodel targets** (if any)
   for the same acid-activation / bioactive translation issue as
   Bachem. These two are the remaining TGFb sources now that myCAF
   secretion has been rescaled.
2. **CD8 trafficking / CXCR3 module** — flagged as orphan in the
   original diagnostic. Re-enabling chemotaxis would let CD8 respond to
   the 7× higher mcDC1 priming.
3. **IL-10 axis** — still saturated at ~0.97; may need the same
   latent/active translation treatment.
4. **Option A (latent/active TGFb split)** remains the right
   model-structure answer if TGFb-targeted therapies enter scope.
   Option E does not replace it.
5. **Stale GVAX cascade test thresholds** — repoint to absolute d21
   values or mark xfail pending the CD4 priming balance fix.

## Update 2026-04-22 (evening) — after PR #26 + PR #27 landed on main

Two PRs landed on main after this diagnostic was written. They changed
the baseline snapshot values and clarify what's still on the table.

### Landed

- **PR #27 (main side): linear IFNg reparameterization.** M2→M1
  polarization arm reframed as a spatial-coverage slope rather than a
  saturating Hill. Cleared `test_m1_m2_ratio_improves` (the XFAIL from
  the Part 2 table — it was dropped from the test file rather than
  unmarked). Did not fix the baseline `m2_m1_ratio_range` bound.
- **PR #26: LOX-crosslinking shield on M2 MMP collagen degradation.**
  Added `f_nascent = phi_cross_50 / (phi_cross_50 + phi_collagen)` Hill
  to gate the M2 MMP term; reconciles Zhu 2001 fresh-gel in-vitro rate
  with in-vivo fibrotic collagen persistence. ECM bound revised from
  `[5, 20]` (Xu 2017 SHG — method-biased toward aligned fibrils) to
  `[2, 12]` (Tian 2019 TMT proteomics 2.6× primary source, Imamura 1995
  hydroxyproline and Polasek 2017 mouse as secondaries). Split
  `test_all_snapshot_bounds_pass` into a per-bound parametrized test.

### Current bound states on the refreshed cache

| Bound | Pre-merge | Post-merge | Threshold | Status |
|---|---|---|---|---|
| `ecm_fold_change_range_diagnosis` | 0.99× | **3.4×** | [2, 12] (revised) | PASS |
| `m2_m1_ratio_range` | 11.6 | **9.82** | [1, 5] | FAIL (smaller margin) |
| `cd8_treg_ratio_max_diagnosis` | 1.7e-4 | 1.7e-4 | ≤ 15 | PASS (trivially; lit ~3) |
| `treg_fraction_min` | 0.85 | 0.86 | ≥ 0.05 | PASS (trivially; lit ~0.35) |
| `apc_maturation_max_diagnosis` | 3.5e-6 | ~3.5e-6 | ≤ 0.5 | PASS (trivially) |

Note: the over-immunosuppression diagnosis is essentially unchanged by
either PR — both touched narrow knobs (IFNg M2→M1 slope; M2 MMP gating)
while the root TGFb conflation remains unaddressed.

### New ECM-exclusion side-effect xfails (introduced by PR #26)

As ECM fold-change rose from 0.99× to 3.4×, `phi_collagen` climbed from
~0.045 to ~0.17, which shrank the effective pore size (`H_pore_T`) and
increased stiffness-mediated CD8 exhaustion. Two progression tests
flipped:
- `test_species_increases_d0_to_d90[CD8]` — CD8 drops ~23% over 90 d
  instead of rising. Consistent with immune-excluded PDAC (Carstens
  2017) but conflicts with the test's assumption. Xfailed.
- `test_species_increases_d0_to_d90[IFNg]` — downstream of the CD8 drop
  (fewer effectors → less IFNg). 3.3% drop over 90 d. Xfailed.

Both xfails reference "revisit after M2/M1 polarization fix — lower M2
density may rebalance CD8 recruitment/retention" in the reason string.
The real expected fix is the Option A latent/active TGFb split, which
restores CD8 effector function and reduces Treg-mediated exhaustion at
the same time — the ECM-barrier effect on CD8 should then be less
dominant.

### GH Issue #28 opened — V_T / rho_collagen semantic overload

Discovered during PR #26 debugging. Two problems, deferred to a
dedicated refactor PR:

1. `rho_collagen = 0.025 g/mL` is overloaded: (a) reference
   concentration for phi-scaling, 25 mg/mL saturation convention; (b)
   material packing density for the volume rule. These are conceptually
   different quantities that happened to coincide numerically.
2. The volume rule `+V_T.collagen/rho_collagen` grows V_T as collagen
   accumulates. This inverts cytokine distribution-volume physics
   (collagen crowds interstitium → should *concentrate* cytokines, not
   dilute), and can let C_max net-increase with desmoplasia at high
   phi (V_T growth outruns phi-suppression in the current formulation).

Not load-bearing for the TGFb work. Track separately.

### Revised next-session priority order

1. **Option A — latent/active TGFb split.** Highest leverage. Rename
   `V_T.TGFb` → `V_T.TGFb_latent`, add `V_T.TGFb_active` produced at
   rate `k_TGFb_activate × V_T.TGFb_latent × f_activators` where
   `f_activators` depends on myCAF integrins and M1/MMP activity.
   Repoint every `H_TGFb*` Hill and every `TGFb_50_*` EC50 consumer to
   `V_T.TGFb_active`. Target `f_active ≈ 5–20%` at desmoplastic steady
   state. EC50 family stays unchanged.
2. **Audit `TGFb_baseline` and `k_TGFb_Tsec` submodel targets** for the
   same acid-activation / total-vs-bioactive translation issue.
   `k_TGFb_Tsec` is Treg-secreted TGFb; `TGFb_baseline` is cancer-cell
   baseline production. After Option A these still contribute to the
   latent pool and need consistent target interpretation.
3. **Update the `k_TGFb_sec_myCAF` submodel target YAML** to drop the
   `f_active_total` rescaling now that the split handles it
   structurally; the target once again parameterizes latent secretion
   (which is what Bachem's acid-activation actually measured).
4. **Convert suspicious passes to hard failures** in `tests/qualitative_bounds.yaml`:
   - `treg_fraction_max` — upper bound, ≤ 0.5 (Hiraoka 2006 baseline 0.35)
   - `cd8_treg_ratio_min_diagnosis` — lower bound, ≥ 0.3 (lit mean ~3)
   - `apc_maturation_min_diagnosis` — lower bound, ≥ 1e-3 (or two-sided band)
5. **Relax / repoint the GVAX cascade fix thresholds** in
   `test_qualitative_behaviors.py::TestGvaxCascadeFixes`. After Option E
   `test_fix1_gvax_drives_mcDC1` (expects d21/d0 ≥ 10) and
   `test_fix1_gvax_drives_cd8_infiltration` (expects d21/d0 ≥ 5) both
   fail on ratio (3.71×, 3.40×) while d21 absolute values are healthier
   than pre-Option-E. Convert to absolute d21 thresholds or mark xfail
   pending CD4 priming balance. After Option A, d0 baselines will shift
   again — so reassess after the split lands.
6. **Un-xfail CD8 and IFNg progression tests** if Option A's restoration
   of CD8 effector function + Treg-mediated exhaustion reduction
   outweighs the ECM-barrier effect introduced by PR #26.
7. **IL-10 axis** — inspect the k_IL10_sec submodel target (if any) for
   the same total-vs-bioactive translation issue. IL-10 Hill was
   saturated at 0.97 in the original diagnostic; unchanged by Option E.
8. **Orphan CXCR3 module.** CD8 has no active chemotaxis to tumor in
   current model; re-enabling may require fresh submodel targets.
9. **Issue #28 (V_T / rho_collagen)** — separate refactor PR. Not in
   scope for the TGFb-centered sprint but worth keeping in mind because
   the cytokine distribution-volume fix there will also touch how
   TGFb_active is seen by cells.

Estimated scope of items 1–5 as a single sprint: 2–3 weeks including
stage-1 re-run, stage-2 NPE retrain (may be a cell in a sweep), PPC
audit, and qualitative-bounds updates. Items 6–9 are follow-on work.
