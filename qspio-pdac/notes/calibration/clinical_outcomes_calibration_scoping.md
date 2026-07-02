# Clinical-outcomes calibration: scoping + surgery/chemo confound

**Date:** 2026-04-22
**Status:** Scoping — no code changes. Outcome: defer clinical-outcomes calibration, proceed with Heumann paired TME (supplement) targets next.

## Triggering question

Use Heumann 2023 DFS/OS as calibration targets for the long-horizon GVAX(+nivo) scenarios?

## Current state of long-horizon infrastructure

Already present:
- Three 1095-day scenarios with `evolve_to_diagnosis`: `clinical_progression`, `gvax_neoadjuvant_longterm`, `gvax_nivo_neoadjuvant_longterm`.
- Death threshold `V_T.C1 >= 1e11 cells` wired into OS observables.
- 12 prediction-only observables per scenario under `calibration_targets/predictions/<scenario>/`: OS@12/24/36mo, DFS@12/24mo, RECIST CR/PR/SD/PD@day90, time-to-progression, tumor diameter@day365/730. Already consumed by PPC / OBED (`_section_predictive_uncertainty`).
- One fit clinical target: `clinical_progression/tumor_doubling_time_PDAC_deriv001.yaml` (Ahn 2017, VDT).

Gap for clinical-outcome calibration: attach empirical `median + CI95` to the prediction observables (i.e. convert prediction-only YAMLs into CalibrationTarget YAMLs in `calibration_targets/<scenario>/`). The two `*_longterm` scenario directories don't exist yet under `calibration_targets/`.

## Surgery + adjuvant-chemo confound (investigated)

From `scenarios/gvax_nivo_neoadjuvant_longterm.yaml:10-21`, explicit header:

> **Important: surgery (resection at day 21) and adjuvant chemotherapy (FOLFIRINOX) are NOT modeled.** Long-horizon predictions therefore capture the pharmacodynamic effect of GVAX+nivo alone over three years — not the clinical OS/DFS of the Zheng 2021 treatment-plus-surgery-plus-adjuvant cohort.

Every patient in Heumann 2023 (all three arms) actually received:
1. Neoadjuvant priming dose ~day -14
2. **Surgery ~day 0**
3. Second priming post-op
4. **SOC adjuvant chemo (FOLFIRINOX)**
5. 4 more GVAX boosters q28d
6. Extended treatment / follow-up

Components missing from the model vs what Heumann measured:

| Component | In model? | Effect on outcome |
|---|---|---|
| Resection of primary tumor | No | Removes the mass driving V_T.C1 → 1e11 |
| Adjuvant FOLFIRINOX | No | Large OS benefit (PRODIGE 24 mOS 54.4 vs 35.0 mo) |
| Cyclophosphamide (all arms) | No | Immunomodulatory; Zheng 2021 flagged it detrimental |
| GVAX booster doses (5 post-op) | No | Current scenario: single day-7 dose |
| Nivo continuation | Partial | Single dose, no continuation |

Fitting Heumann mOS ≈ 27 mo to a scenario that simulates a single-dose, unresected, untreated-otherwise trajectory would force QSP parameters to attribute surgery + adjuvant-chemo survival benefit to immunotherapy. Posteriors on immune-potency parameters would be biased upward by a large factor.

## Calibration-value assessment (Heumann DFS/OS specifically)

Heumann usable signal (drop urelumab arm — not modeled):
- 2 arms × 2 summary stats (mDFS, mOS) = **4 scalar targets**
- Plus maybe 1–2 HR comparisons between arms
- n = 14–16 per arm; 95% CIs span ~13 mo to NR (factor-of-3+ uncertainty)

Current calibration load: 44 submodel targets + ~30 CalibrationTargets ≈ **75 targets** over 194 QSP parameters. PRCC + NPE capacity sweep already identified the bottleneck as **parameter degeneracy** (R² flat across 15× NN capacity), not too-few-targets.

The survival observable gates on `V_T.C1 ≥ 1e11`, so at best DFS/OS nudges: `k_C_growth`, total `k_C_Tcell` kill, maybe IL-2 / priming params via GVAX magnitude. All already hit by `tumor_doubling_time` (Ahn 2017, n=100, tight CI) and Li 2022 Fig 2F fold-change targets (already merged PR #14). Mostly redundant information.

Alternatives compared:

| Option | Targets added | Lift | Likely QSP posterior impact |
|---|---|---|---|
| Heumann DFS/OS (with surgery proxy + lumped chemo) | 4 | Medium (~2 wk) | Low — mostly absorbed by chemo nuisance param |
| Heumann paired TME (Fig S3 R/S/T/U/V/G, S4 I, S5 B) | ~20 | Low (same template as Fig 2F) | Medium — directly constrains CD4/Th/Treg/NK/M1/M2 |
| Keep DFS/OS as prediction-only | 0 | None | Free — remain as PPC sanity checks |

Decision: **next PR is the Heumann paired TME supplement batch** (already planned in `li2022_fig2f_extraction_handoff.md`). Survival targets deferred until the surgery + chemo backbone is modeled.

## Why survival calibration is structurally hard (even after infra is built)

To disentangle the immunotherapy increment from surgery + adjuvant chemo you need either:
- Both a treated-arm and an untreated-arm OS in the same cohort, or
- A chemo-only arm in the same cohort.

Heumann has neither (all three arms are Cy-GVAX + something; no chemo-only control). Zheng 2021's historical controls (adjuvant-chemo-only, mOS 24.8 mo) are the closest reference but cross-study comparison is statistically weak. Fitting survival scalars without a clean control distributes information across the QSP immune biology *and* the new chemo/surgery nuisances — nuisances absorb most of it because they sit closer to the outcome.

## Surgery + chemo modeling difficulty (investigated)

| Component | Effort | Notes |
|---|---|---|
| Surgery (resection event) | 1–2 days | SimBiology event at day_of_surgery: `V_T.C1 *= (1 - resection_fraction)` + proportional TME cell reduction; Beta prior on `resection_fraction` (mass ~0.95 R0, tail to 0.85 R1). Worth doing standalone regardless of calibration plans — makes PPC trajectories realistic. |
| Lumped adjuvant FFX | 3–5 days | Single `k_adjuvant_kill` first-order on `V_T.C1` active between day_surgery+14 and day_surgery+180; optional weak kill on proliferating immune cells for crude myelosuppression. Conditional on resection firing. |
| Two-drug FFX lump | 2–3 weeks | Separate 5-FU vs platinum/irinotecan kill terms. Only worth it if calibrating against dose-intensity variants. |
| Full 4-drug FFX PK/PD | 6–8 weeks | 5-FU + leucovorin + irinotecan/SN-38 + oxaliplatin, each with PK, kill, tox. Needed only for toxicity/pharmacogenomic prediction. Not required for survival calibration. |
| GVAX booster dosing | ~1 hour | Config change: `GVAX_schedule: [7, 28, 6]`. |
| Cyclophosphamide (low-dose, Treg-depleting) | 1–2 days | Pulse Treg-kill event at day -1; magnitude from Ghiringhelli 2007 / Laheurte 2010. |

**Recommended clinical-backbone sprint (~3–4 weeks):**
1. Surgery event
2. Lumped adjuvant FFX (`k_adjuvant_kill`)
3. Neoadjuvant FFX + lumped SBRT proxy (single fractional-kill event ± DAMP pulse)
4. Gem-nabP lumped (reuses existing `nabpaclitaxel_module.m`)

Post-sprint every modern PDAC trial becomes in-bounds.

## Is FOLFIRINOX worth doing eventually? Yes.

FFX / mFFX is SOC for fit PDAC patients. Everything newer layers immunotherapy or targeted agents on top of a FFX backbone. Without a FFX module, the entire modern PDAC clinical-trial dataset is untranslatable for downstream validation or OBED.

Trials that become usable once lumped FFX exists:

**Adjuvant:**
- **PRODIGE 24 / CCTG PA.6 (Conroy 2018, NEJM)** — adjuvant mFFX vs gem, n=493, mOS 54.4 vs 35.0 mo. Single cleanest anchor for `k_adjuvant_kill` prior.

**Metastatic:**
- **PRODIGE 4 / ACCORD 11 (Conroy 2011, NEJM)** — FFX vs gem, mOS 11.1 vs 6.8 mo.
- **NAPOLI-1 / NAPOLI-3** — FFX / nal-IRI combinations.

**Neoadjuvant / borderline resectable:**
- **ALLIANCE A021501** — mFFX vs mFFX+SBRT, BR-PDAC, n=126.
- **SWOG S1505** — neoadjuvant mFFX vs gem-nabP, resectable, n=147.
- **NEOLAP (Kunzmann 2021)** — mFFX vs gem-nabP-FFX, LAPC.
- **Murphy 2018** — neoadjuvant mFFX + SBRT.
- **Agarwal 2025** (already in `heumann2023_extractable_data.md` as deferred) — mFFX + SBRT + GVAX/nivo. Fully usable once FFX+SBRT exist.

**Immunotherapy combinations:** most recent PDAC immunotherapy trial designs run on a FFX backbone. Heumann is an outlier for *not* giving FFX during the neoadjuvant window.

Tier-1 lumped FFX is the sweet spot — decades of published PK/PD, `k_adjuvant_kill` tightly constrained by PRODIGE 24 HR 0.58. Higher-resolution tiers aren't identifiable from trial data anyway (drugs always co-administered).

## Strategic ordering (from highest-leverage per unit work)

1. **Now:** Heumann paired TME supplement batch (Fig S3/S4/S5 — 20 targets, template ready).
2. **Next:** finish stage-2 SBI / submodel target extraction for unidentified params (from `contraction_local_shell < 10%` cohort in parameter audit).
3. **Clinical-backbone sprint (3–4 weeks):** surgery event → lumped adjuvant FFX → lumped neoadjuvant FFX + SBRT → gem-nabP lumped.
4. **Survival calibration:** Heumann DFS/OS, Zheng 2021 Arm A, Agarwal 2025, with PRODIGE 24 as chemo-only anchor. Treat primarily as **validation / PPC** rather than tight fit targets, because nuisance absorption limits QSP-parameter identifiability even after the sprint.

## Key takeaway

Clinical-outcomes calibration via Heumann DFS/OS is **not the next-needle-mover** for QSP parameter identification. The valuable Heumann extractions for SBI are the paired TME readouts. FOLFIRINOX modeling is worth doing eventually, but its primary payoff is unlocking the entire PDAC clinical-trial dataset for validation and OBED — not improving the fit of the existing calibration.
