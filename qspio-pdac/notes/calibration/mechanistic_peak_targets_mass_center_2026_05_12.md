# Mechanistic peak-ordering targets: argmax → mass-center

**Date:** 2026-05-12
**PR:** TBD (feature/mechanistic-peak-mass-center)
**Motivation:** `notes/calibration/cd8fc_vegf_trace_2026_05_12.md` +
`notes/calibration/mechanistic_peak_targets_unreachable_2026_05_12.md`.

## Problem

Four mechanistic cal targets encoded the priming-cascade biology
(mAPC peaks before CD8 peaks; tumor min follows CD8 peak) via
`argmax`/`argmin` of species trajectories. SBI-side problems:

1. **Argmax is discontinuous in θ.** Infinitesimal θ-shifts move the
   argmax index between adjacent timepoints, producing step changes in
   the observable. NPE's smooth surrogate can't fit step functions
   cleanly — it widens the implied likelihood far beyond the stated CI.
2. **Sentinel values land where the lognormal has near-zero mass.**
   When the model produces monotone trajectories (which the post-#180
   audit showed it does at posterior medians AND under every probed
   ±3σ push), `argmax` returns window edges → observable is `0` or
   `−14` against a target distribution centered on 3.74 with CI [1, 14].
   NPE treats those as real measurements at log-infinite distance from
   target.
3. **Across 16k prior draws, the observable was near-constant.** The
   `predictive_summary.csv` for these targets had
   `prior_q05_raw = prior_q95_raw ≈ −23` (or 0) — NPE got zero useful
   gradient from the observable itself, while the sentinel pathology
   in (2) actively distorted posteriors of unrelated parameters
   (post-#180 audit had `vol_myCAFcell` and `cDC1_recruit_target_T`
   appearing as top drivers spuriously).

## Fix

Two parts, applied to all four peak-ordering YAMLs:

### Replace argmax/argmin with excursion mass-center

For each species, compute the time-of-mass-center of its excursion
above (or below) its window-start value:

```python
def _excursion_mass_center(values, tw):
    excursion = np.clip(values - values[0], 0.0, None)   # rise above baseline
    denom = trapz(excursion, tw)
    if not np.isfinite(denom) or denom <= 0:
        return float('nan')
    return float(trapz(tw * excursion, tw) / denom)
```

For `tumor_min_follows_cd8_peak`, the C1 version uses
`np.clip(c1[0] - c1, 0.0, None)` (fall below baseline) instead.

Properties:
- **Smooth in θ.** Trapezoidal integrals are continuous; small θ shifts
  produce small mass-center shifts. NPE-friendly.
- **Biologically faithful.** For a true peaked profile, mass-center
  sits near the peak time. For a monotone rise, it sits near the
  centroid (~2/3 of the integration window). Two monotone-rising
  species can still produce a real positive delta if one rises earlier.
- **NaN-gated when no excursion exists.** If C1 never falls below
  baseline, denominator is 0 → NaN → qsp-hpc-tools' NaN filter drops
  the target from this θ's loss instead of feeding a sentinel.

### Split the magnitude claim into a separate target

The old `tumor_min_follows_cd8_peak` target was carrying two claims:

1. *Tumor should shrink* (magnitude)
2. *If it does, timing should follow CD8 peak* (timing)

With argmax, both were forced through one sentinel observable. With
NaN-gating, the timing target stops firing when no shrinkage occurs,
which would silently lose claim (1). Two new mechanistic-prior cal
targets carry the magnitude claim:

- `tumor_fold_d21_gvax_mechanistic` — C1(d21) / C1(0), CI95 [0.6, 1.4]
- `tumor_fold_d21_gvax_nivo_mechanistic` — same observable, CI95 [0.5, 1.4]

Modeled on the existing `tumor_diameter_fold_d90_*_mechanistic`
targets in `mechanistic/*_longterm/`. CI widths chosen so each acts as
a directional sanity rail rather than a tight likelihood driver.

## Behavior on the current posterior (`multi_v2_nucleated_total`)

Single-θ trace at posterior medians, gvax_nivo scenario:

| target | old argmax value | new mass-center value | target CI95 |
|---|---|---|---|
| `mapc_precedes_cd8` | 0.00 (sentinel) | **+0.35 day** (smooth, below CI lower 1) | [1, 14] |
| `tumor_min_follows_cd8` | −13.95 (sentinel) | **NaN** (no shrinkage → gated out) | [1, 14] |
| `tumor_fold_d21` (new) | n/a | **1.80** (tumor grew 80%, above CI upper 1.4) | [0.5, 1.4] |

The composite signal SBI will see now:

- `tumor_fold_d21_gvax_nivo` provides the *primary* push for tumor
  shrinkage (model output 1.80, target CI top 1.4 — strong gradient).
- `tumor_min_follows_cd8_peak_gvax_nivo` is silent (NaN) while no
  shrinkage exists; will reactivate as a timing constraint once
  shrinkage is achieved.
- `mapc_precedes_cd8_peak_gvax_nivo` provides a soft timing gradient
  (model +0.35d, target ≥1d) pushing for stronger mAPC→CD8 temporal
  lead.

No more `vol_myCAFcell` or `cDC1_recruit_target_T` spuriously topping
the audit driver column.

## Scope notes

- The four peak-ordering YAMLs were all updated (live gvax_nivo
  versions + orphaned gvax-only versions) for hygiene. The orphaned
  ones live in `calibration_targets/mechanistic/gvax_neoadjuvant/`
  which is *not* currently in `workflows/run_config.py`'s cal-target
  dir list for `gvax_neoadjuvant_zheng2022` (only the literature dir
  `calibration_targets/gvax_neoadjuvant` is wired in). The gvax_nivo
  scenario already loads its mechanistic dir. **Follow-up:** decide
  whether to also wire the gvax mechanistic dir into `run_config.py`
  or to delete the orphaned YAMLs.
- The new `tumor_fold_d21_*_mechanistic` targets land in the same
  scenarios as their timing partners — gvax_nivo is live; gvax is
  also written but will only be loaded once the run_config update
  above lands.
- Other mechanistic targets (Group B in the audit:
  `tumor_no_regression`, `tumor_diameter_fold_d90/365/730`,
  `exhaustion_worsens_untreated`, `ecm_concentration_stability`,
  `checkpoint_release_fold`) are endpoint fold-changes — already
  smooth in θ, no fix needed. `time_to_progression` is censored
  threshold-crossing — has its own (milder) discontinuity but matches
  the clinical RECIST definition; left alone.
