# Mechanistic peak-ordering targets aren't reachable — model is monotonic in [7, 21]

**Date:** 2026-05-12
**Audit run:** `figures/multi_v2_nucleated_total/misspec_audit/`
**Trigger:** PR #181 reach-sweep flagged these two targets as ❌ with sentinel-looking outputs (`-23`, `0`).

## What the targets encode

Both `_mechanistic` cal targets compute a time-difference between dynamic
features over the post-treatment window `t ∈ [7, 21]` (post-GVAX±nivo):

- `mapc_peak_precedes_cd8_peak`: `t_peak(V_T.CD8 + V_T.CD8_exh) − t_peak(V_T.mcDC1 + V_T.mcDC2)`
  — expected positive (mAPC peaks first under vaccination), target median 3.74d, CI95 [1, 14].
- `tumor_min_follows_cd8_peak`: `t_min(V_T.C1) − t_peak(V_T.CD8 + V_T.CD8_exh)`
  — expected positive (tumor reaches minimum after CD8 peaks), same target distribution.

Both versions exist for `gvax_neoadjuvant` and `gvax_nivo_neoadjuvant`.

## What the model produces

Single-θ trace at the SBI posterior medians from
`multi_v2_nucleated_total`, evaluated over `[7, 21] d`:

| scenario | species | trajectory in [7, 21]d |
|---|---|---|
| gvax_neo | V_T.CD8 + V_T.CD8_exh | monotone ↑ 3.95e8 → 8.99e8 |
| gvax_neo | V_T.C1 | monotone ↑ 8.88e8 → 1.31e9 |
| gvax_neo | V_T.mcDC1 + V_T.mcDC2 | monotone ↑ 8.10e4 → 1.24e5 |
| gvax_nivo | V_T.CD8 + V_T.CD8_exh | monotone ↑ 3.95e8 → 8.32e8 |
| gvax_nivo | V_T.C1 | monotone ↑ 8.88e8 → 1.30e9 |
| gvax_nivo | V_T.mcDC1 + V_T.mcDC2 | monotone ↑ 8.10e4 → 1.21e5 |

All three species rise monotonically across the window. No CD8 peak, no
tumor minimum, no mAPC peak. So `argmax` falls at the right edge for
CD8 and mAPC, `argmin` falls at the left edge for C1, and the observed
deltas are:

- `mapc_peak_precedes_cd8_peak = 21 − 21 = 0`
- `tumor_min_follows_cd8_peak = 7 − 21 = −14` (the sweep saw `-23` over the
  longer window [7, 30] when the simulator's step-cap let it run past 21d)

These are *correct* evaluations of an observable whose expected dynamic
shape simply doesn't exist in the model's 21-day trajectory.

## Why the model is monotonic here

Three independent mechanisms would have to land in the window for the
ordering target to be measurable:

1. **CD8 contraction** — requires `k_cell_clear / k_CD8_exh_death` fast
   enough to overcome ongoing GVAX-driven priming + nivolumab-relieved
   suppression. At the posterior medians, expansion dominates clearance
   throughout 21 d. Consistent with the cd8_fc ceiling at ~2.4–3.6× in
   `notes/calibration/cd8fc_vegf_trace_2026_05_12.md`.
2. **Tumor shrinkage** — requires effective CD8-killing throughput
   (`k_C_Tcell_eff × CD8 × C1 / (C1 + K_C_T_kill)`). With C1 still rising,
   killing is in the saturated regime and net growth wins.
3. **mAPC contraction** — requires `k_mAPC_death` × mAPC ≳ GVAX-driven
   maturation rate by day 21. Posterior medians give maturation
   throughput well above clearance.

None of these are "tweakable to fit the target" without breaking other
calibration targets. The model is operating in a regime where the
GVAX+nivo intervention is too weak to produce shrinkage and too
sustained to allow contraction.

## Recommended actions

Pick one of:

1. **Retire the four `_mechanistic` peak targets.** They were
   mechanistic-prior assertions (per memory's
   `feedback_mechanistic_prior_cal_targets.md`) inferred from broader
   vaccine + checkpoint literature, not from a within-patient time
   series of the Zheng 2022 cohort. Given the model can't reach them
   from any θ at posterior medians AND the experimental setup (paired
   biopsy + surgical resection, ~2 weeks apart) doesn't actually
   measure these time-deltas in patients, the simplest move is to drop
   them and let the static mIHC observables carry the GVAX+nivo signal.
2. **Re-source if available.** Search for a paper that does provide
   within-patient kinetics of CD8/mAPC/tumor over a vaccination
   schedule — likely some preclinical work on B16-GVAX or 4T1 that
   tracks tumor volume + IHC. The target would then be a primary
   citation rather than a mechanistic prior.
3. **Accept and structurally extend** if the temporal ordering is
   diagnostically important: add a CD8 contraction term active only
   when antigen drops below a threshold (already partially in the
   model via the `k_cell_clear × K_C2_recruit/(C_total² + K_C2_recruit)`
   funnel — could increase `K_C2_recruit`); and / or speed up
   `k_GVAX_clearance` so the mAPC burst contracts before day 21. This
   is real model work and would need its own SBI cycle to verify it
   doesn't break m1tam/m2tam d21 ratios.

Option 1 is the lowest-risk path; option 3 is the only one that
actually delivers the dynamic biology the targets encode.
