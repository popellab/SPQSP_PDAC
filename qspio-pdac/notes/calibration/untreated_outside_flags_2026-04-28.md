# Untreated-scenario OUTSIDE flag triage (post-PR #90 5k SBI run `bq7s6dvmu`)

**Date:** 2026-04-28
**Run:** `figures/sbi_multi/` (5000 raw → 4167 baseline-valid sims)
**Scope:** Four prior-predictive 90% CI misses in untreated scenarios (3 baseline + 1 clinical_progression). Goal: identify which are σ-widenable vs structural.

## Method

For each OUTSIDE flag I (1) ranked all 187 prior parameters by Spearman correlation with the observable across the 4167 prior-predictive sims, then (2) ran a counterfactual subset trace: bin sims by signed-quartile of the top 3-4 dominant drivers and check whether the observation falls inside the bottom (or top) quartile's 90% CI. If even the extreme quartile's CI does not bracket the observation, σ-widening the prior cannot close the gap and a forward-model fix is required.

Per `feedback_trace_before_proposing_fix`: this analysis must precede any prior-edit recommendation.

## Findings

### vegf_tissue_concentration — σ-widenable

- obs=145 pg/mg, full prior 90%CI=[213, 13370], median=1654. Model is ~10× high (geomean).
- Top drivers: `k_vas_Csec` (+0.40), `k_vas_deg` (-0.40), `f_stroma_max` (-0.30), `k_col_myCAF_sec` (-0.27).
- Bottom-quartile subset (low Csec, high deg, high stroma): CI=[116, 2938], median=555 — **obs inside**.
- Provenance note: `k_vas_Csec` cited from Volk 2008 (PMID 18516298, melanoma cell line, not PDAC). Defensible σ-widen target.
- Recommended action: σ-widen `k_vas_Csec` (currently σ=0.7) and/or `k_vas_deg` (σ=0.5). Need a follow-up trace at the proposed σ to confirm prior-predictive p5 reaches obs=145.

### apc_maturation_ratio — σ-widenable (marginal)

- obs=0.0142, full prior 90%CI=[0.0234, 0.751], median=0.227. Model has too many mature DCs.
- Top drivers: `k_APC_mat_cDC2` (+0.35), `d_crit_APC` (+0.34), `k_APC_mig` (-0.31), `d_pore_ref` (-0.31).
- Bottom-quartile subset: CI=[0.0122, 0.501], median=0.094 — **obs at the boundary**.
- Sign of `d_crit_APC` is positive: smaller critical pore size → more APC exclusion → fewer cDCs → higher fraction matured? Counter-intuitive; needs trace to confirm. The cDC pore-exclusion signal is shared with cdc1_density (below) — same collagen-attractor cause.
- Recommended action: σ-widen `k_APC_mat_cDC2` and `k_APC_mat_cDC1` (both σ=0.7). Together with structural cdc1 fix below, this target should resolve.

### cdc1_density_baseline — STRUCTURAL FIX REQUIRED

- obs=7.82 cells/mm², full prior 90%CI=[0.0008, 1.65], median=0.089. Model is 5–10000× too low.
- Top drivers: `d_crit_APC` (-0.39), `d_pore_ref` (+0.37), `APC0_cDC1_T` (+0.32), `k_APC_death` (+0.22).
- Top-quartile subset (most-permissive collagen pore + highest homeostatic density): CI=[0.020, 3.52], median=0.272 — **obs=7.82 is still above the top quartile p95**.
- σ-widening any combination of these priors over current centers will not produce sims with cDC1 density >5 cells/mm² in non-trivial frequency. The model's structure caps cDC1 density below the target regardless of prior placement.
- Root cause already documented in `collagen_attractor_overshoot.md`: phi_collagen lands at 0.58 attractor (5× normal), driving H_pore_APC=0.75 (75% recruitment blocked). Structural fixes:
  1. Recenter `phi_cross_50` median (currently 0.03 — defer until literature-anchoring sprint per `feedback_no_prior_tuning`).
  2. Recenter `d_pore_ref` and/or `d_crit_APC` (collagen→pore-size mapping).
  3. Reformulate H_pore_APC to be less aggressive at PDAC collagen levels.
- σ-widening alone is insufficient. Will not be resolved by prior edits; depends on either the deferred `phi_cross_50` literature anchor or an SBI posterior pull.

### exhaustion_worsens_untreated_mechanistic — STRUCTURAL FIX REQUIRED

- obs=1.73 (CD8 exhaustion ratio frac(d90)/frac(d0)), full prior 90%CI=[0.97, 1.19]. Model exhaustion fraction is essentially flat over 90 days.
- All 187 prior parameters have |Spearman ρ| < 0.07 with this observable. The strongest signal is `k_CD8_basal_exh` at +0.064.
- This means **no prior edit on any current parameter will produce a meaningful exhaustion increase over an untreated 90-day window**. The model's untreated CD8 dynamics are not coupled to a parameter that can drive progressive exhaustion.
- Hypothesis: untreated CD8 dynamics are quasi-steady. New CD8 entering V_T (recruited or activated from LN) ≈ apoptotic loss + exhaustion flux. If activation is steady, frac_exh stays constant. To match obs=1.73 (frac doubling over 90d), need a slow accumulating insult (chronic antigen exposure, progressive TGFb buildup, etc.) that the current model lacks.
- Recommended action: defer prior tweaks. Forward-model investigation needed — likely missing time-dependent exhaustion-driving feedback (e.g., progressive M2/Treg accumulation feeding back on CD8). Possible candidates to inspect: whether `k_CD8_basal_exh` accumulates with antigen exposure history, whether TGFb signaling's effect on exhaustion is dose-cumulative.

## Reframing: reachability is the criterion, not prior PC coverage

(Established 2026-04-28 mid-investigation; see `feedback_reachability_not_prior_coverage`.)

OUTSIDE prior-predictive flags are acceptable as long as the observation is **reachable** — i.e., the prior support extends to the obs in some quartile of the dominant drivers — because the posterior can then pull mass there. The pass criterion lives in the *posterior* PC, not the prior PC. The σ-widenable-vs-structural split below should be read as reachable-vs-unreachable.

| Flag | Reachable under current prior? | Verdict |
|------|-------------------------------|---------|
| vegf_tissue_concentration | Yes (Q1 CI brackets obs) | Leave prior; rely on posterior pull. No edit needed. |
| apc_maturation_ratio | Yes (Q1 lower=0.012, obs=0.014) | Leave prior; rely on posterior pull. |
| cdc1_density_baseline | **No** — top-quartile p95=3.5 << obs=7.8 | Structural fix required (collagen attractor / H_pore_APC). |
| exhaustion_worsens | **No** — all \|ρ\| < 0.07 | Structural fix required (missing time-cumulative CD8 feedback). |

The two real failures are the unreachable ones. The first two flags are not blocking — promote/audit will reveal whether the SBI posterior closes them.

For comparison: `cd8_treg_ratio_golesworthy2022` is *INSIDE* the 90% prior CI (covered by prior tail) but its prior-predictive median (0.0042) is ~870× below obs (3.685). Under the reachability criterion this is also fine — wide prior, posterior can pull. The "INSIDE" label is uninformative on its own; what matters is whether the prior's tail reaches the obs and whether the posterior PC closes.
