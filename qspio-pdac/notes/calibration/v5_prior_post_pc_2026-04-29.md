# v5 Prior + Posterior PC Status (2026-04-29)

**Branch:** `feature/cd8-in-tumor-expansion`  •  **Pool:** `multi_scenario_v5_qual_migration_03862e24` (5k baseline-only)
**Restriction classifier:** `tests/fixtures/restriction_classifier/` — HistGBM(deeper+l2), 50k unfiltered training, CV AUC 0.950, threshold τ=0.7
**Logs:** `/tmp/sbi_run_cd8expansion_v5_20260429_215633.log` (prior PC), `/tmp/sbi_run_cd8expansion_v5_ppc2_20260429_221930.log` (PPC)

## Headline

| Metric | Prior PC | Posterior PC |
|---|---|---|
| Sims used | 4873 / 5000 | 441 / 500 |
| Row-level NaN rate | **2.5%** | 11.8% (posterior tails) |
| D² | 87.0 | **113.4** |
| OUTSIDE 90% CI | 7 / 22 | 6 / 22 |
| Median per-sample reachability | — | 14/22 (63.6%) |

**D²_post / D²_prior = 1.303** — diagnostic message "mostly structural." Posterior tightened priors on what the model can fit, but parts it can't fit got worse — the canonical signature of structural model gaps that no amount of NPE conditioning will close.

## OUTSIDE targets

| Target | Prior reach | Prior 90%CI | PPC reach | PPC 90%CI | obs |
|---|---|---|---|---|---|
| `apcaf_fraction_of_caf` | 10.9% | [0.002, 0.051] | 10.9% | [0.003, 0.053] | 0.064 |
| `cd8_density_baseline` | 2.1% | [8e-5, 5.61] | **12.2%** | [0.04, 100] | 139 |
| `cdc1_density_baseline` | not flagged | [0.345, 3.51] | 35.2% | [0.635, 3.81] | 7.82 |
| `m1_m2_ratio` | not flagged | [0.082, 0.325] | reachable | [0.122, 0.398] | 0.518 |
| `stromal_fraction` | 20.5% | [0.79, 0.96] | structural | [0.78, 0.92] (above obs) | 0.668 |
| `th_exhausted_fraction` | not flagged | [0.15, 0.78] | reachable | [0.26, 0.73] | 0.78 |
| `treg_fraction_baseline` | flagged in prior only | [0.05, 1.00] | within | — | 0.052 |

## Wins from this branch

- **CD8 in-tumor expansion mechanism (commit `3eb3c0d`)** is doing real work. PPC `cd8_density_baseline` upper-CI moved from 5.6 (prior) → 100 (posterior). Obs=139 is now ~6× away vs ~25× pre-branch.
- **`k_C1_growth_intrinsic` submodel target (commit `91752f3`)** pulled cancer growth 5.4× upward, and `stromal_fraction` reach went 13% → 22% in prior (Path B Hwang2008 rework no longer needed).
- **Restriction classifier deeper+l2 retrain** dropped row-level NaN from v3's 51.8% → 2.5% in v5. Effective n doubled.
- **`apcaf_fraction_of_caf`** which went OUTSIDE in v3 (suspected k_C1_growth knock-on) is back to reachable in v5 — was a small-sample artifact, not a structural regression.

## New tensions surfaced

The CD8 fix unmasked the next bottleneck: **`cdc1_density_baseline`** is now the single largest posterior misfit driver (influence 35.2). Obs=7.82 cells/mm² but PPC 90%CI tops out at 3.81. The Treg cluster (`treg_fraction_cd4`, `treg_fraction_cd4_hiraoka2006`, `cd8_treg_ratio_golesworthy2022`) is also pulling hard.

LIKELY MISSPECIFIED diagnostics flagged:
- `apcaf_fraction_of_caf` ↔ `th_exhausted_fraction` (cos = -0.31)
- `th_exhausted_fraction` ↔ `m1_m2_ratio` (cos = -0.47), ↔ `stromal_fraction` (cos = -0.31)

Top Jacobian elasticities point at `k_PSC_to_apCAF` as the dominant lever for `cd4_helper_density` (12.7) and `treg_fraction_baseline` (5.6).

## Next levers (ordered)

1. **`cdc1_density_baseline` reachability** — biggest posterior driver. Need to investigate whether it's a prior-scale issue (`APC0_cDC1_T` median = 5e4, but submodel target may be off) or a structural issue (cDC1 production rate too low under realistic IL-12/cross-presentation signals).
2. **Treg cluster tension** — three Treg-related obs all flagged. Could indicate `k_Treg_pro_tumor` / `rho_Treg_max` priors are too tight, or a structural mismatch in the Treg compartment.
3. **`stromal_fraction` 0.668 vs PPC [0.78, 0.92]** — model now overpredicts stroma (post-`k_C1_growth` 5.4× boost). Consider adding back some balancing via Path B (Hwang2008 custom_ode) or revisit if k_C1_growth shift was too aggressive.
4. **CD8 gap** — final ~6× shortfall. Levers in `notes/calibration/cd8_treg_fix_proposal.md` (widen `p1_50`, `n_T_expansion_burst`, `N_div_costim` priors with literature backing).

## Tooling added this session

- `workflows/restrict_threshold_geometry.py` — diagnose how a classifier threshold reshapes the prior (per-param KS, log-median shift, width ratio). Use to verify marginals stay intact at chosen τ.
- `workflows/restrict_prior_train.py` updated — defaults to deeper+l2 HistGBM (max_iter=1000, depth=8, l2=1.0).
- `workflows/sbi_runner.py` PPC handler patched to unpack CppSimulator's `(theta, pyarrow.Table)` return signature (was crashing with `AttributeError: 'tuple' object has no attribute 'shape'`).
- `tests/fixtures/restriction_classifier/` — promoted retrained artifact, README + DRIFT WATCH banner refreshed.
- Filed `qsp-hpc-tools#63` — top-up downloads test stats twice (pre-topup + post-topup full re-download).
