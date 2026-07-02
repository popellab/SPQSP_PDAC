# Prior-predictive bounds diagnostic — session handoff

_Date: 2026-04-26._

## What was built

`workflows/prior_predictive_bounds_diagnostic.py` — a complement to
`scripts/parameter_audit.py`. The audit answers *"after calibration, are
observables covered?"*; this diagnostic answers *"before calibration, can
the prior + model even reach the literature-feasible region?"*

Pipeline:
1. Sample N thetas from the live prior (base CSV or Stage-1 composed),
   filtered through the restriction classifier.
2. Run baseline + clinical_progression scenarios via the C++ binary.
3. Evaluate every snapshot-checkable qualitative bound on the diagnosis
   snapshot per theta -> binary pass/fail matrix.
4. PRCC of every prior parameter against each binding bound's pass vector.
5. Bucket each bound: prior-range / mechanism / weak-signal.
6. Markdown report + cached probe matrices (`probe_data.npz`) for fast
   re-runs after prior tweaks.

Outputs land under `results/prior_diagnostic/<prior>/`.

## Headline finding (base prior, N=1000)

`results/prior_diagnostic/base_v0/report.md`

- **Joint pass rate: 2/872 (0.2%)** at the live base prior.
- 14 of 16 snapshot bounds are binding (>0% fail rate).
- Most bounds are bucketed `mechanism` (combinatorial drivers) — only
  `apcaf_fraction_max` and `treg_fraction_max` flagged `prior-range`
  (single dominant prior).
- Tightest co-failure cluster: `cd8_treg_ratio + treg_fraction +
  icaf_fraction + m2_m1_ratio` — Treg-dominated, CD8-sparse,
  fibrotic-skewed TME state. ~470–550 sims fail any pair simultaneously.

### Drivers worth chasing (NOT in Stage-1 yet)

The fresh 113-param `notes/calibration/submodel_priors.yaml` covers most
diagnostic-flagged drivers (`k_apCAF_prolif`, `k_Treg_death`, `n_IL1`,
`k_M2_pol`, `k_Mac_death`, plus the CAF interconversion ladder). Drivers
the audit hasn't constrained yet:
- `q_nCD8_LN_in`, `q_nCD8_P_out` (CD8 trafficking — top drivers of
  `cd8_treg_ratio_max`)
- `k_CD8_basal_exh`, `k_CD8_exh_death`, `k_cell_clear` (exhaustion
  dynamics)
- `k_apCAF_death` (apCAF turnover)

These are candidates for new submodel-target extractions.

## What does NOT work yet (the composite-prior path)

The diagnostic supports `--prior composite --submodel-priors-yaml
notes/calibration/submodel_priors.yaml`, which loads the audit's
113-parameter Stage-1 composed prior via `qsp_inference.priors.
load_composite_prior_log`. Three blockers found and partially fixed:

1. **Sample correlation matrix not PSD**: MCMC-derived correlations had
   tiny negative eigenvalues that broke `np.linalg.cholesky`.
   - **Fixed** in `qsp-inference` branch `fix/composite-prior-non-psd`,
     commit `e3f13cf` (eigenvalue clipping + diagonal renormalization).

2. **Non-lognormal marginals with non-positive support**: the audit's
   parameterizer fits Normal/Gamma/InvGamma marginals whose support
   crosses zero, breaking `_log_transform_marginal`'s `np.log(rvs)`.
   - **Fixed** in `qsp-inference` commit `9675029` (truncate samples to
     positive domain before log-fitting).

3. **Degenerate marginals (gamma scale=0)**: `k_IL2_cons` had `scale: 0.0`
   from a posterior pinned at zero, which produced 0 positive samples.
   - **Fixed** in `qsp-inference` commit `a2d46d2` (fall back to CSV
     lognormal for that single param, warn rather than abort).

4. **NEW BLOCKER (not fixed)**: even with the prior loading correctly,
   100% of composite-prior thetas crash CVODE at t=0 with corrector
   convergence failure. Hit rate is not classifier-related — same
   100% failure with `--no-restriction`.

   Likely causes (need investigation):
   - Spurious correlations in the empirical posterior correlation matrix
     pushing samples to ODE-unstable corners.
   - Heavy tails of gamma/invgamma marginals reaching extreme values that
     the simulator hasn't been hardened against.
   - A specific parameter median shift from base that crosses an
     integrability boundary (the audit's mu/sigma are very different
     from base CSV mu/sigma for some parameters — worth diffing
     median-to-median).

   Diagnostic next step: sample 10 thetas from composite prior, run them
   one-by-one through `qsp_sim`, identify which species gradient blows
   up, attribute to a parameter or interaction.

## What was tried but is wrong

Manual prior tweaks in `parameters/pdac_priors.csv` (k_apCAF_prolif sigma
1.0->0.5, k_Treg_death sigma 1.2->0.7, n_IL1 sigma 0.7->0.4) were
**reverted** because all three parameters are already constrained by
Stage-1 (commit reverted in `44eb766`). Under composite prior the CSV
sigmas are overridden by the yaml lognormal copula marginals. The tweaks
had zero effect on what NPE actually sees.

A hand-rolled `scripts/refresh_submodel_priors.py` was deleted because
the canonical refresh path is `scripts/parameter_audit.py` -> writes
`notes/calibration/submodel_priors.yaml` automatically.

## Path mismatch worth fixing

The audit writes `notes/calibration/submodel_priors.yaml` (113 params,
fresh).
Live consumers (`workflows/local_npe_repro.py`, `workflows/run_ppc_clinical.py`,
the prior-predictive diagnostic before the `--submodel-priors-yaml` flag)
default to `calibration_targets/submodel_targets/submodel_priors.yaml`
(23 params, stale 2026-03-18). They should either symlink the audit's
output to that path or be repointed at `notes/calibration/`.

## To resume

1. Investigate the CVODE-at-t=0 crash on composite-prior thetas. Likely
   start: sample 10 thetas from composite, find one that crashes, find
   what's extreme about it relative to base-prior medians.
2. Once composite probe runs, expect joint pass rate to climb
   substantially — Stage-1 already constrains the headline drivers.
3. Repoint or symlink the consumer paths so the live path
   (`calibration_targets/submodel_targets/submodel_priors.yaml`) actually
   reflects the audit's output.

## Branch state

- `pdac-build` (worktree at `pdac-build-prior-diag`): branch
  `diag/prior-predictive-bounds`. Commits: `9e6a97a` (initial
  diagnostic), `a533044` (manual prior tweaks — reverted in `44eb766`),
  `f1ff882` (composite-prior plumbing restored).
- `qsp-inference`: branch `fix/composite-prior-non-psd`. Commits:
  `e3f13cf` (PSD projection), `9675029` (log-domain truncation),
  `a2d46d2` (degenerate-marginal fallback).
