# Restriction classifier (test fixture)

> **DRIFT WATCH.** This fixture is a *trained ML model* — it ages as the prior structure evolves. Run `python tests/fixtures/restriction_classifier/probe_drift.py` periodically (or before any larger prior overhaul) and regenerate if **recall under the live prior drops below 0.95** or the top-15 features change. Last regenerated: 2026-05-05 against pool `multi_scenario_v5_qual_migration_3a0063dd` (20k baseline draws post-PD-L1 rebaseline + route 1 gate; 193 features; 8294 valid / 41.5% raw survival; CV AUC 0.903; threshold geometry at τ=0.5: 42.3% accepted, 87.3% survival, max KS 0.133 — marginals intact). See "When to regenerate" below.

Cached `HistGradientBoostingClassifier` used by `tests/scenarios/test_qualitative_coverage.py` to filter prior draws to thetas likely to evolve to diagnosis successfully. Without this filter, ~55% of prior draws fail evolution and per-bound coverage estimates get noisy.

Loaded by `_load_restriction_classifier()` in `tests/scenarios/_cpp_helper.py` via `qsp_inference.inference.restriction.RestrictionClassifier.load()`.

## Provenance

- Trained on 20,000 baseline draws under the composite prior (8,294 valid; baseline-only joint mask, 41.5% raw survival post-PD-L1 rebaseline + route 1 gate).
- Model: `sklearn.ensemble.HistGradientBoostingClassifier(max_iter=1000, max_depth=8, learning_rate=0.03, l2_regularization=1.0)`, log input transform, default threshold 0.5.
- 5-fold CV AUC = 0.903; in-sample AUC = 0.964.
- Feature order (193 params) and threshold curve are recorded in `metadata.json`.
- Threshold geometry (per `workflows/restrict_threshold_geometry.py`): at τ=0.5 → 42.3% accepted, 87.3% sim survival, max per-param KS = 0.133 (marginals essentially intact). At τ=0.7 → 24.8% accepted, 96.5% survival, max KS = 0.212 (mild squeezing on `k_stroma_support`, `k_C1_growth`, `k_vas_growth`). τ=0.5 selected as best yield/draw (0.369) with marginals under the 0.15 KS target.
- Top-driver shape unchanged from prior regen: `k_stroma_support`, `f_stroma_max`, `k_C1_growth`, `k_vas_growth`, `initial_tumour_diameter` dominate (integrator-blow-up directions).

## Drift behavior

Feature order matches the live `parameters/pdac_priors.csv` at the time of training (190 params including `k_CD8_T_pro`, `rho_CD8_max`). Future param additions/removals will be handled by `_project_to_classifier_features()` in `tests/scenarios/_cpp_helper.py`: dropped at projection if absent from current prior, filled with old prior median if retired.

## When to regenerate

- Recall under the live prior drops below ~0.95 (rerun `tests/fixtures/restriction_classifier/probe_drift.py` to check).
- The dominant top-15 features by permutation importance change (e.g., a major param-set overhaul, not a single addition).
- `pickle.loads(classifier.pkl)` errors out under the current `scikit-learn` pin.

## How to regenerate

The canonical training entrypoint is `workflows/restrict_prior_train.py` (which calls `qsp_inference.inference.restriction.train_restriction_classifier` under the hood). Recipe:

1. Launch an unfiltered SBI sweep against the current priors:
   ```
   python -m workflows.sbi_runner --scenarios baseline_no_treatment \\
       --no-restriction-classifier --num-total-simulations 50000 ...
   ```
   You can kill the runner once the HPC sweep + derivation completes — NPE training is not needed.
2. Update the `POOLS` dict in `workflows/restrict_prior_train.py` to point at the new pool's `test_stats/<hash>/test_stats.parquet`.
3. Run `python -m workflows.restrict_prior_train` (writes to `cache/restriction_classifier/`).
4. Optional: inspect threshold geometry with `python -m workflows.restrict_threshold_geometry --pool <pool_dir>` to confirm marginals are preserved at the threshold you'll use (target: max KS ≤ 0.15 at τ=0.7).
5. Promote: `cp cache/restriction_classifier/* tests/fixtures/restriction_classifier/` and update the DRIFT WATCH banner above with the regen date + pool hash.

The pickle format ties to the installed `scikit-learn` version. If the load fails, regenerate against the current pin.
