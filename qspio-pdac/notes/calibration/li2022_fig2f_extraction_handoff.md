# Li 2022 Fig 2F paired pre/post extraction — session handoff

**Date of last work:** 2026-04-20
**Feature branch:** `feature/li2022-fig2f-paired-targets` (pdac-build)
**Open PR:** https://github.com/jeliason/qspio-pdac/pull/14

## What's done

### Calibration targets (4 YAMLs)

Four paired pre/post CD8 fold-change CalibrationTargets digitized from Li 2022 (Cancer Cell, NCT02451982) Figure 2F, promoted into scenario folders:

| File | n | Median FC | CI95 |
|---|---|---|---|
| `calibration_targets/gvax_neoadjuvant/cd8_fc_nonLA_gvax_d21_PDAC_deriv001.yaml` | 6 | 6.63 | [1.58, 54.31] |
| `calibration_targets/gvax_neoadjuvant/cd8gzmb_fc_nonLA_gvax_d21_PDAC_deriv001.yaml` | 6 | 3.29 | [1.72, 6.31] |
| `calibration_targets/gvax_nivo_neoadjuvant/cd8_fc_nonLA_gvax_nivo_d21_PDAC_deriv001.yaml` | 10 | 5.98 | [2.70, 8.85] |
| `calibration_targets/gvax_nivo_neoadjuvant/cd8gzmb_fc_nonLA_gvax_nivo_d21_PDAC_deriv001.yaml` | 10 | 1.40 | [0.93, 6.52] |

Each declares per-patient pre/post readings as `direct_parameter` inputs with `figure_excerpt` provenance, plus per-patient fold changes as `derived_arithmetic` inputs (`formula: post_ii / pre_ii`, validator evaluates and asserts match within 1%).

Arm A CD8 included one patient (06, OS>2y) whose pre-treatment value was at the digitization floor (0.0%); floored to 0.1% as a conservative mIHC detection limit, with uncertainty documented in `key_study_limitations`.

WPD CSVs relocated to `notes/data-extraction/li2022_fig2F_paired/` for provenance.

### Superseded target

`calibration_targets/gvax_nivo_neoadjuvant/cd8_fold_increase_gvax_nivo_d21_PDAC_deriv001.yaml` (median 3.03 from Fig 1G arm-level read) moved to `excluded/` — superseded by the new per-patient Fig 2F target (median 5.98).

### Maple upstream PRs (both merged on main)

- **#21** https://github.com/popellab/maple/pull/21 — MCP validator now forwards `model_structure` context (fixed a silent-fail bug); `figure_excerpt` / `table_excerpt` structured provenance added to `EstimateInput`; `value_snippet` optional.
- **#22** https://github.com/popellab/maple/pull/22 — `DERIVED_ARITHMETIC` `InputType` on CalibrationTarget with cross-input formula validator; static visitor for `validate_inputs_used` now recognizes string-literal input name references (kills false-positive "unused inputs" warnings on f-string / list-literal keyed lookups).

Maple is required at `main >= e0dc8d4`.

### Local fix on pdac-build (also on the feature branch)

- `tests/scenarios/test_calibration_targets.py` — added `to-review` to `SKIP_DIRS` so staged YAMLs in that directory don't block the slow test suite.
- `scripts/export_posterior_medians.py` — short-circuits the write when `_meta.source_hash` already matches the on-disk file. Without this, every `./scripts/run_slow_tests.sh` invocation bumped the `_meta.generated_at` timestamp in `parameters/posterior_medians.json`, which mutated the `slow_test_hash` content and caused CI receipt verification to always fail. **This change is currently uncommitted** — the session left before a clean committable receipt could be generated (see below).

## What's blocking the PR

CI needs a clean `.slow_test_receipt.json` with 0 failures on the feature branch. Each attempt so far has been blocked by untracked YAMLs in validation-scanned directories that fail the appropriate schema check. Two known offenders:

1. `calibration_targets/to-review/CXCL9_50_Trec_PDAC_deriv001.yaml` — SubmodelTarget YAML that fails `test_calibration_target_validates`. Fixed this session by skipping `to-review/` in the test harness.
2. `calibration_targets/to-review/k_NK_kill_PDAC_deriv001.yaml` — same, also fixed by the `to-review` skip.
3. **Still blocking:** `calibration_targets/submodel_targets/compare_inference_results.yaml` — looks like analysis output (has `metadata.timestamp`, `method: component_wise`, etc.) from `fix/sbi-workflow-cleanup` work sitting untracked in `submodel_targets/`. The slow test harness tries to validate it as a `SubmodelTarget` and fails.

All three are untracked in git, so CI's tracked-only content_hash and tracked-only test scan do not see them. The local test run pulls them in, records a failure, and the receipt's `tests_failed: 1` kills CI.

## To resume next session

1. On the feature branch, decide how to handle `compare_inference_results.yaml`:
   - **Option A** (recommended): move or delete the file if it's sbi-cleanup scratch. It's in the user's WIP on `fix/sbi-workflow-cleanup`, not in their Fig 2F work.
   - **Option B**: add it to `SKIP_FILES` in `tests/scenarios/test_calibration_targets.py`.
   - **Option C**: enhance the SubmodelTarget test to auto-skip files whose top-level keys don't match the SubmodelTarget schema shape.
2. Re-run `./scripts/run_slow_tests.sh` on the feature branch. Expect: 239 passed (or whatever the count is after option A/B/C), 0 failed.
3. Commit `.slow_test_receipt.json` + `scripts/export_posterior_medians.py` together. The export-script fix is what made CI's hash computation agree with the local hash computation.
4. Push, watch CI (`gh pr checks 14 --watch`), merge when green.

## Clinical-outcomes calibration — scoped out (2026-04-22)

See `clinical_outcomes_calibration_scoping.md` for the full analysis. Short version: Heumann DFS/OS is not the next-needle-mover (4 scalar targets × wide CIs vs 75 existing targets in a 194-D degenerate parameter space), and the surgery + adjuvant-FFX confound means fitting survival would pollute immune-potency posteriors. Paired TME supplement batch below is the higher-leverage path. FOLFIRINOX / surgery modeling is a worthwhile ~3–4 week sprint eventually (unlocks PRODIGE 24, Agarwal 2025, ALLIANCE A021501, most modern PDAC trials), but after the next SBI stage stabilises.

## Remaining paired-data work (after this PR merges)

From `notes/data-extraction/heumann2023_extractable_data.md` scouting and `sources/mmc1.pdf` supplement scan:

- **Fig S3 R / S / T / U / V** — CD4 / Th1 / Th2 / Th17 / Treg paired. 10 targets (× 2 arms).
- **Fig S3 G** — NK paired. 2 targets (skip B cells — not modeled).
- **Fig S4 I** — CD4⁺PD-1⁺ / CD8⁺PD-1⁺ paired (skip CD8⁺EOMES⁺ — EOMES not modeled). 4 targets.
- **Fig S5 B** — M1 TAM / M2 TAM paired. 4 targets.

All follow the exact same template as the four Fig 2F YAMLs on this branch. Each subpanel needs WPD digitization in the same CSV format used in `notes/data-extraction/li2022_fig2F_paired/`. Cross-panel patient identity is not recoverable (confirmed this session — no patient labels on individual points); each subset is its own independent within-arm paired fold-change target.

Optional longer-term work:

- Survival targets (Heumann 2023 DFS/OS, Zheng 2021 Arm A DFS/OS). Flagged in `heumann2023_extractable_data.md`. Needs the `clinical_progression` scenario extended from 365d to ~3 years and a "death threshold" defined on tumor volume.
- Agarwal 2025 per-patient CD8 density (n=13, already digitized in that notes file). Borderline-resectable cohort; mFOLFIRINOX + SBRT confound.

## Notable session feedback captured

- The user prefers **not dropping patients** from small-n paired targets — floor zero digitization values to a conservative detection limit and document the uncertainty. Applied in Arm A CD8 (patient 06, ε = 0.1%).
- The user endorses moving **SubmodelTarget schema features into CalibrationTarget** when they cleanly transfer (figure_excerpt, table_excerpt, derived_arithmetic so far). Symmetric evolution is preferred over divergent schemas.
- `to-review/` is a draft/holding area for in-progress extractions, validated individually via `validate_target` during authoring, not by the scheduled schema suite. That's the mental model behind skipping it in the test harness.
