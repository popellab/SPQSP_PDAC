# SBI calibration audit — session findings (2026-04-13)

Joint multi-scenario NPE on the PDAC QSP model, 4 scenarios, 38 observables.
All results against the existing sim pool (priors CSV + model version v4); no new
MATLAB simulations were generated beyond a 2000→5000 bump.

## Stage-2 section added to parameter audit report

`scripts/parameter_audit.py` now takes `--sbi-run-path <dir>` and emits a
"Stage 2: NPE posterior shifts" section built from the pinned run's
`posterior_samples.csv` + `diagnostics/`.

Four comparisons are reported per parameter:

- **vs pre**: against the pre-stage-1 prior (`pdac_priors.csv`). Total calibration effect.
- **vs S1**: against the stage-1 posterior (`submodel_priors.yaml`) for params with a submodel target. Marginal stage-2 effect.
- **Contr global**: mean contraction over `theta_test ~ prior` (from `local_calibration.csv` or `z_score_contraction.csv`). Identifiability under the prior.
- **Contr local-shell**: mean contraction over `theta_test_local` — test obs within a 20% copula-space neighborhood of `x_obs`. Identifiability near the data.

All implemented in `qsp-inference/src/qsp_inference/audit/report.py`
(`load_sbi_run`, `load_submodel_priors`, `_section_stage2_npe`).

### Interpreting the four numbers

- **Local at x_obs only trustworthy if shell-local agrees.** If local >> shell-local, the tight posterior at x_obs is a single-point flow artifact rather than genuine identification.
- **Global + shell-local agreeing** means the NPE behaves consistently near the observation and the local number is believable.
- **Artifact examples** (tier 1, 2467 sims): `k_TGFb_sec_myCAF` local 79%, shell 4%; `k_col_myCAF_sec` local 75%, shell 7%; `d_pore_ref` local 58%, shell 8%. Stage-1 posteriors already constrain these; whole-model data likely sees only one ridge of a larger degeneracy.
- **Genuine identification**: `k_M2_pol` local 54% / shell 51%; `f_stroma_max` local 54% / shell 38%.

## Key library fixes made en route

**`qsp-hpc-tools/qsp_hpc/utils/model_structure_units.py`** (new).
`load_units_from_model_structure()` merges species + compartments + parameters
from `model_structure.json`. Replaces three ad-hoc constructions in
`derive_test_stats_worker.py` and `qsp_simulator.py` that used only `data["species"]`.

- Fixed baseline NaN on observables referencing compartment volume (V_T) or
  scalar parameters (`rho_collagen`) — those were silently defaulting to
  dimensionless and failing `.to("milliliter")`.
- Impact: baseline NaN rate on affected columns dropped from 100% → 45%
  (normal sim-failure level).

**`calibration_targets/gvax_*_neoadjuvant/*pct_nonLA_*.yaml`** (12 files).
Li 2022 (Cancer Cell, Fig 2E) defines mIHC percentages as fraction of CD45+
cells. Extracted YAMLs were summing all cells (cancer + CAFs + immune),
yielding a 10–100× mismatch. Patched denominators to immune-only
(T/TAM/MDSC/DC).

- `scripts/fix_pct_nonLA_denominator.py` implements the patch.
- `scripts/invalidate_test_stats.py` deletes HPC-cached derived test stats
  when library/YAML changes require re-derivation without re-running sims.

## Sim budget: 2000 → 5000 raw

| | n_train effective | best val loss | R² median | shell ≥50% |
|---|---|---|---|---|
| 2000 raw (976 train) | 976 | 31.6 | 0.086 | 4 |
| 5000 raw (2467 train) | 2467 | **26.5** | **0.117** | 5 |

- Val loss improved 5 units; flow was clearly undertrained at 976.
- Learning curve: best epoch 35/55 → slight overfitting, near but not at the data ceiling.
- Identified-core contractions moved meaningfully: `k_M2_pol` 54%→62%,
  `k_cell_clear` 36%→46%, `k_Treg_death` became "partially identified" for the
  first time.
- Non-identified T-cell/IL-2 cluster unchanged — information limit is real
  there (`k_IL2_sec`, `N_IL2_CD8`, `q_nCD8_LN_in`, `q_nCD8_P_out`,
  `CXCL12_50_Texcl`, `k_T_activation_per_APC`, `n_T_expansion_burst`).

## Tier × flow sweep (all at 2467 train sims, seed 2025)

| config | θ | NN params | best val | shell ≥50% | shell <10% | shell <0 |
|---|---|---|---|---|---|---|
| tier 1 (32×3) | 29 | 70k | 26.6 | 3 | 9 | 2 |
| tier 2 (32×3) | 69 | 141k | 66.0 | 5 | 30 | 5 |
| tier 3 (32×3) | 116 | 237k | 113.2 | 1 | 80 | 24 |
| tier 3 (64×5) | 116 | 754k | 104.8 | 3 | 75 | 23 |
| all (64×5) | 217 | 1.4M | 219.7 | 1 | 181 | 60 |
| all (128×7) | 217 | 3.8M | 184.2 | 5 | 134 | 47 |
| **all (192×9)** | 217 | 7.6M | **122.3** | **11** | **51** | **4** |
| all (256×11) | 217 | 13.0M | 108.7 | 9 | 83 | 10 |

### Takeaways

1. **Tier-3 collapse at small flow was capacity, not information.**
   Going from (32×3) to (64×5) at tier 3 recovered `k_cell_clear` from 45% →
   64% shell-contraction (tier-2 had 67%). The flow had been too small to
   represent a 116-dim conditional density, not too little data.

2. **Flow capacity can trade for θ dim up to a point.** Bumping 128×7 → 192×9
   on all 217 params unlocked substantial gains on the identified core
   (e.g., `k_Th_pro_tumor` 48%→64%, `k_stroma_support` 31%→55%).

3. **Diminishing returns hit at 256×11.** Identified-core stable within seed
   noise; shell<10% and shell<0 both got worse. Flow is now big enough; the
   bottleneck is training data.

4. **Non-informative params correctly marginalize when capacity is adequate.**
   At 192×9 all-217, shell<10% count dropped to 51 (from 181 at 64×5 and 134
   at 128×7) — the flow stops drowning weak signals in noise and lets
   unidentified params land near 0 contraction as they should.

5. **R² stayed near zero** at all 217-param configs. That's expected: R²
   averages posterior recovery over `theta_test` drawn from prior, including
   x's far from x_obs. Shell-local contraction is what matters for inference
   at the actual observation.

### Seed stability: 128×7 all-217

| | seed 2025 | seed 2029 | Δ |
|---|---|---|---|
| best val loss | 184.2 | 198.6 | +14 |
| shell ≥50% | 5 | 3 | −2 |
| shell <10% | 134 | 151 | +17 |
| Mean \|Δ\| across 217 params | — | — | 6.2 pp |

Identified core moved 4–10 pp between seeds at 128×7. Not stable enough to
trust individual contraction values. Not yet repeated at 192×9 but the flow
is 2× bigger with same data, so similar or worse seed-wobble is plausible.
**Seed stability at 192×9 is an outstanding verification.**

## Non-identified clusters (consistent across configs)

Params that stayed <10% shell-contraction in every config (with sufficient
capacity):

- T-cell trafficking: `q_CD8_T_in`, `q_nCD8_LN_in`, `q_nCD8_P_out`
- IL-2 axis: `k_IL2_sec`, `N_IL2_CD8`, `k_IL2_cons`
- T activation / expansion: `k_T_activation_per_APC`, `n_T_expansion_burst`,
  `k_CD8_pro`, `k_CD8_death`
- Cytokine secretion/degradation: `k_IL10_sec`, `k_CXCL12_sec_iCAF`,
  `k_CXCL12_sec_cancer`, `CXCL12_50_Texcl`, `IL10_50`, `IL1_50`, `k_IL1_sec`,
  `k_CXCL12_deg`
- APC turnover: `k_APC_mat_cDC1`, `k_APC_mat_cDC2`, `k_APC_death`, `k_mAPC_death`
- CAF interconversion: `k_iCAF_prolif`, `k_apCAF_prolif`, `k_myCAF_to_iCAF`,
  `k_iCAF_to_myCAF`
- Vascular: `k_vas_growth`, `k_vas_decay`

These are information-limited, not capacity-limited. No amount of flow tuning
or additional priors-from-pool sims will identify them. Path forward: new
calibration targets.

## Recommended next data extractions

1. **Paired pre→post mIHC from Fig 1F (Arm A, n=6) / Fig 1G (Arm B, n=10).**
   Ratios constrain dynamics directly. CD8+ *** in Arm B → expansion/trafficking
   rates; M1/M2 ratios → polarization kinetics. Expected to move the T-cell
   and macrophage clusters meaningfully.

2. **KM survival at landmark times** (6, 12, 18, 24 mo) for Arm A and Arm B
   from Heumann 2023. 8 scalar targets; captures shape information that median
   OS throws away. Not as powerful as the paired mIHC for parameter ID but a
   useful out-of-sample mechanistic check, especially for `k_C1_growth` and
   kill/growth balance.

Deprioritized: median OS/DFS alone (single scalar each; survival is
information-poor for mechanistic params at n=14–16).

## Recommended operational defaults

`workflows/sbi_workflow.py`:

- `PRCC_TIER = None` (all 217 params) — no reason to limit once flow is adequate
- `NPE_HIDDEN_FEATURES = 192`, `NPE_NUM_TRANSFORMS = 9` — sweet spot at 2467 train sims
- `NUM_TOTAL_SIMULATIONS = 5000` — current budget produces 2467 effective train
  sims, which 192×9 can actually use

Outstanding questions (not answered this session):

1. Seed stability at 192×9 — repeat with SEED=2029, check identified-core wobble
2. Training-data ceiling at 192×9 — subsample sweep [500, 1000, 1500, 2000,
   2467] to see if still climbing (cheap) or plateaued (would justify
   doubling sim budget)
3. SNPE-C multi-scenario — existing `workflows/sequential_sbi_workflow.py` is
   single-scenario only; adapting to multi-scenario + tier-configurable setup
   is ~1 hour of work and would let us test if focused sampling tightens the
   artifact cluster shell contraction

## Artifacts preserved for cross-run comparison

- `figures/sbi_multi_tier1` — current tier 1 (32×3), sim 2467
- `figures/sbi_multi_tier2` — current tier 2 (32×3), sim 2467
- `figures/sbi_multi_tier3` — tier 3 with big flow (64×5)
- `figures/sbi_multi_tier3_smallnn` — tier 3 with small flow (32×3)
- `figures/sbi_multi_64x5` — all 217 with 64×5
- `figures/sbi_multi_128x7_s2025` — all 217 with 128×7, seed 2025
- `figures/sbi_multi_128x7_s2029` — all 217 with 128×7, seed 2029 (stability check)
- `figures/sbi_multi_192x9_s2025` — all 217 with 192×9 (recommended default)
- `figures/sbi_multi_256x11_s2025` — all 217 with 256×11 (diminishing returns)