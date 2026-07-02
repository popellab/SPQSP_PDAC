# Cluster C structural fix — `stromal_fraction` / V_T composition

**Date:** 2026-04-30
**Triggering signal:** `stromal_fraction` p=0.0000, z=+1.59, top driver `Ve_T` (20k τ=1.0 posterior Jacobian, see `cluster_c_diagnostic_2026-04-30.md`)

## What the simulator is doing

The V_T volume rule (assembled in `core/model_compilation/simbio_init.m:140`, replaced in `core/modules/fibroblast_module_enhanced.m:724`):

```
V_T = V_Tmin
    + ( (C_x + C_total) * vol_cell                           — cancer (alive + dead)
      + (CD8_exh + Th_exh + T_total) * vol_Tcell             — T cells
      + M_total * vol_Mcell                                   — macrophages
      + V_T.qPSC*vol_qPSCcell + V_T.iCAF*vol_iCAFcell
      + V_T.myCAF*vol_myCAFcell + V_T.apCAF*vol_apCAFcell    — PSC + CAFs
      ) / Ve_T
    + V_T.collagen / rho_collagen
```

`Ve_T = 0.37` is the **intracellular volume fraction** — cells fill 37% of V_T, the remaining 63% is interstitial fluid + ECM scaffold per cell. Anchored to Wang 2023 DCE-MRI PDAC tumor-mass measurement (`Ve_DCE = 0.61 ± 0.11`, `Ve_T = 1 - Ve_DCE = 0.39`).

The diagnosis-state evolution (`scripts/evolve_to_diagnosis.m:178`) terminates when V_T-derived diameter `(6 V_T / π)^(1/3)` hits `initial_tumour_diameter` (posterior median 2.77 cm at 10k → V_T = 11.13 mL). So **V_T at diagnosis is fixed by the diameter target, regardless of cell composition.**

`stromal_fraction` observable (`calibration_targets/baseline_no_treatment/stromal_fraction_PDAC_deriv001.yaml`):

```python
stromal_fraction = 1 - (C1 * vol_cancer) / V_T   # bare cancer volume, no /Ve_T
```

Sim at posterior median: `1 - (4.03e8 × 2.57e-9) / 11.21 = 1 - 0.092 = 0.908`.
Target: `0.668` (cellularity 0.332). Bever et al. PDAC TSP / cellularity by H&E.

## Why the model is over-stromal

At posterior median (10k τ=1.0 baseline), the cell census in V_T:

| species | count | bare vol (mL) | V_T contrib (vol/Ve_T, mL) |
|---|---|---|---|
| C1 (cancer) | 4.03e8 | 1.04 | 2.81 |
| iCAF | 1.88e8 | ~0.38 | 1.03 |
| myCAF | 4.12e8 | ~0.82 | 2.21 |
| apCAF | 4.7e6 | ~0.01 | 0.03 |
| Mac_M1 + Mac_M2 | 3.64e8 | ~0.42 | 1.13 |
| T cells (CD8+Treg+Th+exh) | 2.25e8 | ~0.07 | 0.18 |
| MDSC, DCs (not in V_T rule) | 1.1e8 | ~0.13 | n/a |
| collagen | 38.7 mg | n/a | 0.38 (via /rho_collagen) |
| **total V_T** | | | **~11.2 mL ✓** |

**Cancer:non-cancer cell-count ratio at diagnosis = 1 : 2.8.** PDAC pathology benchmarks suggest 1 : 1.5-2 — model is ~40-50% too stromal at the cell-count level. Volumetrically, cancer is 9.2% of V_T vs 33.2% target.

`f_stroma_max` (max CAF:cancer ratio) prior median = 2.43, posterior median = 1.63, posterior 75% = 2.47. Already pulled down by SBI. Parameter notes (`parameters_PDAC.m:1572`) acknowledge: *"Diagnostic sweep at fs=2.4 still leaves stromal_fraction ~0.92; Bengtsson cellularity (cal target) is expected to pull this lower at SBI."* — that pull-down to f_stroma_max ≈ 1.6 is real but insufficient.

## Why this maps onto `Ve_T` in the Jacobian

At fixed diagnosis V_T = 11.2 mL, raising `Ve_T` lets *more bare cell volume* fit (since each cell's V_T contribution shrinks), increasing total cell count uniformly — including cancer, increasing cellularity. So the Jacobian sees `d log stromal_fraction / d log Ve_T < 0`.

But: even at the limit `Ve_T = 1.0` (no interstitium at all), with the current cancer:stromal ratio of 1:2.8, cellularity caps at ~25% (still below target 33%). **`Ve_T` alone cannot close the gap**; it can only contribute about half of the needed correction.

The remaining gap requires a shift in the cancer:stromal cell count ratio. Candidate levers (in order of leverage):

| lever | direction | rationale |
|---|---|---|
| `f_stroma_max` | down | Current prior median 2.43 too high relative to PDAC cancer:CAF ratio. Re-anchor with submodel target on cancer-vs-CAF count ratio in PDAC IHC data. Already on the prior-tuning radar per the param note. |
| `Ve_T` | up | Wang 2023 anchor (`Ve_DCE = 0.61 ± 0.11`) gives 0.39, but DCE-MRI Ve measures the **EES (extravascular extracellular space)** which excludes ECM-bound water. The model's `Ve_T` in the V_T rule conflates "intracellular packing fraction" with "volume share that is cells + ECM". Re-examining the Wang 2023 quantity vs the model's structural meaning is a sub-task. |
| Macrophage / Treg recruitment rates | down | TAM density (3.6e8 macs at posterior median) and Treg density (2.1e8) are large stromal contributors. `tam_density` was over-flagged at 10k (sim too high); Treg already constrained by `treg_density_baseline`. |

## Fix paths to scope

### Path A: re-anchor `Ve_T` upward (smallest blast radius)

- Re-read Wang 2023's Ve_DCE definition to confirm it's the right model anchor. Plausible outcome: it isn't, and the right anchor is something closer to 0.6–0.85, not 0.39.
- Update `Ve_T_PDAC_deriv001.yaml` submodel target accordingly; rerun `regen_submodel_priors.py --invalidate Ve_T`.
- Limit: even max `Ve_T = 1.0` doesn't reach target (see math above). This path alone won't close the gap.

### Path B: re-anchor `f_stroma_max` downward via submodel target on cancer:CAF count ratio

- Currently anchored to Torphy 2018 + Ogawa 2021 ACTA2/FAP **area** fraction (joint posterior, n=215). Area fraction → CAF:cancer count ratio conversion may be the source of upward bias if cancer cells are smaller-area than CAFs in section.
- Audit the Torphy/Ogawa submodel-target forward model. Plausible bug: the conversion from area to cell-count ratio assumed cancer-cell area and CAF-cell area are equal, when CAFs (spindle-shaped) typically have larger 2D footprints in section.
- If audit finds the bias, refit gives lower f_stroma_max (probably 1.0–1.5 anchor).

### Path C: Sweep at posterior median — confirm direction before structural change

Per `feedback_trace_before_proposing_fix.md`. The trace pattern from earlier this session can be reused: sweep `Ve_T` and `f_stroma_max` jointly at posterior median, plot stromal_fraction surface. Pick the operating point that lands on target, work backward to confirm what literature would support that point.

```python
# Joint sweep skeleton:
for ve in [0.37, 0.6, 0.85, 1.0]:
    for fs in [0.5, 1.0, 1.5, 2.4]:
        run qsp_sim with overrides = posterior medians + {Ve_T: ve, f_stroma_max: fs}
        record stromal_fraction, cellularity, cancer:stromal ratio
```

This tells us which (Ve_T, f_stroma_max) combos hit target 0.668 stromal_fraction, and whether the literature anchor zones are achievable.

## Recommended sequence

1. **Trace first** (Path C): joint Ve_T × f_stroma_max sweep at posterior median, identify the operating regions that land on target. ~30 min.
2. **Audit Wang 2023 Ve_DCE definition** (Path A research) and Torphy/Ogawa area→count conversion (Path B research). ~1-2 hours of literature work + maple-driven re-extraction if either anchor is wrong.
3. **If a re-anchor reaches target in the trace and the literature supports it** — update submodel target YAML(s), regen submodel priors, rerun 20k SBI to verify.
4. **If neither reaches target alone** — combined update (both Ve_T and f_stroma_max anchors revised, with explicit covariance in the joint posterior given they trade off in the V_T rule).

## Open questions

- Does the Bever cellularity target's 0.668 value reflect the same V_T volume the model uses, or does it implicitly normalize by tissue area in a way that's not directly comparable to V_T? (Methodological audit of Bever et al. — Delesse principle assumption already noted in the observable's `mapping_rationale`.)
- The trace's V_T composition arithmetic above doesn't quite balance (~3 mL discrepancy) — there are minor V_T contributors (V_Tmin, dead-cell pool C_x, secondary species I haven't enumerated). Confirm by dumping the assembled V_T rule and stepping through term-by-term at trace state.
- Should MDSC and DCs be added to the V_T rule? Currently absent. ~1 mL of cell volume / Ve_T = ~3 mL of V_T contribution missing. Could explain the discrepancy and is a probably-correct addition independent of the stromal_fraction fix.
