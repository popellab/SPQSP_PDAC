# Untreated-scenario pairwise tension analysis

**Date:** 2026-04-28
**Source:** Prior-PC joint reachability + pairwise tension diagnostics (PR #92) on cached pools `d5c2e0e0` (baseline_no_treatment) + `96085258` (clinical_progression). 3742 joint-valid sims, 26 untreated targets, target-specific empirical CI95 as the reach criterion.
**Companion notes:** `untreated_outside_flags_2026-04-28.md` (per-target reachability), `collagen_attractor_overshoot.md` (cdc1_density structural root cause).

## Method

For each pair of cal targets (i, j), let `R_i = 1` if a prior draw's simulated value falls inside target i's empirical CI95. The phi coefficient on the boolean reach matrix measures co-reach correlation:

- φ ≈ 0: independent
- φ > 0: co-reachable (tend to pass together)
- **φ < 0: tensioned** — pass-i sims tend to fail j and vice versa; structural conflict signal

The pairwise tension CSV (`figures/sbi_multi/diagnostics/prior_pairwise_tension.csv`) ranks all 325 untreated pairs. Filtering to pairs with both marginal reach rates ≥ 5% (otherwise tension is uninformative) leaves 15 entries with meaningful negative φ.

## Finding 1: stromal_fraction is a tension hub

`baseline_no_treatment/stromal_fraction` is in mutually-exclusive tension with six other baseline TME observables. Co-reach probabilities are far below the independent expectation `P_i × P_j`:

| Tension partner | φ | P(stromal) | P(partner) | P(both) | Independent expectation | Co-reach deficit |
|---|---:|---:|---:|---:|---:|---:|
| vegf_tissue_concentration | −0.40 | 24% | 39% | **1.0%** | 9.2% | **−89%** |
| mdsc_density | −0.31 | 24% | 92% | 18% | 22% | −18% |
| collagen1_protein_level | −0.26 | 24% | 47% | 5.7% | 11% | −49% |
| apcaf_fraction_of_caf | −0.25 | 24% | 34% | 2.9% | 8.0% | −64% |
| mycaf_fraction_of_caf | −0.25 | 24% | 33% | 3.0% | 8.0% | −63% |
| cdc2_density_baseline | −0.18 | 24% | 56% | 9.5% | 13% | −29% |

**Interpretation:** sims that match Erkan stromal fraction (~50–80% stroma by volume) systematically miss VEGF concentration, the per-subtype CAF fractions, collagen mass, MDSC density, and cDC2 density. This is a **forward-model compositional inconsistency**: the V_T volume rule (cancer cells + 3 CAF subtypes + collagen + immune infiltrate) cannot satisfy all stromal-composition observables simultaneously. Not a prior-width problem — a model-shape problem in `fibroblast_module_enhanced.m` and how the volume budget is allocated.

The `stromal_fraction × vegf_tissue_concentration` pair is the single most extreme conflict: sims pass both targets in only 1.0% of draws vs. the 9.2% expected under independence — a 9× deficit, almost full mutual exclusion.

## Finding 2: CD8 baseline ↔ exhaustion-progression paradox

| Pair | φ | P_i | P_j | P_ij |
|---|---:|---:|---:|---:|
| `cd8_density_baseline` × `exhaustion_worsens_untreated_mechanistic` | −0.34 | 13% | 89% | 7.9% |
| `cd8_treg_ratio_golesworthy2022` × `exhaustion_worsens_untreated_mechanistic` | −0.28 | 17% | 89% | 12% |

**Interpretation:** the two CD8-density-related targets are tensioned against the mechanistic prior that exhaustion fraction grows over 90 days (obs ≈ 1.73× baseline at d90). Sims with realistic CD8 baseline density rarely show progressive exhaustion.

Mechanism hypothesis: the model couples low baseline CD8 with low T-cell turnover, so there is no flux of fresh CD8 entering V_T to convert to exhausted — the population is quasi-steady at the baseline ratio. This is consistent with the earlier finding (per-target Spearman ranking) that `exhaustion_worsens` has |ρ| < 0.07 against all 187 priors: the observable is structurally insensitive to any prior. The negative φ identifies which target it is *trading off against*: fitting CD8 density forces flat exhaustion. Forward-model fix needed (likely a missing time-cumulative feedback in the CD8 exhaustion machinery — chronic antigen-driven accumulation, progressive TGFb buildup, etc.).

## Finding 3: VEGF as a secondary hub

| Tension partner | φ |
|---|---:|
| stromal_fraction | −0.40 |
| treg_density_baseline | −0.20 |
| total_til_density | −0.19 |
| cd4_helper_density | −0.11 |

**Interpretation:** baseline VEGF (146 pg/mg) is structurally tensioned with the immune-infiltrate density observables (Treg, total TIL, CD4). Plausible cause: the angiogenesis-vasculature feedback in the model couples high VEGF with reduced functional vasculature (Provenzano 2012-style suppression), which suppresses immune-cell trafficking into V_T. Same parameters that elevate VEGF to model-prior median (~1700 pg/mg, ~10× above obs) shrink the immune compartment.

This cluster is plausibly downstream of Finding 1 — if the volume budget is rebalanced and stroma + VEGF land in their joint-realistic region, the VEGF–vasculature–trafficking pathway may resolve too. Worth re-checking after a stromal-budget fix.

## Two structural axes for the calibration roadmap

| Axis | Targets coupled | Fixable by | Priority |
|---|---|---|---|
| TME volume budget | stromal_fraction × {VEGF, apCAF/myCAF/iCAF, collagen, MDSC, cDC2} | forward-model edit (V_T volume rule, CAF carrying capacity, collagen mass terms) | high — one knob, six downstream conflicts |
| CD8 turnover dynamics | cd8_density × exhaustion_worsens | forward-model edit (add time-cumulative exhaustion driver) | medium — affects clinical-progression scenario, blocks PFS calibration |

The smaller VEGF↔immune-infiltrate cluster is plausibly downstream of axis 1; revisit after stromal-budget fix.

## What this rules out

`cdc1_density_baseline` (lowest joint reach at 3.1%, OUTSIDE flag in prior PC) does **not** appear in the dominant pairwise tension table because its marginal reach is too low (4–5%) to enter the φ ≥ 5%-marginal filter. Its structural root cause is documented separately in `collagen_attractor_overshoot.md` (collagen attractor at φ_collagen=0.58 → H_pore_APC=0.75 caps cDC1 recruitment). It is a *single-target* structural failure, not a *pairwise* tension. Listed here so future readers don't conclude the diagnostics missed it.

Similarly, the per-target Spearman analysis caught `exhaustion_worsens` as insensitive to all priors (|ρ|<0.07) — the pairwise tension here adds the next-layer information: what the observable is failing *with*, namely cd8_density and cd8_treg_ratio. That pinpoints the missing forward-model coupling more sharply than per-target signals could.

## Reproducing this analysis

```bash
cd /Users/joeleliason/Projects/pdac-build-main && \
.venv/bin/python -m workflows.sbi_runner \
    --scenarios baseline_no_treatment,clinical_progression \
    --num-total-simulations 5000
```

Diagnostics land at `figures/sbi_multi/diagnostics/{prior,posterior}_{joint_reachability,pairwise_tension}.csv`. Both diagnostics fire automatically per PR #92 (merged to main 2026-04-28, commit `33b9a09`).

---

## Follow-up: stromal_fraction cal target audit + V_T volume-rule structural diagnosis

Drilled into Finding 1 (the stromal_fraction tension hub) on 2026-04-28. First validated the cal target itself, then traced the model's V_T volume rule. Cal target is clean; the structural conflict lives in the volume rule.

### Cal target audit: `calibration_targets/baseline_no_treatment/stromal_fraction_PDAC_deriv001.yaml`

Checked at the source, distribution, and observable layer.

**Source (Bengtsson 2026, DOI `10.1016/j.sopen.2026.01.002`).** n=142 treatment-naive resected PDAC, CA19-9 IHC + ImageJ color deconvolution. Median 66.1%, range 27.8–99.9%. Primary human PDAC clinical data with indication-exact match. No flags.

**Empirical distribution.** `distribution_code` derives SD from the reported range using the extreme-order-statistic formula `range ≈ 2·d_n·SD` with `d_n = sqrt(2·ln(142)) ≈ 3.15`, then fits Beta by method of moments on (mean=0.661, var=0.013). Resulting CI95 [0.424, 0.863] is reasonable for that median + breadth.

**Observable code.** `stromal_frac = 1 − C1·V_cell / V_T` with `V_cell = 2.572e−9 mL/cell` from a 17 µm sphere. Arithmetic checks: `(π/6) × (17e−4)³ = 2.572e−9` ✓. Mapping is correct by Delesse: CA19-9-negative area ↔ (V_T − cancer-cell-volume)/V_T for isotropic random sections. Observable uses the model's own V_T directly rather than reconstructing volume from species — eliminates a class of consistency bugs.

**Verdict: cal target is correctly formulated.** The conflict lives downstream in the model.

### Where the model lands

From `prior_predictive_coverage.csv`:

| | obs | pp_q05 | pp_median | pp_q95 |
|---|---:|---:|---:|---:|
| stromal_fraction | **0.668** | 0.644 | **0.944** | 0.990 |

Model produces ~94% stromal fraction at baseline vs. observed 67%. The prior 90% CI bottom barely brushes the cal-target obs. The model is at the **stroma-dominated edge of physical reality** — tumors that are nearly entirely not-cancer-cells.

### V_T volume rule (compiled across all modules)

```
V_T = V_Tmin
    + (C_x + C_total) × vol_cell      / Ve_T   # cancer cells
    + (CD8_exh + Th_exh + T_total) × vol_Tcell / Ve_T  # T cells
    + M_total × vol_Mcell             / Ve_T   # macrophages
    + V_T.NK × vol_NK                 / Ve_T   # NK
    + V_T.qPSC × vol_qPSCcell         / Ve_T   # quiescent PSC
    + V_T.iCAF × vol_iCAFcell         / Ve_T   # iCAF
    + V_T.myCAF × vol_myCAFcell       / Ve_T   # myCAF
    + V_T.apCAF × vol_apCAFcell       / Ve_T   # apCAF
    + V_T.collagen / rho_collagen              # collagen — NO /Ve_T
```

with `V_Tmin = 1e−6 mL`, `Ve_T = 1`, `rho_collagen = 0.025 g/mL`.

Composition: `core/model_compilation/simbio_init.m` line 140 sets the base rule; modules append (line 689 fibroblast_module_enhanced.m, line 225 macrophage_module.m, line 148 NK_module.m).

### Three structural problems in the V_T rule

**Problem 1 — additive volume rule has no packing constraint (dominant issue).**

Each cell type and collagen contribute *independently* to V_T. There is no space-competition constraint that says cells and ECM must share a fixed total volume. When `V_T.collagen` accumulates, V_T expands by `collagen/rho_collagen` and cells **do not get displaced**.

In a real desmoplastic tumor, cells and ECM compete: dense fibrosis squeezes cells out, and high cellularity displaces interstitium. The current model decouples these, so:
- adding collagen → V_T grows → cellularity (C1·V_cell / V_T) shrinks
- adding cancer cells → V_T grows → φ_collagen shrinks
- **but neither displaces the other** — they coexist in a non-physical sum

This directly produces the stromal_fraction × VEGF tension. To fit `stromal_fraction = 0.67` you need lower V_T (less collagen) and/or higher cancer-cell mass; but the same lever raises VEGF concentration `(k_vas_Csec × C1)/V_T` because secretion-per-volume grows on both ends. Two targets push the volume rule in opposite directions and the additive structure forbids resolution.

The right physical model is a packing constraint:
```
phi_collagen + Σ(phi_cells_i) ≤ 1
```
where each component's volume fraction in V_T sums to ≤ 1. Equivalently, a `(1 − phi_collagen)` saturation factor on cell-volume contributions, mirroring the `(1 − phi_collagen)` factor that already exists on collagen secretion in `fibroblast_module_enhanced.m`.

**Problem 2 — collagen contribution lacks `/Ve_T` while all cell terms have it.**

Cell terms divide by `Ve_T` (the cell-packing-density factor — represents intercellular fluid each cell drags into its volume); collagen does not. Dimensionally collagen mass / density gives mL, yes, but it doesn't get the same intercellular-fluid expansion that cells get. Currently `Ve_T = 1` makes this a no-op, but that's a coincidence of the value choice, not a designed equivalence.

**Problem 3 — Ve_T = 1 is outside the PDAC literature range.**

`parameters_PDAC.m` line 68 sets `Ve_T = 1` cited as Finley 2012 (PMID 22547351). But `parameters/cross_model_parameters.csv` documents PDAC's literature range as **0.017–0.529**. Other indications use within-range values:

| Indication | Ve_T (model) | Lit value / source |
|---|---:|---|
| TNBC, CRC, UM | 0.37 | Zhang 2010 |
| NSCLC | 0.75 | — |
| HCC | 1 | — |
| **PDAC** | **1** | **0.017–0.529 documented, model uses 1** |

Setting `Ve_T = 1` means cells take up 100% of the volume they occupy — no intercellular fluid per cell. Realistic PDAC Ve_T should be ~0.3–0.5. **However, lowering Ve_T makes the stromal-fraction problem WORSE** — lower Ve_T amplifies cell-volume contribution to V_T, V_T grows, cellularity shrinks, stromal_fraction grows further from obs 0.67. The choice `Ve_T = 1` was probably tuned (intentionally or not) to keep stromal_fraction from being even higher. So this is a documentation / cited-source issue rather than the central bug.

### Fix options

**Option A (minimal — recommended first attempt).** Add a packing-saturation factor on cell-volume contributions so they shrink as collagen approaches packing limits:

```
V_T = V_Tmin
    + Σ cells_i × vol_cell_i × (1 − phi_collagen) / Ve_T
    + V_T.collagen / rho_collagen
```

At φ_collagen=0.58, cell terms get ×0.42 — cancer cell mass at the same C1 contributes less to V_T → cellularity rises → stromal_fraction drops. Should bring it from 0.94 toward 0.7. One-line edit per module.

**Option B (cleaner — fallback if A insufficient).** Reformulate as a fixed-volume constraint per growth event. Cancer cell birth is allowed only if `phi_collagen + Σ phi_cells < 1`; collagen secretion is allowed only if the same constraint holds. Requires more invasive ODE refactor.

**Option C (orthogonal — pair with A or B).** Fix `Ve_T` to a literature PDAC value (e.g., 0.4) when the volume rule is restructured, since the current `Ve_T = 1` choice was an unspoken correction for the additive bug. Once the bug is fixed, Ve_T can return to its physical range.

### Coupled effects on the other tension-hub conflicts

If Option A resolves stromal_fraction, the spillover should also help:
- **stromal × {apCAF, myCAF, iCAF}**: same mechanism — CAF subtype fractions are also computed against V_T. Restoring proper packing should bring all CAF fractions into joint reach with stromal_fraction.
- **stromal × collagen1_protein**: collagen mass is the same species both targets see; the conflict is via the V_T denominator. Should partially resolve.
- **stromal × VEGF**: VEGF concentration is `secretion / V_T`. With cells displaced by collagen properly, V_T grows less when cancer cells grow, so the per-volume secretion term doesn't blow up the same way. Should partially resolve.
- **stromal × MDSC**: MDSC density is `count / V_T`. Same story.
- **CD8_density × exhaustion_worsens**: probably independent of V_T volume rule — that conflict is in the CD8 turnover dynamics. Won't be helped by Option A.

So one volume-rule fix could unblock 5 of the 6 stromal-hub tensions plus the secondary VEGF cluster. The CD8 exhaustion paradox needs a separate forward-model edit.

### Action items

1. Draft Option A patch (apply `× (1 − phi_collagen)` to cell volume contributions across modules) on a feature branch. **Pending sign-off — touches every module's assembly.**
2. After patch lands, re-run `sbi_runner --scenarios baseline_no_treatment,clinical_progression` and check `prior_joint_reachability.csv` + `prior_pairwise_tension.csv` for the stromal-hub pairs. Expected outcomes: stromal_fraction reach rises from 24% to 50%+; φ on (stromal × VEGF, apCAF, myCAF, iCAF, collagen1, mdsc, cdc2) all rise from negative toward zero.
3. If stromal_fraction reach plateaus before 50%, escalate to Option B (volume-budget constraint) or revisit the collagen attractor anchor sprint deferred from `collagen_attractor_overshoot.md`.
