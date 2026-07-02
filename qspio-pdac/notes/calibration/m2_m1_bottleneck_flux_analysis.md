# M1/M2 Bottleneck + Hill Saturation — Diagnosis-State Flux Analysis (2026-05-05)

Investigation of the residual structural tensions surfaced by the post-PD-L1-rebaseline 20k SBI run (`pool 3a0063dd`, τ=0.5).

## Run context

- Pool: `multi_scenario_v5_qual_migration_3a0063dd_baseline_no_treatment` (20k draws, post-PD-L1-rebaseline + route-1 gate, restriction classifier off, τ_cohort=0.5).
- All 24 marginals pass 90% PPC coverage; D²_post = 1968 / D²_prior = 20 (ratio 98×, "mostly structural").
- Top 5 posterior misfit drivers (τ=0.5): cd8_density_baseline (1746), cd8_density_baseline_golesworthy2022 (1709), cd8_density_baseline_liu2015 (27), cd8_exhausted_fraction_shen2017 (18), mycaf_fraction_of_caf (17).

## Initial hypothesis (refuted)

The cd8 deficit was first read as a **CD8 vs Treg pull-of-war**: posterior pulls toward few CD8 to satisfy treg_fraction targets. Cross-sim correlations on 390 PPC sims rule this out:

- `r(cd8_density, treg_density) = +0.001` — independent, no anti-correlation.
- `r(cd8_density, k_Treg_CD8_exh) = +0.18` — *positive*, so Treg-driven exhaustion isn't the bottleneck.
- Treg/CD8 ratio at posterior median = 0.36 (obs ≈ 0.27) — Treg compartment is roughly correct.

## Real diagnosis: dual lockout on CD8 priming via Hill saturation

CD8 density is dragged 8–10× below the cohort triple by a self-consistent low-cytokine fixed point:

```
r(cd8_density, V_T.IFNg) = +0.935   (cd8 axis is IFNg-driven feedback)
r(cd8_density, V_T.cDC1)  = +0.18
r(cd8_density, V_T.mcDC1) = +0.18
r(m1m2_ratio, k_M2_pol)   = -0.528   (strongest single-param driver of M1/M2)
r(m1m2_ratio, k_M1_pol)   = +0.380
```

Posterior medians:
- M1/M2 = 0.28 (obs 0.52) — model too M2-skewed.
- k_M2_pol = 0.480 vs k_M1_pol = 0.043 (ratio 11×).
- CD8 density ~10× below obs; CD8/CD8_exh ratio fine; CD4 helper -0.5σ; VEGF -1.2σ.

These collapse to **one mechanistic root** when the Hill-function saturation map is read off the same 390 PPC sims:

| Hill | median H | frac sims at H<0.05 | frac sims at H>0.95 | implication |
|---|---|---|---|---|
| ~~**H_IL12** (cDC1 mat / CD8 prime)~~ | ~~**0.000**~~ | ~~**100%**~~ | ~~0%~~ | RETRACTED — unit-confusion bug in analysis. Corrected H_median = 0.19 (functional, mid-band) |
| ~~**H_IL10** (M1 suppression / APC inhibit)~~ | ~~0.001~~ | ~~100%~~ | ~~0%~~ | RETRACTED — corrected H_median = 0.58 (functional, mid-band) |
| **H_TGFb_CAF_act** | 0.953 | 0% | 53% | CAF activation pinned at saturation |
| **H_IL1** (iCAF formation) | 0.952 | 0% | 52% | iCAF formation pinned at saturation |
| **H_TGFb_APC** (APC mat inhibit) | 0.933 | 0% | 43% | **APC maturation suppressed at saturation by TGFb** |
| **H_CCL2** (Mac/MDSC rec) | 0.922 | 0% | 33% | macrophage recruitment maxed |
| H_IL6_M2 / _MDSC / _iCAF | 0.88–0.89 | ≤0.3% | 20–25% | IL-6 axes maxed |
| H_TGFb_Teff (CD8 exh) | 0.851 | 0% | 20% | borderline saturating |
| H_NO_Teff | 0.875 | 0% | 21% | NO Teff suppression saturating |
| H_TGFb_Th_exh | 0.799 | 0% | 11% | functional |
| H_CCL5_Treg | 0.766 | 0% | 10% | functional |
| H_CXCL12_Texcl | 0.366 | 0.5% | 0% | mid-band, healthy |

**Why CD8 lands low at the posterior:**
1. `H_TGFb_APC` at 0.93 → APC maturation is suppressed regardless of upstream APC parameters.
2. `H_IL12` ≈ 0 → no IL-12 amplification of cDC1 / CD8 priming.
3. → Both APC priming axes are dead → no mature DCs in V_LN → no CD8 priming.
4. → Low CD8 → low IFNg → no signal to push macrophage M2→M1 (IFNg axis is now linear via `k_IFNg_response`, not Hill-mediated; with low IFNg the linear input to the M1 polarization rule is small).
5. Meanwhile `H_IL6_M2` ≈ 0.89 saturates the M2 push from iCAF-/cancer-derived IL-6 → high k_M2_pol/k_M1_pol effective ratio is downstream of IL-6 saturation, not the cause.

**The k_M2_pol/k_M1_pol = 11 from the earlier read was a symptom, not a root cause.**

## Cytokine level vs literature audit (PDAC tissue)

`V_T.*` species are nM. Literature ranges from PDAC / pancreatic-tissue cytokine reviews (rough guidance, not authoritative):

| species | model median (nM) | model p05–p95 | lit range (nM) | EC50 (param) | model EC50 | lit EC50 | verdict |
|---|---|---|---|---|---|---|---|
| TGFb | 0.41 | 0.07–2.6 | 0.4–20 (active) | TGFb_50_APC | **0.029** | 0.04–0.4 | EC50 ~5× too low |
| IL-12 | 3e-3 | 7e-4–1.8e-2 | 0.014–0.14 | IL12_50 | **14.1** | 0.014–0.14 | **EC50 100× too high — critical** |
| IL-10 | 0.011 | 3e-3–0.04 | 0.5–50 | IL10_50 | **7.93** | 0.1–1 | level 50–1000× low + EC50 ~5× high |
| IL-6 | 0.12 | 0.02–0.42 | 0.04–4 | IL6_50_M2 | **0.015** | 0.04–0.38 | EC50 3–25× too low |
| IL-1 | 0.64 | 0.26–1.6 | 6e-5 – 6e-3 | IL1_50 | 0.030 | ≈0.01 | **level 100–10000× too high** |
| CCL2 | 3.11 | 0.93–11.4 | 0.01–1 | CCL2_50 | 0.262 | ≈0.01 | level + EC50 both ~25× too high |
| IFNg | 3e-4 | 8e-6–0.018 | 6e-5 – 6e-3 | (linear) | — | — | plausible cold-tumor level |
| VEGF | 2.0e4 | (large) | needs check | VEGF_50 | 1.9e3 | needs target | both unanchored, units may not be nM |

**Two distinct failure modes:**

A. **EC50s mis-anchored despite having submodel targets** — IL12_50 (100×), IL10_50 (5×), IL6_50_M2 (3–25×), TGFb_50_APC (5×). Either the submodel target distributions were extracted from non-tissue-matched experimental readouts, or NPE drift across the joint posterior is dominating the literature pull from the submodel target.

B. **Cytokine *levels* mismatched** despite secretion-rate submodel targets — IL-1 (100–10000× too high in tissue), IL-10 (50–1000× too low — k_IL10_sec posterior median 1e-11 is suspiciously small), CCL2 (~25× high). These are posterior simulation outputs, so they reflect (secretion - degradation - dilution) at steady state.

## RETRACTED: codegen unit bug claim was a unit-confusion bug in my own analysis (2026-05-05)

**Original claim (wrong):** picomolarity params (IL12_50, IL10_50, IL10_50_phago, IFNg_50_ind) were silently 1000× too high in the simulator due to a `* 1e-9` instead of `* 1e-12` conversion factor in `cpp/qsp/ode/ODE_system.cpp`.

**Correction:** the codegen is *correct*. The `* 1e-9` factor for picomolarity parameters converts pM → mol/m³ (the SBML unitDefinition encodes pM as `mole × metre^-3 × dimensionless×1e-9`, which evaluates to 1e-9 mol/m³). The species use `* 1e-12` because their substanceUnits are a *different* derived unit (`MWDERIVEDUNIT_nanomolarity_milliliter` = nM × mL = 1e-12 mol amount). Both are correct for their respective unit declarations.

The "100% of sims at H<0.05 for H_IL12 / H_IL10" was an artifact of my analysis script dividing `V_T.IL12` (nanomolar) by `IL12_50` (picomolar) as raw numbers, without converting to a common unit. With proper conversion (IL12_50_pM × 0.001 → nM), H_IL12 ≈ 0.19 at posterior median (mid-band, functional), and H_IL10 ≈ 0.58 (mid-band, engaged).

This also explains why `make parity` (MATLAB ↔ C++ trajectory comparison) hasn't tripped — there is no divergence to catch.

**Lesson** (added to memory queue): **always harmonize units before computing dimensional ratios in cross-sim analysis scripts**. The species-vs-EC50 unit mismatch is a known SimBiology export pattern (species in concentration × volume substanceUnits, EC50s in plain prefixed molarity). A small `unit_aware_hill()` helper in qsp-inference would prevent this.



**`cpp/qsp/ode/ODE_system.cpp`** (auto-generated by `popellab/qsp-codegen` from the SBML/SimBiology export):

```cpp
//IL10_50, index: 290, units: MWBUILTINPREFIX_pico_MWBUILTINUNIT_molarity
_class_parameter[P_IL10_50] = PFILE(QSP_IL10_50) * 1e-09;     // BUG: should be 1e-12
//IL12_50, index: 291, units: MWBUILTINPREFIX_pico_MWBUILTINUNIT_molarity
_class_parameter[P_IL12_50] = PFILE(QSP_IL12_50) * 1e-09;     // BUG: should be 1e-12
//IL10_50_phago, ... pico_molarity
_class_parameter[P_IL10_50_phago] = PFILE(QSP_IL10_50_phago) * 1e-09;     // BUG
//IFNg_50_ind, ... pico_molarity
_class_parameter[P_IFNg_50_ind] = PFILE(QSP_IFNg_50_ind) * 1e-09;          // BUG

// Compare correct species init (uses 1e-12 for pM→M):
_species_var[SP_V_T_IL12] = PFILE(QSP_V_T_IL12) * PFILE(QSP_V_T) * 1e-12;  // OK
```

**The four picomolarity-declared parameters above are silently 1000× too high in the simulator.** Hill ratios using them are exactly the saturation pattern observed:
- H_IL12 = 3 pM / 15 nM ≈ 2e-4 (≈0 across all 390 posterior sims)
- H_IL10 = same pattern
- → IL-12 amplification of CD8 priming **dead** (root cause of cd8 deficit)
- → IL-10 inhibition of M1 / APC **dead** (currently inert; turning it on may *worsen* M1/M2 — must be re-tested)

**This is the structural root cause of the cd8/M1-M2 cluster tension. All other "EC50 too high" / submodel-posterior-vs-literature mismatches in the table above need to be re-evaluated *after* the codegen fix lands**, because the joint NPE has been compensating for the silent 1000× bug.

## Fix paths

1. **Root fix in `popellab/qsp-codegen`**: the SimBiology unit-prefix decoder is mis-mapping `MWBUILTINPREFIX_pico` to the nanomolar conversion factor (1e-9) instead of picomolar (1e-12). Likely a hard-coded fallback or table miss in the codegen template. Affects every model that uses pM-declared params.
2. **Workaround in pdac-build**: in `parameters/parameters_PDAC.m` + `model_structure.json` + `parameters/pdac_priors.csv`, change the four params from `picomolarity` to `nanomolarity` and divide the values by 1000 (15 pM → 0.015 nM, 8 pM → 0.008 nM, etc.). Re-export SBML, re-run codegen. The Hill ratios will then compute correctly because both species and EC50 will share `* 1e-9`.
3. **Patch generated C++ directly**: not viable — overwritten on next codegen run.

Path 1 is the right fix; path 2 is a same-day stopgap. After either lands, re-run the 20k SBI and re-do the saturation audit. Predict: H_IL12 / H_IL10 will move into mid-band; cd8 density posterior median will rise (dual-lockout removed); M1/M2 posterior median may *drop* (now that IL-10 is actually inhibiting M1 polarization) — verify with PPC.

## Original action items (deferred until codegen fix lands)

| priority | item | owner |
|---|---|---|
| 1 | Crack open `IL12_50_PDAC_deriv001.yaml` + `IL10_50_PDAC_deriv001.yaml` + `k_IL10_sec_PDAC_deriv001.yaml` and read empirical_data vs posterior medians | done — submodel posterior at 14 nM = literature * 1000, consistent with the codegen bug pulling IL12_50 up |
| 2 | Same for `TGFb_50_APC_PDAC_deriv001.yaml`, `IL6_50_M2_PDAC_deriv001.yaml`, `CCL2_50` | open |
| 3 | Trace V_T.IL1 dynamics — is k_IL1_sec posterior pushed too high, or is k_IL1_deg too low? | follow-up |
| 4 | Add submodel targets for VEGF_50, DAMP_50, NO_50_Teff, ArgI_50_Teff, ArgI_50_Treg (all unanchored) | extraction queue |
| 5 | Re-evaluate CCL2_50 / k_CCL2_sec interaction (both targets exist; both look off) | follow-up |
| 6 | **Codegen unit-prefix fix in `popellab/qsp-codegen` and regenerate `cpp/qsp/ode/`** | **NEW priority 0 — blocks all of the above** |

## V_T.TGFb steady-state trace (2026-05-05)

Hill saturation reframing led to "is the cytokine *level* in the model right" question. Trace at posterior median:

```
V_T.TGFb posterior median = 0.41 nM (= 410 pM)

SS contribution breakdown (390 PPC sims, k_TGFb_deg = 14.3 /day fixed):
  baseline (f_active * TGFb_baseline):  0.0012 nM   (0.3%)
  Treg secretion:                        6e-5 nM    (negligible)
  myCAF secretion:                       0.418 nM   (>99%)
  M2 secretion:                          5e-4 nM    (negligible)
  predicted total (sum):                 0.42 nM   ← matches actual 0.41 nM ✓

Per-sim dominance: 390/390 sims have myCAF as the top TGFb source.
```

myCAF compartment at posterior:
- V_T.myCAF = 7.6e8 cells (3e7 cells/mL) — consistent with PDAC desmoplasia
- Total CAFs = 1.07e9 cells (4.5e7/mL), exceeds V_T.C1 = 9.8e8 cancer cells
- myCAF / (myCAF+iCAF+apCAF) = 0.69 (matches mycaf_fraction_of_caf observed 0.72)

`k_TGFb_sec_myCAF` posterior median = 2.75e-9 nmol/cell/day. Bachem1992 (MFBLC, the submodel target's primary source) reports 2e-7 nmol/cell/day. The submodel posterior is already **70× below the literature value** — partly explained by `f_active_TGFb = 0.077` (literature is total TGFb), but even after the bioactive correction the SBI is squeezing secretion-per-cell ~5× below source. **The CAF count is correct; the secretion rate has been compressed by the joint posterior to keep V_T.TGFb tractable.**

Implication: V_T.TGFb at 0.41 nM is **emergent from realistic PDAC CAF abundance with literature-anchored (or below) per-cell secretion**. The question becomes whether **0.41 nM is actually plausible for PDAC tumor TGFb in vivo**. The TGFb_50_* submodel sources peak at ~80 pM (Ahmadzadeh 40 pM max, FogelPetrovic 80 pM max), so the model lives 5–10× above their characterized regime — making `H_TGFb_*` saturate.

**Decisive next step: anchor V_T.TGFb to PDAC tissue concentration data.** If literature PDAC tumor TGFb is ~1 nM, the model is correct and the EC50 source data is incomplete (need wider-range dose-responses). If literature is ~50 pM, V_T.TGFb is 8× too high and CAF abundance / secretion / clearance need rebalancing.

This generalizes to the rest of the saturating cluster (V_T.IL1=0.64 nM, V_T.CCL2=3 nM, V_T.NO=10.5 nM, V_T.ArgI=1.7 nM): **anchor cytokine *concentrations* directly, not just per-cell secretion rates**. The deferred CSV `notes/calibration/pdac_cytokine_chemokine_targets.csv` has 12 rows queued for exactly this — TGFb_tumor, TGFb_plasma, CCL2, IFNg, IL10, IL6, IL12, arginase, CXCL9/12, CCL5, PDGF — all currently unextracted.

## Revised priority queue

| priority | item |
|---|---|
| 1 | **Extract `tgfb_concentration_tumor` calibration target** (currently in `notes/calibration/pdac_cytokine_chemokine_targets.csv`) — anchors V_T.TGFb directly. Decisive on whether the saturating Hill cluster (H_TGFb_APC, _CAF_act, _Teff, _col_sec) is real model misspec or just below the EC50-data dose range |
| 2 | Same for `il1_concentration` (V_T.IL1=0.64 nM, ~100× above source data dose range), `ccl2_concentration` (3 nM in model), `il6_concentration`, `il10_concentration`, `il12_concentration`, `arginase_concentration` |
| 3 | **NO_50_Teff and ArgI_50_Teff submodel target extraction** (just queued in `notes/targets.csv`) — these are saturating and unanchored on the EC50 axis |
| 4 | **k_*_sec posteriors at e^-23 to e^-25**: trace whether each submodel forward model has finite gradient w.r.t. the rate constant or only constrains a ratio. Degenerate posterior at 0 is a forward-model identifiability issue |
| 5 | **IL12_50 submodel target dose range too narrow** (Silveira2025 0.83–6.67 pM); EC50 of 14 pM is extrapolated. Re-extract from a richer IL-12 dose-response |
| 6 | **IL6_50_MDSC posterior CV=0.96** (uninformative; mirrors IL6_50_M2 via prior leakage). Re-extract from a multi-dose source or merge into the M2 EC50 |

## Cross-references

- `project_il10_il6_tissue_targets.md` — earlier flux analysis flagged k_Mac_death + k_M2_pol as priorities; superseded by this analysis (the saturation map shows polarization rates are downstream of IL-6 saturation; primary fix is on the EC50 / level axis, not the rate axis).
- `feedback_no_prior_tuning.md` — this analysis identifies submodel-target / literature mismatches; remediation is at the submodel-target layer (re-extract / re-evaluate sources), not by tuning CSV priors.
- `project_pdl1_rebaseline_landed.md` — PD-L1 rebaseline closed the cd8 marginal coverage gap; this Hill-saturation analysis surfaces the next-deeper layer of structural tension.

## Methodology / repro

```python
# 390 successful PPC sims at diagnosis state (t=0 row of t=[0, 0.5, 1] day)
import pandas as pd, numpy as np
p = "cache/sbi_simulations/multi_scenario_v5_qual_migration_3a0063dd_baseline_no_treatment_posterior_predictive_8d1e31de/species_*.parquet"
df = pd.read_parquet(glob.glob(p)[0])
df = df[df["status"]==0].copy()
for c in df.columns:
    if df[c].dtype == object:
        df[c] = df[c].apply(lambda a: a[0] if isinstance(a, np.ndarray) else a)

# Hill = species / (species + EC50). Compute per-sim, summarize fraction at <0.05 / >0.95.
```

Compute lives in: this conversation only — not yet committed as a script. If we run this analysis again post-extraction-update, factor into `scripts/hill_saturation_audit.py` (new).
