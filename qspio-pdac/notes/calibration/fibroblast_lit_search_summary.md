# Fibroblast Parameter Literature Search Summary

Date: 2026-03-18

## Objective

Find better literature sources for CAF interconversion and activation rate parameters
(Priority 1 in `notes/calibration/fibroblast_param_audit.md`). The existing sources
(Biffi 2019, Elyada 2019) provide qualitative mechanism and steady-state ratios but
no quantitative kinetic data for rate constants.

## Parameters of interest

| Parameter | Current value | Units | Current source | Gap |
|-----------|-------------|-------|---------------|-----|
| `k_PSC_to_myCAF` | 0.4 | 1/day | Biffi 2019 (qualitative) | No rate data |
| `k_PSC_to_iCAF` | 0.2 | 1/day | Biffi 2019 (qualitative) | No rate data |
| `k_PSC_to_apCAF` | 0.01 | 1/day | Elyada 2019 (~2-5%) | No rate data |
| `k_iCAF_to_myCAF` | 0.1 | 1/day | Biffi 2019 (qualitative) | No rate data |
| `k_myCAF_to_iCAF` | 0.05 | 1/day | Biffi 2019 (qualitative) | No rate data |
| `k_myCAF_death` | 0.0165 | 1/day | Hao 2015 | Generic fibroblast |
| `k_iCAF_death` | 0.0165 | 1/day | Hao 2015 | Generic fibroblast |
| `k_myCAF_prolif` | 0.15 | 1/day | Inherited | No source |
| `k_iCAF_prolif` | 0.10 | 1/day | Inherited | No source |
| `k_TGFb_sec_myCAF` | 2e-11 | nM*L/(cell*day) | Calon 2015, Pickup 2013 | Indirect |

## Papers reviewed

### Already in library (existing calibration targets)

**Elyada 2019** (Cancer Discovery, DOI: 10.1158/2159-8290.CD-19-0094)
- KPC flow cytometry: iCAF 44.4+/-3.9%, myCAF 45.3+/-4.1%, apCAF 10.3+/-1.25% (n=20 tumors)
- Human scRNA-seq: 20.9% of CAFs co-express HLA-DRA + CD74 (apCAF markers)
- **Status:** Active calibration targets exist for iCAF/myCAF/apCAF fractions (Ogawa 2021 primary, Elyada excluded as mouse-redundant)

**Biffi 2019** (Cancer Discovery, DOI: 10.1158/2159-8290.CD-18-0710)
- Establishes IL-1/JAK/STAT -> iCAF vs TGFb/SMAD -> myCAF antagonism
- Key time-scales: iCAF gene induction ~1 hour (IL-1alpha exposure); iCAF->myCAF reversion ~24 hours (JAK inhibitor); myCAFs proliferate faster than iCAFs
- TGFb antagonizes IL-1 by downregulating IL-1R1 (SMAD3-mediated)
- IL-1alpha used at 20 ng/mL in organoid assays (~1.2 nM)
- **Assessment:** Mechanism confirmed (already encoded in model). No quantitative rate constants extractable.

**Ohlund 2017** (JEM, DOI: 10.1084/jem.20162024)
- Original paper defining iCAF/myCAF in organoid co-culture
- PSC proliferation curve (Fig 2H): ~3x mCherry fold-change by ~120 hours
- iCAF Ki67: 23% of PDGFRb+/IL-6+ cells are Ki67+ in vivo (Fig 4F-G)
- iCAF->myCAF reversion: "rapidly reverted" when plated in monolayer (qualitative)
- **Assessment:** Qualitative switching data, no rate constants. Ki67 fraction could constrain `k_iCAF_prolif` but hard to map without modeling Ki67 directly.

### New papers identified

**Lee & Kim 2025** (npj Systems Biology, DOI: 10.1038/s41540-025-00578-y)
- ODE model of CAF heterogeneity with 33 parameters
- **Assessment:** NOT USEFUL. Models CAF phenotype as static composition fractions (alpha, beta, gamma, delta for antiimmune/proimmune/anticancer/procancer), NOT dynamic iCAF/myCAF/apCAF interconversion. No transition rate ODEs.

**Bougueon et al. 2024** (PLoS Comput Biol, DOI: 10.1371/journal.pcbi.1011858)
- Rule-based multiscale model of HSC dynamics during liver fibrosis
- 7 HSC physiological states, 37-43 parameters
- Key calibrated values from S2 Table:
  - `HSC_stage_half_life` = 17 hours/step x 7 steps for qHSC->aHSC (total ~5 days)
  - `MFB_stage_half_life` = 17 hours/step x 7 steps for aHSC->MFB (total ~5 more days)
  - `MFB_half_life` = 3.54 days (post-TGFb removal)
  - `HSC_doubling_time` = 15.5 days (aHSC), 31 days (MFB)
  - `react_HSC_stage_half_life` = 4.86 hours/step (3.5x faster reactivation)
  - `inactivation_percentage` = 45% of MFBs inactivate (vs 55% die)
  - `inactivation_percentage_react` = 5% of reactivated MFBs
  - `iHSC_half_life` = 2 years
- Calibrated against Kisseleva 2012 CCl4 mouse data
- **Assessment:** Valuable as a computational reference, but parameters are model-derived (calibrated), not direct measurements. Points us to the primary data sources (Bachem, Kisseleva).

### Primary data sources (newly obtained PDFs)

**Bachem et al. 1992** (J Clin Invest, DOI: 10.1172/JCI115561)
- Rat liver perisinusoidal lipocyte (HSC) activation by MFB-conditioned media
- Activation timescale: 6-10 days on plastic to fully transformed MFBLC
- **Key quantitative data:**
  - Table I: BrdU incorporation (% BrdU+) for PL-3d, PL-7d, MFBLC at 0/0.5/2/10% FCS (mean+/-SD, n=3-4). Shows proliferation increases with activation state.
  - Table II: TGFalpha and TGFbeta1 production rates (fmol/cell/d) for PL-3d, PL-6d, TC-10d, MFBLC. TGFbeta1: n.d. -> 0.02 -> 0.13 -> 0.20 fmol/cell/d.
  - Fig 7a+b: Cell count + alpha-SMA% time course (days 4,7,11,16) with control vs TGFbeta1 vs aMFBcM stimulation (mean+/-SD, n=3). Needs digitization.
  - Fig 8: Retinyl palmitate loss time course (days 1-8) with aMFBcM.
- Statistical analysis: "Data are presented as mean+/-SD" (explicitly stated)

**Bachem et al. 1993** (J Hepatology, DOI: 10.1016/S0168-8278(05)80008-X)
- HSC response to growth regulators during transdifferentiation
- Transdifferentiation within 1-2 weeks on plastic
- **Key quantitative data:**
  - Fig 4a: Dual-axis time course of cell count (x10^3/well) + alpha-SMA% over days 0-16 in primary culture (mean+/-SD, n=3). THE key activation time course. Needs digitization.
    - alpha-SMA%: ~0% (day 2) -> ~30% (day 7, confluence) -> ~90% (day 12-14)
    - Cell count: flat day 1-4, doubles day 4-7, plateaus at confluence
    - At low seeding density (40x10^3 cells/well): confluence at day 12-14, 90% alpha-SMA+
    - At high density (150x10^3 cells/well): 30% alpha-SMA+ at confluence
  - Fig 4b: Retinyl palmitate loss: 60 -> 18 ug/10^6 cells over 16 days primary culture
  - Table 1: TGFalpha/TGFbeta1 effects on PL vs MFBLC proliferation (mean+/-SD, n=3). TGFbeta1 inhibits PL (0.2+/-0.1 at 5ng/ml) but no effect on MFBLC (1.08+/-0.1).
  - Fig 5: Receptor binding (PDGF, TGFbeta1, IGF1, EGF) to PL-1d, PL-4d, MFBLC (mean of 2, n=3)
  - Fig 7: PDGF + IGF1 dose-response on PL and MFBLC proliferation (bars with significance)
- Statistical analysis: "Data are presented as mean +/- S.D." (explicitly stated)

**Kisseleva et al. 2012** (PNAS, DOI: 10.1073/pnas.1201840109)
- In vivo genetic labeling of HSCs during CCl4-induced liver fibrosis + recovery
- Col-alpha1(I)-GFP reporter mice + Cre-loxP fate mapping
- **Key quantitative data (all text-reported with +/-SD):**
  - Baseline: qHSCs = 10.6+/-0.8% of total liver cells (n=4)
  - Fibrosis (2mo CCl4): 98+/-2% of HSCs activated (upregulated YFP)
  - Fibrosis: total HSCs = 14.3+/-1.5% of liver cells; alpha-SMA = 8+/-1.5%
  - Recovery 1mo: alpha-SMA+ myofibroblasts "completely disappeared"
  - Recovery 1mo: total HSCs = 5.6+/-1.8% of liver cells
  - Recovery 1mo: YFP+ in 38+/-8% of Desmin+ HSCs and 41+/-5% of GFAP+ cells (iHSCs)
  - Recovery 1mo: GFP+ iHSCs = 14+/-4% by tamoxifen genetic labeling (n=6)
  - Apoptosis: 2.6+/-0.7% caspase-3+ aHSCs/myofibroblasts at day 7 post-CCl4
  - Alcohol model: 64+/-5% genetic labeling achieved; 36+/-4% vitamin A+ YFP+ HSCs persisted
  - Fig 3B: 2xCCl4 more severe fibrosis than 1xCCl4 (iHSC reactivation)
  - Fig 3C: Adoptive transfer engraftment: iHSCs 19+/-2.3% and 13+/-2.0% vs qHSCs 0.5+/-0.2% (40x reactivation advantage)

## Existing calibration targets (CAF fractions)

| Target | Source | Data | Status |
|--------|--------|------|--------|
| `icaf_fraction_of_caf` | Ogawa 2021 (n=215, human mIHC) | median 0.284 [0.094, 0.604] | Active |
| `mycaf_fraction_of_caf` | Ogawa 2021 (n=215, human mIHC) | median 0.716 [0.396, 0.906] | Active |
| `apcaf_fraction_of_caf` | Elyada 2019 + Ogawa 2021 | median 0.064 [0.02, 0.21] | Active |
| `icaf_fraction_of_caf_elyada2019` | Elyada 2019 KPC (n=20) | 0.444 [0.369, 0.521] | Excluded (mouse, redundant) |
| `apcaf_fraction_of_caf_elyada2019` | Elyada 2019 KPC (n=20) | 0.103 [0.080, 0.129] | Excluded (mouse, redundant) |
| `psc_activation_ratio` | Drifka 2016 (n=356 cores) | 0.575 [0.345, 0.786] | Excluded (denominator mismatch) |

Existing submodel target in to-review/:
| Target | Source | Data | Status |
|--------|--------|------|--------|
| `k_psc_activation_apte1998` | Apte 1998 (alpha-SMA at 48h) | t_act ~ 48h | In to-review/, needs schema update |

## Planned submodel targets

### Target 1: Stellate cell activation time course — DONE
- **Source:** Bachem 1993 Fig 4a
- **Parameters:** `k_PSC_to_myCAF`, `k_myCAF_prolif` (joint)
- **Forward model:** `custom_ode` — Q (quiescent) + A (activated) with logistic proliferation
- **Data:** alpha-SMA% + cell count at days 2, 4, 6, 10, 16 (mean+/-SD, n=3). Digitized.
- **File:** `calibration_targets/submodel_targets/k_psc_activation_PDAC_deriv002.yaml`
- **Validation:** Schema+snippets pass. Prior derivation awaits custom_ode inference support.
- **10 error model entries** (5 time points x 2 observables: alpha-SMA fraction + cell count)

### Target 2: In vivo HSC dynamics during fibrosis recovery — DONE
- **Source:** Kisseleva 2012 (text-reported values)
- **Parameters:** `k_myCAF_death`
- **Forward model:** `algebraic` — MFB exponential decay during 30-day recovery
- **Data:** Total HSC% at fibrosis (14.3+/-1.5%) and recovery (5.6+/-1.8%), alpha-SMA at fibrosis (8+/-1.5%)
- **File:** `calibration_targets/submodel_targets/k_myCAF_death_PDAC_deriv001.yaml`
- **Validation:** All pass. Derived prior median=0.0168/day, confirms current value (0.0165/day).

### Target 3: Stellate cell proliferation by activation state — DONE
- **Source:** Bachem 1992 Table I
- **Parameters:** `k_myCAF_prolif`
- **Forward model:** `algebraic` — BrdU fraction = 1 - exp(-k * t_label)
- **Data:** BrdU% for MFBLC at 4 FCS concentrations (21-60%, mean+/-SD). Bootstrap across conditions.
- **File:** `calibration_targets/submodel_targets/k_myCAF_prolif_PDAC_deriv001.yaml`
- **Validation:** Schema+snippets pass. Prior derivation has JAX tracer issue to debug.

### Target 4: TGFbeta1 secretion rate by stellate cells — DONE
- **Source:** Bachem 1992 Table II
- **Parameters:** `k_TGFb_sec_myCAF`
- **Forward model:** `algebraic` — convert parameter units to fmol/cell/d
- **Data:** TGFbeta1 production: 0.02 (PL-6d), 0.13 (TC-10d), 0.20 (MFBLC) fmol/cell/d
- **File:** `calibration_targets/submodel_targets/k_TGFb_sec_myCAF_PDAC_deriv001.yaml`
- **Validation:** All pass. Derived prior median=1e-9, ~50x shift from current 2e-11.

### Next priority targets

5. **Bachem 1993 Fig 7 — PDGF dose-response on MFBLC proliferation** → `PDGF_50_prolif`
  - PDGF(BB) at 2, 10, 20 ng/ml (~0.08, 0.4, 0.8 nM; PDGF-BB MW ~25 kDa) stimulates
    MFBLC thymidine incorporation. Bars with significance markers, n=3.
  - Forward model: `direct_fit` Hill curve. f_prolif = baseline + (max-baseline)/(1+(PDGF/EC50)^n).
  - Current `PDGF_50_prolif = 0.2 nM` has no source. This is a direct dose-response
    constraining the EC50.
  - Text also reports: "PDGF(AB) acted as a mitogen for MFBLC cultured in fetal calf
    serum deficient (<0.5%) medium only" and "10 ng/ml PDGF(AB) produced a 1.5-fold
    increase" in thymidine incorporation.
  - **Needs digitization** of Fig 7b bars (PDGF panel).

6. **Bachem 1992 Fig 7b — TGFbeta1-accelerated activation** → `TGFb_50_CAF_act`
  - alpha-SMA% time course at days 4, 7, 11, 16 with three conditions:
    control (10% FCS alone), 1 ng/ml TGFbeta1 (~0.04 nM), and 20% acidified MFBcM.
  - Mean+/-SD, n=3 per condition. TGFbeta1 and aMFBcM accelerate activation vs control.
  - Forward model options: (a) ratio of activation rates at known TGFbeta concentration
    → algebraic constraint on TGFb_50_CAF_act; (b) paired ODE fits with/without TGFbeta.
  - The TGFbeta1 concentration (0.04 nM) relative to TGFb_50_CAF_act (0.07 nM) gives
    Hill occupancy ~36%. If TGFbeta1 doubles the activation rate, this constrains the EC50.
  - **Needs digitization** of Fig 7b (all three conditions).

### Lower priority targets

7. **Kisseleva 2012 apoptosis snapshot** → independent `k_myCAF_death` constraint
  - 2.6+/-0.7% caspase-3+ MFBs at day 7 post-CCl4 cessation (text-reported, n not stated).
  - Forward model: fraction_apoptotic = k_death x tau_apoptosis (snapshot relationship).
  - Requires knowing apoptosis clearance time (tau ~6-12h typical), which is not reported.
    Could use literature value for tau as a fixed parameter.
  - Complementary to existing k_myCAF_death target (population decline vs instantaneous
    apoptotic fraction — different systematic errors).

8. **Kisseleva 2012 Fig 3C — Adoptive transfer reactivation ratio**
  - iHSCs engraft at 19+/-2.3% vs qHSCs at 0.5+/-0.2% (text-reported, n not stated).
  - 38x reactivation advantage for pre-activated cells.
  - Could constrain ratio k_PSC_to_iCAF / k_PSC_to_myCAF, interpreting
    iHSC reactivation as analogous to re-engagement of the iCAF activation pathway.
  - Mapping is indirect: adoptive transfer engraftment != activation rate. The 38x
    includes homing, survival, and proliferation advantages, not just activation kinetics.

9. **Bachem 1993 Table 1 — TGFbeta1 differential proliferation effects**
  - TGFbeta1 inhibits PL proliferation (0.2+/-0.1x at 5 ng/ml) but has no effect on
    MFBLC (1.08+/-0.1x). TGFalpha stimulates PL (2.6+/-0.2x) but not MFBLC.
  - Interesting biology but doesn't map to a single model parameter cleanly.
  - Could inform a qualitative bound on TGFbeta-proliferation interaction.

## Key biological insights for parameter interpretation

1. **Activation is multi-step, not instantaneous.** Bachem 1993 shows a gradual S-curve from 0% to 90% alpha-SMA+ over ~14 days, not a step function. Our single first-order rate constant is an effective approximation.

2. **Proliferation increases with activation.** Bachem 1992 Table I: BrdU+ goes from 0.1% (quiescent, 0 FCS) to 21-60% (MFBLC). Our model's `k_myCAF_prolif > k_iCAF_prolif` is directionally correct.

3. **MFB elimination is fast post-stimulus.** Kisseleva 2012: alpha-SMA+ cells disappear within 1 month of CCl4 cessation. Our `k_myCAF_death = 0.0165/day` (t_half = 42 days) may be too slow; Bougueon calibrated `MFB_half_life = 3.54 days`.

4. **Inactivated cells are NOT quiescent.** Kisseleva 2012: iHSCs reactivate 40x faster than qHSCs. This supports our model having separate `k_PSC_to_iCAF` > `k_PSC_to_myCAF` for pre-activated cells.

5. **TGFbeta1 secretion increases with activation.** Bachem 1992 Table II: 0 -> 0.20 fmol/cell/d. This is a positive feedback loop (activated cells produce TGFbeta which activates more cells), consistent with our model's `k_TGFb_sec_myCAF` parameter.

6. **HSC and PSC activation biology is highly conserved.** Both store vitamin A, express alpha-SMA upon activation, respond to TGFbeta/PDGF, produce collagen. The liver-to-pancreas extrapolation is well-supported.
