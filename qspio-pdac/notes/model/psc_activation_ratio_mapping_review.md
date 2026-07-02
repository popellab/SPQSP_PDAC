# PSC Activation Ratio Observable Mapping Review

**Date:** 2026-03-09
**Status:** Under investigation

## The Problem

After excluding `cancer_cell_density_baseline`, the top two misfit drivers are `psc_activation_ratio` (ΔD²=398) and `mycaf_fraction_of_caf` (ΔD²=393), together accounting for 89% of posterior D²=449.

## Observable Definitions

| Target | Formula | Source | Target Value |
|--------|---------|--------|-------------|
| `psc_activation_ratio` | myCAF / (iCAF + myCAF + apCAF + qPSC) | Drifka 2016 (n=114, α-SMA IHC) | 0.575 |
| `mycaf_fraction_of_caf` | myCAF / (iCAF + myCAF + apCAF) | Ogawa 2021 (n=215, ACTA2+FAP mIHC) | 0.716 |

## Key Insight: Targets Are Consistent IF qPSC Pool Is Maintained

The two targets differ only in whether qPSC is in the denominator. Back-solving:

```
M / (A + Q) = 0.575   (Drifka)
M / A       = 0.716   (Ogawa)

=> Q/A = (0.716 - 0.575) / 0.575 = 0.245
=> qPSC ≈ 20% of activated CAFs, or ~20% of all fibroblasts
```

The model produces Q/A ≈ 0.022 (2.2%), so the two observables collapse to nearly the same number (0.600 vs 0.613). The model depletes qPSC too aggressively.

## BUT: Is the Denominator Mapping Correct?

### Drifka's Actual Measurement
- α-SMA IHC with color deconvolution: DAB pixels / (DAB + hematoxylin pixels)
- Measured in **periductal stromal ROIs** on 1mm TMA cores
- This is a **pixel-based (area) measurement**, not cell counts

### Current Mapping Concern
The denominator currently includes only fibroblast-lineage cells (iCAF, myCAF, apCAF, qPSC). But the experimental denominator is "all hematoxylin-stained cells in the stromal ROI," which includes immune infiltrates.

### Two Competing Arguments

**Argument for adding immune cells to denominator:**
- Hematoxylin stains ALL nucleated cells, not just fibroblasts
- Macrophages are large (20µm, 314µm² cross-section) and could contribute meaningfully to pixel area
- At whole-tumor average densities (TAM ~196/mm², TIL ~153/mm²), immune cells could be ~15-25% of total stromal cell area
- This would explain the 0.575 vs 0.716 gap without needing a large qPSC pool

**Argument AGAINST adding immune cells:**
- "Periductal stromal ROIs" are the dense desmoplastic band immediately surrounding cancer nests
- PDAC is famously immune-excluded — immune cells concentrate at tumor periphery, not in dense periductal stroma
- The whole-tumor immune densities are averaged spatially; periductal ROIs likely have much lower immune content
- If immune dilution is only ~5% in periductal stroma, the current fibroblast-only mapping is approximately correct
- The YAML already acknowledges this: "Partially offset by the larger area-per-cell of activated myofibroblasts"

### What We Don't Know
- The actual immune cell density within Drifka's specific periductal ROIs
- Whether Drifka's "periductal stromal regions" include or exclude immune-rich areas
- The spatial distribution of TAMs relative to the desmoplastic band (some studies suggest TAMs are interdigitated with CAFs, not excluded)

## Literature on qPSC Fraction

- No direct IHC measurement of quiescent PSC fraction within PDAC tumors exists
- Literature consensus: PSC activation in PDAC is near-complete
- Normal pancreas: qPSCs are 4-7% of parenchymal cells
- Helms 2022: PSC-derived CAFs are only ~10-15% of total CAFs (most from perivascular/mesenchymal lineages)
- 20% quiescent PSCs is likely too high; 5-10% might be an upper bound within tumor

## Model Dynamics (Why qPSC Depletes)

From fibroblast_module_enhanced.m:
- **Supply:** `k_CAF_encounter * V_T * ((k_CAF_const - qPSC/V_T) / k_CAF_const)` — logistic encounter as tumor grows
- **Depletion:** TGFb/PDGF-gated activation to myCAF/iCAF + constitutive activation to apCAF + death
- At diagnosis: PDGF is very low (H_PDGF ≈ 0.003), heavily suppressing activation
- The steady-state qPSC pool is small because encounter rate and activation balance at a low level

## Options

1. **Re-examine Drifka paper** — understand exactly what "periductal stromal ROIs" contain. If immune cells are present in meaningful numbers, add area-weighted immune terms to the denominator. This would reconcile the two targets without requiring model changes.

2. **Accept current mapping, fix model** — if periductal stroma is truly fibroblast-dominated, the 20% qPSC requirement is real and the model needs to maintain a larger quiescent pool (slower activation, or qPSC replenishment).

3. **Consider the target unusable** — Drifka's measurement may be too ambiguous (pixel-based, ROI composition unknown, no dispersion reported) to serve as a reliable calibration target. The Ogawa target is cleaner (marker-classified, explicit denominator). Could drop Drifka and rely solely on Ogawa for myCAF fraction.

4. **Rename and reframe** — regardless of fix, `psc_activation_ratio` is a misleading name. The observable measures myCAF fraction of fibroblasts (or stromal cells), not PSC activation per se.

## Next Steps

- [ ] Read Drifka 2016 paper carefully for ROI selection methodology
- [ ] Determine if immune cell content in periductal ROIs is quantifiable
- [ ] Decide whether to fix the denominator mapping or drop the target