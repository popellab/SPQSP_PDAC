# CD8 Collapse Diagnosis — Baseline PDAC TME

**Date:** 2026-04-22
**Trigger:** New qualitative bounds added 2026-04-22 (`cd8_treg_ratio_max_diagnosis` lower 0.1; `treg_fraction_min` upper 0.40) fired on cached `tests/scenarios/.sim_outputs.json` base snapshot.

## Observation

At posterior-median baseline (`baseline_no_treatment` after `evolve_to_diagnosis`):

| Species | Count |
|---|---|
| Th | 6.33 × 10⁷ |
| Treg | 6.13 × 10⁸ |
| CD8 | 7.68 × 10³ |
| CD8_exh | 5.19 × 10³ |
| T_total | 6.76 × 10⁸ |

**CD8:Treg = 1.25 × 10⁻⁵** (literature mean ~3; new lower bound 0.1). **Treg/T_total = 0.91** (new upper bound 0.40).

CD8 is **5 orders of magnitude** below Th+Treg. No existing bound caught this — the pre-existing T-cell bounds were all one-sided in the "immune-hot" direction.

## Mechanistic diagnosis

Four asymmetries stack to produce the collapse. CXCL12 exclusion is NOT the primary cause — both Th and CD8 see the same `(1 − H_CXCL12_Texcl)` factor (= 0.18 at baseline; 82% excluded for both).

### #1 — Uncapped Th → Treg conversion (primary driver)

`core/modules/Th_module.m:61-62`:
```
V_T.Th -> V_T.Treg
rate = k_Th_to_Treg * V_T.Th * H_TGFb
```

No Treg-cap saturation term. Meanwhile tumor-local Treg proliferation **is** cap-gated (`core/modules/Treg_module.m:218-219`):
```
k_Treg_pro_tumor * V_T.Treg * (1 - V_T.Treg/(rho_Treg_max*V_T*(1+alpha_Treg_cap*H_apCAF) + cell))
```

At baseline:
- Effective Treg cap = `rho_Treg_max × V_T × (1 + α × H_apCAF)` = 1e7 × 17.16 × (1 + 2 × 0.66) = **3.99 × 10⁸**
- Actual Treg = **6.13 × 10⁸** — i.e. **154% of cap**.
- The excess (~2.1 × 10⁸ cells) comes from the uncapped conversion pathway.
- Conversion flux at baseline: `k_Th_to_Treg × Th × H_TGFb` ≈ 0.022 × 6.33e7 × 0.8 ≈ **1.1 × 10⁶ Treg/day**, comparable to Treg natural death (~1.8 × 10⁶/day).

**Fix candidate:** append `* (1 - V_T.Treg/(rho_Treg_max*V_T*(1+alpha_Treg_cap*H_apCAF) + cell))` to the conversion rate, matching the proliferation cap.

### #2 — No CD8 feeder pool

Th has fresh supply from `nCD4 × mcDC2` priming in V_LN (→ aTh → Th), and can convert to Treg.
CD8 requires a parallel-but-separate `nCD8 × mcDC1` priming cascade — but mcDC1 is much lower than mcDC2 (cached mcDC1 = 1.8e5 vs mcDC2 = 6.4e5), so priming is asymmetric. This is partly deliberate (cDC1 is the rare cross-presenting subtype) but compounds the other losses.

### #3 — CD8 has 5 exhaustion paths + death, no local proliferation

CD8 loss terms in `core/modules/Teff_module.m`:
- `V_T.CD8 → null` (natural death, `k_CD8_death`)
- `V_T.CD8 → V_T.CD8_exh` via: baseline (`k_CD8_basal_exh`), antigen clearance (`k_cell_clear`), Treg-mediated (`k_Treg_CD8_exh`), PD-1 (`k_C_CD8_exh × H_PD1_C1`), stiffness (`k_stiff_CD8_exh × H_stiff_Texh`)

Treg-mediated exhaustion (line 205-207):
```
k_Treg_CD8_exh * V_T.CD8 * Tregs / (V_T.CD8 + Tregs + cell)
```
With Treg ~ 6e8 >> CD8 ~ 1e4, this saturates at `k_Treg_CD8_exh = 0.1/day` — meaning CD8 is exhausted at 10% per day from Tregs alone.

Meanwhile Treg has **local proliferation** (`k_Treg_pro_tumor × V_T.Treg × (1 - …)`) while CD8 has **none** in V_T.

### #4 — CXCL12 exclusion is symmetric (not the cause of asymmetry)

Both Th and CD8 transport into V_T are gated by the same `(1 - H_CXCL12_Texcl)` factor:
- `core/modules/Teff_module.m:232`: CD8 entry
- `core/modules/Th_module.m:94`: Th entry

At CXCL12 = 9.1 nM, CXCL12_50 = 2.0 nM: `H = 0.82`, allowing 18% through for both. This slows both but doesn't explain the ~10⁵× asymmetry.

## Ranked fix candidates

1. **Cap Th → Treg conversion at the same effective Treg carrying capacity as proliferation.** Would prevent Treg from exceeding its stated cap and reduce Treg pool by ~35% immediately, which feeds back into lower Treg-mediated CD8 exhaustion (#3). Smallest, most mechanistic change.
2. **Add a local CD8 proliferation term** (analogous to `k_Treg_pro_tumor`), gated by antigen / IL-2 / IL-12. CD8 TILs do proliferate locally in the TME; the model currently omits this. Would match Treg's structure.
3. **Re-examine `k_T_activation_per_APC`** (currently 1e-4 `1/(day*cell)` — "Calibrated: 5× from baseline"). With mcDC1 ~ 1.8e5, priming flux is `1e-4 × 1.8e5 = 18/day × nCD8 × n_CD8_clones`, which may be low relative to the loss terms in #3.

## Follow-ups

- Verify fix #1 with a targeted sim after editing the Th → Treg rate.
- Re-run PPC (`workflows/run_ppc_clinical.py`) after fix to see if clinical-endpoint uncertainty narrows.
- Re-evaluate the `cd8_treg_ratio_max_diagnosis` and `treg_fraction_min` xfails in `tests/scenarios/test_qualitative_behaviors.py:2118-2140` — they should pass once #1 is in place.

## Related

- Qualitative bounds YAML: `tests/qualitative_bounds.yaml` (entries `cd8_treg_ratio_max_diagnosis`, `treg_fraction_min` now two-sided as of 2026-04-22)
- Draft supplementary bounds: `notes/calibration/qualitative_bounds_tier_a_drafts.yaml`
- Audit: `notes/calibration/qualitative_bounds_gap_audit.md`
