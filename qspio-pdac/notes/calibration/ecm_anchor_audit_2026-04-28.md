---
date: 2026-04-28
status: audit + 3 submodel targets extracted and validated; SBI re-run pending
related: collagen_attractor_overshoot.md, untreated_tension_analysis_2026-04-28.md, GitHub #93
branches_touched: feature/option-a-collagen-damper-fix (rebased onto), feature/ecm-anchor-audit-issue-93 (this work)

## Update 2026-04-28 (post-audit + post-inference)

- Branch `feature/ecm-anchor-audit-issue-93` rebased onto `feature/option-a-collagen-damper-fix`. All work consolidated on this branch.
- Three submodel targets extracted and validated against source PDFs:
  - `Ve_T_PDAC_deriv001.yaml` (Wang 2023 DCE-MRI, n=14, Ve_DCE=0.61±0.11 → Ve_T=0.39±0.11). Algebraic identity Ve_T = 1 - Ve_DCE.
  - `phi_col_ref_grizzi2019_PDAC_deriv001.yaml` (Grizzi 2019 Sirius-red, n=5, 50 ROIs, 2.23±0.28%). Direct picrosirius-area anchor.
  - `rho_collagen_valderrama1991_PDAC_deriv001.yaml` (Valderrama 1991 Sirius-red colorimetry, n=11 normal pancreas, 89.1±11.6 µg/mg). Joint constraint with phi_col_ref via the product (phi × rho) / protein_per_wet.

### Joint submodel inference (parameter_audit.py --invalidate)

Two MCMC components: ECM (5 targets, 6 params) + Ve_T (1 target, 1 param). NumPyro NUTS, 2 chains × 4000 samples after 2000 warmup. All converged (rhat=1.00, n_eff > 9000 for ECM, ~2700 for Ve_T). Initial run with priors inherited from main hit prior edges on rho_collagen and protein_per_wet_volume_pancreas; widened σ on rho_collagen (0.3 → 0.7) and ppwv (0.25 → 0.5) per `feedback_no_prior_tuning` (widen, don't shift). Re-run produced clean posteriors:

| Param | Old default | New posterior median | New σ | Contraction |
|---|---|---|---|---|
| `Ve_T` | 1.0 | **0.37** | 0.32 | 60% |
| `phi_col_ref` | 0.12 | **0.046** | 0.22 | 80% |
| `rho_collagen` | 0.025 g/mL | **0.103 g/mL** | 0.46 | 57% |
| `d_pore_ref` | 3.5 µm | 5.60 µm | (downstream) | 58% |
| `E_ref` | 1.0 | 0.70 | (downstream) | 79% |
| `n_stiff` | 1.8 | 0.72 | (downstream) | 51% |

Forward-model self-consistency on Valderrama observable: predicted c_mass = phi × rho / ppwv × 1000 = 0.046 × 0.103 / 0.07 × 1000 = **67.7 µg/mg** vs observed 89.1 µg/mg (76% match). Remaining gap = cross-paper tension (Bedossa 1989 reports 26.5 ± 7.2 µg/mg vs Valderrama 89.1; primary measurements span an order of magnitude). The posterior is the right Valderrama-only fit given the joint constraints from Wolf/Rubiano/Grizzi.

### Parameter / priors-CSV updates (committed)

- `parameters/parameters_PDAC.m` `.Value` defaults updated to posterior medians: Ve_T 0.37, phi_col_ref 0.046, rho_collagen 0.103.
- `parameters/pdac_priors.csv` median + mu updated to match. σ retained at widened values (Ve_T 0.5, phi_col_ref 0.5, rho_collagen 0.7) for any future re-inference. Submodel-derived posteriors (sigma 0.22 / 0.32 / 0.46) live in `notes/calibration/submodel_priors.yaml` and override at SBI runtime per CLAUDE.md.
- Apte 2004 PMID 15247571 verified WRONG (intervertebral-disc paper); intended Apte 2004 (PMID 15367883) is a review without quantitative phi support. Anchor re-attributed to Imamura 1995 PMID 8532652 + Grizzi 2019.

### Open items (task #14 — separate PR)

- `cross_model_parameters.csv` labels Ve_T "Void fraction" but code uses it as intracellular fraction (opposite).
- `cross_model_parameters.csv` "0.017–0.529 PDAC" range has no traceable primary source.
- PMID 22547351 (Finley 2012) is verified to NOT be a Ve_T primary source.

### Follow-on investigation (next audit step)

Post-anchor SBI showed `stromal_fraction` still OUTSIDE Bengtsson CI; diagnosed as CAF:cancer carrying-cap parameters (`f_stroma_max`, `k_stroma_support`), not the V_T volume rule. Continued in `caf_cancer_coupling_2026-04-28.md` (joint posterior pulls `f_stroma_max` 5.0 → 2.43; full-model SBI multi-observable tradeoffs prevent reaching the single-observable Bengtsson optimum).

---

# ECM anchor audit: re-anchoring (Ve_T, rho_collagen, phi_col_ref) jointly with downstream Hill EC50s — 2026-04-28

## TL;DR

Three structural anchors in the V_T volume rule and the phi_collagen Hill infrastructure
(Ve_T, rho_collagen, phi_col_ref) are tightly coupled and currently jointly compensate
for past tuning decisions. Pulling literature for the joint constraint surface yields:

- **The Apte 2004 PMID currently cited in `parameters_PDAC.m` (15247571) is verifiably wrong** — that PMID is an intervertebral-disc paper. The intended paper is Apte 2004 *Pancreas* 29(3):179–87, **PMID 15367883**, DOI `10.1097/00006676-200410000-00002`. **However**, that paper is a review/morphometric correlation paper that does not actually report the "phi_collagen ~0.10–0.15 normal pancreas" value cited in the parameters file. The 0.12 anchor is propagated from project notes, not directly verifiable in Apte 2004. Per `feedback_no_fabricated_citations`, the citation must be re-attributed.
- The verified primary anchor for normal-pancreas collagen content is **Imamura 1995 *Pancreas* 11(4):357–64, PMID 8532652** — already in `calibration_targets/reference_values.yaml` as `normal_pancreas_collagen` (5 µg/mg dry, range 3–8). At pancreas dry-weight fraction 0.22, that's ~1.1 mg/mL wet-tissue collagen for normal pancreas, with PDAC ~3× higher (Imamura) ≈ 3.3 mg/mL wet.
- The **interpretation** of `phi_collagen` is the actual root cause of the audit confusion. There are three plausible reading conventions, and the model + submodel targets currently mix them. Section 2 below disentangles which convention the existing Hill EC50s and submodel targets bake in.
- **Recommended interim posture (this PR):** keep the existing anchor triple (Ve_T=1, rho_collagen=25 mg/mL, phi_col_ref=0.12), correct the Apte PMID to 15367883, document that the 0.12 anchor is a "scaffold-equivalent fraction" interpretation (not literal fiber volume), and merge the volume-rule fixes from `feature/option-a-collagen-damper-fix`. **Defer joint re-anchoring** until after a fresh SBI run on the unified-phi + (1−phi)² damper code (commits 0c7f799, 9d22f73). The empirical posterior on phi_collagen is the strongest signal for which calibration domain we actually want.
- **Queued (separate maple sprint):** a new submodel target `rho_collagen_imamura1995_PDAC_deriv001.yaml` constraining `rho_collagen` directly via Imamura's hydroxyproline-based collagen-content measurements paired with picrosirius-red collagen area fractions from the same indication (PDAC). Spec drafted in §6 below.

## 1. The three anchors and their structural roles

```
Ve_T            cell-volume scaling factor in V_T rule (cells contribute vol_cell/Ve_T)
rho_collagen    mass-to-volume conversion: V_T.collagen / rho_collagen → "scaffold volume"
phi_col_ref     normal-pancreas reference for Hill normalization (d_pore, stiffness)
```

Downstream coupling (every Hill function listed depends on at least one of these):

| Hill / rule | Formula | Anchors used |
|---|---|---|
| `phi_collagen` | `V_T.collagen / (V_T × rho_collagen)` | rho_collagen, (Ve_T via V_T budget) |
| `d_pore` | `d_pore_ref × (phi_col_ref/phi_collagen)^0.5` | phi_col_ref, (rho via phi) |
| `stiffness` | `E_ref × (phi_collagen/phi_col_ref)^n_stiff` | phi_col_ref, (rho via phi) |
| `H_pore_T`, `H_pore_APC` | sigmoid on `d_pore/d_crit` | (all three indirectly) |
| `H_stiff_fib`, `H_stiff_Texh` | sigmoid on `stiffness/E_50` | (all three indirectly) |
| C_max suppression (`phi_col_50_Kmax`) | sigmoid on `phi_collagen` | rho_collagen |
| V_T volume rule | additive cell + collagen contributions | Ve_T, rho_collagen |

A coordinated re-anchoring sprint must update all six together.

## 2. The phi_collagen interpretation problem

Three plausible readings of `phi_collagen`, each implying a different rho_collagen:

| Convention | What phi means | rho_collagen | Phi at normal pancreas (Imamura 1.1 mg/mL wet) |
|---|---|---|---|
| (A) Literal fiber-volume | fraction of V_T occupied by pure collagen fibers at dry-protein density | ~1340 mg/mL | 0.0008 |
| (B) Fibrillar-scaffold (current model) | fraction of V_T containing scaffold-equivalent at fibrillar packing | 25 mg/mL | 0.044 |
| (C) Histological / picrosirius-area | fraction of V_T staining as collagen on picrosirius red | implicit ~9 mg/mL | 0.10–0.15 (Apte qualitative claim) |

The current parameters mix these:

- `phi_col_ref = 0.12` is the **histological convention** (C), citing "Apte 2004 normal pancreas 0.10–0.15"
- `rho_collagen = 0.025 g/mL` is the **fibrillar-scaffold convention** (B), citing Antoine 2014 9–45 mg/mL cancer interstitium and Kai 2023 ECV F3 ~42.5%
- The Wolf 2013 d_pore submodel target uses convention (B): `phi_at_X = c/25 mg/mL`, giving phi 0.07–0.6 across the in vitro gel range
- The Rubiano 2018 stiffness submodel target uses convention (B): `phi_gel = 3 mg/mL / 25 = 0.12` as the "normal pancreas equivalent"

**The mismatch:** under convention (B) (rho=25), Imamura's wet-tissue normal pancreas (1.1 mg/mL) gives `phi ≈ 0.044`, NOT 0.12. So `phi_col_ref = 0.12` is implicitly using convention (C) while `rho_collagen` is set per (B). Internal inconsistency.

Three ways to resolve:

| Fix | What changes | Pros | Cons |
|---|---|---|---|
| **Stay (B), lower phi_col_ref** | phi_col_ref → 0.044 (Imamura wet-tissue derived) | Self-consistent with rho=25; preserves Wolf/Rubiano gel-data interpretation | Breaks the d_pore_ref / E_ref calibration which assumed phi_col_ref ≈ 0.12 → both EC50s shift; need to re-tune E_50_Texh, E_50_fib, d_crit_T, d_crit_APC |
| **Switch to (C), lower rho_collagen** | rho_collagen → ~9 mg/mL (Imamura wet/0.12) | phi_col_ref=0.12 stays; matches histological intuition | Wolf 2013 in vitro gels at 15 mg/mL would give phi=1.67 (nonphysical); breaks both Wolf and Rubiano submodel targets entirely (`phi_at_15 = 15/9 > 1`) |
| **Re-interpret as "scaffold-volume in V_T including void"** (current implicit) | Document that (B) measures "scaffold occupied volume," not literal fiber, so 0.044 (mass-derived) ≠ 0.12 (histological) is a feature, not a bug | No parameter changes; just doc | Doesn't resolve the Imamura-wet-tissue mass arithmetic; downstream Hill EC50s depend on which "phi" is being thresholded |

**The honest read:** the model phi is none of (A), (B), (C) cleanly — it's an emergent quantity controlled by the rate balance between secretion (TGFb-driven) and degradation (constitutive + MMP-M2). The attractor at 0.58 (per `collagen_attractor_overshoot.md`) is an empirical fact about the rate balance, not a literature value. Whether 0.58 is physiologically realistic depends on the convention chosen.

## 3. Joint constraint surface from literature

All values are for resectable PDAC, treatment-naive, primary human tissue unless noted.

| Quantity | Literature range | Anchor(s) | Source(s) |
|---|---|---|---|
| Cancer-cell area fraction (cellularity) | 5–30% (median ~15%) | Ve_T (via stromal_fraction observable) | Raphael 2017 PMID 28810144 (TCGA PAAD); already in `reference_values.yaml/pdac_cellularity_fraction` |
| Stromal area fraction (1 − cellularity) | 70–95% (median 66.1% in Bengtsson) | Ve_T + rho_collagen + cell volumes | Bengtsson 2026 DOI 10.1016/j.sopen.2026.01.002 (cal target); Erkan 2012 PMID 22710569 |
| Hydroxyproline-derived collagen (PDAC tissue) | 460 ± 108 µg/g wet → ~3.4 mg/mL wet | rho_collagen | Whatling-cohort PDAC papers (e.g. PMC8215985 cite Imamura) |
| Hydroxyproline-derived collagen (normal pancreas) | ~1.1 mg/mL wet (5 µg/mg dry × 0.22 dry fraction) | rho_collagen | Imamura 1995 PMID 8532652; ALREADY in `reference_values.yaml/normal_pancreas_collagen` |
| Picrosirius-red collagen area (normal pancreas) | 13.2 ± 4.9% | phi_col_ref (histological convention) | Whatling/Tian-cohort PDAC papers |
| Picrosirius-red collagen area (PDAC) | 19.7 ± 6.5% | phi_collagen target attractor | Whatling/Tian-cohort PDAC papers |
| Steady-state modulus normal pancreas | 1.06 ± 0.25 kPa (n=8) | E_ref | Rubiano 2018 DOI 10.1016/j.actbio.2017.11.037 (already in submodel target) |
| Steady-state modulus PDAC | 5.46 ± 3.18 kPa (n~10) | n_stiff (paired w/ E_ref) | Rubiano 2018 |
| AFM Young's modulus PDAC core | 4–40+ kPa | n_stiff (sensitive to indentation method) | Rubiano 2021 PMC8234178; multiple |
| In vivo pore size fibrotic PDAC | 1–3 µm (model uses 1.6 µm at attractor) | d_pore_ref | Wolf 2013 DOI 10.1083/jcb.201210152 (already submodel target) |
| Ve_T literature range | 0.017–0.529 PDAC (per `cross_model_parameters.csv`); 0.37 TNBC/CRC/UM, 0.75 NSCLC, 1.0 HCC | Ve_T | Zhang 2010 PMID 20713014; Finley 2012 PMID 22547351 |

**Two key observations:**

1. The Apte "0.10–0.15 normal pancreas" claim cited as the anchor for `phi_col_ref = 0.12` is in fact better attributed to Whatling/Imamura picrosirius-red histology (13.2 ± 4.9%), not the Apte 2004 review. The Apte 2004 paper does morphometric correlation between PSC activation and collagen area but does not report the bare normal-pancreas fraction.
2. Whatling fibrotic-PDAC picrosirius area (19.7 ± 6.5%) is **far below** the model's current attractor of phi=0.58. Even under interpretation (C) (histological), the model overshoots by 3×.

## 4. What the existing diagnostics tell us about which anchor set to choose

Per `untreated_tension_analysis_2026-04-28.md`:

- Stromal_fraction is a tension hub: at the model's current 94% stroma vs. 67% observed, sims that match Bengtsson stromal_fraction systematically miss VEGF (89% co-reach deficit), apCAF, myCAF, iCAF, MDSC, cDC2.
- The structural diagnosis identified the V_T volume rule's lack of packing constraint as the dominant cause — `feature/option-a-collagen-damper-fix` adds (1−phi)² collagen secretion damping (commit 9d22f73) and bare-volume normalization (commit 0c7f799).

Per `collagen_attractor_overshoot.md`:

- At prior medians, phi_collagen converges to 0.58. Under the Bachem-1992-recommended secretion bump (10× k_col_myCAF_sec), it would overshoot to >0.8 unless degradation also rises.
- phi_cross_50 σ was widened (1.2 → 2.0) to give SBI room to navigate.

**Implication for this audit:** the model's current anchor triple was tuned (consciously or not) so that the *attractor* phi_collagen lands somewhere internally consistent with the cited downstream EC50s — not so that any one anchor matches a clean literature value. Re-anchoring requires running the volume-rule patch first and re-measuring where phi_collagen converges.

## 5. Recommended actions

### 5a. This PR (immediate, low-risk, no SBI dependency)

1. **Fix the verified PMID error.** `parameters/parameters_PDAC.m:1888-1891` cites Apte 2004 PMID 15247571 for `phi_col_ref = 0.12`. PMID 15247571 is unrelated (intervertebral disc). The intended paper is **PMID 15367883**, which does not actually report a quantitative phi value — so the citation should be re-attributed:
   - Move the quantitative anchor citation to **Imamura 1995 PMID 8532652** (already in `reference_values.yaml/normal_pancreas_collagen`)
   - Document that 0.12 is a *histological convention* derived from picrosirius-red area in normal pancreas
   - Optionally cite Whatling/Imamura picrosirius-red area (13.2 ± 4.9%) as the more direct source

2. **Document the convention mismatch in the structural-anchor comment block** in `parameters_PDAC.m`. Add a note that `phi_col_ref = 0.12` uses convention (C) (histological/picrosirius area) while `rho_collagen = 25 mg/mL` uses convention (B) (fibrillar scaffold equivalent). The two are reconcilable only because the current Hill EC50s (`phi_col_50_Kmax = 2.0`, `d_pore_ref = 3.5 µm`, `E_ref = 1.0 kPa`) were tuned against this internally-inconsistent baseline.

3. **Merge `feature/option-a-collagen-damper-fix` to main** (after this PR's audit changes are reviewed). The structural-anchor cleanup commit (0f5f292) and damper fix (9d22f73) are independently sound and unblock the Stage 2 sprint.

### 5b. Next session (after merge, before re-anchoring)

4. **Re-run the qsp_sim attractor trace** at prior medians on the merged main with (1−phi)² damper. Per `collagen_attractor_overshoot.md` reproducibility recipe. Empirical signal: where does phi_collagen settle on the patched volume rule? If it lands near 0.30–0.50, anchor (B) is internally consistent; if it stays > 0.55, the rate equation needs further work, separate from anchor choice.

5. **Run a fresh SBI cycle** on baseline_no_treatment + clinical_progression with the patched code. Re-check joint reachability + pairwise tension diagnostics. If stromal_fraction tension hub resolves toward zero, the structural fix was sufficient and re-anchoring can be deferred indefinitely. If tensions persist on phi-derived observables, run the re-anchoring sprint.

### 5c. Re-anchoring sprint (deferred — only if 5b shows persistent tensions)

6. **Maple submodel target for rho_collagen** via the `qsp-calibration-extractor` agent. Spec in §6 below. Routes to `calibration_targets/submodel_targets/`.

7. **Joint re-tune of downstream Hill EC50s** under the chosen anchor convention. Specifically:
   - `phi_col_50_Kmax` (currently 2.0; set to be intentionally weak)
   - `d_pore_ref` (currently 3.5 µm; set against in vitro Wolf gels at convention B)
   - `E_ref` (currently 1.0 dimensionless = 1 kPa; set at phi_col_ref by Rubiano normal pancreas)
   - `n_stiff` (currently 1.8; set against collagen power-law literature)
   - `(1−phi_collagen)^2` damper exponent in `fibroblast_module_enhanced.m` (currently 2; may want different power for new anchor regime)

8. **Update unit-conversion inputs** in the existing submodel targets that bake in `rho_collagen = 25 mg/mL`:
   - `calibration_targets/submodel_targets/d_pore_ref_phi_col_ref_PDAC_deriv001.yaml` (Wolf 2013) — `rho_collagen_mgml` input + 4× `phi_at_X` inputs
   - `calibration_targets/submodel_targets/stiffness_collagen_power_law_PDAC_deriv001.yaml` (Rubiano 2018) — `rho_collagen` input + `phi_gel` input
   - `calibration_targets/submodel_targets/E_ref_PDAC_deriv001.yaml` (Rubiano 2018) — uses `phi_col_ref` implicitly; no rho_collagen input but the identity assumption (phi = phi_col_ref at normal) needs to stay self-consistent

9. **PRCC sensitivity sweep** over the discrete anchor sets (e.g. 4–6 candidate triples spanning convention-B and convention-C points). Verify downstream observable predictions are robust within the chosen literature uncertainty.

10. **Update qualitative bounds** in `tests/qualitative_bounds.yaml` if any phi-derived bounds shift their literature thresholds (phi_collagen, stiffness, d_pore, stromal_fraction-via-V_T-rule). Currently no bounds reference these directly per `tests/scenarios/test_qualitative_behaviors.py` and `tests/qualitative_bounds.py`, so likely no-op — but verify before closing.

## 6. Submodel target spec for `rho_collagen` (deferred extraction)

When the time comes to constrain `rho_collagen` directly, the cleanest empirical path is:

**Target ID:** `rho_collagen_imamura1995_PDAC_deriv001`

**Forward model:** algebraic identity. Given paired hydroxyproline-derived collagen mass density `c_mass` (mg/mL wet tissue, Imamura 1995) and picrosirius-red collagen area fraction `c_area` (Whatling/Tian/Imamura cohort, sample size n), `rho_collagen = c_mass / c_area`. This is the "histological-area-to-mass-density conversion factor" — exactly what the current model parameter is supposed to mean if conventions B and C are unified.

**Inputs (extract via maple):**

- `c_mass_normal_mean` = 1.1 mg/mL (Imamura 1995 normal pancreas wet-tissue derived)
- `c_mass_normal_sd` = ... (Imamura 1995 reported, n=?)
- `c_mass_pdac_mean` = 3.4 mg/mL (Imamura 1995 PDAC wet, ~3× normal)
- `c_mass_pdac_sd` = ... (Imamura 1995, n=?)
- `c_area_normal_mean` = 0.132 (Whatling/Tian-cohort picrosirius normal)
- `c_area_normal_sd` = 0.049
- `c_area_pdac_mean` = 0.197 (Whatling/Tian-cohort picrosirius PDAC)
- `c_area_pdac_sd` = 0.065

**Identifiability:** 1 parameter (`rho_collagen`) with 2 paired (mass, area) data points (normal + PDAC). Robust to between-cohort variability if the same indication's mass and area are paired within a single study; less so if cross-cohort pairing is required (would need a wider error model).

**Note:** This is an interim approximation. The "true" rho_collagen (the volume of V_T occupied by scaffold-equivalent collagen) is mathematically not the same as the mass-to-area ratio because picrosirius-red overstates fiber area for thin fibers and understates for tightly packed bundles. But it's the cleanest available empirical anchor pending direct in vivo SEM measurements.

**Owner:** queue with `qsp-calibration-extractor` agent when sprint priority rises. Reference this audit note + the deferred-cytokine-targets log structure (`notes/calibration/pdac_cytokine_chemokine_targets.csv`) for similar deferred-extraction queueing patterns.

## 7. Reproducibility

For the qsp_sim attractor trace + diagnostics referenced above:

```bash
cd /Users/joeleliason/Projects/pdac-build && \
.venv/bin/python -m workflows.sbi_runner \
    --scenarios baseline_no_treatment,clinical_progression \
    --num-total-simulations 5000
# Diagnostics: figures/sbi_multi/diagnostics/{prior,posterior}_{joint_reachability,pairwise_tension}.csv
```

Single-sim trace at prior-median XML:

```bash
./cpp/sim/build/qsp_sim \
  --param resources/cpp/param_all.xml \
  --evolve-to-diagnosis resources/cpp/healthy_state.yaml \
  --scenario scenarios/baseline_no_treatment.yaml \
  --drug-metadata resources/cpp/drug_metadata.yaml \
  --csv-out /tmp/dc_diag/species.csv \
  --t-end-days 1.0 --dt-days 0.5
```
Inspect `phi_collagen`, `H_pore_APC`, `H_TGFb_APC`, `f_nascent`, `stromal_fraction` at t=0 (post-evolve).

## 8. Cross-references

- GitHub issue #93 (this audit's home)
- `notes/calibration/collagen_attractor_overshoot.md` — phi=0.58 attractor diagnosis at current anchors
- `notes/calibration/untreated_tension_analysis_2026-04-28.md` — stromal_fraction tension hub structural diagnosis
- `parameters/parameters_PDAC.m:68-87, 1855-1958` — current anchor definitions + structural-anchor doc blocks (post-0f5f292)
- `parameters/cross_model_parameters.csv` — Ve_T literature range table
- `calibration_targets/reference_values.yaml` — `normal_pancreas_collagen`, `pdac_cellularity_fraction`, `pdac_stromal_fraction`, `ecm_tissue_density`, `pancreas_dry_weight_fraction`
- `calibration_targets/submodel_targets/{d_pore_ref_phi_col_ref,stiffness_collagen_power_law,E_ref}_PDAC_deriv001.yaml` — submodel targets that bake in `rho_collagen = 25 mg/mL`
- `feedback_no_fabricated_citations`, `feedback_no_prior_tuning`, `feedback_trace_before_proposing_fix` — methodology constraints applied here
