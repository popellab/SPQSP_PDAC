# Tumor cDC1/cDC2 dynamics: homeostatic vs recruitment-rate-limited

**Date:** 2026-05-02
**Trigger:** SBI 10k baseline+clinical_progression run flagged `cdc1_density_baseline`
as MARGINAL with `k_PSC_to_apCAF` as named top driver. Local `qsp_sim` traces showed
the parameter has 2% leverage and the realized cdc1 density (0.043 cells/mm²) is
~180× below Plesca 2022's mIHC target (7.82 cells/mm²) at default `APC0_cDC1_T`.
The V_T cDC1 equation in `core/modules/APC_module.m` is recruit + death, while
the V_LN equation uses clean homeostatic relaxation — a structural asymmetry that
prompted this literature review.

## Question

Should intratumoral immature cDC1 / cDC2 dynamics be modeled as **homeostatic
density relaxation** (DCs equilibrate to a fixed steady-state density set by
`APC0_cDC1_T`) or **recruitment-rate-limited** (continuous influx from circulation
balanced by death/turnover, with steady-state controlled by the rate ratio)?

## Conclusion: recruitment + death is biologically correct. Do not reformulate to homeostatic.

The current V_T equation is right. The mismatch is in how the parameter is *labelled*
and *prior-anchored*, not in the equation form. PDAC's central DC phenotype is
*recruitment failure with elevated tumor apoptosis* — a homeostatic relaxation form
would elide exactly the biology the model needs to capture.

## Evidence summary

### 1. Tumor cDC1/cDC2 are continuously recruited, not locally maintained

- **Liu et al. 2007** (*Nat Immunol*, doi:10.1038/ni1462) and **Liu et al. 2009**
  (*Science*, doi:10.1126/science.1170540): parabiosis + pre-cDC tracking — cDC pools
  in lymphoid organs equilibrate between parabionts on weekly timescales via blood-borne
  pre-cDC intermediate. Steady-state turnover ~4,300 cells/h per lymphoid organ — flux-driven
  steady state, not self-renewal.
- **Salmon et al. 2016** (*Immunity*, doi:10.1016/j.immuni.2016.03.012) and **Böttcher
  et al. 2018** (*Cell*, doi:10.1016/j.cell.2018.01.004): intratumoral cDC1 abundance
  set by chemokine-driven recruitment from blood (NK-derived CCL5/XCL1, FLT3L availability).
  cDC1 density can be increased several-fold by exogenous FLT3L/chemokine — incompatible
  with a fixed homeostatic set-point.
- **Broz et al. 2014** (*Cancer Cell*, doi:10.1016/j.ccell.2014.09.007): tumor cDC1
  are FLT3L/GM-CSF-dependent migratory cells, not a locally proliferating compartment.

### 2. Intratumoral cDC turnover is fast (days), and tumor-specific apoptosis shortens it further

- **Kamath et al. 2002** (*Blood*, doi:10.1182/blood.V100.5.1734...): continuous BrdU
  labelling places splenic CD8α+ (cDC1-equivalent) at ~3-day half-life and CD11b+ cDC2
  at ~2 days.
- **Cytlak/Bigley et al. 2024** (*JEM*, doi:10.1084/jem.20220867): human in vivo deuterium
  labelling corroborates short cDC half-lives (a few days), longer for pDC.
- **Hegde et al. 2020** (*Cancer Cell*, doi:10.1016/j.ccell.2020.02.008) and **Lin
  et al. 2020** (*Cancer Discov*, doi:10.1158/2159-8290.CD-19-0966): tumor cDC1s in
  KPC PDAC show *increased apoptosis* driven by IL-6 dysregulation, FLT3L sequestration,
  and tumor-derived factors. Tumor death rate exceeds peripheral steady-state.

### 3. Density tracks recruitment, doesn't homeostatically self-restore

- FLT3L administration → sustained, dose-dependent increase in tumor cDC1 (Hegde 2020,
  Salmon 2016, intratumoral FLT3L trials).
- NK depletion / CCR5 / XCR1 blockade → sustained decrease in tumor cDC1 (Böttcher 2018).
- KPC PDAC: cDC1 density progressively *declines* through PanIN → invasive disease
  (Lin 2020) — tracks changing kinetics, not a defended attractor.

### 4. PDAC-specific scarcity is mechanistically a recruitment/death imbalance

- **Hegde 2020**: cDC1 in PDAC is ~79-fold lower than lung adenocarcinoma at matched
  stage and ~100-fold lower than tumor macrophages. Driven by (i) reduced FLT3L bioavailability,
  (ii) IL-6-driven cDC1 apoptosis, (iii) impaired chemokine signalling.
- **Lin 2020**: deficit originates in dysregulated bone-marrow → blood pre-cDC1 output.

Both lines are intrinsically recruit + death formulations — no biological basis for
a homeostatic restoring force in PDAC.

## Why the V_T vs V_LN asymmetry is appropriate

- **V_LN**: pre-cDC-derived steady-state pool with relatively long residence and
  weak local death drivers; a homeostatic abstraction is acceptable there.
- **V_T**: recruitment-and-apoptosis dominated; must be modelled as such to capture
  PDAC's recruitment-failure phenotype and predict FLT3L / anti-IL6 / chemokine
  interventions.

## Implication for `APC0_cDC1_T` / `APC0_cDC2_T`

The parameter is currently labelled "cDC1 homeostatic density in tumor" but at steady
state, realized density ≈ `APC0 × (1 − H_pore_APC) / (1 + maturation_drain / k_APC_death)`.
The 180× gap between APC0 (50,000 cell/mL) and Plesca's realized density (7.82 cells/mm²
≈ 460 cells/mm³ ≈ 460,000 cell/mL) is an artifact of treating a *recruitment target*
as a *realized density*.

### Two clean options (preferred = first)

**(A) Re-interpret `APC0_cDC1_T` as a recruitment-target density.** Effectively
`k_recruit = k_APC_death × APC0_cDC1_T` — a rate constant numerator with units of
cells/(volume × time). The literature density (Plesca, etc.) becomes a constraint
on the *realized* observable, not on the parameter directly. SBI then fits the
recruitment-rate parameter against the observed realized density.

Implementation steps:
1. Rename `APC0_cDC1_T` → `cDC1_recruit_target_T` (or similar) in `parameters_PDAC.m`,
   the priors CSV, the C++ XML template, and `APC_module.m`. Update the Notes string
   from "homeostatic density" to "recruitment-target density".
2. Widen the prior σ on this parameter to allow the realized density to span Plesca-scale
   (currently σ=0.7 → ±2σ ≈ [12k, 200k]; need σ closer to 1.5–2.0 to allow ±2σ ≈
   [2.5k, 1M+] given the multiplicative attenuation chain).
3. Same for `APC0_cDC2_T`.
4. Update the docstring in `cdc1_density_baseline_*.yaml` and any submodel target
   YAMLs that anchor against these parameters to reflect the recruit-target semantic.
5. Document in `notes/model/pdac_changelog.md`.

**(B) Treat current parameter name as misnamed but tolerable.** Don't rename, but
widen the prior σ and document in `notes/model/pdac_changelog.md` that the parameter
is a recruit-target despite its name. Lower-friction now but accumulates technical
debt.

### Do NOT do: reformulate V_T to homeostatic

This was the original instinct after the SBI trace, but the literature is clear that
it would be biologically wrong for PDAC and would lose the model's ability to predict
FLT3L / anti-IL6 / chemokine perturbations.

## Caveats

- Direct in-tumor BrdU / deuterium pulse-chase for cDC1/cDC2 is sparse; tumor-DC
  half-life is bounded by inference (≤ peripheral cDC half-life, likely shorter due
  to elevated apoptosis).
- "cDC2" in tumors is heterogeneous (cDC2 ↔ monocyte-derived DC2 continuum); current
  recruitment formulation captures bulk dynamics but doesn't resolve this. Future
  work could split the recruitment term if needed.
- Plesca 2022 and similar mIHC counts measure *realized* density. They are observables
  on `cDC1_density`, not direct priors on `APC0_cDC1_T`. **This is the root cause
  of the 180× attenuation pain.**

## Recommended next step

Pick option A or B and proceed. If A: this is a 5–10 file edit (param rename across
M-files + XML + CSV + YAML), rebuild via `make refresh-cpp`, run `make parity`, then
SBI verification at 5k. If B: just widen the σ in the priors CSV, regenerate the
submodel priors, run SBI verification.

Either way, the next iteration should also reconsider the prior on `APC0_cDC2_T`
(currently calibrated by similar tissue-density anchors) and any submodel targets
that constrain these parameters via in vitro density measurements — those need to
be re-routed through realized-density observables instead of acting as direct priors.
