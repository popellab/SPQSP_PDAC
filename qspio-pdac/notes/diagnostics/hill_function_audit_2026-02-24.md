# Hill Function Sensitivity Audit

**Date:** 2026-02-24
**Context:** Automated audit of Hill function operating points at baseline (diagnosis) using median parameters. Values computed from `test_qualitative_behaviors_helper.m` Tier 1 baseline simulation.

---

## Framework

Not all Hill functions should be centered at their EC50. The right operating point depends on the biological role:

1. **Graded modulators** — rheostats that should be in the sensitive range [0.05, 0.95] so that perturbations produce proportional output changes
2. **Biological switches** — pathways that are genuinely all-or-nothing in the disease context (e.g., HIF in hypoxic PDAC)
3. **Treatment reserves** — dormant at baseline, activated by therapy (e.g., CXCL9 recruitment in cold tumors)

**Red flags:**
- Graded modulator at H < 0.05 or H > 0.95 → EC50 in the wrong decade, parameter insensitive
- Biological switch at the wrong extreme → model contradicts established biology
- Treatment reserve already saturated at baseline → no headroom for therapy response

**N-dependent sensitive range:** The sensitive range for a Hill function with coefficient n is defined as ligand within 3x of EC50: `[1/(1+3^n), 1-1/(1+3^n)]`. For n=1: [0.250, 0.750]. For n=2: [0.100, 0.900]. This replaces the earlier arbitrary [0.05, 0.95] threshold, which was not n-aware and was too permissive for n=1 Hill functions. Several values previously classified as "borderline" (H_CCL5_Treg at 0.113, H_lac_Teff at 0.107) are now correctly flagged as outside the sensitive range.

**Analytical IQR from EC50 priors:** For each Hill function with a mapped EC50 parameter, the partial IQR across the EC50's lognormal prior (Q25 to Q75 of EC50) is computed analytically. This identifies Hill functions that remain insensitive even across the prior range (IQR < 0.1) without requiring simulation. Limitation: this is the partial IQR from EC50 variation only; the full IQR (varying all parameters simultaneously, including ligand production rates) requires simulation.

**Framework limitations:**
- **Median-only evaluation.** This audit evaluates each Hill function at median parameters. For SBI, what matters is sensitivity *across the prior distribution*. The analytical IQR from EC50 priors partially addresses this, but only captures EC50 variation — not ligand production/degradation changes.
- **Context-dependent categories.** Some Hill functions should change category between scenarios. H_IFNg_CXCL9 is graded at baseline but might saturate under checkpoint therapy (more IFNg). H_IL12_NK is graded at baseline but could become a treatment-reserve framing. This audit is baseline-only, which is correct for the current SBI scenario but worth noting. Treatment Hill function data collection infrastructure has been added (Tier 6).
- **Classification is a biological claim.** Categorizing a Hill function as "switch" vs "graded" should be justified by independent biology, not by the model output. Some classifications below are strong (justified by known biology); others are weaker (post-hoc rationalization of the model's current behavior).

---

## Full Audit Table

| Hill Function | Value | Category | Status | Axis |
|---|---|---|---|---|
| H_TGFb | 0.755 | graded | ok | TGFb → myCAF/Treg/M2 differentiation |
| H_TGFb_Teff | 0.606 | graded | ok | TGFb → Teff inhibition |
| H_TGFb_APC | 0.755 | graded | ok | TGFb → APC maturation suppression |
| H_TGFb_NK | 0.683 | graded | ok | TGFb → NK suppression |
| H_IL10 | 0.399 | graded | ok | IL-10 → M2/immune suppression |
| H_IL10_NK | 0.001 | graded | **INSENSITIVE** | IL-10 → NK suppression |
| H_IL12 | 0.975 | graded | **SATURATED** | IL-12 → M1 activation |
| H_IL12_NK | 0.073 | graded | ok | IL-12 → NK activation |
| H_IL6_iCAF | 0.252 | graded | ok | IL-6 → iCAF autocrine feedback |
| H_IL6_M2 | 0.092 | graded | ok (borderline) | IL-6 → M2 via STAT3 |
| H_IL6_MDSC | 0.822 | graded | **SATURATED** | IL-6 → MDSC recruitment |
| H_IFNg_CXCL9 | 0.303 | graded | ok | IFNg → CXCL9 production |
| H_CCL5_Treg | 0.113 | graded | ok (borderline) | CCL5 → Treg recruitment via CCR5 |
| H_CXCL12_Texcl | 0.433 | graded | ok | CXCL12 → T cell exclusion |
| H_CXCL9_Trec | 0.005 | switch_low | ok | CXCL9 → T cell/NK recruitment (treatment-reserve) |
| H_XCL1_cDC1 | 0.001 | switch_low | ok | XCL1 → cDC1 recruitment (treatment-reserve) |
| H_glc_Teff | 0.856 | graded | ok | Glucose → T cell function (positive) |
| H_glc_C | 0.796 | graded | ok | Glucose → cancer proliferation |
| H_lac_Teff | 0.107 | graded | ok (borderline) | Lactate → T cell suppression |
| H_lac_M2 | 0.136 | graded | ok (borderline) | Lactate → M2 polarization |
| H_ECM_APC_mig | 0.823 | graded | ok | ECM → APC migration inhibition |
| H_ECM_T_mot | 0.903 | graded | ok (borderline) | ECM → T cell motility inhibition |
| H_ECM_T_exh | 0.736 | graded | ok | ECM → T cell exhaustion |
| H_NO | 0.767 | graded | ok | NO → T cell dysfunction |
| H_ArgI_Teff | 0.499 | graded | ok | Arginase I → T cell suppression |
| HIF | 0.092 | switch_high | **UNEXPECTED** | Hypoxia response (PDAC median pO2 ~2.7 mmHg) |
| H_DAMP | 0.991 | switch_high | ok (scrutinize) | DAMP → APC maturation |
| H_P1_apCAF | 0.974 | switch_high | ok | Antigen → apCAF Treg induction (EC50=1pM) |

---

## Correctly in the sensitive range (high confidence)

**H_TGFb = 0.755** — TGFb at 0.22 nM vs EC50 0.07 nM. Active but not saturated. Correct — TGFb is the master immunosuppressive signal in PDAC but varies across patients (Ogawa stroma types).

**H_TGFb_Teff = 0.606** — Higher EC50 (0.14 nM) for Teff inhibition. Good — requires more TGFb to suppress effectors than to drive Treg differentiation.

**H_TGFb_APC = 0.755** — Same EC50 as H_TGFb. Consistent — APC maturation suppression tracks with Treg induction.

**H_TGFb_NK = 0.683** — EC50 0.1 nM. NK suppression is intermediate between macrophage and Teff sensitivity. Reasonable.

**H_IL10 = 0.399** — IL-10 ~0.005 nM vs EC50 0.008 nM. Near midpoint — maximally responsive. This is the main IL-10 immunosuppression axis and should be tunable.

**H_IFNg_CXCL9 = 0.303** — IFNg ~0.022 nM vs EC50 0.05 nM. Partially active — CXCL9 production responds to IFNg changes. Treatment (more IFNg) would meaningfully boost this.

**H_CXCL12_Texcl = 0.433** — CXCL12 ~1.5 nM vs EC50 2.0 nM. Near midpoint. The immune exclusion axis is moderately active and responsive — critical for differentiating hot vs cold tumors across patients.

**H_ArgI_Teff = 0.499** — Essentially at the midpoint. Maximally sensitive to arginase changes. Ideal for calibration.

**H_IL6_iCAF = 0.252** — IL-6 ~0.1 nM vs EC50 0.3 nM. The autocrine iCAF feedback is partially active — enough to maintain some iCAFs but not dominant. Boosting IL-6 (e.g., via more iCAFs) would substantially increase the feedback strength.

**H_glc_Teff = 0.856, H_glc_C = 0.796** — Glucose ~907 µM, T cells and cancer cells still mostly glucose-replete. Correct *given current glucose levels*, but would decrease if HIF were properly calibrated (more glycolysis → more glucose depletion).

**H_NO = 0.767** — NO-mediated T cell dysfunction is active but not saturated.

**H_ECM_T_exh = 0.736** — ECM-driven T cell exhaustion is active. *Caveat:* the model overproduces ECM (230x vs observed 10x), so this would drop substantially if ECM is corrected.

**H_IL12_NK = 0.073** — IL-12 ~4 pM vs NK EC50 50 pM. NK cells get only a modest IL-12 boost at baseline. Correct — cDC1-derived IL-12 is limited in untreated PDAC. Room to increase under therapy. *Note:* The boundary between "graded at low end" and "treatment-reserve" is fuzzy here. cDC1-derived IL-12 would increase substantially under therapy (via NK→XCL1→cDC1 loop activation). Could reasonably be classified as treatment-reserve with a non-negligible baseline value.

---

## Classified as switches (varying confidence)

**H_P1_apCAF = 0.974** — EC50 = 1 pM. Any tumor with antigen saturates this. Correct — apCAF Treg induction is limited by apCAF number, not antigen availability. **Strong justification** — antigen availability is genuinely not the bottleneck.

**H_DAMP = 0.991** — DAMPs from tumor killing are saturating for APC maturation. The justification assumes that in an established tumor, antigen release (`R_Tcell`) is always high. **Moderate justification** — this depends on whether `R_Tcell` at diagnosis is dominated by immune killing or by baseline tumor turnover (intrinsic apoptosis/necrosis). In cold PDAC with excluded CD8s and suppressed NK cells, how much immune killing is actually happening? If the death rate is mostly non-immune, then DAMP saturation is reasonable. But if immune killing is low, H_DAMP might not genuinely be saturated. Worth checking whether `R_Tcell` at baseline comes primarily from `k_C_Tcell_eff` terms or from intrinsic cancer cell turnover.

**H_IL6_MDSC = 0.822** — IL-6 ~0.1 nM vs EC50 0.02 nM. MDSC recruitment is essentially always-on if any IL-6 exists. **Weak justification** — the SA showed MDSC-related parameters are significant, meaning MDSC levels do vary across patients. If IL-6 varies from 0.05-2 nM across the prior, an EC50 of 0.02 nM means H_IL6_MDSC ranges from 0.71-0.99 — barely sensitive. Making IL-6→MDSC always-on means `k_MDSC_rec` absorbs all MDSC variation, hiding what should be a graded IL-6 effect. An EC50 of ~0.5 nM would let IL-6 meaningfully differentiate patients. This may be another case of an EC50 in the wrong decade, similar to IL12_50. Should be reclassified as graded if the EC50 provenance doesn't hold up.

---

## Correctly classified treatment-reserves

**H_CXCL9_Trec = 0.005** — CXCL9 ~0.005 nM vs EC50 1.0 nM. Dormant at baseline. Correct for untreated "cold" PDAC — CXCR3-mediated T cell recruitment activates when IFNg increases (immunotherapy → more IFNg → more CXCL9 → T cell recruitment). The IFNg perturbation test confirms the pathway works mechanistically.

**H_XCL1_cDC1 = 0.001** — XCL1 negligible at baseline. Correct — NK-derived XCL1 is low because NK cells are few and suppressed. This is the NK→XCL1→cDC1→IL-12→NK amplification loop; it needs a kick-start (e.g., IL-12 from treatment) to activate.

---

## Genuine concerns

### 1. H_IL10_NK = 0.001 — EC50 likely too high

IL-10 baseline ~0.005 nM. Macrophage EC50 = 0.008 nM (H_IL10 = 0.40). NK EC50 = 5.0 nM (H_IL10_NK = 0.001). That's a **625x difference in sensitivity**.

NK cells express IL-10R and are functionally suppressed by IL-10 in vitro at 1-10 ng/mL (0.05-0.5 nM for IL-10 MW ~18.6 kDa). An EC50 of 5 nM means half-maximal NK suppression requires ~90 ng/mL, which is supraphysiological. This parameter was likely inherited from a different modeling context or is a typo (5.0 should be 0.05?). The consequence: the model says NK cells are completely immune to IL-10 in the TME, which contradicts evidence of IL-10-mediated NK dysfunction in cancer.

**Recommendation:** Reduce `IL10_50_NK` to 0.05–0.5 nM range, or investigate provenance.

### 2. H_IL12 = 0.975 — broken graded modulator, not a genuine switch

The EC50 is 0.14 *picomolar*, meaning any measurable IL-12 saturates the M1 activation response. This makes IL-12 a meaningless modulator of M1 polarization — the rate constant `k_M1_pol` absorbs what should be a graded IL-12 effect.

Contrast with H_IL12_NK = 0.073 (EC50 = 50 pM), where IL-12 IS a meaningful modulator of NK activation. The 357x difference in EC50 between macrophage and NK IL-12 responsiveness is suspicious.

**Consequence:** `IL12_50` is not identifiable in SBI — changing it doesn't change model output. If IL-12 should modulate M1 polarization (which it should biologically), the EC50 needs to increase ~100x to the 10-50 pM range.

**Reclassified from switch_high to graded/INSENSITIVE.** The original switch_high classification was a post-hoc rationalization to avoid false test failures, not a biological claim. IL-12 genuinely modulates M1 polarization in a graded fashion — the saturated response is an artifact of the 0.14 pM EC50, not a biological switch. Treating it as a switch masks a real parameter problem.

### 3. HIF = 0.092 — model pO2 too high (known calibration gap)

The model produces pO2 ~19 mmHg at diagnosis. Clinical PDAC: median ~2.7 mmHg (Koong 2000). With EC50 = 7 mmHg and n=2: HIF = 1/(1+(19/7)²) = 0.09. At clinical pO2 of 2.7: HIF = 1/(1+(2.7/7)²) = 0.87.

**Downstream consequences of low HIF:**
- Lactate production underestimated (HIF amplifies glycolysis): model ~2.4 mM vs expected 5-15 mM
- H_lac_Teff = 0.107 and H_lac_M2 = 0.136 too low — metabolic immunosuppression underrepresented
- HIF-driven VEGF lower → less vascular dysfunction than expected
- The `*(1-HIF)` M2 polarization term barely contributes

**Root cause:** Vascular parameters or oxygen consumption rates need tuning so tumor pO2 drops to the 2-10 mmHg range at diagnosis.

---

## Borderline concerns (not failing, but worth watching)

*Note:* For n=1 Hill functions, the high-sensitivity range is roughly [0.27, 0.73]. Values like 0.113 and 0.092 are genuinely in the insensitive tail, not merely borderline. These are called "borderline" because the underlying ligand concentrations are plausible — the issue is mild EC50 mismatch, not an order-of-magnitude error like IL10_50_NK.

**H_CCL5_Treg = 0.113** — CCL5 Treg recruitment is weak. Mitigated by the new `k_Treg_pro_tumor` (local proliferation sustains Tregs), but means the CCL5/CCR5 pathway barely matters for Treg dynamics. Changing CCL5 production won't significantly affect Treg numbers.

**H_IL6_M2 = 0.092** — IL-6→M2 polarization is almost inert. OK if IL-6's M2 role is secondary to TGFb/IL-10 (which it is), but the pathway is essentially dead at baseline.

**H_ECM_T_mot = 0.903** — Close to saturation. ECM-driven T cell immobility is near-maximal. Consequence of ECM overproduction (230x vs observed 10x). Once ECM is calibrated down (via MMP degradation or parameter changes), this should drop to ~0.3-0.5 and become more sensitive.

**H_lac_Teff = 0.107, H_lac_M2 = 0.136** — Lactate suppression barely active. Downstream consequence of low HIF → low lactate. Would increase to 0.2-0.4 with correct HIF calibration.

---

## Cascade dependencies

Several "ok" Hill functions are only ok because upstream problems compensate:

```
HIF too low (0.09)
  → lactate production underestimated
    → H_lac_Teff (0.107) and H_lac_M2 (0.136) too low
    → metabolic immunosuppression underrepresented
  → VEGF underproduced
    → vascular quality too high
    → glucose not depleted enough
      → H_glc_Teff (0.856) higher than it should be

ECM overproduced (230x vs 10x)
  → H_ECM_T_mot (0.903) near-saturated
  → H_ECM_APC_mig (0.823) near-saturated
  → H_ECM_T_exh (0.736) elevated
  → all three would become more sensitive at correct ECM levels
```

Fixing HIF calibration would cascade through the metabolic branch. Fixing ECM (via MMP degradation or parameter changes) would cascade through the stromal branch. Both sets of Hill functions would shift toward more responsive operating points.

---

## Action items

| Priority | Issue | Fix |
|---|---|---|
| 1 | `IL10_50_NK` = 5 nM (625x too high) | Reduce to 0.05-0.5 nM; verify provenance |
| 2 | HIF = 0.09 (pO2 ~19 mmHg vs clinical ~2.7) | Tune vascular/O2 params for pO2 2-10 mmHg |
| 3 | `IL12_50` = 0.14 pM (IL-12 always saturated) | Increase to 10-50 pM for graded M1 modulation |
| 4 | `IL6_50_MDSC` = 0.02 nM (switch classification weak) | Verify EC50 provenance; consider increase to ~0.5 nM for graded modulation |
| 5 | H_DAMP = 0.991 (switch assumption unverified) | Check whether `R_Tcell` at diagnosis is immune killing or intrinsic turnover |
| — | H_ECM_* near saturation | Resolves after ECM calibration (structural review items) |
| — | H_lac_* barely active | Resolves after HIF calibration (item 2) |

## Future improvements to this audit

- ~~Compute Hill function IQR across prior samples (not just median) to assess sensitivity across the parameter space SBI actually explores~~ **Done** — analytical partial IQR from EC50 prior variation added to test suite (Tier 7). Full IQR (varying all parameters) still requires simulation.
- ~~Re-run audit under treatment scenarios (GVAX+nivo) to check which treatment-reserves activate and which graded modulators saturate~~ **Partially done** — data collection infrastructure added (MATLAB helper outputs treatment Hill values at d0 and d21; Tier 6 tests verify treatment-reserve activation). Full treatment audit pending next MATLAB run.
- ~~Formalize the n-dependent sensitive range: for Hill coefficient n, the high-sensitivity band is approximately [1/(1+3^n), 1/(1+3^{-n})]~~ **Done** — moved to Framework section above; implemented in test suite with per-entry `n` field.
