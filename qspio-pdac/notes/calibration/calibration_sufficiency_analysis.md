# Calibration Sufficiency Analysis

Date: 2026-03-25

## Question

When is parameter calibration "enough" for a QSP paper? Do we have sufficient submodel targets and calibration targets?

---

## Current State (as of 2026-03-25)

### Active Calibration Targets: 83 (updated 2026-03-25)

| Scenario | Count | Type |
|----------|-------|------|
| Baseline TME (diagnosis) | 22 | Full-model (IHC densities, ratios, fractions) |
| GVAX monotherapy (day 21) | 6 | Full-model (treatment response) |
| GVAX+nivo (day 21) | 9 | Full-model (treatment response) |
| Clinical progression (90d) | 1 | Full-model (tumor doubling time) |
| Submodel targets | 45 | Isolated-system (in vitro/ex vivo + clinical IHC) |

### Parameter Coverage: 217 total params in priors

| Category | Count | % |
|----------|-------|---|
| With submodel targets | 45 | 21% |
| In hierarchical groups | 21 | 10% |
| **Total directly/indirectly constrained** | **~57** | **~26%** |
| Unconstrained (SBI with broad priors) | 160 | 74% |

### Module Coverage (updated 2026-03-25)

| Module | Params | With targets | % | Notes |
|--------|--------|-------------|---|-------|
| Fibroblast/CAF | 63 | 19 + 8 in groups | 43% | Best-covered module |
| T cell | 23 | 4 (n_CD8_clones, k_C1_growth, q_CD8_T_in, N_IL2_CD8) | 17% | Trafficking + IL-2 division now covered |
| Macrophage | 14 | 4 (k_M1_pol, k_M2_pol, k_Mac_death, k_Mac_rec) | 29% | Was 0%, now well-covered |
| MDSC | 7 | 2 (k_MDSC_rec, k_MDSC_death) | 29% | Was 0% |
| APC/DC | 13 | 2 | 15% | |
| NK cells | 6 | 0 | 0% | Low priority (small effect in PDAC) |
| GVAX | 5 | 1 | 20% | |
| Checkpoint | 4 | 0 | 0% | Treatment-mechanism; constrained by treatment-response targets |
| CXCL9/recruitment | 7 | 0 | 0% | Treatment-mechanism; IFNg-CXCL9 loop dormant at baseline, activated by immunotherapy. Constrained by GVAX+nivo treatment-response targets in SBI |

---

## Published QSP-IO Calibration Norms

### No gold standard exists

The field has no prescribed ratio of targets to parameters. Key references:

- **Azer et al. 2021** (PMC8821410): "no 'gold standard' for model development and assessment"; universal calibration recommendations are impractical
- **Chelliah et al. 2021** (PMC7983940): QSP parameters should be "based on experimental data on individual interactions, rather than statistically inferred"; focus on plausible ranges, not identifiability
- **Gadkar et al. 2016** (PMC4879472): six-stage workflow avoids numerical prescriptions; calibration "must constrain the relevant biological mechanisms sufficiently to have confidence in subsequent predictions"
- **Tanten et al. 2025** (arxiv 2505.02750): "sloppy parameters" perspective shows models with largely unconstrained parameters can still make precise predictions because predictions depend on a low-dimensional manifold in parameter space

### What published Popel lab models actually do

| Model | Total params | Params varied | Acceptance/calibration endpoints | Ratio |
|-------|-------------|---------------|----------------------------------|-------|
| Wang 2023 (NSCLC, 255 params) | 255 | 30 | 3 acceptance + ~8 categories | ~0.3:1 |
| Arulraj 2023 (TNBC, 737 params) | 737 | 127 | ~5-6 clinical/transcriptomic | ~0.04:1 |
| Milberg 2019 (melanoma) | ~300+ | ~7 | ~4-5 clinical endpoints | ~0.7:1 |
| Sove 2020 (QSP-IO toolbox) | ~280 eqs | "several hundred" | hand-tuned + LHS | N/A |

**Practical norms:**
- Total parameters: 200-750
- Parameters varied for VPop: 7-130 (10-30% of total)
- Explicit calibration endpoints: 3-10 *categories*
- Ratio of acceptance criteria to varied parameters: ~0.1:1 to 0.5:1
- Models are always heavily underdetermined
- Most parameters set from literature; only a subset varied; acceptance criteria filter plausible virtual patients

### Our model in context

With ~80 targets for 217 parameters (~0.4:1), we are at the **upper end** of what published QSP-IO models achieve. The combination of submodel targets (constraining individual parameters) + full-model calibration targets + hierarchical parameter groups is substantially more rigorous than standard practice (hand-tuning + LHS + acceptance criteria). No published Popel lab model has submodel-level parameter constraints -- that's our methodological contribution.

---

## Submodel Target Selection Rationale

Reviewers may ask: "Why did you extract submodel targets for these parameters but not others?"

Three nested selection criteria (to be stated explicitly in the paper):

### 1. Novel module, novel parameters first

The fibroblast/ECM module is a major mechanistic contribution. These parameters don't exist in the base QSP-IO framework -- no inherited defaults, no prior calibration from NSCLC/melanoma. Every fibroblast parameter was a blank slate. That's why ~24/42 targets are fibroblast/ECM. Macrophage and T cell modules inherit plausible ranges from Wang 2023, Arulraj 2023, etc.

### 2. Extractability as isolated-system targets

Not every parameter *can* be constrained by an in vitro/ex vivo experiment:
- **Extractable**: Rate constants measurable in cell culture (proliferation, death, secretion rates), EC50s from dose-response curves, physical constants (pore size, stiffness)
- **Not extractable as submodel targets**: Trafficking rates (require in vivo imaging), feedback-loop parameters (macrophage polarization depends on full cytokine milieu), emergent properties

Parameters not amenable to submodel targets are instead constrained by full-model calibration targets (cell densities, ratios, fold changes).

### 3. PRCC sensitivity screening

Parameters that don't appear in top PRCC tiers for any observable are practically non-identifiable. No value in extracting submodel targets for them.

### Suggested paper framing

> Submodel targets were prioritized for parameters meeting three criteria: (i) **novel to this model** -- parameters introduced in the PDAC-specific fibroblast/ECM module lack prior calibration from existing QSP-IO frameworks; (ii) **amenable to isolated-system calibration** -- the parameter can be linked to a measurable quantity in an in vitro/ex vivo assay describable by a low-dimensional ODE; and (iii) **sensitive** -- PRCC analysis identified the parameter as influential on at least one calibration observable. Parameters inherited from prior models, embedded in multi-parameter feedback loops, or insensitive to model outputs were constrained through full-model calibration targets or left at literature-derived prior ranges.

---

## Three-Paper Strategy

1. **MAPLE methods paper** -- extraction framework, submodel targets as a concept, Bayesian calibration pipeline. The 42+ submodel targets + 7 hierarchical groups are the proof-of-concept dataset.

2. **PDAC application paper** -- 38 full-model calibration targets across 4 scenarios, SBI with NPE, posterior predictive validation, qualitative bounds. Demonstrates the workflow on a real model.

3. **Clinical/translational PDAC paper** -- predictions, virtual trials, treatment optimization. Consumes the calibrated model.

---

## Assessment: Stop Calibrating?

**Yes, with one caveat.** The current target set is sufficient for both the methods and application papers. The main vulnerability is the macrophage module at 0% coverage. There are 27 old-schema submodel targets in `to-review/` covering 12 net-new parameters (including `k_M1_pol` x7 and `k_Mac_rec` x3). Converting these would:

- Break the "macrophage = 0%" gap (biggest reviewer concern)
- Cover Tier 1 trafficking params (`q_CD8_T_in`, `q_Treg_T_in`)
- Touch metabolism (`hypoxic_fraction`)
- Extend the CAF death rate hierarchical group

After converting to-review targets: ~69 submodel targets, ~51 params covered, broad module coverage with only principled gaps remaining (checkpoint and CXCL9 -- treatment-mechanism params best constrained by treatment-response data).

**Remaining gaps are explicitly "future work":**
- Checkpoint module (4 params) -- constrain when anti-PD-1 monotherapy calibration begins
- CXCL9/recruitment (7 params) -- constrain with treatment-response fold changes
- Cytokine concentrations (12 deferred) -- extract after sensitivity analysis
- Metabolism (9 params) -- indirect constraint via tumor growth and hypoxia targets

---

## Audit of `to-review/` Old-Schema Submodel Targets

Date: 2026-03-25

50 YAML files in `to-review/`. ~15 are duplicates of already-promoted targets. 27 files cover 12 net-new parameters. Audited for source quality, forward model correctness, and biological plausibility.

### Tier A — Definitely convert (high quality, fills critical gaps)

| Target | Source | System | Why keep |
|--------|--------|--------|----------|
| k_M1_pol deriv001 | Watkins 2007 | Mouse LLC TAMs, IL-12 in vivo | Solid: "<90 min" reprogramming, clean first-order half-life model |
| k_M1_pol deriv003 | Yates 2020 | Mouse BMDMs, RP-182 peptide | 30-min CD86 upregulation, constrains upper bound well |
| k_M1_pol deriv004 | Gu 2023 | **Human PDAC** IHC | Direct PDAC M1:M2 ratio (CD86/CD163, n=42). Constrains k_M1_pol/k_M2_pol jointly |
| k_M1_pol deriv006 | De Simoni 2023 | **Human PDAC** IHC | Direct PDAC iNOS+:CD163+ ratio ~1:18 (n=38) |
| k_Mac_rec deriv001 | Gu 2023 | **Human PDAC** IHC | CD86+ M1 density → steady-state recruitment. Exact indication match |
| k_Mac_rec deriv002 | Chen 2015 | **Human PDAC** IHC | CD68+ total TAM density, retrospective cohort |
| q_CD8_T_in deriv001 | Boissonnas 2007 + De Boer 2003 | Mouse EG7 intravital imaging | Quantitative (327±50 cells/mm2), clean steady-state derivation, 10x translation uncertainty |
| N_IL2_CD4 | Gett & Hodgkin 1998 | Mouse naive CD4+, CFSE | Classic division-tracking: 7-8 generations under IL-2 |
| N_IL2_CD8 | Castro 2012 | Mouse OT-I CD8+, IL-2 complexes | 40-70x expansion → 5-6 extra divisions. In vivo, quantitative |
| ECM_density | Kai 2023 + Antoine 2014 | Human PDAC CT + collagen review | ECV at F3 fibrosis = 42.5%, clever two-source approach |
| PDGF_50_prolif | Bachem 1993 | Rat HSCs, PDGF dose-response | 2-point dose-response (2 and 10 ng/mL). Sparse but right experiment |

**= ~11 targets, 7 net-new parameters**

### Tier B — Convert with caveats (moderate quality, useful for breadth)

| Target | Source | Caveat |
|--------|--------|--------|
| k_MDSC_rec deriv001 | Hong 2015, mouse B16/LLC | Fix Wikipedia citation for tumor cell density |
| k_MDSC_rec deriv002 | Oo 2022, OSCC Transwell | Effective volume is guessed (0.01 mL); order-of-magnitude constraint only |
| q_Treg_T_in deriv001 | Tremblay 2017, mouse HPV cancer MRI | Qualitative timing (24h appearance) → loose prior anchor |
| k_apsc_death deriv001 | Estaras, rat PSCs + melatonin | Upper-bound only (pharmacological stimulus, supraphysiological dose) |
| k_M1_pol deriv002 | Horhold 2020, mouse PECs siRNA | Non-physiological mechanism (TF knockdown, not cytokine-driven) |
| k_CCL2_sec deriv001 | Fujita 2010, human prostate stroma | BPH proxy, assumes 1 mL medium volume (not reported) |
| hypoxic_fraction | PIMO-PANC study, **human PDAC** | Excellent source but needs pO2/pO2_50/n_HIF species in model |

**= ~7 targets, 4 more parameters**

### Tier C — Skip (source too distant or weak)

| Target | Why skip |
|--------|----------|
| k_M1_pol deriv005 | Manfroi 2021 DLBCL, n=3 cases, completely different TME |
| k_M1_pol deriv007 | Herwig 2013 uveal melanoma, eye tumor, n=20 |
| k_CCL2_sec deriv002 | Steube 1999 leukemia cell lines — different biology |
| k_Mac_rec deriv003 | Chang 2021 — mislabeled, actually M2/M1 ratio (duplicate of k_M1_pol ratio targets) |

### Schema Changes Required

Old schema → current schema differences:
- `observation_code`: old uses `(inputs, sample_size, rng, n_bootstrap)` with bootstrap sampling → new uses `(inputs, sample_size)` returning `{value, sd, sd_uncertain, n}` dict
- `figure_excerpt` field on inputs: removed in current schema
- `batch_accumulation` forward model type: may need conversion to `algebraic`
- `observable` blocks inside error_model: restructured in current schema
- Missing `empirical_data` top-level field in some old targets

---

## Unconstrained EC50 / Cytokine Parameter Audit

Date: 2026-03-25

34 EC50/Hill parameters and 18 cytokine secretion/degradation rates remain unconstrained by submodel targets. Many EC50s are biophysical receptor-ligand constants with published dose-response data that has never been formally extracted through the MAPLE pipeline. The existing default values were set from reading papers (citations in parameter notes) but lack provenance-tracked submodel targets.

EC50s are ideal submodel targets: each maps to a `direct_fit` Hill curve from a single dose-response experiment. Extracting these is low-hanging fruit.

### Tier 1 — CXCL9 axis (treatment-critical, currently no citations)

These control the IFNg→CXCL9→CXCR3 positive feedback loop that drives immune "hot" tumor phenotype. The loop is dormant at baseline PDAC but activates under immunotherapy (GVAX+nivo). Currently set as educated guesses with no literature citations.

| Parameter | Current value | Units | What it controls |
|-----------|-------------|-------|------------------|
| `IFNg_50_CXCL9` | 0.05 | nM | IFNg EC50 for CXCL9/10 induction (JAK-STAT1) |
| `CXCL9_50_Trec` | 1.0 | nM | CXCL9 EC50 for CXCR3-mediated T cell/NK recruitment |

**Literature:** Mikucki 2015 (cited in chemotaxis_module.m), Chow 2019. IFNg dose-response on CXCL9 production by tumor cells/macrophages. CXCL9 chemotaxis dose-response on T cell migration.

### Tier 2 — Core immune EC50s (have citations but no formal extraction)

These have published citations in parameters_PDAC.m but the values were set by reading papers, not by fitting dose-response data. Formally extracting them would verify the defaults and provide proper uncertainty.

| Parameter | Current value | Units | Citation | What it controls |
|-----------|-------------|-------|----------|------------------|
| `IL2_50` | 0.32 | nM | Marchingo 2014 | IL-2 EC50 for T cell proliferation |
| `IL2_50_Treg` | 0.32 | nM | (same as IL2_50) | IL-2 EC50 for Treg proliferation |
| `IFNg_50` | 2.9 | pM | Altan-Bonnet 2019 | IFNg EC50 for M2→M1 polarization |
| `IL10_50` | 8.0 | pM | Altan-Bonnet 2019 | IL-10 EC50 for M1→M2 polarization |
| `IL12_50` | 15 | pM | Altan-Bonnet 2019 (modified) | IL-12 EC50 for M2→M1 / NK activation |
| `TGFb_50` | 0.07 | nM | Robertson-Tessi 2012 | Generic TGFb EC50 (multiple contexts) |
| `TGFb_50_Teff` | 0.14 | nM | (not cited) | TGFb inhibition of T cell function |
| `TGFb_50_NK` | 0.1 | nM | (not cited) | TGFb inhibition of NK function |

**Note:** IL12_50 was manually increased from 0.14 pM (Altan-Bonnet) to 15 pM because the original value made IL-12 always-on. A proper dose-response extraction would resolve whether 15 pM is correct.

### Tier 3 — Chemokine EC50s and niche parameters

| Parameter | Current value | Units | What it controls |
|-----------|-------------|-------|------------------|
| `CCL2_50` | 0.23 | nM | CCL2 EC50 for monocyte/MDSC recruitment |
| `CCL5_50_Treg` | 0.15 | nM | CCL5 EC50 for CCR5-mediated Treg recruitment |
| `IL6_50_MDSC` | 0.06 | nM | IL-6 enhancement of MDSC recruitment (in IL6_EC50s group) |
| `IL6_50_iCAF` | 0.1 | nM | IL-6 EC50 for iCAF maintenance (in IL6_EC50s group) |
| `DAMP_50` | 10000 | cell/day | DAMP release EC50 for APC maturation |
| `EC50_P1_mature` | 10 | nM | Antigen EC50 for APC maturation |
| `pO2_50` | 7 | mmHg | Oxygen tension for hypoxia response |

**Note:** IL6_50_MDSC and IL6_50_iCAF are in the IL6_EC50s hierarchical group anchored by IL6_50_M2 (already constrained from Fernando 2014). Extracting one more member would strengthen the group.

### Unconstrained cytokine secretion rates (high-impact non-EC50s)

These are the cytokine production rates that feed the macrophage polarization OR-gates and the CXCL9 feedback loop. All have sigma > 25 (essentially unconstrained) and are measurable via standard ELISA on activated immune cell cultures (`batch_accumulation` forward model).

| Parameter | Current value | Units | Sigma | What it feeds |
|-----------|-------------|-------|-------|---------------|
| `k_IFNg_Tsec` | 4e-12 | nmol/cell/day | 26.2 | M2→M1 polarization (via IFNg_50), CXCL9 production (via IFNg_50_CXCL9), PD-L1 upregulation |
| `k_IL10_sec` | 3e-12 | nmol/cell/day | 26.5 | M1→M2 polarization (via IL10_50), immunosuppression |
| `k_IL12_sec` | 8.5e-12 | nmol/cell/day | 25.5 | M2→M1 polarization (via IL12_50), NK activation |
| `k_IL12_Msec` | 5e-13 | nmol/cell/day | 28.3 | Same as above, macrophage source (vs cDC1 source) |
| `k_TGFb_Tsec` | 1.2e-10 | nmol/cell/day | 22.8 | M1→M2 polarization, T cell suppression, CAF activation |
| `k_TGFb_Msec` | 2e-11 | nmol/cell/day | 24.6 | Same, macrophage source |

**Why these matter:** IFNg, IL-10, IL-12, and TGFb are the four cytokines that drive macrophage polarization (the OR-gate Hill functions). Their secretion rates set the steady-state cytokine concentrations, which determine the effective Hill terms. Constraining these rates from ELISA data means the Hill terms are no longer nuisance parameters — they become predictions.

**Literature availability:** All of these have published ELISA data from activated immune cell cultures:
- k_IFNg_Tsec: anti-CD3/CD28 activated CD8+ T cells, 24h supernatant ELISA (many papers)
- k_IL10_sec: M2-polarized macrophage cultures (IL-4/IL-13 treated), 24h ELISA
- k_IL12_sec: LPS/IFNg-activated cDC1 or BMDM cultures, 24h ELISA
- k_TGFb_Tsec/Msec: Activated T cell or macrophage cultures, TGFb ELISA

### Other high-value non-EC50 targets

| Parameter | Sigma | Why extract |
|-----------|-------|------------|
| `APC0_cDC2_T` | 14.6 | cDC2 density in tumor. APC0_cDC1_T already constrained; same mIF papers likely have cDC2 data |
| `q_Treg_T_in` | 9.8 | Treg trafficking. Skipped weak MRI target; better adoptive transfer data may exist |
| `k_CCR5_Treg_rec` | 9.2 | CCL5/CCR5 Treg chemotaxis. Standard Transwell data available |
| `k_MMP_M1` | 18.4 | M1-mediated collagen degradation. Pre/post therapy biopsy data would be ideal but rare |

### Prioritized extraction plan

**Round 1 — EC50s (direct_fit Hill, cleanest targets):**
1. **IFNg_50_CXCL9** — IFNg dose-response on CXCL9/CXCL10 production (ELISA readout)
2. **CXCL9_50_Trec** — CXCL9/10 chemotaxis dose-response (Transwell migration)
3. **IL2_50** — extract from Marchingo 2014 (already cited, likely has CFSE vs IL-2 dose)
4. **TGFb_50_Teff** — TGFb suppression of T cell proliferation/function
5. **IL12_50** — verify the manual 15 pM override with actual dose-response data
6. **CCL2_50** — monocyte/MDSC chemotaxis dose-response

**Round 2 — Cytokine secretion rates (batch_accumulation, well-characterized):**
7. **k_IFNg_Tsec** — activated CD8+ T cell IFNg ELISA. **Most impactful unconstrained rate.**
8. **k_IL10_sec** — M2 macrophage IL-10 ELISA
9. **k_IL12_sec** — activated DC IL-12p70 ELISA
10. **k_TGFb_Tsec** — activated T cell TGFb ELISA

**Round 3 — Cell densities and trafficking:**
11. **APC0_cDC2_T** — cDC2 density from same mIF papers as cDC1
12. **q_Treg_T_in** — Treg adoptive transfer kinetics
13. **k_CCR5_Treg_rec** — CCL5 Transwell chemotaxis

All EC50 targets use `direct_fit` Hill models. All secretion rates use `batch_accumulation`. Expected effort: ~1-2 hours per target.

---

## When to stop extracting submodel targets

There is no natural stopping point — every parameter *could* have a submodel target. The question is diminishing returns vs. opportunity cost.

**Arguments to keep extracting:**
- Each submodel target reduces the burden on full-model calibration targets during SBI
- EC50s and secretion rates are low-effort extractions (~1-2 hours each with MAPLE)
- Tighter priors mean faster SBI convergence and more interpretable posteriors
- The cytokine secretion rates (sigma > 25) are essentially free parameters right now

**Arguments to run inference first:**
- PRCC sensitivity analysis identifies which parameters actually matter for model outputs
- Posterior predictive checks reveal which observables are poorly covered
- Some parameters may be structurally non-identifiable regardless of prior width
- Running inference generates the figures and results needed for the paper

**Recommended approach:** Run SBI inference with current targets (83 total). Use posterior predictive checks to identify which parameters have the widest posteriors or worst predictive coverage. Extract submodel targets for *those* parameters specifically. This is data-driven prioritization rather than exhaustive extraction.

The 13-target extraction plan above is a hedge — it addresses the parameters most likely to matter (cytokine axes, treatment-mechanism EC50s) based on mechanistic reasoning. If inference reveals that other parameters dominate, pivot to those instead.