# Calibration Strategy

## Context

The PDAC QSP model has ~177 free parameters in `pdac_priors.csv` and needs a structured calibration strategy. The model is significantly underdetermined, requiring both isolated-experiment data (submodel targets) and full-model observables (calibration targets from clinical/IHC data).

The calibration target framework lives in `qsp-llm-workflows` and supports two target types:
- **IsolatedSystemTarget**: Submodel-based targets from in vitro/ex vivo experiments that constrain individual parameters or small parameter groups
- **CalibrationTarget**: Full-model observables from clinical/IHC data for joint calibration via SBI

Both use a meta-analytic framework for context mismatch handling (see `qsp-llm-workflows/docs/calibration_target_design.md`).

### Provenance Requirement

**Every calibration target must have full provenance**: DOI, value_snippet (verbatim text from the paper), value_location (figure/table/section), source_ref, and experimental context. The existing test statistics CSVs (`baseline_no_treatment.csv`, `clinical_outcomes.csv`, `gvax_nivo_neoadjuvant_zheng2022.csv`) have medians and IQRs but **unknown provenance** — we cannot verify where those values came from. These should be treated as unreliable and replaced with properly extracted targets via `qsp-llm-workflows`.

The `pdac_calibration_targets_computable.csv` in `qsp-llm-workflows/jobs/input_data/` is an *input spec* (what to extract), not extracted data — it defines the target wishlist and is fine as a starting point. Only 6 submodel target YAMLs (all T cell) have been fully extracted with provenance so far.

---

## Calibration Workflow

### Phase 1: Target Design and Extraction

Everything downstream depends on having trustworthy, provenanced calibration targets. This must come first.

#### 1a. Design the target set

The target set must include both **baseline TME targets** (what does untreated PDAC look like at diagnosis?) and **treatment-response targets** (how does the TME change under therapy?). Treatment-response targets are essential for two reasons:

1. **Clinical relevance**: The model's purpose is to predict immunotherapy outcomes. Parameters that determine treatment response (checkpoint binding, exhaustion recovery, cytokine sensitivity) are the clinically important ones. Without treatment-response targets, sensitivity screening would miss these parameters entirely — they'd appear "inert" because no target measures their effect on anything.

2. **Breaking degeneracies**: Baseline snapshots constrain operating-point parameters but cannot distinguish between parameter combinations that produce the same steady state but different dynamic responses. Two parameterizations with identical M1/M2 = 0.57 at diagnosis can predict completely different M1/M2 trajectories under anti-PD-1. Treatment-response targets pin feedback loop *gain*, not just equilibrium.

**Target categories needed:**

| Category | Purpose | Example Targets |
|---|---|---|
| Baseline TME (diagnosis snapshot) | Pin steady-state operating point of each module | Cell densities, ratios, cytokine concentrations |
| Treatment-response biomarkers | Pin feedback loop gain and sensitivity | Fold changes in CD8, M1/M2, cytokines under GVAX/nivo |
| Clinical outcome endpoints | Pin population-level treatment effect | Response rates, PFS, OS, pathologic response |
| Negative controls | Validate model predicts known treatment failures | Anti-PD-1 monotherapy 0% ORR in PDAC |

#### 1b. Expand the target wishlist

The current `pdac_calibration_targets_computable.csv` has 35 baseline TME targets. It needs expansion in two directions:

**Missing baseline mechanistic axes:**

| Axis | Targets to Add | Available Data |
|---|---|---|
| Metabolism | Tumor lactate concentration | Mehta 2017: ~2.4x fold increase vs normal pancreas (HR-MAS NMR, human surgical specimens) |
| Metabolism | Tumor glucose depletion proxy | Koay 2020: FDG-PET SUVmax 3-12, GLUT1 correlation; absolute tissue [glucose] poorly quantified |
| NK cells | NK density in tumor | Lim 2019: <0.5% of TILs by flow cytometry (n=80 resectable PDAC); implies <140 cells/mm^3 |
| NK-cDC1 axis | XCL1 expression / trNK fraction | Trefny 2024: trNK cells are strongest XCL1 source to XCR1+ cDC1s (scRNA-seq, human PDAC) |
| Chemokines | CXCL9 concentration or expression | Romero 2020: CXCL9 RNA correlated with CD8 infiltration (multiple PDAC cohorts); protein levels sparse |
| Chemokines | CCL5 concentration | Singh 2018: elevated CCL5 in metastatic PDAC tissue (IHC); serum CCL5 prognostic |
| DC subtypes | cDC1 density or TAM:cDC1 ratio | Hegde 2020: cDC1 ~100x less abundant than TAMs (mass cytometry, human PDAC) |
| DC subtypes | DC maturation state by subtype | Hegde 2020: lower co-stimulatory markers on PDAC DCs vs lung adenocarcinoma |

**Treatment-response targets to add:**

| Target | Source | Data |
|---|---|---|
| CD8 fold increase d21 (GVAX+nivo) | Zheng 2022 | ~2.8x increase, n=neoadjuvant cohort |
| CD8 exhaustion reduction d21 | Zheng 2022 | ~50% reduction in PD-1+ fraction in OS>2y patients |
| M1/M2 ratio d21 | Zheng 2022 | ~1.6 (shifted from 0.57 baseline) |
| CD8+CD137+ in TLAs post-GVAX | Blair/Heumann 2023 | 27.9% in GVAX+nivo+urelumab arm (n=40, Nature Comms) |
| GZMB+CD8+ increase | Blair 2023 | Significant increase across all arms (p=0.045) |
| Major pathologic response rate | Zheng 2025 | 35% (<10% residual tumor) in GVAX/Cy/nivo/SBRT (n=31, borderline resectable) |
| Anti-PD-1 monotherapy ORR (negative control) | Brahmer 2012, Royal 2010, O'Reilly 2019 | 0% ORR in unselected PDAC across multiple trials (n=14-65) |
| GVAX+ipi 1-year OS | Le 2013 | 27% vs 7% ipi alone (n=30, randomized) |

**Clinical outcome targets for validation/refinement:**

| Target | Source | Data |
|---|---|---|
| mOS GVAX+nivo neoadjuvant | Zheng 2025 | 20.4 months (95% CI 18.2, NA) |
| DFS GVAX+nivo+ure vs GVAX | Blair 2023 | HR 0.30 (95% CI 0.11-0.86) |
| mOS checkpoint mono (negative control) | Multiple | 3-4 months |
| mOS GVAX+ipi | Le 2013 | 5.7 months |

#### 1c. Extract targets with full provenance

Run the expanded target CSVs through `qsp-llm-workflows`:
- `qsp-extract` with `--type calibration_target` for full-model targets
- `qsp-extract` with `--type isolated_system_target` for submodel targets (parameter-level)
- Each extraction produces a YAML with DOI, value_snippets, distribution_code, experimental_context
- Human review using the curation guide
- Approved targets move to canonical store

**The existing test statistics CSVs can serve as a rough cross-check** — if a freshly extracted target value is wildly different from the old test statistic, investigate why. But the extracted version with provenance is the one we trust.

### Phase 2: Submodel Target Extraction (Individual Parameters)

**This is the primary method for calibrating individual parameters.** Instead of extracting parameter values directly from literature, we extract the *experiment* — the raw observable with its uncertainty — and pair it with a small ODE submodel that shares parameter names with the full QSP model.

**Why submodel targets work well:**
- Give proper posteriors on individual parameters, not point estimates
- Handle joint constraints when multiple parameters appear in the same submodel (e.g., proliferation and death in a net growth assay)
- Transparent and auditable — the submodel code, experimental context, and derivation logic are all recorded in the YAML
- Context mismatch (in vitro → in vivo, mouse → human) is handled formally via variance inflation, not ad hoc
- Composable — parameter-level posteriors become informative priors for full-model SBI

**Extraction workflow (via `qsp-llm-workflows`):**
1. Prepare input CSV with target_id, cancer_type, parameters, and notes
2. Run `qsp-extract` with `--type isolated_system_target`
3. LLM finds relevant literature, extracts experimental data, writes submodel code
4. Pydantic validation catches structural issues automatically
5. Human review using the curation guide (check study interpretation, submodel correctness, snippet fidelity, DOI validity)
6. Approved targets move from `to-review/` to the canonical target store

**Batch by calibration block** (see Phase 4), not alphabetically. Extracting all fibroblast/ECM parameters together lets the LLM cross-reference values within a coherent biological context.

**Distinguish parameter types in extraction prompts:**
- Rate constants (k_*): Need kinetic data from time-course experiments
- Hill function parameters (EC50, n): Need dose-response data
- Stoichiometric/fraction parameters: Need mass balance or composition data

**Current state:** 6 extractions completed (T cell proliferation/death). IsolatedSystemTarget CSVs needed for all other calibration blocks.

**Submodel targets needed per calibration block:**

| Block | Example Submodel Targets | Priority |
|---|---|---|
| Fibroblast/ECM | PSC activation rates (culture experiments), ECM secretion rates, MMP degradation kinetics | High (15 params) |
| Macrophage | M1→M2 polarization rates (cytokine stimulation), M2→M1 repolarization, phagocytosis rates | High (10 params) |
| T cell (remaining) | Treg suppression assays, checkpoint binding kinetics, exhaustion dynamics | Medium (mostly done) |
| Metabolism | Glycolysis rates (Warburg effect), lactate production per cancer cell, glucose consumption | High (6 params, zero data) |
| NK cells | NK killing assays (51Cr release), NK degranulation rates, XCL1 secretion | Medium (6 params) |
| APC/DC | DC maturation kinetics (LPS/DAMP stimulation), antigen uptake rates, migration assays | Medium (10 params) |
| Vasculature | VEGF secretion rates, endothelial proliferation, vascular permeability | Medium (10 params) |

### Phase 3: Sensitivity Screening

#### Baseline scenario: COMPLETE

**Method:** Surrogate sensitivity analysis using PRCC (Partial Rank Correlation Coefficients) and Random Forest permutation importance against 31 calibration targets at diagnosis. Parameter significance determined by permutation null test (1,000 permutations, Benjamini-Hochberg FDR < 0.05).

**Results (2026-02-21):**
- 2,852 valid simulations from 5,000 requested (43% NaN rate from failed evolve_to_diagnosis)
- Cross-method agreement: Spearman rho = 0.865 (PRCC vs RF)
- **97/177 parameters significant** at FDR < 0.05
- 80 parameters non-significant → fix at prior medians for baseline SBI
- Top 13 parameters dominate: fibroblast/stroma (5), tumor intrinsic (2), CD8 trafficking (3), immune regulation (3)
- Full results: `results/surrogate_sensitivity/REPORT.md`, `notes/diagnostics/sensitivity_analysis_biological_interpretation.md`

**Key biological findings:**
- Stroma dominance: the full CAF plasticity circuit (iCAF↔myCAF, PSC activation, ECM dynamics) is significant
- CD8 trafficking chain (thymus → LN → tumor) matters at every step; killing efficiency (`p_T_kill_per_contact`) does not
- Metabolism module (glucose, lactate, hypoxia) has real influence — not decorative
- CXCL12 T cell exclusion axis (Feig 2013) is significant
- apCAF→Treg induction axis (Elyada 2019) is significant
- Chemotaxis module (CXCR3, CCR5) is non-significant at baseline — constitutive trafficking dominates inducible recruitment

**Prior sensitivity caveat:** The SA is conditional on the current priors in `pdac_priors.csv`. Many priors were guesstimated (not rigorously literature-derived). A parameter audit (`calibration_targets/sources/parameter_audit.md`) identified ~5 red flags and ~6 yellow flags; most have been corrected. The remaining 2-5x uncertainties in prior centers are within the lognormal sigma ranges (0.7-1.2), so SA membership is probably robust. However:
- Systematic bias in immune cell turnover rates (collectively too slow) could make the TME dynamics too static even if densities are constrained
- SBI posteriors will self-diagnose prior location errors (parameter railing against boundary)
- A prior-perturbation experiment (shift yellow-flag priors to audit-recommended values, re-run SA) would directly test robustness but is not blocking

#### Treatment scenarios: TODO

Sensitivity must also be assessed against treatment-response targets (GVAX+nivo, clinical outcomes). Parameters important for treatment sensitivity (checkpoint, exhaustion recovery, cytokine response) may be non-significant at baseline but critical under therapy. Run per-scenario SA and take the union of significant parameters across scenarios.

**Three-axis classification of active parameters:**

| Sensitivity | Identifiable | Prior Confidence | Action |
|---|---|---|---|
| High | Yes | Poor | **Priority**: submodel target extraction → SBI |
| High | Yes | Good | Include in SBI, prior already informative |
| High | No | Poor | **Submodel target extraction** (isolated experiments) |
| High | No | Good | Fix at prior median |
| Low | — | — | Fix at prior median |

### Phase 4: Modular SBI Calibration

The full parameter set (even post-screening) is too large for joint SBI. But the model is substantially decomposable into modules with strong intra-block coupling but weaker inter-block coupling.

**Natural calibration blocks:**

| Block | Parameters | Key Targets | Coupling to Other Blocks |
|---|---|---|---|
| 1. Tumor growth | ~5 (k_C1_growth, K_initial, initial_tumour_diameter, ...) | Tumor volume, doubling time | Weak (tumor is source term for everything) |
| 2. Fibroblast/ECM | ~15 (PSC activation, ECM secretion/degradation, CAF fractions) | Stromal fraction, ECM density, CAF proportions, αSMA area | Moderate (ECM → immune migration Hill functions) |
| 3. Macrophage polarization | ~10 (M1/M2 rates, cytokine secretion) | M1/M2 ratio, TAM density, M2 fraction, CCL2, post-treatment M1/M2 | Moderate (cytokine Hill functions) |
| 4. T cell trafficking & killing | ~15 (recruitment, exhaustion, killing, checkpoint) | CD8 density, CD8/Treg ratio, exhaustion fraction, GrB+ CD8, CD8 fold change | Strong (depends on APC, cytokines, ECM) |
| 5. Metabolism | ~6 (glucose, lactate, HIF coupling) | Tumor lactate | Moderate (lactate → immune impairment Hill functions) |
| 6. NK/innate | ~6 (NK recruitment, killing, XCL1) | NK density | Moderate (XCL1 → cDC1, IFNg → polarization) |
| 7. Vasculature/hypoxia | ~10 (VEGF, pO2, HIF) | VEGF concentration, vascular density | Moderate (HIF → polarization, VEGF → permeability) |
| 8. APC/DC maturation | ~10 (cDC1/cDC2 rates, migration, maturation) | APC maturation ratio, DC density, cDC1 fraction | Strong (depends on cytokines, ECM, antigen) |

**Modular calibration procedure:**
1. Calibrate blocks with weak external dependencies first (tumor growth, fibroblast/ECM, metabolism)
2. Fix those posteriors, calibrate blocks that depend on them (macrophage, vasculature/hypoxia)
3. Fix those, calibrate remaining blocks (T cell, NK, APC/DC)
4. Each block: ~5-15 parameters, ~5-15 matched targets → tractable for SBI

**Risk:** Modular calibration loses joint uncertainty. If ECM_50_T_mot and k_C_CD8_exh are actually correlated in the true posterior (they trade off in determining intratumoral CD8 density), modular calibration can't capture that.

**Mitigation:** Final joint refinement round (Phase 5) on the ~30-40 most sensitive parameters using modular posteriors as priors.

### Phase 5: Joint Refinement

After modular calibration, run a final SBI round:
- **Parameters**: Top 30-40 by sensitivity (from Phase 3 screening), using modular posteriors as priors
- **Targets**: All calibration targets (baseline TME + treatment response + clinical outcomes)
- **Simulations**: 50-100k evolve_to_diagnosis snapshots + treatment simulations (feasible with simulation pooling)
- **Purpose**: Capture cross-block correlations that modular calibration misses

**Method choice:**
- **SBI (BayesFlow/SNPE)**: Good fit for 30-40 parameters with simulation pooling already implemented. Needs 50-100k simulations for that dimensionality.
- **BSL (Bayesian Synthetic Likelihood)**: More robust to summary statistic misspecification, highly parallelizable. Better if likelihood surface is multimodal.
- **SMC-ABC**: Very parallelizable, no neural network training. Pragmatic choice if SBI struggles.

Recommendation: SBI for modular blocks (Phase 4), then SBI or BSL for joint refinement depending on how modular posteriors look. If unimodal and well-behaved, SBI is fine. If multimodal, BSL is safer.

### Phase 6: Validation

1. **Prior predictive checks** — simulate from priors, verify outputs span clinically plausible ranges. Systematic misses signal structural model issues.
2. **Posterior predictive checks** — simulate from posterior, verify calibration targets are recovered within uncertainty.
3. **Emergent validation** — hold out targets from SBI, predict from posterior. If the model predicts held-out ratios correctly, the feedback balance is real rather than overfit.
4. **Cross-scenario validation** — calibrate on baseline + GVAX, predict clinical progression or vice versa.
5. **Negative control validation** — verify model predicts anti-PD-1 monotherapy failure in PDAC (0% ORR).

---

## What Different Target Types Constrain

- **Absolute densities** (CD8, Th, Treg, TAM, etc.) — Pin the scale of each population. Necessary but not sufficient; compensating parameters can match densities while getting feedback balance wrong.
- **Ratios** (M1/M2, CD8/Treg, stromal fraction, etc.) — Most powerful constraints on relative balance. M1/M2=0.57 directly constrains the net balance of HIF + lactate + IFNg polarization drivers.
- **Cytokine concentrations** (IFNg, IL-10, TGFb, etc.) — Constrain the signaling intensity that mediates feedback loops. These are the "intermediate variables" inside the loops.
- **Dynamic/temporal targets** (growth rate, fold changes, AUCs) — Constrain time derivatives and perturbation responses. Reveal feedback loop *gain*, not just operating point.
- **Treatment-response targets** (CD8 fold change, M1/M2 shift under therapy) — Constrain the *sensitivity* of feedback loops to perturbation. These are the targets that make the model clinically predictive.
- **Clinical outcome targets** (ORR, PFS, OS, pathologic response) — Population-level endpoints that integrate all mechanisms. Validate that the calibrated model produces realistic clinical predictions.
- **Negative controls** (checkpoint mono 0% ORR) — Validate that the model correctly predicts known treatment failures. If the model can't explain *why* anti-PD-1 fails in PDAC (stromal barrier, poor antigen presentation, immunosuppressive TME), the calibration is wrong regardless of how well it fits positive outcomes.

## Quality vs Quantity

More targets of the same type add very little. What matters is:

1. **Unique sensitivity to under-constrained axes** — One lactate measurement constrains 6 metabolism parameters more than 10 additional immune cell ratios.
2. **Sensitivity to interaction terms** — Ratios depend on the net balance of competing mechanisms, making them more informative than absolute counts.
3. **Different timepoints or conditions** — Two measurements of the same species at different times constrain dynamics in a way that two different species at the same time cannot.
4. **Perturbation responses** — Treatment-induced fold changes reveal feedback loop gain, which steady-state snapshots cannot.

With 177 parameters, ~40-60 are likely identifiable from any reasonable target set. The rest remain prior-determined. This is acceptable if priors are well-calibrated from submodel targets.

---

## Available Literature Data by Mechanistic Axis

Summary of key data sources identified for calibration target extraction. Each entry includes the measurement type, quantitative values where available, and the study context (species, n, method).

### Metabolism (Currently Zero Targets)
- **Tumor lactate (absolute)**: Battini 2017 (BMC Medicine, DOI: 10.1186/s12916-017-0810-z) — PDAC: 16.1 ± 4.7 nmol/mg tissue vs normal: 11.0 ± 2.5 nmol/mg (HR-MAS NMR, n=44 PDAC + n=9 normal, chemo-naive). Fold change ~1.47x. Convert nmol/mg → µM via tissue density ~1 g/mL (×1000). Directly constrains `V_T.lactate` and k_lac_Csec/k_lac_clear balance.
- **Tumor glucose**: Same Battini 2017 study reports glucose levels by HR-MAS NMR in matched PDAC vs normal tissue. Constrains k_glc_tumor.
- **Glucose uptake**: Koay 2020 — FDG-PET SUVmax 3-12 in PDAC, GLUT1 expression correlates with SUV (r=0.60). Mapping to model glucose level is indirect (SUVmax ∝ uptake rate, not tissue concentration).

### NK Cells (Currently Zero Targets)
- **NK density**: Lim 2019 — <0.5% of TILs by flow cytometry (n=80 resectable PDAC). Given baseline total immune density ~28,000 cells/mm^3, implies <140 NK cells/mm^3.
- **NK-cDC1 axis**: Trefny 2024 — Tissue-resident NK (trNK) subpopulation with high XCL1/XCL2 expression; strongest signal source to XCR1+ cDC1s (CellChat analysis, scRNA-seq). Higher trNK signature correlates with better OS/DFS.
- **Pan-cancer NK**: Peng 2023 — PDAC has among lowest NK infiltration across cancer types.

### DC Subtypes (Currently Zero Subtype-Specific Targets)
- **cDC1 paucity**: Hegde 2020 — cDC1s ~100-fold less abundant than TAMs or neutrophils (mass cytometry, human PDAC). cDC1s 79-fold less than in lung adenocarcinoma. Lower co-stimulatory/maturation markers.
- **DC states**: Binnewies 2019 — Five conserved DC states across solid cancers; cDC1 critical for CD8 priming.
- **Systemic DC deficiency**: Meyer 2018 — Systemic cDC1 progenitor reduction in PDAC patients.

### Chemokines
- **CXCL9/10**: Romero 2020 — Four-chemokine signature (CCL4, CCL5, CXCL9, CXCL10) strongly associated with CD8 infiltration (RNA-seq TPM, multiple PDAC cohorts). Protein-level concentrations sparse.
- **CXCL9/10 for checkpoint response**: Chow 2019 — Macrophage-derived CXCL9/10 required for checkpoint blockade response (mouse model).
- **CXCL12**: Feig 2013 — FAP+ CAFs are sole tumoral source of CXCL12. AMD3100 (CXCR4 inhibitor) arrests tumor growth; AMD3100 + anti-PD-L1 leads to significant tumor decline (KPC mouse model).
- **CCL5**: Singh 2018 — Elevated CCL5 in metastatic PDAC tissue (IHC); serum CCL5 prognostic.

### Treatment-Response (GVAX-Based)
- **Zheng 2022**: Neoadjuvant GVAX+nivo — CD8 ~2.8x fold increase, ~50% exhaustion reduction, M1/M2 shifts to ~1.6 at d21. (Already defined in test statistics but needs provenance verification.)
- **Blair/Heumann 2023**: Platform trial GVAX ± nivo ± urelumab (n=40, Nature Comms) — CD8+CD137+ in TLAs: 27.9% in triple arm (p=0.003). DFS HR 0.30 (CI 0.11-0.86). Significant GZMB+CD8+ increase.
- **Zheng 2025**: GVAX/Cy/nivo/SBRT in borderline resectable (n=31, CCR) — 35% major pathologic response (<10% residual), 1 pCR. mOS 20.4 months.
- **Le 2013**: GVAX+ipi vs ipi alone (n=30, randomized) — 1-year OS 27% vs 7%. Mesothelin-specific T cell increase in long responders.

### Negative Controls (Checkpoint Monotherapy in PDAC)
- **Brahmer 2012**: Nivolumab Phase I — 0% ORR in PDAC subset (n=14).
- **Royal 2010**: Tremelimumab — 0% ORR (1 PR in 20 patients).
- **O'Reilly 2019**: Durvalumab ± tremelimumab — 0% ORR mono, 3.1% combo (n=65). mOS 3.6 months mono.

### Immune Subtypes (for Model Validation)
- **Wartenberg 2018**: Three PDAC immune subtypes — "immune escape" (54%), "immune rich" (35%), "immune exhausted" (11%). Distinct CD8/Treg profiles per subtype. Useful for validating that the calibrated model can generate patient heterogeneity.

---

## Redundancy in Existing Test Statistics

The `clinical_outcomes.csv` test statistics contain significant redundancy: 7/15 targets are tumor volume derivatives (`best_percent_change`, `time_to_progression`, `tumor_doubling_time`, `tumor_volume_d30/d60/d90`, `tumor_growth_rate_log`). These are all derived from the same V_T trajectory and provide ~2-3 effective independent constraints. If re-extracted, consolidate to 2-3 independent tumor volume targets.

---

## LLM-Assisted Workflow

The scale of this calibration task (~177 parameters, ~50-60 targets needed, literature extraction for each) makes LLM assistance essential rather than optional.

**Where LLMs add the most value:**
1. **Target extraction** (Phase 1-2) — The `qsp-extract` pipeline in `qsp-llm-workflows` automates literature search, data extraction, submodel code generation, and YAML packaging. Human review is still required but the extraction bottleneck is broken.
2. **Curation** — LLM-assisted review of extractions using the curation guide: check study interpretation, submodel correctness, snippet fidelity, DOI validity. Decide whether to accept, fix, or re-extract.
3. **Parameter classification** — LLM domain knowledge can pre-classify parameters by expected importance and data availability before formal sensitivity screening.
4. **Batch extraction at scale** — Organize by calibration block (all fibroblast/ECM together, all macrophage together, etc.) so the LLM cross-references values within a coherent biological context.

---

## Glucose-Lactate Feedback Fix

During earlier analysis, identified that lactate production was glucose-independent (`k_lac_Csec * C_total`), meaning the model assumed immune cells always benefit more from vasculature-delivered glucose. Fixed by making lactate production proportional to glucose availability:

```matlab
% Before (biased):
set(reaction,'ReactionRate','k_lac_Csec * C_total');

% After (glucose-dependent):
set(reaction,'ReactionRate','k_lac_Csec * C_total * glc / glc_normal');
```

This creates the negative feedback: better vasculature → more glucose → more lactate → partial offset of immune benefit.

---

## SBI Dimensionality Considerations

Sensitivity screening reduced 177 → 97 significant parameters for the baseline scenario. This is still high-dimensional for neural posterior estimation but more tractable than 177.

**Strategy for baseline SBI:**
- **Start with all 97 significant parameters** against 31 baseline calibration targets. SBI will naturally constrain the top ~30-40 most identifiable parameters while leaving the remaining ~60 at approximately their priors. This is acceptable — SBI handles "too many parameters" gracefully by producing wide posteriors on uninformative ones.
- **If 97D proves intractable** (poor convergence, uninformative posteriors everywhere), fall back to modular calibration (Phase 4) with 6-8 blocks of 5-15 parameters.
- **Joint refinement** (Phase 5): After baseline calibration, use baseline posteriors as priors for treatment-scenario SBI (GVAX+nivo). This is where treatment-sensitive parameters (checkpoint, exhaustion) get constrained.

**Simulation budget:** evolve_to_diagnosis snapshots are the expensive step (~15-20s each). With HPC parallelism, 10-50k simulations are feasible per SBI round. The simulation pooling infrastructure handles batch accumulation and reuse across rounds.

**Prior sensitivity for SBI:** If a prior center is wrong by >5x and sigma is tight, SBI posteriors will rail against the prior boundary. This is the primary diagnostic for bad priors. After the first SBI round, check all posterior marginals for boundary-piling — any parameter showing this needs its prior recentered and widened before the next round.

---

## Concrete Next Steps

### Completed
1. [x] **Sensitivity screening (baseline)**: Surrogate SA with PRCC + RF + permutation significance test. 97/177 parameters significant. Results in `results/surrogate_sensitivity/`.
2. [x] **Parameter audit**: Prior distribution audit in `calibration_targets/sources/parameter_audit.md`. Major red flags (f_cDC1, thymic output, CXCL12_50, DC death rates, etc.) corrected.
3. [x] **Calibration target extraction (baseline)**: 31 baseline TME targets extracted with provenance in `calibration_targets/baseline_no_treatment/`.

### In Progress
4. [ ] **Run baseline SBI**: 97 significant parameters against 31 baseline calibration targets. Use existing simulation pool (`restriction_estimator_v4`). Start with NPE/BayesFlow, assess convergence.

### Upcoming
5. [ ] **Diagnose prior issues from SBI posteriors**: After first SBI round, check for posterior-railing-against-prior-boundary. Recenter/widen priors for flagged parameters.
6. [ ] **Sensitivity screening (GVAX+nivo scenario)**: Run per-scenario SA to identify treatment-sensitive parameters (checkpoint, exhaustion, cytokine response).
7. [ ] **Expand target wishlist**: Add treatment-response targets (Zheng 2022 CD8 fold change, M1/M2 shift, Blair 2023 pathologic response), clinical outcomes, negative controls.
8. [ ] **Treatment-scenario SBI**: Use baseline posteriors as priors, calibrate against treatment-response + clinical targets.
9. [ ] **Validation**: Prior/posterior predictive checks, held-out target prediction, negative control validation (anti-PD-1 mono 0% ORR).
