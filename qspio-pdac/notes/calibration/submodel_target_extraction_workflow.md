# Submodel Target Extraction Workflow

A collaborative workflow for extracting submodel targets from literature.
Captures the human + AI division of labor that works best in practice.

## Overview

The goal is to turn published experimental data into SubmodelTarget YAMLs
that constrain QSP model parameters via Bayesian inference. Each target
links a paper's quantitative data to one or more model parameters through
a forward model (the "bridge" between parameter space and observable space).

## Phase 1: Parameter Audit + Literature Search

**Who leads:** AI (with human guidance on priorities)

1. **Identify unconstrained parameters.** Check `notes/calibration/fibroblast_param_audit.md`
   (or equivalent) for parameters without submodel targets. Prioritize by
   sensitivity (PRCC ranking) and model impact.

2. **Investigate parameter usage in model code.** Before searching literature,
   look up each parameter in:
   - `parameters/parameters_PDAC.m` (definition, units, notes)
   - `core/modules/*.m` (reaction rate expressions, Hill functions)
   - `parameters/pdac_priors.csv` (current prior)
   - Existing targets in `calibration_targets/submodel_targets/`

   **This determines what data you actually need.** A parameter gated by
   Hill(TGFb) × Hill(PDGF) needs different literature than a standalone rate.

3. **Load the maple extraction tool** (`mcp__maple__extract_target`) BEFORE
   doing any literature search. The guide contains the YAML schema, valid
   enum values, and hard rules that validators enforce.

4. **Web search for candidate papers.** Search for papers with:
   - Time-course data (activation curves, proliferation curves)
   - Dose-response data (EC50 measurements)
   - Quantitative measurements with error bars (mean ± SD, not just qualitative)
   - Multiple conditions (control vs treatment) for paired analysis

5. **Write a consolidated summary** to `notes/calibration/` documenting papers
   found, key data available, and which parameters each paper constrains.

## Phase 2: PDF Collection + Data Inventory

**Who leads:** Human (for PDF acquisition), AI (for data cataloging)

1. **Human obtains PDFs** into `calibration_targets/submodel_targets/papers/<SourceTag>/`

2. **AI reads each paper** and creates a complete inventory of every figure
   and table with quantitative data. For each:
   - What it measures
   - Whether it has error bars / SD
   - Which model parameters it could constrain
   - Whether it needs digitization

3. **Human and AI agree** on which data points to extract and in what order.
   Prioritize by: (a) number of parameters constrained, (b) data quality
   (SD reported, good n), (c) ease of extraction (table > figure).

## Phase 3: Figure Digitization

**Who leads:** Human (for reading figures), AI (for setup and computation)

1. **AI creates CSV template** in `papers/<SourceTag>/` with column headers
   and empty rows for each expected data point.

2. **Convention for error bars:** Human reports either mean+SD or mean-SD
   depending on which error bar direction is easier to read. Specify which
   convention in the CSV column header (e.g., `mean_plus_sd` or `mean_minus_sd`).

3. **Human fills in values** from the figure. Only fill in what you can
   confidently read — leave cells blank if unclear.

4. **AI computes actual SDs** from the reported values and confirms they
   look reasonable (e.g., CV not implausibly small = probably SEM).

5. **Human verifies** the computed SDs match what they see in the figure.

## Phase 4: YAML Writing

**Who leads:** AI (writes YAML), Human (reviews)

### Forward Model Design (CRITICAL)

1. **Look up the actual reaction** in `core/modules/*.m` that uses the parameter.
   Copy the exact rate expression.

2. **Include ALL Hill functions and gating terms** from that reaction in the
   forward model ODE. Do NOT simplify by assuming Hill terms ≈ 1.

3. **Add nuisance parameters** (`nuisance: true` with inline prior) for
   experimental conditions that aren't QSP parameters:
   - Basal cytokine levels in culture (e.g., `tgfb_basal_culture`)
   - Use the same name across targets that share the same culture system
   - Provide a reasonable lognormal prior (wide sigma ~1.0-1.5)

4. **Maximize QSP parameters per target.** If the reaction has 3 Hill functions,
   include all 3 EC50 parameters even if this target only weakly constrains
   them. Joint inference across targets will resolve degeneracies.

### Input Extraction

5. **Keep values in paper units.** Don't convert percent to fraction, hours to
   days, ng/ml to nM in the input values. Do conversions in code.

6. **Use `figure_excerpt` for digitized data, `value_snippet` for text data,
   `table_excerpt` for table data.** The validator treats these differently.

7. **Don't invent uncertainty.** If the paper doesn't report SD:
   - Bootstrap across conditions or subgroups
   - Use the range of measurements as a population
   - If there's genuinely no spread, the target may not be extractable

8. **Check SEM vs SD.** If uncertainty seems implausibly small (CV < 5% for
   biological data), it's probably SEM. Convert: SD = SEM × sqrt(n).

### Error Model

9. **One error model entry per time point** for ODE models. Each with its own
   `evaluation_points`, `uses_inputs`, `observation_code`.

10. **Observable function signature for ODE models:** `def compute(t, y, y_start)`
    — NOT `(t, y, params, inputs)`. The params/inputs are baked in at construction.

## Phase 5: Validation

**Who leads:** AI (runs validator), Human (reviews failures)

1. **Run `mcp__maple__validate_target`** with papers_dir pointing to the papers
   directory. Three checks:
   - Schema validation (Pydantic)
   - Prior derivation (NumPyro MCMC)
   - Snippet verification (text matching against PDFs)

2. **Common failures and fixes:**
   - "Extra inputs are not permitted" → schema field doesn't exist, check a
     validated target for the real field names
   - "Value not found in snippet" → input value doesn't appear in the quoted
     text. Use figure_excerpt instead, or fix the snippet to match the PDF
   - "Hardcoded numeric values" → move the constant to an input with
     `input_type: unit_conversion` and a rationale
   - "Cannot find valid initial parameters" → the prior medians produce
     NaN/inf in the forward model. Check ODE integration span, initial
     conditions, and whether Hill terms go to 0
   - Author name mismatch → CrossRef wants last name only, not "Smith JA"

3. **Prior derivation results to check:**
   - Contraction near 0 = data confirms prior (good)
   - Contraction > 0.5 = data is informative (great)
   - Contraction < -1 = data conflicts with prior (investigate)
   - z_score > 3 = large shift from prior (may indicate unit error)
   - Divergences > 0 = MCMC problems (fix the model)

4. **After validation passes**, copy YAML to `calibration_targets/submodel_targets/`
   and commit.

## Phase 6: Joint Inference + Review

**Who leads:** AI (runs inference), Human (reviews results)

1. **Run `mcp__maple__run_joint_inference`** across all submodel targets.

2. **Check for consistency across targets** constraining the same parameter.
   If two targets give very different estimates (> 5× apart), investigate:
   - Are the forward models parameterized consistently?
   - Is a Hill function missing from one target?
   - Are the experimental conditions actually comparable?

3. **Check copula correlations.** Strong correlations (|r| > 0.3) between
   parameters indicate shared information — these parameters should appear
   together in the same targets.

4. **Update the parameter audit** with new coverage numbers and derived priors.

## Checklist for Each Target

- [ ] Parameter usage investigated in model code
- [ ] Forward model matches QSP reaction structure (all Hill functions included)
- [ ] All input values in paper units with provenance (snippet/figure/table excerpt)
- [ ] No invented uncertainties — SD from data or bootstrap
- [ ] SEM vs SD checked
- [ ] Nuisance parameters named consistently across targets
- [ ] Schema validation passes
- [ ] Prior derivation passes (or documented why it fails)
- [ ] Snippet validation passes
- [ ] Copied to `calibration_targets/submodel_targets/` and committed
- [ ] Parameter audit updated
