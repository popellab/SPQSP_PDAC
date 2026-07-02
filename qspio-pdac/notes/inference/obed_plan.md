# Optimal Bayesian Experimental Design for Literature Search Prioritization

Status: Planning (2026-03-10)

## Goal

Use the calibrated QSP model to identify which **published observables** or **tighter parameter priors** would most reduce uncertainty in clinically relevant predictive outputs (starting with ORR). This is not about designing new experiments or building digital twins — it's about prioritizing where to spend literature search time.

## Decision-relevant output (y_decision)

**ORR across treatment arms** — chosen because it:
- Integrates the entire causal chain (immune priming → effector function → tumor kill)
- Is maximally uncertain in the current model
- Is the endpoint clinicians and regulators care about
- Both pathologic (MPR at resection) and radiographic (RECIST) definitions are viable

Start with **MPR** (major pathologic response at day 21 resection) since we already have the neoadjuvant scenarios and Zheng 2022 calibration data.

## Approach

### Step 1: Calibration

Run full 4-scenario SBI calibration with posterior predictive:
- baseline_no_treatment
- clinical_progression
- gvax_neoadjuvant_zheng2022
- gvax_nivo_neoadjuvant_zheng2022

This gives posterior θ draws.

### Step 2: Full trajectory posterior predictive

Modify the posterior predictive pipeline to return **full species × time matrices**, not just endpoint test statistics. The raw `.mat` simulation files on HPC already contain this — need a retrieval/extraction script.

Output: matrix of shape (n_VPs, n_species, n_timepoints) per scenario.

The model has ~160 species across 4 compartments (V_T, V_LN, V_C, V_P). At daily resolution over 21 days, that's ~3,360 potential univariate observations per scenario.

### Step 3: Response classification

Define MPR from the tumor volume trajectory. Candidate threshold: <35% viable tumor cells at resection (Agarwal 2025). Classify each posterior predictive VP as responder/non-responder.

### Step 4: MI sweep + LOO observable decomposition

Two complementary analyses:

#### 4a: MI heatmap (univariate)

For every species s in every compartment c at every timepoint t, compute MI(s(c,t), response).

Plot as a heatmap: species × time, colored by MI. Hot spots identify the most informative individual observations. This is the "what single measurement would help most?" answer.

#### 4b: LOO observable decomposition

For each observable j currently used in calibration:
1. Retrain NPE without observable j (seconds, cached sims reused)
2. Measure the change in posterior constraint on ORR-relevant parameters

**Two approaches, trading off cost vs directness:**

**Cheap (no new MATLAB sims):** Measure posterior width change directly from the NPE.
- For each LOO config, sample posterior and compute marginal variances on key parameters
- Weight by PRCC sensitivity to ORR: **ΔVar(θ_k) × sensitivity(ORR, θ_k)** summed over parameters k
- This answers "how much parameter constraint does observable j provide?" without posterior predictive sims
- Limitation: misses nonlinear interactions between parameters in determining ORR

**Expensive but direct (small posterior predictive):** Run 50-100 posterior predictive sims per LOO config.
- ~30 observables × 100 sims = 3,000 MATLAB sims total
- Classify each VP, compute ORR per config
- Measure: Bias(ORR)², Variance(ORR), MSE(ORR) = Bias² + Variance
- **ΔMSE_j = MSE(without j) - MSE(with all)** directly ranks observable value
- Decomposition into ΔBias² and ΔVariance shows *why* each observable matters (precision vs accuracy)

**Recommendation:** Start with the cheap approach to get a rough ranking, then run the expensive version on the top ~10 observables to validate and get the bias/variance decomposition.

**Note on importance weighting:** Reweighting prior predictive sims by posterior density (to avoid new MATLAB runs) was considered but rejected — in ~30-dimensional parameter space, ESS collapses to single digits and the estimates are unreliable.

### Step 5: Composite feature discovery (multivariate)

Search over functions of species to find composite biomarkers that are more informative than any single species. Three approaches, in order of priority:

1. **Enumerated ratios/fold-changes**: Pairwise ratios species_i(t)/species_j(t), temporal fold-changes species_i(t2)/species_i(t1), cross-compartment ratios. ~500K candidates, computable in seconds on the posterior predictive matrix.

2. **Symbolic regression (PySR)**: Search over mathematical expressions to find compact, interpretable formulas that maximally predict response. Discovers things like "log(CD8_VT_d14 / Treg_VT_d14) × cDC1_VLN_d10".

3. **Tree-based feature importance**: Random forest on the full matrix as a cross-check on symbolic regression.

### Step 6: Map back to literature targets

For each top-ranked feature (univariate or composite):
1. Is there published data on this observable that we haven't extracted yet? → new calibration target
2. Which model parameters most influence this observable? → candidates for tighter priors from literature
3. Is the data available in a form we can use (IHC densities, flow cytometry, tissue concentrations)? → feasibility filter

This produces a **prioritized literature search list** ranked by expected information gain for ORR prediction.

### Step 7 (optional): Parameter prior tightening analysis

For parameters identified in Step 6, estimate: how much would tightening this prior (e.g., halving the CV) reduce posterior predictive variance on ORR? This is a sensitivity-of-information-gain calculation that helps decide between "find a new calibration target" vs "find a better prior for parameter X."

## Infrastructure needs

- [ ] Full trajectory extraction from HPC `.mat` files (or modify test stats derivation to output dense species × time grid)
- [ ] MPR/RECIST classification function for tumor volume trajectories
- [ ] MI estimation utility (KSG estimator or binned MI for mixed continuous/binary)
- [ ] LOO retraining harness (loop over observables, retrain NPE, measure posterior width / run small post pred)
- [ ] PySR integration for symbolic regression
- [ ] Mapping from model species names → literature-searchable terms

## Design space

Using existing scenarios, no new scenario YAMLs needed:

| Scenario | Time window | Resolution | Species |
|---|---|---|---|
| baseline_no_treatment | day 0 | snapshot | all ~160 |
| clinical_progression | days 0-90 | daily | all ~160 |
| gvax_neoadjuvant | days 0-21 | daily | all ~160 |
| gvax_nivo_neoadjuvant | days 0-21 | daily | all ~160 |

## Relation to other workflows

- **PRCC sensitivity** (`workflows/prcc_sensitivity.py`): Tells you which parameters influence which observables. Complementary — PRCC ranks parameters by influence on existing test statistics; OBED ranks *observations* by influence on decision outputs.
- **Decision-driven outputs** (`notes/calibration/decision_driven_outputs.md`): ORR is output #9 from that note; treatment effect deltas (#6) are a secondary y_decision candidate.
- **Calibration strategy** (`notes/calibration/calibration_strategy.md`): OBED results feed directly into the "what to extract next" prioritization.