# Decision-Driven Predictive Outputs

Status: Ideation (2026-03-09)

Predictive outputs organized by the decisions they inform, not by endpoint type. These guide which model inputs are most needed for calibration.

## Goal 1: Patient Stratification

The question: given a patient's baseline TME profile at diagnosis, can the model predict who will respond to a given regimen?

### Primary outputs

1. **P(RECIST response | baseline TME vector)** — The conditional response probability for a specific patient, not the population-average ORR. Requires simulating each virtual patient under treatment and classifying by RECIST from tumor volume trajectory. The *mapping* from baseline features to response is the deliverable, not just the marginal rate.

2. **Baseline feature sensitivity of response** — Which TME features most strongly modulate P(response)? Partial dependence or SHAP-like decomposition over the posterior predictive. Candidates:
   - CD8 density (effector availability)
   - Treg density / CD8:Treg ratio (immunosuppressive brake)
   - myCAF fraction / stromal fraction (physical barrier to infiltration)
   - M1/M2 ratio (innate immune tone)
   - cDC1 density (cross-presentation capacity → CD8 priming)
   - PD-L1 expression proxy (target availability for nivo)

3. **Virtual patient archetype clusters** — Do posterior predictive virtual patients naturally cluster into TME archetypes (immune-hot, immune-excluded, immune-desert, stroma-dominant)? If so, what are the outcome distributions per archetype? This connects mechanistic QSP output to the clinical taxonomy oncologists already use.

### Supporting outputs

4. **Time-to-response distribution** — Among responders, how quickly does the tumor begin shrinking? Distinguishes fast-priming (pre-existing immunity reinvigorated) from slow-priming (de novo immune response via GVAX).

5. **Probability of durable response** — P(response sustained at 6 months | initial response). Separates transient tumor shrinkage from immunological control.

## Goal 2: Novel Regimen Confidence

The question: when the model predicts outcomes for untested combinations, how much should we trust those predictions?

### Primary outputs

6. **Treatment effect deltas (not absolutes)** — Fold-change or absolute shift in key TME metrics when adding an agent. More trustworthy than absolute predictions because systematic model biases cancel:
   - CD8 fold-change: GVAX+nivo vs GVAX mono
   - M1/M2 ratio shift: combination vs mono
   - Treg depletion depth: with vs without cyclophosphamide
   - Exhausted CD8 fraction change: with vs without PD-1 blockade

7. **Combination interaction type** — For each pair of agents, is the predicted tumor volume reduction superadditive, additive, or subadditive? Tests whether the model captures mechanistic synergy (GVAX primes → nivo reinvigorates) vs simple additivity. Decompose as:
   - Effect_combo vs Effect_A + Effect_B (Bliss independence)
   - Identify which mechanism drives the interaction (e.g., exhaustion reversal, Treg depletion, APC maturation)

8. **Trajectory divergence timing** — When does the combination regimen's tumor volume trajectory separate from monotherapy? Early divergence (days 7-14) suggests direct immune activation; late divergence (days 14-21+) suggests adaptive immune response buildup. Testable against serial imaging or biopsy data if available.

9. **Predicted ORR for hypothetical regimens** — Population-level response rates for untested combinations:
   - GVAX + anti-CTLA4 (ipilimumab) — different checkpoint, Treg depletion mechanism
   - GVAX + anti-PD-L1 (instead of anti-PD-1) — same axis, different target
   - Higher/lower nivo dose — dose-response relationship
   - Modified GVAX timing (day 0 vs day 7 vs day 14) — priming window

   Confidence comes from correctly capturing the *known* GVAX mono → GVAX+nivo delta first.

### Supporting outputs

10. **Parameter sensitivity of treatment deltas** — Which model parameters most influence the predicted combination benefit? If the delta depends on well-constrained parameters, confidence is high. If it depends on unconstrained parameters, that identifies the next calibration priority.

11. **Extrapolation uncertainty quantification** — Posterior predictive intervals should widen for regimens far from calibration data. If they don't, the model is overconfident. Compare posterior width for:
    - GVAX+nivo (calibrated) — should be narrow
    - GVAX+anti-CTLA4 (related mechanism, uncalibrated) — should be wider
    - Novel target (e.g., anti-CD47) — should be widest

## Implementation Strategy

Both goals require the same infrastructure:
1. **Population-level simulation**: Sample N virtual patients from the posterior, simulate each under treatment, aggregate into population statistics
2. **Conditional analysis**: Stratify virtual patients by baseline features, compute outcome distributions per stratum
3. **Sensitivity analysis**: Vary one baseline feature (or parameter) at a time, measure impact on outputs above

The hierarchical SBI approach (learning population μ, ω rather than individual θ) naturally supports this — each posterior draw defines a population, and the outputs above are population-level functionals.

## Priority Ordering

For maximum information gain with minimum implementation effort:

1. **Treatment effect deltas (#6)** — already have GVAX mono and GVAX+nivo scenarios; just need posterior predictive comparison
2. **P(response | baseline TME) (#1)** — needs RECIST classification from volume trajectories + conditional analysis
3. **Parameter sensitivity of deltas (#10)** — identifies next calibration priorities, closing the loop
4. **Combination interaction type (#7)** — tests mechanistic understanding, builds regimen extrapolation trust
5. **Archetype clustering (#3)** — high translational value but requires more analysis infrastructure
