# Predictive Output Targets for Reducing Uncertainty

Status: Ideation (2026-03-09)

The model is structurally sound (posterior D²=22.4, p=0.455) but most constraints are on the diagnosis snapshot (baseline TME composition). The predictive outputs we'd actually use the model for are largely unconstrained.

## Tier 1: High-impact, feasible now

1. **Objective Response Rate (ORR)** — RECIST-style categorization (CR/PR/SD/PD) from virtual patient tumor volume trajectories. Published ORR data exists for multiple PDAC immunotherapy arms.

2. **Overall Survival (OS) / Median OS** — Requires defining a "death" threshold (tumor volume exceeding organ capacity or performance status proxy). Published median OS abundant: anti-PD-1 monotherapy (~3.6 mo) vs GVAX combinations (~20 mo) gives wide dynamic range.

3. **Progression-Free Survival (PFS)** — Time to 20% volume increase from nadir. Population-level but computable from individual trajectories.

4. **Major Pathologic Response (MPR)** — Partially in place (35% from Agarwal 2025), though confounded by unmodeled SBRT. Zheng 2022 surgical cohort may have resection pathology data.

## Tier 2: High-impact, needs some work

5. **Biomarker dynamics under treatment** — CD8 infiltration kinetics, PD-L1 expression changes, ctDNA clearance. Bridge mechanism to clinical response; disambiguate *why* patients respond.

6. **Responder vs non-responder stratification** — Can the model predict which TME archetypes (immune-hot/cold/excluded) respond? Key translational question for patient selection.

7. **Combination therapy differential** — GVAX mono vs GVAX+nivo vs GVAX+nivo+SBRT response rates. Constraining the *incremental benefit* of adding agents is more informative than absolute response rates.

## Tier 3: Aspirational

8. **Duration of response (DoR)** — How long responders maintain response
9. **Time to treatment failure** — Resistance emergence dynamics
10. **Dose-response relationships** — Nivo dose vs PD-1 receptor occupancy vs clinical benefit

## Recommended Starting Point

**OS and ORR across 2-3 treatment arms** — most bang for the buck:
- Endpoints clinicians and regulators care about
- Published data readily available (O'Reilly 2019 anti-PD-1 mono, Le 2013/2015 GVAX+ipi, Zheng 2022 GVAX+nivo)
- Require the model to get *integrated* biology right (immune priming → effector function → tumor kill → volume change → survival)
- Main implementation need: population-level aggregation (simulate N VPs, compute ORR fraction or KM curves, compare against trial statistics)