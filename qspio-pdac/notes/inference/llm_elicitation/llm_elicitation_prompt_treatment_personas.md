You are {PERSONA}. You are being asked to provide your best estimates for what you would expect to observe in PDAC (pancreatic ductal adenocarcinoma) patients under specific treatment scenarios.

IMPORTANT RULES:
1. Do NOT look up, cite, or recall specific numbers from published papers.
2. You MUST show your quantitative derivation for each observable. Think step-by-step: start from anchor quantities you are confident about, then derive each estimate through explicit arithmetic. Show the chain of reasoning with actual numbers at each step.
3. Your derivation should reflect your particular expertise and perspective — different experts will weight different factors and arrive at different numbers. Trust your own reasoning even if it differs from what you think others might say.

## Measurement context

All cell-type observables below (unless otherwise noted) are measured by **multiplex IHC (mIHC)** on resected PDAC tumor tissue sections, reported as **percentage of all nucleated cells** in non-lymphoid-aggregate (non-LA) tumor regions. The denominator is ALL nucleated cells in the tissue section (tumor epithelial + stromal + immune), not just immune cells.

PDAC tissue sections are dominated by tumor epithelial cells and stromal cells (fibroblasts, PSCs). Immune cells are a small minority of total cellularity. Keep this in mind when estimating — individual immune cell-type percentages of all nucleated cells will be much smaller than their percentages of immune cells alone.

**Consistency constraint**: Your estimates must satisfy:
  cd8_pct + cd4_pct + treg_pct + m1tam_pct + m2tam_pct < 0.15
(immune cells are a small minority of all nucleated cells in PDAC)

## Treatment context

- **GVAX**: Irradiated tumor cell vaccine secreting GM-CSF, administered subcutaneously with low-dose cyclophosphamide (Treg depletion). Recruits/activates dendritic cells, primes tumor-specific T cells, induces tertiary lymphoid aggregates.
- **Nivolumab**: Anti-PD-1 checkpoint inhibitor. Blocks PD-1/PD-L1 interaction, reinvigorates exhausted T cells. Minimal efficacy as PDAC monotherapy, but may synergize with vaccine priming.
- **Trial design**: Single neoadjuvant dose at day 7, surgical resection at day 21. mIHC performed on resected specimens. Small cohorts (n=9–12 per arm).

For each observable below:
1. Show your step-by-step derivation (starting from anchor quantities, with arithmetic)
2. State your point estimate (median across patients)
3. State a 95% confidence interval reflecting inter-patient variability and measurement uncertainty

Then at the END of your response, output a CSV block with columns: observable,estimate,ci95_lo,ci95_hi,units,reasoning
The reasoning column should be a 1-sentence summary of your derivation.

## Observables to estimate

### SCENARIO 1 — UNTREATED CLINICAL PROGRESSION:
1. **tumor_doubling_time** — Time for tumor volume to double in untreated PDAC (units: days).

### SCENARIO 2 — NEOADJUVANT GVAX MONOTHERAPY (single dose at day 7, resection at day 21):
All measurements from mIHC on resected tumor, reported as **fraction of all nucleated cells** in non-LA tumor regions. Express as decimal fractions (e.g., 0.02 = 2%).

2. **cd8_pct_nonLA_gvax_d21** — CD8+ T cells as fraction of all nucleated cells after GVAX (dimensionless)
3. **cd4_pct_nonLA_gvax_d21** — CD4+ T cells (helper + Treg combined) as fraction of all nucleated cells after GVAX (dimensionless)
4. **treg_pct_nonLA_gvax_d21** — Foxp3+ Tregs as fraction of all nucleated cells after GVAX (dimensionless). Low-dose cyclophosphamide partially depletes Tregs.
5. **cd8exh_pct_nonLA_gvax_d21** — PD-1+CD8+ T cells as fraction of all nucleated cells after GVAX (dimensionless). Note: numerator is PD-1+CD8+ cells, denominator is ALL nucleated cells.
6. **m1tam_pct_nonLA_gvax_d21** — M1-polarized macrophages (iNOS+/CD80+) as fraction of all nucleated cells after GVAX (dimensionless)
7. **m2tam_pct_nonLA_gvax_d21** — M2-polarized macrophages (CD163+) as fraction of all nucleated cells after GVAX (dimensionless)

### SCENARIO 3 — NEOADJUVANT GVAX + NIVOLUMAB (both given at day 7, resection at day 21):
Same measurement methodology as Scenario 2, but with added PD-1 blockade.

8. **cd8_pct_nonLA_gvax_nivo_d21** — CD8+ T cells as fraction of all nucleated cells after GVAX+nivo (dimensionless)
9. **cd4_pct_nonLA_gvax_nivo_d21** — CD4+ T cells as fraction of all nucleated cells after GVAX+nivo (dimensionless)
10. **treg_pct_nonLA_gvax_nivo_d21** — Tregs as fraction of all nucleated cells after GVAX+nivo (dimensionless)
11. **cd8exh_pct_nonLA_gvax_nivo_d21** — PD-1+CD8+ T cells as fraction of all nucleated cells after GVAX+nivo (dimensionless). Note: nivolumab blocks PD-1 signaling but anti-PD-1 antibody bound to the receptor may interfere with IHC detection of PD-1, potentially reducing apparent PD-1+ staining.
12. **m1tam_pct_nonLA_gvax_nivo_d21** — M1 macrophages as fraction of all nucleated cells after GVAX+nivo (dimensionless)
13. **m2tam_pct_nonLA_gvax_nivo_d21** — M2 macrophages as fraction of all nucleated cells after GVAX+nivo (dimensionless)
14. **cd8_fold_increase_gvax_nivo_d21** — Fold-increase in CD8+ T cell density relative to untreated baseline after GVAX+nivo (dimensionless).
15. **m1_m2_ratio_gvax_nivo_d21** — Ratio of M1 to M2 macrophages after GVAX+nivo (dimensionless).
16. **treg_fraction_gvax_nivo_d21** — Tregs as fraction of all CD3+ T cells after GVAX+nivo (dimensionless). Note: this denominator is CD3+ T cells only, not all nucleated cells.