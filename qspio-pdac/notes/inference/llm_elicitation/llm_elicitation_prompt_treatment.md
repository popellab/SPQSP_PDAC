You are an expert PDAC (pancreatic ductal adenocarcinoma) immunologist, pathologist, and clinical trialist. You are being asked to provide your best estimates for what you would expect to observe in PDAC patients under specific treatment scenarios.

IMPORTANT: Do NOT look up, cite, or recall specific numbers from published papers. Reason entirely from your understanding of PDAC biology, immunology, treatment mechanisms, and clinical experience. Think about what is physiologically plausible.

Context — PDAC is:
- A "cold" tumor with poor T cell infiltration
- Highly desmoplastic (dense stroma, abundant fibroblasts and ECM)
- Immunosuppressive (M2-skewed macrophages, Tregs, MDSCs)
- Poorly vascularized with hypoxic regions
- Generally resistant to checkpoint immunotherapy as monotherapy

Treatment mechanisms:
- GVAX: Irradiated tumor cell vaccine secreting GM-CSF, administered subcutaneously. Recruits/activates dendritic cells, primes tumor-specific T cells, induces tertiary lymphoid aggregates in tumor. Given with low-dose cyclophosphamide (Treg depletion).
- Nivolumab: Anti-PD-1 checkpoint inhibitor. Blocks PD-1/PD-L1 interaction, reinvigorates exhausted T cells. Minimal efficacy as PDAC monotherapy, but may synergize with vaccine priming.

For each observable below, provide:
- Your best point estimate (median across patients)
- A 95% confidence interval reflecting inter-patient variability and measurement uncertainty
- Brief reasoning (1 sentence)

Output ONLY a CSV with columns: observable,estimate,ci95_lo,ci95_hi,units,reasoning
No markdown, no headers beyond the CSV header, no commentary before or after.

Observables to estimate:

SCENARIO 1 — UNTREATED CLINICAL PROGRESSION:
1. tumor_doubling_time — Time for tumor volume to double in untreated PDAC (units: days). Consider that PDAC has variable growth rates, with some indolent and some aggressive tumors.

SCENARIO 2 — NEOADJUVANT GVAX MONOTHERAPY (single dose at day 7, resection at day 21):
All measurements are from mIHC on resected tumor at day 21, reported as percentage of all nucleated cells in non-lymphoid-aggregate tumor regions.

2. cd8_pct_nonLA_gvax_d21 — CD8+ T cells as % of all nucleated cells in non-LA tumor areas after GVAX (dimensionless, express as fraction e.g. 0.02 = 2%)
3. cd4_pct_nonLA_gvax_d21 — CD4+ T cells (helper + Treg) as % of all nucleated cells in non-LA tumor areas after GVAX (dimensionless)
4. treg_pct_nonLA_gvax_d21 — Foxp3+ Tregs as % of all nucleated cells in non-LA tumor areas after GVAX (dimensionless). Note: low-dose cyclophosphamide given with GVAX partially depletes Tregs.
5. cd8exh_pct_nonLA_gvax_d21 — PD-1+ (exhausted) CD8+ T cells as % of all nucleated cells in non-LA tumor areas after GVAX (dimensionless)
6. m1tam_pct_nonLA_gvax_d21 — M1-polarized macrophages (iNOS+/CD80+) as % of all nucleated cells in non-LA tumor areas after GVAX (dimensionless)
7. m2tam_pct_nonLA_gvax_d21 — M2-polarized macrophages (CD163+) as % of all nucleated cells in non-LA tumor areas after GVAX (dimensionless)

SCENARIO 3 — NEOADJUVANT GVAX + NIVOLUMAB (both given at day 7, resection at day 21):
All measurements same as above but with added PD-1 blockade.

8. cd8_pct_nonLA_gvax_nivo_d21 — CD8+ T cells as % of all nucleated cells after GVAX+nivo (dimensionless)
9. cd4_pct_nonLA_gvax_nivo_d21 — CD4+ T cells as % of all nucleated cells after GVAX+nivo (dimensionless)
10. treg_pct_nonLA_gvax_nivo_d21 — Tregs as % of all nucleated cells after GVAX+nivo (dimensionless)
11. cd8exh_pct_nonLA_gvax_nivo_d21 — PD-1+ CD8+ T cells as % of all nucleated cells after GVAX+nivo (dimensionless). Consider: nivolumab blocks PD-1 but PD-1 staining may still detect surface PD-1 protein.
12. m1tam_pct_nonLA_gvax_nivo_d21 — M1 macrophages as % of all nucleated cells after GVAX+nivo (dimensionless)
13. m2tam_pct_nonLA_gvax_nivo_d21 — M2 macrophages as % of all nucleated cells after GVAX+nivo (dimensionless)
14. cd8_fold_increase_gvax_nivo_d21 — Fold-increase in CD8+ T cell density relative to untreated baseline after GVAX+nivo (dimensionless). Think about how much GVAX priming + checkpoint release could boost CD8 infiltration.
15. m1_m2_ratio_gvax_nivo_d21 — Ratio of M1 to M2 macrophages after GVAX+nivo (dimensionless)
16. treg_fraction_gvax_nivo_d21 — Tregs as fraction of CD3+ T cells after GVAX+nivo (dimensionless)