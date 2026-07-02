You are {PERSONA}. You are being asked to provide your best estimates for what you would expect to observe in the tumor microenvironment of a typical PDAC (pancreatic ductal adenocarcinoma) patient at diagnosis, BEFORE any treatment.

IMPORTANT RULES:
1. Do NOT look up, cite, or recall specific numbers from published papers.
2. You MUST show your quantitative derivation for each observable. Think step-by-step: start from anchor quantities you are confident about (e.g., total cellularity per mm², typical immune fraction of tissue, known cell ratios), then derive each estimate through explicit arithmetic. Show the chain of reasoning with actual numbers at each step.
3. Your derivation should reflect your particular expertise and perspective — different experts will weight different factors and arrive at different numbers. Trust your own reasoning even if it differs from what you think others might say.

Context — PDAC is:
- A "cold" tumor with poor T cell infiltration
- Highly desmoplastic (dense stroma, abundant fibroblasts and ECM)
- Immunosuppressive (M2-skewed macrophages, Tregs, MDSCs)
- Poorly vascularized with hypoxic regions

For each observable below:
1. Show your step-by-step derivation (starting from anchor quantities, with arithmetic)
2. State your point estimate (median across patients)
3. State a 95% confidence interval reflecting inter-patient variability and measurement uncertainty

Then at the END of your response, output a CSV block with columns: observable,estimate,ci95_lo,ci95_hi,units,reasoning
The reasoning column should be a 1-sentence summary of your derivation.

Observables to estimate:

CELL DENSITIES (cells per mm² of tumor tissue cross-section):
1. cancer_cell_density_baseline — PanCK+ tumor epithelial cells
2. cd8_density_baseline — CD8+ T cells (both effector and exhausted)
3. cd4_helper_density — CD4+ helper T cells (Th1 + Th2 + Th17, excluding Tregs)
4. treg_density_baseline — Foxp3+ regulatory T cells
5. cdc1_density_baseline — BATF3+/CLEC9A+ type 1 conventional dendritic cells
6. cdc2_density_baseline — IRF4+/CD1c+ type 2 conventional dendritic cells
7. tam_density — Total tumor-associated macrophages (CD68+, includes M1 and M2)
8. mdsc_density — Myeloid-derived suppressor cells
9. total_til_density — Total tumor-infiltrating lymphocytes (CD8 + CD4 helper + Treg)

FRACTIONS AND RATIOS (dimensionless):
10. cd8_exhausted_fraction — Fraction of CD8+ T cells expressing PD-1 (exhausted phenotype)
11. th_exhausted_fraction — Fraction of CD4+ helper T cells expressing PD-1
12. cd8_treg_ratio — Ratio of CD8+ T cells to Foxp3+ Tregs
13. cd8_treg_ratio_alt — Same ratio, second independent estimate
14. treg_fraction_cd4 — Tregs as fraction of all CD4+ cells (Treg + Th)
15. treg_fraction_cd4_alt — Same fraction, second independent estimate
16. treg_fraction_baseline — Foxp3+ cells as fraction of all CD3+ T cells
17. m1_m2_ratio — Ratio of M1 (iNOS+/CD80+) to M2 (CD163+/CD206+) macrophages
18. m2_macrophage_fraction — CD163+ macrophages as fraction of total macrophages
19. apc_maturation_ratio — Mature (DC-LAMP+) DCs as fraction of total DCs
20. icaf_fraction_of_caf — Inflammatory CAFs (FAP+/IL6+) as fraction of all CAFs
21. mycaf_fraction_of_caf — Myofibroblastic CAFs (αSMA+) as fraction of all CAFs
22. apcaf_fraction_of_caf — Antigen-presenting CAFs (MHC-II+) as fraction of all CAFs
23. apcaf_fraction_of_caf_alt — Same fraction, second independent estimate

STROMAL AND ECM:
24. stromal_fraction — Total stromal volume as fraction of tumor volume
25. psc_activation_ratio — Fraction of pancreatic stellate cells that are activated (αSMA+)
26. collagen1_protein_level — Collagen I protein level as fold-change relative to normal pancreas

OTHER:
27. vegf_tissue_concentration — VEGF-A concentration in tumor tissue (units: pg per mg total protein)