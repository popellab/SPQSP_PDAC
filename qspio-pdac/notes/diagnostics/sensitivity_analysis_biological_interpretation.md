# Biological Interpretation of Sensitivity Analysis Results

**Date:** 2026-02-21
**Context:** 97/177 parameters significant at FDR < 0.05 (permutation null test, 1000 perms)
**Scenario:** `baseline_no_treatment` (diagnosis snapshot)

## Key Biological Findings

### Complete CAF plasticity circuit is significant (~24 params)

Not just myCAF proliferation and ECM secretion — all three interconversion pathways (iCAF↔myCAF, PSC→iCAF/myCAF/apCAF), proliferation, and death for each subtype. The model genuinely cares about *which* CAF subtype dominates, not just total CAF numbers. This aligns with Biffi 2019 — the iCAF/myCAF balance is TGFb-driven and determines whether the stroma is inflammatory (iCAF, IL-6) vs desmoplastic (myCAF, ECM/CXCL12).

### apCAF→Treg induction axis made the cut

`k_apCAF_Treg` (rank 42) and `k_PSC_to_apCAF` (rank 46) are both significant. This is Elyada 2019's mechanism — apCAFs presenting antigen on MHC-II to induce Tregs rather than activate effectors. The fact that it's detectable at diagnosis (no treatment) means it's not just a theoretical add-on — apCAF-driven Treg induction contributes measurably to the baseline immunosuppressive TME.

### Every step of the CD8 trafficking chain is significant

`q_nCD8_P_out` → `q_nCD8_LN_in` → `k_T_activation_per_APC` → `n_T_expansion_burst` → `q_CD8_T_in` → `k_CD8_death`. The bottleneck isn't at any single step — the entire supply chain from thymus to tumor matters. Meanwhile, `p_T_kill_per_contact` (rank 171) is *not* significant, confirming that the problem in PDAC isn't killing efficiency, it's getting T cells there in the first place.

### CXCL12 exclusion axis is significant

`CXCL12_50_Texcl` (rank 23) and `k_CXCL12_sec_myCAF` (rank 40). This is the Feig 2013 mechanism — myCAF-derived CXCL12 creating a chemokine barrier that excludes T cells. It validates the model's representation of the "immune desert" phenotype and suggests CXCR4 inhibitors (e.g., AMD3100/plerixafor) would genuinely shift model behavior.

### Metabolism module has real teeth

9 parameters significant, including glucose competition (`glc_50_C`, `k_glc_tumor`, `k_glc_immune`), lactate accumulation (`k_lac_Csec`, `k_lac_clear`), and hypoxia (`pO2_50`, `k_O2_demand`, `f_HIF_glycolysis`). This isn't a decorative module — metabolic competition genuinely shapes the TME at diagnosis. The Warburg effect (tumor glucose hogging → immune starvation) is a real driver in the model.

### Two highly nonlinear mechanisms

Big RF/PRCC discrepancy:
- `k_M2_pol` — rank 3 by RF, rank 26 by PRCC. M1→M2 polarization is switch-like, probably because M2 macrophages create a positive feedback loop (M2 → IL-10 → suppress M1 → more M2 → more TGFb/VEGF).
- `k_cell_clear` — rank 6 by RF, rank 22 by PRCC. Dead cell clearance → DAMP release → APC maturation. There's likely a threshold: below some clearance rate, DAMPs accumulate enough to trigger meaningful antigen presentation. This is essentially the immunogenic cell death loop.

### n_CD8_clones (rank 25) is a proxy for TMB

The number of tumor-reactive clones determines how many naive T cells in the LN can be activated. It's significant — the model's immune response scales with neoantigen diversity, consistent with the TMB-response correlation seen clinically.

### PD1_50 significant but low-ranked (rank 83)

Even without treatment, constitutive PD-L1 expression (baseline + IFNg-driven upregulation) has a detectable effect on CD8 exhaustion. This is a good sign — it means the checkpoint axis is wired in correctly and will become much more important under anti-PD-1 therapy.

### Notable absences

The CXCR3/chemotaxis module parameters (`k_CXCR3_rec`, `k_CXCL9_sec`, `k_CCR5_Treg_rec`) are all non-significant. The newer chemokine-mediated recruitment mechanisms are secondary to the baseline trafficking rates (`q_CD8_T_in`, `q_Treg_T_in`). Either the chemotaxis module needs wider priors, or at baseline the constitutive trafficking dominates the IFNg-inducible recruitment. This might change under treatment when IFNg levels spike.