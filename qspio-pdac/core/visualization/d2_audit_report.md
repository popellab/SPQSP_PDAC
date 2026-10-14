# D2 Schematic Audit Report

**Date:** 2026-03-04
**Source:** `model_structure.json` (exported via `qsp-export-model`) vs `qsp_schematic.d2`

---

## Critical Errors

### 1. `injection.mAPC -> dln.mcDC1` should be `-> dln.mcDC2`

- **D2 line 244:** `injection.mAPC -> dln.mcDC1`
- **Model reaction** `mAPC_migration_ID_LN`: `V_ID.mAPC -> V_LN.mcDC2`
- The injection site APCs become **mcDC2** in the dLN, not mcDC1.

### 2. `tumor.myCAF -> tumor.CXCL12` is wrong — should be `tumor.iCAF -> tumor.CXCL12`

- **D2 line 224:** `tumor.myCAF -> tumor.CXCL12`
- **Model:** CXCL12 is produced by **iCAF** (`Reaction_263`: `k_CXCL12_sec_iCAF*V_T.iCAF`) and cancer (`Reaction_264`: `k_CXCL12_sec_cancer*C_total`). myCAF does NOT produce CXCL12.

### 3. `tumor.iCAF -> tumor.CCL2` is wrong — should be `tumor.Cancer -> tumor.CCL2`

- **D2 line 227:** `tumor.iCAF -> tumor.CCL2`
- **Model:** CCL2 is produced only by **cancer** (`Reaction_203`: `k_CCL2_sec*C_total`). iCAF does NOT produce CCL2.

### 4. `tumor.IL6 -> tumor.Cancer` is wrong

- **D2 line 229:** `tumor.IL6 -> tumor.Cancer`
- **Model:** IL-6 does NOT directly affect cancer. Instead it modulates:
  - MDSC recruitment (`Reaction_205`: `H_IL6_MDSC`)
  - M1→M2 polarization (`Reaction_224`: `H_IL6_M2`)
  - iCAF differentiation (`Reaction_243`, `246`: `H_IL6_iCAF`)
- This edge should be removed or redirected to the actual targets.

### 5. `peripheral <-> dln` transport is wrong

- **D2 lines 120–121:** `peripheral.nTCD4 <-> dln.nTCD4` and `peripheral.nTCD8 <-> dln.nTCD8`
- **Model:** There is NO direct peripheral↔dLN transport. Naive T cells route **peripheral ↔ central ↔ dLN**. The correct peripheral↔central edges already exist (lines 113–117).

---

## Missing Connections

These connections exist in the model but are absent from the D2 schematic.

### 6. `tumor.Teff -> tumor.IFNg`

CD8+ T cells produce IFNγ (`Reaction_71`: `k_IFNg_Teff*V_T.CD8`). Currently only `tumor.Th -> tumor.IFNg` is shown.

### 7. `tumor.iCAF -> tumor.CCL5`

iCAF produces CCL5 (`Reaction_268`: `k_CCL5_sec_iCAF*V_T.iCAF`). Currently only `tumor.Cancer -> tumor.CCL5` is shown.

### 8. `tumor.iCAF -> tumor.CXCL12`

See error #2 above; needs to replace the myCAF attribution.

### 9. `tumor.Cancer -> tumor.CXCL12`

Cancer also produces CXCL12 (`Reaction_264`: `k_CXCL12_sec_cancer*C_total`).

### 10. `tumor.qPSC -> tumor.apCAF`

qPSC differentiates to apCAF (`Reaction_244`). D2 only shows qPSC→myCAF and qPSC→iCAF.

### 11. `tumor.mcDC1 -> tumor.IL12`

Mature cDC1 produces IL-12 in the tumor (`Reaction_218`). D2 only shows `tumor.M1 -> tumor.IL12`.

### 12. `tumor.IL6 -> tumor.MDSC`

IL-6 enhances MDSC recruitment (`Reaction_205`: `H_IL6_MDSC`).

### 13. `tumor.IL6 -> tumor.M2`

IL-6 promotes M1→M2 polarization (`Reaction_224`: `H_IL6_M2`).

### 14. `tumor.IL6 -> tumor.iCAF`

IL-6 promotes iCAF phenotype via positive feedback (`Reaction_243`, `246`: `H_IL6_iCAF`).

### 15. `tumor.PDGF -> tumor.myCAF/iCAF/apCAF`

PDGF modulates all CAF subtypes (`Reactions 242, 243, 247, 248, 249`). D2 only shows `PDGF -> qPSC`.

### 16. `tumor.TGFb -> tumor.myCAF/iCAF`

TGFβ modulates CAF differentiation (`Reactions 242, 243, 245, 246`). D2 only shows `TGFb -> qPSC`.

### 17. `dln.mcDC1 -> dln.aCD8`

mcDC1 primes activated CD8 (`Reaction_53`). D2 shows `mcDC1 -> nTCD8` which is a simplification — the model has mcDC1 driving activation of nCD8→aCD8, not just naive T cell recruitment.

---

## Missing Species

Notable species present in the model but absent from the D2 schematic.

### 18. `tumor.VEGF`

Produced by cancer, M2 macrophages, and myCAF. Regulates tumor vasculature/carrying capacity (K). Biologically significant for tumor growth dynamics.

### 19. `tumor.Th_exh`

Th cell exhaustion exists in the model (`Reactions 191–192`) but isn't shown. The D2 has Texh (CD8 exhaustion) but not Th exhaustion.

### 20. `tumor.P0`, `tumor.P1`

Antigen peptides that mediate DC cross-presentation and T cell priming. Critical for the GVAX mechanism of action.

### 21. `dln.cDC1`, `dln.cDC2`

Immature DCs exist in the dLN in the model but are not shown.

---

## Acceptable Simplifications

These omissions are reasonable for schematic clarity:

- **Antibody species** (aPD1, aPDL1, aCTLA4) across compartments — OK to omit
- **Synapse compartments** (syn_CD8_C1, syn_CD8_APC, syn_M_C) — correctly abstracted
- **Endosomal/surface compartments** (A_e, A_s, V_e) — correctly abstracted
- **`tumor.Debris`** in D2 abstracting C_x, P0, P1 collectively — reasonable
- **`tumor.IL10 -> tumor.cDC2`** shown as inhibition — IL-10 actually inhibits **both** cDC1 and cDC2 maturation (same `H_IL10` term in both `Reaction_76` and `Reaction_83`), so the edge is correct but incomplete