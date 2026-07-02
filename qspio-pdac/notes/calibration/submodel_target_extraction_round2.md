# SubmodelTarget Extraction — Round 2

OBED iteration 2 (2026-03-12) identified these parameters as top priorities for prior tightening via IQR-based dispersion reduction on treatment outcome.

## Checklist

- [ ] **k_IL2_sec** (ΔDisp=+30.1%, σ=0.7) — IL-2 secretion rate by T cells
  - Current: 3e-5 nmol/cell/hour, σ=0.7
  - Paper: Han et al. 2010 (DOI: 10.1039/b926849a) — single-cell microengraving, ~5-20 molecules/s
  - Paper: Busse et al. 2010 (DOI: 10.1073/pnas.0812851107) — ODE model, 5000 molecules/cell/hour; Thurley 2015 refined to ~7200 effective

- [ ] **Q_nTreg_thym** (ΔDisp=+28.6%, σ=1.0) — Thymic output of naive CD4 T cells
  - Current: 1e6 cell/day, σ=1.0
  - Paper: Bains et al. 2009 (DOI: 10.1182/blood-2008-10-184184) — age-dependent formula, ~6.8×10⁷ CD4/day at age 30
  - Paper: den Braber et al. 2012 (DOI: 10.1016/j.immuni.2012.01.014) — thymic output half-life 15.7 years; age 65 estimate ~3-5×10⁶/day

- [ ] **k_TGFb_Tsec** (ΔDisp=+25.4%, σ=0.7) — TGFβ secretion rate by T cells
  - Current: 1.2e-10 nmol/cell/day, σ=0.7
  - Paper: Stockis et al. 2009 (DOI: 10.1002/eji.200939684) — stimulated Tregs ~200 pg/10⁶ cells/day

- [ ] **TGFb_50_APC** (ΔDisp=+25.3%, σ=0.7) — TGFβ EC50 for APC maturation inhibition
  - Current: 0.07 nM, σ=0.7
  - Paper: Fogel-Petrovic et al. 2007 (DOI: 10.1016/j.intimp.2007.07.003) — 3-point dose-response on DCs, EC50 ~0.02-0.04 nM

- [ ] **rho_Th_max** (ΔDisp=+25.0%, σ=0.8) — Max Th density in tumor
  - Current: 5e6 cell/mL, σ=0.8
  - Paper: Wartenberg et al. 2018 (DOI: 10.1158/1078-0432.CCR-17-3401) — max stromal CD4 ~1268 cells/mm² (PDAC, n=110)
  - Paper: Karakhanova et al. 2015 (DOI: 10.1038/bjc.2015.72) — median CD4+ ~229 cells/mm² (PDAC)
