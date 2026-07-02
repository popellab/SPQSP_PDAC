OK, here's my audit based on background knowledge. I'll organize by concern level.

  ---
  Prior Distribution Audit — parameters/pdac_priors.csv

  175 parameters total. I'll focus on the ones where medians or ranges look off from published biology, grouping by severity.

  ---
  RED FLAGS — Likely Needs Correction

  f_cDC1 = 0.6 (σ=0.3, bounds 0.33–1.09)
  cDC1 (BATF3+) dendritic cells are the minority DC subtype in most tissues and especially in PDAC, typically 10–20% of total DCs. cDC2 (IRF4+) are the dominant population. A median of 60%
   cDC1 is inverted from reality. Should be ~0.15–0.25, with cDC2 as the majority.

  k_IL2_deg = 0.2/day (σ=0.5, half-life ~3.5 days)
  IL-2 half-life in vivo is 5–10 minutes systemically. Even accounting for tissue-compartment effective clearance (receptor-mediated consumption + diffusion), a 3.5-day half-life is
  extremely slow. Most QSP models use effective IL-2 clearance rates of 5–50/day. This value could lead to unrealistic IL-2 accumulation.

  Q_nTreg_thym = 7e7 cell/day (σ=1.0, bounds 9.5e6–5.2e8)
  Total thymic output in adults is ~1e8 cells/day in young adults but drops to ~1e7/day by age 50–60 (typical PDAC demographic). Tregs are ~5–10% of thymic output, so realistic nTreg
  thymic output for PDAC patients would be ~5e5–1e6/day. This is ~100x too high.

  Q_nCD8_thym = 3.5e7 cell/day (σ=1.0)
  Same age concern. For age 50–60, total thymic output ~1e7/day, CD8 fraction ~30%, giving ~3e6/day. This is ~10x too high for the patient population.

  glc_50_Teff = 61.25 µM (σ=0.7)
  Normal tissue glucose is ~3–5 mM. Tumor glucose depletion reaches ~0.1–1 mM. T cell functional impairment begins below ~0.5–1 mM in most studies. A half-max of 61 µM (0.06 mM) means T
  cells would only be significantly impaired at near-zero glucose. Check how glc is defined — if it's a normalized quantity (0–1) rather than absolute concentration, this might be fine. If
   it's absolute µM, it's ~10x too low.

  ---
  YELLOW FLAGS — Potentially Problematic

  k_APC_death = 0.01/day (half-life ~69 days) & k_mAPC_death = 0.02/day (half-life ~35 days)
  Dendritic cell lifespan in vivo is typically 3–9 days (immature) and 2–4 days (mature, post-activation). These values give DC lifespans an order of magnitude too long. Would recommend
  k_APC_death ~0.1–0.2/day and k_mAPC_death ~0.2–0.5/day.

  CXCL12_50_Texcl = 0.025 nM (σ=1.2)
  CXCR4 Kd is ~1–5 nM. A half-max of 0.025 nM means even trace CXCL12 would fully activate T cell exclusion. This is ~100x below receptor Kd, which would make the exclusion mechanism
  essentially always-on. More plausible: 0.5–5 nM.

  IL10_50 = 8 nM (σ=0.7)
  IL-10 receptor Kd is ~0.1–1 nM. A half-max of 8 nM means the model requires supraphysiological IL-10 for half-maximal effect. This effectively weakens IL-10 immunosuppression. More
  typical: 0.5–2 nM.

  k_CD8_death = 0.01/day & k_CD4_death = 0.01/day (half-life ~69 days)
  These represent activated effector T cell death in the model. Effector T cells typically have half-lives of 3–7 days in vivo (~0.1–0.23/day). 0.01/day is more appropriate for naive or
  memory T cells. The prior bounds (0.004–0.027) don't even reach the expected range.

  k_MDSC_death = 0.015/day (half-life ~46 days)
  MDSC lifespan in tumors is typically days to ~2 weeks. A 46-day half-life seems 3–5x too slow. Expect ~0.05–0.15/day.

  k_NK_death = 0.02/day (half-life ~35 days)
  NK cell lifespan is typically 7–14 days in blood (~0.05–0.1/day). 35 days is quite long, though tissue-resident NK may persist longer.

  k_nCD8_death = 0.002/day & k_nTreg_death = 0.002/day (half-life ~347 days)
  Naive T cell homeostatic turnover with half-life ~1 year. This is actually reasonable for naive T cells in healthy individuals, which can persist for years. But combined with the
  inflated thymic output above, this would over-accumulate naive cells.

  ---
  MINOR CONCERNS — Worth Noting

  n_CD8_clones = 120 (σ=1.2, bounds 11–1322)
  PDAC has low TMB (~1.5 mut/Mb), yielding estimated neoantigens of 10–50. 120 tumor-specific clones is on the high side. The σ=1.2 gives a very wide range (up to 1322 clones), which seems
   excessive. σ=0.7 would be more appropriate.

  ECM_50_Kmax = 300 mg/mL (σ=1.2, bounds 27–3307)
  300 mg/mL = 30% ECM by weight. PDAC is highly desmoplastic, but the upper bound of 3307 mg/mL (330% by weight) is physically impossible. Tighter bounds or σ would help.

  k_Treg_act = 5.0/day (σ=1.0, upper bound 37/day)
  This is a fast activation rate, and the upper bound allows unrealistically fast Treg activation. Likely acceptable if modulated by antigen/APC availability.

  k_vas_growth = 7.37/day (σ=1.0, upper bound 54.5/day)
  This appears to be an antigen/growth-related rate. Upper bound seems very high.

  ---
  LOOKS GOOD — Major Categories Passing Audit

  - Cancer growth: k_C1_growth = 0.0066/day (~105 day doubling time) — reasonable for PDAC
  - TGF-beta axis: TGFb_50 = 0.07 nM, TGFb_baseline = 0.016 nM — consistent with active TGF-beta biology
  - IFN-gamma degradation: k_IFNg_deg = 7.68/day (~2.2 hr half-life) — matches in vivo data
  - VEGF axis: k_vas_Csec, k_vas_deg = 16.6/day (~1 hr half-life), VEGF_50 = 2410 pg/mL — all reasonable
  - Macrophage dynamics: k_M2_pol, k_M1_pol (M2-skewed), k_Mac_death = 0.02/day — appropriate for PDAC TME
  - CAF biology: Proliferation rates, death rates, interconversion (iCAF⇄myCAF) — all reasonable for desmoplastic stroma
  - CAF activation: myCAF > iCAF >> apCAF activation rates — matches biology
  - ECM dynamics: k_ECM_deg = 0.035/day, ECM_base = 60 mg/mL — fits PDAC desmoplasia
  - CCL2/CCL5: Degradation rates, EC50 values — reasonable
  - Hypoxia: pO2_50 = 7 mmHg for HIF stabilization — correct range
  - PD-L1: PDL1_C1_baseline = 120,000 molecules/cell, r_PDL1_IFNg = 6x — both in literature range
  - Lactate: lac_50_M2 = 16 mM — matches functional threshold data
  - GM-CSF: k_GMCSF_deg = 5/day (~3.3 hr half-life), EC50_GMCSF = 0.5 nM — reasonable
  - IL-2 signaling: IL2_50 = 0.32 nM — appropriate for mixed high/low affinity receptor

  ---
  Summary

  ┌──────────────┬───────┬──────────────────────────────────────────────────────────────────────────┐
  │   Category   │ Count │                                  Notes                                   │
  ├──────────────┼───────┼──────────────────────────────────────────────────────────────────────────┤
  │ Red flags    │ 5     │ f_cDC1, k_IL2_deg, thymic outputs (×2), possibly glc_50_Teff             │
  ├──────────────┼───────┼──────────────────────────────────────────────────────────────────────────┤
  │ Yellow flags │ 6     │ DC death rates, CXCL12_50, IL10_50, effector T cell death, MDSC/NK death │
  ├──────────────┼───────┼──────────────────────────────────────────────────────────────────────────┤
  │ Minor        │ 4     │ Clone number σ, ECM upper bound, fast activation rates                   │
  ├──────────────┼───────┼──────────────────────────────────────────────────────────────────────────┤
  │ Looks good   │ ~160  │ Majority of parameters are reasonable ballpark values                    │
  └──────────────┴───────┴──────────────────────────────────────────────────────────────────────────┘

  The biggest structural issue is that many immune cell death/turnover rates are too slow by ~5–10x, which collectively would make the TME too static. The thymic output values are set for
  a young adult rather than the 60+ PDAC population. And f_cDC1 being inverted is a clear error that would misroute antigen presentation.

 i want you to do a brief audit of the parameters in the priors csv. no web search or zotero or anything, just your own background knowledge - I want you do identify if they are good     
                                                                                                                                                                                            
     ballpark distributions for the almost 200 parameters in there. just keep it to you though, dont launch extra agents or antyhng. btw, i just did a parameter audit in                   
  parameter_audit.md. do you agree with those conclusions?                                                                                                                                  
                                                                                                                                                                                            
⏺ Let me read both files.                                                                                                                                                                   
                                                                                                                                                                                            
⏺ Searched for 4 patterns, read 2 files (ctrl+o to expand)                                                                                                                                  
                                                                                                                                                                                            
⏺ Now let me read the parameters file for cross-reference on a few things.                                                                                                                  
                                                                                                                                                                                            
⏺ Searched for 2 patterns (ctrl+o to expand)                                                                                                                                                
                                                                                                                                                                                            
⏺ Critical find — let me check the IL-2 units discrepancy more carefully.                                                                                                                   
                                                                                                                                                                                            
⏺ Searched for 2 patterns (ctrl+o to expand)                                                                                                                                                
                                                                                                                                                                                            
⏺ This is very informative. Here's my assessment:                                                                                                                                           
                                                                                                                                                                                            
  ---                                                                                                                                                                                       
  My verdict on your audit                                                                                                                                                                  
                                                                                                                                                                                            
  I largely agree with your conclusions, but I have a few additions and corrections:                                                                                                        
                  
  k_IL2_deg — This is actually a unit mismatch bug, not just a bad value

  Your audit flags k_IL2_deg = 0.2/day as having an implausibly long half-life. You're right that 0.2/day is wrong — but the root cause is more specific: parameters_PDAC.m:338-339 defines
  it as 0.2/minute (half-life ~3.5 min, consistent with Lotze 1985 IL-2 PK). The priors CSV has it as 0.2 1/day — a 1440x discrepancy. This is a copy-paste unit error in the priors CSV,
  not a modeling judgment issue.

  glc_50_Teff — I'd soften from red to yellow

  The parameter notes at line 2215 explain the intent: glc_50_Teff = 61.25 µM = 5% of normal tissue glucose (1225 µM). The model defines "normal tissue glucose" as 1225 µM (~1.2 mM), not
  the blood value of 5 mM — this is reasonable for interstitial fluid. However, the notes themselves say "T cells require ~1 mM glucose for optimal IFNg production," yet at 1 mM the Hill
  function gives ~94% function (barely impaired). So there's an internal inconsistency: if the intent is to capture impairment starting around 1 mM, the EC50 should be ~500–1000 µM, not 61
   µM. But it's not as clear-cut as 10x off — it depends on whether the modeler intended "only extreme depletion matters."

  IL12_50 = 0.14 pM — Your audit missed this, and I'd flag it

  IL-12 receptor Kd (IL-12Rβ1/β2 complex) is ~5–50 pM. An EC50 of 0.14 pM is 30–300x below receptor Kd, meaning the model treats IL-12 signaling as essentially always saturated if any
  IL-12 is present. Compare with IL12_50_NK = 0.05 nM (50 pM) on line 170, which is much more biologically reasonable. The comment on line 1404 (2g/L = 2/7e4 pg/mL (2.86e-5 M)) is
  dimensionally incoherent — suggests a calculation error in deriving this value.

  k_C1_death = 0.0001/day — Also missed, and worth noting

  This gives a cancer cell spontaneous death half-life of ~19 years. Even accounting for immune-mediated killing being the dominant death pathway, spontaneous/baseline cancer cell death
  (anoikis, nutrient deprivation, replication errors) is typically 0.001–0.01/day. This means the model relies 100% on immune killing for tumor control — if immune responses are weak,
  tumor growth is essentially unchecked by intrinsic death. Might be intentional modeling choice, but worth flagging.

  Things I agree with strongly

  ┌────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │            Flag            │                                                             My take                                                             │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ f_cDC1 = 0.6               │ Agree, clearly inverted. cDC1 (BATF3+) are the minority subtype, especially in PDAC TME. Should be ~0.15–0.25.                  │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Q_nTreg_thym / Q_nCD8_thym │ Agree, too high for PDAC demographic (age 60+). Thymic involution reduces output by ~10x from the young-adult values used here. │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ k_APC_death / k_mAPC_death │ Agree, DC lifespans of 35–69 days are ~10x too long.                                                                            │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ k_CD8_death / k_CD4_death  │ Agree, 0.01/day is naive/memory T cell turnover, not activated effector.                                                        │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ CXCL12_50_Texcl = 0.025 nM │ Agree, ~100x below CXCR4 Kd. Would make T cell exclusion always-on.                                                             │
  ├────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ ECM_50_Kmax upper bound    │ Agree, 3307 mg/mL is physically impossible (>100% tissue weight).                                                               │
  └────────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Things I'd slightly push back on

  - IL10_50 = 8 nM: You say it should be 0.5–2 nM based on receptor Kd of 0.1–1 nM. But functional EC50 for IL-10 effects on immune cells (suppression of TNF, IL-12 production, etc.) is
  often 1–5 nM in cell-based assays, sometimes higher depending on the readout. 8 nM is high-ish but not as extreme as 8x over — I'd call it mildly elevated (maybe 2–3x over the functional
   EC50).
  - k_NK_death = 0.02/day: You flag this as a yellow. NK cell biology is heterogeneous — CD56dim circulating NK have 7–14 day half-lives, but CD56bright and tissue-resident NK can persist
  much longer. In PDAC tumors with immunosuppressive TME, NK turnover might be slower than blood. I'd call this borderline acceptable.

  Summary of my additions

  ┌────────────┬───────────────────────────────────────────────────────────────────┬────────────┐
  │ Parameter  │                               Issue                               │  Severity  │
  ├────────────┼───────────────────────────────────────────────────────────────────┼────────────┤
  │ k_IL2_deg  │ Unit mismatch: 0.2/min in params file vs 0.2/day in priors        │ Red (bug)  │
  ├────────────┼───────────────────────────────────────────────────────────────────┼────────────┤
  │ IL12_50    │ 0.14 pM is ~100x below IL-12R Kd; essentially always saturated    │ Yellow-Red │
  ├────────────┼───────────────────────────────────────────────────────────────────┼────────────┤
  │ k_C1_death │ 0.0001/day = 19-year intrinsic half-life; no baseline tumor death │ Yellow     │
  └────────────┴───────────────────────────────────────────────────────────────────┴────────────┘

  Overall

  Your audit is solid. ~160/175 parameters passing is consistent with what I'd expect — the QSPIO framework these come from is well-calibrated for most standard immunology. The issues
  cluster in: (1) PDAC-specific adaptations that weren't fully adjusted (thymic output for elderly patients, f_cDC1), (2) immune cell turnover rates that are systematically too slow (DCs,
  effectors, MDSCs), and (3) a few EC50/Kd values that are off from receptor pharmacology. The structural pattern you identify — "TME too static" due to slow death rates — is the most
  impactful systematic issue
