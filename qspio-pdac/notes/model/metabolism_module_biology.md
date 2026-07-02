# Metabolic Competition Module: Biological Rationale

## The Warburg Effect in PDAC

Normal cells metabolize glucose through oxidative phosphorylation (efficient, ~36 ATP/glucose). Cancer cells — especially PDAC — preferentially use aerobic glycolysis even when oxygen is available: they consume glucose rapidly but ferment it to lactate (inefficient, only 2 ATP/glucose). This is the Warburg effect. PDAC is one of the most glycolytic solid tumors, driven by oncogenic KRAS signaling which upregulates glucose transporters (GLUT1) and glycolytic enzymes (HK2, LDHA).

The consequence: PDAC tumors consume glucose at 5-10x the rate of surrounding tissue, creating a glucose-depleted microenvironment (measured at ~0.5-2 mM vs ~5 mM in normal tissue) while flooding the TME with lactate (10-20 mM vs ~1.5 mM normal).

## Glucose Depletion Impairs T Cell Function

This is the key finding from Chang et al. 2015 and Ho et al. 2015 (both published in Cell, same issue). T cells, upon activation, undergo their own metabolic switch to aerobic glycolysis — they need glucose to fuel:

- **Effector cytokine production**: IFNg mRNA translation is directly controlled by GAPDH, which moonlights as an RNA-binding protein. When glucose is scarce, GAPDH is busy in glycolysis and can't release IFNg mRNA for translation. So glucose-deprived T cells make dramatically less IFNg even if they're otherwise properly activated.
- **Cytotoxic function**: Granzyme/perforin-mediated killing requires biosynthetic precursors from glycolytic intermediates. Glucose-starved CD8s show reduced degranulation.
- **Proliferation and survival**: Without glycolytic flux, T cells can't sustain the anabolic demands of clonal expansion. They become anergic or apoptotic.

The critical insight is that tumor cells and T cells are competing for the same glucose pool. The tumor wins this competition because it has more cells, higher per-cell uptake rates, and isn't dependent on activation signals to maintain glycolysis. This creates a metabolic checkpoint on anti-tumor immunity that's independent of the PD-1/PD-L1 or TGFb/Treg axes.

## Lactate Drives M2 Macrophage Polarization

Colegio et al. 2014 (Nature) showed that tumor-derived lactic acid is sufficient to polarize macrophages toward an M2/tumor-promoting phenotype. The mechanism:

- Lactate is taken up by macrophages via MCT transporters
- It stabilizes HIF-1alpha in macrophages (even under normoxic conditions), which drives expression of VEGF, arginase-1, and other M2 markers
- The accompanying acidification (lactate dissociates, dropping pH to ~6.5-6.8) further inhibits M1 inflammatory programs (NF-kB signaling is pH-sensitive)

This is mechanistically distinct from the existing M2 drivers in the model (TGFb, IL-10, IL-6, HIF). Those are cytokine-mediated signals; lactate is a metabolic waste product acting as a signaling molecule. It adds a parallel immunosuppressive axis that operates even if you block all the cytokine pathways.

## Lactate Also Suppresses T Cells (reserved mechanism)

Brand et al. 2016 (Cell Metabolism) showed that lactate concentrations above ~10-15 mM directly suppress T cells through:

- Inhibition of lactate export from T cells (MCT1 reversal due to extracellular lactate gradient), which causes intracellular acidification
- Impaired NFAT signaling, reducing cytokine production
- At very high concentrations (>20 mM), induction of T cell apoptosis

This mechanism is kept as "reserved" in the initial implementation because it's highly correlated with glucose depletion — both are driven by tumor glycolysis. Including both multiplied together risks over-suppressing T cells in the model. But the mechanisms are biologically distinct (glucose depletion = fuel starvation; lactate = waste product toxicity), so it can be activated if the glucose effect alone doesn't create enough metabolic immunosuppression.

## The Feedback Loop This Creates

The key topological contribution to the model:

```
More tumor cells (C1)
  -> more glucose consumed (Warburg)
    -> less glucose available for T cells
      -> impaired CD8 killing + less IFNg
        -> less tumor control
          -> more tumor cells
  -> more lactate produced
    -> more M2 macrophage polarization
      -> more immunosuppressive TME
        -> less tumor control
          -> more tumor cells
```

This is a resource-competition loop, fundamentally different from the cytokine-mediated suppression loops already in the model. It makes immune escape dependent on tumor size — larger tumors create a stronger metabolic barrier. It also links vasculature (K) to immune function: better vasculature delivers more glucose, partially relieving the metabolic barrier. This connects the angiogenesis subsystem to immune dynamics through nutrient delivery.

## Key References

- Chang CH et al. (2015) "Posttranscriptional Control of T Cell Effector Function by Aerobic Glycolysis." Cell 162(6):1229-41. PMID: 25662009.
- Ho PC et al. (2015) "Phosphoenolpyruvate Is a Metabolic Checkpoint of Anti-tumor T Cell Responses." Cell 162(6):1217-28. PMID: 26607794.
- Brand A et al. (2016) "LDHA-Associated Lactic Acid Production Blunts Tumor Immunosurveillance by T and NK Cells." Cell Metab 24(5):657-671. PMID: 27373335.
- Colegio OR et al. (2014) "Functional polarization of tumour-associated macrophages by tumour-derived lactic acid." Nature 513(7519):559-563. PMID: 25417104.
- Halbrook CJ et al. (2023) "Pancreatic cancer: Advances and challenges." Cell 186(8):1729-1754. PMID: 36807998.
- Sullivan MR et al. (2019) "Quantification of microenvironmental metabolites in murine cancers reveals determinants of tumor nutrient availability." eLife 8:e44235. PMID: 31548036.
- Vander Heiden MG et al. (2009) "Understanding the Warburg effect: the metabolic requirements of cell proliferation." Science 324(5930):1029-33. PMID: 19460998.
