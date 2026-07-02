⏺ Yes, that's exactly the right approach, and it's one of the main payoffs of the initialization fix we just made. Here's why:                                                              
                                                                                                                                                                                            
  What the baseline targets actually measure: The calibration targets (CD8 density, M1/M2 ratio, Treg fraction, ECM fraction, etc.) are IHC/pathology measurements from resected PDAC       
  specimens — i.e., the diagnosis-state TME. That's precisely the endpoint of evolve_to_diagnosis.                                                                                          
                                                                                                                                                                                            
  What the evolution constrains: By fixing tumor diameter at diagnosis (via initial_tumour_diameter) and varying rate parameters, you're asking: "which parameter values produce a realistic
   TME at diagnosis?" The composition at that diameter — immune infiltration, polarization balance, stromal development — is what varies. This constrains recruitment rates, polarization 
  kinetics, PSC activation, ECM deposition, etc.

  Why it's better than setting ICs directly: If you just set diagnosis ICs to match targets, you'd have no guarantee those ICs are a dynamically reachable state of the ODE system. The
  model might immediately relax away from them in the first few timesteps of treatment simulation. Evolution ensures the diagnosis state is on the model's natural trajectory.

  One thing to watch: parameter sets where evolution fails to reach the target diameter (rejected by evolve_to_diagnosis) are informative — they correspond to parameter combinations
  incompatible with forming a resectable tumor. That's a legitimate biological constraint, not a bug.

  Practical concern: each evolution adds ~10-15s per simulation. For SBI with thousands of draws, you'd want the simulation pool infrastructure to amortize this. The pooling already
  handles it.

  So the workflow would be: SA/SBI over rate parameters → evolve each to diagnosis → compare diagnosis-state observables against the baseline targets in calibration_targets/. The
  treatment-scenario targets (GVAX+nivo day 21) would be a second calibration stage starting from those diagnosis states.

❯ so initial_tumor_diameter should be fixed for SA/calibration? not a parameter? that probably gives makes the parameters much more identifiable. what about the VPop generation? we        
  were varying it at that point before.                                                                                                                                                     
                                                                                                                                                                                            
⏺ Exactly right. It's a clean separation:                                                                                                                                                   
                                                                                                                                                                                            
  Calibration/SA — fix it (or narrow heavily):                                                                                                                                              
  - The baseline targets are population-level summaries (median CD8 density, median M1/M2 ratio, etc.) from resectable PDAC cohorts with a fairly narrow size range (~2-4 cm)               
  - Fixing diameter isolates the effect of rate parameters on TME composition — otherwise you're conflating "different kinetics" with "different evolution time," which kills
  identifiability                                                                                                                                                                           
  - You're asking: "at 3 cm, what rate parameters produce the observed TME?"                                                                                                              

  VPop — vary it:
  - Real patients present with different tumor sizes
  - Tumor size affects TME: larger tumors tend to be more hypoxic, more fibrotic, more immunosuppressed — simply because they've been evolving longer
  - So initial_tumour_diameter sampled from a clinical distribution (e.g., lognormal centered around 3 cm, range 2-5 cm) is a legitimate axis of inter-patient variability
  - Each virtual patient gets: (calibrated rate parameters drawn from posterior) + (tumor diameter drawn from clinical distribution)

  The workflow becomes:
  1. SA: Fix diameter at ~3 cm, screen which rate parameters drive TME composition
  2. Calibration (SBI): Fix diameter, calibrate rate parameters against baseline targets → posterior
  3. VPop: Sample rate parameters from posterior × sample diameter from clinical prior → virtual population with realistic heterogeneity in both kinetics and tumor burden

  One subtlety: some of the variability in the calibration targets (IQR/CI95) comes from tumor size differences across patients in the source studies. By fixing diameter during
  calibration, the posterior over rate parameters will absorb that variability — making it slightly too broad. But that's a conservative error. When you add diameter variability in VPop,
  you get a bit more spread than observed, which is safer than too little.

  Want me to check the current priors CSV to see how initial_tumour_diameter is configured, and whether it should be pulled out into a separate "VPop-only" parameter list?
