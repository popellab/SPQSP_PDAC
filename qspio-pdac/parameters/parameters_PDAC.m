%% Function to generate model parameters for PDAC (pancreatic ductal adenocarcinoma)
%
% Output: params -- object containing parameters

function params = parameters_PDAC

% Define Cell Dimension
params.cell.Value = 1;
params.cell.Units = 'cell';
params.cell.Notes = 'A unit cell for calculation';
% Avogadro's Number
params.N_avg.Value = 6.0221409e23;
params.N_avg.Units = 'molecule/mole';
params.N_avg.Notes = '(Avogadro 1811)';

%% Compartment Volume Parameters
% Central Compartment Volume
params.V_C.Value = 5;
params.V_C.Units = 'liter';
params.V_C.Notes = '(estimated)';
% Peripheral Compartment Volume
params.V_P.Value = 60;
params.V_P.Units = 'liter';
params.V_P.Notes = '(estimated)';
% Cancer-Free Tumour Volume
params.V_Tmin.Value = 1e-6;
params.V_Tmin.Units = 'milliliter';
params.V_Tmin.Notes = '(estimated)';
% Number of Lymph Nodes
params.nLNs.Value = 17;
params.nLNs.Units = 'dimensionless';
params.nLNs.Notes = '(estimated)';
% Lymph Node Diameter
params.D_LN.Value = 5;
params.D_LN.Units = 'millimeter';
params.D_LN.Notes = '(Schmidt 2007, PMID: 17724531)';
% Lymph Nodes Compartment Volume
params.V_LN.Value = [];
params.V_LN.Units = 'milliliter';
params.V_LN.Notes = ['Volume of lumped tumor-draining lymph nodes calculated based on number of LNs ' params.nLNs.Notes ' and average LN diameter ' params.D_LN.Notes];
params.V_LN.Factors = ["nLNs", "D_LN"];
params.V_LN.Equation = 'p(1)*4/3*pi*(p(2)/2)^3';
% Cancer Cell Diameter
params.D_cell.Value = 17;
params.D_cell.Units = 'micrometer';
params.D_cell.Notes = '(Abramczyk 2015, PMID: 25730442)';
% T Cell Diameter
params.D_Tcell.Value = 6.94;
params.D_Tcell.Units = 'micrometer';
params.D_Tcell.Notes = '(Chapman 1981, PMID: 6975780)';
% Volume of a Cancer Cell
params.vol_cell.Value = [];
params.vol_cell.Units = 'micrometer^3/cell';
params.vol_cell.Notes = ['Volume of a cancer cell calculated based on cancer cell diameter ' params.D_cell.Notes];
params.vol_cell.Factors = ["D_cell", "cell"];
params.vol_cell.Equation = '4/3*pi*(p(1)/2)^3/p(2)';
% Volume of a T cell
params.vol_Tcell.Value = [];
params.vol_Tcell.Units = 'micrometer^3/cell';
params.vol_Tcell.Notes = ['Volume of a T cell calculated based on the average T cell diameter ' params.D_Tcell.Notes];
params.vol_Tcell.Factors = ["D_Tcell", "cell"];
params.vol_Tcell.Equation = '4/3*pi*(p(1)/2)^3/p(2)';
% Dead Cell Clearance Rate
params.k_cell_clear.Value = 0.1;
params.k_cell_clear.Units = '1/day';
params.k_cell_clear.Notes = '(estimated)';
% Tumor Cell Volume Fraction = INTRACELLULAR FRACTION (per simbio_init.m line 95)
% V_T volume rule contribution from cells: cells*vol_cell/Ve_T. Ve_T = 0.39
% means cells occupy 39% of V_T and the remaining 61% is interstitium / ECM.
%
% Structural coupling: Ve_T is part of the (Ve_T, rho_collagen, phi_col_ref)
% triple that defines the V_T volume rule and the calibration domain for
% every Hill function downstream of phi_collagen (H_pore_*, H_stiff_*, C_max
% suppression). Independent priors compensate jointly; coordinated re-anchoring
% requires keeping all three in priors with submodel-target anchors and
% re-tuning the downstream Hill EC50s to remain coherent with their joint
% posteriors. See notes/calibration/ecm_anchor_audit_2026-04-28.md (GitHub #93).
%
% Citation history: previously cited Finley 2012 PMID 22547351 — verified to
% NOT be a primary measurement (Finley 2012 is a VEGF compartment-modeling
% paper inheriting this value from upstream parameter tables). Re-anchored to
% Wang 2023 (DOI 10.3389/fonc.2022.1007134), which reports extended-Tofts-
% model DCE-MRI Ve_DCE = 0.61 +/- 0.11 in the PDAC tumor-mass ROI (n=14).
% Ve_T (intracellular) = 1 - Ve_DCE = 0.39.
%
% NOTE: parameters/cross_model_parameters.csv labels this "Void fraction" and
% lists 0.017-0.529 as the PDAC range — that label is opposite the code's
% mechanistic use (intracellular fraction) and the 0.017-0.529 range has no
% traceable primary source. Both flagged for cleanup in a separate PR.
%
% Now in pdac_priors.csv (lognormal median 0.39, sigma 0.5), anchored via
% submodel inference against
% calibration_targets/submodel_targets/Ve_T_PDAC_deriv001.yaml.
params.Ve_T.Value = 0.37;
params.Ve_T.Units = 'dimensionless';
params.Ve_T.Notes = 'Intracellular volume fraction of the tumor. Submodel posterior median 0.37 (sigma 0.32, 80% contraction) from Wang 2023 DCE-MRI PDAC tumor-mass anchor (Ve_DCE = 0.61 +/- 0.11, n=14).';
params.Ve_T.derivedFrom = {'Wang 2023 DOI: 10.3389/fonc.2022.1007134'};

%% Cancer Parameters
% Chemotherapy-sensitive cancer clone parameters
% Lowered 0.010 -> 0.008 to the clinical volume-doubling-time value (submodel
% targets k_C1_growth_PDAC_deriv001 DOI 10.1007/s12029-016-9876-7 and _deriv002
% DOI 10.1016/j.ejrad.2015.05.012). This ships COUPLED with C_cap0 (the
% C-independent carrying-capacity floor): the old 0.010 default + stroma-
% proportional C_max could not grow a 3.2 cm tumor at the clinical rate without
% the carrying capacity self-limiting (~1.5 cm stall), which is why the submodel
% posterior had inflated to 0.0495 via the organoid target. With C_cap0 ~1.7e9
% the model reaches 3.2 cm at k=0.008 with realized net VDT ~145 d. See
% notes/HANDOFF_growth_carrying_capacity_redesign.md (Q1 results).
params.k_C1_growth.Value = 0.008;
params.k_C1_growth.Units = '1/day';
params.k_C1_growth.Notes = 'clinical PDAC volume-doubling time (k_C1_growth_PDAC_deriv001 DOI 10.1007/s12029-016-9876-7; _deriv002 DOI 10.1016/j.ejrad.2015.05.012); coupled with C_cap0 carrying-capacity floor';
params.k_C1_death.Value = 0.003;
params.k_C1_death.Units = '1/day';
params.k_C1_death.Notes = '(Palsson 2013, PMID: 24074340)';

% Chemotherapy-resistant cancer clone parameters
params.k_C2_growth.Value = 0.010;
params.k_C2_growth.Units = '1/day';
params.k_C2_growth.Notes = '(Ryu 2014, PMID: 24895040; Desai 2006, PMID: 16489089)';
params.k_C2_death.Value = 0.0001;
params.k_C2_death.Units = '1/day';
params.k_C2_death.Notes = '(Palsson 2013, PMID: 24074340)';
% Maximal Tumor Capacity
params.C_max.Value = 2.7e4;
params.C_max.Units = 'cell';
params.C_max.Notes = '(Desai 2006, PMID: 16489089)';
% Initial Tumour Diameter
params.initial_tumour_diameter.Value = 3.2; % 1 - 5 cm
params.initial_tumour_diameter.Units = 'centimeter';
params.initial_tumour_diameter.Notes = '(varied)';
% Cancer Cell Density
params.rho_cell.Value = 2.06e8; % 1.67e8 - 2.06e8
params.rho_cell.Units = 'cell/milliliter';
params.rho_cell.Notes = '(Del Monte 2009, PMID: 19176997; Barnes 2016 PMID: 26332194)';

%% Initial Conditions / Baseline Populations (Density-Based)
% These density-based IC priors are derived from observed baseline test statistics
% (PMID: 30185312). set_baseline_populations.m converts densities to absolute
% counts using tumor volume from initial_tumour_diameter.

% Cancer Cell Density at Baseline
params.C1_density_initial.Value = 376310;
params.C1_density_initial.Units = 'cell/millimeter^3';
params.C1_density_initial.Notes = 'Baseline cancer cell density (varied). From cancer_cell_density_baseline test statistic.';
% CD8+ T Cell Density at Baseline
params.CD8_density_initial.Value = 34831;
params.CD8_density_initial.Units = 'cell/millimeter^3';
params.CD8_density_initial.Notes = 'Baseline CD8+ TIL density (varied). From cd8_density_baseline test statistic (PMID: 30185312).';
% Treg Density at Baseline
params.Treg_density_initial.Value = 2416;
params.Treg_density_initial.Units = 'cell/millimeter^3';
params.Treg_density_initial.Notes = 'Baseline Treg density (varied). From treg_density_baseline test statistic (PMID: 30185312).';
% Th (CD4+ helper) Density at Baseline
params.Th_density_initial.Value = 8028;
params.Th_density_initial.Units = 'cell/millimeter^3';
params.Th_density_initial.Notes = 'Baseline CD4+ helper T cell density (varied). From th_density_baseline test statistic.';
% MDSC Density at Baseline
params.MDSC_density_initial.Value = 12789;
params.MDSC_density_initial.Units = 'cell/millimeter^3';
params.MDSC_density_initial.Notes = 'Baseline MDSC density (varied). From mdsc_density_baseline test statistic (PMID: 31061129).';
% TAM (total macrophage) Density at Baseline
params.TAM_density_initial.Value = 8747;
params.TAM_density_initial.Units = 'cell/millimeter^3';
params.TAM_density_initial.Notes = 'Baseline TAM density (varied). From tam_density_baseline test statistic. Split by m1_m2_ratio_initial.';
% M1/M2 Macrophage Ratio at Baseline
params.m1_m2_ratio_initial.Value = 0.756;
params.m1_m2_ratio_initial.Units = 'dimensionless';
params.m1_m2_ratio_initial.Notes = 'Baseline M1/M2 ratio (varied). From m1_m2_ratio_baseline test statistic. Used to split TAM density.';
% CAF Subtype Densities at Baseline (split from aPSC_density_initial = 242121)
% Approximate split: myCAF ~60%, iCAF ~35%, apCAF ~5% based on scRNA-seq (Elyada 2019)
params.iCAF_density_initial.Value = 84742;
params.iCAF_density_initial.Units = 'cell/millimeter^3';
params.iCAF_density_initial.Notes = 'Baseline iCAF density (varied). ~35% of total CAF density (Elyada 2019 PMID: 31197017).';

params.myCAF_density_initial.Value = 145273;
params.myCAF_density_initial.Units = 'cell/millimeter^3';
params.myCAF_density_initial.Notes = 'Baseline myCAF density (varied). ~60% of total CAF density (Elyada 2019 PMID: 31197017).';

params.apCAF_density_initial.Value = 12106;
params.apCAF_density_initial.Units = 'cell/millimeter^3';
params.apCAF_density_initial.Notes = 'Baseline apCAF density (varied). ~5% of total CAF density (Elyada 2019 PMID: 31197017).';
% Quiescent PSC Density at Baseline
params.qPSC_density_initial.Value = 69575;
params.qPSC_density_initial.Units = 'cell/millimeter^3';
params.qPSC_density_initial.Notes = 'Baseline qPSC density (varied). Derived from aPSC/qPSC ratio = 3.48 (PMID: 24651015).';
% APC (total dendritic cell) Density at Baseline
params.APC_density_initial.Value = 2711;
params.APC_density_initial.Units = 'cell/millimeter^3';
params.APC_density_initial.Notes = 'Baseline total APC density (varied). From apc_density_baseline test statistic.';
% APC Maturation Ratio at Baseline
params.apc_maturation_ratio_initial.Value = 0.33;
params.apc_maturation_ratio_initial.Units = 'dimensionless';
params.apc_maturation_ratio_initial.Notes = 'Fraction of mature APC (varied). From apc_maturation_ratio_baseline. Used to split APC density.';
% ECM Volume Fraction at Baseline
params.ECM_fraction_initial.Value = 0.222;
params.ECM_fraction_initial.Units = 'dimensionless';
params.ECM_fraction_initial.Notes = 'Baseline ECM volume fraction (varied). From ecm_density_baseline test statistic (PMID: 22810696).';
% CCL2 Concentration at Baseline
params.CCL2_conc_initial.Value = 21.17;
params.CCL2_conc_initial.Units = 'nanomolarity';
params.CCL2_conc_initial.Notes = 'Baseline CCL2 concentration (varied). Derived from ccl2_concentration_baseline (184155 pg/mL, MW=8700 Da).';
% TGF-beta Concentration at Baseline
params.TGFb_conc_initial.Value = 0.083;
params.TGFb_conc_initial.Units = 'nanomolarity';
params.TGFb_conc_initial.Notes = 'Baseline TGF-beta concentration (varied). Derived from tgfb_concentration_baseline (2083 pg/mL, MW=25000 Da).';
% VEGF (angiogenic factor) Concentration at Baseline
params.VEGF_conc_initial.Value = 150829;
params.VEGF_conc_initial.Units = 'picogram/milliliter';
params.VEGF_conc_initial.Notes = 'Baseline VEGF concentration (varied). From angiogenic_factor_baseline test statistic.';

%% T Cell Parameters
% Tumor-specific T cell clone number (TMB)
params.n_clones_tum.Value = 120;
params.n_clones_tum.Units = 'dimensionless';
params.n_clones_tum.Notes = '(Morisaki 2021 PMID: 34193879; Narang 2019, PMID: 30832597)';
% Vaccine-driven epitope spreading (Path A): effective fold-broadening of the
% tumor-specific T-cell repertoire under saturating GVAX antigen. GVAX (whole-cell,
% GM-CSF) cross-primes new specificities (Thomas 2004 J Exp Med PMID 15289501:
% mesothelin-specific CD8 induced by a vaccine that does not define that antigen).
% One shared intensity scales n_CD8_clones / n_CD4_clones in the
% naive->aCD8 / naive->aTh commitment reactions via the vaccine-antigen Hill
% H_P1_spread (=0 at diagnosis). A shared parameter retains vaccine-driven
% broadening without asserting an independently identifiable CD4:CD8 asymmetry.
% Its initial value is the geometric mean of the retired medians 4 and 6; refit.
params.f_epitope_spread.Value = 4.898979485566356;
params.f_epitope_spread.Units = 'dimensionless';
params.f_epitope_spread.Notes = 'Shared effective fold-broadening of tumor-specific CD4 and CD8 repertoires under saturating GVAX antigen (qualitatively anchored to Thomas 2004 PMID 15289501; geometric-mean initial value; estimated and requires reduced-model refit).';
params.f_epitope_spread.derivedFrom = {};
% Self-antigen-specific T cell clone number
params.n_clones_slf.Value = 100;
params.n_clones_slf.Units = 'dimensionless';
params.n_clones_slf.Notes = 'Number of self-antigen-specific CD4 T cell clones (estimated). Renamed to n_CD4_clones at assembly.';
% Maximum Rate of CD8+ T Cell Activation by mAPCs
params.k_nCD8_act.Value = 23;
params.k_nCD8_act.Units = '1/day';
params.k_nCD8_act.Notes = '(De Boer and Perelson 1995, PMID: 7475092; Robertson-Tessi 2012, PMID: 22051568)';
% Maximum Rate of CD4+ Treg Activation by APCs
params.k_nCD4_act.Value = 5;
params.k_nCD4_act.Units = '1/day';
params.k_nCD4_act.Notes = '(De Boer and Perelson 1995, PMID: 7475092; Robertson-Tessi 2012, PMID: 22051568)';
% Rate of CD8+ T Cell Proliferation
params.k_CD8_pro.Value = 1.0;
params.k_CD8_pro.Units = '1/day';
params.k_CD8_pro.Notes = '(Marchingo 2014, PMID: 25430770)';
% Rate of CD4+ T Cell Proliferation
params.k_CD4_pro.Value = 1.0;
params.k_CD4_pro.Units = '1/day';
params.k_CD4_pro.Notes = '(Marchingo 2014, PMID: 25430770)';
% Rate of CD8 T Cell Decay
params.k_CD8_death.Value = 0.03;
params.k_CD8_death.Units = '1/day';
params.k_CD8_death.Notes = 'Homeostatic death of functional (non-exhausted) intratumoral CD8 effectors, distinct from the exhaustion routes (V_T.CD8 -> V_T.CD8_pex). Re-centered 2026-07-15 from 0.10 (De Boer & Perelson 1995, PMID 7475092 -- a LUMPED activated-T-cell turnover that predates the exhaustion split) to 0.03/day (t1/2 ~23 d), the functional-effector homeostatic band (~0.02-0.05). Because this model now routes exhaustion/AICD separately, the residual pure-death term must be below the lumped 1995 rate. The Akondy2017 memory-CD8 submodel target (~0.0074) is retained as a lower-bound guardrail, not a point anchor (wrong phenotype: memory-in-blood). See notes/calibration/cd8_retention_anchoring_scope.md.';
% Tumor-local CD8 proliferation rate (PDAC TME-driven, antigen-gated, PD-1-suppressed).
% PLACEHOLDER VALUE - not yet anchored on verified literature. Qualitative starting
% value of 0.05/day chosen as analog to k_Th_pro_tumor (0.15/day) and k_Treg_pro_tumor (0.2/day),
% reflecting that CD8 Ki67+ fraction in PDAC TIL is generally lower than CD4 Treg/Th. To be
% replaced with a submodel target extraction (Ki67+ CD8 fraction in solid-tumor TIL) once
% the mechanism is validated by SBI.
params.k_CD8_T_pro.Value = 0.05;
params.k_CD8_T_pro.Units = '1/day';
params.k_CD8_T_pro.Notes = 'PLACEHOLDER: tumor-local CD8 proliferation rate (qualitative anchor; not yet literature-verified). Replace with submodel target after SBI confirms pathway.';
params.k_CD8_T_pro.derivedFrom = {};
% Maximum CD8 density in tumor (carrying capacity for in-tumor proliferation).
% Set to match rho_Treg_max (1e7 cell/mL) as a default scale; allows inflamed-tumor TIL
% densities of order 100-200 cells/mm^2 without unbounded growth. Not separately anchored.
params.rho_CD8_max.Value = 5e6;
params.rho_CD8_max.Units = 'cell/milliliter';
params.rho_CD8_max.Notes = 'Max CD8 density in tumor (carrying cap). Re-centered 1e7 -> 5e6 on 2026-06-18 (v26) toward the live cd8_density_baseline_pooled target (61.8 cells/mm^2 via the model 11um Abercrombie projection -> ~5.6e6 cell/mL). Free placeholder (NOT submodel-anchored); once the v24 CD8 seed floored the cold-tumor collapse, the realized cd8_density was dominated by the filled branch at rho_CD8_max*V_T (~291/mm^2 at the old posterior 2.65e7 -> 2.7x over the pooled 62). Lowering the cap is an unambiguous multi-target win: cd8_density, cd8_pct (was 1.4x over post-seed) and cd8exh_pct (4x over) ALL want less CD8, and the shrunk nucleated_total LIFTS cd4_pct (Th absolute is correct but its fraction was diluted). Same placeholder-re-center as rho_Treg_max v23; the seed floor prevents re-collapse. sigma 1.5 keeps the densest tumors reachable.';
params.rho_CD8_max.derivedFrom = {};
% Vaccine-driven in-situ (intratumoral) CD8 proliferation increment (2026-07-18).
% Vaccine-primed, antigen-experienced CD8 undergo
% antigen-driven in-situ proliferation on tumoral antigen re-encounter -- the
% treatment-DIFFERENTIAL half of tumor CD8 expansion that the constitutive
% k_CD8_T_pro (vaccine-blind, Blessin-anchored, pins the baseline cd8_density)
% cannot supply. Added as a SEPARATE saturating source on bulk V_T.CD8 in
% gvax_module, gated by the GVAX antigen Hill H_P1_spread (=0 at diagnosis and in
% no-vaccine arms), which decouples cd8_fc from the baseline density. Anchored as
% a FRACTION of the constitutive rate: k_CD8_T_vac_pro = f*k_CD8_T_pro, f~0.6, the
% vaccine-differential increment bounded by intrinsic in-situ proliferation
% capacity. (The earlier 0.5/day PLACEHOLDER was ~6x the Blessin-anchored
% constitutive k_CD8_T_pro=0.079/day -- backwards, and the source of the ~20-29x
% cd8gzmb_fc over-expansion.) Lutz 2014 shows vaccine-induced proliferation
% only QUALITATIVELY (Ki-67 IHC, Fig 1D; no CD8-resolved fraction), so no direct
% Ki-67->rate anchor exists. See notes/calibration/vpop_thexh_cd8vac_anchoring_2026-07-23.md.
params.k_CD8_T_vac_pro.Value = 0.05;
params.k_CD8_T_vac_pro.Units = '1/day';
params.k_CD8_T_vac_pro.Notes = 'GVAX-vaccine-gated in-situ tumor CD8 proliferation increment (Path C). Anchored to f*k_CD8_T_pro (f~0.6): the vaccine-differential increment is a bounded fraction of the Blessin-anchored constitutive in-situ rate (k_CD8_T_pro=0.079/day). H_P1_spread-gated (zero without vaccine), saturating in V_T.CD8 density. Retained in gvax_module after the C13 TLA/urelumab cut. Prior sigma 0.7 (was 1.5 placeholder). See notes/calibration/vpop_thexh_cd8vac_anchoring_2026-07-23.md.';
params.k_CD8_T_vac_pro.derivedFrom = {'k_CD8_T_pro'};
% Density set-point for a quadratic CD8 crowding-death term (V3 stabilizer, 2026-07-17).
% The homeostatic death is now k_CD8_death*CD8*(1 + CD8/(rho_CD8_crowd*V_T)) -- a
% cytokine/nutrient-competition sink that grows with density. Its job is NOT to lower
% the target CD8 level but to REMOVE the near-critical runaway: linear birth vs linear
% death sits at a bifurcation (below it CD8 collapses, above it the coupled tumour/immune
% loop diverges to NaN, ~30% of draws), and the ~62 cell/mm^2 target lives right on that
% edge. A C^2 death bounds the high-density branch, turning the "explode" side into a
% finite steady state so the high-density regime becomes reachable. Set-point 2e7 cell/mL
% (~3.6x the observed ~5.6e6 -> crowding death equals the linear death at ~220 cells/mm^2):
% ~28% extra death at the target (mild), growing to bound the runaway above it. Free
% placeholder, not yet submodel-anchored; first-try value to test the mechanism.
params.rho_CD8_crowd.Value = 2e7;
params.rho_CD8_crowd.Units = 'cell/milliliter';
params.rho_CD8_crowd.Notes = 'Density set-point for the quadratic CD8 crowding-death term (V3 stabilizer, 2026-07-17). Crowding death k_CD8_death*CD8^2/(rho_CD8_crowd*V_T) equals the linear death at density = rho_CD8_crowd. Placeholder 2e7 cell/mL (~3.6x observed ~5.6e6); bounds the near-critical high-density runaway without crushing the target.';
params.rho_CD8_crowd.derivedFrom = {};
% (alpha_IL2_cap / IL2_cap_50 retired 2026-06-22, in-tumor restructure: the
% IL-2-niche cap boost was a workaround for dead division destiny. Now IL2_niche
% fires N_aT, the treated CD8 fold comes from influx/expansion, and a trace showed
% CD8 at <9% of the physical cap with H_IL2_cap saturated flat -- the boost is inert.
% Cap reverts to the physical rho_CD8_max*V_T.)

% Baseline tumor-immunosurveillance CD8 seeding (cancer-gated effector floor; v24, 2026-06-18).
% k_CD8_seed REMOVED 2026-07-13 (dead parameter). Added in v24 (732b9f6) as a constitutive
% CD8 homing flux to stop ~1/3 of virtual patients collapsing to an unphysical ~0 CD8
% (cd8_density q05 = 3e-5 /mm^2 vs real immune-cold PDAC 5-30 /mm^2). It was a band-aid on a
% supply bug, and 7c0366d fixed the supply chain properly (k_T_activation_per_APC bound to
% Bousso + residence re-anchor; cd8_density 0.005 -> 21 /mm^2),
% which retired the need for the seed. Its reaction was commented out then, but the parameter
% was left registered at Value = 0 and never entered pdac_priors.csv -- so it was carried into
% the SBML/C++ as a dead constant with no consumers. Now removed from parameters, Teff_module,
% and the generated model. The CD4 analog k_Th_seed (v25) is deliberately RETAINED and live:
% the cd4_pct bistability it guards against is still unfixed.

% === Progenitor-exhausted (Tpex) reservoir ===
% Gateway exhaustion topology: effector CD8 exhaust into a self-renewing TCF1+
% progenitor-exhausted pool (V_T.CD8_pex), which differentiates to terminal
% CD8_exh, refluxes to effectors under checkpoint blockade (k_reinvig, gated by
% anti-PD-1 occupancy), and is maintained against a niche carrying capacity
% (Jansen 2019, Nature 576:465 -- intratumoral niche maintains stem-like CD8).
% The capacity (not a drain balance) sets the in-tumor CD8 setpoint, taking the
% counted pool off the near-cancellation knife-edge that otherwise makes the
% baseline CD8 count both vanishingly small and SBI-unidentifiable. All four
% rates are SBI knobs (wide priors); cross-indication anchors below.
% Niche carrying-capacity density for the Tpex reservoir.
params.rho_pex_max.Value = 1e7;
params.rho_pex_max.Units = 'cell/milliliter';
params.rho_pex_max.Notes = 'Niche carrying-capacity density for the progenitor-exhausted (Tpex) reservoir (Jansen 2019 intratumoral stem-like CD8 niche). Sets the in-tumor CD8 setpoint; SBI-calibrated, default = rho_CD8_max scale.';
params.rho_pex_max.derivedFrom = {};
% p_pex_renew RETIRED 2026-06-17 (reservoir reformulation): the generative
% self-renewal it scaled self-ignited the reservoir to the niche cap regardless
% of inflow, decoupling terminal exhaustion from its drivers. Reservoir is now
% inflow-sourced + niche-capped on the inflow (Teff_module.m); self-renewal is
% folded into slow turnover (k_pex_death). The niche cap (rho_pex_max) is now a
% ceiling on inflow, not a self-filling target.
% Tpex -> terminal differentiation rate (days timescale; Miller 2019 longitudinal
% progenitor->terminal transition over tumor days 6-21).
params.k_pex_diff.Value = 0.1;
params.k_pex_diff.Units = '1/day';
params.k_pex_diff.Notes = 'Differentiation rate of progenitor-exhausted (Tpex) -> terminal exhausted CD8 (Miller 2019 longitudinal progenitor->terminal transition); SBI-calibrated.';
params.k_pex_diff.derivedFrom = {};
% Tpex death rate, anchored as a RATIO to the terminal exhausted death rate.
% Miller 2019 Fig 3f: progenitor (Slamf6+Tim-3-) ~41% Annexin-V+ vs terminal ~72%.
% Under a flow/residence occupancy model p = d*tau/(1+d*tau) with a shared Annexin
% visibility tau, the HAZARD ratio is the ODDS ratio, not the raw prevalence ratio:
%   d_P/d_E = [0.41/0.59] / [0.72/0.28] = 0.695/2.571 = 0.27
% (the raw 41/72 = 0.57 mapping ignores that both prevalences are high and saturating).
% So k_pex_death = 0.27 * k_CD8_exh_death = 0.27 * 0.15 = 0.041. The ABSOLUTE rate is
% NOT identifiable from an Annexin snapshot (tau unmeasured), which is why the joint
% Miller Tpex-lineage extraction was rejected in derivation review; only the ratio to
% the anchored terminal rate is robust. k_pex_diff is kept as a separate effective-rate
% submodel target (Miller Fig4e transfer, ~0.094/day). Updated 2026-07-15 (was 0.06 = 0.4x).
params.k_pex_death.Value = 0.041;
params.k_pex_death.Units = '1/day';
params.k_pex_death.Notes = 'Death rate of progenitor-exhausted (Tpex) CD8, set at 0.27 * k_CD8_exh_death (=0.041) from Miller 2019 Fig 3f Annexin-V+ (progenitor ~41% vs terminal ~72%) under a shared-tau flow occupancy model: the hazard/odds ratio [0.41/0.59]/[0.72/0.28]=0.27 (NOT the raw 41/72=0.57 prevalence ratio, because prevalences saturate). Absolute rate not identifiable from the snapshot; anchored as a ratio to the anchored terminal death rate.';
params.k_pex_death.derivedFrom = {'Miller2019'};
% Constitutive (therapy-independent) Tpex -> effector reflux. Tpex homeostatically
% replenish the effector pool even without checkpoint blockade (Siddiqui 2019 --
% stem-like cells continuously generate effector progeny); the anti-PD-1 k_reinvig
% arm boosts this. Bounded by the effector carrying cap (rho_CD8_max), so it cannot
% run away. SBI-calibrated; sets the functional-effector fraction of the counted pool.
params.k_pex_reflux.Value = 0.005;
params.k_pex_reflux.Units = '1/day';
params.k_pex_reflux.Notes = 'Constitutive Tpex -> effector reflux rate (homeostatic stem-like replenishment, Siddiqui 2019); capped by rho_CD8_max; SBI-calibrated.';
params.k_pex_reflux.derivedFrom = {};

% === Exhausted-Th (Th_exh) reservoir: CD4 analog of the CD8 Tpex gateway ===
% The tumoral Th pool was bistable (collapse vs saturate) because its three
% Th->Th_exh exhaustion routes are unbounded linear drains with no reservoir-
% fullness throttle and no reflux -- so once the (TGFb-saturated) drain exceeds
% k_Th_pro_tumor the live Th pool drains to ~0 and treg_fraction_cd4 -> 1. CD8
% escapes this exact pathology via its gateway: exhaustion INFLOW is niche-capped
% (shuts off as the reservoir fills) and a constitutive reflux replenishes the
% functional pool. We mirror that for Th: (1) rho_Thexh_max niche-caps the three
% Th->Th_exh inflows in Th_module, (2) k_Th_reflux returns Th_exh -> Th. Toy ODE
% (notes/calibration/d21_composition_tension_diagnosis.md, 2026-06-18) shows the
% niche-cap ALONE fails (reservoir cannot fill from a collapsed Th, throttle stays
% ~1) but cap+reflux together are MONO-STABLE: Th rides its rho_Th_max cap instead
% of the collapse, eliminating the bistability. Biological basis: CD4 exhaustion is
% partially reversible; Tfh/Th1 progenitors are a CD4 analog of CD8 Tpex. Both rates
% are SBI knobs with sibling-informed priors (CD8 analogs rho_pex_max, k_pex_reflux).
% Niche carrying-capacity density for the exhausted-Th reservoir. Data-anchored
% (2026-07-23) from Shibuya PD-1+CD4 fraction x Liudahl CD4 density (see below),
% NOT the old rho_Th_max-scale heuristic. The exhausted fraction is ~0.78
% (Shibuya), so the exhausted reservoir is inherently LARGER than the functional
% Th pool -- rho_Thexh_max > rho_Th_max is CORRECT, and the earlier "reservoir must
% not exceed parent" ordering concern (new_param_magnitude_review.md) is
% contradicted by the data and retired.
params.rho_Thexh_max.Value = 3e7;
params.rho_Thexh_max.Units = 'cell/milliliter';
params.rho_Thexh_max.Notes = 'Niche carrying-capacity density for the exhausted-Th (Th_exh) reservoir; caps the Th->Th_exh exhaustion inflow (mirrors CD8 rho_pex_max). Data-anchored: exhausted-Th density = Shibuya PD-1+CD4 fraction (0.78) x Liudahl total-Th density (p95 ~211 cell/mm^2) / Abercrombie geometry ~1.5e7 cell/mL at the high-Th tail; ceiling = ~2x headroom -> 3e7 cell/mL so the cap does not bind physiological patients. Prior sigma 0.5 (was 1.5 placeholder). See notes/calibration/vpop_thexh_cd8vac_anchoring_2026-07-23.md.';
params.rho_Thexh_max.derivedFrom = {};
% Constitutive Th_exh -> Th reflux (CD4 analog of k_pex_reflux). Breaks the Th
% collapse so the rho_Th_max niche cap binds (mono-stable). SBI-calibrated.
params.k_Th_reflux.Value = 0.01;
params.k_Th_reflux.Units = '1/day';
params.k_Th_reflux.Notes = 'Constitutive exhausted-Th -> functional-Th reflux rate (partial reversibility of CD4 exhaustion; CD4 analog of CD8 k_pex_reflux); capped by rho_Th_max; SBI-calibrated.';
params.k_Th_reflux.derivedFrom = {};

% === Mechanistic T Cell Activation Parameters ===
% Rate of naive T cell activation per mature APC
params.k_T_activation_per_APC.Value =  3.2e-4;  % Bousso2003-derived (was 1e-6 placeholder)
params.k_T_activation_per_APC.Units = '1/(day*cell)';
params.k_T_activation_per_APC.Notes = 'Per-DC naive T cell activation rate in lymph node (encounter frequency × activation probability). Anchored to Bousso 2003 (PMID: 14563317) two-photon imaging: ~55% of cognate P14 CD8 engaged in long-lived contacts with 1uM gp33-pulsed DCs by 20h. Inverting the well-mixed capture model f=1-exp(-k*N_DC*t) with ~3000 antigen-bearing DCs/node gives k~3.2e-4 /(day*cell). Replaces the prior 1e-6 placeholder (the old fraction-valued submodel target did not bind k because a [0,1] observable was swamped by the proxy/species discrepancy sigma; reformulated to a direct per-DC-rate inversion). See k_T_activation_per_APC_PDAC_deriv001.yaml.';
params.k_T_activation_per_APC.derivedFrom = 'Miller et al. 2004 (PMID: 15240696) - T cell scanning rate ~500-5000 APCs/hour';
% CD4-help / CD40L licensing of cDC1->CD8 priming (mechanism #7; H_CD4_help in
% Teff_module.m, rule in Th_module.m). Additive amplifier with a help-independent
% floor: H_CD4_help = f_help_indep + (1-f_help_indep)*V_LN.Th/(V_LN.Th+Th_50_help).
params.f_help_indep.Value = 0.2;  % residual innate (DAMP/IL-12) cross-priming with zero CD4 help; PDAC is innate-poor -> small floor
params.f_help_indep.Units = 'dimensionless';
params.f_help_indep.Notes = 'Help-independent floor fraction of cDC1->CD8 cross-priming. Estimated (PDAC innate-poor); queued for submodel anchoring in notes/reconciliation_targets.csv.';
params.f_help_indep.derivedFrom = {};
params.Th_50_help.Value = 1500;  % cells; centered on the simulated V_LN.Th operating band (trace 2026-06-04)
params.Th_50_help.Units = 'cell';
params.Th_50_help.Notes = 'Half-maximal LN conventional-helper (V_LN.Th) density for CD40L licensing of cDC1. Centered on the simulated V_LN.Th operating range (C++ trace 2026-06-04: ~1023 cells at diagnosis, rising to ~1356/1458 under GVAX / GVAX+nivo) so H_CD4_help is responsive (diagnosis just below half-max -> PDAC help-limited; GVAX Th boost raises help). The earlier 1e5 placeholder pinned H_CD4_help at the floor (~0.21), inert. Still queued for a literature submodel anchor, but the scale is now set.';
params.Th_50_help.derivedFrom = {};
% Dedicated EC50 for the DIVISION-DESTINY help-licensing Hill (H_div_help), used ONLY in
% the N_aT costim term -- separate from Th_50_help (priming throughput). Set into the
% realized V_LN.Th operating band (~2.3e5 d0 -> 3e6 d21 in the 2026-06-29 posterior) so
% H_div_help is cold at diagnosis and swings under GVAX, restoring Path B's costim swing
% WITHOUT throttling baseline priming (which keeps the saturated H_CD4_help). Higher EC50
% than Th_50_help by design: licensing a cell to PRIME needs only baseline help; programming
% EXTENSIVE division needs vaccine-level help. Estimated/scale-centered placeholder, SBI-tunable.
params.Th_50_div.Value = 1e6;
params.Th_50_div.Units = 'cell';
params.Th_50_div.Notes = 'Half-max V_LN.Th for the division-destiny help-licensing Hill H_div_help (N_aT costim only; decoupled from the priming-throughput Th_50_help). Set into the realized V_LN.Th band so the costim divisions swing under vaccine without throttling baseline priming. Estimated, SBI-tunable.';
params.Th_50_div.derivedFrom = {};
% n_T_expansion_burst REMOVED (audit V12, 2026-06-04): it was a fixed fold-expansion
% multiplier on the naive->activated step that DOUBLE-COUNTED the clonal expansion
% already carried by the 2^N_aT proliferation cascade (N_aT = N_div_base +
% N_div_costim*H_CD28 + N_IL2_CD8*IL2hill, Marchingo 2014). Both were anchored to
% total-division-count measurements (Butz1998 ~15 divisions; Marchingo ~16 at
% saturation), and the Butz submodel forward model literally summed
% log2(n_T_expansion_burst) + N_aT. Removing the fixed burst makes expansion
% context-dependent (costim/IL2-modulated via 2^N_aT) instead of a constant 800x
% floor; the activation rate is still anchored by Bousso2003 (k_T_activation_per_APC).
% Requires SBI re-fit. See mechanism_audit_reconciliation.md (V12).

% === Mechanistic Tumor Killing Parameters ===
% T cell motility in tumor (baseline measurement)
params.v_T_motility.Value = 10;
params.v_T_motility.Units = 'micrometer/minute';
params.v_T_motility.Notes = 'Baseline T cell motility speed in tumor tissue (from intravital microscopy)';
params.v_T_motility.derivedFrom = 'Boissonnas et al. 2007 (PMID: 17220890) - range 5-15 μm/min';
% Volume searched by one T cell per day
params.v_T_search_volume.Value = 1e-4;  % Halle 2016 output-anchored (Issue #192 Phase 2)
params.v_T_search_volume.Units = 'millimeter^3/day';
params.v_T_search_volume.Notes = 'Volume of tumor tissue searched by one T cell per day. Output-anchored to Halle et al. 2016 in-vivo CTL kill rate of 2-16 cells/CTL/day at typical TIL densities. Previous value (1e-6) cited Boissonnas 2007 motility but the derivation was over-reaching: Boissonnas reports CTL velocity only (4-10 um/min intratumoral) and explicitly disclaims kill-rate measurement. Halle 2016 anchor implies ~5e-5 to 4e-4 mm^3/day; mid-range adopted. See Issue #192 for full audit.';
params.v_T_search_volume.derivedFrom = 'Halle et al. 2016 (PMID: 26872694) - In vivo CTL kill capacity 2-16 cells/CTL/day (output-side anchor, not a direct measurement of v_T)';
% Kill probability per contact
params.p_T_kill_per_contact.Value = 0.45;  % Calibrated: 1.5× from baseline (0.3)
params.p_T_kill_per_contact.Units = 'dimensionless';
params.p_T_kill_per_contact.Notes = 'Probability of tumor cell death when contacted by cytotoxic T cell (from in vitro killing assays). Calibrated for PDAC (upper range for effective killing)';
params.p_T_kill_per_contact.derivedFrom = 'Ritter et al. 2015 (PMID: 26567510) - range 0.1-0.5';

% Rate of CD4 T Cell Decay
params.k_CD4_death.Value = 0.1;
params.k_CD4_death.Units = '1/day';
params.k_CD4_death.Notes = '(De Boer and Perelson 1995, PMID: 7475092)';
% Activated CD8 Rate of Transmigration
params.k_CD8_mig.Value = 5.8e-12;
params.k_CD8_mig.Units = '1/minute/cell';
params.k_CD8_mig.Notes = '(Zhu 1996, PMID: 8706023)';
% Activated CD4 Rate of Transmigration
params.k_CD4_mig.Value = 5.8e-12;
params.k_CD4_mig.Units = '1/minute/cell';
params.k_CD4_mig.Notes = '(Zhu 1996, PMID: 8706023)';
% T cell Adhesion Density
params.rho_adh.Value = 5e8; % 5e8-1e9
params.rho_adh.Units = 'cell/centimeter^3';
params.rho_adh.Notes = '(Zhu 1996, PMID: 8706023)';
% Peripheral Vascular Volume Fractions
params.gamma_P.Value = 0.014;
params.gamma_P.Units = 'dimensionless';
params.gamma_P.Notes = '(Finley 2012, PMID: 22547351)';
% Tumour Vascular Volume Fractions
params.gamma_T.Value = 0.02;
params.gamma_T.Units = 'dimensionless';
params.gamma_T.Notes = '(Finley 2012, PMID: 22547351; Stamatelos 2014, PMID: 24342178)';
% Activated CD8+ T Cell Transport C->P
% Re-anchored 2026-06-24: was derived as k_CD8_mig*rho_adh*gamma_P*V_P = 2.436/min,
% a 25-second blood residence (146:1 periphery:blood) — unphysical (~100% extravasation
% per cardiac pass). The k_mig*rho_adh composite multiplied by the full 60 L peripheral
% volume is the source of the inflation; the tumor door q_CD8_T_in correctly omits the
% V_P factor. Now set on a steady-state balance: q_P_in = q_P_out * R, with q_P_out =
% 24/day and R = periphery:blood partition ~15 (blood residence ~4 min, periphery
% residence ~1 h, internally consistent). 24/day * 15 = 360/day = 0.25/min. Free SBI
% param anchored by submodel target q_CD8_P_in_PDAC_deriv001.yaml.
params.q_CD8_P_in.Value = 0.25;
params.q_CD8_P_in.Units = '1/minute';
params.q_CD8_P_in.Notes = 'Steady-state balance q_P_in = q_P_out(24/day) * periphery:blood ratio(~15); blood residence ~4 min. Replaces unphysical derived 2.436/min (25-sec residence). See q_CD8_P_in_PDAC_deriv001.yaml';
% Activated CD4+ T Cell Transport C->P
% NB: q_CD4_P_in is not itself a SimBiology parameter — it only SEEDS the value of
% the real model param q_Treg_P_in (Treg_module.m:81), which governs both V_C.Th->V_P.Th
% (Th_module) and V_C.Treg->V_P.Treg. So the submodel target and priors-CSV row are on
% q_Treg_P_in, not q_CD4_P_in.
params.q_CD4_P_in.Value = 0.25;
params.q_CD4_P_in.Units = '1/minute';
params.q_CD4_P_in.Notes = 'Seeds q_Treg_P_in. Steady-state balance q_P_in = q_P_out(24/day) * periphery:blood ratio(~15); blood residence ~4 min. Replaces unphysical derived 2.436/min (25-sec residence). See q_Treg_P_in_PDAC_deriv001.yaml';
% Activated CD8+ T Cell Transport C->T
params.q_CD8_T_in.Value = [];
params.q_CD8_T_in.Units = '1/(centimeter^3*minute)';
params.q_CD8_T_in.Notes = ['calculated based on T cell transmigration rate ' params.k_CD8_mig.Notes ' and T cell adhesion density ' params.rho_adh.Notes];
params.q_CD8_T_in.Factors = ["k_CD8_mig", "rho_adh", "gamma_T"];
params.q_CD8_T_in.Equation = 'p(1)*p(2)*p(3)';
% Activated CD4+ T Cell Transport C->T
params.q_CD4_T_in.Value = [];
params.q_CD4_T_in.Units = '1/(centimeter^3*minute)';
params.q_CD4_T_in.Notes = ['calculated based on T cell transmigration rate ' params.k_CD4_mig.Notes ' and T cell adhesion density ' params.rho_adh.Notes];
params.q_CD4_T_in.Factors = ["k_CD4_mig", "rho_adh", "gamma_T"];
params.q_CD4_T_in.Equation = 'p(1)*p(2)*p(3)';
% Activated CD8+ T Cell Transport P->C
params.q_CD8_P_out.Value = 24;
params.q_CD8_P_out.Units = '1/day';
params.q_CD8_P_out.Notes = '(De Boer and Perelson 1995, PMID: 7475092)';
% Activated CD4+ T Cell Transport P->C
params.q_CD4_P_out.Value = 24;
params.q_CD4_P_out.Units = '1/day';
params.q_CD4_P_out.Notes = '(De Boer and Perelson 1995, PMID: 7475092)';
% Activated CD8+ T Cell Transport LN->C
params.q_CD8_LN_out.Value = 24.0;
params.q_CD8_LN_out.Units = '1/day';
params.q_CD8_LN_out.Notes = '(De Boer and Perelson 1995, PMID: 7475092)';
% Activated CD4+ T Cell Transport LN->C
params.q_CD4_LN_out.Value = 24.0;
params.q_CD4_LN_out.Units = '1/day';
params.q_CD4_LN_out.Notes = '(De Boer and Perelson 1995, PMID: 7475092)';
% Death Rate Due to T Cells
params.k_C_Tcell.Value = 2.2;
params.k_C_Tcell.Units = '1/day';
params.k_C_Tcell.Notes = '(Robertson-Tessi 2012, PMID: 22051568)';
% Rate of T Cell Exhaustion by Cancer Cells
params.k_Tcell.Value = 0.1;
params.k_Tcell.Units = '1/day';
params.k_Tcell.Notes = '(estimated)';
% Rate of Treg-mediated CD8 T cell exhaustion (CD8 -> CD8_exh transition)
params.k_Treg_CD8_exh.Value = 0.1;
params.k_Treg_CD8_exh.Units = '1/day';
params.k_Treg_CD8_exh.Notes = 'Maximal rate of Treg-mediated CD8 exhaustion under saturating IL-10 and Treg:CD8 ratio';
% Baseline exhaustion rate from chronic antigen exposure
params.k_CD8_basal_exh.Value = 0.05;
params.k_CD8_basal_exh.Units = '1/day';
params.k_CD8_basal_exh.Notes = 'Baseline T cell exhaustion rate from chronic TCR stimulation (estimated, lower than death rate)';
params.k_CD8_basal_exh.derivedFrom = {'estimated'};
% Cancer cell number for half-maximal exhaustion induction
params.K_C_exh.Value = 1e8;
params.K_C_exh.Units = 'cell';
params.K_C_exh.Notes = 'Cancer cell threshold for half-maximal exhaustion induction (estimated)';
params.K_C_exh.derivedFrom = {'estimated'};

% Exhausted CD8 T cell death rate (prevents indefinite accumulation)
params.k_CD8_exh_death.Value = 0.15;
params.k_CD8_exh_death.Units = '1/day';
params.k_CD8_exh_death.Notes = 'Death rate of exhausted CD8 T cells (estimated ~1.5x functional T cell death rate)';
params.k_CD8_exh_death.derivedFrom = {'estimated'};

% Anti-PD-1 reinvigoration rate of progenitor-exhausted CD8 (mid-tier cA-C4).
% NOT a free estimate -- DERIVED via the fate-competition identity: at maximal PD-1
% occupancy the rescuable fraction f = k_reinvig/(k_reinvig + k_CD8_exh_death) equals
% the progenitor-exhausted (Tpex, TCF1+) fraction of exhausted CD8, so
% k_reinvig = k_CD8_exh_death * f/(1-f). With k_CD8_exh_death=0.15/day (itself
% submodel-anchored) and a provisional Tpex fraction f~0.20 (human tumor TIL range
% ~10-30%; PDAC-specific data exists), k_reinvig ~ 0.0375/day. The Tpex-fraction
% anchor is QUEUED for maple extraction -> submodel target k_reinvig_PDAC_deriv001
% (closed-form: f = k_reinvig/(k_reinvig+k_CD8_exh_death)); that posterior will
% shadow this provisional value. derivedFrom left empty (no direct literature value;
% constrained via the submodel target once extracted) -- not tagged 'estimated'.
params.k_reinvig.Value = 0.0375;
params.k_reinvig.Units = '1/day';
params.k_reinvig.Notes = 'Max anti-PD-1 reinvigoration rate of exhausted CD8 at full PD-1 occupancy; derived from the Tpex (progenitor-exhausted) fraction via f = k_reinvig/(k_reinvig+k_CD8_exh_death) (provisional f~0.20; submodel target k_reinvig_PDAC_deriv001 queued for extraction).';
params.k_reinvig.derivedFrom = {};

% Th exhaustion (mirrors CD8 routes; previously Th_exh was a relabeled death pool)
% Baseline exhaustion from chronic antigen exposure
params.k_Th_basal_exh.Value = 0.05;
params.k_Th_basal_exh.Units = '1/day';
params.k_Th_basal_exh.Notes = 'Baseline Th exhaustion rate from chronic TCR stimulation (mirrors k_CD8_basal_exh)';
params.k_Th_basal_exh.derivedFrom = {'estimated'};
% TGFβ-driven Th exhaustion (chronic immunosuppression)
params.k_TGFb_Th_exh.Value = 0.3;
params.k_TGFb_Th_exh.Units = '1/day';
params.k_TGFb_Th_exh.Notes = 'Maximal Th exhaustion rate driven by TGFβ (estimated; literature anchor needed via submodel target — TGFβ chronic immunosuppression of CD4 helper compartment)';
params.k_TGFb_Th_exh.derivedFrom = {'estimated'};

%% IL2 Parameters
% Degradation Rate
params.k_IL2_deg.Value = 0.09079844885257181;
params.k_IL2_deg.Units = '1/minute';
params.k_IL2_deg.Notes = '(Lotze 1985, PMID: 3871099)';
% Maximum Consumption Rate by T Cells
params.k_IL2_cons.Value =  5e-11;
params.k_IL2_cons.Units = 'nanomole/cell/hour';
params.k_IL2_cons.Notes = '(Lotze 1985, PMID: 3871099)';
% Secretion Rate from Activated T Cells
params.k_IL2_sec.Value =  1e-11;
params.k_IL2_sec.Units = 'nanomole/cell/hour';
params.k_IL2_sec.Notes = '(Han 2012, PMID: 22160692; Thurley 2015, PMID: 25923703; Liu 2001)';
% IL2 Concentration for Half-Maximal T Cell Proliferation
params.IL2_50.Value = 0.32;
params.IL2_50.Units = 'nanomolarity';
params.IL2_50.Notes = '(Marchingo 2014, PMID: 25430770)';
% (IL2_50_Treg retired: bulk IL-2 consumption removed in the IL2_niche QSS
% reformulation; Treg IL-2 competition is now carried by kappa_IL2_Treg.)
% Baseline Number of Activated T Cell Generations
params.N_div_base.Value = 2;
params.N_div_base.Units = 'dimensionless';
params.N_div_base.Notes = '(Marchingo 2014, PMID: 25430770)';
% Baseline Number of Activated T Cell Generations for co-stimulation
params.N_div_costim.Value = 1.5105;
params.N_div_costim.Units = 'dimensionless';
params.N_div_costim.Notes = 'Effective basal CD28 costimulation contribution to division destiny. Reanchored from 3 to 1.5105 when the separately fitted APC-side PD-1 multiplier was retired: 3*(1-0.5*0.993)=1.5105 preserves the documented nominal untreated costimulation. Requires updated-sampler refit. Marchingo 2014 PMID: 25430770.';
% Additional Number of Activated CD8+ T Cell Generations Due to IL2
params.N_IL2_CD8.Value = 11;
params.N_IL2_CD8.Units = 'dimensionless';
params.N_IL2_CD8.Notes = '(Marchingo 2014, PMID: 25430770)';
% Additional Number of Activated CD4+ T Cell Generations Due to IL2
params.N_IL2_CD4.Value = 8.5;
params.N_IL2_CD4.Units = 'dimensionless';
params.N_IL2_CD4.Notes = '(Marchingo 2014, PMID: 25430770)';
% ---------------------------------------------------------------------------
% Local IL-2 niche params (quasi-steady-state supply/demand reformulation).
% IL-2 is no longer a bulk dynamic species. The bulk V_LN.IL2 pool was
% degradation-limited (~1e-7 nM, ~1e2-1e4x below the local niche), which left the
% division-destiny IL-2 channel dead (H_IL2 ~ 1e-4) and Treg IL-2 competition
% inert. IL-2 is now an algebraic niche concentration at QSS (it equilibrates in
% minutes vs the ~day division-programming window):
%   IL2_niche = k_IL2_sec*(w_IL2_help*aTh + aCD8)
%               / ( k_IL2_cons*(aCD8 + kappa_IL2_Treg*Treg) + k_IL2_deg*V_niche )
% then N_aT += N_IL2_CD8 * IL2_niche/(IL2_50 + IL2_niche). Supply = autocrine CD8
% + paracrine CD4-helper IL-2 (paracrine dominant for primary expansion, Cell
% Reports 2022); demand = priming CD8 + competing Tregs (local IL-2 restriction =
% dominant Treg suppression, J Immunol 2011); clearance volume V_niche << V_LN is
% the physical replacement for the old f_IL2_synaptic enhancement. This keeps the
% k_IL2_sec / k_IL2_cons / k_IL2_deg / IL2_50 submodel anchors all live.
params.w_IL2_help.Value = 5;
params.w_IL2_help.Units = 'dimensionless';
params.w_IL2_help.Notes = 'Paracrine CD4-helper IL-2 weight vs CD8 autocrine in the niche supply (paracrine dominates primary CD8 expansion + higher per-cell CD4 secretion, Cell Reports 2022 PMID 35417701); estimated.';
params.w_IL2_help.derivedFrom = {};
params.kappa_IL2_Treg.Value = 1;
params.kappa_IL2_Treg.Units = 'dimensionless';
params.kappa_IL2_Treg.Notes = 'Treg IL-2 uptake weight in the niche demand (Tregs are CD25-high and out-compete; local IL-2 restriction is the dominant Treg suppression of CD8, J Immunol 2011 PMID 21849684); estimated.';
params.kappa_IL2_Treg.derivedFrom = {};
% V_niche: clearance volume of a single active IL-2 priming niche in the local QSS.
% SUBMODEL-ANCHORED to the measured IL-2 diffusion-consumption length (Oyler-Yaniv
% 2017, lambda 80-120 um) as a sphere of radius lambda: V_niche = (4*pi/3)*lambda^3
% ~ 4.2e-6 mL at lambda~100 um (submodel_targets/V_niche_PDAC_deriv001.yaml). This is
% the single-niche volume, NOT the aggregate N_clusters*v: the single-niche scale
% gives an IL2_niche consumer-sparse cap of ~0.4-4 nM (matching Busse 2010's 0.1-1 nM
% near-source IL-2, PMID 20133667), whereas the aggregate would force a degradation-
% dominated, sub-IL2_50 niche (~0.005-0.05 nM). The Oyler-Yaniv length is set BY
% consumption (their tunable diffusion-consumption thesis), so at the niche's own
% consumer density the clearance is uptake-dominated by construction -- consistent
% with V_niche being sub-dominant when consumers are present and load-bearing only as
% the consumer-sparse saturation cap. Physical replacement for the retired dimensionless
% f_IL2_synaptic enhancement. SBI fits it over a wide prior around the anchored center.
params.V_niche.Value = 4.18879e-6;
params.V_niche.Units = 'milliliter';
params.V_niche.Notes = 'Single IL-2 priming-niche clearance volume for the local QSS (<< V_LN). SUBMODEL-ANCHORED to Oyler-Yaniv 2017 (PMID 28389069): V_niche = (4*pi/3)*lambda^3 with niche radius = measured IL-2 diffusion-consumption length lambda (80-120 um), ~4.2e-6 mL; reproduces the Busse 2010 (PMID 20133667) near-source 0.1-1 nM IL-2 cap. Physical replacement for the retired f_IL2_synaptic; single-niche (not aggregate N_clusters) volume.';
params.V_niche.derivedFrom = {};

%% Naive T Cell Parameters
% Naive CD8+ T Cell Density in Blood
params.rho_nCD8.Value = 5.05e5; % 505.0, 410.5–825.4 cells/uL
params.rho_nCD8.Units = 'cell/milliliter';
params.rho_nCD8.Notes = '(Autissier 2010, PMID: 20099249)';
% Naive CD4+ T Cell Density in Blood
params.rho_nCD4.Value = 8.6e5; % 860.7, 520.8–1173.2 cells/uL
params.rho_nCD4.Units = 'cell/milliliter';
params.rho_nCD4.Notes = '(Autissier 2010, PMID: 20099249)';
% Number of Naive CD8+ T Cell in Blood
params.nCD8_C.Value = [];
params.nCD8_C.Units = 'cell';
params.nCD8_C.Notes = '';
params.nCD8_C.Factors = ["rho_nCD8", "V_C"];
params.nCD8_C.Equation = 'p(1)*p(2)';
% Number of Naive CD4+ T Cell in Blood
params.nCD4_C.Value = [];
params.nCD4_C.Units = 'cell';
params.nCD4_C.Notes = '';
params.nCD4_C.Factors = ["rho_nCD4", "V_C"];
params.nCD4_C.Equation = 'p(1)*p(2)';
% Number of Naive CD8+ T Cell in Peripheral
params.nCD8_P.Value = [];
params.nCD8_P.Units = 'cell';
params.nCD8_P.Notes = ' estimated assuming a 50x T cell number in peripheral compartment (Braber 2012, PMID: 22365666)';
params.nCD8_P.Factors = ["nCD8_C"];
params.nCD8_P.Equation = 'p(1)*50';
% Number of Naive CD4+ T Cell in Peripheral
params.nCD4_P.Value = [];
params.nCD4_P.Units = 'cell';
params.nCD4_P.Notes = ' estimated assuming a 50x T cell number in peripheral compartment (Braber 2012, PMID: 22365666)';
params.nCD4_P.Factors = ["nCD4_C"];
params.nCD4_P.Equation = 'p(1)*50';
% Number of Naive CD8+ T Cell in Lymph Nodes
params.nCD8_LN.Value = [];
params.nCD8_LN.Units = 'cell';
params.nCD8_LN.Notes = ' calculated based on the steady-state naive T cell level in healthy individuals (Autissier 2010, PMID: 20099249)';
params.nCD8_LN.Factors = ["nCD8_C"];
params.nCD8_LN.Equation = 'p(1)*0.05';
% Number of Naive CD4+ T Cell in Lymph Nodes
params.nCD4_LN.Value = [];
params.nCD4_LN.Units = 'cell';
params.nCD4_LN.Notes = ' calculated based on the steady-state naive T cell level in healthy individuals (Autissier 2010, PMID: 20099249)';
params.nCD4_LN.Factors = ["nCD4_C"];
params.nCD4_LN.Equation = 'p(1)*0.042';

% Naive CD8+ T Cell Diversity
params.nCD8_div.Value = 1.11e6;
params.nCD8_div.Units = 'dimensionless';
params.nCD8_div.Notes = '(Robins 2009, PMID: 19706884)';
% Naive CD4+ T Cell Diversity
params.nCD4_div.Value = 1.16e6;
params.nCD4_div.Units = 'dimensionless';
params.nCD4_div.Notes = '(Robins 2009, PMID: 19706884)';
% Naive T Rate of Transmigration
params.k_nT_mig.Value = 4.2e-13;
params.k_nT_mig.Units = '1/minute/cell';
params.k_nT_mig.Notes = '(Zhu 1996, PMID: 8706023)';
% Naive T cell density for half-maximal peripheral proliferation
params.K_nT_pro.Value = 1e9;
params.K_nT_pro.Units = 'cell';
params.K_nT_pro.Notes = '(Braber 2012, PMID: 22365666)';
% Rate of naive T cell death
params.k_nT_death.Value = 0.002;
params.k_nT_death.Units = '1/day';
params.k_nT_death.Notes = '(Braber 2012, PMID: 22365666)';
% Thymic output of naive CD4+ T Cells into the blood
params.Q_nCD4_thym.Value = 1e6; % adjusted for PDAC demographic (age 60+, thymic involution)
params.Q_nCD4_thym.Units = 'cell/day';
params.Q_nCD4_thym.Notes = '(Bains 2009, PMID: 19179300; Yeh 2017, PMID: 28912973)';
% Thymic output of naive CD8+ T Cells into the blood
params.Q_nCD8_thym.Value = 3e6;
params.Q_nCD8_thym.Units = 'cell/day';
params.Q_nCD8_thym.Notes = '(Bains 2009, PMID: 19179300; Ye 2002, PMID: 11994448)';
% Rate of naive CD4+ T Cell proliferation
params.k_nCD4_pro.Value = 3.2e8;
params.k_nCD4_pro.Units = 'cell/day';
params.k_nCD4_pro.Notes = '(Braber 2012, PMID: 22365666)';
% Rate of naive CD8+ T Cell proliferation
params.k_nCD8_pro.Value = 3.2e8;
params.k_nCD8_pro.Units = 'cell/day';
params.k_nCD8_pro.Notes = '(Braber 2012, PMID: 22365666)';
% Naive CD8+ T Cell Transport C->P
params.q_nCD8_P_in.Value = [];
params.q_nCD8_P_in.Units = '1/minute';
params.q_nCD8_P_in.Notes = ['calculated based on T cell transmigration rate ' params.k_nT_mig.Notes ' and T cell adhesion density ' params.rho_adh.Notes];
params.q_nCD8_P_in.Factors = ["k_nT_mig", "rho_adh", "gamma_P", "V_P"];
params.q_nCD8_P_in.Equation = 'p(1)*p(2)*p(3)*p(4)';
% Naive CD4+ T Cell Transport C->P
params.q_nCD4_P_in.Value = [];
params.q_nCD4_P_in.Units = '1/minute';
params.q_nCD4_P_in.Notes = ['calculated based on T cell transmigration rate ' params.k_nT_mig.Notes ' and T cell adhesion density ' params.rho_adh.Notes];
params.q_nCD4_P_in.Factors = ["k_nT_mig", "rho_adh", "gamma_P", "V_P"];
params.q_nCD4_P_in.Equation = 'p(1)*p(2)*p(3)*p(4)';
% Naive CD8+ T Cell Transport C->T
params.q_nCD8_T_in.Value = [];
params.q_nCD8_T_in.Units = '1/(centimeter^3*minute)';
params.q_nCD8_T_in.Notes = ['calculated based on T cell transmigration rate ' params.k_nT_mig.Notes ' and T cell adhesion density ' params.rho_adh.Notes];
params.q_nCD8_T_in.Factors = ["k_nT_mig", "rho_adh", "gamma_T"];
params.q_nCD8_T_in.Equation = 'p(1)*p(2)*p(3)';
% Naive CD4+ T Cell Transport C->T
params.q_nCD4_T_in.Value = [];
params.q_nCD4_T_in.Units = '1/(centimeter^3*minute)';
params.q_nCD4_T_in.Notes = ['calculated based on T cell transmigration rate ' params.k_nT_mig.Notes ' and T cell adhesion density ' params.rho_adh.Notes];
params.q_nCD4_T_in.Factors = ["k_nT_mig", "rho_adh", "gamma_T"];
params.q_nCD4_T_in.Equation = 'p(1)*p(2)*p(3)';
% Naive CD8+ T Cell Transport P->C
params.q_nCD8_P_out.Value = 5.1;
params.q_nCD8_P_out.Units = '1/day';
params.q_nCD8_P_out.Notes = ' calculated based on steady-state naive T cell density in healthy individuals (Autissier 2010, PMID: 20099249)';
% Naive CD4+ T Cell Transport P->C
params.q_nCD4_P_out.Value = 5.1;
params.q_nCD4_P_out.Units = '1/day';
params.q_nCD4_P_out.Notes = ' calculated based on steady-state naive T cell density in healthy individuals (Autissier 2010, PMID: 20099249)';
% Naive CD8+ T Cell Lymph Node Exit Rate
params.q_nCD8_LN_out.Value = 1.8;
params.q_nCD8_LN_out.Units = '1/day';
params.q_nCD8_LN_out.Notes = '(Mandl 2012, PMID: 23071319)';
% Naive CD4+ T Cell Lymph Node Exit Rate
params.q_nCD4_LN_out.Value = 2.88;
params.q_nCD4_LN_out.Units = '1/day';
params.q_nCD4_LN_out.Notes = '(Mandl 2012, PMID: 23071319)';
% Naive CD8+ T Cell Lymph Node Entry Rate
params.q_nCD8_LN_in.Value = 0.076;
params.q_nCD8_LN_in.Units = '1/day';
params.q_nCD8_LN_in.Notes = ' calculated based on the steady-state naive T cell level in healthy individuals (Autissier 2010, PMID: 20099249)';
% Naive CD4+ T Cell Lymph Node Entry Rate
params.q_nCD4_LN_in.Value = 0.1;
params.q_nCD4_LN_in.Units = '1/day';
params.q_nCD4_LN_in.Notes = ' calculated based on the steady-state naive T cell level in healthy individuals (Autissier 2010, PMID: 20099249)';


%% APC Module Parameters
% DC subtype fraction (cDC1 = cross-presenting BATF3+; cDC2 = MHC-II presenting IRF4+)
params.f_cDC1.Value = 0.2;
params.f_cDC1.Units = 'dimensionless';
params.f_cDC1.Notes = 'cDC1/(cDC1+cDC2) fraction (Hegde 2020, PMID: 31932165; Broz 2014, PMID: 25417636)';
params.f_cDC1.derivedFrom = {};
% Rate of cDC1 Maturation (BATF3+ cross-presenting DCs)
params.k_APC_mat_cDC1.Value = 2.0;
params.k_APC_mat_cDC1.Units = '1/day';
params.k_APC_mat_cDC1.Notes = 'cDC1 maturation rate; slightly higher than cDC2 (BATF3 program, Spranger 2017, PMID: 28983061)';
params.k_APC_mat_cDC1.derivedFrom = {};
% Rate of cDC2 Maturation (IRF4+ MHC-II presenting DCs)
params.k_APC_mat_cDC2.Value = 1.5;
params.k_APC_mat_cDC2.Units = '1/day';
params.k_APC_mat_cDC2.Notes = 'cDC2 maturation rate; same as prior generic rate (Chen 2014, PMID: 25184733)';
params.k_APC_mat_cDC2.derivedFrom = {};
% Rate of APC Migration
params.k_APC_mig.Value = 4.0;
params.k_APC_mig.Units = '1/day';
params.k_APC_mig.Notes = '(Russo Halin 2016, PMID: 26876174)';
% Rate of APC Death
params.k_APC_death.Value = 0.15;
params.k_APC_death.Units = '1/day';
params.k_APC_death.Notes = '(Marino and Kirschner 2004, PMID: 15038983)';
% Rate of mAPC Death
params.k_mAPC_death.Value = 0.3;
params.k_mAPC_death.Units = '1/day';
params.k_mAPC_death.Notes = '(Marino and Kirschner 2004, PMID: 15038983)';
% Rate of antigen transfer from migratory (GVAX-derived) mcDC2 to LN-resident cDC1
% Drives V_LN.cDC1 -> V_LN.mcDC1 at rate k * V_LN.cDC1 * V_LN.mcDC2 (bimolecular).
% Baseline flux (V_LN.mcDC2 ~ 27 cell, V_LN.cDC1 ~ 2.4e5*V_LN cell) is negligible;
% under GVAX (V_LN.mcDC2 ~ 3.6e4) flux grows 1e3x and drives CD8 cross-priming via
% the canonical migratory-DC -> resident-cDC1 antigen handoff (Route B).
params.k_cDC1_Ag_transfer.Value = 1e-7;
params.k_cDC1_Ag_transfer.Units = '1/(cell*day)';
params.k_cDC1_Ag_transfer.Notes = 'Antigen-transfer rate constant: migratory mcDC2 -> resident cDC1 cross-licensing (Ferris 2022 PMID: 35648641; Mueller 2017 Eur J Immunol)';
params.k_cDC1_Ag_transfer.derivedFrom = {'Ferris 2022 PMID: 35648641'};
% APC Density in Tumour
params.APC0_T.Value = 2.7e6;
params.APC0_T.Units = 'cell/milliliter';
params.APC0_T.Notes = '(Lavin 2017, PMID: 28475900)';
% APC Density in LN
params.APC0_LN.Value = 1.2e6;
params.APC0_LN.Units = 'cell/milliliter';
params.APC0_LN.Notes = '(Catron 2004, PMID: 15357945)';
% Independent DC subtype recruitment-target densities (decouples cDC1/cDC2 for SBI)
% NOTE: cDC{1,2}_recruit_target_T is *not* a realized tissue density — it is
% the recruitment-rate numerator (k_recruit / k_APC_death). Steady-state realized
% density is attenuated ~50-200× by H_pore_APC (ECM exclusion) and maturation
% drain. Calibrate via realized-density observables (cdc{1,2}_density_baseline)
% through SBI, not as a direct literature anchor. See
% notes/model/cdc1_homeostasis_vs_recruitment.md for the biology rationale
% (Hegde 2020 / Lin 2020 / Salmon 2016: tumor cDC1 are recruitment-and-apoptosis
% dominated, not homeostatically defended).
params.cDC1_recruit_target_T.Value = 5.4e5;   % anchored to cdc1_density_baseline_PDAC (Plesca 2022): realized cDC1 ~9.84 cells/mm^2 needs ~4.6e5 cell/mL pre pore/maturation attenuation. (Prior Lavin/Hegde + 0.2*APC0_T=2.7e6 framing was wrong provenance: Lavin is lung %, Hegde is a cDC-paucity paper. was 5e4 -> cdc1_density ran ~30x low.)
params.cDC1_recruit_target_T.Units = 'cell/milliliter';
params.cDC1_recruit_target_T.Notes = 'cDC1 recruitment-target density in tumor (recruit-rate / k_APC_death numerator); calibrated via the cdc1_density_baseline_PDAC realized-density observable (Plesca 2022, Cancers, DOI 10.3390/cancers14051216: PDAC mIHC CLEC9A+ cDC1 ~9.84/mm^2).';
params.cDC1_recruit_target_T.derivedFrom = {'Plesca 2022 DOI: 10.3390/cancers14051216 (via cdc1_density_baseline_PDAC observable)'};
params.cDC2_recruit_target_T.Value = 2e5;  % anchored to cdc2_density_baseline_PDAC (Plesca 2022): realized cDC2 ~2.98 cells/mm^2 needs ~1.1e5 cell/mL pre attenuation. (NOT 0.8*2.7e6=2.16e6 -- PDAC inverts the usual ordering, cDC1>cDC2, so the legacy f_cDC2*APC0_T formula does not apply.)
params.cDC2_recruit_target_T.Units = 'cell/milliliter';
params.cDC2_recruit_target_T.Notes = 'cDC2 recruitment-target density in tumor (recruit-rate / k_APC_death numerator); calibrated via the cdc2_density_baseline_PDAC realized-density observable (Plesca 2022, Cancers, DOI 10.3390/cancers14051216: PDAC mIHC cDC2 ~2.98/mm^2, below cDC1 ~9.84 -- PDAC-inverted ordering).';
params.cDC2_recruit_target_T.derivedFrom = {'Plesca 2022 DOI: 10.3390/cancers14051216 (via cdc2_density_baseline_PDAC observable)'};
params.APC0_cDC1_LN.Value = 2.4e5;  % = 0.2 * 1.2e6
params.APC0_cDC1_LN.Units = 'cell/milliliter';
params.APC0_cDC1_LN.Notes = 'cDC1 homeostatic density in LN (Catron 2004 PMID: 15357945)';
params.APC0_cDC1_LN.derivedFrom = {'Catron 2004 PMID: 15357945'};
params.APC0_cDC2_LN.Value = 9.6e5;  % = 0.8 * 1.2e6
params.APC0_cDC2_LN.Units = 'cell/milliliter';
params.APC0_cDC2_LN.Notes = 'cDC2 homeostatic density in LN (Catron 2004 PMID: 15357945)';
params.APC0_cDC2_LN.derivedFrom = {'Catron 2004 PMID: 15357945'};
% Half-maximal tumor death rate for HMGB1 (DAMP)-mediated APC maturation.
% Renamed DAMP_50 -> HMGB1_50 (mid-tier cA-G1): HMGB1 is the representative
% maturation DAMP (soluble, TLR4-signaling; calreticulin is "eat-me", ATP is
% "find-me"), so the parameter is named for the mediator it is anchored to.
% H_HMGB1 = R_Tcell/(R_Tcell + HMGB1_50); R_Tcell is the realized T-cell killing
% FLUX (cell/day), used as the HMGB1-release proxy. The old 1e4 estimate assumed a
% 1e6-cell tumor * 0.01/day = 1e4 cell/day, but the diagnosis attractor is ~4e7
% cancer cells with realized R_Tcell ~ 4.3e6 cell/day (overlay trace 2026-06-03).
% At HMGB1_50=1e4 that pins H_HMGB1 = 0.9977 across the ENTIRE prior (median 1e4,
% sigma 1.0 reaches only ~2e5 = +3 sigma -> H_HMGB1 ~ 0.96; de-saturating to 0.5
% needs +6.1 sigma), which makes the IL-12 *amplification* arm of the OR-gate
% (1-(1-H_HMGB1)*(1-H_IL12)) inert and defeats the "DAMP bootstrap, IL-12 amplify"
% design (APC_module.m:8). Re-centered to 1e6 (H_HMGB1 ~ 0.8 at the operating
% R_Tcell, restoring dynamic range for IL-12 / therapy modulation) per the
% parameter's own "half-max ~ operating death flux" definition -- NOT to fit any
% calibration observable. Downstream effect on cDC1/mcDC1 priming is validated by
% SBI re-fit. See mechanism_audit_reconciliation.md (V4).
%
% LITERATURE-ANCHOR ROUTE (deferred to calibration): at QSS (HMGB1 t1/2 ~min-h <<
% day-scale model) the flux half-max derives from concentration-frame quantities,
% HMGB1_50 = EC50_HMGB1 * k_deg_HMGB1 * V_T / q_release, with grounded inputs
% q_release ~0.04-0.1 pg/cell (10^6 molecules/cell; Messmer 2004) and t1/2 ~17 min
% (serum) - 3 h (culture; PMID 23447529, disulfide form). The DC-maturation EC50 is
% the open factor (no published titration; working dose ~1 ug/mL ~40 nM) -> a single
% future submodel-extraction target. The mechanistic estimate suggests tumor HMGB1
% (~tens of ng/mL) sits below that EC50 -> weaker baseline licensing than the
% hand-set 0.8; resolving that magnitude is a calibration question, not done here.
params.HMGB1_50.Value = 1e6;
params.HMGB1_50.Units = 'cell/day';
params.HMGB1_50.Notes = 'Half-max T-cell killing FLUX (cell/day) for HMGB1 (DAMP)-mediated APC maturation; anchored to the realized diagnosis-attractor killing flux (~4e6 cell/day, overlay trace) so H_HMGB1 sits mid-range (~0.8) instead of pinned at 1. Literature-anchor route deferred: HMGB1_50 = EC50*k_deg*V_T/q_release (EC50 the open factor).';
params.HMGB1_50.derivedFrom = {};
% Half-max death FLUX (cell/day) for HMGB1 (DAMP)-mediated maturation at the GVAX
% injection depot. H_HMGB1_ID = (k_GVAX_clearance*V_ID.GVAX_cells)/(... + HMGB1_50_ID).
% Depot operating death flux ~ k_GVAX_clearance(0.1/day) * GVAX_dose(5e8 cell) = 5e7
% cell/day at peak; anchored to 1e7 so the gate sits responsive (~0.83 at peak,
% declining as the depot clears) -- same "half-max ~ operating death flux" idiom as
% HMGB1_50, NOT a fit. Same deferred literature-anchor route (differs from the tumor
% value only through the depot volume V_ID in the QSS derivation).
params.HMGB1_50_ID.Value = 1e7;
params.HMGB1_50_ID.Units = 'cell/day';
params.HMGB1_50_ID.Notes = 'Half-max death FLUX (cell/day) for HMGB1 (DAMP)-mediated APC maturation at the GVAX injection depot; anchored to the depot operating death flux (~5e7 cell/day peak) so H_HMGB1_ID sits responsive. Literature-anchor route deferred (HMGB1_50_ID = EC50*k_deg*V_ID/q_release).';
params.HMGB1_50_ID.derivedFrom = {};
% Half-maximal TGFb for inhibition of APC maturation
% Same sensitivity as macrophage polarization (TGFb_50_M2pol = 0.07 nM)
params.TGFb_50_APC.Value = 0.07;
params.TGFb_50_APC.Units = 'nanomolarity';
params.TGFb_50_APC.Notes = 'Estimated; DCs sensitive to TGFb-mediated suppression (Batlle & Massague 2019, PMID: 30580966)';
params.TGFb_50_APC.derivedFrom = {};
% Maximum Number of T Cells an APC can Interact with
params.n_sites_APC.Value = 10;
params.n_sites_APC.Units = 'dimensionless';
params.n_sites_APC.Notes = '(De Boer and Perelson 1995, PMID: 7475092)';

% Antigen Module Parameters
% Number of MHC Molecule Types
params.N_MHC.Value = 1;
params.N_MHC.Units = 'dimensionless';
params.N_MHC.Notes = '';
% Total Amount of MHC
params.n_MHC_T.Value = 2e6;
params.n_MHC_T.Units = 'molecule';
params.n_MHC_T.Notes = '(Chen 2014, PMID: 25184733)';
% Total Amount of MHC per Area
params.MHC_T.Value = [];
params.MHC_T.Units = 'molecule';
params.MHC_T.Notes = ['calculated based on the total amount of MHC ' params.n_MHC_T.Notes ...
                    ' and the total surface area'];
params.MHC_T.Factors = ["n_MHC_T", "A_endo", "N_endo", "A_s"];
params.MHC_T.Equation = 'p(1)/(p(2)*p(3) + p(4))';
% Rate of MHC Internalization
params.kin.Value = 14.4;
params.kin.Units = '1/day';
params.kin.Notes = '(Chen 2014, PMID: 25184733)';
% Rate of MHC Externalization
params.kout.Value = 28.8;
params.kout.Units = '1/day';
params.kout.Notes = '(Chen 2014, PMID: 25184733)';
% Number of Endosomal Vesicles per Cell
params.N_endo.Value = 10;
params.N_endo.Units = 'dimensionless';
params.N_endo.Notes = '(Agrawal and Linderman 1996, PMID: 8944895)';
% Endosomal Volume
params.V_endo.Value = 4.0e-17;
params.V_endo.Units = 'liter';
params.V_endo.Notes = '(Agrawal and Linderman 1996, PMID: 8944895)';
% Endosomal Surface Area
params.A_endo.Value = 1.5;
params.A_endo.Units = 'micrometer^2';
params.A_endo.Notes = '(Agrawal and Linderman 1996, PMID: 8944895)';
% Endosomal Surface Area
params.A_s.Value = 900.0;
params.A_s.Units = 'micrometer^2';
params.A_s.Notes = '(Agrawal and Linderman 1996, PMID: 8944895)';
% Endosomal Volume
params.V_e.Value = [];
params.V_e.Units = 'liter';
params.V_e.Notes = ['calculated based on the volume of a single endosome ' params.V_endo.Notes ...
                        ' and the number of endosomes per cell ' params.N_endo.Notes];
params.V_e.Factors = ["V_endo", "N_endo"];
params.V_e.Equation = 'p(1)*p(2)';
% Endosomal Surface Area
params.A_e.Value = [];
params.A_e.Units = 'micrometer^2';
params.A_e.Notes = ['calculated based on the surface area of a single endosome ' params.A_endo.Notes ...
                        ' and the number of endosomes per cell ' params.N_endo.Notes];
params.A_e.Factors = ["A_endo", "N_endo"];
params.A_e.Equation = 'p(1)*p(2)';
% Surface Area of T Cells
params.A_Tcell.Value = [];
params.A_Tcell.Units = 'micrometer^2';
params.A_Tcell.Notes = ['calculated based on the average T cell diameter ' params.D_Tcell.Notes];
params.A_Tcell.Factors = ["D_Tcell"];
params.A_Tcell.Equation = '4*pi*(p(1)/2)^2';
% Surface Area of Cancer Cells
params.A_cell.Value = [];
params.A_cell.Units = 'micrometer^2';
params.A_cell.Notes = ['calculated based on the average cancer cell diameter ' params.D_cell.Notes];
params.A_cell.Factors = ["D_cell"];
params.A_cell.Equation = '4*pi*(p(1)/2)^2';
% Surface Area of APC Cells
params.A_APC.Value = 900.0;
params.A_APC.Units = 'micrometer^2';
params.A_APC.Notes = '(Agrawal and Linderman 1996, PMID: 8944895)';
% Rate of Antigen Uptake
params.k_up.Value = 14.4;
params.k_up.Units = '1/day/cell';
params.k_up.Notes = '(Chen 2014, PMID: 25184733)';
% Rate of Extracellular Antigen Degradation
params.k_xP_deg.Value = 2.0;
params.k_xP_deg.Units = '1/day';
params.k_xP_deg.Notes = '(Palsson 2013, PMID: 24074340)';
% Rate of Endosomal Antigen Degradation
params.k_P_deg.Value = 17.28;
params.k_P_deg.Units = '1/day';
params.k_P_deg.Notes = '(Chen 2014, PMID: 25184733)';
% Rate of Endosomal Epitope Degradation
params.k_p_deg.Value = 144.0;
params.k_p_deg.Units = '1/day';
params.k_p_deg.Notes = '(Chen 2014, PMID: 25184733)';
% Rate of Antigen Binding
params.k_on.Value = 1.44e5;
params.k_on.Units = '1/day/molarity';
params.k_on.Notes = '(Agrawal and Linderman 1996, PMID: 8944895)';
% Number of Epitope Molecules for Half-Maximal T Cell Activation
params.N_p_50.Value = 1e-3;
params.N_p_50.Units = 'molecule';
params.N_p_50.Notes = '(Kimachi 1997, PMID: 9464819)';
% TCR-pMHC Concentration for Half-Maximal T Cell Activation
params.p_50.Value = [];
params.p_50.Units = 'molecule/micrometer^2';
params.p_50.Notes = ['calculated based on the number of molecules for half-maximal T cell activation '...
                         params.N_p_50.Notes ' and synapse surface area'];
params.p_50.Factors = ["N_p_50","A_syn"];
params.p_50.Equation = 'p(1)/p(2)';

% Half-maximal endosomal tumor-antigen concentration for MHC-II presentation
% (mid-tier mechanism #3, cA-H4). Gates conventional Th priming via
% H_P1_II = V_e.P1/(V_e.P1 + p1_II_50) (rule in Th_module.m). This is the MHC-II
% analog of the MHC-I p1_50 KPR threshold: Th help against the tumor antigen used
% to borrow H_P1 (the MHC-I cross-presentation pMHC pool that licenses CD8), which
% conflated the two antigen-processing pathways. The MHC-II axis shares the same
% internalized endosomal antigen (V_e.P1, what cDC2 endocytoses) but is decoupled
% from the cDC1 cross-presentation surface-complex/KPR cascade. The MHC-II surface
% loading/display chain is collapsed to QSS (a single saturation on the internalized
% pool), mirroring the no-new-species parsimony of mechanisms #1/#2.
% PROVENANCE = algebraic-derived, not estimated: the literature anchor is the
% ~50-200 surface pMHC-II complexes/APC that trigger CD4 activation (Demotz 1990 /
% Harding 1990; reviewed Frontiers Immunol 2013, 10.3389/fimmu.2013.00340). Mapping
% that surface-count threshold to an internalized-antigen molarity requires the
% MHC-II loading/display geometry we deliberately collapse, so the realized EC50 is
% the OPEN FACTOR (same situation as #1's HMGB1 EC50). SCALE IS UNANCHORED here: the
% V_e.P1 operating range has not been traced this (structural-only) pass, so the
% value below is a placeholder and the prior is wide; a future calibration pass must
% center it on the simulated V_e.P1 band (cf. the Th_50_help re-centering) or it
% risks pinning H_P1_II at 0/1 and rendering the MHC-II axis inert. Submodel target
% p1_II_50_PDAC_deriv queued in notes/reconciliation_targets.csv.
params.p1_II_50.Value = 1e-11;  % molarity; trace-centered 2026-06-17 to V_e.P1 scale
params.p1_II_50.Units = 'molarity';
params.p1_II_50.Notes = 'Half-maximal internalized endosomal tumor-antigen (V_e.P1) concentration for MHC-II presentation gating conventional Th priming (mechanism #3). Algebraic-derived from the ~50-200 pMHC-II/APC CD4-activation threshold; realized molarity is the open factor (loading-geometry mapping collapsed to QSS). Re-centered 1e-9 -> 1e-11 on 2026-06-17: a posterior-cloud trace found V_e.P1 ~1.2e-11 M, so the old 1e-9 placeholder sat ~80x above the antigen supply and clamped H_P1_II ~= 0.01 (CD4 activation throttled ~80x, Th starved, treg_fraction_cd4 pegged at 1). Centering on V_e.P1 puts the gate in its responsive mid-range (H_P1_II ~0.5). Confirmed lever (sweep: gate-open raises aTh ~34x, V_T.Th ~12x) but NECESSARY-AND-INSUFFICIENT: Th remains ~450x below the treg_fraction_cd4 target because Th lacks the tumor-local self-renewal niche Treg has (see notes/calibration/cd4_treg_collapse_rediagnosis.md, Th-niche asymmetry). sigma=2.0 left wide for SBI. submodel target p1_II_50_PDAC_deriv still queued (only worth extracting if the pMHC-II -> V_e.P1 molarity mapping is tractable).';
params.p1_II_50.derivedFrom = {};

% TCR signal - kinetic proofreading with limited signaling
% Number of TCR molecules on naive T cells
params.TCR_tot_abs.Value = 15708;
params.TCR_tot_abs.Units = 'molecule';
params.TCR_tot_abs.Notes = '(Lever 2014, PMID: 25145757)';
% TCR molecules density on naive T cells
params.TCR_tot.Value = [];
params.TCR_tot.Units = 'molecule/micrometer^2';
params.TCR_tot.Notes = ['calculated based on the number of molecules for TCR '...
                         params.TCR_tot_abs.Notes ' and T cell surface area'];
params.TCR_tot.Factors = ["TCR_tot_abs","A_Tcell"];
params.TCR_tot.Equation = 'p(1)/p(2)';

% Rate of modification of TCRs
params.k_TCR_p.Value = 1;
params.k_TCR_p.Units = '1/second';
params.k_TCR_p.Notes = '(Lever 2014, PMID: 25145757)';
% Unbinding rate of ag/MHC to TCR
params.k_TCR_off.Value = 1;
params.k_TCR_off.Units = '1/second';
params.k_TCR_off.Notes = '(Lever 2014, PMID: 25145757)';
% binding rate of ag/MHC to TCR
params.k_TCR_on.Value = 1e-0; %3.1831e-5;
params.k_TCR_on.Units = '1/(second*molecule/micrometer^2)';
params.k_TCR_on.Notes = '(Lever 2014, PMID: 25145757)';
% Rate of modification of TCR that leads to non-signaling
params.phi_TCR.Value = 0.09;
params.phi_TCR.Units = '1/second';
params.phi_TCR.Notes = '(Lever 2014, PMID: 25145757)';
% Number of intermediate steps
params.N_TCR.Value = 10;
params.N_TCR.Units = 'dimensionless';
params.N_TCR.Notes = '(Lever 2014, PMID: 25145757)';

%% Checkpoint Module Parameters
% Surface area of the synapse
params.A_syn.Value = 37.8;
params.A_syn.Units = 'micrometer^2';
params.A_syn.Notes = '(Jansson 2005, PMID: 16034096)';
% The synapse gap distance (kd2D = kd3D*d_syn)
params.d_syn.Value = 3.0;
params.d_syn.Units = 'nanometer';
params.d_syn.Notes = '(Jansson 2005, PMID: 16034096)';
% Expression rate of PDL1 on tumor cells
params.k_out_PDL1.Value = 5e4;
params.k_out_PDL1.Units = 'molecule/day';
params.k_out_PDL1.Notes = '(Mimura 2018, PMID: 29034543)';
% Degradation rate of PDL1 on tumor cells
params.k_in_PDL1.Value = 1;
params.k_in_PDL1.Units = '1/day';
params.k_in_PDL1.Notes = '(Hsu 2018, PMID: 30442814)';
% PD1/PDL1 Parameters
% PD1-PDL1 kd
params.kd_PD1_PDL1.Value = 8.2;
params.kd_PD1_PDL1.Units = 'micromolarity';
params.kd_PD1_PDL1.Notes = '(Cheng 2013, PMID: 23417675)';
% PD1-PDL1 kon
params.kon_PD1_PDL1_3D.Value = 0.18;
params.kon_PD1_PDL1_3D.Units = '1/(micromolarity*second)';
params.kon_PD1_PDL1_3D.Notes = '(Cheng 2013, PMID: 23417675)';
% PD1-PDL1 kon for 2D
params.kon_PD1_PDL1.Value = [];
params.kon_PD1_PDL1.Units = '';
params.kon_PD1_PDL1.Notes = ['kon for PD1-PDL1 ' params.kon_PD1_PDL1_3D.Notes];
params.kon_PD1_PDL1.Factors = ["kon_PD1_PDL1_3D","d_syn"];
params.kon_PD1_PDL1.Equation = 'p(1)/p(2)';
% PD1-PDL2 kd
params.kd_PD1_PDL2.Value = 2.3;
params.kd_PD1_PDL2.Units = 'micromolarity';
params.kd_PD1_PDL2.Notes = '(Cheng 2013, PMID: 23417675)';
% PD1-PDL2 kon
params.kon_PD1_PDL2_3D.Value = 0.24;
params.kon_PD1_PDL2_3D.Units = '1/(micromolarity*second)';
params.kon_PD1_PDL2_3D.Notes = '(Cheng 2013, PMID: 23417675)';
% PD1-PDL2 kon for 2D
params.kon_PD1_PDL2.Value = [];
params.kon_PD1_PDL2.Units = '';
params.kon_PD1_PDL2.Notes = '(Cheng 2013, PMID: 23417675)';
params.kon_PD1_PDL2.Factors = ["kon_PD1_PDL2_3D","d_syn"];
params.kon_PD1_PDL2.Equation = 'p(1)/p(2)';
% PD1-aPD1 kd
params.kd_PD1_aPD1.Value = 4;
params.kd_PD1_aPD1.Units = 'nanomolarity';
params.kd_PD1_aPD1.Notes = '(Wang 2014, PMID: 24872026; Brown 2020, PMID: 32134960)';
% PD1-aPD1 kon
params.kon_PD1_aPD1.Value = 6.7e5;
params.kon_PD1_aPD1.Units = '1/(molarity*second)';
params.kon_PD1_aPD1.Notes = '(Wang 2014, PMID: 24872026; Brown 2020, PMID: 32134960)';
% PD1-aPD1 Chi (antibody cross-arm binding strength)
params.Chi_PD1_aPD1_3D.Value = 100;
params.Chi_PD1_aPD1_3D.Units = 'dimensionless';
params.Chi_PD1_aPD1_3D.Notes = '(estimated based on Wang 2014, PMID: 24872026)';
% PD1-aPD1 Chi for 2D
params.Chi_PD1_aPD1.Value = [];
params.Chi_PD1_aPD1.Units = '';
params.Chi_PD1_aPD1.Notes = '(estimated based on Wang 2014, PMID: 24872026)';
params.Chi_PD1_aPD1.Factors = ["Chi_PD1_aPD1_3D","d_syn","N_avg"];
params.Chi_PD1_aPD1.Equation = 'p(1)/(p(2)*p(3))';
% PD1-PDL1 koff
params.koff_PD1_PDL1.Value = [];
params.koff_PD1_PDL1.Units = '1/second';
params.koff_PD1_PDL1.Notes = ['calculated based on the measured kd and kon ' params.kd_PD1_PDL1.Notes];
params.koff_PD1_PDL1.Factors = ["kon_PD1_PDL1_3D","kd_PD1_PDL1"];
params.koff_PD1_PDL1.Equation = 'p(1)*p(2)';
% PD1-PDL2 koff
params.koff_PD1_PDL2.Value = [];
params.koff_PD1_PDL2.Units = '1/second';
params.koff_PD1_PDL2.Notes = ['calculated based on the measured kd and kon ' params.kd_PD1_PDL2.Notes];
params.koff_PD1_PDL2.Factors = ["kon_PD1_PDL2_3D","kd_PD1_PDL2"];
params.koff_PD1_PDL2.Equation = 'p(1)*p(2)';
% PD1-aPD1 koff
params.koff_PD1_aPD1.Value = [];
params.koff_PD1_aPD1.Units = '1/second';
params.koff_PD1_aPD1.Notes = ['calculated based on the measured kd and kon ' params.kd_PD1_aPD1.Notes];
params.koff_PD1_aPD1.Factors = ["kon_PD1_aPD1","kd_PD1_aPD1"];
params.koff_PD1_aPD1.Equation = 'p(1)*p(2)';
% PD1 Expression on T Cells
params.T8_PD1.Value = 3.1e3*20*.45;
params.T8_PD1.Units = 'molecule';
params.T8_PD1.Notes = '(Cheng 2014, PMID: 23417675; Mkrtichyan 2012, PMID: 22837483; Gatalica 2014, PMID: 25392179)';
% PDL1 Expression on T Cells
params.T8_PDL1.Value = 9.3e3*20*.45;
params.T8_PDL1.Units = 'molecule';
params.T8_PDL1.Notes = '(Cheng 2014, PMID: 23417675; Mkrtichyan 2012, PMID: 22837483; Gatalica 2014, PMID: 25392179)';
% Average Baseline PDL1 Expression on Tumor cells in PDAC tumor.
% Re-anchored from generic 8e4*20/6*.45 = 1.2e5 molecules/cell (legacy QSPIO
% melanoma anchor; PMIDs Cheng 2014 / Mkrtichyan 2012 / Shin 2017 / Gatalica
% 2014 do not contain PDAC-specific quantitative ABC measurements).
%
% Derivation:
%   per-PDL1+-cell density ~8e4 molecules/cell (legacy estimate; no PDAC
%     primary source — treat as nuisance, widen prior in SBI).
%   × cohort-mean PDL1+ fraction ~0.06 derived from PDAC TPS distributions:
%     - Zhang 2022 (PMID: 35037417, n=291 resected PDAC): 30% TPS≥1%
%     - Liang 2018 (PMID: 29378617, n=373 PDAC):
%       8.8% TPS≥5%, 5.9% TPS≥10%, 3.2% TPS≥25%
%     Bin-midpoint integration → mean PDL1+ fraction ≈ 5–6%.
%
% Sanity check (D_cell=17μm → A_cell=908μm², PD1_50=6 mol/μm²):
%   density = 4800/908 = 5.3 mol/μm² → ratio = 0.88 → H_PD1_C1 ≈ 0.44 (n=2)
%   leaves headroom for IFNg-driven upregulation (r_PDL1_IFNg=6) to push
%   H_PD1_C1 toward 0.95 in active immune response — i.e. PD-1 inhibition
%   becomes a meaningful dynamic lever instead of a saturated constant.
%
% No direct PDAC ABC measurement exists; SBI prior σ should be wide.
params.C_PDL1.Value = 8e4*0.06;
params.C_PDL1.Units = 'molecule';
params.C_PDL1.Notes = '(Zhang 2022, PMID: 35037417; Liang 2018, PMID: 29378617) — PDAC cohort-averaged baseline; per-PDL1+-cell density estimated';
% Average Baseline PDL1 Expression on mAPCs in TDLN.
% Decoupled from C_PDL1 because mature DCs upregulate PD-L1 uniformly (every
% mDC is positive — no cohort-mean discount applies).
% 2026-07-14: the ~8e4 estimate, originally a bare legacy guess, is now supported by the
% absolute-count literature and should NOT be lowered. Rieckmann 2017 (Nat Immunol,
% PMID 28263321; copy-number proteomics) measures ~214,000 total CD274 copies/cell on
% activated human mDCs (geo-mean, n=4, resting below detection); Mayoux 2020 (PMID 32161104)
% independently reports DC PD-L1 ~20x B7.1 and far above tumor cells. 8e4 SURFACE out of
% ~214k TOTAL implies a surface fraction of ~0.37 — ordinary for a type-I membrane protein.
% So the 17x DC-vs-cancer ratio (8e4 vs C_PDL1 ~4.8e3) is biologically real, and this value
% is if anything on the low side. IMPORTANT for the PR #248 H_PD1_APC->CD28-costim work:
% H_PD1_APC saturating at ~0.99 is STRUCTURAL (whole 900 um^2 A_APC diluting the count vs a
% PD1_50 of 6), NOT a wrong value here — no defensible DC anchor lands it off saturation, so
% grading that gate is an A_APC-footprint / PD1_50 / Hill-form decision, not a re-anchor of
% this parameter. See notes/model/apc_pdl1_base_anchor_investigation.md. A clean SURFACE
% anchor (Mayoux 2020 Fig 2B, per-donor SABC on iDC+mDC) is queued but deferred — it would
% replace the guess with a measurement without changing the saturation story.
params.APC_PDL1.Value = 8e4;
params.APC_PDL1.Units = 'molecule';
params.APC_PDL1.Notes = 'Per-mDC PD-L1 surface density; ~8e4 surface consistent with Rieckmann 2017 ~214k total copies/cell (~0.37 surface fraction), PMID 28263321; Mayoux 2020 PMID 32161104';
% PD1-PDL1 complex density for half-maximal T cell inhibition
% MECHANISTIC INTERPRETATION: This is the synapse density of PD1:PDL1/PDL2 complexes
% (NOT total PD-1 or PD-L1 expression) at which T cell killing is 50% inhibited.
% LITERATURE MAPPING:
%   - PD-1 expression (MFI) and PD-L1 (% positive) don't directly give this
%   - Checkpoint inhibitor efficacy depends on blocking this interaction
%   - Can estimate from: dose-response of anti-PD1 drugs × known binding kinetics
%   - Lower PD1_50 = T cells more sensitive to PD-1 inhibition (easier to exhaust)
% For SBI: This parameter couples with PD-L1 expression levels on tumor/APC
params.PD1_50.Value = 6;
params.PD1_50.Units = 'molecule/micrometer^2';
params.PD1_50.Notes = 'PD1-PDL1 complex density for 50% T cell inhibition (Jafarnejad 2019, PMID: 31236847)';
% Hill Coefficient for PD1/PDL1
params.n_PD1.Value = 2;
params.n_PD1.Units = 'dimensionless';
params.n_PD1.Notes = '(estimated)';
% PDL2/PDL1 ratio in tumor
params.r_PDL2C.Value = 0.07;
params.r_PDL2C.Units = 'dimensionless';
params.r_PDL2C.Notes = '(Cheng 2014, PMID: 23417675)';
% PDL2/PDL1 ratio on mAPCs
params.r_PDL2APC.Value = 0.07;
params.r_PDL2APC.Units = 'dimensionless';
params.r_PDL2APC.Notes = '(Cheng 2014, PMID: 23417675)';
% Number of folds increase of PDL1 expression by IFNg
params.r_PDL1_IFNg.Value = 6;
params.r_PDL1_IFNg.Units = 'dimensionless';
params.r_PDL1_IFNg.Notes = '(Shin 2017, PMID: 27903500)';
% PD1/PDL2 Concentration for Half-Maximal T Cell Killing
% params.PD1_PDL2_50.Value = 170/6;
% params.PD1_PDL2_50.Units = 'dimensionless';
% params.PD1_PDL2_50.Notes = '(Latchman 2001, PMID: 11224527)';

%% CD28/CTLA4/CD80/CD86 Parameters
% CD28-CD80 kd
params.kd_CD28_CD80.Value = 4.0;
params.kd_CD28_CD80.Units = 'micromolarity';
params.kd_CD28_CD80.Notes = '(van der Merwe 1997, PMID: 9053440)';
% CD28-CD80 kon
params.kon_CD28_CD80_3D.Value = 0.4;
params.kon_CD28_CD80_3D.Units = '1/(micromolarity*second)';
params.kon_CD28_CD80_3D.Notes = '(van der Merwe 1997, PMID: 9053440)';
% CD28-CD80 kon for 2D
params.kon_CD28_CD80.Value = [];
params.kon_CD28_CD80.Units = '';
params.kon_CD28_CD80.Notes = '(van der Merwe 1997, PMID: 9053440)';
params.kon_CD28_CD80.Factors = ["kon_CD28_CD80_3D","d_syn"];
params.kon_CD28_CD80.Equation = 'p(1)/p(2)';
% CD28-CD86 kd
params.kd_CD28_CD86.Value = 20.0;
params.kd_CD28_CD86.Units = 'micromolarity';
params.kd_CD28_CD86.Notes = '(Collins 2002, PMID: 12196291)';
% CD28-CD86 kon
params.kon_CD28_CD86_3D.Value = 1.4;
params.kon_CD28_CD86_3D.Units = '1/(micromolarity*second)';
params.kon_CD28_CD86_3D.Notes = '(Collins 2002, PMID: 12196291)';
% CD28-CD86 kon for 2D
params.kon_CD28_CD86.Value = [];
params.kon_CD28_CD86.Units = '';
params.kon_CD28_CD86.Notes = '(Collins 2002, PMID: 12196291)';
params.kon_CD28_CD86.Factors = ["kon_CD28_CD86_3D","d_syn"];
params.kon_CD28_CD86.Equation = 'p(1)/p(2)';
% CTLA4-CD80 kd
params.kd_CTLA4_CD80.Value = 0.42;
params.kd_CTLA4_CD80.Units = 'micromolarity';
params.kd_CTLA4_CD80.Notes = '(van der Merwe 1997, PMID: 9053440)';
% CTLA4-CD80 kon
params.kon_CTLA4_CD80_3D.Value = 1.0;
params.kon_CTLA4_CD80_3D.Units = '1/(micromolarity*second)';
params.kon_CTLA4_CD80_3D.Notes = '(van der Merwe 1997, PMID: 9053440)';
% CTLA4-CD80 kon for 2D
params.kon_CTLA4_CD80.Value = [];
params.kon_CTLA4_CD80.Units = '';
params.kon_CTLA4_CD80.Notes = '(van der Merwe 1997, PMID: 9053440)';
params.kon_CTLA4_CD80.Factors = ["kon_CTLA4_CD80_3D","d_syn"];
params.kon_CTLA4_CD80.Equation = 'p(1)/p(2)';
% CTLA4-CD86 kd
params.kd_CTLA4_CD86.Value = 2.6;
params.kd_CTLA4_CD86.Units = 'micromolarity';
params.kd_CTLA4_CD86.Notes = '(Jansson 2005, PMID: 16034096; Collins 2002, PMID: 12196291)';
% CTLA4-CD86 kon
params.kon_CTLA4_CD86_3D.Value = 2.0;
params.kon_CTLA4_CD86_3D.Units = '1/(micromolarity*second)';
params.kon_CTLA4_CD86_3D.Notes = '(Jansson 2005, PMID: 16034096; Collins 2002, PMID: 12196291)';
% CTLA4-CD86 kon for 2D
params.kon_CTLA4_CD86.Value = [];
params.kon_CTLA4_CD86.Units = '';
params.kon_CTLA4_CD86.Notes = '(Jansson 2005, PMID: 16034096; Collins 2002, PMID: 12196291)';
params.kon_CTLA4_CD86.Factors = ["kon_CTLA4_CD86_3D","d_syn"];
params.kon_CTLA4_CD86.Equation = 'p(1)/p(2)';
% CD80-PDL1 kd
params.kd_CD80_PDL1.Value = 18.8;
params.kd_CD80_PDL1.Units = 'micromolarity';
params.kd_CD80_PDL1.Notes = '(Cheng 2014, PMID: 23417675; Butte 2008, PMID: 18585785)';
% CD80-PDL1 kon
params.kon_CD80_PDL1_3D.Value = 0.32;
params.kon_CD80_PDL1_3D.Units = '1/(micromolarity*second)';
params.kon_CD80_PDL1_3D.Notes = '(Cheng 2014, PMID: 23417675)';
% CD80-PDL1 kon for 2D
params.kon_CD80_PDL1.Value = [];
params.kon_CD80_PDL1.Units = '';
params.kon_CD80_PDL1.Notes = '(Cheng 2014, PMID: 23417675)';
params.kon_CD80_PDL1.Factors = ["kon_CD80_PDL1_3D","d_syn"];
params.kon_CD80_PDL1.Equation = 'p(1)/p(2)';
% CD80-CD80 kd
params.kd_CD80_CD80.Value = 17;
params.kd_CD80_CD80.Units = 'micromolarity';
params.kd_CD80_CD80.Notes = '(Ikemizu 2000, PMID: 10661405)';
% CD80-CD80 kon
params.kon_CD80_CD80_3D.Value = 5.9e-4;
params.kon_CD80_CD80_3D.Units = '1/(micromolarity*second)';
params.kon_CD80_CD80_3D.Notes = '(Ikemizu 2000, PMID: 10661405)';
% CD80-CD80 kon for 2D
params.kon_CD80_CD80.Value = [];
params.kon_CD80_CD80.Units = '';
params.kon_CD80_CD80.Notes = '(Ikemizu 2000, PMID: 10661405)';
params.kon_CD80_CD80.Factors = ["kon_CD80_CD80_3D","d_syn"];
params.kon_CD80_CD80.Equation = 'p(1)/p(2)';
% CD28-CD80 koff
params.koff_CD28_CD80.Value = [];
params.koff_CD28_CD80.Units = '1/second';
params.koff_CD28_CD80.Notes = ['calculated based on the measured kd and kon ' params.kd_CD28_CD80.Notes];
params.koff_CD28_CD80.Factors = ["kon_CD28_CD80_3D","kd_CD28_CD80"];
params.koff_CD28_CD80.Equation = 'p(1)*p(2)';
% CD28-CD86 koff
params.koff_CD28_CD86.Value = [];
params.koff_CD28_CD86.Units = '1/second';
params.koff_CD28_CD86.Notes = ['calculated based on the measured kd and kon ' params.kd_CD28_CD86.Notes];
params.koff_CD28_CD86.Factors = ["kon_CD28_CD86_3D","kd_CD28_CD86"];
params.koff_CD28_CD86.Equation = 'p(1)*p(2)';
% CTLA4-CD80 koff
params.koff_CTLA4_CD80.Value = [];
params.koff_CTLA4_CD80.Units = '1/second';
params.koff_CTLA4_CD80.Notes = ['calculated based on the measured kd and kon ' params.kd_CTLA4_CD80.Notes];
params.koff_CTLA4_CD80.Factors = ["kon_CTLA4_CD80_3D","kd_CTLA4_CD80"];
params.koff_CTLA4_CD80.Equation = 'p(1)*p(2)';
% CTLA4-CD86 koff
params.koff_CTLA4_CD86.Value = [];
params.koff_CTLA4_CD86.Units = '1/second';
params.koff_CTLA4_CD86.Notes = ['calculated based on the measured kd and kon ' params.kd_CTLA4_CD86.Notes];
params.koff_CTLA4_CD86.Factors = ["kon_CTLA4_CD86_3D","kd_CTLA4_CD86"];
params.koff_CTLA4_CD86.Equation = 'p(1)*p(2)';
% CD80-PDL1 koff
params.koff_CD80_PDL1.Value = [];
params.koff_CD80_PDL1.Units = '1/second';
params.koff_CD80_PDL1.Notes = ['calculated based on the measured kd and kon ' params.kd_CD80_PDL1.Notes];
params.koff_CD80_PDL1.Factors = ["kon_CD80_PDL1_3D","kd_CD80_PDL1"];
params.koff_CD80_PDL1.Equation = 'p(1)*p(2)';
% CD80-CD80 koff
params.koff_CD80_CD80.Value = [];
params.koff_CD80_CD80.Units = '1/second';
params.koff_CD80_CD80.Notes = ['calculated based on the measured kd and kon ' params.kd_CD80_CD80.Notes];
params.koff_CD80_CD80.Factors = ["kon_CD80_CD80_3D","kd_CD80_CD80"];
params.koff_CD80_CD80.Equation = 'p(1)*p(2)';
% CD28 Expression on T Cells
params.T8_CD28.Value = 9200*20;
params.T8_CD28.Units = 'molecule';
params.T8_CD28.Notes = '(Jansson 2005, PMID: 16034096)';
% CTLA4 Expression in T Cell synapse
params.T8_CTLA4.Value = 400*20;
params.T8_CTLA4.Units = 'molecule';
params.T8_CTLA4.Notes = '(Jansson 2005, PMID: 16034096)';
% CD80 Expression on Cancer Cells
params.C_CD80.Value = 4000*20;
params.C_CD80.Units = 'molecule';
params.C_CD80.Notes = '(Jansson 2005, PMID: 16034096)';
% CD86 Expression on Cancer Cells
params.C_CD86.Value = 43000*20;
params.C_CD86.Units = 'molecule';
params.C_CD86.Notes = '(Jansson 2005, PMID: 16034096)';
% CD80 Expression on mAPCs
params.APC_CD80.Value = 4000*20;
params.APC_CD80.Units = 'molecule';
params.APC_CD80.Notes = '(Jansson 2005, PMID: 16034096)';
% CD86 Expression on mAPCs
params.APC_CD86.Value = 43000*20;
params.APC_CD86.Units = 'molecule';
params.APC_CD86.Notes = '(Jansson 2005, PMID: 16034096)';
% CD28/CD80/CD86 Concentration for Half-Maximal T Cell Killing
params.CD28_CD8X_50.Value = 200;
params.CD28_CD8X_50.Units = 'molecule/micrometer^2';
params.CD28_CD8X_50.Notes = 'estimated assuming T cell activation is inhibited by 70% at baseline CTLA-4 expression (Parry 2005, PMID: 16227604)';
% Hill Coefficient for CD28/CD80/CD86
params.n_CD28_CD8X.Value = 2;
params.n_CD28_CD8X.Units = 'dimensionless';
params.n_CD28_CD8X.Notes = '(estimated)';

%% Antibody Transport Properties (parameters are fitted and stored in pk_parameters.m)
% Permeability
% Central <-> Peripheral
params.Dk_P.Value = 3.0e-8;
params.Dk_P.Units = 'centimeter/second';
params.Dk_P.Notes = '(Finley 2012)';
% Central <-> Tumour
params.Dk_T.Value = 3.0e-7;
params.Dk_T.Units = 'centimeter/second';
params.Dk_T.Notes = '(Finley 2012)';
% Central <-> LN
params.Dk_LN.Value = 3.0e-8;
params.Dk_LN.Units = 'centimeter/second';
params.Dk_LN.Notes = '(Padera 2017)';
% Surface Area Density
% Peripheral
params.SA_P.Value = 105000.0;
params.SA_P.Units = 'centimeter^2/liter';
params.SA_P.Notes = '(Finley 2012)';
% Tumour
params.SA_T.Value = 105000.0;
params.SA_T.Units = 'centimeter^2/liter';
params.SA_T.Notes = '(Finley 2012)';
% LN
params.SA_LN.Value = 105000.0;
params.SA_LN.Units = 'centimeter^2/liter';
params.SA_LN.Notes = '(Finley 2012)';
% Lymphatic Drainage
params.Q_LD.Value = 0.0015;
params.Q_LD.Units = '1/minute';
params.Q_LD.Notes = '(Zhu 1996, PMID: 8706023)';

%% MDSC Module Parameters
% MDSC recruitment rate
params.k_MDSC_rec.Value = 3.5e4;
params.k_MDSC_rec.Units = 'cell/(milliliter*day)';
params.k_MDSC_rec.Notes = '(Tomiyama, T.2022; https://doi.org/10.3892/ol.2022.13213) - Increased for PDAC high myeloid infiltration phenotype';
% rate of MDSC death
params.k_MDSC_death.Value = 0.07;
params.k_MDSC_death.Units = '1/day';
params.k_MDSC_death.Notes = '(Lai 2018, PMID: 29735668)';
% rate of CCL2 degradtion
params.k_CCL2_deg.Value = 0.06;
params.k_CCL2_deg.Units = '1/hour';
params.k_CCL2_deg.Notes = '(Tanimoto 2007, PMID: 18089573)';
% rate of NO degradtion
params.k_NO_deg.Value = 135;
params.k_NO_deg.Units = '1/day';
params.k_NO_deg.Notes = '(Hakim 1996, PMID: 8953625)';
% rate of ArgI degradtion
params.k_ArgI_deg.Value = 0.173;
params.k_ArgI_deg.Units = '1/day';
params.k_ArgI_deg.Notes = '(Schimke 1964, PMID: 14257612)';
% rate of CCL2 secretion
params.k_CCL2_sec.Value = 2e-10; % normal: 1.065e-11; TNBC: (14.2 +/- 6)*10^-11
params.k_CCL2_sec.Units = 'nanomole/cell/day';
params.k_CCL2_sec.Notes = '(Huang 2007, PMID: 17257744; Dutta 2018, PMID: 29594759)';
% rate of NO secretion
params.k_NO_sec.Value = 4.8e-7;
params.k_NO_sec.Units = 'nanomole/cell/day';
params.k_NO_sec.Notes = '(Serafini 2008, PMID: 18593947)';
% rate of ArgI secretion (66 ng/5e5 cells /3e5 kDalton/ 4 days)
params.k_ArgI_sec.Value = 1.1e-10;
params.k_ArgI_sec.Units = 'nanomole/cell/day';
params.k_ArgI_sec.Notes = '(Rodriguez PC, PMID: 19201693)';
% rate of ArgI/NO-induced T cell death Park 2018, PMID: 29491381
% half-maximal ArgI concentration for CTL inhibition
params.ArgI_50_Teff.Value = 50;
params.ArgI_50_Teff.Units = 'nanomolarity';
params.ArgI_50_Teff.Notes = 'Half-maximal Arg-I concentration for T cell suppression (free recombinant Arg-I in conditioned media or arginase-depleted media). Recentered 2026-05-05 from 0.42 nM to 50 nM. The original 0.42 nM cited Miret 2019 (PMID 30728077) but Miret reports T-cell-restoration via arginase INHIBITION in tumor lysates, not a clean dose-response of free Arg-I against T cell function — the 0.42 nM value did not have a quantitatively traceable derivation. Submodel target extraction (2026-05-05, Bingisser1998-style) found no published paper reporting graded T cell function vs known molar recombinant Arg-I concentration with measured specific activity (most studies titrate against L-arginine concentration or use undefined conditioned media). Recentered to 50 nM as a conservative estimate from classical Arg-I T-cell-suppression literature reporting suppressive concentrations of recombinant Arg-I in the 1–10 µg/mL range (Arg-I MW ~40 kDa → ~25–250 nM); 50 nM is at the lower end. CSV prior σ widened to 1.0 (covers ~7–370 nM at ±2σ) so the joint NPE has room to settle if cytokine concentration cal targets later constrain V_T.ArgI tightly. Original Miret 2019 reference: PMID 30728077 (Suppression of myeloid cell arginase activity… NSCLC mouse model, J Immunother Cancer).';
% half-maximal NO concentration for CTL inhibition
params.NO_50_Teff.Value = 1500;
params.NO_50_Teff.Units = 'nanomolarity';
params.NO_50_Teff.Notes = 'Half-maximal free NO for T cell suppression. Recentered 2026-05-05 from 1.5 nM to 1500 nM. The prior 1.5 nM had no primary citation (hand-tuned from earlier 0.75 nM "to move H_NO into graded range at baseline NO ~2.8 nM" — the assumed baseline was itself a guess, and at posterior diagnosis state V_T.NO ≈ 10 nM, so the tuning rationale was outdated). Classical MDSC literature (Mazzoni 2002 J Immunol PMID 11777962, Bingisser 1998 J Immunol PMID 9637481, Rodriguez 2004 Cancer Res PMID 15313928) reports NO/MDSC-mediated suppression of T cell function with functional EC50 estimates in the 1–50 µM range when free NO is measured directly or estimated from SNAP-donor titrations with conservative conversion factors. 1500 nM = lower end of µM range, conservative choice. CSV prior σ widened to 1.5 (covers ~75 nM to 30 µM at ±2σ). Bingisser1998 SubmodelTarget moved to excluded/ — the SNAP-donor-only data could not separate free-NO EC50 from the unknown SNAP→NO conversion factor.';
% Half-maximal CCL2 concentration for macrophage/MDSC recruitment
params.CCL2_50.Value = 0.23;
params.CCL2_50.Units = 'nanomolarity';
params.CCL2_50.Notes = '(Opalek 2007, PMID: 17888174; Ernst 1994, PMID: 8144933)';
% half-maximal ArgI concentration for Treg expension
params.ArgI_50_Treg.Value = 0.36;
params.ArgI_50_Treg.Units = 'nanomolarity';
params.ArgI_50_Treg.Notes = '(estimated based on Juan J. Miret 2019, PMID: 30728077)';
% Max fractional arginase amplification of TGFb-driven Th->Treg conversion.
% Additive-amplifier coefficient: rate = k_Th_to_Treg*Th*H_TGFb_Th2Treg*(1+Emax_ArgI_Treg*H_ArgI_Treg).
% The TGFb floor (Emax=0) is anchored to Davidson 2007 (arginase-free iTreg
% induction); arginase amplifies above it. No quantitative anchor yet for the
% magnitude -> wide weakly-informative prior, SBI-varied. Submodel target pending.
params.Emax_ArgI_Treg.Value = 0.75;
params.Emax_ArgI_Treg.Units = 'dimensionless';
params.Emax_ArgI_Treg.Notes = 'Max fractional arginase amplification of TGFb-driven Th->Treg at saturating tumoral arginase; no quantitative anchor yet (placeholder, SBI-varied; submodel target pending)';
params.Emax_ArgI_Treg.derivedFrom = {};

%% T Helper Cell Module parameters
% Th differentiation rate to Treg
params.k_Th_to_Treg.Value = 0.022;
params.k_Th_to_Treg.Units = '1/day';
params.k_Th_to_Treg.Notes = '(Robertson-Tessi 2012, PMID: 22051568)';
% Secretion rate of TGFb by Treg
params.k_TGFb_Tsec.Value = 1.2e-10;
params.k_TGFb_Tsec.Units = 'nanomole/cell/day';
params.k_TGFb_Tsec.Notes = '(Liyanage 2002, PMID: 12193750)';
% Degradation rate of TGFb
params.k_TGFb_deg.Value = 14.3;
params.k_TGFb_deg.Units = '1/day';
params.k_TGFb_deg.Notes = '(Robertson-Tessi 2012, PMID: 22051568)';
% Context-specific TGFb EC50s (mid-tier #13). The generic TGFb_50 (Robertson-
% Tessi 2012) was shared across three distinct TGFb programs — Th->Treg
% differentiation, M1->M2 polarization, chemoresistance — while the rest of the
% TGFb-50 family is already per-context (TGFb_50_Teff/APC/CAF_act/col_sec/
% IL1R1/Th_exh). Split into three independently-calibratable EC50s, each
% defaulting to the Robertson-Tessi 0.07 nM value (zero behaviour change at
% default).
params.TGFb_50_Th2Treg.Value = 0.07;
params.TGFb_50_Th2Treg.Units = 'nanomolarity';
params.TGFb_50_Th2Treg.Notes = 'TGFb EC50 for Th->Treg differentiation (Robertson-Tessi 2012, PMID: 22051568)';
params.TGFb_50_Th2Treg.derivedFrom = {};
params.TGFb_50_M2pol.Value = 0.07;
params.TGFb_50_M2pol.Units = 'nanomolarity';
params.TGFb_50_M2pol.Notes = 'TGFb EC50 for M1->M2 macrophage polarization (initially set equal to TGFb_50, Robertson-Tessi 2012, PMID: 22051568; should be independently calibrated)';
params.TGFb_50_M2pol.derivedFrom = {};
params.TGFb_50_chemores.Value = 0.07;
params.TGFb_50_chemores.Units = 'nanomolarity';
params.TGFb_50_chemores.Notes = 'TGFb EC50 for chemoresistance development (initially set equal to TGFb_50, Robertson-Tessi 2012, PMID: 22051568). FIXED CONSTANT, not SBI-varied: the nab-paclitaxel module (its only consumer, via H_TGFb_chemores in the V_T.C1->V_T.C2 reaction) is currently disabled in immune_oncology_model_PDAC.m, so this would be an inert varied param. Add to pdac_priors.csv and queue a submodel target when nab-paclitaxel is re-enabled.';
params.TGFb_50_chemores.derivedFrom = {};
% Half-Maximal TGFb level for CD8+ T cell inhibition
params.TGFb_50_Teff.Value = 0.14;
params.TGFb_50_Teff.Units = 'nanomolarity';
params.TGFb_50_Teff.Notes = '(Robertson-Tessi 2012, PMID: 22051568)';
% Maximal fractional suppression of CD8 killing by TGFb (partial-antagonist Emax).
% The killing gate is (1 - Emax_TGFb_Teff*H_TGFb_Teff): without this floor the Hill
% (1-H_TGFb_Teff) drives killing to 0 (100% suppression) at saturating TGFb, which
% contradicts the calibrating data. Ahmadzadeh 2005 (Fig 4C, PMID: 15843518) shows
% TGFb suppression of human CD8 effector function plateaus at ~30-50% (max 41% at
% 10 ng/mL), i.e. ~60% of function survives saturating TGFb. The same submodel target
% (TGFb_50_Teff_PDAC_deriv001) estimates this plateau via its baseline_unsupp_Teff
% nuisance (median ~0.6 remaining); Emax_TGFb_Teff = 1 - baseline_unsupp ~ 0.4.
% Fixed for now; follow-up promotes baseline_unsupp to jointly calibrate this.
params.Emax_TGFb_Teff.Value = 0.4;
params.Emax_TGFb_Teff.Units = 'dimensionless';
params.Emax_TGFb_Teff.Notes = 'Max fractional TGFb suppression of CD8 killing; ~40% plateau (Ahmadzadeh 2005, Fig 4C, PMID: 15843518)';
params.Emax_TGFb_Teff.derivedFrom = {};
% Half-Maximal TGFb level for CD4 helper T cell exhaustion (PD-1+, dysfunction)
% — separate EC50 from CD8 (TGFb_50_Teff) and from Treg differentiation (TGFb_50)
% to allow submodel target on Th-side exhaustion to constrain Th biology
% specifically. Default mirrors TGFb_50_Teff; literature anchor pending via
% submodel target extraction (TGFβ + chronic TCR → CD4 dysfunction kinetics).
params.TGFb_50_Th_exh.Value = 0.14;
params.TGFb_50_Th_exh.Units = 'nanomolarity';
params.TGFb_50_Th_exh.Notes = 'Estimated; mirrors TGFb_50_Teff pending submodel target anchoring';
params.TGFb_50_Th_exh.derivedFrom = {'estimated'};
% Half-saturation constant for the SQUARED-cancer recruitment/funnel gates:
%   recruitment   C_total^2/(C_total^2 + K_C2_recruit)   -- tumor-present ON-switch
%   clearance     K_C2_recruit/(C_total^2 + K_C2_recruit)  -- post-eradication drain
% PROVENANCE (verified 2026-06-04): the value 2.02e7 is the CD8-recruitment Hill
% parameter k from de Pillis, Gu & Radunskaya 2006 (J Theor Biol 238(4):841-62,
% PMID 16153659; shared with de Pillis-Radunskaya-Wiseman 2005). [The old note read
% "Phillis 2006" -- a misspelling of de Pillis.] NOTE the repurposing: in de Pillis
% the squared quantity is a DIMENSIONLESS fractional-kill ratio D (term
% j*D^2/(k+D^2)*L), so their k is dimensionless. Here it is applied to the raw
% cancer-cell count C_total^2, hence the cell^2 unit relabel.
% CONSEQUENCE: effective half-max tumor SIZE = sqrt(K_C2_recruit) = sqrt(2.02e7)
% ~ 4.5e3 cells (NOT 2.02e7 cells) -- an INCIDENTAL threshold, not a fitted
% tumor-size half-max; the value's magnitude is misleading on its own.
% STRUCTURAL ROLE (not a free biological constant): the value must sit far below any
% established-tumor burden so the clearance funnel ~0 in real tumors and only opens
% near C->0 -- it is a near-zero "functionally eradicated" switch threshold. At the
% diagnosis attractor (C~4e7) the recruitment gate ~1 and the funnel ~0 (both
% intended); the switch-point only matters below ~4.5k cells (near-eradication, never
% reached by any PDAC scenario). The n=2 exponent is a fixed phenomenological shape
% (de Pillis's D^2 form), non-identifiable at PDAC operating scale.
params.K_C2_recruit.Value = 2.02e7;
params.K_C2_recruit.Units = 'cell^2';
params.K_C2_recruit.Notes = 'Squared-cancer recruitment/funnel half-saturation C^2/(C^2+K_C2_recruit); value = de Pillis CD8-recruitment k (de Pillis, Gu & Radunskaya 2006, J Theor Biol 238:841-62, PMID: 16153659), repurposed from a dimensionless fractional-kill-ratio term D^2/(k+D^2) to raw C_total^2 (units relabeled cell^2). Effective half-max sqrt(K) ~ 4.5e3 cells is incidental; functions as a near-zero post-eradication switch threshold, not a fitted tumor-size constant.';
% Baseline TGFb level (interpreted as total — paper sources report total
% TGFb from tissue IHC/ELISA. In the model, V_T.TGFb represents the
% bioactive-equivalent fraction, so the baseline is scaled by f_active_TGFb
% in the cancer-cell baseline production reaction.)
params.TGFb_baseline.Value = 0.016;
params.TGFb_baseline.Units = 'nanomolarity';
params.TGFb_baseline.Notes = '(Ivanović 2003, PMID: 12751375; Manuyakorn 2008, PMID: 19317336; Panis 2013, PMID: 23393376)';
% Bioactive fraction of secreted TGFb.
% Literature sources that anchor k_TGFb_sec_myCAF (Bachem 1992), k_TGFb_Tsec
% (Yang 2013/Liyanage 2002), k_TGFb_Msec (Matsuo 2000) and TGFb_baseline
% (Ivanović 2003 etc.) all measure total TGFb (acid-activated or ELISA on
% tissue lysate). The EC50 family (TGFb_50, TGFb_50_Teff, TGFb_50_APC,
% TGFb_50_CAF_act, TGFb_50_col_sec, TGFb_50_IL1R1) is calibrated
% against bioactive recombinant TGFb (Robertson-Tessi 2012). f_active_TGFb
% scales total secretion into the bioactive pool represented by V_T.TGFb.
% In vivo latent-to-active fractions vary: Annes 2004 reports basal ~0.1-1%
% in healthy tissue; integrin-dense desmoplastic tumors 1-20% (Wipff 2007).
% Default 0.1 sits at the mid-to-upper end; wide lognormal prior (sigma=1.5)
% lets joint NPE explore 3 decades.
params.f_active_TGFb.Value = 0.1;
params.f_active_TGFb.Units = 'dimensionless';
params.f_active_TGFb.Notes = 'Bioactive fraction of secreted TGFb (integrin/MMP-activated). Reconciles total-TGFb literature secretion rates with bioactive EC50 family. Annes 2004 PMID 15286164 (basal ~0.1-1%), Wipff 2007 PMID 17353357 (desmoplastic up to 25%).';
params.f_active_TGFb.derivedFrom = {'Annes 2004 PMID: 15286164', 'Wipff 2007 PMID: 17353357'};
% IFNg secretion rate by cytotoxic CD8 T cell
params.k_IFNg_Tsec.Value = 2e-13;
params.k_IFNg_Tsec.Units = 'nanomole/cell/day';
params.k_IFNg_Tsec.Notes = '(Autenshlyus 2021, PMID: 34059727; Borj 2017, doi:10.15171/ijbms.2017.05)';
% Ratio of IFNg secretin rate b/t CD8 and CD4 T cells
params.r_IFNg.Value = 3;
params.r_IFNg.Units = 'dimensionless';
params.r_IFNg.Notes = '(Kasahara 1983, PMID: 6403613)';
% IFNg secretion rate by T helper cell
params.k_IFNg_Thsec.Value = [];
params.k_IFNg_Thsec.Units = 'nanomole/cell/day';
params.k_IFNg_Thsec.Notes = ['estimated assuming IFNg secretion is 3 times lower in CD4 than CD8 T cells ' params.r_IFNg.Notes];
params.k_IFNg_Thsec.Factors = ["k_IFNg_Tsec", "r_IFNg"];
params.k_IFNg_Thsec.Equation = 'p(1)/p(2)';
% IFNg degradation rate
params.k_IFNg_deg.Value = 7.68;
params.k_IFNg_deg.Units = '1/day';
params.k_IFNg_deg.Notes = '(Hofstra 1998, PMID: 9806748)';
% Half-Maximal IFNg level for PD-L1 induction
params.IFNg_50_ind.Value = 2.96;
params.IFNg_50_ind.Units = 'picomolarity';
params.IFNg_50_ind.Notes = '(Shin 2017, PMID: 27903500)';

%% vascular module
% Initial Tumor Capacity
params.K_initial.Value = 1.03e4;
params.K_initial.Units = 'cell';
params.K_initial.Notes = '(Xiang 2014, https://doi.org/10.1158/1078-0432.CCR-13-2620)';
% Tumour vasculature growth rate
% Original HCC value: 7.37 (Xiang 2014). PDAC is hypo-vascular (MVD ~30-60/mm²
% vs HCC ~150-300/mm²) with dense stroma compressing vessels (Olive 2009, Provenzano 2012).
params.k_vas_growth.Value = 0.15;
params.k_vas_growth.Units = '1/day';
params.k_vas_growth.Notes = 'Reduced ~50x from HCC value (Xiang 2014) for PDAC hypo-vascularity. PDAC MVD 5-10x lower than HCC.';
% Tumour vasculature inhibition rate
% Original HCC value: 4e-5 (Xiang 2014). This is the INTRINSIC (size-independent)
% rate: production applies the Hahnfeldt surface term (C_total*vas_inhib_vol_scale)^(2/3)
% on top, which is ~50x at the ~9 mm Cao2014 xenograft and ~580x at a 3 cm tumour.
% The former 0.004 value double-counted that geometric amplification (it was the
% macroscopic EFFECTIVE rate, which production then re-multiplied by ^(2/3) → an
% implausible ~2/day vascular collapse at diagnosis size). The corrected intrinsic
% rate is re-derived from the Cao2014 control Fp time course with the surface term
% carried in the forward model (k_vas_decay_PDAC_deriv001.yaml): ~9e-5/day, i.e. only
% ~2x above the HCC baseline, consistent with modestly increased PDAC vessel regression.
params.k_vas_decay.Value =  0.0001;
params.k_vas_decay.Units = '1/day';
params.k_vas_decay.Notes = 'Intrinsic vascular-decay rate; production applies Hahnfeldt (C)^(2/3) surface term on top. Re-derived from Cao2014 with surface term carried in the submodel forward model (was 0.004, which double-counted the geometric factor). ~2x HCC baseline (Xiang 2014).';
% Per-cell tumor-volume scaling for the Hahnfeldt 1999 endogenous-inhibition
% surface term in the vascular-decay reaction: k_vas_decay*V_T.K*(C_total/cell *
% vas_inhib_vol_scale)^(2/3) (cancer_module.m). C_total/cell * vas_inhib_vol_scale
% is the tumor VOLUME (the count of cancer cells x the per-cell volume), and the
% ^(2/3) gives the surface-area scaling of inhibitor production (Hahnfeldt, Panigrahy,
% Folkman & Hlatky 1999, Cancer Res 59:4770, PMID 10519381). Value = a single cancer
% cell's volume in mm^3 = 4/3*pi*(D_cell/2)^3 * 1e-9 ~ 2.57e-6 for D_cell=17 um
% (Abramczyk 2015). Mid-tier #20: replaces the bare hardcoded 2.57e-6 literal with a
% named constant. Treated as dimensionless here (the mm^3 unit is conceptual and
% absorbed into k_vas_decay's calibration), so kept as a fixed dimensionless constant
% rather than a D_cell-derived dimensional param. KEEP CONSISTENT WITH D_cell: if the
% cancer-cell diameter changes, update this to 4/3*pi*(D_cell/2)^3 * 1e-9.
params.vas_inhib_vol_scale.Value = 2.57e-6;
params.vas_inhib_vol_scale.Units = 'dimensionless';
params.vas_inhib_vol_scale.Notes = 'Per-cell tumor-volume scaling (cancer cell volume in mm^3 ~ 4/3*pi*(D_cell/2)^3*1e-9) for the Hahnfeldt 1999 V^(2/3) endogenous vascular-inhibition surface term (PMID 10519381).';
params.vas_inhib_vol_scale.derivedFrom = {};
% Secretion rate of angiogenic factors by cancer cells
params.k_vas_Csec.Value = 0.1;
params.k_vas_Csec.Units = 'picogram/cell/day';
params.k_vas_Csec.Notes = '(Volk 2008, PMID: 18516298)';
% Degradation rate of angiogenic factors
params.k_vas_deg.Value = 16.6;
params.k_vas_deg.Units = '1/day';
params.k_vas_deg.Notes = '(Finley 2011, PMID: 22104283)';
% Half-maximal conc. of angiogenic factor on tumor capacity growth
params.VEGF_50.Value = 2.41e3; 
params.VEGF_50.Units = 'picogram/milliliter';
params.VEGF_50.Notes = '(Xiang 2014, https://doi.org/10.1158/1078-0432.CCR-13-2620)';
% Ipi ADCC (antibody-dependent cellular cytotoxicity) rate of Treg
params.k_CTLA4_ADCC.Value = 0.1;
params.k_CTLA4_ADCC.Units = '1/day';
params.k_CTLA4_ADCC.Notes = 'Anti-CTLA4 ADCC (antibody-dependent cellular cytotoxicity) rate of Treg (Richards 2008, PMID: 18723496)';
% IgG1-FcRIIIa V158 kon
% params.kon_IgG1_FcR.Value =  8.2e3;
% params.kon_IgG1_FcR.Units = '1/(molarity*second)';
% params.kon_IgG1_FcR.Notes = '(Li 2007, PMID: 17202140)';
% IgG1-FcRIIIa V158 Kd
% params.kd_IgG1_FcR.Value = 0.28; % 0.28 - 0.5 uM
% params.kd_IgG1_FcR.Units = 'micromolarity';
% params.kd_IgG1_FcR.Notes = '(Bruhns 2009, PMID: 19018092; Richards 2008, PMID: 18723496)';
% RcRIIIa Expression on Tregs
% params.FcRIIIa_tot.Value = 2.4e5; % 2.51e6 - 3.55e6 - 3.98e6
% params.FcRIIIa_tot.Units = 'molecule';
% params.FcRIIIa_tot.Notes = '(Richards 2008, PMID: 18723496; Robinett 2018, PMID: 29960887)';


%% Macrophage Module
% Macrophage recruitment rate
params.k_Mac_rec.Value = 4.5e5;
params.k_Mac_rec.Units = 'cell/(milliliter*day)';
params.k_Mac_rec.Notes = '(Ren 2017, PMID: 28656201)';
% Death rate
params.k_Mac_death.Value = 0.05;
params.k_Mac_death.Units = '1/day';
params.k_Mac_death.Notes = '(Ginhoux 2016, PMID: 26982352)';
% Macrophage niche carrying capacity (density ceiling on TAM recruitment).
% Finite stromal/spatial niche: TAMs saturate rather than accumulate without
% bound. Sized from the v26 d21 trace (total TAM ~3.2x over target -> ceiling
% ~0.31x current total TAM density); SBI-calibrated. See
% notes/calibration/d21_composition_tension_diagnosis.md (macrophage trace+toy).
params.rho_Mac_max.Value = 1e7;
params.rho_Mac_max.Units = 'cell/milliliter';
params.rho_Mac_max.Notes = 'Macrophage niche density ceiling (finite stromal niche). Set just-non-binding vs the Chen2015-anchored recruitment/death equilibrium Total_Mac = k_Mac_rec*H_CCL2/k_Mac_death (~9e6 cell/mL at H_CCL2->1), so TAM density is governed by the anchored kinetics, not this (unanchored) ceiling. The prior 1.5e6 capped TAM density at ~36/mm^2, below both the anchored balance and the observed PDAC density (Sivagnanam2021 ~100-120/mm^2 = 5e6 cell/mL); confirmed via rho_Mac_max sweep at the diagnosis posterior cloud (tam_density 42->100->143/mm^2 for x1/x3.3/x6.7).';
params.rho_Mac_max.derivedFrom = {};
% Secretion rate of TGFb by M2 macrophage
params.k_TGFb_Msec.Value = 2e-11;
params.k_TGFb_Msec.Units = 'nanomole/cell/day';
params.k_TGFb_Msec.Notes = '(Matsuo 2000, PMID: 10652049)';
% Secretion rate of angiogenic factor by M2 macrophage
params.k_vas_Msec.Value = 0.05;
params.k_vas_Msec.Units = 'picogram/cell/day';
params.k_vas_Msec.Notes = '(Wu 2010, PMID: 20692534)';
% Secretion rate of IL12 by mAPC
params.k_IL12_sec.Value = 8.5e-12;
params.k_IL12_sec.Units = 'nanomole/cell/day';
params.k_IL12_sec.Notes = '(Heufler 1996, PMID: 8605935)';
% Secretion rate of IL12 by M2 macrophage
params.k_IL12_Msec.Value = 5e-13;
params.k_IL12_Msec.Units = 'nanomole/cell/day';
params.k_IL12_Msec.Notes = '(Clough 2007, PMID: 17597818; Sica 2000, PMID: 10623821)';
% Degradation rate of IL12
params.k_IL12_deg.Value = 0.0231;
params.k_IL12_deg.Units = '1/hour';
params.k_IL12_deg.Notes = '(Carreño 2000, PMID: 10707873)';
% Secretion rate of IL10
params.k_IL10_sec.Value = 3e-12;
params.k_IL10_sec.Units = 'nanomole/cell/day';
params.k_IL10_sec.Notes = '(Autenshlyus 2021, PMID: 34059727)';
% IL-10 secretion by Tregs. Pan 2025 (DOI:10.3389/fimmu.2025.1582305) identifies
% Tregs as PDAC TME IL-10 producers; canonical Treg immunosuppression mechanism.
% Seeded at ~half of M2 rate; SBI-calibrated against PDAC_IL10_Herremans2023
% and downstream M2 polarization observables.
params.k_IL10_sec_Treg.Value = 1.5e-12;
params.k_IL10_sec_Treg.Units = 'nanomole/cell/day';
params.k_IL10_sec_Treg.Notes = 'IL-10 secretion rate by Tregs. Value is an initial estimate (SBI-calibrated), NOT a literature-derived rate. Pan 2025 (DOI:10.3389/fimmu.2025.1582305, Front Immunol review) supports the qualitative mechanism (Tregs secrete IL-10 in the PDAC TME) only -- a review carries no per-cell secretion rate. Audit m2-A-MAC-02.';
params.k_IL10_sec_Treg.derivedFrom = {};
% IL-10 secretion by MDSCs. Pan 2025: "MDSCs secrete IL-10 and TGF-β to drive
% Treg differentiation." Co-secreted with TGFβ as part of the MDSC suppression
% program. Seeded at ~one-third M2 rate; SBI-calibrated.
params.k_IL10_sec_MDSC.Value = 1e-12;
params.k_IL10_sec_MDSC.Units = 'nanomole/cell/day';
params.k_IL10_sec_MDSC.Notes = 'IL-10 secretion by MDSCs. Pan 2025 DOI:10.3389/fimmu.2025.1582305. Initial estimate, SBI-calibrated.';
params.k_IL10_sec_MDSC.derivedFrom = {'Pan 2025 DOI:10.3389/fimmu.2025.1582305'};
% Degradation rate of IL10
params.k_IL10_deg.Value = 2.5;
params.k_IL10_deg.Units = '1/day';
params.k_IL10_deg.Notes = 'Effective tumor-tissue IL-10 degradation/clearance (lumps proteolysis + local IL-10R-mediated uptake). SUBMODEL-ANCHORED to Yeung 2025 (JHLT Open, DOI 10.1016/j.jhlto.2024.100192, submodel_targets/k_IL10_deg_PDAC_deriv002.yaml): rIL-10 in human ex-vivo-lung-perfusion perfusate decayed 2493->909 pg/mL over 3-12h, t1/2 ~6-8h (k ~2.0-2.7/day) -- a tissue compartment stripped of renal/hepatic clearance and circulating soluble IL-10R, the correct structural match for this lumped first-order term. Replaces the renal-inflated serum PK (Huhn 1996, t1/2 2.3-3.7h, k~5.4/day), now retained only as the fast-side secondary bound. 2.5/day is the prior center; tumor TAM IL-10R density + acidic pH may push effective clearance back up, so the EVLP rate is a soft lower bound and the wide sigma lets SBI calibrate the IL-10 balance (free pool vs Herremans homogenate via the R_total_to_free_IL10 observation aux).';
params.k_IL10_deg.derivedFrom = {};
% Secretion rate of IL13
params.k_IL13_sec.Value = 5.35e-11;
params.k_IL13_sec.Units = 'nanomole/cell/day';
params.k_IL13_sec.Notes = '(Autenshlyus 2021, PMID: 34059727)';
% Degradation rate of IL13
params.k_IL13_deg.Value = 0.05; % 20mins; 72 day^-1
params.k_IL13_deg.Units = '1/minute';
params.k_IL13_deg.Notes = '(Khodoun 2007, PMID: 17982031)';
% M1 to M2 polarization rate (MAXIMUM RATE interpretation)
% This is the CEILING rate when anti-inflammatory cytokines (TGFb, IL-10, IL-6) saturate.
% Model equation: k_M2_pol * Mac_M1 * [cytokine Hill terms]
% Literature using various M2-polarizing stimuli (IL-4, IL-13, TGFb, tumor CM, EVs)
% can estimate this ceiling if stimulation is strong (>> EC50).
% Time-to-50%-polarization of 2-3 days under saturating conditions gives k ~ 0.1-0.3/day.
params.k_M2_pol.Value = 0.1;
params.k_M2_pol.Units = '1/day';
params.k_M2_pol.Notes = '(Max rate under saturating anti-inflammatory stimulus; t_half ~2-3 days)';

% M2 to M1 polarization rate (MAXIMUM RATE interpretation)
% This is the CEILING rate when pro-inflammatory cytokines (IL-12, IFNg) saturate.
% Model equation: k_M1_pol * Mac_M2 * [cytokine Hill terms]
% Literature using various M1-polarizing stimuli (LPS+IFNg, CD40 agonists, GM-CSF)
% can estimate this ceiling if stimulation is strong (>> EC50).
% Different stimuli may activate different pathways (JAK-STAT vs NF-kB) but converge
% on similar kinetics for full M1 commitment under saturating conditions.
% 2026-07-13: recentered 0.045 -> 1.0 /day. The old value was an unanchored guess whose
% half-time (~15 days) is excluded by every strand of the repolarization literature, which
% consistently finds M2->M1 conversion essentially complete within 24-72 h under saturating
% IFNg(+LPS) (Liu 2020 PMID 32703960; Davis 2013 PMID 23781069; Van den Bossche 2016
% PMID 27732846) — i.e. a ceiling of order 1/day, not 0.05/day. Critically, 0.045/day sat
% BELOW k_Mac_death (0.05/day), so TAM phenotype was fixed at entry and could not respond
% to IL-12/IFNg at all. Anchored by k_M1_pol_PDAC_deriv003 (Liu 2020 GEO GSE158094 M2->M1
% repolarization timecourse, with the IL-4-withdrawal arm as the marker-decay control).
params.k_M1_pol.Value = 1.0;
params.k_M1_pol.Units = '1/day';
params.k_M1_pol.Notes = '(Max rate under saturating pro-inflammatory stimulus; t_half ~17 h. Liu 2020, PMID 32703960)';
% IL-10 EC50s, split per effector context (2026-05-23). Previously a single
% IL10_50 ≈ 8 pM (legacy citation unverified) gated M1→M2 polarization,
% DC maturation, and Treg-mediated CD8 exhaustion simultaneously, which
% forced one shared receptor-affinity scale across three biologies. The
% Seldon 1998 TNF-α IC50 submodel target now anchors IL10_50_M2 only.
% IL10_50_APC and IL10_50_Treg start at the same prior median pending
% dedicated literature anchors.
params.IL10_50_M2.Value = 8;
params.IL10_50_M2.Units = 'picomolarity';
params.IL10_50_M2.Notes = 'Inherited from legacy IL10_50 default (8 pM); legacy citation unverified. Real anchor is the Seldon 1998 IL-10 IC50 submodel target (calibration_targets/submodel_targets/IL10_50_PDAC_deriv001.yaml, repointed to IL10_50_M2 in this branch).';
params.IL10_50_APC.Value = 8;
params.IL10_50_APC.Units = 'picomolarity';
params.IL10_50_APC.Notes = 'Initial estimate inherited from legacy IL10_50 (8 pM); SBI-calibrated. Dedicated DC-maturation dose-response anchor pending.';
params.IL10_50_Treg.Value = 8;
params.IL10_50_Treg.Units = 'picomolarity';
params.IL10_50_Treg.Notes = 'Half-maximal IL-10 (pM, monomer-equiv) for the H_IL10_Treg gate amplifying Treg-mediated CD8 exhaustion. Soft submodel anchor: Gorby 2020 Fig 5C WT-dimer IL-10->pSTAT3 EC50 in activated human CD8+ T cells (n=4 donors, ~466 pM monomer-equiv; signaling-potency proxy, direction-agnostic). Value stays at the legacy 8 pM placeholder, but CSV prior sigma widened 1.2->2.0 so the data/family band is reachable; composed submodel posterior median ~257 pM, consistent with the anchored IL10_50_APC (217 pM) and IL10_50_phago (270 pM) siblings. Cell-type-specific IL-10R potency (myeloid IL10_50_M2 ~7 pM is far lower), so not pooled with the IL10_50 family.';
params.IL10_50_M2.derivedFrom = {};
params.IL10_50_APC.derivedFrom = {};
params.IL10_50_Treg.derivedFrom = {};
% Half-maximal IL-12 level for M2 to M1 polarization
params.IL12_50.Value = 15; % pM — graded M1 modulation; previous 0.14 pM saturated at any measurable IL-12
params.IL12_50.Units = 'picomolarity'; % pM
params.IL12_50.Notes = 'Increased from 0.14 pM to 15 pM so IL-12 is a graded modulator of M1 polarization (not always-on). Original 0.14 pM from Altan-Bonnet 2019 (PMID: 30770905) likely refers to a different readout.';
% Population-averaged slope of IFNg-driven M2 to M1 polarization vs bulk tumor IFNg.
% Replaces IFNg_50 Hill (2026-04-22). Rationale: individual macrophages respond to IFNg
% with a saturating Hill at single-cell EC50 ~1 nM (cell-culture literature), but IFNg
% signaling in tumors is spatially confined to ~30-40 um around producing T cells
% (Centofanti 2023 PMID: 37603742). Bulk V_T.IFNg and M2 are both well-mixed in the
% QSP model, so the relevant population-level quantity is the fraction of macrophages
% within signaling range of a TIL — which scales approximately linearly with TIL density
% and therefore with bulk IFNg. The Hill saturating form at bulk concentration drops to
% zero (since bulk IFNg ~1 pM << 1 nM EC50); the linear form correctly represents the
% population-averaged polarization drive as a spatial-coverage proxy.
%
% IMPORTANT: k_IFNg_response is NOT a receptor EC50. It is a lumped parameter bundling
% (a) IFNg diffusion length ~35 um, (b) TIL spatial distribution, (c) local macrophage
% density. Its physical interpretation is d(near-TIL fraction) / d([IFNg]_bulk).
%
% Default 0.2 /pM is an engineering estimate: at bulk IFNg = 1 pM, ~20% of macrophages
% are assumed to be within a TIL signaling zone, consistent with order-of-magnitude
% TIL-density calculations from Centofanti's ~35 um signaling radius and typical PDAC
% TIL densities. NOT calibrated against any primary EC50 measurement — dose-response
% data at sub-ng/mL tissue concentrations does not exist. To be constrained by SBI via
% covariance with TIL-density-sensitive observables (M1/M2 ratios across tumor cohorts);
% prior kept wide (sigma=1.5).
params.k_IFNg_response.Value = 0.2;
params.k_IFNg_response.Units = '1/picomolarity';
params.k_IFNg_response.Notes = 'Population-averaged slope of M2->M1 polarization vs bulk tumor IFNg (spatial-averaging proxy, not a receptor EC50). Default 0.2/pM is an engineering estimate based on IFNg-signaling spatial confinement (Centofanti 2023 PMID: 37603742) — cell-culture single-cell EC50 literature does not apply to the bulk/well-mixed observable in the QSP model.';
params.k_IFNg_response.derivedFrom = {};
% M1-mediated cancer killing rate
params.k_M1_phago.Value = 0.33;
params.k_M1_phago.Units = '1/day';
params.k_M1_phago.Notes = '(Pan 2020, PMID: 33365025)';
% Half-maximal IL-10 level for phagocytosis
params.IL10_50_phago.Value = 270;
params.IL10_50_phago.Units = 'picomolarity';
params.IL10_50_phago.Notes = '(Bian 2016, PMID: 27578867)';
% Dependence of phagocytosis rate on M1/C ratio
params.K_M1_phago.Value = 2;
params.K_M1_phago.Units = 'dimensionless';
params.K_M1_phago.Notes = '(estimated)'; % Gardner 2020, PMID: 33209523
% Treg:Teff ratio for 50% inhibition of T cell killing (PRIMARY parameter)
% Directly measurable from suppression assays: "X% inhibition at Y:1 Treg:Teff"
% Literature: Blood 2009 showed ~39% inhibition at 1:4 ratio (0.25) -> R50 ~ 0.3-0.4
% Higher R50 = Tregs less suppressive; Lower R50 = Tregs more potent
params.R50_Treg.Value = 0.31;
params.R50_Treg.Units = 'dimensionless';
params.R50_Treg.Notes = '(estimated from suppression assays; K_T_Treg = 1/R50_Treg)';
% K_T_Treg is DERIVED from R50_Treg for use in model equations
% Half-inhibition when Tregs/Teff = R50_Treg, i.e., when K_T_Treg * Tregs = Teff
params.K_T_Treg.Value = [];
params.K_T_Treg.Units = 'dimensionless';
params.K_T_Treg.Notes = '(derived: 1/R50_Treg; original: Robertson-Tessi 2012, PMID: 22051568)';
params.K_T_Treg.Factors = ["R50_Treg"];
params.K_T_Treg.Equation = '1/p(1)';
% Macrophage Diameter
params.D_Mcell.Value = 21;
params.D_Mcell.Units = 'micrometer';
params.D_Mcell.Notes = '(Krombach 1997, PMID: 9400735)';
% Surface Area of a Macrophages
params.A_Mcell.Value = [];
params.A_Mcell.Units = 'micrometer^2';
params.A_Mcell.Notes = ['calculated based on the average macrophage diameter ' params.D_Mcell.Notes];
params.A_Mcell.Factors = ["D_Mcell"];
params.A_Mcell.Equation = '4*pi*(p(1)/2)^2';
% Volume of a Macrophage
params.vol_Mcell.Value = [];
params.vol_Mcell.Units = 'micrometer^3/cell';
params.vol_Mcell.Notes = ['Volume of a macrophage calculated based on the average macrophage diameter ' params.D_Mcell.Notes];
params.vol_Mcell.Factors = ["D_Mcell", "cell"];
params.vol_Mcell.Equation = '4/3*pi*(p(1)/2)^3/p(2)';


%% Phagocytosis Module
% Binding affinity of CD47-SIRPa
params.kd_CD47_SIRPa.Value = 1.6;
params.kd_CD47_SIRPa.Units = 'micromolarity';
params.kd_CD47_SIRPa.Notes = '(Hayes 2020, PMID: 31964705)';
% Dissociation rate of CD47-SIRPa
params.koff_CD47_SIRPa.Value = 0.3;
params.koff_CD47_SIRPa.Units = '1/minute';
params.koff_CD47_SIRPa.Notes = '(Hayes 2020, PMID: 31964705)';
% Association rate of CD47-SIRPa
params.kon_CD47_SIRPa.Value = [];
params.kon_CD47_SIRPa.Units = '';
params.kon_CD47_SIRPa.Notes = '(Hayes 2020, PMID: 31964705)';
params.kon_CD47_SIRPa.Factors = ["koff_CD47_SIRPa","kd_CD47_SIRPa","d_syn"];
params.kon_CD47_SIRPa.Equation = 'p(1)/p(2)/p(3)';

% Half-maximal SIRPa binding for phagocytosis inhibition
params.SIRPa_50.Value = 37;
params.SIRPa_50.Units = 'molecule/micrometer^2';
params.SIRPa_50.Notes = '(Willingham 2012, PMID: 22451913)';
% Hill coefficient for H_SIRPa
params.n_SIRPa.Value = 2;
params.n_SIRPa.Units = 'dimensionless';
params.n_SIRPa.Notes = '(estimated)';
% CD47 expression on cancer cell
params.C_CD47.Value = 400;
params.C_CD47.Units = 'molecule/micrometer^2';
params.C_CD47.Notes = '(Morrissey 2020, PMID: 32768386)';
% SIRPa expression on macrophage
params.M_SIRPa.Value = 100;
params.M_SIRPa.Units = 'molecule/micrometer^2';
params.M_SIRPa.Notes = '(Subramanian 2006, PMID: 16291597)';
% PD-1 expression on macrophage
params.M_PD1.Value = 1.25e4;
params.M_PD1.Units = 'molecule';
params.M_PD1.Notes = 'estimated based on PD-1 expression on T cell (Cheng 2014, PMID: 23417675; Mkrtichyan 2012, PMID: 22837483) and adjusted to fit its inhibitory effect on phagocytosis (Gordon 2017, PMID: 28514441)';

%% fibroblast module
% fibroblast recruitment due to tumor expansion in the tumor compartment
params.k_CAF_encounter.Value = 1e6;
params.k_CAF_encounter.Units = "cell/(milliliter*day)";
params.k_CAF_encounter.Notes = "Fitted to 145 +- 9.38 cells/0.785 mm^2 PMID: 19289585";
% fibroblast recruitment due to tumor expansion in the tumor compartment
params.k_CAF_const.Value = 2e6;
params.k_CAF_const.Units = "cell/milliliter";
params.k_CAF_const.Notes = "Fitted to 145 +- 9.38 cells/0.785 mm^2 PMID: 19289585";
%% CAF Subtype Activation Rates (qPSC -> iCAF/myCAF/apCAF)
% TGFb favors myCAF; low TGFb favors iCAF (Biffi 2019 Cancer Discovery PMID: 30366930)
params.k_PSC_to_myCAF.Value = 0.2;
params.k_PSC_to_myCAF.Units = '1/day';
params.k_PSC_to_myCAF.Notes = 'qPSC->myCAF activation rate. TGFb-driven (Biffi 2019 PMID: 30366930). Reduced from 0.6 to maintain ~2:1 ratio with iCAF base rate.';
params.k_PSC_to_myCAF.derivedFrom = {'Biffi 2019 PMID: 30366930'};

params.k_PSC_to_iCAF.Value = 0.2;
params.k_PSC_to_iCAF.Units = '1/day';
params.k_PSC_to_iCAF.Notes = 'qPSC->iCAF BASE activation rate (before IL-6 amplification). Reduced from 0.4 to account for IL-6 feedback (Biffi 2019 PMID: 30366930).';
params.k_PSC_to_iCAF.derivedFrom = {'Biffi 2019 PMID: 30366930'};

params.k_PSC_to_apCAF.Value = 0.05;
params.k_PSC_to_apCAF.Units = '1/day';
params.k_PSC_to_apCAF.Notes = 'qPSC->apCAF activation rate. Constitutive low rate; apCAFs are rarest subtype ~2-5% of CAFs (Elyada 2019 PMID: 31197017).';
params.k_PSC_to_apCAF.derivedFrom = {'Elyada 2019 PMID: 31197017'};

%% CAF Subtype Interconversion
params.k_iCAF_to_myCAF.Value = 0.1;
params.k_iCAF_to_myCAF.Units = '1/day';
params.k_iCAF_to_myCAF.Notes = 'iCAF->myCAF interconversion rate. TGFb-driven plasticity (Biffi 2019 PMID: 30366930).';
params.k_iCAF_to_myCAF.derivedFrom = {'Biffi 2019 PMID: 30366930'};

params.k_myCAF_to_iCAF.Value = 0.05;
params.k_myCAF_to_iCAF.Units = '1/day';
params.k_myCAF_to_iCAF.Notes = 'myCAF->iCAF interconversion rate. Favored when TGFb drops (Biffi 2019 PMID: 30366930).';
params.k_myCAF_to_iCAF.derivedFrom = {'Biffi 2019 PMID: 30366930'};

% death rate of normal fibroblast in tumor compartment
params.k_qpsc_death.Value = 0.005;
params.k_qpsc_death.Units = '1/day';
params.k_qpsc_death.Notes = 'qPSC death rate. Constrained via hierarchical group (partial pooling from Kisseleva 2012 k_myCAF_death target).';
params.k_qpsc_death.derivedFrom = {};

%% CAF Subtype Death Rates
params.k_iCAF_death.Value = 0.05;
params.k_iCAF_death.Units = '1/day';
params.k_iCAF_death.Notes = 'iCAF death rate. Constrained via hierarchical group (partial pooling from Kisseleva 2012 k_myCAF_death target).';
params.k_iCAF_death.derivedFrom = {};

params.k_myCAF_death.Value = 0.015;
params.k_myCAF_death.Units = '1/day';
params.k_myCAF_death.Notes = 'myCAF death rate. Kisseleva 2012 (10.1073/pnas.1201840109) HSC recovery from CCl4 fibrosis.';
params.k_myCAF_death.derivedFrom = {'Kisseleva 2012, DOI: 10.1073/pnas.1201840109'};

params.k_apCAF_death.Value = 0.03;
params.k_apCAF_death.Units = '1/day';
params.k_apCAF_death.Notes = 'apCAF death rate. Constrained via hierarchical group (partial pooling from Kisseleva 2012 k_myCAF_death target).';
params.k_apCAF_death.derivedFrom = {};

%% CAF Subtype Proliferation Rates
params.k_myCAF_prolif.Value = 0.15;
params.k_myCAF_prolif.Units = '1/day';
params.k_myCAF_prolif.Notes = 'myCAF proliferation rate. Inherits from k_apsc_prolif. Major source of CAF accumulation. NEEDS CALIBRATION.';
params.k_myCAF_prolif.derivedFrom = {};

params.k_iCAF_prolif.Value = 0.10;
params.k_iCAF_prolif.Units = '1/day';
params.k_iCAF_prolif.Notes = 'iCAF proliferation rate. Slightly lower than myCAF (less mitogenic). NEEDS CALIBRATION.';
params.k_iCAF_prolif.derivedFrom = {};

params.k_apCAF_prolif.Value = 0.05;
params.k_apCAF_prolif.Units = '1/day';
params.k_apCAF_prolif.Notes = 'apCAF proliferation rate. Lower than iCAF/myCAF — apCAFs are less mitogenic but require self-renewal to maintain 6-10% fraction (Elyada 2019 PMID: 31197017). NEEDS CALIBRATION.';
params.k_apCAF_prolif.derivedFrom = {'Elyada 2019 PMID: 31197017'};

% Maximum CAF-to-cancer ratio (total carrying-capacity budget for the CAF
% compartment). Reformulated 2026-05-01 from a *shared* cap (where all
% subtypes competed for one logistic slot, producing a winner-take-all
% artifact: more apCAF crowdout myCAF, ↓collagen, ↑tumor) to a
% *partitioned* cap (each subtype's cap = f_*_share × total budget).
% Subtype shares default to PDAC-observed proportions (Elyada 2019 Fig 5D:
% ~70% myCAF, ~20% iCAF, ~10% apCAF). f_stroma_max sets the total budget
% (preserved from the Torphy + Ogawa joint submodel inference); shares
% partition it. See f_myCAF_share / f_iCAF_share / f_apCAF_share below.
params.f_stroma_max.Value = 2.43;
params.f_stroma_max.Units = 'dimensionless';
params.f_stroma_max.Notes = 'Max ratio of activated CAFs (aPSC_total) to cancer cells (C_total). Joint submodel posterior median 2.43 (sigma 0.62) from Torphy 2018 + Ogawa 2021 (n=215 PDAC whole-slide ACTA2 + FAP area, with cancer/CAF nuclear-density nuisance priors). Diagnostic sweep at fs=2.4 still leaves stromal_fraction ~0.92; Bengtsson cellularity (cal target) is expected to pull this lower at SBI. Now partitioned among iCAF/myCAF/apCAF via f_*_share parameters (2026-05-01).';
params.f_stroma_max.derivedFrom = {'Ogawa 2021 DOI: 10.1158/1078-0432.CCR-20-2298'};

% CAF subtype simplex (stick-breaking parameterization).
% Two free parameters parameterize the (apCAF, iCAF, myCAF) shares of the
% total CAF carrying-capacity budget. Subtype caps:
%   apCAF cap  = f_apCAF_of_total * cap_total
%   iCAF cap   = (1 - f_apCAF_of_total) * f_iCAF_of_non_apCAF * cap_total
%   myCAF cap  = (1 - f_apCAF_of_total) * (1 - f_iCAF_of_non_apCAF) * cap_total
% where cap_total = f_stroma_max*C_total + K_stiff_niche*H_stiff_fib.
% Sums to cap_total by construction, so the literature-anchored total CAF
% density is preserved regardless of how SBI samples the simplex.
% Defaults from Elyada 2019 Fig 5D (myCAF 0.70, iCAF 0.20, apCAF 0.10):
%   f_apCAF_of_total = 0.10
%   f_iCAF_of_non_apCAF = 0.222 (so iCAF = 0.9 * 0.222 = 0.20,
%                                 myCAF = 0.9 * 0.778 = 0.70)
% Beta priors used in pdac_priors.csv to keep both bounded in (0,1)
% without truncation gymnastics — see comment block at f_stroma_max for
% the structural rationale (winner-take-all → simplex partition).
params.f_apCAF_of_total.Value = 0.1;
params.f_apCAF_of_total.Units = 'dimensionless';
params.f_apCAF_of_total.Notes = 'apCAF share of the total CAF carrying capacity. Default 0.1 from Elyada 2019 Fig 5D (apCAFs ~10% of PDAC CAFs). Free parameter in the stick-breaking simplex parameterization (paired with f_iCAF_of_non_apCAF).';
params.f_apCAF_of_total.derivedFrom = {'Elyada 2019 PMID: 31197017'};
params.f_iCAF_of_non_apCAF.Value = 0.222;
params.f_iCAF_of_non_apCAF.Units = 'dimensionless';
params.f_iCAF_of_non_apCAF.Notes = 'iCAF share of the non-apCAF CAF mass (stick-breaking second step). Default 0.222 from Elyada 2019 Fig 5D — myCAF 0.7 + iCAF 0.2 of total → iCAF/(iCAF+myCAF) = 0.2/0.9 = 0.222. Free parameter in the stick-breaking simplex parameterization (paired with f_apCAF_of_total).';
params.f_iCAF_of_non_apCAF.derivedFrom = {'Elyada 2019 PMID: 31197017'};

% PDGF parameters removed — PDGF species eliminated from model.
% PDGF-BB is matrix-bound (retention motif) and constitutively present
% in PDAC TME. See DEVLOG 2026-03-20 for rationale.

% TGF-beta secretion by myCAFs (major source in PDAC, creates positive feedback loop)
params.k_TGFb_sec_myCAF.Value =  2e-10;
params.k_TGFb_sec_myCAF.Units = 'nanomolarity*liter/(cell*day)';
params.k_TGFb_sec_myCAF.Notes = 'TGF-β secretion by myCAFs. myCAFs are the dominant TGF-β source in PDAC, creating positive feedback (Cancer->TGF-β->PSC activation->more TGF-β). Was k_TGFb_sec_apsc. (Calon 2015 PMID: 25957691; Pickup 2013 PMID: 24060861). NEEDS CALIBRATION.';
params.k_TGFb_sec_myCAF.derivedFrom = {'Calon 2015 PMID: 25957691', 'Pickup 2013 PMID: 24060861'};
% Molecular weight of ECM (effective, collagen type I)
params.MW_ECM.Value = 300.0;
params.MW_ECM.Units = 'kilodalton';
params.MW_ECM.Notes = "Effective MW for ECM (primarily collagen type I). Used for unit conversions.";
params.MW_ECM.derivedFrom = "Biochemistry reference (collagen type I ~300 kDa)";

% Collagen secretion rate by myCAFs (qPSC secretion removed as negligible <1%)
params.k_col_myCAF_sec.Value = 5e-8;  % TGFb-saturated MAXIMAL rate. NOT 10x over lit (audit V2 false positive): the "1.7-9.3 pg/cell/day" literature is BASAL secretion = k_col_myCAF_sec*f_col_basal; submodel target k_col_myCAF_sec_illsley2000 jointly anchors k*f_col_basal ~10 pg/cell/day basal, satisfied at k~4e-8, f_col_basal~0.25. See mechanism_audit_reconciliation.md.
params.k_col_myCAF_sec.Units = 'milligram/(cell*day)';
params.k_col_myCAF_sec.Notes = "TGFb-SATURATED MAXIMAL collagen secretion rate by myCAFs. The 1.7-9.3 pg/cell/day literature is BASAL secretion = k_col_myCAF_sec*f_col_basal, NOT this maximal rate (do not compare directly). Submodel target k_col_myCAF_sec_illsley2000 anchors the k*f_col_basal product to ~10 pg/cell/day basal.";
params.k_col_myCAF_sec.derivedFrom = {};
% Basal fraction of collagen secretion (TGFb-independent)
params.f_col_basal.Value = 0.15;  % submodel-anchored (k_col_myCAF_sec_illsley2000 + f_col_basal_tgfb_fold) posterior ~0.25; CSV/.m default is shadowed by the submodel composite, so left at the original value (audit V10 is a shadowed/data-anchored no-op).
params.f_col_basal.Units = 'dimensionless';
params.f_col_basal.Notes = "Fraction of myCAF collagen secretion that is TGFb-independent. Submodel-anchored (f_col_basal_tgfb_fold_illsley2000: Illsley 2000 TGFb1 fold-change 1.31-1.91 => f_col_basal=1/fold ~0.52-0.76; Shek 2002 complementary ~0.44). Realized value is the submodel posterior (~0.25, jointly fit with k_col_myCAF_sec); the .m/CSV default 0.15 is shadowed by that composite. The lit fold-change range (~0.44-0.76) is an upper bound (single near-saturating TGFb dose), so the lower joint posterior is expected. Do not read the bare default as the operative value.";
params.f_col_basal.derivedFrom = {};
% TGFb EC50 for CAF activation and iCAF/myCAF interconversion (SMAD-driven lineage)
params.TGFb_50_CAF_act.Value = 0.07;
params.TGFb_50_CAF_act.Units = 'nanomolarity';
params.TGFb_50_CAF_act.Notes = "TGFb EC50 for PSC activation to myCAF and iCAF<->myCAF interconversion. Initially set equal to TGFb_50; should be independently calibrated.";
params.TGFb_50_CAF_act.derivedFrom = {};
% TGFb EC50 for collagen secretion enhancement
params.TGFb_50_col_sec.Value = 0.07;
params.TGFb_50_col_sec.Units = 'nanomolarity';
params.TGFb_50_col_sec.Notes = "TGFb EC50 for collagen secretion by myCAFs. Initially set equal to TGFb_50; Illsley 2000 fold-change data may constrain independently.";
params.TGFb_50_col_sec.derivedFrom = {};
% TGFb EC50 for antagonism of IL-1R1 expression (SMAD3-mediated transcriptional repression)
params.TGFb_50_IL1R1.Value = 0.07;
params.TGFb_50_IL1R1.Units = 'nanomolarity';
params.TGFb_50_IL1R1.Notes = "TGFb EC50 for IL-1R1 downregulation. TGFb antagonizes iCAF formation by repressing IL-1R1 via SMAD3 (Biffi 2019, PMID: 30366930). Initially set equal to TGFb_50.";
params.TGFb_50_IL1R1.derivedFrom = {};
% Maximum fold-shift of IL-1 EC50 by TGFb-mediated IL-1R1 repression
params.Emax_IL1R1.Value = 5.0;
params.Emax_IL1R1.Units = 'dimensionless';
params.Emax_IL1R1.Notes = "Max fold-increase in effective IL-1 EC50 from TGFb-driven IL-1R1 downregulation. Caps the shift at 10x (1+Emax) so even saturating TGFb cannot fully abolish IL-1 signaling. Biffi 2019 shows iCAFs persist in TGFb-rich PDAC tumors.";
params.Emax_IL1R1.derivedFrom = {};
% IL-1alpha secretion rate by cancer cells
params.k_IL1_sec.Value = 1e-12;
params.k_IL1_sec.Units = 'nanomole/(cell*day)';
params.k_IL1_sec.Notes = "IL-1alpha secretion rate by PDAC cancer cells. IL-1alpha is tumor-derived and initiates iCAF program via NF-kB -> LIF -> JAK/STAT3 (Biffi 2019, PMID: 30366930).";
params.k_IL1_sec.derivedFrom = {};
% IL-1alpha degradation rate
params.k_IL1_deg.Value = 6.93;
params.k_IL1_deg.Units = '1/day';
params.k_IL1_deg.Notes = "IL-1alpha degradation rate. Half-life ~2.4h (t1/2 = ln2/k_deg). Similar to other pro-inflammatory cytokines.";
params.k_IL1_deg.derivedFrom = {};
% IL-1 EC50 for iCAF induction
params.IL1_50.Value = 0.01;
params.IL1_50.Units = 'nanomolarity';
params.IL1_50.Notes = "Basal EC50 for IL-1-driven iCAF formation. Effective EC50 is IL1_50*(1 + TGFb/TGFb_50_IL1R1) to capture TGFb antagonism of IL-1R1 (Biffi 2019).";
params.IL1_50.derivedFrom = {};

params.n_IL1.Value = 1.0;
params.n_IL1.Units = 'dimensionless';
params.n_IL1.Notes = 'Hill coefficient for IL-1-driven iCAF formation. Ciminieri 2023 IL-6/CXCL8 dose-response in MRC-5 lung fibroblasts.';
params.n_IL1.derivedFrom = {};

% Constitutive collagen degradation rate
params.k_col_deg.Value = 0.007;
params.k_col_deg.Units = '1/day';
params.k_col_deg.Notes = "Constitutive collagen turnover rate (~100 day half-life). Collagen half-life in fibrotic/desmoplastic tissue is 60-100+ days.";
params.k_col_deg.derivedFrom = {};
% M2 macrophage MMP-mediated collagen degradation rate
params.k_MMP_M2.Value = 1e-8;
params.k_MMP_M2.Units = 'milliliter/(cell*day)';
params.k_MMP_M2.Notes = "M2 macrophage MMP/MR-mediated collagen degradation. M2-like TAMs are the dominant collagen degraders via mannose receptor endocytosis (Madsen 2013) and TAM-mediated collagen turnover in PDAC (LaRue 2022). Effective rate is gated by f_nascent (LOX-crosslinking shield) to reconcile fresh-gel in vitro assays (Zhu 2001) with in vivo fibrotic collagen persistence.";
params.k_MMP_M2.derivedFrom = {};

% phi_cross_50: collagen volume fraction at 50% MMP-susceptibility
% LOX-mediated crosslinking of mature collagen I fibrils makes them MMP-resistant.
% f_nascent = phi_cross_50 / (phi_cross_50 + phi_collagen) — only the non-crosslinked
% (nascent) fraction is susceptible to M2 MMP/MR-mediated degradation.
% Small phi_cross_50 (~0.01-0.05) means crosslinking engages early; this reconciles
% in vitro fresh-gel MMP rates (Zhu 2001, phi~0) with 60-100d fibrotic collagen
% half-life in vivo (phi~0.1-0.5).
params.phi_cross_50.Value = 0.03;
params.phi_cross_50.Units = 'dimensionless';
params.phi_cross_50.Notes = 'Collagen volume fraction at 50% MMP-susceptibility (LOX-crosslinking shield on M2 MMP degradation). Small value reflects early engagement of LOX-mediated collagen maturation.';
params.phi_cross_50.derivedFrom = {};

% Base level of collagen in normal pancreas
params.collagen_base.Value = 3;
params.collagen_base.Units = 'milligram/milliliter';
params.collagen_base.Notes = "Normal pancreas collagen content ~1-5 mg/mL.";
params.collagen_base.derivedFrom = {};

% phi_collagen: collagen volume fraction (PRIMARY ECM state variable)
% phi_collagen = V_T.collagen / (V_T * rho_collagen), dimensionless 0-1
% At diagnosis: ~0.3-0.5. Normal pancreas: ~0.12.
params.phi_collagen.Value = 0.12;
params.phi_collagen.Units = 'dimensionless';
params.phi_collagen.Notes = 'Collagen volume fraction (computed: V_T.collagen / (V_T * rho_collagen)). Primary input for all ECM-dependent Hill functions.';

% phi_col_max: physical packing ceiling for fibrillar collagen volume fraction.
% Logistic (Verhulst) saturation on myCAF collagen secretion: secretion shuts
% off as phi_collagen -> phi_col_max, structurally bounding phi below the
% packing limit regardless of how large the myCAF source grows (audit m2-A-FIB-02).
% Replaces the (1 - phi_collagen)^2 quadratic damper, whose only hard ceiling was
% phi=1 -- under sustained myCAF expansion (post-therapy fibrotic scar) the
% quadratic attractor drifts toward 1. PLACEHOLDER value ~0.5 = peak fibrillar
% collagen packing in dense PDAC stroma; pending a primary-source anchor on peak
% fibrillar packing density. Inert at the untreated-diagnosis attractor (phi well
% below 0.5); engages as a post-therapy / high-myCAF scar bound.
params.phi_col_max.Value = 0.5;
params.phi_col_max.Units = 'dimensionless';
params.phi_col_max.Notes = 'Physical packing ceiling for collagen volume fraction (logistic secretion saturation). ~0.5 = peak fibrillar collagen packing in dense PDAC stroma (placeholder pending primary-source anchor).';
params.phi_col_max.derivedFrom = {};

% phi_col_50_Kmax: collagen volume fraction for 50% carrying capacity inhibition
params.phi_col_50_Kmax.Value = 2.0;  % >1 means ECM barrier is intentionally weak (stroma-depletion paradox)
params.phi_col_50_Kmax.Units = 'dimensionless';
params.phi_col_50_Kmax.Notes = 'phi_collagen for 50% carrying capacity reduction. >1 = ECM barrier never reaches half-max, consistent with stroma-depletion worsening PDAC (Rhim 2014, Ozdemir 2014).';

% C_cap0: C-independent baseline carrying-capacity floor (cells). Additive term
% in the C_max rule representing the local pancreatic parenchymal volume the
% lesion can occupy BEFORE desmoplastic stroma / neovasculature must be built.
% Rationale: the existing C_max is ~proportional to current tumor size (stroma
% scales with C_total, vasculature with C^(1/3)), so headroom C_max/C is pinned
% at ~1.9-2.5x at every size and Gompertz net growth k*log(headroom)-death is
% size-independent and ~0 at the clinical growth rate (k_C1_growth ~0.005-0.008)
% -- the slow tumor self-limits at ~1.5 cm. A constant floor gives the small
% tumor real headroom (C_max/C large when C << C_cap0, shrinking to the
% stroma-set value as C approaches diagnosis size), letting the model reach the
% 3.2 cm diagnosis size at the data-correct VDT instead of inflating
% k_C1_growth ~6x. Default 0 recovers the prior stroma+vascular-only C_max
% EXACTLY. UN-ANCHORED structural handle (candidate future submodel anchor:
% accessible parenchymal/lobular volume). NOT a V_T volume packing limit
% (C_phys) -- it is a carrying-capacity term in the C_max rule and does not
% enter the V_T volume rule.
params.C_cap0.Value = 1.7e9;
params.C_cap0.Units = 'cell';
params.C_cap0.Notes = 'Baseline C-independent carrying-capacity floor (cells): parenchymal volume the lesion occupies pre-desmoplasia. UN-ANCHORED fitted structural handle (derivedFrom={}); value 1.7e9 is calibrated as the tumor_fold handle coupled with k_C1_growth~0.008, NOT a literature value. Candidate future submodel anchor: accessible parenchymal/lobular volume / vol_cell (queued in reconciliation_targets.csv) -- but GitHub #229 (V_T<->C_max unification) would DERIVE and dissolve it. Default 0 recovers prior stroma+vascular-only C_max exactly.';
params.C_cap0.derivedFrom = {};

% Stromal support of carrying capacity
% k_stroma_support now acts as a global multiplier on the subtype-weighted CAF
% contribution: C_max = V_T.K + k_stroma_support * (k_stroma_myCAF*myCAF +
% k_stroma_iCAF*iCAF + k_stroma_apCAF*apCAF) * ECM_factor. Hwang 2008 / Xu 2010
% submodel targets constrain the aggregate. Default lowered 1.05 → 1.0 so
% subtype weights carry the magnitude; baseline preserved by k_stroma_myCAF=1.4.
params.k_stroma_support.Value = 1.0;
params.k_stroma_support.Units = 'dimensionless';
params.k_stroma_support.Notes = 'Aggregate stromal contribution scale (global multiplier on subtype-weighted CAF sum in C_max). Constrained by Xu 2010 + Hwang 2008 orthotopic dose-response submodel targets.';
params.k_stroma_support.derivedFrom = {'Hwang 2008 DOI: 10.1158/0008-5472.CAN-07-5714', 'Rhim 2014 PMID: 25171127', 'Ozdemir 2014 PMID: 25417103'};

% Subtype-specific carrying-capacity contribution per CAF.
% Ratios myCAF : iCAF : apCAF = 1.0 : 0.2 : 0.05 reflect structural vs
% inflammatory vs antigen-presenting function. myCAFs build the desmoplastic
% architecture (αSMA+, FAP+, collagen producers — Elyada 2019, Öhlund 2017);
% iCAFs are inflammatory secretors (IL-6, CXCL12) with minimal structural role;
% apCAFs are rare MHC-II+ immunomodulatory cells (Huang 2022 Cancer Cell).
% Defaults calibrated so 1.0 * (1.4*myCAF + 0.28*iCAF + 0.07*apCAF) at
% submodel-median diagnosis ≈ old 1.05 * aPSC_total (preserves evolve_to_diagnosis).
params.k_stroma_myCAF.Value = 1.4;
params.k_stroma_myCAF.Units = 'dimensionless';
params.k_stroma_myCAF.Notes = 'Per-myCAF contribution to cancer C_max. myCAFs build PDAC desmoplasia (collagen/CXCL12/VEGF) — dominant structural CAF subtype (Elyada 2019; Dominguez 2020).';
params.k_stroma_myCAF.derivedFrom = {};

params.k_stroma_iCAF.Value = 0.28;
params.k_stroma_iCAF.Units = 'dimensionless';
params.k_stroma_iCAF.Notes = 'Per-iCAF contribution to cancer C_max. iCAFs are inflammatory secretors (IL-6/CXCL12), low αSMA, located distally from tumor — minimal structural carrying-capacity role (Elyada 2019, Öhlund 2017).';
params.k_stroma_iCAF.derivedFrom = {};

params.k_stroma_apCAF.Value = 0.07;
params.k_stroma_apCAF.Units = 'dimensionless';
params.k_stroma_apCAF.Notes = 'Per-apCAF contribution to cancer C_max. apCAFs are rare (<5% of CAFs) MHC-II+ tolerogenic antigen-presenters — predominantly immunomodulatory (induce Treg expansion in antigen-specific manner), not structural (Elyada 2019, Huang 2022 Cancer Cell).';
params.k_stroma_apCAF.derivedFrom = {};

% Stiffness-dependent CAF niche capacity
params.K_stiff_niche.Value = 1e5;
params.K_stiff_niche.Units = 'cell';
params.K_stiff_niche.Notes = 'Max CAF population sustained by stiff ECM alone (mechanotransduction niche via YAP/TAZ). Allows fibrotic scar persistence post-therapy.';
params.K_stiff_niche.derivedFrom = {};

%% Pore size parameters (immune cell exclusion)
% d_pore = d_pore_ref * (phi_col_ref / phi_collagen)^0.5
% T cell exclusion: H_pore_T = 1 / (1 + (d_pore/d_Tcell_nuc)^n_pore)
% APC exclusion: H_pore_APC = 1 / (1 + (d_pore/d_APC_nuc)^n_pore)

params.d_pore_ref.Value = 3.5;
params.d_pore_ref.Units = 'micrometer';
params.d_pore_ref.Notes = 'Effective pore size in normal pancreas at phi_col_ref. In vivo pores are larger than in vitro gels (1-3 um at 1-3 mg/mL) due to tissue architecture (channels, fiber bundle gaps).';
params.d_pore_ref.derivedFrom = {};

% phi_col_ref = reference normal-pancreas collagen volume fraction. Used as
% the normalization point for two power-law rules:
%   d_pore = d_pore_ref * (phi_col_ref/phi_collagen)^0.5   (pore-size scaling)
%   stiffness = E_ref * (phi_collagen/phi_col_ref)^n_stiff (stiffness scaling)
%
% Structural coupling: every Hill function downstream of phi_collagen
% (H_pore_T, H_pore_APC, H_stiff_fib, H_stiff_Texh, C_max suppression)
% interprets phi_collagen values relative to phi_col_ref. Re-anchoring
% phi_col_ref shifts the calibration domain for all of them; coordinated
% submodel-target inference is the right framework. See
% notes/calibration/ecm_anchor_audit_2026-04-28.md (GitHub #93).
%
% Convention: phi_col_ref is a HISTOLOGICAL convention (picrosirius/Sirius-red
% collagen-stained area fraction in normal pancreas), NOT a literal fiber
% volume fraction. rho_collagen below uses a fibrillar-scaffold density
% (~25 mg/mL) tuned against in vitro Wolf 2013 collagen gels. These two
% conventions reconcile because the downstream Hill EC50s (phi_col_50_Kmax,
% d_pore_ref, E_ref, n_stiff) were jointly calibrated against this baseline.
%
% Citation history: previously cited Apte 2004 PMID 15247571 — that PMID
% is verifiably WRONG (intervertebral-disc paper). The intended Apte 2004
% Pancreas 29(3):179-87 (PMID 15367883) is a review that does not actually
% report 0.10-0.15. Re-anchored to Grizzi 2019 (DOI 10.1186/s12967-019-1817-3),
% which reports Sirius-red collagen area in normal pancreas = 2.23 +/- 0.28%
% (n=5 cases, 50 ROIs). NOTE: 2.23% is much lower than the 0.12 default —
% the default value is held during initial inference; the joint posterior
% across {Wolf 2013 d_pore_ref, Rubiano 2018 E_ref+n_stiff, Grizzi 2019
% direct picrosirius} submodel targets will pull phi_col_ref to a
% data-consistent value.
%
% Now in pdac_priors.csv (lognormal median 0.12, sigma 0.5), anchored via
% submodel inference against the three targets listed above.
params.phi_col_ref.Value = 0.046;
params.phi_col_ref.Units = 'dimensionless';
params.phi_col_ref.Notes = 'Collagen volume fraction in normal pancreas (histological / picrosirius convention). Submodel posterior median 0.046 (sigma 0.22, 80% contraction) from joint inference across Wolf 2013, Rubiano 2018, Grizzi 2019, Valderrama 1991.';
params.phi_col_ref.derivedFrom = {'Grizzi 2019 DOI: 10.1186/s12967-019-1817-3'};

params.d_crit_T.Value = 2.3;
params.d_crit_T.Units = 'micrometer';
params.d_crit_T.Notes = 'Critical pore diameter for T cell migration arrest. From Wolf 2013: arrest at pore cross-section 4 um^2 -> effective diameter ~2.3 um.';
params.d_crit_T.derivedFrom = {'Wolf 2013 DOI: 10.1083/jcb.201210152'};

params.d_crit_APC.Value = 2.1;
params.d_crit_APC.Units = 'micrometer';
params.d_crit_APC.Notes = 'Critical pore diameter for APC migration arrest. DCs tolerate smaller pores than T cells via podosome/MMP-dependent migration.';
params.d_crit_APC.derivedFrom = {};

params.n_pore.Value = 4;
params.n_pore.Units = 'dimensionless';
params.n_pore.Notes = 'Hill coefficient for pore-size exclusion. n=4 gives gradual transition around d_pore = d_nuc.';
params.n_pore.derivedFrom = {};

%% Stiffness parameters (mechanotransduction)
% stiffness = E_ref * (phi_collagen / phi_col_ref)^n_stiff
% Normal pancreas ~1 kPa, PDAC ~2-10 kPa (Rubiano 2018)
% Units: dimensionless (kPa-scaled, avoids SimBiology unit registration issues)

params.E_ref.Value = 1.0;
params.E_ref.Units = 'dimensionless';
params.E_ref.Notes = 'Reference tissue stiffness at phi_col_ref (normal pancreas ~1 kPa). Dimensionless, kPa-scaled.';
params.E_ref.derivedFrom = {'Rubiano 2018 DOI: 10.1016/j.actbio.2018.04.025'};

params.n_stiff.Value = 1.8;
params.n_stiff.Units = 'dimensionless';
params.n_stiff.Notes = 'Power law exponent for stiffness-collagen relationship. E ~ c^n, n=1.5-2.0 for collagen gels (Latinovic 2010, Motte & Bhatt 2013).';
params.n_stiff.derivedFrom = {};

params.E_50_Texh.Value = 4.0;
params.E_50_Texh.Units = 'dimensionless';
params.E_50_Texh.Notes = 'Half-max stiffness for T cell exhaustion via chronic mechanosensing and PD-L1 upregulation. ~4 kPa = moderate PDAC stiffness.';
params.E_50_Texh.derivedFrom = {};

params.n_Texh.Value = 1.0;
params.n_Texh.Units = 'dimensionless';
params.n_Texh.Notes = 'Hill coefficient for stiffness-mediated T cell exhaustion. Saitakis 2017 cytokine dose-response suggests n~0.3-0.5 (sub-linear, rheostat-like).';
params.n_Texh.derivedFrom = {};

params.E_50_fib.Value = 5.0;
params.E_50_fib.Units = 'dimensionless';
params.E_50_fib.Notes = 'Half-max stiffness for fibroblast activation via YAP/TAZ mechanotransduction. Chang 2025 PDAC CAF YAP dose-response.';
params.E_50_fib.derivedFrom = {};

params.n_fib.Value = 1.0;
params.n_fib.Units = 'dimensionless';
params.n_fib.Notes = 'Hill coefficient for stiffness-mediated fibroblast activation (YAP/TAZ). Chang 2025 PDAC CAF data.';
params.n_fib.derivedFrom = {};

% Collagen molecular weight (legacy)
params.collagen_MW.Value = 3e5;
params.collagen_MW.Units = 'gram/mole';
params.collagen_MW.Notes = 'Effective MW for collagen (type I). 3e5 g/mole = 300 kDa.';
params.collagen_MW.derivedFrom = 'Biochemistry reference (collagen type I ~300 kDa)';
% Collagen protein packing density within the extracellular space (NOT bulk
% hydrated ECM tissue density, which is ~150 mg/mL — see ecm_tissue_density
% in reference_values.yaml). V_T.collagen/rho_collagen gives the extracellular
% volume occupied by collagen. phi_collagen converges asymptotically to 1
% as collagen dominates V_T.
% Antoine 2014: cancerous tissue interstitium 9-45 mg/mL (geometric mean ~20).
% Kai 2023: CT-derived ECV in PDAC F3 fibrosis ~42.5%.
% See calibration_targets/submodel_targets/ECM_density_PDAC_deriv001.yaml.
%
% rho_collagen = mass-to-volume conversion for the collagen contribution to V_T.
% V_T.collagen/rho_collagen gives the V_T volume occupied by collagen scaffold.
% Value 25 mg/mL interprets phi_collagen as "fraction of V_T that is fibrillar
% scaffold including void spaces between fibers", consistent with the in vitro
% Wolf 2013 collagen-gel convention used by the d_pore_ref submodel target.
% Pure fibrillar density (~1340 mg/mL hydrated) would interpret phi as literal
% fiber volume (giving phi ~ 1-3% in fibrotic PDAC); a third candidate is the
% bulk-tissue mass density (~9 mg/mL) implied by Imamura 1995 wet-tissue
% collagen / Apte phi_col_ref pairing.
%
% Structural coupling: rho_collagen couples tightly with Ve_T (cell-volume
% contribution to V_T) and phi_col_ref (Hill normalization). See
% notes/calibration/ecm_anchor_audit_2026-04-28.md (GitHub #93). The submodel
% target framework is the right way to anchor — varying rho_collagen alone
% breaks the Wolf 2013 d_pore submodel target's phi-input convention.
%
% Anchored to Valderrama 1991 (DOI 10.1007/BF02924250) Sirius-red
% colorimetric collagen content in n=11 normal human pancreas:
% 89.1 +/- 11.6 ug collagen / mg total protein. The submodel target
% (calibration_targets/submodel_targets/rho_collagen_valderrama1991_PDAC_deriv001.yaml)
% jointly constrains rho_collagen with phi_col_ref via the algebraic
% identity (phi_col_ref * rho_collagen) / protein_per_wet_pancreas =
% c_mass_per_protein. Joint inference with the three phi_col_ref-only
% targets (Wolf 2013, Rubiano 2018, Grizzi 2019) breaks the product
% degeneracy and yields individual posteriors on rho_collagen and
% phi_col_ref.
%
% Now in pdac_priors.csv (lognormal median 0.025, sigma 0.3) so the
% Valderrama submodel target can pull a data-consistent posterior.
params.rho_collagen.Value = 0.103;
params.rho_collagen.Units = 'gram/milliliter';
params.rho_collagen.Notes = 'Collagen protein packing density (scaffold-equivalent including void spaces). Submodel posterior median 0.103 g/mL (sigma 0.46, 57% contraction) from joint inference jointly with phi_col_ref via Valderrama 1991 colorimetry + Wolf 2013 / Rubiano 2018 / Grizzi 2019 phi anchors.';
% Stiffness-mediated T cell exhaustion rate
params.k_stiff_CD8_exh.Value = 0.02;
params.k_stiff_CD8_exh.Units = '1/day';
params.k_stiff_CD8_exh.Notes = 'Rate of T cell exhaustion driven by stiff collagen matrix (Fitted to Ma 2022 single cell data Texh fraction of all CD8+ T cell).';
% vol_qPSC_cell
params.vol_qPSCcell.Value = 1500;
params.vol_qPSCcell.Units = 'micrometer^3/cell';
params.vol_qPSCcell.Notes = 'https://doi.org/10.1016/B0-12-370879-6' ;
% CAF subtype cell volumes (all inherit from vol_aPSCcell initially)
params.vol_iCAFcell.Value = 2000;
params.vol_iCAFcell.Units = 'micrometer^3/cell';
params.vol_iCAFcell.Notes = 'iCAF cell volume. Same as original vol_aPSCcell.';
params.vol_iCAFcell.derivedFrom = {};

params.vol_myCAFcell.Value = 3500;
params.vol_myCAFcell.Units = 'micrometer^3/cell';
params.vol_myCAFcell.Notes = 'myCAF cell volume. Same as original vol_aPSCcell.';
params.vol_myCAFcell.derivedFrom = {};

params.vol_apCAFcell.Value = 2500;
params.vol_apCAFcell.Units = 'micrometer^3/cell';
params.vol_apCAFcell.Notes = 'apCAF cell volume. Same as original vol_aPSCcell.';
params.vol_apCAFcell.derivedFrom = {};

%% CXCL12 Module Parameters (T Cell Exclusion - Critical for PDAC)
% CXCL12 secretion rate by iCAFs (reassigned from myCAF based on scRNA-seq)
params.k_CXCL12_sec_iCAF.Value = 1e-10;
params.k_CXCL12_sec_iCAF.Units = 'nanomolarity*liter/(cell*day)';
params.k_CXCL12_sec_iCAF.Notes = 'CXCL12/SDF-1 secretion by iCAFs. Reassigned from myCAF: scRNA-seq consistently shows CXCL12 as iCAF marker (Elyada 2019 PMID: 31197017, Ogawa 2021 PMID: 33504555). T cell exclusion via CXCR4 signaling (Feig 2013 PMID: 24101459).';
params.k_CXCL12_sec_iCAF.derivedFrom = {'Elyada 2019 PMID: 31197017', 'Ogawa 2021 PMID: 33504555', 'Feig 2013 PMID: 24101459'};
% CXCL12 secretion rate by tumor cells
params.k_CXCL12_sec_cancer.Value = 5e-11;
params.k_CXCL12_sec_cancer.Units = 'nanomolarity*liter/(cell*day)';
params.k_CXCL12_sec_cancer.Notes = 'CXCL12 secretion by PDAC tumor cells. Lower per-cell rate than aPSCs but significant due to tumor mass. Makes T cell exclusion partially tumor-autonomous (Feig et al. 2013 PNAS PMID: 24101459).';
params.k_CXCL12_sec_cancer.derivedFrom = {};
% CXCL12 degradation/clearance rate
params.k_CXCL12_deg.Value = 2.0;
params.k_CXCL12_deg.Units = '1/day';
params.k_CXCL12_deg.Notes = 'CXCL12 degradation rate. Chemokine half-lives typically 4-12 hours (Proudfoot et al. 2003 PMID: 12646643).';
% EC50 for CXCL12-mediated T cell exclusion
params.CXCL12_50_Texcl.Value = 8.0;
params.CXCL12_50_Texcl.Units = 'nanomolarity';
params.CXCL12_50_Texcl.Notes = 'CXCR4 receptor affinity for CXCL12. Anchored to the CXCR4-CXCL12 Kd (7.1-7.5 nM Scatchard, CXCL12_50_Texcl_PDAC_deriv001 / DOI 10.4049/jimmunol.160.2.877); the former 2.0 nM was a hand-tuned immune-desert knob 4-9x below the measured Kd, over-strengthening CXCL12 exclusion. Set to ~8 nM (geometric-mean of the Kd measurements). Low EC50 = high receptor sensitivity.';

% Vascular-access gating of blood->tumor extravasation (PDAC hypovascularity).
% As PDAC grows it outpaces its vascular support: V_T.K (vasculature-dependent
% carrying capacity) grows slower than V_T, so vascular density V_T.K/V_T falls.
% H_vasc = (V_T.K/V_T)^n_vasc / (K50_vasc^n_vasc + (V_T.K/V_T)^n_vasc) gates the
% per-volume extravasation rate of CD8/Th recruitment reactions, encoding the
% documented progressive immune exclusion of growing PDAC (vessel collapse under
% desmoplastic compression — Olive 2009, Provenzano 2012). Without this gate,
% recruitment scales with raw tumor volume and intratumoral CD8 density rises
% over untreated progression instead of falling.
%
% K50_vasc and n_vasc are FIXED structural constants, NOT SBI-fit priors. As
% free priors they were under-identified — the joint NPE left them at their
% uninformative prior and the gate stayed weak (sbi_multi_hvasc: K50_vasc
% posterior ≈ prior, cd8_density_declines posterior-predictive still ×1.75).
% K50_vasc is anchored to the diagnosis-state vascular density (V_T.K/V_T at
% diagnosis ≈ 9e6–1e7 cell/mL, so the gate is ~half-engaged at diagnosis —
% consistent with PDAC being immune-excluded already at presentation — and
% declines steeply as the tumor outgrows its vasculature). n_vasc = 3 is a
% standard moderate-cooperativity Hill exponent; the supralinear shape is what
% amplifies the raw 2.4x V_T.K/V_T decline into the steeper extravasation
% falloff the progressive-exclusion biology requires (a linear gate only
% passes the 2.4x through, insufficient). Trace-confirmed: clinical_progression
% 90d CD8 density fold 3.77 (no gate) -> 0.58 at these values.
params.K50_vasc.Value = 1e7;
params.K50_vasc.Units = 'cell/milliliter';
params.K50_vasc.Notes = 'EC50 for vascular-access gating of blood->tumor extravasation, anchored to the diagnosis-state vascular density V_T.K/V_T (~9e6-1e7 cell/mL). Fixed structural constant, not SBI-fit.';
params.K50_vasc.derivedFrom = {};
params.n_vasc.Value = 3.0;
params.n_vasc.Units = 'dimensionless';
params.n_vasc.Notes = 'Hill coefficient for vascular-access gating of extravasation. Fixed structural constant (moderate cooperativity); the supralinear shape amplifies the V_T.K/V_T decline. Not SBI-fit.';
params.n_vasc.derivedFrom = {};

%% IL-6 Module Parameters (Myeloid Immunosuppression)
% Effective iCAF-scaled IL-6 source. C11 retired separately fitted cancer and
% TAM source rates and absorbed their diagnosis flux into this coordinate.
params.k_IL6_sec_iCAF.Value = 1.1036219418946422e-9;
params.k_IL6_sec_iCAF.Units = 'nanomolarity*liter/(cell*day)';
params.k_IL6_sec_iCAF.Notes = 'Effective iCAF-scaled IL-6 source rate. Initialized to preserve the pre-C11 total source flux at prior-centre diagnosis (1.1036219419e-9); source fractions stayed within ~1% across all paper arms and through day 365. The underlying iCAF culture anchor is translated by the fixed C11 absorption factor 3.676852. Cancer- and TAM-derived IL-6 are acknowledged but not separately attributed. Requires updated-sampler refit.';
params.k_IL6_sec_iCAF.derivedFrom = {'Saga 2018 DOI: 10.3390/ijms19103043', 'C11 nominal source-flux trace', 'estimated'};
% IL-6 degradation rate
params.k_IL6_deg.Value = 16.0;
params.k_IL6_deg.Units = '1/day';
params.k_IL6_deg.Notes = 'IL-6 degradation/clearance rate. Half-life ~100 minutes in circulation (Castell et al. 1988 PMID: 2455694).';
% EC50 for IL-6 enhancement of MDSC recruitment
% All three IL-6 EC50s harmonized to 3.3x total spread (was 50x).
% IL-6R/gp130/STAT3 Kd ~1 nM, functional EC50 0.01-0.1 nM due to
% receptor clustering. MDSCs most sensitive (high gp130 expression).
params.IL6_50_MDSC.Value = 0.015;
params.IL6_50_MDSC.Units = 'nanomolarity';
params.IL6_50_MDSC.Notes = 'IL-6R/gp130 signaling EC50 for MDSC recruitment (MDSCs are among the most IL-6-sensitive myeloid cells). Value 0.015 nM is a PLACEHOLDER, shadowed at inference time by the IL6_50_MDSC submodel target (calibration_targets/submodel_targets/IL6_50_MDSC_PDAC_deriv001.yaml) -- the realized EC50 comes from that posterior, not this default. Stale prior narrative ("raised from 0.02 to 0.06 to avoid saturation") removed: it never matched the 0.015 placeholder and is superseded by the submodel anchor. Audit x2-A-IL6NOTE-01 / x1-OBS-02.';
params.IL6_50_MDSC.derivedFrom = {};
% EC50 for IL-6 promotion of M2 macrophage polarization
params.IL6_50_M2.Value = 0.015;
params.IL6_50_M2.Units = 'nanomolarity';
params.IL6_50_M2.Notes = 'STAT3 activation via IL-6 promoting M2 polarization. Least sensitive of the three IL-6 targets (M2 polarization primarily driven by IL-10/TGFb; IL-6 is secondary). Lowered from 1.0 to match model IL-6 range.';
params.IL6_50_M2.derivedFrom = {};
% EC50 for IL-6 autocrine feedback on iCAF phenotype
% NOTE: IL-6 is used as a proxy for LIF (leukemia inhibitory factor), the
% actual autocrine mediator of iCAF phenotype maintenance. Biffi 2019 showed
% IL-6 neutralization and IL-6 KO had no effect on iCAF markers (Supp Fig
% S4B-D), while LIF neutralization significantly reduced them (Fig 3E). Both
% IL-6 and LIF signal through gp130/JAK/STAT3, so IL-6 serves as a surrogate
% for the lumped JAK/STAT3 ligand signal. Default value is a placeholder;
% softly anchored via the Donahue2024 submodel target (see Notes).
params.IL6_50_iCAF.Value = 0.015;
params.IL6_50_iCAF.Units = 'nanomolarity';
params.IL6_50_iCAF.Notes = 'EC50 for JAK/STAT3 autocrine feedback promoting iCAF phenotype. IL-6 is a proxy for LIF — the actual mediator (Biffi 2019 Supp Fig S4B-D). Soft submodel anchor: Donahue2024 Fig 6D rIL-6->pSTAT3 dose-response in pancreatic fibroblasts (n=1 blot, EC50 ~0.39 nM); CSV prior sigma widened to 2.0 so the data is reachable in the tail (median kept at 0.015 placeholder). Removed from the IL6_EC50s hierarchical group: the fibroblast EC50 is ~27x the myeloid (M2/MDSC) members, consistent with cell-type-specific IL-6Ralpha:gp130 ratios (Reeh 2019).';
params.IL6_50_iCAF.derivedFrom = {'Biffi 2019 PMID: 30366930'};
% Fold-enhancement of iCAF activation/interconversion by JAK/STAT3 feedback
% See IL6_50_iCAF comment above — IL-6 is a proxy for LIF/JAK/STAT3 signaling.
params.f_IL6_iCAF.Value = 3.0;
params.f_IL6_iCAF.Units = 'dimensionless';
params.f_IL6_iCAF.Notes = 'Max fold-enhancement of iCAF formation by JAK/STAT3 autocrine feedback (IL-6 as proxy for LIF). At saturating signal, rate is (1+f_IL6_iCAF)=4x baseline. Constrained indirectly by full-model iCAF density targets.';
params.f_IL6_iCAF.derivedFrom = {'Biffi 2019 PMID: 30366930'};

%% Angiogenic Factor Production by myCAFs
% VEGF secretion by myCAFs (was k_vas_apsc_sec)
params.k_vas_myCAF_sec.Value =  0.002;
params.k_vas_myCAF_sec.Units = 'picogram/(cell*day)';
params.k_vas_myCAF_sec.Notes = 'Angiogenic factor (VEGF) secretion by myCAFs. Was k_vas_apsc_sec. ~50% of M2 macrophage rate (k_vas_Msec = 1.7e-5).';
params.k_vas_myCAF_sec.derivedFrom = {};

%% Initial Condition Parameters for PSC Populations
% Normal pancreas PSC density
params.PSC_density_normal.Value = 0.04;
params.PSC_density_normal.Units = 'dimensionless';
params.PSC_density_normal.Notes = 'Ratio of PSCs (pancreatic stellate cells) to total cells in normal pancreas. Literature estimates 2-7% (Apte et al. 2012 PMID: 15286120). Use 4% as baseline.';
% Baseline aPSC activation fraction
params.aPSC_fraction_baseline.Value = 0.20;
params.aPSC_fraction_baseline.Units = 'dimensionless';
params.aPSC_fraction_baseline.Notes = 'Fraction of PSCs that are activated (aPSC/total PSC) at baseline PDAC. Highly variable (10-80%). KEY CALIBRATION TARGET - adjust to match αSMA+ cell density in patient biopsies. 20% is moderate baseline estimate.';

% Treg-specific proliferation and death rates (required by Treg_module)
params.k_Treg_pro.Value = 0.05;
params.k_Treg_pro.Units = '1/day';
params.k_Treg_pro.Notes = 'Treg proliferation rate - lower than conventional CD4+ for homeostatic maintenance';
params.k_Treg_pro.derivedFrom = {};
params.k_Treg_death.Value = 0.1;
params.k_Treg_death.Units = '1/day';
params.k_Treg_death.Notes = 'Treg death rate - similar to conventional CD4+ for balanced homeostasis';
params.k_Treg_death.derivedFrom = {};
% Tumor-local Treg proliferation rate
params.k_Treg_pro_tumor.Value = 0.2;
params.k_Treg_pro_tumor.Units = '1/day';
params.k_Treg_pro_tumor.Notes = 'Tumor-local Treg proliferation rate. Reduced from 0.3 to 0.2; Ki67+ data supports ~1.3x Th rate (Treg 30-60% vs Th 20-30%), not the previous 2x. PMID: 28007400 (Balli 2017), PMID: 29089587 (Plitas 2016).';
params.k_Treg_pro_tumor.derivedFrom = {'PMID: 28007400', 'PMID: 29089587'};
% Maximum Treg density in tumor (NICHE CARRYING CAPACITY — now the level-setter)
params.rho_Treg_max.Value = 1.3e6;
params.rho_Treg_max.Units = 'cell/milliliter';
params.rho_Treg_max.Notes = 'Niche carrying capacity for the tumoral Treg pool. Re-centered 9.12e6 -> 1.3e6 on 2026-06-18 (v23). The tumoral linear Treg death was removed (Treg_module.m), so this cap BINDS and sets the in-tumor Treg level (was inert when the level rode the k_Treg_pro_tumor - k_Treg_death knife-edge); the cap therefore equals the OPERATING Treg density. The 9.12e6 value was a STALE residue of the EXCLUDED Brouwer2022 submodel anchor (rho_Treg_max_PDAC_deriv001, now in submodel_targets/excluded/) and carried a PROJECTION BUG: Brouwer mapped 36.5 cells/mm^2 -> 9.12e6 cell/mL using a thickness-only (4 um) 2D->3D projection, but the full-model treg_density observable uses the Abercrombie (thickness + cell_diameter = 11 um) projection, so 9.12e6 reads back as ~100 cells/mm^2 in-model -- 2.75x inflated even on Brouwer''s own data. Because rho_Treg_max is NOT submodel-anchored (Brouwer excluded), SBI could not overcome the high prior median (posterior 3.95e6, treg density 48.6 vs target 14 -- 3.5x over; cd8:treg ratio ~1 vs target ~10-24). Re-centered to the LIVE full-model density target (Sivagnanam 2021, 14 cells/mm^2, n=113 treatment-naive PDAC) via the model''s own 11-um projection: 14 / 1.1e-5 = 1.27e6 cell/mL -> rounded 1.3e6. sigma=1.5 keeps the densest-tumor / Brouwer / Kiryu cohorts reachable in the upper tail. Diagnostic traces (2026-06-18): sweeping rho_Treg_max down lowers Treg proportionally while leaving in-tumor CD8 and cancer C1 UNCHANGED at both diagnosis and gvax_nivo d21 -- the Treg pool is supply/cap-set, not load-bearing for CD8 suppression or tumor control, so the re-center is safe for the growth and CD8 structural wins. Old note: prior 2e7->1e7->9.12e6 lineage had no live lit justification once the Brouwer anchor was excluded.';
params.rho_Treg_max.derivedFrom = {};
% Fraction of naturally-occurring Tregs
params.f_nTreg.Value = 0.25;
params.f_nTreg.Units = 'dimensionless';
params.f_nTreg.Notes = 'Fraction of Tregs that are naturally-occurring (antigen-independent) vs induced (antigen-dependent)';


% (w_IL2_CD8 / w_IL2_Treg retired: the bulk IL-2 consumption reactions they
% weighted were removed in the IL2_niche QSS reformulation. The CD8-vs-Treg uptake
% asymmetry is now carried by kappa_IL2_Treg in the niche demand term.)

%% GVAX Parameters
params.k_GVAX_clearance.Value = 0.1;
params.k_GVAX_clearance.Units = '1/day';
params.k_GVAX_clearance.Notes = 'GVAX cell clearance rate from tumor';

% GM-CSF production and degradation
params.k_GMCSF_prod.Value = 1e-8;
params.k_GMCSF_prod.Units = 'nanomolarity/cell/day';
params.k_GMCSF_prod.Notes = 'GM-CSF production rate per GVAX cell per day';

params.k_GMCSF_deg.Value = 5.0;
params.k_GMCSF_deg.Units = '1/day';
params.k_GMCSF_deg.Notes = 'GM-CSF degradation/clearance rate (half-life ~3 hours)';

% GM-CSF EC50 (still used for injection site effects)
params.EC50_GMCSF.Value = 0.0029400504281744564;
params.EC50_GMCSF.Units = 'nanomolarity';
params.EC50_GMCSF.Notes = 'EC50 for GM-CSF-mediated APC recruitment and maturation';

%% GVAX Intradermal Injection Site Parameters
params.k_APC_recruit_ID.Value = 1e5;
params.k_APC_recruit_ID.Units = 'cell/day';
params.k_APC_recruit_ID.Notes = 'APC recruitment rate to intradermal injection site';

% APC maturation at injection site
params.k_APC_mature_ID.Value = 2.7280598629067656;
params.k_APC_mature_ID.Units = '1/day';
params.k_APC_mature_ID.Notes = 'APC maturation rate at injection site';

% APC migration from injection site to lymph nodes
params.k_APC_mig_ID_LN.Value = 0.7;
params.k_APC_mig_ID_LN.Units = '1/day';
params.k_APC_mig_ID_LN.Notes = 'APC migration rate from injection site to lymph nodes';

% f_mAPC_mcDC1_fate removed (mid-tier #15): the direct depot-moDC->mcDC1
% cross-priming route it scaled was retired. moDCs can cross-prime in principle
% (Briseno 2016, PMC4941620) but require GM-CSF + IL-4 (the depot supplies only
% GM-CSF), and GVAX CD8 priming is host-cDC1-dependent (collapses in Batf3-/-,
% Ferris 2022 PMID 35648641) -- so cross-priming flows through the mcDC2->cDC1
% antigen-handoff route (k_cDC1_Ag_transfer), not a direct moDC path. The old
% value 0.3 was an unanchored placeholder. Re-add with an IL-4 gate + submodel
% anchor if a motivating context arises.

% APC death at injection site
params.k_APC_death_ID.Value = 0.1;
params.k_APC_death_ID.Units = '1/day';
params.k_APC_death_ID.Notes = 'APC death rate at injection site';

% Antigen consumption per APC maturation (replaces separate uptake kinetics)
params.sigma_P1_per_mAPC.Value = 1e-9;
params.sigma_P1_per_mAPC.Units = 'nanomole/cell';
params.sigma_P1_per_mAPC.Notes = 'Antigen consumed per APC maturation event at injection site';

% EC50 for antigen-dependent APC maturation
params.EC50_P1_mature.Value = 0.1;
params.EC50_P1_mature.Units = 'nanomolarity';
params.EC50_P1_mature.Notes = 'EC50 for antigen-dependent APC maturation. Lowered from 10 nM to 0.1 nM on 2026-04-21 (Fix 3): with the 10 nM value, V_ID.P1_GVAX (~0.009 nM steady state) produced a Hill factor of ~0.0009 in J_mature_ID, so GVAX-derived antigen was never a meaningful contributor to APC maturation — maturation proceeded almost entirely via the GMCSF + baseline APC terms. 0.1 nM is an estimated value in the nanomolar-affinity range typical of peptide/antigen-receptor interactions; not anchored to a specific primary source. Candidate for submodel-target extraction.';

% Injection site volume
params.V_ID_volume.Value = 1e-4;
params.V_ID_volume.Units = 'liter';
params.V_ID_volume.Notes = 'Volume of intradermal injection site (~0.1 mL)';

% APC capacity and baseline recruitment at injection site
params.K_APC_ID.Value = 1e6;
params.K_APC_ID.Units = 'cell';
params.K_APC_ID.Notes = 'Maximum APC capacity at injection site';

params.k_APC_recruit0.Value = 1e2;
params.k_APC_recruit0.Units = 'cell/day';
params.k_APC_recruit0.Notes = 'Baseline APC recruitment rate to injection site';

% Separate mature APC death rate
params.k_mAPC_death_ID.Value = 0.7;
params.k_mAPC_death_ID.Units = '1/day';
params.k_mAPC_death_ID.Notes = 'Mature APC death rate at injection site';

% Antigen yield and degradation
params.y_Ag_per_cell.Value = 1e-9;
params.y_Ag_per_cell.Units = 'nanomole/cell';
params.y_Ag_per_cell.Notes = 'Antigen yield per GVAX cell death';

params.k_P1_deg_ID.Value = 10.0;
params.k_P1_deg_ID.Units = '1/day';
params.k_P1_deg_ID.Notes = 'Antigen degradation/drainage rate at injection site';

% Vaccine-antigen presentation-uptake rate: feeds GVAX depot antigen into the
% shared endosomal antigen-1 pool (V_e.P1) so the vaccine raises the pMHC display
% signal H_P1, not just the mature-DC count (audit consensus #2). Direct parallel
% to the tumor-antigen uptake rate k_up (=14.4, same units 1/day/cell); centered
% at k_up because GVAX cells are irradiated tumor cells (same antigens) delivered
% as an adjuvanted bolus to a DC-rich site, so per-unit-antigen presentation is at
% least as efficient as in-situ tumor uptake. SBI-varied; submodel anchor pending.
params.k_GVAX_present.Value = 14.4;
params.k_GVAX_present.Units = '1/day/cell';
params.k_GVAX_present.Notes = 'Vaccine-antigen presentation-uptake rate into the shared endosomal pMHC pool; parallel to tumor k_up=14.4 (audit consensus #2, GVAX->H_P1 coupling); placeholder centered at k_up, SBI-varied, submodel anchor pending';
params.k_GVAX_present.derivedFrom = {};

% APC migration saturation parameter
params.K_APC_mig.Value = 1e5;
params.K_APC_mig.Units = 'cell';
params.K_APC_mig.Notes = 'Saturation constant for APC migration to lymph nodes';

% Cyclophosphamide parameters (low-dose for Treg depletion)
params.k_Cy_clear.Value = 2.5;  % 1/day (Cy half-life ~12-24 hours)
params.k_Cy_clear.Units = "1/day";
params.k_Cy_clear.Notes = "Cyclophosphamide clearance rate";

params.k_Cy_Treg_deplete.Value = 0.15;  % 1/day (depletion rate at saturating Cy)
params.k_Cy_Treg_deplete.Units = "1/day";
params.k_Cy_Treg_deplete.Notes = "Maximum Treg depletion rate by cyclophosphamide";

params.Cy_50_Treg.Value = 50;  % mg (EC50 for Treg depletion)
params.Cy_50_Treg.Units = "milligram";
params.Cy_50_Treg.Notes = "Cyclophosphamide concentration for 50% Treg depletion";

% Cyclophosphamide durable PD effect compartment (added 2026-06-01)
% Low-dose Cy depletes cycling Tregs acutely (k_Cy_Treg_deplete, drug-present)
% AND durably suppresses Treg reconstitution past the parent-drug PK window.
% Parent Cy (t1/2 ~hours, k_Cy_clear) is gone by ~day 8, so the acute depletion
% is near-inert at the day-21 neoadjuvant readout. An indirect-response effect
% compartment (Cy_eff) carries the durable suppression: Cy_eff builds from drug
% exposure (k_Cy_on, gated by the same Cy Hill) and decays at k_Cy_eff_decay,
% throttling tumor-Treg proliferation by up to Emax_Cy_Treg. k_Cy_eff_decay sets
% the suppression duration (PD memory); Emax_Cy_Treg sets its depth. Both anchor
% to the cy_treg_recovery submodel target. k_Cy_on / Cy_eff_50 set the throttle
% saturation jointly and are only weakly identifiable individually -> fixed.
params.k_Cy_eff_decay.Value = 0.05;  % 1/day (PD effect t1/2 ~14 days; durable past d21)
params.k_Cy_eff_decay.Units = "1/day";
params.k_Cy_eff_decay.Notes = "Decay rate of the cyclophosphamide durable PD effect compartment (sets Treg-suppression duration)";

params.Emax_Cy_Treg.Value = 0.8;  % dimensionless (max fractional suppression of tumor-Treg proliferation)
params.Emax_Cy_Treg.Units = "dimensionless";
params.Emax_Cy_Treg.Notes = "Maximum fractional suppression of tumor-Treg proliferation by the Cy effect compartment";

params.k_Cy_on.Value = 10;  % milligram/day (build rate of Cy_eff from drug exposure; fixed, weakly identifiable)
params.k_Cy_on.Units = "milligram/day";
params.k_Cy_on.Notes = "Build rate of the cyclophosphamide effect compartment from drug exposure (structural; weakly identifiable, fixed)";

params.Cy_eff_50.Value = 1;  % milligram (half-saturation of the Cy_eff -> Treg-proliferation throttle; fixed, weakly identifiable)
params.Cy_eff_50.Units = "milligram";
params.Cy_eff_50.Notes = "Half-saturation of the Cy effect compartment on tumor-Treg proliferation (structural; weakly identifiable, fixed)";

%% apCAF-Specific Parameters (Tolerogenic Antigen Presentation)
% apCAFs express MHC-II, engage antigen-specific CD4 T cells, and can induce
% Tregs without canonical costimulation (Elyada 2019; Huang 2022).
params.k_apCAF_iTreg.Value = 0.15;
params.k_apCAF_iTreg.Units = '1/day';
params.k_apCAF_iTreg.Notes = 'Effective antigen-specific induced-Treg rate via apCAF presentation. Multiplied by the apCAF fraction and a local-antigen QSS gate; contact geometry and absolute-count saturation are collapsed into this one fitted coordinate. The initial value preserves the retired formulation at the nominal 10% apCAF share (0.15/day * 0.1 = 0.015/day, matching 0.03/day at its half-max count gate). Requires refit. Elyada 2019 PMID: 31197017; Huang 2022 PMID: 35523176.';
params.k_apCAF_iTreg.derivedFrom = {'Elyada 2019 PMID: 31197017', 'Huang 2022 PMID: 35523176', 'estimated'};

params.P1_50_apCAF.Value = 1e-9;
params.P1_50_apCAF.Units = 'molarity';
params.P1_50_apCAF.Notes = 'Fixed EC50 for the quasi-steady local-antigen gate on apCAF MHC-II-dependent Treg induction. Held at 1e-9 M and not varied across patients pending an operating-range trace.';
params.P1_50_apCAF.derivedFrom = {'estimated'};

%% Th Tumor-Local Proliferation Parameters
% Tumor-local Th proliferation rate
% Th1 cells show moderate Ki67+ (20-30%) in PDAC TME, lower than Tregs (30-60%)
params.k_Th_pro_tumor.Value = 0.15;
params.k_Th_pro_tumor.Units = '1/day';
params.k_Th_pro_tumor.Notes = 'Tumor-local Th proliferation rate. ~0.5x Treg rate (0.3/day). Th1 Ki67+ fraction is lower than Treg in TME.';
params.k_Th_pro_tumor.derivedFrom = {};
% Maximum Th density in tumor (carrying capacity for tumor-local proliferation)
params.rho_Th_max.Value = 5e6;
params.rho_Th_max.Units = 'cell/milliliter';
params.rho_Th_max.Notes = 'Maximum Th density in tumor. Lower than Treg carrying capacity (2e7) reflecting that Th cells are less abundant than Tregs in PDAC TME. Estimated from IHC Th densities (50-200 cells/mm^2 ~ 1e6-1e7 cells/mL volumetrically).';
params.rho_Th_max.derivedFrom = {};

% Baseline tumor-immunosurveillance CD4 Th seeding (cancer-gated floor; v25, 2026-06-18).
% Direct analog of k_CD8_seed (v24, Teff_module.m). The in-tumor Th renewal is logistic
% (proportional to V_T.Th -> V_T.Th=0 is a fixed point), so live Th bistable-collapses for
% draws where k_Th_pro_tumor < its drain stack -- the v20 Th_exh gateway floored Th_exh but
% NOT live Th (cd4_helper_density median 0.1 vs obs 17, q95 1671 = multimodal). Flux =
% k_Th_seed * V_T(mL) * C_total^2/(C_total^2+K_C2_recruit) (cell/day). Restores the missing
% baseline CD4 TIL inflow so cold tumors keep a realistic Th floor and cd4_pct / cd4_helper
% stop cratering (the v24 CD8 seed diluted cd4_pct further by adding CD8 to nucleated_total).
% ROUTE-2 FIX 2026-06-22: FIRMED 3e4 -> 1e5 (and prior sigma 1.5 -> 0.5 in
% pdac_priors.csv). Retiring the inflation Hill bounded the cap and killed the
% saturation (right) tail, but the collapse (left) tail persists: for draws where
% k_Th_pro_tumor < drain stack the renewal-vs-drain net rate goes negative and the
% attractor drops to seed/|net| -- a too-weak seed lets it crater. A 2-compartment
% toy on the real posterior (Th+Th_exh steady state) gives q95/q05: old inflated cap
% 797x -> Hill-retired alone 144x -> Hill-retired + this ~1e5 floor 18x (CD8 family is
% 14-15x). Mirrors the CD8 v24 seed + v26 cap recipe. Still an SBI knob (sigma 0.5 keeps
% it firm but not pinned); the data sets the final floor. See cd4_pct_bistability_formulation.md.
params.k_Th_seed.Value = 1e5;
params.k_Th_seed.Units = 'cell/(milliliter*day)';
params.k_Th_seed.Notes = 'Baseline tumor-immunosurveillance CD4 Th seeding density-rate (cancer-gated effector floor; analog of k_CD8_seed). Route-2 firmed to 1e5 (2026-06-22) so the collapse branch lands at a physical cold level after the inflation-Hill cap fix. NOTE: the analogous CD8 seed (k_CD8_seed) is INTENTIONALLY disabled (=0 + reaction commented out in Teff_module) post in-tumor restructure; the Th seed is retained on purpose because the CD4 pool still has the prolif-vs-drain bistability the CD8 restructure removed (see in_tumor_cd8_terms_realism.md, cd4_pct_bistability_formulation.md). Estimated -- no literature rate anchor; observable-anchored via cd4_helper_density. SBI-tunable (sigma 0.5).';
params.k_Th_seed.derivedFrom = {};

%% Th carrying capacity (route-2 fix, 2026-06-22)
% The CD8-inflammation Hill cap-scaling (alpha_Th_cap, CD8_50_Tcap) was RETIRED.
% It was dead: realized V_T.CD8 (1e6-1e9) >> CD8_50_Tcap (1e5), so the Hill sat
% pinned at ~1 in 94% of posterior draws (verified at overlay), turning
% alpha_Th_cap into an unidentified multiplier that inflated the
% Karakhanova-anchored rho_Th_max cap ~6x and drove the cd4_pct over-dispersion.
% The Th cap is now the clean data-anchored rho_Th_max*V_T (mirrors the CD8 base
% cap). V_T.Treg cap is likewise rho_Treg_max*V_T.
%
% V_T.Treg cap = rho_Treg_max * V_T (no apCAF-driven additive term).
%   The apCAF-driven extra Treg cap (rho_Treg_apCAF_extra * H_apCAF) was
%   retired 2026-05-01: a counterfactual SBI run pinning the parameter at
%   ~0 produced D²_post/D²_prior = 0.85 (vs 0.91 with the pathway active),
%   posterior reachability for treg_density_baseline rose from ~17% to
%   53%, and cd4_helper dropped out of the LIKELY_MISSPECIFIED list. The
%   data prefers the apCAF→Treg-cap pathway off, so the additive-cap
%   formulation introduced in the 2026-05-01 PR was rolled back to the
%   simpler logistic. The remaining apCAF→Treg coupling is the direct
%   antigen-specific Th→Treg contribution via k_apCAF_iTreg in fibroblast_module_enhanced.
% alpha_Th_cap and CD8_50_Tcap RETIRED 2026-06-22 (route-2 Th cap fix): the
% CD8-inflammation Hill they parameterized was dead (H_CD8~1 everywhere) and
% only inflated the data-anchored rho_Th_max cap. Removed from Th_module +
% Treg_module + pdac_priors.csv. See notes/calibration/cd4_pct_bistability_formulation.md.
% Retired 2026-05-01: rho_Treg_apCAF_extra (additive Treg-cap apCAF
% pathway) and apCAF_50_Tcap (Hill EC50 for that pathway) removed after
% counterfactual SBI showed the data prefers the pathway off. See the
% block above for the diagnostic summary.

%% Tfh Module Parameters
% Tfh activation rate in LN (naive CD4 -> aTfh via cDC2 + antigen)
params.k_Tfh_act.Value = 10;
params.k_Tfh_act.Units = '1/day';
params.k_Tfh_act.Notes = 'Rate of naive CD4 activation into Tfh in LN (antigen + cDC2 gated)';
params.k_Tfh_act.derivedFrom = {};
% Tumor-local Th -> Tfh differentiation rate (antigen + IL-6 gated)
params.k_Tfh_diff_tumor.Value = 0.5;
params.k_Tfh_diff_tumor.Units = '1/day';
params.k_Tfh_diff_tumor.Notes = 'Rate of Th->Tfh in tumor (antigen + IL-6 Hill gated). ABM: PARAM_TFH_DIFF_K';
params.k_Tfh_diff_tumor.derivedFrom = {};
% IL-6 EC50 for Tfh differentiation
params.IL6_50_Tfh.Value = 0.001;
params.IL6_50_Tfh.Units = 'nanomolarity';
params.IL6_50_Tfh.Notes = 'IL-6 EC50 for Th->Tfh differentiation in tumor. ABM: PARAM_TFH_EC50_IL6';
params.IL6_50_Tfh.derivedFrom = {};

%% B Cell Module Parameters
% Naive B cell blood density (matches rho_nCD4/rho_nCD8 pattern)
params.rho_nB.Value = 2e5;
params.rho_nB.Units = 'cell/milliliter';
params.rho_nB.Notes = 'Naive B cell density in blood (~200 cells/uL, Blanco 2018 PMID: 29686403). Total B ~250 cells/uL, ~80% naive.';
params.rho_nB.derivedFrom = {};
% Initial conditions (density x volume, matching nCD4_C/nCD8_C pattern)
params.nB_C.Value = [];
params.nB_C.Units = 'cell';
params.nB_C.Notes = 'Naive B cells in blood at steady state = rho_nB * V_C';
params.nB_C.Factors = ["rho_nB", "V_C"];
params.nB_C.Equation = 'p(1)*p(2)';
params.nB_LN.Value = [];
params.nB_LN.Units = 'cell';
params.nB_LN.Notes = 'Naive B cells in LN at steady state (LN fraction ~4% of blood, matching nCD8_LN/nCD4_LN pattern)';
params.nB_LN.Factors = ["nB_C"];
params.nB_LN.Equation = 'p(1)*0.04';
% Marrow output of naive B cells (scaled to match nB_C steady state)
params.k_nB_in.Value = 1e4;
params.k_nB_in.Units = 'cell/day';
params.k_nB_in.Notes = 'Marrow output of naive B cells to blood (LeBien 2008, PMID: 18390730). Scaled to steady-state nB_C via rho_nB*V_C.';
params.k_nB_in.derivedFrom = {};
% LN activation rate
params.k_B_act.Value = 10;
params.k_B_act.Units = '1/day';
params.k_B_act.Notes = 'Naive B activation in LN (antigen + Tfh gated). Scaled from GC kinetics';
params.k_B_act.derivedFrom = {};
% Clonal expansion
params.n_B_clones.Value = 50;
params.n_B_clones.Units = 'dimensionless';
params.n_B_clones.Notes = 'Antigen-specific B cell clones (GC output diversity)';
params.n_B_clones.derivedFrom = {};
% Proliferation rate
params.k_B_pro.Value = 2.0;
params.k_B_pro.Units = '1/day';
params.k_B_pro.Notes = 'Activated B proliferation rate in LN (GC centroblast ~12h cycle)';
params.k_B_pro.derivedFrom = {};
% Mature B death rate
params.k_B_death.Value = 0.1;
params.k_B_death.Units = '1/day';
params.k_B_death.Notes = 'Mature B cell death rate (~10 day half-life). ABM: PARAM_BCELL_LIFE_MEAN derived as 1/(k_B_death*dt)';
params.k_B_death.derivedFrom = {};
% Plasma differentiation rate
params.k_B_to_plasma.Value = 0.5;
params.k_B_to_plasma.Units = '1/day';
params.k_B_to_plasma.Notes = 'Tumor-local aB -> plasma differentiation (Tfh-helped)';
params.k_B_to_plasma.derivedFrom = {};
% Plasma death rate
params.k_plasma_death.Value = 0.033;
params.k_plasma_death.Units = '1/day';
params.k_plasma_death.Notes = 'Plasma cell death rate (~30 day half-life for long-lived plasma). ABM: PARAM_BCELL_LIFE_MEAN_PLASMA';
params.k_plasma_death.derivedFrom = {};
% Tfh EC50 for LN B activation
params.Tfh_50_B.Value = 1e5;
params.Tfh_50_B.Units = 'cell';
params.Tfh_50_B.Notes = 'Half-maximal LN Tfh number for B cell activation help';
params.Tfh_50_B.derivedFrom = {};
% Tumor-local parameters
params.k_B_act_tumor.Value = 1.0;
params.k_B_act_tumor.Units = '1/day';
params.k_B_act_tumor.Notes = 'Tumor-local naive B activation rate (antigen-gated)';
params.k_B_act_tumor.derivedFrom = {};
params.k_B_pro_tumor.Value = 1.0;
params.k_B_pro_tumor.Units = '1/day';
params.k_B_pro_tumor.Notes = 'Tumor-local activated B proliferation rate (saturating form with rho_B_max cap)';
params.k_B_pro_tumor.derivedFrom = {};
% Carrying capacity for tumor-local B proliferation (analogous to rho_Th_max)
params.rho_B_max.Value = 5e6;
params.rho_B_max.Units = 'cell/milliliter';
params.rho_B_max.Notes = 'Maximum activated B cell density in tumor (carrying capacity for local proliferation). Analogous to rho_Th_max (5e6). Added to fix uncapped exponential B cell growth.';
params.rho_B_max.derivedFrom = {};
params.Tfh_50_plasma.Value = 1e6;
params.Tfh_50_plasma.Units = 'cell/milliliter';
params.Tfh_50_plasma.Notes = 'Half-maximal tumor Tfh DENSITY for aB -> plasma differentiation. Uses concentration (Tfh/V_T) not raw count.';
params.Tfh_50_plasma.derivedFrom = {};

%% B Cell Transport Parameters (own rates, replacing reused Treg/nCD4 params)
% Initialized to the same values as the T cell params they replace.
% Naive B death (was k_nTreg_death)
params.k_nB_death.Value = 0.002;
params.k_nB_death.Units = '1/day';
params.k_nB_death.Notes = 'Naive B cell death rate. Was shared with k_nTreg_death; now independent. (Marrow-output balanced)';
params.k_nB_death.derivedFrom = {};
% Naive B LN trafficking (was q_nCD4_LN_in/out)
params.q_nB_LN_in.Value = 0.1;
params.q_nB_LN_in.Units = '1/day';
params.q_nB_LN_in.Notes = 'Naive B cell entry into lymph node (was q_nCD4_LN_in). Initialized to nCD4 value.';
params.q_nB_LN_in.derivedFrom = {};
params.q_nB_LN_out.Value = 2.88;
params.q_nB_LN_out.Units = '1/day';
params.q_nB_LN_out.Notes = 'Naive B cell exit from lymph node (was q_nCD4_LN_out). Initialized to nCD4 value.';
params.q_nB_LN_out.derivedFrom = {};
% Mature B peripheral trafficking (was q_Treg_P_in/out)
params.q_B_P_in.Value = 0.25;
params.q_B_P_in.Units = '1/minute';
params.q_B_P_in.Notes = 'Mature B cell transport C->P (was q_Treg_P_in). Initialized to Treg value.';
params.q_B_P_in.derivedFrom = {};
params.q_B_P_out.Value = 24;
params.q_B_P_out.Units = '1/day';
params.q_B_P_out.Notes = 'Mature B cell transport P->C (was q_Treg_P_out). Initialized to Treg value.';
params.q_B_P_out.derivedFrom = {};
% Mature B LN egress (was q_Treg_LN_out)
params.q_B_LN_out.Value = 24;
params.q_B_LN_out.Units = '1/day';
params.q_B_LN_out.Notes = 'Mature B cell exit from lymph node (was q_Treg_LN_out). Initialized to Treg value.';
params.q_B_LN_out.derivedFrom = {};
% B cell tumor infiltration (was q_Treg_T_in). Direct value, initialized to match
% q_Treg_T_in = 5.8e-5 1/(cm^3*min). Independent for calibration.
params.q_B_T_in.Value = 5.8e-5;
params.q_B_T_in.Units = '1/(centimeter^3*minute)';
params.q_B_T_in.Notes = 'B cell transmigration into tumor (was q_Treg_T_in). Initialized to same value as Treg. B cells are larger and use different adhesion molecules, so calibrate independently.';
params.q_B_T_in.derivedFrom = {};

%% Breg Switching Parameters (tumor-only, TGFb + IL-10 driven)
% Forward switching rate (B -> Breg at maximal signal)
params.k_Breg_on.Value = 1.0;
params.k_Breg_on.Units = '1/day';
params.k_Breg_on.Notes = 'Forward B->Breg switching rate at maximal TGFb+IL10 signal. Breg is a transient functional state, not a lineage (Rosser 2015 PMID:25594691).';
params.k_Breg_on.derivedFrom = {};
% Reverse switching rate (Breg -> B reversion)
params.k_Breg_off.Value = 2.0;
params.k_Breg_off.Units = '1/day';
params.k_Breg_off.Notes = 'Reverse Breg->B reversion rate. Faster than k_Breg_on to ensure transient Breg state (equilibrium fraction < 50% at full signal). Epigenetic HDAC-dependent IL-10 locus (Lal 2015 PMID:25488983).';
params.k_Breg_off.derivedFrom = {};
% TGFb EC50 for Breg switching
params.TGFb_50_Breg.Value = 0.07;
params.TGFb_50_Breg.Units = 'nanomolarity';
params.TGFb_50_Breg.Notes = 'TGF-beta EC50 for Breg switching. Initialized to TGFb_50_M2pol (shared TGFb sensitivity scale).';
params.TGFb_50_Breg.derivedFrom = {};
% IL-10 EC50 for Breg switching
params.IL10_50_Breg.Value = 8.0;
params.IL10_50_Breg.Units = 'nanomolarity';
params.IL10_50_Breg.Notes = 'IL-10 EC50 for Breg switching. Initialized to IL10_50_M2 (shared IL-10 sensitivity scale). Autocrine IL-10 amplification loop.';
params.IL10_50_Breg.derivedFrom = {};
% Breg IL-10 secretion rate
params.k_IL10_sec_Breg.Value = 1.5e-12;
params.k_IL10_sec_Breg.Units = 'nanomole/(cell*day)';
params.k_IL10_sec_Breg.Notes = 'Breg IL-10 secretion rate. Initialized to k_IL10_sec_Treg magnitude (similar per-cell IL-10 output). ABM: PARAM_BCELL_IL10_RELEASE_BREG';
params.k_IL10_sec_Breg.derivedFrom = {};

%% Antibody / ADCC Parameters
% Plasma cell secretion rate
params.k_Ab_sec.Value = 1e-5;
params.k_Ab_sec.Units = 'nanomole/cell/day';
params.k_Ab_sec.Notes = 'Plasma cell IgG secretion (~100 pg/cell/day, Hibi 1986 PMID: 3489799). ABM: PARAM_BCELL_ANTIBODY_RELEASE';
params.k_Ab_sec.derivedFrom = {};
% Antibody degradation
params.k_Ab_deg.Value = 0.033;
params.k_Ab_deg.Units = '1/day';
params.k_Ab_deg.Notes = 'Tumor-local IgG clearance (~21 day serum half-life, shorter in tumor due to FcRn recycling inefficiency)';
params.k_Ab_deg.derivedFrom = {};
% ADCC EC50
params.Ab_50_ADCC.Value = 0.01;
params.Ab_50_ADCC.Units = 'nanomolarity';
params.Ab_50_ADCC.Notes = 'Half-maximal antibody for ADCC macrophage killing boost. ABM: PARAM_ADCC_AB_EC50';
params.Ab_50_ADCC.derivedFrom = {};
% ADCC boost max
params.ADCC_boost_max.Value = 2.0;
params.ADCC_boost_max.Units = 'dimensionless';
params.ADCC_boost_max.Notes = 'Max fold-increase in M1 phagocytosis from ADCC (1x baseline -> 3x at saturation). ABM: PARAM_ADCC_BOOST_MAX';
params.ADCC_boost_max.derivedFrom = {};
% DC immune-complex uptake EC50
params.Ab_50_DC.Value = 29.3;
params.Ab_50_DC.Units = 'nanomolarity';
params.Ab_50_DC.Notes = 'Half-maximal antibody for DC immune-complex uptake boost. ABM: PARAM_DC_ANTIBODY_50';
params.Ab_50_DC.derivedFrom = {};
% DC immune-complex boost max
params.DC_Ab_boost.Value = 3.95;
params.DC_Ab_boost.Units = 'dimensionless';
params.DC_Ab_boost.Notes = 'Max fold-increase in DC antigen uptake from immune complexes. ABM: PARAM_DC_ANTIBODY_BOOST';
params.DC_Ab_boost.derivedFrom = {};
