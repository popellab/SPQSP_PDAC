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
params.k_C1_growth.Value = 0.010;
params.k_C1_growth.Units = '1/day';
params.k_C1_growth.Notes = '(Ryu 2014, PMID: 24895040; Desai 2006, PMID: 16489089)';
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
params.k_CD8_death.Value = 0.1;
params.k_CD8_death.Units = '1/day';
params.k_CD8_death.Notes = '(De Boer and Perelson 1995, PMID: 7475092)';
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
params.rho_CD8_max.Value = 1e7;
params.rho_CD8_max.Units = 'cell/milliliter';
params.rho_CD8_max.Notes = 'PLACEHOLDER: max CD8 density in tumor (carrying cap). Set equal to rho_Treg_max as default; not separately literature-anchored.';
params.rho_CD8_max.derivedFrom = {};

% === Mechanistic T Cell Activation Parameters ===
% Rate of naive T cell activation per mature APC
params.k_T_activation_per_APC.Value =  1e-6;  % Calibrated: 5× from baseline (1e-7)
params.k_T_activation_per_APC.Units = '1/(day*cell)';
params.k_T_activation_per_APC.Notes = 'Rate of naive T cell activation per mature APC in lymph node (encounter frequency × activation probability). Calibrated for PDAC (5× increase from baseline for enhanced activation efficiency)';
params.k_T_activation_per_APC.derivedFrom = 'Miller et al. 2004 (PMID: 15240696) - T cell scanning rate ~500-5000 APCs/hour';
% Clonal expansion burst size (= fold expansion per activated naive T cell)
% LITERATURE MAPPING: Studies report "X-fold expansion" or "Y doublings" (fold = 2^Y)
% This parameter IS the fold expansion: 1 naive T cell → n_T_expansion_burst effector cells
% Example: 10 doublings = 2^10 = 1024-fold ≈ n_T_expansion_burst of 1000
% Typical range for viral/tumor antigens: 100-10,000 fold (7-14 doublings)
params.n_T_expansion_burst.Value = 800;  % ~10 doublings (2^10 ≈ 1000)
params.n_T_expansion_burst.Units = 'dimensionless';
params.n_T_expansion_burst.Notes = 'Fold expansion: effector cells produced per activated naive T cell. 800 = ~9-10 doublings.';
params.n_T_expansion_burst.derivedFrom = 'Marzo 2005 (PMID: 15746364); Kaech 2001 (PMID: 11357157) - range 100-1000 for tumor antigens';

% === Mechanistic Tumor Killing Parameters ===
% T cell motility in tumor (baseline measurement)
params.v_T_motility.Value = 10;
params.v_T_motility.Units = 'micrometer/minute';
params.v_T_motility.Notes = 'Baseline T cell motility speed in tumor tissue (from intravital microscopy)';
params.v_T_motility.derivedFrom = 'Boissonnas et al. 2007 (PMID: 17220890) - range 5-15 μm/min';
% Volume searched by one T cell per day
params.v_T_search_volume.Value = 1e-6;  % Calibrated: 10× from baseline (1e-6)
params.v_T_search_volume.Units = 'millimeter^3/day';
params.v_T_search_volume.Notes = 'Volume of tumor tissue searched by one T cell per day (derived from motility and persistence time). Calibrated for PDAC (10× increase for enhanced search efficiency - accounting for chemokine-guided migration)';
params.v_T_search_volume.derivedFrom = 'Boissonnas et al. 2007 (PMID: 17220890) - T cell motility ~10 μm/min';
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
params.q_CD8_P_in.Value = [];
params.q_CD8_P_in.Units = '1/minute';
params.q_CD8_P_in.Notes = ['calculated based on T cell transmigration rate ' params.k_CD8_mig.Notes ' and T cell adhesion density ' params.rho_adh.Notes];
params.q_CD8_P_in.Factors = ["k_CD8_mig", "rho_adh", "gamma_P", "V_P"];
params.q_CD8_P_in.Equation = 'p(1)*p(2)*p(3)*p(4)';
% Activated CD4+ T Cell Transport C->P
params.q_CD4_P_in.Value = [];
params.q_CD4_P_in.Units = '1/minute';
params.q_CD4_P_in.Notes = ['calculated based on T cell transmigration rate ' params.k_CD4_mig.Notes ' and T cell adhesion density ' params.rho_adh.Notes];
params.q_CD4_P_in.Factors = ["k_CD4_mig", "rho_adh", "gamma_P", "V_P"];
params.q_CD4_P_in.Equation = 'p(1)*p(2)*p(3)*p(4)';
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
% IL2 Concentration for Half-Maximal Treg Proliferation
params.IL2_50_Treg.Value = 0.01; % 0.0032
params.IL2_50_Treg.Units = 'nanomolarity';
params.IL2_50_Treg.Notes = '(Wang and Smith 1987, PMID: 3116143)';
% Baseline Number of Activated T Cell Generations
params.N_div_base.Value = 2;
params.N_div_base.Units = 'dimensionless';
params.N_div_base.Notes = '(Marchingo 2014, PMID: 25430770)';
% Baseline Number of Activated T Cell Generations for co-stimulation
params.N_div_costim.Value = 3;
params.N_div_costim.Units = 'dimensionless';
params.N_div_costim.Notes = '(Marchingo 2014, PMID: 25430770)';
% Additional Number of Activated CD8+ T Cell Generations Due to IL2
params.N_IL2_CD8.Value = 11;
params.N_IL2_CD8.Units = 'dimensionless';
params.N_IL2_CD8.Notes = '(Marchingo 2014, PMID: 25430770)';
% Additional Number of Activated CD4+ T Cell Generations Due to IL2
params.N_IL2_CD4.Value = 8.5;
params.N_IL2_CD4.Units = 'dimensionless';
params.N_IL2_CD4.Notes = '(Marchingo 2014, PMID: 25430770)';

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
params.cDC1_recruit_target_T.Value = 5e4;   % = 0.2 * 2.7e6 (legacy default; σ widened in priors CSV)
params.cDC1_recruit_target_T.Units = 'cell/milliliter';
params.cDC1_recruit_target_T.Notes = 'cDC1 recruitment-target density in tumor (recruit-rate / k_APC_death numerator); calibrated via realized-density observable. Lavin 2017 PMID: 28475900, Hegde 2020 PMID: 31932165';
params.cDC1_recruit_target_T.derivedFrom = {'Lavin 2017 PMID: 28475900', 'Hegde 2020 PMID: 31932165'};
params.cDC2_recruit_target_T.Value = 2e5;  % = 0.8 * 2.7e6
params.cDC2_recruit_target_T.Units = 'cell/milliliter';
params.cDC2_recruit_target_T.Notes = 'cDC2 recruitment-target density in tumor (recruit-rate / k_APC_death numerator); calibrated via realized-density observable. Lavin 2017 PMID: 28475900';
params.cDC2_recruit_target_T.derivedFrom = {'Lavin 2017 PMID: 28475900'};
params.APC0_cDC1_LN.Value = 2.4e5;  % = 0.2 * 1.2e6
params.APC0_cDC1_LN.Units = 'cell/milliliter';
params.APC0_cDC1_LN.Notes = 'cDC1 homeostatic density in LN (Catron 2004 PMID: 15357945)';
params.APC0_cDC1_LN.derivedFrom = {'Catron 2004 PMID: 15357945'};
params.APC0_cDC2_LN.Value = 9.6e5;  % = 0.8 * 1.2e6
params.APC0_cDC2_LN.Units = 'cell/milliliter';
params.APC0_cDC2_LN.Notes = 'cDC2 homeostatic density in LN (Catron 2004 PMID: 15357945)';
params.APC0_cDC2_LN.derivedFrom = {'Catron 2004 PMID: 15357945'};
% Half-maximal tumor death rate for DAMP-mediated APC maturation
% At baseline: ~1e6 cells * 0.01/day death rate = 1e4 cells/day
% DAMP_50 set to baseline death rate so DAMPs provide moderate signal at steady state
params.DAMP_50.Value = 1e4;
params.DAMP_50.Units = 'cell/day';
params.DAMP_50.Notes = 'Estimated from baseline PDAC tumor death rate';
params.DAMP_50.derivedFrom = {};
% Half-maximal TGFb for inhibition of APC maturation
% Same sensitivity as macrophage polarization (TGFb_50 = 0.07 nM)
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
% PDL1-aPDL1 kd
params.kd_PDL1_aPDL1.Value = 0.4;
params.kd_PDL1_aPDL1.Units = 'nanomolarity';
params.kd_PDL1_aPDL1.Notes = '(Wang 2014, PMID: 24872026)';
% PDL1-aPDL1 kon
params.kon_PDL1_aPDL1.Value = 4.3e5;
params.kon_PDL1_aPDL1.Units = '1/(molarity*second)';
params.kon_PDL1_aPDL1.Notes = '(Wang 2014, PMID: 24872026)';
% PDL1-aPDL1 Chi (antibody cross-arm binding strength)
params.Chi_PDL1_aPDL1_3D.Value = 100;
params.Chi_PDL1_aPDL1_3D.Units = 'dimensionless';
params.Chi_PDL1_aPDL1_3D.Notes = '(estimated)';
% PDL1-aPDL1 Chi for 2D
params.Chi_PDL1_aPDL1.Value = [];
params.Chi_PDL1_aPDL1.Units = '';
params.Chi_PDL1_aPDL1.Notes = '(Wang 2014, PMID: 24872026)';
params.Chi_PDL1_aPDL1.Factors = ["Chi_PDL1_aPDL1_3D","d_syn","N_avg"];
params.Chi_PDL1_aPDL1.Equation = 'p(1)/(p(2)*p(3))';

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
% PDL1-aPDL1 koff
params.koff_PDL1_aPDL1.Value = [];
params.koff_PDL1_aPDL1.Units = '1/second';
params.koff_PDL1_aPDL1.Notes = ['calculated based on the measured kd and kon ' params.kd_PDL1_aPDL1.Notes];
params.koff_PDL1_aPDL1.Factors = ["kon_PDL1_aPDL1","kd_PDL1_aPDL1"];
params.koff_PDL1_aPDL1.Equation = 'p(1)*p(2)';

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
%     - Yu 2017 (PMID: 29378617, n=373 PDAC):
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
params.C_PDL1.Notes = '(Zhang 2022, PMID: 35037417; Yu 2017, PMID: 29378617) — PDAC cohort-averaged baseline; per-PDL1+-cell density estimated';
% Average Baseline PDL1 Expression on mAPCs in TDLN.
% Decoupled from C_PDL1 because mature DCs upregulate PD-L1 uniformly (every
% mDC is positive — no cohort-mean discount applies). No PDAC-TDLN-specific
% quantification available; retain the legacy ~8e4 per-cell estimate.
params.APC_PDL1.Value = 8e4;
params.APC_PDL1.Units = 'molecule';
params.APC_PDL1.Notes = 'Estimated per-mDC density; mature DCs in TDLN uniformly express PD-L1 (no PDAC-specific quantitative anchor available)';
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
% CTLA4-aCTLA4 kd
params.kd_CTLA4_aCTLA4.Value = 18.2;
params.kd_CTLA4_aCTLA4.Units = 'nanomolarity';
params.kd_CTLA4_aCTLA4.Notes = '(Wang 2014, PMID: 24872026)';
% CTLA4-aCTLA4 kon
params.kon_CTLA4_aCTLA4.Value = 3.83e5;
params.kon_CTLA4_aCTLA4.Units = '1/(molarity*second)';
params.kon_CTLA4_aCTLA4.Notes = '(Wang 2014, PMID: 24872026)';
% CTLA4-aCTLA4 Chi (antibody cross-arm binding strength)
params.Chi_CTLA4_aCTLA4_3D.Value = 100;
params.Chi_CTLA4_aCTLA4_3D.Units = 'dimensionless';
params.Chi_CTLA4_aCTLA4_3D.Notes = '(estimated)';
% CTLA4-aCTLA4 Chi for 2D
params.Chi_CTLA4_aCTLA4.Value = [];
params.Chi_CTLA4_aCTLA4.Units = '';
params.Chi_CTLA4_aCTLA4.Notes = '(estimated)';
params.Chi_CTLA4_aCTLA4.Factors = ["Chi_CTLA4_aCTLA4_3D","d_syn","N_avg"];
params.Chi_CTLA4_aCTLA4.Equation = 'p(1)/(p(2)*p(3))';

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
% CTLA4-aCTLA4 koff
params.koff_CTLA4_aCTLA4.Value = [];
params.koff_CTLA4_aCTLA4.Units = '1/second';
params.koff_CTLA4_aCTLA4.Notes = ['calculated based on the measured kd and kon ' params.kd_CTLA4_aCTLA4.Notes];
params.koff_CTLA4_aCTLA4.Factors = ["kon_CTLA4_aCTLA4","kd_CTLA4_aCTLA4"];
params.koff_CTLA4_aCTLA4.Equation = 'p(1)*p(2)';

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
% half-maximal entinostat concentration for anti-proliferative effect on tumor cells
params.IC50_inostat_C.Value = 3.74e-7;
params.IC50_inostat_C.Units = 'molarity';
params.IC50_inostat_C.Notes = '(Bouchain 2002, PMID: 12593661; Lee 2001, PMID: 11221885)';

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
% half-maximal entinostat concentration for NO inhibition
params.IC50_inostat_NO.Value = 0.56; % <1 nM
params.IC50_inostat_NO.Units = 'nanomolarity';
params.IC50_inostat_NO.Notes = '(Choo 2010, PMID: 20421217)';
% half-maximal entinostat concentration for CCL2 inhibition
params.IC50_inostat_CCL2.Value = 1.2; % <2 nM
params.IC50_inostat_CCL2.Units = 'nanomolarity';
params.IC50_inostat_CCL2.Notes = '(Choo 2013, PMID: 24241152)';
% half-maximal entinostat concentration for Arg I inhibition
params.IC50_inostat_ArgI.Value = 500;
params.IC50_inostat_ArgI.Units = 'nanomolarity';
params.IC50_inostat_ArgI.Notes = '(Orillian 2017, PMID: 28698201)';

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

%% Tfh + B cell module parameters (coupled QSP+ABM completion)
% Tfh activation rate (mirrors k_Th_act)
params.k_Tfh_act.Value = 10;
params.k_Tfh_act.Units = '1/day';
params.k_Tfh_act.Notes = 'Tfh activation; mirrors k_Th_act (Robertson-Tessi 2012, PMID: 22051568)';
% Naive B initial counts (relax to marrow-source/death steady state during presim)
params.nB_C.Value = 1e8;
params.nB_C.Units = 'cell';
params.nB_C.Notes = 'Naive B in blood, initial seed (order-of-magnitude; relaxes in presim)';
params.nB_LN.Value = 1e7;
params.nB_LN.Units = 'cell';
params.nB_LN.Notes = 'Naive B in LN, initial seed (order-of-magnitude)';
% Marrow output of naive B cells (mirrors thymic-output magnitude)
params.k_nB_in.Value = 1e6;
params.k_nB_in.Units = 'cell/day';
params.k_nB_in.Notes = 'Bone-marrow naive B output; order-of-magnitude from Q_nCD4_thym';
% Naive B activation rate
params.k_B_act.Value = 5;
params.k_B_act.Units = '1/day';
params.k_B_act.Notes = 'Naive B activation (antigen + Tfh help); mirrors k_nCD4_act';
% B cell clonal diversity (tumor antigen)
params.n_B_clones.Value = 120;
params.n_B_clones.Units = 'dimensionless';
params.n_B_clones.Notes = 'Antigen-specific B clones; mirrors n_clones_tum';
% Activated B proliferation
params.k_B_pro.Value = 1.0;
params.k_B_pro.Units = '1/day';
params.k_B_pro.Notes = 'Activated B proliferation; mirrors k_CD4_pro';
% Mature B death
params.k_B_death.Value = 0.1;
params.k_B_death.Units = '1/day';
params.k_B_death.Notes = 'Mature B turnover; mirrors k_CD4_death';
% B -> plasma differentiation (tumor-local)
params.k_B_to_plasma.Value = 0.5;
params.k_B_to_plasma.Units = '1/day';
params.k_B_to_plasma.Notes = 'Tumor-local B to plasma differentiation (order-of-magnitude)';
% Plasma cell death
params.k_plasma_death.Value = 0.1;
params.k_plasma_death.Units = '1/day';
params.k_plasma_death.Notes = 'Plasma cell death (order-of-magnitude)';
% Half-maximal LN Tfh number for B help
params.Tfh_50_B.Value = 1e5;
params.Tfh_50_B.Units = 'cell';
params.Tfh_50_B.Notes = 'Half-maximal LN Tfh number for B-cell help (order-of-magnitude)';

%% T Helper Cell Module parameters
% T helper cell activation rate
params.k_Th_act.Value = 10;
params.k_Th_act.Units = '1/day';
params.k_Th_act.Notes = '(Robertson-Tessi 2012, PMID: 22051568)';
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
% Half-Maximal TGFb level for differentiation of Th to Treg /
% chemoresistance development / M1-to-M2 polarization / fibroblast activation
params.TGFb_50.Value = 0.07;
params.TGFb_50.Units = 'nanomolarity';
params.TGFb_50.Notes = '(Robertson-Tessi 2012, PMID: 22051568)';
% Half-Maximal TGFb level for CD8+ T cell inhibition
params.TGFb_50_Teff.Value = 0.14;
params.TGFb_50_Teff.Units = 'nanomolarity';
params.TGFb_50_Teff.Notes = '(Robertson-Tessi 2012, PMID: 22051568)';
% Half-Maximal TGFb level for CD4 helper T cell exhaustion (PD-1+, dysfunction)
% — separate EC50 from CD8 (TGFb_50_Teff) and from Treg differentiation (TGFb_50)
% to allow submodel target on Th-side exhaustion to constrain Th biology
% specifically. Default mirrors TGFb_50_Teff; literature anchor pending via
% submodel target extraction (TGFβ + chronic TCR → CD4 dysfunction kinetics).
params.TGFb_50_Th_exh.Value = 0.14;
params.TGFb_50_Th_exh.Units = 'nanomolarity';
params.TGFb_50_Th_exh.Notes = 'Estimated; mirrors TGFb_50_Teff pending submodel target anchoring';
params.TGFb_50_Th_exh.derivedFrom = {'estimated'};
% Half-Maximal cancer cell number for T cell recruitment
params.K_C2_recruit.Value = 2.02e7;
params.K_C2_recruit.Units = 'cell^2';
params.K_C2_recruit.Notes = '(Phillis 2006, PMID: 16153659)';
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
% TGFb_50_NK, TGFb_50_CAF_act, TGFb_50_col_sec, TGFb_50_IL1R1) is calibrated
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

%% Paclitaxel Module Parameters
% Max clearance rate from central compartment
params.Vmcl.Value = 8070;
params.Vmcl.Units = 'microgram/hour';
params.Vmcl.Notes = '(Chen 2014, PMID: 24719309)';
% Half-max conc. of clearance rate from central compartment
params.Kcl.Value = 40.2;
params.Kcl.Units = 'microgram/liter';
params.Kcl.Notes = '(Chen 2014, PMID: 24719309)';
% Transport rate b/t central and P compartment
params.Q2.Value = 41.6;
params.Q2.Units = 'liter/hour';
params.Q2.Notes = '(Chen 2014, PMID: 24719309)';
% Max clearance rate from central compartment
params.Vmt.Value = 325000;
params.Vmt.Units = 'microgram/hour';
params.Vmt.Notes = '(Chen 2014, PMID: 24719309)';
% Half-max conc. of clearance rate from central compartment
params.Kt.Value = 4260;
params.Kt.Units = 'microgram/liter';
params.Kt.Notes = '(Chen 2014, PMID: 24719309)';
% Body surface area
params.BSA.Value = 1.9;
params.BSA.Units = 'meter^2';
params.BSA.Notes = '(Chen 2014, PMID: 24719309)';

%% Paclitaxel PD Parameters
% Tumour to plasma concentration ratio of nab-paclitaxel
params.r_nabp.Value = 1.5;
params.r_nabp.Units = 'dimensionless';
params.r_nabp.Notes = '(Yang 2007, PMID: 17828616; Rajeshkumar 2016, PMID: 27441498)';
% Cancer cell killing rate by nab-paclitaxel
params.k_C_nabp.Value = 0.06;
params.k_C_nabp.Units = '1/hour';
params.k_C_nabp.Notes = '(Choi 2012, PMID: 23023313; Volk 2008, PMID: 18516298; Kozłowska 2018, PMID: 29769198)';
% Half-maximal nab-paclitaxel concentration for cancer cell killing
params.IC50_nabp.Value = 92;
params.IC50_nabp.Units = 'nanomolarity';
params.IC50_nabp.Notes = '(Yang 2013, PMID: 23180760)';
% Molecular weight of nab-paclitaxel
params.MW_nabp.Value = 853.9;
params.MW_nabp.Units = 'gram/mole';
params.MW_nabp.Notes = '(calculated)';
% Limited tumor penetration of nab-paclitaxel
params.Kc_nabp.Value = 8e7;
params.Kc_nabp.Units = 'cell';
params.Kc_nabp.Notes = '(estimated based on Kyle 2007, PMID: 17473214)';
% Rate of drug-selected chemo-resistance development (C1 -> C2 transition).
% Generalized from the old nab-paclitaxel-only k_C_resist_nab: the transition is
% now driven by the aggregate chemo drug effect E_chemo (all 5 PASS-01 drugs),
% so resistance is SELECTED by ongoing cytotoxic pressure rather than tied to a
% single agent or the TGFb tone. Calibrated (free) in longitudinal burden fits.
params.k_resist.Value = 1e-4;
params.k_resist.Units = '1/day';
params.k_resist.Notes = 'Drug-selected C1->C2 resistance-acquisition rate per unit E_chemo (estimated)';
% Number of folds increase of cytotoxic IC50 in the resistant clone (cross-drug).
params.r_resist.Value = 100;
params.r_resist.Units = 'dimensionless';
params.r_resist.Notes = '(Nemcova-Furstova 2016, PMID: 27664577)';

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
% Original HCC value: 4e-5 (Xiang 2014). Increased for PDAC where desmoplastic
% stroma compresses vessels, increasing vascular regression (Provenzano 2012, Jacobetz 2013).
params.k_vas_decay.Value =  0.004;
params.k_vas_decay.Units = '1/day';
params.k_vas_decay.Notes = 'Increased 10x from HCC value for PDAC stromal vessel compression (Provenzano 2012, Jacobetz 2013).';
% Secretion rate of angiogenic factors induced by nab-paclitaxel
params.k_vas_nabp.Value = 2.8e-4;
params.k_vas_nabp.Units = 'picogram/cell/day';
params.k_vas_nabp.Notes = '(Volk 2008, PMID: 18516298)';
% Half-maximal conc. of nab-paclitaxel on angiogenic factor induction
params.IC50_nabp_vas.Value = 5.2;
params.IC50_nabp_vas.Units = 'nanomolarity';
params.IC50_nabp_vas.Notes = '(Volk 2008, PMID: 18516298)';
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
% Inhibition rate of maximal tumor capacity by nab-paclitaxel
params.k_K_nabp.Value = 17.8;
params.k_K_nabp.Units = '1/hour/molarity';
params.k_K_nabp.Notes = '(Mollard 2017, PMID: 28416742)';


%% Properties for CTLA4-mediated Treg depletion (ADCC)
% CTLA4-Ipi Concentration for Half-Maximal Treg death
params.Treg_CTLA4_50.Value = 1000;
params.Treg_CTLA4_50.Units = 'molecule';
params.Treg_CTLA4_50.Notes = '(estimated)';
% Hill Coefficient for CTLA4-Ipi
params.n_Treg_CTLA4.Value = 2;
params.n_Treg_CTLA4.Units = 'dimensionless';
params.n_Treg_CTLA4.Notes = '(estimated)';
% CTLA4 Expression on Tregs
params.Treg_CTLA4_tot.Value = 5000;
params.Treg_CTLA4_tot.Units = 'molecule';
params.Treg_CTLA4_tot.Notes = '(Jansson 2005, PMID: 16034096)';
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
% Degradation rate of IL10
params.k_IL10_deg.Value = 4;
params.k_IL10_deg.Units = '1/day';
params.k_IL10_deg.Notes = '(Saxena 2015, PMID: 25481648)';
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
params.k_M1_pol.Value = 0.045;
params.k_M1_pol.Units = '1/day';
params.k_M1_pol.Notes = '(Max rate under saturating pro-inflammatory stimulus; slower than M2 due to PDAC microenvironment)';
% Half-maximal IL10 level for M1 to M2 polarization / maintaining Treg function / mAPC inhibition (*STAT6 related)
params.IL10_50.Value = 8;
params.IL10_50.Units = 'picomolarity';
params.IL10_50.Notes = '(Altan-Bonnet 2019, PMID: 30770905)';
% Half-maximal IL13 level for M1 to M2 polarization
params.IL13_50.Value = 120;
params.IL13_50.Units = 'picomolarity';
params.IL13_50.Notes = '(Altan-Bonnet 2019, PMID: 30770905)';
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
% Dependence of Teff killing rate on Teff/C ratio
params.K_T_C.Value = 1.2;
params.K_T_C.Units = 'dimensionless';
params.K_T_C.Notes = '(Robertson-Tessi 2012, PMID: 22051568)';
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

% Binding affinity of CD47-aCD47
params.kd_CD47_aCD47.Value = 1; % 10-1000 nM PMID: 30133535
params.kd_CD47_aCD47.Units = 'nanomolarity';
params.kd_CD47_aCD47.Notes = '(Jalil 2020, PMID: 32421049)';
% Dissociation rate of CD47-aCD47
params.koff_CD47_aCD47.Value = 1/20; % 1e-4 - 1e-3 1/s
params.koff_CD47_aCD47.Units = '1/minute';
params.koff_CD47_aCD47.Notes = '(Kauder 2018, PMID: 30133535)';
% Association rate of CD47-aCD47
params.kon_CD47_aCD47.Value = [];
params.kon_CD47_aCD47.Units = '';
params.kon_CD47_aCD47.Notes = '(Kauder 2018, PMID: 30133535)';
params.kon_CD47_aCD47.Factors = ["koff_CD47_aCD47","kd_CD47_aCD47"];
params.kon_CD47_aCD47.Equation = 'p(1)/p(2)';

% CD47-aCD47 Chi (antibody cross-arm binding strength)
params.Chi_CD47_aCD47_3D.Value = 100;
params.Chi_CD47_aCD47_3D.Units = 'dimensionless';
params.Chi_CD47_aCD47_3D.Notes = '(estimated)';
% CD47-aCD47 Chi for 2D
params.Chi_CD47_aCD47.Value = [];
params.Chi_CD47_aCD47.Units = '';
params.Chi_CD47_aCD47.Notes = '(estimated)';
params.Chi_CD47_aCD47.Factors = ["Chi_CD47_aCD47_3D","d_syn","N_avg"];
params.Chi_CD47_aCD47.Equation = 'p(1)/(p(2)*p(3))';
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

%% Cabozantinib module
% IC50 for Cabozantinib_MET Complex
params.IC50_MET.Value = 1.3;
params.IC50_MET.Units = "nanomolarity";
params.IC50_MET.Notes = "Reference: doi: 10.1177/1756287217748867";
% IC50 for Cabozantinib_RET Complex
params.IC50_RET.Value = 5.2;
params.IC50_RET.Units = "nanomolarity";
params.IC50_RET.Notes = "Reference: doi: 10.1177/1756287217748867";
% IC50 for Cabozantinib_AXL Complex
params.IC50_AXL.Value = 7;
params.IC50_AXL.Units = "nanomolarity";
params.IC50_AXL.Notes = "Reference: doi: 10.1177/1756287217748867";
% IC50 for Cabozantinib_VEGFR2 Complex
params.IC50_VEGFR2.Value = 0.035;
params.IC50_VEGFR2.Units = "nanomolarity";
params.IC50_VEGFR2.Notes = "Reference: doi: 10.1177/1756287217748867";

% Effect Level for Cancer Cell Growth Inhibition
params.lambda_C_cabo.Value = 0.37;
params.lambda_C_cabo.Units = "dimensionless";
params.lambda_C_cabo.Notes = "fitted using Xiang 2014, PMID: 24700742 DOI: 10.1158/1078-0432.CCR-13-2620";

% Effect Level for Enhanced Vascular Transport
params.lambda_q_cabo.Value = 0.3;
params.lambda_q_cabo.Units = "dimensionless";
params.lambda_q_cabo.Notes = "estimated";

% Transmission rate of cabozantinib through gastro-intestinal tract to absorption site 2
params.k_GI_cabozantinib.Value = 1.0965;
params.k_GI_cabozantinib.Units = "1/hour";
params.k_GI_cabozantinib.Notes = "PMID: 29687244";
% Absorption rate of cabozantinib from site 1 to central compartment
params.k_a1_cabozantinib.Value = 0.568;
params.k_a1_cabozantinib.Units = "1/hour";
params.k_a1_cabozantinib.Notes = "source: https://www.accessdata.fda.gov/drugsatfda_docs/nda/2016/208692Orig1s000ClinPharmR.pdf";
% Absorption rate of cabozantinib from site 2 to central compartment
params.k_a2_cabozantinib.Value = 0.102;
params.k_a2_cabozantinib.Units = "1/hour";
params.k_a2_cabozantinib.Notes = "source: https://www.accessdata.fda.gov/drugsatfda_docs/nda/2016/208692Orig1s000ClinPharmR.pdf";
% Non-linear clearance rate of cabozantinib from central compartment
params.k_cln_cabozantinib.Value = 137.889;
params.k_cln_cabozantinib.Units = "molarity/hour";
params.k_cln_cabozantinib.Notes = "PMID: 29687244,PMID: 27139820 (estimated)";
% Half-maximal concentration of cabozantinib in central compartment for nonlinear clearance
params.Kc_cabozantinib.Value = 77.003;
params.Kc_cabozantinib.Units = "molarity";
params.Kc_cabozantinib.Notes = "estimated";
% Lag time of cabozantinib for fast absorption into central compartment
params.lagP1_cabozantinib.Value = 0.476;
params.lagP1_cabozantinib.Units = "hour";
params.lagP1_cabozantinib.Notes = "https://www.accessdata.fda.gov/drugsatfda_docs/nda/2016/208692Orig1s000ClinPharmR.pdf";
% Lag time of cabozantinib for slow absorption into central compartment
params.lagP2_cabozantinib.Value = 18.4;
params.lagP2_cabozantinib.Units = "hour";
params.lagP2_cabozantinib.Notes = "https://www.accessdata.fda.gov/drugsatfda_docs/nda/2016/208692Orig1s000ClinPharmR.pdf";
% Bioavailability of cabozantinib
params.F_cabozantinib.Value = 0.0232;
params.F_cabozantinib.Units = "dimensionless";
params.F_cabozantinib.Notes = "Estimated";
% Rate of diffusive transport of cabozantinib to the peripheral compartment
params.q_P_cabozantinib.Value = 5.77e-2;
params.q_P_cabozantinib.Units = "1/second";
params.q_P_cabozantinib.Notes = "PMID: 29687244,PMID: 27139820 (estimated)";
% Rate of diffusive transport of cabozantinib to the tumor compartment
params.q_T_cabozantinib.Value = 5.321e-1;
params.q_T_cabozantinib.Units = "1/second";
params.q_T_cabozantinib.Notes = "PMID: 29687244,PMID: 27139820 (estimated)";
% Rate of chemo-resistance development to cabozantinib
params.k_C_resist.Value = 1e-4;
params.k_C_resist.Units = '1/day';
params.k_C_resist.Notes = 'Rate of chemo-resistance development to cabozantinib (estimated)';
% Inhibition rate of maximal tumor capacity by cabozantinib
params.k_K_cabo.Value = 2.55;
params.k_K_cabo.Units = 'dimensionless';
params.k_K_cabo.Notes = 'fitted using Xiang 2014, PMID: 24700742 DOI: 10.1158/1078-0432.CCR-13-2620';

% Rate of diffusive transport of cabozantinib to the Lymph node compartment
params.q_LN_cabozantinib.Value = 9.91e-2;
params.q_LN_cabozantinib.Units = "1/second";
params.q_LN_cabozantinib.Notes = "PMID: 29687244,PMID: 27139820 (estimated)";
% Rate of lymphatic drainage of cabozantinib from TDLN to central compartment
params.q_LD_cabozantinib.Value = 1e-3;
params.q_LD_cabozantinib.Units = "1/second";
params.q_LD_cabozantinib.Notes = "PMID: 29687244,PMID: 27139820 (estimated)";
% Volume fraction of interstitial space available to cabozantinib in central compartment
params.gamma_C_cabozantinib.Value = 0.4729;
params.gamma_C_cabozantinib.Units = "dimensionless";
params.gamma_C_cabozantinib.Notes = "Estimated";
% Peripheral volume fraction for cabozantinib
params.gamma_P_cabozantinib.Value = 6.2037e-5;
params.gamma_P_cabozantinib.Units = "dimensionless";
params.gamma_P_cabozantinib.Notes = "Estimated";
% Tumor volume fraction for cabozantinib
%params.gamma_T_cabozantinib.Value = 0.2448;
params.gamma_T_cabozantinib.Value = 0.002448;
params.gamma_T_cabozantinib.Units = "dimensionless";
params.gamma_T_cabozantinib.Notes = "PMID: 29687244,PMID: 27139820 (estimated)";
% Rate of diffusive transport of cabozantinib to the tumor compartment
params.gamma_LN_cabozantinib.Value = 0.0024;
params.gamma_LN_cabozantinib.Units = "dimensionless";
params.gamma_LN_cabozantinib.Notes = "Estimated";
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
params.k_col_myCAF_sec.Value = 5e-8;
params.k_col_myCAF_sec.Units = 'milligram/(cell*day)';
params.k_col_myCAF_sec.Notes = "Collagen secretion by myCAFs (~5 pg/cell/day). Literature range 1.7-9.3 pg/cell/day.";
params.k_col_myCAF_sec.derivedFrom = {};
% Basal fraction of collagen secretion (TGFb-independent)
params.f_col_basal.Value = 0.15;
params.f_col_basal.Units = 'dimensionless';
params.f_col_basal.Notes = "Fraction of myCAF collagen secretion that is TGFb-independent. Range 0.4-0.65.";
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
params.k_MMP_M2.Notes = "M2 macrophage MMP/MR-mediated collagen degradation. M2-like TAMs are the dominant collagen degraders via mannose receptor endocytosis and MMP-13 (Madsen 2013, LaRue 2022). Effective rate is gated by f_nascent (LOX-crosslinking shield) to reconcile fresh-gel in vitro assays (Zhu 2001) with in vivo fibrotic collagen persistence.";
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

% phi_col_50_Kmax: collagen volume fraction for 50% carrying capacity inhibition
params.phi_col_50_Kmax.Value = 2.0;  % >1 means ECM barrier is intentionally weak (stroma-depletion paradox)
params.phi_col_50_Kmax.Units = 'dimensionless';
params.phi_col_50_Kmax.Notes = 'phi_collagen for 50% carrying capacity reduction. >1 = ECM barrier never reaches half-max, consistent with stroma-depletion worsening PDAC (Rhim 2014, Ozdemir 2014).';

% Stromal support of carrying capacity
params.k_stroma_support.Value = 1.05;
params.k_stroma_support.Units = 'dimensionless';
params.k_stroma_support.Notes = 'Stromal contribution to cancer carrying capacity. Joint submodel posterior median 1.05 (sigma 0.70) from Xu 2010 + Hwang 2008 (orthotopic dose-response). Hwang 1:5 vs 1:1 tumor mass scaling supports a small per-CAF contribution to cancer C_max.';
params.k_stroma_support.derivedFrom = {'Hwang 2008 DOI: 10.1158/0008-5472.CAN-07-5714', 'Rhim 2014 PMID: 25171127', 'Ozdemir 2014 PMID: 25417103'};

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
params.CXCL12_50_Texcl.Value = 2.0;
params.CXCL12_50_Texcl.Units = 'nanomolarity';
params.CXCL12_50_Texcl.Notes = 'CXCR4 receptor affinity for CXCL12 as concentration threshold. CRITICAL PARAMETER - tune to achieve PDAC immune desert phenotype (CD8/Cancer ~1e-4 to 1e-3). Low EC50 = high receptor sensitivity.';

%% IL-6 Module Parameters (Myeloid Immunosuppression)
% IL-6 secretion rate by iCAFs (was k_IL6_sec_apsc)
params.k_IL6_sec_iCAF.Value = 2e-11;
params.k_IL6_sec_iCAF.Units = 'nanomolarity*liter/(cell*day)';
params.k_IL6_sec_iCAF.Notes = 'IL-6 secretion by iCAFs. Was k_IL6_sec_apsc. iCAFs are the major IL-6 source in PDAC (Elyada 2019 PMID: 31197017; Mace 2018 PMID: 29196415).';
params.k_IL6_sec_iCAF.derivedFrom = {'Elyada 2019 PMID: 31197017', 'Mace 2018 PMID: 29196415'};
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
params.IL6_50_MDSC.Notes = 'IL-6R/gp130 signaling EC50 for MDSC recruitment. MDSCs are most IL-6-sensitive. Raised from 0.02 to 0.06 to avoid permanent saturation at baseline IL-6 ~0.09 nM.';
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
% for the lumped JAK/STAT3 ligand signal. EC50 tuned to model IL-6 levels.
params.IL6_50_iCAF.Value = 0.015;
params.IL6_50_iCAF.Units = 'nanomolarity';
params.IL6_50_iCAF.Notes = 'EC50 for JAK/STAT3 autocrine feedback promoting iCAF phenotype. IL-6 is a proxy for LIF — the actual mediator (Biffi 2019 Supp Fig S4B-D). Tuned to match model IL-6 ~0.09 nM.';
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
% Maximum Treg density in tumor (carrying capacity for tumor-local proliferation)
params.rho_Treg_max.Value = 1e7;
params.rho_Treg_max.Units = 'cell/milliliter';
params.rho_Treg_max.Notes = 'Maximum Treg density in tumor. Reduced from 2e7 to 1e7; no lit justification for 4x Th carrying capacity (5e6). 2x asymmetry retained since Tregs are modestly more abundant than effector Th in PDAC IHC data.';
params.rho_Treg_max.derivedFrom = {};
% Fraction of naturally-occurring Tregs
params.f_nTreg.Value = 0.25;
params.f_nTreg.Units = 'dimensionless';
params.f_nTreg.Notes = 'Fraction of Tregs that are naturally-occurring (antigen-independent) vs induced (antigen-dependent)';


% IL2 Consumption Weighting Factors
params.w_IL2_CD8.Value = 1.0;
params.w_IL2_CD8.Units = 'dimensionless';
params.w_IL2_CD8.Notes = 'IL-2 consumption weight for CD8 cells (CD8 << Treg)';
params.w_IL2_Treg.Value = 1.0;
params.w_IL2_Treg.Units = 'dimensionless';
params.w_IL2_Treg.Notes = 'IL-2 consumption weight for Treg cells (Treg dominant)';

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

%% CCL5/CCR5-Mediated Treg Recruitment Parameters
% Based on: Tan et al. (2017) Oncogene - Cancer-FOXP3 directly activates CCL5
%           to recruit FOXP3+ Treg cells in PDAC
% See also: Huffman et al. (2022) J Immunother Cancer - CCR2/CCR5 inhibition

% CCL5 secretion rate by cancer cells
% PDAC cancer cells express FOXP3 which activates CCL5 transcription
% Similar magnitude to CCL2 secretion (k_CCL2_sec = 1.7e-12 nanomole/cell/day)
params.k_CCL5_sec.Value = 1e-12;
params.k_CCL5_sec.Units = 'nanomole/cell/day';
params.k_CCL5_sec.Notes = 'CCL5 secretion rate by PDAC cancer cells. Cancer-FOXP3 activates CCL5 transcription (Tan 2017, DOI: 10.1038/onc.2016.458)';
params.k_CCL5_sec.derivedFrom = {};

% CCL5 degradation rate
% Chemokine half-life typically 1-4 hours; similar to CCL2 (k_CCL2_deg = 0.06/hour = 1.44/day)
params.k_CCL5_deg.Value = 1.5;
params.k_CCL5_deg.Units = '1/day';
params.k_CCL5_deg.Notes = 'CCL5 degradation rate. Chemokine half-life ~1-4 hours (similar to CCL2)';
params.k_CCL5_deg.derivedFrom = {};

% Half-maximal CCL5 concentration for CCR5-mediated Treg recruitment
% CCR5 EC50 for CCL5 binding ~0.1-1 nM (similar to CCL2_50 = 0.23 nM)
params.CCL5_50_Treg.Value = 1.0;
params.CCL5_50_Treg.Units = 'nanomolarity';
params.CCL5_50_Treg.Notes = 'Half-maximal CCL5 for Treg recruitment via CCR5. CCR5-CCL5 Kd ~0.1-1 nM. Lowered from 0.3 to match model CCL5 ~0.09 nM.';
params.CCL5_50_Treg.derivedFrom = {};

% CCR5-dependent Treg recruitment rate (additional to baseline q_Treg_T_in)
% Tregs express CCR5 while CD8 T cells lack CCR5, providing selective recruitment
% Set to ~2x baseline q_Treg_T_in (5.8e-05) to significantly enhance Treg infiltration
params.k_CCR5_Treg_rec.Value = 1e-04;
params.k_CCR5_Treg_rec.Units = '1/(centimeter^3*minute)';
params.k_CCR5_Treg_rec.Notes = 'CCR5-dependent Treg recruitment rate. CCR5 knockdown significantly decreases Treg infiltration in PDAC (Tan 2017). Rate ~2x baseline to provide strong selective Treg recruitment';
params.k_CCR5_Treg_rec.derivedFrom = {};

% CCL5 secretion rate by iCAFs (was k_CCL5_sec_apsc)
% iCAFs secrete CCL5 in addition to cancer cells, contributing to Treg recruitment
params.k_CCL5_sec_iCAF.Value =  3e-11;
params.k_CCL5_sec_iCAF.Units = 'nanomole/cell/day';
params.k_CCL5_sec_iCAF.Notes = 'CCL5 secretion rate by iCAFs. Was k_CCL5_sec_apsc. CCL5 is inflammatory -> iCAF attribution.';
params.k_CCL5_sec_iCAF.derivedFrom = {'estimated'};

%% apCAF-Specific Parameters (Tolerogenic Antigen Presentation)
% apCAFs express MHC-II and present tumor antigen tolerogenically (Elyada 2019)
params.k_P1_up_apCAF.Value = 1.44;
params.k_P1_up_apCAF.Units = '1/(cell*day)';
params.k_P1_up_apCAF.Notes = 'Antigen uptake rate by apCAFs. ~0.1x of k_up (14.4); less efficient than professional APCs (Elyada 2019 PMID: 31197017).';
params.k_P1_up_apCAF.derivedFrom = {'Elyada 2019 PMID: 31197017'};

params.k_apCAF_Treg.Value = 0.03;
params.k_apCAF_Treg.Units = '1/day';
params.k_apCAF_Treg.Notes = 'Max Th-to-Treg conversion rate via apCAF tolerogenic presentation. Reduced from 0.5 to 0.03 (~1.5x TGFb pathway rate 0.022/day); apCAFs lack costimulatory molecules and are less efficient than professional APCs. Original 0.5 had no lit backing (Elyada 2019 PMID: 31197017).';
params.k_apCAF_Treg.derivedFrom = {'Elyada 2019 PMID: 31197017'};

params.K_apCAF_Treg.Value = 1e8;
params.K_apCAF_Treg.Units = 'cell';
params.K_apCAF_Treg.Notes = 'Half-max apCAF count for saturation of tolerogenic Treg induction. Raised from 1e6 to 1e7 so Hill term only saturates when apCAFs are a substantial fraction of total CAFs (~1e7-1e8).';
params.K_apCAF_Treg.derivedFrom = {'estimated'};

params.P1_50_apCAF.Value = 1e-9;
params.P1_50_apCAF.Units = 'molarity';
params.P1_50_apCAF.Notes = 'EC50 for antigen-dependent Treg induction by apCAFs. Raised from 1e-12 to 1e-9 M; 1e-12 was effectively zero (always-on). 1e-9 M requires meaningful antigen levels for activation.';
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

%% Inflammation-scaled carrying capacity
% The logistic cap on V_T.Th is scaled by a CD8-inflammation Hill so the
% Th pool's cap follows the T-cell-inflamed phenotype:
%
%   V_T.Th cap *= (1 + alpha_Th_cap * H_CD8_Tcap)
%       with H_CD8_Tcap = V_T.CD8 / (V_T.CD8 + CD8_50_Tcap)
%
% Th cap scales with effector CD8 infiltration (T-cell-inflamed TME →
%   chemokine/adhesion-molecule upregulation → higher effector Th density;
%   Peng 2012 PMID: 22437871, Spranger 2015 Nature PMID: 25970248). CD8 is
%   not dominated by the Th pool itself, so this avoids a circular
%   Th → IFNg → Th-cap coupling.
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
%   Th→Treg conversion via k_apCAF_Treg in fibroblast_module_enhanced.
params.alpha_Th_cap.Value = 5;
params.alpha_Th_cap.Units = 'dimensionless';
params.alpha_Th_cap.Notes = 'Max fold-increase of Th carrying cap under full CD8 saturation. Supports ~6x baseline cap when tumor is fully T-cell-inflamed.';
params.alpha_Th_cap.derivedFrom = {};
params.CD8_50_Tcap.Value = 1e5;
params.CD8_50_Tcap.Units = 'cell';
params.CD8_50_Tcap.Notes = 'Half-maximal V_T.CD8 count for scaling Th carrying capacity. Chosen between baseline (~5e4) and GVAX-driven (~8e5) values so Hill is responsive in the treatment regime.';
params.CD8_50_Tcap.derivedFrom = {};
% Retired 2026-05-01: rho_Treg_apCAF_extra (additive Treg-cap apCAF
% pathway) and apCAF_50_Tcap (Hill EC50 for that pathway) removed after
% counterfactual SBI showed the data prefers the pathway off. See the
% block above for the diagnostic summary.

% =====================================================================
% CA19-9 serum biomarker (ca199_module) -- added 2026-07-13 for the
% longitudinal tumor-burden calibration (PASS-01 serial CA19-9). The
% absolute scale is nominal and owned by per-patient calibration; these
% defaults set structure/order-of-magnitude only.
% =====================================================================
% CA19-9 secretion rate per live cancer cell (patient-level; non-secretors ~0)
params.k_ca199_sec.Value = 1e-3;
params.k_ca199_sec.Units = 'picogram/cell/day';
params.k_ca199_sec.Notes = 'Nominal placeholder; per-patient calibrated to serum U/mL. PASS-01 longitudinal CA19-9.';
params.k_ca199_sec.derivedFrom = {};
% CA19-9 shedding rate from the dying/dead cancer-cell pool (lysis spike)
params.k_ca199_shed.Value = 1e-3;
params.k_ca199_shed.Units = 'picogram/cell/day';
params.k_ca199_shed.Notes = 'Nominal placeholder; drives the transient CA19-9 rise on chemo-induced tumor lysis (release from V_T.C_x).';
params.k_ca199_shed.derivedFrom = {};
% CA19-9 serum first-order elimination rate (t_1/2 ~ 1-1.4 day)
params.k_ca199_clear.Value = 0.5;
params.k_ca199_clear.Units = '1/day';
params.k_ca199_clear.Notes = 'Serum CA19-9 elimination; t_1/2 ~1.4 day. Prior-anchored placeholder, calibratable.';
params.k_ca199_clear.derivedFrom = {};
% CA19-9 secretion-suppression strength under aggregate chemo drug effect (v2).
% k_sec_eff = k_ca199_sec/(1 + k_sec_suppress*E_chemo). At full drug effect,
% per-cell secretion drops by 1/(1+k_sec_suppress) -> lets serum CA19-9 nadir
% deeper than bulk burden. NOMINAL, calibrated.
params.k_sec_suppress.Value = 5.0;
params.k_sec_suppress.Units = 'dimensionless';
params.k_sec_suppress.Notes = 'Chemo suppression of CA19-9 secretion; nominal, calibrated.';
params.k_sec_suppress.derivedFrom = {};

% Resistant-clone CA19-9 secretion fold-factor (v3/Update 3). Per-cell CA19-9
% secretion of the resistant clone C2 relative to C1: secretion is weighted
% (V_T.C1 + f_ca199_C2*V_T.C2) instead of bulk C_total. Decouples the biomarker
% rise from bulk RECIST volume in progressors -- a modest resistant-nodule
% volume can drive a large CA19-9 rebound (resistant PDAC clones commonly shift
% secretory/glycosylation phenotype). f=1 recovers v3 (uniform secretion).
% Dimensionless; calibrated. Lognormal prior can collapse to ~1 if unsupported.
params.f_ca199_C2.Value = 10.0;
params.f_ca199_C2.Units = 'dimensionless';
params.f_ca199_C2.Notes = 'Fold-increase of per-cell CA19-9 secretion in resistant clone C2 vs C1; nominal, calibrated.';
params.f_ca199_C2.derivedFrom = {};

% =====================================================================
% Cytotoxic chemotherapy PK-PD (chemo_pk_module) -- added 2026-07-13 for
% PASS-01 (mFOLFIRINOX + gemcitabine/nab-paclitaxel). PK values are FIXED
% literature popPK (mass units: L, L/h, mg/L, mg/h). PD potency params
% (r_*, IC50_*, k_kill_*, Kc_*) are NOMINAL and CALIBRATED (the
% population-level chemo-response params of the hierarchical fit).
% Sources: gem Jiang 2008 PMID 17961191; 5-FU Terret 2000 DOI 10.1067/mcp.2000.109352;
% oxaliplatin Nikanjam 2015 PMID 25557868; irinotecan/SN-38 Xie 2002 PMID 12235447.
% =====================================================================
% ---- Gemcitabine: 2-compartment linear ----
params.V_gem_1.Value = 15;    params.V_gem_1.Units = 'liter';        params.V_gem_1.Notes = 'Gemcitabine central volume (Jiang 2008)'; params.V_gem_1.derivedFrom = {};
params.V_gem_2.Value = 15;    params.V_gem_2.Units = 'liter';        params.V_gem_2.Notes = 'Gemcitabine peripheral volume (Jiang 2008)'; params.V_gem_2.derivedFrom = {};
params.CL_gem.Value  = 162;   params.CL_gem.Units  = 'liter/hour';   params.CL_gem.Notes  = 'Gemcitabine clearance (Jiang 2008)'; params.CL_gem.derivedFrom = {};
params.Q_gem_2.Value = 42;    params.Q_gem_2.Units = 'liter/hour';   params.Q_gem_2.Notes = 'Gemcitabine intercompartmental clearance (Jiang 2008)'; params.Q_gem_2.derivedFrom = {};
params.r_gem.Value = 1.0;     params.r_gem.Units = 'dimensionless';  params.r_gem.Notes = 'Gemcitabine tumor:plasma penetration (nominal, calibrated)'; params.r_gem.derivedFrom = {};
params.IC50_gem.Value = 10;   params.IC50_gem.Units = 'milligram/liter'; params.IC50_gem.Notes = 'Gemcitabine half-max kill conc (nominal, calibrated)'; params.IC50_gem.derivedFrom = {};
params.k_kill_gem.Value = 1.0;params.k_kill_gem.Units = '1/day';     params.k_kill_gem.Notes = 'Gemcitabine max cancer-kill rate (nominal, calibrated)'; params.k_kill_gem.derivedFrom = {};
params.Kc_gem.Value = 8e7;    params.Kc_gem.Units = 'cell';          params.Kc_gem.Notes = 'Half-max cancer number for cytotoxic diffusion (nabP analog)'; params.Kc_gem.derivedFrom = {};
% ---- 5-Fluorouracil: 1-compartment Michaelis-Menten ----
params.V_fu5_1.Value = 30;    params.V_fu5_1.Units = 'liter';        params.V_fu5_1.Notes = '5-FU distribution volume (Terret 2000)'; params.V_fu5_1.derivedFrom = {};
params.Vmax_fu5.Value = 1390; params.Vmax_fu5.Units = 'milligram/hour'; params.Vmax_fu5.Notes = '5-FU DPD Vmax (Terret 2000)'; params.Vmax_fu5.derivedFrom = {};
params.Km_fu5.Value = 5.57;   params.Km_fu5.Units = 'milligram/liter'; params.Km_fu5.Notes = '5-FU DPD Km (Terret 2000)'; params.Km_fu5.derivedFrom = {};
params.r_fu5.Value = 1.0;     params.r_fu5.Units = 'dimensionless';  params.r_fu5.Notes = '5-FU tumor:plasma penetration (nominal, calibrated)'; params.r_fu5.derivedFrom = {};
params.IC50_fu5.Value = 2.0;  params.IC50_fu5.Units = 'milligram/liter'; params.IC50_fu5.Notes = '5-FU half-max kill conc (nominal, calibrated)'; params.IC50_fu5.derivedFrom = {};
params.k_kill_fu5.Value = 1.0;params.k_kill_fu5.Units = '1/day';     params.k_kill_fu5.Notes = '5-FU max cancer-kill rate (nominal, calibrated)'; params.k_kill_fu5.derivedFrom = {};
params.Kc_fu5.Value = 8e7;    params.Kc_fu5.Units = 'cell';          params.Kc_fu5.Notes = 'Half-max cancer number for cytotoxic diffusion (nabP analog)'; params.Kc_fu5.derivedFrom = {};
% ---- Oxaliplatin (free ultrafiltrate Pt): 3-compartment linear ----
params.V_oxa_1.Value = 30;    params.V_oxa_1.Units = 'liter';        params.V_oxa_1.Notes = 'Oxaliplatin central volume (Nikanjam 2015)'; params.V_oxa_1.derivedFrom = {};
params.V_oxa_2.Value = 435;   params.V_oxa_2.Units = 'liter';        params.V_oxa_2.Notes = 'Oxaliplatin peripheral volume 2 (Nikanjam 2015)'; params.V_oxa_2.derivedFrom = {};
params.V_oxa_3.Value = 1183;  params.V_oxa_3.Units = 'liter';        params.V_oxa_3.Notes = 'Oxaliplatin deep volume 3 (Nikanjam 2015)'; params.V_oxa_3.derivedFrom = {};
params.CL_oxa.Value = 16;     params.CL_oxa.Units = 'liter/hour';    params.CL_oxa.Notes = 'Oxaliplatin total clearance (Nikanjam 2015)'; params.CL_oxa.derivedFrom = {};
params.Q_oxa_2.Value = 71;    params.Q_oxa_2.Units = 'liter/hour';   params.Q_oxa_2.Notes = 'Oxaliplatin Q2 (Nikanjam 2015)'; params.Q_oxa_2.derivedFrom = {};
params.Q_oxa_3.Value = 5.3;   params.Q_oxa_3.Units = 'liter/hour';   params.Q_oxa_3.Notes = 'Oxaliplatin Q3 (Nikanjam 2015)'; params.Q_oxa_3.derivedFrom = {};
params.r_oxa.Value = 1.0;     params.r_oxa.Units = 'dimensionless';  params.r_oxa.Notes = 'Oxaliplatin tumor:plasma penetration (nominal, calibrated)'; params.r_oxa.derivedFrom = {};
params.IC50_oxa.Value = 1.0;  params.IC50_oxa.Units = 'milligram/liter'; params.IC50_oxa.Notes = 'Oxaliplatin half-max kill conc (nominal, calibrated)'; params.IC50_oxa.derivedFrom = {};
params.k_kill_oxa.Value = 1.0;params.k_kill_oxa.Units = '1/day';     params.k_kill_oxa.Notes = 'Oxaliplatin max cancer-kill rate (nominal, calibrated)'; params.k_kill_oxa.derivedFrom = {};
params.Kc_oxa.Value = 8e7;    params.Kc_oxa.Units = 'cell';          params.Kc_oxa.Notes = 'Half-max cancer number for cytotoxic diffusion (nabP analog)'; params.Kc_oxa.derivedFrom = {};
% ---- Irinotecan: 2-compartment parent + SN-38 metabolite ----
params.V_iri_1.Value = 40;    params.V_iri_1.Units = 'liter';        params.V_iri_1.Notes = 'Irinotecan central volume (Xie 2002)'; params.V_iri_1.derivedFrom = {};
params.V_iri_2.Value = 220;   params.V_iri_2.Units = 'liter';        params.V_iri_2.Notes = 'Irinotecan peripheral volume (Xie 2002)'; params.V_iri_2.derivedFrom = {};
params.CL_iri.Value = 31.6;   params.CL_iri.Units = 'liter/hour';    params.CL_iri.Notes = 'Irinotecan clearance (Xie 2002)'; params.CL_iri.derivedFrom = {};
params.Q_iri_2.Value = 40;    params.Q_iri_2.Units = 'liter/hour';   params.Q_iri_2.Notes = 'Irinotecan intercompartmental clearance (Xie 2002)'; params.Q_iri_2.derivedFrom = {};
params.V_sn38_1.Value = 2880; params.V_sn38_1.Units = 'liter';       params.V_sn38_1.Notes = 'SN-38 volume (Xie 2002, Fm=0.04)'; params.V_sn38_1.derivedFrom = {};
params.CL_sn38.Value = 28.5;  params.CL_sn38.Units = 'liter/hour';   params.CL_sn38.Notes = 'SN-38 clearance (Xie 2002, Fm=0.04)'; params.CL_sn38.derivedFrom = {};
params.Fm_iri.Value = 0.04;   params.Fm_iri.Units = 'dimensionless'; params.Fm_iri.Notes = 'Fraction irinotecan converted to SN-38 (Xie 2002)'; params.Fm_iri.derivedFrom = {};
params.MWr_sn38.Value = 0.6688; params.MWr_sn38.Units = 'dimensionless'; params.MWr_sn38.Notes = 'MW(SN-38 392.4)/MW(irinotecan 586.7) for mass->mass conversion'; params.MWr_sn38.derivedFrom = {};
params.r_iri.Value = 1.0;     params.r_iri.Units = 'dimensionless';  params.r_iri.Notes = 'SN-38 tumor:plasma penetration (nominal, calibrated)'; params.r_iri.derivedFrom = {};
params.IC50_iri.Value = 0.02; params.IC50_iri.Units = 'milligram/liter'; params.IC50_iri.Notes = 'SN-38 half-max kill conc (nominal, calibrated)'; params.IC50_iri.derivedFrom = {};
params.k_kill_iri.Value = 1.0;params.k_kill_iri.Units = '1/day';     params.k_kill_iri.Notes = 'SN-38 max cancer-kill rate (nominal, calibrated)'; params.k_kill_iri.derivedFrom = {};
params.Kc_iri.Value = 8e7;    params.Kc_iri.Units = 'cell';          params.Kc_iri.Notes = 'Half-max cancer number for cytotoxic diffusion (nabP analog)'; params.Kc_iri.derivedFrom = {};

