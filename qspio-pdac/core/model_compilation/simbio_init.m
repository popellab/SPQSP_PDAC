% Function to initialize simbiology object
%
% Inputs: model_name -- string containing model name
%         time       -- array of time points (in days)
%         solver     -- string containing solver (e.g. ode15s)
%         tol_abs    -- absolute tolerance of solver
%         tol_rel    -- relative tolerance of solver
%         params     -- object containing compartment parameters
%                       - V_C--central compartment volume
%                       - V_P--peripheral compartment volume
%                       - V_Tmin--cancer-free tumour compartment volume
%                       - V_LN--lymph node compartment volume
%                       - vol_cell--cancer cell volume
%                       - vol_Tcell--T cell volume
%                       - k_cell_clear--dead cell clearance rate
%
% Outputs: model -- simbiology model object containing user-specified
%                   configuration and four compartments: C,P,T,LN


function model = simbio_init(model_name,time,solver,tol_abs,tol_rel,params)

% Model Object
model = sbiomodel(model_name);

% Configure simulation settings using the generalized function
model = simulation_config(model, ...
    'solver', solver, ...
    'time', time, ...
    'abs_tolerance', tol_abs, ...
    'rel_tolerance', tol_rel, ...
    'time_units', 'day', ...
    'unit_conversion', true);

% Setup Compartments
comp_C = addcompartment(model,'V_C',params.V_C.Value,'CapacityUnits',params.V_C.Units);
    set(comp_C,'Notes','Central compartment (C) volume');
    comp_C.UserData.Source = params.V_C.Notes;
    comp_C.UserData.derivedFrom = params.V_C.derivedFrom;
comp_P = addcompartment(model,'V_P',params.V_P.Value,'CapacityUnits',params.V_P.Units);
    set (comp_P,'Notes','Peripheral compartment (P) volume');
    comp_P.UserData.Source = params.V_P.Notes;
    comp_P.UserData.derivedFrom = params.V_P.derivedFrom;
comp_T = addcompartment(model,'V_T',params.V_Tmin.Value,'CapacityUnits',params.V_Tmin.Units,'ConstantCapacity',false);
    set (comp_T,'Notes','Tumor compartment (T) volume');
    comp_T.UserData.Source = params.V_Tmin.Notes;
    comp_T.UserData.derivedFrom = params.V_Tmin.derivedFrom;
comp_LN = addcompartment(model,'V_LN',params.V_LN.Value,'CapacityUnits',params.V_LN.Units);
    set(comp_LN,'Notes','Lymph node (LN) compartment volume');
    comp_LN.UserData.Source = params.V_LN.Notes;
    comp_LN.UserData.derivedFrom = params.V_LN.derivedFrom;

% Dead Cells
P = addparameter(model,'k_cell_clear',params.k_cell_clear.Value,'ValueUnits',params.k_cell_clear.Units);
    set(P,'Notes','Rate of dead cell clearance from tumour compartment');
    P.UserData.Source = params.k_cell_clear.Notes;
    P.UserData.derivedFrom = params.k_cell_clear.derivedFrom;
S = addspecies(comp_T,'C_x',0,'InitialAmountUnits','cell');
    set(S,'Notes','Dead cancer cells in the tumour compartment');
S = addspecies(comp_T,'CD8_exh',0,'InitialAmountUnits','cell');
    set(S,'Notes','Terminally exhausted CD8 effector T cells (TCF1-)');
S = addspecies(comp_T,'CD8_pex',0,'InitialAmountUnits','cell');
    set(S,'Notes','Progenitor-exhausted (Tpex, TCF1+) CD8 reservoir; self-renews against a niche cap, differentiates to CD8_exh, refluxes to CD8 under checkpoint blockade');
S = addspecies(comp_T,'Th_exh',0,'InitialAmountUnits','cell');
    set(S,'Notes','Exhausted CD4 effector T cells');
R = addreaction(model,'V_T.C_x -> null');
    set(R,'ReactionRate','k_cell_clear*V_T.C_x');
    set(R,'Notes','Clearance of dead cancer cells from tumor');
% NOTE: exhausted-CD8 death is NOT created here. It is handled by the
% purpose-built, submodel-anchored k_CD8_exh_death reaction in Teff_module.m
% ('V_T.CD8_exh -> null' at k_CD8_exh_death). A k_cell_clear clearance of
% CD8_exh was previously ALSO added here, double-counting exhausted-CD8 death
% at the summed rate (k_cell_clear + k_CD8_exh_death) ~ 0.25/day instead of the
% calibrated 0.15/day, which suppressed the exhausted-CD8 fraction (gvax_nivo
% d21 median ~0.20 double-decay vs ~0.25 single-decay). Removed so CD8_exh
% decays solely via its calibrated k_CD8_exh_death rate.
R = addreaction(model,'V_T.Th_exh -> null');
    set(R,'ReactionRate','k_cell_clear*V_T.Th_exh');
    set(R,'Notes','Clearance of exhausted CD4 T cells from tumor');

% Define Cell and Time
p = addparameter(model,'cell',1.0,'ValueUnits','cell');
    set(p,'Notes','unit parameter for calculation');
p = addparameter(model,'day',1.0,'ValueUnits','day');
    set(p,'Notes','unit parameter for calculation');

D_cell = addparameter(model,'D_cell',params.D_cell.Value,'ValueUnits',params.D_cell.Units);
    set(D_cell,'Notes','Average diameter of cancer cell');
    D_cell.UserData.Source = params.D_cell.Notes;
vol_cell = addparameter(model,'vol_cell',params.vol_cell.Value,'ValueUnits',params.vol_cell.Units);
    set(vol_cell,'Notes','Average volume of cancer cell');
    vol_cell.UserData.Source = params.vol_cell.Notes;
    vol_cell.UserData.derivedFrom = params.vol_cell.derivedFrom;
vol_Tcell = addparameter(model,'vol_Tcell',params.vol_Tcell.Value,'ValueUnits',params.vol_Tcell.Units);
    set(vol_Tcell,'Notes','Average volume of T cells');
    vol_Tcell.UserData.Source = params.vol_Tcell.Notes;
    vol_Tcell.UserData.derivedFrom = params.vol_Tcell.derivedFrom;

% rho_cell = addparameter(model,'rho_cell',params.rho_cell.Value,'ValueUnits',params.rho_cell.Units,'ConstantValue',false);
%     set(rho_cell,'Notes',['Tumor density']);
Ve_T = addparameter(model,'Ve_T',params.Ve_T.Value,'ValueUnits',params.Ve_T.Units,'ConstantValue',false);
    set(Ve_T,'Notes','Intracellular fraction of the tumor');
    Ve_T.UserData.Source = params.Ve_T.Notes;
    Ve_T.UserData.derivedFrom = params.Ve_T.derivedFrom;

% Initial Condition Parameters (Density-Based)
% These density priors are converted to absolute counts in set_baseline_populations.m
% Cell density parameters (cells/mm^3)
% p = addparameter(model,'C1_density_initial',params.C1_density_initial.Value,'ValueUnits',params.C1_density_initial.Units);
%     set(p,'Notes',params.C1_density_initial.Notes);
% p = addparameter(model,'CD8_density_initial',params.CD8_density_initial.Value,'ValueUnits',params.CD8_density_initial.Units);
%     set(p,'Notes',params.CD8_density_initial.Notes);
% p = addparameter(model,'Treg_density_initial',params.Treg_density_initial.Value,'ValueUnits',params.Treg_density_initial.Units);
%     set(p,'Notes',params.Treg_density_initial.Notes);
% p = addparameter(model,'Th_density_initial',params.Th_density_initial.Value,'ValueUnits',params.Th_density_initial.Units);
%     set(p,'Notes',params.Th_density_initial.Notes);
% p = addparameter(model,'MDSC_density_initial',params.MDSC_density_initial.Value,'ValueUnits',params.MDSC_density_initial.Units);
%     set(p,'Notes',params.MDSC_density_initial.Notes);
% p = addparameter(model,'TAM_density_initial',params.TAM_density_initial.Value,'ValueUnits',params.TAM_density_initial.Units);
%     set(p,'Notes',params.TAM_density_initial.Notes);
% p = addparameter(model,'m1_m2_ratio_initial',params.m1_m2_ratio_initial.Value,'ValueUnits',params.m1_m2_ratio_initial.Units);
%     set(p,'Notes',params.m1_m2_ratio_initial.Notes);
% p = addparameter(model,'aPSC_density_initial',params.aPSC_density_initial.Value,'ValueUnits',params.aPSC_density_initial.Units);
%     set(p,'Notes',params.aPSC_density_initial.Notes);
% p = addparameter(model,'qPSC_density_initial',params.qPSC_density_initial.Value,'ValueUnits',params.qPSC_density_initial.Units);
%     set(p,'Notes',params.qPSC_density_initial.Notes);
% p = addparameter(model,'APC_density_initial',params.APC_density_initial.Value,'ValueUnits',params.APC_density_initial.Units);
%     set(p,'Notes',params.APC_density_initial.Notes);
% p = addparameter(model,'apc_maturation_ratio_initial',params.apc_maturation_ratio_initial.Value,'ValueUnits',params.apc_maturation_ratio_initial.Units);
%     set(p,'Notes',params.apc_maturation_ratio_initial.Notes);
% % ECM fraction and cytokine concentrations
% p = addparameter(model,'ECM_fraction_initial',params.ECM_fraction_initial.Value,'ValueUnits',params.ECM_fraction_initial.Units);
%     set(p,'Notes',params.ECM_fraction_initial.Notes);
% p = addparameter(model,'CCL2_conc_initial',params.CCL2_conc_initial.Value,'ValueUnits',params.CCL2_conc_initial.Units);
%     set(p,'Notes',params.CCL2_conc_initial.Notes);
% p = addparameter(model,'TGFb_conc_initial',params.TGFb_conc_initial.Value,'ValueUnits',params.TGFb_conc_initial.Units);
%     set(p,'Notes',params.TGFb_conc_initial.Notes);
% p = addparameter(model,'VEGF_conc_initial',params.VEGF_conc_initial.Value,'ValueUnits',params.VEGF_conc_initial.Units);
%     set(p,'Notes',params.VEGF_conc_initial.Notes);

% Minimum tumor volume (cancer-free baseline, prevents 1/V_T singularity)
p = addparameter(model,'V_Tmin',params.V_Tmin.Value,'ValueUnits',params.V_Tmin.Units);
    set(p,'Notes','Minimum tumor compartment volume (cancer-free baseline)');

% Set Tumour Volume (Rule 1)
% V_Tmin floor ensures V_T > 0 even when cell counts are near zero
addrule(model,'V_T = V_Tmin + ((C_x+C_total)*vol_cell+(CD8_exh+CD8_pex+Th_exh+T_total)*vol_Tcell)/Ve_T','repeatedAssignment');

% Set Total Number of Cancer Cells (Rule 2)
p = addparameter(model,'C_total',0,'ValueUnits','cell','ConstantValue',false);
    set(p,'Notes','Total number of cancer cells');
addrule(model,'C_total = 0*cell','repeatedAssignment');

% Set Total Number of T Cells in Tumour (Rule 3)
p = addparameter(model,'T_total',0,'ValueUnits','cell','ConstantValue',false);
    set(p,'Notes','Total number of activated T cells in tumor');
addrule(model,'T_total = 0*cell','repeatedAssignment');

% Set Total Number of T Cells in LN (Rule 4)
p = addparameter(model,'T_total_LN',0,'ValueUnits','cell','ConstantValue',false);
    set(p,'Notes','Total number of activated T cells in TDLNs');
addrule(model,'T_total_LN = 0*cell','repeatedAssignment');

% Set Total Rate of Cancer Death by T Cells (Rule 5)
p = addparameter(model,'R_Tcell','ValueUnits','cell/day','ConstantValue',false);
    set(p,'Notes','Rate of cancer cell death');
addrule(model,'R_Tcell = 0*cell/day','repeatedAssignment');

% Set Default Number of Tregs in Tumor (updated by Treg_module: Tregs_ = V_T.Treg)
p = addparameter(model,'Tregs_',0,'ValueUnits','cell','ConstantValue',false);
    set(p,'Notes','Total number of Tregs in tumor compartment (V_T.Treg)');

% Set Default Hill Function for APCs (Treg priming via cDC2)
p = addparameter(model,'H_APC_Treg',0.5,'ValueUnits','dimensionless','ConstantValue',false);
    set(p,'Notes','Hill function of cDC2 for Treg activation (shared nCD4 pool, self-antigen)');

% H_mAPC retired — CD8 priming now uses mass-action on V_LN.mcDC1 directly
% (Teff_module.m); the old cDC1 Hill was unused (no reaction consumed it).

% H_APC_Th retired — Th priming now uses mass-action on V_LN.mcDC2 directly
% (Th_module.m mechanistic refactor, parallel to CD8 priming in Teff_module.m).

% Set Default Hill Function for PD1 Checkpoint
p = addparameter(model,'H_PD1_C1',0.90,'ValueUnits','dimensionless','ConstantValue',false);
    set(p,'Notes','Hill function of PDL1 on chemotherapy-sensitive tumor cells for T cell exhaustion');

% addparameter(model,'H_PD1_C2',0.90,'ValueUnits','dimensionless','ConstantValue',false);
p = addparameter(model,'H_PD1_APC',0.90,'ValueUnits','dimensionless','ConstantValue',false);
    set(p,'Notes','Hill function of PDL1 on APC for T cell exhaustion');

% Set Default Hill Function for CTLA4 Checkpoint
p = addparameter(model,'H_CD28_C1',0.1,'ValueUnits','dimensionless','ConstantValue',false);
    set(p,'Notes','Hill function of CD28 on tumor cells');

% addparameter(model,'H_CD28_C2',0.1,'ValueUnits','dimensionless','ConstantValue',false);
p = addparameter(model,'H_CD28_APC',0.1,'ValueUnits','dimensionless','ConstantValue',false);
    set(p,'Notes','Hill function of CD28 on APCs for T cell activation');

