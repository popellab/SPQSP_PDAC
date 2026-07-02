% T Follicular Helper (Tfh) Cell Module
%
% Minimal CD4 Tfh subset for the coupled QSP+ABM PDAC model. Mirrors
% Th_module: consumes the shared naive CD4 pool (nCD4, created in Treg_module),
% is primed by cDC2 (reuses H_APC_Th) in an antigen-dependent manner (H_P1),
% traffics C/P/T/LN, and supplies B-cell help in the LN (via V_LN.Tfh, consumed
% by Bcell_module's H_Tfh_help). Kept lean: no tumor-local proliferation or
% exhaustion routes (Tfh here is a helper pool, not an effector).
%
% Reuses (already defined by earlier modules): k_CD4_pro, k_CD4_death, N_aTh,
% n_CD4_clones, H_APC_Th, H_P1, H_CXCL12_Texcl, K_C2_recruit, q_Treg_P_in/out,
% q_Treg_T_in, q_Treg_LN_out. Only new parameter: k_Tfh_act.
%
% Inputs:  model  -- SimBiology model (4 compartments), after Th_module/APC_module
%          params -- parameter struct (parameters_PDAC + load_parameters)
% Outputs: model  -- model with the Tfh module added

function model = Tfh_module(model,params)

species_name = 'Tfh';

% Add Species
% Activated Tfh in the lymph node
addspecies(model.Compartment(4),'aTfh',0,'InitialAmountUnits','cell', 'Notes', ['Number of activated ' species_name ' cells in the lymph node']);
% Mature Tfh across compartments
addspecies(model.Compartment(1),'Tfh',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the central compartment']);
addspecies(model.Compartment(2),'Tfh',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the peripheral compartment']);
addspecies(model.Compartment(3),'Tfh',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the tumor compartment']);
addspecies(model.Compartment(4),'Tfh',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the lymph node compartment']);

% cDC2 priming Hill (shared with Th; both CD4 subsets primed by cDC2/mcDC2)
H_APC = 'H_APC_Th';

% Add Parameters
p = addparameter(model,'k_Tfh_act',params.k_Tfh_act.Value,'ValueUnits',params.k_Tfh_act.Units, 'Notes', 'Rate of naive CD4 activation into Tfh');
p.UserData.Source = params.k_Tfh_act.Notes;
    p.UserData.derivedFrom = params.k_Tfh_act.derivedFrom;

% Add Reactions
% Naive CD4 activation to Tfh (antigen-dependent; consumes shared nCD4 pool)
reaction = addreaction(model,'V_LN.nCD4 -> null');
    set(reaction,'ReactionRate',['k_Tfh_act*' H_APC '*H_P1*V_LN.nCD4']);
    set(reaction,'Notes','Naive CD4 activation to Tfh (antigen-dependent)');
reaction = addreaction(model,'null -> V_LN.aTfh');
    set(reaction,'ReactionRate',['k_Tfh_act*' H_APC '*H_P1*V_LN.nCD4*n_CD4_clones']);
    set(reaction,'Notes','Naive CD4 activation to Tfh (antigen-dependent, clonal expansion)');
% Activated Tfh proliferation
reaction = addreaction(model,'V_LN.aTfh -> null');
    set(reaction,'ReactionRate','(k_CD4_pro/N_aTh)*V_LN.aTfh');
    set(reaction,'Notes','aTfh cell proliferation');
reaction = addreaction(model,'null -> V_LN.Tfh');
    set(reaction,'ReactionRate','(k_CD4_pro/N_aTh)*2^N_aTh*V_LN.aTfh');
    set(reaction,'Notes','aTfh cell proliferation');

% Tfh Death
reaction = addreaction(model,'V_C.Tfh -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_C.Tfh');
    set(reaction,'Notes','Tfh death in the central compartment');
reaction = addreaction(model,'V_P.Tfh -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_P.Tfh');
    set(reaction,'Notes','Tfh death in the peripheral compartment');
reaction = addreaction(model,'V_LN.Tfh -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_LN.Tfh');
    set(reaction,'Notes','Tfh death in the lymph node compartment');
reaction = addreaction(model,'V_T.Tfh -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_T.Tfh');
    set(reaction,'Notes','Tfh death in the tumor compartment');

% Tfh clearance upon antigen clearance (post-tumor-eradication funneling)
reaction = addreaction(model,'V_T.Tfh -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.Tfh*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Tfh clearance upon antigen clearance');

% Tfh transport (mirrors Th/Treg trafficking; reuses q_Treg_* rate constants)
% Central & Peripheral
reaction = addreaction(model,'V_C.Tfh -> V_P.Tfh');
    set(reaction,'ReactionRate','q_Treg_P_in*V_C.Tfh');
    set(reaction,'Notes','Tfh transport into the peripheral compartment');
reaction = addreaction(model,'V_P.Tfh -> V_C.Tfh');
    set(reaction,'ReactionRate','q_Treg_P_out*V_P.Tfh');
    set(reaction,'Notes','Tfh transport out of the peripheral compartment');
% Central & tumor (with CXCL12-mediated exclusion for PDAC, same as CD8/Th)
reaction = addreaction(model,'V_C.Tfh -> V_T.Tfh');
    set(reaction,'ReactionRate','q_Treg_T_in*V_T*V_C.Tfh*(C_total^2/(C_total^2 + K_C2_recruit))*(1-H_CXCL12_Texcl)');
    set(reaction,'Notes','Tfh transport into the tumor compartment (reduced by CXCL12-mediated exclusion)');
% Central & LN
reaction = addreaction(model,'V_LN.Tfh -> V_C.Tfh');
    set(reaction,'ReactionRate','q_Treg_LN_out*V_LN.Tfh');
    set(reaction,'Notes','Tfh transport out of the lymph node compartment');

% Get Model Rules for Updating
model_rules = get(model,'Rules');
% Update Total T Cells in tumor (Rule 3)
Tcell_rule = model_rules(3);
rule = get(Tcell_rule,'Rule');
set(Tcell_rule,'Rule',[rule '+V_T.' species_name]);
% Update Total T Cells in LN (Rule 4)
Tcell_rule = model_rules(4);
rule = get(Tcell_rule,'Rule');
set(Tcell_rule,'Rule',[rule '+V_LN.' species_name]);

warning('off','SimBiology:DimAnalysisNotDone_MatlabFcn_Dimensionless');
