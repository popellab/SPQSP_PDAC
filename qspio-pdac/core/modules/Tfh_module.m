% T Follicular Helper (Tfh) Cell Module
%
% CD4 Tfh subset for the coupled QSP+ABM PDAC model. Tfh are primed in the
% LN from the shared naive CD4 pool (nCD4, created in Treg_module) by cDC2
% in an antigen-dependent manner. They supply B-cell help in the LN (via
% V_LN.Tfh, consumed by Bcell_module's H_Tfh_help).
%
% TUMOR COMPARTMENT: Tfh do NOT traffic from blood to tumor. They arise
% locally from Th -> Tfh differentiation gated by antigen (H_P1) and IL-6
% (pseudo-spatial approximation of ABM's DC contact + IL-6 gate). This
% matches the ABM's t_reg.cuh TH->TCD4_TFH transition and reflects that
% Tfh are GC-resident cells that form in situ at TLS, not recruited from
% circulation.
%
% Tfh in the tumor provide help for B -> Plasma differentiation (consumed
% by Bcell_module's Tfh_50_plasma Hill).
%
% Reuses (already defined by earlier modules): k_CD4_pro, k_CD4_death, N_aTh,
% n_CD4_clones, H_APC_Th, H_P1, K_C2_recruit, q_Treg_P_in/out,
% q_Treg_LN_out, V_T.IL6, IL6_50_MDSC.
% New parameters: k_Tfh_act, k_Tfh_diff_tumor, IL6_50_Tfh.
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

% cDC2 priming Hill: reuse H_APC_Treg (same cDC2-nCD4 saturation formula;
% the old H_APC_Th was retired from APC_module when Th priming moved to
% mass-action, but Tfh priming keeps the saturating Hill form)
H_APC = 'H_APC_Treg';

% Add Parameters
p = addparameter(model,'k_Tfh_act',params.k_Tfh_act.Value,'ValueUnits',params.k_Tfh_act.Units, 'Notes', 'Rate of naive CD4 activation into Tfh');
p.UserData.Source = params.k_Tfh_act.Notes;
    p.UserData.derivedFrom = params.k_Tfh_act.derivedFrom;

p = addparameter(model,'k_Tfh_diff_tumor',params.k_Tfh_diff_tumor.Value,'ValueUnits',params.k_Tfh_diff_tumor.Units, 'Notes', 'Rate of tumor-local Th -> Tfh differentiation (antigen + IL-6 gated)');
p.UserData.Source = params.k_Tfh_diff_tumor.Notes;
    p.UserData.derivedFrom = params.k_Tfh_diff_tumor.derivedFrom;

p = addparameter(model,'IL6_50_Tfh',params.IL6_50_Tfh.Value,'ValueUnits',params.IL6_50_Tfh.Units, 'Notes', 'EC50 for IL-6 Hill function gating Th -> Tfh differentiation in tumor');
p.UserData.Source = params.IL6_50_Tfh.Notes;
    p.UserData.derivedFrom = params.IL6_50_Tfh.derivedFrom;

% IL-6 Hill for Tfh differentiation (assignment rule, computed each step)
addparameter(model,'H_IL6_Tfh',0,'ValueUnits','dimensionless','ConstantValue',false, ...
    'Notes','IL-6 Hill function for tumor-local Th -> Tfh differentiation');

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

% Tumor-local Th -> Tfh differentiation (replaces C->T trafficking)
% Pseudo-spatial approximation: antigen (H_P1) + IL-6 Hill gate.
% Matches ABM's t_reg.cuh:318-331 (DC contact + IL-6 -> Tfh).
% In the QSP, H_P1 proxies antigen presentation (DC encounter) and
% IL-6 provides the cytokine gate that the ABM resolves spatially.
reaction = addreaction(model,'V_T.Th -> V_T.Tfh');
    set(reaction,'ReactionRate','k_Tfh_diff_tumor*H_P1*H_IL6_Tfh*V_T.Th');
    set(reaction,'Notes','Tumor-local Th -> Tfh differentiation (antigen + IL-6 gated, TLS biology)');

% Tfh transport (C/P/LN only — NO C->T trafficking)
% Central & Peripheral
reaction = addreaction(model,'V_C.Tfh -> V_P.Tfh');
    set(reaction,'ReactionRate','q_Treg_P_in*V_C.Tfh');
    set(reaction,'Notes','Tfh transport into the peripheral compartment');
reaction = addreaction(model,'V_P.Tfh -> V_C.Tfh');
    set(reaction,'ReactionRate','q_Treg_P_out*V_P.Tfh');
    set(reaction,'Notes','Tfh transport out of the peripheral compartment');
% Central & LN
reaction = addreaction(model,'V_LN.Tfh -> V_C.Tfh');
    set(reaction,'ReactionRate','q_Treg_LN_out*V_LN.Tfh');
    set(reaction,'Notes','Tfh transport out of the lymph node compartment');

% Assignment rule: IL-6 Hill for Tfh differentiation
addrule(model,'H_IL6_Tfh = V_T.IL6/(IL6_50_Tfh + V_T.IL6)','repeatedAssignment');

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
