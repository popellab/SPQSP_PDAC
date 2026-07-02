% Helper T Cell Module
%
% Models Th activation and transport
%
% Inputs: model        -- SimBiology model object with four compartments
%         params       -- object containing the default parameters
%
% Outputs: model -- SimBiology model object with new Th module


function model = Th_module(model,params)

% nCD4 is the shared naive CD4+ pool (consumed by both Th and Treg activation)
species_name = 'Th';

% Add Species (final names — no rename pattern)
% Activated T cells
aTh = addspecies(model.Compartment(4),'aTh',0,'InitialAmountUnits','cell', 'Notes', ['Number of activated ' species_name ' cells in the lymph node']);
% Mature T cells
Th_C = addspecies(model.Compartment(1),'Th',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the central compartment']);
Th_P = addspecies(model.Compartment(2),'Th',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the peripheral compartment']);
Th_T = addspecies(model.Compartment(3),'Th',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the tumor compartment']);
Th_LN = addspecies(model.Compartment(4),'Th',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the lymph node compartment']);

% Add Hill Functions for APC/mAPC
H_APC = 'H_APC_Th';

% Add Parameters
p = addparameter(model,'k_Th_act',params.k_Th_act.Value,'ValueUnits',params.k_Th_act.Units, 'Notes', 'Rate of T helper cell activation ');
p.UserData.Source = params.k_Th_act.Notes;
    p.UserData.derivedFrom = params.k_Th_act.derivedFrom;
p = addparameter(model,'k_Th_to_Treg',params.k_Th_to_Treg.Value,'ValueUnits',params.k_Th_to_Treg.Units, 'Notes', [species_name ' differentiation rate to Treg']);
p.UserData.Source = params.k_Th_to_Treg.Notes;
    p.UserData.derivedFrom = params.k_Th_to_Treg.derivedFrom;
p = addparameter(model,'k_CD4_pro',params.k_CD4_pro.Value,'ValueUnits',params.k_CD4_pro.Units, 'Notes', 'CD4 proliferation rate');
p.UserData.Source = params.k_CD4_pro.Notes;
    p.UserData.derivedFrom = params.k_CD4_pro.derivedFrom;
p = addparameter(model,'k_CD4_death',params.k_CD4_death.Value,'ValueUnits',params.k_CD4_death.Units, 'Notes', 'CD4 death rate');
p.UserData.Source = params.k_CD4_death.Notes;
    p.UserData.derivedFrom = params.k_CD4_death.derivedFrom;
K_C2_recruit = addparameter(model,'K_C2_recruit',params.K_C2_recruit.Value,'ValueUnits',params.K_C2_recruit.Units, 'Notes', 'Half-Maximal cancer cell number for T cell recruitment ');
K_C2_recruit.UserData.Source = params.K_C2_recruit.Notes;
    K_C2_recruit.UserData.derivedFrom = params.K_C2_recruit.derivedFrom;

% Add Reactions
% Naive T cell activation
reaction = addreaction(model,'V_LN.nCD4 -> null');
    set(reaction,'ReactionRate',['k_Th_act*' H_APC '*H_P1*V_LN.nCD4']);
    set(reaction,'Notes','Naive CD4 activation to Th (antigen-dependent)');
reaction = addreaction(model,'null -> V_LN.aTh');
    set(reaction,'ReactionRate',['k_Th_act*' H_APC '*H_P1*V_LN.nCD4*n_CD4_clones']);
    set(reaction,'Notes','Naive CD4 activation to Th (antigen-dependent, clonal expansion)');
% Activated T cell proliferation
reaction = addreaction(model,'V_LN.aTh -> null');
    set(reaction,'ReactionRate','(k_CD4_pro/N_aTh)*V_LN.aTh');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);
reaction = addreaction(model,'null -> V_LN.Th');
    set(reaction,'ReactionRate','(k_CD4_pro/N_aTh)*2^N_aTh*V_LN.aTh');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);
% Differentiation between Treg and Th Cells
reaction = addreaction(model,'V_T.Th -> V_T.Treg');
    set(reaction,'ReactionRate','k_Th_to_Treg*V_T.Th*H_TGFb');
    set(reaction,'Notes','Differentiation between Treg and Th Cells');

% T cell Death
reaction = addreaction(model,'V_C.Th -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_C.Th');
    set(reaction,'Notes','T cell death in the central compartment');
reaction = addreaction(model,'V_P.Th -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_P.Th');
    set(reaction,'Notes','T cell death in the peripheral compartment');
reaction = addreaction(model,'V_LN.Th -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_LN.Th');
    set(reaction,'Notes','T cell death in the lymph node compartment');
reaction = addreaction(model,'V_T.Th -> null');
    set(reaction,'ReactionRate','k_CD4_death*V_T.Th');
    set(reaction,'Notes','Th cell death in the tumor compartment (regular turnover, not exhaustion)');

% Th exhaustion mechanisms (mirrors CD8 exhaustion routes; previously Th_exh
% was a relabeled death pool, which made Th_exh fraction structurally tied to
% k_CD4_death and crashed CD4 density when forced to match Shibuya 2014).
% Baseline exhaustion from chronic antigen exposure
p = addparameter(model,'k_Th_basal_exh',params.k_Th_basal_exh.Value,'ValueUnits',params.k_Th_basal_exh.Units, 'Notes','Baseline Th exhaustion rate from chronic TCR stimulation (mirrors k_CD8_basal_exh)');
p.UserData.Source = params.k_Th_basal_exh.Notes;
    p.UserData.derivedFrom = params.k_Th_basal_exh.derivedFrom;
reaction = addreaction(model,'V_T.Th -> V_T.Th_exh');
    set(reaction,'ReactionRate','k_Th_basal_exh*V_T.Th*C_total/(C_total+K_C_exh+cell)');
    set(reaction,'Notes','Baseline Th exhaustion from chronic antigen exposure');

% TGFβ-driven Th exhaustion (chronic immunosuppression)
p = addparameter(model,'k_TGFb_Th_exh',params.k_TGFb_Th_exh.Value,'ValueUnits',params.k_TGFb_Th_exh.Units, 'Notes','Maximal Th exhaustion rate driven by TGFβ (chronic immunosuppression)');
p.UserData.Source = params.k_TGFb_Th_exh.Notes;
    p.UserData.derivedFrom = params.k_TGFb_Th_exh.derivedFrom;
reaction = addreaction(model,'V_T.Th -> V_T.Th_exh');
    set(reaction,'ReactionRate','k_TGFb_Th_exh*V_T.Th*H_TGFb_Th_exh');
    set(reaction,'Notes','Th exhaustion driven by TGFβ (cell-specific EC50 via H_TGFb_Th_exh)');

% Th clearance upon antigen clearance (post-tumor-eradication funneling)
reaction = addreaction(model,'V_T.Th -> V_T.Th_exh');
    set(reaction,'ReactionRate','k_cell_clear*V_T.Th*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Th clearance upon antigen clearance');

% T cell transport
% Central & Peripheral
reaction = addreaction(model,'V_C.Th -> V_P.Th');
    set(reaction,'ReactionRate','q_Treg_P_in*V_C.Th');
    set(reaction,'Notes','T cell transport into the peripheral compartment');
reaction = addreaction(model,'V_P.Th -> V_C.Th');
    set(reaction,'ReactionRate','q_Treg_P_out*V_P.Th');
    set(reaction,'Notes','T cell transport out of the peripheral compartment');

% Central & tumor (with CXCL12-mediated exclusion for PDAC, same as CD8)
reaction = addreaction(model,'V_C.Th -> V_T.Th');
    set(reaction,'ReactionRate','q_Treg_T_in*V_T*V_C.Th*(C_total^2/(C_total^2 + K_C2_recruit))*(1-H_CXCL12_Texcl)');
    set(reaction,'Notes','T cell transport into the tumor compartment (reduced by CXCL12-mediated exclusion)');
% Central & LN
reaction = addreaction(model,'V_LN.Th -> V_C.Th');
    set(reaction,'ReactionRate','q_Treg_LN_out*V_LN.Th');
    set(reaction,'Notes','T cell transport out of the lymph node compartment');

% Tumor-local Th proliferation (PDAC TME-driven)
k_Th_pro_tumor = addparameter(model,'k_Th_pro_tumor',params.k_Th_pro_tumor.Value,...
    'ValueUnits',params.k_Th_pro_tumor.Units,...
    'Notes', 'Tumor-local Th proliferation rate (Ki67+ 20-30% in PDAC TME)');
k_Th_pro_tumor.UserData.Source = params.k_Th_pro_tumor.Notes;
    k_Th_pro_tumor.UserData.derivedFrom = params.k_Th_pro_tumor.derivedFrom;
rho_Th_max = addparameter(model,'rho_Th_max',params.rho_Th_max.Value,...
    'ValueUnits',params.rho_Th_max.Units,...
    'Notes', 'Max Th density in tumor (carrying capacity for local proliferation)');
rho_Th_max.UserData.Source = params.rho_Th_max.Notes;
    rho_Th_max.UserData.derivedFrom = params.rho_Th_max.derivedFrom;

% Inflammation-scaled carrying-capacity parameter (Fix 2).
% CD8_50_Tcap is already registered by Treg_module (runs earlier).
alpha_Th_cap = addparameter(model,'alpha_Th_cap',params.alpha_Th_cap.Value,...
    'ValueUnits',params.alpha_Th_cap.Units,...
    'Notes','Max fold-increase of Th cap under CD8 inflammation');
alpha_Th_cap.UserData.Source = params.alpha_Th_cap.Notes;
    alpha_Th_cap.UserData.derivedFrom = params.alpha_Th_cap.derivedFrom;

% Carrying capacity scales with CD8 infiltration (Fix 2): inflamed tumors
% sustain higher T-cell densities. CD8 is used (not IFNg) to avoid a
% circular Th->IFNg->Th-cap coupling: the Th pool dominates IFNg production,
% so an IFNg-scaled cap cannot respond to GVAX when Th itself is capped.
reaction = addreaction(model,'null -> V_T.Th');
    set(reaction,'ReactionRate',...
        'k_Th_pro_tumor*V_T.Th*(1 - V_T.Th/(rho_Th_max*V_T*(1 + alpha_Th_cap*V_T.CD8/(V_T.CD8 + CD8_50_Tcap)) + cell))');
    set(reaction,'Notes','Tumor-local Th proliferation with CD8-scaled carrying capacity');

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
