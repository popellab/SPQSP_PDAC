% Teff Module
%
% Models Teff transport and activation by APCs [Use before antigen corresponding module]
%
% Inputs: model        -- SimBiology model object with four compartments
%         ID           -- T cell-antigen ID number [must be unique]
%         params       -- object containing model parameter Values, Units, and Notes:
%         cancer_types -- cell array of strings containing names of cancer types Teff kill
%
% Outputs: model -- SimBiology model object with new Tcell module


function model = Teff_module(model,params,cancer_types)

% Species Names
species_name = 'CD8';
antigen = 'P1';

% Add Species (final names — no rename pattern)
% Naive T cells
nCD8_C  = addspecies(model.Compartment(1),'nCD8',params.nCD8_C.Value/params.nCD8_div.Value,'InitialAmountUnits','cell');
    set(nCD8_C,'Notes',['Number of naive ' species_name ' cells in the central compartment']);
nCD8_P  = addspecies(model.Compartment(2),'nCD8',params.nCD8_P.Value/params.nCD8_div.Value,'InitialAmountUnits','cell');
    set(nCD8_P,'Notes',['Number of naive ' species_name ' cells in the peripheral compartment']);
nCD8_LN = addspecies(model.Compartment(4),'nCD8',params.nCD8_LN.Value/params.nCD8_div.Value,'InitialAmountUnits','cell', 'Notes', ['Number of naive ' species_name ' cells in the lymph node']);
% Activated T cells
aCD8 = addspecies(model.Compartment(4),'aCD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of activated ' species_name ' cells in the lymph node']);
% Mature T cells
CD8_C = addspecies(model.Compartment(1),'CD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the central compartment']);
CD8_P = addspecies(model.Compartment(2),'CD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the peripheral compartment']);
CD8_T = addspecies(model.Compartment(3),'CD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the tumor compartment']);
CD8_LN = addspecies(model.Compartment(4),'CD8',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the lymph node compartment']);

% Add Hill Functions for APC/mAPC
H_APC = 'H_mAPC';

% Add Parameters (final names — no rename pattern)
div_CD8 = addparameter(model,'div_CD8',params.nCD8_div.Value,'ValueUnits',params.nCD8_div.Units, 'Notes', 'T Cell Diversity ');
div_CD8.UserData.Source = params.nCD8_div.Notes;
    div_CD8.UserData.derivedFrom = params.nCD8_div.derivedFrom;
n_CD8_clones = addparameter(model,'n_CD8_clones',params.n_clones_tum.Value,'ValueUnits',params.n_clones_tum.Units, 'Notes', 'Number of tumor-specific T cell clones ');
n_CD8_clones.UserData.Source = params.n_clones_tum.Notes;
    n_CD8_clones.UserData.derivedFrom = params.n_clones_tum.derivedFrom;
p = addparameter(model,'q_nCD8_LN_in',params.q_nCD8_LN_in.Value,'ValueUnits',params.q_nCD8_LN_in.Units, 'Notes', 'Rate of naive T cell transport into the LN ');
p.UserData.Source = params.q_nCD8_LN_in.Notes;
    p.UserData.derivedFrom = params.q_nCD8_LN_in.derivedFrom;
p = addparameter(model,'q_CD8_LN_out',params.q_CD8_LN_out.Value,'ValueUnits',params.q_CD8_LN_out.Units, 'Notes', 'Rate of activated T cell transport out of the LN ');
p.UserData.Source = params.q_CD8_LN_out.Notes;
    p.UserData.derivedFrom = params.q_CD8_LN_out.derivedFrom;
p = addparameter(model,'q_nCD8_LN_out',params.q_nCD8_LN_out.Value,'ValueUnits',params.q_nCD8_LN_out.Units, 'Notes', 'Rate of naive T cell transport out of the LN ');
p.UserData.Source = params.q_nCD8_LN_out.Notes;
    p.UserData.derivedFrom = params.q_nCD8_LN_out.derivedFrom;
p = addparameter(model,'k_CD8_act',params.k_nCD8_act.Value,'ValueUnits',params.k_nCD8_act.Units, 'Notes', [species_name ' activation rate']);
p.UserData.Source = params.k_nCD8_act.Notes;
    p.UserData.derivedFrom = params.k_nCD8_act.derivedFrom;
p = addparameter(model,'k_CD8_pro',params.k_CD8_pro.Value,'ValueUnits',params.k_CD8_pro.Units, 'Notes', [species_name ' proliferation rate']);
p.UserData.Source = params.k_CD8_pro.Notes;
    p.UserData.derivedFrom = params.k_CD8_pro.derivedFrom;
% Tumor-local CD8 proliferation params (antigen-driven, PD-1-suppressed)
p = addparameter(model,'k_CD8_T_pro',params.k_CD8_T_pro.Value,'ValueUnits',params.k_CD8_T_pro.Units, 'Notes', 'Tumor-local CD8 proliferation rate (antigen-gated, PD-1-suppressed)');
p.UserData.Source = params.k_CD8_T_pro.Notes;
    p.UserData.derivedFrom = params.k_CD8_T_pro.derivedFrom;
p = addparameter(model,'rho_CD8_max',params.rho_CD8_max.Value,'ValueUnits',params.rho_CD8_max.Units, 'Notes', 'Max CD8 density in tumor (carrying capacity for local proliferation)');
p.UserData.Source = params.rho_CD8_max.Notes;
    p.UserData.derivedFrom = params.rho_CD8_max.derivedFrom;
% Mechanistic T cell activation parameters
p = addparameter(model,'k_T_activation_per_APC',params.k_T_activation_per_APC.Value,'ValueUnits',params.k_T_activation_per_APC.Units, 'Notes', 'Rate of naive T cell activation per mature APC');
p.UserData.Source = params.k_T_activation_per_APC.Notes;
    p.UserData.derivedFrom = params.k_T_activation_per_APC.derivedFrom;
p = addparameter(model,'n_T_expansion_burst',params.n_T_expansion_burst.Value,'ValueUnits',params.n_T_expansion_burst.Units, 'Notes', 'Clonal expansion burst size');
p.UserData.Source = params.n_T_expansion_burst.Notes;
    p.UserData.derivedFrom = params.n_T_expansion_burst.derivedFrom;
p = addparameter(model,'k_CD8_death',params.k_CD8_death.Value,'ValueUnits',params.k_CD8_death.Units, 'Notes', [species_name ' death rate']);
p.UserData.Source = params.k_CD8_death.Notes;
    p.UserData.derivedFrom = params.k_CD8_death.derivedFrom;
p = addparameter(model,'q_CD8_P_in',params.q_CD8_P_in.Value,'ValueUnits',params.q_CD8_P_in.Units, 'Notes', 'rate of CD8+ T cell transport into the peripheral compartment ');
p.UserData.Source = params.q_CD8_P_in.Notes;
    p.UserData.derivedFrom = params.q_CD8_P_in.derivedFrom;
p = addparameter(model,'q_CD8_P_out',params.q_CD8_P_out.Value,'ValueUnits',params.q_CD8_P_out.Units, 'Notes', 'rate of CD8+ T cell transport out of the peripheral compartment ');
p.UserData.Source = params.q_CD8_P_out.Notes;
    p.UserData.derivedFrom = params.q_CD8_P_out.derivedFrom;
p = addparameter(model,'q_CD8_T_in',params.q_CD8_T_in.Value,'ValueUnits',params.q_CD8_T_in.Units, 'Notes', 'rate of CD8+ T cell transport into the tumor compartment ');
p.UserData.Source = params.q_CD8_T_in.Notes;
    p.UserData.derivedFrom = params.q_CD8_T_in.derivedFrom;
p = addparameter(model,'q_nCD8_P_in',params.q_nCD8_P_in.Value,'ValueUnits',params.q_nCD8_P_in.Units, 'Notes', 'rate of naive CD8+ T cell transport into the peripheral compartment ');
p.UserData.Source = params.q_nCD8_P_in.Notes;
    p.UserData.derivedFrom = params.q_nCD8_P_in.derivedFrom;
p = addparameter(model,'q_nCD8_P_out',params.q_nCD8_P_out.Value,'ValueUnits',params.q_nCD8_P_out.Units, 'Notes', 'rate of naive CD8+ T cell transport out of the peripheral compartment ');
p.UserData.Source = params.q_nCD8_P_out.Notes;
    p.UserData.derivedFrom = params.q_nCD8_P_out.derivedFrom;
p = addparameter(model,'Q_nCD8_thym',params.Q_nCD8_thym.Value,'ValueUnits',params.Q_nCD8_thym.Units, 'Notes', 'Thymic output of naive CD8+ T cells');
p.UserData.Source = params.Q_nCD8_thym.Notes;
    p.UserData.derivedFrom = params.Q_nCD8_thym.derivedFrom;
p = addparameter(model,'k_nCD8_pro',params.k_nCD8_pro.Value,'ValueUnits',params.k_nCD8_pro.Units, 'Notes', 'rate of naive CD8+ T cell proliferation ');
p.UserData.Source = params.k_nCD8_pro.Notes;
    p.UserData.derivedFrom = params.k_nCD8_pro.derivedFrom;
K_nCD8_pro = addparameter(model,'K_nCD8_pro',params.K_nT_pro.Value,'ValueUnits',params.K_nT_pro.Units, 'Notes', 'half-maximal peripheral proliferation of naive CD8+ T cells');
K_nCD8_pro.UserData.Source = params.K_nT_pro.Notes;
    K_nCD8_pro.UserData.derivedFrom = params.K_nT_pro.derivedFrom;
p = addparameter(model,'k_nCD8_death',params.k_nT_death.Value,'ValueUnits',params.k_nT_death.Units, 'Notes', 'rate of naive CD8+ T cell death ');
p.UserData.Source = params.k_nT_death.Notes;
    p.UserData.derivedFrom = params.k_nT_death.derivedFrom;

k_C_CD8_exh = addparameter(model,'k_C_CD8_exh',params.k_Tcell.Value,'ValueUnits',params.k_Tcell.Units, 'Notes', 'Rate of T cell exhaustion by cancer cells ');
k_C_CD8_exh.UserData.Source = params.k_Tcell.Notes;
    k_C_CD8_exh.UserData.derivedFrom = params.k_Tcell.derivedFrom;
k_C_CD8 = addparameter(model,'k_C_CD8',params.k_C_Tcell.Value,'ValueUnits',params.k_C_Tcell.Units, 'Notes', 'Rate of cancer cell death by T cells ');
k_C_CD8.UserData.Source = params.k_C_Tcell.Notes;
    k_C_CD8.UserData.derivedFrom = params.k_C_Tcell.derivedFrom;

p = addparameter(model,'k_stiff_CD8_exh',params.k_stiff_CD8_exh.Value,'ValueUnits',params.k_stiff_CD8_exh.Units, 'Notes', 'Rate of stiffness-mediated T cell exhaustion');
p.UserData.Source = params.k_stiff_CD8_exh.Notes;
    p.UserData.derivedFrom = params.k_stiff_CD8_exh.derivedFrom;
% Baseline exhaustion parameters (chronic antigen exposure)
p = addparameter(model,'k_CD8_basal_exh',params.k_CD8_basal_exh.Value,'ValueUnits',params.k_CD8_basal_exh.Units, 'Notes', 'Baseline exhaustion rate from chronic TCR stimulation');
p.UserData.Source = params.k_CD8_basal_exh.Notes;
    p.UserData.derivedFrom = params.k_CD8_basal_exh.derivedFrom;
p = addparameter(model,'K_C_exh',params.K_C_exh.Value,'ValueUnits',params.K_C_exh.Units, 'Notes', 'Cancer cell number for half-maximal exhaustion induction');
p.UserData.Source = params.K_C_exh.Notes;
    p.UserData.derivedFrom = params.K_C_exh.derivedFrom;
% Exhausted T cell death rate (prevents indefinite accumulation)
p = addparameter(model,'k_CD8_exh_death',params.k_CD8_exh_death.Value,'ValueUnits',params.k_CD8_exh_death.Units, 'Notes', 'Death rate of exhausted CD8 T cells');
p.UserData.Source = params.k_CD8_exh_death.Notes;
    p.UserData.derivedFrom = params.k_CD8_exh_death.derivedFrom;
% H_pore_T and H_stiff_Texh are defined in fibroblast_module_enhanced.m
% (forward references resolved by SimBiology at simulation time)

% CXCL12-mediated T cell exclusion (PDAC-specific)
H_CXCL12_Texcl = addparameter(model,'H_CXCL12_Texcl',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for CXCL12-mediated T cell exclusion');

% k_Treg_CD8_exh is already in the model (created by Treg_module)

% Antigen Default Hill Function
p = addparameter(model,['H_' antigen],1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', ['Hill function of tumor antigen ' antigen]);

% Add Reactions
% Thymic output of naive T cell
reaction = addreaction(model,'null -> V_C.nCD8');
    set(reaction,'ReactionRate','Q_nCD8_thym/div_CD8');
    set(reaction,'Notes','Thymic output of naive T cell to blood');
% Naive T cell proliferation in the peripheral compartment and the LN compartment
reaction = addreaction(model,'null -> V_P.nCD8');
    set(reaction,'ReactionRate','k_nCD8_pro/div_CD8*V_P.nCD8/(K_nCD8_pro/div_CD8+V_P.nCD8)');
    set(reaction,'Notes','Naive T cell proliferation in the peripheral compartment');
reaction = addreaction(model,'null -> V_LN.nCD8');
    set(reaction,'ReactionRate','k_nCD8_pro/div_CD8*V_LN.nCD8/(K_nCD8_pro/div_CD8+V_LN.nCD8)');
    set(reaction,'Notes','Naive T cell proliferation in the TDLN compartment');
% Naive T cell death
reaction = addreaction(model,'V_P.nCD8 -> null');
    set(reaction,'ReactionRate','k_nCD8_death*V_P.nCD8');
    set(reaction,'Notes','Naive T cell death in the peripheral compartment');
reaction = addreaction(model,'V_C.nCD8 -> null');
    set(reaction,'ReactionRate','k_nCD8_death*V_C.nCD8');
    set(reaction,'Notes','Naive T cell death in the central compartment');
reaction = addreaction(model,'V_LN.nCD8 -> null');
    set(reaction,'ReactionRate','k_nCD8_death*V_LN.nCD8');
    set(reaction,'Notes','Naive T cell death in the TDLN compartment');
% Naive T cell transport into and out of the peripheral compartment
reaction = addreaction(model,'V_C.nCD8 -> V_P.nCD8');
    set(reaction,'ReactionRate','q_nCD8_P_in*V_C.nCD8');
    set(reaction,'Notes','Naive T cell entry into the peripheral compartment');
reaction = addreaction(model,'V_P.nCD8 -> V_C.nCD8');
    set(reaction,'ReactionRate','q_nCD8_P_out*V_P.nCD8');
    set(reaction,'Notes','Naive T cell exit from the peripheral compartment');
% Naive T cell transport into and out of the lymph node
reaction = addreaction(model,'V_C.nCD8 -> V_LN.nCD8');
    set(reaction,'ReactionRate','q_nCD8_LN_in*V_C.nCD8');
    set(reaction,'Notes','Naive T cell entry into the lymph node');
reaction = addreaction(model,'V_LN.nCD8 -> V_C.nCD8');
    set(reaction,'ReactionRate','q_nCD8_LN_out*V_LN.nCD8');
    set(reaction,'Notes','Naive T cell exit from the lymph node');

% Naive T cell activation - mechanistic clonal expansion model
reaction = addreaction(model,'V_LN.nCD8 -> null');
    set(reaction,'ReactionRate',['k_T_activation_per_APC*V_LN.mcDC1*H_' antigen '*V_LN.nCD8']);
    set(reaction,'Notes','Naive CD8 T cell activation by mature cDC1 (cross-presentation)');
reaction = addreaction(model,'null -> V_LN.aCD8');
    set(reaction,'ReactionRate',['k_T_activation_per_APC*V_LN.mcDC1*H_' antigen '*V_LN.nCD8*n_T_expansion_burst*n_CD8_clones']);
    set(reaction,'Notes','Clonal expansion of activated T cells (mechanistic formulation)');
% Activated T cell proliferation
reaction = addreaction(model,'V_LN.aCD8 -> null');
    set(reaction,'ReactionRate','k_CD8_pro/N_aT*V_LN.aCD8');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);
reaction = addreaction(model,'null -> V_LN.CD8');
    set(reaction,'ReactionRate','k_CD8_pro/N_aT*2^(N_aT)*V_LN.aCD8');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);

% T cell Death
reaction = addreaction(model,'V_C.CD8 -> null');
    set(reaction,'ReactionRate','k_CD8_death*V_C.CD8');
    set(reaction,'Notes','T cell death in the central compartment');
reaction = addreaction(model,'V_P.CD8 -> null');
    set(reaction,'ReactionRate','k_CD8_death*V_P.CD8');
    set(reaction,'Notes','T cell death in the peripheral compartment');
reaction = addreaction(model,'V_LN.CD8 -> null');
    set(reaction,'ReactionRate','k_CD8_death*V_LN.CD8');
    set(reaction,'Notes','T cell death in the lymph node compartment');
reaction = addreaction(model,'V_T.CD8 -> null');
    set(reaction,'ReactionRate','k_CD8_death*V_T.CD8');
    set(reaction,'Notes','T cell death in the tumor compartment');
% Tumor-local CD8 proliferation (PDAC TME-driven, antigen-gated, capped by rho_CD8_max).
% Captures in-tumor TIL clonal expansion driven by antigen recognition.
% Gating: V_T.CD8 here is the live (non-exhausted) pool — exhausted cells already partition
% into V_T.CD8_exh via the existing exhaustion routes. So gating by (1-H_PD1_C1) would
% double-count exhaustion suppression. PD-1+ TIL proliferate readily (Ki67+) — PD-1
% suppresses effector killing, not cell division. Antigen presence (H_P1) is the
% physiologically primary gate. TGFb/Treg suppression also act via the exhaustion routes.
reaction = addreaction(model,'null -> V_T.CD8');
    set(reaction,'ReactionRate','k_CD8_T_pro*V_T.CD8*H_P1*(1 - V_T.CD8/(rho_CD8_max*V_T + cell))');
    set(reaction,'Notes','Tumor-local CD8 proliferation (antigen-gated, carrying-cap)');
% Baseline T cell exhaustion from chronic antigen exposure (PDL1-independent route).
% Gated by (1-H_PD1_C1) to avoid additive double-counting with the PDL1-mediated
% chronic-antigen exhaustion route below (k_C_CD8_exh × H_PD1_C1). Both routes
% are driven by the same chronic-antigen pool; the PDL1-dependent mechanism
% should *replace*, not add to, the PDL1-independent baseline as PD-1:PDL1
% engagement increases. (Inverse of the proliferation-side reasoning at lines
% 204-207 above, where gating by (1-H_PD1_C1) would *cause* double-counting.)
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_exh');
    set(reaction,'ReactionRate','k_CD8_basal_exh*V_T.CD8*C_total/(C_total+K_C_exh+cell)*(1-H_PD1_C1)');
    set(reaction,'Notes','Baseline T cell exhaustion from chronic antigen exposure (PDL1-indep; gated by 1-H_PD1_C1 to avoid double-counting with route 4 below)');

% T cell clearance upon Ag clearance
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_exh');
    set(reaction,'ReactionRate','k_cell_clear*V_T.CD8*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','T cell clearance upon antigen clearance');

% T cell exhaustion from Treg (Treg-mediated suppression -> exhausted phenotype)
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_exh');
    set(reaction,'ReactionRate','k_Treg_CD8_exh*V_T.CD8*Tregs_/(V_T.CD8+Tregs_+cell)');
    set(reaction,'Notes','T cell exhaustion from Tregs');
% T cell exhaustion from cancer (PD-1/PD-L1 mediated)
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_exh');
    set(reaction,'ReactionRate','k_C_CD8_exh*V_T.CD8*C_total/(C_total+V_T.CD8+cell)*H_PD1_C1');
    set(reaction,'Notes','T cell exhaustion from cancer (PD-1 mediated)');
% T cell exhaustion from ECM (desmoplasia-induced)
reaction = addreaction(model,'V_T.CD8 -> V_T.CD8_exh');
    set(reaction,'ReactionRate','k_stiff_CD8_exh*V_T.CD8*H_stiff_Texh');
    set(reaction,'Notes','T cell exhaustion from ECM');

% Exhausted T cell death (prevents indefinite accumulation)
reaction = addreaction(model,'V_T.CD8_exh -> null');
    set(reaction,'ReactionRate','k_CD8_exh_death*V_T.CD8_exh');
    set(reaction,'Notes','Death of exhausted CD8 T cells');

% T cell transport
% Central & Peripheral
reaction = addreaction(model,'V_C.CD8 -> V_P.CD8');
    set(reaction,'ReactionRate','q_CD8_P_in*V_C.CD8');
    set(reaction,'Notes','T cell transport into the peripheral compartment');
reaction = addreaction(model,'V_P.CD8 -> V_C.CD8');
    set(reaction,'ReactionRate','q_CD8_P_out*V_P.CD8');
    set(reaction,'Notes','T cell transport out of the peripheral compartment');
% Central & tumor (with CXCL12-mediated exclusion for PDAC)
reaction = addreaction(model,'V_C.CD8 -> V_T.CD8');
    set(reaction,'ReactionRate','q_CD8_T_in*V_T*V_C.CD8*(C_total^2/(C_total^2 + K_C2_recruit))*(1-H_CXCL12_Texcl)');
    set(reaction,'Notes','T cell transport into the tumor compartment (reduced by CXCL12-mediated exclusion)');
% Central & LN
reaction = addreaction(model,'V_LN.CD8 -> V_C.CD8');
    set(reaction,'ReactionRate','q_CD8_LN_out*V_LN.CD8');
    set(reaction,'Notes','T cell transport out of the lymph node compartment');

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

K_T_C = addparameter(model,'K_T_C',params.K_T_C.Value,'ValueUnits',params.K_T_C.Units, 'Notes', 'Dependence of Teff killing rate on Teff/C ratio ');
K_T_C.UserData.Source = params.K_T_C.Notes;
    K_T_C.UserData.derivedFrom = params.K_T_C.derivedFrom;

% R50_Treg: Treg:Teff ratio for 50% inhibition of killing (PRIMARY calibration parameter)
R50_Treg = addparameter(model,'R50_Treg',params.R50_Treg.Value,'ValueUnits',params.R50_Treg.Units, 'Notes', 'Treg:Teff ratio for 50% inhibition of T cell killing');
R50_Treg.UserData.Source = params.R50_Treg.Notes;
    R50_Treg.UserData.derivedFrom = {params.R50_Treg.Notes};

% K_T_Treg: Derived from R50_Treg for use in equations (K_T_Treg = 1/R50_Treg)
K_T_Treg = addparameter(model,'K_T_Treg',params.K_T_Treg.Value,'ValueUnits',params.K_T_Treg.Units,'ConstantValue',false, 'Notes', 'Treg inhibition factor (derived: 1/R50_Treg)');
K_T_Treg.UserData.Source = params.K_T_Treg.Notes;
    K_T_Treg.UserData.derivedFrom = params.K_T_Treg.derivedFrom;
addrule(model, 'K_T_Treg = 1/R50_Treg', 'repeatedAssignment');
% Mechanistic tumor killing parameters
p = addparameter(model,'v_T_search_volume',params.v_T_search_volume.Value,'ValueUnits',params.v_T_search_volume.Units, 'Notes', 'Volume searched by one T cell per day');
p.UserData.Source = params.v_T_search_volume.Notes;
    p.UserData.derivedFrom = params.v_T_search_volume.derivedFrom;
p = addparameter(model,'p_T_kill_per_contact',params.p_T_kill_per_contact.Value,'ValueUnits',params.p_T_kill_per_contact.Units, 'Notes', 'Kill probability per T cell-tumor contact');
p.UserData.Source = params.p_T_kill_per_contact.Notes;
    p.UserData.derivedFrom = params.p_T_kill_per_contact.derivedFrom;

% Effective T cell killing rate (computed parameter)
k_C_Tcell_eff = addparameter(model,'k_C_Tcell_eff',0,'ValueUnits','1/day','ConstantValue',false, 'Notes', 'Effective T cell killing rate including all inhibition factors');

% Add repeated assignment rule for effective T cell killing rate
% TODO: k_C_Tcell_eff conflates spatial access (pore exclusion), functional state
% (Treg/TGFb suppression), and synapse-level inhibition (PD-1) into one rate.
% Future: decompose into infiltration flux × contact rate × per-contact kill probability.
tcellEffRuleTxt = 'k_C_Tcell_eff = p_T_kill_per_contact * v_T_search_volume * V_T.CD8 / V_T / cell * (1-H_pore_T) * V_T.CD8/(V_T.CD8+K_T_Treg*Tregs_+cell) * (1-H_PD1_C1) * (1-H_TGFb_Teff)';

% Get existing rules to check for duplicates
R = sbioselect(model,'Type','rule');
didReplaceTcellEff = false;
for k = 1:numel(R)
    if contains(strrep(R(k).Rule,' ',''), 'k_C_Tcell_eff=')
        set(R(k),'Rule',tcellEffRuleTxt); didReplaceTcellEff = true;
    end
end
if ~didReplaceTcellEff
    addrule(model, tcellEffRuleTxt, 'repeatedAssignment');
end

% Update Cancer Killing by T Cells (Rule 5)
if (exist('cancer_types','var'))
    for i = 1:length(cancer_types)
        reaction = addreaction(model,['V_T.' cancer_types{i} ' -> V_T.C_x']);
            set(reaction,'ReactionRate',['k_C_Tcell_eff*V_T.' cancer_types{i}]);
            set(reaction,'Notes','Cancer cell killing by T cells');
        rule = get(model_rules(5),'Rule');
            set(model_rules(5),'Rule',[rule,'+(k_' cancer_types{i} '_death+k_' cancer_types{i} '_therapy)*V_T.' cancer_types{i} '+k_C_Tcell_eff*V_T.' cancer_types{i}]);
    end
end

warning('off','SimBiology:DimAnalysisNotDone_MatlabFcn_Dimensionless');
