% Treg Module
%
% Models Treg transport and activation by APCs [Use before antigen corresponding module]
%
% Inputs: model        -- SimBiology model object with four compartments
%         params       -- object containing model parameter Values, Units, and Notes:
%
% Outputs: model -- SimBiology model object with new Tcell module


function model = Treg_module(model,params)

% Species Names
species_name = 'Treg';
antigen = 'P0';

% Add Species (final names — no rename pattern)
% Naive T cells
nCD4_C  = addspecies(model.Compartment(1),'nCD4',params.nCD4_C.Value/params.nCD4_div.Value,'InitialAmountUnits','cell');
    set(nCD4_C,'Notes',['Number of naive ' species_name ' cells in the central compartment']);
nCD4_P  = addspecies(model.Compartment(2),'nCD4',params.nCD4_P.Value/params.nCD4_div.Value,'InitialAmountUnits','cell');
    set(nCD4_P,'Notes',['Number of naive ' species_name ' cells in the peripheral compartment']);
nCD4_LN = addspecies(model.Compartment(4),'nCD4',params.nCD4_LN.Value/params.nCD4_div.Value,'InitialAmountUnits','cell', 'Notes', ['Number of naive ' species_name ' cells in the lymph node']);
% Activated T cells
aTreg = addspecies(model.Compartment(4),'aTreg',0,'InitialAmountUnits','cell', 'Notes', ['Number of activated ' species_name ' cells in the lymph node']);
% Mature T cells
Treg_C = addspecies(model.Compartment(1),'Treg',0,'InitialAmountUnits','cell');
Treg_P = addspecies(model.Compartment(2),'Treg',0,'InitialAmountUnits','cell');
Treg_T = addspecies(model.Compartment(3),'Treg',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the tumor compartment']);
Treg_LN = addspecies(model.Compartment(4),'Treg',0,'InitialAmountUnits','cell', 'Notes', ['Number of ' species_name ' cells in the lymph node compartment']);

% Add Hill Functions for APC/mAPC
H_APC = 'H_APC_Treg';
% Add Treg Variable
addrule(model,'Tregs_ = V_T.Treg','repeatedAssignment');

% Add Parameters (final names — no rename pattern)
div_Treg = addparameter(model,'div_Treg',params.nCD4_div.Value,'ValueUnits',params.nCD4_div.Units, 'Notes', 'T Cell Diversity ');
div_Treg.UserData.Source = params.nCD4_div.Notes;
    div_Treg.UserData.derivedFrom = params.nCD4_div.derivedFrom;
n_CD4_clones = addparameter(model,'n_CD4_clones',params.n_clones_slf.Value,'ValueUnits',params.n_clones_slf.Units, 'Notes', 'Number of self-antigen-specific T cell clones ');
n_CD4_clones.UserData.Source = params.n_clones_slf.Notes;
    n_CD4_clones.UserData.derivedFrom = params.n_clones_slf.derivedFrom;
p = addparameter(model,'q_nCD4_LN_in',params.q_nCD4_LN_in.Value,'ValueUnits',params.q_nCD4_LN_in.Units, 'Notes', 'Rate of naive T cell transport into the LN ');
p.UserData.Source = params.q_nCD4_LN_in.Notes;
    p.UserData.derivedFrom = params.q_nCD4_LN_in.derivedFrom;
p = addparameter(model,'q_Treg_LN_out',params.q_CD4_LN_out.Value,'ValueUnits',params.q_CD4_LN_out.Units, 'Notes', 'Rate of activated T cell transport out of the LN ');
p.UserData.Source = params.q_CD4_LN_out.Notes;
    p.UserData.derivedFrom = params.q_CD4_LN_out.derivedFrom;
p = addparameter(model,'q_nCD4_LN_out',params.q_nCD4_LN_out.Value,'ValueUnits',params.q_nCD4_LN_out.Units, 'Notes', 'Rate of naive T cell transport out of the LN ');
p.UserData.Source = params.q_nCD4_LN_out.Notes;
    p.UserData.derivedFrom = params.q_nCD4_LN_out.derivedFrom;
p = addparameter(model,'k_Treg_act',params.k_nCD4_act.Value,'ValueUnits',params.k_nCD4_act.Units, 'Notes', 'Treg activation rate');
p.UserData.Source = params.k_nCD4_act.Notes;
    p.UserData.derivedFrom = params.k_nCD4_act.derivedFrom;
p = addparameter(model,'k_Treg_pro',params.k_Treg_pro.Value,'ValueUnits',params.k_Treg_pro.Units, 'Notes', 'Treg proliferation rate');
p.UserData.Source = params.k_Treg_pro.Notes;
    p.UserData.derivedFrom = params.k_Treg_pro.derivedFrom;
p = addparameter(model,'k_Treg_death',params.k_Treg_death.Value,'ValueUnits',params.k_Treg_death.Units, 'Notes', 'Treg death rate');
p.UserData.Source = params.k_Treg_death.Notes;
    p.UserData.derivedFrom = params.k_Treg_death.derivedFrom;
p = addparameter(model,'k_Treg_pro_tumor',params.k_Treg_pro_tumor.Value,...
    'ValueUnits',params.k_Treg_pro_tumor.Units,...
    'Notes', 'Tumor-local Treg proliferation rate (Ki67+ 30-60% in PDAC TME)');
p.UserData.Source = params.k_Treg_pro_tumor.Notes;
    p.UserData.derivedFrom = params.k_Treg_pro_tumor.derivedFrom;
p = addparameter(model,'rho_Treg_max',params.rho_Treg_max.Value,...
    'ValueUnits',params.rho_Treg_max.Units,...
    'Notes', 'Max Treg density in tumor (carrying capacity for local proliferation)');
p.UserData.Source = params.rho_Treg_max.Notes;
    p.UserData.derivedFrom = params.rho_Treg_max.derivedFrom;

% apCAF→Treg-cap pathway retired 2026-05-01: counterfactual SBI (apCAF rate
% pinned at ~0) showed the data prefers the apCAF-driven additional Treg
% carrying capacity off — D²_post/D²_prior improved 0.91 → 0.85, treg_density
% reachability rose, cd4_helper dropped out of LIKELY_MISSPECIFIED. The
% antigen-specific induced-Treg contribution via k_apCAF_iTreg (appended to
% the existing Th→Treg flux in fibroblast_module_enhanced) remains as the
% biologically supported apCAF→Treg coupling.
% CD8_50_Tcap (formerly registered here for the Th inflammation Hill) was
% RETIRED 2026-06-22 with that Hill — see Th_module route-2 fix.
p = addparameter(model,'q_Treg_P_in',params.q_CD4_P_in.Value,'ValueUnits',params.q_CD4_P_in.Units, 'Notes', 'rate of Treg transport into the peripheral compartment ');
p.UserData.Source = params.q_CD4_P_in.Notes;
    p.UserData.derivedFrom = params.q_CD4_P_in.derivedFrom;
p = addparameter(model,'q_Treg_P_out',params.q_CD4_P_out.Value,'ValueUnits',params.q_CD4_P_out.Units, 'Notes', 'rate of Treg transport out of the peripheral compartment ');
p.UserData.Source = params.q_CD4_P_out.Notes;
    p.UserData.derivedFrom = params.q_CD4_P_out.derivedFrom;
p = addparameter(model,'q_Treg_T_in',params.q_CD4_T_in.Value,'ValueUnits',params.q_CD4_T_in.Units, 'Notes', 'rate of Treg transport into the tumor compartment ');
p.UserData.Source = params.q_CD4_T_in.Notes;
    p.UserData.derivedFrom = params.q_CD4_T_in.derivedFrom;
p = addparameter(model,'q_nCD4_P_in',params.q_nCD4_P_in.Value,'ValueUnits',params.q_nCD4_P_in.Units, 'Notes', 'rate of naive Treg transport into the peripheral compartment ');
p.UserData.Source = params.q_nCD4_P_in.Notes;
    p.UserData.derivedFrom = params.q_nCD4_P_in.derivedFrom;
p = addparameter(model,'q_nCD4_P_out',params.q_nCD4_P_out.Value,'ValueUnits',params.q_nCD4_P_out.Units, 'Notes', 'rate of naive Treg transport out of the peripheral compartment');
p.UserData.Source = params.q_nCD4_P_out.Notes;
    p.UserData.derivedFrom = params.q_nCD4_P_out.derivedFrom;
p = addparameter(model,'Q_nCD4_thym',params.Q_nCD4_thym.Value,'ValueUnits',params.Q_nCD4_thym.Units, 'Notes', 'Thymic output of naive Treg cells');
p.UserData.Source = params.Q_nCD4_thym.Notes;
    p.UserData.derivedFrom = params.Q_nCD4_thym.derivedFrom;
p = addparameter(model,'k_nCD4_pro',params.k_nCD4_pro.Value,'ValueUnits',params.k_nCD4_pro.Units, 'Notes', 'rate of naive Treg proliferation ');
p.UserData.Source = params.k_nCD4_pro.Notes;
    p.UserData.derivedFrom = params.k_nCD4_pro.derivedFrom;
K_nTreg_pro = addparameter(model,'K_nTreg_pro',params.K_nT_pro.Value,'ValueUnits',params.K_nT_pro.Units, 'Notes', 'half-maximal peripheral proliferation of naive Treg');
K_nTreg_pro.UserData.Source = params.K_nT_pro.Notes;
    K_nTreg_pro.UserData.derivedFrom = params.K_nT_pro.derivedFrom;
p = addparameter(model,'k_nTreg_death',params.k_nT_death.Value,'ValueUnits',params.k_nT_death.Units, 'Notes', 'rate of naive Treg death ');
p.UserData.Source = params.k_nT_death.Notes;
    p.UserData.derivedFrom = params.k_nT_death.derivedFrom;

p = addparameter(model,'f_nTreg',params.f_nTreg.Value,'ValueUnits',params.f_nTreg.Units, 'Notes', 'Fraction of naturally-occurring Tregs');
p.UserData.Source = params.f_nTreg.Notes;
    p.UserData.derivedFrom = params.f_nTreg.derivedFrom;

k_Treg_CD8_exh = addparameter(model,'k_Treg_CD8_exh',params.k_Treg_CD8_exh.Value,'ValueUnits',params.k_Treg_CD8_exh.Units, 'Notes', params.k_Treg_CD8_exh.Notes);
k_Treg_CD8_exh.UserData.Source = params.k_Treg_CD8_exh.Notes;
    k_Treg_CD8_exh.UserData.derivedFrom = params.k_Treg_CD8_exh.derivedFrom;

% Antigen Default Hill Function
p = addparameter(model,['H_' antigen],1,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', ['Hill function of tumor antigen' antigen]);

% Add Reactions
% Thymic output of naive T cell
reaction = addreaction(model,'null -> V_C.nCD4');
    set(reaction,'ReactionRate','Q_nCD4_thym/div_Treg');
    set(reaction,'Notes','Thymic output of naive T cell to blood');
% Naive T cell proliferation in the peripheral compartment and the LN compartment
reaction = addreaction(model,'null -> V_P.nCD4');
    set(reaction,'ReactionRate','k_nCD4_pro/div_Treg*V_P.nCD4/(K_nTreg_pro/div_Treg+V_P.nCD4)');
    set(reaction,'Notes','Naive T cell proliferation in the peripheral compartment');
reaction = addreaction(model,'null -> V_LN.nCD4');
    set(reaction,'ReactionRate','k_nCD4_pro/div_Treg*V_LN.nCD4/(K_nTreg_pro/div_Treg+V_LN.nCD4)');
    set(reaction,'Notes','Naive T cell proliferation in the TDLN compartment');
% Naive T cell death
reaction = addreaction(model,'V_P.nCD4 -> null');
    set(reaction,'ReactionRate','k_nTreg_death*V_P.nCD4');
    set(reaction,'Notes','Naive T cell death in the peripheral compartment');
reaction = addreaction(model,'V_C.nCD4 -> null');
    set(reaction,'ReactionRate','k_nTreg_death*V_C.nCD4');
    set(reaction,'Notes','Naive T cell death in the central compartment');
reaction = addreaction(model,'V_LN.nCD4 -> null');
    set(reaction,'ReactionRate','k_nTreg_death*V_LN.nCD4');
    set(reaction,'Notes','Naive T cell death in the TDLN compartment');
% Naive T cell transport into and out of the peripheral compartment
reaction = addreaction(model,'V_C.nCD4 -> V_P.nCD4');
    set(reaction,'ReactionRate','q_nCD4_P_in*V_C.nCD4');
    set(reaction,'Notes','Naive T cell entry into the peripheral compartment');
reaction = addreaction(model,'V_P.nCD4 -> V_C.nCD4');
    set(reaction,'ReactionRate','q_nCD4_P_out*V_P.nCD4');
    set(reaction,'Notes','Naive T cell exit from the peripheral compartment');
% Naive T cell transport into and out of the lymph node
reaction = addreaction(model,'V_C.nCD4 -> V_LN.nCD4');
    set(reaction,'ReactionRate','q_nCD4_LN_in*V_C.nCD4');
    set(reaction,'Notes','Naive T cell entry into the lymph node');
reaction = addreaction(model,'V_LN.nCD4 -> V_C.nCD4');
    set(reaction,'ReactionRate','q_nCD4_LN_out*V_LN.nCD4');
    set(reaction,'Notes','Naive T cell exit from the lymph node');

% Naive T cell activation (antigen-dependent for induced Tregs)
reaction = addreaction(model,'V_LN.nCD4 -> null');
    set(reaction,'ReactionRate',['(1-f_nTreg)*k_Treg_act*' H_APC '*H_' antigen '*V_LN.nCD4']);
    set(reaction,'Notes','Naive T cell activation (antigen-dependent)');
reaction = addreaction(model,'null -> V_LN.aTreg');
    set(reaction,'ReactionRate',['(1-f_nTreg)*k_Treg_act*' H_APC '*H_' antigen '*V_LN.nCD4*n_CD4_clones']);
    set(reaction,'Notes','Naive T cell activation (antigen-dependent)');
% Constitutive activation for naturally occurring Tregs
reaction = addreaction(model,'V_LN.nCD4 -> null');
    set(reaction,'ReactionRate',['f_nTreg*k_Treg_act*' H_APC '*V_LN.nCD4']);
    set(reaction,'Notes','Constitutive Treg activation (antigen-independent)');
reaction = addreaction(model,'null -> V_LN.aTreg');
    set(reaction,'ReactionRate',['f_nTreg*k_Treg_act*' H_APC '*V_LN.nCD4*n_CD4_clones']);
    set(reaction,'Notes','Constitutive Treg activation (antigen-independent)');
% Activated T cell proliferation (homeostatic, not clonal expansion)
reaction = addreaction(model,'V_LN.aTreg -> null');
    set(reaction,'ReactionRate','k_Treg_pro*V_LN.aTreg');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);
reaction = addreaction(model,'null -> V_LN.Treg');
    set(reaction,'ReactionRate','k_Treg_pro*V_LN.aTreg');
    set(reaction,'Notes',['a' species_name ' cell proliferation']);

% T cell Death
reaction = addreaction(model,'V_C.Treg -> null');
    set(reaction,'ReactionRate','k_Treg_death*V_C.Treg');
    set(reaction,'Notes','T cell death in the central compartment');
reaction = addreaction(model,'V_P.Treg -> null');
    set(reaction,'ReactionRate','k_Treg_death*V_P.Treg');
    set(reaction,'Notes','T cell death in the peripheral compartment');
reaction = addreaction(model,'V_LN.Treg -> null');
    set(reaction,'ReactionRate','k_Treg_death*V_LN.Treg');
    set(reaction,'Notes','T cell death in the lymph node compartment');
% Tumor-local homeostatic Treg death. RESTORED 2026-07-13 (see below); it had been
% removed 2026-06-17 ("Proposal 1"), which structurally guaranteed the sign inversion
% documented in notes/model/tcell_carrying_capacity_inversion.md.
%
% WHY IT HAD TO COME BACK. With no tumoral death, the only loss balancing the (strictly
% positive) Treg influx -- trafficking q_Treg_T_in, Th->Treg conversion, apCAF-driven
% conversion -- is the logistic term itself. So the steady state REQUIRES
%   I + k_Treg_pro_tumor*Treg*(1 - Treg/cap) = 0   with I > 0
% which forces the logistic factor NEGATIVE, i.e. Treg > cap. ALWAYS, for EVERY value of
% rho_Treg_max. The pool could not sit below its own carrying capacity, so the renewal
% reaction ran permanently as a quadratic SINK whose strength scales with
% k_Treg_pro_tumor. Consequences, all confirmed by qsp_sim sweeps:
%   - 4x k_Treg_pro_tumor  -> 0.804x tumoral Treg   (MORE proliferation, FEWER Tregs)
%   - low-dose cyclophosphamide, whose whole purpose here is Treg depletion and which acts
%     by damping this very reaction (gvax_module appends *(1 - Emax_Cy_Treg*...)), came out
%     +23% Treg at Emax=0.9 -- damping a sink ADDS cells. The drug ran backwards.
%   - influx sensitivity was square-rooted (dln Treg/dln influx = 0.27), so the observed
%     nivo Treg response (2.64x) would have needed a 14.4x influx change.
%
% The 2026-06-17 rationale was that Treg* = cap*(1 - k_Treg_death/k_Treg_pro_tumor) is a
% near-cancellation of two comparable anchored rates (Ki67 k_Treg_pro_tumor=0.131+/-0.073;
% deuterium k_Treg_death=0.074+/-0.024) and was therefore scale-degenerate. That was a real
% problem -- but it is a property of the LOGISTIC form, not of the death term. With the
% saturating renewal below, there is no such knife-edge: the steady state stays finite,
% positive and monotone straight through k_Treg_pro_tumor == k_Treg_death (verified by
% sweep), because the influx regularizes it. So we get identifiability AND the right sign.
reaction = addreaction(model,'V_T.Treg -> null');
    set(reaction,'ReactionRate','k_Treg_death*V_T.Treg');
    set(reaction,'Notes','Tumor-local homeostatic Treg death (restored 2026-07-13). Bounds the pool so it can sit BELOW its half-saturation density; without it the renewal reaction is forced permanently negative and inverts. Shares the k_Treg_death deuterium anchor with the circulating compartments.');

% T cell clearance upon Ag clearance (only induced Tregs, not nTregs)
reaction = addreaction(model,'V_T.Treg -> null');
    set(reaction,'ReactionRate','(1-f_nTreg)*k_cell_clear*V_T.Treg*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Induced Treg clearance upon antigen clearance (nTregs persist)');

% Tumor-local Treg proliferation (PDAC TME-driven)
% Carrying capacity scales with apCAF density (Fix 2): apCAFs are the
% dedicated Treg-inducing CAF subtype (Elyada 2019, Huang 2022). This
% decouples Treg cap from the effector CD8 pool so that TGFb blockade
% → fewer iCAF->apCAF conversions → apCAF ↓ → Treg cap ↓ (correct
% direction), instead of TGFb blockade → more CD8 → higher Treg cap.
% V_T.apCAF is a forward reference to fibroblast_module_enhanced;
% SimBiology resolves it at simulation time.
reaction = addreaction(model,'null -> V_T.Treg');
    set(reaction,'ReactionRate',...
        'k_Treg_pro_tumor*V_T.Treg/(1 + V_T.Treg/(rho_Treg_max*V_T + cell))');
    set(reaction,'Notes','Tumor-local Treg renewal, SATURATING in density (rho_Treg_max*V_T is the half-saturation density, NOT a hard ceiling). Reformulated 2026-07-13 from the unclamped logistic, which -- with the tumoral death removed on 2026-06-17 -- was forced permanently negative and ran as a quadratic sink (see the death reaction above and notes/model/tcell_carrying_capacity_inversion.md). This form is monotone in k_Treg_pro_tumor and in influx at every density, so the Cy throttle that gvax_module appends here (*(1 - Emax_Cy_Treg*...)) now correctly REMOVES Tregs instead of adding them. The pool is bounded by k_Treg_death (restored above), not by the cap. k_Treg_pro_tumor keeps its Ki67 submodel anchor. Mechanism-specific losses remain explicit: Cy depletion (gvax_module) and post-eradication clearance (below). The apCAF-driven additive cap (rho_Treg_apCAF_extra*H_apCAF) was retired 2026-05-01; an antigen-specific induced-Treg contribution (k_apCAF_iTreg, fibroblast_module_enhanced) remains the residual apCAF→Treg coupling.');

% T cell transport
% Central & Peripheral
reaction = addreaction(model,'V_C.Treg -> V_P.Treg');
    set(reaction,'ReactionRate','q_Treg_P_in*V_C.Treg');
    set(reaction,'Notes','T cell transport into the peripheral compartment');
reaction = addreaction(model,'V_P.Treg -> V_C.Treg');
    set(reaction,'ReactionRate','q_Treg_P_out*V_P.Treg');
    set(reaction,'Notes','T cell transport out of the peripheral compartment');
% Central & tumor
reaction = addreaction(model,'V_C.Treg -> V_T.Treg');
    set(reaction,'ReactionRate','q_Treg_T_in*V_T*V_C.Treg*(C_total^2/(C_total^2 + K_C2_recruit))*H_vasc');
    set(reaction,'Notes','T cell transport into the tumor compartment');
% Central & LN
reaction = addreaction(model,'V_LN.Treg -> V_C.Treg');
    set(reaction,'ReactionRate','q_Treg_LN_out*V_LN.Treg');
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

warning('off','SimBiology:DimAnalysisNotDone_MatlabFcn_Dimensionless');
