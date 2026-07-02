% T-cell Cytokine Module
%
% Centralizes all T-cell related cytokine species, parameters, reactions,
% and activation generation rules.
%
% Assembly position: After Th_module, before MDSC_module.
% All T-cell species (nCD8, aCD8, CD8, nCD4, aTreg, Treg, aTh, Th) exist.
%
% Reactions are split into two sections:
%
%   CLONE-SPECIFIC — Processes performed only by the antigen-specific clone
%     tracked in the model (e.g., IL2/IFNg secretion upon TCR activation).
%
%   SHARED — Consumption, degradation, and constitutive secretion reactions
%     centralized from the individual T-cell modules. Rate expressions match
%     the original implementations exactly (no diversity scaling).
%
% NOTE: Diversity scaling (×div_CD8/div_Treg) on shared reactions is a
% candidate structural change for a future commit.
%
% Inputs: model  -- SimBiology model object with T-cell species already added
%         params -- object containing model parameter Values, Units, and Notes
%
% Outputs: model -- SimBiology model with cytokine reactions added

function model = cytokine_tcell_module(model, params)

%% Species
IL2 = addspecies(model.Compartment(4), 'IL2', 1.9e-4, 'InitialAmountUnits', 'nanomolarity');
    set(IL2, 'Notes', 'Concentration of IL2 in the lymph node compartment');
IFNg = addspecies(model.Compartment(3), 'IFNg', 0, 'InitialAmountUnits', 'nanomolarity');
    set(IFNg, 'Notes', 'IFNg in the tumor compartment');
TGFb = addspecies(model.Compartment(3), 'TGFb', 0, 'InitialAmountUnits', 'nanomolarity');
    set(TGFb, 'Notes', 'TGFb in the tumor compartment');

%% Parameters — IL2
p = addparameter(model, 'k_IL2_deg', params.k_IL2_deg.Value, 'ValueUnits', params.k_IL2_deg.Units, 'Notes', 'rate of IL2 degradation ');
p.UserData.Source = params.k_IL2_deg.Notes;
    p.UserData.derivedFrom = params.k_IL2_deg.derivedFrom;
p = addparameter(model, 'k_IL2_sec', params.k_IL2_sec.Value, 'ValueUnits', params.k_IL2_sec.Units, 'Notes', 'rate of IL2 secretion per activated T cell ');
p.UserData.Source = params.k_IL2_sec.Notes;
    p.UserData.derivedFrom = params.k_IL2_sec.derivedFrom;
p = addparameter(model, 'k_IL2_cons', params.k_IL2_cons.Value, 'ValueUnits', params.k_IL2_cons.Units, 'Notes', 'rate of IL2 consumption per T cell (all clones) ');
p.UserData.Source = params.k_IL2_cons.Notes;
    p.UserData.derivedFrom = params.k_IL2_cons.derivedFrom;
p = addparameter(model, 'IL2_50', params.IL2_50.Value, 'ValueUnits', params.IL2_50.Units, 'Notes', 'T cell activation half-maximal IL2 concentration ');
p.UserData.Source = params.IL2_50.Notes;
    p.UserData.derivedFrom = params.IL2_50.derivedFrom;
p = addparameter(model, 'IL2_50_Treg', params.IL2_50_Treg.Value, 'ValueUnits', params.IL2_50_Treg.Units, 'Notes', 'Treg activation half-maximal IL2 concentration ');
p.UserData.Source = params.IL2_50_Treg.Notes;
    p.UserData.derivedFrom = params.IL2_50_Treg.derivedFrom;
p = addparameter(model, 'w_IL2_CD8', params.w_IL2_CD8.Value, 'ValueUnits', params.w_IL2_CD8.Units, 'Notes', 'IL-2 consumption weight for CD8 cells ');
    p.UserData.derivedFrom = params.w_IL2_CD8.derivedFrom;
p = addparameter(model, 'w_IL2_Treg', params.w_IL2_Treg.Value, 'ValueUnits', params.w_IL2_Treg.Units, 'Notes', 'IL-2 consumption weight for Treg cells ');
    p.UserData.derivedFrom = params.w_IL2_Treg.derivedFrom;

%% Parameters — Activation generations
p = addparameter(model, 'N_div_base', params.N_div_base.Value, 'ValueUnits', params.N_div_base.Units, 'Notes', 'number of activated T cell generations by TCR signaling only');
p.UserData.Source = params.N_div_base.Notes;
    p.UserData.derivedFrom = params.N_div_base.derivedFrom;
p = addparameter(model, 'N_div_costim', params.N_div_costim.Value, 'ValueUnits', params.N_div_costim.Units, 'Notes', 'number of activated T cell generations by co-stimulatory signaling only ');
p.UserData.Source = params.N_div_costim.Notes;
    p.UserData.derivedFrom = params.N_div_costim.derivedFrom;
p = addparameter(model, 'N_IL2_CD8', params.N_IL2_CD8.Value, 'ValueUnits', params.N_IL2_CD8.Units, 'Notes', 'maximum number of activated T cell generations due to IL2 (CD8) ');
p.UserData.Source = params.N_IL2_CD8.Notes;
    p.UserData.derivedFrom = params.N_IL2_CD8.derivedFrom;
p = addparameter(model, 'N_IL2_CD4', params.N_IL2_CD4.Value, 'ValueUnits', params.N_IL2_CD4.Units, 'Notes', 'maximum number of activated T cell generations due to IL2 (CD4) ');
p.UserData.Source = params.N_IL2_CD4.Notes;
    p.UserData.derivedFrom = params.N_IL2_CD4.derivedFrom;

%% Parameters — IFNg
p = addparameter(model, 'k_IFNg_Tsec', params.k_IFNg_Tsec.Value, 'ValueUnits', params.k_IFNg_Tsec.Units, 'Notes', 'rate of IFNg secretion per CD8 T cell in tumor ');
p.UserData.Source = params.k_IFNg_Tsec.Notes;
    p.UserData.derivedFrom = params.k_IFNg_Tsec.derivedFrom;
p = addparameter(model, 'k_IFNg_Thsec', params.k_IFNg_Thsec.Value, 'ValueUnits', params.k_IFNg_Thsec.Units, 'Notes', 'rate of IFNg secretion per Th cell in tumor ');
p.UserData.Source = params.k_IFNg_Thsec.Notes;
    p.UserData.derivedFrom = params.k_IFNg_Thsec.derivedFrom;
p = addparameter(model, 'k_IFNg_deg', params.k_IFNg_deg.Value, 'ValueUnits', params.k_IFNg_deg.Units, 'Notes', 'IFNg degradation rate ');
p.UserData.Source = params.k_IFNg_deg.Notes;
    p.UserData.derivedFrom = params.k_IFNg_deg.derivedFrom;
p = addparameter(model, 'IFNg_50_ind', params.IFNg_50_ind.Value, 'ValueUnits', params.IFNg_50_ind.Units, 'Notes', 'Half-Maximal IFNg level for PD-L1 induction ');
p.UserData.Source = params.IFNg_50_ind.Notes;
    p.UserData.derivedFrom = params.IFNg_50_ind.derivedFrom;

%% Parameters — TGFb
p = addparameter(model, 'k_TGFb_Tsec', params.k_TGFb_Tsec.Value, 'ValueUnits', params.k_TGFb_Tsec.Units, 'Notes', 'TGFb secretion rate per Treg cell (all clones) ');
p.UserData.Source = params.k_TGFb_Tsec.Notes;
    p.UserData.derivedFrom = params.k_TGFb_Tsec.derivedFrom;
p = addparameter(model, 'k_TGFb_deg', params.k_TGFb_deg.Value, 'ValueUnits', params.k_TGFb_deg.Units, 'Notes', 'TGFb degradation rate ');
p.UserData.Source = params.k_TGFb_deg.Notes;
    p.UserData.derivedFrom = params.k_TGFb_deg.derivedFrom;
p = addparameter(model, 'TGFb_baseline', params.TGFb_baseline.Value, 'ValueUnits', params.TGFb_baseline.Units, 'Notes', 'Baseline TGFb level in PDAC tumor ');
p.UserData.Source = params.TGFb_baseline.Notes;
    p.UserData.derivedFrom = params.TGFb_baseline.derivedFrom;
p = addparameter(model, 'TGFb_50', params.TGFb_50.Value, 'ValueUnits', params.TGFb_50.Units, 'Notes', 'Half-Maximal TGFb level for Th-to-Treg differentiation / M1-to-M2 polarization ');
p.UserData.Source = params.TGFb_50.Notes;
    p.UserData.derivedFrom = params.TGFb_50.derivedFrom;
p = addparameter(model, 'TGFb_50_Teff', params.TGFb_50_Teff.Value, 'ValueUnits', params.TGFb_50_Teff.Units, 'Notes', 'Half-Maximal TGFb level for CD8 T cell inhibition ');
p.UserData.Source = params.TGFb_50_Teff.Notes;
    p.UserData.derivedFrom = params.TGFb_50_Teff.derivedFrom;
p = addparameter(model, 'TGFb_50_Th_exh', params.TGFb_50_Th_exh.Value, 'ValueUnits', params.TGFb_50_Th_exh.Units, 'Notes', 'Half-Maximal TGFb level for CD4 helper T cell exhaustion');
p.UserData.Source = params.TGFb_50_Th_exh.Notes;
    p.UserData.derivedFrom = params.TGFb_50_Th_exh.derivedFrom;
% Bioactive fraction scalar: scales every total-TGFb secretion rate into the
% bioactive pool represented by V_T.TGFb. Lets the EC50 family (calibrated
% against bioactive recombinant TGFb) act consistently on the V_T.TGFb
% species without renaming or splitting.
p = addparameter(model, 'f_active_TGFb', params.f_active_TGFb.Value, 'ValueUnits', params.f_active_TGFb.Units, 'Notes', 'Bioactive fraction of secreted TGFb');
p.UserData.Source = params.f_active_TGFb.Notes;
    p.UserData.derivedFrom = params.f_active_TGFb.derivedFrom;

%% Rules — Activation generations
p = addparameter(model, 'N_aT', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Number of Activated CD8+ T Cell Generations (see Rules)');
addrule(model, 'N_aT = N_div_base + N_div_costim*H_CD28_APC + N_IL2_CD8*V_LN.IL2/(IL2_50+V_LN.IL2)', 'repeatedAssignment');

p = addparameter(model, 'N_aT0', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Number of Activated Treg Generations');
addrule(model, 'N_aT0 = N_div_base + N_div_costim*H_CD28_APC + N_IL2_CD4*V_LN.IL2/(IL2_50+V_LN.IL2)', 'repeatedAssignment');

p = addparameter(model, 'N_aTh', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Number of Activated T Helper Cell Generations (see Rules)');
addrule(model, 'N_aTh = N_div_base + N_div_costim*H_CD28_APC + N_IL2_CD4*V_LN.IL2/(IL2_50+V_LN.IL2)', 'repeatedAssignment');

%% Rules — TGFb Hill functions
p = addparameter(model, 'H_TGFb', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Hill function of TGFb for Th-to-Treg differentiation / chemoresistance development / M1-to-M2 polarization');
addrule(model, 'H_TGFb = V_T.TGFb/(V_T.TGFb+TGFb_50)', 'repeatedAssignment');

p = addparameter(model, 'H_TGFb_Teff', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Hill function of TGFb for Teff inhibition and chemoresistance development');
addrule(model, 'H_TGFb_Teff = V_T.TGFb/(V_T.TGFb+TGFb_50_Teff)', 'repeatedAssignment');

p = addparameter(model, 'H_TGFb_Th_exh', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false, 'Notes', 'Hill function of TGFb for CD4 helper T cell exhaustion (PD-1+, dysfunction)');
addrule(model, 'H_TGFb_Th_exh = V_T.TGFb/(V_T.TGFb+TGFb_50_Th_exh)', 'repeatedAssignment');

%% ========================================================================
%  CLONE-SPECIFIC REACTIONS
%  Processes performed only by the antigen-specific clone tracked in the
%  model. These cells were activated by specific antigen presentation and
%  secrete cytokines as part of their effector program. Rate expressions use
%  clone-level species counts directly — no diversity scaling.
%  ========================================================================

% IL2 secretion by activated CD8 T cells (tumor-antigen-specific clone)
reaction = addreaction(model, 'null -> V_LN.IL2');
    set(reaction, 'ReactionRate', 'k_IL2_sec*V_LN.aCD8');
    set(reaction, 'Notes', 'IL2 secretion from activated CD8 T cells (clone-specific, TCR-triggered)');
% IL2 secretion by activated Th cells (tumor-antigen-specific clone)
reaction = addreaction(model, 'null -> V_LN.IL2');
    set(reaction, 'ReactionRate', 'k_IL2_sec*V_LN.aTh');
    set(reaction, 'Notes', 'IL2 secretion from activated Th cells (clone-specific, TCR-triggered)');
% IFNg secretion by CD8 T cells in tumor (tumor-reactive effectors)
reaction = addreaction(model, 'null -> V_T.IFNg');
    set(reaction, 'ReactionRate', 'k_IFNg_Tsec*V_T.CD8');
    set(reaction, 'Notes', 'IFNg secretion from CD8 T cells in tumor (clone-specific effectors)');
% IFNg secretion by Th cells in tumor (tumor-reactive Th1 cells)
reaction = addreaction(model, 'null -> V_T.IFNg');
    set(reaction, 'ReactionRate', 'k_IFNg_Thsec*V_T.Th');
    set(reaction, 'Notes', 'IFNg secretion from Th cells in tumor (clone-specific effectors)');

%% ========================================================================
%  SHARED REACTIONS
%  Cytokine production, consumption, and degradation reactions centralized
%  from the individual T-cell modules. Rate expressions match the original
%  module implementations exactly (no diversity scaling).
%
%  NOTE: Diversity scaling (multiplying by div_CD8/div_Treg to convert
%  clone-level counts to total cell counts) is a candidate structural
%  change for a future commit. It would require simultaneous parameter
%  rescaling to maintain behavioral equivalence.
%  ========================================================================

% IL2 degradation (concentration-dependent, no cell counts)
reaction = addreaction(model, 'V_LN.IL2 -> null');
    set(reaction, 'ReactionRate', 'k_IL2_deg*V_LN.IL2*V_LN');
    set(reaction, 'Notes', 'IL2 degradation');
% IL2 consumption by CD8 T cells
reaction = addreaction(model, 'V_LN.IL2 -> null');
    set(reaction, 'ReactionRate', 'w_IL2_CD8*k_IL2_cons*V_LN.CD8*V_LN.IL2/(IL2_50+V_LN.IL2)');
    set(reaction, 'Notes', 'IL2 consumption by CD8 T cells');
% IL2 consumption by Tregs
reaction = addreaction(model, 'V_LN.IL2 -> null');
    set(reaction, 'ReactionRate', 'w_IL2_Treg*k_IL2_cons*V_LN.Treg*V_LN.IL2/(IL2_50_Treg+V_LN.IL2)');
    set(reaction, 'Notes', 'IL2 consumption by Tregs');
% IFNg degradation (concentration-dependent, no cell counts)
reaction = addreaction(model, 'V_T.IFNg -> null');
    set(reaction, 'ReactionRate', 'k_IFNg_deg*V_T.IFNg');
    set(reaction, 'Notes', 'IFNg degradation in tumor');

% V_T.TGFb represents the bioactive-equivalent TGFb pool. Every
% secretion-rate parameter anchored to total-TGFb measurements (Ivanović
% 2003, Liyanage 2002 via Yang 2013, Matsuo 2000, Bachem 1992) is scaled
% by f_active_TGFb so the EC50 family (calibrated against bioactive
% recombinant TGFb) acts on a physically consistent species.

% TGFb baseline bioactive production by cancer cells (drives V_T.TGFb
% toward f_active_TGFb * TGFb_baseline). TGFb_baseline is the literature
% total; scaling by f_active converts to bioactive-equivalent baseline.
reaction = addreaction(model, 'null -> V_T.TGFb');
    set(reaction, 'ReactionRate', 'k_TGFb_deg*(f_active_TGFb*TGFb_baseline - V_T.TGFb)*V_T');
    set(reaction, 'Notes', 'Bioactive TGFb baseline production by PDAC cancer cells (scaled by f_active_TGFb).');
% TGFb secretion by Tregs (bioactive fraction of total Liyanage 2002 rate)
reaction = addreaction(model, 'null -> V_T.TGFb');
    set(reaction, 'ReactionRate', 'f_active_TGFb*k_TGFb_Tsec*V_T.Treg');
    set(reaction, 'Notes', 'Bioactive TGFb secretion by Tregs in tumor (scaled by f_active_TGFb).');

end
