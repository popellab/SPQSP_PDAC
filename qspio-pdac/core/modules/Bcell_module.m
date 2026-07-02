% B Cell Module (minimal humoral pool)
%
% Minimal B-lineage population for the coupled QSP+ABM PDAC model. The ABM owns
% the spatial antibody/ADCC loop (PDE + macrophage ADCC); the QSP only needs a
% systemic B pool that (a) is primed in the LN by tumor antigen (H_P1) with Tfh
% help, (b) traffics to the tumor to drive ABM B-cell recruitment (qsp_bcell_central,
% recruitment_mode 0), and (c) receives ABM recruitment feedback. No antibody
% species is added here — it would have no QSP-side consumer.
%
% Structure (mirrors the naive->activated->mature T-cell template):
%   naive nB (C, LN)  --marrow source, death, C<->LN traffic
%   -> aB (LN)        --antigen + Tfh-help activation, clonal expansion
%   -> B (C,P,T,LN)   --proliferation, death, C<->P, C<->T (C_total gate), LN->C
%   -> Plasma (T)     --tumor-local differentiation, death
%
% Reuses: k_nTreg_death (naive-T death), N_aTh, K_C2_recruit, H_P1, k_cell_clear,
%   q_nCD4_LN_in/out, q_nCD4_P_in/out (naive/mature traffic),
%   q_Treg_P_in/out, q_Treg_T_in, q_Treg_LN_out (mature B traffic).
% New parameters: k_nB_in, k_B_act, n_B_clones, k_B_pro, k_B_death,
%   k_B_to_plasma, k_plasma_death, Tfh_50_B.
%
% Depends on Tfh_module (uses V_LN.Tfh) — call after Tfh_module.

function model = Bcell_module(model,params)

% Add Species
% Naive B cells (blood + LN)
addspecies(model.Compartment(1),'nB',params.nB_C.Value,'InitialAmountUnits','cell', 'Notes','Number of naive B cells in the central compartment');
addspecies(model.Compartment(4),'nB',params.nB_LN.Value,'InitialAmountUnits','cell', 'Notes','Number of naive B cells in the lymph node');
% Activated B cells (LN)
addspecies(model.Compartment(4),'aB',0,'InitialAmountUnits','cell', 'Notes','Number of activated B cells in the lymph node');
% Mature B cells across compartments
addspecies(model.Compartment(1),'B',0,'InitialAmountUnits','cell', 'Notes','Number of B cells in the central compartment');
addspecies(model.Compartment(2),'B',0,'InitialAmountUnits','cell', 'Notes','Number of B cells in the peripheral compartment');
addspecies(model.Compartment(3),'B',0,'InitialAmountUnits','cell', 'Notes','Number of B cells in the tumor compartment');
addspecies(model.Compartment(4),'B',0,'InitialAmountUnits','cell', 'Notes','Number of B cells in the lymph node compartment');
% Plasma cells (tumor-local differentiated state)
addspecies(model.Compartment(3),'Plasma',0,'InitialAmountUnits','cell', 'Notes','Number of plasma cells in the tumor compartment');

% Add Parameters
p = addparameter(model,'k_nB_in',params.k_nB_in.Value,'ValueUnits',params.k_nB_in.Units, 'Notes','Marrow output of naive B cells to blood');
p.UserData.Source = params.k_nB_in.Notes;  p.UserData.derivedFrom = params.k_nB_in.derivedFrom;
p = addparameter(model,'k_B_act',params.k_B_act.Value,'ValueUnits',params.k_B_act.Units, 'Notes','Naive B activation rate (antigen + Tfh help)');
p.UserData.Source = params.k_B_act.Notes;  p.UserData.derivedFrom = params.k_B_act.derivedFrom;
p = addparameter(model,'n_B_clones',params.n_B_clones.Value,'ValueUnits',params.n_B_clones.Units, 'Notes','Number of antigen-specific B cell clones (clonal expansion)');
p.UserData.Source = params.n_B_clones.Notes;  p.UserData.derivedFrom = params.n_B_clones.derivedFrom;
p = addparameter(model,'k_B_pro',params.k_B_pro.Value,'ValueUnits',params.k_B_pro.Units, 'Notes','Activated B proliferation rate');
p.UserData.Source = params.k_B_pro.Notes;  p.UserData.derivedFrom = params.k_B_pro.derivedFrom;
p = addparameter(model,'k_B_death',params.k_B_death.Value,'ValueUnits',params.k_B_death.Units, 'Notes','Mature B cell death rate');
p.UserData.Source = params.k_B_death.Notes;  p.UserData.derivedFrom = params.k_B_death.derivedFrom;
p = addparameter(model,'k_B_to_plasma',params.k_B_to_plasma.Value,'ValueUnits',params.k_B_to_plasma.Units, 'Notes','Tumor-local B -> plasma differentiation rate');
p.UserData.Source = params.k_B_to_plasma.Notes;  p.UserData.derivedFrom = params.k_B_to_plasma.derivedFrom;
p = addparameter(model,'k_plasma_death',params.k_plasma_death.Value,'ValueUnits',params.k_plasma_death.Units, 'Notes','Plasma cell death rate');
p.UserData.Source = params.k_plasma_death.Notes;  p.UserData.derivedFrom = params.k_plasma_death.derivedFrom;
p = addparameter(model,'Tfh_50_B',params.Tfh_50_B.Value,'ValueUnits',params.Tfh_50_B.Units, 'Notes','Half-maximal LN Tfh number for B-cell help');
p.UserData.Source = params.Tfh_50_B.Notes;  p.UserData.derivedFrom = params.Tfh_50_B.derivedFrom;

% Add Reactions
% Naive B homeostasis (marrow source, death, C<->LN traffic)
reaction = addreaction(model,'null -> V_C.nB');
    set(reaction,'ReactionRate','k_nB_in');
    set(reaction,'Notes','Marrow output of naive B cells to blood');
reaction = addreaction(model,'V_C.nB -> null');
    set(reaction,'ReactionRate','k_nTreg_death*V_C.nB');
    set(reaction,'Notes','Naive B cell death in the central compartment');
reaction = addreaction(model,'V_LN.nB -> null');
    set(reaction,'ReactionRate','k_nTreg_death*V_LN.nB');
    set(reaction,'Notes','Naive B cell death in the lymph node');
reaction = addreaction(model,'V_C.nB -> V_LN.nB');
    set(reaction,'ReactionRate','q_nCD4_LN_in*V_C.nB');
    set(reaction,'Notes','Naive B cell entry into the lymph node');
reaction = addreaction(model,'V_LN.nB -> V_C.nB');
    set(reaction,'ReactionRate','q_nCD4_LN_out*V_LN.nB');
    set(reaction,'Notes','Naive B cell exit from the lymph node');

% Naive B activation (antigen-dependent + Tfh help; consumes nB in LN)
reaction = addreaction(model,'V_LN.nB -> null');
    set(reaction,'ReactionRate','k_B_act*H_P1*(V_LN.Tfh/(V_LN.Tfh+Tfh_50_B))*V_LN.nB');
    set(reaction,'Notes','Naive B activation (antigen + Tfh help)');
reaction = addreaction(model,'null -> V_LN.aB');
    set(reaction,'ReactionRate','k_B_act*H_P1*(V_LN.Tfh/(V_LN.Tfh+Tfh_50_B))*V_LN.nB*n_B_clones');
    set(reaction,'Notes','Naive B activation (antigen + Tfh help, clonal expansion)');
% Activated B proliferation -> mature B (in LN)
reaction = addreaction(model,'V_LN.aB -> null');
    set(reaction,'ReactionRate','(k_B_pro/N_aTh)*V_LN.aB');
    set(reaction,'Notes','Activated B proliferation');
reaction = addreaction(model,'null -> V_LN.B');
    set(reaction,'ReactionRate','(k_B_pro/N_aTh)*2^N_aTh*V_LN.aB');
    set(reaction,'Notes','Activated B proliferation (clonal burst)');

% Mature B death (all compartments)
reaction = addreaction(model,'V_C.B -> null');
    set(reaction,'ReactionRate','k_B_death*V_C.B');
    set(reaction,'Notes','B cell death in the central compartment');
reaction = addreaction(model,'V_P.B -> null');
    set(reaction,'ReactionRate','k_B_death*V_P.B');
    set(reaction,'Notes','B cell death in the peripheral compartment');
reaction = addreaction(model,'V_LN.B -> null');
    set(reaction,'ReactionRate','k_B_death*V_LN.B');
    set(reaction,'Notes','B cell death in the lymph node');
reaction = addreaction(model,'V_T.B -> null');
    set(reaction,'ReactionRate','k_B_death*V_T.B');
    set(reaction,'Notes','B cell death in the tumor compartment');

% Tumor-local B -> Plasma differentiation and plasma death/clearance
reaction = addreaction(model,'V_T.B -> V_T.Plasma');
    set(reaction,'ReactionRate','k_B_to_plasma*V_T.B');
    set(reaction,'Notes','Tumor-local B -> plasma differentiation');
reaction = addreaction(model,'V_T.Plasma -> null');
    set(reaction,'ReactionRate','k_plasma_death*V_T.Plasma');
    set(reaction,'Notes','Plasma cell death in the tumor compartment');
reaction = addreaction(model,'V_T.Plasma -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.Plasma*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Plasma clearance upon antigen clearance');

% Mature B transport (reuses q_Treg_*/q_nCD4_* rate constants)
% Central & Peripheral
reaction = addreaction(model,'V_C.B -> V_P.B');
    set(reaction,'ReactionRate','q_Treg_P_in*V_C.B');
    set(reaction,'Notes','B cell transport into the peripheral compartment');
reaction = addreaction(model,'V_P.B -> V_C.B');
    set(reaction,'ReactionRate','q_Treg_P_out*V_P.B');
    set(reaction,'Notes','B cell transport out of the peripheral compartment');
% Central & tumor (C_total-gated recruitment, same form as T cells)
reaction = addreaction(model,'V_C.B -> V_T.B');
    set(reaction,'ReactionRate','q_Treg_T_in*V_T*V_C.B*(C_total^2/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','B cell transport into the tumor compartment');
% Central & LN
reaction = addreaction(model,'V_LN.B -> V_C.B');
    set(reaction,'ReactionRate','q_Treg_LN_out*V_LN.B');
    set(reaction,'Notes','B cell transport out of the lymph node');

warning('off','SimBiology:DimAnalysisNotDone_MatlabFcn_Dimensionless');
