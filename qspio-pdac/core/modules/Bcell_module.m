% B Cell Module (humoral immunity + antibody/ADCC)
%
% B-lineage population and tumor-local antibody for the coupled QSP+ABM PDAC
% model. The ABM owns the spatial antibody/ADCC loop (PDE diffusion, per-cell
% Fc effects); the QSP provides the well-mixed antibody concentration that
% anchors ABM ADCC parameters and cross-validates bulk-level dynamics.
%
% TUMOR COMPARTMENT mirrors ABM agent states (b_cell.cuh):
%   V_T.nB          = BCELL_NAIVE      — recruited from blood, antigen-naive in tumor
%   V_T.aB          = BCELL_ACTIVATED  — locally activated by antigen, can proliferate
%   V_T.Plasma      = BCELL_PLASMA     — Tfh-helped terminal differentiation (Ab-secreting)
%   V_T.nBreg       = naive Breg       — TGFb+IL10-switched nB (reversible)
%   V_T.aBreg       = activated Breg   — TGFb+IL10-switched aB (reversible, IL-10 secreting)
%   (Plasma_Breg removed — Mirlekar 2022 found no IL-10/IL-35 from PDAC plasma cells)
%   V_T.Ab          = endogenous polyclonal IgG (nanomolarity)
%
% Structure (mirrors the naive->activated->mature T-cell template):
%   naive nB (C, LN)  --marrow source, death, C<->LN traffic
%   -> aB (LN)        --antigen + Tfh-help activation, clonal expansion
%   -> B (C,P,LN)     --proliferation, death, C<->P, LN->C
%   -> V_T.nB         --C->T trafficking (C_total gated), arrives as naive
%   -> V_T.aB         --tumor-local activation (antigen H_P1 gated)
%   -> V_T.Plasma     --Tfh-helped differentiation (V_T.Tfh Hill)
%   -> V_T.Ab         --plasma cell secretion, first-order decay
%
% Antibody consumers (defined in other modules, use H_Ab Hill from here):
%   macrophage_module: ADCC boost on M1 phagocytosis  (1 + ADCC_boost * H_Ab)
%   antigen_module:    immune-complex Fc-receptor boost on DC antigen uptake
%
% Reuses: N_aTh, K_C2_recruit, H_P1, k_cell_clear, H_CXCL12_Texcl, H_vasc.
% Own transport (no longer shared with Treg/nCD4):
%   k_nB_death, q_nB_LN_in/out (naive B traffic),
%   q_B_P_in/out, q_B_LN_out, q_B_T_in (mature B traffic).
% Own parameters: k_nB_in, k_B_act, n_B_clones, k_B_pro, k_B_death,
%   k_B_to_plasma, k_plasma_death, Tfh_50_B (LN activation help),
%   k_B_act_tumor, k_B_pro_tumor, rho_B_max (tumor carrying capacity),
%   Tfh_50_plasma (tumor Tfh help for plasma),
%   k_Ab_sec, k_Ab_deg, Ab_50_ADCC, ADCC_boost_max, Ab_50_DC, DC_Ab_boost,
%   k_Breg_on, k_Breg_off, TGFb_50_Breg, IL10_50_Breg, k_IL10_sec_Breg.
%   Breg aB CANNOT differentiate to plasma (must revert to aB first).
%
% Depends on Tfh_module (uses V_LN.Tfh, V_T.Tfh) — call after Tfh_module.

function model = Bcell_module(model,params)

% Add Species
% Naive B cells (blood + LN)
addspecies(model.Compartment(1),'nB',params.nB_C.Value,'InitialAmountUnits','cell', 'Notes','Number of naive B cells in the central compartment');
addspecies(model.Compartment(4),'nB',params.nB_LN.Value,'InitialAmountUnits','cell', 'Notes','Number of naive B cells in the lymph node');
% Activated B cells (LN)
addspecies(model.Compartment(4),'aB',0,'InitialAmountUnits','cell', 'Notes','Number of activated B cells in the lymph node');
% Mature B cells (C, P, LN — NOT tumor; tumor has nB/aB states)
addspecies(model.Compartment(1),'B',0,'InitialAmountUnits','cell', 'Notes','Number of B cells in the central compartment');
addspecies(model.Compartment(2),'B',0,'InitialAmountUnits','cell', 'Notes','Number of B cells in the peripheral compartment');
addspecies(model.Compartment(4),'B',0,'InitialAmountUnits','cell', 'Notes','Number of B cells in the lymph node compartment');
% Tumor compartment: naive B, activated B, plasma (match ABM states)
addspecies(model.Compartment(3),'nB',0,'InitialAmountUnits','cell', 'Notes','Number of naive B cells in the tumor compartment (ABM: BCELL_NAIVE)');
addspecies(model.Compartment(3),'aB',0,'InitialAmountUnits','cell', 'Notes','Number of activated B cells in the tumor compartment (ABM: BCELL_ACTIVATED)');
addspecies(model.Compartment(3),'Plasma',0,'InitialAmountUnits','cell', 'Notes','Number of plasma cells in the tumor compartment (ABM: BCELL_PLASMA)');
% Tumor Breg species (switching targets — share kinetics with non-Breg)
addspecies(model.Compartment(3),'nBreg',0,'InitialAmountUnits','cell', 'Notes','Naive regulatory B cells in tumor (Breg-switched nB, IL-10 functional state)');
addspecies(model.Compartment(3),'aBreg',0,'InitialAmountUnits','cell', 'Notes','Activated regulatory B cells in tumor (Breg-switched aB, IL-10 secreting)');
% Tumor-local endogenous antibody (polyclonal IgG, nanomolarity)
addspecies(model.Compartment(3),'Ab',0,'InitialAmountUnits','nanomolarity', 'Notes','Endogenous polyclonal IgG in tumor (ABM: PDE_CONC_ANTIBODY)');

% Add Parameters
p = addparameter(model,'k_nB_in',params.k_nB_in.Value,'ValueUnits',params.k_nB_in.Units, 'Notes','Marrow output of naive B cells to blood');
p.UserData.Source = params.k_nB_in.Notes;  p.UserData.derivedFrom = params.k_nB_in.derivedFrom;
p = addparameter(model,'k_B_act',params.k_B_act.Value,'ValueUnits',params.k_B_act.Units, 'Notes','Naive B activation rate in LN (antigen + Tfh help)');
p.UserData.Source = params.k_B_act.Notes;  p.UserData.derivedFrom = params.k_B_act.derivedFrom;
p = addparameter(model,'n_B_clones',params.n_B_clones.Value,'ValueUnits',params.n_B_clones.Units, 'Notes','Number of antigen-specific B cell clones (clonal expansion)');
p.UserData.Source = params.n_B_clones.Notes;  p.UserData.derivedFrom = params.n_B_clones.derivedFrom;
p = addparameter(model,'k_B_pro',params.k_B_pro.Value,'ValueUnits',params.k_B_pro.Units, 'Notes','Activated B proliferation rate in LN');
p.UserData.Source = params.k_B_pro.Notes;  p.UserData.derivedFrom = params.k_B_pro.derivedFrom;
p = addparameter(model,'k_B_death',params.k_B_death.Value,'ValueUnits',params.k_B_death.Units, 'Notes','Mature B cell death rate');
p.UserData.Source = params.k_B_death.Notes;  p.UserData.derivedFrom = params.k_B_death.derivedFrom;
p = addparameter(model,'k_B_to_plasma',params.k_B_to_plasma.Value,'ValueUnits',params.k_B_to_plasma.Units, 'Notes','Tumor-local aB -> plasma differentiation rate (Tfh-helped)');
p.UserData.Source = params.k_B_to_plasma.Notes;  p.UserData.derivedFrom = params.k_B_to_plasma.derivedFrom;
p = addparameter(model,'k_plasma_death',params.k_plasma_death.Value,'ValueUnits',params.k_plasma_death.Units, 'Notes','Plasma cell death rate');
p.UserData.Source = params.k_plasma_death.Notes;  p.UserData.derivedFrom = params.k_plasma_death.derivedFrom;
p = addparameter(model,'Tfh_50_B',params.Tfh_50_B.Value,'ValueUnits',params.Tfh_50_B.Units, 'Notes','Half-maximal LN Tfh number for B-cell activation help');
p.UserData.Source = params.Tfh_50_B.Notes;  p.UserData.derivedFrom = params.Tfh_50_B.derivedFrom;

% New tumor-local parameters
p = addparameter(model,'k_B_act_tumor',params.k_B_act_tumor.Value,'ValueUnits',params.k_B_act_tumor.Units, 'Notes','Tumor-local naive B activation rate (antigen-gated)');
p.UserData.Source = params.k_B_act_tumor.Notes;  p.UserData.derivedFrom = params.k_B_act_tumor.derivedFrom;
p = addparameter(model,'k_B_pro_tumor',params.k_B_pro_tumor.Value,'ValueUnits',params.k_B_pro_tumor.Units, 'Notes','Tumor-local activated B proliferation rate (saturating form)');
p.UserData.Source = params.k_B_pro_tumor.Notes;  p.UserData.derivedFrom = params.k_B_pro_tumor.derivedFrom;
p = addparameter(model,'rho_B_max',params.rho_B_max.Value,'ValueUnits',params.rho_B_max.Units, 'Notes','Max activated B density in tumor (carrying capacity, analogous to rho_Th_max)');
p.UserData.Source = params.rho_B_max.Notes;  p.UserData.derivedFrom = params.rho_B_max.derivedFrom;
p = addparameter(model,'Tfh_50_plasma',params.Tfh_50_plasma.Value,'ValueUnits',params.Tfh_50_plasma.Units, 'Notes','Half-maximal tumor Tfh for aB -> plasma differentiation help');
p.UserData.Source = params.Tfh_50_plasma.Notes;  p.UserData.derivedFrom = params.Tfh_50_plasma.derivedFrom;

% B cell-specific transport parameters (replacing shared Treg/nCD4 params)
p = addparameter(model,'k_nB_death',params.k_nB_death.Value,'ValueUnits',params.k_nB_death.Units, 'Notes','Naive B cell death rate (was k_nTreg_death)');
p.UserData.Source = params.k_nB_death.Notes;  p.UserData.derivedFrom = params.k_nB_death.derivedFrom;
p = addparameter(model,'q_nB_LN_in',params.q_nB_LN_in.Value,'ValueUnits',params.q_nB_LN_in.Units, 'Notes','Naive B cell entry into LN (was q_nCD4_LN_in)');
p.UserData.Source = params.q_nB_LN_in.Notes;  p.UserData.derivedFrom = params.q_nB_LN_in.derivedFrom;
p = addparameter(model,'q_nB_LN_out',params.q_nB_LN_out.Value,'ValueUnits',params.q_nB_LN_out.Units, 'Notes','Naive B cell exit from LN (was q_nCD4_LN_out)');
p.UserData.Source = params.q_nB_LN_out.Notes;  p.UserData.derivedFrom = params.q_nB_LN_out.derivedFrom;
p = addparameter(model,'q_B_P_in',params.q_B_P_in.Value,'ValueUnits',params.q_B_P_in.Units, 'Notes','Mature B cell transport C->P (was q_Treg_P_in)');
p.UserData.Source = params.q_B_P_in.Notes;  p.UserData.derivedFrom = params.q_B_P_in.derivedFrom;
p = addparameter(model,'q_B_P_out',params.q_B_P_out.Value,'ValueUnits',params.q_B_P_out.Units, 'Notes','Mature B cell transport P->C (was q_Treg_P_out)');
p.UserData.Source = params.q_B_P_out.Notes;  p.UserData.derivedFrom = params.q_B_P_out.derivedFrom;
p = addparameter(model,'q_B_LN_out',params.q_B_LN_out.Value,'ValueUnits',params.q_B_LN_out.Units, 'Notes','Mature B cell exit from LN (was q_Treg_LN_out)');
p.UserData.Source = params.q_B_LN_out.Notes;  p.UserData.derivedFrom = params.q_B_LN_out.derivedFrom;
p = addparameter(model,'q_B_T_in',params.q_B_T_in.Value,'ValueUnits',params.q_B_T_in.Units, 'Notes','B cell transmigration into tumor (was q_Treg_T_in)');
p.UserData.Source = params.q_B_T_in.Notes;  p.UserData.derivedFrom = params.q_B_T_in.derivedFrom;

% Antibody parameters
p = addparameter(model,'k_Ab_sec',params.k_Ab_sec.Value,'ValueUnits',params.k_Ab_sec.Units, 'Notes','Plasma cell antibody secretion rate');
p.UserData.Source = params.k_Ab_sec.Notes;  p.UserData.derivedFrom = params.k_Ab_sec.derivedFrom;
p = addparameter(model,'k_Ab_deg',params.k_Ab_deg.Value,'ValueUnits',params.k_Ab_deg.Units, 'Notes','Antibody degradation/clearance rate in tumor');
p.UserData.Source = params.k_Ab_deg.Notes;  p.UserData.derivedFrom = params.k_Ab_deg.derivedFrom;
p = addparameter(model,'Ab_50_ADCC',params.Ab_50_ADCC.Value,'ValueUnits',params.Ab_50_ADCC.Units, 'Notes','Half-maximal antibody for ADCC (macrophage Fc-mediated killing boost)');
p.UserData.Source = params.Ab_50_ADCC.Notes;  p.UserData.derivedFrom = params.Ab_50_ADCC.derivedFrom;
p = addparameter(model,'ADCC_boost_max',params.ADCC_boost_max.Value,'ValueUnits',params.ADCC_boost_max.Units, 'Notes','Max fold-increase in M1 phagocytosis from ADCC');
p.UserData.Source = params.ADCC_boost_max.Notes;  p.UserData.derivedFrom = params.ADCC_boost_max.derivedFrom;
p = addparameter(model,'Ab_50_DC',params.Ab_50_DC.Value,'ValueUnits',params.Ab_50_DC.Units, 'Notes','Half-maximal antibody for DC immune-complex uptake boost');
p.UserData.Source = params.Ab_50_DC.Notes;  p.UserData.derivedFrom = params.Ab_50_DC.derivedFrom;
p = addparameter(model,'DC_Ab_boost',params.DC_Ab_boost.Value,'ValueUnits',params.DC_Ab_boost.Units, 'Notes','Max fold-increase in DC antigen uptake from immune complexes');
p.UserData.Source = params.DC_Ab_boost.Notes;  p.UserData.derivedFrom = params.DC_Ab_boost.derivedFrom;

% Antibody Hill function (consumed by macrophage_module and antigen_module)
H_Ab = addparameter(model,'H_Ab',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes','Hill function for antibody-mediated ADCC and immune-complex effects');
H_Ab_DC = addparameter(model,'H_Ab_DC',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes','Hill function for antibody-mediated DC uptake boost');
addrule(model,'H_Ab = V_T.Ab/(Ab_50_ADCC + V_T.Ab)','repeatedAssignment');
addrule(model,'H_Ab_DC = V_T.Ab/(Ab_50_DC + V_T.Ab)','repeatedAssignment');

% Breg switching parameters
p = addparameter(model,'k_Breg_on',params.k_Breg_on.Value,'ValueUnits',params.k_Breg_on.Units, 'Notes','Forward B->Breg switching rate (TGFb+IL10 gated)');
p.UserData.Source = params.k_Breg_on.Notes;  p.UserData.derivedFrom = params.k_Breg_on.derivedFrom;
p = addparameter(model,'k_Breg_off',params.k_Breg_off.Value,'ValueUnits',params.k_Breg_off.Units, 'Notes','Reverse Breg->B reversion rate');
p.UserData.Source = params.k_Breg_off.Notes;  p.UserData.derivedFrom = params.k_Breg_off.derivedFrom;
p = addparameter(model,'TGFb_50_Breg',params.TGFb_50_Breg.Value,'ValueUnits',params.TGFb_50_Breg.Units, 'Notes','TGFb EC50 for Breg switching');
p.UserData.Source = params.TGFb_50_Breg.Notes;  p.UserData.derivedFrom = params.TGFb_50_Breg.derivedFrom;
p = addparameter(model,'IL10_50_Breg',params.IL10_50_Breg.Value,'ValueUnits',params.IL10_50_Breg.Units, 'Notes','IL-10 EC50 for Breg switching');
p.UserData.Source = params.IL10_50_Breg.Notes;  p.UserData.derivedFrom = params.IL10_50_Breg.derivedFrom;
p = addparameter(model,'k_IL10_sec_Breg',params.k_IL10_sec_Breg.Value,'ValueUnits',params.k_IL10_sec_Breg.Units, 'Notes','Breg IL-10 secretion rate (aBreg only)');
p.UserData.Source = params.k_IL10_sec_Breg.Notes;  p.UserData.derivedFrom = params.k_IL10_sec_Breg.derivedFrom;

% Breg Hill functions (TGFb and IL-10 gates for switching)
H_TGFb_Breg = addparameter(model,'H_TGFb_Breg',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes','TGFb Hill for Breg switching');
H_IL10_Breg = addparameter(model,'H_IL10_Breg',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes','IL-10 Hill for Breg switching');
addrule(model,'H_TGFb_Breg = V_T.TGFb/(TGFb_50_Breg + V_T.TGFb)','repeatedAssignment');
addrule(model,'H_IL10_Breg = V_T.IL10/(IL10_50_Breg + V_T.IL10)','repeatedAssignment');

% ========================================================================
% LYMPH NODE & SYSTEMIC DYNAMICS (unchanged)
% ========================================================================

% Naive B homeostasis (marrow source, death, C<->LN traffic)
% Uses B cell-specific transport params (no longer shared with nCD4/Treg)
reaction = addreaction(model,'null -> V_C.nB');
    set(reaction,'ReactionRate','k_nB_in');
    set(reaction,'Notes','Marrow output of naive B cells to blood');
reaction = addreaction(model,'V_C.nB -> null');
    set(reaction,'ReactionRate','k_nB_death*V_C.nB');
    set(reaction,'Notes','Naive B cell death in the central compartment');
reaction = addreaction(model,'V_LN.nB -> null');
    set(reaction,'ReactionRate','k_nB_death*V_LN.nB');
    set(reaction,'Notes','Naive B cell death in the lymph node');
reaction = addreaction(model,'V_C.nB -> V_LN.nB');
    set(reaction,'ReactionRate','q_nB_LN_in*V_C.nB');
    set(reaction,'Notes','Naive B cell entry into the lymph node');
reaction = addreaction(model,'V_LN.nB -> V_C.nB');
    set(reaction,'ReactionRate','q_nB_LN_out*V_LN.nB');
    set(reaction,'Notes','Naive B cell exit from the lymph node');

% Naive B activation in LN (antigen-dependent + Tfh help; consumes nB in LN)
reaction = addreaction(model,'V_LN.nB -> null');
    set(reaction,'ReactionRate','k_B_act*H_P1*(V_LN.Tfh/(V_LN.Tfh+Tfh_50_B))*V_LN.nB');
    set(reaction,'Notes','Naive B activation in LN (antigen + Tfh help)');
reaction = addreaction(model,'null -> V_LN.aB');
    set(reaction,'ReactionRate','k_B_act*H_P1*(V_LN.Tfh/(V_LN.Tfh+Tfh_50_B))*V_LN.nB*n_B_clones');
    set(reaction,'Notes','Naive B activation in LN (antigen + Tfh help, clonal expansion)');
% Activated B proliferation -> mature B (in LN)
reaction = addreaction(model,'V_LN.aB -> null');
    set(reaction,'ReactionRate','(k_B_pro/N_aTh)*V_LN.aB');
    set(reaction,'Notes','Activated B proliferation in LN');
reaction = addreaction(model,'null -> V_LN.B');
    set(reaction,'ReactionRate','(k_B_pro/N_aTh)*2^N_aTh*V_LN.aB');
    set(reaction,'Notes','Activated B proliferation in LN (clonal burst)');

% Mature B death (C, P, LN compartments)
reaction = addreaction(model,'V_C.B -> null');
    set(reaction,'ReactionRate','k_B_death*V_C.B');
    set(reaction,'Notes','B cell death in the central compartment');
reaction = addreaction(model,'V_P.B -> null');
    set(reaction,'ReactionRate','k_B_death*V_P.B');
    set(reaction,'Notes','B cell death in the peripheral compartment');
reaction = addreaction(model,'V_LN.B -> null');
    set(reaction,'ReactionRate','k_B_death*V_LN.B');
    set(reaction,'Notes','B cell death in the lymph node');

% Mature B transport (C/P/LN) — uses B cell-specific rates
reaction = addreaction(model,'V_C.B -> V_P.B');
    set(reaction,'ReactionRate','q_B_P_in*V_C.B');
    set(reaction,'Notes','B cell transport into the peripheral compartment');
reaction = addreaction(model,'V_P.B -> V_C.B');
    set(reaction,'ReactionRate','q_B_P_out*V_P.B');
    set(reaction,'Notes','B cell transport out of the peripheral compartment');
reaction = addreaction(model,'V_LN.B -> V_C.B');
    set(reaction,'ReactionRate','q_B_LN_out*V_LN.B');
    set(reaction,'Notes','B cell transport out of the lymph node');

% ========================================================================
% TUMOR COMPARTMENT (matched to ABM b_cell.cuh states)
% ========================================================================

% C -> T trafficking: mature B from blood arrives as naive B in tumor
% (tumor-naive cells that will activate locally upon encountering antigen)
% Fix: uses B cell-specific q_B_T_in, adds CXCL12 exclusion + vasculature gates
% (matching Th F175 pattern; previously missing, causing unrestricted influx).
% Niche-capacity gate 1/(1 + B_total/(rho_B_max*V_T)) prevents uncapped
% infiltration: without it, V_C.B → V_T.nB is the dominant B cell source
% and drives Plasma to 2.7e10 cells (97% of V_T volume at 75d).
reaction = addreaction(model,'V_C.B -> V_T.nB');
    set(reaction,'ReactionRate','q_B_T_in*V_T*V_C.B*(C_total^2/(C_total^2 + K_C2_recruit))*(1-H_CXCL12_Texcl)*H_vasc/(1 + (V_T.nB + V_T.aB + V_T.Plasma + V_T.nBreg + V_T.aBreg)/(rho_B_max*V_T + cell))');
    set(reaction,'Notes','B cell trafficking to tumor (arrives as naive, ABM: BCELL_NAIVE). Gated by CXCL12 exclusion, vasculature, and B lineage niche capacity (includes Breg).');

% Tumor-local naive B activation (antigen-dependent, matches ABM)
reaction = addreaction(model,'V_T.nB -> V_T.aB');
    set(reaction,'ReactionRate','k_B_act_tumor*H_P1*V_T.nB');
    set(reaction,'Notes','Tumor-local naive B activation (antigen-gated, ABM: NAIVE->ACTIVATED)');

% Tumor-local activated B proliferation (SATURATING, analogous to Th F182)
% Fix: was pure exponential k_B_pro_tumor*aB (net 0.9/day, 18h doubling).
% Now saturating Michaelis form with rho_B_max carrying capacity.
% Cap uses TOTAL B lineage (nB+aB+Plasma) not just aB, because Plasma is a
% terminal sink with slow death (k_plasma_death=0.033/day, 30d half-life)
% and uncapped aB->Plasma differentiation caused Plasma runaway (2.77e10
% cells at 75d, 1500x cancer). Total-lineage cap prevents this.
reaction = addreaction(model,'null -> V_T.aB');
    set(reaction,'ReactionRate','k_B_pro_tumor*V_T.aB/(1 + (V_T.nB + V_T.aB + V_T.Plasma + V_T.nBreg + V_T.aBreg)/(rho_B_max*V_T + cell))');
    set(reaction,'Notes','Tumor-local activated B cell proliferation (saturating with rho_B_max cap on TOTAL B lineage including Breg, matching Th F182 form)');

% Tumor-local aB -> Plasma differentiation (Tfh-helped, CONCENTRATION-based Hill)
% Uses Tfh DENSITY (V_T.Tfh/V_T) with density-based EC50, not raw count.
% Raw-count form had Tfh_50_plasma=100 cell vs V_T.Tfh~1.6e8 → H≈1.0 (saturated, no regulation).
reaction = addreaction(model,'V_T.aB -> V_T.Plasma');
    set(reaction,'ReactionRate','k_B_to_plasma*V_T.aB*((V_T.Tfh/V_T)/((V_T.Tfh/V_T) + Tfh_50_plasma))');
    set(reaction,'Notes','Tumor-local aB -> plasma differentiation (Tfh-helped density Hill, ABM: ACTIVATED->PLASMA)');

% Tumor B cell death (all states)
reaction = addreaction(model,'V_T.nB -> null');
    set(reaction,'ReactionRate','k_B_death*V_T.nB');
    set(reaction,'Notes','Naive B cell death in the tumor compartment');
reaction = addreaction(model,'V_T.aB -> null');
    set(reaction,'ReactionRate','k_B_death*V_T.aB');
    set(reaction,'Notes','Activated B cell death in the tumor compartment');
reaction = addreaction(model,'V_T.Plasma -> null');
    set(reaction,'ReactionRate','k_plasma_death*V_T.Plasma');
    set(reaction,'Notes','Plasma cell death in the tumor compartment');

% Antigen-clearance funneling (all tumor B states)
reaction = addreaction(model,'V_T.nB -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.nB*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Naive B clearance upon antigen clearance');
reaction = addreaction(model,'V_T.aB -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.aB*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Activated B clearance upon antigen clearance');
reaction = addreaction(model,'V_T.Plasma -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.Plasma*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Plasma clearance upon antigen clearance');

% ========================================================================
% BREG SWITCHING (tumor compartment only — TGFb + IL-10 driven)
% ========================================================================

% Forward: B -> Breg (signal-dependent, reversible)
reaction = addreaction(model,'V_T.nB -> V_T.nBreg');
    set(reaction,'ReactionRate','k_Breg_on*H_TGFb_Breg*H_IL10_Breg*V_T.nB');
    set(reaction,'Notes','Naive B -> naive Breg switching (TGFb + IL-10 gated)');
reaction = addreaction(model,'V_T.aB -> V_T.aBreg');
    set(reaction,'ReactionRate','k_Breg_on*H_TGFb_Breg*H_IL10_Breg*V_T.aB');
    set(reaction,'Notes','Activated B -> activated Breg switching (TGFb + IL-10 gated)');

% Reverse: Breg -> B (first-order reversion)
reaction = addreaction(model,'V_T.nBreg -> V_T.nB');
    set(reaction,'ReactionRate','k_Breg_off*V_T.nBreg');
    set(reaction,'Notes','Naive Breg -> naive B reversion (epigenetic, HDAC-dependent)');
reaction = addreaction(model,'V_T.aBreg -> V_T.aB');
    set(reaction,'ReactionRate','k_Breg_off*V_T.aBreg');
    set(reaction,'Notes','Activated Breg -> activated B reversion');

% Breg activation (same rate as non-Breg nB -> aB)
reaction = addreaction(model,'V_T.nBreg -> V_T.aBreg');
    set(reaction,'ReactionRate','k_B_act_tumor*H_P1*V_T.nBreg');
    set(reaction,'Notes','Tumor-local naive Breg activation (antigen-gated, same rate as nB)');

% Breg proliferation (same saturating form, shared niche cap)
reaction = addreaction(model,'null -> V_T.aBreg');
    set(reaction,'ReactionRate','k_B_pro_tumor*V_T.aBreg/(1 + (V_T.nB + V_T.aB + V_T.Plasma + V_T.nBreg + V_T.aBreg)/(rho_B_max*V_T + cell))');
    set(reaction,'Notes','Tumor-local activated Breg proliferation (saturating, shared B lineage niche cap)');

% Breg aB CANNOT differentiate to plasma — must revert to aB first.
% (Mirlekar 2022: no IL-10/IL-35 from plasma cells in PDAC.)

% Breg death (same rates as non-Breg)
reaction = addreaction(model,'V_T.nBreg -> null');
    set(reaction,'ReactionRate','k_B_death*V_T.nBreg');
    set(reaction,'Notes','Naive Breg death in tumor');
reaction = addreaction(model,'V_T.aBreg -> null');
    set(reaction,'ReactionRate','k_B_death*V_T.aBreg');
    set(reaction,'Notes','Activated Breg death in tumor');
% Breg antigen-clearance funneling (same as non-Breg)
reaction = addreaction(model,'V_T.nBreg -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.nBreg*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Naive Breg clearance upon antigen clearance');
reaction = addreaction(model,'V_T.aBreg -> null');
    set(reaction,'ReactionRate','k_cell_clear*V_T.aBreg*(K_C2_recruit/(C_total^2 + K_C2_recruit))');
    set(reaction,'Notes','Activated Breg clearance upon antigen clearance');
% Breg IL-10 secretion (aBreg only — plasma excluded per Mirlekar 2022)
reaction = addreaction(model,'null -> V_T.IL10');
    set(reaction,'ReactionRate','k_IL10_sec_Breg*V_T.aBreg');
    set(reaction,'Notes','Breg IL-10 secretion (activated Breg only)');

% ========================================================================
% ANTIBODY DYNAMICS (tumor-local endogenous IgG)
% ========================================================================

% Plasma cell secretion (~100 pg/cell/day ≈ 1e-5 pmol/s per cell)
reaction = addreaction(model,'null -> V_T.Ab');
    set(reaction,'ReactionRate','k_Ab_sec*V_T.Plasma');
    set(reaction,'Notes','Endogenous antibody secretion by plasma cells');
% First-order degradation/clearance
reaction = addreaction(model,'V_T.Ab -> null');
    set(reaction,'ReactionRate','k_Ab_deg*V_T.Ab');
    set(reaction,'Notes','Antibody degradation/clearance in tumor');

% ========================================================================
% VOLUME AND AGGREGATE RULE UPDATES
% ========================================================================

% B cells are lymphocytes (~vol_Tcell). Add tumor B cell populations to
% the tumor volume rule (Rule 1) so they contribute to V_T.
model_rules = get(model,'Rules');
volume_rule = model_rules(1);
rule = get(volume_rule,'Rule');
set(volume_rule,'Rule',[rule '+(V_T.nB+V_T.aB+V_T.Plasma+V_T.nBreg+V_T.aBreg)*vol_Tcell/Ve_T']);

% ========================================================================
% ASSEMBLY-TIME REWRITES (modify reactions from earlier modules)
% ========================================================================

% ADCC boost on M1 phagocytosis (macrophage_module, called before Bcell_module).
% Append '*(1 + ADCC_boost_max*H_Ab)' to every reaction whose rate contains
% 'k_M1_phago', matching the ABM's cancer_cell.cuh ADCC block.
all_reactions = model.Reactions;
for i = 1:length(all_reactions)
    rate_str = get(all_reactions(i),'ReactionRate');
    if contains(rate_str,'k_M1_phago')
        set(all_reactions(i),'ReactionRate',[rate_str '*(1+ADCC_boost_max*H_Ab)']);
    end
end

% Immune-complex boost on DC antigen uptake (antigen_module, called before
% Bcell_module). Append '*(1 + DC_Ab_boost*H_Ab_DC)' to every reaction whose
% rate contains 'k_up*APC_total_T', matching ABM dendritic_cell.cuh Fc-receptor
% immune-complex enhancement.
for i = 1:length(all_reactions)
    rate_str = get(all_reactions(i),'ReactionRate');
    if contains(rate_str,'k_up*APC_total_T')
        set(all_reactions(i),'ReactionRate',[rate_str '*(1+DC_Ab_boost*H_Ab_DC)']);
    end
end

% Antibody-immune-complex boost on DC MATURATION signal (matches ABM
% dendritic_cell.cuh lines 186-195). ABM formula:
%   H_antigen_eff = min(H_antigen * (1 + ab_boost * H_ab), 1.0)
%   H_signal = 1 - (1 - H_antigen_eff) * (1 - H_IL12)
% QSP: replace H_HMGB1 (DAMP/antigen signal) with the antibody-boosted
% version in the cDC1 and cDC2 maturation reactions (APC_module.m).
% Reuses DC_Ab_boost and H_Ab_DC from the uptake boost above. The min(.,1)
% cap matches ABM's fminf(H_antigen_eff, 1.0f).
for i = 1:length(all_reactions)
    rate_str = get(all_reactions(i),'ReactionRate');
    if contains(rate_str,'k_APC_mat') && contains(rate_str,'H_HMGB1')
        new_rate = strrep(rate_str,'H_HMGB1','min(H_HMGB1*(1+DC_Ab_boost*H_Ab_DC),1)');
        set(all_reactions(i),'ReactionRate',new_rate);
    end
end

% Update antigen release rate rule (Rule 5) with ADCC boost. The macrophage
% module appended phagocytosis terms to Rule 5; those terms must also get
% the ADCC multiplier. Find and replace k_M1_phago terms in the rule.
model_rules = get(model,'Rules');
rule5_str = get(model_rules(5),'Rule');
if contains(rule5_str,'k_M1_phago') && ~contains(rule5_str,'ADCC_boost_max')
    rule5_str = strrep(rule5_str,'*(1-H_IL10_phago)','*(1-H_IL10_phago)*(1+ADCC_boost_max*H_Ab)');
    set(model_rules(5),'Rule',rule5_str);
end

warning('off','SimBiology:DimAnalysisNotDone_MatlabFcn_Dimensionless');
