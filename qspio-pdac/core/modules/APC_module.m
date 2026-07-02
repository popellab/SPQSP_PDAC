% APC Module — cDC1/cDC2 Dendritic Cell Subtype Split
%
% Models two DC subtypes with distinct functional roles:
%   cDC1 (BATF3+): cross-presenting DCs that prime CD8 T cells via MHC-I
%   cDC2 (IRF4+):  MHC-II presenting DCs that prime CD4 Th/Treg cells
%
% Each subtype undergoes recruitment, maturation, migration, and death.
% Maturation is driven by DAMPs (bootstrap) and IL-12 (amplification) using
% OR logic, inhibited by IL-10 and TGFb. H_IL12 and H_IL10 are created by
% macrophage_module; H_DAMP and H_TGFb_APC are created here with rules.
%
% GVAX-matured APCs route to cDC2 (GM-CSF recruits cDC2/moDCs).
%
% Hill function routing:
%   H_mAPC     (CD8 priming)  -> uses V_LN.mcDC1
%   H_APC_Treg (Treg priming) -> uses V_LN.mcDC2
%   H_APC_Th   (Th priming)   -> uses V_LN.mcDC2
%
% Aggregate aliases (repeatedAssignment rules for backward compatibility):
%   APC_total_T  = V_T.cDC1 + V_T.cDC2   (used by antigen_module)
%   mAPC_total_T = V_T.mcDC1 + V_T.mcDC2
%   mAPC_total_LN = V_LN.mcDC1 + V_LN.mcDC2
%
% Inputs: model        -- simbio model object with four compartments
%         params       -- object containing model parameter Values, Units, and Notes:
%                         - cDC1_recruit_target_T--cDC1 recruitment-target density in tumour
%                         - cDC2_recruit_target_T--cDC2 recruitment-target density in tumour
%                         - APC0_cDC1_LN--cDC1 homeostatic density in LN
%                         - APC0_cDC2_LN--cDC2 homeostatic density in LN
%                         - k_APC_mat_cDC1--rate of cDC1 maturation
%                         - k_APC_mat_cDC2--rate of cDC2 maturation
%                         - k_APC_mig--rate of APC migration
%                         - k_APC_death--rate of APC death
%                         - k_mAPC_death--rate of mAPC death
%                         - DAMP_50--half-max death rate for DAMP-mediated maturation
%                         - TGFb_50_APC--half-max TGFb for APC maturation inhibition
%
% Requires from fibroblast_module_enhanced:
%   H_pore_APC -- pore-size-based APC exclusion (from fibroblast_module_enhanced)
%
% Outputs: model -- SimBiology model object with new APC module


function model = APC_module(model,params)

% Add Species — 4 per compartment (tumor + LN)
% Tumor compartment (V_T)
addspecies(model.Compartment(3),'cDC1',0,'InitialAmountUnits','cell', 'Notes', 'Immature cDC1 (CD11c+CD141+BATF3+CD83-, cross-presenting) in tumor');
addspecies(model.Compartment(3),'cDC2',0,'InitialAmountUnits','cell', 'Notes', 'Immature cDC2 (CD11c+CD1c+IRF4+CD83-, MHC-II presenting) in tumor');
addspecies(model.Compartment(3),'mcDC1',0,'InitialAmountUnits','cell', 'Notes', 'Mature cDC1 (CD83+CD80+CD86+CD141+BATF3+, cross-presenting) in tumor');
addspecies(model.Compartment(3),'mcDC2',0,'InitialAmountUnits','cell', 'Notes', 'Mature cDC2 (CD83+CD80+CD86+CD1c+IRF4+, MHC-II presenting) in tumor');
% Lymph node compartment (V_LN)
addspecies(model.Compartment(4),'cDC1',0,'InitialAmountUnits','cell', 'Notes', 'Immature cDC1 (CD11c+CD141+BATF3+CD83-) in lymph node');
addspecies(model.Compartment(4),'cDC2',0,'InitialAmountUnits','cell', 'Notes', 'Immature cDC2 (CD11c+CD1c+IRF4+CD83-) in lymph node');
addspecies(model.Compartment(4),'mcDC1',0,'InitialAmountUnits','cell', 'Notes', 'Mature cDC1 (CD83+CD80+CD86+CD141+BATF3+CCR7+) in lymph node');
addspecies(model.Compartment(4),'mcDC2',0,'InitialAmountUnits','cell', 'Notes', 'Mature cDC2 (CD83+CD80+CD86+CD1c+IRF4+CCR7+) in lymph node');

% Add Parameters
% Subtype-specific maturation rates
p = addparameter(model,'k_APC_mat_cDC1',params.k_APC_mat_cDC1.Value,'ValueUnits',params.k_APC_mat_cDC1.Units, 'Notes', 'Rate of cDC1 maturation (BATF3 program)');
p.UserData.Source = params.k_APC_mat_cDC1.Notes;
    p.UserData.derivedFrom = params.k_APC_mat_cDC1.derivedFrom;
p = addparameter(model,'k_APC_mat_cDC2',params.k_APC_mat_cDC2.Value,'ValueUnits',params.k_APC_mat_cDC2.Units, 'Notes', 'Rate of cDC2 maturation');
p.UserData.Source = params.k_APC_mat_cDC2.Notes;
    p.UserData.derivedFrom = params.k_APC_mat_cDC2.derivedFrom;

% Shared parameters (migration, death, steady-state densities)
p = addparameter(model,'k_APC_mig',params.k_APC_mig.Value,'ValueUnits',params.k_APC_mig.Units, 'Notes', 'Rate of APC migration');
p.UserData.Source = params.k_APC_mig.Notes;
    if isfield(params.k_APC_mig, 'derivedFrom'), p.UserData.derivedFrom = params.k_APC_mig.derivedFrom; end
p = addparameter(model,'k_APC_death',params.k_APC_death.Value,'ValueUnits',params.k_APC_death.Units, 'Notes', 'Rate of APC death');
p.UserData.Source = params.k_APC_death.Notes;
    if isfield(params.k_APC_death, 'derivedFrom'), p.UserData.derivedFrom = params.k_APC_death.derivedFrom; end
p = addparameter(model,'k_mAPC_death',params.k_mAPC_death.Value,'ValueUnits',params.k_mAPC_death.Units, 'Notes', 'Rate of mAPC death');
p.UserData.Source = params.k_mAPC_death.Notes;
    if isfield(params.k_mAPC_death, 'derivedFrom'), p.UserData.derivedFrom = params.k_mAPC_death.derivedFrom; end
% Independent DC subtype recruitment-target densities (decoupled for SBI calibration).
% NOTE: V_T uses recruit + first-order death (lines below); steady-state realized
% density is attenuated by H_pore_APC and maturation drain, NOT equal to this
% parameter. V_LN uses clean homeostatic relaxation. See
% notes/model/cdc1_homeostasis_vs_recruitment.md for the biology rationale.
p = addparameter(model,'cDC1_recruit_target_T',params.cDC1_recruit_target_T.Value,'ValueUnits',params.cDC1_recruit_target_T.Units, 'Notes', 'cDC1 recruitment-target density in tumor (recruit-rate / k_APC_death)');
p.UserData.Source = params.cDC1_recruit_target_T.Notes;
    p.UserData.derivedFrom = params.cDC1_recruit_target_T.derivedFrom;
p = addparameter(model,'cDC2_recruit_target_T',params.cDC2_recruit_target_T.Value,'ValueUnits',params.cDC2_recruit_target_T.Units, 'Notes', 'cDC2 recruitment-target density in tumor (recruit-rate / k_APC_death)');
p.UserData.Source = params.cDC2_recruit_target_T.Notes;
    p.UserData.derivedFrom = params.cDC2_recruit_target_T.derivedFrom;
p = addparameter(model,'APC0_cDC1_LN',params.APC0_cDC1_LN.Value,'ValueUnits',params.APC0_cDC1_LN.Units, 'Notes', 'cDC1 homeostatic density in LN');
p.UserData.Source = params.APC0_cDC1_LN.Notes;
    p.UserData.derivedFrom = params.APC0_cDC1_LN.derivedFrom;
p = addparameter(model,'APC0_cDC2_LN',params.APC0_cDC2_LN.Value,'ValueUnits',params.APC0_cDC2_LN.Units, 'Notes', 'cDC2 homeostatic density in LN');
p.UserData.Source = params.APC0_cDC2_LN.Notes;
    p.UserData.derivedFrom = params.APC0_cDC2_LN.derivedFrom;
p = addparameter(model,'n_sites_APC',params.n_sites_APC.Value,'ValueUnits',params.n_sites_APC.Units, 'Notes', 'Maximum number of T Cells an APC can interact with');
p.UserData.Source = params.n_sites_APC.Notes;
    if isfield(params.n_sites_APC, 'derivedFrom'), p.UserData.derivedFrom = params.n_sites_APC.derivedFrom; end

% DAMP-mediated APC maturation parameters
p = addparameter(model,'DAMP_50',params.DAMP_50.Value,'ValueUnits',params.DAMP_50.Units, 'Notes', 'Half-maximal tumor death rate for DAMP-mediated APC maturation');
p.UserData.Source = params.DAMP_50.Notes;
    p.UserData.derivedFrom = params.DAMP_50.derivedFrom;
H_DAMP = addparameter(model,'H_DAMP',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for DAMP-mediated APC maturation (uses R_Tcell = total cancer death rate)');

% TGFb inhibition of APC maturation
p = addparameter(model,'TGFb_50_APC',params.TGFb_50_APC.Value,'ValueUnits',params.TGFb_50_APC.Units, 'Notes', 'Half-maximal TGFb for inhibition of APC maturation');
p.UserData.Source = params.TGFb_50_APC.Notes;
    p.UserData.derivedFrom = params.TGFb_50_APC.derivedFrom;
H_TGFb_APC = addparameter(model,'H_TGFb_APC',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for TGFb inhibition of APC maturation');

% H_pore_APC is defined in fibroblast_module_enhanced.m (forward reference)

% Rules for DAMP and TGFb Hill functions
addrule(model,'H_DAMP = R_Tcell/(R_Tcell+DAMP_50)','repeatedAssignment');
addrule(model,'H_TGFb_APC = V_T.TGFb/(V_T.TGFb+TGFb_50_APC)','repeatedAssignment');

% Aggregate alias rules (backward compatibility for downstream modules)
addparameter(model,'APC_total_T',0,'ValueUnits','cell','ConstantValue',false, 'Notes', 'Total immature DCs in tumor (cDC1+cDC2; CD11c+CD83-)');
addparameter(model,'mAPC_total_T',0,'ValueUnits','cell','ConstantValue',false, 'Notes', 'Total mature DCs in tumor (mcDC1+mcDC2; CD83+ by IHC)');
addparameter(model,'mAPC_total_LN',0,'ValueUnits','cell','ConstantValue',false, 'Notes', 'Total mature DCs in LN (mcDC1+mcDC2; CD83+CCR7+)');
addrule(model,'APC_total_T = V_T.cDC1 + V_T.cDC2','repeatedAssignment');
addrule(model,'mAPC_total_T = V_T.mcDC1 + V_T.mcDC2','repeatedAssignment');
addrule(model,'mAPC_total_LN = V_LN.mcDC1 + V_LN.mcDC2','repeatedAssignment');

% =========================================================================
% cDC1 Reactions (BATF3+ cross-presenting DCs)
% =========================================================================

% cDC1 recruitment into tumor (impeded by ECM)
reaction = addreaction(model,'null -> V_T.cDC1');
    set(reaction,'ReactionRate','k_APC_death*cDC1_recruit_target_T*V_T*(1-H_pore_APC)');
    set(reaction,'Notes','cDC1 recruitment into tumor (ECM-impeded)');
% cDC1 death in tumor
reaction = addreaction(model,'V_T.cDC1 -> null');
    set(reaction,'ReactionRate','k_APC_death*V_T.cDC1');
    set(reaction,'Notes','cDC1 death/turnover in tumor');
% cDC1 recruitment/death in LN
reaction = addreaction(model,'null -> V_LN.cDC1');
    set(reaction,'ReactionRate','k_APC_death*(APC0_cDC1_LN*V_LN-V_LN.cDC1)');
    set(reaction,'Notes','cDC1 recruitment/death in LN');
% cDC1 maturation (DAMP+IL12 OR logic, inhibited by IL10+TGFb)
reaction = addreaction(model,'V_T.cDC1 -> V_T.mcDC1');
    set(reaction,'ReactionRate','k_APC_mat_cDC1*V_T.cDC1*(1-(1-H_DAMP)*(1-H_IL12))*(1-H_IL10)*(1-H_TGFb_APC)');
    set(reaction,'Notes','cDC1 maturation in tumor (DAMP+IL12 driven, IL10+TGFb inhibited)');
% cDC1 migration to LN (impeded by ECM)
reaction = addreaction(model,'V_T.mcDC1 -> V_LN.mcDC1');
    set(reaction,'ReactionRate','k_APC_mig*V_T.mcDC1*(1-H_pore_APC)');
    set(reaction,'Notes','mcDC1 migration to lymph node (ECM-impeded)');
% (mcDC1 death in V_T removed: k_mAPC_death is the lumped DC turnover rate
% from Marino & Kirschner 2004 (TB granuloma model with no separate
% migration term). Combining it with explicit migration via
% k_APC_mig*(1-H_pore_APC) double-counts the efflux sink, and silently
% becomes the dominant sink in dense PDAC ECM where migration is impeded.
% Mature DC fate in tumor is migration; the V_LN.mcDC1 -> null reaction
% below retains k_mAPC_death for canonical post-presentation turnover in
% the lymph node.)
% mcDC1 death in LN
reaction = addreaction(model,'V_LN.mcDC1 -> null');
    set(reaction,'ReactionRate','k_mAPC_death*V_LN.mcDC1');
    set(reaction,'Notes','mcDC1 death in lymph node');

% =========================================================================
% cDC2 Reactions (IRF4+ MHC-II presenting DCs)
% =========================================================================

% cDC2 recruitment into tumor (no XCL1 boost, impeded by ECM)
reaction = addreaction(model,'null -> V_T.cDC2');
    set(reaction,'ReactionRate','k_APC_death*cDC2_recruit_target_T*V_T*(1-H_pore_APC)');
    set(reaction,'Notes','cDC2 recruitment into tumor (ECM-impeded)');
% cDC2 death in tumor
reaction = addreaction(model,'V_T.cDC2 -> null');
    set(reaction,'ReactionRate','k_APC_death*V_T.cDC2');
    set(reaction,'Notes','cDC2 death/turnover in tumor');
% cDC2 recruitment/death in LN
reaction = addreaction(model,'null -> V_LN.cDC2');
    set(reaction,'ReactionRate','k_APC_death*(APC0_cDC2_LN*V_LN-V_LN.cDC2)');
    set(reaction,'Notes','cDC2 recruitment/death in LN');
% cDC2 maturation (same OR logic as cDC1, with cDC2-specific rate)
reaction = addreaction(model,'V_T.cDC2 -> V_T.mcDC2');
    set(reaction,'ReactionRate','k_APC_mat_cDC2*V_T.cDC2*(1-(1-H_DAMP)*(1-H_IL12))*(1-H_IL10)*(1-H_TGFb_APC)');
    set(reaction,'Notes','cDC2 maturation in tumor (DAMP+IL12 driven, IL10+TGFb inhibited)');
% cDC2 migration to LN (impeded by ECM)
reaction = addreaction(model,'V_T.mcDC2 -> V_LN.mcDC2');
    set(reaction,'ReactionRate','k_APC_mig*V_T.mcDC2*(1-H_pore_APC)');
    set(reaction,'Notes','mcDC2 migration to lymph node (ECM-impeded)');
% (mcDC2 death in V_T removed: same rationale as mcDC1 above —
% k_mAPC_death from Marino & Kirschner 2004 is a lumped turnover that
% double-counts the explicit migration sink. V_LN side retains it.)
% mcDC2 death in LN
reaction = addreaction(model,'V_LN.mcDC2 -> null');
    set(reaction,'ReactionRate','k_mAPC_death*V_LN.mcDC2');
    set(reaction,'Notes','mcDC2 death in lymph node');

% =========================================================================
% Hill Functions for APC-driven T cell priming (subtype-specific routing)
% =========================================================================
% H_APC_Treg (Treg priming): cDC2 drives Treg activation via MHC-II
% nCD4 = shared naive CD4+ pool; n_CD4_clones = self-antigen clone diversity
addrule(model,'H_APC_Treg = n_sites_APC*V_LN.mcDC2/(n_sites_APC*V_LN.mcDC2+V_LN.nCD4*n_CD4_clones+cell)','repeatedAssignment');
% H_mAPC (CD8 priming): cDC1 drives CD8 activation via cross-presentation
addrule(model,'H_mAPC = n_sites_APC*V_LN.mcDC1/(n_sites_APC*V_LN.mcDC1+V_LN.nCD8*n_CD8_clones+cell)','repeatedAssignment');
% H_APC_Th (Th priming): cDC2 drives Th activation via MHC-II
addrule(model,'H_APC_Th = n_sites_APC*V_LN.mcDC2/(n_sites_APC*V_LN.mcDC2+V_LN.nCD4*n_CD4_clones+cell)','repeatedAssignment');
