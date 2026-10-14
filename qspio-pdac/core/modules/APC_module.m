% APC Module — cDC1/cDC2 Dendritic Cell Subtype Split
%
% Models two DC subtypes with distinct functional roles:
%   cDC1 (BATF3+): cross-presenting DCs that prime CD8 T cells via MHC-I
%   cDC2 (IRF4+):  MHC-II presenting DCs that prime CD4 Th/Treg cells
%
% Each subtype undergoes recruitment, maturation, migration, and death.
% Maturation is driven by HMGB1 (the representative maturation DAMP; bootstrap)
% and IL-12 (amplification) using OR logic, inhibited by IL-10 (H_IL10_APC) and
% TGFb. H_IL12 and H_IL10_APC are created by macrophage_module; H_HMGB1 and
% H_TGFb_APC are created here.
%
% GVAX-matured APCs route to cDC2 (GM-CSF recruits cDC2/moDCs).
%
% Hill function routing:
%   CD8 priming uses mass-action on V_LN.mcDC1 directly (see Teff_module.m)
%   H_APC_Treg (Treg priming) -> uses V_LN.mcDC2
%   (Th priming uses mass-action on V_LN.mcDC2 directly — see Th_module.m;
%    the old H_APC_Th was retired in the Th mechanistic refactor.)
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
%                         - HMGB1_50--half-max death rate for HMGB1 (DAMP)-mediated maturation
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

% HMGB1 (representative maturation DAMP)-mediated APC maturation parameters
p = addparameter(model,'HMGB1_50',params.HMGB1_50.Value,'ValueUnits',params.HMGB1_50.Units, 'Notes', 'Half-maximal tumor death rate for HMGB1 (DAMP)-mediated APC maturation');
p.UserData.Source = params.HMGB1_50.Notes;
    p.UserData.derivedFrom = params.HMGB1_50.derivedFrom;
H_HMGB1 = addparameter(model,'H_HMGB1',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for HMGB1 (DAMP)-mediated APC maturation (uses R_Tcell = total cancer death rate as the HMGB1-release proxy)');

% TGFb inhibition of APC maturation
p = addparameter(model,'TGFb_50_APC',params.TGFb_50_APC.Value,'ValueUnits',params.TGFb_50_APC.Units, 'Notes', 'Half-maximal TGFb for inhibition of APC maturation');
p.UserData.Source = params.TGFb_50_APC.Notes;
    p.UserData.derivedFrom = params.TGFb_50_APC.derivedFrom;
H_TGFb_APC = addparameter(model,'H_TGFb_APC',0,'ValueUnits','dimensionless','ConstantValue',false, 'Notes', 'Hill function for TGFb inhibition of APC maturation');

% H_pore_APC is defined in fibroblast_module_enhanced.m (forward reference)

% Rules for HMGB1 (DAMP) and TGFb Hill functions
addrule(model,'H_HMGB1 = R_Tcell/(R_Tcell+HMGB1_50)','repeatedAssignment');
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
% cDC1 recruitment/death in LN. Split from the prior single relaxation source
% 'null -> V_LN.cDC1' at k_APC_death*(APC0_cDC1_LN*V_LN-V_LN.cDC1) into an
% explicit source + first-order sink -- mathematically IDENTICAL but removes the
% negative-capable null-> pattern (audit mid-tier #10; cf. TGFb baseline V8 split).
reaction = addreaction(model,'null -> V_LN.cDC1');
    set(reaction,'ReactionRate','k_APC_death*APC0_cDC1_LN*V_LN');
    set(reaction,'Notes','cDC1 recruitment into LN (homeostatic source toward APC0_cDC1_LN)');
reaction = addreaction(model,'V_LN.cDC1 -> null');
    set(reaction,'ReactionRate','k_APC_death*V_LN.cDC1');
    set(reaction,'Notes','cDC1 turnover/death in LN (first-order)');
% cDC1 maturation (DAMP+IL12 OR logic, inhibited by IL10+TGFb)
reaction = addreaction(model,'V_T.cDC1 -> V_T.mcDC1');
    set(reaction,'ReactionRate','k_APC_mat_cDC1*V_T.cDC1*(1-(1-H_HMGB1)*(1-H_IL12))*(1-H_IL10_APC)*(1-H_TGFb_APC)');
    set(reaction,'Notes','cDC1 maturation in tumor (DAMP+IL12 driven, IL10+TGFb inhibited)');
% cDC1 migration to LN (impeded by ECM)
reaction = addreaction(model,'V_T.mcDC1 -> V_LN.mcDC1');
    set(reaction,'ReactionRate','k_APC_mig*V_T.mcDC1*(1-H_pore_APC)');
    set(reaction,'Notes','mcDC1 migration to lymph node (ECM-impeded)');
% mcDC1 in-situ death in tumor (mid-tier #17). RE-ADDED after a prior removal
% that argued k_mAPC_death is a lumped death+emigration rate (Marino & Kirschner
% 2004) and would double-count the explicit migration efflux. That rationale is
% internally inconsistent: the model already uses k_mAPC_death as PURE death for
% LN mature DCs (V_LN.mcDC1/mcDC2 -> null below), where there is no further
% migration -- so k_mAPC_death is a death rate, and death (in situ) vs migration
% (to LN) are distinct, non-overlapping fates, not a double-count. Without this
% sink, tumor mature DCs have ONLY ECM-impeded migration: in dense PDAC ECM
% (H_pore_APC high -> k_APC_mig*(1-H_pore_APC) -> 0) they would have a vanishing
% sink and accumulate unboundedly. Mature DCs die even when they cannot migrate.
reaction = addreaction(model,'V_T.mcDC1 -> null');
    set(reaction,'ReactionRate','k_mAPC_death*V_T.mcDC1');
    set(reaction,'Notes','mcDC1 in-situ death/turnover in tumor (pure death; migration to LN is the separate efflux fate)');
% mcDC1 death in LN
reaction = addreaction(model,'V_LN.mcDC1 -> null');
    set(reaction,'ReactionRate','k_mAPC_death*V_LN.mcDC1');
    set(reaction,'Notes','mcDC1 death in lymph node');

% =========================================================================
% cDC2 Reactions (IRF4+ MHC-II presenting DCs)
% =========================================================================

% cDC2 recruitment into tumor (impeded by ECM)
reaction = addreaction(model,'null -> V_T.cDC2');
    set(reaction,'ReactionRate','k_APC_death*cDC2_recruit_target_T*V_T*(1-H_pore_APC)');
    set(reaction,'Notes','cDC2 recruitment into tumor (ECM-impeded)');
% cDC2 death in tumor
reaction = addreaction(model,'V_T.cDC2 -> null');
    set(reaction,'ReactionRate','k_APC_death*V_T.cDC2');
    set(reaction,'Notes','cDC2 death/turnover in tumor');
% cDC2 recruitment/death in LN. Split from the prior single relaxation source
% (see cDC1 above) into an explicit source + first-order sink -- mathematically
% IDENTICAL but removes the negative-capable null-> pattern (audit mid-tier #10).
reaction = addreaction(model,'null -> V_LN.cDC2');
    set(reaction,'ReactionRate','k_APC_death*APC0_cDC2_LN*V_LN');
    set(reaction,'Notes','cDC2 recruitment into LN (homeostatic source toward APC0_cDC2_LN)');
reaction = addreaction(model,'V_LN.cDC2 -> null');
    set(reaction,'ReactionRate','k_APC_death*V_LN.cDC2');
    set(reaction,'Notes','cDC2 turnover/death in LN (first-order)');
% cDC2 maturation (same OR logic as cDC1, with cDC2-specific rate)
reaction = addreaction(model,'V_T.cDC2 -> V_T.mcDC2');
    set(reaction,'ReactionRate','k_APC_mat_cDC2*V_T.cDC2*(1-(1-H_HMGB1)*(1-H_IL12))*(1-H_IL10_APC)*(1-H_TGFb_APC)');
    set(reaction,'Notes','cDC2 maturation in tumor (DAMP+IL12 driven, IL10+TGFb inhibited)');
% cDC2 migration to LN (impeded by ECM)
reaction = addreaction(model,'V_T.mcDC2 -> V_LN.mcDC2');
    set(reaction,'ReactionRate','k_APC_mig*V_T.mcDC2*(1-H_pore_APC)');
    set(reaction,'Notes','mcDC2 migration to lymph node (ECM-impeded)');
% mcDC2 in-situ death in tumor (mid-tier #17, same rationale as mcDC1 above:
% k_mAPC_death is pure death (cf. its LN usage), distinct from migration efflux;
% re-added so tumor mature DCs have a sink when ECM impedes migration).
reaction = addreaction(model,'V_T.mcDC2 -> null');
    set(reaction,'ReactionRate','k_mAPC_death*V_T.mcDC2');
    set(reaction,'Notes','mcDC2 in-situ death/turnover in tumor (pure death; migration to LN is the separate efflux fate)');
% mcDC2 death in LN
reaction = addreaction(model,'V_LN.mcDC2 -> null');
    set(reaction,'ReactionRate','k_mAPC_death*V_LN.mcDC2');
    set(reaction,'Notes','mcDC2 death in lymph node');

% =========================================================================
% Hill Functions for APC-driven T cell priming (subtype-specific routing)
% =========================================================================
% H_APC_Treg (Treg priming): cDC2 drives Treg activation via MHC-II
% nCD4 = shared naive CD4+ pool; n_CD4_clones = self-antigen clone diversity.
% NOTE (audit m1-A20, adjudicated 2026-06-05, ACCEPTED as-is): at the diagnosis
% attractor this Hill is saturated (~0.995: n_sites_APC*mcDC2 >> nCD4*n_CD4_clones),
% so Treg priming is effectively cDC2-count-insensitive, while CD8/Th priming below
% are mass-action (linear in mcDC1/mcDC2). This asymmetry is INTENTIONAL and the
% saturating form here is the biologically faithful one: priming commitment
% saturates with DC density and has a threshold (Henrickson 2008 PMID 18223650;
% Mempel 2004 PMID 14712275), and tumor-specific naive T cells are rare (Moon/Jenkins
% 2007 PMID 17707129) so the tumor-draining LN sits in the DC-abundant / naive-T-limited
% (saturating) regime. A GVAX trace confirmed it is immaterial in current scenarios:
% the vaccine helper burst is antigen-driven (H_P1_II), not cDC2-count-driven (mcDC2
% swings only ~1.7x). If ever harmonized, harmonize TOWARD saturation (restore the
% CD8/Th Hills), not away. See notes/model/mechanism_audit_remaining_checklist.md (m1-A20).
addrule(model,'H_APC_Treg = n_sites_APC*V_LN.mcDC2/(n_sites_APC*V_LN.mcDC2+V_LN.nCD4*n_CD4_clones+cell)','repeatedAssignment');
% CD8 priming uses mass-action k_T_activation_per_APC*V_LN.mcDC1 directly
% (Teff_module.m) — the old H_mAPC Hill was unused and has been removed.
% H_APC_Th retired: Th priming now uses mass-action on V_LN.mcDC2 directly
% (Th_module.m mechanistic refactor for parity with CD8 priming).
