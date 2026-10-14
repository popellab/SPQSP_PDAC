% CXCL9/10 Chemotaxis Module
%
% NOT ASSEMBLED: this module is not called from
% scripts/immune_oncology_model_PDAC.m in the current build. Its parameters
% (IFNg_50_CXCL9, k_CXCL9_sec) are therefore NOT defined in
% parameters/parameters_PDAC.m and would raise a struct-field error if the
% module were invoked without restoring them. The CXCR3 trafficking axis
% (IFNg -> CXCL9/10 -> CD8/NK recruitment) is currently absent from the
% PDAC model, so IFNg-dependent T-cell chemotaxis is uncoupled. Reactivate
% by (a) defining IFNg_50_CXCL9 and k_CXCL9_sec in parameters_PDAC.m,
% (b) adding a chemotaxis_module(model, params, cancer_types) call in the
% main assembly script, and (c) adding the params to pdac_priors.csv if
% they should be varied in SBI.
%
% Models IFNg-induced CXCL9/10 production and CXCR3-mediated recruitment
% of CD8 T cells and NK cells to the tumor. Creates the trafficking-level
% positive feedback loop: T cells -> IFNg -> CXCL9/10 -> more T cell
% recruitment -> more IFNg.
%
% CXCL9 and CXCL10 are lumped into a single species (V_T.CXCL9) representing
% total CXCR3 ligand concentration.
%
% Inputs: model  -- SimBiology model object with four compartments
%         params -- object containing model parameters
%
% Outputs: model -- SimBiology model object with chemotaxis module
%
% Dependencies: macrophage_module (Mac_M1), APC_module (mcDC1),
%               NK_module (NK, NK_P_steady), Teff_module (H_CXCL12_Texcl),
%               Th_module (Th species), fibroblast_module_enhanced (H_CXCL12_Texcl rule)
%
% Assembly order: after metabolism_module, before fibroblast_module_enhanced
%
% References:
%   Mikucki et al. (2015) Non-redundant requirement for CXCR3 signalling
%   during tumoricidal T-cell trafficking. Nat Commun. PMID: 26056112
%
%   Dangaj et al. (2019) Cooperation between constitutive and inducible
%   chemokines enables T cell engraftment and immune attack in solid tumors.
%   Cancer Cell. PMID: 30728497
%
%   Peng et al. (2015) PD-1 blockade enhances T-cell migration to tumors
%   by elevating IFN-gamma inducible chemokines. Cancer Res. PMID: 26297524

function model = chemotaxis_module(model, params)

% Add CXCL9 species in tumor compartment (lumped CXCR3 ligands)
CXCL9 = addspecies(model.Compartment(3), 'CXCL9', 0, 'InitialAmountUnits', 'nanomolarity');
set(CXCL9, 'Notes', 'Concentration of CXCL9/10 (lumped CXCR3 ligands) in the tumor compartment');

% Add Parameters
p = addparameter(model, 'k_CXCL9_sec', params.k_CXCL9_sec.Value, 'ValueUnits', params.k_CXCL9_sec.Units);
set(p, 'Notes', 'CXCL9/10 secretion rate by tumor cells, M1 macrophages, and cDC1s (IFNg-dependent)');
p.UserData.Source = params.k_CXCL9_sec.Notes;
p.UserData.derivedFrom = params.k_CXCL9_sec.derivedFrom;

p = addparameter(model, 'k_CXCL9_deg', params.k_CXCL9_deg.Value, 'ValueUnits', params.k_CXCL9_deg.Units);
set(p, 'Notes', 'CXCL9/10 degradation rate');
p.UserData.Source = params.k_CXCL9_deg.Notes;
p.UserData.derivedFrom = params.k_CXCL9_deg.derivedFrom;

p = addparameter(model, 'IFNg_50_CXCL9', params.IFNg_50_CXCL9.Value, 'ValueUnits', params.IFNg_50_CXCL9.Units);
set(p, 'Notes', 'IFNg EC50 for CXCL9/10 induction (JAK-STAT1 dependent)');
p.UserData.Source = params.IFNg_50_CXCL9.Notes;
p.UserData.derivedFrom = params.IFNg_50_CXCL9.derivedFrom;

p = addparameter(model, 'CXCL9_50_Trec', params.CXCL9_50_Trec.Value, 'ValueUnits', params.CXCL9_50_Trec.Units);
set(p, 'Notes', 'Half-maximal CXCL9/10 concentration for CXCR3-mediated recruitment');
p.UserData.Source = params.CXCL9_50_Trec.Notes;
p.UserData.derivedFrom = params.CXCL9_50_Trec.derivedFrom;

p = addparameter(model, 'k_CXCR3_rec', params.k_CXCR3_rec.Value, 'ValueUnits', params.k_CXCR3_rec.Units);
set(p, 'Notes', 'CXCR3-dependent CD8 T cell recruitment rate (additive to baseline q_CD8_T_in)');
p.UserData.Source = params.k_CXCR3_rec.Notes;
p.UserData.derivedFrom = params.k_CXCR3_rec.derivedFrom;

p = addparameter(model, 'k_CXCR3_NK_rec', params.k_CXCR3_NK_rec.Value, 'ValueUnits', params.k_CXCR3_NK_rec.Units);
set(p, 'Notes', 'CXCR3-dependent NK cell recruitment rate (additive to baseline k_NK_rec)');
p.UserData.Source = params.k_CXCR3_NK_rec.Notes;
p.UserData.derivedFrom = params.k_CXCR3_NK_rec.derivedFrom;

% Hill Functions (repeated assignment rules)
p = addparameter(model, 'H_IFNg_CXCL9', 0, 'ValueUnits', 'dimensionless', 'ConstantValue', false);
set(p, 'Notes', 'Hill function for IFNg induction of CXCL9/10 (JAK-STAT1)');
addrule(model, 'H_IFNg_CXCL9 = V_T.IFNg/(IFNg_50_CXCL9 + V_T.IFNg)', 'repeatedAssignment');

p = addparameter(model, 'H_CXCL9_Trec', 0, 'ValueUnits', 'dimensionless', 'ConstantValue', false);
set(p, 'Notes', 'Hill function for CXCR3-mediated T cell/NK recruitment');
addrule(model, 'H_CXCL9_Trec = V_T.CXCL9/(CXCL9_50_Trec + V_T.CXCL9)', 'repeatedAssignment');

% Reactions
% CXCL9/10 production by tumor cells, M1 macrophages, and mature cDC1s
% All three cell types produce CXCL9/10 in response to IFNg via JAK-STAT1
reaction = addreaction(model, 'null -> V_T.CXCL9');
set(reaction, 'ReactionRate', 'k_CXCL9_sec*(C_total + V_T.Mac_M1 + V_T.mcDC1)*H_IFNg_CXCL9');
set(reaction, 'Notes', 'CXCL9/10 production (IFNg-dependent, by tumor cells, M1 macrophages, and cDC1s)');

% CXCL9/10 degradation
reaction = addreaction(model, 'V_T.CXCL9 -> null');
set(reaction, 'ReactionRate', 'k_CXCL9_deg*V_T.CXCL9');
set(reaction, 'Notes', 'CXCL9/10 degradation in tumor compartment');

% CXCR3-dependent CD8 T cell recruitment (additive to baseline q_CD8_T_in)
% Tregs do NOT express CXCR3; only CD8 T cells are recruited via this axis
% Uses renamed species 'CD8' (not 'T') since Teff_module runs before this module
reaction = addreaction(model, 'V_C.CD8 -> V_T.CD8');
set(reaction, 'ReactionRate', 'k_CXCR3_rec*V_T*V_C.CD8*H_CXCL9_Trec*(1-H_CXCL12_Texcl)');
set(reaction, 'Notes', 'CXCR3-dependent CD8 T cell recruitment (additive, CXCL9-driven, CXCL12-opposed)');

% CXCR3-dependent Th1 cell recruitment (additive to baseline Th transport)
% Th1 cells express CXCR3; uses 'Th' since Th_module runs before this module
p = addparameter(model, 'k_CXCR3_Th_rec', params.k_CXCR3_Th_rec.Value, 'ValueUnits', params.k_CXCR3_Th_rec.Units);
set(p, 'Notes', 'CXCR3-dependent Th1 cell recruitment rate (additive to baseline q_Treg_T_in for Th)');
p.UserData.Source = params.k_CXCR3_Th_rec.Notes;
p.UserData.derivedFrom = params.k_CXCR3_Th_rec.derivedFrom;

reaction = addreaction(model, 'V_C.Th -> V_T.Th');
set(reaction, 'ReactionRate', 'k_CXCR3_Th_rec*V_T*V_C.Th*H_CXCL9_Trec*(1-H_CXCL12_Texcl)');
set(reaction, 'Notes', 'CXCR3-dependent Th1 recruitment (additive, CXCL9-driven, CXCL12-opposed)');

% CXCR3-dependent NK cell recruitment (additive to baseline k_NK_rec)
reaction = addreaction(model, 'null -> V_T.NK');
set(reaction, 'ReactionRate', 'k_CXCR3_NK_rec*V_T*(V_P.NK/(NK_P_steady+cell))*H_CXCL9_Trec*(1-H_CXCL12_Texcl)');
set(reaction, 'Notes', 'CXCR3-dependent NK cell recruitment (additive, CXCL9-driven, CXCL12-opposed)');

warning('off', 'SimBiology:DimAnalysisNotDone_MatlabFcn_Dimensionless');

end
