% CCL5/CCR5-Mediated Treg Recruitment Module
%
% Models CCL5 secretion by cancer cells and CCR5-dependent Treg recruitment
% Based on: Tan et al. (2017) Oncogene - Cancer-FOXP3 directly activates CCL5
%           to recruit FOXP3+ Treg cells in pancreatic ductal adenocarcinoma
%
% Mechanism:
%   - PDAC cancer cells express FOXP3 which activates CCL5 transcription
%   - CCL5 binds CCR5 receptor on Tregs (but not CD8 T cells)
%   - This selectively recruits Tregs to the tumor microenvironment
%
% Inputs: model        -- SimBiology model object with four compartments
%         params       -- object containing model parameters
%
% Outputs: model -- SimBiology model object with CCL5/CCR5 module
%
% Dependencies: Treg_module must be called before this module
%               (V_T.Treg species must exist)
%
% References:
%   Tan MCB et al. (2017). Cancer-FOXP3 directly activated CCL5 to recruit
%   FOXP3+Treg cells in pancreatic ductal adenocarcinoma. Oncogene.
%   DOI: 10.1038/onc.2016.458
%
%   Huffman AP et al. (2022). CCR2/CCR5 inhibitor permits the radiation-induced
%   effector T cell infiltration in pancreatic adenocarcinoma. J Immunother Cancer.
%   PMID: 35404390

function model = CCL5_Treg_module(model, params)

% Add CCL5 species in tumor compartment
CCL5 = addspecies(model.Compartment(3), 'CCL5', 0, 'InitialAmountUnits', 'nanomolarity');
set(CCL5, 'Notes', 'Concentration of CCL5 (RANTES) in the tumor compartment');

% Add Parameters
% CCL5 secretion rate by cancer cells
p = addparameter(model, 'k_CCL5_sec', params.k_CCL5_sec.Value, 'ValueUnits', params.k_CCL5_sec.Units);
set(p, 'Notes', 'Rate of CCL5 secretion by cancer cells (FOXP3-dependent)');
p.UserData.Source = params.k_CCL5_sec.Notes;
p.UserData.derivedFrom = params.k_CCL5_sec.derivedFrom;

% CCL5 degradation rate
p = addparameter(model, 'k_CCL5_deg', params.k_CCL5_deg.Value, 'ValueUnits', params.k_CCL5_deg.Units);
set(p, 'Notes', 'Rate of CCL5 degradation');
p.UserData.Source = params.k_CCL5_deg.Notes;
p.UserData.derivedFrom = params.k_CCL5_deg.derivedFrom;

% Half-maximal CCL5 concentration for Treg recruitment via CCR5
p = addparameter(model, 'CCL5_50_Treg', params.CCL5_50_Treg.Value, 'ValueUnits', params.CCL5_50_Treg.Units);
set(p, 'Notes', 'Half-maximal CCL5 concentration for CCR5-mediated Treg recruitment');
p.UserData.Source = params.CCL5_50_Treg.Notes;
p.UserData.derivedFrom = params.CCL5_50_Treg.derivedFrom;

% CCR5-dependent Treg recruitment rate (additional to baseline q_Treg_T_in)
p = addparameter(model, 'k_CCR5_Treg_rec', params.k_CCR5_Treg_rec.Value, 'ValueUnits', params.k_CCR5_Treg_rec.Units);
set(p, 'Notes', 'CCR5-dependent Treg recruitment rate to tumor');
p.UserData.Source = params.k_CCR5_Treg_rec.Notes;
p.UserData.derivedFrom = params.k_CCR5_Treg_rec.derivedFrom;

% Note: CCL5 secretion by iCAFs (was k_CCL5_sec_apsc) is now handled in
% fibroblast_module_enhanced.m where the iCAF species is defined.

% Add Hill function for CCL5-mediated recruitment
p = addparameter(model, 'H_CCL5_Treg', 1, 'ValueUnits', 'dimensionless', 'ConstantValue', false);
set(p, 'Notes', 'Hill function for CCL5/CCR5-mediated Treg recruitment');
addrule(model, 'H_CCL5_Treg = V_T.CCL5/(V_T.CCL5 + CCL5_50_Treg)', 'repeatedAssignment');

% Add Reactions
% CCL5 secretion by cancer cells
% Based on Tan et al.: Cancer cells in PDAC express FOXP3 which activates CCL5 transcription
reaction = addreaction(model, 'null -> V_T.CCL5');
set(reaction, 'ReactionRate', 'k_CCL5_sec*C_total');
set(reaction, 'Notes', 'CCL5 secretion by PDAC cancer cells (FOXP3-driven)');

% Note: CCL5 secretion by iCAFs is handled in fibroblast_module_enhanced.m

% CCL5 degradation
reaction = addreaction(model, 'V_T.CCL5 -> null');
set(reaction, 'ReactionRate', 'k_CCL5_deg*V_T.CCL5');
set(reaction, 'Notes', 'CCL5 degradation in tumor compartment');

% CCR5-dependent Treg recruitment from central compartment
% This is ADDITIONAL to the existing baseline q_Treg_T_in recruitment
% Tregs express CCR5 while CD8 T cells do not, providing selective recruitment
reaction = addreaction(model, 'V_C.Treg -> V_T.Treg');
set(reaction, 'ReactionRate', 'k_CCR5_Treg_rec*V_T*V_C.Treg*H_CCL5_Treg');
set(reaction, 'Notes', 'CCR5-dependent Treg recruitment to tumor (CCL5-mediated, selective for Tregs)');

warning('off', 'SimBiology:DimAnalysisNotDone_MatlabFcn_Dimensionless');

end
