% Chemo-Resistance Clone Coupling Module
%
% Wires the drug-resistant cancer clone C2 into the tumor model. The C2 SPECIES
% (with its own Gompertzian growth, death -> V_T.C_x, and inclusion in the global
% C_total rule) is created upstream by cancer_module(model,'C2',...). This module
% adds the coupling that makes C2 a *resistant* clone rather than an independent
% tumor:
%
%   1. Growth/death tie -- C2 shares C1's per-cell growth and death kinetics
%      (assignment rules k_C2_growth = k_C1_growth, k_C2_death = k_C1_death),
%      so C2 carries NO free growth/death DOF: it is the same tumor, differing
%      only in drug sensitivity.
%
%   2. Drug-selected acquisition -- C1 converts to C2 in proportion to the
%      aggregate cytotoxic pressure E_chemo (from chemo_effect_module):
%          rate = k_resist * V_T.C1 * E_chemo
%      i.e. resistance is SELECTED by ongoing chemotherapy, not by TGFb tone
%      (the old nab-paclitaxel-only formulation used H_TGFb) and not
%      constitutively. Off-treatment (E_chemo -> 0) there is no new acquisition.
%
% The cross-drug KILL resistance itself (C2 killed at IC50_<drug> * r_resist by
% each cytotoxic agent) lives in the drug modules (nabpaclitaxel_module,
% chemo_pk_module), mirroring their C1 kill reactions -- that is where each
% drug's Ct/IC50/k_kill/Kc are in scope. This module owns only the C1->C2
% selection and the kinetic tie.
%
% Net new CALIBRATED parameters: k_resist (here) and r_resist (added by the drug
% modules). k_C2_growth/k_C2_death are pinned to C1 and are NOT free.
%
% NOTE (scope): C2 is currently invisible to immune killing (CD8/macrophage act
% on C1 only) -- cross-resistance is modeled for cytotoxic drugs, not immunity.
% A per-drug r_resist_<drug> split and immune killing of C2 are deferred.
%
% Dependencies (must run AFTER):
%   cancer_module('C2') -- V_T.C2 species, k_C2_growth/k_C2_death, C_total rule
%   cancer_module('C1') -- k_C1_growth/k_C1_death
%   chemo_effect_module -- E_chemo
%
% Inputs:  model  -- SimBiology model object
%          params -- parameter struct (params_out) with k_resist
% Outputs: model  -- model with the C1->C2 transition + kinetic tie rules

function model = resistance_module(model, params)

	% --- Transition rate parameter (drug-selected acquisition) ---
	p = addparameter(model, 'k_resist', params.k_resist.Value, ...
		'ValueUnits', params.k_resist.Units, 'ConstantValue', false, ...
		'Notes', 'Drug-selected C1->C2 resistance-acquisition rate per unit E_chemo');
	p.UserData.Source = params.k_resist.Notes;
	p.UserData.derivedFrom = params.k_resist.derivedFrom;

	% --- Kinetic tie: resistant clone inherits C1 growth/death (0 free DOF) ---
	addrule(model, 'k_C2_growth = k_C1_growth', 'repeatedAssignment');
	addrule(model, 'k_C2_death = k_C1_death', 'repeatedAssignment');

	% --- Drug-selected C1 -> C2 transition ---
	r = addreaction(model, 'V_T.C1 -> V_T.C2');
	set(r, 'ReactionRate', 'k_resist*V_T.C1*E_chemo', ...
		'Notes', 'Acquisition of chemo-resistance (C1->C2), selected by aggregate drug effect E_chemo');

end
