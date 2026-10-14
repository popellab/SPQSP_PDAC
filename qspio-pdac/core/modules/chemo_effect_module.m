% Aggregate Cytotoxic Drug Effect Module
%
% Defines E_chemo, a single dimensionless [0,1] aggregate of the instantaneous
% cytotoxic pressure from all PASS-01 chemo drugs. Built as the complement-
% product of each drug's Emax engagement (the same Ct/(Ct+IC50) term used in the
% cell-kill reactions, reusing the calibrated kill IC50s):
%
%   E_chemo = 1 - PROD_d ( 1 - Ct_d / (Ct_d + IC50_d) )
%
% over d in {nabp, gem, oxa, iri, fu5}. It is regimen-general: undosed drugs have
% Ct_d = 0 -> their factor is 1 -> they contribute nothing, so the same rule
% serves GnP (nabp+gem) and mFOLFIRINOX (fu5+oxa+iri) without edits.
%
% E_chemo is a SHARED driver, defined once here and consumed by two downstream
% mechanisms:
%   - resistance_module : C1 -> C2 transition is SELECTED by drug pressure
%                         (rate = k_resist * V_T.C1 * E_chemo)
%   - ca199_module      : per-cell CA19-9 secretion is suppressed under chemo
%                         (rate = k_ca199_sec / (1 + k_sec_suppress*E_chemo) * C_total)
%
% Dependencies (must run AFTER every chemo drug module):
%   nabpaclitaxel_module -- Ct_nabp, IC50_nabp
%   chemo_pk_module x4   -- Ct_gem/oxa/iri/fu5, IC50_gem/oxa/iri/fu5
%
% Inputs:  model  -- SimBiology model object
% Outputs: model  -- model with the E_chemo parameter + assignment rule

function model = chemo_effect_module(model)

	addparameter(model, 'E_chemo', 0, 'ValueUnits', 'dimensionless', 'ConstantValue', false, ...
		'Notes', 'Aggregate chemo drug effect on tumor [0,1] (complement-product of per-drug Emax)');
	addrule(model, ['E_chemo = 1 - (1 - Ct_nabp/(Ct_nabp+IC50_nabp))' ...
		'*(1 - Ct_gem/(Ct_gem+IC50_gem))*(1 - Ct_oxa/(Ct_oxa+IC50_oxa))' ...
		'*(1 - Ct_iri/(Ct_iri+IC50_iri))*(1 - Ct_fu5/(Ct_fu5+IC50_fu5))'], 'repeatedAssignment');

end
