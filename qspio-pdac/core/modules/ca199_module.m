% CA19-9 Serum Biomarker Module
%
% Adds serum CA19-9 (sialyl-Lewis^a) as a secreted tumor-burden biomarker for
% longitudinal calibration against PASS-01 serial CA19-9 (measured in kU/L = U/mL).
%
% Model form -- production proportional to live tumor burden PLUS shedding from
% the dying-cell pool, with first-order serum elimination:
%
%   d[V_C.CA199]/dt = k_ca199_sec/(1+k_sec_suppress*E_chemo)
%                        * (V_T.C1 + f_ca199_C2*V_T.C2)  (live-cell secretion, chemo-
%                                                         suppressed, clone-weighted)
%                   + k_ca199_shed * V_T.C_x         (release from the dead-cell pool)
%                   - k_ca199_clear* V_C.CA199       (serum elimination, first order)
%
%   - V_T.C1/C2 : live sensitive / resistant cancer clones; the resistant clone
%                 secretes CA19-9 at f_ca199_C2x the per-cell rate (Update 3),
%                 decoupling the biomarker rise from bulk RECIST volume
%   - V_T.C_x : dead/dying cancer pool (cleared at k_cell_clear); the shed term
%               produces the transient CA19-9 SPIKE on chemotherapy-induced lysis
%               before the decline, matching the clinically observed early rise.
%   - V_C     : central (serum) compartment where the biomarker is measured
%
% The absolute scale of CA199 is nominal: the per-patient k_ca199_sec (and shed
% rate) absorb the mapping to measured U/mL during calibration. CA19-9
% non-secretors (Lewis-null, ~5-10%) correspond to k_ca199_sec ~ 0 and are
% handled at the likelihood level (CA19-9 arm dropped for those patients).
%
% Dependencies (must run AFTER cancer_module, simbio_init, chemo_effect_module):
%   cancer_module       -- V_T.C1, global C_total rule
%   simbio_init         -- V_T.C_x (+ k_cell_clear), V_C compartment
%   chemo_effect_module -- E_chemo (drug-suppressed secretion driver)
%
% Inputs:  model  -- SimBiology model object
%          params -- parameter struct (params_out) with k_ca199_* entries
% Outputs: model  -- model with serum CA19-9 species + secretion/clearance reactions

function model = ca199_module(model, params)

	% --- Serum CA19-9 species (central compartment) ---
	comp_C = sbioselect(model, 'Type', 'compartment', 'Name', 'V_C');
	if isempty(comp_C)
		error('ca199_module: central compartment V_C not found (run simbio_init first).');
	end
	addspecies(comp_C, 'CA199', 0, 'InitialAmountUnits', 'picogram/milliliter', ...
		'Notes', 'Serum CA19-9 tumor-burden biomarker (nominal units; calibrated to U/mL).');

	% --- Parameters ---
	p = addparameter(model, 'k_ca199_sec', params.k_ca199_sec.Value, ...
		'ValueUnits', params.k_ca199_sec.Units, 'ConstantValue', false, ...
		'Notes', 'CA19-9 secretion rate per live cancer cell (patient-level)');
	p.UserData.Source = params.k_ca199_sec.Notes;
	p.UserData.derivedFrom = params.k_ca199_sec.derivedFrom;

	p = addparameter(model, 'k_ca199_shed', params.k_ca199_shed.Value, ...
		'ValueUnits', params.k_ca199_shed.Units, 'ConstantValue', false, ...
		'Notes', 'CA19-9 shedding rate from the dying/dead cancer-cell pool (V_T.C_x)');
	p.UserData.Source = params.k_ca199_shed.Notes;
	p.UserData.derivedFrom = params.k_ca199_shed.derivedFrom;

	p = addparameter(model, 'k_ca199_clear', params.k_ca199_clear.Value, ...
		'ValueUnits', params.k_ca199_clear.Units, 'ConstantValue', false, ...
		'Notes', 'CA19-9 serum first-order elimination rate');
	p.UserData.Source = params.k_ca199_clear.Notes;
	p.UserData.derivedFrom = params.k_ca199_clear.derivedFrom;

	% --- Drug-suppressed secretion (v2, 2026-07-13) ---
	% Viable cancer cells down-regulate CA19-9 shedding under chemotherapy stress,
	% so serum CA19-9 crashes FASTER and DEEPER than bulk tumor volume (the v1
	% secretion = k_sec*C_total was too rigidly tied to burden -> nadir too shallow).
	% Suppress per-cell secretion by the aggregate chemo drug effect E_chemo
	% (defined in chemo_effect_module: complement-product of each drug's Emax,
	% reusing the kill IC50s; regimen-general). One extra param here (k_sec_suppress).
	p = addparameter(model, 'k_sec_suppress', params.k_sec_suppress.Value, ...
		'ValueUnits', params.k_sec_suppress.Units, 'ConstantValue', false, ...
		'Notes', 'CA19-9 secretion-suppression strength under aggregate chemo drug effect');
	p.UserData.Source = params.k_sec_suppress.Notes;
	p.UserData.derivedFrom = params.k_sec_suppress.derivedFrom;

	% --- Clone-weighted secretion (Update 3, 2026-07-14) ---
	% The resistant clone C2 secretes CA19-9 at f_ca199_C2x the per-cell rate of
	% C1, so live-cell secretion is weighted (V_T.C1 + f_ca199_C2*V_T.C2) instead
	% of bulk C_total (= C1 + C2). This DECOUPLES the biomarker rise from bulk
	% RECIST volume: a modest resistant-nodule volume drives a large CA19-9
	% rebound (progressor SLD<->CA19-9 discordance). f_ca199_C2=1 recovers v3.
	p = addparameter(model, 'f_ca199_C2', params.f_ca199_C2.Value, ...
		'ValueUnits', params.f_ca199_C2.Units, 'ConstantValue', false, ...
		'Notes', 'Per-cell CA19-9 secretion of resistant clone C2 relative to C1');
	p.UserData.Source = params.f_ca199_C2.Notes;
	p.UserData.derivedFrom = params.f_ca199_C2.derivedFrom;

	% --- Reactions ---
	r = addreaction(model, 'null -> V_C.CA199');
	set(r, 'ReactionRate', 'k_ca199_sec/(1 + k_sec_suppress*E_chemo)*(V_T.C1 + f_ca199_C2*V_T.C2)', ...
		'Notes', 'CA19-9 secretion by live cancer cells (clone-weighted C1 + f_ca199_C2*C2), chemo-suppressed');

	r = addreaction(model, 'null -> V_C.CA199');
	set(r, 'ReactionRate', 'k_ca199_shed*V_T.C_x', ...
		'Notes', 'CA19-9 release from the dying-cell pool (chemotherapy lysis spike)');

	r = addreaction(model, 'V_C.CA199 -> null');
	set(r, 'ReactionRate', 'k_ca199_clear*V_C.CA199', ...
		'Notes', 'CA19-9 serum elimination (first order)');

end
