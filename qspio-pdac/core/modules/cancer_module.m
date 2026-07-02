function model = cancer_module(model, species_name, params, varargin)
	%CANCER_MODULE Add a tumor growth submodel to a SimBiology model.
	%
	%   model = cancer_module(model, species_name, params, ...) adds a cancer
	%   cell species to the specified compartment and includes logistic or
	%   Gompertzian growth, death, and vascular interactions.
	%
	%   Optional inputs (name-value pairs):
	%       'initial_amount' - initial tumor burden (default: 1)
	%       'model_type'     - 'Logistic' or 'Gompertzian' (default: 'Logistic')
	%       'compartment'    - name of the compartment to place the cancer cell (default: 'V_T')

		in = inputParser;
		addOptional(in, 'initial_amount', 1);
		addOptional(in, 'model_type', 'Logistic', @(s)ischar(s));
		addOptional(in, 'compartment', 'V_T', @(s)ischar(s));
		parse(in, varargin{:});

		init = in.Results.initial_amount;
		model_type = in.Results.model_type;
		comp_name = in.Results.compartment;

		% Lookup compartment
		comp = sbioselect(model, 'Type', 'compartment', 'Name', comp_name);
		if isempty(comp)
				error('Compartment "%s" not found in model.', comp_name);
		end

		% Add cancer species
		C = addspecies(comp, 'C', init, 'InitialAmountUnits', 'cell', 'Notes', 'Number of cancer cells in tumor');

		% Add growth and death parameters
		k_growth = addparameter(model, ['k_' species_name '_growth'], params.(['k_' species_name '_growth']).Value, 'ValueUnits', params.(['k_' species_name '_growth']).Units, 'ConstantValue', false, 'Notes', 'Cancer growth rate');
		k_growth.UserData.Source = params.(['k_' species_name '_growth']).Notes;
		k_growth.UserData.derivedFrom = params.(['k_' species_name '_growth']).derivedFrom;

		k_death = addparameter(model, ['k_' species_name '_death'], params.(['k_' species_name '_death']).Value, 'ValueUnits', params.(['k_' species_name '_death']).Units, 'ConstantValue', false, 'Notes', 'Cancer death rate');
		k_death.UserData.Source = params.(['k_' species_name '_death']).Notes;
		k_death.UserData.derivedFrom = params.(['k_' species_name '_death']).derivedFrom;

		% Add carrying capacity
		try
				C_max = addparameter(model, 'C_max', params.C_max.Value, 'ValueUnits', params.C_max.Units, 'ConstantValue', false, 'Notes', 'Current tumor carrying capacity');
				C_max.UserData.Source = params.C_max.Notes;
				C_max.UserData.derivedFrom = params.C_max.derivedFrom;
				first_call = true;
		catch
				first_call = false;
				% C_max already exists - multiple cancer cell types present, skipping vasculature re-initialization
		end

		% Therapy parameter and rule
		param = addparameter(model, ['k_' species_name '_therapy'], 0, 'ValueUnits', '1/day', 'ConstantValue', false, 'Notes', 'Therapy-induced death rate');
		addrule(model, ['k_' species_name '_therapy = 0/day'], 'repeatedAssignment');

		% Initial tumor diameter (once)
		try
				p = addparameter(model, 'initial_tumour_diameter', params.initial_tumour_diameter.Value, 'ValueUnits', params.initial_tumour_diameter.Units, 'Notes', 'Pre-treatment tumor diameter');
		catch
				% initial_tumour_diameter already exists - this is expected for multiple cancer cell types
		end

		% Add growth reaction
		switch lower(model_type)
				case 'gompertzian'
						rate = ['k_' species_name '_growth*V_T.C*log(max(C_max/(C_total+cell), 1))'];
				otherwise
						rate = ['k_' species_name '_growth*V_T.C*(1 - C_total/C_max)'];
		end
		addreaction(model, 'null -> V_T.C', 'ReactionRate', rate, 'Notes', 'Tumor cell growth');

		% Add death reaction
		addreaction(model, 'V_T.C -> V_T.C_x', 'ReactionRate', ['k_' species_name '_death*V_T.C'], 'Notes', 'Tumor cell death');

		% Add vasculature only on first call
		if first_call
				% Add K_initial as a parameter for calibration
				p = addparameter(model, 'K_initial', params.K_initial.Value, 'ValueUnits', params.K_initial.Units, 'ConstantValue', true, 'Notes', 'Initial tumor carrying capacity');
				p.UserData.Source = params.K_initial.Notes;
				p.UserData.derivedFrom = params.K_initial.derivedFrom;

				addspecies(comp, 'K', params.K_initial.Value, 'InitialAmountUnits', params.K_initial.Units, 'Notes', 'Tumor carrying capacity (vasculature-dependent)');
				addspecies(comp, 'VEGF', 0, 'InitialAmountUnits', 'picogram/milliliter', 'Notes', 'VEGF concentration in tumor');

				p = addparameter(model, 'k_vas_growth', params.k_vas_growth.Value, 'ValueUnits', params.k_vas_growth.Units, 'ConstantValue', true, 'Notes', 'Tumor vasculature growth rate');
				p.UserData.Source = params.k_vas_growth.Notes;
				p.UserData.derivedFrom = params.k_vas_growth.derivedFrom;
				p = addparameter(model, 'k_vas_decay', params.k_vas_decay.Value, 'ValueUnits', params.k_vas_decay.Units, 'ConstantValue', true, 'Notes', 'Tumor vasculature inhibition rate');
				p.UserData.Source = params.k_vas_decay.Notes;
				p.UserData.derivedFrom = params.k_vas_decay.derivedFrom;
				p = addparameter(model, 'k_vas_Csec', params.k_vas_Csec.Value, 'ValueUnits', params.k_vas_Csec.Units, 'ConstantValue', true, 'Notes', 'Angiogenic factor secretion rate');
				p.UserData.Source = params.k_vas_Csec.Notes;
				p.UserData.derivedFrom = params.k_vas_Csec.derivedFrom;
				p = addparameter(model, 'k_vas_deg', params.k_vas_deg.Value, 'ValueUnits', params.k_vas_deg.Units, 'ConstantValue', true, 'Notes', 'Angiogenic factor degradation rate');
				p.UserData.Source = params.k_vas_deg.Notes;
				p.UserData.derivedFrom = params.k_vas_deg.derivedFrom;
				p = addparameter(model, 'VEGF_50', params.VEGF_50.Value, 'ValueUnits', params.VEGF_50.Units, 'ConstantValue', true, 'Notes', 'Half-maximal angiogenic factor concentration for tumor capacity growth');
				p.UserData.Source = params.VEGF_50.Notes;
				p.UserData.derivedFrom = params.VEGF_50.derivedFrom;

				addrule(model, 'C_max = V_T.K', 'repeatedAssignment');
				addreaction(model, 'null -> V_T.VEGF', 'ReactionRate', 'k_vas_Csec*C_total', 'Notes', 'VEGF secretion by cancer cells');
				addreaction(model, 'V_T.VEGF -> null', 'ReactionRate', 'k_vas_deg*V_T.VEGF', 'Notes', 'VEGF degradation');
				addreaction(model, 'null -> V_T.K', 'ReactionRate', 'k_vas_growth*C_total*V_T.VEGF/(V_T.VEGF + VEGF_50)', 'Notes', 'Vascular growth');
				addreaction(model, 'V_T.K -> null', 'ReactionRate', 'k_vas_decay*V_T.K*(max(C_total,0)/cell*2.57e-6)^(2/3)', 'Notes', 'Vascular inhibition');
				addevent(model, 'C_total < 0.5*cell', 'V_T.K = 0.01*cell');
		end

		% Eradication threshold
		addevent(model, 'V_T.C < 0.5*cell', 'V_T.C = 0.01*cell');

		% Update C_total rule
		model_rules = get(model, 'Rules');
		total_cancer_rule = model_rules(2);
		rule = get(total_cancer_rule, 'Rule');
		set(total_cancer_rule, 'Rule', [rule ' + V_T.' species_name]);

		% Rename
		rename(C, species_name);
		rename(k_growth, ['k_' species_name '_growth']);
		rename(k_death, ['k_' species_name '_death']);
end
