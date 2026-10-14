function [model_out, success, simData] = evolve_to_diagnosis(model_in, varargin)
% EVOLVE_TO_DIAGNOSIS Simulate natural history from healthy pancreas to diagnosis
%
% Initializes the model with a healthy pancreatic state (small tumor, homeostatic
% immune populations, minimal fibrosis) and evolves it forward in time to reach
% a diagnosis-state tumor microenvironment. This provides a mechanistic starting
% point for treatment simulations.
%
% Conceptual phases:
%   1. HEALTHY: Small tumor (~1e6 cells), normal tissue
%   2. EVOLUTION: Natural tumor progression (this function)
%   3. DIAGNOSIS: Established TME (output of this function)
%   4. TREATMENT: Treatment simulation starts from diagnosis state
%
% Inputs:
%   model_in  - SimBiology model object (pre-initialized with parameters)
%   varargin  - Optional parameters:
%               'EvolutionDays': time to evolve (default: 365 days)
%               'TumorCells': initial tumor cell count (default: 1e6)
%               'Variant': variant object with rate parameters (optional)
%               'Debug': true/false for verbose output (default: false)
%               'TargetDiameter': stop when tumor reaches this size in cm (default: none)
%
% Outputs:
%   model_out - Model initialized at diagnosis state (ready for treatment)
%   success   - true if evolution completed without errors
%   simData   - Natural history simulation data (healthy→diagnosis)
%
% Example usage:
%   % Evolve for 1 year from healthy state
%   model = evolve_to_diagnosis(model, 'EvolutionDays', 365, 'Debug', true);
%
%   % Evolve until tumor reaches 3 cm diameter
%   model = evolve_to_diagnosis(model, 'TargetDiameter', 3.0);
%
% See also: set_healthy_populations, initial_conditions

% Load shared healthy_state.yaml (same file the C++ side reads). Thresholds
% and default tumor cell count come from here so MATLAB and C++ never drift.
hs = yaml_read(healthy_state_path());

% Parse optional inputs
p = inputParser;
addParameter(p, 'EvolutionDays', 365);          % Default: 1 year evolution
addParameter(p, 'TumorCells', hs.default_tumor_cells);
addParameter(p, 'Variant', []);                 % Optional variant for rate parameters
addParameter(p, 'Debug', false);
addParameter(p, 'TargetDiameter', []);          % Optional: stop at specific tumor size
parse(p, varargin{:});

evolution_days = p.Results.EvolutionDays;
tumor_cells = p.Results.TumorCells;
Variant = p.Results.Variant;
Debug = p.Results.Debug;
target_diameter = p.Results.TargetDiameter;

% Evolution-time guards live in the shared YAML (same values the C++
% evolve_to_diagnosis uses).
min_evolution_days = hs.evolve.min_days;
max_evolution_years_days = hs.evolve.max_days;

% Auto-extract TargetDiameter from variant if not explicitly provided
if isempty(target_diameter) && ~isempty(Variant)
    for i = 1:length(Variant.Content)
        if strcmp(Variant.Content{i}{2}, 'initial_tumour_diameter')
            target_diameter = Variant.Content{i}{4};
            fprintf('DEBUG: Auto-extracted TargetDiameter from variant: %.2f cm\n', target_diameter);
            break;
        end
    end
end

% Fallback: read initial_tumour_diameter from the model parameters
% (handles the case where it's set in parameters_PDAC.m but not in the variant)
if isempty(target_diameter)
    param_obj = sbioselect(model_in, 'Name', 'initial_tumour_diameter', 'Type', 'parameter');
    if ~isempty(param_obj)
        target_diameter = param_obj.Value;
        fprintf('DEBUG: Read TargetDiameter from model parameter: %.2f cm\n', target_diameter);
    end
end

% Always print target diameter if specified
if ~isempty(target_diameter)
    fprintf('DEBUG: evolve_to_diagnosis will evolve to target diameter: %.2f cm\n', target_diameter);
else
    fprintf('DEBUG: evolve_to_diagnosis will evolve for %d days (no target diameter)\n', evolution_days);
end

model = copyobj(model_in);

% =========================================================================
% STEP 1: Apply variant rate parameters (if provided)
% =========================================================================
% Note: We do NOT apply IC parameters from variant - we use healthy state
if ~isempty(Variant) && Debug
    fprintf('Applying rate parameters from variant...\n');
end

% =========================================================================
% STEP 2: Set healthy baseline populations
% =========================================================================
if Debug
    fprintf('\n=== PHASE 1: HEALTHY STATE ===\n');
end

model = set_healthy_populations(model, ...
    'TumorCells', tumor_cells, ...
    'Verbose', Debug);

% =========================================================================
% STEP 3: Configure and run natural history evolution
% =========================================================================
if Debug
    fprintf('\n=== PHASE 2: NATURAL HISTORY EVOLUTION ===\n');
    if ~isempty(target_diameter)
        fprintf('Evolving from healthy → diagnosis (target: %.2f cm diameter)...\n', target_diameter);
    else
        fprintf('Evolving from healthy → diagnosis for %d days...\n', evolution_days);
    end
end

% Get user-defined configuration settings
config = getconfigset(model);
user_output_times = config.SolverOptions.OutputTimes;
user_max_step = config.SolverOptions.MaxStep;

% Configure for natural history simulation
% If TargetDiameter specified, evolve for longer to ensure we reach it
if ~isempty(target_diameter)
    max_evolution_time = max_evolution_years_days;  % from healthy_state.yaml
else
    max_evolution_time = evolution_days;
end

set(config, 'StopTime', max_evolution_time);
set(config.SolverOptions, 'OutputTimes', []);  % Let solver choose
set(config.SolverOptions, 'MaxStep', 1.0);     % Daily resolution

% Run evolution simulation
try
    if ~isempty(Variant)
        simData = sbiosimulate(model, Variant);
    else
        simData = sbiosimulate(model);
    end
    success = true;
catch ME
    if Debug
        fprintf('Natural history evolution failed: %s\n', ME.message);
    end
    model_out = copyobj(model);
    success = false;
    simData = [];
    return
end

% =========================================================================
% STEP 4: Extract diagnosis state from evolution endpoint or target diameter
% =========================================================================

% If TargetDiameter specified, find the timepoint where diameter is reached
if ~isempty(target_diameter)
    % Find C1 (tumor cells) in simulation data
    C1_idx = find(strcmp(simData.DataNames, 'V_T.C1'));
    if isempty(C1_idx)
        C1_idx = find(strcmp(simData.DataNames, 'C1'));
    end

    if ~isempty(C1_idx)
        % Use V_T from simulation (total tumor volume: cancer + immune + stromal + collagen)
        % This matches what imaging measures (total tumor mass)
        V_T_idx_diam = find(strcmp(simData.DataNames, 'V_T'));
        if ~isempty(V_T_idx_diam)
            V_T_trajectory = simData.Data(:, V_T_idx_diam);  % mL (≈ cm³)
            diam_cm = (6.0 * V_T_trajectory / pi).^(1/3);  % diameter in cm
        else
            % Fallback: estimate from cancer cells using vol_cell
            C1_trajectory = simData.Data(:, C1_idx);
            vol_cell_cm3 = 4/3 * pi * (17e-4/2)^3;  % D_cell=17µm → cm → cm³
            diam_cm = (6.0 * C1_trajectory * vol_cell_cm3 / pi).^(1/3);
        end

        % Find first timepoint where diameter >= target
        idx = find(diam_cm >= target_diameter, 1, 'first');

        if isempty(idx)
            fprintf('DEBUG: Target diameter %.2f cm NOT reached (max: %.2f cm) - REJECTING simulation\n', ...
                target_diameter, max(diam_cm));
            % Return failure - this parameter set incompatible with having a resectable tumor
            model_out = copyobj(model);
            success = false;
            return
        end

        % Check minimum evolution time (biologically implausible if too fast).
        % PDAC doubling times are 26-500+ days; ~12 doublings from 1e6 cells
        % to 3.2 cm means the configured minimum is not biologically plausible.
        % min_evolution_days comes from healthy_state.yaml (shared with C++).
        if simData.Time(idx) < min_evolution_days
            fprintf('DEBUG: Target diameter reached too quickly (day %.1f < %d) - REJECTING simulation\n', ...
                simData.Time(idx), min_evolution_days);
            model_out = copyobj(model);
            success = false;
            return
        end

        fprintf('DEBUG: Target diameter reached at day %.1f (diameter: %.2f cm, target: %.2f cm)\n', ...
            simData.Time(idx), diam_cm(idx), target_diameter);
    else
        % Cannot determine tumor diameter without C1 - reject if target diameter was specified
        fprintf('DEBUG: Could not find C1 in simulation data - cannot verify target diameter - REJECTING\n');
        model_out = copyobj(model);
        success = false;
        return
    end
else
    % Use last time point = diagnosis
    idx = size(simData.Data, 1);
end

% Set species to diagnosis state
C1_final = NaN;  % Track C1 for debugging
for i = 1:length(model.Species)
    if i > size(simData.Data, 2)
        continue;
    end

    try
        new_value = simData.Data(idx, i);
        if isfinite(new_value) && new_value >= 0
            model.Species(i).InitialAmount = new_value;
            % Track C1 for debugging
            if strcmp(model.Species(i).Name, 'C1') || strcmp(model.Species(i).Name, 'V_T.C1')
                C1_final = new_value;
            end
        else
            model.Species(i).InitialAmount = 0;
        end
    catch
        model.Species(i).InitialAmount = 0;
    end
end

% Debug: print final C1 value set as initial condition
if ~isnan(C1_final)
    vol_cell_cm3 = 4/3 * pi * (17e-4/2)^3;  % D_cell=17µm → cm → cm³
    C1_diam_cm = (6*C1_final*vol_cell_cm3/pi)^(1/3);
    fprintf('DEBUG: Set C1 initial condition to %.2e cells (cancer-only diameter: %.3f cm)\n', C1_final, C1_diam_cm);
end

% Set parameters to diagnosis state
for i = 1:length(model.Parameters)
    param_idx = find(strcmp(simData.DataNames, model.Parameters(i).Name));
    if ~isempty(param_idx)
        model.Parameters(i).Value = simData.Data(idx, param_idx);
    end
end

% Set compartments to diagnosis state
for i = 1:length(model.Compartments)
    comp_idx = find(strcmp(simData.DataNames, model.Compartments(i).Name));
    if ~isempty(comp_idx)
        model.Compartments(i).Capacity = simData.Data(idx, comp_idx);
    end
end

% Reset initial assignment rules
for k = 1:length(model.Rules)
    if strcmp(model.Rules(k).RuleType, 'initialAssignment')
        model.Rules(k).Active = false;
    end
end

% Reset configuration settings
set(config, 'StopTime', user_output_times(end));
set(config.SolverOptions, 'OutputTimes', user_output_times);
if ~isempty(user_max_step)
    set(config.SolverOptions, 'MaxStep', user_max_step);
end

% =========================================================================
% STEP 5: Report diagnosis state
% =========================================================================
if Debug
    fprintf('\n=== PHASE 3: DIAGNOSIS STATE ===\n');

    % Tumor volume
    V_T_idx = find(strcmp(simData.DataNames, 'V_T'));
    if ~isempty(V_T_idx)
        V_T_final = simData.Data(idx, V_T_idx);
        fprintf('Tumor volume: %.3f mL\n', V_T_final);
    end

    % Tumor diameter from V_T (consistent with stopping criterion)
    if ~isempty(V_T_idx)
        V_T_diag = simData.Data(idx, V_T_idx);
        diam_diag_cm = (6 * V_T_diag / pi)^(1/3);
        fprintf('Tumor diameter (from V_T): %.2f cm\n', diam_diag_cm);
    end

    % Tumor cells
    C1_debug_idx = find(strcmp(simData.DataNames, 'C1'));
    if ~isempty(C1_debug_idx)
        fprintf('Tumor cells (C1): %.3e\n', simData.Data(idx, C1_debug_idx));
    end

    % Key immune populations
    compartment_labels = {'V_C', 'V_P', 'V_T', 'V_LN'};
    CD8_idx = find(strcmp(simData.DataNames, 'CD8'));
    if ~isempty(CD8_idx)
        for ci = 1:length(CD8_idx)
            if ci <= length(compartment_labels)
                fprintf('CD8+ T cells (%s): %.3e\n', compartment_labels{ci}, simData.Data(idx, CD8_idx(ci)));
            else
                fprintf('CD8+ T cells (?): %.3e\n', simData.Data(idx, CD8_idx(ci)));
            end
        end
    end

    MDSC_idx = find(strcmp(simData.DataNames, 'MDSC'));
    if ~isempty(MDSC_idx)
        fprintf('MDSCs: %.3e\n', simData.Data(idx, MDSC_idx));
    end

    % Collagen
    col_idx = find(strcmp(simData.DataNames, 'collagen'));
    if ~isempty(col_idx) && ~isempty(V_T_idx)
        col_mass = simData.Data(idx, col_idx);
        V_mL = simData.Data(idx, V_T_idx);
        col_frac = col_mass / (V_mL * 1400);  % Assuming collagen density 1400 mg/mL
        fprintf('Collagen: %.3e mg (%.1f%% volume fraction)\n', col_mass, col_frac * 100);
    end

    % Cytokines
    CCL2_idx = find(strcmp(simData.DataNames, 'CCL2'));
    if ~isempty(CCL2_idx)
        fprintf('CCL2: %.3e nM\n', simData.Data(idx, CCL2_idx));
    end

    TGFb_idx = find(strcmp(simData.DataNames, 'TGFb'));
    if ~isempty(TGFb_idx)
        fprintf('TGF-β: %.3e nM\n', simData.Data(idx, TGFb_idx));
    end

    fprintf('\nDiagnosis state ready for treatment simulation.\n');
    fprintf('======================================\n');
end

model_out = copyobj(model);
delete(model);

end
