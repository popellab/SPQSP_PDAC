function [results, metadata] = PSA_simulate(model, params_in, dose_schedule, config, varargin)
%PSA_SIMULATE Run batch simulations for Parameter Sensitivity Analysis
%
% Features intelligent caching to avoid re-running identical simulations.
% Caches are stored in project-specific directories and validated for integrity.
% Supports both local and SLURM cluster execution modes.
%
% Inputs:
%   model         - SimBiology model object
%   params_in     - Parameter structure from generate_parameter_samples()
%   dose_schedule - Dosing schedule object
%   config        - Ensemble configuration from ensemble_config()
%   varargin      - Optional parameters:
%                   'overwrite', true/false - Force overwrite cache (default: false)
%                   'project_name', 'name' - Set custom project name (default: auto-detect)
%                   'seed', value - Random seed used for parameter generation (for cache key)
%                   'execution_mode', 'local'|'slurm' - Execution mode (default: 'local')
%                   'batch_config', config - Batch configuration (required for slurm mode)
%
% Outputs:
%   results   - Array of simulation results
%   metadata  - Simulation metadata (includes cache info if loaded)
%
% Examples:
%   % Local execution (default)
%   [results, metadata] = PSA_simulate(model, params_in, dose_schedule, config);
%
%   % SLURM cluster execution
%   batch_config = batch_config();
%   [results, metadata] = PSA_simulate(model, params_in, dose_schedule, config, ...
%       'execution_mode', 'slurm', 'batch_config', batch_config);

% Parse optional inputs
p = inputParser;
addParameter(p, 'overwrite', false, @islogical);
addParameter(p, 'project_name', '', @ischar);
addParameter(p, 'seed', [], @isnumeric);
addParameter(p, 'execution_mode', 'local', @(x) ismember(x, {'local', 'slurm'}));
addParameter(p, 'batch_config', [], @isstruct);
parse(p, varargin{:});
overwrite_cache = p.Results.overwrite;
custom_project_name = p.Results.project_name;
seed_value = p.Results.seed;
execution_mode = p.Results.execution_mode;
batch_config = p.Results.batch_config;

% Validate batch configuration for SLURM execution
if strcmp(execution_mode, 'slurm')
    if isempty(batch_config)
        error('Batch configuration required when execution_mode is ''slurm''. Use batch_config() to create one.');
    end
    % Delegate to batch implementation
    [results, metadata] = batch_simulate(model, params_in, dose_schedule, config, batch_config, varargin{:});
    return;
end

n_patients = config.n_patients;

% Simple caching system
if ~isempty(custom_project_name)
    project_name = custom_project_name;
else
    project_name = get_current_project_name();
end
cache_key = generate_simple_cache_key(params_in, config, seed_value);
cache_dir = fullfile('projects', project_name, 'cache', 'PSA_simulations');
cache_file = fullfile(cache_dir, sprintf('psa_cache_%s.mat', cache_key));

% Check for existing cache (unless overwrite requested)
if ~overwrite_cache && exist(cache_file, 'file')
    fprintf('🔄 Found existing cache, loading results...\n');
    fprintf('   Cache key: %s\n', cache_key);
    fprintf('   n_patients: %d\n', config.n_patients);
    if isfield(params_in, 'all') && ~isempty(params_in.all)
        fprintf('   params matrix size: %dx%d\n', size(params_in.all, 1), size(params_in.all, 2));
    end
    try
        cached_data = load(cache_file);
        results = cached_data.results;
        metadata = cached_data.metadata;
        metadata.cache_loaded = true;
        return;
    catch
        fprintf('⚠️  Cache loading failed, proceeding with new simulation\n');
    end
elseif overwrite_cache && exist(cache_file, 'file')
    fprintf('🗑️  Overwriting existing cache...\n');
end

% No valid cache found, run simulation
fprintf('🚀 Running PSA simulations for %d patients (local execution)...\n', n_patients);
fprintf('   Cache key: %s\n', cache_key);

results = [];
metadata = struct();
metadata.status = zeros(n_patients, 1);  % 1=success, 0=failed_IC, -1=failed_sim
metadata.cache_loaded = false;
metadata.execution_mode = 'local';

for i = 1:n_patients
    fprintf('  Patient %d/%d\n', i, n_patients);

    % Clean up previous model copy
    if i > 1, delete(model_copy); end

    % Create model copy and parameter variant
    model_copy = copyobj(model);
    variant = create_parameter_variant(model_copy, params_in, i);

    % Set initial conditions
    try
        [model_copy, ic_success, ~] = initial_conditions(model_copy, 'Variant', variant);

        if ~ic_success
            metadata.status(i) = 0;  % Failed IC
            results(i).simData = [];
            continue;
        end

    catch
        metadata.status(i) = 0;  % Failed IC
        results(i).simData = [];
        continue;
    end

    % Run simulation
    try
        sim_data = sbiosimulate(model_copy, [], variant, dose_schedule);
        metadata.status(i) = 1;   % Success
        results(i).simData = sim_data;

    catch
        metadata.status(i) = -1;  % Failed simulation
        results(i).simData = [];
    end
end

% Summary
n_success = sum(metadata.status == 1);
n_failed_ic = sum(metadata.status == 0);
n_failed_sim = sum(metadata.status == -1);

fprintf('Simulation complete: %d success, %d failed IC, %d failed sim\n', ...
    n_success, n_failed_ic, n_failed_sim);

metadata.summary = struct('n_success', n_success, 'n_failed_ic', n_failed_ic, 'n_failed_sim', n_failed_sim);

% Save results to cache for future use
if n_success > 0  % Only cache if we have some successful results
    if ~exist(cache_dir, 'dir')
        mkdir(cache_dir);
    end
    save(cache_file, 'results', 'metadata', 'params_in', 'config');
    fprintf('💾 Results cached to: %s\n', cache_file);
end

end

function cache_key = generate_simple_cache_key(params_in, config, seed_value)
%GENERATE_SIMPLE_CACHE_KEY Generate a simple cache key from simulation inputs

% Create a hash from the key simulation parameters
key_parts = {};
key_parts{end+1} = sprintf('n%d', config.n_patients);

% Add sorted parameter names
param_names = sort(params_in.names);
key_parts{end+1} = strjoin(param_names, '-');

% Add seed if provided (this makes the cache key depend on the exact parameter values)
if ~isempty(seed_value)
    key_parts{end+1} = sprintf('s%d', seed_value);
end

% Add parameter checksums from actual LHS values (if available)
if isfield(params_in, 'all') && ~isempty(params_in.all)
    % Use a checksum of the parameter matrix
    param_checksum = round(1000 * sum(params_in.all(:)));
    key_parts{end+1} = sprintf('p%d', param_checksum);
end

% Create readable cache key
key_string = strjoin(key_parts, '_');
hash_val = abs(java.lang.String(key_string).hashCode());
cache_key = sprintf('%08x', hash_val);

end

function project_name = get_current_project_name()
%GET_CURRENT_PROJECT_NAME Determine current project name from working directory
%
% Returns the project name based on current working directory or 'default'
% 
% To set a specific project name for your PSA:
% 1. Create a subdirectory: projects/your_project_name/
% 2. Navigate to that directory: cd projects/your_project_name/
% 3. Run your PSA script from there
%
% Example:
%   mkdir projects/pdac_2025
%   cd projects/pdac_2025
%   run('../../scripts/PSA_script_PDAC_GVAX.m')

current_dir = pwd;

% Look for projects/[project_name] pattern in path
project_match = regexp(current_dir, '.*[/\\]projects[/\\]([^/\\]+)', 'tokens');

if ~isempty(project_match) && ~isempty(project_match{1})
    project_name = project_match{1}{1};
else
    % Fallback: use 'default' if not in a project directory
    project_name = 'default';
    fprintf('📁 Using default project name. To use a custom project name:\n');
    fprintf('   1. cd projects/your_project_name/\n');
    fprintf('   2. Run your PSA script from there\n');
end

end

function variant = create_parameter_variant(model, params_in, patient_index)
%CREATE_PARAMETER_VARIANT Create parameter variant for a specific patient

variant_name = sprintf('patient_%05d', patient_index);
variant = addvariant(model, variant_name);

for j = 1:length(params_in.names)
    param_name = params_in.names{j};
    param_value = params_in.(param_name).LHS(patient_index);

    % Check if parameter exists in model
    param_obj = sbioselect(model, 'Type', 'parameter', 'Name', param_name);
    compartment_obj = sbioselect(model, 'Type', 'compartment', 'Name', param_name);

    if ~isempty(param_obj)
        addcontent(variant, {'parameter', param_name, 'Value', param_value});
    elseif ~isempty(compartment_obj)
        addcontent(variant, {'compartment', param_name, 'Capacity', param_value});
    end
end

end
