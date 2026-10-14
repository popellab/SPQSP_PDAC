function [simData, model, scenario] = run_median_simulation(scenario_yaml, varargin)
%RUN_MEDIAN_SIMULATION Core function to simulate a scenario with median prior parameters
%
% Simulates the QSP model under median parameters from pdac_priors.csv.
% This is the core simulation function used by plotting scripts.
%
% Inputs:
%   scenario_yaml - Path to scenario YAML file, or 'baseline' for no treatment
%   varargin      - Optional name-value pairs:
%                   'paper_root'  - pdac-paper checkout holding the priors and the
%                                   submodel YAML (default: '../pdac-paper', or
%                                   $PAPER_ROOT). Both moved there at the
%                                   2026-08-19 split; this repo is the model.
%                   'priors_csv'  - Path to priors CSV (default: <paper_root>/parameters/pdac_priors.csv)
%                   'sim_time'    - Simulation time vector (default: from scenario or 0:0.5:100)
%                   'verbose'     - Print progress messages (default: true)
%
% Outputs:
%   simData  - SimBiology simulation results
%   model    - Configured SimBiology model
%   scenario - Parsed scenario structure (empty struct for baseline)
%
% Example:
%   % Simulate GVAX + nivolumab scenario
%   [simData, model] = run_median_simulation('scenarios/gvax_nivo_neoadjuvant_zheng2022.yaml');
%
%   % Simulate baseline (no treatment)
%   [simData, model] = run_median_simulation('baseline');
%
%   % Custom simulation time
%   [simData, model] = run_median_simulation('baseline', 'sim_time', 0:1:200);

%% Parse inputs
p = inputParser;
addRequired(p, 'scenario_yaml', @ischar);
% The priors and the submodel posterior live in the pdac-paper checkout since
% the 2026-08-19 split. Resolved through one root so the pair cannot disagree
% about which checkout they came from.
default_paper_root = getenv('PAPER_ROOT');
if isempty(default_paper_root)
    default_paper_root = fullfile(fileparts(fileparts(mfilename('fullpath'))), '..', 'pdac-paper');
end
addParameter(p, 'paper_root', default_paper_root, @ischar);
addParameter(p, 'priors_csv', '', @ischar);
addParameter(p, 'submodel_yaml', '', @ischar);
addParameter(p, 'apply_submodel', true, @islogical);
addParameter(p, 'param_overrides', struct(), @isstruct);
addParameter(p, 'sim_time', [], @isnumeric);
addParameter(p, 'verbose', true, @islogical);
parse(p, scenario_yaml, varargin{:});

paper_root = p.Results.paper_root;
priors_csv = p.Results.priors_csv;
if isempty(priors_csv)
    priors_csv = fullfile(paper_root, 'parameters', 'pdac_priors.csv');
end
submodel_yaml = p.Results.submodel_yaml;
if isempty(submodel_yaml)
    submodel_yaml = fullfile(paper_root, 'parameters', 'submodel_priors.yaml');
end
apply_submodel = p.Results.apply_submodel;
param_overrides = p.Results.param_overrides;
sim_time_input = p.Results.sim_time;
verbose = p.Results.verbose;

% Anchor relative paths to the repo root (this file lives in <repo>/scripts/)
% so the function works regardless of the caller's working directory.
repo_root = fileparts(fileparts(mfilename('fullpath')));
priors_csv = resolve_path(priors_csv, repo_root);
submodel_yaml = resolve_path(submodel_yaml, repo_root);

%% Determine if baseline or scenario
is_baseline = strcmpi(scenario_yaml, 'baseline');

if verbose
    fprintf('=== Median Simulation ===\n');
end

%% Load scenario configuration (if not baseline)
scenario = struct();
if ~is_baseline
    if verbose
        fprintf('Loading scenario: %s\n', scenario_yaml);
    end
    scenario = yaml_read(scenario_yaml);
    if verbose && isfield(scenario, 'scenario_id')
        fprintf('Scenario ID: %s\n', scenario.scenario_id);
    end
else
    if verbose
        fprintf('Running baseline (no treatment) simulation\n');
    end
    scenario.scenario_id = 'baseline';
end

%% Load priors and compute median parameters
if verbose
    fprintf('Loading priors from: %s\n', priors_csv);
end
priors = readtable(priors_csv);

% Compute median (expected value) for each parameter
% For lognormal: median = exp(mu) where mu is dist_param1
median_params = struct();
for i = 1:height(priors)
    param_name = priors.name{i};
    dist_type = priors.distribution{i};
    p1 = priors.dist_param1(i);

    if strcmp(dist_type, 'lognormal')
        % median = exp(mu)
        median_val = exp(p1);
    elseif strcmp(dist_type, 'beta')
        % beta(alpha, beta): dist_param1 = alpha, dist_param2 = beta
        median_val = betainv(0.5, p1, priors.dist_param2(i));
    else
        error('Unsupported distribution type: %s', dist_type);
    end

    median_params.(param_name) = median_val;
end

if verbose
    fprintf('Loaded %d parameters with median values\n', height(priors));
end

%% Setup model
if verbose
    fprintf('Setting up PDAC model...\n');
end
sbioreset;

% Run base model setup
immune_oncology_model_PDAC;

%% Apply median parameters and species initial amounts to model
if verbose
    fprintf('Applying median values from priors...\n');
end
param_names = fieldnames(median_params);
[n_params_set, n_species_set, not_found] = apply_param_struct(model, median_params);

if verbose
    fprintf('Set %d parameters, %d species from priors (%d total)\n', ...
        n_params_set, n_species_set, length(param_names));
    if ~isempty(not_found)
        fprintf('  Not found in model (%d):\n', length(not_found));
        for i = 1:length(not_found)
            fprintf('    - %s\n', not_found{i});
        end
    end
end

%% Overlay submodel posterior medians (shadow base priors, same as sbi_runner)
% submodel_priors.yaml carries a posterior marginal for every QSP parameter
% that has submodel-target data; its median shadows the base CSV prior here so
% the cascade simulation matches the prior the SBI workflow actually trains on.
% Value selection (median / exp(mu) underflow fallback / invgamma) is shared
% with the C++ trace path via scripts/submodel_medians.py — single source.
if apply_submodel && ~isempty(submodel_yaml) && isfile(submodel_yaml)
    submodel_json = [tempname, '.json'];
    cleanup_json = onCleanup(@() delete_if_exists(submodel_json));
    py = python_exe(repo_root);
    exporter = fullfile(repo_root, 'scripts', 'submodel_medians.py');
    cmd = sprintf('"%s" "%s" --submodel-yaml "%s" --out-json "%s"', ...
        py, exporter, submodel_yaml, submodel_json);
    [status, cmdout] = system(cmd);
    if status ~= 0
        warning('run_median_simulation:submodel', ...
            'Could not compose submodel medians (skipping shadow layer):\n%s', cmdout);
    else
        sm_data = jsondecode(fileread(submodel_json));
        [n_sm_p, n_sm_s, ~] = apply_param_struct(model, sm_data.medians);
        if verbose
            fprintf('Overlaid %d submodel medians from %s (%d param, %d species)\n', ...
                n_sm_p + n_sm_s, submodel_yaml, n_sm_p, n_sm_s);
        end
    end
elseif apply_submodel && verbose
    fprintf('Submodel priors not found (%s) — using base CSV priors only\n', submodel_yaml);
end

%% Manual parameter overrides (HIGHEST precedence) — always printed so an
% override is never silently forgotten in a later run. Values are in each
% parameter's native model units.
if ~isempty(fieldnames(param_overrides))
    onames = fieldnames(param_overrides);
    fprintf('  [OVERRIDE] applying %d manual parameter override(s):\n', numel(onames));
    for i = 1:length(onames)
        pn = onames{i};
        pv = param_overrides.(pn);
        pobj = sbioselect(model, 'Name', pn, 'Type', 'parameter');
        if ~isempty(pobj)
            fprintf('    %-22s %12.6g -> %-12.6g [%s]\n', pn, pobj(1).Value, pv, pobj(1).ValueUnits);
            set(pobj, 'Value', pv);
        else
            sobj = sbioselect(model, 'Name', pn, 'Type', 'species');
            if ~isempty(sobj)
                fprintf('    %-22s (species init) -> %-12.6g\n', pn, pv);
                for s = 1:length(sobj), sobj(s).InitialAmount = pv; end
            else
                fprintf('    %-22s NOT FOUND in model (skipped)\n', pn);
            end
        end
    end
end

%% Configure dosing from scenario
% YAML schema: dosing.drugs, dosing.GVAX_dose, dosing.nivolumab_dose, etc.
dose_schedule = [];
drugs = {};

if ~is_baseline && isfield(scenario, 'dosing') && isfield(scenario.dosing, 'drugs')
    dosing_cfg = scenario.dosing;
    drugs = dosing_cfg.drugs;
    if ischar(drugs), drugs = {drugs}; end
    if isempty(drugs) || (iscell(drugs) && all(cellfun(@isempty, drugs)))
        drugs = {};
    end
    % Promote to top-level for downstream use
    scenario.drugs = drugs;

    if ~isempty(drugs)
    if verbose
        fprintf('Configuring dosing: %s\n', strjoin(drugs, ', '));
    end

    patient_weight = 70;
    patient_bsa = 1.9;
    if isfield(dosing_cfg, 'patientWeight'), patient_weight = dosing_cfg.patientWeight; end
    if isfield(dosing_cfg, 'patientBSA'), patient_bsa = dosing_cfg.patientBSA; end

    dosing_args = {'patientWeight', patient_weight, 'patientBSA', patient_bsa};

    % Pass scenario dose fields through in the units documented by the YAML.
    % In particular, nivolumab_dose is already mg/kg; schedule_dosing applies
    % patient weight exactly once. Generic pass-through also preserves the
    % explicit cyclophosphamide_* alias instead of silently using defaults.
    dosing_fields = fieldnames(dosing_cfg);
    for f = 1:length(dosing_fields)
        field_name = dosing_fields{f};
        if any(strcmp(field_name, {'drugs', 'patientWeight', 'patientBSA'}))
            continue;
        end
        field_value = dosing_cfg.(field_name);
        if endsWith(field_name, '_schedule')
            [sched_start, sched_int, sched_repeat] = parse_schedule(field_value);
            field_value = [sched_start, sched_int, sched_repeat];
        end
        dosing_args = [dosing_args, field_name, field_value]; %#ok<AGROW>
    end

    % Convert drug names for schedule_dosing
    drugs_for_dosing = drugs;
    for d = 1:length(drugs_for_dosing)
        if ~strcmp(drugs_for_dosing{d}, 'GVAX') && ~strcmp(drugs_for_dosing{d}, 'Cy')
            drugs_for_dosing{d} = lower(drugs_for_dosing{d});
        end
    end

    dose_schedule = schedule_dosing(drugs_for_dosing, dosing_args{:});
    end % ~isempty(drugs)
end

%% Configure simulation time
% YAML schema: sim_config.start_time, sim_config.stop_time
if ~isempty(sim_time_input)
    sim_time = sim_time_input;
elseif isfield(scenario, 'sim_config') && isfield(scenario.sim_config, 'stop_time')
    t_start = 0;
    if isfield(scenario.sim_config, 'start_time')
        t_start = scenario.sim_config.start_time;
    end
    sim_time = t_start:0.5:scenario.sim_config.stop_time;
else
    sim_time = 0:0.5:100;
end

if verbose
    fprintf('Simulation time: Day %.0f to %.0f\n', sim_time(1), sim_time(end));
end

%% Establish initial conditions via natural history evolution
if verbose
    fprintf('Establishing initial conditions (evolve to diagnosis)...\n');
end
[model, ic_success, ~] = evolve_to_diagnosis(model, 'Debug', verbose);

if ~ic_success
    error('Failed to establish initial conditions via evolve_to_diagnosis');
end

%% Configure and run simulation
if verbose
    fprintf('Running simulation...\n');
end
model = simulation_config(model, 'solver', 'sundials', 'time', sim_time, ...
                         'abs_tolerance', 1e-9, 'rel_tolerance', 1e-6);

tic;
if isempty(dose_schedule)
    simData = sbiosimulate(model);
else
    simData = sbiosimulate(model, dose_schedule);
end
sim_time_elapsed = toc;

if verbose
    fprintf('Simulation completed in %.2f seconds\n', sim_time_elapsed);
end

end


function [start_day, interval, repeat_count] = parse_schedule(sched)
%PARSE_SCHEDULE Parse a dosing schedule from YAML format
% Converts [start, interval, num_doses] to schedule_dosing format

if iscell(sched)
    start_day = sched{1};
    interval_raw = sched{2};
    num_doses = sched{3};
else
    start_day = sched(1);
    interval_raw = sched(2);
    num_doses = sched(3);
end

% Handle null interval for single dose
if isempty(interval_raw) || isnan(interval_raw)
    interval = 1;
    repeat_count = 0;
else
    interval = interval_raw;
    repeat_count = num_doses - 1;
end
end


function scenario = yaml_read(filepath)
%YAML_READ Simple YAML parser for scenario files

fid = fopen(filepath, 'r');
if fid == -1
    error('Cannot open file: %s', filepath);
end

scenario = struct();
current_parent = '';
multiline_key = '';
multiline_indent = 0;

while ~feof(fid)
    line = fgetl(fid);

    if isempty(line) || startsWith(strtrim(line), '#')
        continue;
    end

    % Remove inline comments
    comment_idx = find(line == '#', 1, 'first');
    if ~isempty(comment_idx)
        line = line(1:comment_idx-1);
    end

    stripped = strtrim(line);
    indent = length(line) - length(stripped);

    % Handle multiline strings (|)
    if contains(stripped, ': |')
        key = strtrim(extractBefore(stripped, ':'));
        if indent == 0
            scenario.(key) = '';
            multiline_key = key;
        else
            scenario.(current_parent).(key) = '';
            multiline_key = [current_parent, '.', key];
        end
        multiline_indent = indent;
        continue;
    end

    % Continue multiline string
    if ~isempty(multiline_key) && indent > multiline_indent
        if contains(multiline_key, '.')
            parts = strsplit(multiline_key, '.');
            scenario.(parts{1}).(parts{2}) = [scenario.(parts{1}).(parts{2}), stripped, ' '];
        else
            scenario.(multiline_key) = [scenario.(multiline_key), stripped, ' '];
        end
        continue;
    else
        multiline_key = '';
    end

    if indent == 0
        current_parent = '';
    end

    if contains(stripped, ':')
        colon_idx = find(stripped == ':', 1, 'first');
        key = strtrim(stripped(1:colon_idx-1));
        rest = '';
        if colon_idx < length(stripped)
            rest = strtrim(stripped(colon_idx+1:end));
        end

        if ~isempty(rest)
            value = parse_yaml_value(rest);
            if indent == 0
                scenario.(key) = value;
            elseif indent > 0 && ~isempty(current_parent)
                if ~isfield(scenario, current_parent) || ~isstruct(scenario.(current_parent))
                    scenario.(current_parent) = struct();
                end
                scenario.(current_parent).(key) = value;
            end
        else
            if indent == 0
                scenario.(key) = struct();
                current_parent = key;
            end
        end
    end
end

fclose(fid);
end


function value = parse_yaml_value(str)
%PARSE_YAML_VALUE Parse a YAML value string into MATLAB type

value = str;

if startsWith(value, '"') && endsWith(value, '"')
    value = value(2:end-1);
    return;
end

if startsWith(value, '[') && endsWith(value, ']')
    inner = value(2:end-1);
    inner = strrep(inner, 'null', 'NaN');
    arr_parts = strsplit(inner, ',');
    arr = cell(1, length(arr_parts));
    for i = 1:length(arr_parts)
        v = strtrim(arr_parts{i});
        if startsWith(v, '"') && endsWith(v, '"')
            arr{i} = v(2:end-1);
        else
            num = str2double(v);
            if ~isnan(num) || strcmp(v, 'NaN')
                arr{i} = num;
            else
                arr{i} = v;
            end
        end
    end
    value = arr;
    return;
end

num = str2double(value);
if ~isnan(num)
    value = num;
end
end

function [n_param, n_species, not_found] = apply_param_struct(model, s)
%APPLY_PARAM_STRUCT Set each field of struct s on the model as a parameter
% value or, failing that, a species initial amount. Shared by the CSV-median,
% submodel-shadow, and joint-posterior overlay passes.
names = fieldnames(s);
n_param = 0;
n_species = 0;
not_found = {};
for i = 1:length(names)
    pname = names{i};
    pval = s.(pname);
    if ~(isnumeric(pval) && isscalar(pval)) || ~isfinite(pval)
        warning('run_median_simulation:badValue', ...
            'Skipping %s: value is not a finite numeric scalar', pname);
        not_found{end+1} = pname; %#ok<AGROW>
        continue;
    end
    param_obj = sbioselect(model, 'Name', pname, 'Type', 'parameter');
    if ~isempty(param_obj)
        set(param_obj, 'Value', pval);
        n_param = n_param + 1;
    else
        species_obj = sbioselect(model, 'Name', pname, 'Type', 'species');
        if ~isempty(species_obj)
            for k = 1:length(species_obj)
                species_obj(k).InitialAmount = pval;
            end
            n_species = n_species + 1;
        else
            not_found{end+1} = pname; %#ok<AGROW>
        end
    end
end
end

function py = python_exe(repo_root)
%PYTHON_EXE Path to the repo venv interpreter (per CLAUDE.md: always use .venv).
venv = fullfile(repo_root, '.venv', 'bin', 'python');
if isfile(venv)
    py = venv;
else
    py = 'python';
end
end

function p = resolve_path(p, root)
%RESOLVE_PATH Make a relative path absolute against root (no-op if empty/absolute).
if isempty(p) || isAbsolute(p)
    return;
end
p = fullfile(root, p);
end

function tf = isAbsolute(p)
if ispc
    tf = (numel(p) >= 2 && p(2) == ':') || startsWith(p, '\\') || startsWith(p, '/');
else
    tf = startsWith(p, '/');
end
end

function delete_if_exists(f)
if isfile(f)
    delete(f);
end
end
