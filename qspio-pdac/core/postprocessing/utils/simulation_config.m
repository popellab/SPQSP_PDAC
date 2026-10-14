function model = simulation_config(model, varargin)
%SIMULATION_CONFIG Configure simulation settings for SimBiology models
%
% This function allows you to set simulation configuration parameters
% for any SimBiology model, enabling flexible simulation settings
% across different use cases.
%
% Inputs:
%   model - SimBiology model object
%
% Optional Name-Value Pairs:
%   'solver'         - Solver type (default: 'sundials')
%   'time'           - Time vector in days (default: 0:1:365)
%   'abs_tolerance'  - Absolute tolerance (default: 1e-12)
%   'rel_tolerance'  - Relative tolerance (default: 1e-9)
%   'max_step'       - Maximum step size (default: [])
%   'time_units'     - Time units (default: 'day')
%   'unit_conversion' - Enable unit conversion (default: true)
%   'stop_time'      - Stop time (default: time(end))
%
% Outputs:
%   model - Updated SimBiology model object with new configuration
%
% Example:
%   model = simulation_config(model, 'solver', 'ode23s', ...
%                            'time', 0:0.5:180, ...
%                            'abs_tolerance', 1e-10);

% Parse input arguments
p = inputParser;
addRequired(p, 'model');
addParameter(p, 'solver', 'sundials', @ischar);
addParameter(p, 'time', 0:1:365, @isnumeric);
addParameter(p, 'abs_tolerance', 1e-12, @isnumeric);
addParameter(p, 'rel_tolerance', 1e-9, @isnumeric);
addParameter(p, 'max_step', [], @isnumeric);
addParameter(p, 'time_units', 'day', @ischar);
addParameter(p, 'unit_conversion', true, @islogical);
addParameter(p, 'stop_time', [], @isnumeric);
parse(p, model, varargin{:});

% Extract parsed values
solver = p.Results.solver;
time = p.Results.time;
abs_tol = p.Results.abs_tolerance;
rel_tol = p.Results.rel_tolerance;
max_step = p.Results.max_step;
time_units = p.Results.time_units;
unit_conversion = p.Results.unit_conversion;
stop_time = p.Results.stop_time;

% Default stop_time to time(end) if not provided
if isempty(stop_time)
    stop_time = time(end);
end

% Get configuration set
config = getconfigset(model);

% Set unit conversion first (as in simbio_init)
options = get(config, 'CompileOptions');
set(options, 'UnitConversion', unit_conversion);

% Set time configuration
set(config, 'TimeUnits', time_units);
set(config.SolverOptions, 'OutputTimes', time);
set(config, 'StopTime', stop_time);

% Set solver type
set(config, 'SolverType', solver);

% Set tolerances
set(config.SolverOptions, 'AbsoluteTolerance', abs_tol);
set(config.SolverOptions, 'RelativeTolerance', rel_tol);

% Set maximum step size if provided
if ~isempty(max_step)
    set(config.SolverOptions, 'MaxStep', max_step);
    fprintf('  Maximum step size: %.3f\n', max_step);
end

end
