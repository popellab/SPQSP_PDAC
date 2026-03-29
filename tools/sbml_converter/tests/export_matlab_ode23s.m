% export_matlab_ode23s.m - Run SimBiology model and export trajectories with qualified names.
%
% Uses ode23t solver (Rosenbrock method) — independent from ode15s/CVODE BDF.
% Three-solver comparison showed ode15s, ode23t, and SUNDIALS agree perfectly,
% so any solver gives correct reference trajectories.
%
% Usage:
%   matlab -batch "output_csv='/path/to/out.csv'; pdac_build_dir='/path/to/pdac-build'; run('/path/to/export_matlab_ode23s.m')"

if ~exist('output_csv', 'var')
    error('Set output_csv before running this script');
end
if ~exist('pdac_build_dir', 'var')
    pdac_build_dir = pwd;
end

cd(pdac_build_dir);
run('startup.m');

% Build model
immune_oncology_model_PDAC;

% Switch solver to ode23t (Rosenbrock method)
cs = getconfigset(model, 'active');
cs.SolverType = 'ode23t';
fprintf('Solver: %s, RelTol=%g, AbsTol=%g\n', cs.SolverType, ...
    cs.SolverOptions.RelativeTolerance, cs.SolverOptions.AbsoluteTolerance);

simdata = sbiosimulate(model);

% Map bare DataNames to qualified names (Compartment_Species).
% WARNING: sbiosimulate's DataNames order differs from export(model).ValueInfo.
% Always use qualify_species_names() for sbiosimulate output.
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
col_names = qualify_species_names(model, simdata.DataNames);

T = array2table([simdata.Time, simdata.Data], 'VariableNames', [{'Time'}, col_names]);
writetable(T, output_csv);
fprintf('Wrote %d time points, %d columns to %s\n', size(simdata.Data,1), length(col_names), output_csv);
