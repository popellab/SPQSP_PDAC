% export_trajectories.m - Run SimBiology model and export species trajectories
%
% Usage:
%   matlab -batch "run('scripts/export_trajectories.m')"
%
% Output:
%   trajectories.csv in the project root (time + all species)

%% Build and simulate the model
immune_oncology_model_PDAC;

% Run simulation
simdata = sbiosimulate(model);

%% Export to CSV
% Get time and species data
t = simdata.Time;
data = simdata.Data;
names = simdata.DataNames;

% Build table
T = array2table([t, data], 'VariableNames', ['Time', names']);

% Write
output_file = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'trajectories.csv');
writetable(T, output_file);
fprintf('Trajectories exported to: %s\n', output_file);
fprintf('  %d time points, %d species\n', size(data, 1), size(data, 2));