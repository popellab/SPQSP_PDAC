function [scenario, dose_schedule] = scenario_utils(test_stats_csv, model)
%SCENARIO_UTILS Extract scenario from test statistics CSV filename
%
% Extracts scenario identifier from CSV filename pattern: PDAC_{scenario}_test_statistics
% Always returns baseline scenario (no treatment, no dose schedule).
%
% Usage:
%   [scenario, dose_schedule] = scenario_utils(test_stats_csv, model)
%
% Inputs:
%   test_stats_csv - Path to test statistics CSV file
%   model          - SimBiology model object (unused, for consistency)
%
% Outputs:
%   scenario       - Struct with fields: scenario_id, drugs, time
%   dose_schedule  - Empty (baseline scenario)
%
% Example:
%   [scenario, dose_schedule] = scenario_utils('PDAC_baseline_no_treatment_test_statistics.csv', model);
%   % Returns: scenario.scenario_id = 'baseline_no_treatment', dose_schedule = []

% Extract scenario from filename
[~, csv_name, ~] = fileparts(test_stats_csv);
scenario_match = regexp(csv_name, 'PDAC_(.+)_test_statistics', 'tokens');

if ~isempty(scenario_match)
    scenario_id = scenario_match{1}{1};
else
    error('Could not extract scenario from CSV filename: %s\nExpected pattern: PDAC_{scenario}_test_statistics', csv_name);
end

% Build scenario struct (baseline: no drugs, no dosing)
scenario = struct();
scenario.scenario_id = scenario_id;
scenario.drugs = {};
scenario.time = 0:1:365;  % Default: 1 year simulation with daily output

% Baseline scenario: no dose schedule
dose_schedule = [];

end
