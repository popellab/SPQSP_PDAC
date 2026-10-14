function postprocessed_data = PSA_postprocess(results, sim_metadata)
%PSA_POSTPROCESS Postprocess simulation results for PSA analysis
%
% This function postprocesses batch simulation data and extracts additional
% outputs of interest like clonality and percentage changes, adding them
% to individual simulations as parameters.
%
% Inputs:
%   results       - Array of simulation results from PSA_simulate
%   sim_metadata  - Simulation metadata from PSA_simulate
%
% Outputs:
%   postprocessed_data - Postprocessed data structure

fprintf('Postprocessing simulation results...\n');

% Create params_out structure for compatibility
params_out = struct();
params_out.iPatient = find(sim_metadata.status == 1);
n_PSA = length(params_out.iPatient);
index = params_out.iPatient;

% Initialize structure for postprocess and add first postprocessed
% data as simple tumor volume
postprocessed_data = struct;
for i = 1:n_PSA
    [~,temp,~] = selectbyname(results(index(i)).simData, 'V_T');
    postprocessed_data(index(i)).simData.DataNames = {'Voltum'};
    postprocessed_data(index(i)).simData.Data      = temp;
end

% Calculate T cell clonality and total CD8 T cell in blood
postprocessed_data = clonality(results, postprocessed_data, params_out);

% Add percentage change in tumor size
postprocessed_data = tumSizePerc(results, postprocessed_data, params_out);

% Add T cell subset ratio
postprocessed_data = calculate_ratio(results, postprocessed_data, params_out);

fprintf('PSA data post-processed.\n');

end
