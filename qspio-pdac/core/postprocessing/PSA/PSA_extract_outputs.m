function [output_data, output_names] = PSA_extract_outputs(results, filtered_results, config, postprocessed_data)
%PSA_EXTRACT_OUTPUTS Extract specified outputs from filtered simulation results
%
% Inputs:
%   results              - Array of simulation results
%   filtered_results     - Filtered patient indices
%   config              - PSA configuration
%   postprocessed_data  - Postprocessed data (optional)
%
% Outputs:
%   output_data         - Matrix of extracted outputs (patients x outputs)
%   output_names        - Cell array of output names

n_patients = length(results);
output_names = fieldnames(config.outputs);
n_outputs = length(output_names);

output_data = NaN(n_patients, n_outputs);
plausible_indices = filtered_results.plausible_successful;

fprintf('Extracting %d outputs from %d plausible patients...\n', n_outputs, length(plausible_indices));

for i = 1:n_outputs
    output_name = output_names{i};
    output_config = config.outputs.(output_name);

    for j = 1:length(plausible_indices)
        patient_idx = plausible_indices(j);

        try
            value = extract_single_output(results(patient_idx).simData, output_config, ...
                                        postprocessed_data, patient_idx);
            output_data(patient_idx, i) = value;
        catch
            % Leave as NaN
        end
    end
end

fprintf('Output extraction complete\n');

end

function value = extract_single_output(sim_data, output_config, postprocessed_data, patient_idx)
%EXTRACT_SINGLE_OUTPUT Extract a single output value from simulation data

switch lower(output_config.type)
    case 'species'
        full_name = sprintf('%s.%s', output_config.compartment, output_config.name);
        [~, data, ~] = selectbyname(sim_data, full_name);
        value = get_value_at_timepoint(data, output_config.timepoint);

    case 'parameter'
        [~, data, ~] = selectbyname(sim_data, output_config.name);
        value = data(end);

    case 'compartment'
        [~, data, ~] = selectbyname(sim_data, output_config.name);
        value = get_value_at_timepoint(data, output_config.timepoint);

    case 'calculated'
        data_names = postprocessed_data(patient_idx).simData.DataNames;
        data_idx = find(strcmp(data_names, output_config.name));
        data = postprocessed_data(patient_idx).simData.Data(:, data_idx);
        value = get_value_at_timepoint(data, output_config.timepoint);

    case {'baseline', 'final'}
        data_names = postprocessed_data(patient_idx).simData.DataNames;
        data_idx = find(strcmp(data_names, output_config.name));
        data = postprocessed_data(patient_idx).simData.Data(:, data_idx);

        if strcmp(output_config.type, 'baseline')
            value = data(1);
        else
            value = data(end);
        end

    otherwise
        error('Unknown output type: %s', output_config.type);
end

end

function value = get_value_at_timepoint(data, timepoint)
%GET_VALUE_AT_TIMEPOINT Extract value at specific timepoint

if ischar(timepoint)
    switch lower(timepoint)
        case 'baseline'
            value = data(1);
        case 'final'
            value = data(end);
        otherwise
            error('Unknown timepoint: %s', timepoint);
    end
elseif isnumeric(timepoint)
    value = data(timepoint);
else
    error('Invalid timepoint');
end

end
