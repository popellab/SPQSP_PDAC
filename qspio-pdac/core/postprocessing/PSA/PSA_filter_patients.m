function [filtered_results, filter_metadata] = PSA_filter_patients(results, metadata, config, postprocessed_data)
%PSA_FILTER_PATIENTS Filter patients based on plausibility criteria
%
% Inputs:
%   results              - Array of simulation results
%   metadata            - Simulation metadata
%   config              - PSA configuration
%   postprocessed_data  - Postprocessed data (optional)
%
% Outputs:
%   filtered_results    - Structure with patient indices
%   filter_metadata     - Filter performance info

% Calculate total number of patients from chunked results structure
% Results is organized as chunks x patients_per_chunk
[n_chunks, patients_per_chunk] = size(results);
n_patients = n_chunks * patients_per_chunk;

successful_indices = find(metadata.status == 1);
filter_names = fieldnames(config.filters);

fprintf('Filtering %d successful patients with %d filters...\n', ...
    length(successful_indices), length(filter_names));


% Initialize - track validity for all patients
valid_patients = true(n_patients, 1);
filter_metadata = struct();


% Apply each filter sequentially
for i = 1:length(filter_names)
    filter_name = filter_names{i};
    filter_config = config.filters.(filter_name);

    % Extract values for this filter (only for successful patients)
    filter_values = NaN(n_patients, 1);

    for j = 1:length(successful_indices)
        patient_idx = successful_indices(j);
        
        % Convert linear patient index to chunk/patient coordinates
        chunk_idx = ceil(patient_idx / patients_per_chunk);
        patient_in_chunk_idx = mod(patient_idx - 1, patients_per_chunk) + 1;

        try
            % Access results using 2D indexing
            patient_simData = results(chunk_idx, patient_in_chunk_idx).simData;
            if ~isempty(patient_simData)
                value = extract_filter_value(patient_simData, filter_config, ...
                                           postprocessed_data, patient_idx);
                filter_values(patient_idx) = value;
            end
        catch
            % Leave as NaN if extraction fails (will be marked as failing)
        end
    end

    % Apply bounds filter - identify patients that fail this specific filter
    lower_bound = filter_config.lower_bound;
    upper_bound = filter_config.upper_bound;
    
    % A patient fails this filter if their value is outside bounds or NaN
    current_filter_failures = (filter_values < lower_bound) | ...
                             (filter_values > upper_bound) | ...
                             isnan(filter_values);
    
    % Ensure arrays have consistent dimensions
    if length(valid_patients) ~= length(current_filter_failures)
        error('Dimension mismatch in filter %s: valid_patients(%d) vs failures(%d)', ...
            filter_name, length(valid_patients), length(current_filter_failures));
    end
    
    % Update overall validity: a patient is valid only if they pass ALL filters so far
    valid_patients = valid_patients & ~current_filter_failures;

    % Report results for this filter
    n_failing_this_filter = sum(current_filter_failures);
    fprintf('  %s: %d patients failed this filter\n', filter_name, n_failing_this_filter);
end

% Generate results
filtered_results = struct();
filtered_results.successful_indices = successful_indices;
filtered_results.plausible_indices = find(valid_patients);
filtered_results.plausible_successful = intersect(find(valid_patients), successful_indices);

n_plausible = length(filtered_results.plausible_successful);
fprintf('Result: %d plausible patients\n', n_plausible);

filter_metadata.n_plausible = n_plausible;

end

function value = extract_filter_value(sim_data, filter_config, postprocessed_data, patient_idx)
%EXTRACT_FILTER_VALUE Extract a single filter value from simulation data

switch lower(filter_config.type)
    case 'species'
        full_name = sprintf('%s.%s', filter_config.compartment, filter_config.name);
        [~, data, ~] = selectbyname(sim_data, full_name);
        value = get_value_at_timepoint(data, filter_config.timepoint);

    case 'parameter'
        [~, data, ~] = selectbyname(sim_data, filter_config.name);
        value = data(end);

    case 'compartment'
        [~, data, ~] = selectbyname(sim_data, filter_config.name);
        value = get_value_at_timepoint(data, filter_config.timepoint);

    case 'calculated'
        data_names = postprocessed_data(patient_idx).simData.DataNames;
        data_idx = find(strcmp(data_names, filter_config.name));
        data = postprocessed_data(patient_idx).simData.Data(:, data_idx);
        value = get_value_at_timepoint(data, filter_config.timepoint);

    case {'baseline', 'final'}
        data_names = postprocessed_data(patient_idx).simData.DataNames;
        data_idx = find(strcmp(data_names, filter_config.name));
        data = postprocessed_data(patient_idx).simData.Data(:, data_idx);

        if strcmp(filter_config.type, 'baseline')
            value = data(1);
        else
            value = data(end);
        end

    otherwise
        error('Unknown filter type: %s', filter_config.type);
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
