function [endpoints, endpoint_metadata] = PSA_calculate_endpoints(results, filtered_results)
%PSA_CALCULATE_ENDPOINTS Calculate clinical endpoints from simulation results
%
% This function calculates clinically relevant endpoints such as RECIST
% response, progression-free survival, and tumor response classifications.
%
% Inputs:
%   results              - Array of simulation results from PSA_simulate
%   filtered_results     - Filtered patient indices from PSA_filter_patients
%
% Outputs:
%   endpoints            - Structure containing calculated endpoints
%   endpoint_metadata    - Metadata about endpoint calculations
%
% Example:
%   [endpoints, metadata] = PSA_calculate_endpoints(results, filtered_results);

%% Initialize
n_patients = length(results);
plausible_indices = filtered_results.plausible_successful;

endpoints = initialize_endpoints(n_patients);
endpoint_metadata = struct();

fprintf('Calculating clinical endpoints for %d plausible patients...\n', length(plausible_indices));

%% Process all patients in single loop
fprintf('  Processing tumor volume and classifications...\n');

for i = 1:length(plausible_indices)
    patient_idx = plausible_indices(i);
    
    % Extract tumor volume data
    tumor_data = extract_tumor_volume(results(patient_idx));
    if isempty(tumor_data)
        continue;
    end
    
    % Store tumor volume endpoints
    endpoints.tumor_volume.baseline(patient_idx) = tumor_data.baseline;
    endpoints.tumor_volume.minimum(patient_idx) = tumor_data.minimum;
    endpoints.tumor_volume.final(patient_idx) = tumor_data.final;
    endpoints.tumor_volume.percent_change(patient_idx) = tumor_data.percent_change;
    endpoints.tumor_volume.best_response(patient_idx) = tumor_data.best_response;
    
    % Classify RECIST response
    endpoints.recist.category{patient_idx} = classify_recist_response(tumor_data.best_response);
    
    % Calculate PFS
    endpoints.pfs.time_to_progression(patient_idx) = calculate_time_to_progression(...
        tumor_data, endpoints.recist.category{patient_idx});
    endpoints.pfs.progression_free_at_end(patient_idx) = ~strcmp(endpoints.recist.category{patient_idx}, 'PD');
    
    % Classify responder status
    recist_category = endpoints.recist.category{patient_idx};
    [endpoints.responder.is_responder(patient_idx), endpoints.responder.classification{patient_idx}] = ...
        classify_responder(recist_category);
end

%% Calculate summary statistics
[endpoints, endpoint_metadata] = calculate_summary_statistics(endpoints, plausible_indices);

%% Display results
display_endpoint_summary(endpoints, endpoint_metadata);

end

%% Helper Functions
function endpoints = initialize_endpoints(n_patients)
    % Initialize endpoint structure with NaN/empty values
    endpoints = struct();
    
    % Tumor volume endpoints
    endpoints.tumor_volume = struct(...
        'baseline', NaN(n_patients, 1), ...
        'minimum', NaN(n_patients, 1), ...
        'final', NaN(n_patients, 1), ...
        'percent_change', NaN(n_patients, 1), ...
        'best_response', NaN(n_patients, 1));
    
    % RECIST endpoints
    endpoints.recist = struct(...
        'category', {cell(n_patients, 1)}, ...
        'response_rate', NaN, ...
        'disease_control_rate', NaN);
    
    % PFS endpoints
    endpoints.pfs = struct(...
        'time_to_progression', NaN(n_patients, 1), ...
        'progression_free_at_end', false(n_patients, 1), ...
        'median_pfs', NaN);
    
    % Responder classification
    endpoints.responder = struct(...
        'is_responder', false(n_patients, 1), ...
        'classification', {cell(n_patients, 1)});
end

function tumor_data = extract_tumor_volume(result)
    % Extract and calculate tumor volume metrics from simulation result
    tumor_data = [];
    
    if isempty(result.simData)
        return;
    end
    
    try
        [time_points, tumor_volume, ~] = selectbyname(result.simData, 'V_T');
        
        baseline = tumor_volume(1);
        if baseline <= 0
            return;
        end
        
        tumor_data = struct();
        tumor_data.time_points = time_points;
        tumor_data.tumor_volume = tumor_volume;
        tumor_data.baseline = baseline;
        tumor_data.minimum = min(tumor_volume);
        tumor_data.final = tumor_volume(end);
        tumor_data.percent_change = 100 * (tumor_data.final - baseline) / baseline;
        tumor_data.best_response = 100 * (tumor_data.minimum - baseline) / baseline;
        
    catch ME
        warning('Failed to extract tumor volume: %s', ME.message);
    end
end

function category = classify_recist_response(best_response)
    % Classify RECIST response based on best percent change
    if isnan(best_response)
        category = 'Unknown';
    elseif best_response <= -100
        category = 'CR';  % Complete Response
    elseif best_response <= -30
        category = 'PR';  % Partial Response
    elseif best_response >= 20
        category = 'PD';  % Progressive Disease
    else
        category = 'SD';  % Stable Disease
    end
end

function time_to_progression = calculate_time_to_progression(tumor_data, recist_category)
    % Calculate time to progression for PFS analysis
    if isempty(tumor_data)
        time_to_progression = NaN;
        return;
    end
    
    simulation_time = tumor_data.time_points(end);
    
    if strcmp(recist_category, 'PD')
        % Find when tumor volume increased by 20%
        progression_threshold = tumor_data.baseline * 1.2;
        progression_idx = find(tumor_data.tumor_volume >= progression_threshold, 1);
        
        if ~isempty(progression_idx)
            time_to_progression = tumor_data.time_points(progression_idx);
        else
            time_to_progression = simulation_time;
        end
    else
        time_to_progression = simulation_time;
    end
end

function [is_responder, classification] = classify_responder(recist_category)
    % Classify patient as responder or non-responder
    if strcmp(recist_category, 'CR') || strcmp(recist_category, 'PR')
        is_responder = true;
        classification = 'Responder';
    elseif strcmp(recist_category, 'SD') || strcmp(recist_category, 'PD')
        is_responder = false;
        classification = 'Non-responder';
    else
        is_responder = false;
        classification = 'Unknown';
    end
end

function [endpoints, endpoint_metadata] = calculate_summary_statistics(endpoints, plausible_indices)
    % Calculate response rates and summary statistics
    recist_categories = endpoints.recist.category(plausible_indices);
    n_evaluable = sum(~strcmp(recist_categories, 'Unknown'));
    
    if n_evaluable > 0
        n_cr = sum(strcmp(recist_categories, 'CR'));
        n_pr = sum(strcmp(recist_categories, 'PR'));
        n_sd = sum(strcmp(recist_categories, 'SD'));
        
        endpoints.recist.response_rate = (n_cr + n_pr) / n_evaluable;
        endpoints.recist.disease_control_rate = (n_cr + n_pr + n_sd) / n_evaluable;
    end
    
    % Calculate median PFS
    valid_pfs = endpoints.pfs.time_to_progression(plausible_indices);
    valid_pfs = valid_pfs(~isnan(valid_pfs));
    
    if ~isempty(valid_pfs)
        endpoints.pfs.median_pfs = median(valid_pfs);
    end
    
    % Create metadata summary
    endpoint_metadata.summary = struct(...
        'n_evaluable', length(plausible_indices), ...
        'response_rate', endpoints.recist.response_rate, ...
        'disease_control_rate', endpoints.recist.disease_control_rate, ...
        'median_pfs', endpoints.pfs.median_pfs);
    
    % RECIST distribution
    endpoint_metadata.summary.recist_distribution = struct();
    for category = {'CR', 'PR', 'SD', 'PD'}
        count = sum(strcmp(recist_categories, category{1}));
        endpoint_metadata.summary.recist_distribution.(category{1}) = count;
    end
end

function display_endpoint_summary(endpoints, endpoint_metadata)
    % Display endpoint results to console
    recist_dist = endpoint_metadata.summary.recist_distribution;
    n_eval = endpoint_metadata.summary.n_evaluable;
    
    if n_eval > 0
        fprintf('    CR: %d (%.1f%%), PR: %d (%.1f%%), SD: %d (%.1f%%), PD: %d (%.1f%%)\n', ...
            recist_dist.CR, 100*recist_dist.CR/n_eval, ...
            recist_dist.PR, 100*recist_dist.PR/n_eval, ...
            recist_dist.SD, 100*recist_dist.SD/n_eval, ...
            recist_dist.PD, 100*recist_dist.PD/n_eval);
        fprintf('    Response Rate: %.1f%%, Disease Control Rate: %.1f%%\n', ...
            100*endpoints.recist.response_rate, 100*endpoints.recist.disease_control_rate);
        fprintf('    Median PFS: %.1f days\n', endpoints.pfs.median_pfs);
    end
    
    fprintf('\nEndpoint Summary:\n');
    fprintf('  Evaluable patients: %d\n', n_eval);
    fprintf('  Response rate: %.1f%%\n', 100*endpoint_metadata.summary.response_rate);
    fprintf('  Disease control rate: %.1f%%\n', 100*endpoint_metadata.summary.disease_control_rate);
    fprintf('  Median PFS: %.1f days\n', endpoint_metadata.summary.median_pfs);
end