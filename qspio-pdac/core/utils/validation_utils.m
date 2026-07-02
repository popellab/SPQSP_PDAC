function varargout = validation_utils(action, varargin)
%VALIDATION_UTILS Utility functions for test statistic validation workflows
%
% This function provides helper utilities for PSA test statistic validation.
%
% Usage:
%   cache_key = validation_utils('cache_key', param_csv, test_stats_csv, n_patients, seed)
%   hash = validation_utils('extract_hash', filename)
%   validation_utils('display_summary', params_matrix, test_stats_matrix, metadata)

switch action
    case 'cache_key'
        varargout{1} = generate_validation_cache_key(varargin{:});
    case 'extract_hash'
        varargout{1} = extract_hash_from_filename(varargin{:});
    case 'display_summary'
        display_validation_summary(varargin{:});
    otherwise
        error('Unknown action: %s', action);
end

end

function cache_key = generate_validation_cache_key(param_csv, test_stats_csv, n_patients, seed)
%GENERATE_VALIDATION_CACHE_KEY Generate cache key for validation results

% Extract hashes from CSV filenames
param_hash = extract_hash_from_filename(param_csv);
test_stats_hash = extract_hash_from_filename(test_stats_csv);

% Combine into cache key
key_string = sprintf('p%s_t%s_n%d_s%d', param_hash, test_stats_hash, n_patients, seed);
hash_val = abs(java.lang.String(key_string).hashCode());
cache_key = sprintf('%08x', hash_val);

end

function hash = extract_hash_from_filename(filename)
%EXTRACT_HASH_FROM_FILENAME Extract hash from filename like 'file_abc123.csv'

[~, basename, ~] = fileparts(filename);
hash_match = regexp(basename, '_([a-f0-9]+)$', 'tokens');
if ~isempty(hash_match)
    hash = hash_match{1}{1};
else
    hash = 'unknown';
end

end

function display_validation_summary(params_matrix, test_stats_matrix, metadata)
%DISPLAY_VALIDATION_SUMMARY Display summary of validation results

fprintf('\n=== Results Summary ===\n');
fprintf('Parameters: %d patients × %d parameters\n', size(params_matrix, 1), size(params_matrix, 2));
fprintf('Test statistics: %d patients × %d statistics\n', size(test_stats_matrix, 1), size(test_stats_matrix, 2));
fprintf('\n');

% Parameter ranges
fprintf('Parameter ranges:\n');
for i = 1:length(metadata.param_names)
    param_name = metadata.param_names{i};
    values = params_matrix(:, i);
    fprintf('  %s: [%.3g, %.3g] (median: %.3g)\n', ...
        param_name, min(values), max(values), median(values));
end
fprintf('\n');

% Test statistic comparison
fprintf('Test statistic validation:\n');
for j = 1:length(metadata.test_stat_ids)
    test_stat_id = metadata.test_stat_ids{j};
    values = test_stats_matrix(:, j);

    % Remove NaN values
    valid_values = values(~isnan(values));
    n_valid = length(valid_values);
    n_total = length(values);

    if n_valid == 0
        fprintf('  %s: No valid values\n', test_stat_id);
        continue;
    end

    % Compute statistics
    obs_mean = mean(valid_values);
    obs_std = std(valid_values);

    % Expected values
    exp_mean = metadata.test_stat_expected_mean(j);
    exp_std = sqrt(metadata.test_stat_expected_variance(j));

    % Z-score
    if exp_std > 0
        z_score = (obs_mean - exp_mean) / exp_std;
    else
        z_score = NaN;
    end

    fprintf('  %s (%s):\n', test_stat_id, metadata.test_stat_units{j});
    fprintf('    Observed:  %.3g ± %.3g (n=%d/%d)\n', obs_mean, obs_std, n_valid, n_total);
    fprintf('    Expected:  %.3g ± %.3g\n', exp_mean, exp_std);
    if ~isnan(z_score)
        fprintf('    Z-score:   %.2f %s\n', z_score, format_z_assessment(z_score));
    end
end

end

function assessment = format_z_assessment(z_score)
%FORMAT_Z_ASSESSMENT Format z-score assessment

if abs(z_score) < 1.96
    assessment = '(within 95% CI ✓)';
elseif abs(z_score) < 2.58
    assessment = '(outside 95% CI, within 99% CI)';
else
    assessment = '(outside 99% CI ⚠️)';
end

end
