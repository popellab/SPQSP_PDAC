function PSA_PRCC(params_in, output_data, output_names, filtered_results, config)
%PSA_PRCC Perform Partial Rank Correlation Coefficient analysis
%
% Uses the existing calculate_prcc function for partial correlation analysis.
%
% Inputs:
%   params_in        - Parameter structure from generate_parameter_samples()
%   output_data      - Matrix of outputs (patients x outputs)
%   output_names     - Cell array of output names
%   filtered_results - Filtered patient indices
%   config          - PSA configuration

plausible_indices = filtered_results.plausible_successful;
n_params = length(params_in.names);
n_outputs = length(output_names);

if isempty(plausible_indices)
    fprintf('No plausible patients for sensitivity analysis\n');
    return;
end

fprintf('Running PRCC analysis on %d patients, %d parameters, %d outputs...\n', ...
    length(plausible_indices), n_params, n_outputs);

% Extract parameter matrix
param_matrix = zeros(length(plausible_indices), n_params);
for i = 1:n_params
    param_name = params_in.names{i};
    if isfield(params_in, param_name) && isfield(params_in.(param_name), 'LHS')
        param_matrix(:, i) = params_in.(param_name).LHS(plausible_indices);
    end
end

% Extract output matrix
output_matrix = output_data(plausible_indices, :);

% Remove NaN values
valid_rows = all(~isnan(param_matrix), 2) & all(~isnan(output_matrix), 2);
param_matrix = param_matrix(valid_rows, :);
output_matrix = output_matrix(valid_rows, :);

if size(param_matrix, 1) < 10
    fprintf('Warning: Only %d valid patients for correlation analysis\n', size(param_matrix, 1));
end

% Calculate PRCC using existing function
prcc_results = calculate_prcc(param_matrix, output_matrix, config.significance_level);

% Display significant correlations
fprintf('\nSignificant correlations (p < %.3f, partial correlation):\n', config.significance_level);

n_significant = 0;
for i = 1:n_params
    for j = 1:n_outputs
        if prcc_results.significant(i, j)
            param_name = params_in.names{i};
            output_name = output_names{j};

            fprintf('  %s -> %s: PRCC = %.3f, p = %.3f\n', ...
                param_name, output_name, prcc_results.prcc_values(i, j), prcc_results.p_values(i, j));
            n_significant = n_significant + 1;
        end
    end
end

if n_significant == 0
    fprintf('  No significant correlations found\n');
end

% Create correlation heatmap (always show, regardless of significance)
create_correlation_heatmap(prcc_results.prcc_values, prcc_results.p_values, ...
    params_in.names, output_names, config.significance_level);

end

function create_correlation_heatmap(rho, pval, param_names, output_names, significance_threshold)
%CREATE_CORRELATION_HEATMAP Create a correlation heatmap with significance stars


% Display ALL correlations (not just significant ones)
rho_display = rho;

% Create figure
figure;
imagesc(rho_display);
colorbar;

% Create a simple red-white-blue colormap
n_colors = 64;
red_to_white = [linspace(1, 1, n_colors/2)', linspace(0, 1, n_colors/2)', linspace(0, 1, n_colors/2)'];
white_to_blue = [linspace(1, 0, n_colors/2)', linspace(1, 0, n_colors/2)', linspace(1, 1, n_colors/2)'];
cmap = [red_to_white; white_to_blue];
colormap(cmap);
caxis([-1 1]);

% Set title and axis labels first
title('Significant Parameter-Output Correlations (PRCC)', 'FontSize', 32);
xlabel('Outputs', 'FontSize', 11);
ylabel('Parameters', 'FontSize', 11);

% Adjust figure size for better visibility
fig_width = max(600, 80 * length(output_names));
fig_height = max(400, 30 * length(param_names));
set(gcf, 'Position', [100, 100, fig_width, fig_height]);

% Set up axes AFTER setting figure size
ax = gca;

% Make sure the plot is properly formatted first
axis tight;

% NOW set ticks and labels (after axis tight)

% Ensure cell arrays of strings
if ~iscell(output_names)
    output_names = cellstr(output_names);
end
if ~iscell(param_names)
    param_names = cellstr(param_names);
end

% Set ticks first
set(ax, 'XTick', 1:length(output_names));
set(ax, 'YTick', 1:length(param_names));

% Force tick labels to show and set font size before labels
set(ax, 'FontSize', 11);
set(ax, 'TickLabelInterpreter', 'none');

% Set labels
set(ax, 'XTickLabels', output_names);
set(ax, 'YTickLabels', param_names);

% Rotate labels for better readability
set(ax, 'XTickLabelRotation', 45);
set(ax, 'YTickLabelRotation', 0);

% Remove grid
grid off;

% Add significance stars to tiles
for i = 1:length(param_names)
    for j = 1:length(output_names)
        p_val = pval(i, j);

        % Determine significance level and star notation
        if p_val < 0.001
            star_text = '***';
        elseif p_val < 0.01
            star_text = '**';
        elseif p_val < 0.05
            star_text = '*';
        else
            star_text = '';
        end

        % Add text to tile if significant
        if ~isempty(star_text)
            text(j, i, star_text, 'HorizontalAlignment', 'center', ...
                'VerticalAlignment', 'middle', 'FontSize', 12, ...
                'FontWeight', 'bold', 'Color', 'black');
        end
    end
end

end
