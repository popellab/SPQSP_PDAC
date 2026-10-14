function fig = PSA_plot_tornado(param_values, output_values, param_names, output_name)
%PSA_PLOT_TORNADO Create tornado plot showing parameter impact ranges
%
% Creates a tornado plot showing the range of output values when each
% parameter varies across its range, sorted by impact magnitude.
%
% Inputs:
%   param_values - Matrix of parameter values (patients x parameters)
%   output_values - Vector of output values (patients x 1)
%   param_names  - Cell array of parameter names
%   output_name  - Name of the output being analyzed
%
% Outputs:
%   fig - Figure handle

n_params = length(param_names);
param_impacts = zeros(n_params, 3); % [min_output, max_output, range]

% Calculate parameter impacts using proper tornado methodology
for i = 1:n_params
    param_vals = param_values(:, i);

    % Remove NaN values
    valid_idx = ~isnan(param_vals) & ~isnan(output_values);
    if sum(valid_idx) < 10
        param_impacts(i, :) = [NaN, NaN, 0];
        continue;
    end

    param_vals = param_vals(valid_idx);
    output_vals = output_values(valid_idx);

    % Use actual parameter extremes (10th and 90th percentiles to avoid outliers)
    param_low = prctile(param_vals, 10);
    param_high = prctile(param_vals, 90);

    % Find outputs corresponding to parameter extremes
    % Use more extreme percentiles for clearer separation
    low_idx = param_vals <= prctile(param_vals, 10);
    high_idx = param_vals >= prctile(param_vals, 90);

    if sum(low_idx) > 0 && sum(high_idx) > 0
        low_output = mean(output_vals(low_idx));
        high_output = mean(output_vals(high_idx));

        % Store the actual min/max (regardless of which extreme is higher)
        param_impacts(i, 1) = min(low_output, high_output);
        param_impacts(i, 2) = max(low_output, high_output);
        param_impacts(i, 3) = abs(high_output - low_output);
    else
        % Fallback: use more extreme percentiles
        low_idx = param_vals <= prctile(param_vals, 25);
        high_idx = param_vals >= prctile(param_vals, 75);

        if sum(low_idx) > 0 && sum(high_idx) > 0
            low_output = mean(output_vals(low_idx));
            high_output = mean(output_vals(high_idx));
            param_impacts(i, 1) = min(low_output, high_output);
            param_impacts(i, 2) = max(low_output, high_output);
            param_impacts(i, 3) = abs(high_output - low_output);
        else
            % Final fallback: use overall range
            param_impacts(i, 1) = min(output_vals);
            param_impacts(i, 2) = max(output_vals);
            param_impacts(i, 3) = param_impacts(i, 2) - param_impacts(i, 1);
        end
    end
end

% Sort by impact magnitude
[~, sort_idx] = sort(param_impacts(:, 3), 'descend', 'MissingPlacement', 'last');

% Show top 10 parameters
n_show = min(10, sum(~isnan(param_impacts(:, 3))));
show_idx = sort_idx(1:n_show);

show_impacts = param_impacts(show_idx, :);
show_params = param_names(show_idx);

% Calculate baseline
baseline = median(output_values, 'omitnan');


% Create figure
fig = figure('Position', [300, 100, 800, max(400, 35*n_show)]);

% Create tornado bars - much clearer approach
hold on;

% Use consistent color scheme
bar_color = [0.4, 0.6, 0.8]; % Single blue color for all bars

for i = 1:n_show
    min_val = show_impacts(i, 1);
    max_val = show_impacts(i, 2);

    if ~isnan(min_val) && ~isnan(max_val)
        % Create proper tornado bars using rectangle
        y_pos = i - 0.4;  % Bar starts slightly below center
        bar_height = 0.8; % Bar thickness
        
        % Draw rectangle from min_val to max_val
        rectangle('Position', [min_val, y_pos, max_val - min_val, bar_height], ...
            'FaceColor', bar_color, 'EdgeColor', 'black', 'LineWidth', 1);

        % Add min/max labels at bar ends (scientific notation)
        text(min_val - (max(show_impacts(:, 2)) - min(show_impacts(:, 1))) * 0.02, i, ...
            sprintf('%.2e', min_val), 'FontSize', 9, 'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'middle', 'Color', 'blue', 'FontWeight', 'bold');

        text(max_val + (max(show_impacts(:, 2)) - min(show_impacts(:, 1))) * 0.02, i, ...
            sprintf('%.2e', max_val), 'FontSize', 9, 'HorizontalAlignment', 'left', ...
            'VerticalAlignment', 'middle', 'Color', 'blue', 'FontWeight', 'bold');

        % Add range (impact) in center of bar (scientific notation)
        bar_center = (min_val + max_val) / 2;
        text(bar_center, i, sprintf('Δ=%.2e', show_impacts(i, 3)), ...
            'FontSize', 9, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
            'Color', 'white', 'FontWeight', 'bold');
    end
end

% Add baseline line with clear label
baseline_line = line([baseline, baseline], [0.5, n_show + 0.5], 'Color', 'red', ...
    'LineStyle', '--', 'LineWidth', 2);

% Add baseline label (scientific notation)
text(baseline, n_show + 0.7, sprintf('Baseline = %.2e', baseline), ...
    'FontSize', 10, 'HorizontalAlignment', 'center', 'FontWeight', 'bold', ...
    'Color', 'red');

% Format axes
ax = gca;
ax.FontSize = 10;
set(ax, 'YTick', 1:n_show);
set(ax, 'YTickLabels', format_param_labels(show_params));
ax.YDir = 'reverse';
ylim([0.5, n_show + 0.5]);

% Set proper x-axis limits based on actual data range
data_min = min(show_impacts(:, 1));
data_max = max(show_impacts(:, 2));
data_range = data_max - data_min;
x_padding = data_range * 0.15; % 15% padding on each side

xlim([data_min - x_padding, data_max + x_padding]);

% Labels and title
xlabel(format_output_name(output_name), 'FontSize', 12);
title(sprintf('Parameter Impact on %s', format_output_name(output_name)), ...
    'FontSize', 14, 'FontWeight', 'bold');
grid on;
ax.GridAlpha = 0.3;

% Add explanatory text box
explanation_text = {
    'Each bar shows the range of output values'
    'when the parameter varies across its range.'
    'Δ = impact magnitude (max - min)'
    'Parameters ranked by impact size'
};

annotation('textbox', [0.02, 0.02, 0.4, 0.15], 'String', explanation_text, ...
    'FontSize', 9, 'BackgroundColor', [0.95, 0.95, 0.95], 'EdgeColor', 'black', ...
    'FitBoxToText', 'on');

end

function labels = format_param_labels(param_names)
%FORMAT_PARAM_LABELS Format parameter names for display
labels = cell(size(param_names));
for i = 1:length(param_names)
    name = param_names{i};
    name = strrep(name, '_', ' ');
    if ~isempty(name)
        name(1) = upper(name(1));
    end
    if length(name) > 18
        name = [name(1:15) '...'];
    end
    labels{i} = name;
end
end

function formatted = format_output_name(output_name)
%FORMAT_OUTPUT_NAME Format output name for display
formatted = strrep(output_name, '_', ' ');
words = strsplit(formatted, ' ');
for i = 1:length(words)
    if ~isempty(words{i})
        words{i}(1) = upper(words{i}(1));
    end
end
formatted = strjoin(words, ' ');
end
