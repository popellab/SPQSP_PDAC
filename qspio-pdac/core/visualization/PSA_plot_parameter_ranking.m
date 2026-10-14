function fig = PSA_plot_parameter_ranking(prcc_values, p_values, param_names, output_name, output_idx)
%PSA_PLOT_PARAMETER_RANKING Create parameter ranking bar chart
%
% Creates a horizontal bar chart showing parameters ranked by their
% absolute PRCC values for a specific output.
%
% Inputs:
%   prcc_values  - Matrix of PRCC values (parameters x outputs)
%   p_values     - Matrix of p-values (parameters x outputs)
%   param_names  - Cell array of parameter names
%   output_name  - Name of the output being analyzed
%   output_idx   - Index of the output column to analyze
%
% Outputs:
%   fig - Figure handle

% Extract data for this output
correlations = prcc_values(:, output_idx);
p_vals = p_values(:, output_idx);

% Remove NaN values and sort by absolute correlation
valid_idx = ~isnan(correlations);
correlations = correlations(valid_idx);
p_vals = p_vals(valid_idx);
param_names = param_names(valid_idx);

[~, sort_idx] = sort(abs(correlations), 'descend');

% Show top 10 parameters
n_show = min(10, length(sort_idx));
show_idx = sort_idx(1:n_show);

show_corr = correlations(show_idx);
show_pvals = p_vals(show_idx);
show_params = param_names(show_idx);

% Create figure
fig = figure('Position', [200, 100, 800, max(400, 40*n_show)]);

% Create horizontal bar plot
barh(1:n_show, show_corr);
hold on;

% Color bars by direction
for i = 1:n_show
    if show_corr(i) > 0
        barh(i, show_corr(i), 'FaceColor', [0.2, 0.6, 0.8]);
    else
        barh(i, show_corr(i), 'FaceColor', [0.8, 0.3, 0.2]);
    end
end

% Format plot
ax = gca;
ax.FontSize = 10;
set(ax, 'YTick', 1:n_show);
set(ax, 'YTickLabels', format_param_labels(show_params));
ax.YDir = 'reverse';

% Add zero line
line([0, 0], ylim, 'Color', 'k', 'LineStyle', '--', 'LineWidth', 1);

% Add significance indicators
for i = 1:n_show
    p_val = show_pvals(i);
    corr_val = show_corr(i);

    if p_val < 0.01
        sig_text = '**';
    elseif p_val < 0.05
        sig_text = '*';
    else
        sig_text = '';
    end

    if ~isempty(sig_text)
        x_pos = corr_val + sign(corr_val) * max(abs(show_corr)) * 0.1;
        text(x_pos, i, sig_text, 'FontSize', 12, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle');
    end
end

% Labels and title
xlabel('PRCC', 'FontSize', 12);
title(sprintf('Parameter Ranking: %s', format_output_name(output_name)), ...
    'FontSize', 14, 'FontWeight', 'bold');
grid on;
ax.GridAlpha = 0.3;

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
    if length(name) > 20
        name = [name(1:17) '...'];
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
