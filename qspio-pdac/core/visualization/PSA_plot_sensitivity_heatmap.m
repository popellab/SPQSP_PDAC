function fig = PSA_plot_sensitivity_heatmap(prcc_values, p_values, param_names, output_names)
%PSA_PLOT_SENSITIVITY_HEATMAP Create PRCC sensitivity heatmap
%
% Creates a heatmap showing parameter sensitivity (PRCC values) across
% multiple outputs with statistical significance indicators.
%
% Inputs:
%   prcc_values  - Matrix of PRCC values (parameters x outputs)
%   p_values     - Matrix of p-values (parameters x outputs)
%   param_names  - Cell array of parameter names
%   output_names - Cell array of output names
%
% Outputs:
%   fig - Figure handle

n_params = length(param_names);
n_outputs = length(output_names);

% Create figure
fig = figure('Position', [100, 100, max(600, 80*n_outputs), max(400, 30*n_params)]);

% Create heatmap
imagesc(prcc_values);
hold on;

% Set colormap (red-white-blue)
n_colors = 64;
red_to_white = [linspace(0.8, 1, n_colors/2)', linspace(0.1, 1, n_colors/2)', linspace(0.1, 1, n_colors/2)'];
white_to_blue = [linspace(1, 0.1, n_colors/2)', linspace(1, 0.1, n_colors/2)', linspace(1, 0.8, n_colors/2)'];
colormap([red_to_white; white_to_blue]);
caxis([-1 1]);

% Add colorbar
cb = colorbar;
cb.Label.String = 'PRCC';
cb.FontSize = 10;

% Format axes
ax = gca;
ax.FontSize = 10;
set(ax, 'XTick', 1:n_outputs);
set(ax, 'YTick', 1:n_params);
set(ax, 'XTickLabels', format_labels(output_names));
set(ax, 'YTickLabels', format_labels(param_names));
ax.XTickLabelRotation = 45;

% Add grid
grid on;
ax.GridColor = [0.8 0.8 0.8];
ax.GridAlpha = 0.3;

% Add significance stars
for i = 1:n_params
    for j = 1:n_outputs
        p_val = p_values(i, j);

        if p_val < 0.01
            star_text = '**';
        elseif p_val < 0.05
            star_text = '*';
        else
            continue;
        end

        text(j, i, star_text, 'HorizontalAlignment', 'center', ...
            'VerticalAlignment', 'middle', 'FontSize', 12, ...
            'FontWeight', 'bold', 'Color', 'white');
    end
end

% Labels and title
title('Parameter Sensitivity (PRCC)', 'FontSize', 14, 'FontWeight', 'bold');
xlabel('Outputs', 'FontSize', 12);
ylabel('Parameters', 'FontSize', 12);

end

function labels = format_labels(raw_labels)
%FORMAT_LABELS Clean up labels for display
labels = raw_labels;
for i = 1:length(labels)
    labels{i} = strrep(labels{i}, '_', ' ');
    if length(labels{i}) > 15
        labels{i} = [labels{i}(1:12) '...'];
    end
end
end
