function fig_handles = PSA_plot_response_correlation(output_data, output_names, filtered_results, varargin)
%PSA_PLOT_RESPONSE_CORRELATION Create response correlation matrix visualization
%
% Creates a comprehensive correlation matrix showing relationships between
% clinical endpoints with scatter plots, correlation coefficients, and
% marginal histograms.
%
% Inputs:
%   output_data        - Matrix of outputs (patients x outputs) from PSA_extract_outputs
%   output_names       - Cell array of output names
%   filtered_results   - Filtered patient indices from PSA_filter_patients
%   varargin          - Optional name-value pairs:
%                       'MaxOutputs' - Maximum outputs to show (default: 6)
%                       'SavePath' - Directory for saving (default: none)
%                       'Format' - Output format ('png', 'svg', 'eps', default: 'png')
%                       'ColorScheme' - Color palette ('default', 'viridis', default: 'default')
%                       'FontSize' - Text size (default: 10)
%                       'FigureSize' - [width, height] in inches (default: [12, 10])
%                       'ShowHistograms' - Include marginal histograms (default: true)
%
% Outputs:
%   fig_handles       - Figure handle

% Parse input arguments
p = inputParser;
addParameter(p, 'MaxOutputs', 6, @isnumeric);
addParameter(p, 'SavePath', '', @ischar);
addParameter(p, 'Format', 'png', @ischar);
addParameter(p, 'ColorScheme', 'default', @ischar);
addParameter(p, 'FontSize', 10, @isnumeric);
addParameter(p, 'FigureSize', [12, 10], @isnumeric);
addParameter(p, 'ShowHistograms', true, @islogical);
parse(p, varargin{:});

max_outputs = p.Results.MaxOutputs;
save_path = p.Results.SavePath;
format = p.Results.Format;
color_scheme = p.Results.ColorScheme;
font_size = p.Results.FontSize;
fig_size = p.Results.FigureSize;
show_histograms = p.Results.ShowHistograms;

% Get plausible data
plausible_indices = filtered_results.plausible_successful;
if isempty(plausible_indices)
    warning('No plausible patients for correlation analysis');
    fig_handles = [];
    return;
end

% Limit number of outputs for readability
n_outputs = min(max_outputs, length(output_names));
selected_outputs = 1:n_outputs;
data = output_data(plausible_indices, selected_outputs);
names = output_names(selected_outputs);

% Remove any remaining NaN values
valid_rows = all(~isnan(data), 2);
data = data(valid_rows, :);
n_patients = size(data, 1);

if n_patients < 10
    warning('Insufficient patients (%d) for correlation analysis', n_patients);
    fig_handles = [];
    return;
end

fprintf('Creating response correlation matrix for %d outputs, %d patients...\n', n_outputs, n_patients);

% Create figure
fig = figure('Position', [100, 100, fig_size(1)*100, fig_size(2)*100]);

% Calculate correlations
corr_matrix = corr(data);

% Create subplot layout
if show_histograms
    % With histograms: diagonal shows histograms, upper triangle shows correlations, lower shows scatter
    for i = 1:n_outputs
        for j = 1:n_outputs
            subplot_idx = (i-1)*n_outputs + j;
            subplot(n_outputs, n_outputs, subplot_idx);
            
            if i == j
                % Diagonal: histogram
                create_histogram(data(:, i), names{i}, color_scheme, font_size);
            elseif i < j
                % Upper triangle: correlation coefficient
                create_correlation_text(corr_matrix(i, j), font_size);
            else
                % Lower triangle: scatter plot
                create_scatter_plot(data(:, j), data(:, i), names{j}, names{i}, ...
                                  corr_matrix(i, j), color_scheme, font_size);
            end
        end
    end
else
    % Without histograms: symmetric scatter plot matrix
    for i = 1:n_outputs
        for j = 1:n_outputs
            subplot_idx = (i-1)*n_outputs + j;
            subplot(n_outputs, n_outputs, subplot_idx);
            
            if i == j
                % Diagonal: output name and basic stats
                create_stats_text(data(:, i), names{i}, font_size);
            else
                % Off-diagonal: scatter plot
                create_scatter_plot(data(:, j), data(:, i), names{j}, names{i}, ...
                                  corr_matrix(i, j), color_scheme, font_size);
            end
        end
    end
end

% Add overall title
sgtitle('Clinical Endpoint Correlation Matrix', 'FontSize', font_size + 4, 'FontWeight', 'bold');

% Adjust spacing
set(fig, 'Units', 'normalized');
set(fig, 'Position', [0.1, 0.1, 0.8, 0.8]);

% Save figure if requested
if ~isempty(save_path)
    save_figure(fig, save_path, 'response_correlation', format);
end

fig_handles = fig;

end

function create_histogram(data, name, color_scheme, font_size)
%CREATE_HISTOGRAM Create histogram for diagonal elements

% Get color
switch lower(color_scheme)
    case 'viridis'
        color = [0.267, 0.004, 0.329];
    otherwise
        color = [0.3, 0.5, 0.8];
end

% Create histogram
histogram(data, 'FaceColor', color, 'EdgeColor', 'black', 'FaceAlpha', 0.7);

% Format
title(format_name(name), 'FontSize', font_size, 'FontWeight', 'bold');
ax = gca;
ax.FontSize = font_size-1;
grid on;
ax.GridAlpha = 0.3;

% Add basic statistics
mean_val = mean(data);
std_val = std(data);
text(0.02, 0.98, sprintf('μ=%.2g\nσ=%.2g', mean_val, std_val), ...
    'Units', 'normalized', 'VerticalAlignment', 'top', ...
    'FontSize', font_size-2, 'BackgroundColor', 'white');

end

function create_correlation_text(corr_val, font_size)
%CREATE_CORRELATION_TEXT Display correlation coefficient

% Determine color based on correlation strength
if abs(corr_val) > 0.7
    color = 'red';
    weight = 'bold';
elseif abs(corr_val) > 0.4
    color = 'black';
    weight = 'bold';
else
    color = [0.5, 0.5, 0.5];
    weight = 'normal';
end

% Display correlation
text(0.5, 0.5, sprintf('r = %.3f', corr_val), ...
    'HorizontalAlignment', 'center', 'VerticalAlignment', 'middle', ...
    'FontSize', font_size + 2, 'FontWeight', weight, 'Color', color);

% Add significance stars if strong correlation
if abs(corr_val) > 0.7
    text(0.5, 0.3, '***', 'HorizontalAlignment', 'center', ...
        'FontSize', font_size, 'Color', color);
elseif abs(corr_val) > 0.4
    text(0.5, 0.3, '**', 'HorizontalAlignment', 'center', ...
        'FontSize', font_size, 'Color', color);
end

% Remove axes
set(gca, 'XTick', [], 'YTick', [], 'Box', 'on');

end

function create_scatter_plot(x_data, y_data, x_name, y_name, corr_val, color_scheme, font_size)
%CREATE_SCATTER_PLOT Create scatter plot with trend line

% Get colors
switch lower(color_scheme)
    case 'viridis'
        scatter_color = [0.127, 0.566, 0.551];
        line_color = [0.993, 0.906, 0.144];
    otherwise
        scatter_color = [0.3, 0.5, 0.8];
        line_color = [0.8, 0.3, 0.3];
end

% Create scatter plot
scatter(x_data, y_data, 20, 'MarkerFaceColor', scatter_color, ...
    'MarkerEdgeColor', 'black', 'MarkerFaceAlpha', 0.6, 'LineWidth', 0.5);

hold on;

% Add trend line if correlation is significant
if abs(corr_val) > 0.3
    % Linear fit
    p = polyfit(x_data, y_data, 1);
    x_fit = linspace(min(x_data), max(x_data), 100);
    y_fit = polyval(p, x_fit);
    plot(x_fit, y_fit, 'Color', line_color, 'LineWidth', 2);
    
    % Add R² value
    r_squared = corr_val^2;
    text(0.02, 0.98, sprintf('R² = %.3f', r_squared), ...
        'Units', 'normalized', 'VerticalAlignment', 'top', ...
        'FontSize', font_size-1, 'BackgroundColor', 'white');
end

% Format axes
if font_size <= 8  % For small subplots, reduce label complexity
    xlabel(format_name_short(x_name), 'FontSize', font_size-1);
    ylabel(format_name_short(y_name), 'FontSize', font_size-1);
else
    xlabel(format_name(x_name), 'FontSize', font_size-1);
    ylabel(format_name(y_name), 'FontSize', font_size-1);
end

ax = gca;
ax.FontSize = font_size-2;
grid on;
ax.GridAlpha = 0.3;

end

function create_stats_text(data, name, font_size)
%CREATE_STATS_TEXT Create statistics summary for diagonal

% Calculate statistics
mean_val = mean(data);
median_val = median(data);
std_val = std(data);
min_val = min(data);
max_val = max(data);

% Display name and stats
text(0.5, 0.7, format_name(name), 'HorizontalAlignment', 'center', ...
    'FontSize', font_size, 'FontWeight', 'bold');

stats_text = sprintf(['Mean: %.2g\n' ...
                     'Median: %.2g\n' ...
                     'Std: %.2g\n' ...
                     'Range: [%.2g, %.2g]'], ...
                     mean_val, median_val, std_val, min_val, max_val);

text(0.5, 0.4, stats_text, 'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'middle', 'FontSize', font_size-2);

% Remove axes
set(gca, 'XTick', [], 'YTick', [], 'Box', 'on');

end

function formatted = format_name(name)
%FORMAT_NAME Format output names for display

formatted = strrep(name, '_', ' ');
words = strsplit(formatted, ' ');
for i = 1:length(words)
    if ~isempty(words{i})
        words{i}(1) = upper(words{i}(1));
    end
end
formatted = strjoin(words, ' ');

% Truncate very long names
if length(formatted) > 25
    formatted = [formatted(1:22) '...'];
end

end

function formatted = format_name_short(name)
%FORMAT_NAME_SHORT Format output names for small subplots

formatted = format_name(name);

% Truncate for small subplots
if length(formatted) > 15
    formatted = [formatted(1:12) '...'];
end

end

function save_figure(fig, save_path, filename, format)
%SAVE_FIGURE Save figure to specified path and format

if ~exist(save_path, 'dir')
    mkdir(save_path);
end

full_path = fullfile(save_path, [filename, '.', format]);

switch lower(format)
    case 'png'
        exportgraphics(fig, full_path, 'Resolution', 300);
    case 'svg'
        saveas(fig, full_path, 'svg');
    case 'eps'
        exportgraphics(fig, full_path, 'ContentType', 'vector');
    otherwise
        warning('Unknown format %s, saving as PNG', format);
        exportgraphics(fig, full_path, 'Resolution', 300);
end

fprintf('Saved response correlation plot: %s\n', full_path);

end