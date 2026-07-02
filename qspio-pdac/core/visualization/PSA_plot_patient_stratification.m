function fig_handles = PSA_plot_patient_stratification(params_in, output_data, output_names, filtered_results, varargin)
%PSA_PLOT_PATIENT_STRATIFICATION Create patient stratification visualization
%
% Creates dimensionality reduction plots (t-SNE/PCA) to identify patient subgroups
% based on parameter space and clinical outcomes. Overlays clinical endpoints
% on the embeddings for pattern identification.
%
% Inputs:
%   params_in         - Parameter structure from generate_parameter_samples()
%   output_data       - Matrix of outputs (patients x outputs) from PSA_extract_outputs
%   output_names      - Cell array of output names
%   filtered_results  - Filtered patient indices from PSA_filter_patients
%   varargin         - Optional name-value pairs:
%                      'Method' - Dimensionality reduction ('tsne', 'pca', 'umap', default: 'tsne')
%                      'ColorBy' - What to color by ('outcome', 'parameter', default: 'outcome')
%                      'OutcomeIndex' - Which output to use for coloring (default: 1)
%                      'ParameterName' - Parameter name for coloring (default: first param)
%                      'SavePath' - Directory for saving (default: none)
%                      'Format' - Output format ('png', 'svg', 'eps', default: 'png')
%                      'FontSize' - Text size (default: 12)
%                      'FigureSize' - [width, height] in inches (default: [12, 8])
%                      'ShowClusters' - Identify and show clusters (default: true)
%
% Outputs:
%   fig_handles      - Figure handle(s)

% Parse input arguments
p = inputParser;
addParameter(p, 'Method', 'tsne', @ischar);
addParameter(p, 'ColorBy', 'outcome', @ischar);
addParameter(p, 'OutcomeIndex', 1, @isnumeric);
addParameter(p, 'ParameterName', '', @ischar);
addParameter(p, 'SavePath', '', @ischar);
addParameter(p, 'Format', 'png', @ischar);
addParameter(p, 'FontSize', 12, @isnumeric);
addParameter(p, 'FigureSize', [12, 8], @isnumeric);
addParameter(p, 'ShowClusters', true, @islogical);
parse(p, varargin{:});

method = p.Results.Method;
color_by = p.Results.ColorBy;
outcome_idx = p.Results.OutcomeIndex;
param_name = p.Results.ParameterName;
save_path = p.Results.SavePath;
format = p.Results.Format;
font_size = p.Results.FontSize;
fig_size = p.Results.FigureSize;
show_clusters = p.Results.ShowClusters;

% Get plausible patients
plausible_indices = filtered_results.plausible_successful;
if length(plausible_indices) < 20
    warning('Insufficient patients (%d) for stratification analysis', length(plausible_indices));
    fig_handles = [];
    return;
end

fprintf('Creating patient stratification plot using %s for %d patients...\n', ...
    upper(method), length(plausible_indices));

% Prepare parameter matrix
param_matrix = extract_parameter_matrix(params_in, plausible_indices);
if isempty(param_matrix)
    warning('Could not extract parameter matrix');
    fig_handles = [];
    return;
end

% Prepare outcome data
outcome_data = output_data(plausible_indices, :);

% Perform dimensionality reduction
[embedding, method_used] = perform_dimensionality_reduction(param_matrix, method);

% Determine coloring scheme
[color_values, color_name, colormap_type] = get_coloring_scheme(color_by, outcome_data, ...
    outcome_idx, output_names, params_in, param_name, plausible_indices);

% Create main figure
fig = create_main_stratification_plot(embedding, color_values, color_name, ...
    colormap_type, method_used, font_size, fig_size);

% Add clusters if requested
if show_clusters
    add_cluster_analysis(embedding, color_values, font_size);
end

% Save figure if requested
if ~isempty(save_path)
    save_figure(fig, save_path, sprintf('patient_stratification_%s', method), format);
end

fig_handles = fig;

end

function param_matrix = extract_parameter_matrix(params_in, plausible_indices)
%EXTRACT_PARAMETER_MATRIX Extract parameter matrix for dimensionality reduction

param_names = params_in.names;
n_patients = length(plausible_indices);
n_params = length(param_names);

param_matrix = zeros(n_patients, n_params);

for i = 1:n_params
    param_name = param_names{i};
    if isfield(params_in, param_name) && isfield(params_in.(param_name), 'LHS')
        param_matrix(:, i) = params_in.(param_name).LHS(plausible_indices);
    end
end

% Remove any parameters with zero variance or NaN values
valid_params = ~any(isnan(param_matrix), 1) & (std(param_matrix) > eps);
param_matrix = param_matrix(:, valid_params);

% Standardize parameters for dimensionality reduction
param_matrix = (param_matrix - mean(param_matrix)) ./ std(param_matrix);

end

function [embedding, method_used] = perform_dimensionality_reduction(param_matrix, method)
%PERFORM_DIMENSIONALITY_REDUCTION Apply dimensionality reduction

n_patients = size(param_matrix, 1);

switch lower(method)
    case 'tsne'
        if n_patients >= 30
            try
                embedding = tsne(param_matrix, 'NumDimensions', 2, 'Perplexity', ...
                    min(30, floor(n_patients/4)));
                method_used = 't-SNE';
            catch
                warning('t-SNE failed, falling back to PCA');
                [embedding, method_used] = perform_pca(param_matrix);
            end
        else
            warning('Too few patients for t-SNE, using PCA');
            [embedding, method_used] = perform_pca(param_matrix);
        end
        
    case 'pca'
        [embedding, method_used] = perform_pca(param_matrix);
        
    case 'umap'
        % Simple UMAP-like embedding (or fallback to PCA if not available)
        try
            embedding = perform_simple_umap(param_matrix);
            method_used = 'UMAP-like';
        catch
            warning('UMAP not available, using PCA');
            [embedding, method_used] = perform_pca(param_matrix);
        end
        
    otherwise
        warning('Unknown method %s, using PCA', method);
        [embedding, method_used] = perform_pca(param_matrix);
end

end

function [embedding, method_used] = perform_pca(param_matrix)
%PERFORM_PCA Principal Component Analysis

[coeff, score, ~, ~, explained] = pca(param_matrix);
embedding = score(:, 1:2);
method_used = sprintf('PCA (%.1f%% variance)', sum(explained(1:2)));

end

function embedding = perform_simple_umap(param_matrix)
%PERFORM_SIMPLE_UMAP Simple UMAP-like embedding using local PCA

% This is a simplified version - for production use, consider the UMAP toolbox
n_patients = size(param_matrix, 1);
k_neighbors = min(15, floor(n_patients/3));

% Compute pairwise distances
distances = pdist2(param_matrix, param_matrix);

% For each point, find k nearest neighbors and perform local PCA
embedding = zeros(n_patients, 2);

for i = 1:n_patients
    [~, neighbor_idx] = sort(distances(i, :));
    neighbors = neighbor_idx(1:k_neighbors);
    
    local_data = param_matrix(neighbors, :);
    [~, score] = pca(local_data);
    
    % Find position of current point in local embedding
    point_idx = find(neighbors == i);
    if ~isempty(point_idx) && size(score, 2) >= 2
        embedding(i, :) = score(point_idx, 1:2);
    end
end

end

function [color_values, color_name, colormap_type] = get_coloring_scheme(color_by, outcome_data, ...
    outcome_idx, output_names, params_in, param_name, plausible_indices)
%GET_COLORING_SCHEME Determine what to color points by

switch lower(color_by)
    case 'outcome'
        if outcome_idx <= size(outcome_data, 2)
            color_values = outcome_data(:, outcome_idx);
            color_name = format_name(output_names{outcome_idx});
            colormap_type = 'continuous';
        else
            warning('Invalid outcome index, using first outcome');
            color_values = outcome_data(:, 1);
            color_name = format_name(output_names{1});
            colormap_type = 'continuous';
        end
        
    case 'parameter'
        if isempty(param_name)
            param_name = params_in.names{1};
        end
        
        if isfield(params_in, param_name)
            color_values = params_in.(param_name).LHS(plausible_indices);
            color_name = format_name(param_name);
            colormap_type = 'continuous';
        else
            warning('Parameter %s not found, using outcome coloring', param_name);
            color_values = outcome_data(:, 1);
            color_name = format_name(output_names{1});
            colormap_type = 'continuous';
        end
        
    otherwise
        % Default to first outcome
        color_values = outcome_data(:, 1);
        color_name = format_name(output_names{1});
        colormap_type = 'continuous';
end

end

function fig = create_main_stratification_plot(embedding, color_values, color_name, ...
    colormap_type, method_used, font_size, fig_size)
%CREATE_MAIN_STRATIFICATION_PLOT Create the main stratification plot

fig = figure('Position', [100, 100, fig_size(1)*100, fig_size(2)*100]);

% Create scatter plot
scatter(embedding(:, 1), embedding(:, 2), 60, color_values, 'filled', ...
    'MarkerEdgeColor', 'black', 'LineWidth', 0.5, 'MarkerFaceAlpha', 0.8);

% Set colormap
switch colormap_type
    case 'continuous'
        colormap(viridis(256));
    case 'discrete'
        colormap(lines(max(color_values)));
end

% Add colorbar
c = colorbar;
c.Label.String = color_name;
c.Label.FontSize = font_size;

% Format plot
xlabel(sprintf('%s Component 1', method_used), 'FontSize', font_size);
ylabel(sprintf('%s Component 2', method_used), 'FontSize', font_size);
title(sprintf('Patient Stratification (%s)', method_used), ...
    'FontSize', font_size + 2, 'FontWeight', 'bold');

ax = gca;
ax.FontSize = font_size;
ax.Box = 'on';
grid on;
ax.GridAlpha = 0.3;

end

function add_cluster_analysis(embedding, color_values, font_size)
%ADD_CLUSTER_ANALYSIS Add cluster identification and characterization

n_patients = size(embedding, 1);

% Determine optimal number of clusters using elbow method
max_k = min(8, floor(n_patients/5));
if max_k < 2
    return;
end

silhouette_scores = zeros(max_k-1, 1);
for k = 2:max_k
    try
        cluster_idx = kmeans(embedding, k, 'Replicates', 5);
        silhouette_scores(k-1) = mean(silhouette(embedding, cluster_idx));
    catch
        silhouette_scores(k-1) = 0;
    end
end

[~, optimal_k_idx] = max(silhouette_scores);
optimal_k = optimal_k_idx + 1;

% Perform clustering with optimal k
try
    cluster_idx = kmeans(embedding, optimal_k, 'Replicates', 10);
    
    % Add cluster boundaries
    hold on;
    
    % Draw convex hulls around clusters
    colors = lines(optimal_k);
    for k = 1:optimal_k
        cluster_points = embedding(cluster_idx == k, :);
        if size(cluster_points, 1) >= 3
            try
                hull_idx = convhull(cluster_points(:, 1), cluster_points(:, 2));
                plot(cluster_points(hull_idx, 1), cluster_points(hull_idx, 2), ...
                    '--', 'Color', colors(k, :), 'LineWidth', 2);
            catch
                % Skip if convex hull fails
            end
        end
        
        % Add cluster label
        centroid = mean(cluster_points);
        text(centroid(1), centroid(2), sprintf('C%d', k), ...
            'FontSize', font_size, 'FontWeight', 'bold', ...
            'HorizontalAlignment', 'center', 'BackgroundColor', 'white', ...
            'EdgeColor', colors(k, :));
    end
    
    % Add cluster statistics
    cluster_stats = sprintf('Identified %d clusters\nSilhouette score: %.3f', ...
        optimal_k, silhouette_scores(optimal_k_idx));
    
    text(0.02, 0.98, cluster_stats, 'Units', 'normalized', ...
        'VerticalAlignment', 'top', 'FontSize', font_size-1, ...
        'BackgroundColor', 'white', 'EdgeColor', 'black');
    
catch
    % Clustering failed, skip
end

end

function formatted = format_name(name)
%FORMAT_NAME Format names for display

formatted = strrep(name, '_', ' ');
words = strsplit(formatted, ' ');
for i = 1:length(words)
    if ~isempty(words{i})
        words{i}(1) = upper(words{i}(1));
    end
end
formatted = strjoin(words, ' ');

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

fprintf('Saved patient stratification plot: %s\n', full_path);

end

function cmap = viridis(n)
%VIRIDIS Generate viridis colormap
% Simplified viridis colormap implementation

if n <= 1
    cmap = [0.267, 0.004, 0.329];
    return;
end

% Viridis key colors
key_colors = [
    0.267, 0.004, 0.329;  % Dark purple
    0.229, 0.322, 0.545;  % Blue
    0.127, 0.566, 0.551;  % Teal
    0.369, 0.788, 0.382;  % Green
    0.993, 0.906, 0.144   % Yellow
];

% Interpolate
t = linspace(1, size(key_colors, 1), n);
cmap = interp1(1:size(key_colors, 1), key_colors, t);

end