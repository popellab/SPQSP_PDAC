function fig_handles = PSA_plot_waterfall_enhanced(results, filtered_results, params_in, config, varargin)
%PSA_PLOT_WATERFALL_ENHANCED Enhanced waterfall plot showing treatment response heterogeneity
%
% Creates publication-quality waterfall plots with RECIST response categories,
% parameter overlays, and statistical annotations.
%
% Inputs:
%   results             - PSA simulation results array
%   filtered_results    - Filtered patient indices from PSA_filter_patients
%   params_in          - Parameter structure from generate_parameter_samples()
%   config             - PSA configuration structure
%   varargin           - Optional name-value pairs:
%                        'Parameter' - Parameter name for color coding (default: none)
%                        'ResponseSampling' - How to sample response timepoints:
%                                           'final' - Use final timepoint only
%                                           'best' - Best response from all timepoints
%                                           'clinical' - Clinical sampling (skip 30%, sample every 10%)
%                                           [start_frac, interval_frac] - Custom sampling fractions
%                                           (default: 'clinical')
%                        'SavePath' - Directory for saving (default: none)
%                        'Format' - Output format ('png', 'svg', 'eps', default: 'png')
%                        'ColorScheme' - Color palette ('recist', 'viridis', default: 'recist')
%                        'FontSize' - Text size (default: 12)
%                        'FigureSize' - [width, height] in inches (default: [10, 6])
%
% Outputs:
%   fig_handles        - Figure handle(s)

% Parse input arguments
p = inputParser;
addParameter(p, 'Parameter', '', @ischar);
addParameter(p, 'ResponseSampling', 'clinical');
addParameter(p, 'SavePath', '', @ischar);
addParameter(p, 'Format', 'png', @ischar);
addParameter(p, 'ColorScheme', 'recist', @ischar);
addParameter(p, 'FontSize', 12, @isnumeric);
addParameter(p, 'FigureSize', [10, 6], @isnumeric);
parse(p, varargin{:});

param_name = p.Results.Parameter;
response_sampling = p.Results.ResponseSampling;
save_path = p.Results.SavePath;
format = p.Results.Format;
color_scheme = p.Results.ColorScheme;
font_size = p.Results.FontSize;
fig_size = p.Results.FigureSize;

% Get plausible patients
plausible_indices = filtered_results.plausible_successful;
n_patients = length(plausible_indices);

if n_patients == 0
    warning('No plausible patients for waterfall plot');
    fig_handles = [];
    return;
end

% Extract tumor response data
tumor_responses = extract_tumor_responses(results, plausible_indices, response_sampling);

if isempty(tumor_responses)
    warning('No tumor response data found');
    fig_handles = [];
    return;
end

% Sort patients by best response
[sorted_responses, sort_idx] = sort(tumor_responses, 'descend');
sorted_indices = plausible_indices(sort_idx);

% Create figure
fig = figure('Position', [100, 100, fig_size(1)*100, fig_size(2)*100]);
hold on;

% Get colors based on scheme
colors = get_color_scheme(color_scheme, param_name, params_in, sorted_indices);

% Plot waterfall bars
bar_handles = create_waterfall_bars(sorted_responses, colors, n_patients);

% Add RECIST threshold lines
add_recist_lines();

% Add statistical annotations
add_statistical_annotations(sorted_responses, font_size);

% Add parameter legend if specified
if ~isempty(param_name) && isfield(params_in, param_name)
    add_parameter_legend(param_name, params_in, sorted_indices, colors, font_size);
end

% Format axes and labels
format_waterfall_plot(font_size, response_sampling);

% Save figure if requested
if ~isempty(save_path)
    save_figure(fig, save_path, 'waterfall_enhanced', format);
end

fig_handles = fig;

end

function tumor_responses = extract_tumor_responses(results, plausible_indices, response_sampling)
%EXTRACT_TUMOR_RESPONSES Extract tumor response for each patient

n_patients = length(plausible_indices);
tumor_responses = NaN(n_patients, 1);

for i = 1:n_patients
    patient_idx = plausible_indices(i);
    
    try
        % Handle both flat array and chunked results structure
        if size(results, 2) > 1
            % Chunked results (chunks x patients_per_chunk)
            patients_per_chunk = size(results, 2);
            chunk_idx = ceil(patient_idx / patients_per_chunk);
            patient_in_chunk_idx = mod(patient_idx - 1, patients_per_chunk) + 1;
            sim_data = results(chunk_idx, patient_in_chunk_idx).simData;
        else
            % Flat array
            sim_data = results(patient_idx).simData;
        end
        
        if isempty(sim_data)
            continue;
        end
        
        % First try to find pre-calculated percentage change
        data_names = sim_data.DataNames;
        perc_idx = find(strcmp(data_names, 'D_T_perc'), 1);
        
        if ~isempty(perc_idx)
            % Use pre-calculated percentage change
            D_T_perc = sim_data.Data(:, perc_idx);
            tumor_responses(i) = sample_response(D_T_perc, response_sampling);
        else
            % Fallback: calculate from tumor volume
            [~, tumor_data, ~] = selectbyname(sim_data, 'V_T');
            
            if ~isempty(tumor_data) && length(tumor_data) > 1
                baseline = tumor_data(1);
                if baseline > 0
                    % Calculate percent change from baseline
                    percent_changes = ((tumor_data - baseline) / baseline) * 100;
                    tumor_responses(i) = sample_response(percent_changes, response_sampling);
                end
            end
        end
        
    catch
        % Skip patients with missing or invalid data
        continue;
    end
end

% Remove NaN values
valid_idx = ~isnan(tumor_responses);
tumor_responses = tumor_responses(valid_idx);

end

function response = sample_response(percent_changes, response_sampling)
%SAMPLE_RESPONSE Apply sampling strategy to get tumor response

if ischar(response_sampling)
    switch lower(response_sampling)
        case 'final'
            % Use final timepoint only
            response = percent_changes(end);
            
        case 'best'
            % Best response from all timepoints (minimum percent change)
            response = min(percent_changes);
            
        case 'clinical'
            % Clinical sampling: skip initial period, sample periodically, take best
            n_timepoints = length(percent_changes);
            if n_timepoints > 20
                % Skip first 30%, sample every 10% of timepoints
                start_idx = max(1, round(n_timepoints * 0.3));
                sample_interval = max(1, round(n_timepoints * 0.1));
                sample_indices = start_idx:sample_interval:n_timepoints;
                if isempty(sample_indices)
                    sample_indices = n_timepoints;
                end
                sampled_responses = percent_changes(sample_indices);
                response = min(sampled_responses);
            else
                % For short simulations, use final timepoint
                response = percent_changes(end);
            end
            
        otherwise
            error('Unknown response sampling method: %s', response_sampling);
    end
    
elseif isnumeric(response_sampling) && length(response_sampling) == 2
    % Custom sampling: [start_fraction, interval_fraction]
    start_frac = response_sampling(1);
    interval_frac = response_sampling(2);
    
    n_timepoints = length(percent_changes);
    start_idx = max(1, round(n_timepoints * start_frac));
    sample_interval = max(1, round(n_timepoints * interval_frac));
    sample_indices = start_idx:sample_interval:n_timepoints;
    
    if isempty(sample_indices)
        sample_indices = n_timepoints;
    end
    
    sampled_responses = percent_changes(sample_indices);
    response = min(sampled_responses);
    
else
    error('Invalid response_sampling parameter. Use string or [start_frac, interval_frac]');
end

end

function colors = get_color_scheme(color_scheme, param_name, params_in, sorted_indices)
%GET_COLOR_SCHEME Generate colors based on scheme and parameter

n_patients = length(sorted_indices);

% If parameter is specified, use parameter-based coloring regardless of color_scheme
if ~isempty(param_name) && isfield(params_in, param_name)
    % Parameter-based coloring
    param_values = params_in.(param_name).LHS(sorted_indices);
    colors = get_parameter_colors(param_values);
else
    % Use specified color scheme
    switch lower(color_scheme)
        case 'recist'
            % RECIST-based colors (implemented in create_waterfall_bars)
            colors = [];
            
        case 'viridis'
            % Viridis colormap
            cmap = viridis(n_patients);
            colors = cmap;
            
        otherwise
            % Default blue gradient
            colors = repmat([0.2, 0.4, 0.8], n_patients, 1);
    end
end

end

function colors = get_parameter_colors(param_values)
%GET_PARAMETER_COLORS Generate colors based on parameter values

% Normalize parameter values to [0,1]
min_val = min(param_values);
max_val = max(param_values);
if max_val > min_val
    normalized_vals = (param_values - min_val) / (max_val - min_val);
else
    normalized_vals = ones(size(param_values)) * 0.5;
end

% Use viridis colormap for full spectrum (blue->green->yellow->red)
n_patients = length(param_values);
viridis_map = viridis(256); % Get high-resolution colormap

% Map normalized values to colormap indices
colormap_indices = round(normalized_vals * 255) + 1;
colormap_indices = max(1, min(256, colormap_indices)); % Clamp to valid range

% Extract colors for each patient
colors = viridis_map(colormap_indices, :);

end

function bar_handles = create_waterfall_bars(responses, colors, n_patients)
%CREATE_WATERFALL_BARS Create the main waterfall bars

bar_handles = [];

for i = 1:n_patients
    response = responses(i);
    
    % Determine color based on RECIST criteria if no custom colors
    if isempty(colors)
        if response <= -30
            color = [0.2, 0.7, 0.2]; % Green for PR/CR
        elseif response >= 20
            color = [0.8, 0.2, 0.2]; % Red for PD
        else
            color = [0.8, 0.8, 0.2]; % Yellow for SD
        end
    else
        color = colors(i, :);
    end
    
    h = bar(i, response, 'FaceColor', color, 'EdgeColor', 'black', 'LineWidth', 0.5);
    bar_handles = [bar_handles, h];
end

end

function add_recist_lines()
%ADD_RECIST_LINES Add RECIST threshold lines with clear labels

% Get current axes limits for line positioning
xlims = xlim;
ylims = ylim;

% Partial Response threshold
line(xlims, [-30, -30], 'Color', [0.2, 0.7, 0.2], 'LineStyle', '--', ...
    'LineWidth', 2.5);
% Add text label for PR line
text(xlims(2)*0.95, -30, ' PR (-30%)', 'Color', [0.2, 0.7, 0.2], ...
    'FontWeight', 'bold', 'FontSize', 10, 'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'right', 'BackgroundColor', 'white', ...
    'EdgeColor', [0.2, 0.7, 0.2], 'Margin', 2);

% Progressive Disease threshold  
line(xlims, [20, 20], 'Color', [0.8, 0.2, 0.2], 'LineStyle', '--', ...
    'LineWidth', 2.5);
% Add text label for PD line
text(xlims(2)*0.95, 20, ' PD (+20%)', 'Color', [0.8, 0.2, 0.2], ...
    'FontWeight', 'bold', 'FontSize', 10, 'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'right', 'BackgroundColor', 'white', ...
    'EdgeColor', [0.8, 0.2, 0.2], 'Margin', 2);

% Zero line (baseline)
line(xlims, [0, 0], 'Color', 'black', 'LineStyle', '-', 'LineWidth', 1.5);
% Add text label for baseline
text(xlims(1)*0.05, 0, 'Baseline (0%) ', 'Color', 'black', ...
    'FontWeight', 'bold', 'FontSize', 10, 'VerticalAlignment', 'bottom', ...
    'HorizontalAlignment', 'left', 'BackgroundColor', 'white', ...
    'EdgeColor', 'black', 'Margin', 2);


end

function add_statistical_annotations(responses, font_size)
%ADD_STATISTICAL_ANNOTATIONS Add statistical summary

% Calculate statistics
median_response = median(responses);
mean_response = mean(responses);
q25 = prctile(responses, 25);
q75 = prctile(responses, 75);

% Count response categories
n_pr = sum(responses <= -30);
n_sd = sum(responses > -30 & responses < 20);
n_pd = sum(responses >= 20);
n_total = length(responses);

% Add text annotations
ax = gca;
y_max = max(responses) * 1.1;
y_min = min(responses) * 1.1;

% Summary statistics
annotation_text = sprintf(['Median: %.1f%%\\newline' ...
                          'Mean: %.1f%%\\newline' ...
                          'IQR: [%.1f%%, %.1f%%]'], ...
                          median_response, mean_response, q25, q75);

text(0.02, 0.98, annotation_text, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'FontSize', font_size-1, ...
    'BackgroundColor', 'white', 'EdgeColor', 'black');

% Response categories
category_text = sprintf(['PR: %d (%.1f%%)\\newline' ...
                        'SD: %d (%.1f%%)\\newline' ...
                        'PD: %d (%.1f%%)'], ...
                        n_pr, (n_pr/n_total)*100, ...
                        n_sd, (n_sd/n_total)*100, ...
                        n_pd, (n_pd/n_total)*100);

text(0.98, 0.98, category_text, 'Units', 'normalized', ...
    'VerticalAlignment', 'top', 'HorizontalAlignment', 'right', ...
    'FontSize', font_size-1, 'BackgroundColor', 'white', 'EdgeColor', 'black');

end

function add_parameter_legend(param_name, params_in, sorted_indices, colors, font_size)
%ADD_PARAMETER_LEGEND Add legend for parameter-based coloring

if ~isfield(params_in, param_name)
    return;
end

param_values = params_in.(param_name).LHS(sorted_indices);
min_val = min(param_values);
max_val = max(param_values);

% Set viridis colormap for the plot
colormap(viridis(256));

% Create colorbar
c = colorbar('Location', 'eastoutside');

% Format parameter name for display (simple replacement only)
param_display_name = strrep(param_name, '_', ' ');

% Get units if available
units_str = '';
if isfield(params_in, param_name) && isfield(params_in.(param_name), 'units')
    units = params_in.(param_name).units;
    if ~isempty(units) && ~strcmp(units, 'dimensionless')
        units_str = sprintf(' (%s)', units);
    end
end

c.Label.String = sprintf('Parameter: %s%s', param_display_name, units_str);
c.Label.FontSize = font_size;
c.Label.FontWeight = 'bold';

% Set colormap limits
caxis([min_val, max_val]);

% No legend annotation needed - colorbar is self-explanatory

end

function format_waterfall_plot(font_size, response_sampling)
%FORMAT_WATERFALL_PLOT Format axes and labels

xlabel('Patients (sorted by response)', 'FontSize', font_size);
ylabel('Best Overall Response (%)', 'FontSize', font_size);

% Create main title
title('Treatment Response Waterfall Plot', 'FontSize', font_size + 2, 'FontWeight', 'bold');

% Add sampling explanation in bottom-left box
sampling_explanation = get_sampling_explanation(response_sampling);
if ~isempty(sampling_explanation)
    text(0.02, 0.02, sampling_explanation, 'Units', 'normalized', ...
        'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'left', ...
        'FontSize', font_size-1, 'BackgroundColor', 'white', ...
        'EdgeColor', 'black', 'Margin', 3);
end

% Format axes
ax = gca;
ax.FontSize = font_size;
ax.Box = 'on';
ax.LineWidth = 1;

% Set y-axis limits with some padding
y_data = ylim;
ylim([y_data(1) * 1.1, max(y_data(2) * 1.1, 30)]);

grid on;
ax.GridAlpha = 0.3;

end

function explanation = get_sampling_explanation(response_sampling)
%GET_SAMPLING_EXPLANATION Get explanation text for response sampling method

if ischar(response_sampling)
    switch lower(response_sampling)
        case 'final'
            explanation = 'Response measured at final timepoint only';
            
        case 'best'
            explanation = 'Best response across all timepoints (minimum % change)';
            
        case 'clinical'
            explanation = 'Clinical sampling: best response from days 109-365, measured every 36 days';
            
        otherwise
            explanation = '';
    end
    
elseif isnumeric(response_sampling) && length(response_sampling) == 2
    start_pct = response_sampling(1) * 100;
    interval_pct = response_sampling(2) * 100;
    explanation = sprintf('Custom sampling: starting at %.0f%%, every %.0f%% of timepoints', ...
                         start_pct, interval_pct);
else
    explanation = '';
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

fprintf('Saved waterfall plot: %s\n', full_path);

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