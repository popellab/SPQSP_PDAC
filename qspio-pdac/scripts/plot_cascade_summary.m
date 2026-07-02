function fig = plot_cascade_summary(simBaseline, simScenario, scenario_id, varargin)
%PLOT_CASCADE_SUMMARY Key mechanism cascade plots for presentations
%
% Creates a single 2x3 figure with the most important treatment cascade plots:
%   1. Treatment input (GVAX/drug)
%   2. APC activation (mAPC in LN)
%   3. T cell infiltration (CD8 in tumor)
%   4. Immunosuppression (CD8/Treg ratio)
%   5. Stromal barrier (ECM)
%   6. Tumor outcome (cancer cells)
%
% Usage:
%   [simBase, ~, ~] = run_median_simulation('baseline');
%   [simTx, ~, scenario] = run_median_simulation('scenarios/gvax_nivo_neoadjuvant_zheng2022.yaml');
%   fig = plot_cascade_summary(simBase, simTx, scenario.scenario_id);

%% Parse inputs
p = inputParser;
addRequired(p, 'simBaseline');
addRequired(p, 'simScenario');
addRequired(p, 'scenario_id', @ischar);
addParameter(p, 'time_units', 'days', @(x) ismember(x, {'days', 'weeks'}));
parse(p, simBaseline, simScenario, scenario_id, varargin{:});

time_units = p.Results.time_units;

%% Setup
time_base = simBaseline.time;
time_tx = simScenario.time;
if strcmp(time_units, 'weeks')
    time_base = time_base / 7;
    time_tx = time_tx / 7;
    time_label = 'Time (weeks)';
else
    time_label = 'Time (days)';
end

% Colors
colors.baseline = [0.5, 0.5, 0.5];
colors.treatment = [0.2, 0.4, 0.8];

% Line width for presentation
LW = 3.5;
TITLE_SIZE = 16;
LABEL_SIZE = 14;
TICK_SIZE = 12;

%% Create figure
fig = figure('Position', [100, 100, 1400, 800], 'Color', 'white');
tiledlayout(2, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

%% Panel 1: Treatment Input
nexttile; hold on; box on;
% Try GVAX first, then nivolumab
data_base = select(simBaseline, {'Name', 'GVAX_cells', 'Compartment', 'V_ID'});
data_tx = select(simScenario, {'Name', 'GVAX_cells', 'Compartment', 'V_ID'});
if isempty(data_tx) || all(data_tx.Data == 0)
    data_base = select(simBaseline, {'Name', 'nivolumab', 'Compartment', 'V_C'});
    data_tx = select(simScenario, {'Name', 'nivolumab', 'Compartment', 'V_C'});
    ylabel_str = 'Nivolumab (nM)';
else
    ylabel_str = 'GVAX Cells';
end
plot_panel(time_base, time_tx, data_base, data_tx, colors, 'Treatment Input', ylabel_str, time_label, true, LW, TITLE_SIZE, LABEL_SIZE, TICK_SIZE);

%% Panel 2: APC Activation (Injection Site)
nexttile; hold on; box on;
data_base = select(simBaseline, {'Name', 'mAPC', 'Compartment', 'V_ID'});
data_tx = select(simScenario, {'Name', 'mAPC', 'Compartment', 'V_ID'});
plot_panel(time_base, time_tx, data_base, data_tx, colors, 'moDC Activation (Inj. Site)', 'mAPC (cells)', time_label, true, LW, TITLE_SIZE, LABEL_SIZE, TICK_SIZE);

%% Panel 3: T Cell Infiltration
nexttile; hold on; box on;
data_base = select(simBaseline, {'Name', 'CD8', 'Compartment', 'V_T'});
data_tx = select(simScenario, {'Name', 'CD8', 'Compartment', 'V_T'});
plot_panel(time_base, time_tx, data_base, data_tx, colors, 'T Cell Infiltration', 'CD8 (cells)', time_label, true, LW, TITLE_SIZE, LABEL_SIZE, TICK_SIZE);

%% Panel 4: CD8/Treg Ratio
nexttile; hold on; box on;
cd8_base = select(simBaseline, {'Name', 'CD8', 'Compartment', 'V_T'});
treg_base = select(simBaseline, {'Name', 'Treg', 'Compartment', 'V_T'});
cd8_tx = select(simScenario, {'Name', 'CD8', 'Compartment', 'V_T'});
treg_tx = select(simScenario, {'Name', 'Treg', 'Compartment', 'V_T'});
if ~isempty(cd8_base) && ~isempty(treg_base)
    ratio_base = cd8_base.Data ./ max(treg_base.Data, 1);
    ratio_tx = cd8_tx.Data ./ max(treg_tx.Data, 1);
    semilogy(time_base, ratio_base, '--', 'Color', colors.baseline, 'LineWidth', LW);
    semilogy(time_tx, ratio_tx, '-', 'Color', colors.treatment, 'LineWidth', LW);
    yline(1, 'k:', 'LineWidth', 1.5);
    add_fold_change(ratio_base, ratio_tx, LABEL_SIZE);
end
xlabel(time_label, 'FontSize', LABEL_SIZE, 'FontWeight', 'bold');
ylabel('CD8/Treg (ratio)', 'FontSize', LABEL_SIZE, 'FontWeight', 'bold');
title('Immune Balance', 'FontWeight', 'bold', 'FontSize', TITLE_SIZE);
set(gca, 'FontSize', TICK_SIZE, 'LineWidth', 1.5);
grid on; set(gca, 'GridAlpha', 0.3);

%% Panel 5: Stromal Barrier (collagen)
nexttile; hold on; box on;
data_base = select(simBaseline, {'Name', 'collagen', 'Compartment', 'V_T'});
data_tx = select(simScenario, {'Name', 'collagen', 'Compartment', 'V_T'});
plot_panel(time_base, time_tx, data_base, data_tx, colors, 'Stromal Barrier', 'collagen (mg)', time_label, true, LW, TITLE_SIZE, LABEL_SIZE, TICK_SIZE);

%% Panel 6: Tumor Outcome
nexttile; hold on; box on;
data_base = select(simBaseline, {'Name', 'C1'});
data_tx = select(simScenario, {'Name', 'C1'});
if ~isempty(data_base) && ~isempty(data_tx)
    semilogy(time_base, data_base.Data, '--', 'Color', colors.baseline, 'LineWidth', LW, 'DisplayName', 'Baseline');
    semilogy(time_tx, data_tx.Data, '-', 'Color', colors.treatment, 'LineWidth', LW, 'DisplayName', 'Treatment');
    legend('Location', 'best', 'FontSize', TICK_SIZE);
    % Show reduction
    reduction = (1 - data_tx.Data(end) / data_base.Data(end)) * 100;
    text(0.02, 0.02, sprintf('%.0f%% reduction', reduction), 'Units', 'normalized', ...
        'VerticalAlignment', 'bottom', 'FontSize', LABEL_SIZE, 'FontWeight', 'bold', ...
        'BackgroundColor', [1 1 1 0.8]);
end
xlabel(time_label, 'FontSize', LABEL_SIZE, 'FontWeight', 'bold');
ylabel('Cancer Cells', 'FontSize', LABEL_SIZE, 'FontWeight', 'bold');
title('Tumor Burden', 'FontWeight', 'bold', 'FontSize', TITLE_SIZE);
set(gca, 'FontSize', TICK_SIZE, 'LineWidth', 1.5);
grid on; set(gca, 'GridAlpha', 0.3);

%% Title
sgtitle('GVAX + Nivolumab: Mechanism of Action', 'FontSize', 20, 'FontWeight', 'bold');

end


function plot_panel(time_base, time_tx, data_base, data_tx, colors, title_str, ylabel_str, xlabel_str, use_log, LW, TITLE_SIZE, LABEL_SIZE, TICK_SIZE)
    if ~isempty(data_base) && ~isempty(data_tx)
        if use_log
            semilogy(time_base, data_base.Data, '--', 'Color', colors.baseline, 'LineWidth', LW);
            semilogy(time_tx, data_tx.Data, '-', 'Color', colors.treatment, 'LineWidth', LW);
        else
            plot(time_base, data_base.Data, '--', 'Color', colors.baseline, 'LineWidth', LW);
            plot(time_tx, data_tx.Data, '-', 'Color', colors.treatment, 'LineWidth', LW);
        end
        add_fold_change(data_base.Data, data_tx.Data, LABEL_SIZE);
    else
        text(0.5, 0.5, 'N/A', 'Units', 'normalized', 'HorizontalAlignment', 'center', 'FontSize', LABEL_SIZE);
    end
    xlabel(xlabel_str, 'FontSize', LABEL_SIZE, 'FontWeight', 'bold');
    ylabel(ylabel_str, 'FontSize', LABEL_SIZE, 'FontWeight', 'bold');
    title(title_str, 'FontWeight', 'bold', 'FontSize', TITLE_SIZE);
    set(gca, 'FontSize', TICK_SIZE, 'LineWidth', 1.5);
    grid on; set(gca, 'GridAlpha', 0.3);
end


function add_fold_change(data_base, data_tx, LABEL_SIZE)
    auc_base = trapz(data_base);
    auc_tx = trapz(data_tx);
    if auc_base > 0
        fold = auc_tx / auc_base;
        if fold >= 10
            txt = sprintf('%.0fx', fold);
        else
            txt = sprintf('%.1fx', fold);
        end
        text(0.98, 0.98, txt, 'Units', 'normalized', ...
            'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
            'FontSize', LABEL_SIZE, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.8]);
    end
end
