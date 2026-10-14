function figs = plot_mechanism_cascade(simBaseline, simScenario, scenario_id, varargin)
%PLOT_MECHANISM_CASCADE Visualize treatment mechanism of action cascade
%
% Creates a comprehensive visualization showing the biological cascade from
% treatment input through immune activation to tumor control. This helps
% understand HOW the treatment works, not just WHAT the outcome is.
%
% The cascade follows: Drug Input → APC Activation → T Cell Priming →
%   Effector Response → Checkpoint/Suppression → CAF Subtypes/Desmoplasia →
%   Chemokine Trafficking
%
% Inputs:
%   simBaseline - Baseline (no treatment) simulation results
%   simScenario - Treatment scenario simulation results
%   scenario_id - Scenario identifier string
%   varargin    - Optional name-value pairs:
%                 'time_units' - 'days' or 'weeks' (default: 'days')
%                 'save_plots' - Save plots to file (default: false)
%                 'output_dir' - Directory for saved plots (default: 'figures')
%
% Outputs:
%   figs - Array of figure handles
%
% Example:
%   [simBase, ~, ~] = run_median_simulation('baseline');
%   [simTx, ~, scenario] = run_median_simulation('scenarios/gvax_nivo.yaml');
%   figs = plot_mechanism_cascade(simBase, simTx, scenario.scenario_id);
%
% See also: compare_scenario_to_baseline, run_median_simulation

%% Parse inputs
p = inputParser;
addRequired(p, 'simBaseline');
addRequired(p, 'simScenario');
addRequired(p, 'scenario_id', @ischar);
addParameter(p, 'time_units', 'days', @(x) ismember(x, {'days', 'weeks'}));
addParameter(p, 'save_plots', false, @islogical);
addParameter(p, 'output_dir', 'figures', @ischar);
parse(p, simBaseline, simScenario, scenario_id, varargin{:});

time_units = p.Results.time_units;
save_plots = p.Results.save_plots;
output_dir = p.Results.output_dir;

%% Setup
time = simScenario.time;
if strcmp(time_units, 'weeks')
    time = time / 7;
    time_label = 'Time (weeks)';
else
    time_label = 'Time (days)';
end

% Publication styling
set(groot, 'defaultAxesFontSize', 9);
set(groot, 'defaultAxesFontName', 'Arial');
set(groot, 'defaultLineLineWidth', 1.5);
set(groot, 'defaultAxesLineWidth', 1);
set(groot, 'defaultAxesBox', 'on');

% Colors
colors = struct();
colors.baseline = [0.4, 0.4, 0.4];      % Gray for baseline
colors.treatment = [0.8, 0.2, 0.2];     % Red for treatment
colors.benefit = [0.2, 0.6, 0.3];       % Green for beneficial
colors.apc = [0.5, 0.2, 0.7];           % Purple for APCs
colors.tcell = [0.0, 0.45, 0.74];       % Blue for T cells

figs = gobjects(0);

%% Figure 1: Treatment Input & Early Signals (Panels 1-8)
fig1 = figure('Position', [50, 50, 1400, 900], 'Color', 'white');
figs(end+1) = fig1;
tiledlayout(4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: GVAX Cells (Vaccine Input)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'GVAX_cells', 'V_ID', ...
    colors, '1. GVAX Vaccine Input', 'GVAX Cells', time_label, true);

% Panel 2: GM-CSF Production
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'GMCSF', 'V_ID', ...
    colors, '2. GM-CSF Production', 'GM-CSF (nM)', time_label, false);

% Panel 3: GVAX Antigen Release
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'P1_GVAX', 'V_ID', ...
    colors, '3. GVAX Antigen Release', 'P1_{GVAX} (nM)', time_label, false);

% Panel 4: Total Antigen Pool (Tumor)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'P1', 'V_T', ...
    colors, '4. Total Antigen Pool', 'P1 (nM)', time_label, false, 1e9);

% Panel 5: APC Recruitment (Injection Site)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'APC', 'V_ID', ...
    colors, '5. APC Recruitment', 'APCs (V_{ID})', time_label, true);

% Panel 6: APC Maturation (Injection Site)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'mAPC', 'V_ID', ...
    colors, '6. APC Maturation', 'Mature APCs (V_{ID})', time_label, true);

% Panel 7: mcDC2 Migration to Lymph Node (GVAX routes to cDC2)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'mcDC2', 'V_LN', ...
    colors, '7. mcDC2 Migration to LN', 'mcDC2 (V_{LN})', time_label, true);

% Panel 8: H_APC_Treg (Antigen-driven APC Activation for Treg priming)
nexttile; hold on; box on;
plot_observable_comparison(simBaseline, simScenario, time, 'H_APC_Treg', ...
    colors, '8. APC Activation Signal', 'H_{APC\_Treg} (0-1)', time_label);

% Title omitted - provided by presentation slide

%% Figure 2: Mechanistic T Cell Activation (Panels 9-17)
% Shows the new mechanistic formulation: direct APC-driven activation with clonal burst
% Panels ordered by CD8 flow through the GVAX cascade:
%   LN priming (mcDC1 -> nCD8 -> aCD8 -> effector CD8) -> circulation (V_C)
%   -> peripheral tissue (V_P) -> tumor (V_T) -> CD8/Treg ratio -> IFN-gamma
fig2 = figure('Position', [100, 100, 1400, 900], 'Color', 'white');
figs(end+1) = fig2;
tiledlayout(3, 3, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 9: mcDC1 in Lymph Node (driver of CD8 activation via cross-presentation)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'mcDC1', 'V_LN', ...
    colors, '9. mcDC1 in LN (CD8 Activation)', 'mcDC1 (V_{LN})', time_label, true);

% Panel 10: Naive CD8 in Lymph Node (substrate for activation)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'nCD8', 'V_LN', ...
    colors, '10. Naive CD8 Pool (LN)', 'nCD8 (V_{LN})', time_label, true);

% Panel 11: Activated CD8 in Lymph Node (k_T_activation_per_APC * mAPC * nT)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'aCD8', 'V_LN', ...
    colors, '11. CD8 Activation (Mechanistic)', 'aCD8 (V_{LN})', time_label, true);
% Add annotation about mechanistic formula
text(0.02, 0.85, 'k_{act/APC} \times mcDC1 \times nT', 'Units', 'normalized', ...
    'FontSize', 7, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);

% Panel 12: Effector CD8 in Lymph Node (clonal burst: n_T_expansion_burst)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'CD8', 'V_LN', ...
    colors, '12. Effector CD8 (Clonal Burst)', 'CD8 (V_{LN})', time_label, true);
text(0.02, 0.85, 'n_{burst} \approx 800 cells/activation', 'Units', 'normalized', ...
    'FontSize', 7, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);

% Panel 13: CD8 in Central Compartment (trafficking)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'CD8', 'V_C', ...
    colors, '13. CD8 in Circulation', 'CD8 (V_C)', time_label, true);

% Panel 14: CD8 in Peripheral Compartment (distribution to peripheral tissue)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'CD8', 'V_P', ...
    colors, '14. CD8 in Periphery', 'CD8 (V_P)', time_label, true);

% Panel 15: CD8 in Tumor (infiltration, reduced by CXCL12)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'CD8', 'V_T', ...
    colors, '15. Tumor-Infiltrating CD8', 'CD8 (V_T)', time_label, true);
text(0.02, 0.85, '(1 - H_{CXCL12}) exclusion', 'Units', 'normalized', ...
    'FontSize', 7, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);

% Panel 16: CD8/Treg Ratio in Tumor
nexttile; hold on; box on;
plot_ratio_comparison(simBaseline, simScenario, time, 'CD8', 'Treg', 'V_T', ...
    colors, '16. CD8/Treg Ratio (Tumor)', 'CD8/Treg', time_label);

% Panel 17: IFN-gamma in Tumor (effector function)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'IFNg', 'V_T', ...
    colors, '17. IFN-\gamma (Effector Function)', 'IFN-\gamma (nM)', time_label, true);

% Title omitted - provided by presentation slide

%% Figure 3: Mechanistic Tumor Killing (Panels 17-24)
% Shows the search-based killing: p_kill * v_search * T/V_tumor
fig3 = figure('Position', [150, 150, 1400, 900], 'Color', 'white');
figs(end+1) = fig3;
tiledlayout(4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 17: CD8 T cells in Tumor (numerator of density)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'CD8', 'V_T', ...
    colors, '17. CD8 in Tumor (T cells)', 'CD8 (V_T)', time_label, true);

% Panel 18: Tumor Volume (denominator of density)
nexttile; hold on; box on;
plot_volume_comparison(simBaseline, simScenario, time, colors, time_label, ...
    '18. Tumor Volume (V_T)');

% Panel 19: T Cell Density (T/V_tumor - key for search-based killing)
nexttile; hold on; box on;
plot_tcell_density(simBaseline, simScenario, time, colors, time_label, ...
    '19. T Cell Density (T/V_T)');

% Panel 20: Collagen (pore size barrier)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'collagen', 'V_T', ...
    colors, '20. Collagen (Pore Barrier)', 'collagen (V_T)', time_label, true);

% Panel 21: H_pore_T (pore-size T cell exclusion)
nexttile; hold on; box on;
plot_observable_comparison(simBaseline, simScenario, time, 'H_pore_T', ...
    colors, '21. Pore-Size T Cell Exclusion', 'H_{pore,T} (0-1)', time_label);
text(0.02, 0.15, '(1 - H_{pore}) in k_{eff}', 'Units', 'normalized', ...
    'FontSize', 7, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);

% Panel 22: Tregs in Tumor (suppression factor)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'Treg', 'V_T', ...
    colors, '22. Tregs (Suppression Factor)', 'Treg (V_T)', time_label, true);
text(0.02, 0.85, 'T/(T + K_{Treg} \times Treg)', 'Units', 'normalized', ...
    'FontSize', 7, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);

% Panel 23: Effective Killing Rate (k_C_Tcell_eff)
nexttile; hold on; box on;
plot_observable_comparison(simBaseline, simScenario, time, 'k_C_Tcell_eff', ...
    colors, '23. Effective Killing Rate', 'k_{eff} (1/day)', time_label);
text(0.02, 0.85, 'p_{kill} \times v_{search} \times T/V', 'Units', 'normalized', ...
    'FontSize', 7, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);

% Panel 24: Tumor Burden (outcome)
nexttile; hold on; box on;
plot_tumor_outcome(simBaseline, simScenario, time, colors, time_label, '24. Tumor Burden');

% Title omitted - provided by presentation slide

%% Figure 4: Checkpoint & Immunosuppression (if applicable)
fig4 = figure('Position', [200, 200, 1400, 600], 'Color', 'white');
figs(end+1) = fig4;
tiledlayout(2, 4, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: PD-1 on CD8
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'PD1_CD8', 'V_T', ...
    colors, 'PD-1 on CD8', 'PD1_{CD8}', time_label, true);

% Panel 2: PD-L1 on Cancer
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'PDL1_C1', 'V_T', ...
    colors, 'PD-L1 on Cancer', 'PDL1_{C1}', time_label, true);

% Panel 3: PD-1/PD-L1 Complex
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'PD1_PDL1_C1', 'V_T', ...
    colors, 'PD-1:PD-L1 Complex', 'PD1:PDL1', time_label, true);

% Panel 4: Nivolumab (if present)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'nivolumab', 'V_C', ...
    colors, 'Nivolumab (Central)', 'Nivolumab (nM)', time_label, true);

% Panel 5: M1 Macrophages
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'Mac_M1', 'V_T', ...
    colors, 'M1 Macrophages', 'M1 (V_T)', time_label, true);

% Panel 6: M2 Macrophages
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'Mac_M2', 'V_T', ...
    colors, 'M2 Macrophages', 'M2 (V_T)', time_label, true);

% Panel 7: MDSCs
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'MDSC', 'V_T', ...
    colors, 'MDSCs', 'MDSC (V_T)', time_label, true);

% Panel 8: CD8/Treg Ratio
nexttile; hold on; box on;
plot_ratio_comparison(simBaseline, simScenario, time, 'CD8', 'Treg', 'V_T', ...
    colors, 'CD8/Treg Ratio', 'CD8/Treg', time_label);

% Title omitted - provided by presentation slide

%% Figure 5: CAF Subtypes & Desmoplasia (Panels 1-8)
% Shows 3 CAF subtypes and their distinct functional roles in the TME
fig5 = figure('Position', [250, 250, 1400, 900], 'Color', 'white');
figs(end+1) = fig5;
tiledlayout(4, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: qPSC (Quiescent PSCs - precursor pool)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'qPSC', 'V_T', ...
    colors, '1. Quiescent PSCs', 'qPSC (cells)', time_label, true);

% Panel 2: myCAF (ECM/CXCL12/TGFb producers)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'myCAF', 'V_T', ...
    colors, '2. myCAFs (ECM/CXCL12/TGF\beta)', 'myCAF (cells)', time_label, true);

% Panel 3: iCAF (IL-6/CXCL12 producers)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'iCAF', 'V_T', ...
    colors, '3. iCAFs (IL-6/CXCL12)', 'iCAF (cells)', time_label, true);

% Panel 4: apCAF (Tolerogenic - Treg induction)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'apCAF', 'V_T', ...
    colors, '4. apCAFs (Tolerogenic)', 'apCAF (cells)', time_label, true);

% Panel 5: Collagen Accumulation (myCAF product)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'collagen', 'V_T', ...
    colors, '5. Collagen (myCAF)', 'collagen (mg)', time_label, true);

% Panel 6: Collagen Volume Fraction
nexttile; hold on; box on;
plot_observable_comparison(simBaseline, simScenario, time, 'phi_collagen', ...
    colors, '6. Collagen Volume Fraction', '\phi_{col}', time_label);

% Panel 7: Pore Size -> T Cell Exclusion
nexttile; hold on; box on;
plot_observable_comparison(simBaseline, simScenario, time, 'H_pore_T', ...
    colors, '7. Pore \rightarrow T Cell Exclusion', 'H_{pore,T} (0-1)', time_label);

% Panel 8: Tregs (apCAF-induced)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'Treg', 'V_T', ...
    colors, '8. Tregs (apCAF-induced)', 'Treg (cells)', time_label, true);

% Title omitted - provided by presentation slide

%% Figure 6: Chemokine Networks & Immune Trafficking (Panels 1-6)
% Shows chemokines controlling immune cell infiltration vs exclusion
fig6 = figure('Position', [300, 300, 1400, 700], 'Color', 'white');
figs(end+1) = fig6;
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

% Panel 1: CXCL12 (T Cell Exclusion)
nexttile; hold on; box on;
plot_species_comparison(simBaseline, simScenario, time, 'CXCL12', 'V_T', ...
    colors, '1. CXCL12 (T Cell Exclusion)', 'CXCL12 (nM)', time_label, false);

% Panel 2: CXCL12 -> T Exclusion
nexttile; hold on; box on;
plot_observable_comparison(simBaseline, simScenario, time, 'H_CXCL12_Texcl', ...
    colors, '2. CXCL12 \rightarrow T Exclusion', 'H_{CXCL12} (0-1)', time_label);

% Panel 3: Pore Size -> APC Migration Block
nexttile; hold on; box on;
plot_observable_comparison(simBaseline, simScenario, time, 'H_pore_APC', ...
    colors, '3. Pore \rightarrow APC Migration Block', 'H_{pore,APC} (0-1)', time_label);

% Panel 4: HMGB1 (DAMP) -> APC Maturation
nexttile; hold on; box on;
plot_observable_comparison(simBaseline, simScenario, time, 'H_HMGB1', ...
    colors, '4. HMGB1 (DAMP) \rightarrow APC Maturation', 'H_{HMGB1} (0-1)', time_label);

% Title omitted - provided by presentation slide

%% Save plots if requested
if save_plots
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    fig_names = {'cascade_input_apc', 'cascade_tcell_activation', 'cascade_tcell_killing', ...
                 'cascade_checkpoint', 'cascade_caf_subtypes', 'cascade_chemokine_trafficking'};
    for i = 1:length(figs)
        filename = sprintf('%s/%s_%s.png', output_dir, scenario_id, fig_names{i});
        saveas(figs(i), filename, 'png');
        fprintf('Saved: %s\n', filename);

        filename_pdf = sprintf('%s/%s_%s.pdf', output_dir, scenario_id, fig_names{i});
        saveas(figs(i), filename_pdf, 'pdf');
    end
end

fprintf('Mechanism cascade visualization complete (%d figures)\n', length(figs));
end


%% Helper Functions

function plot_species_comparison(simBase, simTx, time, species_name, compartment, colors, title_str, ylabel_str, xlabel_str, use_log, scale_factor)
%PLOT_SPECIES_COMPARISON Plot baseline vs treatment for a species

if nargin < 11
    scale_factor = 1;
end
if nargin < 10
    use_log = false;
end

try
    if isempty(compartment)
        data_base = select(simBase, {'Name', species_name});
        data_tx = select(simTx, {'Name', species_name});
    else
        data_base = select(simBase, {'Name', species_name, 'Compartment', compartment});
        data_tx = select(simTx, {'Name', species_name, 'Compartment', compartment});
    end

    if ~isempty(data_base) && ~isempty(data_tx)
        if use_log
            semilogy(time, data_base.Data * scale_factor, '--', 'Color', colors.baseline, 'LineWidth', 1.5);
            semilogy(time, data_tx.Data * scale_factor, '-', 'Color', colors.treatment, 'LineWidth', 1.5);
        else
            plot(time, data_base.Data * scale_factor, '--', 'Color', colors.baseline, 'LineWidth', 1.5);
            plot(time, data_tx.Data * scale_factor, '-', 'Color', colors.treatment, 'LineWidth', 1.5);
        end

        xlabel(xlabel_str);
        ylabel(ylabel_str);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
        grid on; set(gca, 'GridAlpha', 0.3);

        % Add AUC ratio annotation
        auc_base = trapz(time, data_base.Data);
        auc_tx = trapz(time, data_tx.Data);
        if auc_base > 0
            auc_ratio = auc_tx / auc_base;
            if auc_ratio >= 100
                txt = sprintf('%.0fx', auc_ratio);
            elseif auc_ratio >= 10
                txt = sprintf('%.1fx', auc_ratio);
            else
                txt = sprintf('%.2fx', auc_ratio);
            end
            text(0.98, 0.98, txt, 'Units', 'normalized', ...
                'HorizontalAlignment', 'right', 'VerticalAlignment', 'top', ...
                'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7]);
        end
    else
        text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
    end
catch
    text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
end
end


function plot_observable_comparison(simBase, simTx, time, obs_name, colors, title_str, ylabel_str, xlabel_str)
%PLOT_OBSERVABLE_COMPARISON Plot baseline vs treatment for an observable

try
    data_base = select(simBase, {'Name', obs_name});
    data_tx = select(simTx, {'Name', obs_name});

    if ~isempty(data_base) && ~isempty(data_tx)
        plot(time, data_base.Data, '--', 'Color', colors.baseline, 'LineWidth', 1.5);
        plot(time, data_tx.Data, '-', 'Color', colors.treatment, 'LineWidth', 1.5);

        xlabel(xlabel_str);
        ylabel(ylabel_str);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
        grid on; set(gca, 'GridAlpha', 0.3);

        % Add mean value annotation
        avg_base = mean(data_base.Data);
        avg_tx = mean(data_tx.Data);
        text(0.98, 0.98, sprintf('%.3f vs %.3f', avg_base, avg_tx), ...
            'Units', 'normalized', 'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'top', 'FontSize', 8, 'BackgroundColor', [1 1 1 0.7]);
    else
        text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
    end
catch
    text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
end
end


function plot_ratio_comparison(simBase, simTx, time, num_species, denom_species, compartment, colors, title_str, ylabel_str, xlabel_str)
%PLOT_RATIO_COMPARISON Plot ratio of two species

try
    num_base = select(simBase, {'Name', num_species, 'Compartment', compartment});
    denom_base = select(simBase, {'Name', denom_species, 'Compartment', compartment});
    num_tx = select(simTx, {'Name', num_species, 'Compartment', compartment});
    denom_tx = select(simTx, {'Name', denom_species, 'Compartment', compartment});

    if ~isempty(num_base) && ~isempty(denom_base) && ~isempty(num_tx) && ~isempty(denom_tx)
        ratio_base = num_base.Data ./ max(denom_base.Data, 1);
        ratio_tx = num_tx.Data ./ max(denom_tx.Data, 1);

        semilogy(time, ratio_base, '--', 'Color', colors.baseline, 'LineWidth', 1.5);
        semilogy(time, ratio_tx, '-', 'Color', colors.treatment, 'LineWidth', 1.5);
        yline(1, 'k:', 'LineWidth', 0.5);

        xlabel(xlabel_str);
        ylabel(ylabel_str);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
        grid on; set(gca, 'GridAlpha', 0.3);

        % Add final ratio annotation
        final_base = ratio_base(end);
        final_tx = ratio_tx(end);
        text(0.98, 0.98, sprintf('%.1f vs %.1f', final_base, final_tx), ...
            'Units', 'normalized', 'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'top', 'FontSize', 8, 'BackgroundColor', [1 1 1 0.7]);
    else
        text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
    end
catch
    text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
end
end


function plot_tumor_outcome(simBase, simTx, time, colors, xlabel_str, title_str)
%PLOT_TUMOR_OUTCOME Plot tumor burden comparison with outcome metrics

if nargin < 6
    title_str = 'Tumor Burden';
end

try
    % Try C_total first, then C1
    tumor_base = select(simBase, {'Name', 'C_total'});
    tumor_tx = select(simTx, {'Name', 'C_total'});

    if isempty(tumor_base) || isempty(tumor_tx)
        tumor_base = select(simBase, {'Name', 'C1'});
        tumor_tx = select(simTx, {'Name', 'C1'});
    end

    if ~isempty(tumor_base) && ~isempty(tumor_tx)
        semilogy(time, tumor_base.Data, '--', 'Color', colors.baseline, 'LineWidth', 2, 'DisplayName', 'Baseline');
        semilogy(time, tumor_tx.Data, '-', 'Color', colors.treatment, 'LineWidth', 2, 'DisplayName', 'Treatment');

        xlabel(xlabel_str);
        ylabel('Cancer Cells');
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
        legend('Location', 'best', 'FontSize', 8);
        grid on; set(gca, 'GridAlpha', 0.3);

        % Calculate and display reduction
        final_base = tumor_base.Data(end);
        final_tx = tumor_tx.Data(end);
        if final_base > 0
            reduction_pct = (1 - final_tx / final_base) * 100;
            if reduction_pct > 0
                txt = sprintf('%.1f%% reduction', reduction_pct);
            else
                txt = sprintf('%.1f%% increase', -reduction_pct);
            end
            text(0.02, 0.02, txt, 'Units', 'normalized', ...
                'HorizontalAlignment', 'left', 'VerticalAlignment', 'bottom', ...
                'FontSize', 9, 'FontWeight', 'bold', 'BackgroundColor', [1 1 1 0.7]);
        end
    else
        text(0.5, 0.5, 'Tumor data unavailable', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
    end
catch
    text(0.5, 0.5, 'Tumor data unavailable', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
end
end


function plot_volume_comparison(simBase, simTx, time, colors, xlabel_str, title_str)
%PLOT_VOLUME_COMPARISON Plot tumor volume comparison

try
    % V_T is the tumor compartment volume
    vol_base = select(simBase, {'Name', 'V_T'});
    vol_tx = select(simTx, {'Name', 'V_T'});

    if ~isempty(vol_base) && ~isempty(vol_tx)
        semilogy(time, vol_base.Data, '--', 'Color', colors.baseline, 'LineWidth', 1.5);
        semilogy(time, vol_tx.Data, '-', 'Color', colors.treatment, 'LineWidth', 1.5);

        xlabel(xlabel_str);
        ylabel('Volume (mL)');
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
        grid on; set(gca, 'GridAlpha', 0.3);

        % Add final volume annotation
        final_base = vol_base.Data(end);
        final_tx = vol_tx.Data(end);
        text(0.98, 0.98, sprintf('%.2f vs %.2f mL', final_base, final_tx), ...
            'Units', 'normalized', 'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'top', 'FontSize', 8, 'BackgroundColor', [1 1 1 0.7]);
    else
        text(0.5, 0.5, 'Volume data unavailable', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
    end
catch
    text(0.5, 0.5, 'Volume data unavailable', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
end
end


function plot_tcell_density(simBase, simTx, time, colors, xlabel_str, title_str)
%PLOT_TCELL_DENSITY Plot T cell density (T cells / tumor volume)
% This is the key metric for search-based killing: k_eff ∝ T/V_tumor

try
    cd8_base = select(simBase, {'Name', 'CD8', 'Compartment', 'V_T'});
    cd8_tx = select(simTx, {'Name', 'CD8', 'Compartment', 'V_T'});
    vol_base = select(simBase, {'Name', 'V_T'});
    vol_tx = select(simTx, {'Name', 'V_T'});

    if ~isempty(cd8_base) && ~isempty(vol_base) && ~isempty(cd8_tx) && ~isempty(vol_tx)
        % Convert volume from mL to mm³ (1 mL = 1000 mm³)
        vol_base_mm3 = vol_base.Data * 1000;
        vol_tx_mm3 = vol_tx.Data * 1000;

        % Density in cells/mm³
        density_base = cd8_base.Data ./ max(vol_base_mm3, 1e-6);
        density_tx = cd8_tx.Data ./ max(vol_tx_mm3, 1e-6);

        semilogy(time, density_base, '--', 'Color', colors.baseline, 'LineWidth', 1.5);
        semilogy(time, density_tx, '-', 'Color', colors.treatment, 'LineWidth', 1.5);

        xlabel(xlabel_str);
        ylabel('CD8 Density (cells/mm³)');
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
        grid on; set(gca, 'GridAlpha', 0.3);

        % Add annotation about mechanistic significance
        text(0.02, 0.15, 'k_{eff} \propto T/V_{tumor}', 'Units', 'normalized', ...
            'FontSize', 7, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);

        % Add final density annotation
        final_base = density_base(end);
        final_tx = density_tx(end);
        if final_base > 0
            ratio = final_tx / final_base;
            text(0.98, 0.98, sprintf('%.1fx', ratio), ...
                'Units', 'normalized', 'HorizontalAlignment', 'right', ...
                'VerticalAlignment', 'top', 'FontSize', 9, 'FontWeight', 'bold', ...
                'BackgroundColor', [1 1 1 0.7]);
        end
    else
        text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
    end
catch
    text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
end
end


function plot_psc_cancer_ratio(simBase, simTx, time, colors, xlabel_str, title_str)
%PLOT_PSC_CANCER_RATIO Plot myCAF/Cancer ratio (CAF density - PDAC hallmark)
% High myCAF/Cancer ratio is characteristic of PDAC's desmoplastic stroma

try
    apsc_base = select(simBase, {'Name', 'myCAF'});
    apsc_tx = select(simTx, {'Name', 'myCAF'});
    c1_base = select(simBase, {'Name', 'C1'});
    c1_tx = select(simTx, {'Name', 'C1'});

    if ~isempty(apsc_base) && ~isempty(c1_base) && ~isempty(apsc_tx) && ~isempty(c1_tx)
        ratio_base = apsc_base.Data ./ max(c1_base.Data, 1);
        ratio_tx = apsc_tx.Data ./ max(c1_tx.Data, 1);

        plot(time, ratio_base, '--', 'Color', colors.baseline, 'LineWidth', 1.5);
        plot(time, ratio_tx, '-', 'Color', colors.treatment, 'LineWidth', 1.5);

        % Add clinical reference range
        yline(0.1, 'k:', 'LineWidth', 0.5);
        yline(3.0, 'k:', 'LineWidth', 0.5);

        xlabel(xlabel_str);
        ylabel('myCAF/Cancer');
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
        grid on; set(gca, 'GridAlpha', 0.3);

        % Add annotation about clinical range
        text(0.02, 0.15, 'Clinical: 0.1-3.0', 'Units', 'normalized', ...
            'FontSize', 7, 'FontAngle', 'italic', 'Color', [0.3 0.3 0.3]);

        % Add final ratio annotation
        final_base = ratio_base(end);
        final_tx = ratio_tx(end);
        text(0.98, 0.98, sprintf('%.3f vs %.3f', final_base, final_tx), ...
            'Units', 'normalized', 'HorizontalAlignment', 'right', ...
            'VerticalAlignment', 'top', 'FontSize', 8, 'BackgroundColor', [1 1 1 0.7]);
    else
        text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
            'HorizontalAlignment', 'center', 'FontSize', 9);
        title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
    end
catch
    text(0.5, 0.5, 'Data unavailable', 'Units', 'normalized', ...
        'HorizontalAlignment', 'center', 'FontSize', 9);
    title(title_str, 'FontWeight', 'bold', 'FontSize', 10);
end
end
