function fig_handles = plot_qsp_comprehensive(simData, model, varargin)
%PLOT_QSP_COMPREHENSIVE Comprehensive diagnostic plotting for QSP-IO simulations
%
% Creates comprehensive visualization of key simulation trajectories with
% improved organization, cleaner styling, and better readability.
%
% Inputs:
%   simData - SimBiology simulation data
%   model   - SimBiology model object
%   varargin - Optional parameters:
%              'patient_id', ID - Label for this patient (default: '')
%              'save_plots', true/false - Save plots to file (default: false)
%              'plot_selection', {...} - Which plot categories to show (default: all)
%              'time_units', 'days'/'weeks'/'months' - Time axis units (default: 'days')
%
% Outputs:
%   fig_handles - Array of figure handles for further customization
%
% Available plot categories:
%   'tumor'      - Tumor growth and antigens
%   'immune'     - T cell populations across compartments  
%   'checkpoints'- Checkpoint proteins and drug concentrations
%   'stroma'     - Fibroblasts, ECM, and tumor microenvironment
%   'macrophages'- Macrophage populations and ratios
%
% Example:
%   % Plot all categories
%   figs = plot_qsp_comprehensive(simData, model, 'patient_id', 'P001');
%   
%   % Plot only tumor and immune trajectories
%   figs = plot_qsp_comprehensive(simData, model, 'plot_selection', {'tumor', 'immune'});


%% Parse inputs
p = inputParser;
addParameter(p, 'patient_id', '', @ischar);
addParameter(p, 'save_plots', false, @islogical);
addParameter(p, 'plot_selection', {'tumor', 'immune', 'checkpoints', 'stroma', 'macrophages'}, @iscell);
addParameter(p, 'time_units', 'days', @(x) ismember(x, {'days', 'weeks', 'months'}));
parse(p, varargin{:});

patient_id = p.Results.patient_id;
save_plots = p.Results.save_plots;
plot_selection = p.Results.plot_selection;
time_units = p.Results.time_units;

%% Setup
numClones = howManyClones(simData);
numMHC = howManyMHC(simData);

% Convert time if needed
time_data = simData.time;
time_factor = 1;
switch time_units
    case 'weeks'
        time_data = time_data / 7;
        time_factor = 7;
    case 'months'
        time_data = time_data / 30.44; % Average days per month
        time_factor = 30.44;
end

fig_handles = [];

%% Plot 1: Tumor Growth and Dynamics
if ismember('tumor', plot_selection)
    fig1 = figure('Position', [100, 100, 1200, 800]);
    fig_handles(end+1) = fig1;
    
    % Title
    if ~isempty(patient_id)
        sgtitle(sprintf('Tumor Dynamics - Patient %s', patient_id), 'FontSize', 16, 'FontWeight', 'bold');
    else
        sgtitle('Tumor Dynamics', 'FontSize', 16, 'FontWeight', 'bold');
    end
    
    % Cancer cell populations
    subplot(2,3,1); hold on; box on;
    
    % Find all cancer clones dynamically
    cancer_clones = findCancerClones(simData);
    colors = lines(length(cancer_clones) + 1); % +1 for carrying capacity
    legend_entries = {};
    
    % Plot each cancer clone
    for i = 1:length(cancer_clones)
        clone_name = cancer_clones{i};
        simbio_plot_time(simData, clone_name, time_data, 'Color', colors(i,:), 'LineWidth', 2);
        legend_entries{i} = clone_name;
    end
    
    % Plot carrying capacity if available
    try
        simbio_plot_time(simData, 'K', time_data, 'Color', colors(end,:), 'LineWidth', 1, 'LineStyle', '--');
        legend_entries{end+1} = 'K (Capacity)';
    catch
        % K might not be available
    end
    
    xlabel(sprintf('Time (%s)', time_units)); 
    ylabel('Cancer Cells');
    set(gca, 'YScale', 'log', 'FontSize', 10);
    legend(legend_entries, 'Location', 'best');
    title('Cancer Cell Populations');
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    
    % Tumor volume and size
    subplot(2,3,2); hold on; box on;
    yyaxis left
    selected_data = select(simData, {'Name', 'V_T','Type','compartment'});
    V_T = selected_data.Data;
    plot(time_data, V_T, 'Color', [0.8, 0.4, 0.2], 'LineWidth', 2);
    ylabel('Tumor Volume (L)');
    set(gca, 'YScale', 'log', 'YColor', [0.8, 0.4, 0.2]);
    
    yyaxis right
    D_T = 2 * (3/(4*pi)*V_T).^(1/3) * 10; % Convert to cm
    plot(time_data, D_T, 'Color', [0.4, 0.2, 0.8], 'LineWidth', 2);
    ylabel('Tumor Diameter (cm)');
    set(gca, 'YColor', [0.4, 0.2, 0.8]);
    
    xlabel(sprintf('Time (%s)', time_units));
    title('Tumor Volume & Size');
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    
    % Tumor antigens
    subplot(2,3,3); hold on; box on;
    colors = lines(numClones+1);
    for i = 0:numClones
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', sprintf('P%d', i)});
        P_data = selected_data.Data;
        plot(time_data, P_data, 'Color', colors(i+1,:), 'LineWidth', 2);
        legend_entries{i+1} = sprintf('P%d', i);
    end
    xlabel(sprintf('Time (%s)', time_units));
    ylabel('Protein Concentration (M)');
    set(gca, 'YScale', 'log', 'FontSize', 10);
    legend(legend_entries, 'Location', 'best');
    title('Tumor Antigens');
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    
    % RECIST response over time
    subplot(2,3,4); hold on; box on;
    baseline_volume = V_T(1);
    percent_change = ((V_T - baseline_volume) / baseline_volume) * 100;
    plot(time_data, percent_change, 'Color', [0.2, 0.6, 0.8], 'LineWidth', 2);
    
    % Add RECIST thresholds
    yline(20, 'r--', 'PD Threshold', 'LineWidth', 1.5);
    yline(-30, 'g--', 'PR Threshold', 'LineWidth', 1.5);
    yline(0, 'k:', 'Baseline', 'LineWidth', 1);
    
    xlabel(sprintf('Time (%s)', time_units));
    ylabel('Tumor Change from Baseline (%)');
    title('RECIST Response');
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    ylim([-100, 200]);
    
    % Tumor growth rate
    subplot(2,3,5); hold on; box on;
    if length(V_T) > 1
        growth_rate = diff(log(V_T)) ./ diff(simData.time) * time_factor;
        plot(time_data(2:end), growth_rate, 'Color', [0.6, 0.2, 0.6], 'LineWidth', 2);
        ylabel(sprintf('Growth Rate (1/%s)', time_units));
        yline(0, 'k:', 'Zero Growth', 'LineWidth', 1);
    end
    xlabel(sprintf('Time (%s)', time_units));
    title('Tumor Growth Rate');
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    
    % Collagen and fibrosis
    subplot(2,3,6); hold on; box on;
    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'collagen'});
    col_data = selected_data.Data;
    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'Fib'});
    Fib = selected_data.Data;
    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'CAF'});
    CAF = selected_data.Data;

    % Plot collagen, CAFs, and Fibroblasts directly
    plot(time_data, col_data, 'Color', [0.7, 0.5, 0.3], 'LineWidth', 2);
    plot(time_data, Fib, 'Color', [0.5, 0.7, 0.3], 'LineWidth', 2);
    plot(time_data, CAF, 'Color', [0.3, 0.5, 0.7], 'LineWidth', 2);

    xlabel(sprintf('Time (%s)', time_units));
    ylabel('Count');
    set(gca, 'YScale', 'log');
    legend('Collagen', 'Fibroblasts', 'CAFs', 'Location', 'best');
    title('Fibrotic Tissue Composition');
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
end

%% Plot 2: Immune Cell Dynamics
if ismember('immune', plot_selection)
    fig2 = figure('Position', [150, 150, 1400, 900]);
    fig_handles(end+1) = fig2;
    
    if ~isempty(patient_id)
        sgtitle(sprintf('Immune Cell Dynamics - Patient %s', patient_id), 'FontSize', 16, 'FontWeight', 'bold');
    else
        sgtitle('Immune Cell Dynamics', 'FontSize', 16, 'FontWeight', 'bold');
    end
    
    % T effector cells in tumor
    subplot(2,4,1); hold on; box on;    colors = lines(5);
    legend_entries = {};
    
    % Plot CD8+ T cells
    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'CD8'});
    CD8_data = selected_data.Data;
    semilogy(time_data, CD8_data, 'Color', colors(1,:), 'LineWidth', 2);
    legend_entries{end+1} = 'CD8+ T cells';
    
    % Plot Treg
    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'Treg'});
    Treg_data = selected_data.Data;
    semilogy(time_data, Treg_data, 'Color', colors(2,:), 'LineWidth', 2);
    legend_entries{end+1} = 'Treg';
    
    % Add exhausted T cells
    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'CD8_exh'});
    T1_exh = selected_data.Data;
    semilogy(time_data, T1_exh, '--', 'Color', colors(end-1,:), 'LineWidth', 2);
    legend_entries{end+1} = 'CD8 exhausted';

    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'Th_exh'});
    Th_exh = selected_data.Data;
    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'Th'});
    Th = selected_data.Data;
    semilogy(time_data, Th_exh, '--', 'Color', colors(end,:), 'LineWidth', 2);
    semilogy(time_data, Th, 'Color', colors(end,:), 'LineWidth', 2);
    legend_entries{end+1} = 'Th exhausted';
    legend_entries{end+1} = 'Th active';
    
    xlabel(sprintf('Time (%s)', time_units));
    ylabel('Cell Count');
    set(gca, 'YScale', 'log');
    title('T Effector Cells (Tumor)');
    legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    clear legend_entries;
    
    % T effector cells in lymph node
    subplot(2,4,2); hold on; box on;
    legend_entries = {};
    
    % Plot CD8+ T cells in LN
    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_LN', 'Name', 'CD8'});
    CD8_LN = selected_data.Data;
    semilogy(time_data, CD8_LN, 'Color', colors(1,:), 'LineWidth', 2);
    legend_entries{end+1} = 'CD8+ LN';
    
    % Plot Treg in LN
    selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_LN', 'Name', 'Treg'});
    Treg_LN = selected_data.Data;
    semilogy(time_data, Treg_LN, 'Color', colors(2,:), 'LineWidth', 2);
    legend_entries{end+1} = 'Treg LN';
    xlabel(sprintf('Time (%s)', time_units));
    ylabel('Cell Count');
    set(gca, 'YScale', 'log');
    title('T Effector Cells (Lymph Node)');
    legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    clear legend_entries;
    
    % Regulatory T cells across compartments
    subplot(2,4,3); hold on; box on;
    compartments = {'V_T', 'V_C', 'V_LN', 'V_P'};
    comp_names = {'Tumor', 'Central', 'LN', 'Peripheral'};
    comp_colors = [0.8, 0.2, 0.2; 0.2, 0.8, 0.2; 0.2, 0.2, 0.8; 0.8, 0.8, 0.2];
    
    for i = 1:length(compartments)
        try
            selected_data = select(simData, {'Type', 'species', 'Compartment', compartments{i}, 'Name', 'Treg'});
            Treg = selected_data.Data;
            semilogy(time_data, Treg, 'Color', comp_colors(i,:), 'LineWidth', 2);
            legend_entries{i} = sprintf('Treg %s', comp_names{i});
        catch
            legend_entries{i} = sprintf('Treg %s (N/A)', comp_names{i});
        end
    end
    xlabel(sprintf('Time (%s)', time_units));
    ylabel('Cell Count');
    set(gca, 'YScale', 'log');
    title('Regulatory T Cells');
    legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    clear legend_entries;
    
    % T cell ratios and clonality
    subplot(2,4,4); hold on; box on;
    try
        % Calculate Teff:Treg ratio in tumor
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'CD8'}); % CD8+ T effector cells
        T_total = selected_data.Data;
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'Treg'});
        Treg_T = selected_data.Data;
        ratio = T_total ./ (Treg_T + 1e-10); % Avoid division by zero
        
        semilogy(time_data, ratio, 'Color', [0.8, 0.4, 0.2], 'LineWidth', 2);
        ylabel('Teff:Treg Ratio');
        set(gca, 'YScale', 'log');
        title('T Cell Ratio (Tumor)');
        grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    catch
        text(0.5, 0.5, 'Ratio data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    xlabel(sprintf('Time (%s)', time_units));
    
    % APC populations
    subplot(2,4,5); hold on; box on;
    compartments = {'V_T', 'V_LN'};
    comp_names = {'Tumor', 'LN'};

    for i = 1:length(compartments)
        try
            % Sum DC subtypes for each compartment
            dc1 = select(simData, {'Type', 'species', 'Compartment', compartments{i}, 'Name', 'cDC1'});
            dc2 = select(simData, {'Type', 'species', 'Compartment', compartments{i}, 'Name', 'cDC2'});
            APC = dc1.Data + dc2.Data;
            mdc1 = select(simData, {'Type', 'species', 'Compartment', compartments{i}, 'Name', 'mcDC1'});
            mdc2 = select(simData, {'Type', 'species', 'Compartment', compartments{i}, 'Name', 'mcDC2'});
            mAPC = mdc1.Data + mdc2.Data;

            semilogy(time_data, APC, '-', 'Color', comp_colors(i,:), 'LineWidth', 2);
            semilogy(time_data, mAPC, '--', 'Color', comp_colors(i,:), 'LineWidth', 2);

            legend_entries{2*i-1} = sprintf('iDC %s', comp_names{i});
            legend_entries{2*i} = sprintf('mDC %s', comp_names{i});
        catch, end
    end
    xlabel(sprintf('Time (%s)', time_units));
    ylabel('Cell Count');
    set(gca, 'YScale', 'log');
    title('Dendritic Cells (cDC1+cDC2)');
    legend(legend_entries, 'Location', 'best', 'FontSize', 8);
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    clear legend_entries;
    
    % IL-2 concentration
    subplot(2,4,6); hold on; box on;
    try
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_LN', 'Name', 'IL2'});
        IL2 = selected_data.Data;
        semilogy(time_data, IL2, 'Color', [0.6, 0.2, 0.8], 'LineWidth', 2);
        xlabel(sprintf('Time (%s)', time_units));
        ylabel('IL-2 Concentration (M)');
        set(gca, 'YScale', 'log');
        title('IL-2 in Lymph Node');
        grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    catch
        text(0.5, 0.5, 'IL-2 data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    
    % T cell densities in tumor
    subplot(2,4,7); hold on; box on;
    try
        selected_data = select(simData, {'Type', 'compartment', 'Name', 'V_T'});
        V_T = selected_data.Data;
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'CD8'});
        CD8 = selected_data.Data;
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'Treg'});
        Treg = selected_data.Data;
        
        CD8_density = CD8 ./ V_T / 1e3; % cells per mm³
        Treg_density = Treg ./ V_T / 1e3;
        
        semilogy(time_data, CD8_density, 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2);
        semilogy(time_data, Treg_density, 'Color', [0.2, 0.2, 0.8], 'LineWidth', 2);
        
        xlabel(sprintf('Time (%s)', time_units));
        ylabel('Cell Density (cells/mm³)');
        set(gca, 'YScale', 'log');
        title('T Cell Density in Tumor');
        legend('CD8+ Teff', 'Treg', 'Location', 'best');
        grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    catch
        text(0.5, 0.5, 'Density data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    
    % Immune activation score
    subplot(2,4,8); hold on; box on;
    try
        % Simple activation score based on Teff density and APC activation
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'CD8'});
        CD8 = selected_data.Data;
        mdc1_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'mcDC1'});
        mdc2_data = select(simData, {'Type', 'species', 'Compartment', 'V_T', 'Name', 'mcDC2'});
        mAPC_T = mdc1_data.Data + mdc2_data.Data;
        selected_data = select(simData, {'Type', 'compartment', 'Name', 'V_T'});
    V_T = selected_data.Data;
        
        activation_score = (CD8 ./ V_T) .* (mAPC_T ./ V_T) / 1e6;
        semilogy(time_data, activation_score, 'Color', [0.2, 0.8, 0.4], 'LineWidth', 2);
        
        xlabel(sprintf('Time (%s)', time_units));
        ylabel('Activation Score (a.u.)');
        set(gca, 'YScale', 'log');
        title('Immune Activation Score');
        grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    catch
        text(0.5, 0.5, 'Activation score unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
end

%% Plot 3: Checkpoint and Drug Dynamics
if ismember('checkpoints', plot_selection)
    fig3 = figure('Position', [200, 200, 1200, 800]);
    fig_handles(end+1) = fig3;
    
    if ~isempty(patient_id)
        sgtitle(sprintf('Checkpoint & Drug Dynamics - Patient %s', patient_id), 'FontSize', 16, 'FontWeight', 'bold');
    else
        sgtitle('Checkpoint & Drug Dynamics', 'FontSize', 16, 'FontWeight', 'bold');
    end
    
    % Drug concentrations
    subplot(2,3,1); hold on; box on;
    drugs = {'aPD1', 'aPDL1', 'aCTLA4', 'cabozantinib', 'NabP'};
    drug_colors = [0.8, 0.2, 0.2; 0.2, 0.8, 0.2; 0.2, 0.2, 0.8; 0.8, 0.4, 0.2; 0.6, 0.2, 0.8];
    
    for i = 1:length(drugs)
        try
            % Different drugs are in different compartments
            if strcmp(drugs{i}, 'NabP')
                % Nab-paclitaxel is in V_1 compartment
                selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_1', 'Name', drugs{i}});
            else
                % Immunotherapies and cabozantinib are in V_C compartment
                selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_C', 'Name', drugs{i}});
            end
            drug_data = selected_data.Data;
            semilogy(time_data, drug_data, 'Color', drug_colors(i,:), 'LineWidth', 2);
            legend_entries{i} = drugs{i};
        catch
            legend_entries{i} = sprintf('%s (N/A)', drugs{i});
        end
    end
    xlabel(sprintf('Time (%s)', time_units));
    ylabel('Drug Concentration (M)');
    title('Systemic Drug Concentrations');
    legend(legend_entries, 'Location', 'best');
    grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    clear legend_entries;
    
    % PD-1/PD-L1 interactions in synapse
    subplot(2,3,2); hold on; box on;
    try
        A_syn = sbioselect(model, 'name', 'A_syn');
        interactions = {'PD1_PDL1', 'PD1_PDL2', 'PD1_aPD1'};
        colors = [0.8, 0.2, 0.2; 0.2, 0.8, 0.2; 0.2, 0.2, 0.8];
        
        for i = 1:length(interactions)
            try
                selected_data = select(simData, {'Type', 'species', 'Compartment', 'syn_T_C1', 'Name', interactions{i}});
                complex_data = selected_data.Data;
                molecules = complex_data * A_syn.Value;
                semilogy(time_data, molecules, 'Color', colors(i,:), 'LineWidth', 2);
                legend_entries{i} = strrep(interactions{i}, '_', '-');
            catch
                legend_entries{i} = sprintf('%s (N/A)', strrep(interactions{i}, '_', '-'));
            end
        end
        xlabel(sprintf('Time (%s)', time_units));
        ylabel('Molecules in Synapse');
        title('PD-1 Pathway Interactions');
        legend(legend_entries, 'Location', 'best');
        grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    catch
        text(0.5, 0.5, 'Synapse data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    clear legend_entries;
    
    % CTLA-4 interactions
    subplot(2,3,3); hold on; box on;
    try
        interactions = {'CD28_CD80', 'CD28_CD86', 'CD80_CTLA4', 'CD86_CTLA4', 'CTLA4_aCTLA4'};
        colors = lines(length(interactions));
        
        for i = 1:length(interactions)
            try
                selected_data = select(simData, {'Type', 'species', 'Compartment', 'syn_T_C1', 'Name', interactions{i}});
                complex_data = selected_data.Data;
                molecules = complex_data * A_syn.Value;
                semilogy(time_data, molecules, 'Color', colors(i,:), 'LineWidth', 2);
                legend_entries{i} = strrep(interactions{i}, '_', '-');
            catch
                legend_entries{i} = sprintf('%s (N/A)', strrep(interactions{i}, '_', '-'));
            end
        end
        xlabel(sprintf('Time (%s)', time_units));
        ylabel('Molecules in Synapse');
        title('CTLA-4 Pathway Interactions');
        legend(legend_entries, 'Location', 'best', 'FontSize', 8);
        grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    catch
        text(0.5, 0.5, 'CTLA-4 data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    clear legend_entries;
    
    % Checkpoint inhibition efficiency
    subplot(2,3,4); hold on; box on;
    try
        % Calculate inhibition as fraction of bound receptors
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'syn_T_C1', 'Name', 'PD1_PDL1'});
        PD1_total = selected_data.Data;
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'syn_T_C1', 'Name', 'PD1_aPD1'});
        PD1_blocked = selected_data.Data;
        
        inhibition_eff = PD1_blocked ./ (PD1_total + PD1_blocked + 1e-10) * 100;
        plot(time_data, inhibition_eff, 'Color', [0.8, 0.2, 0.4], 'LineWidth', 2);
        
        xlabel(sprintf('Time (%s)', time_units));
        ylabel('PD-1 Inhibition (%)');
        title('Checkpoint Inhibition Efficiency');
        ylim([0, 100]);
        grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    catch
        text(0.5, 0.5, 'Inhibition data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    
    % Drug pharmacokinetics
    subplot(2,3,5); hold on; box on;
    try
        selected_data = select(simData, {'Type', 'species', 'Compartment', 'V_C', 'Name', 'aPD1'});
        aPD1 = selected_data.Data;
        if ~isempty(find(aPD1 > 0, 1))
            % Plot on both linear and log scale
            yyaxis left
            plot(time_data, aPD1, 'Color', [0.8, 0.2, 0.2], 'LineWidth', 2);
            ylabel('aPD1 Concentration (M)');
            set(gca, 'YColor', [0.8, 0.2, 0.2]);
            
            yyaxis right
            semilogy(time_data, aPD1, '--', 'Color', [0.4, 0.1, 0.1], 'LineWidth', 1);
            ylabel('aPD1 Log Scale (M)');
            set(gca, 'YColor', [0.4, 0.1, 0.1]);
        end
        xlabel(sprintf('Time (%s)', time_units));
        title('Anti-PD1 Pharmacokinetics');
        grid on; 
    try
        grid(gca, 'alpha', 0.3);
    catch
        % If grid alpha fails, just use basic grid
    end
    catch
        text(0.5, 0.5, 'PK data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
    
    % Checkpoint expression levels
    subplot(2,3,6); hold on; box on;
    try
        checkpoints = {'PD1', 'PDL1', 'CTLA4'};
        colors = [0.8, 0.2, 0.2; 0.2, 0.8, 0.2; 0.2, 0.2, 0.8];
        
        for i = 1:length(checkpoints)
            try
                % This is a simplified example - actual checkpoint expression
                % would need to be extracted differently based on model structure
                text(0.5, 0.8-i*0.1, sprintf('%s expression data needed', checkpoints{i}), ...
                    'Units', 'normalized');
            catch, end
        end
        
        xlabel(sprintf('Time (%s)', time_units));
        ylabel('Expression Level');
        title('Checkpoint Expression');
    catch
        text(0.5, 0.5, 'Expression data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    end
end

%% Save plots if requested
if save_plots
    timestamp = datestr(now, 'yyyymmdd_HHMMSS');
    for i = 1:length(fig_handles)
        if ~isempty(patient_id)
            filename = sprintf('qsp_comprehensive_%s_fig%d_%s.png', patient_id, i, timestamp);
        else
            filename = sprintf('qsp_comprehensive_fig%d_%s.png', i, timestamp);
        end
        saveas(fig_handles(i), filename, 'png');
        fprintf('Saved: %s\n', filename);
    end
end

end

%% Helper Functions
function cancer_clones = findCancerClones(dataObj)
    % Find all cancer clone species (C1, C2, C3, etc.) in simulation data
    try
        data_names = dataObj.DataNames;
    catch
        % If DataNames doesn't exist, try extracting from Species
        data_names = cell(length(dataObj.Species),1);
        for i = 1:length(dataObj.Species)
            data_names{i} = dataObj.Species(i).Name;
        end
    end
    
    % Find species that match pattern C followed by digits
    cancer_clones = {};
    for i = 1:length(data_names)
        if regexp(data_names{i}, '^C\d+$')
            cancer_clones{end+1} = data_names{i};
        end
    end
    
    % Sort clones numerically (C1, C2, C3, etc.)
    if ~isempty(cancer_clones)
        clone_numbers = cellfun(@(x) str2double(x(2:end)), cancer_clones);
        [~, sort_idx] = sort(clone_numbers);
        cancer_clones = cancer_clones(sort_idx);
    end
end
function simbio_plot_time(simData, species_name, time_data, varargin)
    % Enhanced simbio_plot with time conversion support
    try
        % Parse species name to extract compartment and name
        if contains(species_name, '.')
            parts = split(species_name, '.');
            selected_data = select(simData, {'Type', 'species', 'Compartment', parts{1}, 'Name', parts{2}});
        else
            selected_data = select(simData, {'Type', 'species', 'Name', species_name});
        end
        data = selected_data.Data;
        plot(time_data, data, varargin{:});
    catch
        warning('Could not plot species: %s', species_name);
    end
end
