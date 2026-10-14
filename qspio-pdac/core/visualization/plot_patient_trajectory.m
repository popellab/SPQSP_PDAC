function fig = plot_patient_trajectory(results, patient_idx, model)
%PLOT_PATIENT_TRAJECTORY Simple trajectory plotting for individual patients
%
% Creates a clean, comprehensive plot of key simulation trajectories for
% a single patient from PSA results.
%
% Inputs:
%   results     - PSA results cell array
%   patient_idx - Index of patient to plot
%   model       - SimBiology model (optional, for checkpoint data)
%
% Output:
%   fig - Figure handle
%
% Example:
%   load('PSA_results_ssh_simple.mat');
%   fig = plot_patient_trajectory(results, 15, model);

% Handle different result structures
if patient_idx < 1 || patient_idx > length(results)
    error('Patient index %d out of range (1-%d)', patient_idx, length(results));
end

if iscell(results)
    simData = results{patient_idx};
elseif isstruct(results) && isfield(results, 'simData')
    % Handle structure array with simData field
    simData = results(patient_idx).simData;
else
    % Handle array of simulation data directly
    simData = results(patient_idx);
end

if isempty(simData)
    error('No data for patient %d', patient_idx);
end

% Create figure
fig = figure('Position', [100, 100, 1200, 800]);
sgtitle(sprintf('Patient %d Trajectory', patient_idx), 'FontSize', 16, 'FontWeight', 'bold');

time_weeks = simData.time / 7;

%% Tumor Growth
subplot(2,3,1); hold on; box on;
try
    [~, C1, ~] = selectbyname(simData, 'C1');
    [~, C2, ~] = selectbyname(simData, 'C2');
    semilogy(time_weeks, C1, 'r-', 'LineWidth', 2);
    semilogy(time_weeks, C2, 'g-', 'LineWidth', 2);
    xlabel('Time (weeks)');
    ylabel('Cancer Cells');
    title('Cancer Cell Growth');
    legend('Chemo\_sens', 'Chemo\_res', 'Location', 'best');
    grid on;
catch
    text(0.5, 0.5, 'Cancer cell data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
end

%% Tumor Volume
subplot(2,3,2); hold on; box on;
try
    [~, V_T, ~] = selectbyname(simData, 'V_T');
    
    % Plot volume
    yyaxis left
    semilogy(time_weeks, V_T, 'b-', 'LineWidth', 2);
    ylabel('Tumor Volume (L)');
    
    % Plot diameter
    yyaxis right
    D_T = 2 * (3/(4*pi)*V_T).^(1/3) * 10; % cm
    plot(time_weeks, D_T, 'r--', 'LineWidth', 2);
    ylabel('Diameter (cm)');
    
    xlabel('Time (weeks)');
    title('Tumor Size');
    grid on;
catch
    text(0.5, 0.5, 'Tumor volume data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
end

%% RECIST Response
subplot(2,3,3); hold on; box on;
try
    [~, V_T, ~] = selectbyname(simData, 'V_T');
    baseline = V_T(1);
    percent_change = ((V_T - baseline) / baseline) * 100;
    
    plot(time_weeks, percent_change, 'b-', 'LineWidth', 2);
    yline(20, 'r--', 'PD', 'LineWidth', 1.5);
    yline(-30, 'g--', 'PR', 'LineWidth', 1.5);
    yline(0, 'k:', 'Baseline', 'LineWidth', 1);
    
    xlabel('Time (weeks)');
    ylabel('Change from Baseline (%)');
    title('RECIST Response');
    ylim([-100, 200]);
    grid on;
catch
    text(0.5, 0.5, 'RECIST data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
end

%% T Cells in Tumor
subplot(2,3,4); hold on; box on;
try
    % Try flexible T cell species matching
    data_names = simData.DataNames;
    
    % Find T cells using exact species names
    T1_idx = [];
    T0_idx = [];
    
    % Look for CD8 and Treg
    for i = 1:length(simData.DataInfo)
        info = simData.DataInfo{i};

        if strcmp(info.Type, 'species') && strcmp(info.Compartment, 'V_T')
            if strcmp(info.Name, 'CD8')
                T1_idx = i;
            elseif strcmp(info.Name, 'Treg')
                T0_idx = i;
            end
        end
    end
    
    % Extract data if found
    T1 = [];
    T0 = [];
    if ~isempty(T1_idx)
        T1 = simData.Data(:, T1_idx);
    end
    if ~isempty(T0_idx)
        T0 = simData.Data(:, T0_idx);
    end
    
    if ~isempty(T1) && ~isempty(T0)
        semilogy(time_weeks, T1, 'r-', 'LineWidth', 2);
        semilogy(time_weeks, T0, 'b-', 'LineWidth', 2);
        
        xlabel('Time (weeks)');
        ylabel('Cell Count');
        title('T Cells in Tumor');
        legend('CD8', 'Treg', 'Location', 'best');
        grid on;
    else
        if isempty(T1)
            error('CD8 T cells not found in tumor compartment (V_T)');
        end
        if isempty(T0)
            error('Treg cells not found in tumor compartment (V_T)');
        end
    end
catch
    rethrow(lasterror);
end

%% T Cell Ratio
subplot(2,3,5); hold on; box on;
try
    % Use the same T1 and T0 from above if they were found
    if exist('T1', 'var') && exist('T0', 'var') && ~isempty(T1) && ~isempty(T0)
        ratio = T1 ./ (T0 + 1e-10);
        semilogy(time_weeks, ratio, 'g-', 'LineWidth', 2);
        
        xlabel('Time (weeks)');
        ylabel('CD8:Treg Ratio');
        title('T Cell Balance');
        grid on;
    else
        error('T cell ratio unavailable - missing CD8 or Treg data from tumor compartment');
    end
catch
    rethrow(lasterror);
end

%% Drug Concentration
subplot(2,3,6); hold on; box on;
try
    % Try different drug concentration patterns - suppress warnings during search
    drug_patterns = {'V_C.aPD1', 'Central.aPD1', 'nivolumab', 'Central.nivolumab', 'aPD1'};
    drug_data = [];
    drug_name = 'Drug';
    
    warning('off', 'SimBiology:SimData:InvalidDataName');
    for pattern = drug_patterns
        try
            [~, drug_data, ~] = selectbyname(simData, pattern{1});
            if ~isempty(drug_data) && any(drug_data > 0)
                if contains(pattern{1}, 'nivolumab', 'IgnoreCase', true)
                    drug_name = 'Nivolumab';
                elseif contains(pattern{1}, 'aPD1', 'IgnoreCase', true)
                    drug_name = 'Anti-PD1';
                end
                break;
            end
        catch
            continue;
        end
    end
    warning('on', 'SimBiology:SimData:InvalidDataName');
    
    if ~isempty(drug_data) && any(drug_data > 0)
        semilogy(time_weeks, drug_data, 'm-', 'LineWidth', 2);
        xlabel('Time (weeks)');
        ylabel('Concentration (M)');
        title(drug_name);
        grid on;
    else
        text(0.5, 0.5, 'No drug administered', 'Units', 'normalized', 'HorizontalAlignment', 'center');
        title('Drug Concentration');
    end
catch
    text(0.5, 0.5, 'Drug data unavailable', 'Units', 'normalized', 'HorizontalAlignment', 'center');
    title('Drug Concentration');
end

end
