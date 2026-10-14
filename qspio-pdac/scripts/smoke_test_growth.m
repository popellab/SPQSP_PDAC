% Smoke test: evolve to diagnosis with median parameters and plot dynamics
% Verifies that Gompertzian growth + structural changes produce plausible trajectories
cd('/Users/joeleliason/Projects/pdac-build');
startup;

%% Build model
fprintf('=== Building model ===\n');
immune_oncology_model_PDAC;

%% Apply median prior values
fprintf('=== Applying median prior values ===\n');
priors = readtable('parameters/pdac_priors.csv');
for i = 1:height(priors)
    pname = priors.name{i};
    mval = priors.median(i);  % Use median directly (CSV stores median, not log-median)
    pobj = sbioselect(model, 'Name', pname, 'Type', 'parameter');
    if ~isempty(pobj)
        set(pobj, 'Value', mval);
        fprintf('  %-25s = %.4e\n', pname, mval);
    else
        fprintf('  %-25s = NOT FOUND\n', pname);
    end
end

%% Report key parameters
k_g = sbioselect(model, 'Name', 'k_C1_growth', 'Type', 'parameter');
k_d = sbioselect(model, 'Name', 'k_C1_death', 'Type', 'parameter');
fprintf('\n=== Key Growth Parameters ===\n');
fprintf('k_C1_growth = %.6f /day\n', k_g.Value);
fprintf('k_C1_death  = %.6f /day\n', k_d.Value);
fprintf('Growth model: Gompertzian\n');
fprintf('  At C/Cmax=0.01: effective = %.4f * log(100) - %.4f = %.4f /day (VDT %.0f days)\n', ...
    k_g.Value, k_d.Value, k_g.Value*log(100)-k_d.Value, log(2)/(k_g.Value*log(100)-k_d.Value));
fprintf('  At C/Cmax=0.50: effective = %.4f * log(2) - %.4f = %.4f /day (VDT %.0f days)\n', ...
    k_g.Value, k_d.Value, k_g.Value*log(2)-k_d.Value, log(2)/(k_g.Value*log(2)-k_d.Value));

%% Run evolve_to_diagnosis
fprintf('\n=== Running evolve_to_diagnosis ===\n');
[model_out, success, simData] = evolve_to_diagnosis(model, ...
    'TargetDiameter', 3.0, 'Debug', true);

if ~success
    fprintf('\n*** EVOLUTION FAILED — tumor did not reach target diameter ***\n');
    if isempty(simData)
        fprintf('No simulation data returned (solver error).\n');
        return
    end
end

%% Extract trajectories (truncated at diagnosis)
time_full = simData.Time;
names = simData.DataNames;

C1_idx = flex_find(names, 'C1');
K_idx = flex_find(names, 'K');
VT_idx = flex_find(names, 'V_T');

% Compute diameter to find diagnosis timepoint
V_T_full = simData.Data(:, VT_idx);
diam_full = (6.0 * V_T_full / pi).^(1/3);
diag_idx = find(diam_full >= 3.0, 1, 'first');
if isempty(diag_idx)
    diag_idx = length(time_full);  % fallback to end
end

% Add 10% margin past diagnosis for context, but cap at end
plot_end = min(round(diag_idx * 1.1), length(time_full));
tr = 1:plot_end;  % truncation range
diag_day = time_full(diag_idx);

time = time_full(tr);
C1 = simData.Data(tr, C1_idx);
K = simData.Data(tr, K_idx);
V_T_traj = simData.Data(tr, VT_idx);
diam_cm = (6.0 * V_T_traj / pi).^(1/3);

% Immune cells
species_to_plot = {'CD8', 'Mac_M1', 'Mac_M2', 'Treg', ...
                   'iCAF', 'myCAF', 'NK', 'MDSC'};

% Metabolic
glc_idx = flex_find(names, 'glc');
lac_idx = flex_find(names, 'lactate');
HIF_idx = flex_find(names, 'HIF');

% C_max (algebraic parameter)
Cmax_idx = flex_find(names, 'C_max');

%% Plot 1: Tumor growth and carrying capacity
figure('Position', [100 100 1200 800], 'Name', 'Smoke Test: Tumor Growth Dynamics');

subplot(2,3,1);
semilogy(time, C1, 'r-', 'LineWidth', 2); hold on;
semilogy(time, K, 'b--', 'LineWidth', 1.5);
if ~isempty(Cmax_idx)
    semilogy(time, simData.Data(tr, Cmax_idx), 'g:', 'LineWidth', 1.5);
    legend('C1 (cancer)', 'K (vasculature)', 'C_{max} (carrying cap.)');
else
    legend('C1 (cancer)', 'K (vasculature)');
end
xline(diag_day, 'k--', 'Diagnosis', 'FontSize', 7);
xlabel('Time (days)'); ylabel('Cells');
title('Tumor Growth (Gompertzian)');
grid on;

subplot(2,3,2);
plot(time, diam_cm, 'r-', 'LineWidth', 2); hold on;
yline(3.0, 'k--', 'Target 3 cm');
yline(0.17, 'b:', 'Start ~1.7 mm');
xline(diag_day, 'k--');
xlabel('Time (days)'); ylabel('Diameter (cm)');
title('Tumor Diameter');
grid on;

%% Plot 2: Immune cell populations
subplot(2,3,3);
colors = lines(8);
legend_entries = {};
ci = 1;
for j = 1:length(species_to_plot)
    idx = flex_find(names, species_to_plot{j});
    if ~isempty(idx)
        semilogy(time, max(simData.Data(tr, idx), 1), '-', 'Color', colors(ci,:), 'LineWidth', 1.5);
        hold on;
        legend_entries{end+1} = species_to_plot{j};
        ci = ci + 1;
    end
end
xline(diag_day, 'k--', 'Diagnosis', 'FontSize', 7);
xlabel('Time (days)'); ylabel('Cells');
title('Immune/Stromal Populations');
legend(legend_entries, 'Location', 'best', 'FontSize', 7);
grid on;

%% Plot 3: Metabolic environment
subplot(2,3,4);
if ~isempty(glc_idx)
    yyaxis left
    plot(time, simData.Data(tr, glc_idx), 'b-', 'LineWidth', 2);
    ylabel('Glucose (\muM)');
end
if ~isempty(lac_idx)
    yyaxis right
    plot(time, simData.Data(tr, lac_idx), 'r-', 'LineWidth', 2);
    ylabel('Lactate (\muM)');
end
xline(diag_day, 'k--');
xlabel('Time (days)');
title('Metabolic Environment');
grid on;

%% Plot 4: HIF and pO2
subplot(2,3,5);
pO2_idx = flex_find(names, 'pO2');
if ~isempty(pO2_idx) && ~isempty(HIF_idx)
    yyaxis left
    plot(time, simData.Data(tr, pO2_idx), 'b-', 'LineWidth', 2);
    ylabel('pO2 (mmHg)');
    yyaxis right
    plot(time, simData.Data(tr, HIF_idx), 'r-', 'LineWidth', 2);
    ylabel('HIF (dimensionless)');
end
xline(diag_day, 'k--');
xlabel('Time (days)');
title('Hypoxia');
grid on;

%% Plot 5: Stromal dynamics (collagen)
subplot(2,3,6);
phi_col_idx = flex_find(names, 'phi_collagen');
col_idx = flex_find(names, 'collagen');
if ~isempty(phi_col_idx)
    plot(time, simData.Data(tr, phi_col_idx), 'k-', 'LineWidth', 2);
    ylabel('\phi_{collagen}');
elseif ~isempty(col_idx)
    plot(time, simData.Data(tr, col_idx), 'k-', 'LineWidth', 2);
    ylabel('Collagen (mg)');
end
xline(diag_day, 'k--');
xlabel('Time (days)');
title('Collagen Volume Fraction');
grid on;

sgtitle(sprintf('Smoke Test: Gompertzian Growth (success=%d, diagnosis day %.0f)', success, diag_day));

%% Print summary AT DIAGNOSIS
fprintf('\n=== SUMMARY AT DIAGNOSIS (day %.0f) ===\n', diag_day);
fprintf('Evolution success: %d\n', success);
fprintf('C1: %.2e cells\n', simData.Data(diag_idx, C1_idx));
fprintf('K:  %.2e cells\n', simData.Data(diag_idx, K_idx));
if ~isempty(Cmax_idx)
    fprintf('C_max: %.2e cells\n', simData.Data(diag_idx, Cmax_idx));
end
fprintf('Tumor volume: %.1f mL\n', V_T_full(diag_idx));
fprintf('Tumor diameter: %.2f cm\n', diam_full(diag_idx));

if ~isempty(glc_idx)
    fprintf('Glucose: %.0f \x00B5M (normal: 1225 \x00B5M)\n', simData.Data(diag_idx, glc_idx));
end
if ~isempty(lac_idx)
    fprintf('Lactate: %.0f \x00B5M\n', simData.Data(diag_idx, lac_idx));
end
if ~isempty(pO2_idx)
    fprintf('pO2: %.1f mmHg\n', simData.Data(diag_idx, pO2_idx));
end
if ~isempty(HIF_idx)
    fprintf('HIF: %.3f\n', simData.Data(diag_idx, HIF_idx));
end

% Immune densities at diagnosis (cells/mm³)
V_T_diag = V_T_full(diag_idx);  % mL
V_T_mm3 = V_T_diag * 1000; % mm³
fprintf('\n=== Immune Densities at Diagnosis (cells/mm\x00B3) ===\n');
for j = 1:length(species_to_plot)
    idx = flex_find(names, species_to_plot{j});
    if ~isempty(idx)
        count = simData.Data(diag_idx, idx);
        density = count / V_T_mm3;
        fprintf('  %-12s: %8.0f cells/mm\x00B3 (%10.2e total)\n', species_to_plot{j}, density, count);
    end
end

fprintf('\nSave figure: saveas(gcf, ''results/smoke_test_growth.png'')\n');

%% Local functions
function idx = flex_find(names, name)
% Find column index in DataNames, trying name as-is, with V_T. prefix, or stripped
    idx = find(strcmp(names, name), 1);
    if isempty(idx)
        idx = find(strcmp(names, ['V_T.' name]), 1);
    end
    if isempty(idx) && contains(name, '.')
        idx = find(strcmp(names, regexprep(name, '^V_T\.', '')), 1);
    end
end