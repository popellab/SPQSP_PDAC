% Generate cascade figures for presentation
%
% This script generates:
%   1. cascade_summary_gvax_nivo.png - 6-panel summary for main presentation
%   2. 8 detailed cascade figures for backup slides

%% Run simulations (both 90 days for full treatment effect trajectory)
sim_time = 0:0.5:90;
fprintf('Running simulations (0-90 days)...\n');
[simBase, ~, ~] = run_median_simulation('scenarios/clinical_progression.yaml', 'sim_time', sim_time);
[simTx, ~, scenario] = run_median_simulation('scenarios/gvax_nivo_neoadjuvant_zheng2022.yaml', 'sim_time', sim_time);

%% Generate cascade summary (6-panel)
fprintf('Generating cascade summary...\n');
fig = plot_cascade_summary(simBase, simTx, scenario.scenario_id);
saveas(fig, 'figures/cascade_summary_gvax_nivo.png');
fprintf('  Saved: figures/cascade_summary_gvax_nivo.png\n');

%% Generate full mechanism cascade (8 figures)
fprintf('Generating full mechanism cascade...\n');
figs = plot_mechanism_cascade(simBase, simTx, scenario.scenario_id);

% Save with names expected by LaTeX presentation
fig_names = {'cascade_input_apc', 'cascade_tcell_activation', ...
             'cascade_tcell_killing', 'cascade_checkpoint', ...
             'cascade_caf_subtypes', 'cascade_chemokine_trafficking'};

for i = 1:length(figs)
    filename = sprintf('figures/%s.png', fig_names{i});
    saveas(figs(i), filename);
    fprintf('  Saved: %s\n', filename);
end

fprintf('\nAll cascade figures generated successfully.\n');
fprintf('Main slide: figures/cascade_summary_gvax_nivo.png\n');
fprintf('Backup slides: figures/cascade_*.png\n');

%% Dump time-series data at 10-day intervals for review
fprintf('\nDumping time-series data...\n');
dump_times = 0:10:90;

% Species in various compartments
dump_species = {
    % Fig 1: Treatment input & APC
    'GVAX_cells', 'V_ID';
    'GMCSF',      'V_ID';
    'P1_GVAX',    'V_ID';
    'P1',         'V_T';
    'APC',        'V_ID';
    'mAPC',       'V_ID';
    'mcDC2',      'V_LN';
    % Fig 2: T cell activation
    'mcDC1',      'V_LN';
    'nCD8',       'V_LN';
    'aCD8',       'V_LN';
    'CD8',        'V_LN';
    'CD8',        'V_C';
    'CD8',        'V_T';
    'IFNg',       'V_T';
    % Fig 3: Tumor killing
    'collagen',   'V_T';
    'Treg',       'V_T';
    'C1',         'V_T';
    % Fig 4: Checkpoint & immunosuppression
    'PD1_CD8',    'V_T';
    'PDL1_C1',    'V_T';
    'PD1_PDL1_C1','V_T';
    'nivolumab',  'V_C';
    'Mac_M1',     'V_T';
    'Mac_M2',     'V_T';
    'MDSC',       'V_T';
    % Fig 5: CAF subtypes
    'qPSC',       'V_T';
    'myCAF',      'V_T';
    'iCAF',       'V_T';
    'apCAF',      'V_T';
    % Fig 6: Chemokines
    'CCL5',       'V_T';
    'CXCL12',     'V_T';
};

% Observables (repeatedAssignment parameters)
dump_observables = {
    'H_APC_Treg', 'phi_collagen', ...
    'H_pore_T', 'H_pore_APC', 'H_stiff_Texh', 'H_stiff_fib', ...
    'k_C_Tcell_eff', 'C_total', 'V_T', 'T_total', 'aPSC_total', ...
    'H_TGFb', 'H_TGFb_Teff', 'H_TGFb_APC', ...
    'H_IL10', 'H_IL12', ...
    'H_IL6_iCAF', 'H_IL6_MDSC', 'H_IL6_M2', ...
    'H_CCL5_Treg', ...
    'H_CXCL12_Texcl', ...
    'H_NO', 'H_ArgI_Teff', 'H_DAMP', 'H_P1_apCAF', ...
};

% Build JSON output
fid = fopen('figures/cascade_timeseries.json', 'w');
fprintf(fid, '{\n');
fprintf(fid, '  "times": [%s],\n', strjoin(arrayfun(@num2str, dump_times, 'UniformOutput', false), ', '));

sim_list = {simBase, simTx};
sim_names = {'base', 'tx'};

for s = 1:2
    sim = sim_list{s};
    sim_name = sim_names{s};
    fprintf(fid, '  "%s": {\n', sim_name);

    entries = {};

    % Species
    for i = 1:size(dump_species, 1)
        sp_name = dump_species{i, 1};
        comp = dump_species{i, 2};
        key = sprintf('%s.%s', comp, sp_name);
        try
            [~, data, ~] = selectbyname(sim, key);
            vals = interp1(sim.Time, data, dump_times, 'linear');
            val_str = strjoin(arrayfun(@(v) sprintf('%.6e', v), vals, 'UniformOutput', false), ', ');
            entries{end+1} = sprintf('    "%s": [%s]', key, val_str);
        catch
            % Species may not exist in this sim (e.g. GVAX in baseline)
        end
    end

    % Observables
    for i = 1:length(dump_observables)
        obs = dump_observables{i};
        try
            [~, data, ~] = selectbyname(sim, obs);
            vals = interp1(sim.Time, data, dump_times, 'linear');
            val_str = strjoin(arrayfun(@(v) sprintf('%.6e', v), vals, 'UniformOutput', false), ', ');
            entries{end+1} = sprintf('    "%s": [%s]', obs, val_str);
        catch
        end
    end

    fprintf(fid, '%s\n', strjoin(entries, ',\n'));

    if s < 2
        fprintf(fid, '  },\n');
    else
        fprintf(fid, '  }\n');
    end
end

fprintf(fid, '}\n');
fclose(fid);
fprintf('  Saved: figures/cascade_timeseries.json\n');
