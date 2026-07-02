%% Export Model Metadata
% This script runs the PDAC QSP model initialization and exports comprehensive
% metadata about species, compartments, and parameters to a structured
% text file that can be used by AI agents for context.
%
% Usage:
%   Run this script from MATLAB after running startup.m
%   Output: model_catalog.txt in the project root directory

%% Run the model initialization script
fprintf('Running immune_oncology_model_PDAC.m to initialize model...\n');
run('immune_oncology_model_PDAC.m');

fprintf('Model initialized with %d species, %d compartments, %d parameters\n', ...
    length(model.Species), length(model.Compartments), length(model.Parameters));

%% Export metadata to file
output_file = fullfile(fileparts(mfilename('fullpath')), '..', 'model_catalog.txt');
fid = fopen(output_file, 'w');

fprintf(fid, '# PDAC QSP Model Metadata\n');
fprintf(fid, '# Generated: %s\n', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fprintf(fid, '# Model: %s\n\n', model.Name);

%% Compartments
fprintf(fid, '## Compartments\n\n');
compartments = model.Compartments;
for i = 1:length(compartments)
    comp = compartments(i);
    fprintf(fid, '### %s\n', comp.Name);
    fprintf(fid, '- Capacity: %.4g %s\n', comp.Capacity, comp.CapacityUnits);
    fprintf(fid, '- Constant: %s\n', mat2str(comp.ConstantCapacity));
    if ~isempty(comp.Notes)
        fprintf(fid, '- Notes: %s\n', comp.Notes);
    end
    fprintf(fid, '\n');
end

%% Species by Compartment
fprintf(fid, '## Species\n\n');
for i = 1:length(compartments)
    comp = compartments(i);
    comp_species = sbioselect(model, 'Type', 'species', 'Parent', comp);

    if ~isempty(comp_species)
        fprintf(fid, '### %s Compartment (%d species)\n\n', comp.Name, length(comp_species));
        fprintf(fid, '| Species | Initial Amount | Units | Description |\n');
        fprintf(fid, '|---------|----------------|-------|-------------|\n');

        for j = 1:length(comp_species)
            sp = comp_species(j);
            notes = sp.Notes;
            if isempty(notes)
                notes = '-';
            else
                notes = strrep(notes, newline, ' ');
                notes = strrep(notes, '|', '/');
            end
            fprintf(fid, '| %s | %.4g | %s | %s |\n', ...
                sp.Name, sp.InitialAmount, sp.InitialAmountUnits, notes);
        end
        fprintf(fid, '\n');
    end
end

%% Parameters (grouped by prefix/category)
fprintf(fid, '## Parameters\n\n');
params = model.Parameters;

% Group parameters by prefix
param_groups = containers.Map();
for i = 1:length(params)
    p = params(i);
    name = p.Name;

    % Extract prefix (before first underscore)
    parts = strsplit(name, '_');
    if length(parts) > 1
        prefix = parts{1};
    else
        prefix = 'other';
    end

    if ~isKey(param_groups, prefix)
        param_groups(prefix) = {};
    end
    param_groups(prefix) = [param_groups(prefix), {p}];
end

% Sort and write grouped parameters
prefixes = sort(keys(param_groups));
for i = 1:length(prefixes)
    prefix = prefixes{i};
    group_params = param_groups(prefix);

    fprintf(fid, '### %s Parameters (%d)\n\n', upper(prefix), length(group_params));
    fprintf(fid, '| Parameter | Value | Units | Description |\n');
    fprintf(fid, '|-----------|-------|-------|-------------|\n');

    for j = 1:length(group_params)
        p = group_params{j};
        notes = p.Notes;
        if isempty(notes)
            notes = '-';
        else
            notes = strrep(notes, newline, ' ');
            notes = strrep(notes, '|', '/');
        end
        fprintf(fid, '| %s | %.4g | %s | %s |\n', ...
            p.Name, p.Value, p.ValueUnits, notes);
    end
    fprintf(fid, '\n');
end

%% Reactions summary
fprintf(fid, '## Reactions Summary\n\n');
reactions = model.Reactions;
fprintf(fid, 'Total reactions: %d\n\n', length(reactions));

%% Key observables (species commonly used as test statistics)
fprintf(fid, '\n## Key Observables for Calibration\n\n');
fprintf(fid, 'These species are commonly measured in clinical/preclinical studies:\n\n');

key_observables = {
    'V_T.C1', 'Cancer cells (tumor volume proxy)';
    'V_T.T1_exh', 'Exhausted CD8+ T cells in tumor';
    'V_T.Th_exh', 'Exhausted CD4+ T cells in tumor';
    'V_T.T1', 'Effector CD8+ T cells in tumor';
    'V_T.Th', 'Helper CD4+ T cells in tumor';
    'V_T.Treg', 'Regulatory T cells in tumor';
    'V_T.MDSC', 'Myeloid-derived suppressor cells';
    'V_T.Mac_M1', 'M1 macrophages (anti-tumor)';
    'V_T.Mac_M2', 'M2 macrophages (pro-tumor)';
    'V_T.iCAF', 'Inflammatory CAFs (IL-6, CCL5 secretion)';
    'V_T.myCAF', 'Myofibroblastic CAFs (ECM, CXCL12, TGFb)';
    'V_T.apCAF', 'Antigen-presenting CAFs (tolerogenic Treg induction)';
    'V_C.T1', 'Circulating CD8+ T cells';
    'V_C.Treg', 'Circulating regulatory T cells';
    'V_LN.nT1', 'Naive CD8+ T cells in lymph node';
    'V_LN.aT1', 'Activated CD8+ T cells in lymph node';
};

fprintf(fid, '| Observable | Description |\n');
fprintf(fid, '|------------|-------------|\n');
for i = 1:size(key_observables, 1)
    fprintf(fid, '| %s | %s |\n', key_observables{i, 1}, key_observables{i, 2});
end

%% Close file
fclose(fid);
fprintf('\nModel metadata exported to: %s\n', output_file);
fprintf('Done.\n');
