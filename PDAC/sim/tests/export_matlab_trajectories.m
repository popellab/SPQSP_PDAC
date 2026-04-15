% export_matlab_trajectories.m - Run SimBiology model and export trajectories
%
% Usage:
%   matlab -batch "output_csv='/path/to/out.csv'; pdac_build_dir='/path/to/pdac-build'; run('/path/to/export_matlab_trajectories.m')"
%
% Requires: output_csv variable set before calling

if ~exist('output_csv', 'var')
    error('Set output_csv before running this script');
end
if ~exist('pdac_build_dir', 'var')
    pdac_build_dir = pwd;
end

cd(pdac_build_dir);
run('startup.m');

% Use the live .m model script. sbmlimport(sbml_path) would be the
% source-of-truth loader (same SBML the C++ codegen reads), but SimBiology's
% sbmlexport strips the max(x, 0) guards from sqrt/pow kinetic laws in the
% p0_MHC_tot / p1_MHC_tot TCR expressions — reimported models produce
% complex-valued RHS and fail to integrate. Live .m it is.
immune_oncology_model_PDAC;

% Tighten solver tolerances to roughly match the C++ CVODE settings
% (reltol=1e-6, abstol=1e-12). SimBiology defaults (1e-3 / 1e-6) are
% too loose to compare against C++ at day-100+ timescales.
cfg = getconfigset(model);
cfg.SolverOptions.RelativeTolerance = 1e-5;
cfg.SolverOptions.AbsoluteTolerance = 1e-9;
% Output grid. Defaults match the baseline 365-day run; set `stop_time`
% before calling to use a shorter scenario (e.g. event-fire tests).
if ~exist('stop_time', 'var') || isempty(stop_time)
    stop_time = 365;
end
cfg.StopTime = stop_time;
cfg.SolverOptions.OutputTimes = (0:0.1:stop_time)';

% Optional: override model values with those from a param_all XML so MATLAB
% simulates with the exact same ICs / parameters as the C++ run.
if exist('param_xml', 'var') && ~isempty(param_xml)
    fprintf('Applying param overrides from %s\n', param_xml);
    doc = xmlread(param_xml);
    iv = doc.getElementsByTagName('init_value').item(0);
    if isempty(iv)
        error('No <init_value> element found in %s', param_xml);
    end

    % Index model objects by the names we expect in the XML
    comp_map = containers.Map();
    for i = 1:length(model.Compartments)
        comp_map(model.Compartments(i).Name) = model.Compartments(i);
    end
    sp_map = containers.Map();
    for i = 1:length(model.Species)
        s = model.Species(i);
        sp_map([s.Parent.Name '_' s.Name]) = s;
    end
    par_map = containers.Map();
    for i = 1:length(model.Parameters)
        par_map(model.Parameters(i).Name) = model.Parameters(i);
    end

    n_set = struct('comp', 0, 'sp', 0, 'par', 0, 'miss', 0);
    for section_name = {'Compartment', 'Species', 'Parameter'}
        sec = iv.getElementsByTagName(section_name{1}).item(0);
        if isempty(sec); continue; end
        children = sec.getChildNodes();
        for k = 0:children.getLength()-1
            node = children.item(k);
            if node.getNodeType() ~= node.ELEMENT_NODE; continue; end
            name = char(node.getNodeName());
            val = str2double(char(node.getTextContent()));
            if isnan(val); continue; end
            switch section_name{1}
                case 'Compartment'
                    if comp_map.isKey(name)
                        c = comp_map(name); c.Capacity = val;
                        n_set.comp = n_set.comp + 1;
                    else; n_set.miss = n_set.miss + 1; end
                case 'Species'
                    if sp_map.isKey(name)
                        s = sp_map(name); s.InitialAmount = val;
                        n_set.sp = n_set.sp + 1;
                    else; n_set.miss = n_set.miss + 1; end
                case 'Parameter'
                    if par_map.isKey(name)
                        p = par_map(name); p.Value = val;
                        n_set.par = n_set.par + 1;
                    else; n_set.miss = n_set.miss + 1; end
            end
        end
    end
    fprintf('  Overrode: %d compartments, %d species, %d parameters (%d names not in model)\n', ...
        n_set.comp, n_set.sp, n_set.par, n_set.miss);
end

simdata = sbiosimulate(model);

t = simdata.Time;
data = simdata.Data;
raw_names = simdata.DataNames;

% Map bare species name -> qualified "Compartment_Species". A species name may
% repeat across compartments, so keep a list per bare name and index into it.
species_list = model.Species;
bare_to_comp = containers.Map();

for i = 1:length(species_list)
    s = species_list(i);
    comp = s.Parent.Name;
    qname = [comp '_' s.Name];
    if bare_to_comp.isKey(s.Name)
        existing = bare_to_comp(s.Name);
        if ~iscell(existing)
            existing = {existing};
        end
        existing{end+1} = qname;
        bare_to_comp(s.Name) = existing;
    else
        bare_to_comp(s.Name) = qname;
    end
end

n_cols = length(raw_names);
col_names = cell(1, n_cols);
seen_count = containers.Map();

for i = 1:n_cols
    n = raw_names{i};
    if bare_to_comp.isKey(n)
        mapping = bare_to_comp(n);
        if iscell(mapping)
            if seen_count.isKey(n)
                seen_count(n) = seen_count(n) + 1;
            else
                seen_count(n) = 1;
            end
            idx = seen_count(n);
            if idx <= length(mapping)
                col_names{i} = mapping{idx};
            else
                col_names{i} = [n '_' num2str(idx)];
            end
        else
            if seen_count.isKey(n)
                col_names{i} = n;
            else
                seen_count(n) = 1;
                col_names{i} = mapping;
            end
        end
    else
        col_names{i} = n;
    end
end

final_seen = containers.Map();
for i = 1:n_cols
    n = col_names{i};
    if final_seen.isKey(n)
        final_seen(n) = final_seen(n) + 1;
        col_names{i} = [n '_dup' num2str(final_seen(n))];
    else
        final_seen(n) = 1;
    end
end

T = array2table([t, data], 'VariableNames', [{'Time'}, col_names]);
writetable(T, output_csv);
fprintf('Wrote %d time points, %d columns to %s\n', size(data,1), n_cols, output_csv);
