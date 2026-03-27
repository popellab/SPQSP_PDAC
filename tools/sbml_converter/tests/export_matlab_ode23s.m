% export_matlab_ode23s.m - Run SimBiology model with ode23s solver (Rosenbrock method)
% This provides an independent third solver to compare against ode15s and C++ CVODE.
%
% Usage:
%   matlab -batch "output_csv='/path/to/out.csv'; pdac_build_dir='/path/to/pdac-build'; run('/path/to/export_matlab_ode23s.m')"

if ~exist('output_csv', 'var')
    error('Set output_csv before running this script');
end
if ~exist('pdac_build_dir', 'var')
    pdac_build_dir = pwd;
end

cd(pdac_build_dir);
run('startup.m');

% Build model
immune_oncology_model_PDAC;

% Switch solver to ode23s (Rosenbrock method — different algorithm family from ode15s/CVODE BDF)
cs = getconfigset(model, 'active');
cs.SolverType = 'ode23t';
% Keep default tolerances
fprintf('Solver: %s, RelTol=%g, AbsTol=%g\n', cs.SolverType, ...
    cs.SolverOptions.RelativeTolerance, cs.SolverOptions.AbsoluteTolerance);

simdata = sbiosimulate(model);

% Extract and export (same logic as export_matlab_trajectories.m)
t = simdata.Time;
data = simdata.Data;
raw_names = simdata.DataNames;

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
