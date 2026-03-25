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

% Build and simulate
immune_oncology_model_PDAC;
simdata = sbiosimulate(model);

% Extract data
t = simdata.Time;
data = simdata.Data;
raw_names = simdata.DataNames;  % 243x1 column cell array

% Build a map from bare species name -> qualified "Compartment_Species"
% A species name can appear in multiple compartments, so we track index
species_list = model.Species;
qualified = containers.Map('KeyType', 'double', 'ValueType', 'char');
bare_to_comp = containers.Map();

for i = 1:length(species_list)
    s = species_list(i);
    comp = s.Parent.Name;
    qname = [comp '_' s.Name];
    % Map by name; if duplicate, store as cell array
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

% Build unique column names for the CSV
% DataNames order matches Data columns. Some are species, some are rules/params.
n_cols = length(raw_names);
col_names = cell(1, n_cols);
% Track how many times we've seen each bare name to pick the right compartment
seen_count = containers.Map();

for i = 1:n_cols
    n = raw_names{i};
    if bare_to_comp.isKey(n)
        mapping = bare_to_comp(n);
        if iscell(mapping)
            % Multiple compartments — use occurrence count to pick
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
            % Single compartment — but might appear again as a rule variable
            if seen_count.isKey(n)
                % Already used the species name, this must be a rule
                col_names{i} = n;
            else
                seen_count(n) = 1;
                col_names{i} = mapping;
            end
        end
    else
        % Not a species — rule variable or parameter
        col_names{i} = n;
    end
end

% Ensure uniqueness (final pass)
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