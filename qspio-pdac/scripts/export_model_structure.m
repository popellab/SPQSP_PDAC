% export_model_structure.m - Export the assembled SimBiology model to JSON.
%
% The model explorer and inference-side unit checks consume this compact,
% source-level representation. Keeping the exporter in the model repository
% makes `make model-structure` reproducible from a lean checkout with MATLAB.

immune_oncology_model_PDAC;

species_objects = model.Species;
species = repmat(struct( ...
    'name', '', 'compartment', '', 'base_name', '', ...
    'units', '', 'description', ''), numel(species_objects), 1);
for i = 1:numel(species_objects)
    species(i).name = [species_objects(i).Parent.Name, '.', species_objects(i).Name];
    species(i).compartment = species_objects(i).Parent.Name;
    species(i).base_name = species_objects(i).Name;
    species(i).units = species_objects(i).InitialAmountUnits;
    species(i).description = species_objects(i).Notes;
end

compartment_objects = model.Compartments;
compartments = repmat(struct( ...
    'name', '', 'volume', 0, 'volume_units', '', 'description', ''), ...
    numel(compartment_objects), 1);
for i = 1:numel(compartment_objects)
    compartments(i).name = compartment_objects(i).Name;
    compartments(i).volume = compartment_objects(i).Capacity;
    compartments(i).volume_units = compartment_objects(i).CapacityUnits;
    compartments(i).description = compartment_objects(i).Notes;
end

parameter_objects = model.Parameters;
parameters = repmat(struct( ...
    'name', '', 'value', 0, 'units', '', 'description', ''), ...
    numel(parameter_objects), 1);
parameter_names = cell(numel(parameter_objects), 1);
for i = 1:numel(parameter_objects)
    parameter_names{i} = parameter_objects(i).Name;
    parameters(i).name = parameter_objects(i).Name;
    parameters(i).value = parameter_objects(i).Value;
    parameters(i).units = parameter_objects(i).ValueUnits;
    parameters(i).description = parameter_objects(i).Notes;
end

reaction_objects = model.Reactions;
reactions = repmat(struct( ...
    'name', '', 'reactants', {{}}, 'products', {{}}, ...
    'rate_law', '', 'parameters', {{}}), numel(reaction_objects), 1);
for i = 1:numel(reaction_objects)
    reactions(i).name = reaction_objects(i).Name;
    reactions(i).reactants = qualified_species_names(reaction_objects(i).Reactants);
    reactions(i).products = qualified_species_names(reaction_objects(i).Products);
    reactions(i).rate_law = reaction_objects(i).ReactionRate;

    used = false(numel(parameter_names), 1);
    for j = 1:numel(parameter_names)
        token = regexptranslate('escape', parameter_names{j});
        pattern = ['(?<![A-Za-z0-9_])', token, '(?![A-Za-z0-9_])'];
        used(j) = ~isempty(regexp(reactions(i).rate_law, pattern, 'once'));
    end
    reactions(i).parameters = parameter_names(used)';
end

model_structure = struct( ...
    'species', species, ...
    'compartments', compartments, ...
    'parameters', parameters, ...
    'reactions', reactions);

repo_root = fileparts(fileparts(mfilename('fullpath')));
output_file = fullfile(repo_root, 'model_structure.json');
fid = fopen(output_file, 'w');
if fid == -1
    error('Could not open %s for writing', output_file);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(model_structure, 'PrettyPrint', true));
fprintf('Model structure exported to %s (%d species, %d parameters, %d reactions)\n', ...
    output_file, numel(species), numel(parameters), numel(reactions));

function names = qualified_species_names(objects)
names = cell(1, numel(objects));
for k = 1:numel(objects)
    names{k} = [objects(k).Parent.Name, '.', objects(k).Name];
end
end
