function qnames = qualify_species_names(model, raw_names)
% QUALIFY_SPECIES_NAMES Map bare SimBiology DataNames to compartment-qualified names.
%
% SimBiology's sbiosimulate() returns bare species names ("Treg", "nCD4") in
% DataNames. The same bare name appears once per compartment (V_C.Treg,
% V_P.Treg, V_T.Treg, V_LN.Treg). The column ORDER in DataNames matches the
% model's internal species order, NOT the export(model) order.
%
% This function walks model.Species in the same order that sbiosimulate uses
% (verified: sbiosimulate outputs species grouped by compartment, in the order
% compartments and species were added to the model). For each bare name in
% raw_names, it finds the Nth species with that name (N = how many times
% that bare name has appeared so far) and returns Compartment_Species.
%
% WARNING: The export(model).ValueInfo order is DIFFERENT from sbiosimulate's
% DataNames order. Do NOT mix these two orderings — they map different columns
% to different species. Always use this function with sbiosimulate output.
%
% Usage:
%   simdata = sbiosimulate(model);
%   qnames = qualify_species_names(model, simdata.DataNames);

    species = model.Species;
    n = length(raw_names);
    qnames = cell(1, n);
    seen = containers.Map();

    for i = 1:n
        name = raw_names{i};

        % Count how many times we've seen this bare name
        if seen.isKey(name)
            seen(name) = seen(name) + 1;
        else
            seen(name) = 1;
        end
        occurrence = seen(name);

        % Find the Nth species in model.Species with this bare name
        match_count = 0;
        for j = 1:length(species)
            if strcmp(species(j).Name, name)
                match_count = match_count + 1;
                if match_count == occurrence
                    qnames{i} = [species(j).Parent.Name '_' name];
                    break;
                end
            end
        end

        % Fallback if no match found
        if isempty(qnames{i})
            qnames{i} = name;
        end
    end
end