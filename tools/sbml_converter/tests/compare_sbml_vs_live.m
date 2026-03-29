% compare_sbml_vs_live.m - Compare SBML kinetic law expressions against live model.
%
% SimBiology's sbmlexport() modifies kinetic law expressions for SBML compliance:
% - Multiplies concentration-species reaction rates by compartment volume
%   (SBML convention: kinetic laws give amount/time, not concentration/time)
% - But does this INCONSISTENTLY — some reactions get *V_T, others don't
%
% This script extracts ALL reaction rates from both the live model and the SBML,
% resolves SBML IDs to names, and flags differences.
%
% Usage:
%   matlab -batch "run('tools/sbml_converter/tests/compare_sbml_vs_live.m')"

pdac_dir = '/Users/joeleliason/Projects/pdac-build';

%% 1. Build live model and extract reaction rates
cd(pdac_dir);
run('startup.m');
immune_oncology_model_PDAC;

rxns = model.Reactions;
live_rates = containers.Map();
for i = 1:length(rxns)
    live_rates(rxns(i).Name) = rxns(i).ReactionRate;
end
fprintf('Live model: %d reactions\n', length(rxns));

%% 2. Parse SBML and extract kinetic law expressions (with name resolution)
doc = sbmlimport_raw(fullfile(pdac_dir, 'PDAC_model.sbml'));

fprintf('SBML: %d reactions\n', length(doc.rates));
fprintf('\n=== Differences between Live model and SBML export ===\n');
fprintf('(SBML IDs resolved to readable names)\n\n');

n_match = 0;
n_diff = 0;
for i = 1:length(doc.names)
    name = doc.names{i};
    sbml_rate = doc.rates{i};
    if live_rates.isKey(name)
        live_rate = live_rates(name);
        % Normalize whitespace for comparison
        live_norm = strrep(live_rate, ' ', '');
        sbml_norm = strrep(sbml_rate, ' ', '');
        if strcmp(live_norm, sbml_norm)
            n_match = n_match + 1;
        else
            n_diff = n_diff + 1;
            fprintf('  %s:\n', name);
            fprintf('    Live: %s\n', live_rate);
            fprintf('    SBML: %s\n', sbml_rate);
            fprintf('\n');
        end
    end
end

fprintf('Matching: %d, Different: %d\n', n_match, n_diff);
if n_diff == 0
    fprintf('SBML export matches live model perfectly.\n');
else
    fprintf('SBML export has %d differences from live model.\n', n_diff);
    fprintf('These may be SBML compartment-volume scaling (expected for concentration species)\n');
    fprintf('or genuine export bugs.\n');
end


function doc = sbmlimport_raw(sbml_file)
% Parse SBML XML and extract kinetic law formulas with ID->name resolution.
% Does NOT use sbmlimport (which can fail on reimport).

    import javax.xml.parsers.DocumentBuilderFactory;
    import org.w3c.dom.*;

    factory = DocumentBuilderFactory.newInstance();
    builder = factory.newDocumentBuilder();
    xdoc = builder.parse(sbml_file);

    model_node = xdoc.getElementsByTagName('model').item(0);

    % Build ID -> name mapping
    id_to_name = containers.Map();

    % Compartments
    comps = model_node.getElementsByTagName('compartment');
    for i = 0:comps.getLength()-1
        node = comps.item(i);
        id = char(node.getAttribute('id'));
        name = char(node.getAttribute('name'));
        if ~isempty(name); id_to_name(id) = name;
        else; id_to_name(id) = id; end
    end

    % Species
    species = model_node.getElementsByTagName('species');
    for i = 0:species.getLength()-1
        node = species.item(i);
        id = char(node.getAttribute('id'));
        name = char(node.getAttribute('name'));
        comp_id = char(node.getAttribute('compartment'));
        if id_to_name.isKey(comp_id)
            comp_name = id_to_name(comp_id);
        else
            comp_name = comp_id;
        end
        if ~isempty(name)
            id_to_name(id) = [comp_name '.' name];
        else
            id_to_name(id) = id;
        end
    end

    % Parameters
    params = model_node.getElementsByTagName('parameter');
    for i = 0:params.getLength()-1
        node = params.item(i);
        id = char(node.getAttribute('id'));
        name = char(node.getAttribute('name'));
        if ~isempty(name); id_to_name(id) = name;
        else; id_to_name(id) = id; end
    end

    % Extract reactions
    reactions = model_node.getElementsByTagName('reaction');
    doc.names = {};
    doc.rates = {};
    for i = 0:reactions.getLength()-1
        rxn = reactions.item(i);
        name = char(rxn.getAttribute('name'));

        % Get kinetic law math as string via libsbml (more reliable than XML parsing)
        % Fall back to formula attribute
        kl_nodes = rxn.getElementsByTagName('kineticLaw');
        if kl_nodes.getLength() > 0
            math_nodes = kl_nodes.item(0).getElementsByTagName('math');
            if math_nodes.getLength() > 0
                % Get the formula via SBML formula string
                % Use a simple approach: extract ci elements
                formula = extract_formula_from_mathml(math_nodes.item(0), id_to_name);
            else
                formula = '?';
            end
        else
            formula = '?';
        end

        doc.names{end+1} = name;
        doc.rates{end+1} = formula;
    end
end


function str = extract_formula_from_mathml(math_node, id_map)
% Recursively convert MathML to human-readable string with resolved names.
    str = convert_node(math_node, id_map);
end

function str = convert_node(node, id_map)
    if node.getNodeType() == 3  % TEXT_NODE
        str = strtrim(char(node.getTextContent()));
        return;
    end

    tag = char(node.getNodeName());

    switch tag
        case 'math'
            % Process first child
            children = get_element_children(node);
            if length(children) >= 1
                str = convert_node(children{1}, id_map);
            else
                str = '?';
            end

        case 'apply'
            children = get_element_children(node);
            if isempty(children); str = '?'; return; end
            op = char(children{1}.getNodeName());
            args = {};
            for i = 2:length(children)
                args{end+1} = convert_node(children{i}, id_map);
            end

            switch op
                case 'times'
                    str = strjoin(args, ' * ');
                case 'divide'
                    if length(args) == 2
                        str = [args{1} ' / ' args{2}];
                    else
                        str = strjoin(args, ' / ');
                    end
                case 'plus'
                    str = strjoin(args, ' + ');
                case 'minus'
                    if length(args) == 1
                        str = ['-' args{1}];
                    else
                        str = strjoin(args, ' - ');
                    end
                case 'power'
                    str = [args{1} '^' args{2}];
                otherwise
                    str = [op '(' strjoin(args, ', ') ')'];
            end

        case 'ci'
            id = strtrim(char(node.getTextContent()));
            if id_map.isKey(id)
                str = id_map(id);
            else
                str = id;
            end

        case 'cn'
            str = strtrim(char(node.getTextContent()));

        case 'csymbol'
            str = strtrim(char(node.getTextContent()));

        case 'piecewise'
            str = 'piecewise(...)';

        otherwise
            children = get_element_children(node);
            if length(children) == 1
                str = convert_node(children{1}, id_map);
            else
                str = ['<' tag '>'];
            end
    end
end

function children = get_element_children(node)
    children = {};
    child = node.getFirstChild();
    while ~isempty(child)
        if child.getNodeType() == 1  % ELEMENT_NODE
            children{end+1} = child;
        end
        child = child.getNextSibling();
    end
end
