% Extract the 575-element internal state vector from SimBiology's exported model.
cd('/Users/joeleliason/Projects/pdac-build');
run('startup.m');
immune_oncology_model_PDAC;

em = export(model);

% Get the internal state vector and its metadata
iv = em.InitialValues;
vi = em.ValueInfo;

fprintf('InitialValues: %d elements\n', length(iv));
fprintf('ValueInfo: %d elements\n', length(vi));

% Write full state to CSV
fid = fopen('/tmp/simbio_internal_state.csv', 'w');
fprintf(fid, 'Index,Name,Type,Value,Units,QualifiedName\n');
for i = 1:length(vi)
    v = vi(i);
    try
        qname = v.QualifiedName;
    catch
        qname = '';
    end
    try
        units = v.Units;
    catch
        units = '';
    end
    try
        vtype = v.Type;
    catch
        vtype = '';
    end
    fprintf(fid, '%d,%s,%s,%.15e,%s,%s\n', i, v.Name, vtype, iv(i), units, qname);
    if i <= 30
        fprintf('%3d %-30s %-15s %15.6e %s\n', i, v.Name, vtype, iv(i), units);
    end
end
fclose(fid);
fprintf('\nWrote %d entries to /tmp/simbio_internal_state.csv\n', length(vi));

% Also print just the species (not parameters) to compare with C++
fprintf('\n=== Species only ===\n');
n_species = 0;
for i = 1:length(vi)
    v = vi(i);
    if strcmp(v.Type, 'species')
        n_species = n_species + 1;
        if n_species <= 20
            fprintf('%3d %-35s %15.6e %s\n', i, v.Name, iv(i), v.Units);
        end
    end
end
fprintf('Total species: %d\n', n_species);
