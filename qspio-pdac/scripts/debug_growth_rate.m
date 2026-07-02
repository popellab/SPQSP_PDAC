% Debug: check if median parameters can grow tumor to diagnosis
cd('/Users/joeleliason/Projects/pdac-build');
startup;
immune_oncology_model_PDAC;

priors = readtable('parameters/pdac_priors.csv');
for i = 1:height(priors)
    pname = priors.name{i};
    mval = exp(priors.dist_param1(i));
    pobj = sbioselect(model, 'Name', pname, 'Type', 'parameter');
    if ~isempty(pobj)
        set(pobj, 'Value', mval);
    end
end

k_g = sbioselect(model, 'Name', 'k_C1_growth', 'Type', 'parameter');
k_d = sbioselect(model, 'Name', 'k_C1_death', 'Type', 'parameter');
K_K = sbioselect(model, 'Name', 'k_vas_growth', 'Type', 'parameter');
fprintf('DIAG:k_C1_growth=%.6e\n', k_g.Value);
fprintf('DIAG:k_C1_death=%.6e\n', k_d.Value);
fprintf('DIAG:k_vas_growth=%.6e\n', K_K.Value);
net = k_g.Value - k_d.Value;
fprintf('DIAG:net_growth=%.6e per day\n', net);
if net > 0
    fprintf('DIAG:doubling_days=%.0f\n', log(2)/net);
else
    fprintf('DIAG:net_growth_NEGATIVE\n');
end

[~, success, ~] = evolve_to_diagnosis(model, 'TargetDiameter', 3.0, 'Debug', true);
fprintf('DIAG:success=%d\n', success);