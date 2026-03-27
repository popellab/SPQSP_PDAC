% Extract the ACTUAL state vector that the ODE solver sees internally.
% Strategy: use the exported model's simulate() which gives us the ODE function,
% then evaluate it at t=0 to see what y0 looks like.

cd('/Users/joeleliason/Projects/pdac-build');
run('startup.m');
immune_oncology_model_PDAC;

em = export(model);

% The exported model can be simulated directly
% Its InitialValues should be the SOLVER state vector
iv = em.InitialValues;
vi = em.ValueInfo;

% Print species in solver units
fprintf('=== Solver state vector (species only) ===\n');
fprintf('%-35s %15s %20s\n', 'Name', 'SolverValue', 'Units');
fprintf('%s\n', repmat('-', 1, 75));
for i = 1:length(vi)
    if strcmp(vi(i).Type, 'species')
        fprintf('%-35s %15.6e %20s\n', vi(i).Name, iv(i), vi(i).Units);
    end
end

% Now simulate the exported model for one step and check ODE RHS magnitude
fprintf('\n=== Simulating exported model ===\n');
opts = em.SimulationOptions;
fprintf('Solver: %s\n', opts.SolverType);

% Try to evaluate the RHS at t=0
% The exported model should have an ODE function we can call
try
    simdata = em.simulate();
    fprintf('Simulation succeeded: %d time points\n', length(simdata.Time));
    % Print first and second state to see the magnitude of changes
    fprintf('\n=== State at t=0 vs t=dt ===\n');
    fprintf('%-35s %15s %15s %15s\n', 'Name', 't=0', 't=dt', 'diff/dt');
    dt = simdata.Time(2) - simdata.Time(1);
    for i = 1:min(20, size(simdata.Data, 2))
        v0 = simdata.Data(1, i);
        v1 = simdata.Data(2, i);
        deriv = (v1 - v0) / dt;
        fprintf('%-35s %15.6e %15.6e %15.6e\n', simdata.DataNames{i}, v0, v1, deriv);
    end
catch e
    fprintf('Simulation failed: %s\n', e.message);
end

% Check: does the exported model's InitialValues match what we'd get
% by manually applying unit conversion?
fprintf('\n=== Unit conversion check ===\n');
sp = model.Species;
for i = 1:min(10, length(sp))
    s = sp(i);
    comp = s.Parent;
    % The key: what does SimBiology convert InitialAmount to internally?
    fprintf('%s.%s: declaredAmount=%.6e %s\n', comp.Name, s.Name, s.InitialAmount, s.InitialAmountUnits);
end