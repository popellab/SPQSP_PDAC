% Extract SimBiology's internal state vector (what CVODE actually sees).
%
% Strategy: use the exported model's ODE function to evaluate f(t0, y0)
% and compare the y0 values against our C++ initial state.
%
% Usage:
%   matlab -batch "output_csv='/path/to/out.csv'; pdac_build_dir='/path/to/pdac-build'; run('extract_simbio_internal.m')"

if ~exist('output_csv', 'var')
    output_csv = '/tmp/simbio_internal_state.csv';
end
if ~exist('pdac_build_dir', 'var')
    pdac_build_dir = '/Users/joeleliason/Projects/pdac-build';
end

cd(pdac_build_dir);
run('startup.m');
immune_oncology_model_PDAC;

cs = getconfigset(model, 'active');
fprintf('UnitConversion: %d\n', cs.CompileOptions.UnitConversion);
fprintf('DimensionalAnalysis: %d\n', cs.CompileOptions.DimensionalAnalysis);

% Export the model — this gives us access to the compiled ODE function
em = export(model);

% The exported model has methods to get the initial conditions and evaluate the ODE
% Try to access the state vector
fprintf('\n=== Exported model properties ===\n');
props = properties(em);
for i = 1:length(props)
    p = props{i};
    try
        val = em.(p);
        if isnumeric(val) && numel(val) < 20
            fprintf('%s = %s\n', p, mat2str(val, 6));
        elseif isnumeric(val)
            fprintf('%s = [%d x %d numeric]\n', p, size(val,1), size(val,2));
        elseif ischar(val) || isstring(val)
            fprintf('%s = %s\n', p, val);
        else
            fprintf('%s = [%s]\n', p, class(val));
        end
    catch
        fprintf('%s = <error reading>\n', p);
    end
end

% Try to get the ODE function handle and initial state
fprintf('\n=== Trying to get internal state ===\n');
try
    methods_list = methods(em);
    fprintf('Methods: ');
    for i = 1:length(methods_list)
        fprintf('%s ', methods_list{i});
    end
    fprintf('\n');
catch e
    fprintf('methods failed: %s\n', e.message);
end

% Alternative approach: use sbioselect and the model's reactions to trace
% unit conversion. Get the species objects and check their converted values.
fprintf('\n=== Species with unit info ===\n');
sp = model.Species;
% Write CSV: Name, Compartment, InitialAmount, InitialAmountUnits, HOSU, InitialConcentration
fid = fopen(output_csv, 'w');
fprintf(fid, 'Compartment,Species,InitialAmount,AmountUnits,InitialConcentration,CompCapacity,CompUnits\n');
for i = 1:length(sp)
    s = sp(i);
    comp = s.Parent;
    fprintf(fid, '%s,%s,%.15e,%s,%.15e,%s,%s\n', ...
        comp.Name, s.Name, s.InitialAmount, s.InitialAmountUnits, ...
        s.InitialConcentration, num2str(comp.Capacity), comp.CapacityUnits);
    if i <= 20
        fprintf('%s.%s: amount=%.6e %s, conc=%.6e, comp=%s %s\n', ...
            comp.Name, s.Name, s.InitialAmount, s.InitialAmountUnits, ...
            s.InitialConcentration, num2str(comp.Capacity), comp.CapacityUnits);
    end
end
fclose(fid);
fprintf('\nWrote species info to %s\n', output_csv);

% Now the key test: simulate for one tiny step and look at the SUNDIALS stats
% to understand the actual solver state
cs.SolverType = 'sundials';
cs.StopTime = 0.001;  % very short
cs.SolverOptions.OutputTimes = [0 0.001];

simdata = sbiosimulate(model);
stats = simdata.SimulationInfo;
fprintf('\n=== Solver stats ===\n');
try
    fprintf('Struct fields: ');
    fn = fieldnames(stats);
    for i = 1:length(fn)
        fprintf('%s ', fn{i});
    end
    fprintf('\n');
catch
    fprintf('SimulationInfo class: %s\n', class(stats));
    disp(stats);
end
