function test_evolve_to_diagnosis_helper()
%% test_evolve_to_diagnosis_helper
% Helper function for pytest test_evolve_to_diagnosis.py
% Outputs structured TEST_OUTPUT lines for Python parsing.
%
% Usage (from repo root):
%   matlab -batch "cd('REPO_ROOT'); startup; test_evolve_to_diagnosis_helper"

%% Build model
immune_oncology_model_PDAC;

% Use the model's own defaults. This test guards assembly and diagnosis-state
% initialization, not calibration; paper-side priors are tested in pdac-paper.

%% Check initial volume scale via set_healthy_populations
model_ic = copyobj(model);
model_ic = set_healthy_populations(model_ic, 'TumorCells', 1e6, 'Verbose', true);

D_cell_obj = sbioselect(model_ic, 'Type', 'parameter', 'Name', 'D_cell');
D_cell_um = D_cell_obj.Value;
vol_cell_um3 = 4/3 * pi * (D_cell_um/2)^3;
vol_cell_mL = vol_cell_um3 / 1e12;
init_V_mL = 1e6 * vol_cell_mL;

fprintf('TEST_OUTPUT:D_cell_um=%.1f\n', D_cell_um);
fprintf('TEST_OUTPUT:init_V_tumor_mL=%.6e\n', init_V_mL);

% Check initial CD8 count (search within V_T to avoid V_C/V_LN/V_P duplicates)
V_T_comp = sbioselect(model_ic, 'Type', 'compartment', 'Name', 'V_T');
CD8_obj = sbioselect(model_ic, 'Type', 'species', 'Name', 'CD8', 'Parent', V_T_comp);
if ~isempty(CD8_obj)
    fprintf('TEST_OUTPUT:init_CD8=%.6e\n', CD8_obj(1).InitialAmount);
end
delete(model_ic);

%% Run evolve_to_diagnosis with 3 cm target
target_diam = 3.0;
[model_out, success, simData] = evolve_to_diagnosis(model, ...
    'TargetDiameter', target_diam, 'Debug', true);

fprintf('TEST_OUTPUT:evolution_success=%d\n', success);

if success
    %% Extract diagnosis-state values
    V_T_idx = find(strcmp(simData.DataNames, 'V_T'));
    V_T_traj = simData.Data(:, V_T_idx);
    diam_traj = (6.0 * V_T_traj / pi).^(1/3);
    idx = find(diam_traj >= target_diam, 1, 'first');

    fprintf('TEST_OUTPUT:time_to_diagnosis=%.1f\n', simData.Time(idx));
    fprintf('TEST_OUTPUT:V_T_diagnosis_mL=%.6e\n', V_T_traj(idx));
    fprintf('TEST_OUTPUT:diam_diagnosis_cm=%.4f\n', diam_traj(idx));

    V_T_mm3 = V_T_traj(idx) * 1000;

    % Use model.Species to find V_T species indices directly.
    % simData columns 1:N_species correspond to model.Species order.
    C1_val  = get_vt_val(model_out, simData, idx, 'C1');
    CD8_val = get_vt_val(model_out, simData, idx, 'CD8');
    Treg_val = get_vt_val(model_out, simData, idx, 'Treg');
    M1_val  = get_vt_val(model_out, simData, idx, 'Mac_M1');
    M2_val  = get_vt_val(model_out, simData, idx, 'Mac_M2');
    ECM_val = get_vt_val(model_out, simData, idx, 'collagen');

    if ~isnan(C1_val)
        fprintf('TEST_OUTPUT:C1_diagnosis=%.6e\n', C1_val);
        fprintf('TEST_OUTPUT:C1_density=%.2f\n', C1_val / V_T_mm3);
    end

    if ~isnan(CD8_val)
        fprintf('TEST_OUTPUT:CD8_diagnosis=%.6e\n', CD8_val);
        fprintf('TEST_OUTPUT:CD8_density=%.2f\n', CD8_val / V_T_mm3);
    end

    if ~isnan(Treg_val)
        fprintf('TEST_OUTPUT:Treg_diagnosis=%.6e\n', Treg_val);
        fprintf('TEST_OUTPUT:Treg_density=%.2f\n', Treg_val / V_T_mm3);
    end

    if ~isnan(M1_val) && ~isnan(M2_val)
        TAM_val = M1_val + M2_val;
        fprintf('TEST_OUTPUT:TAM_diagnosis=%.6e\n', TAM_val);
        fprintf('TEST_OUTPUT:TAM_density=%.2f\n', TAM_val / V_T_mm3);
        fprintf('TEST_OUTPUT:M1_M2_ratio=%.4f\n', M1_val / max(M2_val, 1));
    end

    if ~isnan(ECM_val)
        fprintf('TEST_OUTPUT:collagen_diagnosis_mg=%.6e\n', ECM_val);
    end
end

%% Test: evolve_to_diagnosis without explicit TargetDiameter or variant
% This mimics the batch_worker.m path where initial_tumour_diameter is NOT
% in the priors (and thus not in the variant). The function must fall back
% to reading initial_tumour_diameter from the model parameters.
fprintf('\n--- Test: model-parameter fallback (no explicit TargetDiameter) ---\n');
model_copy2 = copyobj(model);

% Create an empty variant (mimics batch_worker when initial_tumour_diameter
% is not in pdac_priors.csv)
empty_variant = addvariant(model_copy2, 'empty_test_variant');

[model_out2, success2, ~] = evolve_to_diagnosis(model_copy2, ...
    'Variant', empty_variant, 'Debug', true);

fprintf('TEST_OUTPUT:fallback_success=%d\n', success2);

if success2
    % Verify it used the model's initial_tumour_diameter (3.2 cm)
    V_T_obj2 = sbioselect(model_out2, 'Type', 'compartment', 'Name', 'V_T');
    V_T_val2 = V_T_obj2.Capacity;
    diam2 = (6.0 * V_T_val2 / pi)^(1/3);
    fprintf('TEST_OUTPUT:fallback_diam_cm=%.4f\n', diam2);
    fprintf('TEST_OUTPUT:fallback_V_T_mL=%.6e\n', V_T_val2);
end
delete(model_copy2);

fprintf('MATLAB_DONE\n');
end


function val = get_vt_val(model_obj, sim_data, row_idx, species_name)
%GET_VT_VAL Get value of a V_T species from simulation data.
% Uses model.Species ordering to find the correct column index.
    val = NaN;
    for j = 1:length(model_obj.Species)
        if strcmp(model_obj.Species(j).Name, species_name) && ...
           strcmp(model_obj.Species(j).Parent.Name, 'V_T')
            if j <= size(sim_data.Data, 2)
                val = sim_data.Data(row_idx, j);
            end
            return;
        end
    end
end
