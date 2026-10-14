function test_qualitative_behaviors_helper()
%% test_qualitative_behaviors_helper
% Helper for pytest test_qualitative_behaviors.py
% Runs all simulations needed for qualitative behavior checks and outputs
% TEST_OUTPUT lines for Python parsing.
%
% Scenario YAMLs drive simulation configuration (stop_time, dosing, solver)
% so that tests stay in sync with scenario definitions.
%
% Tiers:
%   1 - Baseline TME snapshot (t=0) via baseline_no_treatment.yaml
%   2 - Temporal dynamics (0-90 days) via clinical_progression.yaml
%   3 - GVAX + nivo treatment response via gvax_nivo_neoadjuvant_zheng2022.yaml
%   4 - Directional perturbation tests (30 days, manual from progression IC)

fprintf('=== Qualitative Behavior Tests ===\n');

%% ============================================================
%% TIER 1: Baseline TME snapshot via baseline_no_treatment.yaml
%% ============================================================
fprintf('\n--- Tier 1: Baseline snapshot (baseline_no_treatment.yaml) ---\n');
try
    [simBase, modelBase, ~] = run_median_simulation( ...
        'scenarios/baseline_no_treatment.yaml', 'verbose', false);
    output_species(simBase, 1, 'base');

    % Extract vol_cell for stromal fraction computation in Python
    D_cell_obj = sbioselect(modelBase, 'Name', 'D_cell', 'Type', 'parameter');
    vol_cell_mL = (4/3) * pi * (D_cell_obj.Value/2 * 1e-4)^3;
    fprintf('TEST_OUTPUT:vol_cell_mL=%.8e\n', vol_cell_mL);
    fprintf('TEST_OUTPUT:evolve_success=%d\n', 1);

    % Hill function audit — output all named Hill function values at baseline
    output_hill_values(simBase, 1, 'hill');

    fprintf('Tier 1: OK\n');
catch ME
    fprintf('TEST_OUTPUT:evolve_success=%d\n', 0);
    fprintf('ERROR (Tier 1): %s\n', ME.message);
end

%% ============================================================
%% TIER 2: Temporal dynamics via clinical_progression.yaml
%% ============================================================
fprintf('\n--- Tier 2: Clinical progression (clinical_progression.yaml) ---\n');
try
    [simProg, modelProg, ~] = run_median_simulation( ...
        'scenarios/clinical_progression.yaml', 'verbose', false);

    output_species(simProg, 1, 'prog_d0');

    idx_d10 = find(simProg.Time >= 10, 1);
    idx_d21 = find(simProg.Time >= 21, 1);
    idx_d30 = find(simProg.Time >= 30, 1);
    idx_d60 = find(simProg.Time >= 60, 1);
    idx_d90 = length(simProg.Time);

    output_species(simProg, idx_d10, 'prog_d10');
    output_species(simProg, idx_d21, 'prog_d21');
    output_species(simProg, idx_d30, 'prog_d30');
    output_species(simProg, idx_d60, 'prog_d60');
    output_species(simProg, idx_d90, 'prog_d90');

    % k_C_Tcell_eff at progression d0 (baseline reference for treatment comparison)
    output_observable(simProg, 1, 'prog_d0', 'k_C_Tcell_eff');

    % Monotonicity check: output min C1 over entire trajectory
    [~, c1_data, ~] = selectbyname(simProg, 'V_T.C1');
    fprintf('TEST_OUTPUT:prog.C1_min=%.8e\n', min(c1_data));

    fprintf('Tier 2: OK\n');
catch ME
    fprintf('ERROR (Tier 2): %s\n', ME.message);
end

%% ============================================================
%% TIER 3: GVAX + nivo via gvax_nivo_neoadjuvant_zheng2022.yaml
%% ============================================================
fprintf('\n--- Tier 3: GVAX + nivo (gvax_nivo_neoadjuvant_zheng2022.yaml) ---\n');
try
    [simTx, ~, ~] = run_median_simulation( ...
        'scenarios/gvax_nivo_neoadjuvant_zheng2022.yaml', 'verbose', false);

    idx_tx_d3 = find(simTx.Time >= 3, 1);
    idx_tx_d7 = find(simTx.Time >= 7, 1);
    idx_tx_d14 = find(simTx.Time >= 14, 1);
    idx_tx_d21 = length(simTx.Time);

    output_species(simTx, 1, 'tx_d0');
    output_species(simTx, idx_tx_d3, 'tx_d3');
    output_species(simTx, idx_tx_d7, 'tx_d7');
    output_species(simTx, idx_tx_d14, 'tx_d14');
    output_species(simTx, idx_tx_d21, 'tx_d21');

    % Injection site and LN species at treatment time points
    output_injection_site(simTx, 1, 'tx_d0');
    output_injection_site(simTx, idx_tx_d3, 'tx_d3');
    output_injection_site(simTx, idx_tx_d7, 'tx_d7');
    output_injection_site(simTx, idx_tx_d14, 'tx_d14');
    output_injection_site(simTx, idx_tx_d21, 'tx_d21');

    % k_C_Tcell_eff (repeatedAssignment observable) at treatment time points
    output_observable(simTx, 1, 'tx_d0', 'k_C_Tcell_eff');
    output_observable(simTx, idx_tx_d3, 'tx_d3', 'k_C_Tcell_eff');
    output_observable(simTx, idx_tx_d7, 'tx_d7', 'k_C_Tcell_eff');
    output_observable(simTx, idx_tx_d14, 'tx_d14', 'k_C_Tcell_eff');
    output_observable(simTx, idx_tx_d21, 'tx_d21', 'k_C_Tcell_eff');

    % Hill function values under treatment (for Tier 6 treatment Hill audit)
    output_hill_values(simTx, 1, 'tx_d0_hill');
    output_hill_values(simTx, idx_tx_d21, 'tx_d21_hill');

    fprintf('Tier 3: OK\n');
catch ME
    fprintf('ERROR (Tier 3): %s\n', ME.message);
end

%% ============================================================
%% TIER 4: Perturbation tests (30 days from diagnosis IC)
%% ============================================================
% Perturbation tests are hypothetical experiments without scenario YAMLs.
% We build a fresh model here because run_median_simulation calls sbioreset,
% which deletes models from earlier tiers. The reference (prog_d30) was
% already output in Tier 2.
fprintf('\n--- Tier 4: Perturbation tests ---\n');
fprintf('  Building fresh model for perturbation tests...\n');

try
    sbioreset;
    immune_oncology_model_PDAC;

    % Apply median parameters from priors
    priors = readtable('parameters/pdac_priors.csv');
    for i = 1:height(priors)
        pname = priors.name{i};
        mval = exp(priors.dist_param1(i));
        pobj = sbioselect(model, 'Name', pname, 'Type', 'parameter');
        if ~isempty(pobj)
            set(pobj, 'Value', mval);
        else
            sobj = sbioselect(model, 'Name', pname, 'Type', 'species');
            if ~isempty(sobj)
                for s = 1:length(sobj)
                    sobj(s).InitialAmount = mval;
                end
            end
        end
    end

    % Overlay posterior medians (if file exists)
    post_json_path = 'parameters/posterior_medians.json';
    if isfile(post_json_path)
        post_text = fileread(post_json_path);
        post_data = jsondecode(post_text);
        post_medians = post_data.medians;
        post_names = fieldnames(post_medians);
        n_post = 0;
        for i = 1:length(post_names)
            pname = post_names{i};
            pval = post_medians.(pname);
            pobj = sbioselect(model, 'Name', pname, 'Type', 'parameter');
            if ~isempty(pobj)
                set(pobj, 'Value', pval);
                n_post = n_post + 1;
            else
                sobj = sbioselect(model, 'Name', pname, 'Type', 'species');
                if ~isempty(sobj)
                    for s = 1:length(sobj)
                        sobj(s).InitialAmount = pval;
                    end
                    n_post = n_post + 1;
                end
            end
        end
        fprintf('  Applied %d posterior medians from %s\n', n_post, post_json_path);
    end

    [model, pert_success, ~] = evolve_to_diagnosis(model, 'Debug', false);
    if ~pert_success
        error('evolve_to_diagnosis failed for perturbation model');
    end

    % 4a: TGFb blockade — zero all TGFb production
    fprintf('  Perturbation: TGFb blockade\n');
    run_perturbation(model, 'pert_tgfb', ...
        {'k_TGFb_sec_myCAF', 0; 'k_TGFb_Tsec', 0; 'k_TGFb_Msec', 0}, {});

    % 4b: CXCL12 blockade — zero CXCL12 production by iCAFs
    fprintf('  Perturbation: CXCL12 blockade\n');
    run_perturbation(model, 'pert_cxcl12', ...
        {'k_CXCL12_sec_iCAF', 0}, {});

    % 4c: IL-10 blockade — zero IL-10 production
    fprintf('  Perturbation: IL-10 blockade\n');
    run_perturbation(model, 'pert_il10', ...
        {'k_IL10_sec', 0}, {});

    % 4d: Collagen deposition blockade — zero collagen production
    fprintf('  Perturbation: collagen blockade\n');
    run_perturbation(model, 'pert_ecm', ...
        {'k_col_myCAF_sec', 0; 'k_col_qpsc_sec', 0}, {});

    % 4e: IFNg blockade — zero all IFNg secretion sources
    fprintf('  Perturbation: IFNg blockade\n');
    run_perturbation(model, 'pert_ifng', ...
        {'k_IFNg_Tsec', 0}, {});

catch ME
    fprintf('ERROR (Tier 4): %s\n', ME.message);
end

fprintf('\nMATLAB_DONE\n');
end


function run_perturbation(base_model, prefix, param_changes, species_changes)
%RUN_PERTURBATION Run a 30-day simulation with parameter/species modifications.
% Copies the base model (with diagnosis ICs already set), modifies the
% specified parameters/species, and simulates for 30 days.
    model_pert = copyobj(base_model);

    % Modify parameters
    for i = 1:size(param_changes, 1)
        pobj = sbioselect(model_pert, 'Name', param_changes{i,1}, 'Type', 'parameter');
        if ~isempty(pobj)
            set(pobj, 'Value', param_changes{i,2});
        else
            fprintf('  WARNING: parameter %s not found\n', param_changes{i,1});
        end
    end

    % Modify species initial amounts
    if ~isempty(species_changes)
        V_T = sbioselect(model_pert, 'Type', 'compartment', 'Name', 'V_T');
        for i = 1:size(species_changes, 1)
            sobj = sbioselect(model_pert, 'Type', 'species', ...
                'Name', species_changes{i,1}, 'Parent', V_T);
            if ~isempty(sobj)
                sobj(1).InitialAmount = species_changes{i,2};
            else
                fprintf('  WARNING: species %s not found in V_T\n', species_changes{i,1});
            end
        end
    end

    model_pert = simulation_config(model_pert, 'solver', 'sundials', ...
        'time', 0:1:30, 'abs_tolerance', 1e-9, 'rel_tolerance', 1e-6);

    try
        sim = sbiosimulate(model_pert);
        output_species(sim, length(sim.Time), prefix);
        fprintf('  %s: OK\n', prefix);
    catch ME
        fprintf('  %s: FAILED (%s)\n', prefix, ME.message);
    end

    delete(model_pert);
end


function output_species(simData, time_idx, prefix)
%OUTPUT_SPECIES Output all relevant species/parameter values at a given time index.
    species_list = {
        'V_T.Mac_M1', 'V_T.Mac_M2', ...
        'V_T.CD8', 'V_T.CD8_exh', ...
        'V_T.Treg', 'V_T.Th', ...
        'V_T.myCAF', 'V_T.iCAF', 'V_T.apCAF', 'V_T.qPSC', ...
        'V_T.cDC1', 'V_T.cDC2', 'V_T.mcDC1', 'V_T.mcDC2', ...
        'V_T.collagen', 'V_T.C1', ...
        'V_T.CXCL12', ...
        'V_T.IL6', 'V_T.IL10', 'V_T.TGFb', ...
        'V_T.VEGF', 'V_T.IFNg', 'V_T.K', 'V_T.MDSC', ...
    };
    param_list = {'C_total', 'T_total', 'M_total', 'aPSC_total'};
    compartment_list = {'V_T'};

    for i = 1:length(species_list)
        name = species_list{i};
        short = strrep(name, 'V_T.', '');
        val = get_val(simData, time_idx, name);
        fprintf('TEST_OUTPUT:%s.%s=%.8e\n', prefix, short, val);
    end

    for i = 1:length(param_list)
        name = param_list{i};
        val = get_val(simData, time_idx, name);
        fprintf('TEST_OUTPUT:%s.%s=%.8e\n', prefix, name, val);
    end

    for i = 1:length(compartment_list)
        name = compartment_list{i};
        val = get_val(simData, time_idx, name);
        fprintf('TEST_OUTPUT:%s.%s=%.8e\n', prefix, name, val);
    end
end


function output_hill_values(simData, time_idx, prefix)
%OUTPUT_HILL_VALUES Output Hill function values at a given time index.
% Hill functions are repeatedAssignment parameters computed during simulation.
% Values are output with the given prefix for Python parsing.
% prefix defaults to 'hill' if not provided.
    if nargin < 3, prefix = 'hill'; end
    hill_list = {
        'H_TGFb', 'H_TGFb_Teff', 'H_TGFb_APC', ...
        'H_IL10', ...
        'H_IL12', ...
        'H_IL6_iCAF', 'H_IL6_MDSC', 'H_IL6_M2', ...
        'H_CCL5_Treg', ...
        'H_CXCL12_Texcl', ...
        'H_pore_APC', 'H_pore_T', 'H_stiff_Texh', 'H_stiff_fib', ...
        'H_NO', 'H_ArgI_Teff', ...
        'H_DAMP', ...
        'H_P1_apCAF', ...
    };
    for i = 1:length(hill_list)
        name = hill_list{i};
        val = get_val(simData, time_idx, name);
        fprintf('TEST_OUTPUT:%s.%s=%.8e\n', prefix, name, val);
    end
end


function output_injection_site(simData, time_idx, prefix)
%OUTPUT_INJECTION_SITE Output injection site (V_ID) and LN species at a time index.
% These species track the GVAX treatment cascade from injection site
% through lymph node priming.
    vid_species = {'V_ID.GVAX_cells', 'V_ID.mAPC', 'V_ID.APC'};
    vln_species = {'V_LN.mcDC1', 'V_LN.mcDC2'};

    for i = 1:length(vid_species)
        name = vid_species{i};
        short = strrep(name, 'V_ID.', 'VID_');
        val = get_val(simData, time_idx, name);
        fprintf('TEST_OUTPUT:%s.%s=%.8e\n', prefix, short, val);
    end

    for i = 1:length(vln_species)
        name = vln_species{i};
        short = strrep(name, 'V_LN.', 'VLN_');
        val = get_val(simData, time_idx, name);
        fprintf('TEST_OUTPUT:%s.%s=%.8e\n', prefix, short, val);
    end
end


function output_observable(simData, time_idx, prefix, obs_name)
%OUTPUT_OBSERVABLE Output a single observable (repeatedAssignment parameter) at a time index.
    val = get_val(simData, time_idx, obs_name);
    fprintf('TEST_OUTPUT:%s.%s=%.8e\n', prefix, obs_name, val);
end


function val = get_val(simData, time_idx, name)
%GET_VAL Extract a single value from SimBiology simulation data.
% Works for species (e.g., 'V_T.Mac_M1'), parameters (e.g., 'pO2'),
% and compartments (e.g., 'V_T').
    try
        [~, data, ~] = selectbyname(simData, name);
        val = data(time_idx);
    catch
        val = NaN;
    end
end