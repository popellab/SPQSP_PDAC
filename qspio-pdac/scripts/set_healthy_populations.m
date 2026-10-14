function model = set_healthy_populations(model, varargin)
% SET_HEALTHY_POPULATIONS Initialize populations for early microinvasive PDAC
%
% Sets initial conditions representing an early microinvasive PDAC with
% homeostatic stroma (~1e6 cells, ~1.7 mm diameter). This provides a
% starting point for natural history simulations that evolve from
% early → diagnosis states.
%
% All density/ratio/cytokine/threshold numbers are read from the shared
% YAML at resources/cpp/healthy_state.yaml. The C++ set_healthy_populations
% reads the same file, so changes to the biology propagate to both sides.
%
% Volume is computed from the model's own D_cell parameter (cell diameter)
% to stay consistent with the V_T rule in simbio_init.m.
%
% Inputs:
%   model    - SimBiology model object
%   varargin - Optional parameters:
%              'TumorCells': initial tumor cell count (default: from YAML)
%              'CancerSpeciesName': cancer species name (default: 'C1')
%              'Verbose': print debug info (default: true)
%
% Output:
%   model - Model with early microinvasive populations set.

p = inputParser;
addParameter(p, 'TumorCells', []);
addParameter(p, 'CancerSpeciesName', 'C1');
addParameter(p, 'Verbose', true);
parse(p, varargin{:});
CancerSpeciesName = p.Results.CancerSpeciesName;
verbose = p.Results.Verbose;

% Load shared spec
hs = yaml_read(healthy_state_path());

if isempty(p.Results.TumorCells)
    tumor_cells = hs.default_tumor_cells;
else
    tumor_cells = p.Results.TumorCells;
end

% =========================================================================
% STEP 1: Calculate tumor volume from cell count using model's cell volume
% =========================================================================
D_cell_obj = sbioselect(model, 'Type', 'parameter', 'Name', 'D_cell');
D_cell_um = D_cell_obj.Value;
vol_cell_um3 = 4/3 * pi * (D_cell_um/2)^3;
vol_cell_mL = vol_cell_um3 / 1e12;
V_tumor_mL = tumor_cells * vol_cell_mL;
V_tumor_mm3 = V_tumor_mL * 1000;

if verbose
    fprintf('=== Setting Early Microinvasive Populations (YAML-driven) ===\n');
    fprintf('Target tumor cells: %.2e\n', tumor_cells);
    fprintf('D_cell: %.1f um -> vol_cell: %.0f um^3 (%.2e mL)\n', ...
        D_cell_um, vol_cell_um3, vol_cell_mL);
    fprintf('Tumor volume: %.2f mm^3 (%.4f mL)\n', V_tumor_mm3, V_tumor_mL);
    fprintf('\n');
end

% =========================================================================
% STEP 2: Convert YAML densities to absolute counts
% =========================================================================
d = hs.cell_densities_per_mm3;
r = hs.ratios;

CD8_count    = d.CD8    * V_tumor_mm3;
Treg_count   = d.Treg   * V_tumor_mm3;
Th_count     = d.Th     * V_tumor_mm3;
MDSC_count   = d.MDSC   * V_tumor_mm3;
TAM_total    = d.TAM_total * V_tumor_mm3;
iCAF_count   = d.iCAF   * V_tumor_mm3;
myCAF_count  = d.myCAF  * V_tumor_mm3;
apCAF_count  = d.apCAF  * V_tumor_mm3;
qPSC_count   = d.qPSC   * V_tumor_mm3;
APC_total    = d.APC_total * V_tumor_mm3;

% Split TAM into M1/M2
Mac_M2_count = TAM_total / (1 + r.m1_m2);
Mac_M1_count = r.m1_m2 * Mac_M2_count;

% Split APC into cDC1/cDC2 and mature/immature
cDC1_count  = r.f_cDC1       * (1 - r.apc_mat) * APC_total;
cDC2_count  = (1 - r.f_cDC1) * (1 - r.apc_mat) * APC_total;
mcDC1_count = r.f_cDC1       * r.apc_mat       * APC_total;
mcDC2_count = (1 - r.f_cDC1) * r.apc_mat       * APC_total;

% Collagen mass
collagen_mass = hs.collagen.volume_fraction * V_tumor_mL ...
              * hs.collagen.density_mg_per_mL;

% =========================================================================
% STEP 3: Set species initial amounts in model
% =========================================================================
V_T_comp = sbioselect(model, 'Type', 'compartment', 'Name', 'V_T');

cell_species = {
    CancerSpeciesName, tumor_cells,  'Cancer cells (C1)';
    'CD8',             CD8_count,    'CD8+ T cells';
    'Treg',            Treg_count,   'Tregs';
    'Th',              Th_count,     'CD4+ helper T cells';
    'MDSC',            MDSC_count,   'MDSCs';
    'Mac_M1',          Mac_M1_count, 'M1 macrophages';
    'Mac_M2',          Mac_M2_count, 'M2 macrophages';
    'iCAF',            iCAF_count,   'iCAFs (inflammatory)';
    'myCAF',           myCAF_count,  'myCAFs (myofibroblastic)';
    'apCAF',           apCAF_count,  'apCAFs (antigen-presenting)';
    'qPSC',            qPSC_count,   'Quiescent PSCs';
    'cDC1',            cDC1_count,   'Immature cDC1 (cross-presenting)';
    'cDC2',            cDC2_count,   'Immature cDC2 (MHC-II)';
    'mcDC1',           mcDC1_count,  'Mature cDC1';
    'mcDC2',           mcDC2_count,  'Mature cDC2';
};

if verbose
    fprintf('Cell populations:\n');
end

for i = 1:size(cell_species, 1)
    species_name = cell_species{i, 1};
    count_val    = cell_species{i, 2};
    display_name = cell_species{i, 3};

    species_obj = sbioselect(model, 'Type', 'species', 'Name', species_name, 'Parent', V_T_comp);
    if isempty(species_obj)
        species_obj = sbioselect(model, 'Type', 'species', 'Name', species_name);
    end

    if ~isempty(species_obj)
        species_obj(1).InitialAmount = count_val;
        if verbose
            fprintf('  %-25s = %.3e cells\n', display_name, count_val);
        end
    elseif verbose
        fprintf('  %-25s = SPECIES NOT FOUND\n', display_name);
    end
end

% Vasculature (K = tumor_cells at start)
K_obj = sbioselect(model, 'Type', 'species', 'Name', 'K', 'Parent', V_T_comp);
if ~isempty(K_obj)
    K_obj(1).InitialAmount = tumor_cells;
    if verbose
        fprintf('  %-25s = %.3e cells\n', 'K (vasculature)', tumor_cells);
    end
end

% Collagen
col_species = sbioselect(model, 'Type', 'species', 'Name', 'collagen');
if ~isempty(col_species)
    col_species(1).InitialAmount = collagen_mass;
    if verbose
        fprintf('  %-25s = %.3e mg (%.1f%% volume)\n', ...
            'collagen', collagen_mass, hs.collagen.volume_fraction * 100);
    end
end

% =========================================================================
% STEP 4: Set cytokine concentrations
% =========================================================================
if verbose
    fprintf('\nCytokine concentrations:\n');
end

c = hs.basal_cytokines;
cytokine_species = {
    'CCL2',    c.CCL2_nM,          'CCL2 (basal)',  'nM';
    'TGFb',    c.TGFb_nM,          'TGF-beta',      'nM';
    'VEGF',    c.VEGF_pg_per_mL,   'VEGF (normal)', 'pg/mL';
};

for i = 1:size(cytokine_species, 1)
    species_name = cytokine_species{i, 1};
    conc_val     = cytokine_species{i, 2};
    display_name = cytokine_species{i, 3};
    unit_str     = cytokine_species{i, 4};

    species_obj = sbioselect(model, 'Type', 'species', 'Name', species_name, 'Parent', V_T_comp);
    if ~isempty(species_obj)
        species_obj(1).InitialAmount = conc_val;
        if verbose
            fprintf('  %-25s = %.3e %s\n', display_name, conc_val, unit_str);
        end
    elseif verbose
        fprintf('  %-25s = SPECIES NOT FOUND\n', display_name);
    end
end

% =========================================================================
% STEP 5: Zero out treatment-specific and dynamic species
% =========================================================================
if verbose
    fprintf('\nZeroing treatment/dynamic species:\n');
end

zero_species = hs.zero_species;
if ~iscell(zero_species); zero_species = {zero_species}; end

for i = 1:length(zero_species)
    species_name = zero_species{i};
    species_obj = sbioselect(model, 'Type', 'species', 'Name', species_name, 'Parent', V_T_comp);
    if isempty(species_obj)
        species_obj = sbioselect(model, 'Type', 'species', 'Name', species_name);
    end

    if ~isempty(species_obj)
        species_obj(1).InitialAmount = 0;
        if verbose
            fprintf('  %-25s = 0\n', species_name);
        end
    end
end

if verbose
    fprintf('\nEarly microinvasive populations set.\n');
    fprintf('Ready for natural history evolution (early -> diagnosis).\n');
    fprintf('=========================================\n');
end

end
