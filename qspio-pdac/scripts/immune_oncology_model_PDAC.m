%% Immune Oncology Model
% Script for setting up and running the immune oncology model in simbiology

%% Add use-defined units into both Simbiology and Symbolic libraries
% Add 'cell' unit to SimBiology and Symbolic Toolboxes
if (isempty(sbioshowunits('cell')))
    cell_unit = sbiounit('cell','molecule');
    sbioaddtolibrary(cell_unit);
end
% Symbolic Unit
u = symunit;
try u.cell;
catch
    newUnit('cell',u.molecule);
end

% Add 'meter_mercury' unit so 'millimeter_mercury' (milli prefix) resolves
if (isempty(sbioshowunits('meter_mercury')))
    mmHg_base = sbiounit('meter_mercury','molecule');
    sbioaddtolibrary(mmHg_base);
end

% Add 'mU' unit to SimBiology and Symbolic Toolboxes
mU_unit = sbiounit('mU','mole/liter');
sbioaddtolibrary(mU_unit);
% Symbolic Unit
u = symunit;
try u.mU;
catch
    newUnit('mU',u.molarity);
end


%% Setup Parameters
% Setup Parameters using hybrid YAML+Legacy approach (includes calculated parameters)
params_in = parameters_PDAC;
params_out = load_parameters(params_in);

%% Create the SimBiology Model
% Model Settings
model_name = 'Immune Oncology Model';
start_time = 0.0; % [days]
time_step = 0.1; % [days] 0.01 days ~ 15 mins
end_time = 365; % [days]
absolute_tolerance = 1e-9;
relative_tolerance = 1e-6;
% solver = 'ode15s';
solver = 'sundials';
% Model Object
time = start_time:time_step:end_time;
model = simbio_init(model_name,time,solver,absolute_tolerance,relative_tolerance,params_out);
%model = simbio_init(model_name,time,params_out);
% Maximal simulation time
config = getconfigset(model);
set(config, 'MaximumWallClock', 120)
% set(config, 'MaximumNumberOfLogs', 1)
set(config.SolverOptions, 'AbsoluteToleranceScaling', false)

%% Add Modules to the Model
% Cancer Modules
% Initial amount is a placeholder; set_baseline_populations sets actual value from density priors
model = cancer_module(model,'C1',params_out,1,'Gompertzian');
model = cancer_module(model,'C2',params_out,0,'Gompertzian'); % chemotherapy-resistant clone (growth/death tied to C1 in resistance_module)
% T cell Modules
model = Treg_module(model,params_out);
model = CCL5_Treg_module(model,params_out); % CCL5/CCR5-mediated Treg recruitment (Tan 2017)
model = Teff_module(model,params_out,{'C1'});
% APC Module
model = APC_module(model,params_out);
% Antigen Modules
antigenCP = create_antigen({'C1'},[5.4e-13],'antigenID',0);
model = antigen_module(model,'0',params_out,antigenCP);
antigen   = create_antigen({'C1'},[5.4e-13],'antigenID',1);
model = antigen_module(model,'1',params_out,antigen);
% PK Modules
params_aPD1    = pk_parameters('nivolumab');
params_aPDL1   = pk_parameters('durvalumab');
params_aCTLA4  = pk_parameters('tremelimumab');
model = pk_module(model,'aPD1',params_aPD1);
model = pk_module(model,'aPDL1',params_aPDL1,'n');
model = pk_module(model,'aCTLA4' ,params_aCTLA4);
% Checkpoint Modules
model = checkpoint_module(model,params_out,'CD8','C1');
%model = checkpoint_module(model,params_out,'CD8','C2');
model = checkpoint_module(model,params_out,'CD8','APC');

% QSPIO-HCC Modules
model = Th_module(model,params_out);
model = cytokine_tcell_module(model, params_out);  % Centralized T-cell cytokine reactions (IL2, IFNg, TGFb)
% Humoral / TLS pool (coupled QSP+ABM completion): Tfh helper subset + minimal
% B/plasma pool. After cytokine_tcell_module (needs N_aTh) and Th_module (needs
% k_CD4_pro/k_CD4_death); before aggregate_observables_module.
model = Tfh_module(model,params_out);
model = Bcell_module(model,params_out);
model = MDSC_module(model,params_out,{'C1'},'inostat',0,'drugName','entinostat');
model = nabpaclitaxel_module(model,params_out);  % trimmed: PK + Emax kill (no resistance/vascular)
% model = cabozantinib_module(model,params_out); % Disabled due to numerical instability
% Cytotoxic chemotherapy PK-PD (PASS-01: mFOLFIRINOX + gemcitabine/nab-paclitaxel).
% Faithful per-drug popPK + Emax kill on V_T.C1. Modules are ADDED unconditionally;
% which drugs are actually dosed is set per-scenario (schedule_dosing). Immuno PK
% (aPD1/aPDL1/aCTLA4) is kept inert (never dosed) per PASS-01 (chemo-only trial).
model = chemo_pk_module(model, 'gem', params_out, 1, 'linear', '');     % gemcitabine (GnP)
model = chemo_pk_module(model, 'fu5', params_out, 0, 'mm', '');         % 5-fluorouracil (mFFX)
model = chemo_pk_module(model, 'oxa', params_out, 2, 'linear', '');     % oxaliplatin (mFFX)
model = chemo_pk_module(model, 'iri', params_out, 1, 'linear', 'sn38'); % irinotecan->SN-38 (mFFX)
model = macrophage_module(model,params_out,{'C1'},'aCD47',0); % PK module in development for aCD47
% Fibroblast Module
model = fibroblast_module_enhanced(model,params_out);

model = gvax_module(model, params_out, 'add_cyclophosphamide', true);

% Aggregate cytotoxic drug effect E_chemo (complement-product of per-drug Emax).
% Must run AFTER all 5 chemo drug modules (needs each Ct_<drug>/IC50_<drug>).
% Shared downstream by resistance selection (C1->C2) and CA19-9 secretion suppression.
model = chemo_effect_module(model);

% Chemo-resistant clone coupling: C1->C2 drug-selected transition + growth/death
% tie to C1. After cancer_module('C2') (needs V_T.C2, k_C2_*) and chemo_effect_module
% (needs E_chemo); the per-drug C2 kill reactions live in the drug modules.
model = resistance_module(model, params_out);

% CA19-9 serum biomarker (secretion by live burden + shed from dying pool).
% After cancer_module (needs C1/C_total/C_x) and chemo_effect_module (needs E_chemo);
% standalone serum readout in V_C.
model = ca199_module(model, params_out);

% Aggregate observables (derived totals for cal-target denominators).
% Must run after all species-adding modules so nucleated_total_T can
% reference every V_T cell species defined above.
model = aggregate_observables_module(model);

%% Baseline Populations
% Note: set_baseline_populations() is called in initial_conditions() after
% variant parameters are applied. This ensures IC fraction parameters from
% priors/variants are used correctly.
