function config = ensemble_config()
%ENSEMBLE_CONFIG Configure ensemble simulation settings
%
% This function provides configuration for running ensemble simulations
% (parameter sweeps, uncertainty quantification, PSA, SBI, etc.).
% Focuses on essential elements: outputs and filters. Simulation parameters
% are handled in the model definition (immune_oncology_model_PDAC.m).
%
% Returns:
%   config - Structure containing ensemble simulation configuration
%
% Example:
%   config = ensemble_config();
%   config.n_patients = 1000;  % Change number of simulations

%% Basic Settings
config.n_patients = 500;           % Number of virtual patients
config.model_script = 'immune_oncology_model_PDAC';  % Model definition script

%% Load Output and Filter Configurations
output_config = PSA_output_config();
config.outputs = output_config.outputs;

filter_config = PSA_filter_config();
config.filters = filter_config.filters;

%% Analysis Settings
config.prcc_method = 'spearman';        % Method for PRCC calculation
config.significance_level = 0.05;       % Significance level for correlation

fprintf('Ensemble configuration loaded: %d outputs, %d filters\n', ...
    length(fieldnames(config.outputs)), length(fieldnames(config.filters)));

end
