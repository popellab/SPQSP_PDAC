# Parameter Sensitivity Analysis (PSA) Framework

## Overview

The PSA framework provides a comprehensive system for conducting virtual clinical trials and parameter sensitivity analysis using SimBiology models. The framework supports Latin Hypercube Sampling, batch simulations, patient filtering, output extraction, and statistical analysis.

## Framework Architecture

### Core Workflow Functions

**`PSA_config.m`** - Central configuration management
- Defines analysis settings (patients, PRCC method, significance)
- Loads output and filter configurations
- Provides single point of configuration control

**`PSA_generate_parameters.m`** - Parameter sampling
- Latin Hypercube Sampling (LHS) for virtual patient generation
- Supports parameter distributions and constraints
- Reproducible sampling with optional seed control
- Creates parameter variants for SimBiology models

**`PSA_simulate.m`** - Batch simulation engine
- Runs simulations across virtual patient cohorts
- Intelligent caching with seed-based cache keys
- Manual cache overwrite and project name options
- Handles simulation failures and error tracking
- Project-specific result storage

**`PSA_postprocess.m`** - Result processing
- Calculates derived metrics (clonality, ratios, tumor changes)
- Extracts biomarkers and clinical endpoints
- Prepares data for downstream analysis

**`PSA_filter_patients.m`** - Patient filtering
- Applies plausibility criteria to simulation results
- Filters based on physiological constraints
- Maintains metadata on filtering decisions

**`PSA_extract_outputs.m`** - Output extraction
- Extracts specified outputs from filtered results
- Supports multiple output types and time points
- Formats data for sensitivity analysis

**`PSA_calculate_endpoints.m`** - Clinical endpoints
- RECIST response evaluation
- Progression-free survival calculations
- Biomarker analysis and correlations

### Configuration Functions

**`PSA_output_config.m`** - Output specifications
- Defines outputs for sensitivity analysis
- Specifies time points and measurement types
- Configurable for different analysis needs

**`PSA_filter_config.m`** - Patient filtering criteria
- Defines plausibility filters
- Physiological constraint specifications
- Customizable filtering logic

### Caching System

The framework uses a simplified, seed-based caching system that ensures reproducible results across MATLAB sessions.

**Key Features:**
- Automatic cache key generation from parameters, config, and random seed
- Project-specific cache storage in `projects/{name}/cache/PSA_simulations/`
- Manual cache invalidation with `'overwrite', true` option
- Persistent caching across MATLAB sessions when using seeds

## Usage Examples

### Basic PSA Workflow

```matlab
%% 1. Configuration
psa_config = PSA_config();
psa_config.n_patients = 500;

%% 2. Model Setup
eval(psa_config.model_script);
dose_schedule = schedule_dosing({'drug'}, 'dose', 100, 'schedule', [0,14,28]);

%% 3. Parameter Generation (with reproducible seed)
params_in = PSA_param_in_PDAC();
params_in = PSA_generate_parameters(model, params_in, psa_config.n_patients, 'seed', 12345);

%% 4. Simulation (with automatic caching and project name)
[results, metadata] = PSA_simulate(model, params_in, dose_schedule, psa_config, ...
    'project_name', 'my_project', 'seed', 12345);

%% 5. Postprocessing
postprocessed_data = PSA_postprocess(results, metadata);

%% 6. Filtering
[filtered_results, filter_metadata] = PSA_filter_patients(results, metadata, psa_config, postprocessed_data);

%% 7. Output Extraction
[output_data, output_names] = PSA_extract_outputs(results, filtered_results, psa_config, postprocessed_data);

%% 8. Clinical Endpoints
[endpoints, endpoint_metadata] = PSA_calculate_endpoints(results, filtered_results);
```

### Advanced Configuration

```matlab
%% Custom output configuration
psa_config = PSA_config();
psa_config.outputs.tumor_volume = struct('time', 30, 'species', 'C1');
psa_config.outputs.cd8_ratio = struct('time', 14, 'species', {'Teff', 'Treg'});

%% Custom filtering
psa_config.filters.min_tumor_size = 1e6;
psa_config.filters.max_immune_ratio = 10;
```

## Configuration System

### PSA_config Structure

```matlab
config = PSA_config();
% Basic settings
config.n_patients          % Number of virtual patients (default: 500)
config.model_script         % Model definition script name
config.prcc_method          % PRCC calculation method ('spearman')
config.significance_level   % Statistical significance (0.05)

% Analysis configuration
config.outputs              % Output specifications from PSA_output_config
config.filters              % Filter criteria from PSA_filter_config
```

### Output Configuration

Define analysis outputs in `PSA_output_config.m`:

```matlab
outputs.tumor_volume = struct(...
    'time', 30, ...
    'species', 'C1', ...
    'description', 'Tumor volume at day 30');

outputs.immune_activation = struct(...
    'time', 14, ...
    'species', {'Teff', 'APC'}, ...
    'operation', 'ratio', ...
    'description', 'T effector to APC ratio');
```

### Filter Configuration

Define patient filters in `PSA_filter_config.m`:

```matlab
filters.physiological_tumor = struct(...
    'type', 'range', ...
    'species', 'C1', ...
    'time', 0, ...
    'min_value', 1e6, ...
    'max_value', 1e10);

filters.immune_response = struct(...
    'type', 'threshold', ...
    'species', 'Teff', ...
    'time', 14, ...
    'min_value', 1e3);
```

## Caching and Reproducibility

The PSA framework includes intelligent caching to dramatically speed up repeated analyses while ensuring reproducible results.

### Reproducible Parameter Generation

Use seeds to ensure identical parameter sets across sessions:

```matlab
% Set seed for reproducible results
seed = 12345;
params_in = PSA_generate_parameters(model, params_in, n_patients, 'seed', seed);
[results, metadata] = PSA_simulate(model, params_in, dose_schedule, config, ...
    'project_name', 'my_project', 'seed', seed);
```

### How Caching Works

- **Seed-based**: Cache keys include random seed for reproducible caching
- **Automatic**: No code changes needed - caching happens transparently
- **Project-specific**: Stored in `projects/{name}/cache/PSA_simulations/`
- **Persistent**: Same seed + config = same cache key across MATLAB sessions

### Performance Benefits

- **First run**: Normal simulation time + cache creation (~1-2s overhead)
- **Subsequent runs**: 10-100x faster (typically 1-2 seconds vs minutes)
- **Identical results**: Cached results are bit-for-bit identical

### Cache Control Options

```matlab
% Force overwrite existing cache
PSA_simulate(model, params_in, dose_schedule, config, 'overwrite', true);

% Set custom project name
PSA_simulate(model, params_in, dose_schedule, config, 'project_name', 'my_analysis');

% Combine options
PSA_simulate(model, params_in, dose_schedule, config, ...
    'project_name', 'pdac_2025', 'seed', 12345, 'overwrite', false);
```

### Cache Invalidation

Caches are automatically invalidated when:
- Random seed changes
- Parameter configuration changes (distributions, bounds)
- Number of patients changes
- Model configuration changes

## File Organization

```
core/postprocessing/PSA/
├── README.md                    # This documentation
├── PSA_config.m                 # Central configuration
├── PSA_generate_parameters.m    # Parameter sampling (with seed support)
├── PSA_simulate.m               # Simulation engine (with caching & project control)
├── PSA_postprocess.m            # Result processing
├── PSA_filter_patients.m        # Patient filtering
├── PSA_extract_outputs.m        # Output extraction
├── PSA_calculate_endpoints.m    # Clinical endpoints
├── PSA_output_config.m          # Output specifications
└── PSA_filter_config.m          # Filter criteria
```

## Integration with Projects

### Project Structure
```
projects/
└── pdac_2025/
    ├── PSA_script_PDAC_GVAX.m          # Main PSA script
    ├── PSA_param_in_PDAC_GVAX.m        # Parameter definitions
    ├── immune_oncology_model_PDAC.m    # Model definition
    └── cache/
        └── PSA_simulations/             # Cached results
```

### Project-Specific Scripts

Each project should have:
1. **PSA script** - Main analysis workflow
2. **Parameter input** - Project-specific parameter distributions
3. **Model definition** - SimBiology model setup
4. **Configuration** - Custom outputs and filters (optional)

## Statistical Analysis

The framework integrates with sensitivity analysis tools:

```matlab
% PRCC analysis (requires additional functions)
PSA_PRCC(params_in, output_data, output_names, filtered_results, psa_config);

% Custom analysis
prcc_results = calculate_prcc(param_matrix, output_matrix, 0.05);
```

## Best Practices

### Configuration
- Use `PSA_config()` for consistent settings across analyses
- Customize outputs and filters in dedicated config files
- Document parameter distributions and filtering rationale

### Performance
- Use caching for iterative analysis (automatic)
- Start with smaller patient counts for development
- Monitor cache disk usage in large projects

### Reproducibility
- **Always use seeds**: Set `'seed', value` in both parameter generation and simulation
- **Consistent project names**: Use `'project_name', 'name'` for organized caching
- Save complete configuration with results
- Document model versions and parameter sources

Example reproducible workflow:
```matlab
seed = 2025;  % Set at top of script
params_in = PSA_generate_parameters(model, params_in, n_patients, 'seed', seed);
[results, metadata] = PSA_simulate(model, params_in, dose_schedule, config, ...
    'project_name', 'pdac_2025', 'seed', seed);
```

### Quality Control
- Review filter criteria and success rates
- Validate simulation results against known behaviors
- Check parameter correlation structures

## Troubleshooting

**Simulation failures**: Check `metadata.status` for error patterns
**Cache issues**: Use `PSA_cache_manager('info', path)` to inspect
**Memory issues**: Reduce `n_patients` or enable result streaming
**Performance**: Check cache hits and consider parallel computing

## Extensions

The framework supports:
- Custom output metrics in postprocessing
- Additional filter types in patient filtering
- Alternative sampling methods in parameter generation
- Integration with visualization tools

For specific implementation examples, see the project-specific PSA scripts in `projects/`.
