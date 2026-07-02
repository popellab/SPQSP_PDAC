# QSPIO PDAC

[![Python Tests](https://github.com/jeliason/qspio-pdac/actions/workflows/python-tests.yml/badge.svg)](https://github.com/jeliason/qspio-pdac/actions/workflows/python-tests.yml)
[![codecov](https://codecov.io/gh/jeliason/qspio-pdac/branch/main/graph/badge.svg)](https://codecov.io/gh/jeliason/qspio-pdac)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A quantitative systems pharmacology (QSP) modeling framework for pancreatic ductal adenocarcinoma (PDAC) immunotherapy simulation, calibration, and analysis.

## Overview

This MATLAB/SimBiology + Python framework provides a platform for simulating immune responses to PDAC therapies. Built on the core QSPIO framework, this repository contains PDAC-specific model modules, literature-derived calibration targets, and simulation-based inference (SBI) workflows. The model captures PDAC tumor dynamics, immune cell populations (T cells, macrophages, DCs, MDSCs), fibroblast/ECM biology, and pharmacokinetics for immunotherapy combinations.

## Quick Start

### MATLAB (model simulation)

```matlab
startup  % Add paths and configure environment

% Build the compiled SimBiology model (creates model object in workspace)
run scripts/immune_oncology_model_PDAC

% Run baseline vs treatment comparison (builds model, simulates, plots)
run scripts/cascade_figures
```

### Python (inference and analysis)

```bash
source .venv/bin/activate  # or use .venv/bin/python directly

# Run tests
pytest tests/ -m "not slow"   # Fast tests (~0.3s)
pytest tests/ -m slow          # Includes MATLAB simulations

# PRCC sensitivity analysis
python workflows/prcc_sensitivity.py

# SBI calibration
python workflows/sbi_workflow.py
```

## Model Components

The model uses a four-compartment architecture (Central, Peripheral, Tumor, Lymph Node) with modular biology:

- **Tumor dynamics**: PDAC growth, antigen release, immune evasion
- **T cells**: CD8 effector/exhausted, CD4 Th, Treg (naive/effector), trafficking between compartments
- **Dendritic cells**: cDC1 (BATF3+, cross-presentation to CD8) and cDC2 (IRF4+, CD4 Th/Treg priming) with subtype-specific maturation
- **Macrophages**: M1/M2 polarization, recruitment, cytokine secretion
- **MDSCs**: Immunosuppression and recruitment
- **Fibroblasts/ECM**: Three CAF subtypes (iCAF, myCAF, apCAF) with IL-1/TGFb-driven interconversion, collagen volume fraction (phi_collagen) driving pore-size immune exclusion and stiffness mechanotransduction
- **Chemokines**: CCL5/CCR5 (Treg recruitment), CXCL12 (T cell exclusion), CCL2 (APC recruitment), IL-1/IL-6/IL-10/IL-12/TGFb/IFNg/VEGF
- **Therapies**: GVAX vaccine, nivolumab (anti-PD-1), combination protocols

## Calibration

Calibration targets are extracted from literature via [`maple`](https://github.com/popellab/maple) with full provenance (DOI, value snippets, experimental context).

| Category | Count | Description |
|----------|------:|-------------|
| Baseline TME (diagnosis) | 22 | Cell densities, cytokines, immune ratios, ECM, DC subtypes |
| GVAX+nivo neoadjuvant | 9 | CD8 fold change, M1/M2 shift, Treg fraction (Zheng 2022, Blair 2023) |
| GVAX monotherapy | 6 | Cell composition changes at day 21 |
| Clinical progression | 1 | Tumor doubling time |
| Submodel targets | 106 | In vitro/ex vivo data constraining individual parameters (fibroblast/ECM, macrophage, T cell, cytokine EC50s) |
| Qualitative bounds | 20 | Literature-backed structural bounds with provenance validation |

Prior distributions for SBI are in `parameters/pdac_priors.csv` (194 parameters, log-normal with sigma 0.5-1.2 based on parameter-specific uncertainty).

### SBI Workflow

1. **Sensitivity screening**: PRCC ranks 194 parameters across 35 observables (111 FDR-significant, 3 tiers)
2. **Submodel calibration**: Component-wise Bayesian inference on 106 isolated-system targets (`python scripts/parameter_audit.py`)
3. **Joint NPE training**: Multi-scenario neural posterior estimation with PRCC tier filtering
4. **Validation**: SBC rank diagnostics, posterior predictive checks (PPC), qualitative bounds

Simulation pools (`cache/sbi_simulations/`) accumulate expensive QSP runs across training iterations. See `notes/calibration/calibration_strategy.md` for the full workflow.

## File Structure

```
core/                    # Core MATLAB framework
  modules/               # SimBiology modules (GVAX, fibroblast, macrophage, APC, etc.)
  visualization/         # Plotting functions
  model_compilation/     # Model building utilities
  postprocessing/        # Analysis workflows
  utils/                 # Helpers
scripts/                 # PDAC model scripts
  immune_oncology_model_PDAC.m   # Main model assembly
  cascade_figures.m      # Baseline vs treatment comparison plots
  export_sbml.m          # SBML export
  test_*.m               # Module testing
parameters/              # Parameter files and prior distributions
  parameters_PDAC.m      # Master parameter definitions
  pdac_priors.csv        # SBI prior distributions
scenarios/               # Treatment scenario YAML configurations
calibration_targets/     # Calibration target YAMLs by scenario
  baseline_no_treatment/ # Diagnosis-state TME targets
  gvax_nivo_neoadjuvant/ # GVAX+nivo treatment response targets
  gvax_neoadjuvant/      # GVAX monotherapy targets
  clinical_progression/  # Clinical outcome targets
  submodel_targets/      # IsolatedSystemTarget YAMLs (in vitro/ex vivo)
  reference_values.yaml  # Shared physical constants for unit conversions
workflows/               # Python inference and analysis
  sbi_workflow.py        # Multi-scenario joint SBI (primary)
  prcc_sensitivity.py    # PRCC sensitivity analysis
  sequential_sbi_workflow.py     # SNPE-C multi-round inference
  hierarchical_sbi_workflow.py   # Hierarchical SBI (population mu+omega)
  npe_capacity_sweep.py  # NSF architecture search
  ablation_study.py      # Noise vs parameter count ablation
  morris_sensitivity.py  # Morris screening
  surrogate_sensitivity.py       # GP surrogate sensitivity
  restriction_estimator.py       # Prior restriction diagnostics
  misspec_diagnostic.py  # Model misspecification checks
tests/                   # Python test suite (pytest)
  qualitative_bounds.yaml        # 20 structural bounds with evidence snippets
notes/                   # Working notes and planning documents
presentations/           # Marp markdown slides
hooks/                   # Git hooks (SBML auto-export on model changes)
species_units.json       # Species-to-unit mapping for Pint conversions
model_structure.json     # Exported model structure (parameters, species, rules)
PDAC_model.sbml          # SBML export of compiled model
```

## Requirements

- MATLAB R2020b+ with SimBiology and Statistics toolboxes
- Python 3.10+ (see `requirements.txt`)
  - [`qsp-inference`](https://github.com/jeliason/qsp-inference) (submodel MCMC, SBI diagnostics, parameter audit)
  - [`maple`](https://github.com/popellab/maple) (calibration target schemas, extraction)
  - `qsp-hpc-tools` (simulation pooling, HPC batch jobs)
  - `sbi`, `torch` (neural posterior estimation)
  - `SALib` (sensitivity analysis)
  - `pandas`, `numpy`, `scipy`, `PyYAML`

## License

MIT License - see [LICENSE](LICENSE) file for details.