# QSPIO-PDAC

A quantitative systems pharmacology (QSP) model of pancreatic ductal adenocarcinoma
(PDAC) immunotherapy, built on the QSPIO framework in MATLAB/SimBiology and
exported to SBML and C++.

The model captures PDAC tumour dynamics, immune cell populations (CD8/CD4 T cells,
Tregs, macrophages, cDC1/cDC2, MDSCs), fibroblast and ECM biology (iCAF/myCAF/
apCAF, collagen), and pharmacokinetics for immunotherapy combinations — GVAX,
nivolumab, with a deliberately compact mechanism set for the retained indications.

## What is here, and what is not

This repository is **the model**. On 2026-08-19 it was split: the published-cohort
corpus, the priors, the elicitation and the Bayesian virtual-population campaign
moved to a separate analysis repository, and the SBI/NPE workflows that preceded
that campaign were retired.

| | |
|---|---|
| the model, here | `core/`, `scripts/*.m`, `cpp/`, `scenarios/`, `parameters/parameters_PDAC.m` |
| the fit, in `pdac-paper` | the corpus, `pdac_priors.csv`, the elicitation, the campaign ladder |
| the pre-split tree | `jeliason/pdac-build`, frozen, for anything retired |

The design record — mechanism reviews, the ECM and TLA investigations, the audit
documents — was removed on 2026-08-19 and lives in that frozen mirror. Comments
throughout `core/modules/*.m` still cite it by path; those resolve there.

Nothing here depends on the analysis repository at run time. The checks that
compare the two — priors against `parameters_PDAC.m`, the live priors against the
C++ parameter XML, the generated observables against the SBML — are run from
`pdac-paper` with `MODEL_ROOT` pointing at this checkout.

## Quick start

```matlab
startup                                  % paths and environment
run scripts/immune_oncology_model_PDAC   % assemble the SimBiology model
run scripts/cascade_figures              % baseline vs treatment comparison
```

```bash
make model-structure      # regenerate model_structure.json (needs MATLAB)
make export-sbml          # regenerate PDAC_model.sbml (needs MATLAB)
make refresh-cpp          # SBML -> codegen -> param XML -> build the simulator
make parity               # 365-day MATLAB <-> C++ trajectory parity
make test-fast            # the model's own tests
```

`make help` lists the rest.

## File structure

```
core/                    # the QSPIO framework, PDAC-specialised
  modules/               # SimBiology modules (GVAX, fibroblast, macrophage, APC, ...)
  model_compilation/     # model assembly
  visualization/         # the D2 schematic source and its ELK renderer
  postprocessing/        # howManyClones and simulation_config, the two still called
scripts/                 # assembly, export, and mechanism diagnostics
  immune_oncology_model_PDAC.m   # main model assembly
  export_sbml.m                  # SBML export
  evolve_to_diagnosis.m          # the diagnosis-state solve
  trace_*.m diag_*.m smoke_*.m   # mechanism traces and probes
cpp/                     # generated C++ simulator (qsp/ode) and its CMake build (sim/)
resources/cpp/           # param_all.xml, the parameter file the simulator reads
parameters/              # parameters_PDAC.m, posterior medians
scenarios/               # treatment scenario YAMLs
tests/                   # model assembly/evolve-to-diagnosis and MATLAB<->C++ parity
model_structure.json     # exported structure (parameters, species, rules)
PDAC_model.sbml          # SBML export of the compiled model
```

## Requirements

- MATLAB R2020b+ with SimBiology and Statistics toolboxes
- CMake and a C++17 compiler, for the simulator
- Python 3.10+ for the build and test helpers only (see `requirements.txt`):
  `qsp-codegen` (SBML→C++ and the parity harness), `qsp-hpc-tools`,
  numpy/scipy/pandas, pytest

## License

See `LICENSE`.
