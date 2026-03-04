# QSP Enum Integration - LymphCentral_wrapper Example

## Overview

This document shows how to integrate the `QSPParam` enum-based access into your `LymphCentral_wrapper` class for QSP-ABM coupling.

## File Structure

```
/PDAC/qsp/
├── ode/
│   ├── Param.h                    # Original parameter class
│   ├── Param.cpp
│   ├── QSPParam.h    (NEW)        # Enum-based wrapper
│   ├── ODE_system.h
│   └── ODE_system.cpp
├── cvode/
│   ├── CVODEBase.h
│   └── CVODEBase.cpp
└── LymphCentral_wrapper.h         # (MODIFY)
└── LymphCentral_wrapper.cpp       # (MODIFY)
```

## Step 1: Update LymphCentral_wrapper.h

**File: `/PDAC/qsp/LymphCentral_wrapper.h`**

### Current Version (Before)
```cpp
#ifndef LYMPHCENTRAL_WRAPPER_H
#define LYMPHCENTRAL_WRAPPER_H

#include "cvode/CVODEBase.h"
#include "ode/Param.h"           // Old: Uses Param with indices
#include <vector>

class LymphCentral_wrapper {
private:
    CVODEBase _cvode_solver;
    Param _qsp_params;              // Old: Uses index-based access
    // ...
};
```

### Updated Version (After)
```cpp
#ifndef LYMPHCENTRAL_WRAPPER_H
#define LYMPHCENTRAL_WRAPPER_H

#include "cvode/CVODEBase.h"
#include "ode/QSPParam.h"            // NEW: Use enum-based wrapper
#include <vector>

namespace PDAC {

// Define the QSP state that gets passed to ABM
struct QSPState {
    // Tumor T cell populations
    double tumor_naive_t = 0.0;
    double tumor_activated_t = 0.0;
    double tumor_cytotoxic_t = 0.0;

    // Tumor cytokine concentrations
    double tumor_ifng_conc = 0.0;
    double tumor_il2_conc = 0.0;
    double tumor_il10_conc = 0.0;
    double tumor_tgfb_conc = 0.0;
    double tumor_il12_conc = 0.0;

    // Tumor immunosuppression
    double tumor_mdsc_count = 0.0;
    double tumor_argi_conc = 0.0;
    double tumor_no_conc = 0.0;

    // Drug concentrations (from QSP)
    double nivo_tumor_conc = 0.0;
    double cabo_tumor_conc = 0.0;
};

class LymphCentral_wrapper {
public:
    LymphCentral_wrapper();
    ~LymphCentral_wrapper();

    // Initialize from XML parameter file
    void initialize(const std::string& param_file);

    // Advance ODE system one timestep
    void time_step(double t, double dt);

    // Get current QSP state for ABM
    QSPState get_state_for_abm() const;

    // Update QSP from ABM feedback (cancer deaths, T cell recruitment, etc.)
    void update_from_abm(const ABMFeedback& feedback);

private:
    CVODEBase _cvode_solver;
    CancerVCT::QSPParam _qsp_params;  // NEW: Use enum-based wrapper!

    // Extract indices for frequently-used species
    void _extract_species_indices();

    // Apply ABM-originated changes to ODE system state
    void _apply_abm_feedback(const ABMFeedback& feedback);

    // Species indices for internal ODE access
    int _idx_V_T_T0;      // Naive T cells in tumor
    int _idx_V_T_T1;      // Activated T cells in tumor
    int _idx_V_T_C1;      // Cytotoxic T cells in tumor
    int _idx_V_T_IFNg;    // IFN-gamma in tumor
    int _idx_V_T_IL2;     // IL-2 in tumor
    // ... more indices as needed
};

} // namespace PDAC

#endif
```

## Step 2: Update LymphCentral_wrapper.cpp

**File: `/PDAC/qsp/LymphCentral_wrapper.cpp`**

### Constructor and Initialization

```cpp
#include "LymphCentral_wrapper.h"
#include <iostream>

namespace PDAC {

LymphCentral_wrapper::LymphCentral_wrapper()
    : _idx_V_T_T0(-1), _idx_V_T_T1(-1), _idx_V_T_C1(-1),
      _idx_V_T_IFNg(-1), _idx_V_T_IL2(-1)
{
    // Constructor - defer initialization to initialize()
}

void LymphCentral_wrapper::initialize(const std::string& param_file) {
    std::cout << "Initializing LymphCentral_wrapper from: " << param_file << std::endl;

    // STEP 1: Load parameters from XML using QSPParam (enum-based!)
    try {
        _qsp_params.initializeParams(param_file);
        std::cout << "✓ QSP parameters loaded from XML" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "✗ Failed to load QSP parameters: " << e.what() << std::endl;
        throw;
    }

    // STEP 2: Extract species indices from ODE system
    _extract_species_indices();

    // STEP 3: Initialize CVODE solver with ODE_system
    // TODO: Setup initial conditions from _qsp_params
    // _cvode_solver.initialize(/* ODE system */, /* initial conditions */);

    std::cout << "✓ LymphCentral_wrapper initialized" << std::endl;
}

void LymphCentral_wrapper::_extract_species_indices() {
    // Extract index mappings for ODE_system species
    // This example assumes you have an ODE_system with named species access
    // TODO: Implement based on your ODE_system structure

    std::cout << "Extracting QSP species indices..." << std::endl;

    // For now, placeholder indices - update based on your ODE system
    _idx_V_T_T0 = 0;      // Replace with actual index
    _idx_V_T_T1 = 1;
    _idx_V_T_C1 = 2;
    _idx_V_T_IFNg = 10;
    _idx_V_T_IL2 = 11;
}
```

### Getting QSP State for ABM (Key Method)

```cpp
QSPState LymphCentral_wrapper::get_state_for_abm() const {
    QSPState state;

    // ========================================================================
    // KEY: Using enum-based access to QSP parameters!
    // ========================================================================

    // TUMOR T CELL POPULATIONS
    // Old way: state.tumor_naive_t = _qsp_params.getVal(50);  // Magic number!
    // New way:
    state.tumor_naive_t = _qsp_params.getFloat(CancerVCT::QSP_V_T_T0);
    state.tumor_activated_t = _qsp_params.getFloat(CancerVCT::QSP_V_T_T1);
    state.tumor_cytotoxic_t = _qsp_params.getFloat(CancerVCT::QSP_V_T_C1);

    // TUMOR CYTOKINE CONCENTRATIONS
    // Or use convenience methods for clarity:
    state.tumor_ifng_conc = _qsp_params.getTumorCytokine_IFNg();
    state.tumor_il2_conc = _qsp_params.getTumorCytokine_IL2();
    state.tumor_il10_conc = _qsp_params.getTumorCytokine_IL10();
    state.tumor_tgfb_conc = _qsp_params.getTumorCytokine_TGFb();
    state.tumor_il12_conc = _qsp_params.getFloat(CancerVCT::QSP_V_T_IL12);

    // TUMOR IMMUNOSUPPRESSION
    state.tumor_mdsc_count = _qsp_params.getTumorImmune_MDSC();
    state.tumor_argi_conc = _qsp_params.getTumorImmune_ArgI();
    state.tumor_no_conc = _qsp_params.getTumorImmune_NO();

    // DRUG CONCENTRATIONS (from QSP if tracked)
    // TODO: Add enum entries for NIVO_tumor, CABO_tumor if QSP tracks them
    // state.nivo_tumor_conc = _qsp_params.getFloat(CancerVCT::QSP_V_T_NIVO);

    return state;
}
```

### Time Stepping

```cpp
void LymphCentral_wrapper::time_step(double t, double dt) {
    // Advance ODE system by dt
    if (_cvode_solver.isInitialized()) {
        _cvode_solver.solve(t, dt);
        // Parameters are updated in the ODE system state
    }
}
```

### Applying ABM Feedback

```cpp
void LymphCentral_wrapper::update_from_abm(const ABMFeedback& feedback) {
    std::cout << "Applying ABM feedback to QSP:" << std::endl;
    std::cout << "  - Cancer deaths: " << feedback.cancer_death_count << std::endl;
    std::cout << "  - T cells recruited: " << feedback.tcell_recruitment << std::endl;

    _apply_abm_feedback(feedback);
}

void LymphCentral_wrapper::_apply_abm_feedback(const ABMFeedback& feedback) {
    // ========================================================================
    // Update QSP state based on ABM changes
    // ========================================================================

    // Example 1: T cells killed cancer → signal to activate more lymph node T cells
    if (feedback.cancer_death_count > 0) {
        // Get current lymph node T0 population using enum
        double ln_t0 = _qsp_params.getFloat(CancerVCT::QSP_V_LN_T0);

        // Simulate activation signal
        double activation_signal = feedback.cancer_death_count * 0.001;  // Per cancer cell

        // Update ODE system (TODO: implement setSpecies method)
        // _cvode_solver.setSpecies(_idx_V_LN_T0, ln_t0 + activation_signal);
    }

    // Example 2: ABM reports low T cell effectiveness → might be due to suppression
    if (feedback.tcell_effectiveness_ratio < 0.5) {
        // Check tumor immunosuppression levels
        double mdsc_count = _qsp_params.getTumorImmune_MDSC();
        double argi_conc = _qsp_params.getTumorImmune_ArgI();

        std::cout << "Low T cell effectiveness - tumor suppressors: "
                  << "MDSC=" << mdsc_count << ", ArgI=" << argi_conc << std::endl;

        // Could trigger feedback to increase T cell recruitment or reduce MDSC
    }

    // Example 3: ABM reports T cell recruitment need
    if (feedback.tcell_recruitment > 0) {
        // Move T cells from lymph node to tumor
        double ln_t0_current = _qsp_params.getFloat(CancerVCT::QSP_V_LN_T0);
        double t0_tumor_current = _qsp_params.getFloat(CancerVCT::QSP_V_T_T0);

        double movement = feedback.tcell_recruitment;

        std::cout << "Moving " << movement << " T cells to tumor" << std::endl;

        // TODO: Update ODE system
        // _cvode_solver.setSpecies(_idx_V_LN_T0, ln_t0_current - movement);
        // _cvode_solver.setSpecies(_idx_V_T_T0, t0_tumor_current + movement);
    }
}
```

## Step 3: Integration into main.cu

**File: `/PDAC/sim/main.cu`** (modifications)

```cpp
#include "../qsp/LymphCentral_wrapper.h"

int main(int argc, const char** argv) {
    // ... existing GPU parameter loading ...

    // NEW: Initialize QSP-ABM coupling
    PDAC::LymphCentral_wrapper qsp_wrapper;
    std::string qsp_param_file = "/path/to/param_all_test.xml";  // Same XML!
    qsp_wrapper.initialize(qsp_param_file);

    // ... main simulation loop ...
    for (int step = 0; step < num_steps; ++step) {
        // STEP 1: Run ABM for this timestep
        simulation.simulate();

        // STEP 2: Collect ABM feedback
        PDAC::ABMFeedback feedback = collectABMFeedback(simulation);

        // STEP 3: Update QSP with ABM feedback
        qsp_wrapper.update_from_abm(feedback);

        // STEP 4: Advance QSP/ODE system
        qsp_wrapper.time_step(current_time, dt);

        // STEP 5: Get updated QSP state and push to GPU environment
        PDAC::QSPState qsp_state = qsp_wrapper.get_state_for_abm();
        updateGPUEnvironmentFromQSP(simulation, qsp_state);
    }
}

void updateGPUEnvironmentFromQSP(
    flamegpu::CUDASimulation& sim,
    const PDAC::QSPState& qsp_state)
{
    // Push QSP state into FLAMEGPU environment for agents to read
    sim.EnvironmentDescription().setProperty<float>("qsp_ifng", qsp_state.tumor_ifng_conc);
    sim.EnvironmentDescription().setProperty<float>("qsp_il2", qsp_state.tumor_il2_conc);
    sim.EnvironmentDescription().setProperty<float>("qsp_mdsc", qsp_state.tumor_mdsc_count);
    // ... etc for other parameters ...
}
```

## Comparison: Before vs After

### Before (Index-Based)
```cpp
// Hard to understand, error-prone
double ifng = _qsp_params.getVal(47);      // What is 47?
double il2 = _qsp_params.getVal(48);       // Adjacent to 47?
double cancer_deaths = feedback.count;
// ... complex logic to figure out which index to update ...

// Easy to make mistakes - no IDE support, no compiler checking
```

### After (Enum-Based)
```cpp
// Clear, self-documenting, type-safe
double ifng = _qsp_params.getFloat(CancerVCT::QSP_V_T_IFNg);
double il2 = _qsp_params.getTumorCytokine_IL2();
double cancer_deaths = feedback.cancer_death_count;

// IDE autocomplete helps you find the right parameter
// Compiler checks that the enum exists
// Easy to search codebase for all usages
```

## Key Benefits in Context

1. **Readability**: `getTumorCytokine_IFNg()` vs `getVal(47)`
2. **Safety**: Compiler ensures enum exists, IDE provides autocomplete
3. **Maintainability**: Search for `QSP_V_T_IFNg` finds all uses
4. **Documentation**: Enum names document what each value means
5. **Debuggability**: Print statements show meaningful names
6. **Extensibility**: Adding new parameters is straightforward

## Next Steps

1. **Extract indices** from Param.cpp using the extraction guide
2. **Fill in enum** in QSPParam.h with correct indices
3. **Add convenience methods** in QSPParam for frequently-used parameters
4. **Implement the TODOs** in LymphCentral_wrapper (setSpecies, etc.)
5. **Test** by running with sample XML and verifying parameter values

The enum-based access will make your QSP-ABM coupling code much more maintainable!
