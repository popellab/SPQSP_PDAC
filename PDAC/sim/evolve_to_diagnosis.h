#ifndef PDAC_EVOLVE_TO_DIAGNOSIS_H
#define PDAC_EVOLVE_TO_DIAGNOSIS_H

// Port of pdac-build/scripts/evolve_to_diagnosis.m. Sets the ODE state to
// a healthy early-microinvasive-PDAC initial condition, then integrates
// forward (no doses, no events beyond what the generated ODE emits) until
// the V_T diameter reaches the target. Returns the diagnosis time in days.
//
// Shared knobs (min/max evolution time, default tumor cells, step size)
// live in PDAC/sim/resource/healthy_state.yaml. Same YAML is read by the
// MATLAB oracle, so both sides evolve from the same biological spec.

#include <string>

namespace CancerVCT {

class ODE_system;

struct EvolveOpts {
    std::string yaml_path;          // healthy_state.yaml absolute path (required)
    double target_diameter_cm = -1; // if <= 0, read initial_tumour_diameter from ODE params
    double tumor_cells = -1;        // if <= 0, read default_tumor_cells from YAML
    double time_factor = 86400.0;   // 86400 (SI sec) or 1.0 (model-unit days) — must
                                    // match the caller's simOdeStep time axis
    bool verbose = false;
};

struct EvolveResult {
    bool success = false;           // true iff target diameter reached within guards
    double t_diagnosis_days = 0.0;  // time (days) at which diameter crossed target
    double diameter_cm = 0.0;       // V_T diameter at stop
    std::string reject_reason;      // populated when success == false
};

// Mutates `ode`: replaces ICs with the healthy state, then integrates to
// diagnosis. Caller owns `ode` and can continue simulating from the
// returned state (e.g. treatment scenario).
EvolveResult evolve_to_diagnosis(ODE_system& ode, const EvolveOpts& opts);

// Compute V_T diameter (cm) from current ODE state. V_T volume is in mL
// (= cm³) so diameter = 2 * cbrt(3 V / (4π)).
double current_vt_diameter_cm(const ODE_system& ode);

}  // namespace CancerVCT

#endif