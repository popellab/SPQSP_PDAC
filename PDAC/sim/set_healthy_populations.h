#ifndef PDAC_SET_HEALTHY_POPULATIONS_H
#define PDAC_SET_HEALTHY_POPULATIONS_H

// Port of pdac-build/scripts/set_healthy_populations.m. Initializes V_T
// species to early-microinvasive-PDAC densities so evolve_to_diagnosis
// has a healthy starting point. Densities, ratios, and cytokine levels
// come from PDAC/sim/resource/healthy_state.yaml (shared with the MATLAB
// validation oracle).

#include <string>

namespace CancerVCT {

class ODE_system;

struct HealthyPopulationOpts {
    // Absolute path to healthy_state.yaml. Required.
    std::string yaml_path;
    // Initial cancer-cell count. If <= 0, read default_tumor_cells from YAML.
    double tumor_cells = -1.0;
    // Print each species value as it's written (for parity logs).
    bool verbose = false;
};

// Write the healthy initial condition into `ode._species_var` and recompute
// assignment-rule variables. Caller should call simOdeStep only after this.
// Throws std::runtime_error on YAML / species-index errors.
void set_healthy_populations(ODE_system& ode, const HealthyPopulationOpts& opts);

}  // namespace CancerVCT

#endif