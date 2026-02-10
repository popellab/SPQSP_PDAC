#include "flamegpu/flamegpu.h"
#include "../qsp/LymphCentral_wrapper.h"

namespace PDAC{

// Global pointer to the lymph wrapper (initialized in main)
static LymphCentralWrapper* g_lymph = nullptr;

void set_lymph_pointer(LymphCentralWrapper* lymph) {
    g_lymph = lymph;
}

FLAMEGPU_HOST_FUNCTION(solve_qsp_step) {
    if (!g_lymph) return;
    
    // Get current simulation time/step from environment
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    
    // Get ABM state to pass to QSP (aggregate quantities from agents)
    
    // Update QSP inputs from ABM state
    // g_lymph->setABMInputs(cancer_count, tcell_count, ...);
    
    // Solve ODE for one ABM timestep (CPU-based)
    double dt_abm = FLAMEGPU->environment.getProperty<float>("dt_abm");
    g_lymph->time_step(static_cast<float>(step), dt_abm);
    
    // Get QSP outputs and update environment for ABM to use
    // These can drive recruitment rates, drug concentrations, etc.
    // FLAMEGPU->environment.setProperty<float>("qsp_drug_conc", g_lymph->getDrugConcentration());
    // FLAMEGPU->environment.setProperty<float>("tcell_recruitment_rate", g_lymph->getTcellRecruitment());
}

}