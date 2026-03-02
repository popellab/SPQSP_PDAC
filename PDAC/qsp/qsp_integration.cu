#include "flamegpu/flamegpu.h"
#include "../qsp/LymphCentral_wrapper.h"
#include "../qsp/ode/ODE_system.h"

#include <fstream>
#include <filesystem>
#include <chrono>
#include <nvtx3/nvToolsExt.h>

// File stream for QSP CSV output — file-scope (not in PDAC namespace) so the
// exportQSPData step function (also at file scope) can access it directly.
static std::ofstream g_qsp_csv;

namespace PDAC{

// Global pointer to the lymph wrapper (initialized in main)
static LymphCentralWrapper* g_lymph = nullptr;
double g_last_qsp_ms = 0.0;  // exposed for timing CSV

void set_lymph_pointer(LymphCentralWrapper* lymph) {
    g_lymph = lymph;
}

// Accessible from main.cu FLAMEGPU step functions via extern declaration
bool is_presim_mode_active() {
    return g_lymph ? g_lymph->is_presimulation_mode() : false;
}

// Getter used by exportQSPData (which is outside this namespace)
LymphCentralWrapper* get_lymph_pointer() {
    return g_lymph;
}

FLAMEGPU_HOST_FUNCTION(solve_qsp_step) {
    nvtxRangePush("QSP Solve");
    if (!g_lymph) { nvtxRangePop(); return; }

    // Get current simulation time/step from environment
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");

    // -----------------------------------------------------------------------
    // ABM → QSP: Compute scaler and feed back discrete events to QSP
    //
    //   abm_scaler = (1 - w) / w  ×  lymphCC / (tumCC + abm_min_cc)
    //
    //   lymphCC = QSP cancer molecule count (raw, retrieved from last state)
    //   tumCC   = ABM discrete cancer cell count
    //   w       = ODE_system::_QSP_weight (coupling weight, 0.8 default)
    // -----------------------------------------------------------------------
    const double w          = CancerVCT::ODE_system::_QSP_weight;
    const double lymphCC    = FLAMEGPU->environment.getProperty<float>("qsp_cc_tumor")
                              * FLAMEGPU->environment.getProperty<float>("AVOGADROS");
    const int    tumCC      = FLAMEGPU->environment.getProperty<unsigned int>("total_cancer_cells");
    const double abm_min_cc = FLAMEGPU->environment.getProperty<float>("PARAM_MIN_CC");
    // params.getVal(SP_QSP_IO::SP_QSP_HCC::PARAM_C1_MIN);

    double abm_scaler = 0.0;
    if (w > 0.0 && w < 1.0) {
        abm_scaler = ((1.0 - w) / w) * lymphCC / (static_cast<double>(tumCC) + abm_min_cc);
    }

    // Read ABM event counts (set by agent functions and recruitment during this step)
    const int cancer_deaths  = FLAMEGPU->environment.getProperty<int>("ABM_cc_death");
    // const int cc_death_t_kill = FLAMEGPU->environment.getProperty<int>("ABM_cc_death_t_kill");
    // const int cc_death_mac_kill = FLAMEGPU->environment.getProperty<int>("ABM_cc_death_mac_kill");
    // const int cc_death_natural = FLAMEGPU->environment.getProperty<int>("ABM_cc_death_natural");
    const int teff_recruited = FLAMEGPU->environment.getProperty<int>("ABM_TEFF_REC");
    const int th_recruited = FLAMEGPU->environment.getProperty<int>("ABM_TH_REC");
    const int treg_recruited = FLAMEGPU->environment.getProperty<int>("ABM_TREG_REC");

    // Pass scaled events to wrapper (applied inside time_step via _apply_abm_feedback)
    // Note: cancer_deaths includes deaths from T cells, macrophages, and senescence
    g_lymph->update_from_abm(
        cancer_deaths, teff_recruited, treg_recruited, th_recruited, abm_scaler);

    // Debug output of event counts (optional)
    // if (step % 50 == 0 && cancer_deaths > 0) {
    //     std::cout << "ABM→QSP Events (step " << step << "): CC_deaths=" << cancer_deaths
    //               << " (T=" << cc_death_t_kill << ", MAC=" << cc_death_mac_kill << ", nat=" << cc_death_natural << ")"
    //               << ", Teff_rec=" << teff_recruited << ", TReg_rec=" << treg_recruited << std::endl;
    // }

    // Solve ODE for one ABM timestep (CPU-based)
    double dt_abm = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    double t = static_cast<double>(step) * dt_abm;
    auto qsp_t0 = std::chrono::high_resolution_clock::now();
    if (g_lymph->is_presimulation_mode()) {
        g_lymph->time_step_preSimulation(t, dt_abm);
    } else {
        g_lymph->time_step(t, dt_abm);
    }
    auto qsp_t1 = std::chrono::high_resolution_clock::now();
    g_last_qsp_ms = std::chrono::duration<double, std::milli>(qsp_t1 - qsp_t0).count();

    // Get QSP outputs and update environment for ABM to use
    auto qsp_state = g_lymph->get_state_for_abm();

    // Update environment properties with QSP T cell concentrations for recruitment
    FLAMEGPU->environment.setProperty<float>("qsp_teff_central", static_cast<float>(qsp_state.teff_central));
    FLAMEGPU->environment.setProperty<float>("qsp_treg_central", static_cast<float>(qsp_state.treg_central));
    FLAMEGPU->environment.setProperty<float>("qsp_th_central", static_cast<float>(qsp_state.th_central));

    FLAMEGPU->environment.setProperty<float>("qsp_teff_tumor", static_cast<float>(qsp_state.teff_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_treg_tumor", static_cast<float>(qsp_state.treg_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_th_tumor", static_cast<float>(qsp_state.th_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_mdsc_tumor", static_cast<float>(qsp_state.mdsc_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_m1_tumor", static_cast<float>(qsp_state.m1_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_m2_tumor", static_cast<float>(qsp_state.m2_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_caf_tumor", static_cast<float>(qsp_state.caf_tumor));

    // Drug concentrations
    FLAMEGPU->environment.setProperty<float>("qsp_nivo_tumor", static_cast<float>(qsp_state.nivo_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_cabo_tumor", static_cast<float>(qsp_state.cabo_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_ipi_tumor", static_cast<float>(qsp_state.ipi_tumor));

    FLAMEGPU->environment.setProperty<float>("qsp_cc_tumor", static_cast<float>(qsp_state.cc_tumor) / 
                                                            FLAMEGPU->environment.getProperty<float>("AVOGADROS"));
    FLAMEGPU->environment.setProperty<float>("qsp_cx_tumor", static_cast<float>(qsp_state.cx_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_t_exh_tumor", static_cast<float>(qsp_state.t_exh_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_tum_vol", static_cast<float>(qsp_state.tum_vol));
    FLAMEGPU->environment.setProperty<float>("qsp_tum_cmax", static_cast<float>(qsp_state.tum_cmax));

    FLAMEGPU->environment.setProperty<float>("qsp_f_tum_cap", static_cast<float>(qsp_state.cc_tumor) /
                                                            FLAMEGPU->environment.getProperty<float>("AVOGADROS") /
                                                            qsp_state.tum_cmax);

    // Update Cabo resistance
    float cabo = static_cast<float>(qsp_state.cabo_tumor);
    FLAMEGPU->environment.setProperty<float>("R_cabo", cabo/ (cabo + FLAMEGPU->environment.getProperty<float>("PARAM_IC50_AXL")));
    nvtxRangePop();
}

// ============================================================================
// Timing Accessor: Last QSP Solve Time (milliseconds)
// ============================================================================
double get_last_qsp_ms() {
    return g_last_qsp_ms;
}

} // namespace PDAC

// QSP CSV export step function — defined at file scope (not inside PDAC namespace)
// so it matches the linkage of exportPDEData/exportABMData/stepCounter in main.cu.
// g_lymph and g_qsp_csv are static in PDAC namespace in the same TU, so accessible here.
FLAMEGPU_STEP_FUNCTION(exportQSPData) {
    PDAC::LymphCentralWrapper* lymph = PDAC::get_lymph_pointer();
    if (!lymph) return;
    if (lymph->is_presimulation_mode()) return;

    CancerVCT::ODE_system* ode = lymph->get_ode_system();
    if (!ode) return;

    const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");

    // Open file and write header on first (main-sim) call
    if (!g_qsp_csv.is_open()) {
        std::filesystem::create_directories("outputs");
        g_qsp_csv.open("outputs/qsp.csv");
        g_qsp_csv << "step" << CancerVCT::ODE_system::getHeader() << "\n";
    }

    // Write step index followed by all ODE species (CVODEBase::operator<<)
    g_qsp_csv << main_step << *ode << "\n";
}