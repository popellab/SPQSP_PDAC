#include "flamegpu/flamegpu.h"
#include "../qsp/LymphCentral_wrapper.h"

namespace PDAC{

// Global pointer to the lymph wrapper (initialized in main)
static LymphCentralWrapper* g_lymph = nullptr;

void set_lymph_pointer(LymphCentralWrapper* lymph) {
    g_lymph = lymph;
}

// Accessible from main.cu FLAMEGPU step functions via extern declaration
bool is_presim_mode_active() {
    return g_lymph ? g_lymph->is_presimulation_mode() : false;
}

FLAMEGPU_HOST_FUNCTION(solve_qsp_step) {
    if (!g_lymph) return;

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
    const double abm_min_cc = 0.5;   // prevents division by zero when tumCC == 0

    double abm_scaler = 0.0;
    if (w > 0.0 && w < 1.0) {
        abm_scaler = ((1.0 - w) / w) * lymphCC / (static_cast<double>(tumCC) + abm_min_cc);
    }

    // Read ABM event counts (set by recruit_t_cells / recruit_mdscs / agent functions)
    const int cancer_deaths  = FLAMEGPU->environment.getProperty<int>("ABM_cc_death");
    const int teff_recruited = FLAMEGPU->environment.getProperty<int>("ABM_TEFF_REC");
    const int treg_recruited = FLAMEGPU->environment.getProperty<int>("ABM_TREG_REC");
    const int mdsc_recruited = FLAMEGPU->environment.getProperty<int>("ABM_MDSC_REC");

    // Pass scaled events to wrapper (applied inside time_step via _apply_abm_feedback)
    g_lymph->update_from_abm(
        cancer_deaths, teff_recruited, treg_recruited, mdsc_recruited,
        /*tumor_volume=*/0.0,  // placeholder (not yet computed from ABM geometry)
        tumCC,
        abm_scaler);

    // Solve ODE for one ABM timestep (CPU-based)
    double dt_abm = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    double t = static_cast<double>(step) * dt_abm;
    if (g_lymph->is_presimulation_mode()) {
        g_lymph->time_step_preSimulation(t, dt_abm);
    } else {
        g_lymph->time_step(t, dt_abm);
    }

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


    // // Debug output
    // if (step % 10 == 0) {
    //     std::cout << "QSP state (step " << step << "): "
    //               << "Teff=" << qsp_state.teff_tumor
    //               << ", Th=" << qsp_state.th_tumor
    //               << ", MDSC=" << qsp_state.mdsc_tumor << std::endl;
    // }
}

}