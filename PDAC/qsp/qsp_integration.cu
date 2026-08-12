#include "flamegpu/flamegpu.h"
#include "../qsp/LymphCentral_wrapper.h"
#include "../qsp/ode/ODE_system.h"
#include "../qsp/ode/QSP_enum.h"    // P_v_T_search_volume / P_K_M1_phago / P_R50_Treg
#include "../core/common.cuh"        // PhiPair enum
#include "../pde/pde_integration.cuh" // get_phi / phi_is_valid

#include <fstream>
#include <filesystem>
#include <chrono>
#include <nvtx3/nvToolsExt.h>

// File-scope variables (accessible to exportQSPData step function outside PDAC namespace)
static std::ofstream g_qsp_csv;
static std::string g_qsp_output_path = "outputs/qsp.csv";
static std::ofstream g_qsp_presim_csv;
static std::string g_qsp_presim_output_path = "outputs/qsp_presim.csv";

namespace PDAC{

// Global pointer to the lymph wrapper (initialized in main)
static LymphCentralWrapper* g_lymph = nullptr;
double g_last_qsp_ms = 0.0;  // exposed for timing CSV

// QSP integration-time offset [s]. The decoupled presim advances the ODE in a QSP-only warmup
// (main.cu Phase A) before the coupled loop starts at ABM step 0; this offset keeps the time
// passed to CVODE monotonic across the handoff (CVodeSetStopTime requires tstop ahead of the
// solver's current internal time). Set once after the warmup; 0 otherwise.
static double g_qsp_time_offset = 0.0;
void set_qsp_time_offset(double t_offset_s) { g_qsp_time_offset = t_offset_s; }

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

void set_qsp_output_path(const std::string& path) {
    g_qsp_output_path = path;
}

void set_qsp_presim_output_path(const std::string& path) {
    g_qsp_presim_output_path = path;
}

// Populate every qsp_* env property from the wrapper's current ODE state, plus
// derived quantities (qsp_f_tum_cap, R_cabo). solve_qsp_step normally does this
// at the end of each step, but when QSP is frozen (step_qsp=0) those writes are
// skipped — leaving env properties at their default 0.0. Call this once after
// the internal QSP warmup so ABM-only modes inherit post-warmup values.
void seed_qsp_env_properties(flamegpu::CUDASimulation& simulation) {
    if (!g_lymph) return;
    auto qsp_state = g_lymph->get_state_for_abm();
    const float avogadros = simulation.getEnvironmentProperty<float>("AVOGADROS");
    const float ic50_axl  = simulation.getEnvironmentProperty<float>("PARAM_IC50_AXL");

    simulation.setEnvironmentProperty<float>("qsp_teff_central", static_cast<float>(qsp_state.teff_central));
    simulation.setEnvironmentProperty<float>("qsp_treg_central", static_cast<float>(qsp_state.treg_central));
    simulation.setEnvironmentProperty<float>("qsp_th_central",   static_cast<float>(qsp_state.th_central));
    simulation.setEnvironmentProperty<float>("qsp_bcell_central", static_cast<float>(qsp_state.bcell_central));

    simulation.setEnvironmentProperty<float>("qsp_teff_tumor",   static_cast<float>(qsp_state.teff_tumor));
    simulation.setEnvironmentProperty<float>("qsp_treg_tumor",   static_cast<float>(qsp_state.treg_tumor));
    simulation.setEnvironmentProperty<float>("qsp_th_tumor",     static_cast<float>(qsp_state.th_tumor));
    simulation.setEnvironmentProperty<float>("qsp_mdsc_tumor",   static_cast<float>(qsp_state.mdsc_tumor));
    simulation.setEnvironmentProperty<float>("qsp_m1_tumor",     static_cast<float>(qsp_state.m1_tumor));
    simulation.setEnvironmentProperty<float>("qsp_m2_tumor",     static_cast<float>(qsp_state.m2_tumor));
    simulation.setEnvironmentProperty<float>("qsp_caf_tumor",    static_cast<float>(qsp_state.caf_tumor));
    simulation.setEnvironmentProperty<float>("qsp_dc_tumor",     static_cast<float>(qsp_state.dc_tumor));
    // These three drifted out of sync with solve_qsp_step's setter, which does write them.
    // Consequence was silent and one-step: after a QSP warm-up the B, Plasma and Tfh targets
    // read 0 on step 0 only, so anything gated on them (the gap-closing deficit, the class
    // audit) saw an empty compartment for exactly one step and then healed. Keep the two
    // setters in lockstep — a property in one and not the other is invisible in any run long
    // enough to overwrite it.
    simulation.setEnvironmentProperty<float>("qsp_bcell_tumor",  static_cast<float>(qsp_state.bcell_tumor));
    simulation.setEnvironmentProperty<float>("qsp_plasma_tumor", static_cast<float>(qsp_state.plasma_tumor));
    simulation.setEnvironmentProperty<float>("qsp_tfh_tumor",    static_cast<float>(qsp_state.tfh_tumor));

    simulation.setEnvironmentProperty<float>("qsp_nivo_tumor",   static_cast<float>(qsp_state.nivo_tumor));
    simulation.setEnvironmentProperty<float>("qsp_cabo_tumor",   static_cast<float>(qsp_state.cabo_tumor));
    simulation.setEnvironmentProperty<float>("qsp_ipi_tumor",    static_cast<float>(qsp_state.ipi_tumor));

    const float cc_tumor_mM = static_cast<float>(qsp_state.cc_tumor) / avogadros;
    simulation.setEnvironmentProperty<float>("qsp_cc_tumor",     cc_tumor_mM);
    simulation.setEnvironmentProperty<float>("qsp_cx_tumor",     static_cast<float>(qsp_state.cx_tumor));
    simulation.setEnvironmentProperty<float>("qsp_t_exh_tumor",  static_cast<float>(qsp_state.t_exh_tumor));
    simulation.setEnvironmentProperty<float>("qsp_tum_vol",      static_cast<float>(qsp_state.tum_vol));
    simulation.setEnvironmentProperty<float>("qsp_tum_cmax",     static_cast<float>(qsp_state.tum_cmax));

    const float f_tum_cap = (qsp_state.tum_cmax > 0.0)
        ? cc_tumor_mM / static_cast<float>(qsp_state.tum_cmax)
        : 0.0f;
    simulation.setEnvironmentProperty<float>("qsp_f_tum_cap", f_tum_cap);

    const float cabo = static_cast<float>(qsp_state.cabo_tumor);
    simulation.setEnvironmentProperty<float>("R_cabo", cabo / (cabo + ic50_axl));
}

// Export step-0 QSP state (initial condition, before any simulation steps).
// Called from main.cu after presim completes and before the main loop.
void exportQSPData_step0() {
    if (!g_lymph) return;
    CancerVCT::ODE_system* ode = g_lymph->get_ode_system();
    if (!ode) return;

    if (!g_qsp_csv.is_open()) {
        std::filesystem::create_directories("outputs");
        g_qsp_csv.open(g_qsp_output_path);
        g_qsp_csv << "step," << CancerVCT::ODE_system::getHeader() << "\n";
    }
    g_qsp_csv << 0 << *ode << "\n";
    g_qsp_csv.flush();
}

FLAMEGPU_HOST_FUNCTION(solve_qsp_step) {
    nvtxRangePush("QSP Solve");
    if (!g_lymph) { nvtxRangePop(); return; }

    // ABM-only mode: skip QSP solve. All qsp_* env properties stay pinned at
    // whatever values they had when step_qsp was flipped to 0.
    if (FLAMEGPU->environment.getProperty<int>("step_qsp") == 0) {
        g_last_qsp_ms = 0.0;
        nvtxRangePop();
        return;
    }

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
    const int bcell_recruited = FLAMEGPU->environment.getProperty<int>("ABM_BCELL_REC");

    // Pass scaled events to wrapper (applied inside time_step via _apply_abm_feedback)
    // Note: cancer_deaths includes deaths from T cells, macrophages, and senescence
    g_lymph->update_from_abm(
        cancer_deaths, teff_recruited, treg_recruited, th_recruited, bcell_recruited, abm_scaler);

    // Debug output of event counts (optional)
    // if (step % 50 == 0 && cancer_deaths > 0) {
    //     std::cout << "ABM→QSP Events (step " << step << "): CC_deaths=" << cancer_deaths
    //               << " (T=" << cc_death_t_kill << ", MAC=" << cc_death_mac_kill << ", nat=" << cc_death_natural << ")"
    //               << ", Teff_rec=" << teff_recruited << ", TReg_rec=" << treg_recruited << std::endl;
    // }

    // -----------------------------------------------------------------------
    // ABM → QSP: SPATIAL COUPLING (contract D-1). Replace the ODE's mean-field
    // encounter estimators with the ABM's measurements for this step.
    //
    // Each phi is a DENSITY SUBSTITUTION, applied by scaling an EXISTING parameter —
    // no rate law is edited, so the codegen's analytic Jacobian (written in terms of
    // PARAM(...)) stays consistent for free:
    //
    //   phi_1  [CD8] -> phi_1*[CD8] in the kill numerator
    //          == v_T_search_volume *= phi_1     (the product v_search*[CD8]/V_T IS
    //                                             the mean-field encounter estimator)
    //   phi_2  [Mac_M1] -> phi_2*[Mac_M1] in Flux302's numerator AND denominator
    //          == K_M1_phago /= phi_2            (algebraically equivalent up to the
    //                                             P_cell epsilon, ~7 orders down)
    //   phi_3  Tregs_ -> phi_3*Tregs_ in the kill denominator
    //          == R50_Treg /= phi_3              (since K_T_Treg = 1/R50_Treg)
    //
    // Directions verified: phi<1 means the partner is DEPLETED around the target, and
    // each substitution above then reduces killing / reduces phagocytosis / reduces
    // suppression respectively — i.e. less co-localization does less of the thing.
    //
    // ⚠️ ALWAYS scale from the CACHED BASE, never from the current value: these
    // parameters are overwritten in place every step, so scaling a scaled value would
    // compound geometrically and diverge within a few steps.
    //
    // phi_is_valid gates each one independently — a pair with no cells on one side
    // falls back to its base (mean-field), which is a different statement from phi=1
    // measured, and must not be applied as if it were data.
    {
        using CancerVCT::ODE_system;
        static bool  base_cached = false;
        static double base_v_search = 0.0, base_K_M1 = 0.0, base_R50_Treg = 0.0;
        if (!base_cached) {
            base_v_search  = ODE_system::get_class_param(CancerVCT::P_v_T_search_volume);
            base_K_M1      = ODE_system::get_class_param(CancerVCT::P_K_M1_phago);
            base_R50_Treg  = ODE_system::get_class_param(CancerVCT::P_R50_Treg);
            base_cached = true;
        }
        // Bit 0 = contact corrections. Mode 0 restores the base values every step, so the
        // control arm is the unmodified model EXACTLY, not merely at initialization.
        const int mode = FLAMEGPU->environment.getProperty<int>("PARAM_COUPLING_MODE");
        const bool contact = (mode & 1) != 0;

        const double p1 = (contact && PDAC::phi_is_valid(PHI_CD8_C1))   ? PDAC::get_phi(PHI_CD8_C1)   : 1.0;
        const double p2 = (contact && PDAC::phi_is_valid(PHI_M1_C1))    ? PDAC::get_phi(PHI_M1_C1)    : 1.0;
        const double p3 = (contact && PDAC::phi_is_valid(PHI_CD8_TREG)) ? PDAC::get_phi(PHI_CD8_TREG) : 1.0;

        // A phi of 0 would divide by zero below and, for phi_1, would switch killing off
        // entirely on a single empty-neighbourhood step. Floor it: the ABM cannot resolve
        // co-density below one cell in the kernel anyway.
        const double eps = 1e-6;
        ODE_system::set_class_param(CancerVCT::P_v_T_search_volume, base_v_search * fmax(p1, eps));
        ODE_system::set_class_param(CancerVCT::P_K_M1_phago,        base_K_M1     / fmax(p2, eps));
        ODE_system::set_class_param(CancerVCT::P_R50_Treg,          base_R50_Treg / fmax(p3, eps));
    }

    // Solve ODE for one ABM timestep (CPU-based). Add the QSP-only-warmup offset so CVODE's
    // integration time stays monotonic across the decoupled-presim handoff.
    double dt_abm = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    double t = g_qsp_time_offset + static_cast<double>(step) * dt_abm;
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
    FLAMEGPU->environment.setProperty<float>("qsp_bcell_central", static_cast<float>(qsp_state.bcell_central));

    FLAMEGPU->environment.setProperty<float>("qsp_teff_tumor", static_cast<float>(qsp_state.teff_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_treg_tumor", static_cast<float>(qsp_state.treg_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_th_tumor", static_cast<float>(qsp_state.th_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_mdsc_tumor", static_cast<float>(qsp_state.mdsc_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_m1_tumor", static_cast<float>(qsp_state.m1_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_m2_tumor", static_cast<float>(qsp_state.m2_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_caf_tumor", static_cast<float>(qsp_state.caf_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_qpsc_tumor", static_cast<float>(qsp_state.qpsc_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_icaf_tumor", static_cast<float>(qsp_state.icaf_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_mycaf_tumor", static_cast<float>(qsp_state.mycaf_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_apcaf_tumor", static_cast<float>(qsp_state.apcaf_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_bcell_tumor", static_cast<float>(qsp_state.bcell_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_plasma_tumor", static_cast<float>(qsp_state.plasma_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_tfh_tumor", static_cast<float>(qsp_state.tfh_tumor));
    FLAMEGPU->environment.setProperty<float>("qsp_dc_tumor", static_cast<float>(qsp_state.dc_tumor));

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

    CancerVCT::ODE_system* ode = lymph->get_ode_system();
    if (!ode) return;

    const bool in_presim = lymph->is_presimulation_mode();

    if (in_presim) {
        const unsigned int presim_step = FLAMEGPU->environment.getProperty<unsigned int>("presim_step");
        if (!g_qsp_presim_csv.is_open()) {
            const auto parent = std::filesystem::path(g_qsp_presim_output_path).parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
            g_qsp_presim_csv.open(g_qsp_presim_output_path);
            g_qsp_presim_csv << "step," << CancerVCT::ODE_system::getHeader() << "\n";
        }
        g_qsp_presim_csv << (presim_step + 1) << *ode << "\n";
    } else {
        const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");
        if (!g_qsp_csv.is_open()) {
            const auto parent = std::filesystem::path(g_qsp_output_path).parent_path();
            if (!parent.empty()) std::filesystem::create_directories(parent);
            g_qsp_csv.open(g_qsp_output_path);
            g_qsp_csv << "step," << CancerVCT::ODE_system::getHeader() << "\n";
        }
        g_qsp_csv << (main_step + 1) << *ode << "\n";
    }
}