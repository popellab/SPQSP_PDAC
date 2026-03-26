#include "LymphCentral_wrapper.h"
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/xml_parser.hpp>

#include "ode/QSP_enum.h" // for species enum, uses CancerVCT::"SP_NameHere"

#define QP(x) CancerVCT::ODE_system::get_class_param(x)
#define getVAR(x) ode_sys->getSpeciesVar(x, false)
#define getVAR_RAW(x) ode_sys->getSpeciesVar(x, true)

namespace PDAC {

// Constructor
LymphCentralWrapper::LymphCentralWrapper()
    : _is_initialized(false), _current_time(0.0)
    , _full_target_vol(0.0), _presimulation_mode(false)
    , _nivo_on(false), _nivo_dose(0.0), _nivo_interval_s(0.0), _nivo_doses_given(0)
    , _cabo_on(false), _cabo_dose(0.0), _cabo_interval_s(0.0), _cabo_doses_given(0)
    , _treatment_started(false), _treatment_start_time(0.0)
    , _nivo_next_dose_t(0.0), _cabo_next_dose_t(0.0) {
    _abm_signals = {0, 0, 0, 0, 0.0};
}

// Destructor
LymphCentralWrapper::~LymphCentralWrapper() {
    // Smart pointers will clean up automatically
}

// Initialize from parameter file, including QSP steady-state warmup
bool LymphCentralWrapper::initialize(const std::string& param_filename) {
    try {
        std::cout << "Initializing QSP LymphCentral model from: " << param_filename << std::endl;

        // =====================================================================
        // STEP 1: Load QSP parameters from XML
        // =====================================================================
        _parameters = std::make_unique<CancerVCT::QSPParam>();
        _parameters->initializeParams(param_filename);

        // Read simulation-level settings directly from XML (not in QSPParam)
        namespace pt = boost::property_tree;
        pt::ptree tree;
        pt::read_xml(param_filename, tree, pt::xml_parser::trim_whitespace);

        double presim_frac = tree.get<double>(
            "Param.QSP.simulation.presimulation_diam_frac", 0.95);
        double weight_qsp  = tree.get<double>(
            "Param.QSP.simulation.weight_qsp", 0.8);
        double dt          = tree.get<double>(
            "Param.ABM.Environment.SecPerSlice", 600.0);

        // Read drug dosing schedule (Param.Pharmacokinetics in XML)
        const double SEC_PER_DAY = 86400.0;
        _nivo_on         = tree.get<int>("Param.ABM.Pharmacokinetics.nivoOn", 0) != 0;
        _nivo_dose       = tree.get<double>("Param.ABM.Pharmacokinetics.nivoDose", 0.0);
        double nivo_days = tree.get<double>("Param.ABM.Pharmacokinetics.nivoDoseIntervalTime", 14.0);
        _nivo_interval_s = nivo_days * SEC_PER_DAY;

        _cabo_on         = tree.get<int>("Param.ABM.Pharmacokinetics.caboOn", 0) != 0;
        _cabo_dose       = tree.get<double>("Param.ABM.Pharmacokinetics.caboDose", 0.0);
        double cabo_days = tree.get<double>("Param.ABM.Pharmacokinetics.caboDoseIntervalTime", 1.0);
        _cabo_interval_s = cabo_days * SEC_PER_DAY;

        std::cout << "  presimulation_diam_frac = " << presim_frac << std::endl;
        std::cout << "  weight_qsp              = " << weight_qsp  << std::endl;
        std::cout << "  dt (sec/slice)          = " << dt          << std::endl;
        std::cout << "  nivo dosing: on=" << _nivo_on
                  << " dose=" << _nivo_dose << " mol"
                  << " interval=" << nivo_days << " days" << std::endl;
        std::cout << "  cabo dosing: on=" << _cabo_on
                  << " dose=" << _cabo_dose << " mol"
                  << " interval=" << cabo_days << " days" << std::endl;

        // =====================================================================
        // STEP 2: Steady-state warmup with a temporary QSP instance
        // Run with use_steady_state=true and _QSP_weight=1 until tumor volume
        // reaches π/6 * (presim_frac * initial_tumour_diameter)^3
        // =====================================================================
        CancerVCT::ODE_system::use_steady_state = true;
        CancerVCT::ODE_system::_QSP_weight      = 1.0;
        CancerVCT::ODE_system::setup_class_parameters(*_parameters);

        // Compute target volume from initial tumour diameter (cm)
        const double D = QP(CancerVCT::P_initial_tumour_diameter);
        const double presim_diam = presim_frac * D;
        const double target_vol  = (M_PI / 6.0) * presim_diam * presim_diam * presim_diam;
        _full_target_vol         = (M_PI / 6.0) * D * D * D;  // 1.0× diameter target

        std::cout << "QSP steady-state warmup:" << std::endl;
        std::cout << "  initial_tumour_diameter = " << D << " cm" << std::endl;
        std::cout << "  presim target diameter  = " << presim_diam << " cm" << std::endl;
        std::cout << "  presim target volume    = " << target_vol  << " cm^3" << std::endl;

        MolecularModelCVode<CancerVCT::ODE_system> ss_model;
        ss_model.getSystem()->setup_instance_tolerance(*_parameters);
        ss_model.getSystem()->setup_instance_variables(*_parameters);
        ss_model.getSystem()->eval_init_assignment();

        const unsigned int n_species = ss_model.getSystem()->get_num_variables();
        std::vector<double> ss_val(n_species, 0.0);

        double tt       = 0.0;
        double tum_vol  = 0.0;
        const double max_time = 5000.0 * 86400.0;  // 5000-day safety limit
        unsigned int step_count = 0;

        while (tt < max_time) {
            ss_model.solve(tt, dt);
            tt += dt;
            step_count++;

            tum_vol = _compute_tumor_volume(ss_model.getSystem());

            if (step_count % 500 == 0) {
                std::cout << "  t=" << tt / 86400.0 << " d  tum_vol=" << tum_vol
                          << " cm^3  (target=" << target_vol << ")" << std::endl;
            }
            if (tum_vol >= target_vol) break;
        }

        if (tum_vol < target_vol) {
            std::cerr << "ERROR: QSP warmup did not reach target volume ("
                      << tum_vol << " < " << target_vol << " cm^3)" << std::endl;
            return false;
        }

        // Save steady-state species values (raw internal units)
        for (unsigned int i = 0; i < n_species; i++) {
            ss_val[i] = ss_model.getSystem()->getSpeciesVar(i);  // raw=true default
        }

        std::cout << "QSP steady-state warmup complete: t=" << tt / 86400.0
                  << " d, tum_vol=" << tum_vol << " cm^3" << std::endl;
        // =====================================================================
        // STEP 3: Setup main QSP model in full simulation mode
        // =====================================================================
        CancerVCT::ODE_system::use_steady_state = false;
        CancerVCT::ODE_system::_QSP_weight      = weight_qsp;
        CancerVCT::ODE_system::setup_class_parameters(*_parameters);

        _qsp_model = std::make_unique<MolecularModelCVode<CancerVCT::ODE_system>>();
        _qsp_model->getSystem()->setup_instance_tolerance(*_parameters);
        _qsp_model->getSystem()->setup_instance_variables(*_parameters);

        // Load steady-state solution into main model
        for (unsigned int i = 0; i < n_species; i++) {
            _qsp_model->getSystem()->setSpeciesVar(i, ss_val[i]);  // raw=true default
        }

        // Re-evaluate initial assignments with the loaded SS state
        _qsp_model->getSystem()->eval_init_assignment();

        _current_time    = tt;   // begin main solve at end of warmup time
        _is_initialized  = true;

        std::cout << "QSP model initialization complete" << std::endl;
        std::cout << "  Species count: " << _qsp_model->getSystem()->get_num_variables() << std::endl;
        std::cout << "  Parameters:   " << _qsp_model->getSystem()->get_num_params()    << std::endl;
        std::cout << "  Start time:   " << _current_time / 86400.0 << " d"              << std::endl;

        return true;

    } catch (const std::exception& e) {
        std::cerr << "Exception during QSP initialization: " << e.what() << std::endl;
        _is_initialized = false;
        return false;
    } catch (...) {
        std::cerr << "Unknown exception during QSP initialization" << std::endl;
        _is_initialized = false;
        return false;
    }
}

// Pre-simulation time step: advance ODE without drug dosing
bool LymphCentralWrapper::time_step_preSimulation(double t, double dt) {
    if (!_is_initialized || !_qsp_model) {
        std::cerr << "Error: QSP model not initialized" << std::endl;
        return false;
    }
    try {
        bool success = _qsp_model->solve(t, dt);
        if (success) {
            _current_time = t + dt;
        }
        return success;
    } catch (const std::exception& e) {
        std::cerr << "Exception during pre-simulation time step: " << e.what() << std::endl;
        return false;
    }
}

// Main simulation time step: advance ODE with drug dosing applied
bool LymphCentralWrapper::time_step(double t, double dt) {
    if (!_is_initialized || !_qsp_model) {
        std::cerr << "Error: QSP model not initialized" << std::endl;
        return false;
    }

    try {
        // Record when treatment started (first main-sim time_step call)
        if (!_treatment_started) {
            _treatment_started = true;
            _treatment_start_time = t;
            std::cout << "QSP: Treatment start time = " << t / 86400.0 << " d" << std::endl;
        }

        // Apply periodic drug boluses for events in [t, t+dt)
        _apply_drug_doses(t, dt);

        // Apply ABM feedback to ODE system if there were signals
        _apply_abm_feedback();

        // Advance ODE system by dt
        // MolecularModelCVode::solve calls ODE_system::simOdeStep internally
        bool success = _qsp_model->solve(t, dt);

        if (success) {
            _current_time = t + dt;
            return true;
        } else {
            std::cerr << "CVODE solver failed during time step" << std::endl;
            return false;
        }

    } catch (const std::exception& e) {
        std::cerr << "Exception during ODE time step: " << e.what() << std::endl;
        return false;
    } catch (...) {
        std::cerr << "Unknown exception during ODE time step" << std::endl;
        return false;
    }
}

// Update QSP from ABM signals — stores scaled counts, applied before next ODE solve
void LymphCentralWrapper::update_from_abm(
    int cancer_deaths,
    int teff_recruited,
    int treg_recruited,
    int th_recruited,
    double abm_scaler)
{
    _abm_signals.cancer_deaths_last_step  = cancer_deaths;
    _abm_signals.teff_recruited_last_step = teff_recruited;
    _abm_signals.treg_recruited_last_step = treg_recruited;
    _abm_signals.th_recruited_last_step   = th_recruited;
    _abm_signals.abm_scaler               = abm_scaler;
}

// Get QSP state for ABM
QSPState LymphCentralWrapper::get_state_for_abm() const {
    QSPState state;

    if (!_is_initialized || !_qsp_model) {
        // Return default state if not initialized
        state.nivo_tumor = 0.0;
        state.cabo_tumor = 0.0;
        state.ipi_tumor = 0.0;

        state.teff_central = 0.0;
        state.treg_central = 0.0;
        state.th_central = 0.0;

        state.teff_tumor = 0.0;
        state.treg_tumor = 0.0;
        state.th_tumor = 0.0;
        state.mdsc_tumor = 0.0;
        state.m1_tumor = 0.0;
        state.m2_tumor = 0.0;
        state.caf_tumor = 0.0;

        state.cc_tumor = 0.0;
        state.cx_tumor = 0.0;
        state.t_exh_tumor = 0.0;
        state.tum_vol = 0.0;
        state.tum_cmax = 0.0;
        state.f_tum_cap = 0.0;
        return state;
    }

    try {
        // Access ODE system to extract species values
        // Species indices are specific to the SBML model in ODE_system.cpp
        // These would be defined based on the actual model structure

        auto* ode_sys = _qsp_model->getSystem();
        if (!ode_sys) {
            return state;
        }

        // Extract relevant species from ODE system using enum indices from QSP_enum.h
        // See PDAC/qsp/ode/QSP_enum.h for species definitions

        // Drug concentrations (tumor compartment)
        state.nivo_tumor = getVAR(CancerVCT::SP_V_T_aPD1);
        state.cabo_tumor = getVAR(CancerVCT::SP_V_T_cabozantinib);
        state.ipi_tumor = 0.0;

        // Central compartment search - for recruitment
        state.teff_central = getVAR(CancerVCT::SP_V_C_T1);
        state.treg_central = getVAR(CancerVCT::SP_V_C_T0); 
        state.th_central = getVAR(CancerVCT::SP_V_C_Th);

        // Tumor compartment search - for initialization
        state.teff_tumor = getVAR(CancerVCT::SP_V_T_T1);
        state.treg_tumor = getVAR(CancerVCT::SP_V_T_T0);
        state.th_tumor = getVAR(CancerVCT::SP_V_T_Th);
        state.mdsc_tumor = getVAR(CancerVCT::SP_V_T_MDSC);
        state.m1_tumor = getVAR(CancerVCT::SP_V_T_Mac_M1);
        state.m2_tumor = getVAR(CancerVCT::SP_V_T_Mac_M2);
        state.caf_tumor = getVAR(CancerVCT::SP_V_T_CAF);

        state.cc_tumor = getVAR_RAW(CancerVCT::SP_V_T_C1);
        state.cx_tumor = getVAR(CancerVCT::SP_V_T_C_x);
        state.t_exh_tumor = getVAR(CancerVCT::SP_V_T_T1_exh);
        state.tum_cmax = getVAR(CancerVCT::SP_V_T_K);

        state.tum_vol = _compute_tumor_volume(ode_sys);

        // std::cout << "Nivo dose going into ABM: " << state.nivo_tumor << std::endl;

        return state;

    } catch (const std::exception& e) {
        std::cerr << "Exception extracting QSP state: " << e.what() << std::endl;
        return state;
    }
}

// Get number of species
int LymphCentralWrapper::get_num_species() const {
    if (!_is_initialized || !_qsp_model) {
        return 0;
    }
    return _qsp_model->getSystem()->get_num_variables();
}

// Get species value by index
double LymphCentralWrapper::get_species_value(int species_idx) const {
    if (!_is_initialized || !_qsp_model) {
        return 0.0;
    }
    try {
        return _qsp_model->getSystem()->getSpeciesVar(species_idx);
    } catch (...) {
        return 0.0;
    }
}

// Set species value by index
void LymphCentralWrapper::set_species_value(int species_idx, double value) {
    if (!_is_initialized || !_qsp_model) {
        return;
    }
    try {
        _qsp_model->getSystem()->setSpeciesVar(species_idx, value);
    } catch (...) {
        // Silently fail
    }
}

// Get tumor volume from current main QSP state (cm^3)
double LymphCentralWrapper::get_tumor_volume() const {
    if (!_is_initialized || !_qsp_model) return 0.0;
    return _compute_tumor_volume(_qsp_model->getSystem());
}

// Compute tumor volume from any ODE system instance using SI species values
// Formula matches get_state_for_abm() - uses getSpeciesVar(x, false) for moles
double LymphCentralWrapper::_compute_tumor_volume(CancerVCT::ODE_system* sys) const {
    using namespace CancerVCT;

    auto getS = [&](int idx) { return sys->getSpeciesVar(idx, false); };

    double C_total = getS(SP_V_T_C1) + getS(SP_V_T_C2);
    double T_total = getS(SP_V_T_T0) + getS(SP_V_T_T1) + getS(SP_V_T_Th);
    double M_total = getS(SP_V_T_Mac_M1) + getS(SP_V_T_Mac_M2) + getS(SP_V_T_MDSC);

    double vol =
        ((getS(SP_V_T_C_x) + C_total) * QP(P_vol_cell) +
         (getS(SP_V_T_T1_exh) + getS(SP_V_T_Th_exh) + T_total) * QP(P_vol_Tcell))
            / QP(P_Ve_T)
        + M_total * QP(P_vol_Mcell) / QP(P_Ve_T)
        + getS(SP_V_T_Fib) * QP(P_vol_Fibcell) / QP(P_Ve_T)
        + getS(SP_V_T_CAF) * QP(P_vol_CAFcell) / QP(P_Ve_T)
        + getS(SP_V_T_ECM) * QP(P_ECM_MW) / QP(P_ECM_density);

    return vol;
}

// Apply drug boluses at scheduled dose times.
//
// Treatment time is measured from _treatment_start_time (set on first time_step call).
// Doses are given at treat_t = 0, interval, 2*interval, ...
// _nivo_next_dose_t / _cabo_next_dose_t track the next scheduled dose time.
// A dose fires when treat_t >= next_dose_t for this step.
//
// Nivolumab (IV): dose added directly to SP_V_C_aPD1 (central compartment)
// Cabozantinib (oral): dose split to SP_V_C_A_site1 + site2 (two-lag absorption model)
void LymphCentralWrapper::_apply_drug_doses(double t, double dt) {
    if (!_is_initialized || !_qsp_model) return;

    using namespace CancerVCT;
    auto* ode = _qsp_model->getSystem();

    const double treat_t = t - _treatment_start_time;  // elapsed treatment time

    const double SEC_PER_DAY = 86400.0;
    double final_dose_week = 8;
    double day_per_week = 7;

    if (treat_t / (SEC_PER_DAY * day_per_week) < final_dose_week){
        // --- Nivolumab (aPD1) ---
        if (_nivo_on && _nivo_interval_s > 0.0 && _nivo_dose > 0.0) {
            while (treat_t >= _nivo_next_dose_t) {
                double cur = ode->getSpeciesVar(SP_V_C_aPD1, false);
                ode->setSpeciesVar(SP_V_C_aPD1, cur + _nivo_dose, false);
                _nivo_doses_given++;
                // std::cout << "  [Dosing] Nivolumab dose #" << _nivo_doses_given
                //           << " at treat_day=" << (_nivo_next_dose_t / 86400.0)
                //           << " (t=" << (t / 86400.0) << " d)"
                //           << " central=" << std::scientific << ode->getSpeciesVar(SP_V_C_aPD1)
                //           << " mol" << std::endl;
                _nivo_next_dose_t += _nivo_interval_s;
            }
        }

        // --- Cabozantinib (oral, two-site absorption) ---
        if (_cabo_on && _cabo_interval_s > 0.0 && _cabo_dose > 0.0) {
            while (treat_t >= _cabo_next_dose_t) {
                // Split dose between absorption sites (ODE uses two-lag model)
                double s1 = ode->getSpeciesVar(SP_V_C_A_site1, false);
                double s2 = ode->getSpeciesVar(SP_V_C_A_site2, false);

                double f1 = 0.675;
                double s1_dose = f1*_cabo_dose / QP(P_V_C);
                double s2_dose = (1-f1)*_cabo_dose / QP(P_V_C);

                ode->setSpeciesVar(SP_V_C_A_site1, s1 + s1_dose, false);
                ode->setSpeciesVar(SP_V_C_A_site2, s2 + s2_dose, false);
                _cabo_doses_given++;
                // std::cout << "  [Dosing] Cabozantinib dose #" << _cabo_doses_given
                //           << " at treat_day=" << (_cabo_next_dose_t / 86400.0)
                //           << " (t=" << (t / 86400.0) << " d)" << std::endl;
                _cabo_next_dose_t += _cabo_interval_s;
            }
        }
    }
}

// Apply ABM feedback to ODE system (called at start of each time_step()).
//
// Converts discrete ABM event counts to ODE species changes:
//   species_change = discrete_count × abm_scaler / AVOGADROS  (moles)
//
// Events applied:
//   1. Cancer deaths  → reduce V_T_C1  (cancer cells in tumor compartment)
//   2. Teff recruited → reduce V_C_T1  (effector T cells in central compartment)
//   3. TReg recruited → reduce V_C_T0  (regulatory T cells in central compartment)
void LymphCentralWrapper::_apply_abm_feedback() {
    if (!_is_initialized || !_qsp_model) return;
    if (_abm_signals.abm_scaler <= 0.0) return;

    using namespace CancerVCT;
    auto* ode = _qsp_model->getSystem();
    const double avogadros = 6.022140857e23;
    const double scaler = _abm_signals.abm_scaler / avogadros;

    // Antigen increase
    double p0 = ode->getSpeciesVar(SP_V_T_P0, false);
    double p1 = ode->getSpeciesVar(SP_V_T_P1, false);
    double factor_p0 = ode->get_class_param(P_n_T0_clones) * ode->get_class_param(P_P0_C1);
    double factor_p1 = ode->get_class_param(P_n_T1_clones) * ode->get_class_param(P_P1_C1);

    ode->setSpeciesVar(SP_V_T_P0, p0 + _abm_signals.cancer_deaths_last_step * scaler * factor_p0, false);
    ode->setSpeciesVar(SP_V_T_P1, p1 + _abm_signals.cancer_deaths_last_step * scaler * factor_p1, false);

    // Recruitment decrease
    double cent_t_eff = ode->getSpeciesVar(SP_V_C_T1, false);
    double cent_t_reg = ode->getSpeciesVar(SP_V_C_T0, false);
    double cent_t_h = ode->getSpeciesVar(SP_V_C_Th, false);

    ode->setSpeciesVar(SP_V_C_T1, std::max(0.0, cent_t_eff - _abm_signals.teff_recruited_last_step * scaler), false);
    ode->setSpeciesVar(SP_V_C_T0, std::max(0.0, cent_t_reg - _abm_signals.treg_recruited_last_step * scaler), false);
    ode->setSpeciesVar(SP_V_C_Th, std::max(0.0, cent_t_h - _abm_signals.th_recruited_last_step * scaler), false);
}

} // namespace PDAC
