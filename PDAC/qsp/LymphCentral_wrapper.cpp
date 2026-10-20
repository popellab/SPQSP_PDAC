#include "LymphCentral_wrapper.h"
#include <iostream>
#include <fstream>
#include <iomanip>
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
    _abm_signals = {0, 0, 0, 0, 0, 0.0};  // cancer_deaths, teff, treg, th, bcell, abm_scaler
}

// Destructor
LymphCentralWrapper::~LymphCentralWrapper() = default;

// Initialize from parameter file (cold-start; no QSP-only warmup).
// ABM and QSP are coupled from step 0; the simulator's presim phase
// (drugs off) is what evolves the system to a pre-treatment state.
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

        std::cout << "  weight_qsp              = " << weight_qsp  << std::endl;
        std::cout << "  dt (sec/slice)          = " << dt          << std::endl;
        std::cout << "  nivo dosing: on=" << _nivo_on
                  << " dose=" << _nivo_dose << " mol"
                  << " interval=" << nivo_days << " days" << std::endl;
        std::cout << "  cabo dosing: on=" << _cabo_on
                  << " dose=" << _cabo_dose << " mol"
                  << " interval=" << cabo_days << " days" << std::endl;

        // =====================================================================
        // STEP 2: Build main QSP model directly in full simulation mode.
        // Initial values come straight from XML; no time stepping here.
        // The simulator's presim phase (drugs off) provides the warmup via
        // coupled ABM+QSP stepping.
        // =====================================================================
        CancerVCT::ODE_system::use_steady_state = false;
        CancerVCT::ODE_system::_QSP_weight      = weight_qsp;
        CancerVCT::ODE_system::setup_class_parameters(*_parameters);

        // Full-target volume kept as a reference for the volume-stopper fallback
        const double D = QP(CancerVCT::P_initial_tumour_diameter);
        _full_target_vol = (M_PI / 6.0) * D * D * D;
        std::cout << "  full target tum_vol     = " << _full_target_vol << " cm^3 (D=" << D << " cm)" << std::endl;

        _qsp_model = std::make_unique<MolecularModelCVode<CancerVCT::ODE_system>>();
        _qsp_model->getSystem()->setup_instance_tolerance(*_parameters);
        _qsp_model->getSystem()->setup_instance_variables(*_parameters);
        _qsp_model->getSystem()->eval_init_assignment();

        _current_time   = 0.0;
        _is_initialized = true;

        std::cout << "QSP model initialization complete (cold start, no warmup)" << std::endl;
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
        // Apply ABM feedback during presim too (the coupled pre-treatment phase is
        // our target: cancer-death→antigen and recruited-cell→central-pool decrement
        // must act while the tumour grows to the diameter target). No drug dosing in presim.
        _apply_abm_feedback();
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
    int bcell_recruited,
    double abm_scaler)
{
    _abm_signals.cancer_deaths_last_step   = cancer_deaths;
    _abm_signals.teff_recruited_last_step  = teff_recruited;
    _abm_signals.treg_recruited_last_step  = treg_recruited;
    _abm_signals.th_recruited_last_step    = th_recruited;
    _abm_signals.bcell_recruited_last_step = bcell_recruited;
    _abm_signals.abm_scaler                = abm_scaler;
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
        state.bcell_central = 0.0;

        state.teff_tumor = 0.0;
        state.treg_tumor = 0.0;
        state.th_tumor = 0.0;
        state.mdsc_tumor = 0.0;
        state.m1_tumor = 0.0;
        state.m2_tumor = 0.0;
        state.caf_tumor = 0.0;
        state.qpsc_tumor = 0.0;
        state.icaf_tumor = 0.0;
        state.mycaf_tumor = 0.0;
        state.apcaf_tumor = 0.0;
        state.bcell_naive_tumor = 0.0;
        state.bcell_act_tumor = 0.0;
        state.plasma_tumor = 0.0;
        state.tfh_tumor = 0.0;
        state.dc_tumor = 0.0;

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
        state.cabo_tumor = 0.0;  // Cabozantinib not in current SBML model
        state.ipi_tumor = 0.0;  // aCTLA4 not in upstream QSP model

        // Central compartment - for recruitment
        state.teff_central = getVAR(CancerVCT::SP_V_C_CD8);
        state.treg_central = getVAR(CancerVCT::SP_V_C_Treg);
        state.th_central = getVAR(CancerVCT::SP_V_C_Th);
        state.bcell_central = getVAR(CancerVCT::SP_V_C_B);

        // Tumor compartment - for initialization
        state.teff_tumor = getVAR(CancerVCT::SP_V_T_CD8);
        state.treg_tumor = getVAR(CancerVCT::SP_V_T_Treg);
        state.th_tumor = getVAR(CancerVCT::SP_V_T_Th);
        state.mdsc_tumor = getVAR(CancerVCT::SP_V_T_MDSC);
        state.m1_tumor = getVAR(CancerVCT::SP_V_T_Mac_M1);
        state.m2_tumor = getVAR(CancerVCT::SP_V_T_Mac_M2);
        state.qpsc_tumor = getVAR(CancerVCT::SP_V_T_qPSC);
        state.icaf_tumor = getVAR(CancerVCT::SP_V_T_iCAF);
        state.mycaf_tumor = getVAR(CancerVCT::SP_V_T_myCAF);
        state.apcaf_tumor = getVAR(CancerVCT::SP_V_T_apCAF);
        state.caf_tumor = state.icaf_tumor + state.mycaf_tumor + state.apcaf_tumor;
        state.bcell_naive_tumor = getVAR(CancerVCT::SP_V_T_nB);
        state.bcell_act_tumor = getVAR(CancerVCT::SP_V_T_aB);
        state.plasma_tumor = getVAR(CancerVCT::SP_V_T_Plasma);
        state.tfh_tumor = getVAR(CancerVCT::SP_V_T_Tfh);
        state.dc_tumor = getVAR(CancerVCT::SP_V_T_cDC1) + getVAR(CancerVCT::SP_V_T_cDC2)
                       + getVAR(CancerVCT::SP_V_T_mcDC1) + getVAR(CancerVCT::SP_V_T_mcDC2);

        state.cc_tumor = getVAR_RAW(CancerVCT::SP_V_T_C1);
        state.cx_tumor = getVAR(CancerVCT::SP_V_T_C_x);
        state.t_exh_tumor = getVAR(CancerVCT::SP_V_T_CD8_exh);
        state.tum_cmax = getVAR(CancerVCT::SP_V_T_K_vasc);

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

bool LymphCentralWrapper::dump_onset_state(const std::string& path, double t_warmup_s) const {
    if (!_is_initialized || !_qsp_model) {
        std::cerr << "ERROR: dump_onset_state called before QSP initialization" << std::endl;
        return false;
    }
    auto* ode_sys = _qsp_model->getSystem();
    if (!ode_sys) return false;

    // Species are held in mol; ×N_A gives discrete cells. Same conversion the
    // recruitment gap-closing applies (pde_integration.cu: qsp_*_tumor × AVOGADROS),
    // done once here so the generator never has to know the internal unit.
    constexpr double NA = 6.022140857e23;
    struct Row { const char* name; int idx; };
    const Row rows[] = {
        {"C1", CancerVCT::SP_V_T_C1},
        {"C_x", CancerVCT::SP_V_T_C_x},
        {"qPSC", CancerVCT::SP_V_T_qPSC}, {"iCAF", CancerVCT::SP_V_T_iCAF},
        {"myCAF", CancerVCT::SP_V_T_myCAF}, {"apCAF", CancerVCT::SP_V_T_apCAF},
        {"CD8", CancerVCT::SP_V_T_CD8}, {"CD8_exh", CancerVCT::SP_V_T_CD8_exh},
        {"Th", CancerVCT::SP_V_T_Th}, {"Th_exh", CancerVCT::SP_V_T_Th_exh},
        {"Treg", CancerVCT::SP_V_T_Treg}, {"Tfh", CancerVCT::SP_V_T_Tfh},
        {"nB", CancerVCT::SP_V_T_nB}, {"aB", CancerVCT::SP_V_T_aB}, {"Plasma", CancerVCT::SP_V_T_Plasma},
        {"Mac_M1", CancerVCT::SP_V_T_Mac_M1}, {"Mac_M2", CancerVCT::SP_V_T_Mac_M2},
        {"MDSC", CancerVCT::SP_V_T_MDSC},
        {"cDC1", CancerVCT::SP_V_T_cDC1}, {"cDC2", CancerVCT::SP_V_T_cDC2},
        {"mcDC1", CancerVCT::SP_V_T_mcDC1}, {"mcDC2", CancerVCT::SP_V_T_mcDC2},
    };

    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot write QSP onset file '" << path << "'" << std::endl;
        return false;
    }
    f << std::setprecision(17);
    f << "{\n  \"schema\": \"qsp_onset/1\",\n";
    f << "  \"t_warmup_s\": " << t_warmup_s << ",\n";
    f << "  \"V_T_m3\": " << _compute_tumor_volume(ode_sys) << ",\n";
    f << "  \"avogadro\": " << NA << ",\n";
    f << "  \"cells\": {\n";
    const int n = static_cast<int>(sizeof(rows) / sizeof(rows[0]));
    for (int i = 0; i < n; ++i) {
        f << "    \"" << rows[i].name << "\": " << (getVAR(rows[i].idx) * NA)
          << (i + 1 < n ? ",\n" : "\n");
    }
    f << "  }\n}\n";
    f.close();
    std::cout << "  QSP onset state written: " << path
              << " (V_T=" << _compute_tumor_volume(ode_sys) << " m^3)" << std::endl;
    return true;
}

bool LymphCentralWrapper::save_qsp_state(const std::string& path, double t_warmup_s) const {
    if (!_is_initialized || !_qsp_model) {
        std::cerr << "ERROR: save_qsp_state called before QSP initialization" << std::endl;
        return false;
    }
    std::ofstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot write QSP state file '" << path << "'" << std::endl;
        return false;
    }
    f.write("QSP1", 4);
    f.write(reinterpret_cast<const char*>(&t_warmup_s), sizeof(double));
    double vol = _compute_tumor_volume(_qsp_model->getSystem());
    f.write(reinterpret_cast<const char*>(&vol), sizeof(double));
    _qsp_model->getSystem()->saveFullState(f);
    f.close();
    std::cout << "  QSP state saved: " << path
              << " (t=" << t_warmup_s << " s, V_T=" << vol << " m^3)" << std::endl;
    return true;
}

bool LymphCentralWrapper::load_qsp_state(const std::string& path, double& t_warmup_s) {
    if (!_is_initialized || !_qsp_model) {
        std::cerr << "ERROR: load_qsp_state called before QSP initialization" << std::endl;
        return false;
    }
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::cerr << "ERROR: cannot read QSP state file '" << path << "'" << std::endl;
        return false;
    }
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "QSP1") {
        std::cerr << "ERROR: invalid QSP state file magic (expected QSP1)" << std::endl;
        return false;
    }
    f.read(reinterpret_cast<char*>(&t_warmup_s), sizeof(double));
    double vol = 0.0;
    f.read(reinterpret_cast<char*>(&vol), sizeof(double));
    try {
        _qsp_model->getSystem()->loadFullState(f);
    } catch (const std::exception& e) {
        std::cerr << "ERROR: load_qsp_state failed: " << e.what() << std::endl;
        return false;
    }
    _qsp_model->getSystem()->updateVar();
    f.close();
    std::cout << "  QSP state loaded: " << path
              << " (t=" << t_warmup_s << " s, V_T=" << vol << " m^3)" << std::endl;
    return true;
}

double LymphCentralWrapper::get_tumor_cancer_cells() const {
    if (!_is_initialized || !_qsp_model) return 0.0;
    // get_species_value → getSpeciesVar(idx, raw=true) = _species_var / unit_conv; for C1/C2
    // unit_conv = 1/N_A, so this is the discrete cell count (matches the qsp_presim CSV + cc_tumor).
    return get_species_value(CancerVCT::SP_V_T_C1);
}

// Compute tumor volume — MUST mirror the ODE's AUX_VAR_V_T exactly (ODE_system.cpp:1320-1323),
// since that is the volume the QSP dynamics themselves use (RF68 trafficking, growth, drug PK).
// The ODE is authoritative; this helper previously deviated on five terms and returned a volume
// ~1e4× too large on collagen-rich tumors (silently inflating qsp_tum_vol → the presim volume
// stopper and any V_T-based coupling). Corrected Jul 2026 to match the ODE term-for-term:
//   C_total = C1;  T_total = Treg + CD8 + Th + Tfh;  M_total = Mac_M1 + Mac_M2 (NOT MDSC);
//   + V_Tmin floor;  collagen contributes collagen / rho_collagen  (a volume = mass ÷ density,
//   NOT the old collagen · rho / phi which is dimensionally mass²/vol).
double LymphCentralWrapper::_compute_tumor_volume(CancerVCT::ODE_system* sys) const {
    using namespace CancerVCT;

    auto getS = [&](int idx) { return sys->getSpeciesVar(idx, false); };

    const double C_total = getS(SP_V_T_C1);
    const double T_total = getS(SP_V_T_Treg) + getS(SP_V_T_CD8) + getS(SP_V_T_Th) + getS(SP_V_T_Tfh);
    const double M_total = getS(SP_V_T_Mac_M1) + getS(SP_V_T_Mac_M2);

    double vol =
        QP(P_V_Tmin)
        + ((((getS(SP_V_T_C_x) + C_total) * QP(P_vol_cell))
            + ((getS(SP_V_T_CD8_exh) + getS(SP_V_T_Th_exh) + T_total) * QP(P_vol_Tcell))
            + M_total * QP(P_vol_Mcell)
            + getS(SP_V_T_qPSC)  * QP(P_vol_qPSCcell)
            + getS(SP_V_T_iCAF)  * QP(P_vol_iCAFcell)
            + getS(SP_V_T_myCAF) * QP(P_vol_myCAFcell)
            + getS(SP_V_T_apCAF) * QP(P_vol_apCAFcell)) / QP(P_Ve_T))
        + getS(SP_V_T_collagen) / QP(P_rho_collagen);

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

        // --- Cabozantinib (oral) ---
        // NOTE: Cabozantinib absorption sites not in current SBML model.
        // Dosing disabled until PK compartments are added to the ODE.
        if (_cabo_on && _cabo_interval_s > 0.0 && _cabo_dose > 0.0) {
            while (treat_t >= _cabo_next_dose_t) {
                _cabo_doses_given++;
                std::cerr << "  [WARNING] Cabozantinib dose #" << _cabo_doses_given
                          << " requested but PK species not in SBML model" << std::endl;
                _cabo_next_dose_t += _cabo_interval_s;
            }
        }
    }
}

// Apply ABM feedback to ODE system (called at start of each time_step()).
//
// SINGLE-LEDGER REDESIGN (Jul 2026): DISABLED — the ABM is now the stochastic,
// spatially-resolved *realization* of the QSP, not a separate mass-exchanging
// compartment. The QSP owns the burden ledger and runs its own complete
// central→tumor trafficking (RF68 for CD8, RF36/193 Treg/Th, RF237 B, …) plus
// its own cancer-death→antigen dynamics. The ABM must NOT edit QSP physical
// state, or those quantities would be double-counted:
//   - central pool would be debited twice (QSP RF68 + this feedback), the exact
//     "two tumors sharing one blood pool" incoherence the redesign removes;
//   - antigen P0/P1 would be fed both by QSP cancer death and by ABM deaths that
//     are already a subset of the same lesion.
// ABM→QSP coupling is now one-way tracking only (QSP central pools × f drive ABM
// recruitment; see build_recruit_kernel_params / compute_qfields_kernel). The old
// event-count feedback is retained below (commented) for reference / future
// intensive (kill-efficiency) coupling, which is deliberately deferred.
void LymphCentralWrapper::_apply_abm_feedback() {
    // Intentionally a no-op: ABM does not mutate QSP species (single ledger).
    return;

    /* --- retired extensive feedback (double-counts under single ledger) ---
    if (!_is_initialized || !_qsp_model) return;
    if (_abm_signals.abm_scaler <= 0.0) return;

    using namespace CancerVCT;
    auto* ode = _qsp_model->getSystem();
    const double avogadros = 6.022140857e23;
    const double scaler = _abm_signals.abm_scaler / avogadros;

    // Antigen increase
    double p0 = ode->getSpeciesVar(SP_V_T_P0, false);
    double p1 = ode->getSpeciesVar(SP_V_T_P1, false);
    double factor_p0 = ode->get_class_param(P_n_CD4_clones) * ode->get_class_param(P_P0_C1);
    double factor_p1 = ode->get_class_param(P_n_CD8_clones) * ode->get_class_param(P_P1_C1);

    ode->setSpeciesVar(SP_V_T_P0, p0 + _abm_signals.cancer_deaths_last_step * scaler * factor_p0, false);
    ode->setSpeciesVar(SP_V_T_P1, p1 + _abm_signals.cancer_deaths_last_step * scaler * factor_p1, false);

    // Recruitment decrease
    double cent_t_eff = ode->getSpeciesVar(SP_V_C_CD8, false);
    double cent_t_reg = ode->getSpeciesVar(SP_V_C_Treg, false);
    double cent_t_h = ode->getSpeciesVar(SP_V_C_Th, false);
    double cent_b   = ode->getSpeciesVar(SP_V_C_B, false);

    ode->setSpeciesVar(SP_V_C_CD8, std::max(0.0, cent_t_eff - _abm_signals.teff_recruited_last_step * scaler), false);
    ode->setSpeciesVar(SP_V_C_Treg, std::max(0.0, cent_t_reg - _abm_signals.treg_recruited_last_step * scaler), false);
    ode->setSpeciesVar(SP_V_C_Th, std::max(0.0, cent_t_h - _abm_signals.th_recruited_last_step * scaler), false);
    ode->setSpeciesVar(SP_V_C_B, std::max(0.0, cent_b - _abm_signals.bcell_recruited_last_step * scaler), false);
    --- end retired feedback --- */
}

} // namespace PDAC
