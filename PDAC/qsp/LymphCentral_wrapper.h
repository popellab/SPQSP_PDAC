#ifndef PDAC_QSP_LYMPHCENTRAL_WRAPPER_H
#define PDAC_QSP_LYMPHCENTRAL_WRAPPER_H

#include <string>
#include <vector>
#include <memory>
#include <fstream>

// Include actual HCC ODE system classes
#include "ode/ODE_system.h"
#include "ode/QSPParam.h"
#include "cvode/MolecularModelCVode.h"

/**
 * @file LymphCentral_wrapper.h
 * @brief CPU-side wrapper for HCC QSP/ODE model integration with GPU ABM
 *
 * This wrapper encapsulates the actual HCC ODE_system (59+ species) and handles:
 * - Initialization from param_all_test.xml
 * - Time stepping with CVODE solver
 * - Data exchange between GPU ABM and CPU QSP:
 *   - ABM → QSP: Tumor microenvironment signals (cancer deaths, T cell recruitment)
 *   - QSP → ABM: Drug concentrations, systemic immune responses
 */

namespace PDAC {

/**
 * @struct QSPState
 * @brief QSP model state relevant for ABM coupling
 */
struct QSPState {
    // Drug concentrations (will be transferred to GPU environment)
    double nivo_tumor;          // Nivolumab concentration in tumor
    double cabo_tumor;          // Cabozantinib concentration in tumor
    double ipi_tumor;

    // Central compartment cell pools
    double teff_central;        // Effector T cells in central compartment
    double treg_central;        // Regulatory T cells in central compartment
    double th_central;

    // Tumor compartment cell pools
    double teff_tumor;          // CD8+ T cells in tumor (SP_V_T_C1)
    double treg_tumor;
    double th_tumor;            // T helper cells in tumor (SP_V_T_Th)
    double mdsc_tumor;          // MDSCs in tumor (SP_V_T_MDSC)
    double m1_tumor;
    double m2_tumor;
    double caf_tumor;

    double cc_tumor;
    double cx_tumor;
    double t_exh_tumor;
    double tum_vol;
    double tum_cmax;
    double f_tum_cap;

};

/**
 * @class LymphCentralWrapper
 * @brief Wrapper for CPU-side HCC QSP model with CVODE ODE solver
 *
 * Wraps the actual CancerVCT::ODE_system (59+ species) from HCC:
 * - Manages CVODE solver lifecycle
 * - Handles data exchange with GPU ABM
 * - Time stepping through ODE system
 * - State management and species access
 */
class LymphCentralWrapper {
public:
    /**
     * Constructor - initializes but does not set up ODE solver
     */
    LymphCentralWrapper();

    /**
     * Destructor - cleans up CVODE resources
     */
    ~LymphCentralWrapper();

    /**
     * Initialize the QSP model from parameter file
     * @param param_filename Path to parameter XML file (e.g., param_all_test.xml)
     * @return true if initialization successful, false otherwise
     */
    bool initialize(const std::string& param_filename);

    /**
     * Advance ODE system by dt seconds (pre-simulation mode: no drug dosing)
     * @param t Current simulation time (seconds)
     * @param dt Time step (seconds)
     * @return true if successful, false if solver failed
     */
    bool time_step_preSimulation(double t, double dt);

    /**
     * Advance ODE system by dt seconds (main simulation mode: drug dosing applied)
     * @param t Current simulation time (seconds)
     * @param dt Time step (seconds)
     * @return true if successful, false if solver failed
     */
    bool time_step(double t, double dt);

    /**
     * Get tumor volume from current QSP state (cm^3)
     */
    double get_tumor_volume() const;

    /**
     * Update QSP state from ABM signals (called before each QSP time step).
     * Stores scaled ABM events for application in _apply_abm_feedback().
     *
     * abm_scaler = (1-w)/w × lymphCC / (tumCC + abm_min_cc)
     *   where lymphCC = QSP raw cancer molecule count,
     *         tumCC   = ABM discrete cancer cell count,
     *         w       = _QSP_weight (coupling weight, 0.8 default)
     *
     * @param cancer_deaths  ABM cancer cell deaths this step (discrete count)
     * @param teff_recruited ABM Teff cells recruited from QSP central this step
     * @param treg_recruited ABM TReg cells recruited from QSP central this step
     * @param mdsc_recruited ABM MDSC cells recruited this step
     * @param tumor_volume   ABM-estimated tumor volume (cm³)
     * @param tumor_cell_count ABM cancer cell count (discrete agents)
     * @param abm_scaler     Pre-computed ABM→QSP scaling factor
     */
    void update_from_abm(
        int cancer_deaths,
        int teff_recruited,
        int treg_recruited,
        int th_recruited,
        double abm_scaler
    );

    /**
     * Get QSP state for transfer to ABM (GPU)
     * @return QSPState containing relevant variables for ABM coupling
     */
    QSPState get_state_for_abm() const;

    /**
     * Get current time in ODE solver
     */
    double get_current_time() const { return _current_time; }

    /**
     * Set current time in ODE solver (for initialization or restart)
     */
    void set_current_time(double t) { _current_time = t; }

    /**
     * Get number of ODE species
     */
    int get_num_species() const;

    /**
     * Get ODE solution at specific species index
     */
    double get_species_value(int species_idx) const;

    /**
     * Set ODE solution at specific species index
     */
    void set_species_value(int species_idx, double value);

    /**
     * Check if ODE solver is initialized
     */
    bool is_initialized() const { return _is_initialized; }

    /**
     * Get the full target tumor volume (1.0× initial diameter, cm^3)
     * Computed during initialize() from XML parameter P_initial_tumour_diameter
     */
    double get_full_target_volume() const { return _full_target_vol; }

    /**
     * Set/clear pre-simulation mode.
     * When true, solve_qsp_step calls time_step_preSimulation (no drug dosing).
     * When false (default), solve_qsp_step calls time_step (drug dosing active).
     */
    void set_presimulation_mode(bool presim) { _presimulation_mode = presim; }
    bool is_presimulation_mode() const { return _presimulation_mode; }

    /**
     * Set output path for per-step QSP presimulation CSV.
     * Must be called before initialize(). If empty (default), no presim CSV is written.
     */
    void set_presim_output_path(const std::string& path) { _presim_output_path = path; }

    /**
     * Get underlying ODE system (for advanced usage)
     */
    CancerVCT::ODE_system* get_ode_system() {
        return _qsp_model ? _qsp_model->getSystem() : nullptr;
    }

private:
    // Actual HCC ODE system wrapped in MolecularModelCVode
    std::unique_ptr<MolecularModelCVode<CancerVCT::ODE_system>> _qsp_model;

    // Parameter container
    std::unique_ptr<CancerVCT::QSPParam> _parameters;

    // Model state
    bool _is_initialized;
    double _current_time;
    double _full_target_vol;    // π/6 × D³ (1.0× initial diameter), cm^3
    bool _presimulation_mode;   // true → no drug dosing in ODE step
    std::string _presim_output_path;  // path for per-step presim QSP CSV (empty = no output)

    // Drug dosing schedule (read from XML Param.Pharmacokinetics)
    bool   _nivo_on;            // Nivolumab dosing enabled
    double _nivo_dose;          // Dose amount (moles, added to V_C_aPD1)
    double _nivo_interval_s;    // Dosing interval (seconds)
    int    _nivo_doses_given;   // Number of doses administered so far

    bool   _cabo_on;            // Cabozantinib dosing enabled
    double _cabo_dose;          // Dose amount (moles, added to V_C_A_site1)
    double _cabo_interval_s;    // Dosing interval (seconds)
    int    _cabo_doses_given;   // Number of doses administered so far

    bool   _treatment_started;    // Set true on first time_step() call
    double _treatment_start_time; // Time (s) when Phase 4 (main sim) begins
    double _nivo_next_dose_t;    // Next nivo dose time (s, relative to treatment start)
    double _cabo_next_dose_t;    // Next cabo dose time (s, relative to treatment start)

    // Apply drug boluses for any dosing events that fall in [t, t+dt)
    void _apply_drug_doses(double t, double dt);

    // ABM coupling variables (updated each step via update_from_abm)
    struct {
        int cancer_deaths_last_step;
        int teff_recruited_last_step;
        int treg_recruited_last_step;
        int th_recruited_last_step;
        double abm_scaler;   // pre-computed (1-w)/w × lymphCC / (tumCC + eps)
    }_abm_signals;

    // Species indices (extracted from ODE_system or defined by SBML model)
    // These will be populated during initialization
    std::vector<int> _drug_species_indices;
    std::vector<int> _immune_species_indices;

    // Helper method to extract species indices from ODE_system
    void _extract_species_indices();

    // Helper method to apply ABM feedback to ODE system
    void _apply_abm_feedback();

    // Compute tumor volume from an ODE system instance (uses SI units via getSpeciesVar false)
    double _compute_tumor_volume(CancerVCT::ODE_system* sys) const;
};

// Global pointer setter for FLAME GPU host functions
// (declared in qsp_integration.cu)
void set_lymph_pointer(LymphCentralWrapper* lymph);

} // namespace PDAC

#endif // PDAC_QSP_LYMPHCENTRAL_WRAPPER_H
