#include "evolve_to_diagnosis.h"
#include "set_healthy_populations.h"

#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include <yaml-cpp/yaml.h>

#include "qsp/ode/ODE_system.h"
#include "qsp/ode/QSP_enum.h"

namespace CancerVCT {

double current_vt_diameter_cm(const ODE_system& ode) {
    const double V_T_mL = ode.get_compartment_volume("V_T");  // mL = cm³
    return 2.0 * std::cbrt(3.0 * V_T_mL / (4.0 * M_PI));
}

EvolveResult evolve_to_diagnosis(ODE_system& ode, const EvolveOpts& opts) {
    EvolveResult r;
    if (opts.yaml_path.empty()) {
        throw std::runtime_error("evolve_to_diagnosis: yaml_path is required");
    }
    YAML::Node y = YAML::LoadFile(opts.yaml_path);
    const auto& ev = y["evolve"];
    const double min_days = ev["min_days"].as<double>();
    const double max_days = ev["max_days"].as<double>();
    const double step_days = ev["step_days"].as<double>();

    // Target diameter: explicit arg, else read from model parameter
    // initial_tumour_diameter (stored in SI meters; convert to cm).
    double target_cm = opts.target_diameter_cm;
    if (target_cm <= 0.0) {
        const double init_diam_m = ODE_system::get_class_param(P_initial_tumour_diameter);
        target_cm = init_diam_m * 1e2;  // m → cm
    }

    // Lay down healthy IC.
    HealthyPopulationOpts hp;
    hp.yaml_path = opts.yaml_path;
    hp.tumor_cells = opts.tumor_cells;
    hp.verbose = opts.verbose;
    set_healthy_populations(ode, hp);

    const double d0 = current_vt_diameter_cm(ode);
    if (opts.verbose) {
        std::cerr << "[evolve] target diameter = " << target_cm
                  << " cm; initial V_T diameter = " << d0 << " cm\n";
    }
    if (d0 >= target_cm) {
        r.reject_reason = "initial diameter already >= target";
        r.diameter_cm = d0;
        return r;
    }

    // Integrate in step_days chunks until crossing, capped at max_days.
    const double dt = step_days * opts.time_factor;
    const double t_max = max_days * opts.time_factor;
    double t = 0.0;
    int steps = 0;
    double prev_d = d0;
    while (t < t_max) {
        ode.simOdeStep(t, dt);
        t += dt;
        steps++;
        const double d = current_vt_diameter_cm(ode);
        if (d >= target_cm) {
            const double t_days = t / opts.time_factor;
            if (t_days < min_days) {
                std::ostringstream os;
                os << "target reached too fast (" << t_days << " d < "
                   << min_days << " d min)";
                r.reject_reason = os.str();
                r.diameter_cm = d;
                r.t_diagnosis_days = t_days;
                return r;
            }
            r.success = true;
            r.t_diagnosis_days = t_days;
            r.diameter_cm = d;
            if (opts.verbose) {
                std::cerr << "[evolve] diagnosis at t=" << t_days
                          << " d; diameter=" << d << " cm\n";
            }
            return r;
        }
        prev_d = d;
        if (opts.verbose && steps % 365 == 0) {
            std::cerr << "[evolve] t=" << t / opts.time_factor
                      << " d; diameter=" << d << " cm\n";
        }
    }

    std::ostringstream os;
    os << "target diameter " << target_cm << " cm not reached by "
       << max_days << " d (max=" << prev_d << " cm)";
    r.reject_reason = os.str();
    r.diameter_cm = prev_d;
    return r;
}

}  // namespace CancerVCT