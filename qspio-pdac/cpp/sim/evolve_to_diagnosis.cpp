#include "qsp_sim_core/model_hooks.h"
#include "qsp_sim_core/trajectory_writer.h"
#include "set_healthy_populations.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <yaml-cpp/yaml.h>

#include "ODE_system.h"
#include "QSP_enum.h"

namespace CancerVCT {

static double current_vt_diameter_cm(const ODE_system& ode) {
    const double V_T_mL = ode.get_compartment_volume("V_T");  // mL = cm³
    return 2.0 * std::cbrt(3.0 * V_T_mL / (4.0 * M_PI));
}

// Binary v3 trajectory format is owned by qsp_sim_core::TrajectoryWriter
// (qsp_sim_core/trajectory_writer.h). Reusing it keeps the burn-in file
// byte-identical to the post-scenario --binary-out file written by
// qsp-codegen's qsp_sim_main.cpp so a single reader
// (qsp_hpc.cpp.runner.read_binary_trajectory) handles both.

// Every reject_reason opens with a machine token: EVOLVE_TOO_FAST,
// EVOLVE_TOO_SLOW, EVOLVE_DEGENERATE. The driver returns 2 and writes the
// reason to stderr, and qsp_hpc.cpp.batch_runner.INIT_REJECT_TOKENS reads the
// token to tell a rejected patient from a dead solver. The prose after the
// colon is free to change; the token is not.
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
    const bool diagnostic_bounds = ev["diagnostic_bounds"]
        && ev["diagnostic_bounds"].as<bool>();

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
    if (!diagnostic_bounds && d0 >= target_cm) {
        r.reject_reason = "EVOLVE_DEGENERATE: initial diameter already >= target";
        r.diameter_cm = d0;
        return r;
    }

    // Integrate in step_days chunks until crossing, capped at max_days.
    const double dt = step_days * opts.time_factor;
    const double t_max = max_days * opts.time_factor;

    // Optional burn-in trajectory dump. When opts.trajectory_path is set,
    // write the v3 binary file (byte-identical to the post-scenario
    // --binary-out writer in qsp_sim_main.cpp). A row is appended every
    // dump_dt_days of model time, defaulting to step_days when
    // opts.trajectory_dt_days is 0. The min_cadence_hours header field
    // is dump_dt_days*24 — the effective row-spacing floor for this
    // writer — and t_offset_days is 0 since evolve starts at t=0.
    std::ofstream traj;
    const double dump_dt_days = opts.trajectory_dt_days > 0.0
        ? opts.trajectory_dt_days : step_days;
    const size_t n_species = static_cast<size_t>(ode.getNumOutputSpecies());
    const size_t n_comp = opts.trajectory_extra_comps.size();
    const size_t n_rules = opts.trajectory_extra_rules.size();
    const size_t n_state_cols = n_species + n_comp + n_rules;
    std::vector<double> traj_row(n_state_cols);
    std::unique_ptr<TrajectoryWriter> traj_writer;
    if (!opts.trajectory_path.empty()) {
        traj.open(opts.trajectory_path, std::ios::binary);
        if (!traj) {
            throw std::runtime_error(
                "evolve_to_diagnosis: failed to open --evolve-trajectory-out "
                "path: " + opts.trajectory_path);
        }
        traj_writer = std::make_unique<TrajectoryWriter>(
            traj,
            static_cast<uint64_t>(n_species),
            static_cast<uint64_t>(n_comp),
            static_cast<uint64_t>(n_rules),
            /*min_cadence_hours=*/dump_dt_days * 24.0,
            /*t_end_days=*/max_days);
    }
    auto write_traj_row = [&](double t_now) {
        if (!traj_writer) return;
        for (size_t i = 0; i < n_species; ++i) {
            traj_row[i] = ode.getSpeciesOutputValue(static_cast<int>(i));
        }
        for (size_t i = 0; i < n_comp; ++i) {
            traj_row[n_species + i] =
                ode.get_compartment_volume(opts.trajectory_extra_comps[i]);
        }
        for (size_t i = 0; i < n_rules; ++i) {
            traj_row[n_species + n_comp + i] =
                ode.get_assignment_rule_value(opts.trajectory_extra_rules[i]);
        }
        traj_writer->write_row(t_now, traj_row.data(), n_state_cols);
    };
    auto finalize_traj = [&](double /*t_evolve_days*/, uint64_t n_cvode_steps) {
        if (!traj_writer) return;
        // t_offset_days = 0 for burn-in (evolve starts at t=0). The header's
        // t_end_days was set at construction to opts max_days; the actual
        // achieved end-time is recoverable from the last row's time column.
        traj_writer->finalize(/*t_offset_days=*/0.0, n_cvode_steps);
        traj.close();
    };
    // Initial (healthy IC) row at t=0.
    write_traj_row(0.0);
    double t_next_dump = dump_dt_days;

    double t = 0.0;
    int steps = 0;
    double prev_d = d0;
    double lower_cm = d0;
    double upper_cm = d0;
    while (t < t_max) {
        ode.simOdeStep(t, dt);
        t += dt;
        steps++;
        const double t_days = t / opts.time_factor;
        const double d = current_vt_diameter_cm(ode);
        upper_cm = std::max(upper_cm, d);
        if (t_days <= min_days) lower_cm = std::max(lower_cm, d);
        // Dump trajectory rows whenever cumulative time crosses the next
        // multiple of dump_dt_days. Float drift is fine here — the header
        // records the actual t_end_days at finalize.
        while (t_days + 1e-9 >= t_next_dump) {
            write_traj_row(t_next_dump);
            t_next_dump += dump_dt_days;
        }
        if (!diagnostic_bounds && d >= target_cm) {
            if (t_days < min_days) {
                std::ostringstream os;
                os << "EVOLVE_TOO_FAST: target reached too fast (" << t_days
                   << " d < " << min_days << " d min)";
                r.reject_reason = os.str();
                r.diameter_cm = d;
                r.t_diagnosis_days = t_days;
                finalize_traj(t_days, static_cast<uint64_t>(steps));
                return r;
            }
            r.success = true;
            r.t_diagnosis_days = t_days;
            r.diameter_cm = d;
            if (opts.verbose) {
                std::cerr << "[evolve] diagnosis at t=" << t_days
                          << " d; diameter=" << d << " cm\n";
            }
            finalize_traj(t_days, static_cast<uint64_t>(steps));
            return r;
        }
        prev_d = d;
        if (opts.verbose && steps % 365 == 0) {
            std::cerr << "[evolve] t=" << t / opts.time_factor
                      << " d; diameter=" << d << " cm\n";
        }
    }

    if (diagnostic_bounds) {
        std::cout << std::setprecision(17)
                  << "EVOLVE_BOUNDS_CM " << lower_cm << " " << upper_cm
                  << "\n";
        r.success = true;
        r.t_diagnosis_days = max_days;
        r.diameter_cm = prev_d;
        finalize_traj(max_days, static_cast<uint64_t>(steps));
        return r;
    }

    std::ostringstream os;
    os << "EVOLVE_TOO_SLOW: target diameter " << target_cm << " cm not reached by "
       << max_days << " d (max=" << prev_d << " cm)";
    r.reject_reason = os.str();
    r.diameter_cm = prev_d;
    finalize_traj(max_days, static_cast<uint64_t>(steps));
    return r;
}

}  // namespace CancerVCT
