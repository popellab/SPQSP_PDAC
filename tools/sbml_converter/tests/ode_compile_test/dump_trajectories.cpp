/**
 * Run the ODE system and dump species trajectories to CSV.
 * For comparison against MATLAB SimBiology output.
 *
 * Usage: ./dump_trajectories <param_xml> <output_csv> [t_end_days] [dt_days]
 */
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "qsp/ode/ODE_system.h"
#include "qsp/ode/QSPParam.h"
#include "qsp/ode/QSP_enum.h"

using namespace CancerVCT;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <param_xml> <output_csv> [t_end_days] [dt_days]" << std::endl;
        return 1;
    }

    std::string param_file = argv[1];
    std::string output_file = argv[2];
    double t_end_days = argc > 3 ? std::stod(argv[3]) : 365.0;
    double dt_days = argc > 4 ? std::stod(argv[4]) : 0.1;

    // Time in days (matching model time units)
    double t_end = t_end_days;
    double dt = dt_days;

    // Load parameters
    QSPParam param;
    param.initializeParams(param_file);
    ODE_system::setup_class_parameters(param);

    // Create and initialize ODE system
    ODE_system ode;
    ode.setup_instance_variables(param);
    ode.setup_instance_tolerance(param);
    ode.eval_init_assignment();

    unsigned int n_sp = ode.get_num_variables();

    // Open output file
    std::ofstream out(output_file);
    out << std::scientific << std::setprecision(12);

    // Write header — only sp_var columns (first n_sp entries from getHeader)
    std::string full_header = ODE_system::getHeader();
    // getHeader() returns ",name1,name2,..."; extract first n_sp names
    std::string header;
    int comma_count = 0;
    for (size_t i = 0; i < full_header.size(); i++) {
        if (full_header[i] == ',') comma_count++;
        if (comma_count > static_cast<int>(n_sp)) break;
        header += full_header[i];
    }
    out << "Time" << header << std::endl;

    // Write initial state (sp_var only, in model units to match MATLAB)
    auto write_state = [&](double t) {
        out << t;  // time already in days
        for (unsigned int i = 0; i < n_sp; i++) {
            out << "," << ode.getSpeciesVar(i, true);  // model units (cells, nM, etc.)
        }
        out << std::endl;
    };

    write_state(0.0);

    // Simulate
    double t = 0.0;
    int step = 0;
    while (t < t_end) {
        ode.simOdeStep(t, dt);
        t += dt;
        step++;

        write_state(t);

        if (step % 1000 == 0) {
            std::cerr << "  t=" << t << " days" << std::endl;
        }
    }

    out.close();
    std::cerr << "Wrote " << step + 1 << " time points to " << output_file << std::endl;
    return 0;
}