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
        std::cerr << "Usage: " << argv[0]
                  << " <param_xml> <output_csv> [t_end_days] [dt_days]" << std::endl;
        return 1;
    }

    std::string param_file = argv[1];
    std::string output_file = argv[2];
    double t_end_days = argc > 3 ? std::stod(argv[3]) : 365.0;
    double dt_days = argc > 4 ? std::stod(argv[4]) : 0.1;

    // Model-unit mode: solver time in days. SI mode: convert days→seconds.
#ifdef MODEL_UNITS
    double time_factor = 1.0;
#else
    double time_factor = 86400.0;
#endif
    double t_end = t_end_days * time_factor;
    double dt = dt_days * time_factor;

    QSPParam param;
    param.initializeParams(param_file);
    ODE_system::setup_class_parameters(param);

    ODE_system ode;
    ode.setup_instance_variables(param);
    ode.setup_instance_tolerance(param);
    ode.eval_init_assignment();

    std::ofstream out(output_file);
    out << std::scientific << std::setprecision(12);

    // operator<< on CVODEBase uses getVarOriginalUnit() via friend access and
    // emits all species (sp_var + sp_other/assignment-rule outputs), prefixed
    // with commas. getHeader() returns the matching column names (no leading
    // comma), so combine as "Time,<header>" for a self-consistent CSV.
    out << "Time," << ODE_system::getHeader() << std::endl;

    auto write_state = [&](double t) {
        out << t / time_factor << ode;
        out << std::endl;
    };

    write_state(0.0);

    double t = 0.0;
    int step = 0;
    while (t < t_end) {
        ode.simOdeStep(t, dt);
        t += dt;
        step++;
        write_state(t);

        if (step % 1000 == 0) {
            std::cerr << "  t=" << t / time_factor << " days" << std::endl;
        }
    }

    out.close();
    std::cerr << "Wrote " << step + 1 << " time points to " << output_file << std::endl;
    return 0;
}
