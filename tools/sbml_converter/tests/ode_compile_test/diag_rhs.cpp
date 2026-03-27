// Evaluate f(t=0, y0) directly and dump all species with model-unit values.
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <vector>

#include "qsp/ode/ODE_system.h"
#include "qsp/ode/QSPParam.h"
#include "qsp/ode/QSP_enum.h"

#include <nvector/nvector_serial.h>
#include <sundials/sundials_context.h>

using namespace CancerVCT;

int main() {
    QSPParam param;
    param.initializeParams(PARAM_XML_PATH);
    ODE_system::setup_class_parameters(param);

    ODE_system ode;
    ode.setup_instance_variables(param);
    ode.setup_instance_tolerance(param);
    ode.eval_init_assignment();

    int neq = ode.get_num_variables();

    SUNContext sunctx;
    SUNContext_Create(SUN_COMM_NULL, &sunctx);
    N_Vector y = N_VNew_Serial(neq, sunctx);
    N_Vector ydot = N_VNew_Serial(neq, sunctx);

    // Copy initial state (internal units)
    for (int i = 0; i < neq; i++) {
        NV_DATA_S(y)[i] = ode.getSpeciesVar(i, false);
    }

    // Evaluate f directly
    ODE_system::f(0.0, y, ydot, &ode);

    // Get header for species names
    std::string full_header = ODE_system::getHeader();
    // Parse comma-separated names
    std::vector<std::string> names;
    std::string token;
    for (char c : full_header) {
        if (c == ',') {
            if (!token.empty()) names.push_back(token);
            token.clear();
        } else {
            token += c;
        }
    }
    if (!token.empty()) names.push_back(token);

    // Dump CSV: model-unit y0 and ydot (convert back from internal)
    std::ofstream out("/tmp/cpp_rhs.csv");
    out << "Index,Name,y0,dydt" << std::endl;
    out << std::scientific << std::setprecision(15);
    for (int i = 0; i < neq; i++) {
        double y0_model = ode.getSpeciesVar(i, true);  // model units
        // ydot is in internal units per day; convert to model units per day
        // getSpeciesVar(raw=true) divides by scale, so scale = internal/model
        // ydot is in internal units/day; model ydot = ydot / (internal/model) = ydot * model/internal
        double y_internal = ode.getSpeciesVar(i, false);
        double y_model = ode.getSpeciesVar(i, true);
        double scale = (std::abs(y_model) > 1e-30) ? y_internal / y_model : 1.0;
        double dydt_model = NV_DATA_S(ydot)[i] / scale;
        std::string name = (i < (int)names.size()) ? names[i] : "?";
        out << i << "," << name << "," << y0_model << "," << dydt_model << std::endl;
    }
    out.close();
    std::cerr << "Wrote " << neq << " entries to /tmp/cpp_rhs.csv" << std::endl;

    N_VDestroy(y);
    N_VDestroy(ydot);
    SUNContext_Free(&sunctx);
    return 0;
}
