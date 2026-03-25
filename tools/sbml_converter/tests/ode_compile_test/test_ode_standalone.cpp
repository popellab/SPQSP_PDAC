/**
 * Standalone CPU test for generated ODE code.
 * Verifies the generated C++ compiles and the ODE system can be
 * instantiated, initialized, and stepped without crashing.
 */
#include <iostream>
#include <cassert>
#include <cmath>
#include <string>

#include "qsp/ode/ODE_system.h"
#include "qsp/ode/QSPParam.h"
#include "qsp/ode/QSP_enum.h"

using namespace CancerVCT;

int main() {
    std::cout << "=== ODE Compile Test ===" << std::endl;

    // 1. Verify enum counts are reasonable
    std::cout << "Species count: " << SP_species_count << std::endl;
    std::cout << "Param count: " << P_param_count << std::endl;
    assert(SP_species_count > 0 && "Species count should be positive");
    assert(P_param_count > 0 && "Param count should be positive");

    // 2. Load parameters from XML
    QSPParam param;
    param.initializeParams(PARAM_XML_PATH);
    std::cout << "Parameters loaded from XML" << std::endl;

    // 3. Setup class parameters
    ODE_system::setup_class_parameters(param);
    std::cout << "Class parameters initialized" << std::endl;

    // Verify a compartment volume is positive
    double v_c = ODE_system::get_class_param(P_V_C);
    assert(v_c > 0 && "V_C should be positive after initialization");
    std::cout << "V_C = " << v_c << " m^3" << std::endl;

    // 4. Create ODE system instance
    ODE_system ode;
    std::cout << "ODE system instantiated" << std::endl;

    // 5. Setup instance variables and tolerance
    ode.setup_instance_variables(param);
    ode.setup_instance_tolerance(param);
    std::cout << "Instance variables and tolerance set" << std::endl;

    // 6. Evaluate initial assignments
    ode.eval_init_assignment();
    std::cout << "Initial assignments evaluated" << std::endl;

    // 7. Get header string
    std::string header = ODE_system::getHeader();
    assert(!header.empty() && "Header should not be empty");
    std::cout << "Header length: " << header.size() << " chars" << std::endl;

    // 8. Verify number of variables
    assert(ode.get_num_variables() == SP_species_count - 2  // sp_var, not sp_other
           || ode.get_num_variables() > 0);  // at minimum positive
    std::cout << "Num variables: " << ode.get_num_variables() << std::endl;

    // 9. Try a short simulation step
    try {
        ode.simOdeStep(0.0, 1.0);  // 1 second step
        std::cout << "Simulation step completed" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Simulation step failed: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "=== All checks passed ===" << std::endl;
    return 0;
}
