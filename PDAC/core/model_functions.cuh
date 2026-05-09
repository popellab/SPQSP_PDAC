#ifndef FLAMEGPU_MODEL_FUNCTIONS_CUH
#define FLAMEGPU_MODEL_FUNCTION_CUH

#include "flamegpu/flamegpu.h"
#include "../qsp/LymphCentral_wrapper.h"
#include <string>

namespace PDAC {

    void set_internal_params(flamegpu::ModelDescription& model,
                             const PDAC::LymphCentralWrapper& lymph,
                             const std::string& xml_path);

}

#endif