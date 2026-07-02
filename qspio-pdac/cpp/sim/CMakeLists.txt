cmake_minimum_required(VERSION 3.18)
project(qsp_sim CXX)

# Thin consumer of qsp_sim_core (shipped in the qsp-codegen Python wheel).
# The generic driver main, CVODEBase, ParamBase, SUNDIALS/yaml-cpp wiring,
# and the model-init hook interface live upstream; this file only adds the
# PDAC-specific sources: the qsp-codegen-emitted ODE/param code plus the
# evolve-to-diagnosis override that replaces the library's default no-op.

# Resolve the CMake config dir that qsp-codegen ships in its wheel. If the
# user already set CMAKE_PREFIX_PATH to point at it, skip the lookup.
if(NOT DEFINED QSP_SIM_CORE_PREFIX)
    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -m qsp_codegen.cmake --prefix
        OUTPUT_VARIABLE QSP_SIM_CORE_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _qsp_rc)
    if(NOT _qsp_rc EQUAL 0)
        message(FATAL_ERROR
            "Could not locate qsp_sim_core. Install the qsp-codegen wheel "
            "(`pip install ~/Projects/qsp-codegen`) into the active Python "
            "env, or pass -DQSP_SIM_CORE_PREFIX=<path-to-qsp_codegen/cpp/cmake>.")
    endif()
endif()
list(APPEND CMAKE_PREFIX_PATH "${QSP_SIM_CORE_PREFIX}")
find_package(qsp_sim_core CONFIG REQUIRED)

set(PDAC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/..")

add_executable(qsp_sim
    ${QSP_SIM_CORE_DRIVER_SOURCE}
    set_healthy_populations.cpp
    evolve_to_diagnosis.cpp  # PDAC override of the model-init hook
    ${PDAC_ROOT}/qsp/ode/ODE_system.cpp
    ${PDAC_ROOT}/qsp/ode/QSPParam.cpp
)
target_include_directories(qsp_sim PRIVATE
    ${PDAC_ROOT}/qsp/ode
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(qsp_sim PRIVATE qsp_sim_core::qsp_sim_core)

add_executable(profile_jacobian
    profile_jacobian.cpp
    ${PDAC_ROOT}/qsp/ode/ODE_system.cpp
    ${PDAC_ROOT}/qsp/ode/QSPParam.cpp
)
target_include_directories(profile_jacobian PRIVATE
    ${PDAC_ROOT}/qsp/ode
)
target_link_libraries(profile_jacobian PRIVATE qsp_sim_core::qsp_sim_core)
