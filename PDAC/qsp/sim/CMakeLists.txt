cmake_minimum_required(VERSION 3.18)
project(qsp_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PDAC/qsp/sim -> ../.. -> PDAC/
set(PDAC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# --- SUNDIALS ---
set(SUNDIALS_DIR "" CACHE PATH "SUNDIALS installation")
if(NOT SUNDIALS_DIR AND DEFINED ENV{SUNDIALS_DIR})
    set(SUNDIALS_DIR "$ENV{SUNDIALS_DIR}")
endif()

include(FetchContent)
set(_SUNDIALS_READY FALSE)

if(SUNDIALS_DIR)
    list(APPEND CMAKE_PREFIX_PATH "${SUNDIALS_DIR}")
    find_package(SUNDIALS 7.0 QUIET COMPONENTS cvode)
    if(SUNDIALS_FOUND)
        set(_SUNDIALS_READY TRUE)
        set(SUNDIALS_LIBRARIES SUNDIALS::cvode)
        message(STATUS "Found SUNDIALS: ${SUNDIALS_DIR}")
    endif()
endif()

if(NOT _SUNDIALS_READY)
    message(STATUS "Fetching SUNDIALS 7.6.0...")
    FetchContent_Declare(sundials
        GIT_REPOSITORY https://github.com/LLNL/sundials.git
        GIT_TAG v7.6.0
        GIT_SHALLOW ON
    )
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(SUNDIALS_BUILD_WITH_MONITORING OFF CACHE BOOL "" FORCE)
    set(EXAMPLES_ENABLE_C OFF CACHE BOOL "" FORCE)
    set(EXAMPLES_ENABLE_CXX OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(sundials)
    set(SUNDIALS_LIBRARIES sundials_cvode)
endif()

# --- Boost ---
find_package(Boost 1.70 REQUIRED COMPONENTS serialization)

# --- QSP standalone simulator ---
# Runs the PDAC QSP ODE system from a parameter XML and writes trajectories
# as CSV (for SimBiology comparison) or raw binary (for param-sweep callers).
add_executable(qsp_sim
    qsp_sim.cpp
    ${PDAC_ROOT}/qsp/ode/ODE_system.cpp
    ${PDAC_ROOT}/qsp/ode/QSPParam.cpp
    ${PDAC_ROOT}/qsp/cvode/CVODEBase.cpp
    ${PDAC_ROOT}/core/ParamBase.cpp
)

target_include_directories(qsp_sim PRIVATE ${PDAC_ROOT})
target_link_libraries(qsp_sim PRIVATE
    ${SUNDIALS_LIBRARIES}
    Boost::serialization
)

# --- Jacobian sparsity profiler (one-off diagnostic) ---
add_executable(profile_jacobian
    profile_jacobian.cpp
    ${PDAC_ROOT}/qsp/ode/ODE_system.cpp
    ${PDAC_ROOT}/qsp/ode/QSPParam.cpp
    ${PDAC_ROOT}/qsp/cvode/CVODEBase.cpp
    ${PDAC_ROOT}/core/ParamBase.cpp
)
target_include_directories(profile_jacobian PRIVATE ${PDAC_ROOT})
target_link_libraries(profile_jacobian PRIVATE
    ${SUNDIALS_LIBRARIES}
    Boost::serialization
)