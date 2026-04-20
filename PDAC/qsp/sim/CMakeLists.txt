cmake_minimum_required(VERSION 3.18)
project(qsp_sim CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# PDAC/qsp/sim -> ../.. -> PDAC/
set(PDAC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# --- KLU sparse linear solver opt-in ---
# When ON, build SUNDIALS with KLU support (requires SuiteSparse) and
# define USE_KLU so CVODEBase uses the analytical sparse Jacobian emitted
# by qsp_codegen.py. ~2-5x faster per integration step on PDAC's
# 164-species, ~5%-dense system vs dense LU. Default ON when
# SuiteSparse is discoverable.
# USE_KLU: opt-in. Infrastructure (sparse matrix alloc, jac emission,
# CSC sparsity pattern, CVodeSetJacFn wiring) is complete, but the
# emitted analytical Jacobian has boundary singularities at states
# where species = 0. Multiple remediation attempts (NaN clamp, IC
# floor, constraint removal) got to "CVODE setup passes, then crashes
# mid-integration". Deferring until either (a) rate laws are
# regularized to avoid 1/x factors in derivatives, or (b) we add a
# per-entry FD fallback inside jac(). Leaving default OFF so normal
# builds stay on the dense path (which works fine).
option(USE_KLU "Enable KLU sparse linear solver via SuiteSparse (currently WIP)" OFF)
if(USE_KLU)
    find_package(SuiteSparse QUIET COMPONENTS KLU)
    if(NOT SuiteSparse_FOUND)
        # Homebrew on macOS / Debian package both install CMake files under
        # suite-sparse's own paths — try a direct find_library fallback.
        find_path(KLU_INCLUDE_DIR klu.h PATH_SUFFIXES suitesparse)
        find_library(KLU_LIBRARY klu)
        find_library(AMD_LIBRARY amd)
        find_library(COLAMD_LIBRARY colamd)
        find_library(BTF_LIBRARY btf)
        find_library(SUITESPARSECONFIG_LIBRARY suitesparseconfig)
        if(KLU_INCLUDE_DIR AND KLU_LIBRARY)
            set(SuiteSparse_FOUND TRUE)
            set(KLU_LIBRARIES ${KLU_LIBRARY} ${AMD_LIBRARY} ${COLAMD_LIBRARY}
                              ${BTF_LIBRARY} ${SUITESPARSECONFIG_LIBRARY})
            message(STATUS "KLU found via find_library: ${KLU_LIBRARY}")
            message(STATUS "KLU include: ${KLU_INCLUDE_DIR}")
        else()
            message(WARNING "USE_KLU=ON but SuiteSparse/KLU not found. "
                            "Falling back to dense linear solver. "
                            "brew install suite-sparse  (macOS)  or  "
                            "apt install libsuitesparse-dev  (Debian)")
            set(USE_KLU OFF)
        endif()
    endif()
endif()

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
        if(USE_KLU)
            list(APPEND SUNDIALS_LIBRARIES SUNDIALS::sunlinsolklu)
        endif()
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
    if(USE_KLU)
        set(ENABLE_KLU ON CACHE BOOL "" FORCE)
        if(KLU_INCLUDE_DIR)
            set(KLU_INCLUDE_DIR ${KLU_INCLUDE_DIR} CACHE PATH "" FORCE)
        endif()
        if(KLU_LIBRARIES)
            set(KLU_LIBRARY_DIR "" CACHE STRING "" FORCE)
            # SUNDIALS expects KLU_LIBRARIES as the full paths.
        endif()
    endif()
    FetchContent_MakeAvailable(sundials)
    set(SUNDIALS_LIBRARIES sundials_cvode)
    if(USE_KLU)
        list(APPEND SUNDIALS_LIBRARIES sundials_sunlinsolklu)
    endif()
endif()

# --- Boost ---
find_package(Boost 1.70 REQUIRED COMPONENTS serialization)

# --- yaml-cpp (for scenario + drug metadata parsing) ---
find_package(yaml-cpp QUIET)
if(NOT yaml-cpp_FOUND)
    message(STATUS "Fetching yaml-cpp 0.8.0...")
    FetchContent_Declare(yaml-cpp
        GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
        GIT_TAG 0.8.0
        GIT_SHALLOW ON
    )
    set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(yaml-cpp)
endif()

# --- QSP standalone simulator ---
# Runs the PDAC QSP ODE system from a parameter XML and writes trajectories
# as CSV (for SimBiology comparison) or raw binary (for param-sweep callers).
add_executable(qsp_sim
    qsp_sim.cpp
    ${PDAC_ROOT}/sim/set_healthy_populations.cpp
    ${PDAC_ROOT}/sim/evolve_to_diagnosis.cpp
    ${PDAC_ROOT}/qsp/ode/ODE_system.cpp
    ${PDAC_ROOT}/qsp/ode/QSPParam.cpp
    ${PDAC_ROOT}/qsp/cvode/CVODEBase.cpp
    ${PDAC_ROOT}/core/ParamBase.cpp
)

target_include_directories(qsp_sim PRIVATE
    ${PDAC_ROOT}
    ${PDAC_ROOT}/sim
)
target_link_libraries(qsp_sim PRIVATE
    ${SUNDIALS_LIBRARIES}
    Boost::serialization
    yaml-cpp::yaml-cpp
)
if(USE_KLU)
    target_compile_definitions(qsp_sim PRIVATE USE_KLU)
    if(KLU_INCLUDE_DIR)
        target_include_directories(qsp_sim PRIVATE ${KLU_INCLUDE_DIR})
    endif()
    if(KLU_LIBRARIES)
        target_link_libraries(qsp_sim PRIVATE ${KLU_LIBRARIES})
    endif()
endif()

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
if(USE_KLU)
    target_compile_definitions(profile_jacobian PRIVATE USE_KLU)
    if(KLU_INCLUDE_DIR)
        target_include_directories(profile_jacobian PRIVATE ${KLU_INCLUDE_DIR})
    endif()
    if(KLU_LIBRARIES)
        target_link_libraries(profile_jacobian PRIVATE ${KLU_LIBRARIES})
    endif()
endif()