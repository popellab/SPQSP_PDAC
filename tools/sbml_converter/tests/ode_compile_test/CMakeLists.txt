cmake_minimum_required(VERSION 3.18)
project(ode_compile_test CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Path to the PDAC source tree (generated ODE files must already exist here)
set(PDAC_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../../PDAC")

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

# --- Parameter XML path (compile-time constant for test) ---
set(PARAM_XML_PATH "${PDAC_ROOT}/sim/resource/param_all.xml"
    CACHE FILEPATH "Path to parameter XML file for testing")

# --- Build the test ---
add_executable(ode_compile_test
    test_ode_standalone.cpp
    ${PDAC_ROOT}/qsp/ode/ODE_system.cpp
    ${PDAC_ROOT}/qsp/ode/QSPParam.cpp
    ${PDAC_ROOT}/qsp/cvode/CVODEBase.cpp
    ${PDAC_ROOT}/core/ParamBase.cpp
)

target_include_directories(ode_compile_test PRIVATE
    ${PDAC_ROOT}
)

target_link_libraries(ode_compile_test PRIVATE
    ${SUNDIALS_LIBRARIES}
    Boost::serialization
)

target_compile_definitions(ode_compile_test PRIVATE
    PARAM_XML_PATH="${PARAM_XML_PATH}"
)

# --- Trajectory dumper (for MATLAB vs C++ validation) ---
add_executable(dump_trajectories
    dump_trajectories.cpp
    ${PDAC_ROOT}/qsp/ode/ODE_system.cpp
    ${PDAC_ROOT}/qsp/ode/QSPParam.cpp
    ${PDAC_ROOT}/qsp/cvode/CVODEBase.cpp
    ${PDAC_ROOT}/core/ParamBase.cpp
)

target_include_directories(dump_trajectories PRIVATE ${PDAC_ROOT})
target_link_libraries(dump_trajectories PRIVATE ${SUNDIALS_LIBRARIES} Boost::serialization)

# --- RHS diagnostics ---
add_executable(diag_rhs
    diag_rhs.cpp
    ${PDAC_ROOT}/qsp/ode/ODE_system.cpp
    ${PDAC_ROOT}/qsp/ode/QSPParam.cpp
    ${PDAC_ROOT}/qsp/cvode/CVODEBase.cpp
    ${PDAC_ROOT}/core/ParamBase.cpp
)
target_include_directories(diag_rhs PRIVATE ${PDAC_ROOT})
target_link_libraries(diag_rhs PRIVATE ${SUNDIALS_LIBRARIES} Boost::serialization)
target_compile_definitions(diag_rhs PRIVATE PARAM_XML_PATH="${PARAM_XML_PATH}")

# --- CTest ---
enable_testing()
add_test(NAME ode_standalone_test COMMAND ode_compile_test)
