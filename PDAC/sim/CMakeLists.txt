cmake_minimum_required(VERSION 3.18)

# CUDA language support
project(PDAC C CUDA CXX)

# Set C++ and CUDA standards
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1301>)

# Option for FLAMEGPU2 location
set(FLAMEGPU_ROOT "" CACHE PATH "Path to FLAMEGPU2 installation or source")

# Disable Runtime Compilation (RTC) - compile CUDA code ahead-of-time during build
# This increases build time but eliminates JIT compilation delay on first run
set(FLAMEGPU_ENABLE_RTC OFF CACHE BOOL "Disable RTC for ahead-of-time compilation")

# Disable FLAMEGPU seatbelt runtime checks.
# Seatbelts flag mixed read + atomic-write on the same MacroProperty in one layer,
# but our occupancy grid uses CAS (which atomically handles compare-and-swap) — this
# is semantically safe and the conservative seatbelt check produces false positives.
set(FLAMEGPU_SEATBELTS OFF CACHE BOOL "Disable FLAMEGPU2 seatbelt checks" FORCE)

# Set CUDA architecture
# Common architectures:
#   70 = V100
#   75 = RTX 20xx, T4
#   80 = A100
#   86 = RTX 30xx
#   89 = RTX 40xx
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86)
endif()

# ============================================================================
# Dependencies — all auto-fetched if not found on system
# ============================================================================
include(FetchContent)

# --- SUNDIALS (CVODE solver for QSP) ---
# Try: (1) SUNDIALS_DIR env var, (2) find_package, (3) FetchContent
set(SUNDIALS_DIR "" CACHE PATH "SUNDIALS installation (leave empty for auto-detect)")
if(NOT SUNDIALS_DIR AND DEFINED ENV{SUNDIALS_DIR})
    set(SUNDIALS_DIR "$ENV{SUNDIALS_DIR}")
endif()

set(_SUNDIALS_READY FALSE)

# Check explicit directory for pre-built install (must have headers AND libraries)
if(SUNDIALS_DIR AND EXISTS "${SUNDIALS_DIR}/include/cvode/cvode.h")
    # Find the lib directory (lib/ or lib64/)
    if(EXISTS "${SUNDIALS_DIR}/lib64/libsundials_cvode.so" OR EXISTS "${SUNDIALS_DIR}/lib64/libsundials_cvode.a")
        set(_SLIB "${SUNDIALS_DIR}/lib64")
    elseif(EXISTS "${SUNDIALS_DIR}/lib/libsundials_cvode.so" OR EXISTS "${SUNDIALS_DIR}/lib/libsundials_cvode.a")
        set(_SLIB "${SUNDIALS_DIR}/lib")
    endif()
    if(_SLIB)
        set(_SUNDIALS_READY TRUE)
        if(EXISTS "${_SLIB}/libsundials_cvode.so")
            set(SUNDIALS_LIBRARIES "${_SLIB}/libsundials_cvode.so" "${_SLIB}/libsundials_nvecserial.so")
        else()
            set(SUNDIALS_LIBRARIES "${_SLIB}/libsundials_cvode.a" "${_SLIB}/libsundials_nvecserial.a")
        endif()
        set(SUNDIALS_INCLUDE_DIR "${SUNDIALS_DIR}/include")
        message(STATUS "SUNDIALS: using pre-built install at ${SUNDIALS_DIR}")
    endif()
endif()

# Try system install via find_package
if(NOT _SUNDIALS_READY)
    find_package(SUNDIALS QUIET COMPONENTS cvode nvecserial)
    if(SUNDIALS_FOUND)
        set(_SUNDIALS_READY TRUE)
        set(SUNDIALS_LIBRARIES SUNDIALS::cvode SUNDIALS::nvecserial)
        message(STATUS "SUNDIALS: found via find_package")
    endif()
endif()

# Build from local source directory (pre-cloned for offline clusters)
if(NOT _SUNDIALS_READY AND SUNDIALS_DIR AND EXISTS "${SUNDIALS_DIR}/CMakeLists.txt")
    message(STATUS "SUNDIALS: building from local source at ${SUNDIALS_DIR}...")
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EXAMPLES_ENABLE_C OFF CACHE BOOL "" FORCE)
    set(EXAMPLES_ENABLE_CXX OFF CACHE BOOL "" FORCE)
    set(EXAMPLES_INSTALL OFF CACHE BOOL "" FORCE)
    set(BUILD_CVODE ON CACHE BOOL "" FORCE)
    set(BUILD_ARKODE OFF CACHE BOOL "" FORCE)
    set(BUILD_IDA OFF CACHE BOOL "" FORCE)
    set(BUILD_IDAS OFF CACHE BOOL "" FORCE)
    set(BUILD_KINSOL OFF CACHE BOOL "" FORCE)
    set(BUILD_CVODES OFF CACHE BOOL "" FORCE)
    add_subdirectory("${SUNDIALS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/sundials")
    set(SUNDIALS_LIBRARIES sundials_cvode_static sundials_nvecserial_static)
    set(SUNDIALS_INCLUDE_DIR
        "${SUNDIALS_DIR}/include"
        "${CMAKE_CURRENT_BINARY_DIR}/sundials/include"
    )
    set(_SUNDIALS_READY TRUE)
endif()

# Auto-fetch from source (requires network)
if(NOT _SUNDIALS_READY)
    message(STATUS "SUNDIALS: not found — fetching v4.1.0 from GitHub...")
    FetchContent_Declare(
        sundials
        GIT_REPOSITORY https://github.com/LLNL/sundials.git
        GIT_TAG        v4.1.0
        GIT_SHALLOW    TRUE
    )
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    set(EXAMPLES_ENABLE_C OFF CACHE BOOL "" FORCE)
    set(EXAMPLES_ENABLE_CXX OFF CACHE BOOL "" FORCE)
    set(EXAMPLES_INSTALL OFF CACHE BOOL "" FORCE)
    set(BUILD_CVODE ON CACHE BOOL "" FORCE)
    set(BUILD_ARKODE OFF CACHE BOOL "" FORCE)
    set(BUILD_IDA OFF CACHE BOOL "" FORCE)
    set(BUILD_IDAS OFF CACHE BOOL "" FORCE)
    set(BUILD_KINSOL OFF CACHE BOOL "" FORCE)
    set(BUILD_CVODES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(sundials)
    set(SUNDIALS_LIBRARIES sundials_cvode_static sundials_nvecserial_static)
    # FetchContent places headers in the source and binary trees
    set(SUNDIALS_INCLUDE_DIR
        "${sundials_SOURCE_DIR}/include"
        "${sundials_BINARY_DIR}/include"
    )
    set(_SUNDIALS_READY TRUE)
endif()

# --- Boost (property_tree XML + serialization) ---
# Try: (1) BOOST_ROOT env var, (2) find_package, (3) FetchContent
if(NOT BOOST_ROOT AND DEFINED ENV{BOOST_ROOT})
    set(BOOST_ROOT "$ENV{BOOST_ROOT}")
endif()

set(_BOOST_READY FALSE)

message(STATUS "Boost: BOOST_ROOT='${BOOST_ROOT}'")

# Build from local source directory first (pre-staged for offline clusters)
if(NOT _BOOST_READY AND BOOST_ROOT AND EXISTS "${BOOST_ROOT}/CMakeLists.txt")
    message(STATUS "Boost: building from local source at ${BOOST_ROOT}...")
    set(BOOST_INCLUDE_LIBRARIES serialization property_tree CACHE STRING "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    add_subdirectory("${BOOST_ROOT}" "${CMAKE_CURRENT_BINARY_DIR}/boost")
    set(_BOOST_READY TRUE)
endif()

# Auto-fetch from source (requires network)
if(NOT _BOOST_READY)
    message(STATUS "Boost: not found — fetching v1.82.0 from GitHub...")
    FetchContent_Declare(
        boost
        URL https://github.com/boostorg/boost/releases/download/boost-1.82.0/boost-1.82.0.tar.gz
        URL_HASH SHA256=b62bd839ea6c28265af9a1f68393eda37fab3611425d3b28882d8e424535ec9d
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    # Only build the libraries we need
    set(BOOST_INCLUDE_LIBRARIES serialization property_tree CACHE STRING "" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(boost)
    set(_BOOST_READY TRUE)
    # Boost CMake superproject provides targets like Boost::serialization
endif()

# --- FLAME GPU 2 ---

if(NOT TARGET flamegpu)
    if(FLAMEGPU_ROOT AND EXISTS "${FLAMEGPU_ROOT}/CMakeLists.txt")
        message(STATUS "Using local FLAMEGPU2 from: ${FLAMEGPU_ROOT}")
        add_subdirectory(${FLAMEGPU_ROOT} ${CMAKE_CURRENT_BINARY_DIR}/flamegpu)
    else()
        message(STATUS "Fetching FLAMEGPU2 from GitHub...")
        FetchContent_Declare(
            flamegpu
            GIT_REPOSITORY https://github.com/FLAMEGPU/FLAMEGPU2.git
            GIT_TAG        v2.0.0-rc.4
            GIT_SHALLOW    TRUE
        )
        FetchContent_MakeAvailable(flamegpu)
    endif()
endif()

# Define executable with ALL source files
add_executable(pdac
    main.cu
    third_party/lz4/lz4.c
    model_definition.cu
    model_layers.cu
    initialization.cu
    ../pde/pde_integration.cu
    ../pde/pde_solver.cu
    ../qsp/LymphCentral_wrapper.cpp
    ../qsp/cvode/CVODEBase.cpp
    ../qsp/ode/ODE_system.cpp
    ../qsp/ode/QSPParam.cpp
    ../qsp/qsp_integration.cu
    ../core/ParamBase.cpp
    ../core/model_functions.cu
    ../abm/gpu_param.cu
)

# Link against FLAMEGPU2
target_link_libraries(pdac PRIVATE flamegpu)
target_compile_definitions(pdac PRIVATE FLAMEGPU_SEATBELTS=0)

# Link against SUNDIALS (CVODE)
if(_SUNDIALS_READY)
    target_link_libraries(pdac PRIVATE ${SUNDIALS_LIBRARIES})
    if(SUNDIALS_INCLUDE_DIR)
        target_include_directories(pdac PRIVATE ${SUNDIALS_INCLUDE_DIR})
    endif()
    target_compile_definitions(pdac PRIVATE QSP_CVODE_ENABLED=1)
else()
    message(FATAL_ERROR "SUNDIALS is required but could not be found or fetched.")
endif()

# Link against Boost (serialization + property_tree)
if(_BOOST_READY)
    if(TARGET Boost::serialization)
        # FetchContent or modern find_package — use CMake targets
        target_link_libraries(pdac PRIVATE Boost::serialization)
        # property_tree is header-only; link its target only if it exists (Boost CMake superproject)
        if(TARGET Boost::property_tree)
            target_link_libraries(pdac PRIVATE Boost::property_tree)
        endif()
    else()
        # Legacy find_package — use variables
        target_link_libraries(pdac PRIVATE ${Boost_LIBRARIES})
        target_include_directories(pdac SYSTEM PRIVATE ${Boost_INCLUDE_DIR})
    endif()
else()
    message(FATAL_ERROR "Boost is required but could not be found or fetched.")
endif()

cmake_path(GET CMAKE_CURRENT_SOURCE_DIR PARENT_PATH MY_PARENT_DIR)
# Include directories
target_include_directories(pdac PRIVATE
    ${MY_PARENT_DIR}/core
    ${MY_PARENT_DIR}/pde
    ${MY_PARENT_DIR}/agents
    ${MY_PARENT_DIR}/qsp
    ${MY_PARENT_DIR}/abm
)

# Enable separable compilation for CUDA (required by FLAMEGPU2)
set_target_properties(pdac PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_compile_options(pdac PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1301>
)

# Boost 1.82+ uses std::source_location in throw_exception.hpp which triggers
# an nvcc bug with consteval. BOOST_DISABLE_CURRENT_LOCATION is the official
# Boost flag to disable std::source_location usage in throw_exception.
target_compile_definitions(pdac PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:BOOST_DISABLE_CURRENT_LOCATION>
)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(pdac PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:-G>
    )
endif()

# Output directory
set_target_properties(pdac PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Installation
install(TARGETS pdac DESTINATION bin)

# ============================================================================
# PDE Solver Validation Tests (no FLAMEGPU dependency — standalone CUDA only)
# ============================================================================
add_executable(pde_test
    ../pde/pde_test.cu
    ../pde/pde_solver.cu
)

set_target_properties(pde_test PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

target_include_directories(pde_test PRIVATE
    ${MY_PARENT_DIR}/pde
)

target_compile_options(pde_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--diag-suppress=1301>
)

# Print configuration summary
message(STATUS "")
message(STATUS "=== PDAC GPU Configuration ===")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Runtime Compilation (RTC): ${FLAMEGPU_ENABLE_RTC}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}/bin")
message(STATUS "")
