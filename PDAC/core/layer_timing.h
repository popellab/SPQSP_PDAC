#pragma once
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// ============================================================================
// Per-Layer Timing Infrastructure
//
// Usage pattern:
//   1. Insert "timing_step_start" as the first host function layer each step.
//      This calls reset_step_timer(), capturing t0 for that step.
//   2. Insert "timing_after_<phase>" host layers at phase boundaries.
//      Each calls record_checkpoint("name"), recording elapsed ms since t0
//      and advancing t0 to now.
//   3. In main.cu after each simulation.step(), iterate g_layer_timings,
//      write to CSV, then clear the vector.
//
// Note: FLAMEGPU2 guarantees that all GPU kernels in a layer complete before
// the next layer starts, so wall-clock chrono between host function calls
// accurately captures both CPU and GPU time for each phase.
// ============================================================================

namespace PDAC {

using ClockPoint = std::chrono::time_point<std::chrono::high_resolution_clock>;

struct LayerTime {
    std::string name;
    double ms;
};

// Per-step accumulator — filled by timing checkpoint host functions,
// drained (written to CSV + cleared) by main.cu after each simulation.step().
extern std::vector<LayerTime> g_layer_timings;

// Rolling checkpoint time — reset at start of each step, advanced after each phase.
extern ClockPoint g_checkpoint_t;

// Opt-in (env PDAC_PROFILE_SYNC=1): cudaDeviceSynchronize at each checkpoint so the
// recorded host wall-time reflects each phase's TRUE GPU time (resolves async
// mis-attribution). Default off → zero production impact. Checked once.
inline bool profile_sync_enabled() {
    static int e = []{ const char* v = std::getenv("PDAC_PROFILE_SYNC"); return (v && v[0] == '1') ? 1 : 0; }();
    return e != 0;
}

// Opt-in (env PDAC_CUDA_LAYER_CHECK=1): cudaDeviceSynchronize + cudaGetLastError
// at each timing checkpoint to catch illegal-address and other async CUDA errors
// at the phase boundary where they actually originate.
inline bool cuda_layer_check_enabled() {
    static int e = []{ const char* v = std::getenv("PDAC_CUDA_LAYER_CHECK"); return (v && v[0] == '1') ? 1 : 0; }();
    return e != 0;
}

inline void cuda_layer_check(const char* checkpoint_name) {
    if (!cuda_layer_check_enabled()) return;
    fprintf(stderr, "CUDA_LAYER_CHECK[%s]: syncing...\n", checkpoint_name);
    fflush(stderr);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA_LAYER_CHECK[%s]: cudaDeviceSynchronize -> %s (%d)\n",
                checkpoint_name, cudaGetErrorString(err), static_cast<int>(err));
        fflush(stderr);
    }
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA_LAYER_CHECK[%s]: cudaGetLastError -> %s (%d)\n",
                checkpoint_name, cudaGetErrorString(err), static_cast<int>(err));
        fflush(stderr);
    }
}

// Record elapsed time since last checkpoint under `name`, then reset checkpoint.
inline void record_checkpoint(const char* name) {
    cuda_layer_check(name);
    if (profile_sync_enabled()) cudaDeviceSynchronize();
    auto now = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - g_checkpoint_t).count();
    g_layer_timings.push_back({name, ms});
    g_checkpoint_t = now;
}

// Reset checkpoint timer (call at very start of each step).
inline void reset_step_timer() {
    cuda_layer_check("step_start");
    if (profile_sync_enabled()) cudaDeviceSynchronize();
    g_checkpoint_t = std::chrono::high_resolution_clock::now();
}

} // namespace PDAC
