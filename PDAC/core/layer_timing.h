#pragma once
#include <chrono>
#include <string>
#include <vector>

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

// Record elapsed time since last checkpoint under `name`, then reset checkpoint.
inline void record_checkpoint(const char* name) {
    auto now = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(now - g_checkpoint_t).count();
    g_layer_timings.push_back({name, ms});
    g_checkpoint_t = now;
}

// Reset checkpoint timer (call at very start of each step).
inline void reset_step_timer() {
    g_checkpoint_t = std::chrono::high_resolution_clock::now();
}

} // namespace PDAC
