#ifndef PACK_FOR_EXPORT_CUH
#define PACK_FOR_EXPORT_CUH

#include "flamegpu/flamegpu.h"

// ============================================================================
// GPU-side agent packing for async ABM export
//
// Each agent writes 8 int32s to a shared device buffer via atomicAdd on a
// counter. The buffer is then async-copied to host for disk writing.
//
// Row format: [type_id, agent_id, x, y, z, cell_state, life, extra]
// ============================================================================

constexpr int ABM_EXPORT_NCOLS = 8;

// Shared helper: claim a row and write common fields
__device__ __forceinline__ void pack_row(
    flamegpu::DeviceAPI<flamegpu::MessageNone, flamegpu::MessageNone>* FLAMEGPU,
    int type_id, int cell_state, int life, int extra)
{
    int32_t* buf = reinterpret_cast<int32_t*>(
        FLAMEGPU->environment.getProperty<uint64_t>("abm_export_buf_ptr"));
    unsigned int* counter = reinterpret_cast<unsigned int*>(
        FLAMEGPU->environment.getProperty<uint64_t>("abm_export_counter_ptr"));

    unsigned int row = atomicAdd(counter, 1u);
    unsigned int off = row * ABM_EXPORT_NCOLS;

    buf[off + 0] = type_id;
    buf[off + 1] = static_cast<int32_t>(FLAMEGPU->getID());
    buf[off + 2] = FLAMEGPU->getVariable<int>("x");
    buf[off + 3] = FLAMEGPU->getVariable<int>("y");
    buf[off + 4] = FLAMEGPU->getVariable<int>("z");
    buf[off + 5] = cell_state;
    buf[off + 6] = life;
    buf[off + 7] = extra;
}

// type_id=0: has cell_state, divideCD, life (but export uses life=0)
FLAMEGPU_AGENT_FUNCTION(pack_export_cancer, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (!FLAMEGPU->environment.getProperty<int>("do_abm_export")) return flamegpu::ALIVE;
    pack_row(FLAMEGPU, 0,
        FLAMEGPU->getVariable<int>("cell_state"),
        0,
        FLAMEGPU->getVariable<int>("divideCD"));
    return flamegpu::ALIVE;
}

// type_id=1: has cell_state, life
FLAMEGPU_AGENT_FUNCTION(pack_export_tcell, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (!FLAMEGPU->environment.getProperty<int>("do_abm_export")) return flamegpu::ALIVE;
    pack_row(FLAMEGPU, 1,
        FLAMEGPU->getVariable<int>("cell_state"),
        FLAMEGPU->getVariable<int>("life"),
        0);
    return flamegpu::ALIVE;
}

// type_id=2: has cell_state, life
FLAMEGPU_AGENT_FUNCTION(pack_export_treg, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (!FLAMEGPU->environment.getProperty<int>("do_abm_export")) return flamegpu::ALIVE;
    pack_row(FLAMEGPU, 2,
        FLAMEGPU->getVariable<int>("cell_state"),
        FLAMEGPU->getVariable<int>("life"),
        0);
    return flamegpu::ALIVE;
}

// type_id=3: no cell_state
FLAMEGPU_AGENT_FUNCTION(pack_export_mdsc, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (!FLAMEGPU->environment.getProperty<int>("do_abm_export")) return flamegpu::ALIVE;
    pack_row(FLAMEGPU, 3,
        0,
        FLAMEGPU->getVariable<int>("life"),
        0);
    return flamegpu::ALIVE;
}

// type_id=4: has cell_state, life
FLAMEGPU_AGENT_FUNCTION(pack_export_mac, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (!FLAMEGPU->environment.getProperty<int>("do_abm_export")) return flamegpu::ALIVE;
    pack_row(FLAMEGPU, 4,
        FLAMEGPU->getVariable<int>("cell_state"),
        FLAMEGPU->getVariable<int>("life"),
        0);
    return flamegpu::ALIVE;
}

// type_id=5: has cell_state, life
FLAMEGPU_AGENT_FUNCTION(pack_export_fib, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (!FLAMEGPU->environment.getProperty<int>("do_abm_export")) return flamegpu::ALIVE;
    pack_row(FLAMEGPU, 5,
        FLAMEGPU->getVariable<int>("cell_state"),
        FLAMEGPU->getVariable<int>("life"),
        0);
    return flamegpu::ALIVE;
}

// type_id=6: has cell_state, no life. Skip regressed vessels (cell_state==1)
FLAMEGPU_AGENT_FUNCTION(pack_export_vas, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (!FLAMEGPU->environment.getProperty<int>("do_abm_export")) return flamegpu::ALIVE;
    int st = FLAMEGPU->getVariable<int>("cell_state");
    if (st == 1) return flamegpu::ALIVE;  // skip regressed
    pack_row(FLAMEGPU, 6, st, 0, 0);
    return flamegpu::ALIVE;
}

#endif // PACK_FOR_EXPORT_CUH
