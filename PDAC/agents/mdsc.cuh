#ifndef FLAMEGPU_TNBC_MDSC_CUH
#define FLAMEGPU_TNBC_MDSC_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// MDSC agent function: Broadcast location
FLAMEGPU_AGENT_FUNCTION(mdsc_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_MDSC);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("cell_state", 0);  // MDSCs have single state
    FLAMEGPU->message_out.setVariable<float>("PDL1", FLAMEGPU->getVariable<float>("PDL1_syn"));
    FLAMEGPU->message_out.setVariable<float>("kill_factor", 0.0f);  // N/A for MDSC
    FLAMEGPU->message_out.setVariable<int>("dead", 0);
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);

    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );

    // Count this agent into per-state population snapshot
    auto* sc_mdsc = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    atomicAdd(&sc_mdsc[SC_MDSC], 1u);

    return flamegpu::ALIVE;
}

// MDSC agent function: Scan neighbors and cache available voxels
// MDSCs check for voxels without other MDSCs (1 MDSC per voxel max)
FLAMEGPU_AGENT_FUNCTION(mdsc_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const int my_x = FLAMEGPU->getVariable<int>("x");
    const int my_y = FLAMEGPU->getVariable<int>("y");
    const int my_z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int size_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    const float my_pos_x = (my_x + 0.5f) * voxel_size;
    const float my_pos_y = (my_y + 0.5f) * voxel_size;
    const float my_pos_z = (my_z + 0.5f) * voxel_size;

    int cancer_count = 0;
    int tcell_count = 0;
    int treg_count = 0;
    int mdsc_count = 0;

    // Adhesion: count all type+state neighbors
    int adh_counts[ABM_STATE_COUNTER_SIZE] = {0};

    for (const auto& msg : FLAMEGPU->message_in(my_pos_x, my_pos_y, my_pos_z)) {
        const int msg_x = msg.getVariable<int>("voxel_x");
        const int msg_y = msg.getVariable<int>("voxel_y");
        const int msg_z = msg.getVariable<int>("voxel_z");

        const int dx = msg_x - my_x;
        const int dy = msg_y - my_y;
        const int dz = msg_z - my_z;

        // Moore neighborhood (excluding self)
        if (abs(dx) <= 1 && abs(dy) <= 1 && abs(dz) <= 1 && !(dx == 0 && dy == 0 && dz == 0)) {
            const int agent_type = msg.getVariable<int>("agent_type");
            const int agent_state = msg.getVariable<int>("cell_state");
            const float kill_factor = msg.getVariable<float>("kill_factor");

            // Adhesion: accumulate into type+state count vector
            adh_counts[msg_to_sc_idx(agent_type, agent_state, kill_factor)]++;

            // Agent-specific interaction counts
            if (agent_type == CELL_TYPE_CANCER) {
                cancer_count++;
            } else if (agent_type == CELL_TYPE_T) {
                tcell_count++;
            } else if (agent_type == CELL_TYPE_TREG) {
                treg_count++;
            } else if (agent_type == CELL_TYPE_MDSC) {
                mdsc_count++;
            }
        }
    }

    // Compute adhesion p_move from matrix
    const int my_sc = SC_MDSC;  // MDSC has single state
    const float adh_pmove = compute_adhesion_pmove(my_sc, adh_counts, ADH_MATRIX_PTR(FLAMEGPU));

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_Tcell_count", tcell_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_MDSC_count", mdsc_count);
    FLAMEGPU->setVariable<float>("adh_p_move", adh_pmove);

    return flamegpu::ALIVE;
}

// MDSC agent function: State step (life countdown only - MDSCs don't divide)
FLAMEGPU_AGENT_FUNCTION(mdsc_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    // Life countdown
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    if (life <= 0) {
        FLAMEGPU->setVariable<int>("dead", 1);
        auto* evts_md = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        atomicAdd(&evts_md[EVT_DEATH_MDSC], 1u);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    // ========== READ CHEMICAL CONCENTRATIONS DIRECTLY FROM PDE ==========
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;
    float local_IFNg = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_CONC_IFN))[voxel];
    
    // ========== COMPUTE DERIVED STATES ==========
    float PDL1 = update_PDL1(local_IFNg,
         FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_HALF"),
         FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_N"),
         FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_SYN_MAX"),
         FLAMEGPU->getVariable<float>("PDL1_syn"),
         FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_INTERNALIZATION_RATE"),
         FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE"));

    FLAMEGPU->setVariable<float>("PDL1_syn", PDL1);

    return flamegpu::ALIVE;
}

// Helper: Compare two agents for priority (lower wins)
__device__ __forceinline__ bool has_higher_priority_mdsc(unsigned int id1, int sx1, int sy1, int sz1,
                                                          unsigned int id2, int sx2, int sy2, int sz2) {
    if (id1 != id2) return id1 < id2;
    if (sx1 != sx2) return sx1 < sx2;
    if (sy1 != sy2) return sy1 < sy2;
    return sz1 < sz2;
}
// Volume occupancy
FLAMEGPU_AGENT_FUNCTION(mdsc_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int gz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    if (x < 0 || x >= gx || y < 0 || y >= gy || z < 0 || z >= gz) {
        return flamegpu::ALIVE;
    }
    const int vidx = z * (gx * gy) + y * gx + x;

    float my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MDSC");
    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

    return flamegpu::ALIVE;
}

// MDSC agent function: Update chemicals from PDE
FLAMEGPU_AGENT_FUNCTION(mdsc_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

// MDSC agent function: Compute chemical sources
// atomicAdds directly to PDE source/uptake arrays
FLAMEGPU_AGENT_FUNCTION(mdsc_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead = FLAMEGPU->getVariable<int>("dead");

    float ArgI_release_rate = 0.0f;
    float NO_release_rate = 0.0f;
    float CCL2_uptake_rate = 0.0f;

    if (dead == 0) {
        ArgI_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_ARGI_RELEASE");
        NO_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_NO_RELEASE");
        CCL2_uptake_rate = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_UPTAKE");
    }

    // Compute voxel index and volume
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;

    const float vs_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
    const float voxel_volume = vs_cm * vs_cm * vs_cm;

    // ArgI secretion → src ptr 6 (ARGI)
    PDE_SECRETE(FLAMEGPU, PDE_SRC_ARGI, voxel, ArgI_release_rate / voxel_volume);

    // NO secretion → src ptr 7 (NO)
    PDE_SECRETE(FLAMEGPU, PDE_SRC_NO, voxel, NO_release_rate / voxel_volume);

    // CCL2 uptake → upt ptr 5 (CCL2), positive [1/s], no volume scaling
    PDE_UPTAKE(FLAMEGPU, PDE_UPT_CCL2, voxel, CCL2_uptake_rate);

    return flamegpu::ALIVE;
}

// Single-phase MDSC movement using run-tumble chemotaxis.
// Replaces two-phase select_move_target + execute_move.
// MDSCs are exclusive per voxel (CAS) but can share voxels with cancer cells.
// Chemotaxis biased toward CCL2 gradient.
// MDSC movement via unified movement framework.
// CCL2 chemotaxis with persistence.
FLAMEGPU_AGENT_FUNCTION(mdsc_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    // Read CCL2 gradient at current voxel
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (grid_x * grid_y) + y * grid_x + x;
    const float gx = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_X))[vidx];
    const float gy = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_Y))[vidx];
    const float gz = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_Z))[vidx];

    MoveParams mp;
    mp.grid_x = grid_x;
    mp.grid_y = grid_y;
    mp.grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    mp.vol_used = VOL_PTR(FLAMEGPU);
    mp.my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MDSC");
    mp.capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    mp.ecm_density = ECM_DENSITY_PTR(FLAMEGPU);
    mp.ecm_crosslink = ECM_CROSSLINK_PTR(FLAMEGPU);
    mp.density_cap = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DENSITY_CAP");
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_MDSC");
    mp.p_move = FLAMEGPU->getVariable<float>("adh_p_move");
    mp.p_persist = FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_MDSC");
    mp.bias_strength = ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_MDSC"));
    mp.grad_x = gx; mp.grad_y = gy; mp.grad_z = gz;
    mp.orient_x = ECM_ORIENT_X_PTR(FLAMEGPU);
    mp.orient_y = ECM_ORIENT_Y_PTR(FLAMEGPU);
    mp.orient_z = ECM_ORIENT_Z_PTR(FLAMEGPU);
    mp.barrier_strength = FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_MDSC");

    MoveResult r = move_cell(mp, x, y, z,
        FLAMEGPU->getVariable<int>("persist_dir_x"),
        FLAMEGPU->getVariable<int>("persist_dir_y"),
        FLAMEGPU->getVariable<int>("persist_dir_z"),
        FLAMEGPU->random.uniform<float>(),
        FLAMEGPU->random.uniform<float>(),
        FLAMEGPU->random.uniform<float>());

    if (r.moved) {
        FLAMEGPU->setVariable<int>("x", r.new_x);
        FLAMEGPU->setVariable<int>("y", r.new_y);
        FLAMEGPU->setVariable<int>("z", r.new_z);
        FLAMEGPU->setVariable<int>("persist_dir_x", r.persist_dx);
        FLAMEGPU->setVariable<int>("persist_dir_y", r.persist_dy);
        FLAMEGPU->setVariable<int>("persist_dir_z", r.persist_dz);
    }

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_MDSC_CUH