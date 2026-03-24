// Macrophage Agent Behavior Functions
// M1/M2 polarization, CCL2-based recruitment and chemotaxis, cancer cell killing

#ifndef MACROPHAGE_CUH
#define MACROPHAGE_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// Macrophage: Broadcast location (spatial messaging)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_MAC);
    FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);
    const int mac_cs = FLAMEGPU->getVariable<int>("cell_state");
    FLAMEGPU->message_out.setVariable<int>("cell_state", mac_cs);
    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );
    // Count this agent into per-state population snapshot
    auto* sc_mac = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    atomicAdd(&sc_mac[mac_cs == MAC_M1 ? SC_MAC_M1 : SC_MAC_M2], 1u);
    return flamegpu::ALIVE;
}

// ============================================================================
// Macrophage: Write to occupancy grid (exclusive per voxel)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // Macrophages are exclusive (1 per voxel)
    occ[x][y][z][CELL_TYPE_MAC].exchange(1u);

    // Flat arrays for GPU recruitment kernel (MAC exclusive placement)
    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (gx * gy) + y * gx + x;
    unsigned int* mac_occ = reinterpret_cast<unsigned int*>(
        FLAMEGPU->environment.getProperty<uint64_t>("mac_occ_ptr"));
    atomicAdd(&mac_occ[vidx], 1u);

    return flamegpu::ALIVE;
}

// ============================================================================
// Macrophage: Scan neighbors (count adjacent cancer cells for killing)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    // Convert voxel indices to world-space coordinates for spatial query
    const float world_x = (x + 0.5f) * voxel_size;
    const float world_y = (y + 0.5f) * voxel_size;
    const float world_z = (z + 0.5f) * voxel_size;

    int neighbor_cancer_count = 0;

    for (auto& msg : FLAMEGPU->message_in(world_x, world_y, world_z)) {
        int agent_type = msg.getVariable<int>("agent_type");
        if (agent_type == CELL_TYPE_CANCER) {
            int msg_x = msg.getVariable<int>("voxel_x");
            int msg_y = msg.getVariable<int>("voxel_y");
            int msg_z = msg.getVariable<int>("voxel_z");
            int dx = msg_x - x;
            int dy = msg_y - y;
            int dz = msg_z - z;

            // Only count if in Moore neighborhood (26 adjacent voxels)
            if (std::abs(dx) <= 1 && std::abs(dy) <= 1 && std::abs(dz) <= 1) {
                neighbor_cancer_count++;
            }
        }
    }

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", neighbor_cancer_count);
    return flamegpu::ALIVE;
}

// ============================================================================
// Macrophage: Update local chemicals (read from PDE)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

// ============================================================================
// Macrophage: Compute chemical sources
// atomicAdds directly to PDE source/uptake arrays
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const int dead = FLAMEGPU->getVariable<int>("dead");

    float IFNg_release_rate = 0.0f;
    float IL12_release_rate = 0.0f;
    float TGFB_release_rate = 0.0f;
    float IL10_release_rate = 0.0f;
    float VEGFA_release_rate = 0.0f;
    float CCL2_uptake_rate = 0.0f;

    int cancer_count = FLAMEGPU->getVariable<int>("neighbor_cancer_count");
    if (dead == 0) {
        if (cell_state == MAC_M1){
            // if (cancer_count > 0 || FLAMEGPU->getVariable<int>("ifng_active") == 1){
            //     FLAMEGPU->setVariable<int>("ifng_active", 1);
            //     IFNg_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_RELEASE");
            //     IL12_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL12_RELEASE");
            // }
            IFNg_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_RELEASE");
            IL12_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL12_RELEASE");
        } else {
            TGFB_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_RELEASE");
            IL10_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL10_RELEASE");
            VEGFA_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_VEGFA_RELEASE");
        }
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

    // CCL2 uptake → upt ptr 5 (CCL2), positive [1/s], no volume scaling
    PDE_UPTAKE(FLAMEGPU, PDE_UPT_CCL2, voxel, CCL2_uptake_rate);

    // IFN-γ secretion → src ptr 1 (IFN)
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IFN, voxel, IFNg_release_rate / voxel_volume);

    // IL-12 secretion → src ptr 8 (IL12)
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IL12, voxel, IL12_release_rate / voxel_volume);

    // TGF-β secretion → src ptr 4 (TGFB)
    PDE_SECRETE(FLAMEGPU, PDE_SRC_TGFB, voxel, TGFB_release_rate / voxel_volume);

    // IL-10 secretion → src ptr 3 (IL10)
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IL10, voxel, IL10_release_rate / voxel_volume);

    // VEGF-A secretion → src ptr 9 (VEGFA)
    PDE_SECRETE(FLAMEGPU, PDE_SRC_VEGFA, voxel, VEGFA_release_rate / voxel_volume);

    return flamegpu::ALIVE;
}


// ============================================================================
// Macrophage: Single-phase movement using occupancy grid (exclusive per voxel)
// Uses run-tumble chemotaxis toward CCL2 gradient
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");

    // ECM based movement probability: higher ECM → more likely to be blocked
    {
        const float* ecm_ptr = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("ecm_grid_ptr"));
        float ECM_density = ecm_ptr[z * (grid_x * grid_y) + y * grid_x + x];
        float ECM_50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50");
        float ECM_sat = ECM_density / (ECM_density + ECM_50);
        if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;
    }

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    // Use CCL2 gradient for chemotaxis — read directly from PDE
    const int nx_mv = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_mv = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_mv = z * ny_mv*nx_mv + y * nx_mv + x;
    const float grad_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_X))[voxel_mv];
    const float grad_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_Y))[voxel_mv];
    const float grad_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_Z))[voxel_mv];

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const uint8_t* face_flags = reinterpret_cast<const uint8_t*>(
        FLAMEGPU->environment.getProperty<uint64_t>("face_flags_ptr"));

    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        float v_x = move_dir_x / dt;
        float v_y = move_dir_y / dt;
        float v_z = move_dir_z / dt;

        float norm_gradient = std::sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);
        float dot_product = v_x * grad_x + v_y * grad_y + v_z * grad_z;
        float norm_v = std::sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
        float cos_theta = dot_product / (norm_v * norm_gradient);

        const float EC50_grad = 1.0f;
        float H_grad = norm_gradient / (norm_gradient + EC50_grad);
        if (cos_theta < 0) H_grad = -H_grad;

        const float lambda = 0.0000168f;
        float tumble_rate = (lambda / 2.0f) * (1.0f - cos_theta) * (1.0f - H_grad) * dt;
        float p_tumble = 1.0f - std::exp(-tumble_rate);

        if (FLAMEGPU->random.uniform<float>() < p_tumble) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }

        int tx = x + static_cast<int>(std::round(move_dir_x));
        int ty = y + static_cast<int>(std::round(move_dir_y));
        int tz = z + static_cast<int>(std::round(move_dir_z));

        if (tx < 0 || tx >= grid_x || ty < 0 || ty >= grid_y || tz < 0 || tz >= grid_z) {
            return flamegpu::ALIVE;
        }

        // Ductal wall check
        if (is_ductal_wall_blocked(face_flags, x, y, z, tx-x, ty-y, tz-z, grid_x, grid_y)) {
            return flamegpu::ALIVE;
        }

        if (occ[tx][ty][tz][CELL_TYPE_MAC].CAS(0u, 1u) == 0u) {
            occ[x][y][z][CELL_TYPE_MAC].exchange(0u);
            FLAMEGPU->setVariable<int>("x", tx);
            FLAMEGPU->setVariable<int>("y", ty);
            FLAMEGPU->setVariable<int>("z", tz);
        } else {
            // FLAMEGPU->setVariable<int>("tumble", 1);
        }
    }
    // === TUMBLE PHASE (tumble == 1) ===
    // Collect all free Moore neighbors, shuffle, try each until CAS wins.
    else {
        int cand_x[26], cand_y[26], cand_z[26];
        int n_cands = 0;
        for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++) {
            if (di==0 && dj==0 && dk==0) continue;
            int nx = x+di, ny = y+dj, nz = z+dk;
            if (nx<0||nx>=grid_x||ny<0||ny>=grid_y||nz<0||nz>=grid_z) continue;
            if (is_ductal_wall_blocked(face_flags, x, y, z, di, dj, dk, grid_x, grid_y)) continue;
            if (occ[nx][ny][nz][CELL_TYPE_MAC] != 0u) continue;
            cand_x[n_cands] = nx; cand_y[n_cands] = ny; cand_z[n_cands] = nz;
            n_cands++;
        }
        if (n_cands == 0) return flamegpu::ALIVE;
        for (int i = n_cands-1; i > 0; i--) {
            int j = static_cast<int>(FLAMEGPU->random.uniform<float>() * (i+1));
            if (j > i) j = i;
            int tx=cand_x[i]; cand_x[i]=cand_x[j]; cand_x[j]=tx;
            int ty=cand_y[i]; cand_y[i]=cand_y[j]; cand_y[j]=ty;
            int tz=cand_z[i]; cand_z[i]=cand_z[j]; cand_z[j]=tz;
        }
        for (int i = 0; i < n_cands; i++) {
            if (occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_MAC].CAS(0u, 1u) == 0u) {
                occ[x][y][z][CELL_TYPE_MAC].exchange(0u);
                FLAMEGPU->setVariable<int>("x", cand_x[i]);
                FLAMEGPU->setVariable<int>("y", cand_y[i]);
                FLAMEGPU->setVariable<int>("z", cand_z[i]);
                FLAMEGPU->setVariable<float>("move_direction_x", static_cast<float>(cand_x[i]-x));
                FLAMEGPU->setVariable<float>("move_direction_y", static_cast<float>(cand_y[i]-y));
                FLAMEGPU->setVariable<float>("move_direction_z", static_cast<float>(cand_z[i]-z));
                FLAMEGPU->setVariable<int>("tumble", 0);
                break;
            }
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Macrophage: State step (M1/M2 polarization, cancer cell killing, lifespan)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    int cell_state = FLAMEGPU->getVariable<int>("cell_state");  // 0=M1, 1=M2
    int life = FLAMEGPU->getVariable<int>("life");

    // Decrement lifespan
    life--;
    if (life <= 0) {
        auto* evts_mac = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        atomicAdd(&evts_mac[cell_state == MAC_M1 ? EVT_DEATH_MAC_M1 : EVT_DEATH_MAC_M2], 1u);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    // M1/M2 polarization dynamics — read chemicals directly from PDE
    const int nx_ss = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_ss = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_ss = z * ny_ss*nx_ss + y * nx_ss + x;

    if (cell_state == MAC_M1) {
        float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel_ss);
        float IL10 = PDE_READ(FLAMEGPU, PDE_CONC_IL10, voxel_ss);

        double alpha = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M2_POL") * 
                            ((TGFB / (TGFB + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_EC50"))) + 
                            (IL10 / (IL10 + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_10_EC50"))));

        double p_M2_polar = 1 - std::exp(-alpha);

        if (FLAMEGPU->random.uniform<float>() < p_M2_polar) {
            FLAMEGPU->setVariable<int>("cell_state", MAC_M2);
        }
    }

    if (cell_state == MAC_M2){
        float IL12 = PDE_READ(FLAMEGPU, PDE_CONC_IL12, voxel_ss);
        float IFNg = PDE_READ(FLAMEGPU, PDE_CONC_IFN, voxel_ss);

        double alpha = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M1_POL") * 
                            ((IFNg / (IFNg + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IFN_G_EC50"))) + 
                            (IL12 / (IL12 + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_12_EC50"))));

        double p_M1_polar = 1 - std::exp(-alpha);

        if (FLAMEGPU->random.uniform<float>() < p_M1_polar) {
            FLAMEGPU->setVariable<int>("cell_state", MAC_M1);
        }
    }

    // ========== COMPUTE DERIVED STATES ==========
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;
    float local_IFNg = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_CONC_IFN))[voxel];

    float PDL1 = update_PDL1(local_IFNg,
         FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_HALF"),
         FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_PDL1_N"),
         FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_SYN_MAX"),
         FLAMEGPU->getVariable<float>("PDL1_syn"));

    FLAMEGPU->setVariable<float>("PDL1_syn", PDL1);

    return flamegpu::ALIVE;
}

}  // namespace PDAC

#endif
