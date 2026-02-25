// Macrophage Agent Behavior Functions
// M1/M2 polarization, CCL2-based recruitment and chemotaxis, cancer cell killing

#ifndef MACROPHAGE_CUH
#define MACROPHAGE_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// Macrophage Utility Functions
// ============================================================================

// Hill equation for ECM saturation
__device__ float mac_ecm_saturation(float ecm_density, float ec50) {
    if (ec50 <= 0.0f) return 0.0f;
    return ecm_density / (ecm_density + ec50);
}

// Hill equation for CCL2 recruitment
__device__ float mac_ccl2_hill(float ccl2, float ec50) {
    if (ec50 <= 0.0f) return 0.0f;
    return ccl2 / (ccl2 + ec50);
}

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
    FLAMEGPU->message_out.setVariable<int>("x", x);
    FLAMEGPU->message_out.setVariable<int>("y", y);
    FLAMEGPU->message_out.setVariable<int>("z", z);
    FLAMEGPU->message_out.setLocation(-voxel_size, -voxel_size, -voxel_size);
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
    return flamegpu::ALIVE;
}

// ============================================================================
// Macrophage: Scan neighbors (count adjacent cancer cells for killing)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    int neighbor_cancer_count = 0;

    for (auto& msg : FLAMEGPU->message_in(FLAMEGPU->getVariable<int>("x"),
                                           FLAMEGPU->getVariable<int>("y"),
                                           FLAMEGPU->getVariable<int>("z"))) {
        int agent_type = msg.getVariable<int>("agent_type");
        if (agent_type == CELL_TYPE_CANCER) {
            neighbor_cancer_count++;
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
    const int cell_state = FLAMEGPU->getVariable<int>("mac_state");
    const int dead = FLAMEGPU->getVariable<int>("dead");

    // Dead cells don't produce or consume
    if (dead == 1) {
        return flamegpu::ALIVE;
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
    const float CCL2_uptake = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_UPTAKE");
    atomicAdd(&reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_uptake_ptr_5"))[voxel],
        CCL2_uptake);

    if (cell_state == MAC_M1) {
        // IFN-γ secretion → src ptr 1 (IFN)
        atomicAdd(&reinterpret_cast<float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_1"))[voxel],
            FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_RELEASE") / voxel_volume);
        // IL-12 secretion → src ptr 8 (IL12)
        atomicAdd(&reinterpret_cast<float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_8"))[voxel],
            FLAMEGPU->environment.getProperty<float>("PARAM_IL12_RELEASE") / voxel_volume);
    } else if (cell_state == MAC_M2) {
        // TGF-β secretion → src ptr 4 (TGFB)
        atomicAdd(&reinterpret_cast<float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_4"))[voxel],
            FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_RELEASE") / voxel_volume);
        // IL-10 secretion → src ptr 3 (IL10)
        atomicAdd(&reinterpret_cast<float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_3"))[voxel],
            FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL10_RELEASE") / voxel_volume);
        // VEGF-A secretion → src ptr 9 (VEGFA)
        atomicAdd(&reinterpret_cast<float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_source_ptr_9"))[voxel],
            FLAMEGPU->environment.getProperty<float>("PARAM_MAC_VEGFA_RELEASE") / voxel_volume);
    }

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

    // ECM based movement probability - TEMPORARILY DISABLED
    // auto ecm = FLAMEGPU->environment.getMacroProperty<float,
    //     OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("ecm_grid");
    // float ECM_density = ecm[x][y][z];
    // double ECM_sat = ECM_density / (ECM_density + FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50"));
    // if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;
    if (FLAMEGPU->random.uniform<float>() < 0.2f) return flamegpu::ALIVE;

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    // Use CCL2 gradient for chemotaxis — read directly from PDE
    const int nx_mv = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_mv = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_mv = z * ny_mv*nx_mv + y * nx_mv + x;
    const float grad_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_grad_CCL2_x"))[voxel_mv];
    const float grad_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_grad_CCL2_y"))[voxel_mv];
    const float grad_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("pde_grad_CCL2_z"))[voxel_mv];

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    int target_x = x;
    int target_y = y;
    int target_z = z;

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

        // Continue running: move in current direction
        target_x = x + static_cast<int>(std::round(move_dir_x));
        target_y = y + static_cast<int>(std::round(move_dir_y));
        target_z = z + static_cast<int>(std::round(move_dir_z));

        if (target_x < 0 || target_x >= grid_x ||
            target_y < 0 || target_y >= grid_y ||
            target_z < 0 || target_z >= grid_z) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }
    }
    // === TUMBLE PHASE (tumble == 1) ===
    else {
        const float sigma = 0.524f;
        float prob_sum = 0.0f;
        float probs[26];
        int dirs[26][3];
        int n_dirs = 0;

        float norm_movedir = std::sqrt(move_dir_x * move_dir_x +
                                       move_dir_y * move_dir_y +
                                       move_dir_z * move_dir_z);
        if (norm_movedir < 1e-6f) norm_movedir = 1.0f;

        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                for (int k = -1; k <= 1; k++) {
                    if (i == 0 && j == 0 && k == 0) continue;
                    float dot_product = i * move_dir_x + j * move_dir_y + k * move_dir_z;
                    float norm_dir = std::sqrt(static_cast<float>(i*i + j*j + k*k));
                    float cos_theta = dot_product / (norm_dir * norm_movedir);
                    if (cos_theta > 0) {
                        float rho = std::exp(cos_theta / (sigma * sigma)) / std::exp(1.0f / (sigma * sigma));
                        prob_sum += rho;
                        probs[n_dirs] = prob_sum;
                        dirs[n_dirs][0] = i;
                        dirs[n_dirs][1] = j;
                        dirs[n_dirs][2] = k;
                        n_dirs++;
                    }
                }
            }
        }

        if (n_dirs == 0) {
            FLAMEGPU->setVariable<int>("tumble", 0);
            return flamegpu::ALIVE;
        }

        for (int i = 0; i < n_dirs; i++) probs[i] /= prob_sum;

        float r = FLAMEGPU->random.uniform<float>();
        int selected_idx = 0;
        for (int i = 0; i < n_dirs; i++) {
            if (r < probs[i]) { selected_idx = i; break; }
        }

        FLAMEGPU->setVariable<float>("move_direction_x", static_cast<float>(dirs[selected_idx][0]));
        FLAMEGPU->setVariable<float>("move_direction_y", static_cast<float>(dirs[selected_idx][1]));
        FLAMEGPU->setVariable<float>("move_direction_z", static_cast<float>(dirs[selected_idx][2]));
        FLAMEGPU->setVariable<int>("tumble", 0);

        target_x = x + dirs[selected_idx][0];
        target_y = y + dirs[selected_idx][1];
        target_z = z + dirs[selected_idx][2];

        if (target_x < 0 || target_x >= grid_x ||
            target_y < 0 || target_y >= grid_y ||
            target_z < 0 || target_z >= grid_z) {
            return flamegpu::ALIVE;
        }
    }

    // Try to claim target voxel (CAS — macrophages are exclusive per voxel)
    if (target_x != x || target_y != y || target_z != z) {
        if (occ[target_x][target_y][target_z][CELL_TYPE_MAC].CAS(0u, 1u) == 0u) {
            occ[x][y][z][CELL_TYPE_MAC].exchange(0u);
            FLAMEGPU->setVariable<int>("x", target_x);
            FLAMEGPU->setVariable<int>("y", target_y);
            FLAMEGPU->setVariable<int>("z", target_z);
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
    int mac_state = FLAMEGPU->getVariable<int>("mac_state");  // 0=M1, 1=M2
    int life = FLAMEGPU->getVariable<int>("life");

    // Decrement lifespan
    life--;
    if (life <= 0) {
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    // M1/M2 polarization dynamics — read chemicals directly from PDE
    const int nx_ss = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_ss = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_ss = z * ny_ss*nx_ss + y * nx_ss + x;

    if (mac_state == MAC_M1) {
        float TGFB = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_concentration_ptr_4"))[voxel_ss];
        float IL10 = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_concentration_ptr_3"))[voxel_ss];

        double alpha = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M2_POL") * 
                            ((TGFB / (TGFB + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_EC50"))) + 
                            (IL10 / (IL10 + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_10_EC50"))));

        double p_M2_polar = 1 - std::exp(-alpha);

        if (FLAMEGPU->random.uniform<float>() < p_M2_polar) {
            FLAMEGPU->setVariable<int>("mac_state", MAC_M2);
        }
    }

    if (mac_state == MAC_M2){
        float IL12 = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_concentration_ptr_8"))[voxel_ss];
        float IFNg = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>("pde_concentration_ptr_1"))[voxel_ss];

        double alpha = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M1_POL") * 
                            ((IFNg / (IFNg + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IFN_G_EC50"))) + 
                            (IL12 / (IL12 + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_12_EC50"))));

        double p_M1_polar = 1 - std::exp(-alpha);

        if (FLAMEGPU->random.uniform<float>() < p_M1_polar) {
            FLAMEGPU->setVariable<int>("mac_state", MAC_M1);
        }
    }

    return flamegpu::ALIVE;
}

}  // namespace PDAC

#endif
