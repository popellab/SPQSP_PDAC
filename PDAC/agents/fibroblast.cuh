// Fibroblast / Cancer-Associated Fibroblast (CAF) Agent Behavior Functions
// States: FIB_NORMAL (quiescent fibroblast), FIB_CAF (activated CAF)
// Activation: TGFB-driven NORMAL->CAF transition
// Chemotaxis: Run-tumble toward TGFB gradient
// CAFs secrete TGFB (positive feedback) and support tumor

#ifndef FIBROBLAST_CUH
#define FIBROBLAST_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// ============================================================================
// Fibroblast: Broadcast location (spatial messaging)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float fx = (x + 0.5f) * voxel_size;
    const float fy = (y + 0.5f) * voxel_size;
    const float fz = (z + 0.5f) * voxel_size;

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_FIB);
    FLAMEGPU->message_out.setVariable<unsigned int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("x", x);
    FLAMEGPU->message_out.setVariable<int>("y", y);
    FLAMEGPU->message_out.setVariable<int>("z", z);
    FLAMEGPU->message_out.setLocation(fx, fy, fz);
    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Write to occupancy grid (exclusive per voxel)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    occ[x][y][z][CELL_TYPE_FIB].exchange(1u);
    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Compute chemical sources
// CAFs secrete TGFB; normal fibroblasts do not
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int fib_state = FLAMEGPU->getVariable<int>("fib_state");

    if (fib_state == FIB_CAF) {
        FLAMEGPU->setVariable<float>("TGFB_release_rate",
            FLAMEGPU->environment.getProperty<float>("PARAM_FIB_TGFB_RELEASE"));
    } else {
        FLAMEGPU->setVariable<float>("TGFB_release_rate", 0.0f);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Single-phase movement using occupancy grid (exclusive per voxel)
// Uses run-tumble chemotaxis toward TGFB gradient
// CAFs use a higher tumble rate (10x lambda) for greater motility
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");
    const int fib_state = FLAMEGPU->getVariable<int>("fib_state");

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

    // Use TGFB gradient for chemotaxis
    const float grad_x = FLAMEGPU->getVariable<float>("tgfb_grad_x");
    const float grad_y = FLAMEGPU->getVariable<float>("tgfb_grad_y");
    const float grad_z = FLAMEGPU->getVariable<float>("tgfb_grad_z");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    // CAFs are 10x more motile than normal fibroblasts (HCC reference)
    const float lambda = (fib_state == FIB_CAF) ? 0.000168f : 0.0000168f;

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

    // Try to claim target voxel (CAS — fibroblasts are exclusive per voxel)
    // Also require no cancer cell in the target voxel
    if (target_x != x || target_y != y || target_z != z) {
        if (occ[target_x][target_y][target_z][CELL_TYPE_CANCER] == 0u &&
            occ[target_x][target_y][target_z][CELL_TYPE_FIB].CAS(0u, 1u) == 0u) {
            occ[x][y][z][CELL_TYPE_FIB].exchange(0u);
            FLAMEGPU->setVariable<int>("x", target_x);
            FLAMEGPU->setVariable<int>("y", target_y);
            FLAMEGPU->setVariable<int>("z", target_z);
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: State step — TGFB-driven activation (NORMAL -> CAF) and lifespan
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    int fib_state = FLAMEGPU->getVariable<int>("fib_state");
    int life = FLAMEGPU->getVariable<int>("life");

    // Decrement lifespan
    life--;
    if (life <= 0) {
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    // NORMAL → CAF activation driven by TGFB concentration
    if (fib_state == FIB_NORMAL) {
        float TGFB = FLAMEGPU->getVariable<float>("local_TGFB");
        float ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_EC50");
        float activation = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_ACTIVATION");

        // activation_coeff = activation * 5 * (1 + TGFB / (TGFB + EC50))  [HCC formula]
        double activation_coeff = activation * 5.0 * (1.0 + (TGFB / (TGFB + ec50)));
        double p_activation = 1.0 - std::exp(-activation_coeff);

        if (FLAMEGPU->random.uniform<float>() < p_activation) {
            FLAMEGPU->setVariable<int>("fib_state", FIB_CAF);
        }
    }

    return flamegpu::ALIVE;
}

}  // namespace PDAC

#endif  // FIBROBLAST_CUH
