// Fibroblast / Cancer-Associated Fibroblast (CAF) Agent Behavior Functions
// States: FIB_NORMAL (quiescent fibroblast), FIB_CAF (activated CAF)
// Activation: TGFB-driven NORMAL->CAF transition
//
// Movement: Follower-leader chain model
//   - Cells form chains (HEAD → MIDDLE → ... → TAIL)
//   - HEAD (leader_slot == -1): runs TGFB run-tumble chemotaxis
//   - Followers: move to where their leader WAS last step (caterpillar motion)
//   - MacroProperty arrays fib_pos_x/y/z store snapshot positions
//   - MacroProperty array fib_moved flags which cells moved this step

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
// CAFs secrete TGFB — atomicAdd directly to PDE source array
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    if (cell_state == FIB_CAF) {
        const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int ax = FLAMEGPU->getVariable<int>("x");
        const int ay = FLAMEGPU->getVariable<int>("y");
        const int az = FLAMEGPU->getVariable<int>("z");
        const int voxel = az * ny*nx + ay * nx + ax;

        const float vs_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
        const float voxel_volume = vs_cm * vs_cm * vs_cm;

        // TGF-β secretion → src ptr 4 (TGFB)
        PDE_SECRETE(FLAMEGPU, PDE_SRC_TGFB, voxel, FLAMEGPU->environment.getProperty<float>("PARAM_FIB_TGFB_RELEASE") / voxel_volume);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Write position snapshot to MacroProperty
// Resets fib_moved[my_slot] = 0 for this step
// Must run BEFORE sensor_move and follow_move each step
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_write_pos, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int my_slot = FLAMEGPU->getVariable<int>("my_slot");
    if (my_slot < 0) return flamegpu::ALIVE;  // Uninitialized, skip

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    auto fib_pos_x = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_x");
    auto fib_pos_y = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_y");
    auto fib_pos_z = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_z");
    auto fib_moved = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_moved");

    fib_pos_x[my_slot].exchange(x);
    fib_pos_y[my_slot].exchange(y);
    fib_pos_z[my_slot].exchange(z);
    fib_moved[my_slot].exchange(0);

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: HEAD/sensor movement — TGFB run-tumble chemotaxis
// Only runs for cells where leader_slot == -1 (they are the chain front)
// Uses CAS on occ_grid for exclusivity
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_sensor_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int leader_slot = FLAMEGPU->getVariable<int>("leader_slot");
    if (leader_slot != -1) return flamegpu::ALIVE;  // Not a sensor (HEAD), skip

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // ECM-based movement probability (simplified)
    if (FLAMEGPU->random.uniform<float>() < 0.2f) return flamegpu::ALIVE;

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    // Use TGFB gradient for chemotaxis — read directly from PDE
    const int nx_mv = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_mv = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_mv = z * ny_mv*nx_mv + y * nx_mv + x;
    const float grad_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_X))[voxel_mv];
    const float grad_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Y))[voxel_mv];
    const float grad_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Z))[voxel_mv];

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // CAFs are more motile than normal fibroblasts
    const float lambda = (cell_state == FIB_CAF) ? 2.0f : 0.5f;
    const float delta = 1.0f;
    const float EC50_grad = 1e-10f;
    const float sigma = 0.524f;

    int target_x = x;
    int target_y = y;
    int target_z = z;

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        float norm_gradient = std::sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

        // Switch to tumble if gradient is too weak
        if (norm_gradient < EC50_grad) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }

        float norm_v = std::sqrt(move_dir_x * move_dir_x + move_dir_y * move_dir_y + move_dir_z * move_dir_z);
        if (norm_v < 1e-6f) norm_v = 1.0f;

        float dot_product = move_dir_x * grad_x + move_dir_y * grad_y + move_dir_z * grad_z;
        float cos_theta = dot_product / (norm_v * norm_gradient);

        float H_grad = norm_gradient / (norm_gradient + EC50_grad);
        if (cos_theta < 0) H_grad = -H_grad;

        float p_tumble = 1.0f - std::exp(-0.5f * lambda * (1.0f - cos_theta) * (1.0f - H_grad) + delta);
        p_tumble = fmaxf(0.0f, fminf(1.0f, p_tumble));

        if (FLAMEGPU->random.uniform<float>() < p_tumble) {
            FLAMEGPU->setVariable<int>("tumble", 1);
            return flamegpu::ALIVE;
        }

        // Continue running in current direction
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
            // Successfully moved: release old voxel
            occ[x][y][z][CELL_TYPE_FIB].exchange(0u);
            FLAMEGPU->setVariable<int>("x", target_x);
            FLAMEGPU->setVariable<int>("y", target_y);
            FLAMEGPU->setVariable<int>("z", target_z);

            // Signal to followers that we moved
            const int my_slot = FLAMEGPU->getVariable<int>("my_slot");
            if (my_slot >= 0) {
                auto fib_moved = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_moved");
                fib_moved[my_slot].exchange(1);
            }
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Follower movement — moves to leader's snapshot position if leader moved
// Runs in separate layers to propagate movement through chain
// Layer 0: cells whose leader is the HEAD move (1 step from HEAD)
// Layer 1: cells whose leader moved in layer 0 move (2 steps from HEAD)
// Uses CAS on occ_grid for correctness with cross-chain conflicts
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_follow_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int leader_slot = FLAMEGPU->getVariable<int>("leader_slot");
    if (leader_slot == -1) return flamegpu::ALIVE;  // HEAD/sensor, handled by fib_sensor_move

    const int my_slot = FLAMEGPU->getVariable<int>("my_slot");
    if (my_slot < 0) return flamegpu::ALIVE;

    // Check if our leader moved this step
    auto fib_moved = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_moved");
    if (static_cast<int>(fib_moved[leader_slot]) == 0) return flamegpu::ALIVE;  // Leader didn't move

    // Get leader's snapshot position (from fib_write_pos at start of step)
    auto fib_pos_x = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_x");
    auto fib_pos_y = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_y");
    auto fib_pos_z = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_z");

    const int target_x = static_cast<int>(fib_pos_x[leader_slot]);
    const int target_y = static_cast<int>(fib_pos_y[leader_slot]);
    const int target_z = static_cast<int>(fib_pos_z[leader_slot]);

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    if (target_x == x && target_y == y && target_z == z) return flamegpu::ALIVE;  // Already there

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // CAS to claim the target voxel (leader should have vacated it)
    if (occ[target_x][target_y][target_z][CELL_TYPE_FIB].CAS(0u, 1u) == 0u) {
        // Claimed successfully: release old voxel, update position
        occ[x][y][z][CELL_TYPE_FIB].exchange(0u);
        FLAMEGPU->setVariable<int>("x", target_x);
        FLAMEGPU->setVariable<int>("y", target_y);
        FLAMEGPU->setVariable<int>("z", target_z);

        // Signal to our followers that we moved
        fib_moved[my_slot].exchange(1);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: State step — TGFB-driven activation (NORMAL -> CAF) and lifespan
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {

    const int cell_state   = FLAMEGPU->getVariable<int>("cell_state");
    const int my_slot     = FLAMEGPU->getVariable<int>("my_slot");
    const int leader_slot = FLAMEGPU->getVariable<int>("leader_slot");

    // Only HEAD cells in a chain that are still NORMAL can divide
    // HEAD = my_slot >= 0 AND leader_slot != -1 (not TAIL)
    if (my_slot < 0 || leader_slot == -1 || cell_state != FIB_NORMAL) {
        return flamegpu::ALIVE;
    }

    const int nx_ss = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_ss = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax_ss = FLAMEGPU->getVariable<int>("x");
    const int ay_ss = FLAMEGPU->getVariable<int>("y");
    const int az_ss = FLAMEGPU->getVariable<int>("z");
    const int voxel_ss = az_ss * ny_ss*nx_ss + ay_ss * nx_ss + ax_ss;
    const float TGFB = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_CONC_TGFB))[voxel_ss];
    const float ec50     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_EC50");
    const float caf_act  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_ACTIVATION");
    const float activation = caf_act * 5 * (1 + TGFB / (TGFB + ec50));
    const float p_div  = 1.0f - expf(-activation);

    if (FLAMEGPU->random.uniform<float>() < p_div) {
        FLAMEGPU->setVariable<int>("divide_flag", 1);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Build Gaussian density field for ECM deposition
// Each fibroblast scatters a 3D Gaussian kernel (radius=10, sigma=3) to the
// fib_density_field MacroProperty. CAFs contribute with scale=1.0, normals 0.5.
// TGFB amplification is pre-multiplied at the fibroblast's voxel.
// update_ecm_grid host fn then uses this field to deposit ECM.
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_build_density_field, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cx = FLAMEGPU->getVariable<int>("x");
    const int cy = FLAMEGPU->getVariable<int>("y");
    const int cz = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Read TGFB directly from PDE
    const int voxel_bd = cz * grid_y*grid_x + cy * grid_x + cx;
    const float local_TGFB = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_CONC_TGFB))[voxel_bd];

    const float ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_EC50");
    const float scale = (cell_state == FIB_CAF) ? 1.0f : 0.5f;

    // Gaussian parameters matching HCC (radius=10, sigma=3)
    const int radius = 10;
    const float variance = 9.0f;  // sigma^2 = 3^2
    // normalizer = 1 / (sigma^3 * sqrt(2*pi)) = 1 / (27 * 2.5066) ≈ 0.014784
    const float normalizer = 0.014784f;

    // TGFB amplification factor at fibroblast's own voxel (pre-multiplied approximation)
    const float H_TGFB = local_TGFB / (local_TGFB + ec50);
    const float tgfb_scale = scale * (1.0f + H_TGFB);

    auto field = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("fib_density_field");

    for (int dx = -radius; dx <= radius; dx++) {
        const int nx = cx + dx;
        if (nx < 0 || nx >= grid_x) continue;
        for (int dy = -radius; dy <= radius; dy++) {
            const int ny = cy + dy;
            if (ny < 0 || ny >= grid_y) continue;
            for (int dz = -radius; dz <= radius; dz++) {
                const int nz = cz + dz;
                if (nz < 0 || nz >= grid_z) continue;
                float dist_sq = static_cast<float>(dx*dx + dy*dy + dz*dz);
                float kernel_val = tgfb_scale * normalizer * expf(-dist_sq / (2.0f * variance));
                field[nx][ny][nz] += kernel_val;  // atomicAdd via MacroProperty +=
            }
        }
    }

    return flamegpu::ALIVE;
}

}  // namespace PDAC

#endif  // FIBROBLAST_CUH
