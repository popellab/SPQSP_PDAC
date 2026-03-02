#ifndef FLAMEGPU_TNBC_T_CELL_CUH
#define FLAMEGPU_TNBC_T_CELL_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// TCell agent function: Broadcast location
FLAMEGPU_AGENT_FUNCTION(tcell_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_T);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    FLAMEGPU->message_out.setVariable<int>("cell_state", FLAMEGPU->getVariable<int>("cell_state"));
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f);  // T cells don't have PDL1
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);

    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );

    return flamegpu::ALIVE;
}

// TCell agent function: Scan neighbors and cache available voxels
// Counts cancer cells, Tregs, tracks max PDL1, looks for progenitors
// Also builds bitmask of neighbor voxels with room for T cells
FLAMEGPU_AGENT_FUNCTION(tcell_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
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
    int treg_count = 0;
    int all_count = 0;
    float max_PDL1 = 0.0f;
    int found_progenitor = 0;

    // Per-neighbor counts: [direction][0=cancer, 1=tcell, 2=treg]
    int neighbor_counts[26][3] = {{0}};

    for (const auto& msg : FLAMEGPU->message_in(my_pos_x, my_pos_y, my_pos_z)) {
        const int msg_x = msg.getVariable<int>("voxel_x");
        const int msg_y = msg.getVariable<int>("voxel_y");
        const int msg_z = msg.getVariable<int>("voxel_z");

        const int dx = msg_x - my_x;
        const int dy = msg_y - my_y;
        const int dz = msg_z - my_z;

        // Moore neighborhood (excluding self)
        if (abs(dx) <= 1 && abs(dy) <= 1 && abs(dz) <= 1 && !(dx == 0 && dy == 0 && dz == 0)) {
            all_count++;
            const int agent_type = msg.getVariable<int>("agent_type");
            const int agent_state = msg.getVariable<int>("cell_state");
            const float PDL1 = msg.getVariable<float>("PDL1");

            if (PDL1 > max_PDL1) {
                max_PDL1 = PDL1;
            }

            // Find direction index
            int dir_idx = -1;
            for (int i = 0; i < 26; i++) {
                int ddx, ddy, ddz;
                get_moore_direction(i, ddx, ddy, ddz);
                if (ddx == dx && ddy == dy && ddz == dz) {
                    dir_idx = i;
                    break;
                }
            }

            if (dir_idx >= 0) {
                if (agent_type == CELL_TYPE_CANCER) {
                    cancer_count++;
                    neighbor_counts[dir_idx][0]++;
                    if (agent_state == CANCER_PROGENITOR) {
                        found_progenitor = 1;
                    }
                } else if (agent_type == CELL_TYPE_T) {
                    neighbor_counts[dir_idx][1]++;
                } else if (agent_type == CELL_TYPE_TREG) {
                    treg_count++;
                    neighbor_counts[dir_idx][2]++;
                }
            }
        }
    }

    // Build available_neighbors mask (voxels with room for T cells)
    // Only scan Von Neumann neighbors for availability, can just skip other directions
    // Counts for interactions already calculated
    // unsigned int available_neighbors = 0;
    // for (int i = 0; i < 6; i++) {
    //     int dx, dy, dz;
    //     get_moore_direction(i, dx, dy, dz);
    //     int nx = my_x + dx;
    //     int ny = my_y + dy;
    //     int nz = my_z + dz;

    //     if (is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) {
    //         bool has_cancer = (neighbor_counts[i][0] > 0);
    //         int t_count = neighbor_counts[i][1] + neighbor_counts[i][2];
    //         int max_cap = has_cancer ? MAX_T_PER_VOXEL_WITH_CANCER : MAX_T_PER_VOXEL;

    //         if ((t_count < max_cap)) {
    //             available_neighbors |= (1u << i);
    //         }
    //     }
    // }

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_all_count", all_count);
    FLAMEGPU->setVariable<float>("max_neighbor_PDL1", max_PDL1);
    FLAMEGPU->setVariable<int>("found_progenitor", found_progenitor);
    // FLAMEGPU->setVariable<unsigned int>("available_neighbors", available_neighbors);

    return flamegpu::ALIVE;
}

// TCell agent function: State transitions
FLAMEGPU_AGENT_FUNCTION(tcell_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    // TODO: need coupling from QSP to get this
    float nivo = FLAMEGPU->environment.getProperty<float>("qsp_nivo_tumor");

    int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    // Life countdown
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    if (life <= 0) {
        FLAMEGPU->setVariable<int>("dead", 1);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    //Get cell specific parameters
    const int found_progenitor = FLAMEGPU->getVariable<int>("found_progenitor");

    // Get environment parameters
    const float sec_per_slice = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    // Update IL2 exposure — read directly from PDE
    const int nx_t = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_t = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax_t = FLAMEGPU->getVariable<int>("x");
    const int ay_t = FLAMEGPU->getVariable<int>("y");
    const int az_t = FLAMEGPU->getVariable<int>("z");
    const int voxel_t = az_t * ny_t*nx_t + ay_t * nx_t + ax_t;

    float IL2 = PDE_READ(FLAMEGPU, PDE_CONC_IL2, voxel_t);
    float IL2_exposure = FLAMEGPU->getVariable<float>("IL2_exposure");
    IL2_exposure += IL2 * sec_per_slice;

    // Effector cells proliferate on IL2 exposure
    if (IL2_exposure > FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_PROLIF_TH")) {
        FLAMEGPU->setVariable<int>("divide_flag", 1);
        FLAMEGPU->setVariable<float>("IL2_exposure", 0.0f);
    } else {
        FLAMEGPU->setVariable<float>("IL2_exposure",IL2_exposure);
    }

    // EFF -> CYT: Found cancer progenitor in neighborhood
    if (cell_state == T_CELL_EFF) {
        if (found_progenitor == 1) {
            FLAMEGPU->setVariable<int>("cell_state", T_CELL_CYT);
            FLAMEGPU->setVariable<int>("divide_flag", 1);
        }
    }

    // CYT state: update release timers and check exhaustion
    if (cell_state == T_CELL_CYT) {
        // Decrement release timer for IL2
        float IL2_release_remain = FLAMEGPU->getVariable<float>("IL2_release_remain");
        if (IL2_release_remain > 0) {
            IL2_release_remain -= sec_per_slice;
        }

        // Check for exhaustion
        const int neighbor_Treg = FLAMEGPU->getVariable<int>("neighbor_Treg_count");
        const int neighbor_all = FLAMEGPU->getVariable<int>("neighbor_all_count");
        const float max_PDL1 = FLAMEGPU->getVariable<float>("max_neighbor_PDL1");

        bool exhausted = false;
        if (neighbor_Treg > 0){
            const float param_cell = FLAMEGPU->environment.getProperty<float>("PARAM_CELL");
            const float exhaust_base_Treg = FLAMEGPU->environment.getProperty<float>("PARAM_EXHUAST_BASE_TREG");

            float denominator = neighbor_all + param_cell;
            const float q_exh = (denominator > 1e-12f) ? static_cast<float>(neighbor_Treg) / denominator : 1.0f;  // Prevent division by zero
            const float p_exhaust_treg = 1.0f - powf(exhaust_base_Treg, q_exh);

            if (FLAMEGPU->random.uniform<float>() < p_exhaust_treg){
                exhausted = true;
            }
        }
        if (neighbor_all > 0){
            float bond = get_PD1_PDL1(max_PDL1, nivo,
                        FLAMEGPU->environment.getProperty<float>("PARAM_PD1_SYN"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K1"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K2"),
                        FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_K3"));

            const float PD1_PDL1_half = FLAMEGPU->environment.getProperty<float>("PARAM_PD1_PDL1_HALF");
            const float n_PD1_PDL1 = FLAMEGPU->environment.getProperty<float>("PARAM_N_PD1_PDL1");
            float supp = hill_equation(bond, PD1_PDL1_half, n_PD1_PDL1);

            const float exhaust_base_PDL1 = FLAMEGPU->environment.getProperty<float>("PARAM_EXHUAST_BASE_PDL1");
            float p_exhaust_pdl1 = 1 - powf(exhaust_base_PDL1, supp);

            if (FLAMEGPU->random.uniform<float>() < p_exhaust_pdl1){
                exhausted = true;
            }
        }

        if (exhausted) {
                // Transition to suppressed
                FLAMEGPU->setVariable<int>("cell_state", T_CELL_SUPP);
                FLAMEGPU->setVariable<int>("divide_flag", 0);
                FLAMEGPU->setVariable<int>("divide_limit", 0);
        }
    }

    // Division cooldown
    int divide_cd = FLAMEGPU->getVariable<int>("divide_cd");
    if (divide_cd > 0) {
        divide_cd--;
        FLAMEGPU->setVariable<int>("divide_cd", divide_cd);
    }

    return flamegpu::ALIVE;
}

// ============================================================
// Occupancy Grid Functions
// ============================================================

// Write this T cell's position to the occupancy grid.
FLAMEGPU_AGENT_FUNCTION(tcell_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ[x][y][z][CELL_TYPE_T] += 1u;
    return flamegpu::ALIVE;
}

// Single-phase T cell division using occupancy grid.
// T cells can share a voxel up to MAX_T_PER_VOXEL.
// Uses atomicAdd to claim a slot; reverts with atomicSub if over capacity.
FLAMEGPU_AGENT_FUNCTION(tcell_divide, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int divide_flag  = FLAMEGPU->getVariable<int>("divide_flag");
    const int divide_cd    = FLAMEGPU->getVariable<int>("divide_cd");
    const int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
    const int cell_state   = FLAMEGPU->getVariable<int>("cell_state");

    if (FLAMEGPU->getVariable<int>("dead") == 1 ||
        divide_flag == 0 || divide_cd > 0 || divide_limit <= 0 ||
        cell_state != T_CELL_CYT) {
        return flamegpu::ALIVE;
    }

    const int my_x  = FLAMEGPU->getVariable<int>("x");
    const int my_y  = FLAMEGPU->getVariable<int>("y");
    const int my_z  = FLAMEGPU->getVariable<int>("z");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int size_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // Collect Von Neumann neighbors below T cell capacity
    int cand_x[6], cand_y[6], cand_z[6];
    int n_cands = 0;
    unsigned int max_cap[6];
    for (int i = 0; i < 6; i++) {
        int dx, dy, dz;
        get_moore_direction(i, dx, dy, dz);
        const int nx = my_x + dx, ny = my_y + dy, nz = my_z + dz;
        if (!is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) continue;

        bool has_cancer = (occ[nx][ny][nz][CELL_TYPE_CANCER] == 0u);
        max_cap[n_cands] = static_cast<unsigned int>(has_cancer ? MAX_T_PER_VOXEL_WITH_CANCER : MAX_T_PER_VOXEL);

        if (occ[nx][ny][nz][CELL_TYPE_T] < max_cap[n_cands]) {
            cand_x[n_cands] = nx;
            cand_y[n_cands] = ny;
            cand_z[n_cands] = nz;
            n_cands++;
        }
    }

    if (n_cands == 0) return flamegpu::ALIVE;

    const int div_interval  = FLAMEGPU->environment.getProperty<int>("PARAM_TCELL_DIV_INTERNAL");
    const float IL2_release_time = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME");
    const float tcell_life_mean  = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
    const float IL2_exposure     = FLAMEGPU->getVariable<float>("IL2_exposure");

    // Fisher-Yates partial shuffle; try each candidate
    for (int i = 0; i < n_cands; i++) {
        const int j = i + static_cast<int>(FLAMEGPU->random.uniform<float>() * (n_cands - i));
        int tx = cand_x[i]; cand_x[i] = cand_x[j]; cand_x[j] = tx;
        int ty = cand_y[i]; cand_y[i] = cand_y[j]; cand_y[j] = ty;
        int tz = cand_z[i]; cand_z[i] = cand_z[j]; cand_z[j] = tz;
        unsigned int max_curr = max_cap[i]; max_cap[i] = max_cap[j]; max_cap[j] = max_curr;

        // Atomically increment; undo if this pushed count over capacity
        // operator+ performs atomicAdd and returns the OLD value
        const unsigned int old_count = occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_T] + 1u;
        if (old_count >= max_curr) {
            occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_T] -= 1u;  // undo
            continue;  // Voxel was full, try next
        }

        // Won a slot — create daughter
        const float tcell_life_sd = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");
        float tLifeD = tcell_life_mean + FLAMEGPU->random.normal<float>() * tcell_life_sd;
        int daughter_life = static_cast<int>(tLifeD + 0.5f);
        daughter_life = (daughter_life > 0) ? daughter_life : 1;

        FLAMEGPU->agent_out.setVariable<int>("x", cand_x[i]);
        FLAMEGPU->agent_out.setVariable<int>("y", cand_y[i]);
        FLAMEGPU->agent_out.setVariable<int>("z", cand_z[i]);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", cell_state);
        FLAMEGPU->agent_out.setVariable<int>("divide_flag", 0);
        FLAMEGPU->agent_out.setVariable<int>("divide_cd", div_interval);
        FLAMEGPU->agent_out.setVariable<int>("divide_limit", divide_limit - 1);
        FLAMEGPU->agent_out.setVariable<float>("IL2_exposure", IL2_exposure);
        FLAMEGPU->agent_out.setVariable<float>("IL2_release_remain", IL2_release_time);
        FLAMEGPU->agent_out.setVariable<int>("life", daughter_life > 0 ? daughter_life : 1);

        // Update parent
        FLAMEGPU->setVariable<int>("divide_limit", divide_limit - 1);
        FLAMEGPU->setVariable<int>("divide_cd", div_interval);

        // Track T cell proliferation event
        atomicAdd(&reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_tcell_prolif_ptr"))[0], 1u);

        break;  // Division done
    }

    // Clear parent intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    return flamegpu::ALIVE;
}

FLAMEGPU_AGENT_FUNCTION(tcell_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    return flamegpu::ALIVE;
}

// T Cell agent function: Compute chemical sources
// atomicAdds directly to PDE source/uptake arrays
FLAMEGPU_AGENT_FUNCTION(tcell_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead = FLAMEGPU->getVariable<int>("dead");
    int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    float IFNg_release_rate = 0.0f;
    float IL2_release_rate = 0.0f;
    float IL2_uptake_rate = 0.0f;
    if (dead == 0){
        int found_progenitor = FLAMEGPU->getVariable<int>("found_progenitor");
        if (cell_state == T_CELL_EFF && found_progenitor == 1){
            IFNg_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_RELEASE");
            IL2_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_RELEASE");
            IL2_uptake_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_UPTAKE");
        } else if (cell_state == T_CELL_CYT) {
            IFNg_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_RELEASE");
            IL2_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_RELEASE");
            IL2_uptake_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_UPTAKE");
            if (found_progenitor == 0){
                IFNg_release_rate = 0.0f;
            }
            float IL2_release_remain = FLAMEGPU->getVariable<float>("IL2_release_remain");
            if (IL2_release_remain <= 0){
                IL2_release_rate = 0.0f;
            }
        }
    }

    // Compute voxel index
    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * ny*nx + ay * nx + ax;

    const float vs_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1.0e-4f;
    const float voxel_volume = vs_cm * vs_cm * vs_cm;

    // IFN-γ secretion → src ptr 1 (IFN), divide by voxel_volume
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IFN, voxel, IFNg_release_rate / voxel_volume);

    // IL-2 secretion → src ptr 2 (IL2), divide by voxel_volume
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IL2, voxel, IL2_release_rate / voxel_volume);

    // IL-2 uptake → upt ptr 2 (IL2), positive [1/s], no volume scaling
    PDE_UPTAKE(FLAMEGPU, PDE_UPT_IL2, voxel, IL2_uptake_rate);

    return flamegpu::ALIVE;
}

// Single-phase T cell movement using occupancy grid.
// Replaces two-phase select_move_target + execute_move.
// T cells use atomicAdd+undo to allow up to MAX_T_PER_VOXEL per voxel.
// Fewer T cells are allowed in voxels occupied by cancer (MAX_T_PER_VOXEL_WITH_CANCER).
// T cell movement using run-tumble chemotaxis toward IFN-γ/CCL2
FLAMEGPU_AGENT_FUNCTION(tcell_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // ECM based movement probability: higher ECM → more likely to be blocked
    auto ecm = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("ecm_grid");
    float ECM_density = static_cast<float>(ecm[x][y][z]);
    float ECM_50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50");
    float ECM_sat = ECM_density / (ECM_density + ECM_50);
    if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    // Use IFN-γ gradient for chemotaxis (T cell primary attractant) — read directly from PDE
    const int nx_mv = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_mv = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_mv = z * ny_mv*nx_mv + y * nx_mv + x;
    const float grad_x = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IFN_X))[voxel_mv];
    const float grad_y = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IFN_Y))[voxel_mv];
    const float grad_z = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IFN_Z))[voxel_mv];

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    int target_x = x;
    int target_y = y;
    int target_z = z;

    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        float v_x = move_dir_x / dt;
        float v_y = move_dir_y / dt;
        float v_z = move_dir_z / dt;

        float norm_gradient = std::sqrt(grad_x * grad_x + grad_y * grad_y + grad_z * grad_z);

        // Compute alignment to gradient
        float dot_product = v_x * grad_x + v_y * grad_y + v_z * grad_z;
        float norm_v = std::sqrt(v_x * v_x + v_y * v_y + v_z * v_z);
        float cos_theta = dot_product / (norm_v * norm_gradient);

        // Hill function: bias tumble rate based on gradient alignment and strength
        const float EC50_grad = 1.0f;
        float H_grad = norm_gradient / (norm_gradient + EC50_grad);
        if (cos_theta < 0) H_grad = -H_grad;

        const float lambda = 0.0000168;  // Base tumble rate
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

        // Check bounds
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

        // Sample all 26 Moore neighbors
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

        // Weighted random selection of next direction
        float r = FLAMEGPU->random.uniform<float>();
        int selected_idx = 0;
        for (int i = 0; i < n_dirs; i++) {
            if (r < probs[i]) {
                selected_idx = i;
                break;
            }
        }

        // Set new movement direction
        FLAMEGPU->setVariable<float>("move_direction_x", static_cast<float>(dirs[selected_idx][0]));
        FLAMEGPU->setVariable<float>("move_direction_y", static_cast<float>(dirs[selected_idx][1]));
        FLAMEGPU->setVariable<float>("move_direction_z", static_cast<float>(dirs[selected_idx][2]));
        FLAMEGPU->setVariable<int>("tumble", 0);

        target_x = x + dirs[selected_idx][0];
        target_y = y + dirs[selected_idx][1];
        target_z = z + dirs[selected_idx][2];
    }

    // Try to move to target voxel (if valid and has capacity)
    if (target_x >= 0 && target_x < grid_x &&
        target_y >= 0 && target_y < grid_y &&
        target_z >= 0 && target_z < grid_z) {

        unsigned int max_t = (occ[target_x][target_y][target_z][CELL_TYPE_CANCER] > 0u)
            ? static_cast<unsigned int>(MAX_T_PER_VOXEL_WITH_CANCER)
            : static_cast<unsigned int>(MAX_T_PER_VOXEL);

        if (occ[target_x][target_y][target_z][CELL_TYPE_T] < max_t) {
            // Won: release old voxel and move
            occ[x][y][z][CELL_TYPE_T] -= 1u;
            FLAMEGPU->setVariable<int>("x", target_x);
            FLAMEGPU->setVariable<int>("y", target_y);
            FLAMEGPU->setVariable<int>("z", target_z);
        }
    }

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_T_CELL_CUH