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
    const int tcell_cs = FLAMEGPU->getVariable<int>("cell_state");
    FLAMEGPU->message_out.setVariable<int>("cell_state", tcell_cs);
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f);  // T cells don't have PDL1
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);

    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );

    // Count this agent into per-state population snapshot
    auto* sc = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    const int sc_slot = (tcell_cs == T_CELL_EFF) ? SC_CD8_EFF :
                        (tcell_cs == T_CELL_CYT)  ? SC_CD8_CYT : SC_CD8_SUP;
    atomicAdd(&sc[sc_slot], 1u);

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
                    if (agent_state == TCD4_TREG) {
                        treg_count++;
                    }
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
        auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        const int ds = (cell_state == T_CELL_EFF) ? EVT_DEATH_CD8_EFF :
                       (cell_state == T_CELL_CYT)  ? EVT_DEATH_CD8_CYT : EVT_DEATH_CD8_SUP;
        atomicAdd(&evts[ds], 1u);
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
            cell_state = T_CELL_CYT;
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
            FLAMEGPU->setVariable<float>("IL2_release_remain", IL2_release_remain);
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
            const float q_exh = static_cast<float>(neighbor_Treg) / denominator;
            const float p_exhaust_treg = 1.0f - powf(exhaust_base_Treg, q_exh);

            if (FLAMEGPU->random.uniform<float>() < p_exhaust_treg){
                exhausted = true;
            }
        }
        else if (neighbor_all > 0){
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

    // === WAVE ASSIGNMENT ===
    {
        const int divide_flag_now = FLAMEGPU->getVariable<int>("divide_flag");
        const int divide_limit_now = FLAMEGPU->getVariable<int>("divide_limit");
        if (divide_flag_now == 1 && divide_cd <= 0 && divide_limit_now > 0) {
            const int w = static_cast<int>(FLAMEGPU->random.uniform<float>() * N_DIVIDE_WAVES);
            FLAMEGPU->setVariable<int>("divide_wave", w < N_DIVIDE_WAVES ? w : N_DIVIDE_WAVES - 1);
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================
// Occupancy Grid Functions
// ============================================================

// Write this T cell's volume to the volume occupancy grid.
FLAMEGPU_AGENT_FUNCTION(tcell_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (gx * gy) + y * gx + x;

    // Volume-based occupancy
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    float my_vol = (cell_state == T_CELL_EFF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_EFF") :
        (cell_state == T_CELL_CYT) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_CYT") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_SUP");
    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

    return flamegpu::ALIVE;
}

// Single-phase T cell division using occupancy grid.
// T cell division uses volume-based occupancy for daughter placement.
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
    // Wave gate: only execute in the assigned wave round
    if (FLAMEGPU->getVariable<int>("divide_wave") !=
        FLAMEGPU->environment.getProperty<int>("divide_current_wave")) {
        return flamegpu::ALIVE;
    }

    const int my_x  = FLAMEGPU->getVariable<int>("x");
    const int my_y  = FLAMEGPU->getVariable<int>("y");
    const int my_z  = FLAMEGPU->getVariable<int>("z");
    const int size_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int size_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int size_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Volume-based occupancy
    float* vol_used = VOL_PTR(FLAMEGPU);
    const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    float daughter_vol = (cell_state == T_CELL_EFF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_EFF") :
        (cell_state == T_CELL_CYT) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_CYT") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_SUP");

    // Collect Moore (26-direction) neighbors with volume capacity
    int cand_x[26], cand_y[26], cand_z[26];
    int n_cands = 0;
    for (int i = 0; i < 26; i++) {
        int dx, dy, dz;
        get_moore_direction(i, dx, dy, dz);
        const int nx = my_x + dx, ny = my_y + dy, nz = my_z + dz;
        if (!is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) continue;
        int nvidx = nz * (size_x * size_y) + ny * size_x + nx;
        if (vol_used[nvidx] + daughter_vol <= capacity) {
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

        // Atomically claim volume for daughter
        int tvidx = cand_z[i] * (size_x * size_y) + cand_y[i] * size_x + cand_x[i];
        if (!volume_try_claim(vol_used, tvidx, daughter_vol, capacity)) continue;

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
        FLAMEGPU->agent_out.setVariable<int>("tumble", 1);

        // Update parent
        FLAMEGPU->setVariable<int>("divide_limit", divide_limit - 1);
        FLAMEGPU->setVariable<int>("divide_cd", div_interval);

        // Track T cell proliferation event
        {
            auto* evts = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            const int cs_div = FLAMEGPU->getVariable<int>("cell_state");
            const int ps = (cs_div == T_CELL_EFF) ? EVT_PROLIF_CD8_EFF :
                           (cs_div == T_CELL_CYT)  ? EVT_PROLIF_CD8_CYT : EVT_PROLIF_CD8_SUP;
            atomicAdd(&evts[ps], 1u);
        }

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
        // EFF cells secrete IFN-γ only when in contact with cancer (matching HCC)
        // EFF cells do NOT secrete IL-2 — only CYT cells do, after activation
        if (cell_state == T_CELL_EFF && found_progenitor == 1){

        } else if (cell_state == T_CELL_CYT) {
            if (found_progenitor == 1){
                IFNg_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IFNG_RELEASE");
            }
            IL2_uptake_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_UPTAKE");
            if (FLAMEGPU->getVariable<float>("IL2_release_remain") > 0){
                IL2_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_RELEASE");
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
// T cell movement uses volume-based occupancy (volume_try_claim/volume_release).
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

    // ECM porosity: filter target voxels by T cell porosity threshold
    const int old_vidx = z * (grid_x * grid_y) + y * grid_x + x;
    const float* ecm_d = ECM_DENSITY_PTR(FLAMEGPU);
    const float* ecm_c = ECM_CROSSLINK_PTR(FLAMEGPU);
    const float density_cap = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DENSITY_CAP");
    const float min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_TCELL");

    const float move_dir_x = FLAMEGPU->getVariable<float>("move_direction_x");
    const float move_dir_y = FLAMEGPU->getVariable<float>("move_direction_y");
    const float move_dir_z = FLAMEGPU->getVariable<float>("move_direction_z");

    // Volume-based occupancy
    float* vol_used = VOL_PTR(FLAMEGPU);
    const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    float my_vol = (cell_state == T_CELL_EFF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_EFF") :
        (cell_state == T_CELL_CYT) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_CYT") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TCELL_SUP");

    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        // Use IFN-γ gradient for chemotaxis (T cell primary attractant) — read directly from PDE
        const float grad_x = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IFN_X))[old_vidx];
        const float grad_y = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IFN_Y))[old_vidx];
        const float grad_z = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IFN_Z))[old_vidx];

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

        const float lambda = 0.0000168;
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

        int target_vidx = tz * (grid_x * grid_y) + ty * grid_x + tx;
        if (ecm_porosity(ecm_d, ecm_c, target_vidx, density_cap) >= min_porosity &&
            volume_try_claim(vol_used, target_vidx, my_vol, capacity)) {
            volume_release(vol_used, old_vidx, my_vol);
            FLAMEGPU->setVariable<int>("x", tx);
            FLAMEGPU->setVariable<int>("y", ty);
            FLAMEGPU->setVariable<int>("z", tz);
        }
    }
    // === TUMBLE PHASE (tumble == 1) ===
    // Collect candidate neighbors, shuffle, then atomically claim the first available.
    else {
        int cand_x[26], cand_y[26], cand_z[26];
        int n_cands = 0;
        for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++) {
            if (di==0 && dj==0 && dk==0) continue;
            int nx = x+di, ny = y+dj, nz = z+dk;
            if (nx<0||nx>=grid_x||ny<0||ny>=grid_y||nz<0||nz>=grid_z) continue;
            int nvidx = nz * (grid_x * grid_y) + ny * grid_x + nx;
            if (vol_used[nvidx] + my_vol > capacity) continue;
            if (ecm_porosity(ecm_d, ecm_c, nvidx, density_cap) < min_porosity) continue;
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
            int tvidx = cand_z[i] * (grid_x * grid_y) + cand_y[i] * grid_x + cand_x[i];
            if (volume_try_claim(vol_used, tvidx, my_vol, capacity)) {
                volume_release(vol_used, old_vidx, my_vol);
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

} // namespace PDAC

#endif // PDAC_T_CELL_CUH