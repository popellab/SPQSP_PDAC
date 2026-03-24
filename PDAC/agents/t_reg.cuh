#ifndef FLAMEGPU_TNBC_T_REG_CUH
#define FLAMEGPU_TNBC_T_REG_CUH

#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"

namespace PDAC {

// TReg agent function: Broadcast location
FLAMEGPU_AGENT_FUNCTION(treg_broadcast_location, flamegpu::MessageNone, flamegpu::MessageSpatial3D) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    FLAMEGPU->message_out.setVariable<int>("agent_type", CELL_TYPE_TREG);
    FLAMEGPU->message_out.setVariable<int>("agent_id", FLAMEGPU->getID());
    const int treg_cs = FLAMEGPU->getVariable<int>("cell_state");
    FLAMEGPU->message_out.setVariable<int>("cell_state", treg_cs);
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f); // TCD4 cells don't express PDL1
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);

    FLAMEGPU->message_out.setLocation(
        (x + 0.5f) * voxel_size,
        (y + 0.5f) * voxel_size,
        (z + 0.5f) * voxel_size
    );

    // Count this agent into per-state population snapshot
    auto* sc_treg = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    atomicAdd(&sc_treg[treg_cs == TCD4_TH ? SC_TH : SC_TREG], 1u);

    return flamegpu::ALIVE;
}

// TReg agent function: Scan neighbors and cache available voxels
// Counts T cells, TRegs, cancer cells and builds bitmask of available neighbor voxels
FLAMEGPU_AGENT_FUNCTION(treg_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
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

    int tcell_count = 0;
    int treg_count = 0;
    int cancer_count = 0;
    int all_count = 0;
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
                if (agent_type == CELL_TYPE_T) {
                    tcell_count++;
                    neighbor_counts[dir_idx][1]++;
                } else if (agent_type == CELL_TYPE_TREG) {
                    treg_count++;
                    neighbor_counts[dir_idx][2]++;
                } else if (agent_type == CELL_TYPE_CANCER) {
                    cancer_count++;
                    neighbor_counts[dir_idx][0]++;
                    if (agent_state == CANCER_PROGENITOR) {
                        found_progenitor = 1;
                    }
                }
            }
        }
    }

    // Build available_neighbors mask (voxels with room for T cells/TRegs)
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

    FLAMEGPU->setVariable<int>("neighbor_Tcell_count", tcell_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_all_count", all_count);
    // FLAMEGPU->setVariable<unsigned int>("available_neighbors", available_neighbors);
    FLAMEGPU->setVariable<int>("found_progenitor", found_progenitor);

    return flamegpu::ALIVE;
}

__device__ __noinline__ float get_CTLA4_ipi(float ipi, float k_on, float k_off,
                                                float gamma_T_ipi,float chi_CTLA4, float a_Tcell,
                                                 float n_CTLA4_TCD4, float CTLA4_50, float K_ADCC) {
    double a = k_on * chi_CTLA4 / a_Tcell / k_off;
    double b = k_on * ipi / k_off / gamma_T_ipi;
    double c = n_CTLA4_TCD4;
    double d = -1;
    int max_iter = 20;
    double tol_rel = 1E-5;
    double root = 0;
    double res, root_new, f, f1;
    int i = 0;
    while (i < max_iter) {
        f = 2 * a * b * std::pow(root, 2) + (b + 1) * root - c;
        f1 = 4 * a * b * root + (b + 1);

        root_new = root - f / f1;
        res = std::abs(root_new - root) / root_new;
        if (res > tol_rel) {
            i++;
            root = root_new;
        }
        else {
            break;
        }
    }
    double free_CTLA4 = root;
    double CTLA4_Ipi = b * free_CTLA4;
    double CTLA4_Ipi_CTLA4 = a * b * free_CTLA4 * free_CTLA4;
    double H_TCD4 = std::pow(((CTLA4_Ipi + 2 * CTLA4_Ipi_CTLA4) / CTLA4_50), 2) / 
                        (std::pow(((CTLA4_Ipi + 2 * CTLA4_Ipi_CTLA4) / CTLA4_50), 2) + 1);
    double rate = H_TCD4 * K_ADCC;
    return 1 - exp(-rate);
}

// TReg agent function: State transitions
// Handles life countdown and division triggering
FLAMEGPU_AGENT_FUNCTION(treg_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) {
        return flamegpu::DEAD;
    }

    // Life countdown
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    if (life <= 0) {
        FLAMEGPU->setVariable<int>("dead", 1);
        auto* evts_tr = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        const int cs_tr = FLAMEGPU->getVariable<int>("cell_state");
        atomicAdd(&evts_tr[cs_tr == TCD4_TH ? EVT_DEATH_TH : EVT_DEATH_TREG], 1u);
        return flamegpu::DEAD;
    }
    FLAMEGPU->setVariable<int>("life", life);

    // Division cooldown
    int divide_cd = FLAMEGPU->getVariable<int>("divide_cd");
    if (divide_cd > 0) {
        divide_cd--;
        FLAMEGPU->setVariable<int>("divide_cd", divide_cd);
    }

    const int found_progenitor = FLAMEGPU->getVariable<int>("found_progenitor");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    const int nx_ts = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_ts = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz_ts = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int ax_ts = FLAMEGPU->getVariable<int>("x");
    const int ay_ts = FLAMEGPU->getVariable<int>("y");
    const int az_ts = FLAMEGPU->getVariable<int>("z");

    // Bounds check before PDE access
    if (ax_ts < 0 || ax_ts >= nx_ts || ay_ts < 0 || ay_ts >= ny_ts || az_ts < 0 || az_ts >= nz_ts) {
        printf("[TREG BOUNDS] id=%u bad coords (%d,%d,%d) grid=(%d,%d,%d) state=%d\n",
               FLAMEGPU->getID(), ax_ts, ay_ts, az_ts, nx_ts, ny_ts, nz_ts, cell_state);
        return flamegpu::ALIVE;
    }

    const int voxel_ts = az_ts * ny_ts*nx_ts + ay_ts * nx_ts + ax_ts;
    float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel_ts);
    float K_TH_TREG = FLAMEGPU->environment.getProperty<float>("PARAM_K_TH_TREG");
    float MAC_TGFB_EC50 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_EC50");
    int TCD4_DIV_INTERNAL = FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_INTERNAL");
    float CTLA4 = FLAMEGPU->environment.getProperty<float>("PARAM_CTLA4_TREG");

    float TGFB_release_remain = FLAMEGPU->getVariable<float>("TGFB_release_remain");
    float sec_per_slice = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    int divide_flag = 0;  // initialize to prevent UB for TH cells that don't convert
    if (cell_state == TCD4_TH) {
        float denominator = TGFB + MAC_TGFB_EC50;
        float alpha = (denominator > 1e-12f) ? K_TH_TREG * (1 + TGFB / denominator) : K_TH_TREG;  // Prevent division by zero
        float p_th_treg = 1 - std::exp(-alpha);
        //convert TH to TREG
        if (FLAMEGPU->random.uniform<float>() < p_th_treg) {
            FLAMEGPU->setVariable<int>("cell_state", TCD4_TREG);  // was T_CELL_CYT (wrong enum value)
            FLAMEGPU->setVariable<int>("divide_cd", TCD4_DIV_INTERNAL);  // was setVariable<float> (type mismatch)
            FLAMEGPU->setVariable<float>("CTLA4", CTLA4);
            return flamegpu::ALIVE;
        }
        // TH cells can divide when cooldown is ready
        int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
        if (divide_limit > 0 && divide_cd <= 0) {
            divide_flag = 1;
        } else {
            divide_flag = 0;
        }

    } else if (cell_state == TCD4_TREG) {
        if (found_progenitor == 1 && TGFB_release_remain >= 0){
            FLAMEGPU->setVariable<float>("TGFB_release_remain", TGFB_release_remain - sec_per_slice);
        }
        // NOTE: CTLA4-ipi ADCC death disabled: tumor_ipi=0 always → p_ADCC_death=0.
        // Disabled to reduce register pressure / stack usage (caused cudaErrorIllegalAddress). * Claude found this but I think its just an environment access error
        // TODO: re-enable when IPI drug coupling from QSP is implemented.
        // float k_on = FLAMEGPU->environment.getProperty<float>("PARAM_KON_CTLA4_IPI");
        // float k_off = FLAMEGPU->environment.getProperty<float>("PARAM_KOFF_CTLA4_IPI");
        // float gamma_T_ipi_on = FLAMEGPU->environment.getProperty<float>("PARAM_GAMMA_T_IPI");
        // float chi_CTLA4 = FLAMEGPU->environment.getProperty<float>("PARAM_CHI_CTLA4_IPI");
        // float a_Tcell = FLAMEGPU->environment.getProperty<float>("PARAM_A_TCELL");
        // float n_CTLA4_TCD4 = FLAMEGPU->environment.getProperty<float>("PARAM_CTLA4_TREG");
        // float TREG_CTLA4_50 = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_CTLA4_50");
        // float K_ADCC = FLAMEGPU->environment.getProperty<float>("PARAM_K_ADCC");
        // float tumor_ipi = 0.0f; // TODO: replace with IPI from QSP model
        // float p_ADCC_death = get_CTLA4_ipi(tumor_ipi, k_on, k_off, gamma_T_ipi_on,
        //                                     chi_CTLA4, a_Tcell, n_CTLA4_TCD4, TREG_CTLA4_50, K_ADCC);
        // if (FLAMEGPU->random.uniform<float>() < p_ADCC_death) { return flamegpu::DEAD; }

        float MDSC_EC50_ArgI_Treg = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_EC50_ArgI_Treg");
        float ArgI = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_CONC_ARGI))[voxel_ts];
        float H_ArgI = ArgI / (ArgI + MDSC_EC50_ArgI_Treg);
        int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
        if (FLAMEGPU->random.uniform<float>() < H_ArgI && divide_limit > 0){
            divide_cd = 0;
        }
        if (divide_limit > 0 && divide_cd <= 0) {
            divide_flag = 1;
        } else {
            divide_flag = 0;
        }
    }

    FLAMEGPU->setVariable<int>("divide_cd", divide_cd);
    FLAMEGPU->setVariable<int>("divide_flag", divide_flag);

    // === WAVE ASSIGNMENT ===
    if (divide_flag == 1 && divide_cd <= 0) {
        const int w = static_cast<int>(FLAMEGPU->random.uniform<float>() * N_DIVIDE_WAVES);
        FLAMEGPU->setVariable<int>("divide_wave", w < N_DIVIDE_WAVES ? w : N_DIVIDE_WAVES - 1);
    }

    return flamegpu::ALIVE;
}

// ============================================================
// Occupancy Grid Functions
// ============================================================

FLAMEGPU_AGENT_FUNCTION(treg_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ[x][y][z][CELL_TYPE_TREG] += 1u;

    return flamegpu::ALIVE;
}

// Single-phase TReg division using occupancy grid.
FLAMEGPU_AGENT_FUNCTION(treg_divide, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int divide_flag  = FLAMEGPU->getVariable<int>("divide_flag");
    const int divide_cd    = FLAMEGPU->getVariable<int>("divide_cd");
    const int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");

    if (FLAMEGPU->getVariable<int>("dead") == 1 ||
        divide_flag == 0 || divide_cd > 0 || divide_limit <= 0) {
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

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const uint8_t* face_flags = reinterpret_cast<const uint8_t*>(
        FLAMEGPU->environment.getProperty<uint64_t>("face_flags_ptr"));

    int cand_x[26], cand_y[26], cand_z[26];
    int n_cands = 0;
    unsigned int max_cap[26];
    for (int i = 0; i < 26; i++) {
        int dx, dy, dz;
        get_moore_direction(i, dx, dy, dz);
        const int nx = my_x + dx, ny = my_y + dy, nz = my_z + dz;
        if (!is_in_bounds(nx, ny, nz, size_x, size_y, size_z)) continue;
        if (is_ductal_wall_blocked(face_flags, my_x, my_y, my_z, dx, dy, dz, size_x, size_y)) continue;

        bool has_cancer = (occ[nx][ny][nz][CELL_TYPE_CANCER] > 0u);
        max_cap[n_cands] = static_cast<unsigned int>(has_cancer ? MAX_T_PER_VOXEL_WITH_CANCER : MAX_T_PER_VOXEL);

        // TRegs check T cell count for capacity (HCC: TRegs invisible to T cap)
        if (occ[nx][ny][nz][CELL_TYPE_T] < max_cap[n_cands]) {
            cand_x[n_cands] = nx;
            cand_y[n_cands] = ny;
            cand_z[n_cands] = nz;
            n_cands++;
        }
    }

    if (n_cands == 0) return flamegpu::ALIVE;

    const int cell_state     = FLAMEGPU->getVariable<int>("cell_state");
    const int div_interval   = FLAMEGPU->environment.getProperty<int>("PARAM_TCD4_DIV_INTERNAL");
    const float treg_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");

    for (int i = 0; i < n_cands; i++) {
        const int j = i + static_cast<int>(FLAMEGPU->random.uniform<float>() * (n_cands - i));
        int tx = cand_x[i]; cand_x[i] = cand_x[j]; cand_x[j] = tx;
        int ty = cand_y[i]; cand_y[i] = cand_y[j]; cand_y[j] = ty;
        int tz = cand_z[i]; cand_z[i] = cand_z[j]; cand_z[j] = tz;
        unsigned int max_curr = max_cap[i]; max_cap[i] = max_cap[j]; max_cap[j] = max_curr;

        // operator+ performs atomicAdd and returns the OLD value
        // Claim TREG slot (not T slot) — TRegs are invisible to T cell cap checks
        const unsigned int old_count = occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_TREG] + 1u;
        if (old_count >= max_curr) {
            occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_TREG] -= 1u;  // undo
            continue;
        }

        const float treg_life_sd = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");
        float tLifeD = treg_life_mean + FLAMEGPU->random.normal<float>() * treg_life_sd;
        int daughter_life = static_cast<int>(tLifeD + 0.5f);
        daughter_life = (daughter_life > 0) ? daughter_life : 1;

        FLAMEGPU->agent_out.setVariable<int>("x", cand_x[i]);
        FLAMEGPU->agent_out.setVariable<int>("y", cand_y[i]);
        FLAMEGPU->agent_out.setVariable<int>("z", cand_z[i]);
        FLAMEGPU->agent_out.setVariable<int>("cell_state", cell_state);
        FLAMEGPU->agent_out.setVariable<int>("divide_flag", 1);
        FLAMEGPU->agent_out.setVariable<int>("divide_cd", div_interval);
        FLAMEGPU->agent_out.setVariable<int>("divide_limit", divide_limit - 1);
        FLAMEGPU->agent_out.setVariable<int>("life", daughter_life > 0 ? daughter_life : 1);
        FLAMEGPU->agent_out.setVariable<int>("tumble", 1);

        // Update parent: division state only (life continues counting down)
        FLAMEGPU->setVariable<int>("divide_flag", 1);
        FLAMEGPU->setVariable<int>("divide_limit", divide_limit - 1);
        FLAMEGPU->setVariable<int>("divide_cd", div_interval);

        // Track TH or TReg proliferation event
        {
            auto* evts_div = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            atomicAdd(&evts_div[cell_state == TCD4_TH ? EVT_PROLIF_TH : EVT_PROLIF_TREG], 1u);
        }

        break;
    }

    // Clear parent intent
    FLAMEGPU->setVariable<int>("intent_action", INTENT_NONE);
    FLAMEGPU->setVariable<int>("target_x", -1);
    FLAMEGPU->setVariable<int>("target_y", -1);
    FLAMEGPU->setVariable<int>("target_z", -1);

    return flamegpu::ALIVE;
}

// TReg agent function: Update chemicals from PDE
FLAMEGPU_AGENT_FUNCTION(treg_update_chemicals, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Chemical concentrations are now read directly from PDE device pointers where needed.
    return flamegpu::ALIVE;
}

// TReg agent function: Compute chemical sources
// atomicAdds directly to PDE source/uptake arrays
FLAMEGPU_AGENT_FUNCTION(treg_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int dead = FLAMEGPU->getVariable<int>("dead");

    float IL10_release = 0.0f;
    float TGFB_release = 0.0f;
    float IL2_release = 0.0f;

    int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    if (dead == 0){
        if (cell_state == TCD4_TH){
            if (FLAMEGPU->getVariable<int>("found_progenitor") == 1){
                IL2_release = FLAMEGPU->environment.getProperty<float>("PARAM_IL2_RELEASE");
            }
        } else if (cell_state == TCD4_TREG){
            IL10_release = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_IL10_RELEASE");
            if (FLAMEGPU->getVariable<int>("found_progenitor") == 1 &&
                    FLAMEGPU->getVariable<float>("TGFB_release_remain") > 0.0){
                TGFB_release = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_TGFB_RELEASE");
            }
        }
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

    // IL-10 secretion → src ptr 3 (IL10)
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IL10, voxel, IL10_release / voxel_volume);

    // TGF-β secretion → src ptr 4 (TGFB)
    PDE_SECRETE(FLAMEGPU, PDE_SRC_TGFB, voxel, TGFB_release / voxel_volume);

    // IL-2 secrete → src ptr 2 (IL2), positive [1/s], no volume scaling
    PDE_SECRETE(FLAMEGPU, PDE_SRC_IL2, voxel, IL2_release / voxel_volume);

    return flamegpu::ALIVE;
}

// Single-phase TReg movement using occupancy grid.
// Replaces two-phase select_move_target + execute_move.
// Same capacity rules as T cells (up to MAX_T_PER_VOXEL).
// TReg movement using run-tumble chemotaxis toward IFN-γ
FLAMEGPU_AGENT_FUNCTION(treg_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int tumble = FLAMEGPU->getVariable<int>("tumble");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

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

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    const uint8_t* face_flags = reinterpret_cast<const uint8_t*>(
        FLAMEGPU->environment.getProperty<uint64_t>("face_flags_ptr"));

    int target_x = x;
    int target_y = y;
    int target_z = z;

    const float dt = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");

    // === RUN PHASE (tumble == 0) ===
    if (tumble == 0) {
        // Use IFN-γ gradient for chemotaxis (TReg primary attractant) — read directly from PDE
        const int nx_mv = FLAMEGPU->environment.getProperty<int>("grid_size_x");
        const int ny_mv = FLAMEGPU->environment.getProperty<int>("grid_size_y");
        const int voxel_mv = z * ny_mv*nx_mv + y * nx_mv + x;
        const float grad_x = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IFN_X))[voxel_mv];
        const float grad_y = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IFN_Y))[voxel_mv];
        const float grad_z = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IFN_Z))[voxel_mv];

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

        // Ductal wall check
        if (is_ductal_wall_blocked(face_flags, x, y, z, tx-x, ty-y, tz-z, grid_x, grid_y)) {
            return flamegpu::ALIVE;
        }

        // TRegs check T cell count for capacity but claim TREG slot
        unsigned int max_t = (occ[tx][ty][tz][CELL_TYPE_CANCER] > 0u)
            ? static_cast<unsigned int>(MAX_T_PER_VOXEL_WITH_CANCER)
            : static_cast<unsigned int>(MAX_T_PER_VOXEL);

        // Check T cell count against cap (TReg uses T cell conditions)
        if (occ[tx][ty][tz][CELL_TYPE_T] >= max_t) {
            return flamegpu::ALIVE;
        }

        // Claim TREG slot (not T slot) — just for tracking, no cap on TREGs
        occ[tx][ty][tz][CELL_TYPE_TREG] += 1u;
        occ[x][y][z][CELL_TYPE_TREG] -= 1u;
        FLAMEGPU->setVariable<int>("x", tx);
        FLAMEGPU->setVariable<int>("y", ty);
        FLAMEGPU->setVariable<int>("z", tz);
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
            if (is_ductal_wall_blocked(face_flags, x, y, z, di, dj, dk, grid_x, grid_y)) continue;
            unsigned int max_t = (occ[nx][ny][nz][CELL_TYPE_CANCER] > 0u)
                ? static_cast<unsigned int>(MAX_T_PER_VOXEL_WITH_CANCER)
                : static_cast<unsigned int>(MAX_T_PER_VOXEL);
            if (occ[nx][ny][nz][CELL_TYPE_T] >= max_t) continue;
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
            // Re-check T cell cap (may have changed since pre-filter)
            unsigned int max_t = (occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_CANCER] > 0u)
                ? static_cast<unsigned int>(MAX_T_PER_VOXEL_WITH_CANCER)
                : static_cast<unsigned int>(MAX_T_PER_VOXEL);
            if (occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_T] >= max_t) continue;

            // Claim TREG slot (not T slot)
            occ[cand_x[i]][cand_y[i]][cand_z[i]][CELL_TYPE_TREG] += 1u;
            occ[x][y][z][CELL_TYPE_TREG] -= 1u;
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

    return flamegpu::ALIVE;
}

} // namespace PDAC

#endif // PDAC_T_REG_CUH