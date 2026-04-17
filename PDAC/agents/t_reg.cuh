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
    FLAMEGPU->message_out.setVariable<float>("PDL1", 0.0f);       // TCD4 cells don't express PDL1
    FLAMEGPU->message_out.setVariable<float>("kill_factor", 0.0f); // N/A for TReg
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
    auto* sc_treg = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    const int sc_slot = (treg_cs == TCD4_TH) ? SC_TH :
                        (treg_cs == TCD4_TREG)  ? SC_TREG :
                        (treg_cs == TCD4_NAIVE) ? SC_TCD4_NAIVE : SC_TFH;
    atomicAdd(&sc_treg[sc_slot], 1u);

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
    int dc_mature_count = 0;
    int bcell_count = 0;

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
            all_count++;
            const int agent_type = msg.getVariable<int>("agent_type");
            const int agent_state = msg.getVariable<int>("cell_state");
            const float kill_factor = msg.getVariable<float>("kill_factor");

            // Adhesion: accumulate into type+state count vector
            adh_counts[msg_to_sc_idx(agent_type, agent_state, kill_factor)]++;

            // Agent-specific interaction counts
            if (agent_type == CELL_TYPE_T) {
                tcell_count++;
            } else if (agent_type == CELL_TYPE_TREG) {
                treg_count++;
            } else if (agent_type == CELL_TYPE_CANCER) {
                cancer_count++;
                if (agent_state == CANCER_PROGENITOR) {
                    found_progenitor = 1;
                }
            } else if (agent_type == CELL_TYPE_DC) {
                if (agent_state == DC_MATURE) {
                    if (kill_factor > 0.5f) dc_mature_count++;  // cDC2 only
                }
            } else if (agent_type == CELL_TYPE_BCELL) {
                bcell_count++;
            }
        }
    }

    // Compute adhesion p_move from matrix
    const int my_sc = self_sc_idx(CELL_TYPE_TREG, FLAMEGPU->getVariable<int>("cell_state"));
    const float adh_pmove = compute_adhesion_pmove(my_sc, adh_counts, ADH_MATRIX_PTR(FLAMEGPU));

    FLAMEGPU->setVariable<int>("neighbor_Tcell_count", tcell_count);
    FLAMEGPU->setVariable<int>("neighbor_Treg_count", treg_count);
    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_all_count", all_count);
    FLAMEGPU->setVariable<int>("found_progenitor", found_progenitor);
    FLAMEGPU->setVariable<int>("neighbor_dc_mature_count", dc_mature_count);
    FLAMEGPU->setVariable<int>("neighbor_bcell_count", bcell_count);
    FLAMEGPU->setVariable<float>("adh_p_move", adh_pmove);

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
        const int death_slot = (cs_tr == TCD4_TH)    ? EVT_DEATH_TH :
                               (cs_tr == TCD4_TREG)  ? EVT_DEATH_TREG :
                               (cs_tr == TCD4_NAIVE) ? EVT_DEATH_TCD4_NAIVE : EVT_DEATH_TFH;
        atomicAdd(&evts_tr[death_slot], 1u);
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

    // ── NAIVE CD4: DC priming only, skip all other logic ──
    if (cell_state == TCD4_NAIVE) {
        const int dc_cdc2 = FLAMEGPU->getVariable<int>("neighbor_dc_mature_count");
        if (dc_cdc2 > 0) {
            const float sec_per_slice_n = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
            const float n_sites = FLAMEGPU->environment.getProperty<float>("PARAM_DC_N_SITES");
            const int n_local = FLAMEGPU->getVariable<int>("neighbor_all_count");
            const float cell_p = FLAMEGPU->environment.getProperty<float>("PARAM_CELL");
            const float H = n_sites * dc_cdc2 / (n_sites * dc_cdc2 + n_local + cell_p);
            // Combined activation rate: k_Th + k_Treg (both consume naive CD4)
            const float k_th = FLAMEGPU->environment.getProperty<float>("PARAM_DC_PRIME_K_TH");
            const float k_treg = FLAMEGPU->environment.getProperty<float>("PARAM_DC_PRIME_K_TREG");
            const float p = 1.0f - expf(-(k_th + k_treg) * H * sec_per_slice_n);
            if (FLAMEGPU->random.uniform<float>() < p) {
                auto* evts_n = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
                // Fate decision: f_nTreg → Treg, else → TH
                const float f_treg = FLAMEGPU->environment.getProperty<float>("PARAM_F_NTREG");
                if (FLAMEGPU->random.uniform<float>() < f_treg) {
                    FLAMEGPU->setVariable<int>("cell_state", TCD4_TREG);
                    FLAMEGPU->setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_TREG_DIV_INTERVAL"));
                    FLAMEGPU->setVariable<float>("CTLA4", FLAMEGPU->environment.getProperty<float>("PARAM_CTLA4_TREG"));
                    atomicAdd(&evts_n[EVT_PRIME_TREG], 1u);
                } else {
                    FLAMEGPU->setVariable<int>("cell_state", TCD4_TH);
                    FLAMEGPU->setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_TH_DIV_INTERVAL"));
                    atomicAdd(&evts_n[EVT_PRIME_TH], 1u);
                }
                // Division burst from DC priming
                const int N_prime = FLAMEGPU->environment.getProperty<int>("PARAM_PRIME_DIV_BURST");
                FLAMEGPU->setVariable<int>("divide_limit",
                    FLAMEGPU->getVariable<int>("divide_limit") + N_prime);
                FLAMEGPU->setVariable<int>("divide_cd", 0);
                FLAMEGPU->setVariable<int>("divide_flag", 1);
                // Extend lifespan: naive→differentiated gets a fresh lifespan
                const float life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
                const float life_sd = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFESPAN_SD");
                int new_life = static_cast<int>(life_mean + FLAMEGPU->random.normal<float>() * life_sd + 0.5f);
                FLAMEGPU->setVariable<int>("life", new_life > 1 ? new_life : 1);
            }
        }
        // NAIVE cells do nothing else — no TGF-β, no IL-10, no division, no conversion
        return flamegpu::ALIVE;
    }

    const int nx_ts = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_ts = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz_ts = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int ax_ts = FLAMEGPU->getVariable<int>("x");
    const int ay_ts = FLAMEGPU->getVariable<int>("y");
    const int az_ts = FLAMEGPU->getVariable<int>("z");

    // Bounds check before PDE access
    if (ax_ts < 0 || ax_ts >= nx_ts || ay_ts < 0 || ay_ts >= ny_ts || az_ts < 0 || az_ts >= nz_ts) {
        return flamegpu::ALIVE;
    }

    const int voxel_ts = az_ts * ny_ts*nx_ts + ay_ts * nx_ts + ax_ts;
    float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel_ts);
    float K_TH_TREG = FLAMEGPU->environment.getProperty<float>("PARAM_K_TH_TREG");
    float MAC_TGFB_EC50 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_EC50");
    float CTLA4 = FLAMEGPU->environment.getProperty<float>("PARAM_CTLA4_TREG");

    float TGFB_release_remain = FLAMEGPU->getVariable<float>("TGFB_release_remain");
    float sec_per_slice = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    int divide_flag = 0;  // initialize to prevent UB for TH cells that don't convert
    if (cell_state == TCD4_TH) {
        // ODE RF182: k_Th_Treg * Th * H_TGFb * H_ArgI_Treg
        float H_TGFb = TGFB / (TGFB + MAC_TGFB_EC50);
        float ArgI = PDE_READ(FLAMEGPU, PDE_CONC_ARGI, voxel_ts);
        float H_ArgI_Treg = ArgI / (ArgI + FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_EC50_ArgI_Treg"));
        float alpha = K_TH_TREG * H_TGFb * H_ArgI_Treg;
        float p_th_treg = 1.0f - std::exp(-alpha);
        //convert TH to TREG
        if (FLAMEGPU->random.uniform<float>() < p_th_treg) {
            FLAMEGPU->setVariable<int>("cell_state", TCD4_TREG);
            FLAMEGPU->setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<int>("PARAM_TREG_DIV_INTERVAL"));
            FLAMEGPU->setVariable<float>("CTLA4", CTLA4);
            return flamegpu::ALIVE;
        }

        // TH → Tfh: requires mature DC contact + IL-6 signal
        int dc_mature = FLAMEGPU->getVariable<int>("neighbor_dc_mature_count");
        if (dc_mature > 0) {
            float IL6 = PDE_READ(FLAMEGPU, PDE_CONC_IL6, voxel_ts);
            float ec50_il6 = FLAMEGPU->environment.getProperty<float>("PARAM_TFH_EC50_IL6");
            float k_tfh = FLAMEGPU->environment.getProperty<float>("PARAM_TFH_DIFF_K");
            float H_IL6 = IL6 / (IL6 + ec50_il6);
            float p_tfh = 1.0f - expf(-k_tfh * H_IL6);
            if (FLAMEGPU->random.uniform<float>() < p_tfh) {
                FLAMEGPU->setVariable<int>("cell_state", TCD4_TFH);
                return flamegpu::ALIVE;
            }
        }

        // TH cells can divide when cooldown is ready
        int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
        if (divide_limit > 0 && divide_cd <= 0) {
            divide_flag = 1;
        } else {
            divide_flag = 0;
        }

    } else if (cell_state == TCD4_TFH) {
        // Tfh division: same cooldown logic as TH
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

        // Treg division: cooldown-based, rate from QSP k_Treg_pro_tumor
        // ArgI gating removed — ArgI modulates Th→Treg conversion (RF182), not Treg proliferation
        int divide_limit = FLAMEGPU->getVariable<int>("divide_limit");
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

    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int gz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    if (x < 0 || x >= gx || y < 0 || y >= gy || z < 0 || z >= gz) {
        return flamegpu::ALIVE;
    }
    const int vidx = z * (gx * gy) + y * gx + x;

    // Volume-based occupancy — naive CD4 uses TH volume
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    float my_vol = (cell_state == TCD4_TREG) ? FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_REG") :
                   (cell_state == TCD4_TFH)  ? FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_TFH") :
                                                FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_TH");
    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

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

    // Volume-based occupancy
    float* vol_used = VOL_PTR(FLAMEGPU);
    const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    const int cell_state     = FLAMEGPU->getVariable<int>("cell_state");
    float daughter_vol = (cell_state == TCD4_TREG) ? FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_REG") :
                         (cell_state == TCD4_TFH)  ? FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_TFH") :
                                                      FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_TH");

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

    const int div_interval   = (cell_state == TCD4_TREG)
        ? FLAMEGPU->environment.getProperty<int>("PARAM_TREG_DIV_INTERVAL")
        : FLAMEGPU->environment.getProperty<int>("PARAM_TH_DIV_INTERVAL");
    const float treg_life_mean = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");

    for (int i = 0; i < n_cands; i++) {
        const int j = i + static_cast<int>(FLAMEGPU->random.uniform<float>() * (n_cands - i));
        int tx = cand_x[i]; cand_x[i] = cand_x[j]; cand_x[j] = tx;
        int ty = cand_y[i]; cand_y[i] = cand_y[j]; cand_y[j] = ty;
        int tz = cand_z[i]; cand_z[i] = cand_z[j]; cand_z[j] = tz;

        // Atomically claim volume for daughter
        int tvidx = cand_z[i] * (size_x * size_y) + cand_y[i] * size_x + cand_x[i];
        if (!volume_try_claim(vol_used, tvidx, daughter_vol, capacity)) continue;

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
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_x", 0);
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_y", 0);
        FLAMEGPU->agent_out.setVariable<int>("persist_dir_z", 0);

        // Update parent: division state only (life continues counting down)
        FLAMEGPU->setVariable<int>("divide_flag", 1);
        FLAMEGPU->setVariable<int>("divide_limit", divide_limit - 1);
        FLAMEGPU->setVariable<int>("divide_cd", div_interval);

        // Track TH/TReg/Tfh proliferation event
        {
            auto* evts_div = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
            const int prolif_slot = (cell_state == TCD4_TH) ? EVT_PROLIF_TH : (cell_state == TCD4_TREG) ? EVT_PROLIF_TREG : EVT_PROLIF_TFH;
            atomicAdd(&evts_div[prolif_slot], 1u);
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
    // Naive CD4 cells don't secrete any chemicals
    if (cell_state == TCD4_NAIVE) return flamegpu::ALIVE;
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

    // Tfh cells secrete CXCL13 (TLS co-organizing with B cells)
    if (cell_state == TCD4_TFH && dead == 0) {
        float cxcl13_rate = FLAMEGPU->environment.getProperty<float>("PARAM_TFH_CXCL13_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CXCL13, voxel, cxcl13_rate / voxel_volume);
    }

    return flamegpu::ALIVE;
}

// Single-phase TReg movement using occupancy grid.
// Replaces two-phase select_move_target + execute_move.
// TReg movement uses volume-based occupancy (volume_try_claim/volume_release).
// TReg movement using run-tumble chemotaxis toward IFN-γ
// TReg movement via unified movement framework.
// TReg: TGF-β chemotaxis + persistence. TH: persistent random walk (no chemotaxis).
FLAMEGPU_AGENT_FUNCTION(treg_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("dead") == 1) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    float my_vol, p_persist, bias;
    if (cell_state == TCD4_TREG) {
        my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_REG");
        p_persist = FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_TREG_REG");
        bias = ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_TREG_REG"));
    } else if (cell_state == TCD4_TFH) {
        my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_TFH");
        p_persist = FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_TREG_TFH");
        bias = ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_TREG_TFH"));
    } else {
        my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_TREG_TH");
        p_persist = FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_TREG_TH");
        bias = 0.0f;
    }

    // Read gradient: TReg→CXCL12 (CXCR4-mediated, Iellem 2001), Tfh→CXCL13, TH→none
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (grid_x * grid_y) + y * grid_x + x;
    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    if (cell_state == TCD4_TREG) {
        gx = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CXCL12_X))[vidx];
        gy = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CXCL12_Y))[vidx];
        gz = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CXCL12_Z))[vidx];
    } else if (cell_state == TCD4_TFH) {
        gx = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CXCL13_X))[vidx];
        gy = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CXCL13_Y))[vidx];
        gz = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CXCL13_Z))[vidx];
    }

    MoveParams mp;
    mp.grid_x = grid_x;
    mp.grid_y = grid_y;
    mp.grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    mp.vol_used = VOL_PTR(FLAMEGPU);
    mp.my_vol = my_vol;
    mp.capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    mp.ecm_density = ECM_DENSITY_PTR(FLAMEGPU);
    mp.ecm_crosslink = ECM_CROSSLINK_PTR(FLAMEGPU);
    mp.density_cap = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DENSITY_CAP");
    mp.min_porosity = (cell_state == TCD4_TFH)
        ? FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_TFH")
        : FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_TREG");
    mp.p_move = FLAMEGPU->getVariable<float>("adh_p_move");
    mp.p_persist = p_persist;
    mp.bias_strength = bias;
    mp.grad_x = gx; mp.grad_y = gy; mp.grad_z = gz;
    mp.orient_x = ECM_ORIENT_X_PTR(FLAMEGPU);
    mp.orient_y = ECM_ORIENT_Y_PTR(FLAMEGPU);
    mp.orient_z = ECM_ORIENT_Z_PTR(FLAMEGPU);
    mp.barrier_strength = FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_TREG");

    // Contact guidance
    {
        float w_cg = FLAMEGPU->environment.getProperty<float>("PARAM_CONTACT_GUIDANCE_TREG");
        float ox = ECM_ORIENT_X_PTR(FLAMEGPU)[vidx];
        float oy = ECM_ORIENT_Y_PTR(FLAMEGPU)[vidx];
        float oz = ECM_ORIENT_Z_PTR(FLAMEGPU)[vidx];
        if (bias > 0.0f) {
            apply_contact_guidance(mp.grad_x, mp.grad_y, mp.grad_z, ox, oy, oz, w_cg);
        } else {
            apply_contact_guidance_persist(mp.grad_x, mp.grad_y, mp.grad_z, mp.bias_strength,
                ox, oy, oz, w_cg,
                FLAMEGPU->getVariable<int>("persist_dir_x"),
                FLAMEGPU->getVariable<int>("persist_dir_y"),
                FLAMEGPU->getVariable<int>("persist_dir_z"));
        }
    }

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

#endif // PDAC_T_REG_CUH