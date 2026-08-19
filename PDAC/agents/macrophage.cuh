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
    FLAMEGPU->message_out.setVariable<float>("kill_factor", 0.0f);  // N/A for macrophage
    FLAMEGPU->message_out.setVariable<int>("dead", 0);
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
// Macrophage: Write volume to occupancy grid
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    // Reset this step's move budget (batched-round movement; drained by mac_move).
    FLAMEGPU->setVariable<int>("moves_remaining",
        FLAMEGPU->environment.getProperty<int>("PARAM_MAC_MOVE_STEPS"));
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

    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    float my_vol = (cell_state == MAC_M1) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M1") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M2");
    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

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
    int neighbor_fib_count = 0;

    // Adhesion: count all type+state neighbors
    int adh_counts[ABM_STATE_COUNTER_SIZE] = {0};

    for (auto& msg : FLAMEGPU->message_in(world_x, world_y, world_z)) {
        int agent_type = msg.getVariable<int>("agent_type");
        int agent_state = msg.getVariable<int>("cell_state");
        float kill_factor = msg.getVariable<float>("kill_factor");
        int msg_x = msg.getVariable<int>("voxel_x");
        int msg_y = msg.getVariable<int>("voxel_y");
        int msg_z = msg.getVariable<int>("voxel_z");
        int dx = msg_x - x;
        int dy = msg_y - y;
        int dz = msg_z - z;

        // Only count if in Moore neighborhood (26 adjacent voxels)
        if (std::abs(dx) <= 1 && std::abs(dy) <= 1 && std::abs(dz) <= 1) {
            // Adhesion: accumulate into type+state count vector
            adh_counts[msg_to_sc_idx(agent_type, agent_state, kill_factor)]++;

            if (agent_type == CELL_TYPE_CANCER) {
                neighbor_cancer_count++;
            } else if (agent_type == CELL_TYPE_FIB) {
                neighbor_fib_count++;
            }
        }
    }

    // Compute adhesion p_move from matrix
    const int my_sc = self_sc_idx(CELL_TYPE_MAC, FLAMEGPU->getVariable<int>("cell_state"));
    const float adh_pmove = compute_adhesion_pmove(my_sc, adh_counts, ADH_MATRIX_PTR(FLAMEGPU));

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", neighbor_cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_fib_count", neighbor_fib_count);
    FLAMEGPU->setVariable<float>("adh_p_move", adh_pmove);
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
            // M1 macrophages also secrete IL-1 (drives iCAF activation)
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

    // IL-1 + MMP secretion (M1 only)
    if (dead == 0 && cell_state == MAC_M1) {
        float IL1_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M1_IL1_RELEASE");
        float MMP_release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M1_MMP_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_IL1, voxel, IL1_release_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_MMP, voxel, MMP_release_rate / voxel_volume);
    }

    return flamegpu::ALIVE;
}


// ============================================================================
// Macrophage: Single-phase movement using occupancy grid (exclusive per voxel)
// Uses run-tumble chemotaxis toward CCL2 gradient
// ============================================================================
// Macrophage movement via unified movement framework.
// CCL2 chemotaxis with state-dependent persistence and bias.
// Macrophage movement — Step-5 deterministic WAVE propose/commit (snapshot per wave). Each wave
// the cell walks up to per_round steps READ-ONLY (run_move_batch_pick, no claim) from its current
// voxel, reserves the wave's landing voxel by (source-voxel, det_tag); the commit moves the unique
// winner. A loser forfeits only this wave's per_round steps and retries next wave. Mirrors
// mdsc_move_propose/commit. (state-dependent my_vol: MAC_M1 vs MAC_M2 — computed identically in both.)
FLAMEGPU_AGENT_FUNCTION(mac_move_propose, flamegpu::MessageNone, flamegpu::MessageNone) {
    FLAMEGPU->setVariable<int>("move_target_vidx", -1);  // reset proposal each wave
    int moves_remaining = FLAMEGPU->getVariable<int>("moves_remaining");
    if (moves_remaining <= 0) return flamegpu::ALIVE;
    const int move_K = FLAMEGPU->environment.getProperty<int>("MOVE_K");
    const int per_round = (FLAMEGPU->environment.getProperty<int>("PARAM_MAC_MOVE_STEPS") + move_K - 1) / move_K;
    const int n_moves = min(per_round, moves_remaining);

    const int sx = FLAMEGPU->getVariable<int>("x");
    const int sy = FLAMEGPU->getVariable<int>("y");
    const int sz = FLAMEGPU->getVariable<int>("z");
    int x = sx, y = sy, z = sz;
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    float my_vol = (cell_state == MAC_M1) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M1") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M2");

    float p_persist = (cell_state == MAC_M1) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_MAC_M1") :
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_MAC_M2");

    float bias = (cell_state == MAC_M1) ?
        ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_MAC_M1")) :
        ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_MAC_M2"));

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");

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
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_MAC");
    // Adhesion: pre-computed from matrix in scan_neighbors
    mp.p_move = FLAMEGPU->getVariable<float>("adh_p_move");
    mp.p_persist = p_persist;
    mp.bias_strength = bias;
    mp.orient_x = ECM_ORIENT_X_PTR(FLAMEGPU);
    mp.orient_y = ECM_ORIENT_Y_PTR(FLAMEGPU);
    mp.orient_z = ECM_ORIENT_Z_PTR(FLAMEGPU);
    mp.barrier_strength = FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_MAC");
    const float w_cg = FLAMEGPU->environment.getProperty<float>("PARAM_CONTACT_GUIDANCE_MAC");

    // M1: IL-12 chemotaxis (toward DC/immune niche). M2: CCL2 (toward cancer).
    const float* gax = (cell_state == MAC_M1)
        ? reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IL12_X))
        : reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_X));
    const float* gay = (cell_state == MAC_M1)
        ? reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IL12_Y))
        : reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_Y));
    const float* gaz = (cell_state == MAC_M1)
        ? reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_IL12_Z))
        : reinterpret_cast<const float*>(FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_CCL2_Z));

    int pdx = FLAMEGPU->getVariable<int>("persist_dir_x");
    int pdy = FLAMEGPU->getVariable<int>("persist_dir_y");
    int pdz = FLAMEGPU->getVariable<int>("persist_dir_z");

    // Read-only snapshot walk (no claim) → wave landing voxel.
    run_move_batch_pick(FLAMEGPU, mp, n_moves, gax, gay, gaz, bias, w_cg, x, y, z, pdx, pdy, pdz);
    FLAMEGPU->setVariable<int>("moves_remaining", moves_remaining - n_moves);

    if (x != sx || y != sy || z != sz) {
        const int tvidx = z * (grid_x * grid_y) + y * grid_x + x;
        const int my_vidx = sz * (grid_x * grid_y) + sy * grid_x + sx;
        unsigned long long* owner = reinterpret_cast<unsigned long long*>(
            FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
        reserve_voxel(owner, tvidx, make_claim_priority(my_vidx, FLAMEGPU->getVariable<unsigned int>("det_tag")));
        FLAMEGPU->setVariable<int>("move_target_vidx", tvidx);
    }
    return flamegpu::ALIVE;
}

// Macrophage movement — COMMIT pass. Unique winner of each reserved landing voxel moves (claim
// final, release source); losers stay (retry next wave). my_vol matches the propose expression.
FLAMEGPU_AGENT_FUNCTION(mac_move_commit, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int tvidx = FLAMEGPU->getVariable<int>("move_target_vidx");
    if (tvidx < 0) return flamegpu::ALIVE;

    const int sx = FLAMEGPU->getVariable<int>("x");
    const int sy = FLAMEGPU->getVariable<int>("y");
    const int sz = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int my_vidx = sz * (grid_x * grid_y) + sy * grid_x + sx;
    const unsigned int det_tag = FLAMEGPU->getVariable<unsigned int>("det_tag");
    unsigned long long* owner = reinterpret_cast<unsigned long long*>(
        FLAMEGPU->environment.getProperty<unsigned long long>("voxel_owner_ptr"));
    if (!won_voxel(owner, tvidx, make_claim_priority(my_vidx, det_tag))) return flamegpu::ALIVE;

    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    const float my_vol = (cell_state == MAC_M1) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M1") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M2");
    const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    float* vol_used = VOL_PTR(FLAMEGPU);
    // Unconditional claim: snapshot capacity verified in propose + reserve gives a unique winner
    // per target → integer atomicAdd sums are order-independent, no check to flip → deterministic.
    (void)capacity;
    atomicAdd(&vol_used[tvidx], my_vol);
    volume_release(vol_used, my_vidx, my_vol);

    const int tx = tvidx % grid_x;
    const int ty = (tvidx / grid_x) % grid_y;
    const int tz = tvidx / (grid_x * grid_y);
    FLAMEGPU->setVariable<int>("x", tx);
    FLAMEGPU->setVariable<int>("y", ty);
    FLAMEGPU->setVariable<int>("z", tz);
    // Persistence = net wave displacement, clamped to a unit direction per axis.
    FLAMEGPU->setVariable<int>("persist_dir_x", (tx > sx) - (tx < sx));
    FLAMEGPU->setVariable<int>("persist_dir_y", (ty > sy) - (ty < sy));
    FLAMEGPU->setVariable<int>("persist_dir_z", (tz > sz) - (tz < sz));
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

    // Deterministic RNG for the polarization draws below (FLAMEGPU->random is nondeterministic
    // for recruited/born macs — slot-seeded in thread-schedule order). det_tag-seeded streams
    // (distinct salt per decision) are stable. A mac is in exactly one state at step start, so
    // only one branch draws — but distinct salts keep them independent regardless.
    const unsigned int sim_seed = FLAMEGPU->environment.getProperty<unsigned int>("sim_seed");
    const unsigned int det_tag  = FLAMEGPU->getVariable<unsigned int>("det_tag");

    // M1/M2 polarization dynamics — read chemicals directly from PDE
    const int nx_ss = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny_ss = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel_ss = z * ny_ss*nx_ss + y * nx_ss + x;

    // Hypoxic M2 bias: low O2 shifts polarization toward M2
    // bias_factor < 1 in hypoxia → harder to become/stay M1
    float local_O2 = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel_ss);
    const float mac_hyp_th = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_HYPOXIA_TH");
    const float m2_bias_str = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M2_BIAS_STRENGTH");
    float bias_factor = 1.0f;
    if (local_O2 < mac_hyp_th && mac_hyp_th > 0.0f) {
        bias_factor = 1.0f - m2_bias_str * (1.0f - local_O2 / mac_hyp_th);
        bias_factor = fmaxf(bias_factor, 0.01f);  // floor to avoid division by zero
    }

    if (cell_state == MAC_M1) {
        // QSP RF287: k_M2_pol * M1 * (H_TGFb + H_IL10 + H_IL6) / 3
        float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel_ss);
        float IL10 = PDE_READ(FLAMEGPU, PDE_CONC_IL10, voxel_ss);
        float IL6  = PDE_READ(FLAMEGPU, PDE_CONC_IL6, voxel_ss);

        float H_TGFb = TGFB / (TGFB + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_EC50") + 1e-30f);
        float H_IL10 = IL10 / (IL10 + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_10_EC50") + 1e-30f);
        float H_IL6  = IL6  / (IL6  + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL6_M2_EC50") + 1e-30f);

        double alpha = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M2_POL") *
                            (H_TGFb + H_IL10 + H_IL6) / 3.0;

        alpha /= static_cast<double>(bias_factor);

        double p_M2_polar = 1 - std::exp(-alpha);

        unsigned int rs = det_rng_seed(static_cast<int>(det_tag), FLAMEGPU->getStepCounter(), 0xAC1u, sim_seed);
        if (det_rng_uniform(rs) < p_M2_polar) {
            FLAMEGPU->setVariable<int>("cell_state", MAC_M2);
        }
    }

    if (cell_state == MAC_M2){
        // QSP RF288: k_M1_pol * M2 * (1 - (1-H_IL12) * max(0, 1-k_IFNg*C_IFNg))
        float IL12 = PDE_READ(FLAMEGPU, PDE_CONC_IL12, voxel_ss);
        float IFNg = PDE_READ(FLAMEGPU, PDE_CONC_IFN, voxel_ss);

        float H_IL12 = IL12 / (IL12 + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_12_EC50") + 1e-30f);
        float k_ifng = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IFNG_RESPONSE");
        float ifng_suppression = fmaxf(0.0f, 1.0f - k_ifng * IFNg);

        double alpha = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M1_POL") *
                            (1.0 - (1.0 - static_cast<double>(H_IL12)) * static_cast<double>(ifng_suppression));

        alpha *= static_cast<double>(bias_factor);

        double p_M1_polar = 1 - std::exp(-alpha);

        unsigned int rs = det_rng_seed(static_cast<int>(det_tag), FLAMEGPU->getStepCounter(), 0xAC2u, sim_seed);
        if (det_rng_uniform(rs) < p_M1_polar) {
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
         FLAMEGPU->getVariable<float>("PDL1_syn"),
         FLAMEGPU->environment.getProperty<float>("PARAM_PDL1_INTERNALIZATION_RATE"),
         FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE"));

    FLAMEGPU->setVariable<float>("PDL1_syn", PDL1);

    return flamegpu::ALIVE;
}


// ── phi: Mac_M1 is the KDE source for phi_2 (M1×C1), Mac_M2 for phi_4 (M2×C1). ──
FLAMEGPU_AGENT_FUNCTION(mac_phi_deposit, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    if (is_qsp_Mac_M1(cs))
        phi_deposit(FLAMEGPU, PHI_KDE_MAC_M1);
    if (is_qsp_Mac_M2(cs))
        phi_deposit(FLAMEGPU, PHI_KDE_MAC_M2);
    return flamegpu::ALIVE;
}

}  // namespace PDAC

#endif
