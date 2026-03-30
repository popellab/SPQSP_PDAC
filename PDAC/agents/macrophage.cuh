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
// Macrophage: Write volume to occupancy grid
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(mac_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
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

    for (auto& msg : FLAMEGPU->message_in(world_x, world_y, world_z)) {
        int agent_type = msg.getVariable<int>("agent_type");
        int msg_x = msg.getVariable<int>("voxel_x");
        int msg_y = msg.getVariable<int>("voxel_y");
        int msg_z = msg.getVariable<int>("voxel_z");
        int dx = msg_x - x;
        int dy = msg_y - y;
        int dz = msg_z - z;

        // Only count if in Moore neighborhood (26 adjacent voxels)
        if (std::abs(dx) <= 1 && std::abs(dy) <= 1 && std::abs(dz) <= 1) {
            if (agent_type == CELL_TYPE_CANCER) {
                neighbor_cancer_count++;
            } else if (agent_type == CELL_TYPE_FIB) {
                neighbor_fib_count++;
            }
        }
    }

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", neighbor_cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_fib_count", neighbor_fib_count);
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
FLAMEGPU_AGENT_FUNCTION(mac_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");

    float my_vol = (cell_state == MAC_M1) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M1") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_MAC_M2");

    float p_persist = (cell_state == MAC_M1) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_MAC_M1") :
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_MAC_M2");

    float bias = (cell_state == MAC_M1) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_BIAS_MAC_M1") :
        FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_BIAS_MAC_M2");

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
    mp.my_vol = my_vol;
    mp.capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    mp.ecm_density = ECM_DENSITY_PTR(FLAMEGPU);
    mp.ecm_crosslink = ECM_CROSSLINK_PTR(FLAMEGPU);
    mp.density_cap = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_DENSITY_CAP");
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_MAC");
    // Adhesion: M2 semi-sessile niche retention
    if (cell_state == MAC_M2) {
        const int n_cancer = FLAMEGPU->getVariable<int>("neighbor_cancer_count");
        const int n_fib = FLAMEGPU->getVariable<int>("neighbor_fib_count");
        float local_ecm = mp.ecm_density[vidx];
        float ecm_th = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_ECM_DENSITY_TH");
        mp.p_move = compute_adhesion_pmove(
            FLAMEGPU->environment.getProperty<float>("PARAM_ADH_MAC_M2_CANCER"), n_cancer,
            FLAMEGPU->environment.getProperty<float>("PARAM_ADH_MAC_M2_FIB"), n_fib,
            FLAMEGPU->environment.getProperty<float>("PARAM_ADH_MAC_M2_ECM"), local_ecm, ecm_th);
    } else {
        mp.p_move = 1.0f;
    }
    mp.p_persist = p_persist;
    mp.bias_strength = bias;
    mp.grad_x = gx; mp.grad_y = gy; mp.grad_z = gz;

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
        float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel_ss);
        float IL10 = PDE_READ(FLAMEGPU, PDE_CONC_IL10, voxel_ss);

        double alpha = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_M2_POL") *
                            ((TGFB / (TGFB + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_TGFB_EC50"))) +
                            (IL10 / (IL10 + FLAMEGPU->environment.getProperty<float>("PARAM_MAC_IL_10_EC50"))));

        // Hypoxia increases M1→M2 transition probability
        alpha /= static_cast<double>(bias_factor);

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

        // Hypoxia decreases M2→M1 reversion probability
        alpha *= static_cast<double>(bias_factor);

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
