// Fibroblast Agent Behavior Functions
// States: FIB_QUIESCENT, FIB_MYCAF (myofibroblastic CAF), FIB_ICAF (inflammatory CAF)
// Single-cell, single-voxel agents. No chain model.
//
// Activation: Quiescent → myCAF (TGF-β driven), Quiescent → iCAF (IL-1 driven, TGF-β suppressed)
// ECM deposition: myCAF only (Gaussian kernel, radius from XML)
// Chemical sources: myCAF → TGF-β + CCL2; iCAF → IL-6 + CXCL13 + CCL2
// Division: activated fibroblasts can divide locally (cancer-neighbor PDGF proxy)

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

    // Population count by state
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    auto* sc = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    int sc_idx;
    if (cs == FIB_QUIESCENT)  sc_idx = SC_FIB_QUIESCENT;
    else if (cs == FIB_MYCAF) sc_idx = SC_FIB_MYCAF;
    else                      sc_idx = SC_FIB_ICAF;
    atomicAdd(&sc[sc_idx], 1u);

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Write single voxel to occupancy grid
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
// myCAF: TGF-β + CCL2
// iCAF:  IL-6 + CXCL13 + CCL2
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_compute_chemical_sources, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    if (cs == FIB_QUIESCENT) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int voxel = z * (grid_x * grid_y) + y * grid_x + x;

    const float voxel_size_cm = FLAMEGPU->environment.getProperty<float>("voxel_size") * 1e-4f;
    const float voxel_volume = voxel_size_cm * voxel_size_cm * voxel_size_cm;

    if (cs == FIB_MYCAF) {
        const float tgfb_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_TGFB_RELEASE");
        const float ccl2_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_CCL2_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_TGFB, voxel, tgfb_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL2, voxel, ccl2_rate / voxel_volume);
    } else {  // FIB_ICAF
        const float il6_rate    = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL6_RELEASE");
        const float cxcl13_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_CXCL13_RELEASE");
        const float ccl2_rate   = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_CCL2_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_IL6,    voxel, il6_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CXCL13, voxel, cxcl13_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL2,   voxel, ccl2_rate / voxel_volume);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Movement — single-cell random walk, state-dependent probability
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const int cs = FLAMEGPU->getVariable<int>("cell_state");

    // State-dependent movement probability
    float move_prob;
    if (cs == FIB_QUIESCENT)  move_prob = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MOVE_PROB_QUIESCENT");
    else if (cs == FIB_MYCAF) move_prob = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MOVE_PROB_MYCAF");
    else                      move_prob = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MOVE_PROB_ICAF");

    if (FLAMEGPU->random.uniform<float>() >= move_prob) return flamegpu::ALIVE;

    // ECM-gated: higher ECM → less likely to move
    const float* ecm_ptr = reinterpret_cast<const float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("ecm_grid_ptr"));
    float ECM_density = ecm_ptr[z * (grid_x * grid_y) + y * grid_x + x];
    float ECM_50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_MOT_EC50");
    float ECM_sat = ECM_density / (ECM_density + ECM_50);
    if (FLAMEGPU->random.uniform<float>() < ECM_sat) return flamegpu::ALIVE;

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // Random walk: pick random Moore neighbor
    int cand_x[26], cand_y[26], cand_z[26];
    int n_cands = 0;
    for (int di = -1; di <= 1; di++) for (int dj = -1; dj <= 1; dj++) for (int dk = -1; dk <= 1; dk++) {
        if (di == 0 && dj == 0 && dk == 0) continue;
        int nx = x + di, ny = y + dj, nz = z + dk;
        if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
        if (occ[nx][ny][nz][CELL_TYPE_CANCER] != 0u) continue;
        if (occ[nx][ny][nz][CELL_TYPE_FIB] != 0u) continue;
        cand_x[n_cands] = nx; cand_y[n_cands] = ny; cand_z[n_cands] = nz;
        n_cands++;
    }
    if (n_cands == 0) return flamegpu::ALIVE;

    // Pick random candidate
    int pick = static_cast<int>(FLAMEGPU->random.uniform<float>() * n_cands);
    if (pick >= n_cands) pick = n_cands - 1;

    int nx = cand_x[pick], ny = cand_y[pick], nz = cand_z[pick];

    // CAS claim
    if (occ[nx][ny][nz][CELL_TYPE_FIB].CAS(0u, 1u) == 0u) {
        // Release old voxel
        occ[x][y][z][CELL_TYPE_FIB].exchange(0u);
        FLAMEGPU->setVariable<int>("x", nx);
        FLAMEGPU->setVariable<int>("y", ny);
        FLAMEGPU->setVariable<int>("z", nz);
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: State step — activation, lifespan, division cooldown
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_state_step, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cs = FLAMEGPU->getVariable<int>("cell_state");

    // Lifespan check
    int life = FLAMEGPU->getVariable<int>("life");
    life--;
    FLAMEGPU->setVariable<int>("life", life);
    if (life <= 0) {
        auto* ec = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        int death_idx;
        if (cs == FIB_QUIESCENT)  death_idx = EVT_DEATH_FIB_QUIESCENT;
        else if (cs == FIB_MYCAF) death_idx = EVT_DEATH_FIB_MYCAF;
        else                      death_idx = EVT_DEATH_FIB_ICAF;
        atomicAdd(&ec[death_idx], 1u);
        return flamegpu::DEAD;
    }

    // Read local concentrations for activation
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * grid_y * grid_x + ay * grid_x + ax;

    // Activation: only quiescent fibroblasts can activate
    if (cs == FIB_QUIESCENT) {
        const float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel);
        const float IL1  = PDE_READ(FLAMEGPU, PDE_CONC_IL1, voxel);
        const float k_act = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ACTIVATION_RATE");

        // myCAF activation: TGF-β Hill function
        const float mycaf_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_TGFB_EC50");
        const float mycaf_n    = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_TGFB_HILL_N");
        const float p_mycaf = k_act * hill_equation(TGFB, mycaf_ec50, mycaf_n);

        if (FLAMEGPU->random.uniform<float>() < p_mycaf) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_MYCAF);
            return flamegpu::ALIVE;
        }

        // iCAF activation: IL-1 Hill * (1 - TGF-β suppression Hill)
        const float icaf_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_EC50");
        const float icaf_n     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_HILL_N");
        const float supp_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_TGFB_SUPPRESS_EC50");
        const float supp_n     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_TGFB_SUPPRESS_N");
        const float p_icaf = k_act * hill_equation(IL1, icaf_ec50, icaf_n)
                            * (1.0f - hill_equation(TGFB, supp_ec50, supp_n));

        if (FLAMEGPU->random.uniform<float>() < p_icaf) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_ICAF);
            return flamegpu::ALIVE;
        }
    }

    // Division cooldown decrement
    int cooldown = FLAMEGPU->getVariable<int>("divide_cooldown");
    if (cooldown > 0) {
        FLAMEGPU->setVariable<int>("divide_cooldown", cooldown - 1);
    }

    // Division check: only activated fibroblasts, off cooldown, under max count
    if (cs != FIB_QUIESCENT && cooldown <= 0) {
        const int div_count = FLAMEGPU->getVariable<int>("divide_count");
        const int div_max = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_MAX"));
        if (div_count < div_max) {
            const float div_prob = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_PROB");
            if (FLAMEGPU->random.uniform<float>() < div_prob) {
                FLAMEGPU->setVariable<int>("divide_flag", 1);
            }
        }
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Division — device-side local mitosis
// Daughter placed in adjacent free Von Neumann voxel
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_divide, flamegpu::MessageNone, flamegpu::MessageNone) {
    if (FLAMEGPU->getVariable<int>("divide_flag") == 0) return flamegpu::ALIVE;
    FLAMEGPU->setVariable<int>("divide_flag", 0);

    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    if (cs == FIB_QUIESCENT) return flamegpu::ALIVE;

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");

    // Try Von Neumann neighbors (6 face-adjacent) in random order
    int dx6[6] = {1, -1, 0, 0, 0, 0};
    int dy6[6] = {0, 0, 1, -1, 0, 0};
    int dz6[6] = {0, 0, 0, 0, 1, -1};
    // Fisher-Yates shuffle
    for (int i = 5; i > 0; i--) {
        int j = static_cast<int>(FLAMEGPU->random.uniform<float>() * (i + 1));
        if (j > i) j = i;
        int tx = dx6[i]; dx6[i] = dx6[j]; dx6[j] = tx;
        int ty = dy6[i]; dy6[i] = dy6[j]; dy6[j] = ty;
        int tz = dz6[i]; dz6[i] = dz6[j]; dz6[j] = tz;
    }

    for (int d = 0; d < 6; d++) {
        int nx = x + dx6[d], ny = y + dy6[d], nz = z + dz6[d];
        if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
        if (occ[nx][ny][nz][CELL_TYPE_CANCER] != 0u) continue;
        if (occ[nx][ny][nz][CELL_TYPE_FIB].CAS(0u, 1u) != 0u) continue;

        // Success — create daughter
        const int cd = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_COOLDOWN"));

        // Update parent
        FLAMEGPU->setVariable<int>("divide_cooldown", cd);
        FLAMEGPU->setVariable<int>("divide_count", FLAMEGPU->getVariable<int>("divide_count") + 1);

        // Create daughter agent
        auto daughter = FLAMEGPU->agent_out;
        daughter.setVariable<int>("x", nx);
        daughter.setVariable<int>("y", ny);
        daughter.setVariable<int>("z", nz);
        daughter.setVariable<int>("cell_state", cs);  // inherits parent state
        daughter.setVariable<int>("life", FLAMEGPU->getVariable<int>("life"));
        daughter.setVariable<int>("divide_flag", 0);
        daughter.setVariable<int>("divide_cooldown", cd);
        daughter.setVariable<int>("divide_count", 0);
        daughter.setVariable<float>("move_direction_x", 0.0f);
        daughter.setVariable<float>("move_direction_y", 0.0f);
        daughter.setVariable<float>("move_direction_z", 0.0f);
        daughter.setVariable<int>("tumble", 1);

        // Record proliferation event
        auto* ec = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("event_counters_ptr"));
        int prolif_idx = (cs == FIB_MYCAF) ? EVT_PROLIF_FIB_MYCAF : EVT_PROLIF_FIB_ICAF;
        atomicAdd(&ec[prolif_idx], 1u);

        break;  // only one daughter per step
    }

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Build Gaussian density field for ECM deposition (myCAF only)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_build_density_field, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    if (cs != FIB_MYCAF) return flamegpu::ALIVE;  // only myCAF deposits ECM

    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    const int radius = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_RADIUS"));
    const float variance = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_VARIANCE");

    // Normalizer for 3D Gaussian: 1 / ((2π σ²)^(3/2))
    const float normalizer = 1.0f / (powf(2.0f * 3.14159265f * variance, 1.5f));

    float* field_ptr = reinterpret_cast<float*>(
        FLAMEGPU->environment.getProperty<uint64_t>("fib_density_field_ptr"));

    for (int dx = -radius; dx <= radius; dx++) {
        const int nx = x + dx;
        if (nx < 0 || nx >= grid_x) continue;
        for (int dy = -radius; dy <= radius; dy++) {
            const int ny = y + dy;
            if (ny < 0 || ny >= grid_y) continue;
            for (int dz = -radius; dz <= radius; dz++) {
                const int nz = z + dz;
                if (nz < 0 || nz >= grid_z) continue;
                float dist_sq = static_cast<float>(dx * dx + dy * dy + dz * dz);
                float kernel_val = normalizer * expf(-dist_sq / (2.0f * variance));
                atomicAdd(&field_ptr[nz * (grid_x * grid_y) + ny * grid_x + nx], kernel_val);
            }
        }
    }

    return flamegpu::ALIVE;
}

}  // namespace PDAC

#endif  // FIBROBLAST_CUH
