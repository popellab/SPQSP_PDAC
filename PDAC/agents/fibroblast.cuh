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
    FLAMEGPU->message_out.setVariable<int>("voxel_x", x);
    FLAMEGPU->message_out.setVariable<int>("voxel_y", y);
    FLAMEGPU->message_out.setVariable<int>("voxel_z", z);
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
// Fibroblast: Scan neighbors (for adhesion)
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_scan_neighbors, flamegpu::MessageSpatial3D, flamegpu::MessageNone) {
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const float wx = (x + 0.5f) * voxel_size;
    const float wy = (y + 0.5f) * voxel_size;
    const float wz = (z + 0.5f) * voxel_size;

    int cancer_count = 0;
    int fib_count = 0;

    for (const auto& msg : FLAMEGPU->message_in(wx, wy, wz)) {
        const int mx = msg.getVariable<int>("voxel_x");
        const int my = msg.getVariable<int>("voxel_y");
        const int mz = msg.getVariable<int>("voxel_z");
        if (abs(mx - x) <= 1 && abs(my - y) <= 1 && abs(mz - z) <= 1
            && !(mx == x && my == y && mz == z)) {
            const int at = msg.getVariable<int>("agent_type");
            if (at == CELL_TYPE_CANCER) cancer_count++;
            else if (at == CELL_TYPE_FIB) fib_count++;
        }
    }

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_fib_count", fib_count);
    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Write volume to occupancy grid
// ============================================================================
FLAMEGPU_AGENT_FUNCTION(fib_write_to_occ_grid, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");

    const int gx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int gy = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (gx * gy) + y * gx + x;
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    float my_vol = (cell_state == FIB_QUIESCENT) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_QUIESCENT") :
        (cell_state == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");
    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Compute chemical sources
// myCAF: TGF-β + CCL2 (HIF-boosted under hypoxia, reduced at severe hypoxia)
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

    // Hypoxia modulation for myCAF secretion
    float hif_boost = 1.0f;
    float metabolic_factor = 1.0f;
    if (cs == FIB_MYCAF) {
        const float local_O2 = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel);
        const float hyp_th = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_HYPOXIA_TH");
        if (local_O2 < hyp_th && hyp_th > 0.0f) {
            hif_boost = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_HIF_TGFB_BOOST");
            // PSC metabolic decline: at severe hypoxia (<50% of threshold), secretion drops
            const float severe_th = hyp_th * 0.5f;
            if (local_O2 < severe_th) {
                metabolic_factor = fmaxf(local_O2 / severe_th, 0.1f);
            }
        }
    }

    if (cs == FIB_MYCAF) {
        const float tgfb_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_TGFB_RELEASE");
        const float ccl2_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_CCL2_RELEASE");
        float eff_boost = hif_boost * metabolic_factor;
        PDE_SECRETE(FLAMEGPU, PDE_SRC_TGFB, voxel, tgfb_rate * eff_boost / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL2, voxel, ccl2_rate * eff_boost / voxel_volume);
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
// Fibroblast movement via unified movement framework.
// myCAF: TGF-β chemotaxis + persistence. iCAF: random walk + weak persistence.
// Quiescent: no movement. Activated: adhesion-gated via neighbor counts + ECM.
FLAMEGPU_AGENT_FUNCTION(fib_move, flamegpu::MessageNone, flamegpu::MessageNone) {
    const int x = FLAMEGPU->getVariable<int>("x");
    const int y = FLAMEGPU->getVariable<int>("y");
    const int z = FLAMEGPU->getVariable<int>("z");
    const int cs = FLAMEGPU->getVariable<int>("cell_state");

    // Quiescent fibroblasts don't move (preserve existing behavior)
    if (cs == FIB_QUIESCENT) return flamegpu::ALIVE;

    // Adhesion-based movement probability (ECM anchorage + cell-cell)

    float my_vol = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");

    float p_persist = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_FIB_ICAF");

    float bias = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_BIAS_FIB_MYCAF") : 0.0f;

    // Read TGF-β gradient for myCAF chemotaxis
    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int vidx = z * (grid_x * grid_y) + y * grid_x + x;
    float gx = 0.0f, gy = 0.0f, gz = 0.0f;
    if (cs == FIB_MYCAF) {
        gx = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_X))[vidx];
        gy = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Y))[vidx];
        gz = reinterpret_cast<const float*>(
            FLAMEGPU->environment.getProperty<uint64_t>(PDE_GRAD_TGFB_Z))[vidx];
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
    mp.min_porosity = FLAMEGPU->environment.getProperty<float>("PARAM_ECM_POROSITY_FIB");
    // Adhesion from neighbor counts + ECM
    {
        const int n_cancer = FLAMEGPU->getVariable<int>("neighbor_cancer_count");
        const int n_fib = FLAMEGPU->getVariable<int>("neighbor_fib_count");
        float local_ecm = mp.ecm_density[vidx];
        float ecm_th = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_ECM_DENSITY_TH");
        float a_cancer, a_fib, a_ecm;
        if (cs == FIB_MYCAF) {
            a_cancer = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_FIB_MYCAF_CANCER");
            a_fib    = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_FIB_MYCAF_FIB");
            a_ecm    = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_FIB_MYCAF_ECM");
        } else {
            a_cancer = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_FIB_ICAF_CANCER");
            a_fib    = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_FIB_ICAF_FIB");
            a_ecm    = FLAMEGPU->environment.getProperty<float>("PARAM_ADH_FIB_ICAF_ECM");
        }
        mp.p_move = compute_adhesion_pmove(a_cancer, n_cancer, a_fib, n_fib, a_ecm, local_ecm, ecm_th);
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

    // Volume-based occupancy
    float* vol_used = VOL_PTR(FLAMEGPU);
    const float capacity = FLAMEGPU->environment.getProperty<float>("PARAM_VOXEL_CAPACITY");
    float daughter_vol = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");

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
        int tvidx = nz * (grid_x * grid_y) + ny * grid_x + nx;
        if (!volume_try_claim(vol_used, tvidx, daughter_vol, capacity)) continue;

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
        daughter.setVariable<int>("persist_dir_x", 0);
        daughter.setVariable<int>("persist_dir_y", 0);
        daughter.setVariable<int>("persist_dir_z", 0);

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
// HIF-boosted under hypoxia, PSC metabolic decline at severe hypoxia.
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
    const int voxel = z * (grid_x * grid_y) + y * grid_x + x;

    const int radius = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_RADIUS"));
    const float variance = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_VARIANCE");

    // HIF ECM boost: hypoxic myCAFs produce more ECM, but severe hypoxia causes
    // metabolic decline (PSC self-limiting). Net effect = hif_boost * metabolic_factor.
    float ecm_multiplier = 1.0f;
    const float local_O2 = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel);
    const float hyp_th = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_HYPOXIA_TH");
    if (local_O2 < hyp_th && hyp_th > 0.0f) {
        ecm_multiplier = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_HIF_ECM_BOOST");
        // PSC metabolic decline at severe hypoxia (<50% threshold)
        const float severe_th = hyp_th * 0.5f;
        if (local_O2 < severe_th) {
            ecm_multiplier *= fmaxf(local_O2 / severe_th, 0.1f);
        }
    }

    // Normalizer for 3D Gaussian: 1 / ((2π σ²)^(3/2))
    const float normalizer = ecm_multiplier / (powf(2.0f * 3.14159265f * variance, 1.5f));

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
