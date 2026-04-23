// Fibroblast Agent Behavior Functions
// States: FIB_QUIESCENT, FIB_MYCAF (myofibroblastic CAF), FIB_ICAF (inflammatory CAF)
// Single-cell, single-voxel agents. No chain model.
//
// Activation: Quiescent → myCAF (TGF-β driven), Quiescent → iCAF (IL-1 driven, TGF-β suppressed)
// ECM deposition: myCAF only (Gaussian kernel, radius from XML)
// Chemical sources: myCAF → TGF-β + CCL2; iCAF → IL-6 + CCL2 + CXCL12 + CCL5
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
    FLAMEGPU->message_out.setVariable<float>("kill_factor", 0.0f);  // N/A for fibroblast
    FLAMEGPU->message_out.setVariable<int>("dead", 0);
    FLAMEGPU->message_out.setLocation(fx, fy, fz);

    // Population count by state
    const int cs = FLAMEGPU->getVariable<int>("cell_state");
    auto* sc = reinterpret_cast<unsigned int*>(FLAMEGPU->environment.getProperty<uint64_t>("state_counters_ptr"));
    int sc_idx;
    if (cs == FIB_QUIESCENT)  sc_idx = SC_FIB_QUIESCENT;
    else if (cs == FIB_MYCAF) sc_idx = SC_FIB_MYCAF;
    else if (cs == FIB_ICAF)  sc_idx = SC_FIB_ICAF;
    else                       sc_idx = SC_FIB_FRC;
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
    int ltb_lymph_count = 0;  // LTβ-competent lymphocytes (drives iCAF→FRC)

    // Adhesion: count all type+state neighbors
    int adh_counts[ABM_STATE_COUNTER_SIZE] = {0};

    for (const auto& msg : FLAMEGPU->message_in(wx, wy, wz)) {
        const int mx = msg.getVariable<int>("voxel_x");
        const int my = msg.getVariable<int>("voxel_y");
        const int mz = msg.getVariable<int>("voxel_z");
        if (abs(mx - x) <= 1 && abs(my - y) <= 1 && abs(mz - z) <= 1
            && !(mx == x && my == y && mz == z)) {
            const int at = msg.getVariable<int>("agent_type");
            const int ms = msg.getVariable<int>("cell_state");
            const float kf = msg.getVariable<float>("kill_factor");

            // Adhesion: accumulate into type+state count vector
            adh_counts[msg_to_sc_idx(at, ms, kf)]++;

            if (at == CELL_TYPE_CANCER) { cancer_count++; continue; }
            if (at == CELL_TYPE_FIB) { fib_count++; continue; }
            // LTβ-competent lymphocyte filter (state-gated)
            // Ansel2000/Fütterer1998: naive+activated B cells express surface LTα1β2; plasma downregulates (Mebius2003)
            // Luther2002/Scheu2002: TCR-activated CD8 (eff+cyt) produce LTβ; naive/suppressed excluded
            // Furtado2007/Kumar2015: Th+Tfh are canonical LTβ producers; Tregs excluded (TLS forms at effector, not regulatory, sites)
            if (at == CELL_TYPE_BCELL) {
                if (ms == BCELL_NAIVE || ms == BCELL_ACTIVATED) ltb_lymph_count++;
            } else if (at == CELL_TYPE_T) {
                if (ms == T_CELL_EFF || ms == T_CELL_CYT) ltb_lymph_count++;
            } else if (at == CELL_TYPE_TREG) {
                if (ms == TCD4_TH || ms == TCD4_TFH) ltb_lymph_count++;
            }
        }
    }

    // Compute adhesion p_move from matrix
    const int my_sc = self_sc_idx(CELL_TYPE_FIB, FLAMEGPU->getVariable<int>("cell_state"));
    const float adh_pmove = compute_adhesion_pmove(my_sc, adh_counts, ADH_MATRIX_PTR(FLAMEGPU));

    FLAMEGPU->setVariable<int>("neighbor_cancer_count", cancer_count);
    FLAMEGPU->setVariable<int>("neighbor_fib_count", fib_count);
    FLAMEGPU->setVariable<int>("neighbor_ltb_lymph_count", ltb_lymph_count);
    FLAMEGPU->setVariable<float>("adh_p_move", adh_pmove);
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
    const int gz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    if (x < 0 || x >= gx || y < 0 || y >= gy || z < 0 || z >= gz) {
        return flamegpu::ALIVE;
    }
    const int vidx = z * (gx * gy) + y * gx + x;
    const int cell_state = FLAMEGPU->getVariable<int>("cell_state");
    float my_vol;
    if (cell_state == FIB_QUIESCENT) my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_QUIESCENT");
    else if (cell_state == FIB_MYCAF) my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF");
    else if (cell_state == FIB_ICAF)  my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");
    else                               my_vol = FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_FRC");
    float* vol_used = VOL_PTR(FLAMEGPU);
    atomicAdd(&vol_used[vidx], my_vol);

    return flamegpu::ALIVE;
}

// ============================================================================
// Fibroblast: Compute chemical sources
// myCAF: TGF-β + CCL2 (HIF-boosted under hypoxia, reduced at severe hypoxia)
// iCAF:  IL-6 + CCL2 + CXCL12 + CCL5
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
        const float tgfb_rate  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_TGFB_RELEASE");
        const float ccl2_rate  = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_RELEASE");
        const float vegfa_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_VEGFA_RELEASE");
        float eff_boost = hif_boost * metabolic_factor;
        PDE_SECRETE(FLAMEGPU, PDE_SRC_TGFB,  voxel, tgfb_rate  * eff_boost / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL2,  voxel, ccl2_rate  * eff_boost / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_VEGFA, voxel, vegfa_rate * eff_boost / voxel_volume);
    } else if (cs == FIB_ICAF) {
        const float il6_rate    = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL6_RELEASE");
        const float ccl2_rate   = FLAMEGPU->environment.getProperty<float>("PARAM_CCL2_RELEASE");
        const float cxcl12_rate = FLAMEGPU->environment.getProperty<float>("PARAM_CXCL12_ICAF_RELEASE");
        const float ccl5_rate   = FLAMEGPU->environment.getProperty<float>("PARAM_CCL5_ICAF_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_IL6,    voxel, il6_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL2,   voxel, ccl2_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CXCL12, voxel, cxcl12_rate / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL5,   voxel, ccl5_rate / voxel_volume);
    } else {  // FIB_FRC — TLS T-zone scaffold, CCL21 source (Luther2000, Link2007)
        const float ccl21_rate  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_FRC_CCL21_RELEASE");
        const float cxcl12_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_FRC_CXCL12_RELEASE");
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL21,  voxel, ccl21_rate  / voxel_volume);
        PDE_SECRETE(FLAMEGPU, PDE_SRC_CXCL12, voxel, cxcl12_rate / voxel_volume);
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

    // Quiescent + FRC fibroblasts are sessile (preserve tissue architecture / TLS scaffold)
    if (cs == FIB_QUIESCENT || cs == FIB_FRC) return flamegpu::ALIVE;

    // Adhesion-based movement probability (ECM anchorage + cell-cell)

    float my_vol = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_VOLUME_FIB_ICAF");

    float p_persist = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_PERSIST_FIB_ICAF");

    float bias = (cs == FIB_MYCAF) ?
        ci_to_bias(FLAMEGPU->environment.getProperty<float>("PARAM_CHEMO_CI_FIB_MYCAF")) : 0.0f;

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
    // Adhesion: pre-computed from matrix in scan_neighbors
    mp.p_move = FLAMEGPU->getVariable<float>("adh_p_move");
    mp.p_persist = p_persist;
    mp.bias_strength = bias;
    mp.grad_x = gx; mp.grad_y = gy; mp.grad_z = gz;
    mp.orient_x = ECM_ORIENT_X_PTR(FLAMEGPU);
    mp.orient_y = ECM_ORIENT_Y_PTR(FLAMEGPU);
    mp.orient_z = ECM_ORIENT_Z_PTR(FLAMEGPU);
    mp.barrier_strength = (cs == FIB_MYCAF) ?
        FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_FIB_MYCAF") :
        FLAMEGPU->environment.getProperty<float>("PARAM_FIBER_BARRIER_FIB_ICAF");

    // Contact guidance
    {
        float w_cg = (cs == FIB_MYCAF) ?
            FLAMEGPU->environment.getProperty<float>("PARAM_CONTACT_GUIDANCE_FIB_MYCAF") :
            FLAMEGPU->environment.getProperty<float>("PARAM_CONTACT_GUIDANCE_FIB_ICAF");
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
        else if (cs == FIB_ICAF)  death_idx = EVT_DEATH_FIB_ICAF;
        else                       death_idx = EVT_DEATH_FIB_FRC;
        atomicAdd(&ec[death_idx], 1u);
        return flamegpu::DEAD;
    }

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int ax = FLAMEGPU->getVariable<int>("x");
    const int ay = FLAMEGPU->getVariable<int>("y");
    const int az = FLAMEGPU->getVariable<int>("z");
    const int voxel = az * grid_y * grid_x + ay * grid_x + ax;

    // Read local concentrations for activation and interconversion
    const float TGFB = PDE_READ(FLAMEGPU, PDE_CONC_TGFB, voxel);
    const float IL1  = PDE_READ(FLAMEGPU, PDE_CONC_IL1, voxel);
    const float IL6  = PDE_READ(FLAMEGPU, PDE_CONC_IL6, voxel);
    const float caf_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_CAF_EC50");
    const float H_TGFb = TGFB / (TGFB + caf_ec50 + 1e-30f);  // shared TGFb_50_CAF_act

    // IL-6 amplification for iCAF pathways: (1 + f_IL6 * H_IL6_iCAF)
    const float f_IL6  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_IL6_ICAF_FACTOR");
    const float il6_ec50 = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_IL6_ICAF_EC50");
    const float H_IL6 = IL6 / (IL6 + il6_ec50 + 1e-30f);
    const float il6_boost = 1.0f + f_IL6 * H_IL6;

    // Activation: only quiescent fibroblasts can activate
    if (cs == FIB_QUIESCENT) {
        // myCAF activation (ODE RF242): k_myCAF * H_TGFb(TGFb_50_CAF_act)
        const float k_mycaf = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_ACTIVATION");
        const float p_mycaf = k_mycaf * H_TGFb;

        if (FLAMEGPU->random.uniform<float>() < p_mycaf) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_MYCAF);
            return flamegpu::ALIVE;
        }

        // iCAF activation (ODE RF243): k_iCAF * H_IL1_eff * (1 + f_IL6 * H_IL6_iCAF)
        // ABM: IL-1 Hill * (1 - TGF-β suppression) * IL-6 boost
        const float k_icaf   = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_ACTIVATION");
        const float icaf_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_EC50");
        const float icaf_n     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_HILL_N");
        const float supp_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_TGFB_SUPPRESS_EC50");
        const float supp_n     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_TGFB_SUPPRESS_N");
        const float p_icaf = k_icaf * hill_equation(IL1, icaf_ec50, icaf_n)
                            * (1.0f - hill_equation(TGFB, supp_ec50, supp_n))
                            * il6_boost;

        if (FLAMEGPU->random.uniform<float>() < p_icaf) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_ICAF);
            return flamegpu::ALIVE;
        }
    }

    // iCAF ↔ myCAF interconversion (ODE RF245, RF246)
    if (cs == FIB_ICAF) {
        // iCAF → FRC: LTβ-competent lymphocyte density gating (takes priority over iCAF→myCAF)
        // Sustained threshold for PARAM_FIB_FRC_DWELL_STEPS → irreversible FRC commitment.
        // Refs: Bénézech2012 (FRC maturation kinetics), Drayton2003 (ectopic TLS cell density)
        const int n_ltb = FLAMEGPU->getVariable<int>("neighbor_ltb_lymph_count");
        const int frc_th = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_FRC_LYMPH_THRESHOLD"));
        int dwell = FLAMEGPU->getVariable<int>("frc_dwell_counter");
        if (n_ltb >= frc_th) {
            dwell++;
        } else {
            dwell = 0;  // reset if threshold not sustained
        }
        const int frc_dwell_max = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_FRC_DWELL_STEPS"));
        if (dwell >= frc_dwell_max) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_FRC);
            FLAMEGPU->setVariable<int>("frc_dwell_counter", 0);
            return flamegpu::ALIVE;
        }
        FLAMEGPU->setVariable<int>("frc_dwell_counter", dwell);

        // iCAF → myCAF: k_iCAF_to_myCAF * H_TGFb(TGFb_50_CAF_act)
        const float k_i2m = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_TO_MYCAF");
        if (FLAMEGPU->random.uniform<float>() < k_i2m * H_TGFb) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_MYCAF);
        }
    } else if (cs == FIB_MYCAF) {
        // myCAF → iCAF: k_myCAF_to_iCAF * H_IL1_eff * (1 + f_IL6 * H_IL6_iCAF)
        const float k_m2i = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_MYCAF_TO_ICAF");
        const float icaf_ec50  = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_EC50");
        const float icaf_n     = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ICAF_IL1_HILL_N");
        float H_IL1 = hill_equation(IL1, icaf_ec50, icaf_n);
        if (FLAMEGPU->random.uniform<float>() < k_m2i * H_IL1 * il6_boost) {
            FLAMEGPU->setVariable<int>("cell_state", FIB_ICAF);
        }
    }

    // Division cooldown decrement
    int cooldown = FLAMEGPU->getVariable<int>("divide_cooldown");
    if (cooldown > 0) {
        FLAMEGPU->setVariable<int>("divide_cooldown", cooldown - 1);
    }

    // Low-rate CAF proliferation: k_*CAF_prolif in XML is dialed down to the death
    // rate so births balance deaths (stable population with turnover). Kept the
    // divide_count cap as a safeguard; remove or raise once the QSP stroma
    // carrying-cap term (RF247/248) is mirrored on the ABM side.
    if ((cs == FIB_MYCAF || cs == FIB_ICAF) && cooldown <= 0) {
        const int div_count = FLAMEGPU->getVariable<int>("divide_count");
        const int div_max = static_cast<int>(FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_MAX"));
        if (div_count < div_max) {
            FLAMEGPU->setVariable<int>("divide_flag", 1);
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
    if (cs == FIB_QUIESCENT || cs == FIB_FRC) return flamegpu::ALIVE;

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

        // Success — create daughter (per-subtype cooldown from QSP k_prolif)
        const int cd = static_cast<int>((cs == FIB_MYCAF) ?
            FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_COOLDOWN_MYCAF") :
            FLAMEGPU->environment.getProperty<float>("PARAM_FIB_DIV_COOLDOWN_ICAF"));

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
