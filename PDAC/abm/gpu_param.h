#ifndef GPU_PARAM_H
#define GPU_PARAM_H

#include "../core/ParamBase.h"
#include <string>

// Forward declaration to avoid including CUDA headers in .cpp compilation
namespace flamegpu {
    class EnvironmentDescription;
}

namespace PDAC {

// Type-safe enums for parameter access
enum GPUParamFloat {
    // Environment parameters
    PARAM_SEC_PER_SLICE,
    PARAM_REC_SITE_FACTOR,
    PARAM_ADH_SITE_DENSITY,
    PARAM_WEIGHT_QSP,
    //*************************************************************************/
    // Drug parameters
    PARAM_NIVO_DOSE_INTERVAL_TIME,
    PARAM_NIVO_DOSE,
    PARAM_IPI_DOSE_INTERVAL_TIME,
    PARAM_IPI_DOSE,
    PARAM_CABO_DOSE_INTERVAL_TIME,
    PARAM_CABO_DOSE,
    //*************************************************************************/
    // Base cell parameters
    PARAM_PDL1_TH,
    PARAM_IFNG_PDL1_HALF,
    PARAM_IFNG_PDL1_N,
    PARAM_PDL1_HALF,
    //*************************************************************************/
    // Cancer cell parameters
    PARAM_PROG_GROWTH_RATE,
    PARAM_SEN_DEATH_RATE,
    PARAM_ASYM_DIV_PROB,
    PARAM_CANCER_IFNG_UPTAKE,
    PARAM_CANCER_HYPOXIA_TH,
    PARAM_HIF_VEGF_BOOST,             // HIF VEGF-A secretion multiplier (e.g., 5×)
    PARAM_HIF_CCL2_BOOST,             // HIF CCL2 secretion multiplier (e.g., 3×)
    PARAM_HIF_PDL1_BOOST,             // HIF additive PDL1 boost (e.g., 0.3)
    PARAM_HIF_MHC_REDUCTION,          // Kill prob multiplier for HIF-active targets (e.g., 0.5)
    PARAM_DENSITY_STEM,
    PARAM_TKILL_SCALAR,
    PARAM_MIN_CC,
    PARAM_C1_CD47,
    //*************************************************************************/
    // T cell parameters
    PARAM_TCELL_IFNG_RECRUIT_HALF,
    PARAM_TCELL_LIFESPAN_SD,
    PARAM_TCELL_IL2_RELEASE_TIME,
    PARAM_TCELL_IL2_PROLIF_TH,
    PARAM_TCELL_IFNG_RELEASE_TIME,
    PARAM_TEFF_IFN_EC50,
    PARAM_TCELL_HYPOXIA_TH,           // O2 threshold for T cell hypoxia [mM]
    PARAM_TCELL_HYPOXIA_TIER1,        // Exposure steps for low impairment onset
    PARAM_TCELL_HYPOXIA_TIER2,        // Exposure steps for medium impairment onset
    PARAM_TCELL_HYPOXIA_RECOVERY,     // Recovery decrement per normoxic step
    PARAM_TCELL_HYPOXIA_FACTOR1,      // Kill/secretion multiplier: low tier
    PARAM_TCELL_HYPOXIA_FACTOR2,      // All multiplier: medium tier
    PARAM_TCELL_HYPOXIA_FACTOR3,      // All multiplier: high tier (near exhaustion)
    //*************************************************************************/
    // TCD4 cell parameters
    PARAM_TCD4_LIFESPAN_SD,
    PARAM_TCD4_TGFB_RELEASE_TIME,
    //*************************************************************************/
    // MDSC cell parameters
    PARAM_MDSC_LIFESPAN_SD,
    //*************************************************************************/
    // Macrophage cell parameters
    PARAM_MAC_HYPOXIA_TH,             // O2 threshold for hypoxic M2 bias [mM]
    PARAM_MAC_M2_BIAS_STRENGTH,       // Strength of hypoxic M2 polarization shift (0-1)
    //*************************************************************************/
    // Fibroblast cell parameters
    PARAM_FIB_MYCAF_TGFB_EC50,        // TGF-β EC50 for quiescent→myCAF activation
    PARAM_FIB_MYCAF_TGFB_HILL_N,      // Hill coefficient for myCAF activation
    PARAM_FIB_ICAF_IL1_EC50,          // IL-1 EC50 for quiescent→iCAF activation
    PARAM_FIB_ICAF_IL1_HILL_N,        // Hill coefficient for iCAF activation
    PARAM_FIB_ICAF_TGFB_SUPPRESS_EC50,// TGF-β EC50 for iCAF suppression
    PARAM_FIB_ICAF_TGFB_SUPPRESS_N,   // Hill coefficient for TGF-β suppression of iCAF
    PARAM_FIB_ACTIVATION_RATE,        // Base activation probability per step
    PARAM_FIB_MOVE_PROB_QUIESCENT,    // Movement probability per step (quiescent)
    PARAM_FIB_MOVE_PROB_MYCAF,        // Movement probability per step (myCAF)
    PARAM_FIB_MOVE_PROB_ICAF,         // Movement probability per step (iCAF)
    PARAM_FIB_MYCAF_TGFB_RELEASE,     // myCAF TGF-β secretion rate
    PARAM_FIB_MYCAF_CCL2_RELEASE,     // myCAF CCL2 secretion rate
    PARAM_FIB_ICAF_IL6_RELEASE,       // iCAF IL-6 secretion rate
    PARAM_FIB_ICAF_CXCL13_RELEASE,    // iCAF CXCL13 secretion rate
    PARAM_FIB_ICAF_CCL2_RELEASE,      // iCAF CCL2 secretion rate
    PARAM_FIB_ECM_RADIUS,             // ECM Gaussian kernel radius (voxels)
    PARAM_FIB_ECM_VARIANCE,           // ECM Gaussian kernel variance (σ²)
    PARAM_FIB_DIV_PROB,               // Base division probability per step (activated)
    PARAM_FIB_DIV_COOLDOWN,           // Steps between divisions
    PARAM_FIB_DIV_MAX,                // Max division count before senescence
    //*************************************************************************/
    // Vas cell parameters
    PARAM_VAS_MAXPERVOXEL,
    PARAM_VAS_50,
    PARAM_VAS_O2_CONC,
    PARAM_VAS_RC,
    PARAM_VAS_SIGMA,
    PARAM_VAS_FRAC,
    PARAM_VAS_INIT_DENS,
    PARAM_VAS_TUMBLE,
    PARAM_VAS_DELTA,
    PARAM_VAS_BRANCH_PROB,
    PARAM_VAS_MIN_NEIGHBOR,
    PARAM_VAS_MATURITY_RESISTANCE,    // How much maturity reduces compression/regression
    PARAM_VAS_INITIAL_MATURITY,       // Maturity value for pre-existing vessels
    PARAM_VAS_COLLAPSE_THRESHOLD,     // ECM density above which collapse is possible
    PARAM_VAS_COLLAPSE_EC50,          // Hill EC50 for collapse probability
    PARAM_VAS_RECOVERY_THRESHOLD,     // ECM density below which recovery is possible
    PARAM_VAS_RECOVERY_RATE,          // Per-step recovery probability
    PARAM_VAS_REGRESS_RATE,           // Per-step regression base rate at zero VEGF
    PARAM_VAS_VEGFA_SURVIVAL_EC50,    // VEGF-A for 50% vessel survival
    PARAM_VAS_KVL_DYSFUNCTIONAL,      // Fraction of K_vl for dysfunctional sprouts (0-1)
    PARAM_VAS_HYPOXIA_TH,            // O2 threshold for dysfunctional sprout flag
    //*************************************************************************/
    // Volume-based occupancy parameters
    PARAM_VOXEL_CAPACITY,
    PARAM_VOLUME_CANCER_STEM,
    PARAM_VOLUME_CANCER_PROG,
    PARAM_VOLUME_CANCER_SEN,
    PARAM_VOLUME_TCELL_EFF,
    PARAM_VOLUME_TCELL_CYT,
    PARAM_VOLUME_TCELL_SUP,
    PARAM_VOLUME_TREG_REG,
    PARAM_VOLUME_TREG_TH,
    PARAM_VOLUME_MAC_M1,
    PARAM_VOLUME_MAC_M2,
    PARAM_VOLUME_FIB_QUIESCENT,
    PARAM_VOLUME_FIB_MYCAF,
    PARAM_VOLUME_FIB_ICAF,
    PARAM_VOLUME_MDSC,
    PARAM_VOLUME_VAS_PHALANX,
    PARAM_VOLUME_VAS_TIP,
    //*************************************************************************/
    // Molecular parameters
    PARAM_IFNG_DIFFUSIVITY,
    PARAM_IL2_DIFFUSIVITY,
    PARAM_CCL2_DIFFUSIVITY,
    PARAM_ARGI_DIFFUSIVITY,
    PARAM_NO_DIFFUSIVITY,
    PARAM_TGFB_DIFFUSIVITY,
    PARAM_IL10_DIFFUSIVITY,
    PARAM_IL12_DIFFUSIVITY,
    PARAM_VEGFA_DIFFUSIVITY,
    PARAM_O2_DIFFUSIVITY,
    PARAM_IL1_DIFFUSIVITY,
    PARAM_IL6_DIFFUSIVITY,
    PARAM_CXCL13_DIFFUSIVITY,

    PARAM_IFNG_DECAY_RATE,
    PARAM_IL2_DECAY_RATE,
    PARAM_CCL2_DECAY_RATE,
    PARAM_ARGI_DECAY_RATE,
    PARAM_NO_DECAY_RATE,
    PARAM_TGFB_DECAY_RATE,
    PARAM_IL10_DECAY_RATE,
    PARAM_IL12_DECAY_RATE,
    PARAM_VEGFA_DECAY_RATE,
    PARAM_O2_DECAY_RATE,
    PARAM_IL1_DECAY_RATE,
    PARAM_IL6_DECAY_RATE,
    PARAM_CXCL13_DECAY_RATE,

    PARAM_IFNG_RELEASE,
    PARAM_IL2_RELEASE,
    PARAM_CCL2_RELEASE,
    PARAM_ARGI_RELEASE,
    PARAM_NO_RELEASE,
    PARAM_TREG_TGFB_RELEASE,
    PARAM_STEM_TGFB_RELEASE,
    PARAM_PROG_TGFB_RELEASE,
    PARAM_MAC_TGFB_RELEASE,
    PARAM_TREG_IL10_RELEASE,
    PARAM_MAC_IL10_RELEASE,
    PARAM_IL12_RELEASE,
    PARAM_STEM_VEGFA_RELEASE,
    PARAM_PROG_VEGFA_RELEASE,
    PARAM_MAC_VEGFA_RELEASE,
    PARAM_CANCER_IL1_RELEASE_STEM,    // Stem IL-1 secretion (highest, drives iCAF activation)
    PARAM_CANCER_IL1_RELEASE_PROG,    // Progenitor IL-1 secretion
    PARAM_CANCER_IL1_RELEASE_SEN,     // Senescent IL-1 secretion (SASP)
    PARAM_MAC_M1_IL1_RELEASE,         // M1 macrophage IL-1 secretion

    PARAM_IL2_UPTAKE,
    PARAM_CCL2_UPTAKE,
    PARAM_VEGFA_UPTAKE,
    PARAM_O2_UPTAKE,

    PARAM_IFNG_MOLECULAR_WEIGHT,
    PARAM_IL2_MOLECULAR_WEIGHT,
    PARAM_CCL2_MOLECULAR_WEIGHT,
    PARAM_ARGI_MOLECULAR_WEIGHT,
    PARAM_NO_MOLECULAR_WEIGHT,
    PARAM_TGFB_MOLECULAR_WEIGHT,
    PARAM_IL10_MOLECULAR_WEIGHT,
    PARAM_IL12_MOLECULAR_WEIGHT,
    PARAM_VEGFA_MOLECULAR_WEIGHT,
    PARAM_O2_MOLECULAR_WEIGHT,
    PARAM_IL1_MOLECULAR_WEIGHT,
    PARAM_IL6_MOLECULAR_WEIGHT,
    PARAM_CXCL13_MOLECULAR_WEIGHT,
    PARAM_MMP_DIFFUSIVITY,
    PARAM_MMP_DECAY_RATE,
    PARAM_MMP_MOLECULAR_WEIGHT,
    PARAM_CANCER_MMP_RELEASE,            // Cancer cell MMP secretion rate
    PARAM_MAC_M1_MMP_RELEASE,            // M1 macrophage MMP secretion rate
    //*************************************************************************/
    // ECM mechanics parameters
    PARAM_ECM_DEPOSITION_RATE,           // k_depo: myCAF deposition rate [1/s]
    PARAM_ECM_DENSITY_CAP,               // Maximum ECM density per voxel
    PARAM_ECM_MMP_DEGRADE_RATE,          // k_mmp: MMP degradation rate [1/(s*mM)]
    PARAM_ECM_CROSSLINK_RATE,            // k_lox: LOX crosslink accumulation [1/s]
    PARAM_ECM_CROSSLINK_RESISTANCE,      // alpha: crosslink MMP resistance factor
    PARAM_ECM_DECAY_RATE,                // k_decay: baseline ECM decay [1/s]
    PARAM_ECM_BASELINE,                  // Floor density
    PARAM_ECM_TGFB_EC50,                // TGF-β EC50 for deposition Hill function
    PARAM_ECM_POROSITY_CANCER,           // Min porosity for cancer movement
    PARAM_ECM_POROSITY_TCELL,            // Min porosity for T cell movement
    PARAM_ECM_POROSITY_TREG,             // Min porosity for TReg movement
    PARAM_ECM_POROSITY_MDSC,             // Min porosity for MDSC movement
    PARAM_ECM_POROSITY_MAC,              // Min porosity for macrophage movement
    PARAM_ECM_POROSITY_FIB,              // Min porosity for fibroblast movement
    PARAM_ECM_POROSITY_VAS_TIP,          // Min porosity for vascular TIP movement
    PARAM_VAS_ECM_COMPRESS_K,            // Half-max ECM density for vascular compression
    PARAM_ECM_YAP_EC50,                  // YAP/TAZ Hill ceiling: stiffness EC50 for deposition feedback
    //*************************************************************************/
    // Fibroblast HIF/hypoxia parameters
    PARAM_FIB_HYPOXIA_TH,               // O2 threshold for HIF activation in myCAFs
    PARAM_FIB_HIF_TGFB_BOOST,           // TGF-β secretion multiplier under hypoxia
    PARAM_FIB_HIF_ECM_BOOST,            // ECM deposition multiplier under hypoxia
    //*************************************************************************/
    // Domain initialization parameters
    PARAM_DOMAIN_LOBULE_SPACING,         // Voxels between lobule centers (Poisson disk)
    PARAM_DOMAIN_SEPTUM_THICKNESS,       // Voxels, Voronoi boundary width
    PARAM_DOMAIN_TUMOR_RADIUS_FRAC,      // Fraction of grid_x for tumor hemisphere radius
    PARAM_DOMAIN_MARGIN_THICKNESS,       // Voxels around tumor surface → VOXEL_MARGIN
    PARAM_DOMAIN_ECM_SEPTUM_DENSITY,     // ECM density in septum voxels
    PARAM_DOMAIN_ECM_SEPTUM_CROSSLINK,   // ECM crosslink in septum voxels
    PARAM_DOMAIN_ECM_STROMA_DENSITY,     // ECM density in stroma voxels
    PARAM_DOMAIN_ECM_LOBULE_DENSITY,     // ECM density in lobule voxels
    PARAM_DOMAIN_ECM_MARGIN_DENSITY,     // ECM density in margin voxels
    PARAM_DOMAIN_ECM_MARGIN_CROSSLINK,   // ECM crosslink in margin voxels
    PARAM_DOMAIN_ECM_TUMOR_DENSITY,      // ECM density in tumor voxels
    PARAM_DOMAIN_FIB_P_SEPTUM,           // Fibroblast placement prob in septum
    PARAM_DOMAIN_FIB_P_STROMA,           // Fibroblast placement prob in stroma
    PARAM_DOMAIN_FIB_P_LOBULE,           // Fibroblast placement prob in lobule
    PARAM_DOMAIN_FIB_P_MARGIN,           // Fibroblast placement prob in margin
    PARAM_DOMAIN_FIB_P_TUMOR,            // Fibroblast placement prob in tumor
    PARAM_DOMAIN_FIB_MARGIN_MYCAF_FRAC,  // Fraction of margin fibroblasts pre-activated as myCAF
    PARAM_DOMAIN_VAS_P_SEPTUM,           // Vascular placement prob in septum
    PARAM_DOMAIN_VAS_P_STROMA,           // Vascular placement prob in stroma
    PARAM_DOMAIN_VAS_P_LOBULE,           // Vascular placement prob in lobule
    PARAM_DOMAIN_VAS_P_MARGIN,           // Vascular placement prob in margin
    PARAM_DOMAIN_VAS_P_TUMOR,            // Vascular placement prob in tumor
    PARAM_DOMAIN_VAS_MARGIN_TIP_FRAC,    // Fraction of margin vascular cells as TIP
    PARAM_DOMAIN_MAC_P_STROMA,           // Macrophage placement prob in stroma
    PARAM_DOMAIN_MAC_P_MARGIN,           // Macrophage placement prob in margin
    PARAM_DOMAIN_MAC_P_TUMOR,            // Macrophage placement prob in tumor
    PARAM_DOMAIN_MAC_MARGIN_M1_FRAC,     // Fraction of margin macrophages as M1
    PARAM_DOMAIN_TH_P_STROMA,            // TH cell placement prob in stroma
    //*************************************************************************/
    // Movement framework parameters (persistence, chemotaxis bias)
    // Persistence: probability of continuing previous direction each substep
    PARAM_PERSIST_TCELL_EFF,
    PARAM_PERSIST_TCELL_CYT,
    PARAM_PERSIST_TCELL_SUP,
    PARAM_PERSIST_MAC_M1,
    PARAM_PERSIST_MAC_M2,
    PARAM_PERSIST_FIB_MYCAF,
    PARAM_PERSIST_FIB_ICAF,
    PARAM_PERSIST_VAS_TIP,
    PARAM_PERSIST_TREG_REG,
    PARAM_PERSIST_TREG_TH,
    PARAM_PERSIST_MDSC,
    // Chemotaxis bias: gradient-weighted direction selection strength (0=uniform)
    PARAM_CHEMO_BIAS_CANCER_STEM,        // O2 (weak, defer until O2 gradient added)
    PARAM_CHEMO_BIAS_MAC_M1,             // CCL2
    PARAM_CHEMO_BIAS_MAC_M2,             // CCL2
    PARAM_CHEMO_BIAS_FIB_MYCAF,          // TGF-β
    PARAM_CHEMO_BIAS_VAS_TIP,            // VEGF-A
    PARAM_CHEMO_BIAS_TREG_REG,           // TGF-β
    PARAM_CHEMO_BIAS_MDSC,               // CCL2
    // Adhesion: ECM density threshold for pseudo-neighbor count
    PARAM_ADH_ECM_DENSITY_TH,
    // Adhesion coefficients (non-zero entries only)
    PARAM_ADH_CANCER_STEM_CANCER,
    PARAM_ADH_CANCER_STEM_FIB,
    PARAM_ADH_CANCER_STEM_ECM,
    PARAM_ADH_CANCER_PROG_CANCER,
    PARAM_ADH_CANCER_PROG_FIB,
    PARAM_ADH_CANCER_PROG_ECM,
    PARAM_ADH_TCELL_CYT_CANCER,
    PARAM_ADH_MAC_M2_CANCER,
    PARAM_ADH_MAC_M2_FIB,
    PARAM_ADH_MAC_M2_ECM,
    PARAM_ADH_FIB_MYCAF_CANCER,
    PARAM_ADH_FIB_MYCAF_FIB,
    PARAM_ADH_FIB_MYCAF_ECM,
    PARAM_ADH_FIB_ICAF_CANCER,
    PARAM_ADH_FIB_ICAF_FIB,
    PARAM_ADH_FIB_ICAF_ECM,

    GPU_PARAM_FLOAT_COUNT
};

enum GPUParamInt {
    // Environment parameters
    PARAM_SHUFFLE_INTERVAL,
    PARAM_GRID_SHIFT_INTERVAL,
    PARAM_X_SIZE,
    PARAM_Y_SIZE,
    PARAM_Z_SIZE,
    PARAM_VOXEL_SIZE,
    PARAM_NR_T_VOXELS,
    PARAM_NR_T_VOXELS_C,
    PARAM_STEM_MODE,
    PARAM_MOLECULAR_STEPS,
    //*************************************************************************/
    // Drug parameters
    //*************************************************************************/
    // Base cell parameters
    //*************************************************************************/
    // Cancer cell parameters
    PARAM_PROG_DIV_MAX,
    PARAM_CANCER_MOVE_STEPS,
    PARAM_CANCER_MOVE_STEPS_STEM,
    //*************************************************************************/
    // T cell parameters
    PARAM_TCELL_DIV_INTERNAL,
    PARAM_TCELL_DIV_LIMIT,
    PARAM_TCELL_MOVE_STEPS,
    //*************************************************************************/
    // TCD4 cell parameters
    PARAM_TCD4_DIV_INTERNAL,
    PARAM_TCD4_DIV_LIMIT,
    //*************************************************************************/
    // MDSC cell parameters
    PARAM_MDSC_MOVE_STEPS,
    //*************************************************************************/
    // Mac cell parameters
    PARAM_MAC_MOVE_STEPS,
    //*************************************************************************/
    // Fibroblast cell parameters
    PARAM_FIB_MOVE_STEPS,
    //*************************************************************************/
    // Domain initialization parameters
    PARAM_DOMAIN_PDE_WARMUP_SUBSTEPS,    // PDE-only substeps before first ABM step
    //*************************************************************************/
    // Molecular parameters

    GPU_PARAM_INT_COUNT
};

enum GPUParamBool {
    // Environment parameters
    //*************************************************************************/
    // Drug parameters
    PARAM_NIVO_ON,
    PARAM_IPI_ON,
    PARAM_CABO_ON,
    PARAM_ENT_ON,
    //*************************************************************************/
    // Base cell parameters
    //*************************************************************************/
    // Cancer cell parameters
    //*************************************************************************/
    // T cell parameters
    //*************************************************************************/
    // TCD4 cell parameters
    //*************************************************************************/
    // MDSC cell parameters
    //*************************************************************************/
    // Molecular parameters

    GPU_PARAM_BOOL_COUNT
};

class GPUParam : public SP_QSP_IO::ParamBase {
public:
    GPUParam();
    ~GPUParam() {}

    // Type-safe accessors
    inline float getFloat(GPUParamFloat idx) const {
        if (idx >= GPU_PARAM_FLOAT_COUNT) return 0.0f;
        return static_cast<float>(_paramFloat[idx]);
    }
    inline int getInt(GPUParamInt idx) const {
        if (idx >= GPU_PARAM_INT_COUNT) return 0;
        return _paramInt[idx];
    }
    inline bool getBool(GPUParamBool idx) const {
        if (idx >= GPU_PARAM_BOOL_COUNT) return false;
        return _paramBool[idx];
    }

    // Populate FLAMEGPU environment from loaded parameters
    void populateFlameGPUEnvironment(flamegpu::EnvironmentDescription& env) const;

private:
    virtual void setupParam() override;
    virtual void processInternalParams() override;
};

} // namespace PDAC
#endif
