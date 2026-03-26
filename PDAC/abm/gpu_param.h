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
    //*************************************************************************/
    // TCD4 cell parameters
    PARAM_TCD4_LIFESPAN_SD,
    PARAM_TCD4_TGFB_RELEASE_TIME,
    //*************************************************************************/
    // MDSC cell parameters
    PARAM_MDSC_LIFESPAN_SD,
    //*************************************************************************/
    // Macrophage cell parameters

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
    PARAM_CANCER_IL1_RELEASE,         // Cancer cell IL-1 secretion (drives iCAF activation)
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
