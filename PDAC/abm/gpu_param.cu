#include "gpu_param.h"
#include "flamegpu/flamegpu.h"
#include <iostream>

namespace PDAC {

// XML path mappings - maps enum index to XML path, units, and validation rules
// Format: {XML_path, units, validation_type}
// validation_type: "pr" = positive real , "pos" = positive integer, "" = unconstrained
const char* _gpu_param_description[][3] = {
    ////////////////////////////////////////////////////////////////////////////
    // Float parameters (must match GPUParamFloat enum order)
    // Environment parameters
    {"Param.ABM.Environment.SecPerSlice", "seconds", "pos"},       // PARAM_SEC_PER_SLICE
    {"Param.ABM.Environment.recSiteFactor", "", "pos"},            // PARAM_REC_SITE_FACTOR
    {"Param.ABM.Environment.adhSiteDens", "", "pos"},           // PARAM_ADH_SITE_DENSITY
    {"Param.QSP.simulation.weight_qsp","","pr"},
    //*************************************************************************/
    // Drug parameters
    {"Param.ABM.Pharmacokinetics.nivoDoseIntervalTime", "", "pr"},  // PARAM_NIVO_DOSE_INTERVAL_TIME
    {"Param.ABM.Pharmacokinetics.nivoDose", "", "pr"},              // PARAM_NIVO_DOSE
    {"Param.ABM.Pharmacokinetics.ipiDoseIntervalTime", "", "pr"},   // PARAM_IPI_DOSE_INTERVAL_TIME
    {"Param.ABM.Pharmacokinetics.ipiDose", "", "pr"},               // PARAM_IPI_DOSE
    {"Param.ABM.Pharmacokinetics.caboDoseIntervalTime", "", "pr"},  // PARAM_CABO_DOSE_INTERVAL_TIME
    {"Param.ABM.Pharmacokinetics.caboDose", "", "pr"},              // PARAM_CABO_DOSE
    //*************************************************************************/
    // Base cell parameters
    {"Param.ABM.cell.PDL1_th", "", "pr"},          // PARAM_PDL1_TH
    {"Param.ABM.cell.IFNg_PDL1_half", "", "pr"},   // PARAM_IFNG_PDL1_HALF
    {"Param.ABM.cell.IFNg_PDL1_n", "", "pr"},      // PARAM_IFNG_N
    {"Param.ABM.cell.PDL1_halflife", "", "pr"},    // PARAM_PDL1_HALFLIFE
    //*************************************************************************/
    // Cancer cell parameters
    {"Param.ABM.CancerCell.progGrowthRate", "", "pr"},        // PARAM_PROG_GROWTH_RATE
    {"Param.ABM.CancerCell.senescentDeathRate", "", "pr"},    // PARAM_SEN_DEATH_RATE
    {"Param.ABM.CancerCell.asymmetricDivProb", "", "pr"},     // PARAM_ASYM_DIV_PROB
    {"Param.ABM.CancerCell.IFNgUptake", "", "pr"},            // PARAM_CANCER_IFNG_UPTAKE
    {"Param.ABM.CancerCell.hypoxia_th", "", "pr"},            // PARAM_CANCER_HYPOXIA_TH
    {"Param.ABM.CancerCell.density_csc", "", "pr"},           // PARAM_DENSITY_STEM
    {"Param.ABM.CancerCell.Tkill_scaler", "", "pos"},         // PARAM_TKILL_SCALAR
    {"Param.ABM.CancerCell.mincc", "", "pos"},                // PARAM_MIN_CC
    {"Param.ABM.CancerCell.C1_CD47", "", "pos"},              // PARAM_C1_CD47
    //*************************************************************************/
    // T cell parameters
    {"Param.ABM.TCell.IFNg_recruit_Half", "", "pr"},   // PARAM_TCELL_IFNG_RECRUIT_HALF
    {"Param.ABM.TCell.lifespanSD", "", "pos"},        // PARAM_TCELL_LIFESPAN_SD
    {"Param.ABM.TCell.IL2_release_time", "", "pos"},  // PARAM_TCELL_IL2_RELEASE_TIME
    {"Param.ABM.TCell.IL2_prolif_th", "", "pos"},     // PARAM_TCELL_IL2_PROLIF_TH
    {"Param.ABM.TCell.IFNg_release_time", "", "pos"}, // PARAM_TCELL_IFNG_RELEASE_TIME
    {"Param.ABM.TCell.IFNg_recruit_Half", "", "pos"}, 
    //*************************************************************************/
    // TCD4 cell parameters
    {"Param.ABM.TCD4.TGFB_release_time", "", "pos"},  // PARAM_TCD4_TGFB_RELEASE_TIME
    //*************************************************************************/
    // MDSC cell parameters
    {"Param.ABM.MDSC.lifespanSD", "", "pos"},  // PARAM_MDSC_LIFESPAN_SD
    //*************************************************************************/
    // Vas cell parameters
    {"Param.ABM.Vas.maxPerVoxel", "", "pos"},  // 
    {"Param.ABM.Vas.vas_50", "", "pr"},  // 
    {"Param.ABM.Vas.O2_conc", "", "pr"},  // 
    {"Param.ABM.Vas.Rc", "", "pr"},  // 
    {"Param.ABM.Vas.sigma", "", "pr"},  // 
    {"Param.ABM.Vas.ref_vas_frac", "", "pr"},  // 
    {"Param.ABM.Vas.init_density", "", "pr"},  // 
    {"Param.ABM.Vas.tumble", "", "pr"},  // 
    {"Param.ABM.Vas.delta", "", "pr"},  // 
    {"Param.ABM.Vas.branch_prob", "", "pr"},  // 
    {"Param.ABM.Vas.min_neighbor", "", "pos"},  // 
    //*************************************************************************/
    // Molecular parameters
    {"Param.Molecular.biofvm.IFNg.diffusivity", "", "pr"},  // PARAM_IFNG_DIFFUSIVITY
    {"Param.Molecular.biofvm.IL_2.diffusivity", "", "pr"},  // PARAM_IL2_DIFFUSIVITY
    {"Param.Molecular.biofvm.CCL2.diffusivity", "", "pr"},  // PARAM_CCL2_DIFFUSIVITY
    {"Param.Molecular.biofvm.ArgI.diffusivity", "", "pr"},  // PARAM_ARGI_DIFFUSIVITY
    {"Param.Molecular.biofvm.NO.diffusivity", "", "pr"},    // PARAM_NO_DIFFUSIVITY
    {"Param.Molecular.biofvm.TGFB.diffusivity", "", "pr"},  // PARAM_TGFB_DIFFUSIVITY
    {"Param.Molecular.biofvm.IL10.diffusivity", "", "pr"},  // PARAM_IL10_DIFFUSIVITY
    {"Param.Molecular.biofvm.IL12.diffusivity", "", "pr"},  // PARAM_IL12_DIFFUSIVITY
    {"Param.Molecular.biofvm.VEGFA.diffusivity", "", "pr"}, // PARAM_VEGFA_DIFFUSIVITY
    {"Param.Molecular.biofvm.O2.diffusivity", "", "pr"},    // PARAM_O2_DIFFUSIVITY

    {"Param.Molecular.biofvm.IFNg.decayRate", "", "pr"},    // PARAM_IFNG_DECAY_RATE
    {"Param.Molecular.biofvm.IL_2.decayRate", "", "pr"},    // PARAM_IL2_DECAY_RATE
    {"Param.Molecular.biofvm.CCL2.decayRate", "", "pr"},    // PARAM_CCL2_DECAY_RATE
    {"Param.Molecular.biofvm.ArgI.decayRate", "", "pr"},    // PARAM_ARGI_DECAY_RATE
    {"Param.Molecular.biofvm.NO.decayRate", "", "pr"},      // PARAM_NO_DECAY_RATE
    {"Param.Molecular.biofvm.TGFB.decayRate", "", "pr"},    // PARAM_TGFB_DECAY_RATE
    {"Param.Molecular.biofvm.IL10.decayRate", "", "pr"},    // PARAM_IL10_DECAY_RATE
    {"Param.Molecular.biofvm.IL12.decayRate", "", "pr"},    // PARAM_IL12_DECAY_RATE
    {"Param.Molecular.biofvm.VEGFA.decayRate", "", "pr"},   // PARAM_VEGFA_DECAY_RATE
    {"Param.Molecular.biofvm.O2.decayRate", "", "pr"},      // PARAM_O2_DECAY_RATE

    {"Param.Molecular.biofvm.IFNg.release", "", "pr"},       // PARAM_IFNG_RELEASE_RATE
    {"Param.Molecular.biofvm.IL_2.release", "", "pr"},       // PARAM_IL2_RELEASE_RATE
    {"Param.Molecular.biofvm.CCL2.release", "", "pr"},       // PARAM_CCL2_RELEASE_RATE
    {"Param.Molecular.biofvm.ArgI.release", "", "pr"},       // PARAM_ARGI_RELEASE_RATE
    {"Param.Molecular.biofvm.NO.release", "", "pr"},         // PARAM_NO_RELEASE_RATE
    {"Param.Molecular.biofvm.TGFB.release.Treg", "", "pr"},              // PARAM_TREG_TGFB_RELEASE_RATE
    {"Param.Molecular.biofvm.TGFB.release.CancerStem", "", "pr"},        // PARAM_STEM_TGFB_RELEASE_RATE
    {"Param.Molecular.biofvm.TGFB.release.CancerProgenitor", "", "pr"},  // PARAM_PROG_TGFB_RELEASE_RATE
    {"Param.Molecular.biofvm.TGFB.release.Mac", "", "pr"},
    {"Param.Molecular.biofvm.IL10.release.Treg", "", "pr"},              // PARAM_TREG_IL10_RELEASE_RATE
    {"Param.Molecular.biofvm.IL10.release.Mac", "", "pr"},  
    {"Param.Molecular.biofvm.IL12.release", "", "pr"},                   // PARAM_IL12_RELEASE_RATE
    {"Param.Molecular.biofvm.VEGFA.release.CancerStem", "", "pr"},       // PARAM_STEM_VEGFA_RELEASE_RATE
    {"Param.Molecular.biofvm.VEGFA.release.CancerProgenitor", "", "pr"}, // PARAM_PROG_VEGFA_RELEASE_RATE
    {"Param.Molecular.biofvm.VEGFA.release.Mac", "", "pr"},

    {"Param.Molecular.biofvm.IL_2.uptake", "", "pr"},    // PARAM_IL2
    {"Param.Molecular.biofvm.CCL2.uptake", "", "pr"},    // PARAM_CCL2_UPTAKE
    {"Param.Molecular.biofvm.VEGFA.uptake", "", "pr"},   // PARAM_VEGFA_UPTAKE
    {"Param.Molecular.biofvm.O2.uptake", "", "pr"},      // PARAM_O2_UPTAKE

    {"Param.Molecular.biofvm.IFNg.molecularWeight", "", "pr"},  // PARAM_IFNG_MOLECULAR_WEIGHT
    {"Param.Molecular.biofvm.IL_2.molecularWeight", "", "pr"},  // PARAM_IL2_MOLECULAR_WEIGHT
    {"Param.Molecular.biofvm.CCL2.molecularWeight", "", "pr"},  // PARAM_CCL2_MOLECULAR_WEIGHT
    {"Param.Molecular.biofvm.ArgI.molecularWeight", "", "pr"},  // PARAM_ARGI_MOLECULAR_WEIGHT
    {"Param.Molecular.biofvm.NO.molecularWeight", "", "pr"},    // PARAM_NO_MOLECULAR_WEIGHT
    {"Param.Molecular.biofvm.TGFB.molecularWeight", "", "pr"},  // PARAM_TGFB_MOLECULAR_WEIGHT
    {"Param.Molecular.biofvm.IL10.molecularWeight", "", "pr"},  // PARAM_IL10_MOLECULAR_WEIGHT
    {"Param.Molecular.biofvm.IL12.molecularWeight", "", "pr"},  // PARAM_IL12_MOLECULAR_WEIGHT
    {"Param.Molecular.biofvm.VEGFA.molecularWeight", "", "pr"}, // PARAM_VEGFA_MOLECULAR_WEIGHT
    {"Param.Molecular.biofvm.O2.molecularWeight", "", "pr"},    // PARAM_O2_MOLECULAR_WEIGHT

    ////////////////////////////////////////////////////////////////////////////
    // Integer parameters (must match GPUParamInt enum order)
    // Environment parameters
    {"Param.ABM.Environment.ShuffleInterval", "slices", "pos"},    // PARAM_SHUFFLE_INTERVAL
    {"Param.ABM.Environment.gridshiftInterval", "slices", "pos"},  // PARAM_GRID_SHIFT_INTERVAL
    {"Param.ABM.Environment.Tumor.XSize", "microns", "pos"},       // PARAM_X_SIZE
    {"Param.ABM.Environment.Tumor.YSize", "microns", "pos"},       // PARAM_Y_SIZE
    {"Param.ABM.Environment.Tumor.ZSize", "microns", "pos"},       // PARAM_Z_SIZE
    {"Param.ABM.Environment.Tumor.VoxelSize", "microns", "pos"},   // PARAM_VOXEL_SIZE
    {"Param.ABM.Environment.Tumor.nr_T_voxel", "", "pos"},        // PARAM_NR_T_VOXELS
    {"Param.ABM.Environment.Tumor.nr_T_voxel_C", "", "pos"},       // PARAM_NR_T_VOXELS_C
    {"Param.ABM.Environment.Tumor.stem_mode", "", "pos"},           // PARAM_STEM_MODE
    {"Param.Molecular.stepPerSlice", "", "pos"},                   // PARAM_MOLECULAR_STEPS
    //*************************************************************************/
    // Drug parameters
    //*************************************************************************/
    // Base cell parameters
    //*************************************************************************/
    // Cancer cell parameters
    {"Param.ABM.CancerCell.progenitorDivMax", "", "pos"},  // PARAM_PROG_DIV_MAX
    {"Param.ABM.CancerCell.moveSteps", "", "pos"},         // PARAM_CANCER_MOVE_STEPS
    {"Param.ABM.CancerCell.moveSteps_csc", "", "pos"},     // PARAM_CANCER_MOVE_STEPS_STEM
    //*************************************************************************/
    // T cell parameters
    {"Param.ABM.TCell.div_interval", "", "pos"},       // PARAM_TCELL_DIV_INTERNAL
    {"Param.ABM.TCell.div_limit", "", "pos"},                // PARAM_TCELL_DIV_LIMIT
    {"Param.ABM.TCell.moveSteps", "", "pos"},                // PARAM_TCELL_MOVE_STEPS
    //*************************************************************************/
    // TCD4 cell parameters
    {"Param.ABM.TCD4.div_interval", "", "pos"},       // PARAM_TCD4_DIV_INTERNAL
    {"Param.ABM.TCD4.div_limit", "", "pos"},          // PARAM_TCD4_DIV_LIMIT
    //*************************************************************************/
    // MDSC cell parameters
    {"Param.ABM.MDSC.moveSteps", "", "pos"},   // PARAM_MDSC_MOVE_STEPS
    //*************************************************************************/
    // Macrophage cell parameters
    {"Param.ABM.Mac.moveSteps", "", "pos"},   // PARAM_MAC_MOVE_STEPS
    //*************************************************************************/
    // Fibroblast cell parameters
    {"Param.ABM.Fib.moveSteps", "", "pos"},   // PARAM_FIB_MOVE_STEPS
    //*************************************************************************/
    // Molecular parameters

    ////////////////////////////////////////////////////////////////////////////
    // Boolean parameters (must match GPUParamBool enum order)
    // Environment parameters
    //*************************************************************************/
    // Drug parameters
    {"Param.ABM.Pharmacokinetics.nivoOn", "", ""},  // PARAM_NIVO_ON
    {"Param.ABM.Pharmacokinetics.ipiOn", "", ""},   // PARAM_IPI_ON
    {"Param.ABM.Pharmacokinetics.caboOn", "", ""},  // PARAM_CABO_ON
    {"Param.ABM.Pharmacokinetics.entOn", "", ""},   // PARAM_ENT_ON
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
    
};

// Verify that _gpu_param_description has exactly GPU_PARAM_FLOAT_COUNT entries
static_assert(sizeof(_gpu_param_description) / sizeof(_gpu_param_description[0]) == 
        static_cast<int>(GPU_PARAM_FLOAT_COUNT)+static_cast<int>(GPU_PARAM_INT_COUNT)+static_cast<int>(GPU_PARAM_BOOL_COUNT),
        "GPU parameter description array size mismatch");

GPUParam::GPUParam() : ParamBase() {
    setupParam();
}

void GPUParam::setupParam() {
    _paramFloat.resize(GPU_PARAM_FLOAT_COUNT, 0.0);
    _paramInt.resize(GPU_PARAM_INT_COUNT, 0);
    _paramBool.resize(GPU_PARAM_BOOL_COUNT, false);

    // Copy description strings from _gpu_param_description to _paramDesc
    for (size_t i = 0; i < static_cast<int>(GPU_PARAM_FLOAT_COUNT)+
                                static_cast<int>(GPU_PARAM_INT_COUNT)+
                                static_cast<int>(GPU_PARAM_BOOL_COUNT); i++) {
        std::vector<std::string> desc(
            _gpu_param_description[i],
            _gpu_param_description[i] + 3);
        _paramDesc.push_back(desc);
    }
}

void GPUParam::processInternalParams() {
    // No internal parameter processing needed for GPU parameters
    // ParamBase handles validation based on the "pr" and "pos" tags
}

void GPUParam::populateFlameGPUEnvironment(flamegpu::EnvironmentDescription& env) const {
    // Populate FLAMEGPU environment with all GPU parameters loaded from XML
    //ENVIRONMENT PARAMETERS////////////////////////////////////////////////////////////
    env.newProperty<float>("PARAM_SEC_PER_SLICE", getFloat(PARAM_SEC_PER_SLICE));
    env.newProperty<float>("PARAM_REC_SITE_FACTOR", getFloat(PARAM_REC_SITE_FACTOR));
    env.newProperty<float>("PARAM_ADH_SITE_DENSITY", getFloat(PARAM_ADH_SITE_DENSITY));
    env.newProperty<int>("PARAM_SHUFFLE_INTERVAL", getInt(PARAM_SHUFFLE_INTERVAL));
    env.newProperty<int>("PARAM_GRID_SHIFT_INTERVAL", getInt(PARAM_GRID_SHIFT_INTERVAL));
    env.newProperty<int>("PARAM_X_SIZE", getInt(PARAM_X_SIZE));
    env.newProperty<int>("PARAM_Y_SIZE", getInt(PARAM_Y_SIZE));
    env.newProperty<int>("PARAM_Z_SIZE", getInt(PARAM_Z_SIZE));
    env.newProperty<int>("PARAM_VOXEL_SIZE", getInt(PARAM_VOXEL_SIZE));
    env.newProperty<int>("PARAM_NR_T_VOXELS", getInt(PARAM_NR_T_VOXELS));
    env.newProperty<int>("PARAM_NR_T_VOXELS_C", getInt(PARAM_NR_T_VOXELS_C));
    env.newProperty<int>("PARAM_STEM_MODE", getInt(PARAM_STEM_MODE));
    env.newProperty<int>("PARAM_MOLECULAR_STEPS", getInt(PARAM_MOLECULAR_STEPS));
    env.newProperty<float>("PARAM_WEIGHT_QSP", getFloat(PARAM_WEIGHT_QSP));

    env.newProperty<float>("PARAM_GRID_SIZE", static_cast<float>(getInt(PARAM_X_SIZE) 
                                            * getInt(PARAM_Y_SIZE) 
                                            * getInt(PARAM_Z_SIZE)));

    //DRUG PARAMETERS///////////////////////////////////////////////////////////////////
    env.newProperty<float>("PARAM_NIVO_DOSE_INTERVAL_TIME", getFloat(PARAM_NIVO_DOSE_INTERVAL_TIME));
    env.newProperty<float>("PARAM_NIVO_DOSE", getFloat(PARAM_NIVO_DOSE));
    env.newProperty<float>("PARAM_IPI_DOSE_INTERVAL_TIME", getFloat(PARAM_IPI_DOSE_INTERVAL_TIME));
    env.newProperty<float>("PARAM_IPI_DOSE", getFloat(PARAM_IPI_DOSE));
    env.newProperty<float>("PARAM_CABO_DOSE_INTERVAL_TIME", getFloat(PARAM_CABO_DOSE_INTERVAL_TIME));
    env.newProperty<float>("PARAM_CABO_DOSE", getFloat(PARAM_CABO_DOSE));
    env.newProperty<bool>("PARAM_NIVO_ON", getBool(PARAM_NIVO_ON));
    env.newProperty<bool>("PARAM_IPI_ON", getBool(PARAM_IPI_ON));
    env.newProperty<bool>("PARAM_CABO_ON", getBool(PARAM_CABO_ON));
    env.newProperty<bool>("PARAM_ENT_ON", getBool(PARAM_ENT_ON));   
    //BASE CELL PARAMETERS
    env.newProperty<float>("PARAM_PDL1_TH", getFloat(PARAM_PDL1_TH));
    env.newProperty<float>("PARAM_IFNG_PDL1_HALF", getFloat(PARAM_IFNG_PDL1_HALF));
    env.newProperty<float>("PARAM_IFNG_PDL1_N", getFloat(PARAM_IFNG_PDL1_N));
    env.newProperty<float>("PARAM_PDL1_HALF", getFloat(PARAM_PDL1_HALF));
    //CANCER CELL PARAMETERS
    env.newProperty<float>("PARAM_PROG_GROWTH_RATE", getFloat(PARAM_PROG_GROWTH_RATE));
    env.newProperty<float>("PARAM_SEN_DEATH_RATE", getFloat(PARAM_SEN_DEATH_RATE));
    env.newProperty<float>("PARAM_ASYM_DIV_PROB", getFloat(PARAM_ASYM_DIV_PROB));
    env.newProperty<float>("PARAM_CANCER_IFNG_UPTAKE", getFloat(PARAM_CANCER_IFNG_UPTAKE));
    env.newProperty<float>("PARAM_CANCER_HYPOXIA_TH", getFloat(PARAM_CANCER_HYPOXIA_TH));
    env.newProperty<float>("PARAM_DENSITY_STEM", getFloat(PARAM_DENSITY_STEM));
    env.newProperty<int>("PARAM_PROG_DIV_MAX", getInt(PARAM_PROG_DIV_MAX));
    env.newProperty<int>("PARAM_CANCER_MOVE_STEPS", getInt(PARAM_CANCER_MOVE_STEPS));
    env.newProperty<int>("PARAM_CANCER_MOVE_STEPS_STEM", getInt(PARAM_CANCER_MOVE_STEPS_STEM));
    env.newProperty<float>("PARAM_TKILL_SCALAR", getFloat(PARAM_TKILL_SCALAR));
    env.newProperty<float>("PARAM_MIN_CC", getFloat(PARAM_MIN_CC));
    env.newProperty<float>("PARAM_C1_CD47", getFloat(PARAM_C1_CD47));
    //T CELL PARAMETERS
    env.newProperty<float>("PARAM_TCELL_IFNG_RECRUIT_HALF", getFloat(PARAM_TCELL_IFNG_RECRUIT_HALF));
    env.newProperty<float>("PARAM_TCELL_LIFESPAN_SD", getFloat(PARAM_TCELL_LIFESPAN_SD));
    env.newProperty<int>("PARAM_TCELL_DIV_INTERNAL", getInt(PARAM_TCELL_DIV_INTERNAL));
    env.newProperty<int>("PARAM_TCELL_DIV_LIMIT", getInt(PARAM_TCELL_DIV_LIMIT));
    env.newProperty<int>("PARAM_TCELL_MOVE_STEPS", getInt(PARAM_TCELL_MOVE_STEPS));
    env.newProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME", getFloat(PARAM_TCELL_IL2_RELEASE_TIME));
    env.newProperty<float>("PARAM_TCELL_IL2_PROLIF_TH", getFloat(PARAM_TCELL_IL2_PROLIF_TH));
    env.newProperty<float>("PARAM_TCELL_IFNG_RELEASE_TIME", getFloat(PARAM_TCELL_IFNG_RELEASE_TIME));
    env.newProperty<float>("PARAM_TEFF_IFN_EC50", getFloat(PARAM_TEFF_IFN_EC50));
    //TCD4 CELL PARAMETERS
    env.newProperty<int>("PARAM_TCD4_DIV_INTERNAL", getInt(PARAM_TCD4_DIV_INTERNAL));
    env.newProperty<int>("PARAM_TCD4_DIV_LIMIT", getInt(PARAM_TCD4_DIV_LIMIT));
    env.newProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME", getFloat(PARAM_TCD4_TGFB_RELEASE_TIME));
    //MDSC CELL PARAMETERS
    env.newProperty<float>("PARAM_MDSC_LIFESPAN_SD", getFloat(PARAM_MDSC_LIFESPAN_SD));
    env.newProperty<int>("PARAM_MDSC_MOVE_STEPS", getInt(PARAM_MDSC_MOVE_STEPS));
    //MACROPHAGE CELL PARAMETERS
    env.newProperty<int>("PARAM_MAC_MOVE_STEPS", getInt(PARAM_MAC_MOVE_STEPS));
    //FIBROBLAST CELL PARAMETERS
    env.newProperty<int>("PARAM_FIB_MOVE_STEPS", getInt(PARAM_FIB_MOVE_STEPS));
    //VAS CELL PARAMETERS
    env.newProperty<float>("PARAM_VAS_MAXPERVOXEL", getFloat(PARAM_VAS_MAXPERVOXEL));
    env.newProperty<float>("PARAM_VAS_50", getFloat(PARAM_VAS_50));
    env.newProperty<float>("PARAM_VAS_O2_CONC", getFloat(PARAM_VAS_O2_CONC));
    env.newProperty<float>("PARAM_VAS_RC", getFloat(PARAM_VAS_RC));
    env.newProperty<float>("PARAM_VAS_SIGMA", getFloat(PARAM_VAS_SIGMA));
    env.newProperty<float>("PARAM_VAS_FRAC", getFloat(PARAM_VAS_FRAC));
    env.newProperty<float>("PARAM_VAS_INIT_DENS", getFloat(PARAM_VAS_INIT_DENS));
    env.newProperty<float>("PARAM_VAS_TUMBLE", getFloat(PARAM_VAS_TUMBLE));
    env.newProperty<float>("PARAM_VAS_DELTA", getFloat(PARAM_VAS_DELTA));
    env.newProperty<float>("PARAM_VAS_BRANCH_PROB", getFloat(PARAM_VAS_BRANCH_PROB));
    env.newProperty<float>("PARAM_VAS_MIN_NEIGHBOR", getFloat(PARAM_VAS_MIN_NEIGHBOR));
    //MOLECULAR PARAMETERS
    env.newProperty<float>("PARAM_IFNG_DIFFUSIVITY", getFloat(PARAM_IFNG_DIFFUSIVITY));
    env.newProperty<float>("PARAM_IL2_DIFFUSIVITY", getFloat(PARAM_IL2_DIFFUSIVITY));
    env.newProperty<float>("PARAM_CCL2_DIFFUSIVITY", getFloat(PARAM_CCL2_DIFFUSIVITY));
    env.newProperty<float>("PARAM_ARGI_DIFFUSIVITY", getFloat(PARAM_ARGI_DIFFUSIVITY));
    env.newProperty<float>("PARAM_NO_DIFFUSIVITY", getFloat(PARAM_NO_DIFFUSIVITY));
    env.newProperty<float>("PARAM_TGFB_DIFFUSIVITY", getFloat(PARAM_TGFB_DIFFUSIVITY));
    env.newProperty<float>("PARAM_IL10_DIFFUSIVITY", getFloat(PARAM_IL10_DIFFUSIVITY));
    env.newProperty<float>("PARAM_IL12_DIFFUSIVITY", getFloat(PARAM_IL12_DIFFUSIVITY));
    env.newProperty<float>("PARAM_VEGFA_DIFFUSIVITY", getFloat(PARAM_VEGFA_DIFFUSIVITY));
    env.newProperty<float>("PARAM_O2_DIFFUSIVITY", getFloat(PARAM_O2_DIFFUSIVITY));

    env.newProperty<float>("PARAM_IFNG_DECAY_RATE", getFloat(PARAM_IFNG_DECAY_RATE));
    env.newProperty<float>("PARAM_IL2_DECAY_RATE", getFloat(PARAM_IL2_DECAY_RATE));
    env.newProperty<float>("PARAM_CCL2_DECAY_RATE", getFloat(PARAM_CCL2_DECAY_RATE));
    env.newProperty<float>("PARAM_ARGI_DECAY_RATE", getFloat(PARAM_ARGI_DECAY_RATE));
    env.newProperty<float>("PARAM_NO_DECAY_RATE", getFloat(PARAM_NO_DECAY_RATE));
    env.newProperty<float>("PARAM_TGFB_DECAY_RATE", getFloat(PARAM_TGFB_DECAY_RATE));
    env.newProperty<float>("PARAM_IL10_DECAY_RATE", getFloat(PARAM_IL10_DECAY_RATE));
    env.newProperty<float>("PARAM_IL12_DECAY_RATE", getFloat(PARAM_IL12_DECAY_RATE));
    env.newProperty<float>("PARAM_VEGFA_DECAY_RATE", getFloat(PARAM_VEGFA_DECAY_RATE));
    env.newProperty<float>("PARAM_O2_DECAY_RATE", getFloat(PARAM_O2_DECAY_RATE));

    env.newProperty<float>("PARAM_IFNG_RELEASE", getFloat(PARAM_IFNG_RELEASE));              // Cyt TCells
    env.newProperty<float>("PARAM_IL2_RELEASE", getFloat(PARAM_IL2_RELEASE));                // Cyt TCells
    env.newProperty<float>("PARAM_CCL2_RELEASE", getFloat(PARAM_CCL2_RELEASE));              // Cancer
    env.newProperty<float>("PARAM_ARGI_RELEASE", getFloat(PARAM_ARGI_RELEASE));
    env.newProperty<float>("PARAM_NO_RELEASE", getFloat(PARAM_NO_RELEASE));
    env.newProperty<float>("PARAM_TREG_TGFB_RELEASE", getFloat(PARAM_TREG_TGFB_RELEASE));    // TRegs
    env.newProperty<float>("PARAM_STEM_TGFB_RELEASE", getFloat(PARAM_STEM_TGFB_RELEASE));    // Stem Cancer
    env.newProperty<float>("PARAM_PROG_TGFB_RELEASE", getFloat(PARAM_PROG_TGFB_RELEASE));    // Prog Cancer
    env.newProperty<float>("PARAM_MAC_TGFB_RELEASE", getFloat(PARAM_MAC_TGFB_RELEASE)); 
    env.newProperty<float>("PARAM_TREG_IL10_RELEASE", getFloat(PARAM_TREG_IL10_RELEASE));    // TRegs
    env.newProperty<float>("PARAM_MAC_IL10_RELEASE", getFloat(PARAM_MAC_IL10_RELEASE));
    env.newProperty<float>("PARAM_IL12_RELEASE", getFloat(PARAM_IL12_RELEASE));
    env.newProperty<float>("PARAM_STEM_VEGFA_RELEASE", getFloat(PARAM_STEM_VEGFA_RELEASE));  // Stem Cancer
    env.newProperty<float>("PARAM_PROG_VEGFA_RELEASE", getFloat(PARAM_PROG_VEGFA_RELEASE));  // Prog Cancer
    env.newProperty<float>("PARAM_MAC_VEGFA_RELEASE", getFloat(PARAM_MAC_VEGFA_RELEASE));

    env.newProperty<float>("PARAM_IL2_UPTAKE", getFloat(PARAM_IL2_UPTAKE));      // Cyt TCells
    env.newProperty<float>("PARAM_CCL2_UPTAKE", getFloat(PARAM_CCL2_UPTAKE));
    env.newProperty<float>("PARAM_VEGFA_UPTAKE", getFloat(PARAM_VEGFA_UPTAKE));
    env.newProperty<float>("PARAM_O2_UPTAKE", getFloat(PARAM_O2_UPTAKE));

    env.newProperty<float>("PARAM_IFNG_MOLECULAR_WEIGHT", getFloat(PARAM_IFNG_MOLECULAR_WEIGHT));
    env.newProperty<float>("PARAM_IL2_MOLECULAR_WEIGHT", getFloat(PARAM_IL2_MOLECULAR_WEIGHT));
    env.newProperty<float>("PARAM_CCL2_MOLECULAR_WEIGHT", getFloat(PARAM_CCL2_MOLECULAR_WEIGHT));
    env.newProperty<float>("PARAM_ARGI_MOLECULAR_WEIGHT", getFloat(PARAM_ARGI_MOLECULAR_WEIGHT));
    env.newProperty<float>("PARAM_NO_MOLECULAR_WEIGHT", getFloat(PARAM_NO_MOLECULAR_WEIGHT));
    env.newProperty<float>("PARAM_TGFB_MOLECULAR_WEIGHT", getFloat(PARAM_TGFB_MOLECULAR_WEIGHT));
    env.newProperty<float>("PARAM_IL10_MOLECULAR_WEIGHT", getFloat(PARAM_IL10_MOLECULAR_WEIGHT));
    env.newProperty<float>("PARAM_IL12_MOLECULAR_WEIGHT", getFloat(PARAM_IL12_MOLECULAR_WEIGHT));
    env.newProperty<float>("PARAM_VEGFA_MOLECULAR_WEIGHT", getFloat(PARAM_VEGFA_MOLECULAR_WEIGHT));
    env.newProperty<float>("PARAM_O2_MOLECULAR_WEIGHT", getFloat(PARAM_O2_MOLECULAR_WEIGHT));

    std::cout << "Populated " << (static_cast<int>(GPU_PARAM_FLOAT_COUNT) + static_cast<int>(GPU_PARAM_INT_COUNT) + static_cast<int>(GPU_PARAM_BOOL_COUNT))
              << " parameters into FLAMEGPU environment from XML" << std::endl;
}

} // namespace PDAC
