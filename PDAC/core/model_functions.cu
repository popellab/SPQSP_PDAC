#include "flamegpu/flamegpu.h"
#include "../core/common.cuh"
#include "../qsp/LymphCentral_wrapper.h"
#include "../qsp/ode/QSP_enum.h"


#define QP(x) CancerVCT::ODE_system::get_class_param(x)

#define AVOGADROS 6.022140857E23 
#define PI 3.1415926525897932384626
static int SEC_PER_DAY = 86400;
static int HOUR_PER_DAY = 24;

namespace PDAC {

void set_internal_params(flamegpu::ModelDescription& model, const PDAC::LymphCentralWrapper& lymph){
    flamegpu::EnvironmentDescription env = model.Environment();

    env.newProperty<float>("AVOGADROS", AVOGADROS);

    env.newProperty<float>("PARAM_VOXEL_SIZE_CM", 
        env.getProperty<int>("PARAM_VOXEL_SIZE") / 1e4f);

    env.newProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE", 
                    env.getProperty<float>("PARAM_TCELL_LIFESPAN_SD") 
                    / env.getProperty<float>("PARAM_SEC_PER_SLICE")
                    * SEC_PER_DAY);

    // Volume params
    // cell (need to be 1,  not in mole)
    env.newProperty<float>("PARAM_CELL", QP(CancerVCT::P_cell) * AVOGADROS);

    // volume of cancer cells, vol_cell
    env.newProperty<float>("PARAM_V_CELL", QP(CancerVCT::P_vol_cell));

    // volume of T cells, vol_Tcell
    env.newProperty<float>("PARAM_V_TCELL", QP(CancerVCT::P_vol_Tcell));

    // initial tumor volume calculated by initial tumor diameter.
    env.newProperty<float>("PARAM_INIT_TUM_VOL", 
                    (PI * std::pow(QP(CancerVCT::P_initial_tumour_diameter),
                     3)) / 6);

    env.newProperty<float>("PARAM_INIT_TUM_DIAM", QP(CancerVCT::P_initial_tumour_diameter));
	
    //unit: nanomolarity (1e-9 mole/L) -> ng/ml; conversion factor: 1e9 (mole/L to mole/m^3), 1e6 (m^3 to ml)
    env.newProperty<float>("PARAM_MDSC_EC50_CCL2_REC",
                    QP(CancerVCT::P_CCL2_50) 
                    * env.getProperty<float>("PARAM_CCL2_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);

    // half maximal inhibitory concentration of NO on inhibition of CD8+ T cell cytotoxic activity (ng/ml)
    env.newProperty<float>("PARAM_MDSC_IC50_NO_CTL",
                    QP(CancerVCT::P_NO_50_Teff) 
                    * env.getProperty<float>("PARAM_NO_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);

    // half maximal inhibitory concentration of Arg I on inhibition of CD8+ T cell cytotoxic activity (ng/ml) 
    env.newProperty<float>("PARAM_MDSC_IC50_ArgI_CTL",
                    QP(CancerVCT::P_ArgI_50_Teff) 
                    * env.getProperty<float>("PARAM_ARGI_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);

    // half maximal effective concentration of arginase I on Treg expansion (ng/ml) (molecular weights in kDa
    env.newProperty<float>("PARAM_MDSC_EC50_ArgI_Treg",
                    QP(CancerVCT::P_ArgI_50_Treg) 
                    * env.getProperty<float>("PARAM_ARGI_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);

    // Area of synapse
    env.newProperty<float>("PARAM_A_SYN", QP(CancerVCT::P_A_syn));
    // Area of Cancer cell
    env.newProperty<float>("PARAM_A_CELL", QP(CancerVCT::P_A_cell));
    // Area of T cell
    env.newProperty<float>("PARAM_A_TCELL", QP(CancerVCT::P_A_Tcell));
    // Area of Macrophages
    env.newProperty<float>("PARAM_A_MAC", QP(CancerVCT::P_A_Mcell));
    // number of PD1/PDL1 binding for half maximal inhibition, PD1_50
    env.newProperty<float>("PARAM_PD1_PDL1_HALF", QP(CancerVCT::P_PD1_50) 
                    * env.getProperty<float>("PARAM_A_CELL") * 3);

    // total number of PD1 per synapse on T cell = T_PD1_total*A_syn/A_Tcell; T_PD1_total in density (molecule per micrometer^2)
    env.newProperty<float>("PARAM_PD1_SYN", QP(CancerVCT::P_T_PD1_total));  /// env.getProperty<float>("PARAM_A_TCELL");
    // total number of PD1 per synapse on Macrophages = M_PD1_total*A_syn; T_PD1_total in density (molecule per micrometer^2)
    // env.getProperty<float>("PARAM_MAC_PD1_AREA") in molecule / micrometer^2, 1m^2 = 1e12 micrometer^2
    env.newProperty<float>("PARAM_MAC_PD1_SYN", QP(CancerVCT::P_M_PD1_total)); ///	 env.getProperty<float>("PARAM_A_MAC");

    // env.getProperty<float>("PARAM_C1_CD47_SYN") in molecule / micrometer^2, 1m^2 = 1e12 micrometer^2
    env.newProperty<float>("PARAM_C1_CD47_SYN", QP(CancerVCT::P_C_CD47));

    // env.getProperty<float>("PARAM_MAC_SIRPa") in molecule / micrometer^2, mole/m^2 1m^2 = 1e12 micrometer^2
    env.newProperty<float>("PARAM_MAC_SIRPa_SYN", QP(CancerVCT::P_M_SIRPa));

    // Number of PDL1 per synapse = C1_PDL1_total; C1_PDL1_total in density (molecule per micrometer^2)
    env.newProperty<float>("PARAM_PDL1_SYN_MAX", QP(CancerVCT::P_C1_PDL1_base)); 

    // total number of PDL1 per cancer cell = C1_PDL1_total
    env.newProperty<float>("PARAM_PDL1_CELL", QP(CancerVCT::P_C1_PDL1_base));

    // k1 for PDL1-PD1 calculation =  kon_PD1_PDL1 / (koff_PD1_PDL1* A_CELL)
    env.newProperty<float>("PARAM_PDL1_K1", QP(CancerVCT::P_kon_PD1_PDL1) 
                    / (QP(CancerVCT::P_koff_PD1_PDL1) 
                    * env.getProperty<float>("PARAM_A_CELL")));

    // k2 for PDL1-aPD1 calculation = 2* kon_PD1_aPD1 / (koff_PD1_aPD1 * gamma_T_nivo)
    env.newProperty<float>("PARAM_PDL1_K2", 2 * QP(CancerVCT::P_kon_PD1_aPD1) 
                    / (QP(CancerVCT::P_koff_PD1_aPD1) 
                    * QP(CancerVCT::P_gamma_T_aPD1)));

    // k3 for PDL1-PD1 calculation = (Chi_PD1 * kon_PD1_aPD1) / (2 * koff_PD1_aPD1)
    env.newProperty<float>("PARAM_PDL1_K3", QP(CancerVCT::P_Chi_PD1_aPD1) 
                    * QP(CancerVCT::P_kon_PD1_aPD1) 
                    / (2 * QP(CancerVCT::P_koff_PD1_aPD1)));
    // hill coefficient
    env.newProperty<float>("PARAM_N_PD1_PDL1", QP(CancerVCT::P_n_PD1));

    // Unbinding rate between CTLA4 and ipi
    //env.newProperty<float>("PARAM_KOFF_CTLA4_IPI", QP(CancerVCT::P_r_PDL1_IFNg));
    env.newProperty<float>("PARAM_KOFF_CTLA4_IPI", 6.96e-06);

    // Volume fraction available to ipi in tumor compartment
    //env.newProperty<float>("PARAM_GAMMA_T_IPI", QP(CancerVCT::P_q_LD_aPD1));
    env.newProperty<float>("PARAM_GAMMA_T_IPI", 0.718);

    // Antibody cross-arm binding efficiency  that also includes the conversion of kon from 3D to 2D (estimated)
    //env.newProperty<float>("PARAM_CHI_CTLA4_IPI", QP(CancerVCT::P_kon_PD1_PDL2));
    env.newProperty<float>("PARAM_CHI_CTLA4_IPI", 3);

    // CTLA4 occupancy for half-maximal Treg inactivation by macrophages (estimated)  = CTLA_50 * A_treg
    env.newProperty<float>("PARAM_TREG_CTLA4_50", QP(CancerVCT::P_koff_PD1_aPD1) 
                    * QP(CancerVCT::P_k_xP1_deg));

    // total number of CTLA4 per synapse = Treg_CTLA4_total * A_treg; T_CTLA_total in density (molecule per micrometer^2)
    env.newProperty<float>("PARAM_CTLA4_TREG", QP(CancerVCT::P_q_LN_aCTLA4) 
                    * QP(CancerVCT::P_k_xP1_deg));

    // Anti-CTLA4 ADCC (antibody-dependent cellular cytotoxicity) rate of Treg (Richards 2008, PMID: 18723496)
    //env.newProperty<float>("PARAM_K_ADCC", QP(CancerVCT::P_kon_CTLA4_aCTLA4) * env.getProperty<float>("PARAM_SEC_PER_SLICE"));
    env.newProperty<float>("PARAM_K_ADCC", 0.1 
                    * env.getProperty<float>("PARAM_SEC_PER_SLICE"));

    // Parameters calculated from QSP parameter values
    double t_step_sec = env.getProperty<float>("PARAM_SEC_PER_SLICE");

    // Update cabozantinib module from QSP model (values are temperary)
    // IC50	for receptors inhibited by cabozantinib
    env.newProperty<float>("PARAM_IC50_AXL", QP(CancerVCT::P_IC50_AXL));
    env.newProperty<float>("PARAM_IC50_VEGFR2", QP(CancerVCT::P_IC50_VEGFR2));
    env.newProperty<float>("PARAM_IC50_MET", QP(CancerVCT::P_IC50_MET));
    env.newProperty<float>("PARAM_IC50_RET", QP(CancerVCT::P_IC50_RET));

    // theraputic effects parameters elicited by cabozantinib
    env.newProperty<float>("PARAM_LAMBDA_C_CABO", QP(CancerVCT::P_k_K_cabo));

    // poisson process is calculated in TCD4.cpp, here is the rate of the process
    // T cell killing of Cancer cell
    // QP(CancerVCT::P_N_costim): k_C_death_by_T (day^-1, sec^-1 internal)
    // Becuase the ABM contain modules that QSP, which contains extra immunesuppresive modules
    env.newProperty<float>("PARAM_ESCAPE_BASE", 
                    std::exp(-t_step_sec * QP(CancerVCT::P_k_C_T1)));
    // Macrophage killing of Cancer cell
    //env.getProperty<float>("PARAM_MAC_K_M1_PHAGO") k_C_death_by_Macrophage (day^-1, sec^-1 internal)
    // Becuase the ABM contain modules that QSP, which contains extra immunesuppresive modules
    // 5 is a arbitary coefficient to compensate immuno-suppresive module not present in QSP model
    env.newProperty<float>("PARAM_ESCAPE_MAC_BASE", 
                    std::exp(-t_step_sec * QP(CancerVCT::P_k_M1_phago)));

    // T cell exhaustion from PDL1; 
    env.newProperty<float>("PARAM_EXHUAST_BASE_PDL1", 
                    std::exp(-t_step_sec * QP(CancerVCT::P_k_T1)));

    // T cell exhaustion from Treg inhibition, k_Treg;
    env.newProperty<float>("PARAM_EXHUAST_BASE_TREG", 
                    std::exp(-t_step_sec * QP(CancerVCT::P_k_Treg)));

    // rate of Th to Treg transformation , units: 1/days -> 1/timestep
    // poisson process is calculated in TCD4.cpp, here is the rate of the process
    env.newProperty<float>("PARAM_K_TH_TREG", 
                    t_step_sec * QP(CancerVCT::P_k_Th_Treg));
    // Macrophage module related parameter covnersion
    // Rate of M1 to M2 macrophage polarization, units: 1/days -> 1/timestep

    env.newProperty<float>("PARAM_MAC_M2_POL", 
                    t_step_sec * QP(CancerVCT::P_k_M2_pol));
    // Rate of M2 to M1 macrophage polarization, units: 1/days -> 1/timestep
    env.newProperty<float>("PARAM_MAC_M1_POL", 
                    t_step_sec * QP(CancerVCT::P_k_M1_pol));
    //unit: nanomolarity (1e-9 mole/L) -> ng/ml; conversion factor: 1e9 (mole/L to mole/m^3), 1e6 (m^3 to ml)
    env.newProperty<float>("PARAM_MAC_EC50_CCL2_REC", QP(CancerVCT::P_CCL2_50) 
    * env.getProperty<float>("PARAM_CCL2_MOLECULAR_WEIGHT") * 1e3 * 1e9 / 1e6);
    // TGFb_50 Half-Maximal TGFb level for Th-to-Treg differentiation / chemoresistance development / M1-to-M2 polarization
    //unit: nanomolarity (1e-9 mole/L) -> ng/ml; conversion factor: 1e9 (mole/L to mole/m^3), 1e6 (m^3 to ml)
    env.newProperty<float>("PARAM_MAC_TGFB_EC50", QP(CancerVCT::P_TGFb_50) 
                    * env.getProperty<float>("PARAM_TGFB_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);
    //unit: nanomolarity (1e-9 mole/L) -> ng/ml
    env.newProperty<float>("PARAM_TEFF_TGFB_EC50", QP(CancerVCT::P_TGFb_50_Teff) 
                    * env.getProperty<float>("PARAM_TGFB_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);
    //unit: picomolarity (1e-12 mole/L) -> ng/ml
    env.newProperty<float>("PARAM_MAC_IL_10_EC50", QP(CancerVCT::P_IL10_50) 
                    * env.getProperty<float>("PARAM_IL10_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);
    //unit: picomolarity (1e-12 mole/L) -> ng/ml
    env.newProperty<float>("PARAM_MAC_IL_10_HALF_PHAGO", QP(CancerVCT::P_IL10_50_phago) 
                    * env.getProperty<float>("PARAM_IL10_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);
    //unit: picomolarity (1e-12 mole/L) -> ng/ml
    env.newProperty<float>("PARAM_MAC_IFN_G_EC50", QP(CancerVCT::P_IFNg_50) 
                    * env.getProperty<float>("PARAM_IFNG_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);
    //unit: picomolarity (1e-12 mole/L) -> ng/ml
    env.newProperty<float>("PARAM_MAC_IL_12_EC50", QP(CancerVCT::P_IL12_50) 
                    * env.getProperty<float>("PARAM_IL12_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);
    //unit: moles/m^2
    env.newProperty<float>("PARAM_MAC_SIRPa_HALF", QP(CancerVCT::P_SIRPa_50));
    // 1/(micromolarity*minute*nanometer) ->   0.001 (micromolarity to mole/m^3), 60 (minute to second), 1e-9 (nanometer to meter)
    env.newProperty<float>("PARAM_KON_SIRPa_CD47", QP(CancerVCT::P_kon_CD47_SIRPa));
    // 1/minute
    env.newProperty<float>("PARAM_KOFF_SIRPa_CD47", QP(CancerVCT::P_koff_CD47_SIRPa));
    // hill coefficient
    env.newProperty<float>("PARAM_N_SIRPa_CD47", QP(CancerVCT::P_n_SIRPa));

    // ECM secretion per time step for both fibroblast and caf (parameter inputs are ng/s) 6970071747.68519 is the conversion factor to nanomole/cell/day
    env.newProperty<float>("PARAM_FIB_ECM_RELEASE_FIB", QP(CancerVCT::P_k_ECM_fib_sec) 
                    / 6970071747.68519);
    env.newProperty<float>("PARAM_FIB_ECM_RELEASE_CAF", QP(CancerVCT::P_k_ECM_CAF_sec) 
                    / 6970071747.68519);

    //fibroblast activation rate due to tgfb (1/day)
    env.newProperty<float>("PARAM_FIB_CAF_ACTIVATION", QP(CancerVCT::P_k_caf_tran) 
                    * t_step_sec);
    env.newProperty<float>("PARAM_FIB_CAF_EC50", QP(CancerVCT::P_TGFb_50) 
                    * env.getProperty<float>("PARAM_TGFB_MOLECULAR_WEIGHT") 
                    * 1e3 * 1e9 / 1e6);
    env.newProperty<float>("PARAM_FIB_ECM_TGFB_FACTOR", 2);

    env.newProperty<float>("PARAM_FIB_ECM_DECAY_RATE", QP(CancerVCT::P_k_ECM_deg));
    // unit: mole/m^3 -> nmole/ml
    env.newProperty<float>("PARAM_FIB_ECM_BASELINE", QP(CancerVCT::P_ECM_base) * 1e3);
    env.newProperty<float>("PARAM_FIB_ECM_SATURATION", QP(CancerVCT::P_ECM_max) * 1e3);
    env.newProperty<float>("PARAM_FIB_ECM_MOT_EC50", QP(CancerVCT::P_ECM_50_T_mot) * 5e3);
    // CAF TGFB secretion rate per cell per timestep (using M2 macrophage rate as proxy)
    // units: mole/cell/s * t_step_sec -> mole/cell/timestep
    env.newProperty<float>("PARAM_FIB_TGFB_RELEASE",
                    QP(CancerVCT::P_k_TGFb_Msec) * t_step_sec
                    * env.getProperty<float>("PARAM_TGFB_MOLECULAR_WEIGHT") * 1e9);
    // mean lifespan of fibroblast in timesteps (use avg of fib and CAF death rates)
    env.newProperty<float>("PARAM_FIB_LIFE_MEAN",
                    1.0f / ((QP(CancerVCT::P_k_fib_death) + QP(CancerVCT::P_k_CAF_death)) / 2.0)
                    / t_step_sec);

    // time for resection
    // env.newProperty<float>("PARAM_RESECT_TIME_STEP", 
    //                 env.getProperty<float>("PARAM_QSP_T_RESECTION") 
    //                 * SEC_PER_DAY / t_step_sec);

    //The number of adhesion site per voxel is:
    double site_per_voxel = env.getProperty<float>("PARAM_ADH_SITE_DENSITY") 
                                    * std::pow(env.getProperty<int>("PARAM_VOXEL_SIZE"), 3);
    //number of adhesion sites needed to recruit a single cell (becoming a recruitment port)
    double site_per_port = env.getProperty<float>("PARAM_REC_SITE_FACTOR");
    //how many port per voxel have
    env.newProperty<float>("PARAM_REC_PORT", site_per_voxel / site_per_port);
    /*When calculating recruitment probability:
    */
    double w = env.getProperty<float>("PARAM_WEIGHT_QSP");
    // Teff -> k (1/mol) // p = k (1/mol) * Cent.T (mol), q_T1_T_in
    //env.newProperty<float>("PARAM_TEFF_RECRUIT_K", QP(CancerVCT::P_q_nT0_P_in) * site_per_port * t_step_sec  / w / env.getProperty<float>("PARAM_ADH_SITE_DENSITY"));

    env.newProperty<float>("PARAM_TEFF_RECRUIT_K", QP(CancerVCT::P_q_T1_T_in) 
                    * t_step_sec * AVOGADROS 
                    * std::pow(env.getProperty<int>("PARAM_VOXEL_SIZE") / 1e6, 3) 
                    * env.getProperty<float>("PARAM_REC_PORT"));
    // TCD4 -> k (1/mol) // p = k (1/mol) * Cent.T (mol), q_T0_T_in
    //env.newProperty<float>("PARAM_TCD4_RECRUIT_K", QP(CancerVCT::P_k_T1_death) * site_per_port * t_step_sec  / w / env.getProperty<float>("PARAM_ADH_SITE_DENSITY"));
    env.newProperty<float>("PARAM_TREG_RECRUIT_K", QP(CancerVCT::P_q_T0_T_in) 
                    * t_step_sec * AVOGADROS 
                    * std::pow(env.getProperty<int>("PARAM_VOXEL_SIZE") / 1e6, 3) 
                    * env.getProperty<float>("PARAM_REC_PORT"));
    env.newProperty<float>("PARAM_TH_RECRUIT_K", QP(CancerVCT::P_q_T0_T_in) 
                    * t_step_sec * AVOGADROS 
                    * std::pow(env.getProperty<int>("PARAM_VOXEL_SIZE") / 1e6, 3) 
                    * env.getProperty<float>("PARAM_REC_PORT"));
    env.newProperty<float>("PARAM_MDSC_RECRUIT_K", QP(CancerVCT::P_k_MDSC_rec) 
                    * t_step_sec * AVOGADROS 
                    * std::pow(env.getProperty<int>("PARAM_VOXEL_SIZE") / 1e6, 3));
    env.newProperty<float>("PARAM_MAC_RECRUIT_K", QP(CancerVCT::P_k_Mac_rec) 
                    * t_step_sec * AVOGADROS 
                    * std::pow(env.getProperty<int>("PARAM_VOXEL_SIZE") / 1e6, 3));

    // APC -> k (m^3/mol) // p = k (m^3/mol) * (APC0_T*V_T-V_T.APC)
    env.newProperty<float>("PARAM_APC_RECRUIT_K", 0);

    // APC density in the tumour
    env.newProperty<float>("PARAM_APC0_T", QP(CancerVCT::P_APC0_T));

    // APC transmigration rate from tumor to lymph node
    env.newProperty<float>("PARAM_APC_TRANSMIG", QP(CancerVCT::P_k_APC_mig));

    // Number of T0 cell Clonality to lymph node
    env.newProperty<float>("PARAM_T0_CLONE", QP(CancerVCT::P_n_T0_clones));

    // Number of T1 cell Clonality to lymph node
    env.newProperty<float>("PARAM_T1_CLONE", QP(CancerVCT::P_n_T1_clones));

    // antigen concentration in cancer cell
    env.newProperty<float>("PARAM_ANTIGEN_PER_CELL", QP(CancerVCT::P_P1_C1));

    // antigen uptake rate by mAPC
    env.newProperty<float>("PARAM_ANTIGEN_UP", QP(CancerVCT::P_k_P0_up));

    // antigen degrdation rate in the tumor
    env.newProperty<float>("PARAM_K_xP_DEG", QP(CancerVCT::P_k_xP0_deg));

    // mean life of Tcell, unit: time step, 
    // which is different from the QSP model.
    env.newProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE", 
                    1 / QP(CancerVCT::P_k_T1_death) / t_step_sec / 5);
    // mean life of TCD4, unit: time step
    env.newProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE", 
                    1 / QP(CancerVCT::P_k_T0_death) / t_step_sec / 5);

    // mean life of MDSC, unit: time step
    env.newProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE", 
                    1 / QP(CancerVCT::P_k_MDSC_death) / t_step_sec);
    // mean life of MAC, unit: time step
    env.newProperty<float>("PARAM_MAC_LIFE_MEAN", 
                    1 / QP(CancerVCT::P_k_Mac_death) / t_step_sec);
    // mean life of APC, unit: time step
    env.newProperty<float>("PARAM_APC_LIFE_MEAN", 
                    1 / QP(CancerVCT::P_k_APC_death) / t_step_sec);

    // Maximum rate of APC maturation
    env.newProperty<float>("PARAM_K_APC_MAT", QP(CancerVCT::P_k_APC_mat));

    // stem cell division rate is calculated from QSP parameter
    // unit: s^-1, k_C1_growth
    double rs = QP(CancerVCT::P_k_C1_growth) 
                    / (1 - env.getProperty<float>("PARAM_ASYM_DIV_PROB"));
    // unit: day^-1
    env.newProperty<float>("PARAM_CSC_GROWTH_RATE", rs * SEC_PER_DAY);

    env.newProperty<float>("PARAM_FLOAT_CANCER_CELL_STEM_DIV_INTERVAL_SLICE",
                    std::log(2)/rs / env.getProperty<float>("PARAM_SEC_PER_SLICE"));

    env.newProperty<float>("PARAM_CANCER_SENESCENT_MEAN_LIFE",
                    1 / env.getProperty<float>("PARAM_SEN_DEATH_RATE") 
                    / env.getProperty<float>("PARAM_SEC_PER_SLICE") * SEC_PER_DAY);

    env.newProperty<float>("PARAM_FLOAT_CANCER_CELL_PROGENITOR_DIV_INTERVAL_SLICE",
                    std::log(2)/ env.getProperty<float>("PARAM_PROG_GROWTH_RATE") 
                    * SEC_PER_DAY 
                    / env.getProperty<float>("PARAM_SEC_PER_SLICE")  + .5);

}

FLAMEGPU_HOST_FUNCTION(update_agent_counts) {
    // Get counts for each agent type
    int cancer_count = FLAMEGPU->agent(AGENT_CANCER_CELL).count();
    int tcell_count = FLAMEGPU->agent(AGENT_TCELL).count();
    int treg_count = FLAMEGPU->agent(AGENT_TREG).count();
    int mdsc_count = FLAMEGPU->agent(AGENT_MDSC).count();

    // Update environment properties
    FLAMEGPU->environment.setProperty<unsigned int>("total_cancer_cells", cancer_count);
    FLAMEGPU->environment.setProperty<unsigned int>("total_tcells", tcell_count);
    FLAMEGPU->environment.setProperty<unsigned int>("total_tregs", treg_count);
    FLAMEGPU->environment.setProperty<unsigned int>("total_mdscs", mdsc_count);
    FLAMEGPU->environment.setProperty<unsigned int>("total_agents",
    cancer_count + tcell_count + treg_count + mdsc_count);
}

FLAMEGPU_HOST_FUNCTION(check_cancer_count_before_movement) {
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    int cancer_count = FLAMEGPU->agent(AGENT_CANCER_CELL).count();

    std::cout << "Step " << step << " - Cancer cells before movement: " << cancer_count << std::endl;

}

FLAMEGPU_HOST_FUNCTION(reset_division_counters) {
    FLAMEGPU->environment.setProperty<unsigned int>("cancer_divide_attempts", 0u);
    FLAMEGPU->environment.setProperty<unsigned int>("cancer_divide_successes", 0u);
    FLAMEGPU->environment.setProperty<unsigned int>("tcell_divide_attempts", 0u);
    FLAMEGPU->environment.setProperty<unsigned int>("tcell_divide_successes", 0u);
    FLAMEGPU->environment.setProperty<unsigned int>("treg_divide_attempts", 0u);
    FLAMEGPU->environment.setProperty<unsigned int>("treg_divide_successes", 0u);
}

FLAMEGPU_HOST_FUNCTION(report_division_statistics) {
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    unsigned int cancer_attempts = FLAMEGPU->environment.getProperty<unsigned int>("cancer_divide_attempts");
    unsigned int cancer_successes = FLAMEGPU->environment.getProperty<unsigned int>("cancer_divide_successes");
    unsigned int tcell_attempts = FLAMEGPU->environment.getProperty<unsigned int>("tcell_divide_attempts");
    unsigned int tcell_successes = FLAMEGPU->environment.getProperty<unsigned int>("tcell_divide_successes");
    unsigned int treg_attempts = FLAMEGPU->environment.getProperty<unsigned int>("treg_divide_attempts");
    unsigned int treg_successes = FLAMEGPU->environment.getProperty<unsigned int>("treg_divide_successes");

    std::cout << "Step " << step << " - Division Stats:"
              << " Cancer(A:" << cancer_attempts << " S:" << cancer_successes << ")"
              << " TCell(A:" << tcell_attempts << " S:" << tcell_successes << ")"
              << " TReg(A:" << treg_attempts << " S:" << treg_successes << ")"
              << std::endl;
}

FLAMEGPU_HOST_FUNCTION(check_cancer_count_after_movement) {
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    int cancer_count = FLAMEGPU->agent(AGENT_CANCER_CELL).count();

    std::cout << "Step " << step << " - Cancer cells after movement: " << cancer_count << std::endl;

}

FLAMEGPU_HOST_FUNCTION(check_voxel_occupancy) {
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    int cancer_count = FLAMEGPU->agent(AGENT_CANCER_CELL).count();
    std::cout << "Step " << step << " - Cancer cell count (before division): " << cancer_count << std::endl;
}

FLAMEGPU_HOST_FUNCTION(check_voxel_packing_after_movement) {
    // Placeholder - the real issue is in execute_move and execute_divide
    // Both need to check MSG_CELL_LOCATION to verify target voxel is physically empty
}

}

