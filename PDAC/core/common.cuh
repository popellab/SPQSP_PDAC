#ifndef FLAMEGPU_PDAC_COMMON_CUH
#define FLAMEGPU_PDAC_COMMON_CUH

#include <cstdint>
#include "flamegpu/flamegpu.h"

namespace PDAC {

// Agent type enumeration (matches CPU AgentTypeEnum)
enum AgentType : int {
    AGENT_DUMMY = 0,
    CELL_TYPE_CANCER = 1,
    CELL_TYPE_T = 2,
    CELL_TYPE_TREG = 3,
    CELL_TYPE_MDSC = 4,
    CELL_TYPE_MAC = 5,
    CELL_TYPE_FIB = 6,
    CELL_TYPE_VASCULAR = 7,
    CELL_TYPE_BCELL = 8,
    CELL_TYPE_DC = 9
};

// Cancer cell state enumeration (matches CPU AgentStateEnum)
enum CancerState : int {
    CANCER_STEM = 0,
    CANCER_PROGENITOR = 1,
    CANCER_SENESCENT = 2,
    CANCER_PDL1_POS = 3,
    CANCER_PDL1_NEG = 4,
};

// T cell state enumeration
enum TCellState : int {
    T_CELL_EFF = 0,
    T_CELL_CYT = 1,
    T_CELL_SUPP = 2,
    T_CELL_NAIVE = 3
};

// T reg state enumeration (TH=0 matches HCC convention: TCD4_Th < TCD4_TREG)
enum TCD4State : int {
    TCD4_TH = 0,
    TCD4_TREG = 1,
    TCD4_TFH = 2,
    TCD4_NAIVE = 3,
};

// Macrophage state enumeration (M1/M2 polarization)
enum MacrophageState : int {
    MAC_M1 = 0,          // Pro-inflammatory (IFN-γ activated)
    MAC_M2 = 1,          // Anti-inflammatory (IL-10/TGF-β activated)
    MAC_INTERMEDIATE = 2 // Mixed phenotype
};

// Fibroblast state enumeration
enum FibroblastState : int {
    FIB_QUIESCENT = 0,  // Quiescent fibroblast (tissue maintenance)
    FIB_MYCAF = 1,      // Myofibroblastic CAF (ECM deposition, contractile)
    FIB_ICAF = 2,       // Inflammatory CAF (IL-6, CXCL13 secretion)
    FIB_FRC = 3         // Fibroblastic reticular cell (TLS T-zone stromal, CCL21 source)
};

// Vascular cell cell_states
enum VascularCellState : int {
    VAS_TIP = 0,      // Actively sprouting tip cell
    VAS_STALK = 1,    // Connecting stalk cell
    VAS_PHALANX = 2,  // Mature vessel (O2 secreting)
    VAS_PHALANX_COLLAPSED = 3, // Lumen collapsed by ECM pressure (no O2, recoverable)
    VAS_HEV = 4               // High endothelial venule (boosted immune recruitment, O2 delivery)
};

// B cell state enumeration
enum BCellState : int {
    BCELL_NAIVE = 0,      // Circulating/tissue-resident, searching for antigen
    BCELL_ACTIVATED = 1,  // Antigen-loaded, proliferating, forming GC clusters
    BCELL_PLASMA = 2      // Terminal differentiation, antibody/IL-10 factory
};

// Dendritic cell state enumeration
enum DCState : int {
    DC_IMMATURE = 0,  // Tissue-resident, antigen sampling
    DC_MATURE = 1     // Antigen-loaded, presenting, secreting IL-12/CCL21
};
// Dendritic cell subtype (stored as agent variable "dc_subtype")
enum DCSubtype : int {
    DC_CDC1 = 0,   // cDC1: cross-presents to CD8 T cells, secretes IL-12
    DC_CDC2 = 1    // cDC2: MHC-II presents to CD4/Th/Treg
};

// Message names
constexpr const char* MSG_CELL_LOCATION = "cell_location";
constexpr const char* MSG_INTENT = "intent_message";

// Agent names
constexpr const char* AGENT_CANCER_CELL = "CancerCell";
constexpr const char* AGENT_TCELL = "TCell";
constexpr const char* AGENT_TREG = "TReg";
constexpr const char* AGENT_MDSC = "MDSC";
constexpr const char* AGENT_MACROPHAGE = "Macrophage";
constexpr const char* AGENT_FIBROBLAST = "Fibroblast";
constexpr const char* AGENT_VASCULAR = "VascularCell";
constexpr const char* AGENT_BCELL = "BCell";
constexpr const char* AGENT_DC = "DC";

// Environment property names for grid dimensions
constexpr const char* ENV_GRID_SIZE_X = "grid_size_x";
constexpr const char* ENV_GRID_SIZE_Y = "grid_size_y";
constexpr const char* ENV_GRID_SIZE_Z = "grid_size_z";
constexpr const char* ENV_VOXEL_SIZE = "voxel_size";

constexpr int N_DIVIDE_WAVES = 1;               // Wave-interleaved division rounds (cancer/tcell/treg)
// MAX_FIB_CHAIN_LENGTH removed — fibroblasts are now single-cell agents
// ── Per-step event counters (device_event_counters[], env prop "event_counters_ptr") ──────────────
// Reset each step. Written by agent device functions via atomicAdd.
enum EventCounterIdx : int {
    // Proliferation (daughter cell created by division)
    EVT_PROLIF_CD8_EFF = 0,
    EVT_PROLIF_CD8_CYT,
    EVT_PROLIF_CD8_SUP,
    EVT_PROLIF_TH,
    EVT_PROLIF_TREG,
    EVT_PROLIF_TFH,
    EVT_PROLIF_MDSC,          // 0 — division not implemented
    EVT_PROLIF_CANCER_STEM,
    EVT_PROLIF_CANCER_PROG,
    EVT_PROLIF_CANCER_SEN,    // 0 — senescent cells don't divide
    EVT_PROLIF_MAC_M1,        // 0 — division not implemented
    EVT_PROLIF_MAC_M2,        // 0
    EVT_PROLIF_FIB_QUIESCENT, // 0 — quiescent fibroblasts don't divide
    EVT_PROLIF_FIB_MYCAF,
    EVT_PROLIF_FIB_ICAF,
    EVT_PROLIF_FIB_FRC,
    EVT_PROLIF_VAS_TIP,
    EVT_PROLIF_VAS_PHALANX,   // 0
    EVT_PROLIF_BCELL_NAIVE,
    EVT_PROLIF_BCELL_ACT,
    EVT_PROLIF_BCELL_PLASMA,  // 0 — plasma cells don't divide
    // Deaths (all causes combined, by cell type/state)
    EVT_DEATH_CD8_EFF,
    EVT_DEATH_CD8_CYT,
    EVT_DEATH_CD8_SUP,
    EVT_DEATH_TH,
    EVT_DEATH_TREG,
    EVT_DEATH_TFH,
    EVT_DEATH_MDSC,
    EVT_DEATH_CANCER_STEM,
    EVT_DEATH_CANCER_PROG,
    EVT_DEATH_CANCER_SEN,
    EVT_DEATH_MAC_M1,
    EVT_DEATH_MAC_M2,
    EVT_DEATH_FIB_QUIESCENT,
    EVT_DEATH_FIB_MYCAF,
    EVT_DEATH_FIB_ICAF,
    EVT_DEATH_FIB_FRC,
    EVT_DEATH_VAS_TIP,
    EVT_DEATH_VAS_PHALANX,
    EVT_DEATH_VAS_COLLAPSED,
    EVT_DEATH_VAS_HEV,
    EVT_DEATH_BCELL_NAIVE,
    EVT_DEATH_BCELL_ACT,
    EVT_DEATH_BCELL_PLASMA,
    EVT_DEATH_DC_CDC1_IMMATURE,
    EVT_DEATH_DC_CDC1_MATURE,
    EVT_DEATH_DC_CDC2_IMMATURE,
    EVT_DEATH_DC_CDC2_MATURE,
    EVT_DEATH_CD8_NAIVE,
    EVT_DEATH_TCD4_NAIVE,
    // DC priming events
    EVT_PRIME_CD8,
    EVT_PRIME_TH,
    EVT_PRIME_TREG,
    // PDL1 expression numerator (divide by total cancer for PDL1_frac)
    EVT_PDL1_COUNT,
    ABM_EVENT_COUNTER_SIZE    // = 31
};

// ── Per-step population counts by state (device_state_counters[], env prop "state_counters_ptr") ─
// Accumulated during final_broadcast_* functions (start-of-step snapshot). Reset each step.
enum StateCounterIdx : int {
    SC_CANCER_STEM = 0,
    SC_CANCER_PROG,
    SC_CANCER_SEN,
    SC_CD8_EFF,
    SC_CD8_CYT,
    SC_CD8_SUP,
    SC_CD8_NAIVE,
    SC_TH,
    SC_TREG,
    SC_TFH,
    SC_TCD4_NAIVE,
    SC_MDSC,
    SC_MAC_M1,
    SC_MAC_M2,
    SC_FIB_QUIESCENT,
    SC_FIB_MYCAF,
    SC_FIB_ICAF,
    SC_FIB_FRC,
    SC_VAS_TIP,
    SC_VAS_PHALANX,
    SC_VAS_COLLAPSED,
    SC_VAS_HEV,
    SC_BCELL_NAIVE,
    SC_BCELL_ACT,
    SC_BCELL_PLASMA,
    SC_DC_CDC1_IMMATURE,
    SC_DC_CDC1_MATURE,
    SC_DC_CDC2_IMMATURE,
    SC_DC_CDC2_MATURE,
    ABM_STATE_COUNTER_SIZE    // = 16
};
constexpr int MAX_RECRUITS_PER_STEP = 4096;    // Max recruitment requests per ABM step (GPU buffer size)

// GPU recruitment request: filled by recruit_all_kernel, consumed by place_recruited_agents host fn.
struct RecruitRequest {
    int x, y, z;            // Placement voxel coordinates
    int cell_type;          // CELL_TYPE_T, CELL_TYPE_TREG, CELL_TYPE_MAC, CELL_TYPE_MDSC, CELL_TYPE_DC
    int cell_state;         // Sub-state within type (e.g., T_CELL_EFF, TCD4_TREG, MAC_M1)
    int life;               // Pre-sampled lifespan (normal or exponential)
    int divide_cd;          // Division cooldown (T/TReg only)
    int divide_limit;       // Max divisions (T/TReg only)
    float IL2_release_remain;   // T cells only
    float TGFB_release_remain;  // TReg/TH only
    float CTLA4;                // TReg only
    int subtype;            // DC subtype: DC_CDC1=0, DC_CDC2=1 (DC only)
};

// Action types for intent messages
enum IntentAction : int {
    INTENT_NONE = 0,
    INTENT_MOVE = 1,
    INTENT_DIVIDE = 2
};

// ABM event counter indices (for abm_event_counters MacroProperty array)
enum ABMEventCounterIndex : int {
    ABM_COUNT_CC_DEATH = 0,           // Total cancer cell deaths
    ABM_COUNT_CC_DEATH_T_KILL = 1,    // Cancer deaths from T cell killing
    ABM_COUNT_CC_DEATH_MAC_KILL = 2,  // Cancer deaths from macrophage killing
    ABM_COUNT_CC_DEATH_NATURAL = 3,   // Cancer deaths from senescence
    ABM_COUNT_TEFF_REC = 4,           // T effector cells recruited to tumor
    ABM_COUNT_TH_REC = 5,             // T helper cells recruited to tumor
    ABM_COUNT_TREG_REC = 6,           // T regulatory cells recruited to tumor
    ABM_COUNT_MDSC_REC = 7,           // MDSCs recruited to tumor
    ABM_COUNT_MAC_REC = 8,            // Macrophages recruited to tumor
    // Diagnostic counters (Phase 1 debug instrumentation)
    ABM_COUNT_CC_DIVIDE_ATTEMPT = 9,  // Cancer cells that entered division candidate search
    ABM_COUNT_CC_DIVIDE_NO_SPACE = 10,// Cancer cells that found no open neighbor (n_cands==0)
    ABM_COUNT_CC_SENESCENCE = 11,     // Progenitors that transitioned to senescent this step
    ABM_COUNT_CC_T_KILL_EVAL = 12,    // Cancer cells evaluated for T kill (had T neighbors)
    ABM_COUNT_CC_P_KILL_SUM_INT = 13, // Sum of p_kill * 10000 (integer encoding for atomicAdd)
    ABM_COUNT_CC_MAC_KILL_EVAL = 14,  // Cancer cells evaluated for MAC kill (had M1 neighbors)
    ABM_COUNT_BCELL_REC = 15,         // B cells recruited to tumor
    ABM_COUNT_DC_REC = 16             // DCs recruited to tumor
};

// ============================================================
// PDE Environment Property Name Constants
// Maps chemical names to their environment pointer property names.
// Indices match ChemicalSubstrate enum in pde_solver.cuh:
//   O2=0, IFN=1, IL2=2, IL10=3, TGFB=4, CCL2=5, ARGI=6, NO=7, IL12=8, VEGFA=9
// ============================================================

// PDE environment pointer property name constants.
// Using #define so these work as string literals in both host and device code
// (FLAMEGPU's getProperty requires a string literal / const char(&)[N]).
// Indices match ChemicalSubstrate enum in pde_solver.cuh.

// Concentration pointers (read-only in agent functions)
#define PDE_CONC_O2    "pde_concentration_ptr_0"
#define PDE_CONC_IFN   "pde_concentration_ptr_1"
#define PDE_CONC_IL2   "pde_concentration_ptr_2"
#define PDE_CONC_IL10  "pde_concentration_ptr_3"
#define PDE_CONC_TGFB  "pde_concentration_ptr_4"
#define PDE_CONC_CCL2  "pde_concentration_ptr_5"
#define PDE_CONC_ARGI  "pde_concentration_ptr_6"
#define PDE_CONC_NO    "pde_concentration_ptr_7"
#define PDE_CONC_IL12  "pde_concentration_ptr_8"
#define PDE_CONC_VEGFA "pde_concentration_ptr_9"
#define PDE_CONC_IL1   "pde_concentration_ptr_10"
#define PDE_CONC_IL6   "pde_concentration_ptr_11"
#define PDE_CONC_CXCL13 "pde_concentration_ptr_12"
#define PDE_CONC_MMP   "pde_concentration_ptr_13"
#define PDE_CONC_ANTIBODY "pde_concentration_ptr_14"
#define PDE_CONC_CCL21 "pde_concentration_ptr_15"
#define PDE_CONC_CXCL12 "pde_concentration_ptr_16"
#define PDE_CONC_CCL5  "pde_concentration_ptr_17"

// Source pointers (atomicAdd secretion rate / voxel_volume → [conc/s])
#define PDE_SRC_O2    "pde_source_ptr_0"
#define PDE_SRC_IFN   "pde_source_ptr_1"
#define PDE_SRC_IL2   "pde_source_ptr_2"
#define PDE_SRC_IL10  "pde_source_ptr_3"
#define PDE_SRC_TGFB  "pde_source_ptr_4"
#define PDE_SRC_CCL2  "pde_source_ptr_5"
#define PDE_SRC_ARGI  "pde_source_ptr_6"
#define PDE_SRC_NO    "pde_source_ptr_7"
#define PDE_SRC_IL12  "pde_source_ptr_8"
#define PDE_SRC_VEGFA "pde_source_ptr_9"
#define PDE_SRC_IL1   "pde_source_ptr_10"
#define PDE_SRC_IL6   "pde_source_ptr_11"
#define PDE_SRC_CXCL13 "pde_source_ptr_12"
#define PDE_SRC_MMP    "pde_source_ptr_13"
#define PDE_SRC_ANTIBODY "pde_source_ptr_14"
#define PDE_SRC_CCL21 "pde_source_ptr_15"
#define PDE_SRC_CXCL12 "pde_source_ptr_16"
#define PDE_SRC_CCL5  "pde_source_ptr_17"

// Uptake pointers (atomicAdd first-order decay rate [1/s], no volume scaling)
#define PDE_UPT_O2    "pde_uptake_ptr_0"
#define PDE_UPT_IFN   "pde_uptake_ptr_1"
#define PDE_UPT_IL2   "pde_uptake_ptr_2"
#define PDE_UPT_IL10  "pde_uptake_ptr_3"
#define PDE_UPT_TGFB  "pde_uptake_ptr_4"
#define PDE_UPT_CCL2  "pde_uptake_ptr_5"
#define PDE_UPT_ARGI  "pde_uptake_ptr_6"
#define PDE_UPT_NO    "pde_uptake_ptr_7"
#define PDE_UPT_IL12  "pde_uptake_ptr_8"
#define PDE_UPT_VEGFA "pde_uptake_ptr_9"
#define PDE_UPT_IL1   "pde_uptake_ptr_10"
#define PDE_UPT_IL6   "pde_uptake_ptr_11"
#define PDE_UPT_CXCL13 "pde_uptake_ptr_12"
#define PDE_UPT_MMP    "pde_uptake_ptr_13"
#define PDE_UPT_ANTIBODY "pde_uptake_ptr_14"
#define PDE_UPT_CCL21 "pde_uptake_ptr_15"
#define PDE_UPT_CXCL12 "pde_uptake_ptr_16"
#define PDE_UPT_CCL5  "pde_uptake_ptr_17"

// Gradient pointers (read-only, filled by compute_pde_gradients each step)
#define PDE_GRAD_IFN_X   "pde_grad_IFN_x"
#define PDE_GRAD_IFN_Y   "pde_grad_IFN_y"
#define PDE_GRAD_IFN_Z   "pde_grad_IFN_z"
#define PDE_GRAD_TGFB_X  "pde_grad_TGFB_x"
#define PDE_GRAD_TGFB_Y  "pde_grad_TGFB_y"
#define PDE_GRAD_TGFB_Z  "pde_grad_TGFB_z"
#define PDE_GRAD_CCL2_X  "pde_grad_CCL2_x"
#define PDE_GRAD_CCL2_Y  "pde_grad_CCL2_y"
#define PDE_GRAD_CCL2_Z  "pde_grad_CCL2_z"
#define PDE_GRAD_VEGFA_X "pde_grad_VEGFA_x"
#define PDE_GRAD_VEGFA_Y "pde_grad_VEGFA_y"
#define PDE_GRAD_VEGFA_Z "pde_grad_VEGFA_z"
#define PDE_GRAD_CXCL13_X "pde_grad_CXCL13_x"
#define PDE_GRAD_CXCL13_Y "pde_grad_CXCL13_y"
#define PDE_GRAD_CXCL13_Z "pde_grad_CXCL13_z"
#define PDE_GRAD_CCL21_X "pde_grad_CCL21_x"
#define PDE_GRAD_CCL21_Y "pde_grad_CCL21_y"
#define PDE_GRAD_CCL21_Z "pde_grad_CCL21_z"
#define PDE_GRAD_CXCL12_X "pde_grad_CXCL12_x"
#define PDE_GRAD_CXCL12_Y "pde_grad_CXCL12_y"
#define PDE_GRAD_CXCL12_Z "pde_grad_CXCL12_z"
#define PDE_GRAD_CCL5_X  "pde_grad_CCL5_x"
#define PDE_GRAD_CCL5_Y  "pde_grad_CCL5_y"
#define PDE_GRAD_CCL5_Z  "pde_grad_CCL5_z"

// ============================================================
// PDE Access Helper Macros
// Usage examples:
//   float o2  = PDE_READ(FLAMEGPU, PDE_CONC_O2, voxel);
//   PDE_SECRETE(FLAMEGPU, PDE_SRC_CCL2, voxel, rate / voxel_volume);
//   PDE_UPTAKE(FLAMEGPU, PDE_UPT_IFN, voxel, rate_per_sec);
// ============================================================
#define PDE_READ(fgpu, name, voxel) \
    (reinterpret_cast<const float*>((fgpu)->environment.getProperty<uint64_t>(name))[(voxel)])

#define PDE_SECRETE(fgpu, name, voxel, rate_per_vol) \
    atomicAdd(&reinterpret_cast<float*>((fgpu)->environment.getProperty<uint64_t>(name))[(voxel)], (rate_per_vol))

#define PDE_UPTAKE(fgpu, name, voxel, rate_per_sec) \
    atomicAdd(&reinterpret_cast<float*>((fgpu)->environment.getProperty<uint64_t>(name))[(voxel)], (rate_per_sec))

// Helper device function to check grid bounds
__device__ __forceinline__ bool is_in_bounds(int x, int y, int z, int size_x, int size_y, int size_z) {
    return x >= 0 && x < size_x && y >= 0 && y < size_y && z >= 0 && z < size_z;
}

// Helper device function to compute linear voxel index
__device__ __forceinline__ int voxel_index(int x, int y, int z, int size_x, int size_y) {
    return x + y * size_x + z * size_x * size_y;
}

// Volume-based occupancy replaces the old per-type occ_grid MacroProperty.
// See volume_try_claim / volume_release helpers below.

// Helper function: Hill equation
__device__ __forceinline__ float hill_equation(float x, float k50, float n) {
    if (x <= 0.0f) return 0.0f;
    const float xn = powf(x, n);
    const float kn = powf(k50, n);
    return xn / (kn + xn);
}

__device__ __forceinline__ float update_PDL1(float local_IFNg, float IFNg_PDL1_EC50, float IFNg_PDL1_hill,
                                             float PDL1_syn_max, float PDL1_current,
                                             float k_internalization, float dt) {
    // Target PDL1 level: IFN-γ driven Hill equilibrium
    float H_IFNg = hill_equation(local_IFNg, IFNg_PDL1_EC50, IFNg_PDL1_hill);
    float target = PDL1_syn_max * H_IFNg;

    // Exponential approach toward target (both up and down)
    // Models ODE externalization (k_out_PDL1 * H_IFNg) / internalization (k_in_PDL1)
    // Using k_internalization as the relaxation rate
    float decay = expf(-k_internalization * dt);
    return target + (PDL1_current - target) * decay;
}

__device__ __forceinline__ float get_PD1_PDL1(float PDL1, float Nivo,
     float T1, float k1, float k2, float k3) {
    
    double T2 = PDL1;
    double a = 1;
	double b = (Nivo*k2/k1*(2*k3/k1-1) - 2*T2 - T1 - 1/k1)/T2;
	double c = (Nivo*k2/k1 + 1/k1  +T2 + 2*T1 )/T2;
	double d = -T1/T2;

	//Newton_Raphson_root
	int max_iter = 20;
	double tol_rel = 1E-5;
	double root = 0;
	double res, root_new, f, f1;
	int i = 0;
	while (i < max_iter){
		f = a*std::pow(root, 3) + b*std::pow(root, 2)+ c*root + d;
		f1 = 3.0*a*std::pow(root, 2) + 2.0*b*root + c;
		root_new = root - f/f1;
		res = std::abs(root_new - root) / root_new;
		if (res > tol_rel){
			i++;
			root = root_new;
		}
		else{
			break;
		}
	}

	return T2*root;
}

// Helper: Calculate T cell killing probability
__device__ __forceinline__ float get_kill_probability_supp(float supp, float q, float kill_rate) {
    return 1 - std::pow(kill_rate, q*(1-supp));
}

// Helper: Calculate M1 macrophage killing probability
__device__ __forceinline__ float get_kill_probability(float q, float kill_rate) {
    return 1 - std::pow(kill_rate, q);
}

// Helper function to get Moore neighborhood direction
// Returns direction offset for index 0-25
// Indices 0-5 are Von Neumann (face) neighbors, 6-25 are edge/corner neighbors
__device__ __forceinline__ void get_moore_direction(int idx, int& dx, int& dy, int& dz) {
    // Face neighbors (6): indices 0-5 - these are Von Neumann neighbors
    // Edge neighbors (12): indices 6-17
    // Corner neighbors (8): indices 18-25
    const int dirs[26][3] = {
        // Face neighbors (Von Neumann)
        {-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1},
        // Edge neighbors
        {-1, -1, 0}, {-1, 1, 0}, {1, -1, 0}, {1, 1, 0},
        {-1, 0, -1}, {-1, 0, 1}, {1, 0, -1}, {1, 0, 1},
        {0, -1, -1}, {0, -1, 1}, {0, 1, -1}, {0, 1, 1},
        // Corner neighbors
        {-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
        {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}
    };
    dx = dirs[idx][0];
    dy = dirs[idx][1];
    dz = dirs[idx][2];
}

// Not using currently but saving for reference
// Von Neumann mask: only face neighbors (bits 0-5)
constexpr unsigned int VON_NEUMANN_MASK = 0x3Fu;  // binary: 00111111

// ============================================================
// Volume-Based Occupancy Helpers
// ============================================================
// Environment property name for the volume_used device array pointer
#define VOL_USED_PTR "volume_used_ptr"

// Read volume_used pointer from FLAMEGPU environment
#define VOL_PTR(fgpu) \
    reinterpret_cast<float*>((fgpu)->environment.getProperty<uint64_t>(VOL_USED_PTR))

// Read ECM density and crosslink pointers from FLAMEGPU environment
#define ECM_DENSITY_PTR(fgpu) \
    reinterpret_cast<const float*>((fgpu)->environment.getProperty<uint64_t>("ecm_density_ptr"))
#define ECM_CROSSLINK_PTR(fgpu) \
    reinterpret_cast<const float*>((fgpu)->environment.getProperty<uint64_t>("ecm_crosslink_ptr"))

// Read antigen grid pointer (persistent antigen deposited by dying cancer cells)
#define ANTIGEN_GRID_PTR(fgpu) \
    reinterpret_cast<float*>((fgpu)->environment.getProperty<uint64_t>("antigen_grid_ptr"))

// ECM fiber orientation pointers (per-voxel axis vector, magnitude = alignment strength)
#define ECM_ORIENT_X_PTR(fgpu) \
    reinterpret_cast<const float*>((fgpu)->environment.getProperty<uint64_t>("ecm_orient_x_ptr"))
#define ECM_ORIENT_Y_PTR(fgpu) \
    reinterpret_cast<const float*>((fgpu)->environment.getProperty<uint64_t>("ecm_orient_y_ptr"))
#define ECM_ORIENT_Z_PTR(fgpu) \
    reinterpret_cast<const float*>((fgpu)->environment.getProperty<uint64_t>("ecm_orient_z_ptr"))

// Mechanical stress field pointers (transient per-voxel stress from cancer movement)
#define STRESS_X_PTR(fgpu) \
    reinterpret_cast<float*>((fgpu)->environment.getProperty<uint64_t>("stress_x_ptr"))
#define STRESS_Y_PTR(fgpu) \
    reinterpret_cast<float*>((fgpu)->environment.getProperty<uint64_t>("stress_y_ptr"))
#define STRESS_Z_PTR(fgpu) \
    reinterpret_cast<float*>((fgpu)->environment.getProperty<uint64_t>("stress_z_ptr"))

// Adhesion matrix pointer (flat float[ABM_STATE_COUNTER_SIZE * ABM_STATE_COUNTER_SIZE])
#define ADH_MATRIX_PTR(fgpu) \
    reinterpret_cast<const float*>((fgpu)->environment.getProperty<uint64_t>("adh_matrix_ptr"))

// Map broadcast message fields (agent_type, cell_state, kill_factor) to StateCounterIdx.
// DC encodes subtype in kill_factor (0.0=cDC1, 1.0=cDC2); other agents ignore it here.
__device__ __forceinline__ int msg_to_sc_idx(int agent_type, int cell_state, float kill_factor) {
    switch (agent_type) {
        case CELL_TYPE_CANCER:
            switch (cell_state) {
                case CANCER_STEM:       return SC_CANCER_STEM;
                case CANCER_PROGENITOR: return SC_CANCER_PROG;
                default:                return SC_CANCER_SEN;
            }
        case CELL_TYPE_T:
            switch (cell_state) {
                case T_CELL_EFF:  return SC_CD8_EFF;
                case T_CELL_CYT:  return SC_CD8_CYT;
                case T_CELL_SUPP: return SC_CD8_SUP;
                default:          return SC_CD8_NAIVE;
            }
        case CELL_TYPE_TREG:
            switch (cell_state) {
                case TCD4_TH:   return SC_TH;
                case TCD4_TREG: return SC_TREG;
                case TCD4_TFH:  return SC_TFH;
                default:        return SC_TCD4_NAIVE;
            }
        case CELL_TYPE_MDSC: return SC_MDSC;
        case CELL_TYPE_MAC:
            return (cell_state == MAC_M1) ? SC_MAC_M1 : SC_MAC_M2;
        case CELL_TYPE_FIB:
            switch (cell_state) {
                case FIB_QUIESCENT: return SC_FIB_QUIESCENT;
                case FIB_MYCAF:     return SC_FIB_MYCAF;
                case FIB_ICAF:      return SC_FIB_ICAF;
                default:            return SC_FIB_FRC;
            }
        case CELL_TYPE_VASCULAR:
            switch (cell_state) {
                case VAS_TIP:               return SC_VAS_TIP;
                case VAS_PHALANX:           return SC_VAS_PHALANX;
                case VAS_PHALANX_COLLAPSED: return SC_VAS_COLLAPSED;
                default:                    return SC_VAS_HEV;
            }
        case CELL_TYPE_BCELL:
            switch (cell_state) {
                case BCELL_NAIVE:     return SC_BCELL_NAIVE;
                case BCELL_ACTIVATED: return SC_BCELL_ACT;
                default:              return SC_BCELL_PLASMA;
            }
        case CELL_TYPE_DC: {
            int subtype = (kill_factor < 0.5f) ? DC_CDC1 : DC_CDC2;
            if (subtype == DC_CDC1)
                return (cell_state == DC_IMMATURE) ? SC_DC_CDC1_IMMATURE : SC_DC_CDC1_MATURE;
            else
                return (cell_state == DC_IMMATURE) ? SC_DC_CDC2_IMMATURE : SC_DC_CDC2_MATURE;
        }
        default: return 0;
    }
}

// Convenience: map own agent type + cell state to SC index (no kill_factor needed)
__device__ __forceinline__ int self_sc_idx(int agent_type, int cell_state) {
    return msg_to_sc_idx(agent_type, cell_state, 0.0f);
}
// DC variant that needs subtype
__device__ __forceinline__ int dc_self_sc_idx(int cell_state, int dc_subtype) {
    return msg_to_sc_idx(CELL_TYPE_DC, cell_state, static_cast<float>(dc_subtype));
}

// Compute ECM porosity at a voxel: porosity = max(0, 1 - density/cap * (1 + crosslink))
__device__ __forceinline__ float ecm_porosity(const float* ecm_density, const float* ecm_crosslink,
                                               int voxel_idx, float density_cap) {
    float d = ecm_density[voxel_idx];
    float c = ecm_crosslink[voxel_idx];
    return fmaxf(0.0f, 1.0f - (d / density_cap) * (1.0f + c));
}

// Attempt to claim volume in a target voxel via atomicAdd.
// Returns true if the claim succeeded (total <= capacity), false otherwise (undo performed).
__device__ __forceinline__ bool volume_try_claim(float* vol_used, int voxel_idx, float my_volume, float capacity) {
    float old = atomicAdd(&vol_used[voxel_idx], my_volume);
    if (old + my_volume <= capacity) {
        return true;  // Claim succeeded
    }
    // Over capacity — undo the claim
    atomicAdd(&vol_used[voxel_idx], -my_volume);
    return false;
}

// Release volume from a voxel (e.g., after movement or death).
__device__ __forceinline__ void volume_release(float* vol_used, int voxel_idx, float my_volume) {
    atomicAdd(&vol_used[voxel_idx], -my_volume);
}

// ============================================================
// Contact Guidance Helper
// ============================================================
// Blends a chemotaxis gradient with local fiber orientation.
// Amplifies the gradient component parallel to fibers, suppresses perpendicular.
// Uses nematic (bidirectional) decomposition: fibers have no preferred sign.
// w_contact: contact guidance strength (0 = no guidance, 1 = full channeling)
// orient_x/y/z: fiber axis vector at voxel (magnitude = alignment strength, 0 = isotropic)
// grad_x/y/z: chemotaxis gradient (modified in-place)
__device__ __forceinline__ void apply_contact_guidance(
    float& grad_x, float& grad_y, float& grad_z,
    float orient_x, float orient_y, float orient_z,
    float w_contact)
{
    float fiber_mag = sqrtf(orient_x * orient_x + orient_y * orient_y + orient_z * orient_z);
    if (fiber_mag < 0.01f || w_contact < 1e-6f) return;  // isotropic or no guidance

    float inv_mag = 1.0f / fiber_mag;
    float fhat_x = orient_x * inv_mag;
    float fhat_y = orient_y * inv_mag;
    float fhat_z = orient_z * inv_mag;

    // Decompose gradient into fiber-parallel and fiber-perpendicular
    float dot_cf = grad_x * fhat_x + grad_y * fhat_y + grad_z * fhat_z;
    float para_x = dot_cf * fhat_x;
    float para_y = dot_cf * fhat_y;
    float para_z = dot_cf * fhat_z;
    float perp_x = grad_x - para_x;
    float perp_y = grad_y - para_y;
    float perp_z = grad_z - para_z;

    // Amplify parallel, suppress perpendicular (scaled by alignment strength)
    float boost = 1.0f + w_contact * fiber_mag;
    float suppress = fmaxf(0.0f, 1.0f - w_contact * fiber_mag);

    grad_x = para_x * boost + perp_x * suppress;
    grad_y = para_y * boost + perp_y * suppress;
    grad_z = para_z * boost + perp_z * suppress;
}

// Apply contact guidance to persistence direction for cells with no chemotaxis.
// Projects persistence direction onto fiber axis to create an effective gradient.
__device__ __forceinline__ void apply_contact_guidance_persist(
    float& grad_x, float& grad_y, float& grad_z,
    float& bias_strength,
    float orient_x, float orient_y, float orient_z,
    float w_contact,
    int persist_dx, int persist_dy, int persist_dz)
{
    float fiber_mag = sqrtf(orient_x * orient_x + orient_y * orient_y + orient_z * orient_z);
    if (fiber_mag < 0.01f || w_contact < 1e-6f) return;

    float inv_mag = 1.0f / fiber_mag;
    float fhat_x = orient_x * inv_mag;
    float fhat_y = orient_y * inv_mag;
    float fhat_z = orient_z * inv_mag;

    float pdot = static_cast<float>(persist_dx) * fhat_x
               + static_cast<float>(persist_dy) * fhat_y
               + static_cast<float>(persist_dz) * fhat_z;
    float sign = (pdot >= 0.0f) ? 1.0f : -1.0f;

    grad_x = sign * orient_x * w_contact;
    grad_y = sign * orient_y * w_contact;
    grad_z = sign * orient_z * w_contact;
    bias_strength = fiber_mag;
}

// ============================================================
// Unified Movement Framework
// ============================================================
// Replaces per-agent run-tumble and random walk with a composable
// 3-behavior pipeline: adhesion → persistence → gradient-biased selection.

// Convert chemotaxis index (CI) to internal bias_strength for CDF weighting.
// On the 26-neighbor Moore lattice, direction weights are w_i = max(0, 1 + b*cos(theta_i)).
// The expected CI (= E[displacement along gradient] / E[step length]) is exactly CI = b/3
// for b <= 1 (no clamping), due to lattice symmetry: sum(cos_i*dx_i)/sum(mag_i) = 1/3.
// For CI > 1/3 (b > 1), clamping saturates the relationship; max achievable CI ~ 0.68.
// Typical experimental immune cell CIs are 0.05-0.40, well within the linear regime.
__device__ __forceinline__ float ci_to_bias(float ci) {
    return 3.0f * ci;
}

// Compute adhesion-based move probability from type+state neighbor counts and matrix.
//   p_move = max(0, 1 - sum_j(M[my_idx][j] * n_j))
// M[i][j] is the per-contact adhesion coefficient: each neighbor of type j reduces
// i's move probability by M[i][j]. Paired cells with M=0.5 give p_move=0.5 at n=1.
// adh_matrix is flat float[ABM_STATE_COUNTER_SIZE * ABM_STATE_COUNTER_SIZE], row-major.
// neighbor_counts is int[ABM_STATE_COUNTER_SIZE] from scan_neighbors.
__device__ __forceinline__ float compute_adhesion_pmove(
    int my_sc_idx,
    const int* neighbor_counts,
    const float* adh_matrix)
{
    const float* row = &adh_matrix[my_sc_idx * ABM_STATE_COUNTER_SIZE];
    float sum = 0.0f;
    for (int j = 0; j < ABM_STATE_COUNTER_SIZE; j++) {
        sum += row[j] * static_cast<float>(neighbor_counts[j]);
    }
    return fmaxf(0.0f, 1.0f - sum);
}

struct MoveParams {
    // Grid dimensions
    int grid_x, grid_y, grid_z;
    // Volume occupancy
    float* vol_used;
    float my_vol;
    float capacity;
    // ECM filtering
    const float* ecm_density;
    const float* ecm_crosslink;
    float density_cap;
    float min_porosity;
    // Behavior: adhesion (pre-computed move probability, 1.0 = no adhesion)
    float p_move;
    // Behavior: persistence (probability of keeping previous direction)
    float p_persist;
    // Behavior: chemotaxis (gradient-biased direction selection)
    float bias_strength;   // internal CDF weight param; set via ci_to_bias(CI)
    float grad_x, grad_y, grad_z;  // gradient vector at current voxel
    // Fiber barrier: anisotropic restriction from aligned ECM fibers at destination voxel.
    // Penalizes perpendicular movement through aligned fibers; parallel movement unaffected.
    const float* orient_x;
    const float* orient_y;
    const float* orient_z;
    float barrier_strength;  // 0 = no barrier, 1 = full perpendicular block (scaled by fiber_mag)
};

struct MoveResult {
    int new_x, new_y, new_z;
    int persist_dx, persist_dy, persist_dz;
    bool moved;
};

// Unified movement: adhesion check → build candidates → persistence → gradient-biased select → claim.
// Caller provides 3 pre-rolled uniform [0,1) random floats.
__device__ __forceinline__ MoveResult move_cell(
    const MoveParams& p,
    int x, int y, int z,
    int persist_dx, int persist_dy, int persist_dz,
    float r_move,       // for adhesion gate
    float r_persist,    // for persistence check
    float r_direction)  // for gradient-weighted CDF sampling
{
    MoveResult result;
    result.new_x = x; result.new_y = y; result.new_z = z;
    result.persist_dx = persist_dx;
    result.persist_dy = persist_dy;
    result.persist_dz = persist_dz;
    result.moved = false;

    // --- Step 1: Adhesion check ---
    if (r_move >= p.p_move) return result;

    // --- Step 2: Build candidate list (26 Moore neighbors, volume + porosity filter) ---
    int cand_dx[26], cand_dy[26], cand_dz[26];
    int n_cands = 0;

    for (int i = 0; i < 26; i++) {
        int ddx, ddy, ddz;
        get_moore_direction(i, ddx, ddy, ddz);
        int nx = x + ddx, ny = y + ddy, nz = z + ddz;
        if (nx < 0 || nx >= p.grid_x || ny < 0 || ny >= p.grid_y || nz < 0 || nz >= p.grid_z)
            continue;
        int vidx = nz * (p.grid_x * p.grid_y) + ny * p.grid_x + nx;
        if (p.vol_used[vidx] + p.my_vol > p.capacity) continue;
        if (ecm_porosity(p.ecm_density, p.ecm_crosslink, vidx, p.density_cap) < p.min_porosity)
            continue;
        cand_dx[n_cands] = ddx;
        cand_dy[n_cands] = ddy;
        cand_dz[n_cands] = ddz;
        n_cands++;
    }
    if (n_cands == 0) return result;

    // --- Step 3: Persistence check ---
    int sel_dx = 0, sel_dy = 0, sel_dz = 0;
    bool selected = false;

    if (r_persist < p.p_persist && (persist_dx != 0 || persist_dy != 0 || persist_dz != 0)) {
        // Check if previous direction is in candidate list
        for (int i = 0; i < n_cands; i++) {
            if (cand_dx[i] == persist_dx && cand_dy[i] == persist_dy && cand_dz[i] == persist_dz) {
                sel_dx = persist_dx; sel_dy = persist_dy; sel_dz = persist_dz;
                selected = true;
                break;
            }
        }
    }

    // --- Step 4: Gradient-biased direction selection (if persistence didn't fire) ---
    if (!selected) {
        // Normalize gradient
        float gmag = sqrtf(p.grad_x * p.grad_x + p.grad_y * p.grad_y + p.grad_z * p.grad_z);
        float ghat_x = 0.0f, ghat_y = 0.0f, ghat_z = 0.0f;
        if (gmag > 1e-12f) {
            float inv_gmag = 1.0f / gmag;
            ghat_x = p.grad_x * inv_gmag;
            ghat_y = p.grad_y * inv_gmag;
            ghat_z = p.grad_z * inv_gmag;
        }

        // Compute weights: w_i = max(0, 1 + bias * cos_angle) * w_barrier
        // Fiber barrier: penalize perpendicular movement at destination voxel.
        // w_barrier = 1 - barrier_strength * fiber_mag * sin²(angle_to_fiber)
        // Parallel to fiber: w_barrier = 1.0 (no effect). Perpendicular: reduced.
        float weights[26];
        float total_w = 0.0f;
        for (int i = 0; i < n_cands; i++) {
            float dx_f = static_cast<float>(cand_dx[i]);
            float dy_f = static_cast<float>(cand_dy[i]);
            float dz_f = static_cast<float>(cand_dz[i]);
            float dir_mag = sqrtf(dx_f * dx_f + dy_f * dy_f + dz_f * dz_f);
            float cos_angle = (dx_f * ghat_x + dy_f * ghat_y + dz_f * ghat_z) / dir_mag;
            float w = fmaxf(0.0f, 1.0f + p.bias_strength * cos_angle);

            // Fiber barrier at destination voxel
            if (p.barrier_strength > 1e-6f) {
                int dest_vidx = (z + cand_dz[i]) * (p.grid_x * p.grid_y)
                              + (y + cand_dy[i]) * p.grid_x
                              + (x + cand_dx[i]);
                float fox = p.orient_x[dest_vidx];
                float foy = p.orient_y[dest_vidx];
                float foz = p.orient_z[dest_vidx];
                float fmag2 = fox * fox + foy * foy + foz * foz;
                if (fmag2 > 1e-4f) {
                    float fmag = sqrtf(fmag2);
                    float cos_fiber = (dx_f * fox + dy_f * foy + dz_f * foz) / (dir_mag * fmag);
                    float sin2 = 1.0f - cos_fiber * cos_fiber;
                    w *= fmaxf(0.0f, 1.0f - p.barrier_strength * fmag * sin2);
                }
            }

            weights[i] = w;
            total_w += w;
        }

        // Sample from weighted CDF
        if (total_w <= 0.0f) {
            // All weights zero (shouldn't happen with bias <= 1.0) — uniform fallback
            int pick = static_cast<int>(r_direction * n_cands);
            if (pick >= n_cands) pick = n_cands - 1;
            sel_dx = cand_dx[pick]; sel_dy = cand_dy[pick]; sel_dz = cand_dz[pick];
        } else {
            float r = r_direction * total_w;
            float cumsum = 0.0f;
            int pick = n_cands - 1;  // fallback to last
            for (int i = 0; i < n_cands; i++) {
                cumsum += weights[i];
                if (r < cumsum) { pick = i; break; }
            }
            sel_dx = cand_dx[pick]; sel_dy = cand_dy[pick]; sel_dz = cand_dz[pick];
        }
    }

    // --- Step 5: Attempt atomic volume claim ---
    int nx = x + sel_dx, ny = y + sel_dy, nz = z + sel_dz;
    int target_vidx = nz * (p.grid_x * p.grid_y) + ny * p.grid_x + nx;
    int old_vidx = z * (p.grid_x * p.grid_y) + y * p.grid_x + x;

    if (volume_try_claim(p.vol_used, target_vidx, p.my_vol, p.capacity)) {
        volume_release(p.vol_used, old_vidx, p.my_vol);
        result.new_x = nx; result.new_y = ny; result.new_z = nz;
        result.persist_dx = sel_dx; result.persist_dy = sel_dy; result.persist_dz = sel_dz;
        result.moved = true;
    }

    return result;
}

} // namespace PDAC

#endif // PDAC_COMMON_CUH
