#ifndef FLAMEGPU_PDAC_COMMON_CUH
#define FLAMEGPU_PDAC_COMMON_CUH

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
    CELL_TYPE_VASCULAR = 7
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
    T_CELL_SUPP = 2
};

// T reg state enumeration
enum TCD4State : int {
    TCD4_TREG = 0,
    TCD4_TH = 1,
};

// Macrophage state enumeration (M1/M2 polarization)
enum MacrophageState : int {
    MAC_M1 = 0,          // Pro-inflammatory (IFN-γ activated)
    MAC_M2 = 1,          // Anti-inflammatory (IL-10/TGF-β activated)
    MAC_INTERMEDIATE = 2 // Mixed phenotype
};

// Fibroblast state enumeration
enum FibroblastState : int {
    FIB_NORMAL = 0,  // Normal fibroblast
    FIB_CAF = 1      // Cancer-associated fibroblast
};

// Vascular cell cell_states
enum VascularCellState : int {
    VAS_TIP = 0,      // Actively sprouting tip cell
    VAS_STALK = 1,    // Connecting stalk cell
    VAS_PHALANX = 2   // Mature vessel (O2 secreting)
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

// Environment property names for grid dimensions
constexpr const char* ENV_GRID_SIZE_X = "grid_size_x";
constexpr const char* ENV_GRID_SIZE_Y = "grid_size_y";
constexpr const char* ENV_GRID_SIZE_Z = "grid_size_z";
constexpr const char* ENV_VOXEL_SIZE = "voxel_size";

// Voxel capacity from CPU params (param_all.xml: nr_T_voxel, nr_T_voxel_C)
constexpr int MAX_T_PER_VOXEL = 8;              // Max T cells in empty voxel
constexpr int MAX_T_PER_VOXEL_WITH_CANCER = 1;  // Max T cells when cancer present
constexpr int MAX_CANCER_PER_VOXEL = 1;         // Max cancer cells per voxel
constexpr int MAX_MDSC_PER_VOXEL = 1;           // Max MDSC per voxel (exclusive)
constexpr int MAX_MAC_PER_VOXEL = 1;            // Max macrophage per voxel (exclusive)
constexpr int MAX_FIB_SLOTS = 5000;             // Max fibroblast slots in chain position MacroProperty
constexpr int MAX_FIB_CHAIN_LENGTH = 5;         // Max cells per fibroblast chain (HEAD + N-1 followers); grows via division
constexpr int ABM_EVENT_COUNTER_SIZE = 9;      // Array size for ABM→QSP event counters (deaths + recruitment)

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
    ABM_COUNT_MAC_REC = 8             // Macrophages recruited to tumor
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

// ============================================================
// Occupancy Grid Constants
// ============================================================
// Max grid dimension for compile-time MacroProperty allocation.
// Only voxels [0..runtime_grid_size-1] are actually used.
// Memory: 128^3 * 8 types * 4 bytes = ~67 MB (acceptable for modern GPU).
constexpr int OCC_GRID_MAX = 320;

// Number of occupancy type slots (matches AgentType enum max index + 1).
// Index 0 (AGENT_DUMMY) is unused; indices 1-7 map directly to AgentType values.
constexpr int NUM_OCC_TYPES = 8;

// Helper function: Hill equation
__device__ __forceinline__ float hill_equation(float x, float k50, float n) {
    if (x <= 0.0f) return 0.0f;
    const float xn = powf(x, n);
    const float kn = powf(k50, n);
    return xn / (kn + xn);
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

} // namespace PDAC

#endif // PDAC_COMMON_CUH
