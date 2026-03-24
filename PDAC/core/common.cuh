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

// T reg state enumeration (TH=0 matches HCC convention: TCD4_Th < TCD4_TREG)
enum TCD4State : int {
    TCD4_TH = 0,
    TCD4_TREG = 1,
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
constexpr int N_DIVIDE_WAVES = 1;               // Wave-interleaved division rounds (cancer/tcell/treg)
constexpr int MAX_MAC_PER_VOXEL = 1;            // Max macrophage per voxel (exclusive)
constexpr int MAX_FIB_CHAIN_LENGTH = 5;         // Max segments per fibroblast chain (3 normal, 5 CAF)
// ── Per-step event counters (device_event_counters[], env prop "event_counters_ptr") ──────────────
// Reset each step. Written by agent device functions via atomicAdd.
enum EventCounterIdx : int {
    // Proliferation (daughter cell created by division)
    EVT_PROLIF_CD8_EFF = 0,
    EVT_PROLIF_CD8_CYT,
    EVT_PROLIF_CD8_SUP,
    EVT_PROLIF_TH,
    EVT_PROLIF_TREG,
    EVT_PROLIF_MDSC,          // 0 — division not implemented
    EVT_PROLIF_CANCER_STEM,
    EVT_PROLIF_CANCER_PROG,
    EVT_PROLIF_CANCER_SEN,    // 0 — senescent cells don't divide
    EVT_PROLIF_MAC_M1,        // 0 — division not implemented
    EVT_PROLIF_MAC_M2,        // 0
    EVT_PROLIF_FIB_NORM,      // 0 — division disabled
    EVT_PROLIF_FIB_CAF,       // 0
    EVT_PROLIF_VAS_TIP,
    EVT_PROLIF_VAS_PHALANX,   // 0
    // Deaths (all causes combined, by cell type/state)
    EVT_DEATH_CD8_EFF,
    EVT_DEATH_CD8_CYT,
    EVT_DEATH_CD8_SUP,
    EVT_DEATH_TH,
    EVT_DEATH_TREG,
    EVT_DEATH_MDSC,
    EVT_DEATH_CANCER_STEM,
    EVT_DEATH_CANCER_PROG,
    EVT_DEATH_CANCER_SEN,
    EVT_DEATH_MAC_M1,
    EVT_DEATH_MAC_M2,
    EVT_DEATH_FIB_NORM,
    EVT_DEATH_FIB_CAF,
    EVT_DEATH_VAS_TIP,
    EVT_DEATH_VAS_PHALANX,
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
    SC_TH,
    SC_TREG,
    SC_MDSC,
    SC_MAC_M1,
    SC_MAC_M2,
    SC_FIB_NORM,
    SC_FIB_CAF,
    SC_VAS_TIP,
    SC_VAS_PHALANX,
    ABM_STATE_COUNTER_SIZE    // = 15
};
constexpr int MAX_RECRUITS_PER_STEP = 4096;    // Max recruitment requests per ABM step (GPU buffer size)

// GPU recruitment request: filled by recruit_all_kernel, consumed by place_recruited_agents host fn.
struct RecruitRequest {
    int x, y, z;            // Placement voxel coordinates
    int cell_type;          // CELL_TYPE_T, CELL_TYPE_TREG, CELL_TYPE_MAC, CELL_TYPE_MDSC
    int cell_state;         // Sub-state within type (e.g., T_CELL_EFF, TCD4_TREG, MAC_M1)
    int life;               // Pre-sampled lifespan (normal or exponential)
    int divide_cd;          // Division cooldown (T/TReg only)
    int divide_limit;       // Max divisions (T/TReg only)
    float IL2_release_remain;   // T cells only
    float TGFB_release_remain;  // TReg/TH only
    float CTLA4;                // TReg only
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
    ABM_COUNT_CC_MAC_KILL_EVAL = 14   // Cancer cells evaluated for MAC kill (had M1 neighbors)
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

__device__ __forceinline__ float update_PDL1(float local_IFNg, float IFNg_PDL1_EC50, float IFNg_PDL1_hill, float PDL1_syn_max, float PDL1_current) {
    float H_IFNg = hill_equation(local_IFNg, IFNg_PDL1_EC50, IFNg_PDL1_hill);
    float minPDL1 = PDL1_syn_max * H_IFNg;

    if (PDL1_current < minPDL1) {
        return minPDL1;
    } else {
        return PDL1_current;
    }
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

// ── Ductal wall face flag helpers ───────────────────────────────────────────
// Face flag bit layout (from ductal_init.cuh constants):
//   bit 0 = -x, bit 1 = +x, bit 2 = -y, bit 3 = +y, bit 4 = -z, bit 5 = +z
// Maps Von Neumann direction index (0-5) to the face bit crossed when moving.
__device__ constexpr uint8_t DIR_TO_FACE_BIT[6] = {
    0x01,  // dir 0: (-1, 0, 0) → FACE_NEG_X
    0x02,  // dir 1: (+1, 0, 0) → FACE_POS_X
    0x04,  // dir 2: (0, -1, 0) → FACE_NEG_Y
    0x08,  // dir 3: (0, +1, 0) → FACE_POS_Y
    0x10,  // dir 4: (0, 0, -1) → FACE_NEG_Z
    0x20,  // dir 5: (0, 0, +1) → FACE_POS_Z
};

// Check whether movement from (x,y,z) in Von Neumann direction dir_idx (0-5)
// is blocked by a ductal wall face flag.
__device__ __forceinline__
bool is_wall_blocked(const uint8_t* face_flags,
                     int x, int y, int z,
                     int dir_idx,
                     int size_x, int size_y) {
    int voxel = x + y * size_x + z * size_x * size_y;
    return (face_flags[voxel] & DIR_TO_FACE_BIT[dir_idx]) != 0;
}

// Check whether a diagonal (edge/corner) move is blocked by any ductal wall.
// For diagonal moves we check all face-aligned components: moving (+1,+1,0)
// is blocked if EITHER the +x or +y face carries a wall flag.
__device__ __forceinline__
bool is_wall_blocked_diagonal(const uint8_t* face_flags,
                              int x, int y, int z,
                              int dx, int dy, int dz,
                              int size_x, int size_y) {
    int voxel = x + y * size_x + z * size_x * size_y;
    uint8_t flags = face_flags[voxel];
    if (dx == -1 && (flags & 0x01)) return true;
    if (dx == +1 && (flags & 0x02)) return true;
    if (dy == -1 && (flags & 0x04)) return true;
    if (dy == +1 && (flags & 0x08)) return true;
    if (dz == -1 && (flags & 0x10)) return true;
    if (dz == +1 && (flags & 0x20)) return true;
    return false;
}

// Unified wall check for any Moore direction (dx, dy, dz).
// Returns true if movement from (x,y,z) by (dx,dy,dz) crosses a ductal wall.
// Safe to call with face_flags == nullptr (always returns false → no ductal structure).
__device__ __forceinline__
bool is_ductal_wall_blocked(const uint8_t* face_flags,
                            int x, int y, int z,
                            int dx, int dy, int dz,
                            int size_x, int size_y) {
    if (face_flags == nullptr) return false;
    int voxel = x + y * size_x + z * size_x * size_y;
    uint8_t flags = face_flags[voxel];
    if (dx == -1 && (flags & 0x01)) return true;
    if (dx == +1 && (flags & 0x02)) return true;
    if (dy == -1 && (flags & 0x04)) return true;
    if (dy == +1 && (flags & 0x08)) return true;
    if (dz == -1 && (flags & 0x10)) return true;
    if (dz == +1 && (flags & 0x20)) return true;
    return false;
}

} // namespace PDAC

#endif // PDAC_COMMON_CUH
