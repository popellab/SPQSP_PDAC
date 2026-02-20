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
constexpr int MAX_FIB_CHAIN_LENGTH = 3;         // Max cells per fibroblast chain (HEAD + N-1 followers)

// Action types for intent messages
enum IntentAction : int {
    INTENT_NONE = 0,
    INTENT_MOVE = 1,
    INTENT_DIVIDE = 2
};

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
constexpr int OCC_GRID_MAX = 128;

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

} // namespace PDAC

#endif // PDAC_COMMON_CUH
