#ifndef PDAC_DUCTAL_INIT_CUH
#define PDAC_DUCTAL_INIT_CUH

#include <cstdint>
#include <vector>

// Forward-declare FLAMEGPU types to avoid pulling in the full header
namespace flamegpu { class ModelDescription; }

namespace PDAC {

// Minimum grid side length for ductal initialization (-i 1).
// Below this the ductal tree and septum Voronoi are under-resolved.
constexpr int DUCTAL_INIT_MIN_GRID = 200;

// ── Face flag bit layout ────────────────────────────────────────────────────
// Each voxel stores a uint8_t with 6 bits indicating ductal wall presence.
// Bit i is set if the face in direction i carries a wall.
// When building walls, BOTH voxels sharing a face get their respective bit set.
constexpr uint8_t FACE_NEG_X = 0x01;  // bit 0: wall on -x face
constexpr uint8_t FACE_POS_X = 0x02;  // bit 1: wall on +x face
constexpr uint8_t FACE_NEG_Y = 0x04;  // bit 2: wall on -y face
constexpr uint8_t FACE_POS_Y = 0x08;  // bit 3: wall on +y face
constexpr uint8_t FACE_NEG_Z = 0x10;  // bit 4: wall on -z face
constexpr uint8_t FACE_POS_Z = 0x20;  // bit 5: wall on +z face

// ── Tissue type labels (used internally during rasterization) ───────────────
// These are local working labels for face-flag construction, NOT exported to GPU.
constexpr uint8_t TISSUE_NONE      = 0;
constexpr uint8_t TISSUE_LUMEN     = 1;  // inside ductal lumen
constexpr uint8_t TISSUE_WALL      = 2;  // ductal wall voxel (has face flags)

// ── Tree node for the ductal skeleton ───────────────────────────────────────
struct DuctNode {
    float x, y, z;       // position in voxel coordinates (float for sub-voxel)
    float radius;         // duct radius in voxels
    int   parent;         // index of parent node (-1 for root)
    int   generation;     // branching generation (0 = entry branch)
};

// ── Result structure from ductal generation ─────────────────────────────────
struct DuctalNetwork {
    std::vector<DuctNode> nodes;
    std::vector<uint8_t>  wall_mask;       // [total_voxels] TISSUE_NONE/LUMEN/WALL (host-only)
    std::vector<float>    septum_density;  // [total_voxels] lobular septum ECM density (host-only)

    // Device arrays (allocated on GPU, owned by this struct)
    uint8_t* d_face_flags;      // [total_voxels] ductal wall face flags
    float*   d_septum_density;  // [total_voxels] static septum ECM density

    int grid_x, grid_y, grid_z;
    int total_voxels;

    DuctalNetwork() : d_face_flags(nullptr), d_septum_density(nullptr),
                      grid_x(0), grid_y(0), grid_z(0), total_voxels(0) {}
    ~DuctalNetwork();  // frees GPU memory

    // Move semantics (transfers GPU memory ownership)
    DuctalNetwork(DuctalNetwork&& o) noexcept
        : nodes(std::move(o.nodes)),
          wall_mask(std::move(o.wall_mask)),
          septum_density(std::move(o.septum_density)),
          d_face_flags(o.d_face_flags),
          d_septum_density(o.d_septum_density),
          grid_x(o.grid_x), grid_y(o.grid_y), grid_z(o.grid_z),
          total_voxels(o.total_voxels)
    { o.d_face_flags = nullptr; o.d_septum_density = nullptr; }

    DuctalNetwork& operator=(DuctalNetwork&& o) noexcept;

    // No copy (GPU memory is owned)
    DuctalNetwork(const DuctalNetwork&) = delete;
    DuctalNetwork& operator=(const DuctalNetwork&) = delete;
};

// ── Public API ──────────────────────────────────────────────────────────────

// Generate the ductal network wall structure and septum density field.
// Call once during -i 1 initialization, before agent placement.
DuctalNetwork generate_ductal_network(
    int grid_x, int grid_y, int grid_z,
    float voxel_size_um,
    unsigned int seed);

// Register the ductal network device pointers as FLAMEGPU environment properties.
// Must be called after model is built but before simulation starts.
void register_ductal_pointers(
    flamegpu::ModelDescription& model,
    const DuctalNetwork& network);

// Device-side wall check helpers live in core/common.cuh alongside
// get_moore_direction, is_in_bounds, voxel_index, etc.

} // namespace PDAC

#endif // PDAC_DUCTAL_INIT_CUH
