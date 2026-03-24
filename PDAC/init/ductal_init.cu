#include "ductal_init.cuh"
#include "flamegpu/flamegpu.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <queue>
#include <iostream>
#include <cassert>

namespace PDAC {

// ════════════════════════════════════════════════════════════════════════════
// Helpers
// ════════════════════════════════════════════════════════════════════════════

static inline int vidx(int x, int y, int z, int gx, int gy) {
    return x + y * gx + z * gx * gy;
}

static inline float dist3(float ax, float ay, float az,
                           float bx, float by, float bz) {
    float dx = ax - bx, dy = ay - by, dz = az - bz;
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

static inline float length3(float x, float y, float z) {
    return std::sqrt(x*x + y*y + z*z);
}

// ════════════════════════════════════════════════════════════════════════════
// Space Colonization Algorithm — branching ductal skeleton
//
// Model: tissue block sits downstream of the main pancreatic duct.
// Secondary branches enter from one face (x=0) and arborize throughout,
// producing interlobular → intralobular → terminal ductules.
// ════════════════════════════════════════════════════════════════════════════

struct Attractor {
    float x, y, z;
    bool  consumed;
};


static std::vector<DuctNode> generate_skeleton(
    int gx, int gy, int gz,
    float voxel_size_um,
    std::mt19937& rng)
{
    // ── Parameters ──────────────────────────────────────────────────────
    // Entry branches (secondary/interlobular): 1.0-1.5 mm diameter at entry,
    // tapering to 0.75-1.0 mm at their tips before further branching.
    // Radius tapers smoothly with cumulative path distance, not per-segment.
    const float domain_diag = std::sqrt(float(gx*gx + gy*gy + gz*gz));

    std::uniform_real_distribution<float> entry_r_dist(0.10f, 0.20f);  // mm radius at entry → 0.2-0.4 mm diam (5-10 vox at 20µm)
    const float end_radius       = 1.0f;    // terminal ductules: 1.0 voxel radius (~40 µm diam)
    const float min_duct_radius  = 0.8f;    // absolute minimum: ~32 µm diam

    // Space colonization parameters
    const float segment_length   = 4.0f;
    const float kill_distance    = segment_length * 1.2f;
    const float influence_radius = domain_diag * 0.25f;

    // Moderate attractor density
    const int   n_attractors     = static_cast<int>(gx * gy * gz * 0.001f);
    const int   max_iterations   = 3000;
    const float max_path_dist    = domain_diag * 0.75f;  // stop growing beyond this path length

    // Fewer entry branches — leave room for agents
    const int   n_entries        = std::max(3, static_cast<int>(std::sqrt(gy * gz) / 50.0f));

    // Direction noise added to each growth step (radians)
    std::normal_distribution<float> dir_noise(0.0f, 0.15f);

    std::cout << "  Entry branches: " << n_entries
              << ", attractors: " << n_attractors
              << ", influence_r: " << influence_radius << " vox" << std::endl;

    // ── Place attractors throughout the domain ──────────────────────────
    std::uniform_real_distribution<float> dist_x(gx * 0.02f, gx * 0.98f);
    std::uniform_real_distribution<float> dist_y(gy * 0.02f, gy * 0.98f);
    std::uniform_real_distribution<float> dist_z(gz * 0.02f, gz * 0.98f);

    std::vector<Attractor> attractors(n_attractors);
    for (auto& a : attractors) {
        a.x = dist_x(rng);
        a.y = dist_y(rng);
        a.z = dist_z(rng);
        a.consumed = false;
    }

    // ── Seed entry branches on x=0 face ─────────────────────────────────
    // Each branch enters at a random angle (not perpendicular) and may
    // start partially outside the domain (negative x, clamped at boundary).
    std::vector<DuctNode> nodes;
    std::vector<bool> is_tip;
    std::vector<float> node_path_dist;  // cumulative path distance from root

    std::uniform_real_distribution<float> angle_dist(-0.6f, 0.6f);   // entry angle spread (rad)
    std::uniform_real_distribution<float> jitter_y(gy * 0.08f, gy * 0.92f);
    std::uniform_real_distribution<float> jitter_z(gz * 0.08f, gz * 0.92f);

    int entries_placed = 0;
    for (int e = 0; e < n_entries; e++) {
        float entry_radius = entry_r_dist(rng) * 1000.0f / voxel_size_um;  // mm → voxels
        float ey = jitter_y(rng);
        float ez = jitter_z(rng);

        // Random entry direction: mostly +x but with y/z angular spread
        float angle_y = angle_dist(rng);
        float angle_z = angle_dist(rng);
        float dir_x = std::cos(angle_y) * std::cos(angle_z);
        float dir_y = std::sin(angle_y);
        float dir_z = std::sin(angle_z);
        float dlen = length3(dir_x, dir_y, dir_z);
        dir_x /= dlen; dir_y /= dlen; dir_z /= dlen;

        // Root node at domain boundary
        DuctNode root;
        root.x = 1.0f;
        root.y = ey;
        root.z = ez;
        root.radius = entry_radius;
        root.parent = -1;
        root.generation = 0;
        int root_idx = static_cast<int>(nodes.size());
        nodes.push_back(root);
        is_tip.push_back(false);
        node_path_dist.push_back(0.0f);

        // Grow initial trunk along entry direction (10-25% of domain) — short trunk, branch early
        std::uniform_real_distribution<float> pen_dist(0.10f, 0.25f);
        float penetration = gx * pen_dist(rng);
        int n_pen = static_cast<int>(penetration / segment_length);
        float cum_dist = 0;
        for (int i = 0; i < n_pen; i++) {
            // Add slight drift to direction
            float nx = nodes.back().x + (dir_x + dir_noise(rng) * 0.3f) * segment_length;
            float ny = nodes.back().y + (dir_y + dir_noise(rng) * 0.3f) * segment_length;
            float nz = nodes.back().z + (dir_z + dir_noise(rng) * 0.3f) * segment_length;
            nx = std::clamp(nx, 1.0f, static_cast<float>(gx - 2));
            ny = std::clamp(ny, 1.0f, static_cast<float>(gy - 2));
            nz = std::clamp(nz, 1.0f, static_cast<float>(gz - 2));

            cum_dist += segment_length;
            // Smooth taper: radius interpolates from entry to end over 30% of domain diagonal
            float taper_frac = std::min(1.0f, cum_dist / (domain_diag * 0.3f));
            float r = entry_radius * (1.0f - taper_frac) + end_radius * taper_frac;

            DuctNode n;
            n.x = nx; n.y = ny; n.z = nz;
            n.radius = std::max(min_duct_radius, r);
            n.parent = static_cast<int>(nodes.size()) - 1;
            n.generation = 0;
            nodes.push_back(n);
            is_tip.push_back(false);
            node_path_dist.push_back(cum_dist);

            // Consume nearby attractors along trunk
            for (auto& a : attractors) {
                if (!a.consumed && dist3(a.x, a.y, a.z, n.x, n.y, n.z) < kill_distance * 2.0f) {
                    a.consumed = true;
                }
            }
        }
        is_tip.back() = true;  // terminus is a tip

        // Mark every 3rd intermediate node as a branch point
        for (size_t i = root_idx + 3; i < nodes.size(); i += 3) {
            is_tip[i] = true;
        }

        entries_placed++;
    }

    std::cout << "  Placed " << entries_placed << " entry branches ("
              << nodes.size() << " initial nodes)" << std::endl;

    // ── Build tip index for fast lookup ──────────────────────────────────
    // Instead of scanning all nodes each iteration, maintain a compact list
    std::vector<int> tip_indices;
    for (size_t i = 0; i < is_tip.size(); i++) {
        if (is_tip[i]) tip_indices.push_back(static_cast<int>(i));
    }

    // ── Space colonization loop ─────────────────────────────────────────
    for (int iter = 0; iter < max_iterations; iter++) {
        if (tip_indices.empty()) break;

        // Count remaining attractors
        int n_remaining = 0;
        for (const auto& a : attractors) {
            if (!a.consumed) n_remaining++;
        }
        if (n_remaining == 0) break;

        // For each attractor, find closest tip within influence radius
        struct Vote { float dx, dy, dz; int count; };
        std::vector<Vote> votes(nodes.size(), {0, 0, 0, 0});

        for (const auto& a : attractors) {
            if (a.consumed) continue;

            int   closest_idx  = -1;
            float closest_dist = influence_radius;

            for (int ti : tip_indices) {
                float d = dist3(a.x, a.y, a.z,
                                nodes[ti].x, nodes[ti].y, nodes[ti].z);
                if (d < closest_dist) {
                    closest_dist = d;
                    closest_idx  = ti;
                }
            }

            if (closest_idx >= 0) {
                float dx = a.x - nodes[closest_idx].x;
                float dy = a.y - nodes[closest_idx].y;
                float dz = a.z - nodes[closest_idx].z;
                float len = length3(dx, dy, dz);
                if (len > 1e-6f) {
                    votes[closest_idx].dx += dx / len;
                    votes[closest_idx].dy += dy / len;
                    votes[closest_idx].dz += dz / len;
                    votes[closest_idx].count++;
                }
            }
        }

        // Grow new segments from tips with votes.
        // If a tip has votes from multiple divergent attractors, SPLIT into
        // two children (actual branching) rather than averaging the direction.
        bool any_growth = false;
        size_t n_before = nodes.size();
        std::vector<int> new_tips;

        // Helper to create a child node from a tip in a given direction
        auto make_child = [&](int ti, float dx, float dy, float dz) -> int {
            float nx = nodes[ti].x + dx;
            float ny = nodes[ti].y + dy;
            float nz = nodes[ti].z + dz;
            if (nx < 1 || nx >= gx-1 || ny < 1 || ny >= gy-1 || nz < 1 || nz >= gz-1)
                return -1;

            float parent_dist = node_path_dist[ti];
            float child_dist = parent_dist + segment_length;
            if (child_dist > max_path_dist) return -1;  // stop growing beyond limit
            float taper_frac = std::min(1.0f, child_dist / (domain_diag * 0.3f));

            // Walk to root for entry radius
            float root_r = nodes[ti].radius;
            { int ri = ti; while (nodes[ri].parent >= 0) ri = nodes[ri].parent; root_r = nodes[ri].radius; }

            float r = root_r * (1.0f - taper_frac) + end_radius * taper_frac;

            DuctNode child;
            child.x = nx; child.y = ny; child.z = nz;
            child.generation = nodes[ti].generation + 1;
            child.radius = std::max(min_duct_radius, r);
            child.parent = ti;
            int idx = static_cast<int>(nodes.size());
            nodes.push_back(child);
            is_tip.push_back(true);
            node_path_dist.push_back(child_dist);
            return idx;
        };

        for (int ti : tip_indices) {
            if (votes[ti].count == 0) continue;

            float dx = votes[ti].dx;
            float dy = votes[ti].dy;
            float dz = votes[ti].dz;
            float len = length3(dx, dy, dz);
            if (len < 1e-6f) continue;

            // Detect branching: if the aggregated vote vector is much shorter
            // than the vote count, attractors are pulling in divergent directions.
            // vote_coherence = |sum of unit vectors| / count. Low → divergent.
            float coherence = len / votes[ti].count;
            bool do_branch = (votes[ti].count >= 2 && coherence < 0.65f
                              && nodes[ti].radius > min_duct_radius * 1.5f);

            if (do_branch) {
                // Split into two children: averaged direction ± perpendicular offset
                float ndx = dx / len, ndy = dy / len, ndz = dz / len;
                // Find a perpendicular vector
                float px, py, pz;
                if (std::abs(ndx) < 0.9f) { px = 0; py = -ndz; pz = ndy; }
                else                       { px = -ndz; py = 0; pz = ndx; }
                float plen = length3(px, py, pz);
                if (plen > 1e-6f) { px /= plen; py /= plen; pz /= plen; }

                // Splay angle: ~30-50 degrees
                std::uniform_real_distribution<float> splay(0.4f, 0.7f);
                float s = splay(rng);
                // Random rotation of perpendicular around main axis
                float rot = std::uniform_real_distribution<float>(0.0f, 6.283f)(rng);
                float cos_r = std::cos(rot), sin_r = std::sin(rot);
                // Second perpendicular via cross product
                float qx = ndy * pz - ndz * py;
                float qy = ndz * px - ndx * pz;
                float qz = ndx * py - ndy * px;
                float offx = (px * cos_r + qx * sin_r) * s;
                float offy = (py * cos_r + qy * sin_r) * s;
                float offz = (pz * cos_r + qz * sin_r) * s;

                float d1x = (ndx + offx + dir_noise(rng)) * segment_length;
                float d1y = (ndy + offy + dir_noise(rng)) * segment_length;
                float d1z = (ndz + offz + dir_noise(rng)) * segment_length;
                float d2x = (ndx - offx + dir_noise(rng)) * segment_length;
                float d2y = (ndy - offy + dir_noise(rng)) * segment_length;
                float d2z = (ndz - offz + dir_noise(rng)) * segment_length;

                int c1 = make_child(ti, d1x, d1y, d1z);
                int c2 = make_child(ti, d2x, d2y, d2z);
                if (c1 >= 0) new_tips.push_back(c1);
                if (c2 >= 0) new_tips.push_back(c2);
                if (c1 >= 0 || c2 >= 0) {
                    is_tip[ti] = false;
                    any_growth = true;
                }
            } else {
                // Single growth: average direction + noise
                float d1x = (dx / len + dir_noise(rng)) * segment_length;
                float d1y = (dy / len + dir_noise(rng)) * segment_length;
                float d1z = (dz / len + dir_noise(rng)) * segment_length;
                int c = make_child(ti, d1x, d1y, d1z);
                if (c >= 0) {
                    new_tips.push_back(c);
                    is_tip[ti] = false;
                    any_growth = true;
                }
            }
        }

        // Update tip index: remove deactivated, add new
        {
            std::vector<int> updated;
            // Keep tips that weren't deactivated
            for (int ti : tip_indices) {
                if (is_tip[ti]) updated.push_back(ti);
            }
            // Add new tips
            for (int ti : new_tips) updated.push_back(ti);
            tip_indices = std::move(updated);
        }

        // Consume attractors near any new node
        for (auto& a : attractors) {
            if (a.consumed) continue;
            for (size_t ni = n_before; ni < nodes.size(); ni++) {
                if (dist3(a.x, a.y, a.z,
                          nodes[ni].x, nodes[ni].y, nodes[ni].z) < kill_distance) {
                    a.consumed = true;
                    break;
                }
            }
        }

        if (!any_growth) break;

        // Progress logging
        if (iter > 0 && iter % 500 == 0) {
            std::cout << "    iter " << iter << ": " << nodes.size() << " nodes, "
                      << tip_indices.size() << " tips, " << n_remaining << " attractors left"
                      << std::endl;
        }
    }

    // Stats
    int max_gen = 0;
    for (const auto& n : nodes) max_gen = std::max(max_gen, n.generation);
    float max_r = 0, min_r = 1e9f;
    for (const auto& n : nodes) { max_r = std::max(max_r, n.radius); min_r = std::min(min_r, n.radius); }
    std::cout << "  Ductal skeleton: " << nodes.size() << " nodes, "
              << max_gen << " max depth" << std::endl;
    std::cout << "    Radius range: " << min_r << " — " << max_r << " voxels ("
              << (min_r * 2 * voxel_size_um / 1000.0f) << " — "
              << (max_r * 2 * voxel_size_um / 1000.0f) << " mm diam)" << std::endl;

    return nodes;
}

// ════════════════════════════════════════════════════════════════════════════
// Rasterize skeleton to wall voxels + face flags
// ════════════════════════════════════════════════════════════════════════════

// Set a wall face between voxel (x,y,z) and its neighbor in dir (+/-1 on one axis).
// Sets the appropriate bit on BOTH sides of the shared face.
static void set_wall_face(std::vector<uint8_t>& face_flags,
                          int x, int y, int z,
                          int axis, int sign,  // axis: 0=x, 1=y, 2=z; sign: -1 or +1
                          int gx, int gy, int gz) {
    uint8_t src_bit;
    if (axis == 0) src_bit = (sign < 0) ? FACE_NEG_X : FACE_POS_X;
    else if (axis == 1) src_bit = (sign < 0) ? FACE_NEG_Y : FACE_POS_Y;
    else src_bit = (sign < 0) ? FACE_NEG_Z : FACE_POS_Z;

    int src_idx = vidx(x, y, z, gx, gy);
    face_flags[src_idx] |= src_bit;

    // Neighbor voxel gets the opposite face
    int nx = x + (axis == 0 ? sign : 0);
    int ny = y + (axis == 1 ? sign : 0);
    int nz = z + (axis == 2 ? sign : 0);
    if (nx >= 0 && nx < gx && ny >= 0 && ny < gy && nz >= 0 && nz < gz) {
        uint8_t dst_bit;
        if (axis == 0) dst_bit = (sign < 0) ? FACE_POS_X : FACE_NEG_X;
        else if (axis == 1) dst_bit = (sign < 0) ? FACE_POS_Y : FACE_NEG_Y;
        else dst_bit = (sign < 0) ? FACE_POS_Z : FACE_NEG_Z;
        face_flags[vidx(nx, ny, nz, gx, gy)] |= dst_bit;
    }
}

// Rasterize a cylinder (segment between two nodes).
// Marks interior as TISSUE_LUMEN and shell as TISSUE_WALL in a local working array.
// These labels are only used to determine where face flags go.
static void rasterize_segment(const DuctNode& a, const DuctNode& b,
                              std::vector<uint8_t>& wall_mask,
                              int gx, int gy, int gz) {
    float dx = b.x - a.x, dy = b.y - a.y, dz = b.z - a.z;
    float seg_len = length3(dx, dy, dz);
    if (seg_len < 0.01f) return;

    int n_samples = std::max(1, static_cast<int>(std::ceil(seg_len)));

    for (int s = 0; s <= n_samples; s++) {
        float t = static_cast<float>(s) / static_cast<float>(n_samples);
        float cx = a.x + dx * t;
        float cy = a.y + dy * t;
        float cz = a.z + dz * t;
        float r  = a.radius + (b.radius - a.radius) * t;

        int r_ceil = static_cast<int>(std::ceil(r)) + 1;
        int ix = static_cast<int>(std::round(cx));
        int iy = static_cast<int>(std::round(cy));
        int iz = static_cast<int>(std::round(cz));

        for (int vz = iz - r_ceil; vz <= iz + r_ceil; vz++) {
            for (int vy = iy - r_ceil; vy <= iy + r_ceil; vy++) {
                for (int vx = ix - r_ceil; vx <= ix + r_ceil; vx++) {
                    if (vx < 0 || vx >= gx || vy < 0 || vy >= gy || vz < 0 || vz >= gz)
                        continue;

                    float px = vx + 0.5f - a.x;
                    float py = vy + 0.5f - a.y;
                    float pz = vz + 0.5f - a.z;
                    float proj = (px*dx + py*dy + pz*dz) / (seg_len * seg_len);
                    proj = std::max(0.0f, std::min(1.0f, proj));
                    float closest_x = a.x + dx * proj;
                    float closest_y = a.y + dy * proj;
                    float closest_z = a.z + dz * proj;
                    float local_r = a.radius + (b.radius - a.radius) * proj;

                    float dist = dist3(vx + 0.5f, vy + 0.5f, vz + 0.5f,
                                       closest_x, closest_y, closest_z);

                    int idx = vidx(vx, vy, vz, gx, gy);

                    if (dist < local_r - 0.5f) {
                        // Interior: lumen (used for face flag computation)
                        wall_mask[idx] = TISSUE_LUMEN;
                    }
                    else if (dist < local_r + 0.5f) {
                        if (wall_mask[idx] != TISSUE_LUMEN) {
                            wall_mask[idx] = TISSUE_WALL;
                        }
                    }
                }
            }
        }
    }
}

// Build face flags at lumen/non-lumen boundaries.
static void build_face_flags_from_mask(std::vector<uint8_t>& face_flags,
                                       const std::vector<uint8_t>& wall_mask,
                                       int gx, int gy, int gz) {
    const int offsets[6][3] = {
        {-1,0,0}, {1,0,0}, {0,-1,0}, {0,1,0}, {0,0,-1}, {0,0,1}
    };

    for (int z = 0; z < gz; z++) {
        for (int y = 0; y < gy; y++) {
            for (int x = 0; x < gx; x++) {
                int idx = vidx(x, y, z, gx, gy);
                if (wall_mask[idx] != TISSUE_LUMEN) continue;

                for (int d = 0; d < 6; d++) {
                    int nx = x + offsets[d][0];
                    int ny = y + offsets[d][1];
                    int nz = z + offsets[d][2];

                    bool is_boundary = false;
                    if (nx < 0 || nx >= gx || ny < 0 || ny >= gy || nz < 0 || nz >= gz) {
                        is_boundary = true;
                    } else {
                        int nidx = vidx(nx, ny, nz, gx, gy);
                        if (wall_mask[nidx] != TISSUE_LUMEN) {
                            is_boundary = true;
                        }
                    }

                    if (is_boundary) {
                        int axis = d / 2;
                        int sign = (d % 2 == 0) ? -1 : +1;
                        set_wall_face(face_flags, x, y, z, axis, sign, gx, gy, gz);
                    }
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// Septum generation: Voronoi boundaries from terminal duct branches
// ════════════════════════════════════════════════════════════════════════════

static void generate_septum_field(std::vector<float>& septum_density,
                                  const std::vector<DuctNode>& /* nodes */,
                                  const std::vector<uint8_t>& wall_mask,
                                  int gx, int gy, int gz,
                                  std::mt19937& rng) {
    // Generate a jittered grid of Voronoi seeds with ~50-voxel spacing.
    // This produces lobular boundaries independent of the ductal tree topology.
    // Lobule diameter: 1-2 mm (50-100 voxels at 20µm). Use ~75 vox average spacing
    // with high jitter to get natural size variation.
    const float lobule_spacing = 75.0f;  // average lobule diameter in voxels (~1.5 mm)
    int nx_seeds = std::max(1, static_cast<int>(std::round(gx / lobule_spacing)));
    int ny_seeds = std::max(1, static_cast<int>(std::round(gy / lobule_spacing)));
    int nz_seeds = std::max(1, static_cast<int>(std::round(gz / lobule_spacing)));

    float cell_x = static_cast<float>(gx) / nx_seeds;
    float cell_y = static_cast<float>(gy) / ny_seeds;
    float cell_z = static_cast<float>(gz) / nz_seeds;

    // Jitter: up to ±45% of cell size — high jitter gives lobule size variation
    // (some ~50 vox, some ~100 vox, matching real 1-2 mm range)
    std::uniform_real_distribution<float> jx(-cell_x * 0.45f, cell_x * 0.45f);
    std::uniform_real_distribution<float> jy(-cell_y * 0.45f, cell_y * 0.45f);
    std::uniform_real_distribution<float> jz(-cell_z * 0.45f, cell_z * 0.45f);

    struct Seed3D { float x, y, z; };
    std::vector<Seed3D> seeds;
    seeds.reserve(nx_seeds * ny_seeds * nz_seeds);

    for (int iz = 0; iz < nz_seeds; iz++) {
        for (int iy = 0; iy < ny_seeds; iy++) {
            for (int ix = 0; ix < nx_seeds; ix++) {
                float sx = (ix + 0.5f) * cell_x + jx(rng);
                float sy = (iy + 0.5f) * cell_y + jy(rng);
                float sz = (iz + 0.5f) * cell_z + jz(rng);
                sx = std::clamp(sx, 0.5f, gx - 0.5f);
                sy = std::clamp(sy, 0.5f, gy - 0.5f);
                sz = std::clamp(sz, 0.5f, gz - 0.5f);
                seeds.push_back({sx, sy, sz});
            }
        }
    }

    std::cout << "  Septum: " << seeds.size() << " Voronoi seeds ("
              << nx_seeds << "x" << ny_seeds << "x" << nz_seeds
              << " grid, ~" << lobule_spacing << " vox spacing)" << std::endl;

    if (seeds.size() < 2) return;

    const float septum_thickness = 4.0f;  // voxels of transition zone
    const float septum_max_density = 0.6f;

    for (int z = 0; z < gz; z++) {
        for (int y = 0; y < gy; y++) {
            for (int x = 0; x < gx; x++) {
                int idx = vidx(x, y, z, gx, gy);
                if (wall_mask[idx] == TISSUE_LUMEN) continue;

                float vx = x + 0.5f, vy = y + 0.5f, vz = z + 0.5f;

                // Find two closest seeds
                float d1 = 1e9f, d2 = 1e9f;
                for (const auto& s : seeds) {
                    float d = dist3(vx, vy, vz, s.x, s.y, s.z);
                    if (d < d1) { d2 = d1; d1 = d; }
                    else if (d < d2) { d2 = d; }
                }

                float boundary_dist = (d2 - d1) * 0.5f;
                if (boundary_dist < septum_thickness) {
                    float frac = 1.0f - (boundary_dist / septum_thickness);
                    septum_density[idx] = septum_max_density * frac * frac;
                }
            }
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
// DuctalNetwork lifetime management
// ════════════════════════════════════════════════════════════════════════════

DuctalNetwork& DuctalNetwork::operator=(DuctalNetwork&& o) noexcept {
    if (this != &o) {
        if (d_face_flags)     cudaFree(d_face_flags);
        if (d_septum_density) cudaFree(d_septum_density);
        nodes = std::move(o.nodes);
        wall_mask = std::move(o.wall_mask);
        septum_density = std::move(o.septum_density);
        d_face_flags = o.d_face_flags;       o.d_face_flags = nullptr;
        d_septum_density = o.d_septum_density; o.d_septum_density = nullptr;
        grid_x = o.grid_x; grid_y = o.grid_y; grid_z = o.grid_z;
        total_voxels = o.total_voxels;
    }
    return *this;
}

DuctalNetwork::~DuctalNetwork() {
    if (d_face_flags)     cudaFree(d_face_flags);
    if (d_septum_density) cudaFree(d_septum_density);
}

// ════════════════════════════════════════════════════════════════════════════
// Public API
// ════════════════════════════════════════════════════════════════════════════

DuctalNetwork generate_ductal_network(
    int gx, int gy, int gz,
    float voxel_size_um,
    unsigned int seed)
{
    std::cout << "[DuctalInit] Generating ductal network for "
              << gx << "x" << gy << "x" << gz << " domain (seed=" << seed << ")"
              << std::endl;

    if (gx < DUCTAL_INIT_MIN_GRID || gy < DUCTAL_INIT_MIN_GRID || gz < DUCTAL_INIT_MIN_GRID) {
        std::cerr << "[DuctalInit] ERROR: all grid dimensions must be >= "
                  << DUCTAL_INIT_MIN_GRID << " for ductal initialization (-i 1). "
                  << "Got " << gx << "x" << gy << "x" << gz << ". "
                  << "Use -i 0 for small grids or increase -g." << std::endl;
        std::exit(1);
    }

    std::mt19937 rng(seed);
    int total = gx * gy * gz;

    // Step 1: Generate skeleton
    std::vector<DuctNode> nodes = generate_skeleton(gx, gy, gz, voxel_size_um, rng);

    // Step 2: Rasterize to working wall mask + face flags
    // wall_mask is local — only used for face flag construction, not exported
    std::vector<uint8_t> wall_mask(total, TISSUE_NONE);
    std::vector<uint8_t> h_face_flags(total, 0);
    std::vector<float>   h_septum(total, 0.0f);

    for (size_t i = 0; i < nodes.size(); i++) {
        if (nodes[i].parent < 0) continue;
        rasterize_segment(nodes[nodes[i].parent], nodes[i],
                          wall_mask, gx, gy, gz);
    }

    // Step 3: Build face flags from lumen boundaries
    build_face_flags_from_mask(h_face_flags, wall_mask, gx, gy, gz);

    // Step 4: Generate septum density
    generate_septum_field(h_septum, nodes, wall_mask, gx, gy, gz, rng);

    // Stats
    int n_lumen = 0, n_wall_vox = 0;
    for (int i = 0; i < total; i++) {
        if (wall_mask[i] == TISSUE_LUMEN) n_lumen++;
        else if (wall_mask[i] == TISSUE_WALL) n_wall_vox++;
    }
    int n_face_voxels = 0;
    for (int i = 0; i < total; i++) {
        if (h_face_flags[i] != 0) n_face_voxels++;
    }
    int n_septum_voxels = 0;
    for (int i = 0; i < total; i++) {
        if (h_septum[i] > 0.01f) n_septum_voxels++;
    }
    std::cout << "  Rasterized: " << n_lumen << " lumen, " << n_wall_vox << " wall voxels" << std::endl;
    std::cout << "  Face flags set on " << n_face_voxels << " voxels" << std::endl;
    std::cout << "  Septum density > 0.01 on " << n_septum_voxels << " voxels" << std::endl;

    // Step 5: Upload to GPU (face_flags + septum_density only)
    DuctalNetwork net;
    net.nodes = std::move(nodes);
    net.wall_mask = std::move(wall_mask);          // preserve for agent init
    net.septum_density = std::move(h_septum);      // preserve for fibroblast weighting
    net.grid_x = gx;
    net.grid_y = gy;
    net.grid_z = gz;
    net.total_voxels = total;

    cudaMalloc(&net.d_face_flags,     total * sizeof(uint8_t));
    cudaMalloc(&net.d_septum_density, total * sizeof(float));

    cudaMemcpy(net.d_face_flags,     h_face_flags.data(), total * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(net.d_septum_density, h_septum.data(),     total * sizeof(float),   cudaMemcpyHostToDevice);

    std::cout << "[DuctalInit] Complete. GPU memory: "
              << (total * (sizeof(uint8_t) + sizeof(float))) / (1024*1024)
              << " MB" << std::endl;

    return net;
}

void register_ductal_pointers(
    flamegpu::ModelDescription& model,
    const DuctalNetwork& net)
{
    // face_flags_ptr is registered with default 0 in model_definition.cu;
    // overwrite with the actual device pointer here.
    model.Environment().setProperty<uint64_t>(
        "face_flags_ptr",
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(net.d_face_flags)));
    model.Environment().newProperty<uint64_t>(
        "septum_density_ptr",
        static_cast<uint64_t>(reinterpret_cast<uintptr_t>(net.d_septum_density)));
}

} // namespace PDAC
