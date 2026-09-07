#ifndef PDAC_DIAGNOSTICS_RIPLEY_CUH
#define PDAC_DIAGNOSTICS_RIPLEY_CUH

// Cross-type Ripley's L(r) - r diagnostic.
//
// Measures spatial clustering between all unique agent-subtype pairs at
// multiple distance scales. Computed on the host from per-subtype voxel
// count grids, using FFT-based cross-correlation (cuFFT) for efficiency.
//
// Output: one CSV row per step with L(r)-r for every (pair, r) combination.
// Toggle: --ripley CLI flag enables computation (off by default).

#include "flamegpu/flamegpu.h"
#include <string>
#include <vector>

namespace PDAC {

// Agent subtypes tracked by the Ripley diagnostic.
// Order here defines column order in the output CSV.
enum RipleyType : int {
    RIP_CANCER = 0,
    RIP_CD8,
    RIP_TREG,
    RIP_TH,
    RIP_M1,
    RIP_M2,
    RIP_ICAF,
    RIP_MYCAF,
    RIP_APCAF,
    RIP_MDSC,
    RIP_BCELL,
    RIP_DC,
    N_RIPLEY_TYPE
};

inline const char* ripley_type_name(int t) {
    static const char* names[] = {
        "cancer", "cd8", "treg", "th", "m1", "m2",
        "icaf", "mycaf", "apcaf", "mdsc", "bcell", "dc"
    };
    return (t >= 0 && t < N_RIPLEY_TYPE) ? names[t] : "?";
}

void ripley_init(int gx, int gy, int gz, const std::string& output_root,
                 int seed);
void ripley_cleanup();

// Per-subtype voxel count grids used by the Ripley diagnostic.
// Filled by ripley_fill_counts() in pde_integration.cu.
extern std::vector<int> g_ripley_counts[N_RIPLEY_TYPE];

extern flamegpu::FLAMEGPU_HOST_FUNCTION_POINTER compute_ripley;

}  // namespace PDAC

#endif  // PDAC_DIAGNOSTICS_RIPLEY_CUH
