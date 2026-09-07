// Cross-type Ripley's L(r) - r diagnostic (2D, per-z-slice pooled).
//
// For each z-slice the 3D count grid is treated as an independent 2D field.
// Cross-spectra are accumulated in frequency domain across slices and a
// single IFFT yields the pooled cross-correlation — only same-z pairs
// contribute, matching what a single tissue section (IMC) would capture.
//
// Edge correction uses the 2D "translation" estimator: at lag (dx,dy)
// (gx−|dx|)(gy−|dy|) voxel-column pairs are valid.
//
// K_AB(r) = A / Σ_z(N_A_z N_B_z) × Σ_{|d|≤r} C_pooled(d) × w(d)
// L_AB(r) = sqrt( K_AB(r) / π )
// Diagnostic = L(r) − r   (>0 clustered, <0 regular, 0 Poisson)

#include "../core/common.cuh"
#include "../pde/pde_integration.cuh"
#include "ripley.cuh"

#include <cufft.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <vector>

namespace PDAC {
std::vector<int> g_ripley_counts[N_RIPLEY_TYPE];
}  // definition of extern declared in ripley.cuh

namespace {

// ── distance bins in voxel units ──
constexpr int MAX_RADII = 50;
int g_n_radii = 0;
int g_radii[MAX_RADII];

// ── state ──
int g_gx = 0, g_gy = 0, g_gz = 0;
int g_px = 0, g_py = 0;        // 2D padded dims (2× for acyclic correlation)
size_t g_pvox2d = 0;            // padded 2D total pixels
size_t g_fft_complex = 0;       // complex elements after 2D R2C
std::ofstream g_ripley_csv;
std::string g_ripley_csv_path;
bool g_initialised = false;

// cuFFT 2D plans
cufftHandle g_plan_r2c;
cufftHandle g_plan_c2r;

// Device buffers (2D)
cufftReal*    d_real_a     = nullptr;
cufftReal*    d_corr       = nullptr;
cufftComplex* d_freq_a     = nullptr;   // temp for one slice FFT
cufftComplex* d_freq_b     = nullptr;   // temp for one slice FFT
cufftComplex* d_freq_accum = nullptr;   // accumulated cross-spectrum across z

// Host buffers
std::vector<float> h_corr;
std::vector<int>   h_slice;  // single z-slice 2D count grid (gx × gy)

// Per-type, per-slice FFTs cached on host
// Layout: h_freq_slices[type * gz + z], each of size g_fft_complex
std::vector<std::vector<cufftComplex>> h_freq_slices;
// Per-type, per-slice agent counts
std::vector<double> h_Nz;  // [type * gz + z]

// ── precomputed 2D radial bins ──
struct RadialBin {
    float r;
    std::vector<int> offsets;
    std::vector<double> edge_weight;
};
std::vector<RadialBin> g_bins;

int pad_idx_2d(int dx, int dy) {
    int wx = (dx + g_px) % g_px;
    int wy = (dy + g_py) % g_py;
    return wy * g_px + wx;
}

void build_radial_bins() {
    int min_dim = std::min(g_gx, g_gy);
    g_n_radii = std::min(min_dim / 2, MAX_RADII);
    if (g_n_radii < 1) g_n_radii = 1;
    for (int i = 0; i < g_n_radii; ++i)
        g_radii[i] = i + 1;

    g_bins.resize(g_n_radii);
    float r_prev = 0.0f;
    for (int b = 0; b < g_n_radii; ++b) {
        float r_hi = static_cast<float>(g_radii[b]);
        g_bins[b].r = r_hi;
        float r_hi2 = r_hi * r_hi;
        float r_lo2 = r_prev * r_prev;
        int ir = g_radii[b];
        double A = static_cast<double>(g_gx) * g_gy;

        for (int dy = -ir; dy <= ir; ++dy)
        for (int dx = -ir; dx <= ir; ++dx) {
            if (dx == 0 && dy == 0) continue;
            float d2 = static_cast<float>(dx*dx + dy*dy);
            if (d2 > r_lo2 && d2 <= r_hi2) {
                g_bins[b].offsets.push_back(pad_idx_2d(dx, dy));
                double valid = static_cast<double>(g_gx - std::abs(dx))
                             * (g_gy - std::abs(dy));
                g_bins[b].edge_weight.push_back(A / valid);
            }
        }
        r_prev = r_hi;
    }
}

// ── CUDA kernels ──

__global__ void pointwise_mul_conj_accum(cufftComplex* accum,
                                         const cufftComplex* a,
                                         const cufftComplex* b,
                                         float inv_n, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float ar = a[i].x, ai = a[i].y;
    float br = b[i].x, bi = b[i].y;
    accum[i].x += (ar * br + ai * bi) * inv_n;
    accum[i].y += (ai * br - ar * bi) * inv_n;
}

__global__ void scatter_to_padded_2d(cufftReal* padded,
                                     const int* counts_2d,
                                     int gx, int gy, int px) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= gx * gy) return;
    int y = v / gx;
    int x = v - y * gx;
    int pi = y * px + x;
    padded[pi] = static_cast<cufftReal>(counts_2d[v]);
}

// Device buffer for slice count grid
int* d_counts_2d = nullptr;

void fft_one_slice(const int* host_slice, cufftComplex* d_freq_out) {
    const int nxy = g_gx * g_gy;
    cudaMemcpy(d_counts_2d, host_slice, nxy * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_real_a, 0, g_pvox2d * sizeof(cufftReal));
    int block = 256;
    int grid = (nxy + block - 1) / block;
    scatter_to_padded_2d<<<grid, block>>>(d_real_a, d_counts_2d, g_gx, g_gy, g_px);
    cudaDeviceSynchronize();
    cufftExecR2C(g_plan_r2c, d_real_a, d_freq_out);
    cudaDeviceSynchronize();
}

}  // anonymous namespace

namespace PDAC {

void ripley_init(int gx, int gy, int gz,
                 const std::string& output_root, int seed) {
    g_gx = gx; g_gy = gy; g_gz = gz;
    g_px = 2 * gx; g_py = 2 * gy;
    g_pvox2d = static_cast<size_t>(g_px) * g_py;
    g_fft_complex = static_cast<size_t>(g_px) * (g_py / 2 + 1);

    cufftPlan2d(&g_plan_r2c, g_px, g_py, CUFFT_R2C);
    cufftPlan2d(&g_plan_c2r, g_px, g_py, CUFFT_C2R);

    cudaMalloc(&d_real_a,      g_pvox2d * sizeof(cufftReal));
    cudaMalloc(&d_corr,        g_pvox2d * sizeof(cufftReal));
    cudaMalloc(&d_freq_a,      g_fft_complex * sizeof(cufftComplex));
    cudaMalloc(&d_freq_b,      g_fft_complex * sizeof(cufftComplex));
    cudaMalloc(&d_freq_accum,  g_fft_complex * sizeof(cufftComplex));
    cudaMalloc(&d_counts_2d,   gx * gy * sizeof(int));

    h_corr.resize(g_pvox2d);
    h_slice.resize(gx * gy);
    h_freq_slices.resize(N_RIPLEY_TYPE * gz);
    h_Nz.resize(N_RIPLEY_TYPE * gz, 0.0);

    build_radial_bins();

    g_ripley_csv_path = output_root + "/ripley_" + std::to_string(seed) + ".csv";
    g_initialised = true;
}

void ripley_cleanup() {
    if (!g_initialised) return;
    if (d_real_a)      { cudaFree(d_real_a);      d_real_a      = nullptr; }
    if (d_corr)        { cudaFree(d_corr);        d_corr        = nullptr; }
    if (d_freq_a)      { cudaFree(d_freq_a);      d_freq_a      = nullptr; }
    if (d_freq_b)      { cudaFree(d_freq_b);      d_freq_b      = nullptr; }
    if (d_freq_accum)  { cudaFree(d_freq_accum);  d_freq_accum  = nullptr; }
    if (d_counts_2d)   { cudaFree(d_counts_2d);   d_counts_2d   = nullptr; }
    cufftDestroy(g_plan_r2c);
    cufftDestroy(g_plan_c2r);
    if (g_ripley_csv.is_open()) g_ripley_csv.close();
    g_initialised = false;
}

FLAMEGPU_HOST_FUNCTION(compute_ripley) {
    if (!g_initialised) return;

    ripley_fill_counts(FLAMEGPU, g_gx, g_gy, g_gz);

    const int nxy = g_gx * g_gy;
    const double A = static_cast<double>(g_gx) * g_gy;
    const float inv_n = 1.0f / static_cast<float>(g_pvox2d);

    // ── Phase 1: FFT every z-slice of every type, cache on host ──
    double N_total[N_RIPLEY_TYPE] = {};
    for (int t = 0; t < N_RIPLEY_TYPE; ++t) {
        const auto& c3d = g_ripley_counts[t];
        for (int z = 0; z < g_gz; ++z) {
            int idx = t * g_gz + z;
            const int* slice_ptr = c3d.data() + z * nxy;
            double nz = 0;
            for (int i = 0; i < nxy; ++i) nz += slice_ptr[i];
            h_Nz[idx] = nz;
            N_total[t] += nz;

            if (nz < 1) {
                h_freq_slices[idx].clear();
                continue;
            }
            h_freq_slices[idx].resize(g_fft_complex);
            fft_one_slice(slice_ptr, d_freq_a);
            cudaMemcpy(h_freq_slices[idx].data(), d_freq_a,
                       g_fft_complex * sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        }
    }

    // ── CSV header (once) ──
    if (!g_ripley_csv.is_open()) {
        g_ripley_csv.open(g_ripley_csv_path);
        g_ripley_csv << "step";
        for (int a = 0; a < N_RIPLEY_TYPE; ++a)
        for (int b = a; b < N_RIPLEY_TYPE; ++b) {
            const char* na = ripley_type_name(a);
            const char* nb = ripley_type_name(b);
            for (int ri = 0; ri < g_n_radii; ++ri)
                g_ripley_csv << ",L_" << na << "_" << nb
                             << "_r" << g_radii[ri];
        }
        for (int t = 0; t < N_RIPLEY_TYPE; ++t)
            g_ripley_csv << ",n_" << ripley_type_name(t);
        g_ripley_csv << "\n";
    }

    g_ripley_csv << FLAMEGPU->getStepCounter();

    int block = 256;
    int freq_grid = (static_cast<int>(g_fft_complex) + block - 1) / block;

    // ── Phase 2: for each pair, accumulate cross-spectrum across z ──
    for (int a = 0; a < N_RIPLEY_TYPE; ++a)
    for (int b = a; b < N_RIPLEY_TYPE; ++b) {
        // Pair normalization: Σ_z N_a_z × N_b_z  (auto: N_z × (N_z - 1))
        bool is_auto = (a == b);
        double denom = 0;
        for (int z = 0; z < g_gz; ++z) {
            double na_z = h_Nz[a * g_gz + z];
            double nb_z = h_Nz[b * g_gz + z];
            denom += is_auto ? na_z * (na_z - 1.0) : na_z * nb_z;
        }

        if (denom < 1.0) {
            for (int ri = 0; ri < g_n_radii; ++ri)
                g_ripley_csv << ",nan";
            continue;
        }

        // Zero accumulator
        cudaMemset(d_freq_accum, 0, g_fft_complex * sizeof(cufftComplex));

        // Accumulate cross-spectrum across z-slices
        for (int z = 0; z < g_gz; ++z) {
            int ia = a * g_gz + z;
            int ib = b * g_gz + z;
            if (h_freq_slices[ia].empty() || h_freq_slices[ib].empty())
                continue;

            cudaMemcpy(d_freq_a, h_freq_slices[ia].data(),
                       g_fft_complex * sizeof(cufftComplex), cudaMemcpyHostToDevice);
            cudaMemcpy(d_freq_b, h_freq_slices[ib].data(),
                       g_fft_complex * sizeof(cufftComplex), cudaMemcpyHostToDevice);

            pointwise_mul_conj_accum<<<freq_grid, block>>>(
                d_freq_accum, d_freq_a, d_freq_b, inv_n,
                static_cast<int>(g_fft_complex));
            cudaDeviceSynchronize();
        }

        // Single IFFT of accumulated cross-spectrum
        cufftExecC2R(g_plan_c2r, d_freq_accum, d_corr);
        cudaDeviceSynchronize();

        cudaMemcpy(h_corr.data(), d_corr,
                   g_pvox2d * sizeof(cufftReal), cudaMemcpyDeviceToHost);

        // Cumulative K with 2D edge correction
        double K_cum = 0.0;
        const double norm = A / denom;

        for (int ri = 0; ri < g_n_radii; ++ri) {
            double shell_sum = 0.0;
            const auto& bin = g_bins[ri];
            for (size_t j = 0; j < bin.offsets.size(); ++j)
                shell_sum += static_cast<double>(h_corr[bin.offsets[j]])
                           * bin.edge_weight[j];

            K_cum += shell_sum * norm;
            double r = static_cast<double>(g_radii[ri]);
            double L = std::sqrt(K_cum / M_PI);
            g_ripley_csv << "," << (L - r);
        }
    }

    for (int t = 0; t < N_RIPLEY_TYPE; ++t)
        g_ripley_csv << "," << static_cast<int>(N_total[t]);

    g_ripley_csv << "\n";
    g_ripley_csv.flush();
}

}  // namespace PDAC
