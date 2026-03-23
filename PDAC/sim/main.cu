#include "flamegpu/flamegpu.h"
#include <iostream>
#include <memory>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>
#include <nvtx3/nvToolsExt.h>
#include <chrono>
#include <filesystem>
#include <thread>

#include "third_party/lz4/lz4.h"

#include "../core/common.cuh"
#include "../core/layer_timing.h"
#include "../pde/pde_integration.cuh"
#include "initialization.cuh"
#include "gpu_param.h"
#include "../qsp/LymphCentral_wrapper.h"
#include "../core/model_functions.cuh"

constexpr int ABM_EXPORT_NCOLS = 8; // must match pack_for_export.cuh

// Exposed by qsp_integration.cu — true during Phase 3 pre-simulation
namespace PDAC {
extern bool is_presim_mode_active();
extern double get_last_pde_ms();
extern double get_last_qsp_ms();
extern RecruitStats get_last_recruit_stats();
}

// QSP CSV export step function (defined in qsp_integration.cu)
extern flamegpu::FLAMEGPU_STEP_FUNCTION_POINTER exportQSPData;
// QSP step-0 export (initial condition, called before main loop)
namespace PDAC {
    extern void exportQSPData_step0();
    extern void set_qsp_output_path(const std::string& path);
}

namespace PDAC {
    std::unique_ptr<flamegpu::ModelDescription> buildModel(
        int grid_x, int grid_y, int grid_z, float voxel_size,
        const PDAC::GPUParam& gpu_params);

    // void set_internal_params(flamegpu::ModelDescription& model, 
    //                          const LymphCentralWrapper& lymph);
}

// ============================================================================
// Output Directory Management
// ============================================================================

// Ensure output directories exist, creating them if necessary
void ensureOutputDirectories() {
    try {
        std::filesystem::create_directories("outputs/pde");
        std::filesystem::create_directories("outputs/abm");
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not create output directories: " << e.what() << std::endl;
    }
}

// ============================================================================
// Simulation Monitoring Functions
// ============================================================================

// ============================================================================
// LZ4-compressed binary writer for PDE output
//
// File format (.pde.lz4):
//   Bytes 0-3:   magic "PDE1"
//   Bytes 4-7:   grid_x (int32)
//   Bytes 8-11:  grid_y (int32)
//   Bytes 12-15: grid_z (int32)
//   Bytes 16-19: num_substrates (int32)
//   Bytes 20-23: uncompressed_size (int32, bytes)
//   Bytes 24-27: compressed_size (int32, bytes)
//   Bytes 28+:   LZ4-compressed float32 data
//
// Python reader:
//   import lz4.block, struct, numpy as np
//   with open("pde_step_000001.pde.lz4", "rb") as f:
//       magic = f.read(4)
//       gx, gy, gz, ns, raw_sz, comp_sz = struct.unpack('<6i', f.read(24))
//       data = np.frombuffer(lz4.block.decompress(f.read(), raw_sz), dtype=np.float32)
//       data = data.reshape(ns, gz, gy, gx)
// ============================================================================
static void write_pde_lz4(const char* path, int grid_x, int grid_y, int grid_z,
                           const float* data, size_t n_floats) {
    const int ns = PDAC::NUM_SUBSTRATES;
    const int raw_bytes = static_cast<int>(n_floats * sizeof(float));
    const int max_comp = LZ4_compressBound(raw_bytes);

    std::vector<char> comp_buf(max_comp);
    int comp_bytes = LZ4_compress_default(
        reinterpret_cast<const char*>(data), comp_buf.data(), raw_bytes, max_comp);

    if (comp_bytes <= 0) {
        std::cerr << "[WARN] LZ4 compression failed for: " << path << std::endl;
        return;
    }

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        std::cerr << "[WARN] Could not open file for write: " << path << std::endl;
        return;
    }

    fwrite("PDE1", 1, 4, fp);
    fwrite(&grid_x, 4, 1, fp);
    fwrite(&grid_y, 4, 1, fp);
    fwrite(&grid_z, 4, 1, fp);
    fwrite(&ns, 4, 1, fp);
    fwrite(&raw_bytes, 4, 1, fp);
    fwrite(&comp_bytes, 4, 1, fp);
    fwrite(comp_buf.data(), 1, comp_bytes, fp);
    fclose(fp);
}

// ============================================================================
// Async I/O with pinned memory + CUDA streams
//
// Optimizations over the previous version:
//   1. PDE: single cudaMemcpyAsync of all substrates (was 10 separate cudaMemcpy)
//   2. Pinned (page-locked) host memory for DMA-capable async D2H transfers
//   3. Dedicated CUDA stream so D2H can overlap with next step's GPU compute
//   4. Double-buffered: one buffer receives D2H while the other is written to disk
// ============================================================================

// Pinned-memory double buffers for PDE output
static float*   g_pde_pinned[2] = {nullptr, nullptr};
static size_t   g_pde_buf_floats = 0;       // NUM_SUBSTRATES * total_voxels
static cudaStream_t g_pde_stream = nullptr;  // dedicated D2H stream
static cudaEvent_t  g_pde_event  = nullptr;  // signals D2H completion

// ABM GPU-side packing: device buffer + atomic counter + pinned host double buffers
static int32_t*      g_abm_device_buf = nullptr;   // device buffer for packed agent data
static unsigned int* g_abm_device_counter = nullptr; // atomic row counter on device
static int32_t*      g_abm_pinned[2] = {nullptr, nullptr}; // pinned host double buffers
static size_t        g_abm_max_agents = 0;          // max agents the buffer can hold
static std::vector<int32_t> g_abm_bufs[2];          // fallback for step-0 (before sim starts)

static std::vector<float>   g_ecm_bufs[2];
static std::thread g_pde_io_thread;
static std::thread g_abm_io_thread;
static std::thread g_ecm_io_thread;
static int g_pde_buf_idx = 0;
static int g_abm_buf_idx = 0;
static int g_ecm_buf_idx = 0;

// Allocate pinned PDE buffers and CUDA stream (call once after grid size is known)
static void init_pde_io(int grid_x, int grid_y, int grid_z) {
    g_pde_buf_floats = static_cast<size_t>(PDAC::NUM_SUBSTRATES) * grid_x * grid_y * grid_z;
    size_t bytes = g_pde_buf_floats * sizeof(float);
    cudaMallocHost(&g_pde_pinned[0], bytes);
    cudaMallocHost(&g_pde_pinned[1], bytes);
    cudaStreamCreateWithFlags(&g_pde_stream, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&g_pde_event, cudaEventDisableTiming);
}

// Allocate GPU buffer + pinned host buffers for ABM export
static void init_abm_io(size_t max_agents) {
    g_abm_max_agents = max_agents;
    size_t buf_bytes = max_agents * ABM_EXPORT_NCOLS * sizeof(int32_t);
    cudaMalloc(&g_abm_device_buf, buf_bytes);
    cudaMalloc(&g_abm_device_counter, sizeof(unsigned int));
    cudaMemset(g_abm_device_counter, 0, sizeof(unsigned int));
    cudaMallocHost(&g_abm_pinned[0], buf_bytes);
    cudaMallocHost(&g_abm_pinned[1], buf_bytes);
}

static void cleanup_abm_io() {
    if (g_abm_device_buf) { cudaFree(g_abm_device_buf); g_abm_device_buf = nullptr; }
    if (g_abm_device_counter) { cudaFree(g_abm_device_counter); g_abm_device_counter = nullptr; }
    if (g_abm_pinned[0]) { cudaFreeHost(g_abm_pinned[0]); g_abm_pinned[0] = nullptr; }
    if (g_abm_pinned[1]) { cudaFreeHost(g_abm_pinned[1]); g_abm_pinned[1] = nullptr; }
}

// Free pinned PDE buffers and stream (call at cleanup)
static void cleanup_pde_io() {
    if (g_pde_pinned[0]) { cudaFreeHost(g_pde_pinned[0]); g_pde_pinned[0] = nullptr; }
    if (g_pde_pinned[1]) { cudaFreeHost(g_pde_pinned[1]); g_pde_pinned[1] = nullptr; }
    if (g_pde_stream) { cudaStreamDestroy(g_pde_stream); g_pde_stream = nullptr; }
    if (g_pde_event)  { cudaEventDestroy(g_pde_event);  g_pde_event  = nullptr; }
}

static void flush_async_io() {
    if (g_pde_io_thread.joinable()) g_pde_io_thread.join();
    if (g_abm_io_thread.joinable()) g_abm_io_thread.join();
    if (g_ecm_io_thread.joinable()) g_ecm_io_thread.join();
}

// Collect all PDE substrates via single async D2H into pinned buffer, then
// launch background file write once D2H completes.
// Kick off async D2H + background write (caller must join previous thread first)
static void export_pde_async_no_join(int grid_x, int grid_y, int grid_z, const std::string& path) {
    int bi = g_pde_buf_idx;
    float* dst = g_pde_pinned[bi];

    // Single async D2H of all substrates on dedicated stream
    PDAC::g_pde_solver->get_all_concentrations_async(dst, g_pde_stream);
    cudaEventRecord(g_pde_event, g_pde_stream);

    size_t n_floats = g_pde_buf_floats;
    g_pde_io_thread = std::thread([dst, n_floats, path, grid_x, grid_y, grid_z]() {
        // Wait for D2H to finish (non-spinning — yields CPU)
        cudaEventSynchronize(g_pde_event);
        write_pde_lz4(path.c_str(), grid_x, grid_y, grid_z, dst, n_floats);
    });
    g_pde_buf_idx = 1 - bi;
}

// Called manually from main() after presim completes — captures true day-0 PDE state.
void exportPDEData_step0(int grid_x, int grid_y, int grid_z) {
    ensureOutputDirectories();
    if (!PDAC::g_pde_solver) return;
    if (g_pde_io_thread.joinable()) g_pde_io_thread.join();
    export_pde_async_no_join(grid_x, grid_y, grid_z, "outputs/pde/pde_step_000000.pde.lz4");
}

FLAMEGPU_STEP_FUNCTION(exportPDEData) {
    if (PDAC::is_presim_mode_active()) return;
    if (!PDAC::g_pde_solver) return;

    const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");
    const int interval = FLAMEGPU->environment.getProperty<int>("interval_out");
    if (main_step % interval != 0) return;

    ensureOutputDirectories();

    auto t0 = std::chrono::high_resolution_clock::now();

    if (g_pde_io_thread.joinable()) g_pde_io_thread.join();

    auto t1 = std::chrono::high_resolution_clock::now();

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    char path_buf[256];
    snprintf(path_buf, sizeof(path_buf), "outputs/pde/pde_step_%06d.pde.lz4",
             static_cast<int>(main_step + 1));
    export_pde_async_no_join(grid_x, grid_y, grid_z, std::string(path_buf));

    auto t2 = std::chrono::high_resolution_clock::now();
    PDAC::g_layer_timings.push_back({"io_pde_join",
        std::chrono::duration<double, std::milli>(t1 - t0).count()});
    PDAC::g_layer_timings.push_back({"io_pde_export",
        std::chrono::duration<double, std::milli>(t2 - t1).count()});
}

// ============================================================================
// NPY Writer for ECM (stroma) Output
// Writes ECM density and fibroblast density field as a single NPY file.
// Shape: (2, grid_z, grid_y, grid_x), dtype float32, C-order.
//   channel 0: ECM_density  (d_ecm_grid)
//   channel 1: Fib_field    (d_fib_density_field)
// Python: arr = np.load("ecm_step_000001.npy")  → shape (2, nz, ny, nx)
// ============================================================================
static void write_ecm_npy_buf(const char* path, int grid_x, int grid_y, int grid_z,
                               const std::vector<float>& buf) {
    char header_str[256];
    int header_len = snprintf(header_str, sizeof(header_str),
        "{'descr': '<f4', 'fortran_order': False, 'shape': (2, %d, %d, %d), }",
        grid_z, grid_y, grid_x);

    const int prefix_size = 10;
    int padded_header_len = header_len + 1;
    int block = ((prefix_size + padded_header_len + 63) / 64) * 64;
    int pad_spaces = block - prefix_size - padded_header_len;

    FILE* fp = fopen(path, "wb");
    if (!fp) {
        std::cerr << "[WARN] Could not open ECM NPY file for write: " << path << std::endl;
        return;
    }
    const unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 0x01, 0x00};
    fwrite(magic, 1, 8, fp);
    uint16_t hdr_total = static_cast<uint16_t>(padded_header_len + pad_spaces);
    fwrite(&hdr_total, sizeof(uint16_t), 1, fp);
    fwrite(header_str, 1, header_len, fp);
    for (int i = 0; i < pad_spaces; i++) fputc(' ', fp);
    fputc('\n', fp);
    fwrite(buf.data(), sizeof(float), buf.size(), fp);
    fclose(fp);
}

// Collect ECM + fib density from device into a host buffer.
// buf layout: [ecm_grid (total_voxels floats), fib_field (total_voxels floats)]
static void collect_ecm_to_buf(std::vector<float>& buf, int grid_x, int grid_y, int grid_z) {
    const int total_voxels = grid_x * grid_y * grid_z;
    buf.resize(static_cast<size_t>(2) * total_voxels);
    float* d_ecm = PDAC::get_ecm_grid_device_ptr();
    float* d_fib = PDAC::get_fib_density_field_device_ptr();
    if (d_ecm) cudaMemcpy(buf.data(),                  d_ecm, total_voxels * sizeof(float), cudaMemcpyDeviceToHost);
    if (d_fib) cudaMemcpy(buf.data() + total_voxels,   d_fib, total_voxels * sizeof(float), cudaMemcpyDeviceToHost);
}

void exportECMData_step0(int grid_x, int grid_y, int grid_z) {
    try { std::filesystem::create_directories("outputs/ecm"); } catch (...) {}
    if (g_ecm_io_thread.joinable()) g_ecm_io_thread.join();
    int bi = g_ecm_buf_idx;
    collect_ecm_to_buf(g_ecm_bufs[bi], grid_x, grid_y, grid_z);
    std::string path = "outputs/ecm/ecm_step_000000.npy";
    g_ecm_io_thread = std::thread([bi, path, grid_x, grid_y, grid_z]() {
        write_ecm_npy_buf(path.c_str(), grid_x, grid_y, grid_z, g_ecm_bufs[bi]);
    });
    g_ecm_buf_idx = 1 - bi;
}

FLAMEGPU_STEP_FUNCTION(exportECMData) {
    if (PDAC::is_presim_mode_active()) return;

    const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");
    const int interval = FLAMEGPU->environment.getProperty<int>("interval_out");
    if (main_step % interval != 0) return;

    try { std::filesystem::create_directories("outputs/ecm"); } catch (...) {}

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    if (g_ecm_io_thread.joinable()) g_ecm_io_thread.join();
    int bi = g_ecm_buf_idx;
    collect_ecm_to_buf(g_ecm_bufs[bi], grid_x, grid_y, grid_z);

    char path_buf[256];
    snprintf(path_buf, sizeof(path_buf), "outputs/ecm/ecm_step_%06d.npy",
             static_cast<int>(main_step + 1));
    std::string path = path_buf;
    g_ecm_io_thread = std::thread([bi, path, grid_x, grid_y, grid_z]() {
        write_ecm_npy_buf(path.c_str(), grid_x, grid_y, grid_z, g_ecm_bufs[bi]);
    });
    g_ecm_buf_idx = 1 - bi;
}

// ============================================================================
// Binary NPY ABM Output
//
// Shape: (N_agents, 8), dtype int32
// Columns: [type_id, agent_id, x, y, z, cell_state, life, extra]
//   type_id:    0=CANCER 1=TCELL 2=TREG 3=MDSC 4=MAC 5=FIB 6=VAS
//   cell_state: enum int (STEM=0/PROG=1/SEN=2; EFF=0/CYT=1/SUPP=2; etc.)
//   life:       age counter (0 for cancer/vascular which don't use it)
//   extra:      divideCD for cancer, 0 for all others
//
// Python:
//   import numpy as np, pandas as pd
//   arr = np.load("agents_step_000001.npy")
//   df = pd.DataFrame(arr, columns=["type","id","x","y","z","state","life","extra"])
// ============================================================================
static constexpr int ABM_NCOLS = 8;
static constexpr int32_t ABM_TYPE_CANCER = 0;
static constexpr int32_t ABM_TYPE_TCELL  = 1;
static constexpr int32_t ABM_TYPE_TREG   = 2;
static constexpr int32_t ABM_TYPE_MDSC   = 3;
static constexpr int32_t ABM_TYPE_MAC    = 4;
static constexpr int32_t ABM_TYPE_FIB    = 5;
static constexpr int32_t ABM_TYPE_VAS    = 6;

// g_abm_buf kept for step-0 (uses AgentVector API outside step functions)
static std::vector<int32_t> g_abm_buf;

// ============================================================================
// LZ4-compressed binary writer for ABM output
//
// File format (.abm.lz4):
//   Bytes 0-3:   magic "ABM1"
//   Bytes 4-7:   n_agents (int32)
//   Bytes 8-11:  n_cols (int32)
//   Bytes 12-15: uncompressed_size (int32, bytes)
//   Bytes 16-19: compressed_size (int32, bytes)
//   Bytes 20+:   LZ4-compressed int32 data
//
// Python reader:
//   import lz4.block, struct, numpy as np
//   with open("agents_step_000001.abm.lz4", "rb") as f:
//       magic = f.read(4)
//       n, nc, raw_sz, comp_sz = struct.unpack('<4i', f.read(16))
//       data = np.frombuffer(lz4.block.decompress(f.read(), raw_sz), dtype=np.int32)
//       data = data.reshape(n, nc)
// ============================================================================
static void write_abm_lz4_buf(const char* path, const int32_t* data, int n_agents) {
    int raw_bytes = n_agents * ABM_NCOLS * static_cast<int>(sizeof(int32_t));
    int max_comp = LZ4_compressBound(raw_bytes);

    std::vector<char> comp_buf(max_comp);
    int comp_bytes = LZ4_compress_default(
        reinterpret_cast<const char*>(data), comp_buf.data(), raw_bytes, max_comp);

    if (comp_bytes <= 0) {
        std::cerr << "[WARN] LZ4 compression failed for: " << path << std::endl;
        return;
    }

    FILE* fp = fopen(path, "wb");
    if (!fp) { std::cerr << "[WARN] Cannot open " << path << "\n"; return; }

    fwrite("ABM1", 1, 4, fp);
    fwrite(&n_agents, 4, 1, fp);
    int ncols = ABM_NCOLS;
    fwrite(&ncols, 4, 1, fp);
    fwrite(&raw_bytes, 4, 1, fp);
    fwrite(&comp_bytes, 4, 1, fp);
    fwrite(comp_buf.data(), 1, comp_bytes, fp);
    fclose(fp);
}

// Vector overload for step-0 export (uses CPU-side collection)
static void write_abm_lz4(const char* path, const std::vector<int32_t>& buf) {
    write_abm_lz4_buf(path, buf.data(), static_cast<int>(buf.size()) / ABM_NCOLS);
}

static inline void abm_push(std::vector<int32_t>& buf,
    int32_t type, int32_t id,
    int32_t x, int32_t y, int32_t z,
    int32_t state, int32_t life, int32_t extra)
{
    buf.push_back(type); buf.push_back(id);
    buf.push_back(x);    buf.push_back(y);    buf.push_back(z);
    buf.push_back(state); buf.push_back(life); buf.push_back(extra);
}

// Step-0 version: uses host-side AgentVector API (called outside step functions)
static void collect_abm_step0(flamegpu::CUDASimulation& sim,
                               flamegpu::ModelDescription& model,
                               std::vector<int32_t>& buf) {
    buf.clear();
    {
        flamegpu::AgentVector p(model.Agent(PDAC::AGENT_CANCER_CELL));
        sim.getPopulationData(p);
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_CANCER, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                p[i].getVariable<int>("cell_state"), 0,
                p[i].getVariable<int>("divideCD"));
    }
    {
        flamegpu::AgentVector p(model.Agent(PDAC::AGENT_TCELL));
        sim.getPopulationData(p);
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_TCELL, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                p[i].getVariable<int>("cell_state"), p[i].getVariable<int>("life"), 0);
    }
    {
        flamegpu::AgentVector p(model.Agent(PDAC::AGENT_TREG));
        sim.getPopulationData(p);
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_TREG, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                p[i].getVariable<int>("cell_state"), p[i].getVariable<int>("life"), 0);
    }
    {
        flamegpu::AgentVector p(model.Agent(PDAC::AGENT_MDSC));
        sim.getPopulationData(p);
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_MDSC, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                0, p[i].getVariable<int>("life"), 0);
    }
    {
        flamegpu::AgentVector p(model.Agent(PDAC::AGENT_MACROPHAGE));
        sim.getPopulationData(p);
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_MAC, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                p[i].getVariable<int>("cell_state"), p[i].getVariable<int>("life"), 0);
    }
    {
        flamegpu::AgentVector p(model.Agent(PDAC::AGENT_FIBROBLAST));
        sim.getPopulationData(p);
        for (unsigned i = 0; i < p.size(); ++i) {
            const int32_t id    = (int32_t)p[i].getID();
            const int     state = p[i].getVariable<int>("cell_state");
            const int     life  = p[i].getVariable<int>("life");
            const int     clen  = p[i].getVariable<int>("chain_len");
            const auto    sx    = p[i].getVariable<int, PDAC::MAX_FIB_CHAIN_LENGTH>("seg_x");
            const auto    sy    = p[i].getVariable<int, PDAC::MAX_FIB_CHAIN_LENGTH>("seg_y");
            const auto    sz    = p[i].getVariable<int, PDAC::MAX_FIB_CHAIN_LENGTH>("seg_z");
            for (int s = 0; s < clen; ++s)
                abm_push(buf, ABM_TYPE_FIB, id, sx[s], sy[s], sz[s], state, life, 0);
        }
    }
    {
        flamegpu::AgentVector p(model.Agent(PDAC::AGENT_VASCULAR));
        sim.getPopulationData(p);
        for (unsigned i = 0; i < p.size(); ++i) {
            int st = p[i].getVariable<int>("cell_state");
            if (st == 1) continue;
            abm_push(buf, ABM_TYPE_VAS, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                st, 0, 0);
        }
    }
}

// In-step version: uses DeviceAgentVector API, writes to provided buffer
static void collect_abm_step(flamegpu::HostAPI* FLAMEGPU, std::vector<int32_t>& buf) {
    buf.clear();
    {
        flamegpu::DeviceAgentVector p = FLAMEGPU->agent(PDAC::AGENT_CANCER_CELL).getPopulationData();
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_CANCER, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                p[i].getVariable<int>("cell_state"), 0,
                p[i].getVariable<int>("divideCD"));
    }
    {
        flamegpu::DeviceAgentVector p = FLAMEGPU->agent(PDAC::AGENT_TCELL).getPopulationData();
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_TCELL, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                p[i].getVariable<int>("cell_state"), p[i].getVariable<int>("life"), 0);
    }
    {
        flamegpu::DeviceAgentVector p = FLAMEGPU->agent(PDAC::AGENT_TREG).getPopulationData();
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_TREG, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                p[i].getVariable<int>("cell_state"), p[i].getVariable<int>("life"), 0);
    }
    {
        flamegpu::DeviceAgentVector p = FLAMEGPU->agent(PDAC::AGENT_MDSC).getPopulationData();
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_MDSC, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                0, p[i].getVariable<int>("life"), 0);
    }
    {
        flamegpu::DeviceAgentVector p = FLAMEGPU->agent(PDAC::AGENT_MACROPHAGE).getPopulationData();
        for (unsigned i = 0; i < p.size(); ++i)
            abm_push(buf, ABM_TYPE_MAC, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                p[i].getVariable<int>("cell_state"), p[i].getVariable<int>("life"), 0);
    }
    {
        flamegpu::DeviceAgentVector p = FLAMEGPU->agent(PDAC::AGENT_FIBROBLAST).getPopulationData();
        for (unsigned i = 0; i < p.size(); ++i) {
            const int32_t id    = (int32_t)p[i].getID();
            const int     state = p[i].getVariable<int>("cell_state");
            const int     life  = p[i].getVariable<int>("life");
            const int     clen  = p[i].getVariable<int>("chain_len");
            for (int s = 0; s < clen; ++s)
                abm_push(buf, ABM_TYPE_FIB, id,
                    p[i].getVariable<int, PDAC::MAX_FIB_CHAIN_LENGTH>("seg_x", s),
                    p[i].getVariable<int, PDAC::MAX_FIB_CHAIN_LENGTH>("seg_y", s),
                    p[i].getVariable<int, PDAC::MAX_FIB_CHAIN_LENGTH>("seg_z", s),
                    state, life, 0);
        }
    }
    {
        flamegpu::DeviceAgentVector p = FLAMEGPU->agent(PDAC::AGENT_VASCULAR).getPopulationData();
        for (unsigned i = 0; i < p.size(); ++i) {
            int st = p[i].getVariable<int>("cell_state");
            if (st == 1) continue;
            abm_push(buf, ABM_TYPE_VAS, (int32_t)p[i].getID(),
                p[i].getVariable<int>("x"), p[i].getVariable<int>("y"), p[i].getVariable<int>("z"),
                st, 0, 0);
        }
    }
}

// Host function: prepare GPU buffer for ABM export (runs as a layer before pack_for_export)
FLAMEGPU_HOST_FUNCTION(prepare_abm_export) {
    if (PDAC::is_presim_mode_active()) {
        FLAMEGPU->environment.setProperty<int>("do_abm_export", 0);
        return;
    }
    const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");
    const int interval = FLAMEGPU->environment.getProperty<int>("interval_out");
    if (main_step % interval != 0) {
        FLAMEGPU->environment.setProperty<int>("do_abm_export", 0);
        return;
    }
    // Reset counter and enable export for this step
    cudaMemset(g_abm_device_counter, 0, sizeof(unsigned int));
    FLAMEGPU->environment.setProperty<int>("do_abm_export", 1);
}

// Called manually from main() after presim completes — captures true day-0 agent state.
void exportABMData_step0(flamegpu::CUDASimulation& sim, flamegpu::ModelDescription& model) {
    ensureOutputDirectories();
    if (g_abm_io_thread.joinable()) g_abm_io_thread.join();
    int bi = g_abm_buf_idx;
    collect_abm_step0(sim, model, g_abm_bufs[bi]);
    std::string path = "outputs/abm/agents_step_000000.abm.lz4";
    g_abm_io_thread = std::thread([bi, path]() {
        write_abm_lz4(path.c_str(), g_abm_bufs[bi]);
    });
    g_abm_buf_idx = 1 - bi;
}

// Diagnostic: dump cancer cell initial distributions to CSV for comparison with HCC.
// Writes: cell_state, divideCD, divideCountRemaining, life (senescent cells only)
static void write_cancer_init_diagnostic(flamegpu::CUDASimulation& sim,
                                          flamegpu::ModelDescription& model) {
    flamegpu::AgentVector p(model.Agent(PDAC::AGENT_CANCER_CELL));
    sim.getPopulationData(p);

    FILE* fp = fopen("outputs/abm/cancer_init_diagnostic.csv", "w");
    if (!fp) { std::cerr << "[WARN] Cannot open cancer_init_diagnostic.csv\n"; return; }
    fprintf(fp, "cell_state,divideCD,divideCountRemaining,life\n");
    for (unsigned i = 0; i < p.size(); ++i) {
        const int state = p[i].getVariable<int>("cell_state");
        const int dcd   = p[i].getVariable<int>("divideCD");
        const int dcr   = p[i].getVariable<int>("divideCountRemaining");
        const int life  = p[i].getVariable<int>("life");
        fprintf(fp, "%d,%d,%d,%d\n", state, dcd, dcr, life);
    }
    fclose(fp);
    std::cout << "[DIAG] Wrote cancer_init_diagnostic.csv (" << p.size() << " cells)\n";
}

FLAMEGPU_STEP_FUNCTION(exportABMData) {
    if (PDAC::is_presim_mode_active()) return;
    const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");
    const int interval = FLAMEGPU->environment.getProperty<int>("interval_out");
    if (main_step % interval != 0) return;
    ensureOutputDirectories();

    auto t0 = std::chrono::high_resolution_clock::now();

    if (g_abm_io_thread.joinable()) g_abm_io_thread.join();

    auto t1 = std::chrono::high_resolution_clock::now();

    // Read agent count from GPU counter
    unsigned int n_agents = 0;
    cudaMemcpy(&n_agents, g_abm_device_counter, sizeof(unsigned int), cudaMemcpyDeviceToHost);

    // Async D2H of packed buffer into pinned host buffer
    int bi = g_abm_buf_idx;
    size_t copy_bytes = static_cast<size_t>(n_agents) * ABM_EXPORT_NCOLS * sizeof(int32_t);
    cudaMemcpyAsync(g_abm_pinned[bi], g_abm_device_buf, copy_bytes,
                    cudaMemcpyDeviceToHost, g_pde_stream);
    cudaEventRecord(g_pde_event, g_pde_stream);

    auto t2 = std::chrono::high_resolution_clock::now();

    char path_buf[256];
    snprintf(path_buf, sizeof(path_buf), "outputs/abm/agents_step_%06d.abm.lz4",
             static_cast<int>(main_step + 1));
    std::string path = path_buf;
    int32_t* host_buf = g_abm_pinned[bi];
    int n_agents_copy = static_cast<int>(n_agents);
    g_abm_io_thread = std::thread([host_buf, n_agents_copy, path]() {
        cudaEventSynchronize(g_pde_event);
        write_abm_lz4_buf(path.c_str(), host_buf, n_agents_copy);
    });
    g_abm_buf_idx = 1 - bi;

    double join_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double collect_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    PDAC::g_layer_timings.push_back({"io_abm_join", join_ms});
    PDAC::g_layer_timings.push_back({"io_abm_collect", collect_ms});
}

FLAMEGPU_STEP_FUNCTION(stepCounter) {
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    FLAMEGPU->environment.setProperty<unsigned int>("current_step", step + 1);
    
    // Suppress output during Phase 3 pre-simulation warmup
    if (PDAC::is_presim_mode_active()) return;

    const unsigned int main_step = FLAMEGPU->environment.getProperty<unsigned int>("main_sim_step");

    // Compute treatment day from main_sim_step (0 = start of Phase 4)
    const float dt_abm  = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    const float treat_day = static_cast<float>(main_step) * dt_abm / 86400.0f;

    // Agent counts
    const unsigned int cancer_count = FLAMEGPU->agent(PDAC::AGENT_CANCER_CELL).count();
    const unsigned int tcell_count  = FLAMEGPU->agent(PDAC::AGENT_TCELL).count();
    const unsigned int treg_count   = FLAMEGPU->agent(PDAC::AGENT_TREG).count();
    const unsigned int mdsc_count   = FLAMEGPU->agent(PDAC::AGENT_MDSC).count();
    const unsigned int mac_count   = FLAMEGPU->agent(PDAC::AGENT_MACROPHAGE).count();
    const unsigned int fib_count   = FLAMEGPU->agent(PDAC::AGENT_FIBROBLAST).count();
    const unsigned int vas_count   = FLAMEGPU->agent(PDAC::AGENT_VASCULAR).count();

    // QSP state (set by solve_qsp_step each step)
    const float tum_vol  = FLAMEGPU->environment.getProperty<float>("qsp_tum_vol");
    const float cc_tumor = FLAMEGPU->environment.getProperty<float>("qsp_cc_tumor");
    const float nivo     = FLAMEGPU->environment.getProperty<float>("qsp_nivo_tumor");
    const float cabo     = FLAMEGPU->environment.getProperty<float>("qsp_cabo_tumor");
    const float teff_t   = FLAMEGPU->environment.getProperty<float>("qsp_teff_tumor");
    const float treg_t   = FLAMEGPU->environment.getProperty<float>("qsp_treg_tumor");
    const float mdsc_t   = FLAMEGPU->environment.getProperty<float>("qsp_mdsc_tumor");

    std::cout << std::fixed << std::setprecision(2)
              << "[Day " << std::setw(7) << treat_day << "]" << std::endl;
    std::cout << std::fixed << std::setprecision(2) << "[ABM] CC=" << cancer_count 
              << "  TC=" << tcell_count
              << "  TR=" << treg_count
              << "  MD=" << mdsc_count
              << "  MAC=" << mac_count
              << "  FIB=" << fib_count
              << "  VAS=" << vas_count
              << std::endl;

    std::cout << std::scientific << std::setprecision(2) << "[QSP] vol=" << tum_vol
              << " cc=" << cc_tumor
              << " nivo=" << nivo
              << " cabo=" << cabo
              << " Teff=" << teff_t
              << " Treg=" << treg_t
              << " MDSC=" << mdsc_t
              << std::endl;

    FLAMEGPU->environment.setProperty<unsigned int>("main_sim_step", main_step + 1);
}

FLAMEGPU_EXIT_CONDITION(checkSimulationEnd) {
    const unsigned int cancer_count = FLAMEGPU->agent(PDAC::AGENT_CANCER_CELL).count();
    if (cancer_count == 0) {
        std::cout << "\nAll cancer cells eliminated!" << std::endl;
        return flamegpu::EXIT;
    }
    return flamegpu::CONTINUE;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, const char** argv) {
    auto start = std::chrono::high_resolution_clock::now();

    // Check for -p flag (XML path override)
    // Default: resource/param_all_test.xml relative to the executable location
    std::string exe_dir;
    {
        std::filesystem::path exe_path = std::filesystem::canonical("/proc/self/exe");
        exe_dir = exe_path.parent_path().parent_path().parent_path().string(); // bin/ -> build/ -> sim/
    }
    std::string param_file = exe_dir + "/resource/param_all_test.xml";
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "-p" && i + 1 < argc) {
            param_file = argv[++i];
            break;
        }
    }

    // Load XML parameters
    std::cout << "Loading parameters from: " << param_file << std::endl;
    PDAC::GPUParam gpu_params;
    try {
        gpu_params.initializeParams(param_file);
        std::cout << "Parameters loaded successfully." << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "ERROR: Failed to load parameters from XML: " << e.what() << std::endl;
        return 1;
    }

    // Parse configuration from command line
    PDAC::SimulationConfig config;
    config.parseCommandLine(argc, argv, gpu_params);
    config.print();

    // Seed random number generator
    srand(config.random_seed);

    // ========== INITIALIZATION TIMING ==========
    std::ofstream init_file("outputs/init_timing.csv");
    init_file << "phase,ms\n";
    auto init_t0 = std::chrono::high_resolution_clock::now();
    auto init_lap = [&](const std::string& label) {
        auto init_t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(init_t1 - init_t0).count();
        init_file << label << "," << ms << "\n";
        init_file.flush();
        init_t0 = init_t1;
    };

    // ========== BUILD MODEL ==========
    std::cout << "Building FLAME GPU 2 model..." << std::endl;
    auto model = PDAC::buildModel(
        config.grid_x, config.grid_y, config.grid_z,
        config.voxel_size,
        gpu_params);

    // Store output interval in environment for step functions
    model->Environment().newProperty<int>("interval_out", config.interval_out);
    init_lap("build_model");

    // ========== INITIALIZE PDE SOLVER ==========
    std::cout << "Initializing PDE solver..." << std::endl;
    PDAC::initialize_pde_solver(
        config.grid_x, config.grid_y, config.grid_z,
        config.voxel_size, config.dt_abm, config.molecular_steps,
         gpu_params);

    // Store PDE device pointers in model environment
    PDAC::set_pde_pointers_in_environment(*model);

    // Allocate pinned host buffers and CUDA stream for async PDE output
    if (config.grid_out & 2) {
        init_pde_io(config.grid_x, config.grid_y, config.grid_z);
    }
    init_lap("init_pde");

    // ========== GPU MEMORY QUERY (after PDE allocation) ==========
    size_t free_mem_1, total_mem_1;
    cudaMemGetInfo(&free_mem_1, &total_mem_1);
    size_t used_mem_1 = total_mem_1 - free_mem_1;
    std::cout << "[MEM] After PDE init: " << (used_mem_1 / (1024*1024)) << " MB used / "
              << (total_mem_1 / (1024*1024)) << " MB total" << std::endl;

    // Process internal parameters from env params and new QSP params
    // ========== INITIALIZE QSP SOLVER ==========
    PDAC::LymphCentralWrapper _lymph;
    _lymph.initialize(param_file);
    PDAC::set_internal_params(*model, _lymph);
    PDAC::set_lymph_pointer(&_lymph);  // Set global pointer for QSP host functions
    init_lap("init_qsp");

    // ========== ADD STEP FUNCTIONS ==========
    if (config.grid_out & 2) {
        model->addStepFunction(exportPDEData);
        model->addStepFunction(exportECMData);
    }
    if (config.grid_out & 1) {
        model->addStepFunction(exportABMData);
    }
    model->addStepFunction(exportQSPData);
    model->addStepFunction(stepCounter);
    model->addExitCondition(checkSimulationEnd);

    // ========== ALLOCATE GPU BUFFER FOR ABM EXPORT ==========
    // Always allocate (pack_for_export layer runs unconditionally; agents check do_abm_export flag)
    {
        size_t max_agents = static_cast<size_t>(config.grid_x) * config.grid_y * config.grid_z * 2;
        init_abm_io(max_agents);
        model->Environment().setProperty<uint64_t>("abm_export_buf_ptr",
            reinterpret_cast<uint64_t>(g_abm_device_buf));
        model->Environment().setProperty<uint64_t>("abm_export_counter_ptr",
            reinterpret_cast<uint64_t>(g_abm_device_counter));
    }

    // ========== ALLOCATE GPU MEMORY FOR EVENT/STATE COUNTERS ==========
    // Do this BEFORE creating CUDASimulation so environment properties are synced
    unsigned int* device_event_counters = nullptr;
    unsigned int* device_state_counters = nullptr;
    cudaMalloc(&device_event_counters, PDAC::ABM_EVENT_COUNTER_SIZE * sizeof(unsigned int));
    cudaMalloc(&device_state_counters, PDAC::ABM_STATE_COUNTER_SIZE * sizeof(unsigned int));
    cudaMemset(device_event_counters, 0, PDAC::ABM_EVENT_COUNTER_SIZE * sizeof(unsigned int));
    cudaMemset(device_state_counters, 0, PDAC::ABM_STATE_COUNTER_SIZE * sizeof(unsigned int));

    // Store base pointers in model environment (before CUDASimulation init)
    model->Environment().setProperty<uint64_t>("event_counters_ptr",
        reinterpret_cast<uint64_t>(device_event_counters));
    model->Environment().setProperty<uint64_t>("state_counters_ptr",
        reinterpret_cast<uint64_t>(device_state_counters));

    // ========== CREATE SIMULATION ==========
    // Increase CUDA per-thread stack size for complex kernels (default 1KB is too small
    // for cancer_cell_state_step with inlined Newton-Raphson double-precision math)
    cudaDeviceSetLimit(cudaLimitStackSize, 16384);  // 16KB per thread
    std::cout << "Creating CUDA simulation..." << std::endl;
    flamegpu::CUDASimulation simulation(*model);
    simulation.SimulationConfig().steps = config.steps;
    simulation.SimulationConfig().random_seed = config.random_seed;
    init_lap("cuda_sim_create");

    // ========== INITIALIZE AGENTS ==========
    if (config.init_method == 1) {
        std::cout << "Initializing agents from QSP steady-state (init_method=1)..." << std::endl;
        PDAC::initializeToQSP(simulation, *model, config, _lymph);
    } else if (config.init_method == 2) {
        std::cout << "Initializing for neighbor scan test (init_method=2)..." << std::endl;
        PDAC::initializeNeighborTest(simulation, *model, config);
    } else {
        std::cout << "Initializing agents with default distribution (init_method=0)..." << std::endl;
        PDAC::initializeAllAgents(simulation, *model, config);
    }
    std::cout << "[DEBUG] Agent initialization complete" << std::endl;
    std::cout.flush();
    init_lap("init_agents");

    // ========== GPU MEMORY QUERY (after agent allocation) ==========
    size_t free_mem_2, total_mem_2;
    cudaMemGetInfo(&free_mem_2, &total_mem_2);
    size_t used_mem_2 = total_mem_2 - free_mem_2;
    std::cout << "[MEM] After agent init: " << (used_mem_2 / (1024*1024)) << " MB used / "
              << (total_mem_2 / (1024*1024)) << " MB total" << std::endl;

    // ========== PHASE 3: PRE-SIMULATION (QSP-seeded init only) ==========
    // Run ABM+QSP (no drugs) until QSP tumor volume reaches 1.0× target diameter.
    // This fills the gap between the 0.95× warmup and the treatment start.
    if (config.init_method == 1) {
        const double full_target_vol = _lymph.get_full_target_volume();
        double cur_vol = _lymph.get_tumor_volume();

        std::cout << "\n=== Phase 3: Pre-simulation (ABM+QSP, no drugs) ===" << std::endl;
        std::cout << "  Target volume (1.0x diam): " << full_target_vol << " cm^3" << std::endl;
        std::cout << "  Current QSP volume       : " << cur_vol         << " cm^3" << std::endl;

        _lymph.set_presimulation_mode(true);

        const unsigned int max_presim_steps = 100000;
        unsigned int presim_step = 0;

        while (cur_vol < full_target_vol && presim_step < max_presim_steps) {
            // std::cout << "[DEBUG] About to call simulation.step() for presim step " << presim_step << std::endl;
            // std::cout.flush();
            bool ok = simulation.step();
            // std::cout << "[DEBUG] Returned from simulation.step() successfully" << std::endl;
            // std::cout.flush();
            if (!ok) {
                std::cout << "  Pre-simulation: ABM terminated early (all cancer cells gone)" << std::endl;
                break;
            }
            cur_vol = _lymph.get_tumor_volume();
            presim_step++;

            if (presim_step % 50 == 0) {
                std::cout << "  Presim step " << presim_step
                          << ": QSP tum_vol=" << cur_vol
                          << " cm^3  (target=" << full_target_vol << ")" << std::endl;
            }
        }

        _lymph.set_presimulation_mode(false);

        std::cout << "  Pre-simulation complete: " << presim_step << " steps, "
                  << "QSP tum_vol=" << cur_vol << " cm^3" << std::endl;
        init_lap("presim");
    } else {
        init_lap("presim");  // Log presim time even if not run
    }
    init_file.close();

    // ========== BUILD SEED-STAMPED FILE NAMES ==========
    char stats_path[256], timing_path[256], qsp_seed_path[256];
    snprintf(stats_path,    sizeof(stats_path),    "outputs/stats_%u.csv",  config.random_seed);
    snprintf(timing_path,   sizeof(timing_path),   "outputs/timing_%u.csv", config.random_seed);
    snprintf(qsp_seed_path, sizeof(qsp_seed_path), "outputs/qsp_%u.csv",    config.random_seed);
    PDAC::set_qsp_output_path(qsp_seed_path);

    // ========== EXPORT DAY-0 STATE (after presim, before first treatment step) ==========
    if (config.grid_out & 2) exportPDEData_step0(config.grid_x, config.grid_y, config.grid_z);
    if (config.grid_out & 2) exportECMData_step0(config.grid_x, config.grid_y, config.grid_z);
    if (config.grid_out & 1) exportABMData_step0(simulation, *model);
    write_cancer_init_diagnostic(simulation, *model);
    PDAC::exportQSPData_step0();

    // ========== RUN SIMULATION ==========
    std::cout << "\n=== Starting Simulation ===" << std::endl;

    // Write NPY definition text files once at init
    {
        std::filesystem::create_directories("outputs");
        {
            std::ofstream f("outputs/abm_npy_def.txt");
            f << "ABM snapshot NPY definition\n"
              << "dtype: int32, shape: (N_agents, 8)\n"
              << "Columns: [type_id, agent_id, x, y, z, cell_state, life, extra]\n"
              << "  type_id:    0=CANCER 1=TCELL 2=TREG 3=MDSC 4=MAC 5=FIB 6=VAS\n"
              << "  cell_state: state enum per type\n"
              << "    CANCER: STEM=0, PROGENITOR=1, SENESCENT=2\n"
              << "    TCELL:  EFFECTOR=0, CYTOTOXIC=1, SUPPRESSED=2\n"
              << "    TREG:   TH=0, TREG=1\n"
              << "    MDSC:   (single state=0)\n"
              << "    MAC:    M1=0, M2=1\n"
              << "    FIB:    NORMAL=0, CAF=1  (one row per segment in chain)\n"
              << "    VAS:    TIP=0, PHALANX=2\n"
              << "  life:  age counter (steps alive)\n"
              << "  extra: divideCD for CANCER, 0 otherwise\n"
              << "Loading: arr=np.load('agents_step_000001.npy')\n"
              << "         df=pd.DataFrame(arr, columns=['type','id','x','y','z','state','life','extra'])\n";
        }
        {
            std::ofstream f("outputs/pde_npy_def.txt");
            f << "PDE concentration NPY definition\n"
              << "dtype: float32, shape: (NUM_SUBSTRATES, grid_z, grid_y, grid_x)\n"
              << "NUM_SUBSTRATES = 10\n"
              << "Channel index -> chemical:\n"
              << "  0: O2\n"
              << "  1: IFNg\n"
              << "  2: IL2\n"
              << "  3: IL10\n"
              << "  4: TGFB\n"
              << "  5: CCL2\n"
              << "  6: ArgI\n"
              << "  7: NO\n"
              << "  8: IL12\n"
              << "  9: VEGFA\n"
              << "Loading: arr=np.load('pde_step_000001.npy')  # shape (10,nz,ny,nx)\n"
              << "         o2=arr[0]  # shape (nz,ny,nx)\n";
        }
        {
            std::ofstream f("outputs/ecm_npy_def.txt");
            f << "ECM snapshot NPY definition\n"
              << "dtype: float32, shape: (2, grid_z, grid_y, grid_x)\n"
              << "Channel index -> field:\n"
              << "  0: ECM_density   (d_ecm_grid, smoothed ECM density [0..1])\n"
              << "  1: Fib_field     (d_fib_density_field, raw Gaussian fib density)\n"
              << "Loading: arr=np.load('ecm_step_000001.npy')  # shape (2,nz,ny,nx)\n"
              << "         ecm=arr[0]; fib=arr[1]\n";
        }
    }

    // Open stats output file (always written, seed-stamped)
    std::ofstream stats_file(stats_path);
    if (stats_file.is_open()) {
        stats_file << "step,"
            // Agent counts by state
            << "agentCount.cancer.stem,agentCount.cancer.prog,agentCount.cancer.sen,"
            << "agentCount.CD8.effector,agentCount.CD8.cytotoxic,agentCount.CD8.suppressed,"
            << "agentCount.Th.default,agentCount.Treg.default,"
            << "agentCount.MDSC.default,"
            << "agentCount.MAC.M1,agentCount.MAC.M2,"
            << "agentCount.FIB.normal,agentCount.FIB.CAF,"
            << "agentCount.VAS.tip,agentCount.VAS.default,"
            // Recruitment
            << "recruit.CD8.effector,recruit.Th.default,recruit.Treg.default,"
            << "recruit.MDSC.default,recruit.MAC.M1,recruit.MAC.M2,"
            // Proliferation by state
            << "prolif.CD8.effector,prolif.CD8.cytotoxic,prolif.CD8.suppressed,"
            << "prolif.Th.default,prolif.Treg.default,"
            << "prolif.MDSC.default,"
            << "prolif.cancer.stem,prolif.cancer.prog,prolif.cancer.sen,"
            << "prolif.MAC.M1,prolif.MAC.M2,"
            << "prolif.FIB.normal,prolif.FIB.CAF,"
            << "prolif.VAS.tip,prolif.VAS.phalanx,"
            // Death by state
            << "death.CD8.effector,death.CD8.cytotoxic,death.CD8.suppressed,"
            << "death.Th.default,death.Treg.default,"
            << "death.MDSC.default,"
            << "death.cancer.stem,death.cancer.prog,death.cancer.sen,"
            << "death.MAC.M1,death.MAC.M2,"
            << "death.FIB.normal,death.FIB.CAF,"
            << "death.VAS.tip,death.VAS.phalanx,"
            // PDL1 fraction
            << "PDL1_frac\n";
        // Step 0: initial counts (all events zero)
        // (State counts not available yet before first broadcast — write zeros)
        stats_file << "0";
        for (int c = 0; c < 15 + 6 + 15 + 15 + 1; c++) stats_file << ",0";
        stats_file << "\n";
        stats_file.flush();
    }

    // Open timing output file for per-step timing CSV
    std::ofstream timing_file(timing_path);
    timing_file << "step,total_ms,pde_ms,qsp_ms,abm_ms\n";

    // Open per-layer timing CSV (long format: step, layer_name, time_ms)
    std::ofstream layer_file("outputs/layer_timing.csv");
    layer_file << "step,layer,ms\n";

    // Manual stepping loop with NVTX markers for profiling
    const unsigned int total_steps = simulation.SimulationConfig().steps;
    for (unsigned int i = 0; i < total_steps; i++) {
        nvtxRangePush("ABM Step");
        auto step_t0 = std::chrono::high_resolution_clock::now();
        bool continue_sim = simulation.step();
        auto step_t1 = std::chrono::high_resolution_clock::now();
        nvtxRangePop();

        // Capture high-level timing
        double step_ms = std::chrono::duration<double, std::milli>(step_t1 - step_t0).count();
        double pde_ms = PDAC::get_last_pde_ms();
        double qsp_ms = PDAC::get_last_qsp_ms();
        double abm_ms = step_ms - pde_ms - qsp_ms;
        timing_file << i << "," << step_ms << "," << pde_ms << "," << qsp_ms << "," << abm_ms << "\n";
        timing_file.flush();

        // Write per-layer timings collected by checkpoint host functions
        {
            // GPU memory snapshot after this step
            size_t free_m = 0, total_m = 0;
            cudaMemGetInfo(&free_m, &total_m);
            int gpu_used_mb = static_cast<int>((total_m - free_m) / (1024 * 1024));

            layer_file << i << ",gpu_mem_mb," << gpu_used_mb << "\n";
            layer_file << i << ",total_ms," << step_ms << "\n";
            layer_file << i << ",pde_solve_ms," << pde_ms << "\n";
            layer_file << i << ",qsp_solve_ms," << qsp_ms << "\n";

            // Checkpoint-recorded phases (filled by timing_after_* host functions)
            for (const auto& lt : PDAC::g_layer_timings) {
                layer_file << i << "," << lt.name << "," << lt.ms << "\n";
            }
            PDAC::g_layer_timings.clear();

            layer_file.flush();
        }

        // Read event/state counts from GPU and write stats row
        if (stats_file.is_open()) {
            unsigned int host_events[PDAC::ABM_EVENT_COUNTER_SIZE];
            unsigned int host_states[PDAC::ABM_STATE_COUNTER_SIZE];
            cudaMemcpy(host_events, device_event_counters,
                PDAC::ABM_EVENT_COUNTER_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            cudaMemcpy(host_states, device_state_counters,
                PDAC::ABM_STATE_COUNTER_SIZE * sizeof(unsigned int), cudaMemcpyDeviceToHost);

            PDAC::RecruitStats rs = PDAC::get_last_recruit_stats();

            // Compute PDL1 fraction
            unsigned int total_cancer = host_states[PDAC::SC_CANCER_STEM]
                                      + host_states[PDAC::SC_CANCER_PROG]
                                      + host_states[PDAC::SC_CANCER_SEN];
            float pdl1_frac = (total_cancer > 0)
                ? static_cast<float>(host_events[PDAC::EVT_PDL1_COUNT]) / total_cancer
                : 0.0f;

            stats_file << (i + 1) << ","
                // agentCount (15 states, from broadcast at start of this step)
                << host_states[PDAC::SC_CANCER_STEM] << "," << host_states[PDAC::SC_CANCER_PROG] << "," << host_states[PDAC::SC_CANCER_SEN] << ","
                << host_states[PDAC::SC_CD8_EFF] << "," << host_states[PDAC::SC_CD8_CYT] << "," << host_states[PDAC::SC_CD8_SUP] << ","
                << host_states[PDAC::SC_TH] << "," << host_states[PDAC::SC_TREG] << ","
                << host_states[PDAC::SC_MDSC] << ","
                << host_states[PDAC::SC_MAC_M1] << "," << host_states[PDAC::SC_MAC_M2] << ","
                << host_states[PDAC::SC_FIB_NORM] << "," << host_states[PDAC::SC_FIB_CAF] << ","
                << host_states[PDAC::SC_VAS_TIP] << "," << host_states[PDAC::SC_VAS_PHALANX] << ","
                // recruit (6 cols)
                << rs.teff_rec << "," << rs.th_rec << "," << rs.treg_rec << ","
                << rs.mdsc_rec << "," << rs.mac_m1_rec << "," << rs.mac_m2_rec << ","
                // prolif by state (15 cols)
                << host_events[PDAC::EVT_PROLIF_CD8_EFF] << "," << host_events[PDAC::EVT_PROLIF_CD8_CYT] << "," << host_events[PDAC::EVT_PROLIF_CD8_SUP] << ","
                << host_events[PDAC::EVT_PROLIF_TH] << "," << host_events[PDAC::EVT_PROLIF_TREG] << ","
                << host_events[PDAC::EVT_PROLIF_MDSC] << ","
                << host_events[PDAC::EVT_PROLIF_CANCER_STEM] << "," << host_events[PDAC::EVT_PROLIF_CANCER_PROG] << "," << host_events[PDAC::EVT_PROLIF_CANCER_SEN] << ","
                << host_events[PDAC::EVT_PROLIF_MAC_M1] << "," << host_events[PDAC::EVT_PROLIF_MAC_M2] << ","
                << host_events[PDAC::EVT_PROLIF_FIB_NORM] << "," << host_events[PDAC::EVT_PROLIF_FIB_CAF] << ","
                << host_events[PDAC::EVT_PROLIF_VAS_TIP] << "," << host_events[PDAC::EVT_PROLIF_VAS_PHALANX] << ","
                // death by state (15 cols)
                << host_events[PDAC::EVT_DEATH_CD8_EFF] << "," << host_events[PDAC::EVT_DEATH_CD8_CYT] << "," << host_events[PDAC::EVT_DEATH_CD8_SUP] << ","
                << host_events[PDAC::EVT_DEATH_TH] << "," << host_events[PDAC::EVT_DEATH_TREG] << ","
                << host_events[PDAC::EVT_DEATH_MDSC] << ","
                << host_events[PDAC::EVT_DEATH_CANCER_STEM] << "," << host_events[PDAC::EVT_DEATH_CANCER_PROG] << "," << host_events[PDAC::EVT_DEATH_CANCER_SEN] << ","
                << host_events[PDAC::EVT_DEATH_MAC_M1] << "," << host_events[PDAC::EVT_DEATH_MAC_M2] << ","
                << host_events[PDAC::EVT_DEATH_FIB_NORM] << "," << host_events[PDAC::EVT_DEATH_FIB_CAF] << ","
                << host_events[PDAC::EVT_DEATH_VAS_TIP] << "," << host_events[PDAC::EVT_DEATH_VAS_PHALANX] << ","
                // PDL1 fraction
                << std::fixed << std::setprecision(4) << pdl1_frac << "\n";
            stats_file.flush();

            // Reset event counters for next step (state counters reset before next step)
            cudaMemset(device_event_counters, 0, PDAC::ABM_EVENT_COUNTER_SIZE * sizeof(unsigned int));
            cudaMemset(device_state_counters, 0, PDAC::ABM_STATE_COUNTER_SIZE * sizeof(unsigned int));
        }

        if (!continue_sim) {
            std::cout << "Simulation terminated early at step " << i << std::endl;
            break;
        }
    }

    if (stats_file.is_open()) {
        stats_file.close();
        std::cout << "Created: " << stats_path << std::endl;
    }

    if (timing_file.is_open()) {
        timing_file.close();
        std::cout << "Created: " << timing_path << std::endl;
    }

    if (layer_file.is_open()) {
        layer_file.close();
        std::cout << "Created: outputs/layer_timing.csv" << std::endl;
    }

    // ========== REPORT RESULTS ==========
    std::cout << "\n=== Simulation Complete ===" << std::endl;
    
    flamegpu::AgentVector final_cancer(model->Agent(PDAC::AGENT_CANCER_CELL));
    flamegpu::AgentVector final_tcells(model->Agent(PDAC::AGENT_TCELL));
    flamegpu::AgentVector final_tregs(model->Agent(PDAC::AGENT_TREG));
    flamegpu::AgentVector final_mdscs(model->Agent(PDAC::AGENT_MDSC));
    flamegpu::AgentVector final_macs(model->Agent(PDAC::AGENT_MACROPHAGE));
    flamegpu::AgentVector final_fibs(model->Agent(PDAC::AGENT_FIBROBLAST));

    simulation.getPopulationData(final_cancer);
    simulation.getPopulationData(final_tcells);
    simulation.getPopulationData(final_tregs);
    simulation.getPopulationData(final_mdscs);
    simulation.getPopulationData(final_macs);
    simulation.getPopulationData(final_fibs);

    std::cout << "\nFinal Population Counts:" << std::endl;
    std::cout << "  Cancer cells: " << final_cancer.size() << std::endl;
    std::cout << "  T cells: " << final_tcells.size() << std::endl;
    std::cout << "  TRegs: " << final_tregs.size() << std::endl;
    std::cout << "  MDSCs: " << final_mdscs.size() << std::endl;
    std::cout << "  Macrophages: " << final_macs.size() << std::endl;
    std::cout << "  Fibroblasts: " << final_fibs.size() << std::endl;

    // Count T cell states
    if (final_tcells.size() > 0) {
        int eff_count = 0, cyt_count = 0, supp_count = 0;
        for (unsigned int i = 0; i < final_tcells.size(); i++) {
            int state = final_tcells[i].getVariable<int>("cell_state");
            if (state == PDAC::T_CELL_EFF) eff_count++;
            else if (state == PDAC::T_CELL_CYT) cyt_count++;
            else if (state == PDAC::T_CELL_SUPP) supp_count++;
        }
        std::cout << "  T cell states - Effector: " << eff_count
                  << ", Cytotoxic: " << cyt_count
                  << ", Suppressed: " << supp_count << std::endl;
    }

    // ========== DRAIN ASYNC I/O ==========
    flush_async_io();

    // ========== CLEANUP ==========
    cleanup_pde_io();
    cleanup_abm_io();
    PDAC::cleanup_pde_solver();

    // Free GPU event counter memory
    if (device_event_counters != nullptr) {
        cudaFree(device_event_counters);
    }
    
    std::cout << "\nSimulation finished successfully." << std::endl;

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(stop - start);
    std::cout << "Time taken: " << duration.count() << " seconds" << std::endl;

    return 0;
}