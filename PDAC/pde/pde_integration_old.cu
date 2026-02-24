#include "pde_integration.cuh"
#include "../core/common.cuh"
#include <iostream>
#include <vector>
#include <nvtx3/nvToolsExt.h>

namespace PDAC {

// Global PDE solver instance
PDESolver* g_pde_solver = nullptr;

// ============================================================================
// Helper: Read Chemical from PDE to Agents
// ============================================================================

void read_chemical_to_agents(
    flamegpu::HostAPI& host_api,
    const std::string& agent_name,
    int substrate_idx,
    const std::string& output_var_name)
{
    if (!g_pde_solver) return;
    
    // Get agent API and population data
    flamegpu::HostAgentAPI agent_api = host_api.agent(agent_name);
    unsigned int agent_count = agent_api.count();
    if (agent_count == 0) return;
    
    const int grid_x = host_api.environment.getProperty<int>("grid_size_x");
    const int grid_y = host_api.environment.getProperty<int>("grid_size_y");
    const int grid_z = host_api.environment.getProperty<int>("grid_size_z");
    
    // Export agent data to AgentVector
    flamegpu::DeviceAgentVector agent_vec = agent_api.getPopulationData();
    
    // Allocate temporary host arrays
    std::vector<int> h_x(agent_count);
    std::vector<int> h_y(agent_count);
    std::vector<int> h_z(agent_count);
    std::vector<float> h_concentrations(agent_count);
    
    // Copy agent positions to host using indexed access
    for (unsigned int idx = 0; idx < agent_count; idx++) {
        h_x[idx] = agent_vec[idx].getVariable<int>("x");
        h_y[idx] = agent_vec[idx].getVariable<int>("y");
        h_z[idx] = agent_vec[idx].getVariable<int>("z");
    }
    
    // Allocate device arrays
    int* d_x;
    int* d_y;
    int* d_z;
    float* d_concentrations;
    
    cudaMalloc(&d_x, agent_count * sizeof(int));
    cudaMalloc(&d_y, agent_count * sizeof(int));
    cudaMalloc(&d_z, agent_count * sizeof(int));
    cudaMalloc(&d_concentrations, agent_count * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_x, h_x.data(), agent_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), agent_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), agent_count * sizeof(int), cudaMemcpyHostToDevice);
    
    // Get PDE concentration device pointer
    const float* d_pde_concentrations = g_pde_solver->get_device_concentration_ptr(substrate_idx);
    
    // Launch kernel
    int threads = 256;
    int blocks = (agent_count + threads - 1) / threads;
    
    read_concentrations_at_voxels<<<blocks, threads>>>(
        d_pde_concentrations,
        d_x, d_y, d_z,
        d_concentrations,
        agent_count,
        substrate_idx,
        grid_x, grid_y, grid_z
    );
    
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(h_concentrations.data(), d_concentrations, 
               agent_count * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Write back to agents using indexed access
    for (unsigned int idx = 0; idx < agent_count; idx++) {
        agent_vec[idx].setVariable<float>(output_var_name, h_concentrations[idx]);
    }
    
    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_concentrations);
}

// ============================================================================
// Helper: Collect Chemical Sources from Agents to PDE
// ============================================================================

void collect_chemical_from_agents(
    flamegpu::HostAPI& host_api,
    const std::string& agent_name,
    int substrate_idx,
    const std::string& source_var_name)
{
    if (!g_pde_solver) return;
    
    // Get agent API
    flamegpu::HostAgentAPI agent_api = host_api.agent(agent_name);
    unsigned int agent_count = agent_api.count();
    if (agent_count == 0) return;
    
    const int grid_x = host_api.environment.getProperty<int>("grid_size_x");
    const int grid_y = host_api.environment.getProperty<int>("grid_size_y");
    const int grid_z = host_api.environment.getProperty<int>("grid_size_z");
    
    // Export agent data to AgentVector
    flamegpu::DeviceAgentVector agent_vec = agent_api.getPopulationData();
    
    // Allocate temporary host arrays
    std::vector<int> h_x(agent_count);
    std::vector<int> h_y(agent_count);
    std::vector<int> h_z(agent_count);
    std::vector<float> h_source_rates(agent_count);
    
    // Copy agent data to host using indexed access
    for (unsigned int idx = 0; idx < agent_count; idx++) {
        h_x[idx] = agent_vec[idx].getVariable<int>("x");
        h_y[idx] = agent_vec[idx].getVariable<int>("y");
        h_z[idx] = agent_vec[idx].getVariable<int>("z");

        h_source_rates[idx] = agent_vec[idx].getVariable<float>(source_var_name);
    }
    
    // Allocate device arrays
    int* d_x;
    int* d_y;
    int* d_z;
    float* d_source_rates;
    
    cudaMalloc(&d_x, agent_count * sizeof(int));
    cudaMalloc(&d_y, agent_count * sizeof(int));
    cudaMalloc(&d_z, agent_count * sizeof(int));
    cudaMalloc(&d_source_rates, agent_count * sizeof(float));
    
    // Copy to device
    cudaMemcpy(d_x, h_x.data(), agent_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y.data(), agent_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, h_z.data(), agent_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_source_rates, h_source_rates.data(), 
               agent_count * sizeof(float), cudaMemcpyHostToDevice);
    
    // Get PDE source and uptake device pointers
    float* d_pde_sources = g_pde_solver->get_device_source_ptr(substrate_idx);
    float* d_pde_uptakes = g_pde_solver->get_device_uptake_ptr(substrate_idx);

    // Calculate voxel volume in cm^3 for unit conversion
    // Release rates (amount/cell/time) need to be multiplied by dt and divided by volume to get concentration change
    // Uptake rates are already in correct units and should NOT be divided
    const float voxel_size_cm = host_api.environment.getProperty<float>("voxel_size") * 1e-4f;  // µm to cm
    const float voxel_volume = voxel_size_cm * voxel_size_cm * voxel_size_cm;  // cm^3
    const float dt = host_api.environment.getProperty<float>("PARAM_SEC_PER_SLICE");  // seconds per ABM step

    // Launch kernel
    int threads = 256;
    int blocks = (agent_count + threads - 1) / threads;

    add_sources_from_agents<<<blocks, threads>>>(
        d_pde_sources,
        d_pde_uptakes,
        d_x, d_y, d_z,
        d_source_rates,
        agent_count,
        substrate_idx,
        grid_x, grid_y, grid_z,
        voxel_volume,
        dt
    );

    cudaDeviceSynchronize();

    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_source_rates);
}

// ============================================================================
// Helper: Calculate Chemical Gradient for Agents
// ============================================================================
// Calculates spatial gradient of a chemical and stores as agent variables
// More efficient than having each agent calculate independently
void calculate_chemical_gradient_for_agents(
    flamegpu::HostAPI& host_api,
    const std::string& agent_name,
    int substrate_idx,
    const std::string& var_prefix)  // e.g., "CCL2_gradient" -> sets CCL2_gradient_x, _y, _z
{
    if (!g_pde_solver) return;

    // Get agent API
    flamegpu::HostAgentAPI agent_api = host_api.agent(agent_name);
    unsigned int agent_count = agent_api.count();
    if (agent_count == 0) return;
    
    const int grid_x = host_api.environment.getProperty<int>("grid_size_x");
    const int grid_y = host_api.environment.getProperty<int>("grid_size_y");
    const int grid_z = host_api.environment.getProperty<int>("grid_size_z");
    const float voxel_size = host_api.environment.getProperty<float>("voxel_size");
    const float dx = voxel_size * 1.0e-4f;  // Convert µm to cm

    // Get PDE concentration data
    const float* d_chem = g_pde_solver->get_device_concentration_ptr(substrate_idx);
    const int total_voxels = grid_x * grid_y * grid_z;

    // Copy to host for gradient calculation
    std::vector<float> h_chem(total_voxels);
    cudaMemcpy(h_chem.data(), d_chem, total_voxels * sizeof(float), cudaMemcpyDeviceToHost);

    // Iterate through agents and calculate gradient at their position
    flamegpu::DeviceAgentVector agents = agent_api.getPopulationData();
    for (auto agent : agents) {  // Note: can't use reference with DeviceAgentVector
        int x = agent.getVariable<int>("x");
        int y = agent.getVariable<int>("y");
        int z = agent.getVariable<int>("z");
        int voxel_idx = z * (grid_x * grid_y) + y * grid_x + x;

        float grad_x = 0.0f, grad_y = 0.0f, grad_z = 0.0f;

        // X gradient (central difference where possible)
        if (x > 0 && x < grid_x - 1) {
            int idx_left = voxel_idx - 1;
            int idx_right = voxel_idx + 1;
            grad_x = (h_chem[idx_right] - h_chem[idx_left]) / (2.0f * dx);
        } else if (x == 0 && grid_x > 1) {
            // Forward difference at left boundary
            grad_x = (h_chem[voxel_idx + 1] - h_chem[voxel_idx]) / dx;
        } else if (x == grid_x - 1 && grid_x > 1) {
            // Backward difference at right boundary
            grad_x = (h_chem[voxel_idx] - h_chem[voxel_idx - 1]) / dx;
        }

        // Y gradient
        if (y > 0 && y < grid_y - 1) {
            int idx_front = voxel_idx - grid_x;
            int idx_back = voxel_idx + grid_x;
            grad_y = (h_chem[idx_back] - h_chem[idx_front]) / (2.0f * dx);
        } else if (y == 0 && grid_y > 1) {
            grad_y = (h_chem[voxel_idx + grid_x] - h_chem[voxel_idx]) / dx;
        } else if (y == grid_y - 1 && grid_y > 1) {
            grad_y = (h_chem[voxel_idx] - h_chem[voxel_idx - grid_x]) / dx;
        }

        // Z gradient
        if (z > 0 && z < grid_z - 1) {
            int idx_bottom = voxel_idx - grid_x * grid_y;
            int idx_top = voxel_idx + grid_x * grid_y;
            grad_z = (h_chem[idx_top] - h_chem[idx_bottom]) / (2.0f * dx);
        } else if (z == 0 && grid_z > 1) {
            grad_z = (h_chem[voxel_idx + grid_x * grid_y] - h_chem[voxel_idx]) / dx;
        } else if (z == grid_z - 1 && grid_z > 1) {
            grad_z = (h_chem[voxel_idx] - h_chem[voxel_idx - grid_x * grid_y]) / dx;
        }

        // Set gradient variables
        agent.setVariable<float>(var_prefix + "_x", grad_x);
        agent.setVariable<float>(var_prefix + "_y", grad_y);
        agent.setVariable<float>(var_prefix + "_z", grad_z);
    }
}

// ============================================================================
// Host Function: Update Agent Chemicals (Read from PDE)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(update_agent_chemicals) {
    nvtxRangePush("Update Agent Chemicals");
    if (!g_pde_solver) {
        nvtxRangePop();
        return;
    }

    // Cancer reads
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_O2, "local_O2");
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_IFN, "local_IFNg");
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_TGFB, "local_TGFB");
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_ARGI, "local_ArgI");
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_NO, "local_NO");
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_NO, "local_IL10");

    // T-Cell reads
    read_chemical_to_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IL2, "local_IL2");

    // T-Reg reads
    read_chemical_to_agents(*FLAMEGPU, AGENT_TREG, CHEM_IFN, "local_IFNg");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TREG, CHEM_TGFB, "local_TGFB");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TREG, CHEM_ARGI, "local_ArgI");

    // MDSC reads
    read_chemical_to_agents(*FLAMEGPU, AGENT_MDSC, CHEM_IFN, "local_IFNg");

    // Macrophage reads
    read_chemical_to_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_IFN, "local_IFNg");
    read_chemical_to_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_IL10, "local_IL10");
    read_chemical_to_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_TGFB, "local_TGFB");
    read_chemical_to_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_CCL2, "local_CCL2");
    read_chemical_to_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_IL12, "local_IL12");

    // Fibroblast reads
    read_chemical_to_agents(*FLAMEGPU, AGENT_FIBROBLAST, CHEM_TGFB, "local_TGFB");

    // Vasculature reads
    read_chemical_to_agents(*FLAMEGPU, AGENT_VASCULAR, CHEM_O2, "local_O2");
    read_chemical_to_agents(*FLAMEGPU, AGENT_VASCULAR, CHEM_IFN, "local_IFNg");
    read_chemical_to_agents(*FLAMEGPU, AGENT_VASCULAR, CHEM_VEGFA, "local_VEGFA");

    // Note: Nivolumab and Cabozantinib are now handled by QSP compartments
    // They will be transferred to GPU environment properties by the QSP coupling wrapper

    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    if (step % 50 == 0) {
        std::cout << "Updated agent chemicals from PDE (step " << step << ")" << std::endl;
    }

    // Force synchronization to catch any CUDA errors immediately
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error after update_agent_chemicals: " << cudaGetErrorString(err) << std::endl;
    }

    // ========== CALCULATE CHEMICAL GRADIENTS FOR CHEMOTAXIS ==========
    // Pre-compute gradients on host side to avoid redundant per-agent calculations

    // T Cell: IFN-γ gradient for chemotaxis
    calculate_chemical_gradient_for_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IFN, "ifng_grad");

    // TReg: IFN-γ gradient for chemotaxis
    calculate_chemical_gradient_for_agents(*FLAMEGPU, AGENT_TREG, CHEM_IFN, "ifng_grad");

    // MDSC: CCL2 gradient for chemotaxis
    calculate_chemical_gradient_for_agents(*FLAMEGPU, AGENT_MDSC, CHEM_CCL2, "ccl2_grad");

    // Macrophage: CCL2 gradient for chemotaxis
    calculate_chemical_gradient_for_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_CCL2, "ccl2_grad");

    // Fibroblast: TGFB gradient for chemotaxis
    calculate_chemical_gradient_for_agents(*FLAMEGPU, AGENT_FIBROBLAST, CHEM_TGFB, "tgfb_grad");

    // VascularCell: calculate VEGF-A gradient for tip cell chemotaxis
    calculate_chemical_gradient_for_agents(*FLAMEGPU, AGENT_VASCULAR, CHEM_VEGFA, "vegfa_grad");

    nvtxRangePop();
}

// ============================================================================
// Host Function: Collect Agent Sources (Write to PDE)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(collect_agent_sources) {
    if (!g_pde_solver) return;

    nvtxRangePush("Collect Agent Sources");

    // Reset sources and uptakes for this timestep
    g_pde_solver->reset_sources();
    g_pde_solver->reset_uptakes();  // NEW: reset uptakes for BioFVM approach
    
    // Collect O2 consumption from cancer cells (should be negative)
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_O2, "O2_uptake_rate");
    
    // Collect IFN-gamma production from T cells
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IFN, "IFNg_release_rate");
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_IFN, "IFNg_uptake_rate");

    // Collect IL-2 production from T cells
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IL2, "IL2_release_rate");
    
    // Collect IL-2 consumption from Tregs (should be negative)
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TREG, CHEM_IL2, "IL2_release_rate");

    // Collect IL-10 production from Tregs
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TREG, CHEM_IL10, "IL10_release_rate");

    // Collect TGF-beta production from Tregs
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TREG, CHEM_TGFB, "TGFB_release_rate");
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_TGFB, "TGFB_release_rate");
    
    // Collect CCL2 production from cancer cells
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_CCL2, "CCL2_release_rate");
    // Collect VEGF-A production from cancer cells
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_VEGFA, "VEGFA_release_rate");

    // Collect NO production from MDSCs
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MDSC, CHEM_NO, "NO_release_rate");
    // Collect ArgI production from MDSCs
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MDSC, CHEM_ARGI, "ArgI_release_rate");

    // Collect macrophage production (state-dependent)
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_IFN, "IFNg_release_rate");
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_IL12, "IL12_release_rate");
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_TGFB, "TGFB_release_rate");
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_IL10, "IL10_release_rate");
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_VEGFA, "VEGFA_release_rate");
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MACROPHAGE, CHEM_CCL2, "CCL2_uptake_rate");

    // Fibroblast: collect TGFB production (CAFs only, filtered by agent function)
    collect_chemical_from_agents(*FLAMEGPU, AGENT_FIBROBLAST, CHEM_TGFB, "TGFB_release_rate");

    // VascularCell: collect O2 sources (phalanx only, filtered by agent function)
    collect_chemical_from_agents(*FLAMEGPU, AGENT_VASCULAR, CHEM_O2, "O2_source");
    // VascularCell: collect VEGF-A sinks (all states)
    collect_chemical_from_agents(*FLAMEGPU, AGENT_VASCULAR, CHEM_VEGFA, "VEGFA_sink");

    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    if (step % 50 == 0) {
        std::cout << "Collected agent sources to PDE (step " << step << ")" << std::endl;
    }

    nvtxRangePop();
}

// ============================================================================
// Host Function: Solve PDE
// ============================================================================

FLAMEGPU_HOST_FUNCTION(solve_pde_step) {
    nvtxRangePush("PDE Solve");
    if (!g_pde_solver) {
        nvtxRangePop();
        return;
    }

    g_pde_solver->solve_timestep();

    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    if (step % 50 == 0) {
        std::cout << "PDE solved for step " << step << std::endl;
    }
    nvtxRangePop();
}

// ============================================================================
// Initialize/Cleanup
// ============================================================================

void initialize_pde_solver(int grid_x, int grid_y, int grid_z,
                           float voxel_size, float dt_abm, int molecular_steps,
                            const PDAC::GPUParam& gpu_params) {
    PDEConfig config;
    config.nx = grid_x;
    config.ny = grid_y;
    config.nz = grid_z;
    config.num_substrates = NUM_SUBSTRATES;
    config.voxel_size = voxel_size * 1.0e-4f;  // Convert µm to cm
    config.dt_abm = dt_abm;
    config.dt_pde = dt_abm / molecular_steps;
    config.substeps_per_abm = molecular_steps;
    config.boundary_type = 0;  // Neumann (no-flux)
    
    // Set diffusion coefficients (cm²/s) from params file
    config.diffusion_coeffs[CHEM_O2]    = gpu_params.getFloat(PARAM_O2_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IFN]   = gpu_params.getFloat(PARAM_IFNG_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL2]   = gpu_params.getFloat(PARAM_IL2_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL10]  = gpu_params.getFloat(PARAM_IL10_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_TGFB]  = gpu_params.getFloat(PARAM_TGFB_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_CCL2]  = gpu_params.getFloat(PARAM_CCL2_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_ARGI]  = gpu_params.getFloat(PARAM_ARGI_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_NO]    = gpu_params.getFloat(PARAM_NO_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_IL12]  = gpu_params.getFloat(PARAM_IL12_DIFFUSIVITY);
    config.diffusion_coeffs[CHEM_VEGFA] = gpu_params.getFloat(PARAM_VEGFA_DIFFUSIVITY);

    // Set decay rates (1/s) from params file
    config.decay_rates[CHEM_O2]    = gpu_params.getFloat(PARAM_O2_DECAY_RATE);
    config.decay_rates[CHEM_IFN]   = gpu_params.getFloat(PARAM_IFNG_DECAY_RATE);
    config.decay_rates[CHEM_IL2]   = gpu_params.getFloat(PARAM_IL2_DECAY_RATE);
    config.decay_rates[CHEM_IL10]  = gpu_params.getFloat(PARAM_IL10_DECAY_RATE);
    config.decay_rates[CHEM_TGFB]  = gpu_params.getFloat(PARAM_TGFB_DECAY_RATE);
    config.decay_rates[CHEM_CCL2]  = gpu_params.getFloat(PARAM_CCL2_DECAY_RATE);
    config.decay_rates[CHEM_ARGI]  = gpu_params.getFloat(PARAM_ARGI_DECAY_RATE);
    config.decay_rates[CHEM_NO]    = gpu_params.getFloat(PARAM_NO_DECAY_RATE);
    config.decay_rates[CHEM_IL12]  = gpu_params.getFloat(PARAM_IL12_DECAY_RATE);
    config.decay_rates[CHEM_VEGFA] = gpu_params.getFloat(PARAM_VEGFA_DECAY_RATE);
    
    g_pde_solver = new PDESolver(config);
    g_pde_solver->initialize();
    
    // Set initial O2 concentration, all others start at 0.0
    g_pde_solver->set_initial_concentration(CHEM_O2, 0.673);  // Oxygen starts at 0.673 (amount/mL)
    
    std::cout << "PDE Solver initialized and coupled to FLAME GPU 2" << std::endl;
}

// Call this after model initialization but before simulation starts
void set_pde_pointers_in_environment(flamegpu::ModelDescription& model) {
    if (!g_pde_solver) return;
    
    flamegpu::EnvironmentDescription env = model.Environment();
    
    // Store device pointers as unsigned long long (can be cast back to float*)
    for (int sub = 0; sub < NUM_SUBSTRATES; sub++) {
        std::string concentration_key = "pde_concentration_ptr_" + std::to_string(sub);
        std::string source_key = "pde_source_ptr_" + std::to_string(sub);
        
        uintptr_t conc_ptr = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_concentration_ptr(sub));
        uintptr_t src_ptr = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_source_ptr(sub));
        
        env.newProperty<unsigned long long>(concentration_key, static_cast<unsigned long long>(conc_ptr));
        env.newProperty<unsigned long long>(source_key, static_cast<unsigned long long>(src_ptr));
    }

    // Store recruitment sources pointer
    uintptr_t recruit_ptr = reinterpret_cast<uintptr_t>(g_pde_solver->get_device_recruitment_sources_ptr());
    env.newProperty<unsigned long long>("pde_recruitment_sources_ptr", static_cast<unsigned long long>(recruit_ptr));

    std::cout << "PDE device pointers stored in FLAME GPU environment" << std::endl;
}

void cleanup_pde_solver() {
    if (g_pde_solver) {
        delete g_pde_solver;
        g_pde_solver = nullptr;
    }
}

// ============================================================================
// Recruitment System Implementation
// ============================================================================

// CUDA kernel to mark MDSC recruitment sources based on CCL2
__global__ void mark_mdsc_sources_kernel(
    int* d_recruitment_sources,
    const float* d_ccl2,
    int nx, int ny, int nz,
    float ec50_ccl2,
    unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = z * (nx * ny) + y * nx + x;

    float ccl2 = d_ccl2[idx];
    float H_CCL2 = ccl2 / (ccl2 + ec50_ccl2);

    // Simple random number generation (thread-local)
    unsigned int rng_state = seed + idx;
    rng_state = rng_state * 1103515245u + 12345u;
    float rand_val = (rng_state & 0x7FFFFFFF) / float(0x7FFFFFFF);

    if (rand_val < H_CCL2) {
        atomicOr(&d_recruitment_sources[idx], 2);  // Set MDSC bit (bit 1)
    }
}

// CUDA kernel to mark macrophage recruitment sources based on CCL2
__global__ void mark_mac_sources_kernel(
    int* d_recruitment_sources,
    const float* d_ccl2,
    int nx, int ny, int nz,
    float ec50_ccl2,
    unsigned int seed)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= nx || y >= ny || z >= nz) return;

    int idx = z * (nx * ny) + y * nx + x;

    float ccl2 = d_ccl2[idx];
    float H_CCL2 = ccl2 / (ccl2 + ec50_ccl2);

    // Simple random number generation (thread-local)
    unsigned int rng_state = seed + idx;
    rng_state = rng_state * 1103515245u + 12345u;
    float rand_val = (rng_state & 0x7FFFFFFF) / float(0x7FFFFFFF);

    if (rand_val < H_CCL2) {
        atomicOr(&d_recruitment_sources[idx], 4);  // Set macrophage bit (bit 2)
    }
}

// Reset recruitment sources at start of each step
FLAMEGPU_HOST_FUNCTION(reset_recruitment_sources) {
    if (!g_pde_solver) return;
    g_pde_solver->reset_recruitment_sources();
}

// Mark MDSC sources based on CCL2 concentration
FLAMEGPU_HOST_FUNCTION(mark_mdsc_sources) {
    if (!g_pde_solver) return;

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    int* d_recruitment_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    const float* d_ccl2 = g_pde_solver->get_device_concentration_ptr(CHEM_CCL2);

    // Get parameter
    float ec50_ccl2 = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_EC50_CCL2_REC");

    // Generate random seed from step number
    unsigned int seed = static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 12345u;

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    mark_mdsc_sources_kernel<<<grid, block>>>(
        d_recruitment_sources, d_ccl2, nx, ny, nz, ec50_ccl2, seed);

    cudaDeviceSynchronize();
}

// Recruit T cells at marked T source voxels
FLAMEGPU_HOST_FUNCTION(recruit_t_cells) {
    if (!g_pde_solver) return;

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const float voxel_size = FLAMEGPU->environment.getProperty<float>("voxel_size");

    // Get QSP concentrations and recruitment rates from environment
    float qsp_teff_conc = FLAMEGPU->environment.getProperty<float>("qsp_teff_central");
    float k_teff = FLAMEGPU->environment.getProperty<float>("PARAM_TEFF_RECRUIT_K");
    float p_recruit_teff = std::min(qsp_teff_conc * k_teff, 1.0f);

    float qsp_treg_conc = FLAMEGPU->environment.getProperty<float>("qsp_treg_central");
    float k_treg = FLAMEGPU->environment.getProperty<float>("PARAM_TREG_RECRUIT_K");
    float p_recruit_treg = std::min(qsp_treg_conc * k_treg, 1.0f);

    float qsp_th_conc = FLAMEGPU->environment.getProperty<float>("qsp_th_central");
    float k_th = FLAMEGPU->environment.getProperty<float>("PARAM_TH_RECRUIT_K");
    float p_recruit_th = std::min(qsp_th_conc * k_th, 1.0f);

    // Debug output (periodic)
    // static int debug_counter = 0;
    // if (debug_counter % 10 == 0) {
    //     std::cout << "T cell recruitment probabilities: Teff=" << p_recruit_teff
    //               << " (conc=" << qsp_teff_conc << ", k=" << k_teff << ")"
    //               << ", Treg=" << p_recruit_treg
    //               << " (conc=" << qsp_th_conc << ", k=" << k_treg << ")" << std::endl;
    // }
    // debug_counter++;

    // Copy recruitment sources to host
    int total_voxels = nx * ny * nz;
    std::vector<int> h_sources(total_voxels);
    int* d_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);

    // Get agent APIs for creating new agents
    auto tcell_api = FLAMEGPU->agent(AGENT_TCELL);
    auto treg_api = FLAMEGPU->agent(AGENT_TREG);

    int teff_recruited = 0;
    int treg_recruited = 0;
    int th_recruited = 0;

    // Count total sources available
    int total_t_sources = 0;
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 1) != 0) total_t_sources++;
    }
    if (total_t_sources > 0) {
        std::cout << "[DEBUG] Found " << total_t_sources << " T cell recruitment sources" << std::endl;
    }

    // Scan for T source voxels (bit 0 set)
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 1) == 0) continue;  // Not a T source

        int z = idx / (nx * ny);
        int y = (idx % (nx * ny)) / nx;
        int x = idx % nx;

        // Try to recruit Teff
        if (FLAMEGPU->random.uniform<float>() < p_recruit_teff) {
            // Find empty neighbor voxel for placement (Moore neighborhood)
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            // Create new T cell (simplified initialization)
                            auto new_agent = tcell_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);
                            new_agent.setVariable<int>("cell_state", 0);  // Effector state

                            double lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_T_CELL_LIFE_MEAN_SLICE");
                            double lifeSd = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");

                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<float>("life", life);

                            new_agent.setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_DIV_INTERNAL"));
                            new_agent.setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_DIV_LIMIT"));

                            new_agent.setVariable<float>("IL2_release_remain", FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_IL2_RELEASE_TIME"));

                            teff_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }

        // Try to recruit Treg
        if (FLAMEGPU->random.uniform<float>() < p_recruit_treg) {
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = treg_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);
                            new_agent.setVariable<int>("cell_state", 0);

                            double lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
                            double lifeSd = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");

                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<int>("life", life);

                            new_agent.setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_DIV_INTERNAL"));
                            new_agent.setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_DIV_LIMIT"));

                            new_agent.setVariable<float>("TGFB_release_remain", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME"));

                            new_agent.setVariable<float>("CTLA4", FLAMEGPU->environment.getProperty<float>("PARAM_CTLA4_TREG"));

                            treg_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
        // Try to recruit TH
        if (FLAMEGPU->random.uniform<float>() < p_recruit_th) {
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = treg_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);
                            new_agent.setVariable<int>("cell_state", 1);

                            double lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_LIFE_MEAN_SLICE");
                            double lifeSd = FLAMEGPU->environment.getProperty<float>("PARAM_TCELL_LIFESPAN_SD_SLICE");

                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<int>("life", life);

                            new_agent.setVariable<int>("divide_cd", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_DIV_INTERNAL"));
                            new_agent.setVariable<int>("divide_limit", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_DIV_LIMIT"));

                            new_agent.setVariable<float>("TGFB_release_remain", FLAMEGPU->environment.getProperty<float>("PARAM_TCD4_TGFB_RELEASE_TIME"));

                            new_agent.setVariable<float>("CTLA4", 0.0);

                            th_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
    }

    // Update MacroProperty counters (will be copied to environment by copy_abm_counters_to_environment)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_TEFF_REC] += teff_recruited;
    counters[ABM_COUNT_TH_REC] += th_recruited;
    counters[ABM_COUNT_TREG_REC] += treg_recruited;
}

// Recruit MDSCs at marked MDSC source voxels
FLAMEGPU_HOST_FUNCTION(recruit_mdscs) {
    if (!g_pde_solver) return;

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Calculate recruitment probability: p = min(concentration * k_recruit, 1.0)
    float p_recruit_mdsc = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_RECRUIT_K");

    int total_voxels = nx * ny * nz;
    std::vector<int> h_sources(total_voxels);
    int* d_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);

    // Get agent API for creating new agents
    auto mdsc_api = FLAMEGPU->agent(AGENT_MDSC);
    int mdsc_recruited = 0;

    // Scan for MDSC source voxels (bit 1 set)
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 2) == 0) continue;  // Not an MDSC source

        if (FLAMEGPU->random.uniform<float>() < p_recruit_mdsc) {
            int z = idx / (nx * ny);
            int y = (idx % (nx * ny)) / nx;
            int x = idx % nx;

            // Find empty neighbor voxel
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = mdsc_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);

                            double lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_MDSC_LIFE_MEAN_SLICE");
                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<int>("life", life);

                            mdsc_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
    }

    // Update MacroProperty counters (will be copied to environment by copy_abm_counters_to_environment)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_MDSC_REC] += mdsc_recruited;
}

// Mark macrophage sources based on CCL2 concentration
FLAMEGPU_HOST_FUNCTION(mark_mac_sources) {
    if (!g_pde_solver) return;

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    int* d_recruitment_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    const float* d_ccl2 = g_pde_solver->get_device_concentration_ptr(CHEM_CCL2);

    // Get parameter
    float ec50_ccl2 = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_EC50_CCL2_REC");

    // Generate random seed from step number
    unsigned int seed = static_cast<unsigned int>(FLAMEGPU->getStepCounter()) * 12345u + 54321u;

    dim3 block(8, 8, 8);
    dim3 grid((nx + 7) / 8, (ny + 7) / 8, (nz + 7) / 8);

    mark_mac_sources_kernel<<<grid, block>>>(
        d_recruitment_sources, d_ccl2, nx, ny, nz, ec50_ccl2, seed);

    cudaDeviceSynchronize();
}

// Recruit macrophages at marked macrophage source voxels
FLAMEGPU_HOST_FUNCTION(recruit_macrophages) {
    if (!g_pde_solver) return;

    const int nx = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int ny = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int nz = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // Calculate recruitment probability
    float p_recruit_mac = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_RECRUIT_K");

    int total_voxels = nx * ny * nz;
    std::vector<int> h_sources(total_voxels);
    int* d_sources = g_pde_solver->get_device_recruitment_sources_ptr();
    cudaMemcpy(h_sources.data(), d_sources, total_voxels * sizeof(int), cudaMemcpyDeviceToHost);

    // Get agent API for creating new agents
    auto mac_api = FLAMEGPU->agent(AGENT_MACROPHAGE);
    int mac_recruited = 0;

    // Scan for macrophage source voxels (bit 2 set)
    for (int idx = 0; idx < total_voxels; idx++) {
        if ((h_sources[idx] & 4) == 0) continue;  // Not a macrophage source

        if (FLAMEGPU->random.uniform<float>() < p_recruit_mac) {
            int z = idx / (nx * ny);
            int y = (idx % (nx * ny)) / nx;
            int x = idx % nx;

            // Find empty neighbor voxel
            bool placed = false;
            for (int dz = -1; dz <= 1 && !placed; dz++) {
                for (int dy = -1; dy <= 1 && !placed; dy++) {
                    for (int dx = -1; dx <= 1 && !placed; dx++) {
                        if (dx == 0 && dy == 0 && dz == 0) continue;

                        int nx_new = x + dx;
                        int ny_new = y + dy;
                        int nz_new = z + dz;

                        if (nx_new >= 0 && nx_new < nx &&
                            ny_new >= 0 && ny_new < ny &&
                            nz_new >= 0 && nz_new < nz) {

                            auto new_agent = mac_api.newAgent();
                            new_agent.setVariable<int>("x", nx_new);
                            new_agent.setVariable<int>("y", ny_new);
                            new_agent.setVariable<int>("z", nz_new);

                            // Recruit as M1 state
                            int mac_state = MAC_M1;

                            // 30% chance to become M2
                            if (FLAMEGPU->random.uniform<float>() < 0.3f) {
                                mac_state = MAC_M2;
                            }
                            new_agent.setVariable<int>("mac_state", mac_state);

                            // Set lifespan
                            double lifeMean = FLAMEGPU->environment.getProperty<float>("PARAM_MAC_LIFE_MEAN");
                            float rnd = static_cast<float>(rand()) / RAND_MAX;
                            int life = static_cast<int>(lifeMean * std::log(1.0f / (rnd + 0.0001f)) + 0.5f);
                            if (life < 1) life = 1;

                            new_agent.setVariable<int>("life", life);

                            mac_recruited++;
                            placed = true;
                        }
                    }
                }
            }
        }
    }

    // Update MacroProperty counters (will be copied to environment by copy_abm_counters_to_environment)
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    counters[ABM_COUNT_MAC_REC] += mac_recruited;
}

// ============================================================================
// Occupancy Grid: Zero the grid at the start of each step's division phase
// ============================================================================
FLAMEGPU_HOST_FUNCTION(zero_occupancy_grid) {
    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    occ.zero();
}

// ============================================================================
// Zero Fibroblast Density Field (reset before scatter)
// ============================================================================
FLAMEGPU_HOST_FUNCTION(zero_fib_density_field) {
    auto field = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("fib_density_field");
    field.zero();
}

// ============================================================================
// ECM Grid: Apply decay, deposition from fibroblast density field, and clamp
// ============================================================================
FLAMEGPU_HOST_FUNCTION(update_ecm_grid) {
    auto ecm = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("ecm_grid");
    auto field = FLAMEGPU->environment.getMacroProperty<float,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX>("fib_density_field");

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");

    // ECM parameters
    float decay_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_DECAY_RATE");
    float ecm_baseline = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_BASELINE");
    float ecm_saturation = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_SATURATION");
    float release_rate = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_ECM_RELEASE_CAF");
    float dt_sec = FLAMEGPU->environment.getProperty<float>("PARAM_SEC_PER_SLICE");
    float dt = dt_sec / 86400.0f;  // seconds → days

    // ============================================================================
    // Apply ECM dynamics: decay + deposition from Gaussian density field + saturation
    // ============================================================================
    for (int i = 0; i < grid_x; i++) {
        for (int j = 0; j < grid_y; j++) {
            for (int k = 0; k < grid_z; k++) {
                float curr_ecm = ecm[i][j][k];

                // Exponential decay: ECM_n = ECM_{n-1} * exp(-decay_rate * dt)
                float decayed = curr_ecm * expf(-decay_rate * dt);

                // Deposition from Gaussian density field
                float fib_field_val = field[i][j][k];
                float saturation = fminf(curr_ecm / ecm_saturation, 1.0f);
                float deposition = fib_field_val * release_rate / 3.0f * (1.0f - saturation);

                float new_ecm = decayed + deposition;

                // Enforce bounds [baseline, saturation]
                if (new_ecm < ecm_baseline) new_ecm = ecm_baseline;
                if (new_ecm > ecm_saturation) new_ecm = ecm_saturation;

                ecm[i][j][k] = new_ecm;
            }
        }
    }
}

// ============================================================================
// Aggregate ABM Event Counters from Agent States
// Counts cancer cell deaths by cause from agents marked as dead
// ============================================================================
FLAMEGPU_HOST_FUNCTION(aggregate_abm_events) {
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");
    auto cc_api = FLAMEGPU->agent("CancerCell");

    // Get population data for iteration
    flamegpu::DeviceAgentVector cc_agents = cc_api.getPopulationData();
    const unsigned int cc_count = cc_agents.size();

    // Count cancer cell deaths by cause from dead agents
    int cc_death_total = 0;
    int cc_death_natural = 0;
    int cc_death_t_kill = 0;
    int cc_death_mac_kill = 0;

    for (unsigned int i = 0; i < cc_count; i++) {
        if (cc_agents[i].getVariable<int>("dead") != 0) {
            cc_death_total++;
            int reason = cc_agents[i].getVariable<int>("death_reason");
            if (reason == 0) {
                cc_death_natural++;
            } else if (reason == 1) {
                cc_death_t_kill++;
            } else if (reason == 2) {
                cc_death_mac_kill++;
            }
        }
    }

    // Update counter MacroProperty with aggregated values
    counters[ABM_COUNT_CC_DEATH] = cc_death_total;
    counters[ABM_COUNT_CC_DEATH_NATURAL] = cc_death_natural;
    counters[ABM_COUNT_CC_DEATH_T_KILL] = cc_death_t_kill;
    counters[ABM_COUNT_CC_DEATH_MAC_KILL] = cc_death_mac_kill;
}

// ============================================================================
// Copy ABM Event Counters from MacroProperty to Environment Properties
// Called BEFORE QSP so the ODE model can read accumulated counts this step
// ============================================================================
FLAMEGPU_HOST_FUNCTION(copy_abm_counters_to_environment) {
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");

    // Copy from MacroProperty array to environment properties for QSP access
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death", static_cast<int>(counters[ABM_COUNT_CC_DEATH]));
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death_t_kill", static_cast<int>(counters[ABM_COUNT_CC_DEATH_T_KILL]));
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death_mac_kill", static_cast<int>(counters[ABM_COUNT_CC_DEATH_MAC_KILL]));
    FLAMEGPU->environment.setProperty<int>("ABM_cc_death_natural", static_cast<int>(counters[ABM_COUNT_CC_DEATH_NATURAL]));
    FLAMEGPU->environment.setProperty<int>("ABM_TEFF_REC", static_cast<int>(counters[ABM_COUNT_TEFF_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_TH_REC", static_cast<int>(counters[ABM_COUNT_TH_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_TREG_REC", static_cast<int>(counters[ABM_COUNT_TREG_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_MDSC_REC", static_cast<int>(counters[ABM_COUNT_MDSC_REC]));
    FLAMEGPU->environment.setProperty<int>("ABM_MAC_REC", static_cast<int>(counters[ABM_COUNT_MAC_REC]));
}

// ============================================================================
// Reset ABM → QSP Event Counters (called at END of each step)
// Clears MacroProperty array for next step's accumulation
// ============================================================================
FLAMEGPU_HOST_FUNCTION(reset_abm_event_counters) {
    auto counters = FLAMEGPU->environment.getMacroProperty<int, ABM_EVENT_COUNTER_SIZE>("abm_event_counters");

    // Reset all counter elements to zero
    for (int i = 0; i < ABM_EVENT_COUNTER_SIZE; i++) {
        counters[i] = 0;
    }
}

// ============================================================================
// Fibroblast HEAD Division: Create 2 new HEAD cells and convert chain to CAF
// ============================================================================
FLAMEGPU_HOST_FUNCTION(fib_execute_divide) {
    auto fib_api = FLAMEGPU->agent(AGENT_FIBROBLAST);
    const unsigned int fib_count = fib_api.count();
    if (fib_count == 0) return;

    flamegpu::DeviceAgentVector fib_vec = fib_api.getPopulationData();

    const int grid_x = FLAMEGPU->environment.getProperty<int>("grid_size_x");
    const int grid_y = FLAMEGPU->environment.getProperty<int>("grid_size_y");
    const int grid_z = FLAMEGPU->environment.getProperty<int>("grid_size_z");
    const float mean_life = FLAMEGPU->environment.getProperty<float>("PARAM_FIB_LIFE_MEAN");

    auto occ = FLAMEGPU->environment.getMacroProperty<unsigned int,
        OCC_GRID_MAX, OCC_GRID_MAX, OCC_GRID_MAX, NUM_OCC_TYPES>("occ_grid");
    auto fib_pos_x = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_x");
    auto fib_pos_y = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_y");
    auto fib_pos_z = FLAMEGPU->environment.getMacroProperty<int, MAX_FIB_SLOTS>("fib_pos_z");

    // --- Build slot->index map and find max slot ---
    std::unordered_map<int, unsigned int> slot_to_idx;
    int max_slot = -1;
    for (unsigned int i = 0; i < fib_count; i++) {
        int ms = fib_vec[i].getVariable<int>("my_slot");
        if (ms >= 0) {
            slot_to_idx[ms] = i;
            if (ms > max_slot) max_slot = ms;
        }
    }
    int next_slot = max_slot + 1;  // Next free slot index

    // Claimed positions this function call (to avoid duplicate placement)
    std::set<std::tuple<int,int,int>> claimed;

    // 6 face-adjacent directions
    const int dx6[] = {1,-1,0,0,0,0};
    const int dy6[] = {0,0,1,-1,0,0};
    const int dz6[] = {0,0,0,0,1,-1};

    for (unsigned int i = 0; i < fib_count; i++) {
        if (fib_vec[i].getVariable<int>("divide_flag") != 1) continue;

        const int head_slot = fib_vec[i].getVariable<int>("my_slot");
        const int ls        = fib_vec[i].getVariable<int>("leader_slot");
        const int fs        = fib_vec[i].getVariable<int>("fib_state");

        // Safety: must be a HEAD in a chain and still NORMAL
        if (head_slot < 0 || ls == -1 || fs != FIB_NORMAL) {
            fib_vec[i].setVariable<int>("divide_flag", 0);
            continue;
        }

        const int hx = fib_vec[i].getVariable<int>("x");
        const int hy = fib_vec[i].getVariable<int>("y");
        const int hz = fib_vec[i].getVariable<int>("z");

        // --- Find 2 adjacent free voxels in sequence ---
        // NEW_HEAD_2: adjacent to old HEAD
        // NEW_HEAD_1: adjacent to NEW_HEAD_2 (not necessarily adjacent to old HEAD)
        int new_x[2], new_y[2], new_z[2];
        int n_found = 0;

        // Search from HEAD for first free voxel (NEW_HEAD_2 position)
        for (int d = 0; d < 6 && n_found < 1; d++) {
            int nx = hx + dx6[d];
            int ny = hy + dy6[d];
            int nz = hz + dz6[d];
            if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
            if (static_cast<unsigned int>(occ[nx][ny][nz][CELL_TYPE_FIB]) > 0u) continue;
            if (static_cast<unsigned int>(occ[nx][ny][nz][CELL_TYPE_CANCER]) > 0u) continue;
            if (claimed.count({nx, ny, nz})) continue;
            new_x[0] = nx; new_y[0] = ny; new_z[0] = nz;
            n_found = 1;
            claimed.insert({nx, ny, nz});
        }

        // If first cell found, search from IT for second free voxel (NEW_HEAD_1 position)
        if (n_found == 1) {
            for (int d = 0; d < 6; d++) {
                int nx = new_x[0] + dx6[d];
                int ny = new_y[0] + dy6[d];
                int nz = new_z[0] + dz6[d];
                if (nx < 0 || nx >= grid_x || ny < 0 || ny >= grid_y || nz < 0 || nz >= grid_z) continue;
                if (static_cast<unsigned int>(occ[nx][ny][nz][CELL_TYPE_FIB]) > 0u) continue;
                if (static_cast<unsigned int>(occ[nx][ny][nz][CELL_TYPE_CANCER]) > 0u) continue;
                if (claimed.count({nx, ny, nz})) continue;
                new_x[1] = nx; new_y[1] = ny; new_z[1] = nz;
                n_found = 2;
                claimed.insert({nx, ny, nz});
                break;
            }
        }

        // If can't find 2 free voxels, still convert chain but create fewer new cells
        // Assign slots: slot A = NEW_HEAD_2 (next to old HEAD), slot B = NEW_HEAD_1 (outermost)
        int slot_A = next_slot++;    // NEW_HEAD_2: leader_slot = head_slot (points to old HEAD)
        int slot_B = next_slot++;    // NEW_HEAD_1: leader_slot = slot_A   (points to NEW_HEAD_2)

        // Guard against slot overflow
        if (slot_A >= MAX_FIB_SLOTS || slot_B >= MAX_FIB_SLOTS) {
            fib_vec[i].setVariable<int>("divide_flag", 0);
            continue;
        }

        // --- Create NEW_HEAD_2 (adjacent to old HEAD) ---
        if (n_found >= 1) {
            auto cell2 = fib_api.newAgent();
            cell2.setVariable<int>("x", new_x[0]);
            cell2.setVariable<int>("y", new_y[0]);
            cell2.setVariable<int>("z", new_z[0]);
            cell2.setVariable<int>("fib_state", FIB_NORMAL);
            cell2.setVariable<int>("my_slot", slot_A);
            cell2.setVariable<int>("leader_slot", head_slot);  // Points to old HEAD
            cell2.setVariable<int>("life", static_cast<int>(mean_life));
            cell2.setVariable<int>("divide_flag", 0);
            // Write initial pos snapshot to MacroProperty
            fib_pos_x[slot_A] = new_x[0];
            fib_pos_y[slot_A] = new_y[0];
            fib_pos_z[slot_A] = new_z[0];
            // Mark occupancy
            occ[new_x[0]][new_y[0]][new_z[0]][CELL_TYPE_FIB] = 1u;
        }

        // --- Create NEW_HEAD_1 (the new outermost HEAD) ---
        if (n_found >= 2) {
            auto cell1 = fib_api.newAgent();
            cell1.setVariable<int>("x", new_x[1]);
            cell1.setVariable<int>("y", new_y[1]);
            cell1.setVariable<int>("z", new_z[1]);
            cell1.setVariable<int>("fib_state", FIB_NORMAL);
            cell1.setVariable<int>("my_slot", slot_B);
            cell1.setVariable<int>("leader_slot", slot_A);   // Points to NEW_HEAD_2
            cell1.setVariable<int>("life", static_cast<int>(mean_life));
            cell1.setVariable<int>("divide_flag", 0);
            fib_pos_x[slot_B] = new_x[1];
            fib_pos_y[slot_B] = new_y[1];
            fib_pos_z[slot_B] = new_z[1];
            occ[new_x[1]][new_y[1]][new_z[1]][CELL_TYPE_FIB] = 1u;
        }

        // --- Convert entire chain to CAF ---
        // HEAD itself
        fib_vec[i].setVariable<int>("fib_state", FIB_CAF);
        fib_vec[i].setVariable<int>("divide_flag", 0);

        // Follow leader_slot chain to TAIL
        int follow = ls;  // leader_slot of HEAD points to MIDDLE (or TAIL)
        while (follow != -1) {
            auto it = slot_to_idx.find(follow);
            if (it == slot_to_idx.end()) break;
            unsigned int idx = it->second;
            fib_vec[idx].setVariable<int>("fib_state", FIB_CAF);
            follow = fib_vec[idx].getVariable<int>("leader_slot");
        }
    }
}

// ---- Debug checkpoints: disabled for production ----
FLAMEGPU_HOST_FUNCTION(chk_after_zero_occ)    { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_write_occ)   { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_move_cancer) { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_move_tcell)  { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_move_treg)   { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_move_mdsc)   { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_move_vas)    { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_div_cancer)  { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_div_tcell)   { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_div_treg)    { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_after_div_vas)     { /* disabled */ }
FLAMEGPU_HOST_FUNCTION(chk_start_step)     {std::cout << "[debug] Step starting" << std::endl;}
FLAMEGPU_HOST_FUNCTION(chk_break)     {std::cout << "[debug] Made it here" << std::endl;}
} // namespace PDAC