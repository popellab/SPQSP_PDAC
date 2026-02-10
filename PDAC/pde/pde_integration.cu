#include "pde_integration.cuh"
#include "../core/common.cuh"
#include <iostream>
#include <vector>

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
    
    // Get PDE source device pointer
    float* d_pde_sources = g_pde_solver->get_device_source_ptr(substrate_idx);
    
    // Launch kernel
    int threads = 256;
    int blocks = (agent_count + threads - 1) / threads;
    
    add_sources_from_agents<<<blocks, threads>>>(
        d_pde_sources,
        d_x, d_y, d_z,
        d_source_rates,
        agent_count,
        substrate_idx,
        grid_x, grid_y, grid_z
    );
    
    cudaDeviceSynchronize();
    
    // Cleanup
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_source_rates);
}

// ============================================================================
// Host Function: Update Agent Chemicals (Read from PDE)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(update_agent_chemicals) {
    if (!g_pde_solver) return;
    
    // Read O2 for all agents
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_O2, "local_O2");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TCELL, CHEM_O2, "local_O2");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TREG, CHEM_O2, "local_O2");
    read_chemical_to_agents(*FLAMEGPU, AGENT_MDSC, CHEM_O2, "local_O2");
    
    // Read IFN-gamma
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_IFN, "local_IFNg");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IFN, "local_IFNg");
    
    // Read IL-2
    read_chemical_to_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IL2, "local_IL2");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TREG, CHEM_IL2, "local_IL2");
    
    // Read IL-10 (immunosuppressive)
    read_chemical_to_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IL10, "local_IL10");
    
    // Read TGF-beta (immunosuppressive)
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_TGFB, "local_TGFB");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TCELL, CHEM_TGFB, "local_TGFB");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TREG, CHEM_TGFB, "local_TGFB");
    read_chemical_to_agents(*FLAMEGPU, AGENT_MDSC, CHEM_TGFB, "local_TGFB");
    
    // Read CCL2 (chemotaxis)
    read_chemical_to_agents(*FLAMEGPU, AGENT_MDSC, CHEM_CCL2, "local_CCL2");

    // Read ArgI (MDSC-produced, T cell response modifier)
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_ARGI, "local_ArgI");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TCELL, CHEM_ARGI, "local_ArgI");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TREG, CHEM_ARGI, "local_ArgI");

    // Read NO (MDSC-produced, T cell response modifier)
    read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_NO, "local_NO");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TCELL, CHEM_NO, "local_NO");
    read_chemical_to_agents(*FLAMEGPU, AGENT_TREG, CHEM_NO, "local_NO");

    // Read IL-12 (macrophage-produced, T cell activator)
    read_chemical_to_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IL12, "local_IL12");

    // Read VEGF-A (cancer-produced, pro-angiogenic)
    // read_chemical_to_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_VEGFA, "local_VEGFA");

    // Note: Nivolumab and Cabozantinib are now handled by QSP compartments
    // They will be transferred to GPU environment properties by the QSP coupling wrapper
    
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    if (step % 50 == 0) {
        std::cout << "Updated agent chemicals from PDE (step " << step << ")" << std::endl;
    }
}

// ============================================================================
// Host Function: Collect Agent Sources (Write to PDE)
// ============================================================================

FLAMEGPU_HOST_FUNCTION(collect_agent_sources) {
    if (!g_pde_solver) return;
    
    // Reset sources for this timestep
    g_pde_solver->reset_sources();
    
    // Collect O2 consumption from cancer cells (should be negative)
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_O2, "O2_uptake_rate");
    
    // Collect IFN-gamma production from T cells
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IFN, "PARAM_IFNG_RELEASE");
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_IFN, "IFNg_uptake_rate");

    // Collect IL-2 production from T cells
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TCELL, CHEM_IL2, "PARAM_IL2_RELEASE");
    
    // Collect IL-2 consumption from Tregs (should be negative)
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TREG, CHEM_IL2, "PARAM_IL2_UPTAKE");
    
    // Collect IL-10 production from Tregs
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TREG, CHEM_IL10, "PARAM_TREG_IL10_RELEASE");
    
    // Collect TGF-beta production from Tregs
    collect_chemical_from_agents(*FLAMEGPU, AGENT_TREG, CHEM_TGFB, "PARAM_TREG_TGFB_RELEASE");
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_TGFB, "TGFB_release_rate");
    
    // Collect CCL2 production from cancer cells
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_CCL2, "CCL2_release_rate");

    // Collect VEGF-A production from cancer cells
    collect_chemical_from_agents(*FLAMEGPU, AGENT_CANCER_CELL, CHEM_VEGFA, "VEGFA_release_rate");

    // Collect NO production from MDSCs
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MDSC, CHEM_NO, "PARAM_NO_RELEASE");

    // Collect ArgI production from MDSCs
    collect_chemical_from_agents(*FLAMEGPU, AGENT_MDSC, CHEM_ARGI, "PARAM_ARGI_RELEASE");

    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    if (step % 50 == 0) {
        std::cout << "Collected agent sources to PDE (step " << step << ")" << std::endl;
    }
}

// ============================================================================
// Host Function: Solve PDE
// ============================================================================

FLAMEGPU_HOST_FUNCTION(solve_pde_step) {
    if (!g_pde_solver) return;
    
    g_pde_solver->solve_timestep();
    
    unsigned int step = FLAMEGPU->environment.getProperty<unsigned int>("current_step");
    if (step % 50 == 0) {
        std::cout << "PDE solved for step " << step << std::endl;
    }
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
    
    std::cout << "PDE device pointers stored in FLAME GPU environment" << std::endl;
}

void cleanup_pde_solver() {
    if (g_pde_solver) {
        delete g_pde_solver;
        g_pde_solver = nullptr;
    }
}

} // namespace PDAC