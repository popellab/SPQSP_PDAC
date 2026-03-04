# PDAC GPU Parameter Reference

## Summary
Complete reference of all behavioral, chemical, and simulation parameters for the SPQSP PDAC GPU-accelerated agent-based model with CPU QSP coupling.

**Last Updated**: 2026-02-04
**Version**: 1.0
**Status**: GPU implementation aligned with CPU HCC param_all_test.xml

---

## Table of Contents
1. [Movement Parameters](#1-movement-parameters)
2. [Division & Lifespan Parameters](#2-division--lifespan-parameters)
3. [Chemical Parameters](#3-chemical-parameters)
4. [Hill Function Parameters](#4-hill-function-parameters)
5. [Voxel Capacity Parameters](#5-voxel-capacity-parameters)
6. [Simulation Parameters](#6-simulation-parameters)
7. [Environment Setup](#7-environment-setup)
8. [Parameter Sources](#8-parameter-sources)

---

## 1. Movement Parameters

### Cancer Cells
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `cancer_move_prob` | 0.1 | [0-1] | Probability of movement per timestep (Progenitor/Senescent) |
| `cancer_stem_move_prob` | 0.05 | [0-1] | Probability of movement per timestep (Stem cells) |

**Location**: `model_definition.cu:471-472`

### T Cells
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `tcell_move_prob` | 0.5 | [0-1] | Movement probability (Effector state) |
| `tcell_cyt_move_prob` | 0.3 | [0-1] | Movement probability (Cytotoxic state) |

**Location**: `model_definition.cu:476-477`

### Regulatory T Cells (Tregs)
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `treg_move_prob` | 0.3 | [0-1] | Movement probability |

**Location**: `model_definition.cu:487`

### MDSCs
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `mdsc_move_prob` | 0.3 | [0-1] | Movement probability |

**Location**: `model_definition.cu:508`

---

## 2. Division & Lifespan Parameters

### Cancer Cells
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `cancer_stem_div_interval` | 24.0 | timesteps | Division interval for stem cells |
| `cancer_progenitor_div_interval` | 12.0 | timesteps | Division interval for progenitors |
| `cancer_progenitor_div_max` | 10 | [count] | Max divisions before senescence |
| `cancer_asymmetric_div_prob` | 0.8 | [0-1] | Probability stem → (stem + progenitor) |
| `cancer_senescent_mean_life` | 48.0 | timesteps | Mean lifespan of senescent cells |

**Location**: `model_definition.cu:429-445`

### T Cells
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `tcell_div_interval` | 24 | timesteps | Division cooldown |
| `tcell_div_limit` | 10 | [count] | Maximum divisions per cell |
| `tcell_life_mean` | 100.0 | timesteps | Mean lifespan |
| `tcell_IL2_release_time` | 86400.0 | seconds | Duration of IL-2 release (24 hours) |
| `tcell_IFN_release_time` | 86400.0 | seconds | Duration of IFN-gamma release (24 hours) |

**Location**: `model_definition.cu:449-466`

### Tregs
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `treg_div_interval` | 48 | timesteps | Division cooldown |
| `treg_div_limit` | 10 | [count] | Maximum divisions |
| `treg_div_prob` | 0.01 | [0-1] | Base division probability |
| `treg_density_factor` | 0.1 | [0-1] | Density-dependent suppression |
| `treg_life_mean` | 100.0 | timesteps | Mean lifespan |

**Location**: `model_definition.cu:479-493`

### MDSCs
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `mdsc_life_mean` | 100.0 | timesteps | Mean lifespan (no division) |

**Location**: `model_definition.cu:498`

---

## 3. Chemical Parameters

### Diffusion Coefficients (cm²/s)
| Chemical | Value | Source | Notes |
|----------|-------|--------|-------|
| O2 | 2.8e-5 | CPU HCC | Oxygen |
| IFN | 1.0e-7 | CPU HCC | IFN-gamma |
| IL2 | 4.0e-8 | CPU HCC | Interleukin-2 |
| IL10 | 1.4e-8 | CPU HCC | Interleukin-10 |
| TGFB | 2.6e-7 | CPU HCC | TGF-beta |
| CCL2 | 1.31e-8 | CPU HCC | CCL2 chemokine |
| ARGI | 1.0e-6 | CPU HCC | Arginase I |
| NO | 3.8e-5 | CPU HCC | Nitric Oxide |
| IL12 | 2.4e-8 | CPU HCC | Interleukin-12 |
| VEGFA | 2.9e-7 | CPU HCC | VEGF-A |

**Location**: `pde_integration.cu:317-326`

### Decay Rates (1/s)
| Chemical | Value | Source | Notes |
|----------|-------|--------|-------|
| O2 | 1.0e-5 | CPU HCC | Consumption/metabolism |
| IFN | 6.5e-5 | CPU HCC |  |
| IL2 | 2.78e-5 | CPU HCC |  |
| IL10 | 4.6e-5 | CPU HCC |  |
| TGFB | 1.65e-4 | CPU HCC |  |
| CCL2 | 1.67e-5 | CPU HCC |  |
| ARGI | 2.0e-6 | CPU HCC |  |
| NO | 1.56e-3 | CPU HCC | Highly unstable |
| IL12 | 6.4e-5 | CPU HCC |  |
| VEGFA | 1.921e-4 | CPU HCC |  |

**Location**: `pde_integration.cu:329-338`

### Production/Consumption Rates (mol/cell/step)
| Chemical | Agent Type | Value | Parameter Name |
|----------|-----------|-------|---|
| O2 | Cancer (base) | 1.0e-5 | `O2_uptake_cancer` |
| O2 | Cancer (progenitor) | 1.5e-5 | `O2_uptake_cancer * 1.5` |
| IFN-gamma | T cell (base) | 1.0e-6 | `IFNg_release_rate_base` |
| IFN-gamma | Cancer (uptake) | 1.0e-7 | `IFN_uptake_cancer` |
| IL-2 | T cell (base) | 5.0e-7 | `IL2_release_rate_base` |
| IL-2 | Treg (consumption) | 2.0e-7 | `IL2_consumption_treg` |
| IL-10 | Treg (base) | 8.0e-7 | `IL10_release_rate_base` |
| TGF-beta | Treg (base) | 5.0e-7 | `TGFB_release_rate_base` |
| CCL2 | Cancer (base) | 1.0e-6 | `CCL2_release_rate_base` |
| CCL2 | Cancer (hypoxia) | 2.0e-6 | `CCL2_release_rate_base * 2.0` when hypoxic |
| CCL2 | Cancer (stem) | 1.5e-6 | `CCL2_release_rate_base * 1.5` for stem |
| ArgI | MDSC | 1.62e-10 | `ArgI_release_rate_base` |
| NO | MDSC | 1.67e-10 | `NO_release_rate_base` |
| IL12 | MDSC (proxy) | 6.88e-12 | `IL12_release_rate_base` |
| VEGFA | Cancer | 1.27e-12 | `VEGFA_release_rate_base` |

**Location**: Various agent_*.cuh files and model_definition.cu

---

## 4. Hill Function Parameters

### PDL1 Upregulation (IFN-gamma dependent)
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `IFN_PDL1_EC50` | 5.0e-7 | M | IFN concentration for 50% PDL1 upregulation |
| `IFN_PDL1_hill` | 2.0 | [unitless] | Hill coefficient (cooperativity) |
| `PDL1_syn_max` | 0.5 | per step | Maximum synthesis rate |
| `PDL1_decay_rate` | 0.1 | per step | Fraction decaying per timestep |
| `PDL1_basal` | 0.1 | [0-1] | Basal expression level |
| `PDL1_max` | 1.0 | [0-1] | Maximum PDL1 level |

**Formula**: `PDL1_syn = PDL1_syn_max * Hill(IFN, IFN_PDL1_EC50, IFN_PDL1_hill)`

**Location**: `model_definition.cu:531-542`

### T Cell Exhaustion (PD1-PDL1 suppression)
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `PD1_PDL1_half` | 0.5 | [0-1] | EC50 for PD1-PDL1 suppression |
| `n_PD1_PDL1` | 2.0 | [unitless] | Hill coefficient |
| `tcell_exhaust_base_PDL1` | 0.99 | [0-1] | Base exhaustion rate from PDL1 |
| `tcell_exhaust_base_Treg` | 0.99 | [0-1] | Base exhaustion rate from Treg |

**Exhaustion Probability**:
- From PDL1: `p_exhaust_PDL1 = 1 - 0.99^(supp_PDL1 * q)`
- From Treg: `p_exhaust_Treg = 1 - 0.99^q`

**Location**: `model_definition.cu:470-475`, `cancer_cell.cuh:353-387`

### IL10/TGF-beta T Cell Suppression
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `IL10_T_cell_suppress_EC50` | 5.0e-7 | M | IL-10 concentration for 50% suppression |
| `TGFB_T_cell_suppress_EC50` | 3.0e-7 | M | TGF-beta concentration for 50% suppression |
| `suppression_max` | 0.9 | [0-1] | Maximum combined suppression |

**Formula**: `suppression = suppression_max * (Hill(IL10, EC50_IL10, 2.0) + Hill(TGFB, EC50_TGFB, 2.0)) / 2.0`

**Location**: `t_cell.cuh:760-768`

### Nivolumab-PD1 Blockade
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `NIVO_PD1_EC50` | 1.0e-8 | M | Nivolumab concentration for 50% blockade |
| `NIVO_PD1_hill` | 1.0 | [unitless] | Hill coefficient |

**Location**: `model_definition.cu:525-526`

### Cabozantinib Anti-angiogenic Effect
| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `CABO_EC50` | 1.0e-7 | M | Cabozantinib EC50 |
| `CABO_hill` | 1.0 | [unitless] | Hill coefficient |

**Location**: `cancer_cell.cuh:821-823`

---

## 5. Voxel Capacity Parameters

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `MAX_T_PER_VOXEL` | 8 | [count] | Maximum T cells in empty voxel |
| `MAX_T_PER_VOXEL_WITH_CANCER` | 1 | [count] | Maximum T cells when cancer present |
| `MAX_CANCER_PER_VOXEL` | 1 | [count] | Maximum cancer cells per voxel |
| `MAX_MDSC_PER_VOXEL` | 1 | [count] | Maximum MDSCs per voxel (exclusive) |

**Location**: `core/common.cuh:52-55`

---

## 6. Simulation Parameters

| Parameter | Value | Units | Description |
|-----------|-------|-------|-------------|
| `voxel_size` | 10.0 | µm | Spatial resolution |
| `sec_per_slice` | 1200.0 | seconds | Time per ABM step (20 minutes) |
| `dt_abm` | 1200.0 | seconds | ABM timestep for PDE coupling |
| `molecular_steps` | 10 | [count] | PDE substeps per ABM step |
| `dt_pde` | 120.0 | seconds | PDE timestep |

**Location**: `main.cu:509-510`, `pde_integration.cu:292-293`

---

## 7. Environment Setup

### Initialization Parameters
| Parameter | Default Value | Type | Description |
|-----------|--------------|------|-------------|
| `grid_size` | 51 | [voxels] | Simulation domain size |
| `steps` | 500 | [timesteps] | Total simulation duration |
| `cluster_radius` | 5 | [voxels] | Initial tumor radius |
| `num_tcells` | 50 | [count] | Initial T cell count |
| `num_tregs` | 10 | [count] | Initial Treg count |
| `num_mdscs` | 5 | [count] | Initial MDSC count |

**Command-line options**:
```bash
./pdac -g 51 -s 500 -r 5 -t 50 --tregs 10 --mdscs 5 -m 111
```

---

## 8. Parameter Sources

### GPU Parameters (Current)
- **File**: `PDAC/sim/model_definition.cu` (lines 429-548)
- **File**: `PDAC/pde/pde_integration.cu` (lines 317-338)
- **File**: `PDAC/agents/*.cuh` (agent-specific parameters)
- **File**: `PDAC/core/common.cuh` (voxel capacities)

### CPU Parameters (Reference)
- **File**: `/home/chase/SPQSP/SPQSP_HCC-main/HCC/HCC_single/resource/param_all_test.xml`
- **File**: `/home/chase/SPQSP/SPQSP_HCC-main/HCC/SP_QSP_HCC/abm/agent/*.cpp`

### Alignment Status
- ✓ Phase 1: Chemical parameters (diffusion, decay) aligned with CPU HCC
- ✓ Phase 2: Agent killing/exhaustion formulas verified
- ✓ Phase 4: All behavioral parameters documented above
- ⏳ Phase 5: Pending validation testing

---

## Validation and Testing

### Test Scenarios
1. **Unit Test**: Single voxel with 1 cancer, 1 T cell
2. **Proliferation Test**: 10x10x10 grid, measure growth rates
3. **Immune Response**: Measure killing probability vs. T cell count
4. **Comparative**: Identical parameters, GPU vs. CPU output within 10%

### Success Criteria
- Population trajectories match within 10% error
- Spatial distributions qualitatively similar
- State distributions (stem/progenitor, eff/cyt) match
- No numerical instabilities (NaN, overflow)

---

## Future Parameter Extensions

### QSP-specific parameters (Phase 5+)
- Drug PK parameters (absorption, distribution, clearance)
- Systemic T cell recruitment rates
- Central compartment volumes
- Drug-target binding kinetics

### Advanced features
- Spatial heterogeneity in parameters (normoxia vs. hypoxia)
- Genetic heterogeneity in cancer cells
- Immune memory/training effects
- Angiogenesis dynamics

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2026-02-04 | Claude Code | Initial comprehensive reference |

For questions or updates, consult the CLAUDE.md file in the project root.
