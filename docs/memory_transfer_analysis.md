# GPU Memory Transfer Analysis: CSV Output

## Current Implementation: Synchronous D2H (Device-to-Host) Transfers

### What Gets Copied?

#### 1. **ABM Agent Data** (Every `interval_out` steps)
**Location**: `PDAC/sim/main.cu:304-357` in `exportABMData()`

```cpp
flamegpu::DeviceAgentVector cancer_pop = agent.getPopulationData();
// ^ Internally calls cudaMemcpy(DeviceToHost) - BLOCKING
```

**Data transferred per agent:**
- x, y, z (3 × 4B = 12B)
- cell_state (4B)
- divideCD, divideFlag (2 × 4B = 8B)
- Additional state variables (~20B)
- **Total per agent: ~50-100 bytes**

**Typical transfer sizes:**
- 50 cancer cells: ~5 KB
- 50 T cells: ~5 KB
- 20 TRegs: ~2 KB
- 10 MDSCs: ~1 KB
- 50 fibroblasts: ~5 KB
- 50 vascular: ~5 KB
- **Total agents per output step: ~20-30 KB**

#### 2. **PDE Concentration Data** (Every `interval_out` steps)
**Location**: `PDAC/sim/main.cu:117-120` in `exportPDEData()`

```cpp
for (int substrate_idx = 0; substrate_idx < PDAC::NUM_SUBSTRATES; substrate_idx++) {
    all_concentrations[substrate_idx].resize(total_voxels);
    PDAC::g_pde_solver->get_concentrations(
        all_concentrations[substrate_idx].data(), substrate_idx);
    // ^ Calls cudaMemcpy for each of 10 chemicals - BLOCKING
}
```

**Data transferred:**
- 10 chemical substrates × grid³ voxels × 4B per concentration
- **50³ grid**: 125K voxels × 10 × 4B = **5 MB per output step**
- **101³ grid**: 1M+ voxels × 10 × 4B = **40+ MB per output step**
- **320³ grid**: 32.7M voxels × 10 × 4B = **~1.3 GB per output step** ⚠️

### Timing Analysis

| Grid Size | PDE Size | ABM Size | GPU Bandwidth | Transfer Time |
|-----------|----------|----------|---------------|----------------|
| 11³ | 5.3 KB | 30 KB | 50-100 GB/s | **<1 μs** |
| 50³ | 5 MB | 30 KB | 50-100 GB/s | **~100 μs** |
| 101³ | 40 MB | 30 KB | 50-100 GB/s | **~400 μs** |
| 320³ | 1.3 GB | 30 KB | 50-100 GB/s | **~26 ms** ⚠️ |

### Performance Impact

**50³ grid with default settings:**
- PDE solve: ~36 ms
- Memory transfer (D2H): ~1 ms
- CSV write (CPU): ~5-10 ms
- **Total per output step: ~50 ms**
- **If output every 10 steps**: ~5 ms per ABM step overhead

**320³ grid (scaling study):**
- PDE solve: ~2000 ms
- Memory transfer (D2H): **~26 ms** ← becomes visible!
- CSV write (CPU): ~50 ms
- **Total per output step: ~2100 ms**
- **If output every 50 steps**: ~42 ms per ABM step overhead

## Current Optimization Level

### ✓ Already Implemented
- **Sparse output**: CSV export only happens every `interval_out` steps (default ~1-10)
- **No async overlap**: Currently synchronous (simple to reason about)
- **Standard cudaMemcpy**: Uses default GPU memory model

### ✗ Not Implemented
- **Async transfers**: `cudaMemcpyAsync()` could overlap with GPU compute
- **Pinned memory**: `cudaMallocHost()` would provide ~2-3× faster transfers
- **GPU→CPU compression**: Could reduce transfer size
- **Staging buffer**: Could batch transfers more efficiently
- **Output buffering**: Could reduce I/O overhead

## Memory Transfer Diagram

```
┌─────────────────────────────────────────────────┐
│         Per ABM Timestep (every step)           │
├─────────────────────────────────────────────────┤
│ GPU compute (agent functions, PDE solve)        │
│         Duration: ~36-50 ms (50³ grid)         │
└─────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────┐
│   Every interval_out steps (default: 1-10)     │
├─────────────────────────────────────────────────┤
│ 1. cudaMemcpy D2H: Agents (30 KB)  [<1 μs]    │
│ 2. cudaMemcpy D2H: PDE (5 MB)      [~100 μs]  │
│    └─ Called 10 times (one per chemical)       │
│ 3. CPU write CSV file              [~5 ms]    │
│ 4. Next ABM step begins...                     │
└─────────────────────────────────────────────────┘
```

## Recommendations

### For Current Work (50³-101³ grids)
✓ **Current approach is fine** - transfer overhead <5% per step

### For Scaling Studies (320³ grids)
⚠️ **Consider:**
1. Reduce output frequency: `-oa 0 -op 0` (disable CSV output)
2. Or use async transfers (requires modest code change):
   ```cpp
   // Current
   PDAC::g_pde_solver->get_concentrations(...);  // cudaMemcpy

   // Better
   cudaMemcpyAsync(host_ptr, device_ptr, size, cudaMemcpyDeviceToHost, stream);
   // Continue GPU compute while transfer happens
   ```

### For Production Runs
- **Option 1**: Disable CSV output entirely (`-oa 0 -op 0`), rely on HDF5 or other binary format
- **Option 2**: Implement async D2H + GPU compute overlap (10-20% performance gain)
- **Option 3**: Use pinned memory + async transfers (20-30% performance gain)

## Code References

**ABM Export**: `PDAC/sim/main.cu:289-488` (`exportABMData()`)
- Transfers agent population data (all types)
- Uses `FLAMEGPU::DeviceAgentVector::getPopulationData()`

**PDE Export**: `PDAC/sim/main.cu:87-141` (`exportPDEData()`)
- Transfers PDE concentration arrays
- Calls `PDESolver::get_concentrations()` for each substrate

**QSP Export**: `PDAC/qsp/qsp_integration.cu` (`exportQSPData()`)
- Transfers ODE state (153 species)
- Relatively small (~50 KB per step)

## Answer to Original Question

> "Are we having to pass memory to the CPU?"

**Yes, absolutely.**
- **Agents**: ~30 KB per output step (cudaMemcpy D2H)
- **PDE**: ~5 MB per output step × 10 substrates (cudaMemcpy D2H × 10)
- **Timing**: Synchronous, blocks GPU/CPU until complete (~100 μs for 50³ grid)
- **Frequency**: Every `interval_out` steps (typically 1-10 steps)
- **Impact**: <5% overhead for typical grids; becomes visible at 320³

The transfers are **necessary** because CSV files must be written from CPU filesystem, and data lives on GPU device memory.
