# I/O Performance Optimization

## Problem

The PDAC simulation writes two types of output every N steps:

1. **PDE data** — chemical concentration fields (10 substrates × grid voxels × float32)
2. **ABM data** — agent state snapshots (all agents × 8 fields × int32)

At production scale (250×250×500 grid, 6M+ agents), the original implementation added **3,726 ms/step of I/O overhead — more than doubling the total runtime (127% of compute time)**. Most of that time was spent copying agent data from GPU to CPU one variable at a time, iterating over millions of agents on the CPU, then writing synchronously to disk.

## What Changed

Three optimizations, each targeting a different bottleneck:

### 1. Async PDE export with pinned memory

**Before:** Each PDE output step made 10 separate synchronous `cudaMemcpy` calls (one per chemical substrate), waited for each to finish, then wrote to disk — all blocking the simulation.

**After:** A single `cudaMemcpyAsync` copies all 10 substrates at once into a pre-allocated "pinned" host buffer on a dedicated CUDA stream. A background thread waits for the copy to finish, then writes to disk. The simulation continues immediately.

**Key concepts:**
- **Pinned (page-locked) memory** — a special allocation (`cudaMallocHost`) that the GPU can DMA-transfer to directly, without the OS paging it out. Regular `malloc`'d memory requires an extra copy through a staging buffer.
- **CUDA streams** — independent queues of GPU operations. By putting the data transfer on its own stream (`cudaStreamNonBlocking`), it runs in parallel with the simulation's compute kernels on the default stream.
- **Double buffering** — two pinned buffers alternate: while one is being written to disk by the background thread, the other receives the next transfer. This prevents stalls.

### 2. GPU-side ABM packing

**Before:** Agent data was collected using FLAMEGPU's `DeviceAgentVector` API, which copies each agent variable from GPU to CPU in bulk, then iterates over every agent on the CPU to pack it into an output buffer. With 6M+ agents at production scale, this CPU-side iteration was the dominant I/O bottleneck.

**After:** Each agent type has a lightweight GPU kernel (`pack_for_export`) that reads its own data and writes it directly into a shared device buffer using `atomicAdd` for thread-safe indexing. The packed buffer is then async-copied to host (same pattern as PDE). Total time: **0.11 ms/step** (vs thousands of ms before).

**Key concepts:**
- **Agent functions** — in FLAMEGPU, each agent runs a small GPU function in parallel. We added a `pack_for_export` function that each agent executes to write its own row to the output buffer.
- **atomicAdd** — a GPU primitive that lets thousands of threads safely increment a shared counter without conflicts. Each agent atomically claims a row index, then writes its data.
- **Conditional execution** — a `do_abm_export` flag in the simulation environment controls whether agents actually pack data. On non-export steps, the kernels return immediately with near-zero cost.

### 3. LZ4 compression

**Before:** Output files were uncompressed NumPy (`.npy`) format.

**After:** Output is LZ4-compressed binary (`.pde.lz4` / `.abm.lz4`). LZ4 compresses at ~4 GB/s, so the CPU cost is negligible, but the smaller files reduce disk write time and storage. At production scale, each step writes ~1.45 GB uncompressed; LZ4 reduces this 3-5×.

## Benchmark Results

All benchmarks ran on Purdue Anvil A100 GPUs with 9 repetitions per configuration.

### Production scale: 250×250×500 grid, 6M+ agents, 50 steps

| | Old | New |
|---|---|---|
| No I/O (baseline) | 2,945 ± 18 ms/step | 3,332 ± 13 ms/step |
| I/O every step | 6,672 ± 73 ms/step | 3,701 ± 17 ms/step |
| **I/O overhead** | **3,726 ms (127%)** | **369 ms (11%)** |

**90% reduction in I/O overhead. 1.80× speedup with I/O enabled.**

Note: the new binary's no-I/O baseline is higher than the old binary's because the main branch added new compute (ECM grid updates). The I/O optimization itself is what matters: the new binary's overhead is 369 ms vs the old binary's 3,726 ms.

#### Where the time goes (new binary, io_every_step)

| Phase | ms/step | % |
|-------|---------|---|
| PDE solve | 1,971 | 53% |
| ECM update | 576 | 16% |
| Movement | 194 | 5% |
| Neighbor scan | 191 | 5% |
| Division | 62 | 2% |
| Recruitment | 51 | 1% |
| State transitions + chemical sources | 27 | 1% |
| **I/O (ABM collect + PDE export)** | **0.17** | **<0.01%** |
| Other (QSP, gradients) | 8 | <1% |
| Unaccounted (step function overhead) | ~370 | 10% |

### Medium scale: 100³ grid, 750k agents, 200 steps, 9 reps

| Config | Old (ms/step) | New (ms/step) | Speedup |
|--------|---------------|---------------|---------|
| No I/O (baseline) | 917 ± 81 | 850 ± 97 | 1.08× |
| I/O every step | 1,238 ± 59 | 922 ± 83 | **1.34×** |

**I/O overhead reduced from 321 ms/step (35%) to 72 ms/step (8.4%).**

## Memory Cost

The optimization allocates additional memory for double-buffered async transfers:

| Buffer | Size formula | 250×250×500 | 100³ |
|--------|-------------|-------------|------|
| PDE pinned (×2) | 2 × 10 × voxels × 4B | 2,384 MB | 76 MB |
| ABM pinned (×2) | 2 × max_agents × 8 × 4B | ~400 MB | ~96 MB |
| ABM device buffer | max_agents × 8 × 4B | ~200 MB | ~48 MB |

At production scale: ~3 GB pinned host RAM + ~200 MB GPU memory. The simulation itself uses ~18 GB GPU memory, so the export buffers add ~1%.

## Disk Usage

With I/O every step at 250×250×500:
- PDE: ~1.25 GB/step uncompressed (~250-400 MB with LZ4)
- ABM: ~200 MB/step uncompressed (~50-100 MB with LZ4)
- **300 steps ≈ 100-150 GB compressed** on disk

For production runs, writing every 5-10 steps reduces this to 10-30 GB while keeping I/O overhead under 2%.

## Files Changed

- `main.cu` — async PDE/ABM export, pinned memory management, LZ4 writers
- `agents/pack_for_export.cuh` — GPU kernels for agent data packing (7 per-type functions)
- `model_definition.cu` — register pack functions, add export environment properties
- `model_layers.cu` — add prepare + pack layers to simulation step
- `CMakeLists.txt` — add C language support and vendored LZ4 source
- `third_party/lz4/` — vendored LZ4 library (single .c/.h, ~3700 lines)
- `benchmark_io.sh` — A/B benchmark runner with repetitions
- `analyze_benchmark.py` — benchmark analysis with per-layer timing and statistical aggregation
