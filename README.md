# SPQSP PDAC — GPU Agent-Based Model

GPU-accelerated agent-based model (FLAME GPU 2) with CPU QSP coupling (SUNDIALS CVODE) for simulating pancreatic ductal adenocarcinoma tumor microenvironment dynamics.

## Requirements

- **CUDA Toolkit** 11.0+ (tested up to 12.8)
- **CMake** 3.18+
- **C++17 compiler** (g++ 7+)
- **git**

FLAME GPU 2, SUNDIALS, and Boost are **automatically downloaded and built** if not already present on the system.

## Quick Start (HPC)

```bash
cd PDAC/sim
cp cluster.conf.example cluster.conf   # edit ACCOUNT for your allocation
./setup_deps.sh                        # one command — fetches FLAMEGPU2, SUNDIALS, Boost
sbatch submit.sh -s 10 -g 11           # build + test run (first build ~8 min)
```

`submit.sh` auto-detects the cluster (Delta, Anvil), loads modules, builds if needed, runs the simulation on scratch, and copies outputs back to `PDAC/sim/outputs/<job_id>/`.

### Supported Clusters

| Cluster | Partition (default) | CUDA arch | Scratch |
|---------|-------------------|-----------|---------|
| Delta (NCSA) | `gpuA100x4` | 80 (A100), 86 (A40), 90 (H200) | `/work/hdd/<project>/<user>` |
| Anvil (Purdue) | `gpu` | 80 (A100) | `/anvil/scratch/<user>` |

Adding a new cluster: add a `setup_<name>()` function in `submit.sh` and a hostname pattern in `detect_cluster()`.

## Quick Start (Local)

If you have CUDA and cmake available locally (workstation, Docker, etc.):

```bash
cd PDAC/sim
./build.sh                    # auto-fetches deps via network
./build/bin/pdac -s 10 -g 11  # quick test run
```

First build takes ~8 minutes (downloads and compiles all dependencies). Subsequent builds are incremental (~1-2 min).

## Setup Details

### `cluster.conf`

Per-user SLURM settings. Only `ACCOUNT` is required — everything else is auto-detected:

```bash
ACCOUNT="bgre-delta-gpu"     # required
# PARTITION=""                # override auto-detected partition
# CUDA_ARCH=""                # override auto-detected GPU arch
# SCRATCH_BASE=""             # override auto-detected scratch path
```

### `setup_deps.sh`

Fetches FLAMEGPU2, SUNDIALS, and Boost into `external/`. Idempotent — skips deps already present. Run on a node with internet access (login nodes).

```bash
./setup_deps.sh              # fetch all deps
./setup_deps.sh --status     # check what's fetched
./setup_deps.sh --clean      # remove external/ and re-fetch
```

If your cluster has internet on GPU nodes, you can skip `setup_deps.sh` — CMake will auto-fetch deps during the build.

### `build.sh`

Portable build script. Works anywhere with cmake, nvcc, g++, and git on PATH.

```
./build.sh [options]

  --cuda-arch ARCH        Target GPU architecture (e.g., 80 for A100, 90 for H100)
  --debug                 Debug build with CUDA device debugging
  -j, --jobs N            Parallel build jobs (default: nproc)
  --flamegpu PATH         Use local FLAME GPU 2 source instead of fetching
  --clean                 Remove build directory
```

System-installed libraries can be used via environment variables:

```bash
SUNDIALS_DIR=/path/to/sundials BOOST_ROOT=/path/to/boost ./build.sh
```

## Running

```bash
./build/bin/pdac [options]

  -g, --grid-size N       Grid dimensions [8-320] (default: 50 = 1mm^3 at 20um)
  -s, --steps N           Simulation steps (default: 500, each step = 6 hours)
  -r, --radius N          Initial tumor radius in voxels (default: 5)
  -t, --tcells N          Initial T cell count (default: 50)
  -p, --param-file PATH   XML parameter file (default: resource/param_all_test.xml)
  -oa, --output-agents N  0=no agent output, 1=output (default: 1)
  -op, --output-pde N     0=no PDE output, 1=output (default: 1)
  -i, --qsp-init FLAG     0=skip, 1=run QSP to steady state before ABM (default: 0)
```

### SLURM Submission

```bash
sbatch submit.sh                       # defaults: 500 steps, 50^3 grid
sbatch submit.sh -s 1000 -g 101       # custom parameters
```

What `submit.sh` does:
1. Reads `cluster.conf` for account name
2. Auto-detects cluster, loads modules, picks partition and CUDA arch
3. Builds the binary on the GPU node if it doesn't exist
4. Creates a run directory on scratch for fast I/O
5. Runs the simulation
6. Copies outputs back to `PDAC/sim/outputs/<job_id>/`

### Rebuilding

```bash
./build.sh --clean          # removes old build
sbatch submit.sh            # next submission rebuilds
```

## Output Files

Outputs are written to `./outputs/` relative to the working directory. On SLURM, they are also copied to `PDAC/sim/outputs/<job_id>/`.

| File | Contents |
|------|----------|
| `outputs/abm/agents_step_NNNNNN.abm.lz4` | Agent positions, states, properties (LZ4 compressed) |
| `outputs/pde/pde_step_NNNNNN.pde.lz4` | Chemical concentrations (10 species, LZ4 compressed) |
| `outputs/ecm/ecm_step_NNNNNN.npy` | ECM density field (NumPy format) |
| `outputs/qsp_<seed>.csv` | QSP ODE state (153 species) per step |
| `outputs/stats_<seed>.csv` | Per-step agent counts, recruitment, proliferation, death events |
| `outputs/timing_<seed>.csv` | Per-step wall-time breakdown |
| `outputs/layer_timing.csv` | Per-layer wall-time breakdown |

### CUDA Architecture Reference

| GPU | Architecture |
|-----|-------------|
| V100 | 70 |
| RTX 2080 / T4 | 75 |
| A100 | 80 |
| A40 / RTX 3090 | 86 |
| RTX 4090 | 89 |
| H100 / H200 | 90 |

## Notes

- **First run** after build takes 5-10 minutes for CUDA JIT warmup (not a hang).
- **Memory**: Grid 50^3 uses ~2 GB VRAM; 320^3 uses ~8 GB.
- SLURM logs go to `pdac_<job_id>.out` / `.err` in the directory you submit from.
- The param XML is resolved relative to the executable, so it works from any working directory.
