# SPQSP PDAC — GPU Agent-Based Model

GPU-accelerated agent-based model (FLAME GPU 2) with CPU QSP coupling (SUNDIALS CVODE) for simulating pancreatic ductal adenocarcinoma tumor microenvironment dynamics.

## Requirements

- **CUDA Toolkit** 11.0+ (tested up to 12.5)
- **CMake** 3.18+
- **C++17 compiler** (g++ 7+)
- **git**

FLAME GPU 2, SUNDIALS, and Boost are **automatically downloaded and built** if not already present on the system.

## Quick Start

```bash
cd sim
./build.sh                    # That's it — deps are auto-fetched
./build/bin/pdac -s 10 -g 11  # Quick test run
```

First build takes ~8 minutes (downloads and compiles all dependencies). Subsequent builds are incremental (~1-2 min).

## Build Options

```
./build.sh [options]

  --cuda-arch ARCH        Target GPU architecture (e.g., 80 for A100, 90 for H100)
  --debug                 Debug build with CUDA device debugging
  -j, --jobs N            Parallel build jobs (default: nproc)
  --flamegpu PATH         Use local FLAME GPU 2 source instead of fetching
  --clean                 Remove build directory
```

### Using System Libraries

If SUNDIALS or Boost are already installed (e.g., via `module load`), point to them to skip the fetch:

```bash
module load cuda cmake boost sundials      # typical HPC module names
./build.sh

# Or explicitly:
SUNDIALS_DIR=/path/to/sundials BOOST_ROOT=/path/to/boost ./build.sh
```

### CUDA Architecture Reference

| GPU | Architecture |
|-----|-------------|
| V100 | 70 |
| RTX 2080 / T4 | 75 |
| A100 | 80 |
| RTX 3090 | 86 |
| RTX 4090 | 89 |
| H100 | 90 |

Example: `./build.sh --cuda-arch 80` for A100 cluster.

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

## Output Files

Outputs are written to `./outputs/` relative to the **current working directory** (not the executable location):

| File | Contents |
|------|----------|
| `outputs/abm/agents_step_NNNNNN.npy` | Agent positions, states, properties |
| `outputs/pde/pde_step_NNNNNN.npy` | Chemical concentrations (10 species, NumPy format) |
| `outputs/ecm/ecm_step_NNNNNN.npy` | ECM density field |
| `outputs/qsp.csv` | QSP ODE state (153 species) per step |
| `outputs/timing.csv` | Per-step wall-time breakdown |

## Anvil HPC Workflow

Anvil GPU nodes have no internet access, so dependencies must be pre-staged on a login node. Simulation outputs go to `/anvil/scratch/` for I/O performance. CUDA is only available on GPU nodes, so building and running both happen via SLURM.

### One-Time Setup

Run these on a **login node** (which has internet access):

```bash
# Clone the project
cd /anvil/scratch/$USER
git clone <repo-url> SPQSP_PDAC-main
cd SPQSP_PDAC-main

# Pre-fetch dependencies into PDAC/sim/external/ (GPU nodes cannot download these)
cd PDAC/sim
mkdir -p external
git clone --branch v2.0.0-rc.4 --depth 1 https://github.com/FLAMEGPU/FLAMEGPU2.git external/flamegpu2
cd external/flamegpu2 && git submodule update --init --recursive && cd ../..
git clone --branch v4.1.0 --depth 1 https://github.com/LLNL/sundials.git external/sundials

# Download Boost source (module system may not work on all clusters)
cd external
wget https://github.com/boostorg/boost/releases/download/boost-1.82.0/boost-1.82.0.tar.gz
tar xzf boost-1.82.0.tar.gz && mv boost-1.82.0 boost && rm boost-1.82.0.tar.gz
cd ..
```

### Submitting Jobs

```bash
cd /anvil/scratch/$USER/SPQSP_PDAC-main/PDAC/sim

# First submission builds automatically, then runs (500 steps, 50^3 grid)
sbatch submit.sh

# Debug queue (30 min limit, faster scheduling)
sbatch submit_debug.sh -s 50 -g 50

# Custom parameters — pass any pdac flags after the script
sbatch submit.sh -s 1000 -g 101

# Outputs land in /anvil/scratch/$USER/pdac_runs/<job_id>/outputs/
```

On first submission, `submit.sh` detects no binary exists and runs `build.sh` on the GPU node (~8 min) using the pre-staged dependencies. Subsequent submissions skip the build and run immediately.

### What `submit.sh` Does

1. Loads modules: `cmake`, `gcc`, `cuda`, `boost/1.86.0`
2. Builds the binary on the GPU node if it doesn't exist (A100, `--cuda-arch 80`), using local FLAME GPU and SUNDIALS from `external/`
3. Creates a run directory on scratch: `/anvil/scratch/$USER/pdac_runs/<job_id>/`
4. Copies the XML parameter file there for reproducibility
5. `cd`s to scratch and runs the binary (so `./outputs/` writes to scratch)
6. Prints the output path when done

### Rebuilding

To force a rebuild (e.g., after code changes):

```bash
cd /anvil/scratch/$USER/SPQSP_PDAC-main/PDAC/sim
./build.sh --clean          # removes old build (can run from login node)
sbatch submit.sh            # next submission rebuilds on GPU node
```

### Notes

- **First run** after build takes 5-10 minutes for CUDA JIT warmup (not a hang).
- **Memory**: Grid 50^3 uses ~2 GB VRAM; 320^3 uses ~8 GB.
- SLURM logs go to `pdac_<job_id>.out` / `.err` in the directory you submit from.
- The param XML is resolved relative to the executable, so it works from any working directory.
- If you need to update FLAME GPU or SUNDIALS, re-clone into `external/` on a login node.
