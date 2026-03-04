# SPQSP PDAC Framework Diagrams

This directory contains visual diagrams documenting the architecture and design of the SPQSP PDAC GPU-accelerated ABM simulator.

## Diagram Overview

### Figure 1: System Architecture (GPU/CPU Split)
**File:** `figure1_system_architecture.mmd`

High-level overview of the complete system showing:
- GPU components (ABM, PDE solver)
- CPU components (QSP model, parameters)
- Data flow between GPU and CPU
- Coupling mechanisms (PDE-ABM and QSP-ABM)

**Best used for:** Introductory slides, context setting

---

### Figure 2: Simulation Loop & Layer Execution Order
**File:** `figure2_simulation_loop.mmd`

Detailed flowchart of each ABM timestep (600s), showing:
- 13 GPU layers in execution order
- Recruitment source marking
- Movement and sensing
- Chemical response and state transitions
- Chemical source computation
- Division mechanics
- PDE solve and QSP integration

**Best used for:** Explaining the simulation flow, understanding synchronization points

---

### Figure 3: FLAME GPU Agent Operation Model
**File:** `figure3_flamegpu_model.mmd`

Explanation of FLAME GPU's parallel execution model:
- Agent population in device memory
- Parallel kernel execution (1 thread = 1 agent)
- Occupancy grid and atomic operations
- Message passing between layers
- GPU synchronization

**Best used for:** GPU parallelism concepts, explaining thread model

---

### Figure 4: PDE Solver Architecture
**File:** `figure4_pde_solver.mmd` + PNG visualizations

Complete PDE solver documentation:
- Problem formulation (diffusion-decay-reaction)
- Backward Euler discretization
- Matrix-free CG iteration
- 7-point stencil spatial discretization
- Boundary conditions (Neumann)
- 10 chemical substrates
- Solver properties (unconditionally stable)

**Supporting visualizations:**
- `figure4_stencil_visualization.png` — Visual of 7-point stencil and discretization
- `figure4_solver_workflow.png` — CG iteration algorithm flow

**Best used for:** Technical deep-dive on PDE solver, explaining stability and accuracy

---

### Figure 5: PDE-ABM Coupling
**File:** `figure5_pde_abm_coupling.mmd`

Bidirectional coupling between PDE and agents:
- Layer 7: Agents read PDE concentrations
- Layers 8-9: Chemical response (hypoxia, PDL1, chemotaxis)
- Layer 10: Agents compute chemical sources/sinks
- Layer 11: Atomic writes to PDE grid
- PDE solver: Updates concentrations
- Unit conversion (uptake vs. secretion)

**Best used for:** Explaining agent-environment interaction, data flow between scales

---

### Figure 6: Device Memory & Pointer Management
**File:** `figure6_device_memory.dot` (Graphviz source)

Memory layout on GPU device:
- FLAME GPU agent state (properties, positions)
- Occupancy grid (exclusive cells via CAS)
- PDE grids (10 chemicals, concentrations/sources/uptakes/gradients)
- Environment properties (uint64_t pointers)
- Agent read/write patterns (reinterpret_cast)

**Note:** The `.dot` file is a Graphviz source file. To convert to PNG:
```bash
dot -Tpng figure6_device_memory.dot -o figure6_device_memory.png
```

**Best used for:** Understanding memory layout, explaining pointer-based architecture

---

### Figure 7: Cancer Cell Lifecycle Example
**File:** `figure7_cancer_cell_lifecycle.mmd`

Concrete walkthrough of a cancer cell through each layer:
- Layer 5: Random walk movement with CAS
- Layer 6: Neighbor scanning (26-voxel Moore)
- Layer 7: Reading PDE (O2, IFN-γ)
- Layer 8: State updates (hypoxia, PDL1 response)
- Layer 9: State transitions (division countdown)
- Layer 10: Chemical source computation
- Layer 12: Division (intent + execute)
- Layer 13: PDE solver
- Special: T cell killing

**Best used for:** Concrete example, understanding full agent lifecycle

---

## Viewing & Exporting Diagrams

### Mermaid Diagrams (`.mmd` files)

Mermaid files can be viewed and converted in multiple ways:

#### Option 1: GitHub/GitLab (automatic rendering)
- Push diagrams to repository
- GitHub/GitLab automatically render `.mmd` files in README.md

#### Option 2: Mermaid Live Editor
1. Visit https://mermaid.live
2. Paste `.mmd` file content
3. Export as PNG, SVG, or PDF

#### Option 3: VS Code
1. Install "Markdown Preview Mermaid Support" extension
2. Open `.mmd` file or `.md` containing mermaid block
3. Preview renders diagram in real-time
4. Export via Print → Save as PDF

#### Option 4: Command Line (requires mermaid-cli)
```bash
npm install -g @mermaid-js/mermaid-cli

# Convert to PNG
mmdc -i figure1_system_architecture.mmd -o figure1_system_architecture.png

# Convert to PDF
mmdc -i figure1_system_architecture.mmd -o figure1_system_architecture.pdf
```

### Graphviz Diagrams (`.dot` files)

Graphviz files require Graphviz tools:

```bash
# Install (Ubuntu/Debian)
sudo apt-get install graphviz

# Convert to PNG
dot -Tpng figure6_device_memory.dot -o figure6_device_memory.png

# Convert to PDF
dot -Tpdf figure6_device_memory.dot -o figure6_device_memory.pdf

# Convert to SVG
dot -Tsvg figure6_device_memory.dot -o figure6_device_memory.svg
```

### PNG/Visualization Files
- `figure4_stencil_visualization.png` — Ready to use
- `figure4_solver_workflow.png` — Ready to use

---

## Integration into Slides

### PowerPoint/Google Slides
1. Export Mermaid diagrams to PNG/PDF using Mermaid Live Editor or mermaid-cli
2. Insert image into slide
3. Add caption and annotations

### LaTeX Beamer
```latex
\begin{frame}
  \frametitle{System Architecture}
  \includegraphics[width=0.9\textwidth]{figure1_system_architecture.png}
\end{frame}
```

### Markdown Presentations (reveal.js, remark)
```markdown
# System Architecture

![System Architecture](docs/diagrams/figure1_system_architecture.png)
```

---

## File Summary

| Figure | File | Type | Status | Description |
|--------|------|------|--------|-------------|
| 1 | `figure1_system_architecture.mmd` | Mermaid | ✓ Ready | GPU/CPU split overview |
| 2 | `figure2_simulation_loop.mmd` | Mermaid | ✓ Ready | 13 layers per ABM step |
| 3 | `figure3_flamegpu_model.mmd` | Mermaid | ✓ Ready | Parallel kernel model |
| 4 | `figure4_pde_solver.mmd` | Mermaid | ✓ Ready | Solver architecture |
| 4b | `figure4_stencil_visualization.png` | PNG | ✓ Ready | 7-point stencil visual |
| 4c | `figure4_solver_workflow.png` | PNG | ✓ Ready | CG algorithm flow |
| 5 | `figure5_pde_abm_coupling.mmd` | Mermaid | ✓ Ready | Bidirectional coupling |
| 6 | `figure6_device_memory.dot` | Graphviz | ✓ Ready | Memory layout (needs conversion) |
| 7 | `figure7_cancer_cell_lifecycle.mmd` | Mermaid | ✓ Ready | Agent lifecycle example |

---

## Quick Start for Presentations

**Fastest approach** (no installation needed):
1. Open each `.mmd` file in Mermaid Live Editor (https://mermaid.live)
2. Export directly to PNG/PDF
3. Insert into presentation slides

**Most flexible approach** (for automation):
```bash
# Install mermaid-cli
npm install -g @mermaid-js/mermaid-cli

# Batch convert all Mermaid files
for f in *.mmd; do
  mmdc -i "$f" -o "${f%.mmd}.png"
done
```

---

## Notes

- All Mermaid diagrams are version-controlled in the repository
- PNG visualizations can be regenerated anytime using the Python script or Mermaid tools
- Diagrams are designed for 1920×1080 presentation resolution
- Colors are consistent across all diagrams (GPU=blue, CPU=orange, coupling=green)

---

## Future Enhancements

- [ ] Add T cell recruitment flowchart (Figure 8)
- [ ] Add vascular cell sprouting mechanism (Figure 9)
- [ ] Add QSP-ABM coupling details (Figure 10)
- [ ] Add occupancy grid CAS operations (Figure 11)
- [ ] Add parameter flow from XML to GPU (Figure 12)
