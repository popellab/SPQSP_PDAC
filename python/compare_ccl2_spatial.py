#!/usr/bin/env python3
"""
Spatial CCL2 distribution comparison: HCC vs PDAC
- Central z-slice heatmaps at multiple timesteps
- 1D radial profiles from grid center
- Histogram of voxel concentrations
"""
import csv, numpy as np, glob, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

SIM_DIR    = os.path.join(os.path.dirname(__file__), "../PDAC/sim")
HCC_DIR    = os.path.join(SIM_DIR, "HCC_outputs2/snapShots")
PDAC_DIR   = os.path.join(SIM_DIR, "outputs/pde")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "outputs/comparison")
os.makedirs(OUT_DIR, exist_ok=True)

NX = NY = NZ = 50

def load_pde_grid(path, col_name, col_map=None):
    """Load one chemical from a CSV into a (NX,NY,NZ) numpy array."""
    grid = np.zeros((NX, NY, NZ), dtype=np.float32)
    with open(path) as f:
        r = csv.reader(f)
        raw_h = [h.strip() for h in next(r)]
        # Apply column name mapping if provided
        h = [col_map.get(c, c) if col_map else c for c in raw_h]
        xi = h.index('x')
        yi = h.index('y')
        zi = h.index('z')
        ci = h.index(col_name)
        for row in r:
            if len(row) <= ci: continue
            try:
                x,y,z = int(row[xi]), int(row[yi]), int(row[zi])
                grid[x,y,z] = float(row[ci])
            except (ValueError, IndexError):
                pass
    return grid

HCC_COL_MAP = {'IFNg':'IFN','IL_2':'IL2','ArgI':'ARGI','CCL2':'CCL2',
                'NO':'NO','TGFB':'TGFB','IL10':'IL10','IL12':'IL12',
                'VEGFA':'VEGFA','O2':'O2'}

def get_hcc_steps():
    files = sorted(glob.glob(os.path.join(HCC_DIR, "grid_core_*.csv")),
                   key=lambda x: int(x.split('_')[-1].replace('.csv','')))
    return [(int(f.split('_')[-1].replace('.csv','')), f) for f in files]

def get_pdac_steps():
    files = sorted(glob.glob(os.path.join(PDAC_DIR, "pde_step_*.csv")),
                   key=lambda x: int(x.split('_')[-1].replace('.csv','')))
    return [(int(f.split('_')[-1].replace('.csv','')), f) for f in files]

# ── Pick comparison steps ──────────────────────────────────────────────────────
hcc_steps  = get_hcc_steps()
pdac_steps = get_pdac_steps()

# Use every ~50th step, up to 6 panels
n_panels = 6
hcc_sample  = hcc_steps[::max(1, len(hcc_steps)//n_panels)][:n_panels]
pdac_sample = pdac_steps[::max(1, len(pdac_steps)//n_panels)][:n_panels]
n_panels = min(len(hcc_sample), len(pdac_sample))

center_z = NZ // 2

# ── Figure 1: Central z-slice heatmaps ────────────────────────────────────────
fig, axes = plt.subplots(2, n_panels, figsize=(4*n_panels, 8))
fig.suptitle('CCL2 — Central Z-Slice (z=25)', fontsize=14, fontweight='bold')

# Find global color scale across all panels
all_vals = []
for i in range(n_panels):
    gh = load_pde_grid(hcc_sample[i][1],  'CCL2', HCC_COL_MAP)
    gp = load_pde_grid(pdac_sample[i][1], 'CCL2')
    all_vals.append(gh[:,:,center_z].max())
    all_vals.append(gp[:,:,center_z].max())
vmax = np.percentile(all_vals, 95)

for i in range(n_panels):
    hs, hp = hcc_sample[i][0],  hcc_sample[i][1]
    ps, pp = pdac_sample[i][0], pdac_sample[i][1]

    gh = load_pde_grid(hp, 'CCL2', HCC_COL_MAP)
    gp = load_pde_grid(pp, 'CCL2')

    slh = gh[:,:,center_z].T   # shape (NY, NX)
    slp = gp[:,:,center_z].T

    im0 = axes[0,i].imshow(slh, vmin=0, vmax=vmax, cmap='hot', origin='lower')
    axes[0,i].set_title(f'HCC step {hs}\nmean={gh.mean():.3f}', fontsize=9)
    axes[0,i].set_xlabel('x'); axes[0,i].set_ylabel('y')

    im1 = axes[1,i].imshow(slp, vmin=0, vmax=vmax, cmap='hot', origin='lower')
    axes[1,i].set_title(f'PDAC step {ps}\nmean={gp.mean():.3f}', fontsize=9)
    axes[1,i].set_xlabel('x'); axes[1,i].set_ylabel('y')

plt.colorbar(im0, ax=axes[0,:].tolist(), label='CCL2 conc', shrink=0.6)
plt.colorbar(im1, ax=axes[1,:].tolist(), label='CCL2 conc', shrink=0.6)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ccl2_slices.png'), dpi=120, bbox_inches='tight')
print("Saved: ccl2_slices.png")
plt.close()

# ── Figure 2: Radial CCL2 profiles ────────────────────────────────────────────
cx, cy, cz = NX//2, NY//2, NZ//2
# Precompute radial distances for every voxel
ix, iy, iz = np.meshgrid(np.arange(NX), np.arange(NY), np.arange(NZ), indexing='ij')
r_vox = np.sqrt((ix-cx)**2 + (iy-cy)**2 + (iz-cz)**2)
r_bins = np.arange(0, NX//2+1, 1)

def radial_profile(grid):
    """Mean CCL2 concentration in each radial shell."""
    means = []
    for rb in range(len(r_bins)-1):
        mask = (r_vox >= r_bins[rb]) & (r_vox < r_bins[rb+1])
        vals = grid[mask]
        means.append(vals.mean() if len(vals) > 0 else 0)
    return np.array(means)

fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 4), sharey=False)
fig.suptitle('CCL2 — Radial Profile from Grid Center', fontsize=13, fontweight='bold')

for i in range(n_panels):
    gh = load_pde_grid(hcc_sample[i][1],  'CCL2', HCC_COL_MAP)
    gp = load_pde_grid(pdac_sample[i][1], 'CCL2')

    rh = radial_profile(gh)
    rp = radial_profile(gp)
    r_um = (r_bins[:-1] + 0.5) * 20  # convert voxels → µm

    axes[i].plot(r_um, rh, 'b-o', ms=3, label='HCC')
    axes[i].plot(r_um, rp, 'r-o', ms=3, label='PDAC')
    axes[i].set_title(f'Step HCC={hcc_sample[i][0]} PDAC={pdac_sample[i][0]}', fontsize=9)
    axes[i].set_xlabel('Radius (µm)')
    axes[i].set_ylabel('Mean CCL2')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ccl2_radial.png'), dpi=120, bbox_inches='tight')
print("Saved: ccl2_radial.png")
plt.close()

# ── Figure 3: Concentration histogram ─────────────────────────────────────────
fig, axes = plt.subplots(1, n_panels, figsize=(4*n_panels, 4))
fig.suptitle('CCL2 — Voxel Concentration Histogram', fontsize=13, fontweight='bold')

for i in range(n_panels):
    gh = load_pde_grid(hcc_sample[i][1],  'CCL2', HCC_COL_MAP)
    gp = load_pde_grid(pdac_sample[i][1], 'CCL2')

    bins = np.linspace(0, max(gh.max(), gp.max()), 60)
    axes[i].hist(gh.ravel(), bins=bins, alpha=0.5, color='blue', label='HCC', density=True)
    axes[i].hist(gp.ravel(), bins=bins, alpha=0.5, color='red',  label='PDAC', density=True)
    axes[i].set_title(f'HCC={hcc_sample[i][0]} PDAC={pdac_sample[i][0]}\n'
                      f'HCC mean={gh.mean():.3f} PDAC mean={gp.mean():.3f}', fontsize=8)
    axes[i].set_xlabel('CCL2 conc')
    axes[i].set_ylabel('Density')
    axes[i].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'ccl2_histogram.png'), dpi=120, bbox_inches='tight')
print("Saved: ccl2_histogram.png")
plt.close()

# ── Print summary stats ────────────────────────────────────────────────────────
print("\n── CCL2 Summary Stats ──")
print(f"{'Step':>6}  {'HCC mean':>10}  {'HCC max':>10}  {'PDAC mean':>10}  {'PDAC max':>10}  {'ratio mean':>10}")
for i in range(n_panels):
    gh = load_pde_grid(hcc_sample[i][1],  'CCL2', HCC_COL_MAP)
    gp = load_pde_grid(pdac_sample[i][1], 'CCL2')
    print(f"  H{hcc_sample[i][0]:3d}/P{pdac_sample[i][0]:3d}  "
          f"{gh.mean():10.4f}  {gh.max():10.4f}  "
          f"{gp.mean():10.4f}  {gp.max():10.4f}  "
          f"{gp.mean()/gh.mean():10.2f}x")
