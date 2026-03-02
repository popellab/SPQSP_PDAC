#!/usr/bin/env python3
"""
Compare PDAC GPU model outputs vs HCC CPU model outputs.
ABM agent populations and PDE mean concentrations over time.

HCC cell_N.csv columns: x, y, z, Type, State, extra
  Type: 1=CANCER, 2=T, 3=TCD4, 4=MDSC, 5=MAC, 6=FIB, 8=VAS
  State (cancer): 6=STEM, 7=PROGENITOR, 8=SENESCENT
  State (T):      3=EFF, 4=CYT, 5=SUPP
  State (TCD4):   9=Th, 10=TREG
  State (MDSC):   0=DEFAULT
  State (MAC):    13=M1, 14=M2
  State (FIB):    16=CAF
  State (VAS):    17=TIP, 18=STALK(excluded), 19=PHALANX

PDAC agents_step_NNNNNN.csv columns: agent_type, agent_id, x, y, z, cell_state, additional_info
  agent_type: CANCER, TCELL, TREG, MDSC, MACROPHAGE, FIBROBLAST, VAS
  cell_state (cancer): STEM, PROGENITOR, SENESCENT
  cell_state (T):      EFFECTOR (or numeric 0)
  cell_state (VAS):    TIP, PHALANX

PDAC pde_step_NNNNNN.csv columns: x, y, z, O2, IFN, IL2, IL10, TGFB, CCL2, ARGI, NO, IL12, VEGFA
"""

import os
import glob
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Paths ───────────────────────────────────────────────────────────────────
SIM_DIR      = os.path.join(os.path.dirname(__file__), "../PDAC/sim")
PDAC_ABM_DIR = os.path.join(SIM_DIR, "outputs/abm")
PDAC_PDE_DIR = os.path.join(SIM_DIR, "outputs/pde")
PDAC_QSP_CSV = os.path.join(SIM_DIR, "outputs/qsp.csv")
HCC_DIR      = os.path.join(SIM_DIR, "HCC_outputs2/snapShots")  # Use HCC_outputs2 (no presim)
HCC_STATS_DIR = os.path.join(SIM_DIR, "HCC_outputs2")  # For stats_*.csv files
OUT_DIR      = os.path.join(os.path.dirname(__file__), "outputs/comparison")
os.makedirs(OUT_DIR, exist_ok=True)

# ─── Parse HCC ABM ────────────────────────────────────────────────────────────
def parse_hcc_step(step):
    path = os.path.join(HCC_DIR, f"cell_{step}.csv")
    counts = dict(cancer_stem=0, cancer_prog=0, cancer_sen=0,
                  tcell=0, teff=0, tcyt=0, tsup=0,
                  th=0, treg=0,
                  mdsc=0, mac=0, mac_m1=0, mac_m2=0, fib=0,
                  vas_tip=0, vas_phalanx=0)
    if not os.path.exists(path):
        return counts
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 5:
                continue
            t, s = int(row[3]), int(row[4])
            if   t == 1 and s == 6:  counts['cancer_stem']  += 1
            elif t == 1 and s == 7:  counts['cancer_prog']  += 1
            elif t == 1 and s == 8:  counts['cancer_sen']   += 1
            elif t == 2:
                counts['tcell'] += 1
                if   s == 3: counts['teff'] += 1
                elif s == 4: counts['tcyt'] += 1
                elif s == 5: counts['tsup'] += 1
            elif t == 3 and s == 9:  counts['th']           += 1
            elif t == 3 and s == 10: counts['treg']          += 1
            elif t == 4:             counts['mdsc']          += 1
            elif t == 5:
                counts['mac']           += 1
                if   s == 13: counts['mac_m1'] += 1
                elif s == 14: counts['mac_m2'] += 1
            elif t == 6:             counts['fib']           += 1
            elif t == 8 and s == 17: counts['vas_tip']       += 1
            elif t == 8 and s == 19: counts['vas_phalanx']   += 1
            # s==18 STALK excluded
    return counts

# ─── Parse PDAC ABM ───────────────────────────────────────────────────────────
def parse_pdac_step(step):
    path = os.path.join(PDAC_ABM_DIR, f"agents_step_{step:06d}.csv")
    counts = dict(cancer_stem=0, cancer_prog=0, cancer_sen=0,
                  tcell=0, teff=0, tcyt=0, tsup=0,
                  th=0, treg=0,
                  mdsc=0, mac=0, mac_m1=0, mac_m2=0, fib=0,
                  vas_tip=0, vas_phalanx=0)
    if not os.path.exists(path):
        return counts
    with open(path) as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 6:
                continue
            atype, state = row[0].strip(), row[5].strip()
            if atype == 'CANCER':
                if   state == 'STEM':       counts['cancer_stem'] += 1
                elif state == 'PROGENITOR': counts['cancer_prog'] += 1
                elif state == 'SENESCENT':  counts['cancer_sen']  += 1
            elif atype == 'TCELL':
                counts['tcell'] += 1
                if   state == 'EFFECTOR':   counts['teff'] += 1
                elif state == 'CYTOTOXIC':  counts['tcyt'] += 1
                elif state == 'SUPPRESSED': counts['tsup'] += 1
            elif atype == 'TREG':
                if   state == 'TH':         counts['th']   += 1
                else:                       counts['treg'] += 1  # REGULATORY or fallback
            elif atype == 'MDSC':        counts['mdsc']       += 1
            elif atype == 'MAC':
                counts['mac']        += 1
                if   state == 'M1' or state == '0': counts['mac_m1'] += 1
                elif state == 'M2' or state == '1': counts['mac_m2'] += 1
            elif atype == 'FIB':         counts['fib']        += 1
            elif atype == 'VAS':
                if   state == 'TIP':     counts['vas_tip']     += 1
                elif state == 'PHALANX': counts['vas_phalanx'] += 1
    return counts

# ─── Parse PDE means (generic) ───────────────────────────────────────────────
# Canonical chemical names used throughout this script
PDE_CHEMS = ['O2','IFN','IL2','IL10','TGFB','CCL2','ARGI','NO','IL12','VEGFA']

# Map from HCC grid_core column names → canonical names
HCC_PDE_COL_MAP = {
    'O2':    'O2',
    'IFNg':  'IFN',
    'IL_2':  'IL2',
    'IL10':  'IL10',
    'TGFB':  'TGFB',
    'CCL2':  'CCL2',
    'ArgI':  'ARGI',
    'NO':    'NO',
    'IL12':  'IL12',
    'VEGFA': 'VEGFA',
}

def _parse_pde_means(path, col_map=None):
    """Read a CSV with x,y,z,chem... columns; return dict of mean per canonical chem."""
    means = {c: np.nan for c in PDE_CHEMS}
    if not os.path.exists(path):
        return means
    data = {c: [] for c in PDE_CHEMS}
    with open(path) as f:
        reader = csv.reader(f)
        raw_header = [h.strip() for h in next(reader)]
        # Build index: canonical_name -> column index
        idx = {}
        for raw_col, col_idx in zip(raw_header, range(len(raw_header))):
            canonical = col_map.get(raw_col, raw_col) if col_map else raw_col
            if canonical in PDE_CHEMS:
                idx[canonical] = col_idx
        for row in reader:
            for c, i in idx.items():
                if i < len(row) and row[i].strip():  # Skip if column missing or empty
                    try:
                        data[c].append(float(row[i]))
                    except ValueError:
                        pass  # Skip malformed values
    for c in PDE_CHEMS:
        if data[c]:
            means[c] = np.mean(data[c])
    return means

def parse_pdac_pde(step):
    path = os.path.join(PDAC_PDE_DIR, f"pde_step_{step:06d}.csv")
    return _parse_pde_means(path, col_map=None)  # PDAC columns already canonical

def parse_hcc_pde(step):
    path = os.path.join(HCC_DIR, f"grid_core_{step}.csv")
    return _parse_pde_means(path, col_map=HCC_PDE_COL_MAP)

# ─── Parse HCC stats for recruitment/proliferation events ─────────────────────
def parse_hcc_stats():
    """Read HCC stats_*.csv file and extract recruitment/proliferation events."""
    import glob as glob_module
    stats_files = sorted(glob_module.glob(os.path.join(HCC_STATS_DIR, "stats_*.csv")))
    if not stats_files:
        print("WARNING: No HCC stats files found in", HCC_STATS_DIR)
        return {}, {}

    # Read the most recent stats file (should be one per run)
    stats_file = stats_files[-1] if stats_files else None
    if not stats_file or not os.path.exists(stats_file):
        return {}, {}

    events = {}
    try:
        with open(stats_file) as f:
            header = f.readline().strip().split(',')
            # Find column indices for T cell and TReg events
            col_map = {h.strip(): i for i, h in enumerate(header)}

            for line in f:
                row = line.strip().split(',')
                step = int(row[0]) if row else 0

                # Extract recruitment and proliferation events
                event_data = {}
                if 'recruit.CD8.effector' in col_map:
                    event_data['tc_recruit'] = float(row[col_map['recruit.CD8.effector']]) if col_map['recruit.CD8.effector'] < len(row) else 0
                if 'prolif.CD8.cytotoxic' in col_map:
                    event_data['tc_prolif'] = float(row[col_map['prolif.CD8.cytotoxic']]) if col_map['prolif.CD8.cytotoxic'] < len(row) else 0
                if 'recruit.Th.default' in col_map:
                    event_data['th_recruit'] = float(row[col_map['recruit.Th.default']]) if col_map['recruit.Th.default'] < len(row) else 0
                if 'prolif.Th.default' in col_map:
                    event_data['th_prolif'] = float(row[col_map['prolif.Th.default']]) if col_map['prolif.Th.default'] < len(row) else 0
                if 'prolif.Treg.default' in col_map:
                    event_data['treg_prolif'] = float(row[col_map['prolif.Treg.default']]) if col_map['prolif.Treg.default'] < len(row) else 0

                events[step] = event_data
    except Exception as e:
        print(f"Error parsing HCC stats file: {e}")
        return {}, {}

    # Convert to lists indexed by step
    steps = sorted(events.keys())

    # HCC stats are cumulative, so convert to per-step differences
    tc_recruit_cumul = [events.get(s, {}).get('tc_recruit', 0) for s in steps]
    tc_prolif_cumul = [events.get(s, {}).get('tc_prolif', 0) for s in steps]
    th_recruit_cumul = [events.get(s, {}).get('th_recruit', 0) for s in steps]
    th_prolif_cumul = [events.get(s, {}).get('th_prolif', 0) for s in steps]
    treg_prolif_cumul = [events.get(s, {}).get('treg_prolif', 0) for s in steps]

    # Compute per-step rates: current_step - previous_step
    tc_recruit = [tc_recruit_cumul[i] - (tc_recruit_cumul[i-1] if i > 0 else 0) for i in range(len(tc_recruit_cumul))]
    tc_prolif = [tc_prolif_cumul[i] - (tc_prolif_cumul[i-1] if i > 0 else 0) for i in range(len(tc_prolif_cumul))]
    th_recruit = [th_recruit_cumul[i] - (th_recruit_cumul[i-1] if i > 0 else 0) for i in range(len(th_recruit_cumul))]
    th_prolif = [th_prolif_cumul[i] - (th_prolif_cumul[i-1] if i > 0 else 0) for i in range(len(th_prolif_cumul))]
    treg_prolif = [treg_prolif_cumul[i] - (treg_prolif_cumul[i-1] if i > 0 else 0) for i in range(len(treg_prolif_cumul))]

    return steps, {'tc_recruit': tc_recruit, 'tc_prolif': tc_prolif,
                   'th_recruit': th_recruit, 'th_prolif': th_prolif,
                   'treg_prolif': treg_prolif}

# ─── Parse PDAC event tracking ────────────────────────────────────────────────
def parse_pdac_events():
    """Read PDAC outputs/event.csv file and extract recruitment/proliferation events."""
    event_file = os.path.join(SIM_DIR, "outputs/event.csv")
    if not os.path.exists(event_file):
        print("WARNING: No PDAC event.csv found in", SIM_DIR)
        return {}, {}

    events = {}
    try:
        with open(event_file) as f:
            header = f.readline().strip().split(',')
            # Find column indices
            col_map = {h.strip(): i for i, h in enumerate(header)}

            for line_num, line in enumerate(f, 2):
                row = line.strip().split(',')
                if not row or not row[0]:
                    continue
                try:
                    step = int(row[0])
                except ValueError:
                    continue

                # Extract event counts (columns: prolif.CD8.cytotoxic, recruit.CD8.effector, prolif.Th.default, prolif.Treg.default)
                event_data = {}
                if 'prolif.CD8.cytotoxic' in col_map:
                    try:
                        idx = col_map['prolif.CD8.cytotoxic']
                        event_data['tc_prolif'] = float(row[idx]) if idx < len(row) else 0
                    except (ValueError, IndexError):
                        event_data['tc_prolif'] = 0
                if 'recruit.CD8.effector' in col_map:
                    try:
                        idx = col_map['recruit.CD8.effector']
                        event_data['tc_recruit'] = float(row[idx]) if idx < len(row) else 0
                    except (ValueError, IndexError):
                        event_data['tc_recruit'] = 0
                if 'prolif.Th.default' in col_map:
                    try:
                        idx = col_map['prolif.Th.default']
                        event_data['th_prolif'] = float(row[idx]) if idx < len(row) else 0
                    except (ValueError, IndexError):
                        event_data['th_prolif'] = 0
                if 'recruit.Th.default' in col_map:
                    try:
                        idx = col_map['recruit.Th.default']
                        event_data['th_recruit'] = float(row[idx]) if idx < len(row) else 0
                    except (ValueError, IndexError):
                        event_data['th_recruit'] = 0
                if 'prolif.Treg.default' in col_map:
                    try:
                        idx = col_map['prolif.Treg.default']
                        event_data['treg_prolif'] = float(row[idx]) if idx < len(row) else 0
                    except (ValueError, IndexError):
                        event_data['treg_prolif'] = 0

                events[step] = event_data
    except Exception as e:
        print(f"Error parsing PDAC event.csv file: {e}")
        return {}, {}

    # Convert to lists indexed by step
    # PDAC events are already per-step rates (counters reset each step), NOT cumulative
    steps = sorted(events.keys())
    tc_recruit = [events.get(s, {}).get('tc_recruit', 0) for s in steps]
    tc_prolif = [events.get(s, {}).get('tc_prolif', 0) for s in steps]
    th_recruit = [events.get(s, {}).get('th_recruit', 0) for s in steps]
    th_prolif = [events.get(s, {}).get('th_prolif', 0) for s in steps]
    treg_prolif = [events.get(s, {}).get('treg_prolif', 0) for s in steps]

    return steps, {'tc_recruit': tc_recruit, 'tc_prolif': tc_prolif,
                   'th_recruit': th_recruit, 'th_prolif': th_prolif, 'treg_prolif': treg_prolif}

# ─── Collect over all steps ───────────────────────────────────────────────────
# Determine available steps from both models
hcc_steps  = sorted([int(f.split('_')[1].split('.')[0])
                     for f in os.listdir(HCC_DIR)
                     if f.startswith('cell_') and f.endswith('.csv')
                     and 'Zone' not in f])
pdac_files = sorted(glob.glob(os.path.join(PDAC_ABM_DIR, "agents_step_*.csv")))
pdac_steps = [int(os.path.basename(f).split('_')[2].split('.')[0]) for f in pdac_files]

print(f"HCC steps:  {len(hcc_steps)}  ({hcc_steps[0]}–{hcc_steps[-1]})")
print(f"PDAC steps: {len(pdac_steps)} ({pdac_steps[0]}–{pdac_steps[-1]})")

print("Parsing HCC ABM...", flush=True)
hcc_counts = {k: [] for k in ['cancer_stem','cancer_prog','cancer_sen',
                                'tcell','teff','tcyt','tsup',
                                'th','treg',
                                'mdsc','mac','mac_m1','mac_m2','fib',
                                'vas_tip','vas_phalanx']}
for s in hcc_steps:
    c = parse_hcc_step(s)
    for k in hcc_counts:
        hcc_counts[k].append(c[k])

print("Parsing PDAC ABM...", flush=True)
pdac_counts = {k: [] for k in hcc_counts}
for s in pdac_steps:
    c = parse_pdac_step(s)
    for k in pdac_counts:
        pdac_counts[k].append(c[k])

print("Parsing PDAC PDE...", flush=True)
pdac_pde = {c: [] for c in PDE_CHEMS}
for s in pdac_steps:
    m = parse_pdac_pde(s)
    for c in PDE_CHEMS:
        pdac_pde[c].append(m[c])

print("Parsing HCC PDE...", flush=True)
hcc_pde = {c: [] for c in PDE_CHEMS}
for s in hcc_steps:
    m = parse_hcc_pde(s)
    for c in PDE_CHEMS:
        hcc_pde[c].append(m[c])

print("Parsing HCC stats (recruitment/proliferation)...", flush=True)
hcc_event_steps, hcc_events = parse_hcc_stats()

print("Parsing PDAC events (recruitment/proliferation)...", flush=True)
pdac_event_steps, pdac_events = parse_pdac_events()

# ─── Print summary at final step ─────────────────────────────────────────────
print("\n═══ Final step comparison ═══")
print(f"{'Population':<20} {'HCC':>10} {'PDAC':>10}")
print("-" * 42)
pop_labels = [
    ('Cancer Stem',    'cancer_stem'),
    ('Cancer Prog',    'cancer_prog'),
    ('Cancer Sen',     'cancer_sen'),
    ('T Cells',        'tcell'),
    ('  T Effector',   'teff'),
    ('  T Cytotoxic',  'tcyt'),
    ('  T Suppressed', 'tsup'),
    ('T Helper (Th)',  'th'),
    ('TRegs',          'treg'),
    ('MDSCs',          'mdsc'),
    ('Macrophages',    'mac'),
    ('  MAC M1',       'mac_m1'),
    ('  MAC M2',       'mac_m2'),
    ('Fibroblasts',    'fib'),
    ('VAS TIP',        'vas_tip'),
    ('VAS Phalanx',    'vas_phalanx'),
]
for label, key in pop_labels:
    h = hcc_counts[key][-1]  if hcc_counts[key]  else 0
    p = pdac_counts[key][-1] if pdac_counts[key] else 0
    print(f"{label:<20} {h:>10} {p:>10}")

print("\n═══ PDE mean concentrations at final step ═══")
print(f"  {'Chem':<8} {'HCC':>12} {'PDAC':>12}")
print("  " + "-" * 34)
for c in PDE_CHEMS:
    h = hcc_pde[c][-1]  if hcc_pde[c]  else float('nan')
    p = pdac_pde[c][-1] if pdac_pde[c] else float('nan')
    print(f"  {c:<8} {h:>12.6g} {p:>12.6g}")

# ─── Plots ────────────────────────────────────────────────────────────────────
HCC_COLOR  = 'tab:blue'
PDAC_COLOR = 'tab:orange'

hcc_t  = np.array(hcc_steps,  dtype=float)
pdac_t = np.array(pdac_steps, dtype=float)

# --- ABM populations ---
# Each panel shows one cell type; multi-state types use different line styles.
# HCC = blue, PDAC = orange throughout.
abm_panels = [
    ("Cancer (all)",     [('Stem', 'cancer_stem'), ('Prog', 'cancer_prog'), ('Sen', 'cancer_sen')]),
    ("T Cells (states)", [('Eff', 'teff'), ('Cyt', 'tcyt'), ('Sup', 'tsup')]),
    ("Th + TReg",        [('Th', 'th'), ('TReg', 'treg')]),
    ("MDSCs",            [('',   'mdsc')]),
    ("Macrophages",      [('M1', 'mac_m1'), ('M2', 'mac_m2')]),
    ("Fibroblasts",      [('',   'fib')]),
    ("VAS TIP",          [('',   'vas_tip')]),
    ("VAS Phalanx",      [('',   'vas_phalanx')]),
    ("Cancer stem",      [('',   'cancer_stem')]),
    ("Cancer prog",      [('',   'cancer_prog')]),
]
linestyles = ['-', '--', ':']

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
fig.suptitle("ABM Population Comparison  ■ HCC (blue)  ■ PDAC GPU (orange)", fontsize=13)
for ax, (title, series) in zip(axes.flat, abm_panels):
    for i, (lbl, key) in enumerate(series):
        ls = linestyles[i]
        suffix = f" {lbl}" if lbl else ""
        ax.plot(hcc_t,  hcc_counts[key],  color=HCC_COLOR,  lw=1.8, ls=ls, label=f"HCC{suffix}")
        ax.plot(pdac_t, pdac_counts[key], color=PDAC_COLOR, lw=1.8, ls=ls, label=f"PDAC{suffix}")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("Step")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "abm_comparison.png"), dpi=120)
print(f"\nSaved: {OUT_DIR}/abm_comparison.png")

# --- PDE mean concentrations: HCC (blue) vs PDAC (orange) ---
fig2, axes2 = plt.subplots(2, 5, figsize=(20, 8))
fig2.suptitle("Mean PDE Concentrations  ■ HCC (blue)  ■ PDAC GPU (orange)", fontsize=13)
for ax, chem in zip(axes2.flat, PDE_CHEMS):
    hcc_vals  = np.array(hcc_pde[chem],  dtype=float)
    pdac_vals = np.array(pdac_pde[chem], dtype=float)
    ax.plot(hcc_t,  hcc_vals,  color=HCC_COLOR,  lw=1.8, label="HCC")
    ax.plot(pdac_t, pdac_vals, color=PDAC_COLOR, lw=1.8, label="PDAC")
    ax.set_title(chem, fontsize=10)
    ax.set_xlabel("Step")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "pde_means.png"), dpi=120)
print(f"Saved: {OUT_DIR}/pde_means.png")

# --- T Cell and TReg dynamics ---
fig3, axes3 = plt.subplots(2, 3, figsize=(18, 10))
fig3.suptitle("T Cell and TReg Dynamics  ■ HCC (blue)  ■ PDAC GPU (orange)", fontsize=13)

# T Cells: Effector
ax = axes3[0, 0]
ax.plot(hcc_t,  hcc_counts['teff'],  color=HCC_COLOR,  lw=2.0, label="HCC")
ax.plot(pdac_t, pdac_counts['teff'], color=PDAC_COLOR, lw=2.0, label="PDAC")
ax.set_title("T Effector", fontsize=11)
ax.set_xlabel("Step"); ax.set_ylabel("Count")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# T Cells: Cytotoxic
ax = axes3[0, 1]
ax.plot(hcc_t,  hcc_counts['tcyt'],  color=HCC_COLOR,  lw=2.0, label="HCC")
ax.plot(pdac_t, pdac_counts['tcyt'], color=PDAC_COLOR, lw=2.0, label="PDAC")
ax.set_title("T Cytotoxic", fontsize=11)
ax.set_xlabel("Step"); ax.set_ylabel("Count")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# T Cells: Suppressed
ax = axes3[0, 2]
ax.plot(hcc_t,  hcc_counts['tsup'],  color=HCC_COLOR,  lw=2.0, label="HCC")
ax.plot(pdac_t, pdac_counts['tsup'], color=PDAC_COLOR, lw=2.0, label="PDAC")
ax.set_title("T Suppressed", fontsize=11)
ax.set_xlabel("Step"); ax.set_ylabel("Count")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# T Helper
ax = axes3[1, 0]
ax.plot(hcc_t,  hcc_counts['th'],  color=HCC_COLOR,  lw=2.0, label="HCC")
ax.plot(pdac_t, pdac_counts['th'], color=PDAC_COLOR, lw=2.0, label="PDAC")
ax.set_title("T Helper (Th)", fontsize=11)
ax.set_xlabel("Step"); ax.set_ylabel("Count")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# TRegs
ax = axes3[1, 1]
ax.plot(hcc_t,  hcc_counts['treg'],  color=HCC_COLOR,  lw=2.0, label="HCC")
ax.plot(pdac_t, pdac_counts['treg'], color=PDAC_COLOR, lw=2.0, label="PDAC")
ax.set_title("TReg (Regulatory)", fontsize=11)
ax.set_xlabel("Step"); ax.set_ylabel("Count")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

# Suppressed fraction of total T cells
ax = axes3[1, 2]
hcc_sup_frac  = np.array(hcc_counts['tsup'],  dtype=float) / (np.array(hcc_counts['tcell'],  dtype=float) + 1)
pdac_sup_frac = np.array(pdac_counts['tsup'], dtype=float) / (np.array(pdac_counts['tcell'], dtype=float) + 1)
ax.plot(hcc_t,  hcc_sup_frac,  color=HCC_COLOR,  lw=2.0, label="HCC")
ax.plot(pdac_t, pdac_sup_frac, color=PDAC_COLOR, lw=2.0, label="PDAC")
ax.set_title("Suppressed Fraction of T Cells", fontsize=11)
ax.set_xlabel("Step"); ax.set_ylabel("Fraction")
ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "tcell_treg_dynamics.png"), dpi=120)
print(f"Saved: {OUT_DIR}/tcell_treg_dynamics.png")

# --- HCC T Cell and TReg recruitment/proliferation events ---
# if hcc_event_steps:
#     fig4, axes4 = plt.subplots(2, 2, figsize=(14, 10))
#     fig4.suptitle("HCC T Cell and TReg Events (Recruitment & Proliferation)", fontsize=13)

#     hcc_event_t = np.array(hcc_event_steps, dtype=float)

#     # T Cell recruitment
#     ax = axes4[0, 0]
#     if 'tc_recruit' in hcc_events and hcc_events['tc_recruit']:
#         ax.plot(hcc_event_t, hcc_events['tc_recruit'], color='blue', lw=2.0, label='CD8 Recruitment')
#         ax.fill_between(hcc_event_t, hcc_events['tc_recruit'], alpha=0.3, color='blue')
#     ax.set_title("T Cell (CD8) Recruitment Events", fontsize=11)
#     ax.set_xlabel("Step")
#     ax.set_ylabel("Recruitment Rate")
#     ax.grid(True, alpha=0.3)

#     # T Cell proliferation
#     ax = axes4[0, 1]
#     if 'tc_prolif' in hcc_events and hcc_events['tc_prolif']:
#         ax.plot(hcc_event_t, hcc_events['tc_prolif'], color='green', lw=2.0, label='CD8 Proliferation')
#         ax.fill_between(hcc_event_t, hcc_events['tc_prolif'], alpha=0.3, color='green')
#     ax.set_title("T Cell (CD8) Proliferation Events", fontsize=11)
#     ax.set_xlabel("Step")
#     ax.set_ylabel("Proliferation Rate")
#     ax.grid(True, alpha=0.3)

#     # TH proliferation
#     ax = axes4[1, 0]
#     if 'th_prolif' in hcc_events and hcc_events['th_prolif']:
#         ax.plot(hcc_event_t, hcc_events['th_prolif'], color='purple', lw=2.0, label='TH Proliferation')
#         ax.fill_between(hcc_event_t, hcc_events['th_prolif'], alpha=0.3, color='purple')
#     ax.set_title("T Helper (TH) Proliferation Events", fontsize=11)
#     ax.set_xlabel("Step")
#     ax.set_ylabel("Proliferation Rate")
#     ax.grid(True, alpha=0.3)

#     # TReg proliferation
#     ax = axes4[1, 1]
#     if 'treg_prolif' in hcc_events and hcc_events['treg_prolif']:
#         ax.plot(hcc_event_t, hcc_events['treg_prolif'], color='red', lw=2.0, label='TReg Proliferation')
#         ax.fill_between(hcc_event_t, hcc_events['treg_prolif'], alpha=0.3, color='red')
#     ax.set_title("TReg Proliferation Events", fontsize=11)
#     ax.set_xlabel("Step")
#     ax.set_ylabel("Proliferation Rate")
#     ax.grid(True, alpha=0.3)

#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_DIR, "hcc_events.png"), dpi=120)
#     print(f"Saved: {OUT_DIR}/hcc_events.png")

# --- PDAC and HCC event comparison with normalization ---
if pdac_event_steps and hcc_event_steps:
    fig5, axes5 = plt.subplots(2, 3, figsize=(16, 10))
    fig5.suptitle("T Cell and TReg Events (Normalized per cell)  ■ HCC (blue)  ■ PDAC GPU (orange)", fontsize=13)

    pdac_event_t = np.array(pdac_event_steps, dtype=float)
    hcc_event_t = np.array(hcc_event_steps, dtype=float)

    # Normalize proliferation by cell count, recruitment by vasculature count
    pdac_tc_prolif_norm = [pdac_events['tc_prolif'][i] / max(pdac_counts['tcell'][i], 1) if i < len(pdac_counts['tcell']) else 0
                           for i in range(len(pdac_events['tc_prolif']))]
    pdac_th_prolif_norm = [pdac_events['th_prolif'][i] / max(pdac_counts['tcell'][i] + pdac_counts['treg'][i], 1) if i < len(pdac_counts['tcell']) else 0
                           for i in range(len(pdac_events['th_prolif']))]
    pdac_treg_prolif_norm = [pdac_events['treg_prolif'][i] / max(pdac_counts['treg'][i], 1) if i < len(pdac_counts['treg']) else 0
                             for i in range(len(pdac_events['treg_prolif']))]
    pdac_recruit_norm = [pdac_events['tc_recruit'][i] / max(pdac_counts['vas_phalanx'][i] + pdac_counts['vas_tip'][i], 1) if i < len(pdac_counts['vas_phalanx']) else 0
                         for i in range(len(pdac_events['tc_recruit']))]
    pdac_th_recruit_norm = [pdac_events['th_recruit'][i] / max(pdac_counts['vas_phalanx'][i] + pdac_counts['vas_tip'][i], 1) if i < len(pdac_counts['vas_phalanx']) else 0
                                for i in range(len(pdac_events['th_recruit']))]

    hcc_tc_prolif_norm = [hcc_events['tc_prolif'][i] / max(hcc_counts['tcell'][i], 1) if i < len(hcc_counts['tcell']) else 0
                          for i in range(len(hcc_events['tc_prolif']))]
    hcc_th_prolif_norm = [hcc_events['th_prolif'][i] / max(hcc_counts['tcell'][i], 1) if i < len(hcc_counts['tcell']) else 0
                          for i in range(len(hcc_events['th_prolif']))]
    hcc_treg_prolif_norm = [hcc_events['treg_prolif'][i] / max(hcc_counts['treg'][i], 1) if i < len(hcc_counts['treg']) else 0
                            for i in range(len(hcc_events['treg_prolif']))]
    hcc_recruit_norm = [hcc_events['tc_recruit'][i] / max(hcc_counts['vas_phalanx'][i] + hcc_counts['vas_tip'][i], 1) if i < len(hcc_counts['vas_phalanx']) else 0
                        for i in range(len(hcc_events['tc_recruit']))]
    hcc_th_recruit_norm = [hcc_events['th_recruit'][i] / max(hcc_counts['vas_phalanx'][i] + hcc_counts['vas_tip'][i], 1) if i < len(hcc_counts['vas_phalanx']) else 0
                               for i in range(len(hcc_events['th_recruit']))]
    
    # T Cell proliferation (per T cell)
    ax = axes5[0, 0]
    if pdac_tc_prolif_norm:
        ax.plot(pdac_event_t, pdac_tc_prolif_norm, color=PDAC_COLOR, lw=2.0, label='PDAC')
    if hcc_tc_prolif_norm:
        ax.plot(hcc_event_t, hcc_tc_prolif_norm, color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("T Cell Prolif / T Cell", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate per cell")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # T Cell recruitment (per vasculature cell)
    ax = axes5[0, 1]
    if pdac_recruit_norm:
        ax.plot(pdac_event_t, pdac_recruit_norm, color=PDAC_COLOR, lw=2.0, label='PDAC')
    if hcc_recruit_norm:
        ax.plot(hcc_event_t, hcc_recruit_norm, color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("T Recruit / Vasculature", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate per vascular cell")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # TH proliferation (per TH+T cell)
    ax = axes5[0, 2]
    if pdac_th_prolif_norm:
        ax.plot(pdac_event_t, pdac_th_prolif_norm, color=PDAC_COLOR, lw=2.0, label='PDAC')
    if hcc_th_prolif_norm:
        ax.plot(hcc_event_t, hcc_th_prolif_norm, color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("TH Prolif / (T+TReg)", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate per cell")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # TH recruitment (per vasculature)
    ax = axes5[1, 0]
    if pdac_events['th_recruit']:
        ax.plot(pdac_event_t, pdac_th_recruit_norm, color=PDAC_COLOR, lw=2.0, label='PDAC')
    if hcc_events['th_recruit']:
        ax.plot(hcc_event_t, hcc_th_recruit_norm, color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("TH Recruit / Vasculature", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate per vascular cell")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # TReg proliferation (per TReg)
    ax = axes5[1, 1]
    if pdac_treg_prolif_norm:
        ax.plot(pdac_event_t, pdac_treg_prolif_norm, color=PDAC_COLOR, lw=2.0, label='PDAC')
    if hcc_treg_prolif_norm:
        ax.plot(hcc_event_t, hcc_treg_prolif_norm, color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("TReg Prolif / TReg", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate per cell")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Empty space (for layout balance)
    axes5[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "event_comparison_normalized.png"), dpi=120)
    print(f"Saved: {OUT_DIR}/event_comparison_normalized.png")

# --- PDAC and HCC event comparison with normalization ---
if pdac_event_steps and hcc_event_steps:
    fig5, axes5 = plt.subplots(2, 3, figsize=(16, 10))
    fig5.suptitle("T Cell and TReg Events (Normalized per cell)  ■ HCC (blue)  ■ PDAC GPU (orange)", fontsize=13)

    pdac_event_t = np.array(pdac_event_steps, dtype=float)
    hcc_event_t = np.array(hcc_event_steps, dtype=float)

    # Normalize proliferation by cell count, recruitment by vasculature count
    pdac_tc_prolif_norm = [pdac_events['tc_prolif'][i] / max(pdac_counts['tcell'][i], 1) if i < len(pdac_counts['tcell']) else 0
                           for i in range(len(pdac_events['tc_prolif']))]
    pdac_th_prolif_norm = [pdac_events['th_prolif'][i] / max(pdac_counts['tcell'][i] + pdac_counts['treg'][i], 1) if i < len(pdac_counts['tcell']) else 0
                           for i in range(len(pdac_events['th_prolif']))]
    pdac_treg_prolif_norm = [pdac_events['treg_prolif'][i] / max(pdac_counts['treg'][i], 1) if i < len(pdac_counts['treg']) else 0
                             for i in range(len(pdac_events['treg_prolif']))]
    pdac_recruit_norm = [pdac_events['tc_recruit'][i] / max(pdac_counts['vas_phalanx'][i] + pdac_counts['vas_tip'][i], 1) if i < len(pdac_counts['vas_phalanx']) else 0
                         for i in range(len(pdac_events['tc_recruit']))]
    pdac_th_recruit_norm = [pdac_events['th_recruit'][i] / max(pdac_counts['vas_phalanx'][i] + pdac_counts['vas_tip'][i], 1) if i < len(pdac_counts['vas_phalanx']) else 0
                                for i in range(len(pdac_events['th_recruit']))]

    hcc_tc_prolif_norm = [hcc_events['tc_prolif'][i] / max(hcc_counts['tcell'][i], 1) if i < len(hcc_counts['tcell']) else 0
                          for i in range(len(hcc_events['tc_prolif']))]
    hcc_th_prolif_norm = [hcc_events['th_prolif'][i] / max(hcc_counts['tcell'][i], 1) if i < len(hcc_counts['tcell']) else 0
                          for i in range(len(hcc_events['th_prolif']))]
    hcc_treg_prolif_norm = [hcc_events['treg_prolif'][i] / max(hcc_counts['treg'][i], 1) if i < len(hcc_counts['treg']) else 0
                            for i in range(len(hcc_events['treg_prolif']))]
    hcc_recruit_norm = [hcc_events['tc_recruit'][i] / max(hcc_counts['vas_phalanx'][i] + hcc_counts['vas_tip'][i], 1) if i < len(hcc_counts['vas_phalanx']) else 0
                        for i in range(len(hcc_events['tc_recruit']))]
    hcc_th_recruit_norm = [hcc_events['th_recruit'][i] / max(hcc_counts['vas_phalanx'][i] + hcc_counts['vas_tip'][i], 1) if i < len(hcc_counts['vas_phalanx']) else 0
                               for i in range(len(hcc_events['th_recruit']))]
    
    # T Cell proliferation (per T cell)
    ax = axes5[0, 0]
    ax.plot(pdac_event_t, pdac_events['tc_prolif'], color=PDAC_COLOR, lw=2.0, label='PDAC')
    ax.plot(hcc_event_t, hcc_events['tc_prolif'], color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("T Cell Prolif", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # T Cell recruitment (per vasculature cell)
    ax = axes5[0, 1]
    ax.plot(pdac_event_t, pdac_events['tc_recruit'], color=PDAC_COLOR, lw=2.0, label='PDAC')
    ax.plot(hcc_event_t, hcc_events['tc_recruit'], color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("T Recruit", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # TH proliferation (per TH+T cell)
    ax = axes5[0, 2]
    ax.plot(pdac_event_t, pdac_events['th_prolif'], color=PDAC_COLOR, lw=2.0, label='PDAC')
    ax.plot(hcc_event_t, hcc_events['th_prolif'], color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("TH Prolif", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # TH recruitment (per vasculature)
    ax = axes5[1, 0]
    ax.plot(pdac_event_t, pdac_events['th_recruit'], color=PDAC_COLOR, lw=2.0, label='PDAC')
    ax.plot(hcc_event_t, hcc_events['th_recruit'], color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("TH Recruit", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # TReg proliferation (per TReg)
    ax = axes5[1, 1]
    ax.plot(pdac_event_t, pdac_events['treg_prolif'], color=PDAC_COLOR, lw=2.0, label='PDAC')
    ax.plot(hcc_event_t, hcc_events['treg_prolif'], color=HCC_COLOR, lw=2.0, label='HCC')
    ax.set_title("TReg Prolif", fontsize=11)
    ax.set_xlabel("Step")
    ax.set_ylabel("Rate")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Empty space (for layout balance)
    axes5[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "event_comparison.png"), dpi=120)
    print(f"Saved: {OUT_DIR}/event_comparison.png")

# ─── QSP V_T comparison ───────────────────────────────────────────────────────
def parse_qsp_all(path):
    """Read a QSP CSV and return (steps_list, {species: values_list}) for all V_*.* species."""
    if not os.path.exists(path):
        print(f"WARNING: QSP file not found: {path}")
        return [], {}
    steps = []
    series = {}
    with open(path) as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        # First column is 'step' (PDAC) or 'time' (HCC); all V_.* columns are data
        data_cols = {i: h for i, h in enumerate(header) if '.' in h and i > 0}
        for col in data_cols.values():
            series[col] = []
        for row in reader:
            if not row or not row[0].strip():
                continue
            try:
                steps.append(int(float(row[0])))
            except ValueError:
                continue
            for i, col in data_cols.items():
                try:
                    series[col].append(float(row[i]) if i < len(row) and row[i].strip() else float('nan'))
                except ValueError:
                    series[col].append(float('nan'))
    return steps, series

print("Parsing PDAC QSP...", flush=True)
pdac_qsp_steps, pdac_qsp = parse_qsp_all(PDAC_QSP_CSV)

print("Parsing HCC QSP...", flush=True)
hcc_qsp_files = sorted(glob.glob(os.path.join(HCC_STATS_DIR, "QSP_*.csv")))
hcc_qsp_steps, hcc_qsp = [], {}
if hcc_qsp_files:
    hcc_qsp_steps, hcc_qsp = parse_qsp_all(hcc_qsp_files[-1])
else:
    print("WARNING: No HCC QSP_*.csv found in", HCC_STATS_DIR)

def plot_qsp_compartment(prefix, pdac_qsp, pdac_qsp_t, hcc_qsp, hcc_qsp_t, out_path, n_cols=7):
    """Plot all QSP species with the given prefix as an N×n_cols grid."""
    available = set(pdac_qsp.keys()) | set(hcc_qsp.keys())
    species = sorted(s for s in available if s.startswith(prefix + '.'))
    if not species:
        print(f"WARNING: no species found for prefix {prefix}")
        return
    n_rows = (len(species) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.0))
    fig.suptitle(f"QSP {prefix} Compartment  ■ HCC (blue)  ■ PDAC GPU (orange)", fontsize=13)
    for idx, ax in enumerate(axes.flat):
        if idx >= len(species):
            ax.axis('off')
            continue
        sp = species[idx]
        short = sp.replace(prefix + '.', '')
        if sp in hcc_qsp and hcc_qsp_t.size > 0:
            ax.plot(hcc_qsp_t,  np.array(hcc_qsp[sp]),  color=HCC_COLOR,  lw=1.5, label='HCC')
        if sp in pdac_qsp and pdac_qsp_t.size > 0:
            ax.plot(pdac_qsp_t, np.array(pdac_qsp[sp]), color=PDAC_COLOR, lw=1.5, label='PDAC')
        ax.set_title(short, fontsize=9)
        ax.set_xlabel("Step", fontsize=7)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")

pdac_qsp_t = np.array(pdac_qsp_steps, dtype=float)
hcc_qsp_t  = np.array(hcc_qsp_steps,  dtype=float)

plot_qsp_compartment('V_T', pdac_qsp, pdac_qsp_t, hcc_qsp, hcc_qsp_t,
                     os.path.join(OUT_DIR, "qsp_vt_comparison.png"))
plot_qsp_compartment('V_C', pdac_qsp, pdac_qsp_t, hcc_qsp, hcc_qsp_t,
                     os.path.join(OUT_DIR, "qsp_vc_comparison.png"))

plt.close('all')
