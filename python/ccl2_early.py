import csv, numpy as np, glob

def parse_hcc_step(path):
    stem=prog=sen=0
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if len(row) < 5: continue
            if int(row[3]) != 1: continue
            s = int(row[4])
            if s == 6: stem += 1
            elif s == 7: prog += 1
            elif s == 8: sen += 1
    return stem, prog, sen

def parse_pdac_step(path):
    stem=prog=sen=0
    with open(path) as f:
        r = csv.reader(f)
        next(r)
        for row in r:
            if row[0].strip() != 'CANCER': continue
            s = row[5].strip()
            if s == 'STEM': stem += 1
            elif s == 'PROGENITOR': prog += 1
            elif s == 'SENESCENT': sen += 1
    return stem, prog, sen

def mean_ccl2_hcc(step):
    p = f'../PDAC/sim/HCC_outputs2/snapShots/grid_core_{step}.csv'
    files = glob.glob(p)
    if not files: return float('nan')
    vals = [float(r.split(',')[5]) for r in open(files[0]).readlines()[1:] if r.strip()]
    return np.mean(vals)

def mean_ccl2_pdac(path):
    h = open(path).readline().strip().split(',')
    ci = h.index('CCL2')
    vals = [float(r.split(',')[ci]) for r in open(path).readlines()[1:] if r.strip()]
    return np.mean(vals)

hcc_cells  = sorted(glob.glob('../PDAC/sim/HCC_outputs2/snapShots/cell_*.csv'),
                    key=lambda x: int(x.split('cell_')[-1].replace('.csv','')))[:16]
pdac_cells = sorted(glob.glob('../PDAC/sim/outputs/abm/agents_step_*.csv'),
                    key=lambda x: int(x.split('_')[-1].replace('.csv','')))[:16]

print('Step  HCC stem/prog/sen  HCC_secr  HCCL2   | PDAC stem/prog/sen  PDAC_secr  PCCL2   CCL2/secCell ratio')
for i in range(min(15, len(hcc_cells), len(pdac_cells))):
    hs = int(hcc_cells[i].split('cell_')[-1].replace('.csv',''))
    hs2, hp2, hsen = parse_hcc_step(hcc_cells[i])
    hccl2 = mean_ccl2_hcc(hs)
    h_sec = hs2 + hp2

    ps = int(pdac_cells[i].split('_')[-1].replace('.csv',''))
    ps2, pp2, psen = parse_pdac_step(pdac_cells[i])
    pfile = pdac_cells[i].replace('abm/agents_step_', 'pde/pde_step_')
    pfiles = glob.glob(pfile)
    pccl2 = mean_ccl2_pdac(pfiles[0]) if pfiles else float('nan')
    p_sec = ps2 + pp2

    if h_sec > 0 and p_sec > 0 and hccl2 > 0 and not np.isnan(pccl2):
        ratio = (pccl2 / p_sec) / (hccl2 / h_sec)
    else:
        ratio = float('nan')

    print(f'{hs:3d}/{ps:<3d}  {hs2:4d}/{hp2:6d}/{hsen:6d}  {h_sec:8d}  {hccl2:.4f}  |'
          f'  {ps2:4d}/{pp2:6d}/{psen:6d}  {p_sec:9d}  {pccl2:.4f}  {ratio:.2f}x')
