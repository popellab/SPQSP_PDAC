import csv, numpy as np

def load_events(path):
    data = {}
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            step = int(row['Step'])
            for k,v in row.items():
                if k != 'Step':
                    data.setdefault(k, {})[step] = int(v)
    return data

hcc  = load_events('../PDAC/sim/HCC_outputs2/event.csv')
pdac = load_events('../PDAC/sim/outputs/event.csv')

print('Cumulative event totals over 280 steps:')
print(f'  {"Event":<35} {"HCC":>8}  {"PDAC":>8}  {"ratio":>7}')
print('  ' + '-'*65)
all_cols = sorted(set(hcc.keys()) | set(pdac.keys()))
for col in all_cols:
    ht = sum(hcc.get(col, {}).values())
    pt = sum(pdac.get(col, {}).values())
    ratio = f'{pt/ht:.2f}x' if ht > 0 else 'N/A'
    print(f'  {col:<35} {ht:>8}  {pt:>8}  {ratio:>7}')

# Per-step recruit.CD8.effector over first 50 steps
print('\nrecruit.CD8.effector per-step (first 50 steps):')
print(f'  {"Step":>5}  {"HCC":>6}  {"PDAC":>6}')
for s in range(50):
    h = hcc.get('recruit.CD8.effector', {}).get(s, 0)
    p = pdac.get('recruit.CD8.effector', {}).get(s, 0)
    print(f'  {s:>5}  {h:>6}  {p:>6}')
