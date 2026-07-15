#!/usr/bin/env python
"""
Cohort extraction: for EVERY PASS-01 subject, pull the 1ST-LINE tumor-burden
trajectory (SLD + CA19-9), truncated at end-of-1st-line-treatment. The
`InstanceName` column encodes line-of-therapy; we cut at the '1st Line - End of
Treatment' date (fallback: first 2nd-line instance). Writes per-patient CSVs +
a cohort manifest. GnP arm only (current NPE bank is GnP-template); mFFX flagged
for a future mFFX bank.

Filters (usable): GnP arm, CA19-9 secretor, >=3 SLD AND >=3 CA19-9 within 1st
line, window W >= 100 d.
"""
import pandas as pd, numpy as np, json
from pathlib import Path

XL   = Path("/home/chase/SPQSP/SPQSP_PDAC-main/data/PASS01/metadata/ClinicalTable_filtered_DeID.xlsx")
OUT  = Path("/home/chase/SPQSP/SPQSP_PDAC-main/results/burden_sbi_7-15-26")
PDIR = OUT/"cohort"; PDIR.mkdir(exist_ok=True)
MIN_SLD, MIN_CA, MIN_W = 3, 3, 100.0

xl = pd.ExcelFile(XL)
tl = xl.parse('Target Lesions'); ca = xl.parse('All CA 19.9')
ev = xl.parse('135 events').rename(columns={'PFS Eligible':'Subject','1st Line Treatment':'arm'})
arm = dict(zip(ev.Subject, ev.arm))

def line_of(inst):
    s=str(inst)
    if '2nd Line' in s: return 2
    if '3rd Line' in s: return 3
    if '4th Line' in s: return 4
    return 1

def firstline_end(subj, base):
    d=tl[tl.Subject==subj].copy(); d['dt']=pd.to_datetime(d['LES_TAR_TEST_DT'])
    inst=d.groupby('InstanceName')['dt'].first()
    end=[(v-base).days for k,v in inst.items() if '1st Line - End' in str(k)]
    second=[(v-base).days for k,v in inst.items() if '2nd Line' in str(k)]
    if end: return end[0]
    if second: return min(second)
    return (d['dt'].max()-base).days

rows=[]
subjects = sorted(set(tl.Subject) & set(ca.Subject))
for subj in subjects:
    a = str(arm.get(subj,''))
    # base date = first target-lesion scan
    d=tl[tl.Subject==subj].copy(); d['sum']=pd.to_numeric(d['LES_TAR_SUM'],errors='coerce')
    d=d.dropna(subset=['LES_TAR_TEST_DT','sum']);
    if d.empty: continue
    d['dt']=pd.to_datetime(d['LES_TAR_TEST_DT']); base=d['dt'].min()
    W=firstline_end(subj, base)
    g=d.groupby('dt')['sum'].first().sort_index()
    s=pd.DataFrame({'day':(g.index-base).days,'SLD':g.values.astype(int)}); s=s[(s.day>=0)&(s.day<=W)]
    # CA19-9
    c=ca[ca.Subject==subj].copy()
    nonsec = c['Harry Harvey Comments'].astype(str).str.contains('Non-Secretor',case=False,na=False).any()
    c['val']=pd.to_numeric(c['CA199_VAL'],errors='coerce'); c=c.dropna(subset=['CA199_DATE','val'])
    if c.empty: continue
    c['dt']=pd.to_datetime(c['CA199_DATE'])
    cc=pd.DataFrame({'day':(c['dt']-base).dt.days.values,'CA199':c['val'].values}).groupby('day',as_index=False)['CA199'].first()
    cc=cc[(cc.day>=0)&(cc.day<=W)].sort_values('day')
    is_gnp = 'GnP' in a
    usable = is_gnp and (not nonsec) and len(s)>=MIN_SLD and len(cc)>=MIN_CA and W>=MIN_W
    row=dict(subject=subj, arm=('GnP' if is_gnp else ('mFFX' if 'mFFX' in a else a)),
             W=int(W), nSLD=len(s), nCA=len(cc), secretor=(not nonsec), usable=bool(usable))
    if usable:
        row['SLD_fin']=round(s.SLD.iloc[-1]/s.SLD.iloc[0],3)
        row['CA_nadir']=round(cc.CA199.min()/cc.CA199.iloc[0],3)
        row['CA_fin']=round(cc.CA199.iloc[-1]/cc.CA199.iloc[0],3)
        tag=subj.replace('100-','').replace('-','')
        row['tag']=tag
        s.to_csv(PDIR/f"{tag}_SLD.csv",index=False); cc.to_csv(PDIR/f"{tag}_CA.csv",index=False)
    rows.append(row)

man=pd.DataFrame(rows)
man.to_csv(OUT/"cohort_manifest.csv",index=False)
print(f"subjects scanned: {len(subjects)}")
print(f"  GnP arm: {(man.arm=='GnP').sum()}  | mFFX: {(man.arm=='mFFX').sum()}")
print(f"  usable (GnP, secretor, >={MIN_SLD}SLD & >={MIN_CA}CA in 1st-line, W>={MIN_W:.0f}): {man.usable.sum()}")
print(f"  excluded GnP: nonsecretor={((man.arm=='GnP')&~man.secretor).sum()}, "
      f"too-few-pts={((man.arm=='GnP')&man.secretor&~man.usable).sum()}")
print(f"manifest -> cohort_manifest.csv ; per-patient CSVs -> cohort/")
print("\nusable cohort summary (response profile):")
u=man[man.usable].sort_values('CA_fin')
print(u[['tag','W','nSLD','nCA','SLD_fin','CA_nadir','CA_fin']].to_string(index=False))
