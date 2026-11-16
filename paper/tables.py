import numpy as np, pandas as pd, json
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root

from scipy.stats import wilcoxon
RES=f'{ROOT}/results'; T=f'{ROOT}/paper/tables'; os.makedirs(T, exist_ok=True)
D=[('ctg3','CTG-3'),('ctg10','CTG-10')]
def L(d,m): return pd.read_pickle(f'{RES}/{d}__{m}.pkl')
def v(d,m,k): return np.array([r[k] for r in L(d,m)])
def minrec(d,m): return np.array([r['recall'].min() for r in L(d,m)])
def pv(a,b):
    x=a-b; return wilcoxon(x).pvalue if np.any(x) else 1.0
def star(p): return '$^{\\ast\\ast}$' if p<0.01 else ('$^{\\ast}$' if p<0.05 else '')
NUM={}
# ---------------- main table
ALM=[('random','Random sampling'),('entropy','Entropy'),('margin','Margin'),('qbc','QBC'),('coreset','Core-set'),
     ('margin_kmeans','Margin + $k$-means'),('paper','Bhardwaj et al.~\\cite{bhardwaj2025}'),('C1C2','Proposed (C1+C2)'),('proposed','Proposed (C1+C2+C3)')]
REF=[('full','Supervised, full pool'),('full_bal','Supervised, full pool, balanced')]
lines=[]
for d,dn in D:
    lines.append(f'\\multicolumn{{6}}{{l}}{{\\textbf{{{dn}}}}}\\\\')
    rows={m:dict(acc=v(d,m,'acc'),f1=v(d,m,'macroF1'),bal=v(d,m,'balAcc'),mr=minrec(d,m),lab=v(d,m,'labels')) for m,_ in ALM+REF}
    best={k:max(rows[m][k].mean() for m,_ in ALM) for k in ('acc','f1','bal','mr')}
    for m,name in ALM+REF:
        if m=='full': lines.append('\\cmidrule(lr){1-6}')
        r=rows[m]; cells=[]
        for k in ('acc','f1','bal','mr'):
            s=f"{r[k].mean():.2f}$\\pm${r[k].std():.2f}"
            if m not in ('full','full_bal') and abs(r[k].mean()-best[k])<0.005: s='\\textbf{'+s+'}'
            cells.append(s)
        lines.append(f"{name} & "+' & '.join(cells)+f" & {r['lab'].mean():.0f}\\\\")
        NUM[f'{d}_{m}']={k:round(float(r[k].mean()),2) for k in r}
    lines.append('\\midrule')
lines=lines[:-1]
open(f'{T}/main.tex','w').write('\n'.join(lines))
# ---------------- ablation (paired diffs vs base method)
AB=[('C1','C1: leaf-embedding diversity'),('C2','C2: class-balanced final model'),('C1C2','C1 + C2'),('proposed','C1 + C2 + C3 (stopping rule)')]
lines=[]
for d,dn in D:
    lines.append(f'\\multicolumn{{5}}{{l}}{{\\textbf{{{dn}}}}}\\\\')
    for m,name in AB:
        cells=[]
        for k in ('acc','macroF1','balAcc'):
            a,b=v(d,m,k),v(d,'paper',k); p=pv(a,b); cells.append(f"{(a-b).mean():+.2f}{star(p)}")
            NUM[f'abl_{d}_{m}_{k}']=[round(float((a-b).mean()),2),float(p)]
        lines.append(f"{name} & "+' & '.join(cells)+f" & {v(d,m,'labels').mean():.0f}\\\\")
    lines.append('\\midrule')
open(f'{T}/ablation.tex','w').write('\n'.join(lines[:-1]))
# ---------------- significance of C1+C2 vs every baseline on balanced accuracy
BL=[m for m,_ in ALM if m not in ('C1C2','proposed')]
names=dict(ALM)
lines=[]
for m in BL:
    cells=[]
    for d,_ in D:
        a,b=v(d,'C1C2','balAcc'),v(d,m,'balAcc'); p=pv(a,b); dd=(a-b).mean()
        cells.append(f"{dd:+.2f} ({'$<$0.001' if p<0.001 else f'{p:.3f}'})")
    lines.append(names[m].replace(' et al.~\\cite{bhardwaj2025}',' et al.')+' & '+' & '.join(cells)+'\\\\')
open(f'{T}/signif.tex','w').write('\n'.join(lines))
# ---------------- per-class CTG-3 precision/recall/F1 (pooled confusion matrices)
lines=[]; cls=['Normal','Suspect','Pathological']
for m,name in (('paper','Bhardwaj et al.~\\cite{bhardwaj2025}'),('C1C2','Proposed (C1+C2)'),('proposed','Proposed (C1+C2+C3)')):
    cm=np.sum([r['cm'] for r in L('ctg3',m)],0); P=np.diag(cm)/cm.sum(0); R=np.diag(cm)/cm.sum(1); F=2*P*R/(P+R)
    for i,c in enumerate(cls):
        lines.append((f'\\multirow{{3}}{{*}}{{{name}}}' if i==0 else '')+f' & {c} & {P[i]*100:.2f} & {R[i]*100:.2f} & {F[i]*100:.2f}\\\\')
    lines.append('\\midrule')
open(f'{T}/perclass.tex','w').write('\n'.join(lines[:-1]))
# ---------------- stopping rule summary
for d,_ in D:
    lab=v(d,'proposed','labels'); NUM[f'stop_{d}']=dict(mean=float(lab.mean()),min=float(lab.min()),max=float(lab.max()),
        saved=float((1-lab.mean()/420)*100))
    for k in ('acc','macroF1','balAcc'):
        a,b=v(d,'proposed',k),v(d,'C1C2',k); NUM[f'stopvs420_{d}_{k}']=[round(float((a-b).mean()),2),float(pv(a,b))]
# recall per class
for d,_ in D:
    for m in ('paper','C1C2','proposed','full'):
        NUM[f'rec_{d}_{m}']=np.mean([r['recall'] for r in L(d,m)],0).round(2).tolist()
json.dump(NUM,open(f'{ROOT}/paper/numbers.json','w'),indent=1)
print('tables ok')
