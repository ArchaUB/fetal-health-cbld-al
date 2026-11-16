import os, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from scipy.stats import wilcoxon
D = ['ctg3','ctg10']; M = ['random','entropy','margin','qbc','coreset','margin_kmeans','paper','C1','C2','C1C2','proposed','full','full_bal']
R = {(d,m): pd.read_pickle(f'{ROOT}/results/{d}__{m}.pkl') for d in D for m in M}
def v(d,m,k): return np.array([r[k] for r in R[(d,m)]])
def p(a,b):
    d=a-b; return wilcoxon(d).pvalue if np.any(d) else 1.0
for d in D:
    print(f'\n######## {d}')
    rows=[]
    for m in M:
        rec = np.array([r['recall'] for r in R[(d,m)]]).mean(0)
        rows.append(dict(method=m, acc=v(d,m,'acc').mean(), F1=v(d,m,'macroF1').mean(), bal=v(d,m,'balAcc').mean(),
                         minrec=rec.min(), labels=v(d,m,'labels').mean()))
    print(pd.DataFrame(rows).set_index('method').round(2).to_string())
    print('-- paired vs paper (diff, p)')
    for m in ['C1','C2','C1C2','proposed','qbc','margin','entropy','margin_kmeans']:
        print(f"  {m:14s}", "  ".join(f"{k} {(v(d,m,k)-v(d,'paper',k)).mean():+.2f} p={p(v(d,m,k),v(d,'paper',k)):.3f}" for k in ['acc','macroF1','balAcc']))
    # learning curve: C1 vs paper
    S = R[(d,'paper')][0]['sizes']
    for name,key in (('acc','curve'),('bal','curve_bal')):
        B = np.array([r[key] for r in R[(d,'paper')]]); L = np.array([r[key] for r in R[(d,'C1')]])
        s = "  ".join(f"@{n}:{(L[:,S.index(n)]-B[:,S.index(n)]).mean():+.2f}(p={wilcoxon(L[:,S.index(n)]-B[:,S.index(n)]).pvalue:.2f})" for n in (100,150,200,250,300))
        print(f'  C1 vs paper curve {name}: ALC {L.mean()-B.mean():+.2f} (p={wilcoxon(L.mean(1)-B.mean(1)).pvalue:.3f}) | {s}')
