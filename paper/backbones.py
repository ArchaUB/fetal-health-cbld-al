"""Backbone comparison (XGBoost, LightGBM, CatBoost, random forest) under the same AL protocol.
Writes paper/tables/backbones.tex and prints paired tests."""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = f'{ROOT}/results'; T = f'{ROOT}/paper/tables'; os.makedirs(T, exist_ok=True)
D = [('ctg3', 'CTG-3'), ('ctg10', 'CTG-10')]
BB = [('xgb', 'XGBoost'), ('lgbm', 'LightGBM'), ('cat', 'CatBoost'), ('rf', 'Random forest')]
M = [('paper', 'Base method'), ('C1C2', 'CBLD-AL (C1+C2)'), ('proposed', 'CBLD-AL (C1+C2+C3)'), ('full', 'Supervised, full pool')]
def L(d, b, m): return pd.read_pickle(f"{RES}/{d}__{'' if b == 'xgb' else b + '__'}{m}.pkl")
def v(d, b, m, k): return np.array([r[k] for r in L(d, b, m)])
def star(p, diff=1.0):
    up = diff > 0
    if p < 0.01: return '$^{\\ast\\ast}$' if up else '$^{\\dagger\\dagger}$'
    if p < 0.05: return '$^{\\ast}$' if up else '$^{\\dagger}$'
    return ''
lines = []
for b, bn in BB:
    for i, (m, mn) in enumerate(M):
        cells = []
        for d, _ in D:
            for k in ('acc', 'balAcc'):
                x = v(d, b, m, k); s = f'{x.mean():.2f}'
                if m in ('C1C2', 'proposed'):
                    y = v(d, b, 'paper', k); d_ = x - y
                    s += star(wilcoxon(d_).pvalue if np.any(d_) else 1, d_.mean())
                cells.append(s)
            cells.append(f"{v(d, b, m, 'labels').mean():.0f}")
        lines.append((f'\\multirow{{4}}{{*}}{{{bn}}}' if i == 0 else '') + f' & {mn} & ' + ' & '.join(cells) + '\\\\')
    lines.append('\\midrule')
open(f'{T}/backbones.tex', 'w').write('\n'.join(lines[:-1]))
for d, dn in D:
    print('==', dn)
    for b, bn in BB:
        out = []
        for m in ('paper', 'C1C2', 'proposed', 'full'):
            out.append(f"{m}: acc {v(d,b,m,'acc').mean():.2f} bal {v(d,b,m,'balAcc').mean():.2f} F1 {v(d,b,m,'macroF1').mean():.2f} lab {v(d,b,m,'labels').mean():.0f}")
        dd = v(d, b, 'C1C2', 'balAcc') - v(d, b, 'paper', 'balAcc'); da = v(d, b, 'C1C2', 'acc') - v(d, b, 'paper', 'acc')
        print(f'  {bn:14s}', ' | '.join(out), f'|| C1C2-base bal {dd.mean():+.2f} p={wilcoxon(dd).pvalue:.3g} acc {da.mean():+.2f} p={wilcoxon(da).pvalue:.3g}')
    # XGBoost vs others, CBLD-AL
    for b, bn in BB[1:]:
        for k in ('acc', 'balAcc'):
            dd = v(d, 'xgb', 'C1C2', k) - v(d, b, 'C1C2', k)
            print(f'  XGB-{bn} (C1C2) {k}: {dd.mean():+.2f} p={wilcoxon(dd).pvalue:.3g}')
