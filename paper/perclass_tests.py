"""Per-run per-class precision / recall / F1 of CBLD-AL (C1+C2) vs the base method, with paired Wilcoxon tests."""
import os, numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def per(d, m):
    out = []
    for r in pd.read_pickle(f'{ROOT}/results/{d}__{m}.pkl'):
        cm = r['cm'].astype(float); tp = np.diag(cm)
        P = tp / np.maximum(cm.sum(0), 1); R = tp / cm.sum(1); F = np.where(P + R > 0, 2 * P * R / (P + R + 1e-12), 0)
        out.append(np.stack([P, R, F]))
    return np.array(out) * 100          # runs x {precision, recall, F1} x classes
B, C = per('ctg3', 'paper'), per('ctg3', 'C1C2')
for ci, c in enumerate(['Normal', 'Suspect', 'Pathological']):
    print(c, ' | '.join(f"{mn} {B[:, mi, ci].mean():.2f} -> {C[:, mi, ci].mean():.2f} "
                        f"({(C[:, mi, ci] - B[:, mi, ci]).mean():+.2f}, p={wilcoxon(C[:, mi, ci] - B[:, mi, ci]).pvalue:.3g})"
                        for mi, mn in enumerate(['precision', 'recall', 'F1'])))
B, C = per('ctg10', 'paper'), per('ctg10', 'C1C2')
for mi, mn in enumerate(['macro precision', 'macro recall', 'macro F1']):
    x = C[:, mi, :].mean(1) - B[:, mi, :].mean(1)
    print('CTG-10', mn, f'{B[:, mi, :].mean():.2f} -> {C[:, mi, :].mean():.2f} ({x.mean():+.2f}, p={wilcoxon(x).pvalue:.3g})')
