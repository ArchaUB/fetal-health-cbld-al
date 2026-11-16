"""Replays the authors' notebook loop and records, per iteration, the notebook's (leaky) accuracy,
the clean held-out accuracy, the test-set size and the share of test rows that are training samples."""
import warnings, os; warnings.filterwarnings('ignore')
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root

import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from scipy.spatial import distance
from joblib import Parallel, delayed
df = pd.read_csv(f'{ROOT}/data/ctg3.csv'); df['fetal_health'] = LabelEncoder().fit_transform(df['fetal_health'])
x = df.drop(columns=['fetal_health']); y = df['fetal_health']
x_init, x_left, y_init, y_left = train_test_split(x, y, train_size=0.01415, random_state=42)
def one(seed):
    x_tr, y_tr = x_init, y_init
    unl, x_te, lab, y_te = train_test_split(x_left, y_left, test_size=0.5, random_state=seed)
    cx, cy = x_te.copy(), y_te.copy(); rows = []
    for i in range(78):
        m = XGBClassifier(objective='multi:softprob', num_class=3, colsample_bytree=0.5, learning_rate=0.05,
                          max_depth=5, n_estimators=200, n_jobs=1).fit(x_tr, y_tr)
        p = m.predict_proba(unl)
        x_te = pd.concat([x_te, unl]); y_te = pd.concat([y_te, lab])
        leaky = accuracy_score(y_te, m.predict_proba(x_te).argmax(1)) * 100
        clean = accuracy_score(cy, m.predict_proba(cx).argmax(1)) * 100
        rows.append(dict(seed=seed, it=i, n_train=len(x_tr), leaky=leaky, clean=clean, test_rows=len(x_te),
                         leak_pct=x_te.index.isin(x_tr.index).mean() * 100))
        s = np.sort(p, 1); top = np.argsort(s[:, -1] - s[:, -2])[:20]
        sel = [top[0]]; rem = top[1:]
        while len(sel) < 5:
            d = [sum(distance.euclidean(np.ravel(a), np.ravel(r)) for a in sel) for r in rem]
            j = int(np.argmax(d)); sel.append(rem[j]); rem = np.delete(rem, j)
        x_tr = pd.concat([x_tr, unl.iloc[sel]]); y_tr = pd.concat([y_tr, lab.iloc[sel]])
        unl = unl.drop(unl.index[sel]); lab = lab.drop(lab.index[sel])
    m2 = XGBClassifier(objective='multi:softprob', num_class=3, random_state=42, n_jobs=1).fit(x_tr, y_tr)
    x_te = pd.concat([x_te, unl]); y_te = pd.concat([y_te, lab])
    rows.append(dict(seed=seed, it=78, n_train=len(x_tr), leaky=accuracy_score(y_te, m2.predict_proba(x_te).argmax(1))*100,
                     clean=accuracy_score(cy, m2.predict_proba(cx).argmax(1))*100, test_rows=len(x_te),
                     leak_pct=x_te.index.isin(x_tr.index).mean()*100))
    return rows
R = Parallel(n_jobs=-1)(delayed(one)(s) for s in range(10))
out = pd.DataFrame([r for rr in R for r in rr]); out.to_csv(f'{ROOT}/results/leak_trace.csv', index=False)
print(out[out.it == 78][['leaky', 'clean', 'test_rows', 'leak_pct']].agg(['mean', 'std']).round(2))
