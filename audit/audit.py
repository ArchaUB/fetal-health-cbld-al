"""Audit of the original notebook: runs its loop as written plus corrected variants (10 runs each).
usage: python audit/audit.py data/ctg3.csv [config ...]"""
import sys, time
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from scipy.spatial import distance

df = pd.read_csv(sys.argv[1])
df['fetal_health'] = LabelEncoder().fit_transform(df['fetal_health'])
x = df.drop(columns=['fetal_health']); y = df['fetal_health']
x_init, x_left, y_init, y_left = train_test_split(x, y, train_size=0.01415, random_state=42)
print("initial labeled:", len(x_init), "| remaining:", len(x_left))

TUNED = dict(colsample_bytree=0.5, learning_rate=0.05, max_depth=5, n_estimators=200)

def model(params):
    return XGBClassifier(objective="multi:softprob", num_class=3, random_state=42, **params)

def run(seed, loop_params, final_params, diversity="index", query="uncert_div", scaler=None):
    x_tr, y_tr = x_init, y_init
    unl, x_te, lab, y_te = train_test_split(x_left, y_left, test_size=0.5, random_state=seed)
    clean_te, clean_y = x_te.copy(), y_te.copy()          # never-touched held-out half
    rng = np.random.default_rng(seed)
    orig_te, orig_y = x_te, y_te                          # the notebook's growing test set
    for i in range(78):
        m = model(loop_params); m.fit(x_tr, y_tr)
        orig_te = pd.concat([orig_te, unl]); orig_y = pd.concat([orig_y, lab])  # replicate notebook
        if query == "random":
            sel = list(rng.choice(len(unl), 5, replace=False))
        else:
            p = m.predict_proba(unl)
            srt = np.sort(p, axis=1); diff = srt[:, -1] - srt[:, -2]
            top = np.argsort(diff)[:20]
            sel = [top[0]]; rem = list(top[1:])
            feats = unl.values if scaler is None else scaler.transform(unl)
            while len(sel) < 5:
                if diversity == "index":      # what the notebook does: distance between integer positions
                    sums = [sum(distance.euclidean(np.ravel(s), np.ravel(r)) for s in sel) for r in rem]
                else:                         # what the paper describes: distance in feature space
                    sums = [sum(distance.euclidean(feats[s], feats[r]) for s in sel) for r in rem]
                sel.append(rem.pop(int(np.argmax(sums))))
        x_tr = pd.concat([x_tr, unl.iloc[sel]]); y_tr = pd.concat([y_tr, lab.iloc[sel]])
        unl = unl.drop(unl.index[sel]); lab = lab.drop(lab.index[sel])
    fm = model(final_params); fm.fit(x_tr, y_tr)
    orig_te = pd.concat([orig_te, unl]); orig_y = pd.concat([orig_y, lab])
    acc_orig = accuracy_score(orig_y, fm.predict(orig_te)) * 100
    leak = orig_te.index.isin(x_tr.index).mean() * 100
    # notebook-style test but with duplicates removed and training rows excluded
    dedup = orig_te[~orig_te.index.duplicated() & ~orig_te.index.isin(x_tr.index)]
    acc_dedup = accuracy_score(y.loc[dedup.index], fm.predict(dedup)) * 100
    acc_clean = accuracy_score(clean_y, fm.predict(clean_te)) * 100
    f1_clean = f1_score(clean_y, fm.predict(clean_te), average='macro') * 100
    return dict(n_train=len(x_tr), test_rows=len(orig_te), leak_pct=leak, acc_notebook=acc_orig,
                acc_dedup_noleak=acc_dedup, acc_clean_heldout=acc_clean, macroF1_clean=f1_clean)

configs = {
    "1_as_in_notebook":            dict(loop_params=TUNED, final_params={}, diversity="index"),
    "2_feature_space_diversity":   dict(loop_params=TUNED, final_params={}, diversity="feature"),
    "3_paper_faithful":            dict(loop_params={}, final_params=TUNED, diversity="feature"),
    "4_random_baseline":           dict(loop_params={}, final_params=TUNED, query="random"),
}
which = sys.argv[2:] or list(configs)
for name in which:
    t = time.time()
    res = pd.DataFrame([run(s, **configs[name]) for s in range(10)])
    print(f"\n== {name}  ({time.time()-t:.0f}s, 10 runs)")
    print(res.mean().round(2).to_string())
    print("std acc_clean_heldout:", round(res.acc_clean_heldout.std(), 2))
