"""Benchmark: active-learning methods x datasets x seeds, all scored on a clean held-out test set.

usage: python benchmark/bench.py <ctg3|ctg10> [method ...]   results -> results/<dataset>__<method>.pkl
Environment: NSEEDS (default 30), RESULTS_DIR (default <repo>/results), BACKBONE (xgb|lgbm|cat|rf, default xgb). Existing result files are skipped.
"""
import os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd
from xgboost import XGBClassifier
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import (accuracy_score, f1_score, recall_score, balanced_accuracy_score,
                             cohen_kappa_score, confusion_matrix)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root
RESULTS = os.environ.get('RESULTS_DIR', f'{ROOT}/results')
N_SEEDS = int(os.environ.get('NSEEDS', 30))
M_INIT, N_UNC, K, N_ITER = 30, 20, 5, 78          # paper protocol: 30 + 78*5 = 420 labels
TUNED = dict(n_estimators=200, learning_rate=0.05, max_depth=5, min_child_weight=1,
             gamma=0, subsample=1, colsample_bytree=0.5)
STOP_KAPPA, STOP_WINDOW = 0.998, 5
QBC_SIZE = 5

# ---------------------------------------------------------------- data
def load(name):
    if name == 'ctg3':
        df = pd.read_csv(f'{ROOT}/data/ctg3.csv'); y = df.pop('fetal_health')
    else:
        df = pd.read_csv(f'{ROOT}/data/{name}.csv'); y = df.pop('target')
    le = LabelEncoder().fit(y.astype(str))
    return df.reset_index(drop=True), pd.Series(le.transform(y.astype(str))), list(le.classes_)

def make_split(X, y, seed, ensure_all_classes):
    rng = np.random.default_rng(seed)
    if ensure_all_classes:      # one random sample per class, the rest uniformly at random (cold start)
        first = [rng.choice(np.where(y == c)[0]) for c in np.unique(y)]
        rest = rng.choice(np.setdiff1d(np.arange(len(y)), first), M_INIT - len(first), replace=False)
        init = np.r_[first, rest]
    else:                       # paper protocol: purely random initial set
        init = rng.choice(len(y), M_INIT, replace=False)
    others = np.setdiff1d(np.arange(len(y)), init)
    pool, test = train_test_split(others, train_size=0.5, random_state=seed)
    return init, pool, test

# ---------------------------------------------------------------- model
BACKBONE = os.environ.get('BACKBONE', 'xgb')      # xgb | lgbm | cat | rf

def _make_estimator(n_cls, seed, final):
    """Tree-ensemble backbones. Loop models use library defaults (CatBoost capped at 200 trees for run time);
    final models use the settings tuned for XGBoost in the base paper, mapped to each library."""
    if BACKBONE == 'xgb':
        return XGBClassifier(objective='multi:softprob', num_class=n_cls, random_state=seed, n_jobs=1, **(TUNED if final else {}))
    if BACKBONE == 'lgbm':
        from lightgbm import LGBMClassifier
        p = dict(n_estimators=200, learning_rate=0.05, max_depth=5, colsample_bytree=0.5) if final else {}
        return LGBMClassifier(random_state=seed, n_jobs=1, verbose=-1, **p)
    if BACKBONE == 'cat':
        from catboost import CatBoostClassifier
        p = dict(iterations=200, learning_rate=0.05, depth=5, rsm=0.5) if final else dict(iterations=200)
        return CatBoostClassifier(loss_function='MultiClass', random_seed=seed, thread_count=1, verbose=0,
                                  allow_writing_files=False, **p)
    if BACKBONE == 'rf':
        from sklearn.ensemble import RandomForestClassifier
        p = dict(n_estimators=200, max_features=0.5) if final else {}
        return RandomForestClassifier(random_state=seed, n_jobs=1, **p)
    raise ValueError(BACKBONE)

class KXGB:
    """Tree-ensemble classifier (backbone set by BACKBONE) that tolerates classes missing from the training set;
    predict_proba always returns K columns. The name is kept from the XGBoost-only version."""
    def __init__(self, K, seed, final=False):
        self.K, self.seed, self.final = K, seed, final
    def fit(self, X, y, sample_weight=None):
        self.present = np.unique(y)
        if len(self.present) == 1:
            self.const = self.present[0]; return self
        self.const = None
        ymap = np.searchsorted(self.present, y)
        self.m = _make_estimator(len(self.present), self.seed, self.final)
        self.m.fit(X, ymap, sample_weight=sample_weight); return self
    def predict_proba(self, X):
        P = np.zeros((len(X), self.K))
        if self.const is not None: P[:, self.const] = 1; return P
        P[:, self.present] = self.m.predict_proba(X); return P
    def predict(self, X): return self.predict_proba(X).argmax(1)
    def apply(self, X):
        """Leaf index of every sample in every tree (leaf embedding used by C1)."""
        if self.const is not None: return np.zeros((len(X), 1))
        if BACKBONE == 'lgbm': L = self.m.predict(X, pred_leaf=True)
        elif BACKBONE == 'cat': L = self.m.calc_leaf_indexes(X)
        else: L = self.m.apply(X)
        return np.asarray(L).reshape(len(X), -1).astype(float)

def margin(P):
    s = np.sort(P, 1); return s[:, -1] - s[:, -2]
def entropy(P):
    return -(P * np.log(P + 1e-12)).sum(1)

# ---------------------------------------------------------------- query strategies (return positions in pool)
def greedy_div(F, metric, cand):
    sel, rem = [cand[0]], list(cand[1:])
    while len(sel) < K and rem:
        d = cdist(F[rem], F[sel], metric=metric).sum(1)
        sel.append(rem.pop(int(np.argmax(d))))
    return sel

def q_random(ctx):   return list(ctx['rng'].choice(len(ctx['Xp']), K, replace=False))
def q_entropy(ctx):  return list(np.argsort(-entropy(ctx['Pp']))[:K])
def q_margin(ctx):   return list(np.argsort(margin(ctx['Pp']))[:K])

def q_qbc(ctx):
    """Query-by-committee: bagged XGBoost committee, KL divergence of members from the consensus."""
    Xl, yl, rng = ctx['Xl'], ctx['yl'], ctx['rng']; probs = []
    for j in range(QBC_SIZE):
        idx = rng.choice(len(yl), len(yl), replace=True)
        probs.append(KXGB(ctx['K'], ctx['seed'] * 31 + j).fit(Xl[idx], yl[idx]).predict_proba(ctx['Xp']))
    probs = np.array(probs); mean = probs.mean(0)
    kl = (probs * (np.log(probs + 1e-12) - np.log(mean + 1e-12))).sum(2).mean(0)
    return list(np.argsort(-kl)[:K])

def q_coreset(ctx):
    """Core-set / k-center greedy (Sener & Savarese 2018) on standardized features."""
    Fp, Fl = ctx['Sp'], ctx['Sl']
    dmin = cdist(Fp, Fl).min(1); sel = []
    for _ in range(K):
        i = int(np.argmax(dmin)); sel.append(i)
        dmin = np.minimum(dmin, cdist(Fp, Fp[i:i + 1]).ravel())
    return sel

def q_margin_kmeans(ctx):
    """Uncertainty + clustering diversity: k-means on the n most uncertain, take the sample nearest each centroid."""
    top = np.argsort(margin(ctx['Pp']))[:N_UNC]; F = ctx['Sp'][top]
    km = KMeans(K, n_init=5, random_state=ctx['seed']).fit(F)
    sel = []
    for c in range(K):
        members = np.where(km.labels_ == c)[0]
        sel.append(int(top[members[np.argmin(((F[members] - km.cluster_centers_[c]) ** 2).sum(1))]]))
    return sel

def q_paper(ctx):    # base paper: margin top-n, greedy summed Euclidean distance on raw features
    top = list(np.argsort(margin(ctx['Pp']))[:N_UNC]); return greedy_div(ctx['Xp'], 'euclidean', top)

def q_leaf(ctx):     # C1: same, but Hamming distance between XGBoost leaf indices
    top = list(np.argsort(margin(ctx['Pp']))[:N_UNC])
    return greedy_div(ctx['model'].apply(ctx['Xp']), 'hamming', top)

METHODS = {   # name: (query, class-balanced final model, stopping rule)
    'random':        (q_random, False, False),
    'entropy':       (q_entropy, False, False),
    'margin':        (q_margin, False, False),
    'qbc':           (q_qbc, False, False),
    'coreset':       (q_coreset, False, False),
    'margin_kmeans': (q_margin_kmeans, False, False),
    'paper':         (q_paper, False, False),
    'C1':            (q_leaf, False, False),
    'C2':            (q_paper, True, False),
    'C1C2':          (q_leaf, True, False),
    'proposed':      (q_leaf, True, True),
}

# ---------------------------------------------------------------- one run
def metrics(yt, yp, Kc):
    return dict(acc=accuracy_score(yt, yp) * 100, macroF1=f1_score(yt, yp, average='macro') * 100,
                balAcc=balanced_accuracy_score(yt, yp) * 100,
                recall=recall_score(yt, yp, average=None, labels=list(range(Kc)), zero_division=0) * 100,
                cm=confusion_matrix(yt, yp, labels=list(range(Kc))))

def run(dname, method, seed):
    X, y, classes = load(dname); Kc = len(classes)
    Xv, yv = X.values.astype(float), y.values
    scaler = StandardScaler().fit(Xv); Sv = scaler.transform(Xv)
    init, pool, test = make_split(X, y, seed, ensure_all_classes=(dname != 'ctg3'))
    query, balanced, stop = METHODS[method]
    rng = np.random.default_rng(seed + 12345)
    lab, pool = list(init), list(pool)
    stop_set = Xv[pool]; prev = None; kappas = []; curve = []; curve_bal = []; sizes = []
    for it in range(N_ITER):
        model = KXGB(Kc, seed).fit(Xv[lab], yv[lab])
        yp = model.predict(Xv[test])
        curve.append(accuracy_score(yv[test], yp) * 100); curve_bal.append(balanced_accuracy_score(yv[test], yp) * 100)
        sizes.append(len(lab))
        cur = model.predict(stop_set)
        if prev is not None: kappas.append(cohen_kappa_score(prev, cur))
        prev = cur
        if stop and len(kappas) >= STOP_WINDOW and min(kappas[-STOP_WINDOW:]) >= STOP_KAPPA:
            break
        ctx = dict(rng=rng, seed=seed, K=Kc, model=model, Xp=Xv[pool], Pp=model.predict_proba(Xv[pool]),
                   Xl=Xv[lab], yl=yv[lab], Sp=Sv[pool], Sl=Sv[lab])
        sel = query(ctx)
        chosen = [pool[i] for i in sel]
        lab += chosen; pool = [p for p in pool if p not in set(chosen)]
    assert not set(lab) & set(test)
    sw = compute_sample_weight('balanced', yv[lab]) if balanced else None
    final = KXGB(Kc, seed, final=True).fit(Xv[lab], yv[lab], sample_weight=sw)
    out = metrics(yv[test], final.predict(Xv[test]), Kc)
    out.update(labels=len(lab), curve=curve, curve_bal=curve_bal, sizes=sizes, seed=seed)
    return out

def run_full(dname, seed, balanced=False):
    X, y, classes = load(dname); Kc = len(classes); Xv, yv = X.values.astype(float), y.values
    init, pool, test = make_split(X, y, seed, ensure_all_classes=(dname != 'ctg3'))
    tr = np.r_[init, pool]
    sw = compute_sample_weight('balanced', yv[tr]) if balanced else None
    m = KXGB(Kc, seed, final=True).fit(Xv[tr], yv[tr], sample_weight=sw)
    out = metrics(yv[test], m.predict(Xv[test]), Kc); out.update(labels=len(tr), seed=seed); return out

if __name__ == '__main__':
    dname = sys.argv[1]; methods = sys.argv[2:] or list(METHODS) + ['full', 'full_bal']
    os.makedirs(RESULTS, exist_ok=True)
    for m in methods:
        tag = '' if BACKBONE == 'xgb' else f'{BACKBONE}__'
        path = f'{RESULTS}/{dname}__{tag}{m}.pkl'
        if os.path.exists(path): print('skip', path, flush=True); continue
        t = time.time()
        if m.startswith('full'):
            res = [run_full(dname, s, balanced=(m == 'full_bal')) for s in range(N_SEEDS)]
        else:
            res = Parallel(n_jobs=-1)(delayed(run)(dname, m, s) for s in range(N_SEEDS))
        pd.to_pickle(res, path)
        print(f'{dname:8s} {BACKBONE:5s} {m:14s} acc={np.mean([r["acc"] for r in res]):.2f} '
              f'F1={np.mean([r["macroF1"] for r in res]):.2f} bal={np.mean([r["balAcc"] for r in res]):.2f} '
              f'labels={np.mean([r["labels"] for r in res]):.0f}  ({time.time() - t:.0f}s)', flush=True)
