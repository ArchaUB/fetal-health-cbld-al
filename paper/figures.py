"""Publication figures, generated directly from the saved benchmark results."""
import numpy as np, pandas as pd, matplotlib
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repository root

matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = f'{ROOT}/results'; OUT = f'{ROOT}/paper/figs'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 8, 'axes.titlesize': 8.5, 'axes.labelsize': 8, 'legend.fontsize': 7,
                     'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
                     'grid.color': '#e4e4e0', 'grid.linewidth': 0.5, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})
INK, MUTED = '#2b2b2b', '#6b6b66'
# validated categorical palette (fixed order) + markers/line styles as secondary encoding
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
DS = [('ctg3', 'CTG-3 (fetal state)'), ('ctg10', 'CTG-10 (FHR pattern)')]
def load(d, m): return pd.read_pickle(f'{RES}/{d}__{m}.pkl')

# ---------------------------------------------------------------- Fig. 2: leakage in the original protocol
lt = pd.read_csv(f'{RES}/leak_trace.csv'); lt = lt[lt.it < 78]
g = lt.groupby('n_train')
fig, ax = plt.subplots(2, 1, figsize=(3.45, 3.6), sharex=True, gridspec_kw=dict(hspace=0.18, height_ratios=[1.6, 1]))
for col, lab, c, mk, ls in (('leaky', 'Original protocol (pool appended to test set)', C[1], 's', '-'),
                            ('clean', 'Held-out test set only', C[0], 'o', '--')):
    m, s = g[col].mean(), g[col].std()
    ax[0].plot(m.index, m.values, color=c, lw=1.6, ls=ls, marker=mk, ms=3, markevery=8, label=lab)
    ax[0].fill_between(m.index, m - s, m + s, color=c, alpha=0.15, lw=0)
ax[0].axhline(99.38, color=MUTED, lw=0.8, ls=':'); ax[0].text(35, 99.55, 'Accuracy reported by Bhardwaj et al. (99.38%)', color=MUTED, fontsize=6.5)
ax[0].set_ylabel('Accuracy (%)'); ax[0].set_ylim(80, 101); ax[0].grid(axis='y'); ax[0].legend(loc='lower right', frameon=False)
lp = g['leak_pct'].mean()
ax[1].plot(lp.index, lp.values, color=C[6], lw=1.6, marker='^', ms=3, markevery=8)
ax[1].set_ylabel('Test rows that are\ntraining samples (%)'); ax[1].set_xlabel('Labeled training samples'); ax[1].grid(axis='y')
ax[1].set_ylim(0, 25)
fig.savefig(f'{OUT}/fig_leakage.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig. 3: learning curves
CURVE_M = [('random', 'Random'), ('coreset', 'Core-set'), ('qbc', 'QBC'), ('margin', 'Margin'),
           ('paper', 'Base method (Bhardwaj et al.)'), ('C1', 'Proposed query (leaf diversity)')]
MK = ['x', 'v', 'D', 's', 'o', '^']; LS = [':', ':', '-.', '--', '-', '-']
fig, axes = plt.subplots(2, 2, figsize=(7.16, 4.0), gridspec_kw=dict(wspace=0.2, hspace=0.32))
for j, (d, title) in enumerate(DS):
    for i, (m, lab) in enumerate(CURVE_M):
        R = load(d, m); S = np.array(R[0]['sizes'])
        for row, key in enumerate(('curve', 'curve_bal')):
            v = np.mean([r[key] for r in R], 0)
            axes[row, j].plot(S, v, color=C[i], lw=1.5 if m in ('paper', 'C1') else 1.1, ls=LS[i],
                              marker=MK[i], ms=2.8, markevery=10, label=lab)
    axes[0, j].set_title(title)
    for row in range(2): axes[row, j].grid(axis='y')
    axes[1, j].set_xlabel('Labeled training samples')
lims = {'ctg3': ((84, 95.5), (75, 91)), 'ctg10': ((70, 90), (55, 84))}
for j, (d, _) in enumerate(DS):
    axes[0, j].set_ylim(*lims[d][0]); axes[1, j].set_ylim(*lims[d][1])
axes[0, 0].set_ylabel('Accuracy (%)'); axes[1, 0].set_ylabel('Balanced accuracy (%)')
h, l = axes[0, 0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06), handlelength=2.4)
fig.savefig(f'{OUT}/fig_curves.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig. 4: per-class recall, base vs proposed
NAMES = {'ctg3': ['Norm.', 'Susp.', 'Path.'],
         'ctg10': ['A', 'AD', 'B', 'C', 'D', 'DE', 'E', 'FS', 'LD', 'SUSP']}
fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.7), gridspec_kw=dict(hspace=0.45))
for ax, (d, title) in zip(axes, DS):
    k = len(NAMES[d]); x = np.arange(k); w = 0.27
    for i, (m, lab, c) in enumerate((('paper', 'Base method (Bhardwaj et al.)', C[0]), ('C1C2', 'Proposed (C1+C2)', C[1]),
                                     ('proposed', 'Proposed + stopping (C1+C2+C3)', C[2]))):
        rec = np.mean([r['recall'] for r in load(d, m)], 0)
        ax.bar(x + (i - 1) * w, rec, w * 0.9, color=c, label=lab, edgecolor='white', linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels(NAMES[d]); ax.set_title(title); ax.grid(axis='y')
    ax.set_ylim(30 if d == 'ctg10' else 60, 100)
axes[0].set_ylabel('Recall (%)'); axes[1].set_ylabel('Recall (%)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=1, frameon=False, bbox_to_anchor=(0.5, 1.13))
fig.savefig(f'{OUT}/fig_recall.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig. 5: confusion matrices CTG-3
from matplotlib.colors import LinearSegmentedColormap
blues = LinearSegmentedColormap.from_list('b', ['#f7fbff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
fig, axes = plt.subplots(1, 2, figsize=(3.45, 1.85), gridspec_kw=dict(wspace=0.12))
for ax, (m, title) in zip(axes, (('paper', '(a) Base method'), ('C1C2', '(b) Proposed (C1+C2)'))):
    cm = np.sum([r['cm'] for r in load('ctg3', m)], 0); cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap=blues, vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f'{cmn[a, b]:.3f}', ha='center', va='center', fontsize=6.5, color='white' if cmn[a, b] > 0.55 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels(['Normal','Suspect','Patho.']); ax.set_yticks(range(3))
    ax.set_yticklabels(['Normal','Suspect','Patho.'] if m == 'paper' else []); ax.set_title(title, fontsize=7.5)
    ax.set_xlabel('Predicted'); ax.spines[:].set_visible(False); ax.tick_params(length=0)
axes[0].set_ylabel('True')
fig.savefig(f'{OUT}/fig_cm.pdf'); plt.close(fig)

# ---------------------------------------------------------------- Fig. 6: stopping rule, labels used vs accuracy change
fig, axes = plt.subplots(1, 2, figsize=(3.45, 2.0), gridspec_kw=dict(wspace=0.35))
for ax, (d, title) in zip(axes, DS):
    P = load(d, 'proposed'); F = load(d, 'C1C2')
    labs = np.array([r['labels'] for r in P])
    for key, lab, c, mk in (('acc', 'Accuracy', C[0], 'o'), ('balAcc', 'Balanced accuracy', C[1], 's')):
        dv = np.array([p[key] - f[key] for p, f in zip(P, F)])
        ax.scatter(labs, dv, s=12, color=c, marker=mk, alpha=0.8, edgecolors='white', linewidths=0.4, label=lab)
    ax.axhline(0, color=MUTED, lw=0.7); ax.set_title(title, fontsize=7.5); ax.grid(axis='y'); ax.set_xlabel('Labels used at stop')
    ax.set_xlim(min(100, labs.min() - 20), 430)
axes[0].set_ylabel('Change vs. 420-label\nbudget (points)')
h, l = axes[0].get_legend_handles_labels()
fig.legend(h, l, loc='upper center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.14))
fig.savefig(f'{OUT}/fig_stop.pdf'); plt.close(fig)
print('figures written')
