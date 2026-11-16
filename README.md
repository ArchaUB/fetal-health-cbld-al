# Leakage-free re-evaluation and class-balanced leaf-diversity active learning (CBLD-AL) for fetal health classification

Code and data for the paper *"Leakage-Free Re-Evaluation and Class-Balanced Leaf-Diversity Active Learning for Fetal Health Classification with XGBoost"*.

It contains:
1. an **audit** of the XGBoost active-learning method of Bhardwaj et al., *IEEE Access* 13 (2025), doi:10.1109/ACCESS.2025.3527151 (original code: `niyg7/fetal-health-dataset`);
2. the **benchmark** behind every table and figure in the paper: 2 fetal monitoring tasks × 11 active-learning methods + supervised references × 30 paired runs, and a comparison of four tree-ensemble classifiers (XGBoost, LightGBM, CatBoost, random forest) under the same active-learning method.

## Repository layout

| Path | Contents |
|---|---|
| `data/` | `ctg3.csv` (CTG fetal state, 3 classes; same file as the original repository), `ctg10.csv` (same CTG recordings, 10 FHR-pattern classes) |
| `audit/audit.py` | Runs the original notebook's loop as written and the corrected variants; reports the leaky accuracy, the share of test rows that are training samples, and held-out accuracy |
| `audit/leak_trace.py` | Per-iteration leaky vs. held-out accuracy (Fig. 2 of the paper) |
| `benchmark/bench.py` | All methods: random, entropy, margin, QBC, core-set, margin + k-means, base method (corrected), C1, C2, C1+C2 (CBLD-AL), C1+C2+C3, plus full-pool supervised references. The classifier is chosen with `BACKBONE=xgb|lgbm|cat|rf` |
| `benchmark/analyse.py` | Summary tables and paired Wilcoxon tests printed to the console |
| `paper/tables.py`, `paper/figures.py`, `paper/backbones.py`, `paper/perclass_tests.py` | Regenerate the paper's tables (`paper/tables/`) and figures (`paper/figs/`) from `results/` |
| `paper/fig_arch.tex` | Source of the architecture diagram (compile with `pdflatex`) |
| `results/` | Saved per-run results of the full benchmark (pickle files, 30 seeds) and the leakage trace |
| `notebooks/` | `Fetal_Health_AL_corrected.ipynb` (corrected base method) and `Fetal_Health_AL_improved.ipynb` (CTG-3 experiments with the proposed components) |

## Setup

```bash
pip install -r requirements.txt
```
Tested with Python 3.12 and the versions pinned in `requirements.txt`.

## Reproducing the results

Run everything from the repository root.

```bash
# 1. Audit of the original implementation (about 2 min per configuration)
python audit/audit.py data/ctg3.csv
python audit/leak_trace.py

# 2. Full benchmark (about 1 h on 2 CPU cores; existing files in results/ are skipped,
#    so delete or move results/*.pkl to recompute)
python benchmark/bench.py ctg3
python benchmark/bench.py ctg10

# 2b. Classifier comparison (about 2 h): same method with other tree ensembles
for bb in lgbm rf cat; do
  BACKBONE=$bb python benchmark/bench.py ctg3 paper C1C2 proposed full
  BACKBONE=$bb python benchmark/bench.py ctg10 paper C1C2 proposed full
done

# 3. Statistics, tables and figures
python benchmark/analyse.py
python paper/tables.py
python paper/figures.py
python paper/backbones.py
python paper/perclass_tests.py
```

Quick check with fewer runs: `NSEEDS=2 RESULTS_DIR=/tmp/test python benchmark/bench.py ctg3 paper proposed`.

## Evaluation protocol

For every run, 30 initial samples are drawn at random and the remaining data are split 50/50 into the unlabelled pool and a held-out test set. The test set is never queried or trained on, and the code asserts this after every run. The original implementation instead appends the unlabelled pool to the test set in every iteration, so that about 22.5% of the final test rows are copies of training samples; this produces its reported 99.38% accuracy, against about 94.5% on held-out data.

## Data sources

Both datasets come from the UCI Machine Learning Repository and are redistributed here, with attribution, for reproducibility:
- **Cardiotocography**, D. Campos and J. Bernardes, 2010, doi:10.24432/C51S4N. `ctg3.csv` uses the fetal-state label (NSP); `ctg10.csv` uses the 10-class FHR-pattern label (CLASS) and the 21 SisPorto features.

## Citation

If you use this code, please cite the paper (citation to be added after publication).
