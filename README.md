# Multi-Task CNN-LSTM for Joint Dry/Wet-Bulb Temperature Forecasting

Deep-learning extension of a prior classical-ML internship project that predicted
Dry Bulb Temperature (DBT) and Wet Bulb Temperature (WBT) over Kolkata from ERA5
reanalysis variables. Same underlying 300-row dataset, deliberately different
framing: one multi-task CNN-LSTM with self-attention predicting both targets
jointly from an engineered sequence representation, trained and evaluated with a
stricter, leakage-aware methodology than the original project used.

## Objective

Design and train a multi-task CNN-LSTM (TensorFlow/Keras) to jointly predict DBT
and WBT from ERA5 meteorological reanalysis data, using a custom leakage-aware
sequence-windowing pipeline, a constrained Keras Tuner hyperparameter search, and
SHAP-based interpretability — benchmarked honestly against classical ML baselines
and the original internship's reported results.

## Key results

| Target | Best model overall | Test R² | This project's best DL model | Test R² |
|---|---|---|---|---|
| DBT | Random Forest (this split) | 0.711 | CNN-LSTM hybrid (tuned) | **0.662** (beats internship's original Ridge R²=0.621) |
| WBT | Internship LightGBM (original) | 0.709 | CNN-LSTM hybrid (tuned) | **0.611** |

Full leaderboard, ablations, optimizer comparison, and SHAP findings:
[`reports/final_comparison.md`](reports/final_comparison.md). Reported as the actual
finding, not adjusted to look better — at n≈300 the deep learning model does **not**
uniformly beat well-tuned classical ML, and that is stated plainly rather than
worked around.

**Also tested and reported honestly:**
- Gaussian-noise (jittering) data augmentation was hypothesized to help, tested
  empirically, and found to **hurt** test R² on both targets — see
  [`notebooks/07_augmentation_ablation.ipynb`](notebooks/07_augmentation_ablation.ipynb).
- The Keras Tuner search's best configuration **disabled** the self-attention
  layer — added on JD-analysis evidence, not selected as helpful on this dataset.

## Why this isn't "the internship with an LSTM"

The internship trained two **independent single-target** regressors (Ridge for
DBT, LightGBM for WBT) on a flat feature table, selected by a random 80:20 split.
This project instead:

- Predicts DBT and WBT **jointly** from one multi-task CNN-LSTM (shared backbone,
  two output heads), rather than two separate single-target models.
- Treats the four lag-temperature features `T(i-1)…T(i-4)` as an explicit **ordered
  sequence** fed through convolution + recurrence, instead of four unordered flat
  columns handed to a tree/linear model with no notion of temporal adjacency.
- Uses a **chronological, burst-aware** split instead of a random one (see below) —
  a stricter leakage standard than the internship applied.
- Adds an auxiliary **heat-stress-day classification** head, where SMOTE/ADASYN are
  legitimately applicable, instead of forcing class-imbalance techniques onto a
  continuous regression target.

## Data integrity findings that shape the design

Both established during exploratory analysis, before any modeling:

1. **The 300 rows are not a continuous daily series.** Sorting by `Date` reveals 71
   disjoint "bursts" of consecutive calendar days (median length 3–4, max 10),
   spanning 1980–2020, separated by gaps sometimes over a year. A naive sliding
   window across the full sorted table would silently splice together physically
   unrelated days.
2. **`T(i-1)…T(i-4)` are already leakage-checked lag features**, verified to equal
   the DBT of the previous row within the same burst (0 mismatches across 299
   burst-internal transitions). They are the only genuine temporal signal available
   per row — the raw table does not carry each lag day's full meteorological
   vector, only its temperature — which constrains what a "sequence" can honestly
   represent (see [`reports/phase2_design.md`](reports/phase2_design.md)).
3. Consequently: any train/val/test split, and any windowing scheme, must cut only
   at **burst boundaries**, never mid-burst — otherwise a lag feature in one split
   would be derived from a target value sitting in another split.

## Tech stack

TensorFlow / Keras · Keras Tuner · SHAP · scikit-learn · imbalanced-learn (SMOTE,
ADASYN) · pandas · NumPy · matplotlib / seaborn · Jupyter

## Project layout

```
data/raw/             Original dataset, untouched (FINAL_DATASET_allpara.xlsx)
data/processed/       Deterministic pipeline outputs (git-ignored — regenerate from notebooks)
notebooks/            Numbered, runnable notebooks — the reproducibility trail
src/                  Reusable pipeline code imported by the notebooks
reports/              Written findings: JD analysis, design rationale, final comparison
models/checkpoints/   Saved best-epoch models (git-ignored — regenerate by re-running notebooks)
```

### Notebooks (run in order)

| # | Notebook | Produces |
|---|---|---|
| 01 | `01_eda.ipynb` | Burst-structure discovery, leakage checks, target/feature distributions |
| 02 | `02_preprocessing_windowing.ipynb` | Chronological burst-aware split, flat + sequence feature representations |
| 03 | `03_classical_baselines.ipynb` | Internship's model roster refit on this project's split (the DL yardstick) |
| 04 | `04_cnn_lstm.ipynb` | LSTM/CNN/CNN-LSTM ablation, optimizer comparison, Keras Tuner search |
| 05 | `05_classification_subtask.ipynb` | Auxiliary heat-stress-day classifier (SMOTE/ADASYN, scoped) |
| 06 | `06_interpretability.ipynb` | SHAP on the tuned CNN-LSTM vs. the internship's SHAP findings |
| 07 | `07_augmentation_ablation.ipynb` | Tests the jittering-augmentation proposal empirically (helps / doesn't) |

### Reports

| Report | Contents |
|---|---|
| [`reports/phase1_jd_analysis.md`](reports/phase1_jd_analysis.md) | JD-frequency analysis of 58 real placement postings that shaped which DL techniques this project prioritizes |
| [`reports/phase2_design.md`](reports/phase2_design.md) | Full design rationale: problem reframing, split/windowing justification, small-sample discipline, augmentation approach |
| [`reports/final_comparison.md`](reports/final_comparison.md) | Complete results: headline table, ablations, optimizer comparison, overfitting check, SHAP comparison, honest takeaways |

## Setup

TensorFlow does not currently ship wheels for Python 3.14, so this project pins a
dedicated **Python 3.10** virtual environment:

```bash
py -3.10 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Run notebooks with the `.venv` kernel, in numeric order — each is a standalone,
re-runnable record of one pipeline stage (see individual notebook headers for what
each stage produces and consumes, and `data/processed/` for intermediate
artifacts).

## Methodology highlights

- **Leakage prevention**: chronological, burst-aware train/val/test split (no
  random shuffling of a time series); scalers fit on the training split only.
- **Small-sample discipline**: shallow architectures (≤3.2k parameters on 210
  training rows), Dropout + L2 + EarlyStopping(val_loss) +
  ModelCheckpoint(best-epoch) on every model, a deliberately narrow Keras Tuner
  search space, and every result disclosed as a single run's outcome given
  measured run-to-run variance at this scale.
- **Ablations, not assertions**: CNN-only vs. LSTM-only vs. CNN-LSTM-hybrid,
  Adam vs. SGD vs. RMSprop, and jittering-augmentation on/off — each tested
  head-to-head rather than assumed.
- **Interpretability parity**: SHAP applied to the DL model using the same
  framework the internship used, enabling a direct feature-importance
  comparison rather than two incompatible explanation methods.
