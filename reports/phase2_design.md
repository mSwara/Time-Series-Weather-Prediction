# Phase 2 — Design Rationale

This document records the design decisions for the deep-learning project and the
evidence behind each one. It is written to be checked against, not taken on faith —
every claim below traces to a reproducible cell in `notebooks/01_eda.ipynb` or
`notebooks/02_preprocessing_windowing.ipynb`.

> Phase 1 (JD-derived skill priorities) is summarized in
> [`reports/phase1_jd_analysis.md`](phase1_jd_analysis.md). The skill choices below
> (CNN-LSTM, Dropout/L2/EarlyStopping/ModelCheckpoint, Keras Tuner, optimizer
> comparison, SMOTE/ADASYN scoped to the classification sub-task) are used *because*
> they were the ones Phase 1 found evidenced in real JDs — not picked first and
> justified after.

## 1. Problem statement — how this differs from the internship

| | Internship (classical ML) | This project (deep learning) |
|---|---|---|
| Targets | DBT and WBT modeled **independently** (2 separate models, 2 separate feature sets) | DBT and WBT modeled **jointly** — one shared backbone, two output heads (multi-task) |
| Temporal features | `T(i-1)..T(i-4)` handed to the model as 4 unordered flat columns | Same 4 values, but explicitly encoded as an **ordered sequence** (see §3) fed through convolution + recurrence |
| Split | Random 80:20 (`random_state=42`) | Chronological, **burst-aware** 70/15/15 (train/val/test) — see §2 |
| Model family | Linear/Ridge, tree ensembles, boosting, KNN, SVR | CNN, LSTM, CNN-LSTM hybrid (+ a lightweight self-attention layer over the 4-step sequence, added per Phase 1 evidence — see below), compared against the same classical baselines refit on the new split |
| Regularization | GridSearchCV, α tuning (Ridge), num_leaves (LightGBM) | Dropout + L2 + EarlyStopping + ModelCheckpoint(best epoch) + constrained Keras Tuner search |
| Class-imbalance tooling | None used (regression-only) | SMOTE/ADASYN applied **only** to an auxiliary heat-stress-day classification head — never to the continuous regression target (see §5) |
| Interpretability | SHAP (tree/linear), Ridge coefficients | SHAP (KernelExplainer/DeepExplainer on the DL model) compared directly against the internship's SHAP rankings |
| Evaluation | R², RMSE, MAE, MAPE, KGE | Same metrics (KGE retained specifically to stay comparable), plus explicit train/val/test gap reporting as an overfitting check |

The reframing chosen is **joint multi-task regression via a CNN-LSTM that treats the
lag-temperature sequence + same-day atmospheric state as a short "temporal image"**
(the first of the three options the brief offered), with a secondary auxiliary
classification head added where legitimate (see §5). The multi-step-ahead forecasting
option was evaluated and rejected — see §2.3 for the data evidence.

Phase 1's JD analysis (`reports/phase1_jd_analysis.md`) found CNN-LSTM, Keras Tuner,
and SMOTE/ADASYN essentially absent as *named* skills in the 58 fetched JDs (0-1
hits each), while attention/Transformer-style architectures appeared far more often
(8/58, ahead of plain CNN's 3/58). Rather than silently keep the brief's original
architecture or silently swap to what Phase 1 found, one concrete adjustment is
made: a lightweight self-attention layer is added over the 4-timestep sequence
ahead of pooling — small enough not to compromise the parameter-count discipline in
§4, and the one architectural change actually evidenced by Phase 1 data. GenAI/LLM/RAG
(the single largest JD theme by far) is deliberately **not** adopted — see
`phase1_jd_analysis.md` for why forcing it in here would repeat the SMOTE-on-regression
mistake this brief explicitly asked to avoid.

## 2. Why chronological, burst-aware splitting (not the internship's random split)

### 2.1 The data is not a continuous series

Sorting the 300 rows by `Date` shows 70 of 299 consecutive-row gaps are **not** 1 day
— the data is 71 disjoint bursts of 1-10 consecutive days scattered across 1980-2020
(`notebooks/01_eda.ipynb`, cells 4-6). This is almost certainly a curated sample of
individual multi-day heat events, not a continuous daily record.

### 2.2 Random splitting would leak

`T(i-1)..T(i-4)` are lag temperatures, verified (0 mismatches, 299 checks) to equal
the DBT of the previous day *within the same burst*. A random 80:20 split — what the
internship used — can and does place a burst's early days in train and its later days
in test (or vice versa), meaning a test-set row's lag feature would be *derived from*
a target value that the model saw during training. This is a subtler version of the
leakage the internship explicitly guarded against (excluding DBT when predicting WBT
and vice versa) — same principle, different axis (time, not columns).

This project instead splits **whole bursts**, in chronological order, into
train/val/test (`src/data.block_chronological_split`), producing 210/46/44 rows
(70.0% / 15.3% / 14.7%) with no burst crossing a split boundary and no split
overlapping another in time. This is a *stricter* standard than the internship
applied, not merely a different one — worth stating plainly since it means any
head-to-head comparison of R² between the two projects is already on an uneven
footing in this project's favor (harder, non-random test set) rather than the DL
model's.

### 2.3 Why multi-step-ahead forecasting was rejected

The brief's third reframing option — predicting N days ahead instead of 1 — was
evaluated and rejected on data grounds, not preference. Multi-step windows need
even longer unbroken bursts than single-step windows, and burst lengths are already
short (median 3-4 days, max 10). Quantified in `01_eda.ipynb`: a raw single-step
sliding window at k=4 already only recovers 50/300 rows (17%); 2-step-ahead
compounds this further. With training data already this scarce, fragmenting it
further for multi-step-ahead was judged indefensible — it would produce a model
whose reported metrics are not trustworthy at n<40. Joint multi-output single-step
prediction was chosen instead specifically because it does *not* require discarding
rows this way (see §3).

## 3. Why the sequence is built from the lag block, not re-derived from raw days

A "true" sequence model would ideally consume each prior day's full meteorological
vector. The dataset does not provide that — only each day's *own* meteorological
state, plus scalar lag temperatures for the previous 4 days. Re-deriving raw 4-day
windows from the burst structure would use only 50/300 rows (17%); the precomputed
lag block is available, leakage-checked, for all 300 (299 after dropping the 1 row
missing `T(i-1)`).

The chosen tensor per sample is therefore `(4 timesteps x 10 channels)`:
- Channel 0: the ordered lag-temperature sequence `T(i-4) -> T(i-3) -> T(i-2) -> T(i-1)`
  — the one quantity that genuinely varies across the 4 steps.
- Channels 1-9: the current day's static features (8 meteorological variables +
  `Hr`), broadcast identically across all 4 timesteps.

This is presented honestly as a **proxy temporal image**, not a true multivariate
history — broadcasting lets a CNN learn local joint patterns between the recent
warming/cooling trend and the current atmospheric state, and an LSTM/CNN-LSTM model
how that combination evolves across the lag window. The raw information content
handed to the model is the same 12 numbers Ridge/LightGBM saw; what changes is the
inductive bias — order and locality are now explicit and structural, not something
a flat model has to infer (or fail to) from column position.

## 4. Small-sample architecture discipline (n=210 train rows)

- Kept small/shallow throughout: target parameter counts are reasoned about
  explicitly against the 210-row training set in `notebooks/03_models_baselines.ipynb`
  and `04_cnn_lstm.ipynb` (rule of thumb applied: aim for well under 210 trainable
  parameters per "effective" degree of freedom is impossible for any real net, so the
  real constraint enforced is aggressive regularization + early stopping + a
  validation-gated, narrow Keras Tuner search — not a parameter-count target that
  can't be met).
- Dropout + L2 + EarlyStopping(monitor=val_loss) + ModelCheckpoint(save_best_only,
  monitor=val_loss) are used as the core anti-overfitting stack on every DL model.
- Keras Tuner search space is deliberately narrow (few units/layers/dropout/lr
  choices) and selected against the **validation** split, never the test split;
  the test set is touched exactly once, at the end, per model.
- The real held-out test set (44 rows) is never augmented or synthesized — augmentation
  (see §5) only ever touches the training fold.

## 5. Synthetic data / augmentation — what it can and cannot do here

For the regression task: light **jittering (Gaussian noise injection)** on the
scaled training sequences is used as a regularizer during training (comparable in
spirit to how Dropout regularizes activations, applied instead to inputs). This is
explicitly *not* presented as creating new real observations — it cannot add
information the physical system didn't produce, only discourage the small network
from memorizing exact training points. Block bootstrap of whole bursts was
considered as an alternative/addition; jittering was judged sufficient and more
transparent for a first version, and is called out here as the honest choice made.

**SMOTE/ADASYN are not applied to the regression target.** They are class-imbalance
resampling techniques for classification and have no valid meaning against a
continuous physical quantity — applying them to DBT/WBT directly would fabricate
implausible interpolated temperature values and misrepresent what the model was
trained on. Instead, they are applied to the auxiliary **heat-stress-day
classification** sub-task: a binary label (`is_heat_stress_day`) derived from
whether WBT exceeds a threshold set from the *training* distribution only (avoiding
leakage from val/test into the threshold itself). This is where the rare/extreme
class genuinely is imbalanced, and where SMOTE/ADASYN are the textbook-correct tool
— full detail and the exact threshold/class balance are in
`notebooks/05_classification_subtask.ipynb`.

## 6. Evaluation

R², RMSE, MAE, and KGE are reported for every model on train/val/test, specifically
to stay comparable to the internship's Table 6 (DBT Ridge: R²=0.6213, RMSE=1.2064,
MAE=0.8792, KGE=0.7815; WBT LightGBM: R²=0.7092, RMSE=0.7372, MAE=0.5442,
KGE=0.7474). Training/validation loss curves and an explicit train-vs-test gap are
reported for every DL model as the overfitting check the brief asked for.
