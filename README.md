# LSTM Weather Forecasting — Kolkata Daily Temperature

An LSTM-based forecasting pipeline that predicts next-day dry-bulb temperature
(DBT) for Kolkata from 40 years of real daily weather data, with a full
leakage-safe methodology: walk-forward cross-validation, a genuinely untouched
holdout test set, constrained hyperparameter tuning, an optimizer comparison,
a naive-baseline sanity check, and SHAP interpretability.

## Objective

Design and train an LSTM (TensorFlow/Keras) to forecast next-day temperature
from a 7-day historical window of real meteorological data, using a
leakage-aware walk-forward validation methodology, a constrained Keras Tuner
hyperparameter search, and SHAP-based interpretability — benchmarked against
a naive persistence baseline to confirm the model adds genuine value.

## Key results

| Metric | Persistence baseline | Tuned LSTM |
|---|---|---|
| R² | 0.9605 | **0.9681** |
| RMSE | 0.808°C | **0.725°C** (−10.2%) |
| MAE | 0.598°C | **0.555°C** (−7.2%) |
| MAPE | 2.414% | **2.204%** |
| Walk-forward CV R² (5 folds) | — | mean 0.969, std 0.001 |

Temperature has strong day-to-day persistence, so a naive
"tomorrow = today" baseline already scores R²=0.96 for free. The LSTM adds
real, measured value on top of it (10% lower RMSE, 7% lower MAE) — a genuine
but modest improvement, reported as-is rather than inflated. Full numbers,
methodology, and discussion: [`reports/large_scale_extension.md`](reports/large_scale_extension.md).

## Data

- **Source**: [Open-Meteo's historical weather API](https://open-meteo.com/en/docs/historical-weather-api)
  — ERA5/ERA5-Land reanalysis-backed, free, no account required.
- **Coverage**: 14,610 continuous daily records, 1980-01-01 to 2019-12-31,
  Kolkata — verified strictly continuous (every gap between consecutive rows
  is exactly 1 day) and complete (zero missing values).
- **Features**: 15 daily variables per timestep — DBT, WBT, temperature
  max/min, humidity, wind, cloud cover, shortwave radiation, surface
  pressure, evapotranspiration, precipitation, vapour pressure deficit,
  sunshine duration, plus cyclical day-of-year encoding (`sin`/`cos`) for
  seasonality.
- Fetched and versioned via [`src/fetch_large_data.py`](src/fetch_large_data.py)
  — fully reproducible, no manual download step.

## Methodology

- **Sequence windows**: each sample is the past **7 real days'** full
  15-feature vectors, predicting day *t*'s DBT — genuine historical windows,
  not a synthetic proxy, since the underlying series is truly continuous.
- **Walk-forward (expanding-window) cross-validation**: 5 folds, each trained
  on all data up to a point and validated on the next block forward in time
  — never validating on a period older than its own training data, which is
  what prevents look-ahead bias.
- **Held-out test set**: the final 4 years (2016–2019), carved out first and
  scored exactly once, at the end.
- **Regularization**: Dropout + L2 (on both kernel and recurrent LSTM
  weights) + EarlyStopping(val_loss, best-weights restored) +
  ModelCheckpoint(best-epoch only).
- **Hyperparameter tuning**: constrained Keras Tuner random search (units,
  dropout, L2 strength, dense-layer size, learning rate), validated against
  the largest walk-forward fold.
- **Optimizer comparison**: Adam vs. SGD vs. RMSprop, same tuned
  architecture, each scored once on the real holdout set.
- **Baseline check**: naive persistence (predict day *t* = day *t-1*'s
  actual value) — the honest bar the model has to clear.
- **Interpretability**: SHAP (`KernelExplainer` on the flattened sequence
  input), channel-summed across the 7-day window for a per-feature ranking.

## Model architecture

```
Input(7 timesteps, 15 features)
  -> LSTM(64 units, L2=1e-4 kernel + recurrent regularization)
  -> Dropout(0.2)
  -> Dense(64, ReLU, L2=1e-4)
  -> Dropout(0.2)
  -> Dense(1, linear)
```

**24,705 trainable parameters** (verified via `model.count_params()`), found
via the Keras Tuner search described above. Full parameter-count derivation
in [`notebooks/09_large_scale_lstm_training.ipynb`](notebooks/09_large_scale_lstm_training.ipynb).

## Interpretability (SHAP)

| Feature | Share of total importance |
|---|---|
| Recent DBT history | **35.6%** |
| Recent WBT history | 21.9% |
| Recent max temperature | 17.1% |
| Day-of-year (season) | 6.7% |
| Recent min temperature | 3.5% |

The model's own recent temperature history dominates — consistent with the
persistence-baseline finding above, from a completely independent analysis
method. Full breakdown: [`notebooks/10_large_scale_interpretability.ipynb`](notebooks/10_large_scale_interpretability.ipynb).

## Tech stack

TensorFlow / Keras · Keras Tuner · SHAP · scikit-learn · pandas · NumPy ·
Open-Meteo API · matplotlib / seaborn · Jupyter

## Project layout

```
data/raw_large/            Raw fetched dataset (kolkata_1980_2019.csv, committed)
data/processed_large/      Deterministic pipeline outputs (git-ignored — regenerate from notebooks)
notebooks/                 08, 09, 10 — this project's notebooks (numbered within the wider repo)
src/                       fetch_large_data.py, data_large.py — reusable pipeline code
reports/                   large_scale_extension.md — full methodology and results write-up
models/checkpoints_large/  Saved best-epoch models (git-ignored — regenerate by re-running notebooks)
```

### Notebooks (run in order)

| # | Notebook | Produces |
|---|---|---|
| 08 | `08_large_scale_eda.ipynb` | Continuity/quality checks, seasonal structure, sequence-window + walk-forward/holdout pipeline |
| 09 | `09_large_scale_lstm_training.ipynb` | Keras Tuner search, walk-forward CV evaluation, optimizer comparison, final holdout scoring, persistence-baseline comparison |
| 10 | `10_large_scale_interpretability.ipynb` | SHAP feature-importance analysis on the tuned model |

## Setup

TensorFlow does not currently ship wheels for Python 3.14, so this project
pins a dedicated **Python 3.10** virtual environment:

```bash
py -3.10 -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Run notebooks with the `.venv` kernel, in numeric order (08 → 09 → 10) — each
is a standalone, re-runnable record of one pipeline stage.

