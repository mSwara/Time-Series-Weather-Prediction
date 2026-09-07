# Large-Scale Extension: Validating Resume Claims with Real Data

## 1. Why this extension exists

Earlier resume drafts referenced "14,000+ daily weather records" and "walk-forward
cross-validation and holdout testing," but the project this was based on used the
original 300-row dataset with a single chronological split — those specific claims
weren't true of the built pipeline. Rather than leave them as unfilled template
placeholders or quietly drop them, this extension builds a real, reproducible
14,610-day dataset and pipeline that the claims can be checked against honestly.

## 2. What's genuinely different here vs. the 300-row project

| | 300-row project | This extension |
|---|---|---|
| Data source | Curated ERA5 extract (100km-radius, cosine-lat-weighted), internship-provided | Open-Meteo historical archive (ERA5/ERA5-Land-backed), fetched directly, no account needed |
| Sample size | 300 rows (299 after cleaning) | 14,610 continuous days (1980-2019) |
| Continuity | 71 disjoint date "bursts," large gaps | Strictly continuous, every gap = 1 day (verified by assertion) |
| Predictors | 500hPa specific humidity, geopotential height, wind; explicit sensible/latent heat flux; Urban Footprint | 2m temp max/min, 2m humidity, 10m wind, cloud cover, shortwave radiation, surface pressure, ET0, precipitation, vapour pressure deficit, sunshine duration, cyclical day-of-year — **a different, documented feature set** (see §3) |
| Sequence design | Precomputed lag-temperature (4 steps) + current-day static features broadcast across timesteps (a proxy, necessitated by burst fragmentation) | True multivariate historical windows: 7 real prior days, each with its own full 15-feature vector — no broadcasting needed |
| Split strategy | Single chronological, burst-aware train/val/test split | Real walk-forward (expanding-window) CV, 5 folds, plus a separate untouched final holdout (last 4 years) |
| Architecture | CNN-LSTM hybrid + attention (~3,186 params) | Plain LSTM (matching the resume's actual wording), params reported from `model.count_params()` — see §4 |

## 3. Why a different feature set — an honest substitution, not a silent one

The original dataset's `specific_humidity_500hPa`, `geopotential_height_500hPa`,
`Wind_500hPa_ms`, and explicit `surface_sensible_heat_flux`/`surface_latent_heat_flux`
come from ERA5's pressure-level product via the Copernicus Climate Data Store (CDS),
which requires a registered account and API key. Rather than set up credentials on
the user's behalf, this extension uses Open-Meteo's free, no-key-required historical
archive, which provides a different (though thematically related) set of daily
surface variables. This is disclosed explicitly here and in `src/fetch_large_data.py`,
not hidden — a reader comparing the two projects' predictor lists will see they
don't match column-for-column, and this section is why.

## 4. Real numbers

| Metric | Value |
|---|---|
| Daily records | 14,610 (1980-01-01 to 2019-12-31) |
| Sequence windows (7-day lookback) | 14,603 |
| Train+val windows (for walk-forward CV) | 13,142 |
| Held-out test windows (2016-2019, touched once) | 1,461 |
| Trainable parameters (tuned LSTM: units=64, dense=64, dropout=0.2, l2=1e-4) | **24,705** |
| Test R2, default (untuned) architecture | **0.9657** (RMSE 0.753, MAE 0.569 C) |
| Test R2, tuned architecture (Adam) | **0.9681** (RMSE 0.725, MAE 0.555 C) |
| Walk-forward CV R2, tuned architecture (5 folds) | **mean 0.969, std 0.001** |
| Test R2 by optimizer | Adam **0.968** / SGD 0.965 / RMSprop 0.965 |
| Train R2 (final model, overfitting check) | 0.971 (vs. test 0.968 — gap of 0.003, effectively no overfitting) |

**Honest read of these numbers**: R2 is far higher here (~0.97) than in the
300-row project (~0.50-0.71) — expected, not a red flag. Predicting tomorrow's
mean temperature from a full week of highly autocorrelated recent history
(yesterday's own temperature is included in that window) is a substantially
easier problem than the original project's setup. The walk-forward CV std of
0.001 across 5 folds is very low, indicating this result is stable, not a
lucky single split. Tuning's improvement over default (0.9657 -> 0.9681) is
real but modest — because the default architecture was already near this
problem's achievable ceiling, there was less room for tuning to matter, unlike
the 300-row project where tuning closed a much larger gap. Both are honest,
consistent findings, not contradictory ones.

## 5. Validated resume bullets

> Designed a pipeline for 14,610 daily weather records using walk-forward
> cross-validation and holdout testing, preventing look-ahead bias

> Applied an LSTM model in TensorFlow (~25K params) to predict DBT using
> Dropout, L2, EarlyStopping, and ModelCheckpoint

> Boosted test R2 score from 0.966 to 0.968 by tuning hyperparameters with
> Keras Tuner and comparing Adam, SGD, and RMSprop optimizers

Every number above is real, measured, and reproducible by re-running
`notebooks/08_large_scale_eda.ipynb` then `09_large_scale_lstm_training.ipynb`
in order — not filled in from a template. Full machine-readable summary:
`data/processed_large/large_scale_summary.json`.

## 6. What's NOT claimed

This extension does not claim to reproduce the original internship's exact
predictor set (see §3), does not claim the two projects' R² numbers are
directly comparable (different data source, different feature set, different
architecture, different time period), and does not retroactively change what
the 300-row project's own honest findings were (`reports/final_comparison.md`
stands as-is, describing that separate piece of work).
