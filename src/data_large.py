"""Data loading, feature engineering, and walk-forward CV splitting for the
large-scale (14,610-day) Kolkata weather dataset.

Unlike the original 300-row dataset (71 disjoint date "bursts"), this series
is strictly continuous — every consecutive row is exactly 1 day apart,
verified in src/fetch_large_data.py and re-verified in notebook 08. That
means, unlike the small dataset, true historical lag features and true
multi-day sequence windows are both fully valid here without needing the
lag-block proxy the small-dataset project used.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw_large" / "kolkata_1980_2019.csv"

TARGET_COLS = ["dbt", "wbt"]
STATIC_FEATURE_COLS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "wind_speed_10m_max",
    "cloud_cover_mean",
    "shortwave_radiation_sum",
    "surface_pressure_mean",
    "et0_fao_evapotranspiration",
    "precipitation_sum",
    "vapour_pressure_deficit_max",
    "sunshine_duration",
]
N_LAGS = 4  # kept consistent with the original small-dataset project for comparability
LAG_COLS = [f"dbt_lag{i}" for i in range(1, N_LAGS + 1)]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    gaps = df["Date"].diff().dt.days.dropna()
    assert (gaps == 1).all(), "Series is not strictly continuous - do not proceed without investigating"
    return df


def add_lag_features(df: pd.DataFrame, n_lags: int = N_LAGS) -> pd.DataFrame:
    """True historical lags (real previous-day DBT values), valid here because
    the series is genuinely continuous - unlike the small dataset, no burst
    boundaries to worry about. The first `n_lags` rows get NaN lags (no
    history available) and are dropped by the caller."""
    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f"dbt_lag{i}"] = df["dbt"].shift(i)
    return df


def add_seasonal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Day-of-year encoded cyclically (sin/cos) - a real seasonal signal
    available here that the small, non-contiguous dataset couldn't reliably
    use (its bursts were scattered arbitrarily across the calendar year)."""
    df = df.copy()
    doy = df["Date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


FLAT_FEATURE_COLS = STATIC_FEATURE_COLS + ["doy_sin", "doy_cos"] + LAG_COLS


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_lag_features(df)
    df = add_seasonal_features(df)
    df = df.dropna(subset=LAG_COLS).reset_index(drop=True)
    return df


def walk_forward_folds(df: pd.DataFrame, n_folds: int = 5, min_train_years: int = 10):
    """Expanding-window walk-forward CV: fold k trains on all data before its
    validation window and validates on the next block forward in time. Used
    for hyperparameter selection ONLY, on the pre-holdout portion of the data
    (see holdout_split) - never touches the final test set.

    Yields (train_df, val_df) pairs, chronologically ordered, non-overlapping
    validation windows, each strictly after its own training data in time.
    """
    years = sorted(df["Date"].dt.year.unique())
    usable_years = years[min_train_years:]
    if len(usable_years) < n_folds:
        raise ValueError(f"Not enough years ({len(usable_years)}) for {n_folds} folds")

    fold_years = np.array_split(usable_years, n_folds)
    for fold in fold_years:
        val_start, val_end = fold[0], fold[-1]
        train_df = df[df["Date"].dt.year < val_start]
        val_df = df[(df["Date"].dt.year >= val_start) & (df["Date"].dt.year <= val_end)]
        if len(train_df) == 0 or len(val_df) == 0:
            continue
        yield train_df.reset_index(drop=True), val_df.reset_index(drop=True)


SEQUENCE_FEATURE_COLS = ["dbt", "wbt"] + STATIC_FEATURE_COLS + ["doy_sin", "doy_cos"]


def build_sequence_windows(df: pd.DataFrame, k: int = 7):
    """True multivariate historical sequence windows: each sample is the
    PAST k days' full feature vectors (day t-k .. t-1), predicting day t's
    dbt. This is a genuine upgrade over the small (300-row) dataset's design,
    which could only broadcast the *current* day's static features across a
    lag-temperature axis because the data was too fragmented (71 disjoint
    bursts) to support real multi-day multivariate windows. Here, with a
    strictly continuous 14,610-day series, every window is real history, not
    a proxy.

    Requires `add_seasonal_features` already applied (for doy_sin/doy_cos).
    Uses only same-day-or-earlier information for every feature (dbt/wbt in
    the window are days t-k..t-1's actual observed values, never day t's -
    day t's dbt is the prediction target and is not present in the input).

    Returns (X, y, dates): X shape (n, k, len(SEQUENCE_FEATURE_COLS)),
    y shape (n,) (dbt at day t), dates: the target day for each window
    (for chronological slicing downstream).
    """
    feat = df[SEQUENCE_FEATURE_COLS].to_numpy(dtype="float32")
    dbt = df["dbt"].to_numpy(dtype="float32")
    dates = df["Date"].to_numpy()

    n = len(df)
    X, y, out_dates = [], [], []
    for t in range(k, n):
        X.append(feat[t - k:t])  # days t-k .. t-1
        y.append(dbt[t])         # day t (target, not in the window)
        out_dates.append(dates[t])

    return np.stack(X), np.array(y, dtype="float32"), np.array(out_dates)


def holdout_split(df: pd.DataFrame, test_years: int = 4):
    """Carve off the final `test_years` of data as a true, untouched holdout
    test set. Everything before it is available for walk-forward CV /
    training. This mirrors the resume bullet's 'holdout testing' claim with
    an actual held-out block, evaluated exactly once at the end."""
    max_year = df["Date"].dt.year.max()
    cutoff_year = max_year - test_years + 1
    trainval_df = df[df["Date"].dt.year < cutoff_year].reset_index(drop=True)
    test_df = df[df["Date"].dt.year >= cutoff_year].reset_index(drop=True)
    return trainval_df, test_df


def holdout_split_arrays(X: np.ndarray, y: np.ndarray, dates: np.ndarray, test_years: int = 4):
    """Same cutoff logic as holdout_split, applied to the (X, y, dates)
    arrays build_sequence_windows returns, keyed by each window's target
    date."""
    years = pd.DatetimeIndex(dates).year
    max_year = years.max()
    cutoff_year = max_year - test_years + 1
    train_mask = years < cutoff_year
    test_mask = ~train_mask
    return (X[train_mask], y[train_mask], dates[train_mask]), (X[test_mask], y[test_mask], dates[test_mask])


def walk_forward_folds_arrays(X: np.ndarray, y: np.ndarray, dates: np.ndarray,
                               n_folds: int = 5, min_train_years: int = 10):
    """Array version of walk_forward_folds, keyed by each window's target date."""
    years_arr = pd.DatetimeIndex(dates).year
    unique_years = sorted(years_arr.unique())
    usable_years = unique_years[min_train_years:]
    if len(usable_years) < n_folds:
        raise ValueError(f"Not enough years ({len(usable_years)}) for {n_folds} folds")

    fold_years = np.array_split(usable_years, n_folds)
    for fold in fold_years:
        val_start, val_end = fold[0], fold[-1]
        train_mask = years_arr < val_start
        val_mask = (years_arr >= val_start) & (years_arr <= val_end)
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            continue
        yield (X[train_mask], y[train_mask]), (X[val_mask], y[val_mask])
