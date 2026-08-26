"""
Data loading, burst detection, leakage checks, and chronological splitting for the
Kolkata DBT/WBT dataset.

Central fact this module encodes (established in notebooks/01_eda.ipynb): the 300
rows are NOT a continuous daily series. Sorting by Date reveals 71 disjoint "bursts"
of consecutive calendar days separated by gaps (sometimes years). The lag features
T(i-1)..T(i-4) are only valid *within* a burst (verified: T(i-1) of row n equals dbt
of row n-1 whenever both sit in the same burst). Any split or windowing scheme must
therefore operate on whole bursts, never cut a burst in half — otherwise a lag
feature in one split would be derived from a target value sitting in another split.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

RAW_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "FINAL_DATASET_allpara.xlsx"

LAG_COLS = ["T(i-4)", "T(i-3)", "T(i-2)", "T(i-1)"]
STATIC_FEATURE_COLS = [
    "specific_humidity_500hPa",
    "geopotential_height_500hPa",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "Wind_500hPa_ms",
    "OLR_Wm2",
    "TCC",
    "Urban_Footprint",
]
TARGET_COLS = ["dbt", "wbt"]


def load_raw(path: Path | str = RAW_PATH) -> pd.DataFrame:
    """Load the raw dataset, sorted chronologically, with a clean RangeIndex.

    A single missing T(i-1) value is known to exist at the start of a burst (the
    dataset's own first lag point with no predecessor) and is left as NaN here —
    callers decide how to handle it (row 0 of the whole table has no lag available
    either way, since nothing precedes it).
    """
    df = pd.read_excel(path)
    df = df.sort_values("Date", kind="mergesort").reset_index(drop=True)
    return df


def detect_bursts(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a `burst_id` to each row: consecutive calendar days share an id.

    A new burst starts whenever the gap to the previous row's Date is not exactly
    1 day (this includes the very first row, and any row following a multi-day or
    multi-year gap).
    """
    df = df.copy()
    day_gap = df["Date"].diff().dt.days
    new_burst = (day_gap != 1).fillna(True)
    df["burst_id"] = new_burst.cumsum()
    return df


def burst_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per burst: id, start/end date, length, row index range."""
    if "burst_id" not in df.columns:
        df = detect_bursts(df)
    g = df.groupby("burst_id")
    summary = g.agg(
        start_date=("Date", "min"),
        end_date=("Date", "max"),
        n_rows=("Date", "size"),
        first_row=("Date", lambda s: s.index.min()),
        last_row=("Date", lambda s: s.index.max()),
    ).reset_index()
    return summary


def check_lag_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Verify T(i-1) equals the previous row's dbt *within the same burst*.

    Returns the (hopefully empty) DataFrame of mismatching rows. A non-empty
    result means the lag features cannot be trusted as-is and must be
    investigated before any modeling.
    """
    if "burst_id" not in df.columns:
        df = detect_bursts(df)
    df = df.copy()
    df["prev_dbt"] = df.groupby("burst_id")["dbt"].shift(1)
    within_burst = df.dropna(subset=["prev_dbt"])
    mismatches = within_burst[np.abs(within_burst["T(i-1)"] - within_burst["prev_dbt"]) > 1e-6]
    return mismatches


def block_chronological_split(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split that cuts only at burst boundaries.

    Rationale: a plain row-count chronological split (e.g. "first 70% of rows")
    could slice a burst in half, letting a test-set row's T(i-1)..T(i-4) be
    derived from a dbt value that sits in the training set (or vice versa) —
    exactly the kind of leakage the internship guarded against for its flat
    lag features, adapted here for burst/window boundaries. Instead, whole
    bursts (already in chronological order) are accumulated into train, then
    val, then test, stopping as close to the target fractions as possible
    without splitting a burst.

    The internship used a random 80:20 split; this is intentionally stricter.
    """
    if not 0 < train_frac < 1 or not 0 < val_frac < 1 or train_frac + val_frac >= 1:
        raise ValueError("train_frac and val_frac must be in (0,1) and sum to < 1")

    df = detect_bursts(df) if "burst_id" not in df.columns else df.copy()
    bsum = burst_summary(df).sort_values("start_date").reset_index(drop=True)

    n_total = len(df)
    n_train_target = round(n_total * train_frac)
    n_val_target = round(n_total * val_frac)

    train_ids, val_ids, test_ids = [], [], []
    running = 0
    for _, row in bsum.iterrows():
        if running < n_train_target:
            train_ids.append(row.burst_id)
        elif running < n_train_target + n_val_target:
            val_ids.append(row.burst_id)
        else:
            test_ids.append(row.burst_id)
        running += row.n_rows

    train_df = df[df.burst_id.isin(train_ids)].reset_index(drop=True)
    val_df = df[df.burst_id.isin(val_ids)].reset_index(drop=True)
    test_df = df[df.burst_id.isin(test_ids)].reset_index(drop=True)

    # Hard invariant: splits must be chronologically ordered and non-overlapping,
    # and no burst id may appear in more than one split.
    assert train_df["Date"].max() < val_df["Date"].min()
    assert val_df["Date"].max() < test_df["Date"].min()
    assert set(train_ids) & set(val_ids) == set()
    assert set(val_ids) & set(test_ids) == set()
    assert set(train_ids) & set(test_ids) == set()

    return train_df, val_df, test_df
