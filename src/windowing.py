"""
Build the per-row sequence tensor fed to the CNN-LSTM.

Why this shape, given the data (see notebooks/01_eda.ipynb for the evidence):

- The only genuine ordered temporal signal available for *every* row is the
  precomputed lag-temperature block T(i-4), T(i-3), T(i-2), T(i-1) -> current day.
  Re-deriving raw multi-day windows from the burst structure instead would
  discard 47-83% of the 300 rows depending on window length (see EDA notebook),
  which is not affordable at this sample size.
- The raw table does not carry each lag day's full meteorological vector (only
  its temperature), so we cannot honestly claim a "multivariate historical
  sequence" in the way a fully time-indexed dataset would allow.
- Instead, each sample is represented as a (4 timesteps x C channels) tensor:
  channel 0 is the ordered lag-temperature sequence (the one quantity that
  genuinely varies across the 4 timesteps); the remaining channels are the
  current day's static meteorological features, broadcast identically across
  all 4 timesteps. This is a documented, honest proxy for a "temporal image":
  it lets a CNN learn local joint patterns across (recent-temperature-trend x
  same-day atmospheric state), and an LSTM/CNN-LSTM model how that combination
  evolves across the 4 lag steps — a genuinely different inductive bias than
  handing Ridge/LightGBM the same 12 numbers as unordered flat columns, even
  though the raw information content is the same. It is presented here as
  exactly that: a proxy, not a claim of true multi-day multivariate history.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data import LAG_COLS, STATIC_FEATURE_COLS, TARGET_COLS

SEQUENCE_STATIC_COLS = STATIC_FEATURE_COLS + ["Hr"]
N_TIMESTEPS = len(LAG_COLS)  # 4
N_CHANNELS = 1 + len(SEQUENCE_STATIC_COLS)  # lag-temperature channel + static channels


@dataclass
class SequenceScalers:
    """Scalers fit on the training split ONLY, then applied everywhere else."""
    lag_scaler: StandardScaler = field(default_factory=StandardScaler)
    static_scaler: StandardScaler = field(default_factory=StandardScaler)
    target_scaler: StandardScaler = field(default_factory=StandardScaler)

    def fit(self, train_df: pd.DataFrame) -> "SequenceScalers":
        # Pool all 4 lag columns into one scaler: they are the same physical
        # quantity (temperature) at different lags, so they should share one
        # loc/scale rather than getting four artificially different ones.
        pooled_lags = train_df[LAG_COLS].to_numpy().reshape(-1, 1)
        self.lag_scaler.fit(pooled_lags)
        self.static_scaler.fit(train_df[SEQUENCE_STATIC_COLS].to_numpy())
        self.target_scaler.fit(train_df[TARGET_COLS].to_numpy())
        return self


def build_sequence_tensor(df: pd.DataFrame, scalers: SequenceScalers) -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y): X shape (n, 4, C) scaled sequence tensor, y shape (n, 2) scaled targets.

    `scalers` must already be fit on the training split (see SequenceScalers.fit).
    """
    n = len(df)

    lag_raw = df[LAG_COLS].to_numpy()  # (n, 4), chronological order T(i-4)..T(i-1)
    lag_scaled = scalers.lag_scaler.transform(lag_raw.reshape(-1, 1)).reshape(n, N_TIMESTEPS, 1)

    static_raw = df[SEQUENCE_STATIC_COLS].to_numpy()  # (n, S)
    static_scaled = scalers.static_scaler.transform(static_raw)  # (n, S)
    static_broadcast = np.repeat(static_scaled[:, np.newaxis, :], N_TIMESTEPS, axis=1)  # (n, 4, S)

    X = np.concatenate([lag_scaled, static_broadcast], axis=2)  # (n, 4, 1+S)

    y_raw = df[TARGET_COLS].to_numpy()
    y = scalers.target_scaler.transform(y_raw)

    return X.astype("float32"), y.astype("float32")


def inverse_transform_targets(y_scaled: np.ndarray, scalers: SequenceScalers) -> np.ndarray:
    return scalers.target_scaler.inverse_transform(y_scaled)
