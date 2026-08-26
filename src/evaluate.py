"""Shared evaluation metrics, kept identical to the internship's metric set
(R2, RMSE, MAE, KGE) so every model in this project is directly comparable to
the internship's Table 6 numbers, and to each other."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def kge(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Kling-Gupta Efficiency (Gupta et al., 2009 formulation).

    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
    where r = Pearson correlation, alpha = std(pred)/std(true) (variability
    ratio), beta = mean(pred)/mean(true) (bias ratio). KGE=1 is a perfect
    match; used by the internship alongside R2/RMSE/MAE specifically because
    it separates correlation, variability, and bias into one score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r = np.corrcoef(y_true, y_pred)[0, 1]
    alpha = y_pred.std() / y_true.std()
    beta = y_pred.mean() / y_true.mean()
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "KGE": kge(y_true, y_pred),
    }


def metrics_table(results: dict[str, dict[str, dict[str, float]]]) -> pd.DataFrame:
    """results: {model_name: {split_name: {target_name: metrics_dict}}} ->
    a tidy long-format DataFrame (model, split, target, R2, RMSE, MAE, KGE)."""
    rows = []
    for model_name, splits in results.items():
        for split_name, targets in splits.items():
            for target_name, m in targets.items():
                rows.append({"model": model_name, "split": split_name, "target": target_name, **m})
    return pd.DataFrame(rows)


# Internship's reported test-set numbers (report Table 6), for direct comparison.
INTERNSHIP_BENCHMARK = {
    "dbt": {"model": "Ridge", "R2": 0.6213, "RMSE": 1.2064, "MAE": 0.8792, "KGE": 0.7815},
    "wbt": {"model": "LightGBM", "R2": 0.7092, "RMSE": 0.7372, "MAE": 0.5442, "KGE": 0.7474},
}
