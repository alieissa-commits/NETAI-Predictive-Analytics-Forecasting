"""Evaluation metrics for time-series forecasting models.

Provides standard forecasting metrics and model comparison utilities
for benchmarking ARIMA, Prophet, LSTM, Transformer, and ensemble models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def compute_mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(actual - predicted)))


def compute_rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def compute_mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Mean Absolute Percentage Error.

    Handles zero values by adding small epsilon.
    """
    epsilon = 1e-8
    return float(np.mean(np.abs((actual - predicted) / (actual + epsilon))) * 100)


def compute_smape(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Symmetric Mean Absolute Percentage Error.

    More robust than MAPE when actual values are near zero.
    """
    denominator = np.abs(actual) + np.abs(predicted) + 1e-8
    return float(np.mean(2.0 * np.abs(actual - predicted) / denominator) * 100)


def compute_mase(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_actual: np.ndarray,
    seasonality: int = 1,
) -> float:
    """Mean Absolute Scaled Error.

    Scales error by the in-sample naive forecast error.
    """
    naive_errors = np.abs(train_actual[seasonality:] - train_actual[:-seasonality])
    scale = np.mean(naive_errors)
    if scale == 0:
        return float("inf")
    return float(np.mean(np.abs(actual - predicted)) / scale)


def evaluate_forecast(
    actual: np.ndarray,
    predicted: np.ndarray,
    train_actual: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute all evaluation metrics for a forecast.

    Args:
        actual: Ground truth values.
        predicted: Predicted values.
        train_actual: Training data (for MASE computation).

    Returns:
        Dictionary of metric names to values.
    """
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    metrics = {
        "mae": compute_mae(actual, predicted),
        "rmse": compute_rmse(actual, predicted),
        "mape": compute_mape(actual, predicted),
        "smape": compute_smape(actual, predicted),
    }

    if train_actual is not None and len(train_actual) > 1:
        metrics["mase"] = compute_mase(actual, predicted, train_actual)

    return metrics


def compare_models(
    actual: np.ndarray,
    predictions: dict[str, np.ndarray],
    train_actual: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compare multiple models using all evaluation metrics.

    Args:
        actual: Ground truth values.
        predictions: Dict mapping model names to prediction arrays.
        train_actual: Training data for MASE computation.

    Returns:
        DataFrame with models as rows and metrics as columns, sorted by RMSE.
    """
    results = []
    for model_name, pred in predictions.items():
        metrics = evaluate_forecast(actual, pred, train_actual)
        metrics["model"] = model_name
        results.append(metrics)

    df = pd.DataFrame(results).set_index("model")
    return df.sort_values("rmse")
