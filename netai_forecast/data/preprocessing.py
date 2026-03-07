"""Data preprocessing and feature engineering for time-series forecasting.

Provides normalization, sequence creation for deep learning models,
and feature engineering utilities for network metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Optional, Literal


def preprocess_timeseries(
    df: pd.DataFrame,
    metric_columns: Optional[list[str]] = None,
    scaler_type: Literal["minmax", "standard"] = "minmax",
    fill_method: str = "ffill",
) -> tuple[pd.DataFrame, dict[str, MinMaxScaler | StandardScaler]]:
    """Preprocess time-series data for model consumption.

    Args:
        df: Raw DataFrame with timestamp and metric columns.
        metric_columns: Columns to normalize. Defaults to all numeric columns.
        scaler_type: Normalization strategy.
        fill_method: Method to fill missing values.

    Returns:
        Tuple of (preprocessed DataFrame, dict of fitted scalers per column).
    """
    df = df.copy()

    if metric_columns is None:
        metric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        metric_columns = [c for c in metric_columns if c not in ("is_anomaly",)]

    # Fill missing values
    df[metric_columns] = df[metric_columns].ffill() if fill_method == "ffill" else df[metric_columns].fillna(method=fill_method)

    # Add time-based features
    if "timestamp" in df.columns:
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Add rolling statistics
    for col in metric_columns:
        df[f"{col}_rolling_mean_12"] = df[col].rolling(window=12, min_periods=1).mean()
        df[f"{col}_rolling_std_12"] = df[col].rolling(window=12, min_periods=1).std().fillna(0)
        df[f"{col}_diff_1"] = df[col].diff().fillna(0)

    # Normalize metric columns
    scalers = {}
    ScalerClass = MinMaxScaler if scaler_type == "minmax" else StandardScaler
    for col in metric_columns:
        scaler = ScalerClass()
        df[col] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler

    return df, scalers


def create_sequences(
    data: np.ndarray,
    sequence_length: int = 48,
    forecast_horizon: int = 12,
    target_col_idx: int = 0,
    stride: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Create input/output sequences for deep learning models.

    Args:
        data: 2D array of shape (n_samples, n_features).
        sequence_length: Number of past time steps for input.
        forecast_horizon: Number of future steps to predict.
        target_col_idx: Index of the target column in data.
        stride: Step size between consecutive sequences.

    Returns:
        Tuple of (X, y) where:
            X has shape (n_sequences, sequence_length, n_features)
            y has shape (n_sequences, forecast_horizon)
    """
    X, y = [], []
    total = len(data) - sequence_length - forecast_horizon + 1

    for i in range(0, total, stride):
        X.append(data[i : i + sequence_length])
        y.append(
            data[i + sequence_length : i + sequence_length + forecast_horizon, target_col_idx]
        )

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split time-series data chronologically (no shuffling).

    Args:
        df: Input DataFrame sorted by time.
        train_ratio: Fraction for training.
        val_ratio: Fraction for validation.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def inverse_transform(
    values: np.ndarray,
    scaler: MinMaxScaler | StandardScaler,
) -> np.ndarray:
    """Inverse-transform normalized values back to original scale."""
    values_2d = values.reshape(-1, 1)
    return scaler.inverse_transform(values_2d).flatten()
