"""Base class for all forecasting models."""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Optional, Any


class BaseForecaster(ABC):
    """Abstract base class for time-series forecasting models.

    All forecasting models implement this interface to ensure
    consistent usage across ARIMA, Prophet, LSTM, Transformer,
    and ensemble approaches.
    """

    def __init__(self, name: str, forecast_horizon: int = 12):
        self.name = name
        self.forecast_horizon = forecast_horizon
        self._is_fitted = False

    @abstractmethod
    def fit(self, train_data: pd.DataFrame, metric: str = "throughput_mbps", **kwargs) -> None:
        """Fit the model on training data.

        Args:
            train_data: DataFrame with at least 'timestamp' and metric columns.
            metric: The target metric column name.
        """
        ...

    @abstractmethod
    def predict(self, steps: Optional[int] = None, **kwargs) -> np.ndarray:
        """Generate forecasts for future time steps.

        Args:
            steps: Number of steps to forecast. Defaults to self.forecast_horizon.

        Returns:
            Array of predicted values.
        """
        ...

    @abstractmethod
    def predict_with_intervals(
        self, steps: Optional[int] = None, confidence: float = 0.95, **kwargs
    ) -> dict[str, np.ndarray]:
        """Generate forecasts with prediction intervals.

        Args:
            steps: Number of steps to forecast.
            confidence: Confidence level for prediction intervals.

        Returns:
            Dict with keys: 'forecast', 'lower', 'upper'.
        """
        ...

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted

    def get_params(self) -> dict[str, Any]:
        """Return model parameters as a dictionary."""
        return {"name": self.name, "forecast_horizon": self.forecast_horizon}
