"""Facebook Prophet forecasting model for network metrics.

Leverages Prophet's additive decomposition model to capture
multiple seasonalities, trend changes, and holiday effects in
network performance data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
import logging

from .base import BaseForecaster

# Suppress Prophet's verbose logging
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ProphetForecaster(BaseForecaster):
    """Prophet-based forecasting model.

    Handles daily and weekly seasonality in network metrics automatically.
    """

    def __init__(
        self,
        yearly_seasonality: bool = False,
        weekly_seasonality: bool = True,
        daily_seasonality: bool = True,
        changepoint_prior_scale: float = 0.05,
        forecast_horizon: int = 12,
        interval_width: float = 0.95,
        freq_minutes: int = 5,
    ):
        super().__init__(name="Prophet", forecast_horizon=forecast_horizon)
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.changepoint_prior_scale = changepoint_prior_scale
        self.interval_width = interval_width
        self.freq_minutes = freq_minutes
        self._model = None
        self._metric = None

    def fit(self, train_data: pd.DataFrame, metric: str = "throughput_mbps", **kwargs) -> None:
        """Fit Prophet model.

        Args:
            train_data: DataFrame with 'timestamp' and metric columns.
            metric: Target metric column name.
        """
        from prophet import Prophet

        self._metric = metric

        # Prophet requires 'ds' and 'y' columns
        prophet_df = pd.DataFrame({
            "ds": train_data["timestamp"],
            "y": train_data[metric].values,
        })

        self._model = Prophet(
            yearly_seasonality=self.yearly_seasonality,
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            changepoint_prior_scale=self.changepoint_prior_scale,
            interval_width=self.interval_width,
        )
        self._model.fit(prophet_df)
        self._is_fitted = True

    def predict(self, steps: Optional[int] = None, **kwargs) -> np.ndarray:
        """Generate point forecasts."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        steps = steps or self.forecast_horizon
        future = self._model.make_future_dataframe(
            periods=steps, freq=f"{self.freq_minutes}min", include_history=False
        )
        forecast = self._model.predict(future)
        return forecast["yhat"].values

    def predict_with_intervals(
        self, steps: Optional[int] = None, confidence: float = 0.95, **kwargs
    ) -> dict[str, np.ndarray]:
        """Generate forecasts with prediction intervals."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        steps = steps or self.forecast_horizon

        # Adjust interval width if different from default
        if abs(confidence - self.interval_width) > 0.01:
            self._model.interval_width = confidence

        future = self._model.make_future_dataframe(
            periods=steps, freq=f"{self.freq_minutes}min", include_history=False
        )
        forecast = self._model.predict(future)

        return {
            "forecast": forecast["yhat"].values,
            "lower": forecast["yhat_lower"].values,
            "upper": forecast["yhat_upper"].values,
        }

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "yearly_seasonality": self.yearly_seasonality,
            "weekly_seasonality": self.weekly_seasonality,
            "daily_seasonality": self.daily_seasonality,
            "changepoint_prior_scale": self.changepoint_prior_scale,
        })
        return params
