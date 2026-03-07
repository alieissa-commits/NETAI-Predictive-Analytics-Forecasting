"""ARIMA/SARIMA forecasting model for network metrics.

Uses statsmodels SARIMAX to capture trend, seasonality, and
autocorrelation in network performance time-series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy import stats

from .base import BaseForecaster


class ARIMAForecaster(BaseForecaster):
    """ARIMA/SARIMA forecasting model.

    Supports automatic order selection or manual specification.
    Handles both non-seasonal ARIMA(p,d,q) and seasonal SARIMA(p,d,q)(P,D,Q,s).
    """

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 1, 2),
        seasonal_order: Optional[tuple[int, int, int, int]] = None,
        forecast_horizon: int = 12,
    ):
        super().__init__(name="ARIMA", forecast_horizon=forecast_horizon)
        self.order = order
        self.seasonal_order = seasonal_order or (0, 0, 0, 0)
        self._model = None
        self._result = None
        self._metric = None

    def fit(self, train_data: pd.DataFrame, metric: str = "throughput_mbps", **kwargs) -> None:
        """Fit SARIMA model on training data.

        Args:
            train_data: DataFrame with metric column.
            metric: Target metric column name.
        """
        self._metric = metric
        series = train_data[metric].values.astype(np.float64)

        self._model = SARIMAX(
            series,
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        self._result = self._model.fit(disp=False, maxiter=200)
        self._is_fitted = True

    def predict(self, steps: Optional[int] = None, **kwargs) -> np.ndarray:
        """Generate point forecasts."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        steps = steps or self.forecast_horizon
        forecast = self._result.forecast(steps=steps)
        return forecast

    def predict_with_intervals(
        self, steps: Optional[int] = None, confidence: float = 0.95, **kwargs
    ) -> dict[str, np.ndarray]:
        """Generate forecasts with prediction intervals."""
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before predicting.")

        steps = steps or self.forecast_horizon
        forecast_obj = self._result.get_forecast(steps=steps)
        forecast = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=1 - confidence)

        return {
            "forecast": forecast.values if hasattr(forecast, "values") else forecast,
            "lower": conf_int.iloc[:, 0].values if hasattr(conf_int, "iloc") else conf_int[:, 0],
            "upper": conf_int.iloc[:, 1].values if hasattr(conf_int, "iloc") else conf_int[:, 1],
        }

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "order": self.order,
            "seasonal_order": self.seasonal_order,
        })
        if self._result is not None:
            params["aic"] = self._result.aic
            params["bic"] = self._result.bic
        return params
