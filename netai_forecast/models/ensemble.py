"""Weighted ensemble forecasting model.

Combines predictions from ARIMA, Prophet, LSTM, and Transformer
models using configurable weights to produce robust forecasts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

from .base import BaseForecaster


class EnsembleForecaster(BaseForecaster):
    """Ensemble model combining multiple forecasting approaches.

    Supports static weights and dynamic weight optimization
    based on recent validation performance.
    """

    def __init__(
        self,
        models: Optional[list[BaseForecaster]] = None,
        weights: Optional[dict[str, float]] = None,
        forecast_horizon: int = 12,
    ):
        super().__init__(name="Ensemble", forecast_horizon=forecast_horizon)
        self.models: list[BaseForecaster] = models or []
        self._weights = weights or {}
        self._normalized_weights: dict[str, float] = {}

    def add_model(self, model: BaseForecaster, weight: float = 1.0) -> None:
        """Add a model to the ensemble.

        Args:
            model: A fitted or unfitted forecaster instance.
            weight: Relative weight for this model.
        """
        self.models.append(model)
        self._weights[model.name] = weight
        self._normalize_weights()

    def fit(
        self,
        train_data: pd.DataFrame,
        metric: str = "throughput_mbps",
        val_data: Optional[pd.DataFrame] = None,
        **kwargs,
    ) -> None:
        """Fit all sub-models that are not already fitted.

        Args:
            train_data: Training DataFrame.
            metric: Target metric column.
            val_data: Optional validation data for early stopping (DL models).
        """
        for model in self.models:
            if not model.is_fitted:
                fit_kwargs = {"metric": metric}
                if val_data is not None and hasattr(model, "fit"):
                    fit_kwargs["val_data"] = val_data
                model.fit(train_data, **fit_kwargs, **kwargs)

        self._normalize_weights()
        self._is_fitted = True

    def predict(self, steps: Optional[int] = None, **kwargs) -> np.ndarray:
        """Generate weighted ensemble forecasts."""
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before predicting.")

        steps = steps or self.forecast_horizon
        predictions = []
        weights = []

        for model in self.models:
            if model.is_fitted:
                pred = model.predict(steps=steps, **kwargs)
                predictions.append(pred)
                weights.append(self._normalized_weights.get(model.name, 1.0))

        if not predictions:
            raise RuntimeError("No fitted models available in ensemble.")

        predictions = np.array(predictions)
        weights = np.array(weights)
        weights /= weights.sum()

        return np.average(predictions, axis=0, weights=weights)

    def predict_with_intervals(
        self, steps: Optional[int] = None, confidence: float = 0.95, **kwargs
    ) -> dict[str, np.ndarray]:
        """Generate ensemble forecasts with prediction intervals.

        Combines individual model intervals using weighted averaging.
        """
        if not self._is_fitted:
            raise RuntimeError("Ensemble must be fitted before predicting.")

        steps = steps or self.forecast_horizon
        all_forecasts = []
        all_lowers = []
        all_uppers = []
        weights = []

        for model in self.models:
            if model.is_fitted:
                result = model.predict_with_intervals(steps=steps, confidence=confidence, **kwargs)
                all_forecasts.append(result["forecast"])
                all_lowers.append(result["lower"])
                all_uppers.append(result["upper"])
                weights.append(self._normalized_weights.get(model.name, 1.0))

        weights = np.array(weights)
        weights /= weights.sum()

        return {
            "forecast": np.average(np.array(all_forecasts), axis=0, weights=weights),
            "lower": np.average(np.array(all_lowers), axis=0, weights=weights),
            "upper": np.average(np.array(all_uppers), axis=0, weights=weights),
        }

    def optimize_weights(
        self,
        val_data: pd.DataFrame,
        metric: str = "throughput_mbps",
        steps: int = 12,
    ) -> dict[str, float]:
        """Optimize ensemble weights based on validation performance.

        Uses inverse-MSE weighting: models with lower error get higher weight.

        Args:
            val_data: Validation DataFrame.
            metric: Target metric column.
            steps: Forecast horizon for evaluation.

        Returns:
            Optimized weight dictionary.
        """
        actual = val_data[metric].values[:steps]
        mse_scores = {}

        for model in self.models:
            if model.is_fitted:
                pred = model.predict(steps=steps)
                min_len = min(len(pred), len(actual))
                mse = np.mean((pred[:min_len] - actual[:min_len]) ** 2)
                mse_scores[model.name] = mse

        if not mse_scores:
            return self._weights

        # Inverse MSE weighting
        inv_mse = {name: 1.0 / (mse + 1e-8) for name, mse in mse_scores.items()}
        total = sum(inv_mse.values())
        self._weights = {name: w / total for name, w in inv_mse.items()}
        self._normalize_weights()

        return self._weights

    def _normalize_weights(self) -> None:
        """Normalize weights to sum to 1."""
        total = sum(self._weights.get(m.name, 1.0) for m in self.models)
        if total > 0:
            self._normalized_weights = {
                m.name: self._weights.get(m.name, 1.0) / total for m in self.models
            }

    def get_params(self) -> dict:
        params = super().get_params()
        params["models"] = [m.name for m in self.models]
        params["weights"] = self._normalized_weights
        return params
