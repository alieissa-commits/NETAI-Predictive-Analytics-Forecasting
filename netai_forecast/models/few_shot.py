"""Few-shot learning adapter for network forecasting models.

Enables rapid adaptation of pre-trained forecasting models to
new network topologies and measurement patterns using only a
small number of examples (support set).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from copy import deepcopy

from .base import BaseForecaster
from .lstm_model import LSTMForecaster, LSTMNetwork
from ..data.preprocessing import create_sequences


class FewShotAdapter:
    """Few-shot learning adapter using MAML-inspired fine-tuning.

    Takes a pre-trained deep learning model and rapidly adapts it
    to new network topologies using a small support set of measurements.

    This implements a simplified Model-Agnostic Meta-Learning (MAML)
    approach where:
    1. A base model is pre-trained on source topology data
    2. The adapter fine-tunes a copy with few gradient steps on support data
    3. The adapted model makes predictions for the target topology
    """

    def __init__(
        self,
        base_model: LSTMForecaster,
        support_set_size: int = 10,
        adaptation_steps: int = 5,
        adaptation_lr: float = 0.01,
    ):
        """Initialize the few-shot adapter.

        Args:
            base_model: A pre-trained LSTMForecaster (or compatible model).
            support_set_size: Number of examples for adaptation.
            adaptation_steps: Number of gradient steps for fine-tuning.
            adaptation_lr: Learning rate for adaptation.
        """
        if not base_model.is_fitted:
            raise ValueError("Base model must be pre-trained before adaptation.")

        self.base_model = base_model
        self.support_set_size = support_set_size
        self.adaptation_steps = adaptation_steps
        self.adaptation_lr = adaptation_lr
        self._adapted_model: Optional[LSTMNetwork] = None
        self._adapted = False

    def adapt(
        self,
        support_data: pd.DataFrame,
        metric: str = "throughput_mbps",
        feature_columns: Optional[list[str]] = None,
    ) -> None:
        """Adapt the base model to a new topology using support data.

        Args:
            support_data: Small DataFrame from the target topology.
            metric: Target metric column.
            feature_columns: Feature columns to use.
        """
        if feature_columns is None:
            feature_columns = self.base_model._feature_columns

        # Create support sequences
        support_arr = support_data[feature_columns].values.astype(np.float32)
        target_idx = feature_columns.index(metric)

        X_support, y_support = create_sequences(
            support_arr,
            self.base_model.sequence_length,
            self.base_model.forecast_horizon,
            target_idx,
        )

        if len(X_support) == 0:
            raise ValueError(
                "Support data too short to create sequences. "
                f"Need at least {self.base_model.sequence_length + self.base_model.forecast_horizon} points."
            )

        # Limit to support set size
        if len(X_support) > self.support_set_size:
            indices = np.random.choice(len(X_support), self.support_set_size, replace=False)
            X_support = X_support[indices]
            y_support = y_support[indices]

        # Deep copy the base model's network
        self._adapted_model = deepcopy(self.base_model._model)
        device = self.base_model.device

        # Fine-tune with few gradient steps
        optimizer = torch.optim.SGD(
            self._adapted_model.parameters(), lr=self.adaptation_lr
        )
        criterion = nn.MSELoss()

        X_tensor = torch.from_numpy(X_support).to(device)
        y_tensor = torch.from_numpy(y_support).to(device)

        self._adapted_model.train()
        for _ in range(self.adaptation_steps):
            optimizer.zero_grad()
            pred = self._adapted_model(X_tensor)
            loss = criterion(pred, y_tensor)
            loss.backward()
            optimizer.step()

        self._adapted_model.eval()
        self._last_sequence = support_arr[-self.base_model.sequence_length:]
        self._adapted = True

    def predict(self, steps: Optional[int] = None) -> np.ndarray:
        """Generate predictions using the adapted model."""
        if not self._adapted or self._adapted_model is None:
            raise RuntimeError("Model must be adapted before predicting.")

        steps = steps or self.base_model.forecast_horizon
        device = self.base_model.device

        with torch.no_grad():
            x = torch.from_numpy(self._last_sequence).unsqueeze(0).to(device)
            pred = self._adapted_model(x).cpu().numpy().flatten()

        return pred[:steps]

    def predict_with_intervals(
        self, steps: Optional[int] = None, confidence: float = 0.95, n_samples: int = 50
    ) -> dict[str, np.ndarray]:
        """Generate predictions with uncertainty via MC Dropout."""
        if not self._adapted or self._adapted_model is None:
            raise RuntimeError("Model must be adapted before predicting.")

        steps = steps or self.base_model.forecast_horizon
        device = self.base_model.device

        self._adapted_model.train()
        predictions = []
        with torch.no_grad():
            x = torch.from_numpy(self._last_sequence).unsqueeze(0).to(device)
            for _ in range(n_samples):
                pred = self._adapted_model(x).cpu().numpy().flatten()[:steps]
                predictions.append(pred)

        predictions = np.array(predictions)
        alpha = (1 - confidence) / 2

        self._adapted_model.eval()

        return {
            "forecast": predictions.mean(axis=0),
            "lower": np.quantile(predictions, alpha, axis=0),
            "upper": np.quantile(predictions, 1 - alpha, axis=0),
        }

    def evaluate_adaptation(
        self,
        query_data: pd.DataFrame,
        metric: str = "throughput_mbps",
    ) -> dict[str, float]:
        """Evaluate adaptation quality on query (test) data.

        Args:
            query_data: Test data from the target topology.
            metric: Target metric column.

        Returns:
            Dict with MAE and RMSE on query set.
        """
        steps = min(self.base_model.forecast_horizon, len(query_data))
        actual = query_data[metric].values[:steps]
        pred = self.predict(steps=steps)

        min_len = min(len(pred), len(actual))
        actual = actual[:min_len]
        pred = pred[:min_len]

        mae = np.mean(np.abs(pred - actual))
        rmse = np.sqrt(np.mean((pred - actual) ** 2))

        return {"mae": float(mae), "rmse": float(rmse)}
