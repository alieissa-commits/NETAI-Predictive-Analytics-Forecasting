"""LSTM-based deep learning forecasting model for network metrics.

Implements a multi-layer LSTM with optional attention mechanism
for sequence-to-sequence forecasting of network performance.
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
from ..data.preprocessing import create_sequences


class LSTMNetwork(nn.Module):
    """Multi-layer LSTM network for time-series forecasting."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        forecast_horizon: int = 12,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # Use last hidden state
        last_out = lstm_out[:, -1, :]
        last_out = self.dropout(last_out)
        return self.fc(last_out)


class LSTMForecaster(BaseForecaster):
    """LSTM-based forecasting model with training loop.

    Handles data preparation, training with early stopping,
    and multi-step forecasting with uncertainty estimates.
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 50,
        patience: int = 10,
        sequence_length: int = 48,
        forecast_horizon: int = 12,
        device: Optional[str] = None,
    ):
        super().__init__(name="LSTM", forecast_horizon=forecast_horizon)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.sequence_length = sequence_length

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[LSTMNetwork] = None
        self._feature_columns: list[str] = []
        self._target_col_idx: int = 0
        self._train_losses: list[float] = []
        self._val_losses: list[float] = []
        self._last_sequence: Optional[np.ndarray] = None

    def fit(
        self,
        train_data: pd.DataFrame,
        metric: str = "throughput_mbps",
        val_data: Optional[pd.DataFrame] = None,
        feature_columns: Optional[list[str]] = None,
        **kwargs,
    ) -> None:
        """Train the LSTM model.

        Args:
            train_data: Training DataFrame with numeric features.
            metric: Target metric column.
            val_data: Optional validation DataFrame for early stopping.
            feature_columns: Columns to use as features. If None, uses metric only.
        """
        if feature_columns is None:
            feature_columns = [metric]
        self._feature_columns = feature_columns
        self._target_col_idx = feature_columns.index(metric)

        # Prepare sequences
        train_arr = train_data[feature_columns].values.astype(np.float32)
        X_train, y_train = create_sequences(
            train_arr, self.sequence_length, self.forecast_horizon, self._target_col_idx
        )

        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Validation set
        val_loader = None
        if val_data is not None:
            val_arr = val_data[feature_columns].values.astype(np.float32)
            X_val, y_val = create_sequences(
                val_arr, self.sequence_length, self.forecast_horizon, self._target_col_idx
            )
            val_dataset = TensorDataset(
                torch.from_numpy(X_val), torch.from_numpy(y_val)
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Initialize model
        input_size = len(feature_columns)
        self._model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)

        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
            # Training
            self._model.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item() * X_batch.size(0)

            epoch_loss /= len(train_dataset)
            self._train_losses.append(epoch_loss)

            # Validation
            if val_loader is not None:
                val_loss = self._evaluate(val_loader, criterion)
                self._val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = deepcopy(self._model.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        break

        # Restore best model
        if best_model_state is not None:
            self._model.load_state_dict(best_model_state)

        # Store last sequence for prediction
        self._last_sequence = train_arr[-self.sequence_length:]
        self._is_fitted = True

    def predict(self, steps: Optional[int] = None, **kwargs) -> np.ndarray:
        """Generate point forecasts."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predicting.")

        steps = steps or self.forecast_horizon
        self._model.eval()

        with torch.no_grad():
            x = torch.from_numpy(self._last_sequence).unsqueeze(0).to(self.device)
            pred = self._model(x).cpu().numpy().flatten()

        return pred[:steps]

    def predict_with_intervals(
        self, steps: Optional[int] = None, confidence: float = 0.95, n_samples: int = 100, **kwargs
    ) -> dict[str, np.ndarray]:
        """Generate forecasts with uncertainty via MC Dropout.

        Uses Monte Carlo dropout to estimate prediction uncertainty.
        """
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predicting.")

        steps = steps or self.forecast_horizon
        self._model.train()  # Enable dropout for MC sampling

        predictions = []
        with torch.no_grad():
            x = torch.from_numpy(self._last_sequence).unsqueeze(0).to(self.device)
            for _ in range(n_samples):
                pred = self._model(x).cpu().numpy().flatten()[:steps]
                predictions.append(pred)

        predictions = np.array(predictions)
        alpha = (1 - confidence) / 2

        self._model.eval()

        return {
            "forecast": predictions.mean(axis=0),
            "lower": np.quantile(predictions, alpha, axis=0),
            "upper": np.quantile(predictions, 1 - alpha, axis=0),
        }

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> float:
        """Evaluate model on a data loader."""
        self._model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                pred = self._model(X_batch)
                loss = criterion(pred, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                total_samples += X_batch.size(0)

        return total_loss / total_samples

    def get_params(self) -> dict:
        params = super().get_params()
        params.update({
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "sequence_length": self.sequence_length,
            "epochs_trained": len(self._train_losses),
            "device": self.device,
        })
        if self._train_losses:
            params["final_train_loss"] = self._train_losses[-1]
        if self._val_losses:
            params["best_val_loss"] = min(self._val_losses)
        return params
