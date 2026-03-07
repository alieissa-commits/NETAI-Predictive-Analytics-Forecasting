"""Transformer-based forecasting model for network metrics.

Implements a Transformer encoder architecture for time-series
forecasting, using positional encoding and multi-head attention
to capture long-range dependencies in network performance data.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional
from copy import deepcopy

from .base import BaseForecaster
from ..data.preprocessing import create_sequences


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerForecastNet(nn.Module):
    """Transformer encoder network for time-series forecasting."""

    def __init__(
        self,
        input_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        forecast_horizon: int = 12,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

        self.output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, forecast_horizon),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)
        return self.output_head(x)


class TransformerForecaster(BaseForecaster):
    """Transformer-based time-series forecasting model.

    Uses a Transformer encoder with positional encoding to capture
    complex temporal patterns in network performance metrics.
    """

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        learning_rate: float = 0.0005,
        batch_size: int = 64,
        epochs: int = 50,
        patience: int = 10,
        sequence_length: int = 48,
        forecast_horizon: int = 12,
        device: Optional[str] = None,
    ):
        super().__init__(name="Transformer", forecast_horizon=forecast_horizon)
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.sequence_length = sequence_length

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model: Optional[TransformerForecastNet] = None
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
        """Train the Transformer model.

        Args:
            train_data: Training DataFrame.
            metric: Target metric column.
            val_data: Optional validation DataFrame.
            feature_columns: Feature columns. Defaults to [metric].
        """
        if feature_columns is None:
            feature_columns = [metric]
        self._feature_columns = feature_columns
        self._target_col_idx = feature_columns.index(metric)

        train_arr = train_data[feature_columns].values.astype(np.float32)
        X_train, y_train = create_sequences(
            train_arr, self.sequence_length, self.forecast_horizon, self._target_col_idx
        )

        train_dataset = TensorDataset(
            torch.from_numpy(X_train), torch.from_numpy(y_train)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

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

        input_size = len(feature_columns)
        self._model = TransformerForecastNet(
            input_size=input_size,
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.num_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            forecast_horizon=self.forecast_horizon,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self.learning_rate, weight_decay=1e-5
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )
        criterion = nn.MSELoss()
        best_val_loss = float("inf")
        best_model_state = None
        patience_counter = 0

        for epoch in range(self.epochs):
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
            scheduler.step()

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

        if best_model_state is not None:
            self._model.load_state_dict(best_model_state)

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
        """Generate forecasts with uncertainty via MC Dropout."""
        if not self._is_fitted or self._model is None:
            raise RuntimeError("Model must be fitted before predicting.")

        steps = steps or self.forecast_horizon
        self._model.train()  # Enable dropout

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
            "d_model": self.d_model,
            "nhead": self.nhead,
            "num_encoder_layers": self.num_encoder_layers,
            "dim_feedforward": self.dim_feedforward,
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
