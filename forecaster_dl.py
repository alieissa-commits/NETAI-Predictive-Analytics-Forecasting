"""Lightweight LSTM-based latency forecasting for network telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SlidingWindowDataset(Dataset):
    """Convert a 1D time series into (window, next-step target) samples."""

    def __init__(self, values: np.ndarray, window_size: int) -> None:
        """Initialize a sliding window dataset.

        Args:
            values: 1D normalized time-series array.
            window_size: Number of timesteps per input sequence.
        """
        if window_size <= 0:
            raise ValueError("window_size must be positive.")
        if values.ndim != 1:
            raise ValueError("values must be a 1D numpy array.")
        self.values = torch.tensor(values, dtype=torch.float32)
        self.window_size = window_size

    def __len__(self) -> int:
        """Return total number of generated windows."""
        return max(len(self.values) - self.window_size, 0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Fetch one window and its next-step target.

        Args:
            idx: Window start index.

        Returns:
            Tuple of (features, target) tensors.
        """
        x = self.values[idx : idx + self.window_size].unsqueeze(-1)
        y = self.values[idx + self.window_size]
        return x, y


class LatencyLSTM(nn.Module):
    """Small LSTM regressor for next-step latency prediction."""

    def __init__(
        self,
        hidden_size: int = 32,
        num_layers: int = 1,
        dropout: float = 0.0,
    ) -> None:
        """Initialize the LSTM forecaster.

        Args:
            hidden_size: Number of hidden units in LSTM.
            num_layers: Number of recurrent layers.
            dropout: Dropout between recurrent layers (when num_layers > 1).
        """
        super().__init__()
        effective_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=effective_dropout,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass.

        Args:
            x: Input tensor with shape (batch, seq_len, 1).

        Returns:
            Predicted next value for each batch item.
        """
        sequence_out, _ = self.lstm(x)
        return self.head(sequence_out[:, -1, :]).squeeze(-1)


def load_latency_series(data_path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load and return telemetry and latency series.

    Args:
        data_path: CSV path containing telemetry.

    Returns:
        Tuple of sorted telemetry DataFrame and latency ndarray.
    """
    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "latency_ms" not in df.columns:
        raise ValueError("Input data must include 'latency_ms'.")
    return df, df["latency_ms"].to_numpy(dtype=np.float32)


def split_train_test(series: np.ndarray, horizon_steps: int) -> tuple[np.ndarray, np.ndarray]:
    """Split a series into training and holdout test segments.

    Args:
        series: Full latency series.
        horizon_steps: Number of tail points reserved for test.

    Returns:
        Tuple of (train, test) arrays.
    """
    if len(series) <= horizon_steps:
        raise ValueError("Series is shorter than requested horizon split.")
    return series[:-horizon_steps], series[-horizon_steps:]


def build_dataloader(
    values: np.ndarray,
    window_size: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a DataLoader from a 1D series.

    Args:
        values: Input series values.
        window_size: Window size for each sample.
        batch_size: Batch size for training/evaluation.
        shuffle: Whether to shuffle sample order.

    Returns:
        Torch DataLoader over sliding windows.
    """
    dataset = SlidingWindowDataset(values=values, window_size=window_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_lstm_model(
    model: LatencyLSTM,
    train_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> list[float]:
    """Train an LSTM model with MSE + Adam.

    Args:
        model: LSTM model.
        train_loader: Training DataLoader.
        epochs: Number of epochs.
        learning_rate: Adam learning rate.
        device: Torch device.

    Returns:
        Mean training loss per epoch.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()
    epoch_losses: list[float] = []

    for _ in range(epochs):
        batch_losses: list[float] = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            batch_losses.append(float(loss.item()))

        epoch_loss = float(np.mean(batch_losses)) if batch_losses else float("nan")
        epoch_losses.append(epoch_loss)

    return epoch_losses


def predict_batches(
    model: LatencyLSTM,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run batched inference for one-step forecasts.

    Args:
        model: Trained LSTM model.
        loader: Inference DataLoader.
        device: Torch device.

    Returns:
        Numpy array of predictions in normalized scale.
    """
    model.eval()
    preds: list[np.ndarray] = []

    with torch.no_grad():
        for x_batch, _ in loader:
            x_batch = x_batch.to(device)
            y_hat = model(x_batch).cpu().numpy()
            preds.append(y_hat)

    if not preds:
        return np.array([], dtype=np.float32)
    return np.concatenate(preds)


def forecast_recursive(
    model: LatencyLSTM,
    seed_window: np.ndarray,
    horizon_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Forecast multiple future steps recursively.

    Args:
        model: Trained LSTM model.
        seed_window: Last normalized window from observed history.
        horizon_steps: Number of future steps.
        device: Torch device.

    Returns:
        Forecast array in normalized scale.
    """
    model.eval()
    history = seed_window.astype(np.float32).copy()
    outputs: list[float] = []

    with torch.no_grad():
        for _ in range(horizon_steps):
            x = torch.tensor(history, dtype=torch.float32, device=device).view(1, -1, 1)
            pred = float(model(x).item())
            outputs.append(pred)
            history = np.append(history[1:], pred)

    return np.array(outputs, dtype=np.float32)


def evaluate_lstm_mae(
    model: LatencyLSTM,
    train_scaled: np.ndarray,
    test_scaled: np.ndarray,
    scaler: StandardScaler,
    window_size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[float, np.ndarray]:
    """Evaluate one-step-ahead holdout MAE.

    Args:
        model: Trained LSTM model.
        train_scaled: Normalized train values.
        test_scaled: Normalized test values.
        scaler: Fitted scaler for inverse transform.
        window_size: Input sequence length.
        batch_size: Inference batch size.
        device: Torch device.

    Returns:
        Tuple of (MAE in ms, denormalized predictions).
    """
    eval_series = np.concatenate([train_scaled[-window_size:], test_scaled])
    eval_loader = build_dataloader(
        values=eval_series,
        window_size=window_size,
        batch_size=batch_size,
        shuffle=False,
    )

    pred_scaled = predict_batches(model, eval_loader, device=device)
    pred_scaled = pred_scaled[: len(test_scaled)]

    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    actual = scaler.inverse_transform(test_scaled.reshape(-1, 1)).flatten()
    mae = mean_absolute_error(actual, pred)
    return float(mae), pred


def fit_model_on_series(
    series: np.ndarray,
    window_size: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[LatencyLSTM, StandardScaler, list[float]]:
    """Scale a series, train an LSTM, and return artifacts.

    Args:
        series: Raw latency series.
        window_size: Sequence length.
        batch_size: Training batch size.
        epochs: Number of epochs.
        learning_rate: Optimizer learning rate.
        device: Torch device.

    Returns:
        Tuple of (trained model, fitted scaler, epoch losses).
    """
    scaler = StandardScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

    train_loader = build_dataloader(
        values=scaled,
        window_size=window_size,
        batch_size=batch_size,
        shuffle=True,
    )

    model = LatencyLSTM(hidden_size=32, num_layers=1, dropout=0.0)
    losses = train_lstm_model(
        model=model,
        train_loader=train_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        device=device,
    )
    return model, scaler, losses


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for LSTM training and forecasting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/synthetic_network_telemetry.csv"),
        help="Path to input telemetry CSV.",
    )
    parser.add_argument(
        "--forecast-output",
        type=Path,
        default=Path("outputs/lstm_forecast_24h.csv"),
        help="Path to write 24h LSTM forecast CSV.",
    )
    parser.add_argument(
        "--backtest-output",
        type=Path,
        default=Path("outputs/lstm_backtest.csv"),
        help="Path to write backtest predictions CSV.",
    )
    parser.add_argument("--window-size", type=int, default=36, help="Sliding window length.")
    parser.add_argument("--epochs", type=int, default=12, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=288,
        help="Forecast horizon in 5-minute steps.",
    )
    return parser.parse_args()


def main() -> None:
    """Train/evaluate LSTM and emit both backtest and 24-hour forecast files."""
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    telemetry_df, latency_series = load_latency_series(args.data_path)
    train_series, test_series = split_train_test(latency_series, args.horizon_steps)

    train_scaler = StandardScaler()
    train_scaled = train_scaler.fit_transform(train_series.reshape(-1, 1)).flatten()
    test_scaled = train_scaler.transform(test_series.reshape(-1, 1)).flatten()

    train_loader = build_dataloader(
        values=train_scaled,
        window_size=args.window_size,
        batch_size=args.batch_size,
        shuffle=True,
    )

    model = LatencyLSTM(hidden_size=32, num_layers=1, dropout=0.0)
    losses = train_lstm_model(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    mae, pred_test = evaluate_lstm_mae(
        model=model,
        train_scaled=train_scaled,
        test_scaled=test_scaled,
        scaler=train_scaler,
        window_size=args.window_size,
        batch_size=args.batch_size,
        device=device,
    )

    backtest = pd.DataFrame(
        {
            "timestamp": telemetry_df["timestamp"].iloc[-args.horizon_steps :].values,
            "expected_latency_ms": test_series,
            "predicted_latency_ms": pred_test,
            "packet_loss_pct": telemetry_df["packet_loss_pct"].iloc[-args.horizon_steps :].values,
        }
    )

    final_model, final_scaler, _ = fit_model_on_series(
        series=latency_series,
        window_size=args.window_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=device,
    )

    full_scaled = final_scaler.transform(latency_series.reshape(-1, 1)).flatten()
    seed_window = full_scaled[-args.window_size :]
    pred_scaled = forecast_recursive(
        model=final_model,
        seed_window=seed_window,
        horizon_steps=args.horizon_steps,
        device=device,
    )
    pred_future = final_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()

    future_timestamps = pd.date_range(
        start=telemetry_df["timestamp"].iloc[-1] + pd.Timedelta(minutes=5),
        periods=args.horizon_steps,
        freq="5min",
    )
    forecast = pd.DataFrame(
        {
            "timestamp": future_timestamps,
            "predicted_latency_ms": pred_future,
        }
    )

    args.backtest_output.parent.mkdir(parents=True, exist_ok=True)
    args.forecast_output.parent.mkdir(parents=True, exist_ok=True)
    backtest.to_csv(args.backtest_output, index=False)
    forecast.to_csv(args.forecast_output, index=False)

    print(f"LSTM MAE on holdout: {mae:.3f} ms")
    print(f"Final training loss: {losses[-1]:.5f}")
    print(f"Backtest saved to {args.backtest_output}")
    print(f"24h forecast saved to {args.forecast_output}")


if __name__ == "__main__":
    main()
