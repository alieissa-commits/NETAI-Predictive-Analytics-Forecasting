"""Baseline latency forecasting with Prophet for network telemetry."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from pandas import DatetimeTZDtype
from prophet import Prophet
from sklearn.metrics import mean_absolute_error


def load_telemetry(path: Path) -> pd.DataFrame:
    """Load telemetry data from CSV and enforce timestamp ordering.

    Args:
        path: Path to telemetry CSV.

    Returns:
        Timestamp-sorted telemetry DataFrame.
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if isinstance(df["timestamp"].dtype, DatetimeTZDtype):
        df["timestamp"] = df["timestamp"].dt.tz_localize(None)
    if "latency_ms" not in df.columns:
        raise ValueError("Input data must include a 'latency_ms' column.")
    return df.sort_values("timestamp").reset_index(drop=True)


def prepare_prophet_frame(df: pd.DataFrame, target_col: str = "latency_ms") -> pd.DataFrame:
    """Create Prophet-compatible frame with ds/y columns.

    Args:
        df: Source telemetry DataFrame.
        target_col: Column to forecast.

    Returns:
        DataFrame containing ds and y columns.
    """
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in dataframe.")

    return df[["timestamp", target_col]].rename(
        columns={"timestamp": "ds", target_col: "y"}
    )


def train_prophet_model(train_df: pd.DataFrame) -> Prophet:
    """Fit a Prophet model with daily and weekly seasonality.

    Args:
        train_df: Prophet-formatted training DataFrame.

    Returns:
        Trained Prophet model.
    """
    model = Prophet(
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.08,
    )
    model.fit(train_df)
    return model


def evaluate_prophet_mae(
    telemetry_df: pd.DataFrame,
    horizon_steps: int = 288,
    freq: str = "5min",
) -> tuple[float, pd.DataFrame]:
    """Backtest Prophet on the tail split and calculate MAE.

    Args:
        telemetry_df: Raw telemetry DataFrame.
        horizon_steps: Forecast horizon in rows for the test split.
        freq: Sampling frequency.

    Returns:
        A tuple of (MAE value, evaluation DataFrame).
    """
    prophet_df = prepare_prophet_frame(telemetry_df)
    if len(prophet_df) <= horizon_steps:
        raise ValueError("Not enough rows for requested horizon split.")

    train_df = prophet_df.iloc[:-horizon_steps].copy()
    test_df = prophet_df.iloc[-horizon_steps:].copy()
    test_context = telemetry_df.iloc[-horizon_steps:].copy().reset_index(drop=True)

    model = train_prophet_model(train_df)
    future = model.make_future_dataframe(periods=horizon_steps, freq=freq, include_history=False)
    forecast = model.predict(future)

    compare = pd.DataFrame(
        {
            "timestamp": test_df["ds"].values,
            "expected_latency_ms": test_df["y"].values,
            "predicted_latency_ms": forecast["yhat"].values,
        }
    )
    if "packet_loss_pct" in test_context.columns:
        compare["packet_loss_pct"] = test_context["packet_loss_pct"].values

    mae = mean_absolute_error(compare["expected_latency_ms"], compare["predicted_latency_ms"])
    return float(mae), compare


def forecast_next_24h(
    telemetry_df: pd.DataFrame,
    horizon_steps: int = 288,
    freq: str = "5min",
) -> pd.DataFrame:
    """Train on full telemetry history and forecast the next 24 hours.

    Args:
        telemetry_df: Raw telemetry DataFrame.
        horizon_steps: Number of future points to forecast.
        freq: Sampling frequency.

    Returns:
        DataFrame with future timestamp and predicted latency.
    """
    prophet_df = prepare_prophet_frame(telemetry_df)
    model = train_prophet_model(prophet_df)
    future = model.make_future_dataframe(periods=horizon_steps, freq=freq, include_history=False)
    forecast = model.predict(future)

    return forecast[["ds", "yhat"]].rename(
        columns={"ds": "timestamp", "yhat": "predicted_latency_ms"}
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for Prophet forecasting."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/synthetic_network_telemetry.csv"),
        help="Path to synthetic telemetry CSV.",
    )
    parser.add_argument(
        "--forecast-output",
        type=Path,
        default=Path("outputs/prophet_forecast_24h.csv"),
        help="Path to write 24h future forecast CSV.",
    )
    parser.add_argument(
        "--backtest-output",
        type=Path,
        default=Path("outputs/prophet_backtest.csv"),
        help="Path to write backtest predictions CSV.",
    )
    parser.add_argument(
        "--horizon-steps",
        type=int,
        default=288,
        help="Forecast horizon in 5-minute steps.",
    )
    return parser.parse_args()


def main() -> None:
    """Execute Prophet backtest and 24-hour forecast workflows."""
    args = parse_args()
    telemetry_df = load_telemetry(args.data_path)

    mae, backtest_df = evaluate_prophet_mae(
        telemetry_df=telemetry_df,
        horizon_steps=args.horizon_steps,
    )
    forecast_df = forecast_next_24h(
        telemetry_df=telemetry_df,
        horizon_steps=args.horizon_steps,
    )

    args.backtest_output.parent.mkdir(parents=True, exist_ok=True)
    args.forecast_output.parent.mkdir(parents=True, exist_ok=True)
    backtest_df.to_csv(args.backtest_output, index=False)
    forecast_df.to_csv(args.forecast_output, index=False)

    print(f"Prophet MAE on holdout: {mae:.3f} ms")
    print(f"Backtest saved to {args.backtest_output}")
    print(f"24h forecast saved to {args.forecast_output}")


if __name__ == "__main__":
    main()
