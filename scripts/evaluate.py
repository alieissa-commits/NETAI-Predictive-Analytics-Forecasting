"""Evaluation script for NETAI forecasting models.

Runs trained models against test data and generates comparison reports.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netai_forecast.data.generator import NetworkDataGenerator
from netai_forecast.data.preprocessing import train_val_test_split
from netai_forecast.models.arima_model import ARIMAForecaster
from netai_forecast.models.lstm_model import LSTMForecaster
from netai_forecast.models.transformer_model import TransformerForecaster
from netai_forecast.models.ensemble import EnsembleForecaster
from netai_forecast.evaluation.metrics import evaluate_forecast, compare_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
logger = logging.getLogger("netai.evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate NETAI forecasting models")
    parser.add_argument("--metric", type=str, default="throughput_mbps")
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--profile", type=str, default="wan_research")
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Generate and split data
    gen = NetworkDataGenerator(profile=args.profile, num_days=args.days, seed=args.seed)
    data = gen.generate()
    train, val, test = train_val_test_split(data)

    logger.info(f"Data: {len(data)} points | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    # Train lightweight models for evaluation
    models = {}

    logger.info("Training ARIMA...")
    arima = ARIMAForecaster(order=(2, 1, 2), forecast_horizon=args.horizon)
    arima.fit(train, metric=args.metric)
    models["arima"] = arima

    logger.info("Training LSTM...")
    lstm = LSTMForecaster(
        hidden_size=64, num_layers=1, epochs=10, sequence_length=48,
        forecast_horizon=args.horizon, batch_size=32, patience=5,
    )
    lstm.fit(train, metric=args.metric, val_data=val, feature_columns=[args.metric])
    models["lstm"] = lstm

    logger.info("Training Transformer...")
    transformer = TransformerForecaster(
        d_model=32, nhead=2, num_encoder_layers=2, epochs=10,
        sequence_length=48, forecast_horizon=args.horizon, batch_size=32, patience=5,
    )
    transformer.fit(train, metric=args.metric, val_data=val, feature_columns=[args.metric])
    models["transformer"] = transformer

    # Ensemble
    ensemble = EnsembleForecaster(forecast_horizon=args.horizon)
    for m in models.values():
        ensemble.add_model(m)
    ensemble._is_fitted = True
    ensemble.optimize_weights(val, metric=args.metric, steps=args.horizon)
    models["ensemble"] = ensemble

    # Evaluate
    actual = test[args.metric].values[:args.horizon]
    predictions = {name: m.predict(steps=args.horizon) for name, m in models.items()}
    comparison = compare_models(actual, predictions, train[args.metric].values)

    print("\n" + "=" * 70)
    print(f"  MODEL COMPARISON — {args.metric} — {args.horizon}-step forecast")
    print("=" * 70)
    print(comparison.to_string())
    print()


if __name__ == "__main__":
    main()
