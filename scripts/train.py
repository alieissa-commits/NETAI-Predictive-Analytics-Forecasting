"""Training script for NETAI forecasting models.

Trains all forecasting models on synthetic or real perfSONAR data,
evaluates performance, and saves model artifacts.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from netai_forecast.data.generator import NetworkDataGenerator
from netai_forecast.data.preprocessing import preprocess_timeseries, train_val_test_split
from netai_forecast.models.arima_model import ARIMAForecaster
from netai_forecast.models.prophet_model import ProphetForecaster
from netai_forecast.models.lstm_model import LSTMForecaster
from netai_forecast.models.transformer_model import TransformerForecaster
from netai_forecast.models.ensemble import EnsembleForecaster
from netai_forecast.evaluation.metrics import evaluate_forecast, compare_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("netai.train")


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="Train NETAI forecasting models")
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument("--metric", type=str, default="throughput_mbps", help="Target metric")
    parser.add_argument("--days", type=int, default=30, help="Days of synthetic data")
    parser.add_argument("--profile", type=str, default="wan_research", help="Network profile")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--models", nargs="+", default=["arima", "prophet", "lstm", "transformer"],
        help="Models to train",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    config = load_config(args.config)
    metric = args.metric

    # ── Generate data ──────────────────────────────────────────────
    logger.info(f"Generating {args.days} days of synthetic data (profile={args.profile})")
    gen = NetworkDataGenerator(profile=args.profile, num_days=args.days, seed=args.seed)
    data = gen.generate()
    logger.info(f"Generated {len(data)} data points")

    # Split
    train_data, val_data, test_data = train_val_test_split(data)
    logger.info(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    # ── Train models ───────────────────────────────────────────────
    models = {}
    forecast_horizon = config["preprocessing"]["forecast_horizon"]

    if "arima" in args.models:
        logger.info("Training ARIMA model...")
        t0 = time.time()
        arima_cfg = config["models"]["arima"]
        arima = ARIMAForecaster(
            order=tuple(arima_cfg["order"]),
            forecast_horizon=forecast_horizon,
        )
        arima.fit(train_data, metric=metric)
        models["arima"] = arima
        logger.info(f"ARIMA trained in {time.time()-t0:.1f}s")

    if "prophet" in args.models:
        logger.info("Training Prophet model...")
        t0 = time.time()
        prophet_cfg = config["models"]["prophet"]
        prophet = ProphetForecaster(
            yearly_seasonality=prophet_cfg["yearly_seasonality"],
            weekly_seasonality=prophet_cfg["weekly_seasonality"],
            daily_seasonality=prophet_cfg["daily_seasonality"],
            changepoint_prior_scale=prophet_cfg["changepoint_prior_scale"],
            forecast_horizon=forecast_horizon,
        )
        prophet.fit(train_data, metric=metric)
        models["prophet"] = prophet
        logger.info(f"Prophet trained in {time.time()-t0:.1f}s")

    seq_len = config["preprocessing"]["sequence_length"]

    if "lstm" in args.models:
        logger.info("Training LSTM model...")
        t0 = time.time()
        lstm_cfg = config["models"]["lstm"]
        lstm = LSTMForecaster(
            hidden_size=lstm_cfg["hidden_size"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"],
            learning_rate=lstm_cfg["learning_rate"],
            batch_size=lstm_cfg["batch_size"],
            epochs=lstm_cfg["epochs"],
            patience=lstm_cfg["patience"],
            sequence_length=seq_len,
            forecast_horizon=forecast_horizon,
        )
        lstm.fit(train_data, metric=metric, val_data=val_data, feature_columns=[metric])
        models["lstm"] = lstm
        logger.info(f"LSTM trained in {time.time()-t0:.1f}s | {lstm.get_params().get('epochs_trained', '?')} epochs")

    if "transformer" in args.models:
        logger.info("Training Transformer model...")
        t0 = time.time()
        tf_cfg = config["models"]["transformer"]
        transformer = TransformerForecaster(
            d_model=tf_cfg["d_model"],
            nhead=tf_cfg["nhead"],
            num_encoder_layers=tf_cfg["num_encoder_layers"],
            dim_feedforward=tf_cfg["dim_feedforward"],
            dropout=tf_cfg["dropout"],
            learning_rate=tf_cfg["learning_rate"],
            batch_size=tf_cfg["batch_size"],
            epochs=tf_cfg["epochs"],
            patience=tf_cfg["patience"],
            sequence_length=seq_len,
            forecast_horizon=forecast_horizon,
        )
        transformer.fit(train_data, metric=metric, val_data=val_data, feature_columns=[metric])
        models["transformer"] = transformer
        logger.info(f"Transformer trained in {time.time()-t0:.1f}s | {transformer.get_params().get('epochs_trained', '?')} epochs")

    # ── Ensemble ──────────────────────────────────────────────────
    if len(models) >= 2:
        logger.info("Creating ensemble model...")
        ens_cfg = config["models"]["ensemble"]
        ensemble = EnsembleForecaster(forecast_horizon=forecast_horizon)
        for name, model in models.items():
            weight = ens_cfg["weights"].get(name, 1.0 / len(models))
            ensemble.add_model(model, weight)
        ensemble._is_fitted = True

        # Optimize weights
        ensemble.optimize_weights(val_data, metric=metric, steps=forecast_horizon)
        models["ensemble"] = ensemble
        logger.info(f"Ensemble weights: {ensemble._normalized_weights}")

    # ── Evaluate ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("MODEL EVALUATION RESULTS")
    logger.info("=" * 60)

    actual = test_data[metric].values[:forecast_horizon]
    predictions = {}
    for name, model in models.items():
        pred = model.predict(steps=forecast_horizon)
        predictions[name] = pred

    comparison = compare_models(actual, predictions, train_data[metric].values)
    logger.info(f"\n{comparison.to_string()}")

    # Save results
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(out_dir / "evaluation_results.csv")
        logger.info(f"Results saved to {out_dir}")

    logger.info("\nTraining complete!")


if __name__ == "__main__":
    main()
