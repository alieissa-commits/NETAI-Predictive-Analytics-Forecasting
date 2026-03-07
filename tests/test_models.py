"""Tests for forecasting models."""

import numpy as np
import pandas as pd
import pytest

from netai_forecast.data.generator import NetworkDataGenerator
from netai_forecast.data.preprocessing import train_val_test_split
from netai_forecast.models.arima_model import ARIMAForecaster
from netai_forecast.models.lstm_model import LSTMForecaster
from netai_forecast.models.transformer_model import TransformerForecaster
from netai_forecast.models.ensemble import EnsembleForecaster
from netai_forecast.models.few_shot import FewShotAdapter


# Shared fixtures
@pytest.fixture(scope="module")
def sample_data():
    """Generate sample data once for all tests in module."""
    gen = NetworkDataGenerator(profile="wan_research", num_days=14, seed=42)
    return gen.generate()


@pytest.fixture(scope="module")
def split_data(sample_data):
    train, val, test = train_val_test_split(sample_data, 0.7, 0.15)
    return train, val, test


METRIC = "throughput_mbps"
HORIZON = 12


class TestARIMAForecaster:
    def test_fit_predict(self, split_data):
        train, val, test = split_data
        model = ARIMAForecaster(order=(1, 1, 1), forecast_horizon=HORIZON)
        model.fit(train, metric=METRIC)

        assert model.is_fitted
        pred = model.predict(steps=HORIZON)
        assert len(pred) == HORIZON
        assert not np.any(np.isnan(pred))

    def test_predict_with_intervals(self, split_data):
        train, val, test = split_data
        model = ARIMAForecaster(order=(1, 1, 1), forecast_horizon=HORIZON)
        model.fit(train, metric=METRIC)

        result = model.predict_with_intervals(steps=HORIZON)
        assert "forecast" in result
        assert "lower" in result
        assert "upper" in result
        assert len(result["forecast"]) == HORIZON
        # Lower bound should be <= forecast <= upper bound
        assert np.all(result["lower"] <= result["forecast"] + 1e-6)
        assert np.all(result["forecast"] <= result["upper"] + 1e-6)

    def test_not_fitted_error(self):
        model = ARIMAForecaster()
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict()

    def test_get_params(self, split_data):
        train, _, _ = split_data
        model = ARIMAForecaster(order=(2, 1, 2))
        model.fit(train, metric=METRIC)
        params = model.get_params()
        assert params["order"] == (2, 1, 2)
        assert "aic" in params


class TestLSTMForecaster:
    def test_fit_predict(self, split_data):
        train, val, test = split_data
        model = LSTMForecaster(
            hidden_size=32, num_layers=1, epochs=3,
            sequence_length=48, forecast_horizon=HORIZON, batch_size=32,
        )
        model.fit(train, metric=METRIC, val_data=val, feature_columns=[METRIC])

        assert model.is_fitted
        pred = model.predict(steps=HORIZON)
        assert len(pred) == HORIZON
        assert not np.any(np.isnan(pred))

    def test_predict_with_intervals(self, split_data):
        train, val, _ = split_data
        model = LSTMForecaster(
            hidden_size=32, num_layers=1, epochs=3,
            sequence_length=48, forecast_horizon=HORIZON, batch_size=32,
        )
        model.fit(train, metric=METRIC, val_data=val, feature_columns=[METRIC])

        result = model.predict_with_intervals(steps=HORIZON, n_samples=20)
        assert "forecast" in result
        assert "lower" in result
        assert "upper" in result
        assert len(result["forecast"]) == HORIZON

    def test_training_history(self, split_data):
        train, val, _ = split_data
        model = LSTMForecaster(
            hidden_size=32, num_layers=1, epochs=3,
            sequence_length=48, forecast_horizon=HORIZON, batch_size=32,
        )
        model.fit(train, metric=METRIC, val_data=val, feature_columns=[METRIC])

        assert len(model._train_losses) > 0
        assert len(model._val_losses) > 0

    def test_get_params(self, split_data):
        train, val, _ = split_data
        model = LSTMForecaster(hidden_size=64, num_layers=2, epochs=2,
                               sequence_length=48, forecast_horizon=HORIZON)
        model.fit(train, metric=METRIC, feature_columns=[METRIC])
        params = model.get_params()
        assert params["hidden_size"] == 64
        assert params["num_layers"] == 2


class TestTransformerForecaster:
    def test_fit_predict(self, split_data):
        train, val, _ = split_data
        model = TransformerForecaster(
            d_model=16, nhead=2, num_encoder_layers=1, epochs=3,
            sequence_length=48, forecast_horizon=HORIZON, batch_size=32,
        )
        model.fit(train, metric=METRIC, val_data=val, feature_columns=[METRIC])

        assert model.is_fitted
        pred = model.predict(steps=HORIZON)
        assert len(pred) == HORIZON
        assert not np.any(np.isnan(pred))

    def test_predict_with_intervals(self, split_data):
        train, val, _ = split_data
        model = TransformerForecaster(
            d_model=16, nhead=2, num_encoder_layers=1, epochs=3,
            sequence_length=48, forecast_horizon=HORIZON, batch_size=32,
        )
        model.fit(train, metric=METRIC, val_data=val, feature_columns=[METRIC])

        result = model.predict_with_intervals(steps=HORIZON, n_samples=20)
        assert len(result["forecast"]) == HORIZON


class TestEnsembleForecaster:
    def test_ensemble_basic(self, split_data):
        train, val, test = split_data

        arima = ARIMAForecaster(order=(1, 1, 1), forecast_horizon=HORIZON)
        arima.fit(train, metric=METRIC)

        lstm = LSTMForecaster(
            hidden_size=32, num_layers=1, epochs=2,
            sequence_length=48, forecast_horizon=HORIZON, batch_size=32,
        )
        lstm.fit(train, metric=METRIC, feature_columns=[METRIC])

        ensemble = EnsembleForecaster(forecast_horizon=HORIZON)
        ensemble.add_model(arima, weight=0.4)
        ensemble.add_model(lstm, weight=0.6)
        ensemble._is_fitted = True

        pred = ensemble.predict(steps=HORIZON)
        assert len(pred) == HORIZON
        assert not np.any(np.isnan(pred))

    def test_ensemble_with_intervals(self, split_data):
        train, val, _ = split_data

        arima = ARIMAForecaster(order=(1, 1, 1), forecast_horizon=HORIZON)
        arima.fit(train, metric=METRIC)

        ensemble = EnsembleForecaster(forecast_horizon=HORIZON)
        ensemble.add_model(arima)
        ensemble._is_fitted = True

        result = ensemble.predict_with_intervals(steps=HORIZON)
        assert "forecast" in result
        assert len(result["forecast"]) == HORIZON

    def test_optimize_weights(self, split_data):
        train, val, _ = split_data

        arima = ARIMAForecaster(order=(1, 1, 1), forecast_horizon=HORIZON)
        arima.fit(train, metric=METRIC)

        lstm = LSTMForecaster(
            hidden_size=32, num_layers=1, epochs=2,
            sequence_length=48, forecast_horizon=HORIZON, batch_size=32,
        )
        lstm.fit(train, metric=METRIC, feature_columns=[METRIC])

        ensemble = EnsembleForecaster(forecast_horizon=HORIZON)
        ensemble.add_model(arima)
        ensemble.add_model(lstm)
        ensemble._is_fitted = True

        weights = ensemble.optimize_weights(val, metric=METRIC)
        assert abs(sum(weights.values()) - 1.0) < 1e-6


class TestFewShotAdapter:
    def test_adaptation(self, split_data):
        train, val, test = split_data

        # Pre-train base model
        base = LSTMForecaster(
            hidden_size=32, num_layers=1, epochs=3,
            sequence_length=48, forecast_horizon=HORIZON, batch_size=32,
        )
        base.fit(train, metric=METRIC, feature_columns=[METRIC])

        # Generate "new topology" data
        gen = NetworkDataGenerator(profile="datacenter_10g", num_days=3, seed=99)
        new_data = gen.generate()

        # Adapt
        adapter = FewShotAdapter(
            base_model=base, support_set_size=10,
            adaptation_steps=3, adaptation_lr=0.01,
        )
        adapter.adapt(new_data, metric=METRIC, feature_columns=[METRIC])

        assert adapter._adapted
        pred = adapter.predict(steps=HORIZON)
        assert len(pred) == HORIZON

    def test_few_shot_intervals(self, split_data):
        train, _, _ = split_data

        base = LSTMForecaster(
            hidden_size=32, num_layers=1, epochs=2,
            sequence_length=48, forecast_horizon=HORIZON, batch_size=32,
        )
        base.fit(train, metric=METRIC, feature_columns=[METRIC])

        gen = NetworkDataGenerator(profile="campus_1g", num_days=3, seed=77)
        new_data = gen.generate()

        adapter = FewShotAdapter(base_model=base, support_set_size=10)
        adapter.adapt(new_data, metric=METRIC, feature_columns=[METRIC])

        result = adapter.predict_with_intervals(steps=HORIZON, n_samples=20)
        assert "forecast" in result
        assert len(result["forecast"]) == HORIZON

    def test_unfitted_base_error(self):
        base = LSTMForecaster()
        with pytest.raises(ValueError, match="pre-trained"):
            FewShotAdapter(base_model=base)
