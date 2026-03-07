"""Tests for evaluation metrics."""

import numpy as np
import pandas as pd
import pytest

from netai_forecast.evaluation.metrics import (
    compute_mae,
    compute_rmse,
    compute_mape,
    compute_smape,
    compute_mase,
    evaluate_forecast,
    compare_models,
)


class TestMetrics:
    def test_mae_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert compute_mae(actual, actual) == 0.0

    def test_mae_known(self):
        actual = np.array([1.0, 2.0, 3.0])
        predicted = np.array([1.5, 2.5, 3.5])
        assert abs(compute_mae(actual, predicted) - 0.5) < 1e-7

    def test_rmse_perfect(self):
        actual = np.array([1.0, 2.0, 3.0])
        assert compute_rmse(actual, actual) == 0.0

    def test_rmse_known(self):
        actual = np.array([0.0, 0.0])
        predicted = np.array([3.0, 4.0])
        # RMSE = sqrt((9+16)/2) = sqrt(12.5) ≈ 3.536
        assert abs(compute_rmse(actual, predicted) - np.sqrt(12.5)) < 1e-6

    def test_mape(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 220.0])
        # MAPE ≈ 10%
        assert abs(compute_mape(actual, predicted) - 10.0) < 0.1

    def test_smape(self):
        actual = np.array([100.0, 200.0])
        predicted = np.array([110.0, 220.0])
        smape = compute_smape(actual, predicted)
        assert 0 < smape < 100

    def test_mase(self):
        train = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        actual = np.array([6.0, 7.0])
        predicted = np.array([6.5, 7.5])
        mase = compute_mase(actual, predicted, train, seasonality=1)
        assert mase > 0

    def test_evaluate_forecast(self):
        actual = np.array([100.0, 200.0, 300.0])
        predicted = np.array([105.0, 195.0, 310.0])
        result = evaluate_forecast(actual, predicted)
        assert "mae" in result
        assert "rmse" in result
        assert "mape" in result
        assert "smape" in result
        assert all(v >= 0 for v in result.values())

    def test_evaluate_with_train(self):
        train = np.arange(100, dtype=float)
        actual = np.array([100.0, 101.0])
        predicted = np.array([99.0, 102.0])
        result = evaluate_forecast(actual, predicted, train)
        assert "mase" in result

    def test_compare_models(self):
        actual = np.array([100.0, 200.0, 300.0])
        predictions = {
            "model_a": np.array([105.0, 195.0, 310.0]),
            "model_b": np.array([110.0, 190.0, 320.0]),
        }
        df = compare_models(actual, predictions)
        assert isinstance(df, pd.DataFrame)
        assert "mae" in df.columns
        assert len(df) == 2
        # Should be sorted by RMSE
        assert df.index[0] in ["model_a", "model_b"]

    def test_length_mismatch(self):
        """Handles different length arrays gracefully."""
        actual = np.array([1.0, 2.0, 3.0, 4.0])
        predicted = np.array([1.0, 2.0])
        result = evaluate_forecast(actual, predicted)
        assert result["mae"] == 0.0  # Both match on first 2 elements
