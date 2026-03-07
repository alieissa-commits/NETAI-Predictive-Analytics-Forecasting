"""Tests for data generation and preprocessing."""

import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from netai_forecast.data.generator import (
    NetworkDataGenerator,
    NetworkProfile,
    AnomalyConfig,
    PROFILES,
)
from netai_forecast.data.preprocessing import (
    preprocess_timeseries,
    create_sequences,
    train_val_test_split,
    inverse_transform,
)
from netai_forecast.data.perfsonar_loader import PerfSONARLoader


class TestNetworkDataGenerator:
    """Test synthetic data generation."""

    def test_default_generation(self):
        gen = NetworkDataGenerator(num_days=7, seed=42)
        df = gen.generate()

        assert isinstance(df, pd.DataFrame)
        assert "timestamp" in df.columns
        assert "throughput_mbps" in df.columns
        assert "latency_ms" in df.columns
        assert "packet_loss_pct" in df.columns
        assert "retransmits" in df.columns
        assert "is_anomaly" in df.columns

    def test_data_shape(self):
        gen = NetworkDataGenerator(num_days=1, sampling_interval_minutes=5, seed=42)
        df = gen.generate()
        expected_points = (1 * 24 * 60) // 5
        assert len(df) == expected_points

    def test_all_profiles(self):
        for profile_name in PROFILES:
            gen = NetworkDataGenerator(profile=profile_name, num_days=1, seed=42)
            df = gen.generate()
            assert len(df) > 0
            assert df["throughput_mbps"].mean() > 0
            assert df["latency_ms"].mean() > 0

    def test_custom_profile(self):
        profile = NetworkProfile(
            name="custom",
            base_throughput_mbps=1000.0,
            base_latency_ms=10.0,
        )
        gen = NetworkDataGenerator(profile=profile, num_days=1, seed=42)
        df = gen.generate()
        assert len(df) > 0

    def test_anomaly_injection(self):
        config = AnomalyConfig(probability=0.1, severity_range=(3.0, 5.0))
        gen = NetworkDataGenerator(num_days=7, anomaly_config=config, seed=42)
        df = gen.generate()
        assert df["is_anomaly"].sum() > 0

    def test_no_anomalies(self):
        config = AnomalyConfig(probability=0.0)
        gen = NetworkDataGenerator(num_days=7, anomaly_config=config, seed=42)
        df = gen.generate()
        assert df["is_anomaly"].sum() == 0

    def test_physical_bounds(self):
        gen = NetworkDataGenerator(num_days=7, seed=42)
        df = gen.generate()
        assert df["latency_ms"].min() >= 0.1
        assert df["packet_loss_pct"].min() >= 0.0
        assert df["packet_loss_pct"].max() <= 100.0
        assert df["retransmits"].min() >= 0

    def test_multi_link_generation(self):
        gen = NetworkDataGenerator(num_days=1, seed=42)
        result = gen.generate_multi_link()
        assert isinstance(result, dict)
        assert len(result) > 0
        for name, df in result.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0

    def test_reproducibility(self):
        gen1 = NetworkDataGenerator(num_days=1, seed=42)
        gen2 = NetworkDataGenerator(num_days=1, seed=42)
        df1 = gen1.generate()
        df2 = gen2.generate()
        pd.testing.assert_frame_equal(df1, df2)

    def test_invalid_profile(self):
        with pytest.raises(ValueError, match="Unknown profile"):
            NetworkDataGenerator(profile="nonexistent")


class TestPreprocessing:
    """Test data preprocessing."""

    @pytest.fixture
    def sample_data(self):
        gen = NetworkDataGenerator(num_days=7, seed=42)
        return gen.generate()

    def test_preprocess_basic(self, sample_data):
        df, scalers = preprocess_timeseries(sample_data)
        assert isinstance(df, pd.DataFrame)
        assert isinstance(scalers, dict)
        assert "throughput_mbps" in scalers

    def test_preprocess_adds_features(self, sample_data):
        df, _ = preprocess_timeseries(sample_data)
        assert "hour_sin" in df.columns
        assert "hour_cos" in df.columns
        assert "dow_sin" in df.columns
        assert "throughput_mbps_rolling_mean_12" in df.columns
        assert "throughput_mbps_diff_1" in df.columns

    def test_preprocess_normalization(self, sample_data):
        df, scalers = preprocess_timeseries(sample_data, scaler_type="minmax")
        for col in ["throughput_mbps", "latency_ms", "packet_loss_pct"]:
            assert df[col].min() >= -0.01  # Small tolerance
            assert df[col].max() <= 1.01

    def test_create_sequences(self):
        data = np.random.randn(200, 3).astype(np.float32)
        X, y = create_sequences(data, sequence_length=10, forecast_horizon=5, target_col_idx=0)
        assert X.shape[1] == 10  # sequence length
        assert X.shape[2] == 3   # features
        assert y.shape[1] == 5   # forecast horizon
        assert len(X) == len(y)

    def test_create_sequences_stride(self):
        data = np.random.randn(100, 2).astype(np.float32)
        X1, y1 = create_sequences(data, 10, 5, stride=1)
        X2, y2 = create_sequences(data, 10, 5, stride=2)
        assert len(X2) < len(X1)

    def test_train_val_test_split(self, sample_data):
        train, val, test = train_val_test_split(sample_data, 0.7, 0.15)
        assert len(train) + len(val) + len(test) == len(sample_data)
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_inverse_transform(self, sample_data):
        _, scalers = preprocess_timeseries(sample_data)
        scaler = scalers["throughput_mbps"]
        original = sample_data["throughput_mbps"].values[:5]
        normalized = scaler.transform(original.reshape(-1, 1)).flatten()
        recovered = inverse_transform(normalized, scaler)
        np.testing.assert_allclose(original, recovered, rtol=1e-5)


class TestPerfSONARLoader:
    """Test data loading utilities."""

    def test_csv_roundtrip(self):
        gen = NetworkDataGenerator(num_days=1, seed=42)
        df = gen.generate()

        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df.to_csv(f, index=False)
            tmp_path = f.name

        try:
            loader = PerfSONARLoader()
            loaded = loader.from_csv(tmp_path)
            assert len(loaded) == len(df)
            assert "timestamp" in loaded.columns
        finally:
            os.unlink(tmp_path)

    def test_validation_good_data(self):
        gen = NetworkDataGenerator(num_days=1, seed=42)
        df = gen.generate()
        loader = PerfSONARLoader()
        warnings = loader.validate(df)
        assert len(warnings) == 0

    def test_validation_missing_columns(self):
        df = pd.DataFrame({"timestamp": pd.date_range("2025-01-01", periods=10, freq="5min")})
        loader = PerfSONARLoader()
        warnings = loader.validate(df)
        assert any("Missing metric columns" in w for w in warnings)

    def test_sqlite_not_found(self):
        loader = PerfSONARLoader()
        with pytest.raises(FileNotFoundError):
            loader.from_sqlite("/nonexistent/path.db")
