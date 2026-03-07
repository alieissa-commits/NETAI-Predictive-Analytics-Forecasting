"""Tests for early warning system."""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from netai_forecast.data.generator import NetworkDataGenerator
from netai_forecast.early_warning.detector import (
    DegradationDetector,
    DegradationEvent,
    DegradationType,
    AlertSeverity,
    ThresholdConfig,
)
from netai_forecast.early_warning.alerting import AlertManager, Alert


class TestDegradationDetector:
    @pytest.fixture
    def detector(self):
        return DegradationDetector()

    @pytest.fixture
    def baseline_data(self):
        gen = NetworkDataGenerator(num_days=7, seed=42)
        return gen.generate()

    def test_no_degradation(self, detector, baseline_data):
        """No alerts when forecast is normal."""
        recent = baseline_data.tail(288)
        mean_tp = recent["throughput_mbps"].mean()
        # Normal forecast near baseline
        normal_forecast = np.full(12, mean_tp)

        events = detector.detect(
            forecast=normal_forecast,
            recent_history=recent,
            metric="throughput_mbps",
        )
        # Should have few or no events for normal data
        critical_events = [e for e in events if e.severity == AlertSeverity.CRITICAL]
        assert len(critical_events) == 0

    def test_throughput_drop_detection(self, detector, baseline_data):
        """Detect throughput drops."""
        recent = baseline_data.tail(288)
        mean_tp = recent["throughput_mbps"].mean()
        # Severe drop: 70% below baseline
        degraded_forecast = np.full(12, mean_tp * 0.3)

        events = detector.detect(
            forecast=degraded_forecast,
            recent_history=recent,
            metric="throughput_mbps",
        )
        assert len(events) > 0
        assert any(e.event_type == DegradationType.THROUGHPUT_DROP for e in events)

    def test_latency_spike_detection(self, detector, baseline_data):
        """Detect latency spikes."""
        recent = baseline_data.tail(288)
        # Very high latency
        high_latency = np.full(12, 500.0)

        events = detector.detect(
            forecast=high_latency,
            recent_history=recent,
            metric="latency_ms",
        )
        assert len(events) > 0

    def test_trend_detection(self, detector, baseline_data):
        """Detect sustained degradation trends."""
        recent = baseline_data.tail(288)
        mean_latency = recent["latency_ms"].mean()
        std_latency = recent["latency_ms"].std()
        # Increasing latency trend
        trending_up = np.linspace(mean_latency, mean_latency + 5 * std_latency, 12)

        events = detector.detect(
            forecast=trending_up,
            recent_history=recent,
            metric="latency_ms",
        )
        trend_events = [e for e in events if e.event_type == DegradationType.TREND_DEGRADATION]
        assert len(trend_events) > 0

    def test_custom_thresholds(self, baseline_data):
        custom = {
            "throughput_mbps": ThresholdConfig(
                warning_threshold=0.05, critical_threshold=0.10, is_upper_bound=False
            ),
        }
        detector = DegradationDetector(thresholds=custom, confidence_threshold=0.5)
        recent = baseline_data.tail(288)
        mean_tp = recent["throughput_mbps"].mean()

        # Small drop should trigger with low thresholds
        small_drop = np.full(12, mean_tp * 0.85)
        events = detector.detect(
            forecast=small_drop,
            recent_history=recent,
            metric="throughput_mbps",
        )
        assert len(events) > 0


class TestAlertManager:
    def test_process_events(self):
        manager = AlertManager()
        event = DegradationEvent(
            event_type=DegradationType.THROUGHPUT_DROP,
            severity=AlertSeverity.WARNING,
            metric="throughput_mbps",
            current_value=3000.0,
            threshold_value=0.2,
            predicted_values=[3000.0] * 12,
            timestamp=datetime.utcnow(),
            confidence=0.9,
            description="Test warning",
        )

        alerts = manager.process_events([event])
        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.WARNING

    def test_deduplication(self):
        manager = AlertManager()
        event = DegradationEvent(
            event_type=DegradationType.LATENCY_SPIKE,
            severity=AlertSeverity.WARNING,
            metric="latency_ms",
            current_value=100.0,
            threshold_value=50.0,
            predicted_values=[100.0] * 12,
            timestamp=datetime.utcnow(),
            confidence=0.85,
            description="Latency warning",
        )

        alerts1 = manager.process_events([event])
        alerts2 = manager.process_events([event])
        # Second time should not create new alert (dedup)
        assert len(alerts2) == 0

    def test_severity_escalation(self):
        manager = AlertManager()

        warning = DegradationEvent(
            event_type=DegradationType.LATENCY_SPIKE,
            severity=AlertSeverity.WARNING,
            metric="latency_ms",
            current_value=100.0,
            threshold_value=50.0,
            predicted_values=[100.0] * 12,
            timestamp=datetime.utcnow(),
            confidence=0.85,
            description="Warning",
        )
        critical = DegradationEvent(
            event_type=DegradationType.LATENCY_SPIKE,
            severity=AlertSeverity.CRITICAL,
            metric="latency_ms",
            current_value=200.0,
            threshold_value=100.0,
            predicted_values=[200.0] * 12,
            timestamp=datetime.utcnow(),
            confidence=0.95,
            description="Critical",
        )

        manager.process_events([warning])
        escalated = manager.process_events([critical])
        assert len(escalated) == 1
        assert escalated[0].severity == AlertSeverity.CRITICAL

    def test_acknowledge_resolve(self):
        manager = AlertManager()
        event = DegradationEvent(
            event_type=DegradationType.THROUGHPUT_DROP,
            severity=AlertSeverity.WARNING,
            metric="throughput_mbps",
            current_value=2000.0,
            threshold_value=0.2,
            predicted_values=[2000.0] * 12,
            timestamp=datetime.utcnow(),
            confidence=0.9,
            description="Test",
        )

        alerts = manager.process_events([event])
        alert_id = alerts[0].id

        assert manager.acknowledge(alert_id)
        assert alerts[0].acknowledged

        assert manager.resolve(alert_id)
        assert alerts[0].resolved
        assert len(manager.get_active_alerts()) == 0

    def test_summary(self):
        manager = AlertManager()
        summary = manager.get_summary()
        assert "total_active" in summary
        assert summary["total_active"] == 0

    def test_alert_serialization(self):
        alert = Alert(
            id="NETAI-00001",
            severity=AlertSeverity.WARNING,
            metric="throughput_mbps",
            title="Test Alert",
            description="Test",
            predicted_values=[1.0, 2.0],
            confidence=0.9,
            created_at=datetime.utcnow(),
        )
        d = alert.to_dict()
        assert d["id"] == "NETAI-00001"
        json_str = alert.to_json()
        assert "NETAI-00001" in json_str
