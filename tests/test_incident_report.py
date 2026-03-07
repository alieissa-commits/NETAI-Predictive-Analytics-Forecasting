"""Tests for incident report generation."""

import pytest
from datetime import datetime

from netai_forecast.early_warning.detector import (
    DegradationEvent,
    DegradationType,
    AlertSeverity,
)
from netai_forecast.incident_report.report_generator import (
    IncidentReportGenerator,
    IncidentReport,
)


@pytest.fixture
def sample_events():
    return [
        DegradationEvent(
            event_type=DegradationType.THROUGHPUT_DROP,
            severity=AlertSeverity.CRITICAL,
            metric="throughput_mbps",
            current_value=2000.0,
            threshold_value=0.5,
            predicted_values=[2000.0, 1800.0, 1600.0, 1500.0],
            timestamp=datetime.utcnow(),
            confidence=0.92,
            description="Critical throughput drop detected",
        ),
        DegradationEvent(
            event_type=DegradationType.LATENCY_SPIKE,
            severity=AlertSeverity.WARNING,
            metric="latency_ms",
            current_value=100.0,
            threshold_value=50.0,
            predicted_values=[100.0, 110.0, 120.0, 115.0],
            timestamp=datetime.utcnow(),
            confidence=0.85,
            description="Latency increase detected",
        ),
    ]


class TestIncidentReportGenerator:
    def test_template_report(self, sample_events):
        """Test template-based report generation (fallback)."""
        gen = IncidentReportGenerator(use_fallback=True)
        report = gen.generate(sample_events)

        assert isinstance(report, IncidentReport)
        assert report.report_id.startswith("IR-")
        assert report.severity == "critical"
        assert report.generated_by == "template"
        assert len(report.recommended_actions) > 0
        assert report.confidence > 0

    def test_report_markdown(self, sample_events):
        gen = IncidentReportGenerator(use_fallback=True)
        report = gen.generate(sample_events)
        md = report.to_markdown()
        assert "# Incident Report:" in md
        assert "Summary" in md
        assert "Recommended Actions" in md

    def test_report_json(self, sample_events):
        gen = IncidentReportGenerator(use_fallback=True)
        report = gen.generate(sample_events)
        json_str = report.to_json()
        assert "report_id" in json_str
        assert "severity" in json_str

    def test_report_dict(self, sample_events):
        gen = IncidentReportGenerator(use_fallback=True)
        report = gen.generate(sample_events)
        d = report.to_dict()
        assert isinstance(d, dict)
        assert d["severity"] == "critical"
        assert isinstance(d["recommended_actions"], list)

    def test_no_events_error(self):
        gen = IncidentReportGenerator(use_fallback=True)
        with pytest.raises(ValueError, match="(?i)at least one"):
            gen.generate([])

    def test_single_event_report(self):
        gen = IncidentReportGenerator(use_fallback=True)
        event = DegradationEvent(
            event_type=DegradationType.PACKET_LOSS_INCREASE,
            severity=AlertSeverity.WARNING,
            metric="packet_loss_pct",
            current_value=3.0,
            threshold_value=1.0,
            predicted_values=[3.0, 3.5, 4.0],
            timestamp=datetime.utcnow(),
            confidence=0.88,
            description="Packet loss increase",
        )
        report = gen.generate([event])
        assert "packet_loss" in report.affected_metric

    def test_multiple_reports_unique_ids(self, sample_events):
        gen = IncidentReportGenerator(use_fallback=True)
        r1 = gen.generate(sample_events)
        r2 = gen.generate(sample_events)
        assert r1.report_id != r2.report_id

    def test_template_types(self):
        """Ensure all degradation types have templates."""
        gen = IncidentReportGenerator(use_fallback=True)

        for dtype, metric in [
            (DegradationType.THROUGHPUT_DROP, "throughput_mbps"),
            (DegradationType.LATENCY_SPIKE, "latency_ms"),
            (DegradationType.PACKET_LOSS_INCREASE, "packet_loss_pct"),
            (DegradationType.RETRANSMIT_INCREASE, "retransmits"),
        ]:
            event = DegradationEvent(
                event_type=dtype,
                severity=AlertSeverity.WARNING,
                metric=metric,
                current_value=100.0,
                threshold_value=50.0,
                predicted_values=[100.0],
                timestamp=datetime.utcnow(),
                confidence=0.85,
                description="Test",
            )
            report = gen.generate([event])
            assert report.summary  # Non-empty summary
            assert report.predicted_impact  # Non-empty impact
            assert report.root_cause_hypothesis  # Non-empty hypothesis
