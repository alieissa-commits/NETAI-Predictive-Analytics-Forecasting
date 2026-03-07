"""Network performance degradation detector.

Analyzes forecasted metrics to detect potential performance
degradation before it impacts users. Combines threshold-based
detection with trend analysis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from datetime import datetime


class DegradationType(str, Enum):
    THROUGHPUT_DROP = "throughput_drop"
    LATENCY_SPIKE = "latency_spike"
    PACKET_LOSS_INCREASE = "packet_loss_increase"
    RETRANSMIT_INCREASE = "retransmit_increase"
    TREND_DEGRADATION = "trend_degradation"


class AlertSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class DegradationEvent:
    """Represents a detected degradation event."""
    event_type: DegradationType
    severity: AlertSeverity
    metric: str
    current_value: float
    threshold_value: float
    predicted_values: list[float]
    timestamp: datetime
    confidence: float
    description: str


@dataclass
class ThresholdConfig:
    """Threshold configuration for a metric."""
    warning_threshold: float
    critical_threshold: float
    is_upper_bound: bool = True  # True = alert when exceeding, False = alert when below


class DegradationDetector:
    """Detects network performance degradation from forecasted metrics.

    Combines multiple detection strategies:
    1. Absolute threshold violation
    2. Relative change from recent baseline
    3. Trend-based degradation (sustained worsening)
    4. Rate-of-change analysis
    """

    DEFAULT_THRESHOLDS = {
        "throughput_mbps": ThresholdConfig(
            warning_threshold=0.20,
            critical_threshold=0.50,
            is_upper_bound=False,  # Alert when throughput drops
        ),
        "latency_ms": ThresholdConfig(
            warning_threshold=0.30,
            critical_threshold=1.00,
            is_upper_bound=True,  # Alert when latency increases
        ),
        "packet_loss_pct": ThresholdConfig(
            warning_threshold=1.0,
            critical_threshold=5.0,
            is_upper_bound=True,
        ),
        "retransmits": ThresholdConfig(
            warning_threshold=50,
            critical_threshold=200,
            is_upper_bound=True,
        ),
    }

    def __init__(
        self,
        thresholds: Optional[dict[str, ThresholdConfig]] = None,
        trend_window: int = 12,
        confidence_threshold: float = 0.8,
    ):
        """Initialize the degradation detector.

        Args:
            thresholds: Custom threshold configuration per metric.
            trend_window: Number of forecast steps for trend analysis.
            confidence_threshold: Minimum confidence to raise alerts.
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS
        self.trend_window = trend_window
        self.confidence_threshold = confidence_threshold

    def detect(
        self,
        forecast: dict[str, np.ndarray],
        recent_history: pd.DataFrame,
        metric: str = "throughput_mbps",
        forecast_intervals: Optional[dict[str, np.ndarray]] = None,
    ) -> list[DegradationEvent]:
        """Detect degradation events from forecasted values.

        Args:
            forecast: Dict with 'forecast' (and optionally 'lower', 'upper') arrays.
            recent_history: Recent historical data for baseline comparison.
            metric: The metric being analyzed.
            forecast_intervals: Optional prediction intervals.

        Returns:
            List of detected degradation events.
        """
        events = []
        predicted = forecast if isinstance(forecast, np.ndarray) else forecast.get("forecast", forecast)

        if isinstance(predicted, dict):
            predicted = predicted["forecast"]

        # Compute baseline from recent history
        baseline = recent_history[metric].mean()
        baseline_std = recent_history[metric].std()

        # 1. Threshold-based detection
        threshold_events = self._check_thresholds(
            predicted, baseline, metric, forecast_intervals
        )
        events.extend(threshold_events)

        # 2. Relative change detection
        change_events = self._check_relative_change(predicted, baseline, metric)
        events.extend(change_events)

        # 3. Trend-based detection
        trend_events = self._check_trend(predicted, baseline, baseline_std, metric)
        events.extend(trend_events)

        return events

    def _check_thresholds(
        self,
        predicted: np.ndarray,
        baseline: float,
        metric: str,
        intervals: Optional[dict[str, np.ndarray]] = None,
    ) -> list[DegradationEvent]:
        """Check absolute threshold violations."""
        events = []
        config = self.thresholds.get(metric)
        if config is None:
            return events

        for i, val in enumerate(predicted):
            severity = None

            if config.is_upper_bound:
                if val >= config.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                elif val >= config.warning_threshold:
                    severity = AlertSeverity.WARNING
            else:
                # For throughput: threshold is percentage drop
                pct_drop = (baseline - val) / baseline if baseline > 0 else 0
                if pct_drop >= config.critical_threshold:
                    severity = AlertSeverity.CRITICAL
                elif pct_drop >= config.warning_threshold:
                    severity = AlertSeverity.WARNING

            if severity is not None:
                confidence = self._compute_confidence(val, intervals, i)
                if confidence >= self.confidence_threshold:
                    events.append(DegradationEvent(
                        event_type=self._metric_to_type(metric),
                        severity=severity,
                        metric=metric,
                        current_value=float(val),
                        threshold_value=config.critical_threshold if severity == AlertSeverity.CRITICAL else config.warning_threshold,
                        predicted_values=predicted.tolist(),
                        timestamp=datetime.utcnow(),
                        confidence=confidence,
                        description=f"Predicted {metric} {'exceeds' if config.is_upper_bound else 'drops below'} {severity.value} threshold at step {i+1}",
                    ))
                    break  # One event per metric per detection run

        return events

    def _check_relative_change(
        self,
        predicted: np.ndarray,
        baseline: float,
        metric: str,
    ) -> list[DegradationEvent]:
        """Check for significant relative changes from baseline."""
        events = []
        if baseline == 0:
            return events

        mean_pred = predicted.mean()
        pct_change = (mean_pred - baseline) / abs(baseline)

        config = self.thresholds.get(metric)
        if config is None:
            return events

        # Determine if change is in the degradation direction
        if config.is_upper_bound and pct_change > 0.5:
            events.append(DegradationEvent(
                event_type=self._metric_to_type(metric),
                severity=AlertSeverity.WARNING,
                metric=metric,
                current_value=float(mean_pred),
                threshold_value=baseline,
                predicted_values=predicted.tolist(),
                timestamp=datetime.utcnow(),
                confidence=min(abs(pct_change), 1.0),
                description=f"Predicted {metric} shows {pct_change*100:.1f}% increase from baseline",
            ))
        elif not config.is_upper_bound and pct_change < -0.3:
            events.append(DegradationEvent(
                event_type=self._metric_to_type(metric),
                severity=AlertSeverity.WARNING,
                metric=metric,
                current_value=float(mean_pred),
                threshold_value=baseline,
                predicted_values=predicted.tolist(),
                timestamp=datetime.utcnow(),
                confidence=min(abs(pct_change), 1.0),
                description=f"Predicted {metric} shows {abs(pct_change)*100:.1f}% decrease from baseline",
            ))

        return events

    def _check_trend(
        self,
        predicted: np.ndarray,
        baseline: float,
        baseline_std: float,
        metric: str,
    ) -> list[DegradationEvent]:
        """Detect sustained degradation trends."""
        events = []
        if len(predicted) < 3:
            return events

        # Compute linear trend
        x = np.arange(len(predicted))
        coeffs = np.polyfit(x, predicted, 1)
        slope = coeffs[0]

        config = self.thresholds.get(metric)
        if config is None:
            return events

        # Significant trend if slope moves value by >1 std over forecast window
        trend_magnitude = abs(slope * len(predicted))
        if baseline_std > 0 and trend_magnitude > baseline_std:
            is_degrading = (config.is_upper_bound and slope > 0) or (
                not config.is_upper_bound and slope < 0
            )

            if is_degrading:
                confidence = min(trend_magnitude / (baseline_std * 2), 1.0)
                if confidence >= self.confidence_threshold:
                    events.append(DegradationEvent(
                        event_type=DegradationType.TREND_DEGRADATION,
                        severity=AlertSeverity.WARNING,
                        metric=metric,
                        current_value=float(predicted[-1]),
                        threshold_value=baseline,
                        predicted_values=predicted.tolist(),
                        timestamp=datetime.utcnow(),
                        confidence=confidence,
                        description=f"Sustained {'increasing' if slope > 0 else 'decreasing'} trend detected in {metric}",
                    ))

        return events

    def _compute_confidence(
        self,
        value: float,
        intervals: Optional[dict[str, np.ndarray]],
        index: int,
    ) -> float:
        """Compute confidence based on prediction intervals."""
        if intervals is None:
            return 0.85  # Default moderate confidence

        lower = intervals.get("lower")
        upper = intervals.get("upper")
        if lower is None or upper is None or index >= len(lower):
            return 0.85

        interval_width = upper[index] - lower[index]
        if interval_width == 0:
            return 0.95

        # Narrower intervals → higher confidence
        return min(0.95, max(0.5, 1.0 - interval_width / (abs(value) + 1e-8) * 0.5))

    @staticmethod
    def _metric_to_type(metric: str) -> DegradationType:
        mapping = {
            "throughput_mbps": DegradationType.THROUGHPUT_DROP,
            "latency_ms": DegradationType.LATENCY_SPIKE,
            "packet_loss_pct": DegradationType.PACKET_LOSS_INCREASE,
            "retransmits": DegradationType.RETRANSMIT_INCREASE,
        }
        return mapping.get(metric, DegradationType.TREND_DEGRADATION)
