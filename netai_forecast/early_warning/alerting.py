"""Alert management system for network degradation events.

Manages alert lifecycle: creation, deduplication, severity escalation,
and notification formatting.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Optional
from collections import defaultdict

from .detector import DegradationEvent, AlertSeverity


@dataclass
class Alert:
    """A network performance alert."""
    id: str
    severity: AlertSeverity
    metric: str
    title: str
    description: str
    predicted_values: list[float]
    confidence: float
    created_at: datetime
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["created_at"] = self.created_at.isoformat()
        if self.resolved_at:
            d["resolved_at"] = self.resolved_at.isoformat()
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class AlertManager:
    """Manages alert lifecycle and deduplication.

    Features:
    - Converts degradation events to alerts
    - Deduplicates alerts for the same metric
    - Tracks alert history
    - Provides formatted alert summaries
    """

    def __init__(self, max_history: int = 1000):
        self._alerts: dict[str, Alert] = {}
        self._history: list[Alert] = []
        self._counter: int = 0
        self.max_history = max_history

    def process_events(self, events: list[DegradationEvent]) -> list[Alert]:
        """Convert degradation events to alerts with deduplication.

        Args:
            events: List of detected degradation events.

        Returns:
            List of new or updated alerts.
        """
        new_alerts = []

        for event in events:
            alert_key = f"{event.metric}:{event.event_type.value}"

            if alert_key in self._alerts and not self._alerts[alert_key].resolved:
                # Update existing alert if severity escalated
                existing = self._alerts[alert_key]
                if self._severity_rank(event.severity) > self._severity_rank(existing.severity):
                    existing.severity = event.severity
                    existing.description = event.description
                    existing.predicted_values = event.predicted_values
                    existing.confidence = event.confidence
                    new_alerts.append(existing)
            else:
                self._counter += 1
                alert = Alert(
                    id=f"NETAI-{self._counter:05d}",
                    severity=event.severity,
                    metric=event.metric,
                    title=f"{event.severity.value.upper()}: {event.metric} - {event.event_type.value}",
                    description=event.description,
                    predicted_values=event.predicted_values,
                    confidence=event.confidence,
                    created_at=event.timestamp,
                )
                self._alerts[alert_key] = alert
                self._history.append(alert)
                new_alerts.append(alert)

                # Trim history
                if len(self._history) > self.max_history:
                    self._history = self._history[-self.max_history:]

        return new_alerts

    def acknowledge(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts.values():
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def resolve(self, alert_id: str) -> bool:
        """Resolve an alert."""
        for alert in self._alerts.values():
            if alert.id == alert_id:
                alert.resolved = True
                alert.resolved_at = datetime.utcnow()
                return True
        return False

    def get_active_alerts(self) -> list[Alert]:
        """Get all unresolved alerts."""
        return [a for a in self._alerts.values() if not a.resolved]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> list[Alert]:
        """Get active alerts filtered by severity."""
        return [
            a for a in self._alerts.values()
            if not a.resolved and a.severity == severity
        ]

    def get_summary(self) -> dict:
        """Get a summary of current alert state."""
        active = self.get_active_alerts()
        by_severity = defaultdict(int)
        for alert in active:
            by_severity[alert.severity.value] += 1

        return {
            "total_active": len(active),
            "by_severity": dict(by_severity),
            "total_historical": len(self._history),
            "alerts": [a.to_dict() for a in active],
        }

    @staticmethod
    def _severity_rank(severity: AlertSeverity) -> int:
        return {AlertSeverity.INFO: 0, AlertSeverity.WARNING: 1, AlertSeverity.CRITICAL: 2}[
            severity
        ]
