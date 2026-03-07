"""Automated incident report generation using LLMs.

Generates structured incident reports from degradation events
using NRP's managed LLM service (Qwen3-VL, GLM-4.7, GPT-OSS)
with fallback to template-based reports for offline operation.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional

import httpx

from ..early_warning.detector import DegradationEvent, AlertSeverity

logger = logging.getLogger(__name__)


@dataclass
class IncidentReport:
    """Structured incident report."""
    report_id: str
    title: str
    severity: str
    summary: str
    affected_metric: str
    predicted_impact: str
    root_cause_hypothesis: str
    recommended_actions: list[str]
    forecast_data: list[float]
    confidence: float
    generated_at: str
    generated_by: str  # "llm" or "template"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    def to_markdown(self) -> str:
        actions = "\n".join(f"  {i+1}. {a}" for i, a in enumerate(self.recommended_actions))
        return f"""# Incident Report: {self.report_id}

**Title:** {self.title}
**Severity:** {self.severity}
**Generated:** {self.generated_at}
**Confidence:** {self.confidence:.0%}
**Generator:** {self.generated_by}

## Summary
{self.summary}

## Affected Metric
{self.affected_metric}

## Predicted Impact
{self.predicted_impact}

## Root Cause Hypothesis
{self.root_cause_hypothesis}

## Recommended Actions
{actions}

## Forecast Data
Values: {', '.join(f'{v:.2f}' for v in self.forecast_data[:12])}
"""


class IncidentReportGenerator:
    """Generates incident reports using LLMs with template fallback.

    Primary: Calls NRP's managed LLM service for natural language reports.
    Fallback: Uses structured templates when LLM is unavailable.
    """

    def __init__(
        self,
        llm_api_url: str = "https://llm.nrp.ai/v1/chat/completions",
        llm_model: str = "qwen3-vl",
        api_key: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.3,
        use_fallback: bool = True,
        timeout: float = 30.0,
    ):
        self.llm_api_url = llm_api_url
        self.llm_model = llm_model
        self.api_key = api_key
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.use_fallback = use_fallback
        self.timeout = timeout
        self._report_counter = 0

    def generate(
        self,
        events: list[DegradationEvent],
        context: Optional[dict] = None,
    ) -> IncidentReport:
        """Generate an incident report from degradation events.

        Args:
            events: List of degradation events to report on.
            context: Optional additional context (topology info, etc.).

        Returns:
            Structured incident report.
        """
        if not events:
            raise ValueError("At least one degradation event is required.")

        self._report_counter += 1
        report_id = f"IR-{datetime.utcnow().strftime('%Y%m%d')}-{self._report_counter:04d}"

        # Try LLM first, fall back to templates
        try:
            return self._generate_with_llm(report_id, events, context)
        except Exception as e:
            logger.warning(f"LLM report generation failed: {e}")
            if self.use_fallback:
                return self._generate_from_template(report_id, events, context)
            raise

    def _generate_with_llm(
        self,
        report_id: str,
        events: list[DegradationEvent],
        context: Optional[dict],
    ) -> IncidentReport:
        """Generate report using LLM API."""
        prompt = self._build_prompt(events, context)

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.llm_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a network operations AI assistant for the National Research Platform. "
                        "Generate concise, actionable incident reports for network performance issues. "
                        "Respond in JSON format with fields: summary, predicted_impact, "
                        "root_cause_hypothesis, recommended_actions (array of strings)."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        response = httpx.post(
            self.llm_api_url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        response.raise_for_status()

        result = response.json()
        content = result["choices"][0]["message"]["content"]

        # Parse LLM response
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = self._parse_freeform(content)

        primary_event = max(events, key=lambda e: self._severity_rank(e.severity))

        return IncidentReport(
            report_id=report_id,
            title=f"Network {primary_event.event_type.value.replace('_', ' ').title()} Detected",
            severity=primary_event.severity.value,
            summary=parsed.get("summary", "Network performance degradation detected."),
            affected_metric=primary_event.metric,
            predicted_impact=parsed.get("predicted_impact", "Potential service degradation."),
            root_cause_hypothesis=parsed.get("root_cause_hypothesis", "Under investigation."),
            recommended_actions=parsed.get("recommended_actions", ["Monitor the situation."]),
            forecast_data=primary_event.predicted_values,
            confidence=primary_event.confidence,
            generated_at=datetime.utcnow().isoformat(),
            generated_by="llm",
        )

    def _generate_from_template(
        self,
        report_id: str,
        events: list[DegradationEvent],
        context: Optional[dict],
    ) -> IncidentReport:
        """Generate report using structured templates (fallback)."""
        primary_event = max(events, key=lambda e: self._severity_rank(e.severity))
        all_metrics = list(set(e.metric for e in events))

        # Template-based generation
        summary = self._template_summary(primary_event, len(events))
        impact = self._template_impact(primary_event)
        hypothesis = self._template_hypothesis(primary_event)
        actions = self._template_actions(primary_event)

        return IncidentReport(
            report_id=report_id,
            title=f"Network {primary_event.event_type.value.replace('_', ' ').title()} Detected",
            severity=primary_event.severity.value,
            summary=summary,
            affected_metric=", ".join(all_metrics),
            predicted_impact=impact,
            root_cause_hypothesis=hypothesis,
            recommended_actions=actions,
            forecast_data=primary_event.predicted_values,
            confidence=primary_event.confidence,
            generated_at=datetime.utcnow().isoformat(),
            generated_by="template",
        )

    def _build_prompt(self, events: list[DegradationEvent], context: Optional[dict]) -> str:
        event_descriptions = []
        for e in events:
            event_descriptions.append(
                f"- Metric: {e.metric}, Type: {e.event_type.value}, "
                f"Severity: {e.severity.value}, Current Value: {e.current_value:.4f}, "
                f"Confidence: {e.confidence:.2f}"
            )

        prompt = (
            "Generate an incident report for the following network performance events "
            "detected by the NETAI predictive analytics system:\n\n"
            + "\n".join(event_descriptions)
        )

        if context:
            prompt += f"\n\nAdditional context:\n{json.dumps(context, indent=2)}"

        prompt += (
            "\n\nRespond with a JSON object containing: summary, predicted_impact, "
            "root_cause_hypothesis, and recommended_actions (array)."
        )
        return prompt

    def _template_summary(self, event: DegradationEvent, n_events: int) -> str:
        templates = {
            "throughput_drop": (
                f"Predictive analytics detected potential throughput degradation. "
                f"Forecasted throughput shows a {event.severity.value} level deviation from baseline "
                f"with {event.confidence:.0%} confidence. {n_events} related event(s) detected."
            ),
            "latency_spike": (
                f"Predictive analytics detected potential latency increase. "
                f"Forecasted latency shows a {event.severity.value} level elevation "
                f"with {event.confidence:.0%} confidence. {n_events} related event(s) detected."
            ),
            "packet_loss_increase": (
                f"Predictive analytics detected potential packet loss increase. "
                f"Forecasted packet loss rate is at {event.severity.value} level "
                f"with {event.confidence:.0%} confidence. {n_events} related event(s) detected."
            ),
            "retransmit_increase": (
                f"Predictive analytics detected elevated retransmit rates. "
                f"Forecasted retransmits at {event.severity.value} level "
                f"with {event.confidence:.0%} confidence. {n_events} related event(s) detected."
            ),
        }
        return templates.get(
            event.event_type.value,
            f"Network performance degradation detected ({event.event_type.value}) "
            f"at {event.severity.value} severity with {event.confidence:.0%} confidence.",
        )

    def _template_impact(self, event: DegradationEvent) -> str:
        impacts = {
            "throughput_drop": "Reduced data transfer rates may affect large-scale scientific data workflows and distributed computing jobs on NRP.",
            "latency_spike": "Increased network latency may impact real-time applications, interactive sessions, and latency-sensitive workloads.",
            "packet_loss_increase": "Elevated packet loss may cause TCP retransmissions, reduced throughput, and degraded application performance.",
            "retransmit_increase": "High retransmit rates indicate network congestion or link issues, impacting overall connection reliability.",
        }
        return impacts.get(event.event_type.value, "Potential impact on network service quality.")

    def _template_hypothesis(self, event: DegradationEvent) -> str:
        hypotheses = {
            "throughput_drop": "Possible causes: link congestion, routing changes, hardware degradation, or increased traffic from competing flows.",
            "latency_spike": "Possible causes: network congestion, routing path changes, DNS resolution delays, or intermediate node issues.",
            "packet_loss_increase": "Possible causes: buffer overflow at intermediate switches/routers, optical signal degradation, or interface errors.",
            "retransmit_increase": "Possible causes: network congestion causing packet drops, receiver window limitations, or middlebox interference.",
        }
        return hypotheses.get(event.event_type.value, "Root cause under investigation.")

    def _template_actions(self, event: DegradationEvent) -> list[str]:
        base_actions = [
            f"Monitor {event.metric} closely over the next hour",
            "Check perfSONAR dashboard for corroborating measurements",
            "Verify traceroute paths for any routing changes",
        ]

        severity_actions = {
            AlertSeverity.WARNING: [
                "Review recent network configuration changes",
                "Check for scheduled maintenance windows",
            ],
            AlertSeverity.CRITICAL: [
                "Escalate to network operations team immediately",
                "Check physical layer status of affected links",
                "Initiate failover to backup path if available",
                "Open incident ticket with upstream provider if external link",
            ],
        }

        return base_actions + severity_actions.get(event.severity, [])

    @staticmethod
    def _parse_freeform(content: str) -> dict:
        """Best-effort parse of non-JSON LLM response."""
        return {
            "summary": content[:500],
            "predicted_impact": "See full report above.",
            "root_cause_hypothesis": "See full report above.",
            "recommended_actions": ["Review the full LLM analysis above."],
        }

    @staticmethod
    def _severity_rank(severity: AlertSeverity) -> int:
        return {AlertSeverity.INFO: 0, AlertSeverity.WARNING: 1, AlertSeverity.CRITICAL: 2}[
            severity
        ]
