"""Threshold alerting and automated LLM incident reporting for forecast anomalies."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


@dataclass(frozen=True)
class AnomalyRecord:
    """Single anomaly event context passed to the reporter."""

    timestamp: pd.Timestamp
    expected_latency_ms: float
    predicted_latency_ms: float
    packet_loss_pct: float


def detect_latency_threshold_breaches(
    forecast_df: pd.DataFrame,
    threshold_ms: float,
    predicted_col: str = "predicted_latency_ms",
    expected_col: str = "expected_latency_ms",
    packet_loss_col: str = "packet_loss_pct",
) -> pd.DataFrame:
    """Select forecast points that breach a latency threshold.

    Args:
        forecast_df: DataFrame containing forecast values and optional context.
        threshold_ms: Alert threshold for predicted latency.
        predicted_col: Name of predicted latency column.
        expected_col: Name of expected baseline latency column.
        packet_loss_col: Name of packet-loss context column.

    Returns:
        DataFrame with anomaly rows and normalized schema.
    """
    df = forecast_df.copy()
    if "timestamp" not in df.columns or predicted_col not in df.columns:
        raise ValueError("forecast_df must include 'timestamp' and predicted latency columns.")

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if expected_col not in df.columns:
        df[expected_col] = threshold_ms * 0.75
    if packet_loss_col not in df.columns:
        df[packet_loss_col] = np.nan

    anomalies = df[df[predicted_col] > threshold_ms].copy()
    if anomalies.empty:
        return anomalies

    anomalies = anomalies.rename(
        columns={
            predicted_col: "predicted_latency_ms",
            expected_col: "expected_latency_ms",
            packet_loss_col: "packet_loss_pct",
        }
    )

    anomalies["threshold_ms"] = threshold_ms
    anomalies["breach_ms"] = anomalies["predicted_latency_ms"] - threshold_ms
    anomalies["severity_ratio"] = anomalies["predicted_latency_ms"] / threshold_ms

    return anomalies[
        [
            "timestamp",
            "expected_latency_ms",
            "predicted_latency_ms",
            "packet_loss_pct",
            "threshold_ms",
            "breach_ms",
            "severity_ratio",
        ]
    ]


class LLMReporter:
    """Generate incident narratives from anomaly telemetry using an LLM."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> None:
        """Initialize the reporter.

        Args:
            model: OpenAI model name used for report generation.
            api_key: API key. If omitted, OPENAI_API_KEY is used.
        """
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.api_key) if (OpenAI and self.api_key) else None

    def format_prompt(self, anomaly: AnomalyRecord) -> str:
        """Create a concise prompt to request a 3-sentence incident report.

        Args:
            anomaly: Anomaly event context.

        Returns:
            Prompt string ready for LLM submission.
        """
        packet_loss_text = (
            f"{anomaly.packet_loss_pct:.2f}%"
            if not np.isnan(anomaly.packet_loss_pct)
            else "N/A"
        )

        return (
            "You are an AIOps incident analyst. Write exactly 3 short sentences. "
            "Sentence 1: summarize what happened and when. "
            "Sentence 2: assess likely impact to users/services. "
            "Sentence 3: give one concrete mitigation and one follow-up investigation.\n\n"
            f"Timestamp: {anomaly.timestamp}\n"
            f"Expected latency: {anomaly.expected_latency_ms:.2f} ms\n"
            f"Predicted latency: {anomaly.predicted_latency_ms:.2f} ms\n"
            f"Packet loss: {packet_loss_text}"
        )

    def _mock_request_payload(self, prompt: str) -> dict[str, Any]:
        """Build a mock OpenAI request payload when API access is unavailable.

        Args:
            prompt: Prompt sent to the model.

        Returns:
            Dictionary matching the expected request shape.
        """
        return {
            "model": self.model,
            "input": [
                {"role": "system", "content": "You write concise incident reports."},
                {"role": "user", "content": prompt},
            ],
            "max_output_tokens": 180,
            "temperature": 0.2,
        }

    def generate_incident_report(self, anomaly: AnomalyRecord) -> tuple[str, dict[str, Any] | None]:
        """Generate a 3-sentence incident report, with mock fallback.

        Args:
            anomaly: Anomaly context.

        Returns:
            Tuple of (report text, request payload if mocked else None).
        """
        prompt = self.format_prompt(anomaly)

        if self.client is None:
            payload = self._mock_request_payload(prompt)
            report = (
                f"At {anomaly.timestamp}, forecast latency rose to "
                f"{anomaly.predicted_latency_ms:.1f} ms versus an expected "
                f"{anomaly.expected_latency_ms:.1f} ms, indicating a clear performance degradation. "
                "Users are likely to see slower response times and intermittent failures on latency-sensitive flows. "
                "Mitigate by rerouting or rate-limiting affected traffic immediately, then investigate correlated interface errors and upstream congestion signals."
            )
            return report, payload

        response = self.client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": "You write concise network incident reports."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_output_tokens=180,
        )
        report_text = response.output_text.strip()
        return report_text, None


def run_alerting_pipeline(
    forecast_df: pd.DataFrame,
    threshold_ms: float,
    reporter: LLMReporter,
) -> pd.DataFrame:
    """Detect anomalies and attach generated LLM incident reports.

    Args:
        forecast_df: Forecast DataFrame with timestamp and prediction columns.
        threshold_ms: Threshold for anomaly detection.
        reporter: Reporter instance to generate narratives.

    Returns:
        DataFrame with anomaly context and incident report fields.
    """
    anomalies = detect_latency_threshold_breaches(forecast_df=forecast_df, threshold_ms=threshold_ms)
    if anomalies.empty:
        return anomalies

    reports: list[str] = []
    payloads: list[str] = []

    for _, row in anomalies.iterrows():
        anomaly = AnomalyRecord(
            timestamp=pd.Timestamp(row["timestamp"]),
            expected_latency_ms=float(row["expected_latency_ms"]),
            predicted_latency_ms=float(row["predicted_latency_ms"]),
            packet_loss_pct=float(row["packet_loss_pct"])
            if pd.notna(row["packet_loss_pct"])
            else float("nan"),
        )
        report, payload = reporter.generate_incident_report(anomaly)
        reports.append(report)
        payloads.append(json.dumps(payload) if payload else "")

    anomalies = anomalies.copy()
    anomalies["incident_report"] = reports
    anomalies["mock_request_payload"] = payloads
    return anomalies


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for alerting workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--forecast-path",
        type=Path,
        default=Path("outputs/prophet_backtest.csv"),
        help="CSV path with forecasted latency values.",
    )
    parser.add_argument(
        "--threshold-ms",
        type=float,
        default=35.0,
        help="Latency threshold used for anomaly detection.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/incident_reports.csv"),
        help="Path to write detected anomalies and reports.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="OpenAI model for report generation.",
    )
    return parser.parse_args()


def main() -> None:
    """Run threshold alerting and generate incident summaries."""
    args = parse_args()
    forecast_df = pd.read_csv(args.forecast_path, parse_dates=["timestamp"])

    reporter = LLMReporter(model=args.model)
    alert_df = run_alerting_pipeline(
        forecast_df=forecast_df,
        threshold_ms=args.threshold_ms,
        reporter=reporter,
    )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    alert_df.to_csv(args.output_path, index=False)

    print(f"Detected {len(alert_df)} anomalies above {args.threshold_ms:.1f} ms.")
    print(f"Alert report written to {args.output_path}")


if __name__ == "__main__":
    main()
