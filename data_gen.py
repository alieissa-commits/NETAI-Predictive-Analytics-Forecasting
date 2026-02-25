"""Generate synthetic network telemetry with realistic seasonality and degradations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EventRecord:
    """Metadata for an injected degradation event."""

    event_id: int
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    peak_latency_increase_ms: float
    peak_packet_loss_increase_pct: float


def inject_degradation_events(
    telemetry: pd.DataFrame,
    rng: np.random.Generator,
    min_events: int = 3,
    max_events: int = 4,
) -> tuple[pd.DataFrame, list[EventRecord]]:
    """Inject abrupt degradation windows into latency and packet loss signals.

    Args:
        telemetry: Input telemetry DataFrame with baseline signal columns.
        rng: Numpy random generator for deterministic event placement.
        min_events: Minimum number of degradation events.
        max_events: Maximum number of degradation events.

    Returns:
        Tuple containing modified telemetry and metadata for injected events.
    """
    degraded = telemetry.copy()
    degraded["degradation_event"] = False
    event_records: list[EventRecord] = []

    n_points = len(degraded)
    n_events = int(rng.integers(min_events, max_events + 1))
    occupied = np.zeros(n_points, dtype=bool)

    for event_id in range(1, n_events + 1):
        length = int(rng.integers(12, 49))  # 1 to 4 hours at 5-minute resolution
        max_start = max(100, n_points - length - 100)

        start_idx = None
        for _ in range(200):
            candidate = int(rng.integers(100, max_start))
            if not occupied[candidate : candidate + length].any():
                start_idx = candidate
                break

        if start_idx is None:
            continue

        end_idx = start_idx + length
        occupied[start_idx:end_idx] = True

        profile_x = np.linspace(-1.0, 1.0, length)
        profile = np.exp(-3.5 * (profile_x**2))
        severity = float(rng.uniform(0.9, 1.6))

        peak_latency_boost = float(severity * rng.uniform(35.0, 95.0))
        peak_packet_loss_boost = float(severity * rng.uniform(1.4, 4.8))
        throughput_drop = float(severity * rng.uniform(120.0, 300.0))

        degraded.loc[start_idx:end_idx - 1, "latency_ms"] += peak_latency_boost * profile
        degraded.loc[start_idx:end_idx - 1, "packet_loss_pct"] += peak_packet_loss_boost * profile
        degraded.loc[start_idx:end_idx - 1, "throughput_mbps"] -= throughput_drop * profile
        degraded.loc[start_idx:end_idx - 1, "degradation_event"] = True

        event_records.append(
            EventRecord(
                event_id=event_id,
                start_timestamp=degraded.loc[start_idx, "timestamp"],
                end_timestamp=degraded.loc[end_idx - 1, "timestamp"],
                peak_latency_increase_ms=peak_latency_boost,
                peak_packet_loss_increase_pct=peak_packet_loss_boost,
            )
        )

    degraded["throughput_mbps"] = degraded["throughput_mbps"].clip(lower=40.0)
    degraded["latency_ms"] = degraded["latency_ms"].clip(lower=1.0)
    degraded["packet_loss_pct"] = degraded["packet_loss_pct"].clip(lower=0.0)

    return degraded, event_records


def generate_synthetic_telemetry(
    days: int = 30,
    freq_minutes: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, list[EventRecord]]:
    """Create synthetic telemetry with diurnal patterns and degradation events.

    Args:
        days: Number of days of telemetry to generate.
        freq_minutes: Sampling interval in minutes.
        seed: Random seed for reproducibility.

    Returns:
        A tuple of (telemetry DataFrame, degradation event metadata list).
    """
    rng = np.random.default_rng(seed)
    points_per_day = int(24 * 60 / freq_minutes)
    total_points = points_per_day * days

    end_timestamp = pd.Timestamp.now(tz="UTC").floor(f"{freq_minutes}min").tz_localize(None)
    timestamps = pd.date_range(
        end=end_timestamp,
        periods=total_points,
        freq=f"{freq_minutes}min",
    )

    hour_of_day = timestamps.hour.to_numpy(dtype=float) + (
        timestamps.minute.to_numpy(dtype=float) / 60.0
    )
    day_of_week = timestamps.dayofweek.to_numpy(dtype=int)

    diurnal = np.sin(2.0 * np.pi * (hour_of_day / 24.0 - 0.2))
    harmonic = np.sin(4.0 * np.pi * (hour_of_day / 24.0 + 0.1))
    weekend_factor = np.where(day_of_week >= 5, 1.0, 0.0)

    throughput = (
        700.0
        + 180.0 * diurnal
        + 40.0 * harmonic
        - 65.0 * weekend_factor
        + rng.normal(0.0, 18.0, total_points)
    )

    latency = (
        20.0
        + 7.5 * (1.0 - diurnal)
        + 1.8 * weekend_factor
        + rng.normal(0.0, 1.7, total_points)
    )

    packet_loss = (
        0.18
        + 0.10 * (1.0 - diurnal)
        + 0.06 * weekend_factor
        + rng.normal(0.0, 0.045, total_points)
    )

    telemetry = pd.DataFrame(
        {
            "timestamp": timestamps,
            "throughput_mbps": np.clip(throughput, a_min=80.0, a_max=None),
            "latency_ms": np.clip(latency, a_min=1.0, a_max=None),
            "packet_loss_pct": np.clip(packet_loss, a_min=0.0, a_max=None),
        }
    )

    degraded, events = inject_degradation_events(telemetry, rng=rng)
    return degraded, events


def save_dataset(telemetry: pd.DataFrame, output_path: Path) -> None:
    """Persist telemetry to CSV and ensure parent directory exists.

    Args:
        telemetry: DataFrame to save.
        output_path: Target CSV path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    telemetry.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for telemetry generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_network_telemetry.csv"),
        help="Output CSV path for generated telemetry.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation.",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days to synthesize.",
    )
    return parser.parse_args()


def main() -> None:
    """Generate telemetry data and save it to disk."""
    args = parse_args()
    telemetry, events = generate_synthetic_telemetry(days=args.days, seed=args.seed)
    save_dataset(telemetry, args.output)

    print(f"Saved {len(telemetry)} rows to {args.output}")
    for event in events:
        print(
            "Injected event "
            f"#{event.event_id}: {event.start_timestamp} to {event.end_timestamp}, "
            f"latency +{event.peak_latency_increase_ms:.1f} ms, "
            f"packet loss +{event.peak_packet_loss_increase_pct:.2f}%"
        )


if __name__ == "__main__":
    main()
