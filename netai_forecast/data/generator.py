"""Synthetic network data generator mimicking perfSONAR measurements.

Generates realistic time-series data for network metrics including
throughput, latency, packet loss, and retransmits with configurable
anomalies, trends, and seasonal patterns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class NetworkProfile:
    """Defines baseline characteristics for a network link."""
    name: str
    base_throughput_mbps: float = 9500.0
    base_latency_ms: float = 12.0
    base_packet_loss_pct: float = 0.01
    base_retransmits: float = 5.0
    throughput_noise_std: float = 300.0
    latency_noise_std: float = 1.5
    packet_loss_noise_std: float = 0.005
    retransmits_noise_std: float = 3.0


# Pre-defined profiles representing common NRP link types
PROFILES: dict[str, NetworkProfile] = {
    "datacenter_10g": NetworkProfile(
        name="datacenter_10g",
        base_throughput_mbps=9500.0,
        base_latency_ms=2.0,
        base_packet_loss_pct=0.001,
        base_retransmits=2.0,
        throughput_noise_std=200.0,
        latency_noise_std=0.3,
    ),
    "campus_1g": NetworkProfile(
        name="campus_1g",
        base_throughput_mbps=900.0,
        base_latency_ms=5.0,
        base_packet_loss_pct=0.05,
        base_retransmits=8.0,
        throughput_noise_std=80.0,
        latency_noise_std=1.0,
    ),
    "wan_research": NetworkProfile(
        name="wan_research",
        base_throughput_mbps=5000.0,
        base_latency_ms=45.0,
        base_packet_loss_pct=0.1,
        base_retransmits=15.0,
        throughput_noise_std=500.0,
        latency_noise_std=5.0,
    ),
    "transatlantic": NetworkProfile(
        name="transatlantic",
        base_throughput_mbps=3000.0,
        base_latency_ms=85.0,
        base_packet_loss_pct=0.2,
        base_retransmits=25.0,
        throughput_noise_std=600.0,
        latency_noise_std=8.0,
    ),
}


@dataclass
class AnomalyConfig:
    """Configuration for anomaly injection."""
    probability: float = 0.03
    severity_range: tuple[float, float] = (2.0, 5.0)
    duration_range: tuple[int, int] = (3, 24)  # in time steps


class NetworkDataGenerator:
    """Generates synthetic perfSONAR-like network measurement data.

    Produces realistic time-series with:
    - Daily and weekly seasonality patterns
    - Gaussian noise with realistic variance
    - Configurable anomaly injection
    - Gradual trend components
    - Correlation between metrics (e.g., high loss → low throughput)
    """

    def __init__(
        self,
        profile: str | NetworkProfile = "wan_research",
        num_days: int = 90,
        sampling_interval_minutes: int = 5,
        anomaly_config: Optional[AnomalyConfig] = None,
        seed: Optional[int] = None,
    ):
        if isinstance(profile, str):
            if profile not in PROFILES:
                raise ValueError(
                    f"Unknown profile '{profile}'. Choose from: {list(PROFILES.keys())}"
                )
            self.profile = PROFILES[profile]
        else:
            self.profile = profile

        self.num_days = num_days
        self.interval_min = sampling_interval_minutes
        self.anomaly_config = anomaly_config or AnomalyConfig()
        self.rng = np.random.default_rng(seed)
        self._n_points = (num_days * 24 * 60) // sampling_interval_minutes

    def generate(self) -> pd.DataFrame:
        """Generate a complete synthetic dataset.

        Returns:
            DataFrame with columns: timestamp, throughput_mbps, latency_ms,
            packet_loss_pct, retransmits, is_anomaly
        """
        timestamps = pd.date_range(
            start="2025-01-01",
            periods=self._n_points,
            freq=f"{self.interval_min}min",
        )

        # Time features for seasonality (convert to numpy for fast math)
        hour_of_day = np.asarray(timestamps.hour + timestamps.minute / 60.0, dtype=np.float64)
        day_of_week = np.asarray(timestamps.dayofweek, dtype=np.float64)

        # Daily seasonality: network busier during work hours
        daily_pattern = self._daily_seasonality(hour_of_day)
        # Weekly seasonality: less traffic on weekends
        weekly_pattern = self._weekly_seasonality(day_of_week)
        # Combined seasonal factor
        seasonal = daily_pattern * weekly_pattern

        # Generate base metrics with seasonal modulation
        throughput = self._gen_throughput(seasonal)
        latency = self._gen_latency(seasonal)
        packet_loss = self._gen_packet_loss(seasonal)
        retransmits = self._gen_retransmits(seasonal, packet_loss)

        # Inject anomalies
        is_anomaly = np.zeros(self._n_points, dtype=bool)
        anomaly_mask = self._generate_anomaly_mask()
        is_anomaly[anomaly_mask] = True

        throughput, latency, packet_loss, retransmits = self._inject_anomalies(
            throughput, latency, packet_loss, retransmits, anomaly_mask
        )

        # Clip to physical bounds
        throughput = np.clip(throughput, 0, self.profile.base_throughput_mbps * 1.05)
        latency = np.clip(latency, 0.1, None)
        packet_loss = np.clip(packet_loss, 0.0, 100.0)
        retransmits = np.clip(retransmits, 0, None).astype(int)

        return pd.DataFrame({
            "timestamp": timestamps,
            "throughput_mbps": np.round(throughput, 2),
            "latency_ms": np.round(latency, 3),
            "packet_loss_pct": np.round(packet_loss, 4),
            "retransmits": retransmits,
            "is_anomaly": is_anomaly,
        })

    def _daily_seasonality(self, hour: np.ndarray) -> np.ndarray:
        """Model daily usage pattern peaking during business hours."""
        return 1.0 + 0.15 * np.sin(2 * np.pi * (hour - 14) / 24)

    def _weekly_seasonality(self, dow: np.ndarray) -> np.ndarray:
        """Model weekly pattern with lower weekend traffic."""
        return np.where(dow < 5, 1.0, 0.85)

    def _gen_throughput(self, seasonal: np.ndarray) -> np.ndarray:
        base = self.profile.base_throughput_mbps
        noise = self.rng.normal(0, self.profile.throughput_noise_std, self._n_points)
        # Higher seasonal → higher throughput demand → slightly reduced available throughput
        return base - (seasonal - 1.0) * base * 0.1 + noise

    def _gen_latency(self, seasonal: np.ndarray) -> np.ndarray:
        base = self.profile.base_latency_ms
        noise = self.rng.normal(0, self.profile.latency_noise_std, self._n_points)
        # Higher load → higher latency
        return base * seasonal + noise

    def _gen_packet_loss(self, seasonal: np.ndarray) -> np.ndarray:
        base = self.profile.base_packet_loss_pct
        noise = np.abs(self.rng.normal(0, self.profile.packet_loss_noise_std, self._n_points))
        return base * seasonal + noise

    def _gen_retransmits(self, seasonal: np.ndarray, packet_loss: np.ndarray) -> np.ndarray:
        base = self.profile.base_retransmits
        noise = self.rng.normal(0, self.profile.retransmits_noise_std, self._n_points)
        # Retransmits correlated with packet loss
        return base * seasonal + packet_loss * 10 + noise

    def _generate_anomaly_mask(self) -> np.ndarray:
        """Generate contiguous anomaly windows."""
        mask = np.zeros(self._n_points, dtype=bool)
        i = 0
        while i < self._n_points:
            if self.rng.random() < self.anomaly_config.probability:
                duration = self.rng.integers(
                    self.anomaly_config.duration_range[0],
                    self.anomaly_config.duration_range[1] + 1,
                )
                end = min(i + duration, self._n_points)
                mask[i:end] = True
                i = end
            else:
                i += 1
        return mask

    def _inject_anomalies(
        self,
        throughput: np.ndarray,
        latency: np.ndarray,
        packet_loss: np.ndarray,
        retransmits: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Inject realistic anomalies into metrics."""
        n_anomaly = mask.sum()
        if n_anomaly == 0:
            return throughput, latency, packet_loss, retransmits

        severity = self.rng.uniform(
            self.anomaly_config.severity_range[0],
            self.anomaly_config.severity_range[1],
            n_anomaly,
        )

        # Anomalies: throughput drops, latency spikes, loss increases
        throughput[mask] /= severity
        latency[mask] *= severity
        packet_loss[mask] *= severity * 3
        retransmits[mask] *= severity * 2

        return throughput, latency, packet_loss, retransmits

    def generate_multi_link(
        self, profiles: Optional[list[str]] = None, n_links: int = 4
    ) -> dict[str, pd.DataFrame]:
        """Generate data for multiple network links.

        Args:
            profiles: List of profile names. If None, uses all built-in profiles.
            n_links: Number of links (used only if profiles is None).

        Returns:
            Dictionary mapping link names to DataFrames.
        """
        if profiles is None:
            profiles = list(PROFILES.keys())[:n_links]

        result = {}
        for prof_name in profiles:
            gen = NetworkDataGenerator(
                profile=prof_name,
                num_days=self.num_days,
                sampling_interval_minutes=self.interval_min,
                anomaly_config=self.anomaly_config,
                seed=self.rng.integers(0, 2**31),
            )
            result[prof_name] = gen.generate()
        return result
