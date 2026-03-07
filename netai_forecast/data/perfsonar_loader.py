"""Loader for perfSONAR measurement data stored in SQLite databases.

Supports loading throughput test results and traceroute data
from perfSONAR's SQLite archive format, as well as CSV/Parquet files.
"""

from __future__ import annotations

import sqlite3
import pandas as pd
from pathlib import Path
from typing import Optional


class PerfSONARLoader:
    """Load and prepare perfSONAR measurement data for forecasting.

    Supports multiple data sources:
    - SQLite databases (perfSONAR archive format)
    - CSV files
    - Parquet files

    Example:
        >>> loader = PerfSONARLoader()
        >>> df = loader.from_csv("measurements.csv")
        >>> loader.validate(df)
    """

    REQUIRED_COLUMNS = {"timestamp"}
    METRIC_COLUMNS = {"throughput_mbps", "latency_ms", "packet_loss_pct", "retransmits"}

    def from_sqlite(
        self,
        db_path: str | Path,
        query: Optional[str] = None,
        table: str = "measurements",
    ) -> pd.DataFrame:
        """Load data from a perfSONAR SQLite database.

        Args:
            db_path: Path to the SQLite database.
            query: Custom SQL query. If None, selects all from table.
            table: Table name to query (used only if query is None).

        Returns:
            DataFrame with network metrics.
        """
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        try:
            if query is None:
                query = f"SELECT * FROM {table} ORDER BY timestamp"
            df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
        finally:
            conn.close()

        return self._standardize(df)

    def from_csv(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Load data from a CSV file.

        Args:
            path: Path to the CSV file.
            **kwargs: Additional arguments passed to pd.read_csv.

        Returns:
            DataFrame with network metrics.
        """
        df = pd.read_csv(path, parse_dates=["timestamp"], **kwargs)
        return self._standardize(df)

    def from_parquet(self, path: str | Path, **kwargs) -> pd.DataFrame:
        """Load data from a Parquet file.

        Args:
            path: Path to the Parquet file.
            **kwargs: Additional arguments passed to pd.read_parquet.

        Returns:
            DataFrame with network metrics.
        """
        df = pd.read_parquet(path, **kwargs)
        return self._standardize(df)

    def validate(self, df: pd.DataFrame) -> list[str]:
        """Validate that the DataFrame has the expected schema.

        Returns:
            List of warning messages (empty if all is well).
        """
        warnings = []
        missing_required = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_required:
            warnings.append(f"Missing required columns: {missing_required}")

        missing_metrics = self.METRIC_COLUMNS - set(df.columns)
        if missing_metrics:
            warnings.append(f"Missing metric columns: {missing_metrics}")

        if "timestamp" in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
                warnings.append("'timestamp' column is not datetime type")
            elif df["timestamp"].is_monotonic_increasing is False:
                warnings.append("'timestamp' is not monotonically increasing")

        # Check for excessive missing data
        for col in self.METRIC_COLUMNS & set(df.columns):
            pct_missing = df[col].isna().mean() * 100
            if pct_missing > 10:
                warnings.append(f"Column '{col}' has {pct_missing:.1f}% missing values")

        return warnings

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names and types."""
        df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

        if "timestamp" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["timestamp"]
        ):
            df["timestamp"] = pd.to_datetime(df["timestamp"])

        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
