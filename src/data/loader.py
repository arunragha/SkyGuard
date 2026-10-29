"""
Data ingestion and time series preprocessing pipeline for weather station sensors.
Loads CSV/Parquet files, aligns timestamps, resamples to regular grid, and interpolates gaps.
"""

from io import BytesIO, StringIO
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import AppConfig, SamplingConfig, load_config

STANDARD_CHANNELS = [
    "temperature",
    "humidity",
    "barometric_pressure",
    "wind_speed",
    "wind_direction",
    "rainfall",
    "solar_radiation",
]

CHANNEL_ALIASES = {
    "temp": "temperature",
    "ambient_temp": "temperature",
    "rh": "humidity",
    "rel_humidity": "humidity",
    "pressure": "barometric_pressure",
    "baro_pressure": "barometric_pressure",
    "atm_pressure": "barometric_pressure",
    "wind_spd": "wind_speed",
    "wind": "wind_speed",
    "wind_dir": "wind_direction",
    "dir": "wind_direction",
    "rain": "rainfall",
    "precipitation": "rainfall",
    "solar": "solar_radiation",
    "radiation": "solar_radiation",
    "ghi": "solar_radiation",
}


class DataLoader:
    """
    Ingests weather sensor telemetry from CSV/Parquet, regularizes timestamps,
    and handles missing sensor values.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.sampling_cfg: SamplingConfig = self.config.sampling

    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardizes sensor column names according to known aliases."""
        renamed = {}
        for col in df.columns:
            lower = str(col).strip().lower()
            if lower in CHANNEL_ALIASES:
                renamed[col] = CHANNEL_ALIASES[lower]
            elif lower in STANDARD_CHANNELS:
                renamed[col] = lower
            elif lower in ["timestamp", "datetime", "time", "date", "ts"]:
                renamed[col] = "timestamp"
            elif lower in ["station", "station_id", "id"]:
                renamed[col] = "station_id"
            else:
                renamed[col] = lower
        return df.rename(columns=renamed)

    def load_file(
        self, file_path_or_bytes: Union[str, Path, bytes, StringIO, BytesIO]
    ) -> pd.DataFrame:
        """Loads data from CSV or Parquet path or byte stream."""
        if isinstance(file_path_or_bytes, (str, Path)):
            path_str = str(file_path_or_bytes)
            if path_str.endswith(".parquet") or path_str.endswith(".pq"):
                df = pd.read_parquet(path_str)
            else:
                df = pd.read_csv(path_str)
        elif isinstance(file_path_or_bytes, bytes):
            # Attempt parquet first, fallback to csv
            try:
                df = pd.read_parquet(BytesIO(file_path_or_bytes))
            except Exception:
                df = pd.read_csv(BytesIO(file_path_or_bytes))
        elif isinstance(file_path_or_bytes, StringIO):
            df = pd.read_csv(file_path_or_bytes)
        elif isinstance(file_path_or_bytes, BytesIO):
            try:
                df = pd.read_parquet(file_path_or_bytes)
            except Exception:
                file_path_or_bytes.seek(0)
                df = pd.read_csv(file_path_or_bytes)
        else:
            raise ValueError(f"Unsupported input type: {type(file_path_or_bytes)}")

        return self.preprocess(df)

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocesses raw weather telemetry:
        - Resolves timestamp column
        - Sorts by timestamp
        - Handles multiple stations if present
        - Regularizes time grid to specified cadence (e.g., 10 minutes)
        - Fills small missing value gaps
        """
        df = df.copy()
        df = self.standardize_column_names(df)

        if "timestamp" not in df.columns:
            # Check if index is datetime
            if isinstance(df.index, pd.DatetimeIndex):
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: "timestamp"})
            else:
                raise ValueError("Telemetry DataFrame must contain a 'timestamp' column or DatetimeIndex")

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp")

        # Multi-station support
        if "station_id" in df.columns and df["station_id"].nunique() > 1:
            processed_stations = []
            for st_id, st_df in df.groupby("station_id"):
                p_df = self._regularize_station_series(st_df)
                p_df["station_id"] = st_id
                processed_stations.append(p_df)
            combined = pd.concat(processed_stations, axis=0).reset_index()
            return combined.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
        else:
            if "station_id" not in df.columns:
                df["station_id"] = "STATION_001"
            regularized = self._regularize_station_series(df)
            regularized["station_id"] = df["station_id"].iloc[0]
            return regularized.reset_index().sort_values("timestamp").reset_index(drop=True)

    def _regularize_station_series(self, df: pd.DataFrame) -> pd.DataFrame:
        """Resamples single-station data to uniform frequency and handles missing values."""
        df = df.drop_duplicates(subset=["timestamp"]).set_index("timestamp")
        freq_str = f"{self.sampling_cfg.expected_interval_minutes}min"

        # Separate numeric and non-numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in df.columns if c not in numeric_cols and c != "station_id"]

        # Resample numeric columns by mean
        resampled_numeric = df[numeric_cols].resample(freq_str).mean()

        # Handle interpolation within maximum allowable gap
        max_gap = self.sampling_cfg.max_fill_gap_intervals
        if self.sampling_cfg.fill_method == "linear":
            resampled_numeric = resampled_numeric.interpolate(
                method="time", limit=max_gap, limit_direction="both"
            )
        elif self.sampling_cfg.fill_method == "ffill":
            resampled_numeric = resampled_numeric.ffill(limit=max_gap).bfill(limit=max_gap)

        # Non-numeric columns resampled by ffill
        if non_numeric_cols:
            resampled_non_numeric = df[non_numeric_cols].resample(freq_str).ffill(limit=max_gap)
            resampled = pd.concat([resampled_numeric, resampled_non_numeric], axis=1)
        else:
            resampled = resampled_numeric

        return resampled


def load_weather_data(path_or_bytes: Union[str, Path, bytes], config: Optional[AppConfig] = None) -> pd.DataFrame:
    """Convenience functional loader."""
    loader = DataLoader(config)
    return loader.load_file(path_or_bytes)
