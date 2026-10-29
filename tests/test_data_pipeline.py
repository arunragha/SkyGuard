"""
Unit tests for data loader and synthetic generator.
"""

from io import StringIO
import numpy as np
import pandas as pd
import pytest

from src.config import load_config
from src.data.loader import DataLoader, load_weather_data
from src.data.synthetic import WeatherAnomalyGenerator


def test_synthetic_generator_shape_and_columns():
    gen = WeatherAnomalyGenerator(seed=42)
    df = gen.generate(stations=["STATION_ALPHA"], days=2, freq_minutes=10, anomaly_ratio=0.05)

    expected_cols = [
        "timestamp",
        "station_id",
        "temperature",
        "humidity",
        "barometric_pressure",
        "wind_speed",
        "wind_direction",
        "rainfall",
        "solar_radiation",
        "is_anomaly",
        "anomaly_type",
        "faulty_sensor",
    ]
    for col in expected_cols:
        assert col in df.columns

    # 2 days at 10-min cadence = 2 * 24 * 6 = 288 samples
    assert len(df) == 288
    assert df["is_anomaly"].sum() > 0

    # Ensure valid types and physics ranges
    assert (df["humidity"] >= 0).all() and (df["humidity"] <= 100).all()
    assert (df["wind_speed"] >= 0).all()
    assert (df["rainfall"] >= 0).all()


def test_synthetic_generator_multi_station():
    gen = WeatherAnomalyGenerator(seed=100)
    stations = ["STATION_ALPHA", "STATION_BETA"]
    df = gen.generate(stations=stations, days=1, freq_minutes=10)

    assert set(df["station_id"].unique()) == set(stations)
    assert len(df) == 144 * 2


def test_data_loader_csv():
    csv_data = """timestamp,temperature,humidity,barometric_pressure,wind_speed,wind_direction,rainfall,solar_radiation
2026-06-01 00:00:00,20.5,80.0,1013.2,3.5,180.0,0.0,0.0
2026-06-01 00:10:00,20.3,81.0,1013.1,3.2,185.0,0.0,0.0
2026-06-01 00:20:00,20.1,82.5,1013.0,3.0,190.0,0.0,0.0
"""
    loader = DataLoader()
    df = loader.load_file(StringIO(csv_data))
    assert len(df) == 3
    assert "temperature" in df.columns
    assert "station_id" in df.columns


def test_data_loader_missing_timestamps_and_resampling():
    # Gap between 00:00 and 00:30 (missing 00:10, 00:20)
    csv_data = """timestamp,temperature,humidity,barometric_pressure,wind_speed,wind_direction,rainfall,solar_radiation
2026-06-01 00:00:00,20.0,80.0,1013.0,3.0,180.0,0.0,0.0
2026-06-01 00:30:00,23.0,74.0,1013.3,4.5,195.0,0.0,0.0
"""
    loader = DataLoader()
    df = loader.load_file(StringIO(csv_data))

    # Resampled to 10min -> should have 4 rows: 00:00, 00:10, 00:20, 00:30
    assert len(df) == 4
    # Check linear interpolation of temperature: 20.0, 21.0, 22.0, 23.0
    assert np.isclose(df.loc[df["timestamp"] == pd.to_datetime("2026-06-01 00:10:00", utc=True), "temperature"].iloc[0], 21.0)
    assert np.isclose(df.loc[df["timestamp"] == pd.to_datetime("2026-06-01 00:20:00", utc=True), "temperature"].iloc[0], 22.0)
