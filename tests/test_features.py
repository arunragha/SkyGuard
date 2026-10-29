"""
Unit tests for meteorological and time series feature engineer.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import WeatherAnomalyGenerator
from src.features.engineer import FeatureEngineer


@pytest.fixture
def clean_telemetry():
    gen = WeatherAnomalyGenerator(seed=42)
    return gen.generate(stations=["STATION_ALPHA"], days=3, anomaly_ratio=0.0)


def test_feature_engineer_rolling_stats(clean_telemetry):
    engineer = FeatureEngineer()
    df_feat = engineer.transform(clean_telemetry)

    # Check rolling statistics for temperature
    assert "temperature_roll_mean_6" in df_feat.columns
    assert "temperature_roll_std_6" in df_feat.columns
    assert "temperature_roll_spread_6" in df_feat.columns
    assert "temperature_local_zscore_6" in df_feat.columns

    # Check no NaNs remain
    assert not df_feat["temperature_roll_mean_6"].isnull().any()


def test_feature_engineer_rates_of_change(clean_telemetry):
    engineer = FeatureEngineer()
    df_feat = engineer.transform(clean_telemetry)

    assert "temperature_roc" in df_feat.columns
    assert "temperature_accel" in df_feat.columns
    assert "barometric_pressure_roc" in df_feat.columns


def test_feature_engineer_thermodynamics(clean_telemetry):
    engineer = FeatureEngineer()
    df_feat = engineer.transform(clean_telemetry)

    # Check Magnus-Tetens dew point and VPD
    assert "dew_point" in df_feat.columns
    assert "dew_point_depression" in df_feat.columns
    assert "vapor_pressure_deficit" in df_feat.columns
    assert "physics_rh_violation" in df_feat.columns

    # In clean data, dew point must generally be <= temperature
    depression = df_feat["dew_point_depression"].values
    assert (depression >= -0.6).all()


def test_feature_engineer_cyclical_encodings(clean_telemetry):
    engineer = FeatureEngineer()
    df_feat = engineer.transform(clean_telemetry)

    assert "hour_sin" in df_feat.columns
    assert "hour_cos" in df_feat.columns
    assert "season_sin" in df_feat.columns
    assert "season_cos" in df_feat.columns

    # Sin/cos bounds
    assert (df_feat["hour_sin"] >= -1.0).all() and (df_feat["hour_sin"] <= 1.0).all()
    assert (df_feat["hour_cos"] >= -1.0).all() and (df_feat["hour_cos"] <= 1.0).all()
