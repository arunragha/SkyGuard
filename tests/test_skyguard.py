"""
Unit Tests for SkyGuard Weather Station Anomaly Detection System
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

# Ensure skyguard is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skyguard.data.weather_generator import WeatherDataGenerator
from skyguard.features.engineer import WeatherFeatureEngineer
from skyguard.models.isolation_forest import IsolationForestDetector
from skyguard.models.pca_detector import PCADetector
from skyguard.models.lof_detector import LOFDetector
from skyguard.models.rules import PhysicsRuleDetector
from skyguard.models.ensemble import SkyGuardEnsemble
from skyguard.pipeline import SkyGuardPipeline
from skyguard.evaluation.metrics import evaluate_anomaly_detection, evaluate_by_fault_type


def _create_sample_data():
    gen = WeatherDataGenerator(seed=123)
    return gen.generate_telemetry(periods=500, freq="10min", inject_faults=True, anomaly_ratio=0.06)


@pytest.fixture
def sample_data():
    return _create_sample_data()


def test_weather_generator_columns(sample_data):
    expected_cols = [
        "timestamp", "station_id", "temperature_c", "humidity_pct",
        "pressure_hpa", "wind_speed_ms", "wind_direction_deg",
        "solar_radiation_wm2", "precipitation_mm", "battery_voltage_v",
        "internal_temp_c", "is_anomaly", "anomaly_type", "faulty_sensor"
    ]
    for col in expected_cols:
        assert col in sample_data.columns
    assert len(sample_data) == 500
    assert sample_data["is_anomaly"].sum() > 0


def test_feature_engineer(sample_data):
    eng = WeatherFeatureEngineer(rolling_windows=(6, 12))
    feat_df = eng.fit_transform(sample_data)

    # Check thermodynamic features
    assert "dew_point_c" in feat_df.columns
    assert "vpd_kpa" in feat_df.columns
    assert "potential_temp_k" in feat_df.columns

    # Check differencing and rolling features
    assert "temperature_c_diff1" in feat_df.columns
    assert "temperature_c_roll_std_6" in feat_df.columns
    assert "humidity_pct_roll_spread_6" in feat_df.columns

    # Verify no unhandled NaNs
    feature_names = eng.get_feature_names()
    assert len(feature_names) > 20
    assert not feat_df[feature_names].isna().any().any()


def test_individual_detectors(sample_data):
    eng = WeatherFeatureEngineer(rolling_windows=(6,))
    feat_df = eng.fit_transform(sample_data)
    X = feat_df[eng.get_feature_names()]

    # 1. Isolation Forest
    iforest = IsolationForestDetector(n_estimators=50, random_state=42)
    iforest.fit(X)
    s_if = iforest.score(X)
    assert len(s_if) == len(X)
    assert np.all((s_if >= 0.0) & (s_if <= 1.0))
    p_if = iforest.predict(X)
    assert set(p_if).issubset({0, 1})

    # 2. PCA Reconstruction
    pca = PCADetector(variance_ratio=0.85, random_state=42)
    pca.fit(X)
    s_pca = pca.score(X)
    assert len(s_pca) == len(X)
    assert np.all((s_pca >= 0.0) & (s_pca <= 1.0))

    # 3. LOF
    lof = LOFDetector(n_neighbors=15)
    lof.fit(X)
    s_lof = lof.score(X)
    assert len(s_lof) == len(X)

    # 4. Rules
    rules = PhysicsRuleDetector()
    s_rules, diag = rules.score_and_diagnose(sample_data)
    assert len(s_rules) == len(sample_data)
    assert len(diag) == len(sample_data)


def test_skyguard_ensemble_and_rca(sample_data):
    eng = WeatherFeatureEngineer(rolling_windows=(6,))
    feat_df = eng.fit_transform(sample_data)
    X = feat_df[eng.get_feature_names()]

    ensemble = SkyGuardEnsemble()
    ensemble.fit(X, sample_data)

    scores_df = ensemble.score(X, sample_data)
    assert "ensemble_score" in scores_df.columns
    assert "severity" in scores_df.columns
    assert "rule_diagnosis" in scores_df.columns

    # Check RCA on highest anomaly
    top_anomaly_idx = int(scores_df["ensemble_score"].argmax())
    rca = ensemble.diagnose_root_cause(top_anomaly_idx, X, sample_data)
    assert "fault_mode" in rca
    assert "primary_culprit_sensor" in rca
    assert "top_contributing_features" in rca


def test_skyguard_pipeline(sample_data):
    pipeline = SkyGuardPipeline()
    result_df = pipeline.process(sample_data)

    assert "ensemble_score" in result_df.columns
    assert "is_anomaly_pred" in result_df.columns

    # Test evaluation metrics
    metrics = evaluate_anomaly_detection(
        sample_data["is_anomaly"].values,
        result_df["ensemble_score"].values,
        result_df["is_anomaly_pred"].values
    )
    assert "f1_score" in metrics
    assert "roc_auc" in metrics
    assert metrics["roc_auc"] > 0.65  # Baseline sanity check

    breakdown = evaluate_by_fault_type(result_df)
    assert len(breakdown) > 1


if __name__ == "__main__":
    print("Running SkyGuard test suite directly...")
    data = _create_sample_data()
    print("1. Testing weather generator columns...")
    test_weather_generator_columns(data)
    print("2. Testing feature engineer...")
    test_feature_engineer(data)
    print("3. Testing individual detectors...")
    test_individual_detectors(data)
    print("4. Testing ensemble and RCA...")
    test_skyguard_ensemble_and_rca(data)
    print("5. Testing end-to-end pipeline...")
    test_skyguard_pipeline(data)
    print("ALL TESTS PASSED SUCCESSFULLY!")

