"""
Unit tests for detector pipeline, dynamic thresholding, and fault categorization.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.synthetic import WeatherAnomalyGenerator
from src.detector import SkyGuardDetector


@pytest.fixture
def detector_and_data():
    from src.config import load_config
    cfg = load_config()
    cfg.autoencoder.epochs = 3
    gen = WeatherAnomalyGenerator(cfg, seed=42)
    train_df = gen.generate(stations=["STATION_ALPHA"], days=2, anomaly_ratio=0.0)
    test_df = gen.generate(stations=["STATION_ALPHA"], days=2, anomaly_ratio=0.06)

    detector = SkyGuardDetector(cfg)
    detector.fit(train_df)
    return detector, test_df


def test_detector_predict_columns(detector_and_data):
    detector, test_df = detector_and_data
    results = detector.predict(test_df, sensitivity=0.5)

    assert "anomaly_score" in results.columns
    assert "anomaly_threshold" in results.columns
    assert "predicted_anomaly" in results.columns
    assert "predicted_type" in results.columns
    assert "diagnosed_sensor" in results.columns

    # Verify score range
    scores = results["anomaly_score"].values
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

    # Threshold must be within (0, 1)
    thresh = results["anomaly_threshold"].iloc[0]
    assert 0.0 < thresh < 1.0


def test_dynamic_threshold_sensitivity_scaling():
    detector = SkyGuardDetector()
    scores = np.linspace(0.1, 0.9, 1000)

    # Higher sensitivity should produce lower threshold (more alerts)
    thresh_low_sens = detector.compute_dynamic_threshold(scores, sensitivity=0.1, method="percentile")
    thresh_mid_sens = detector.compute_dynamic_threshold(scores, sensitivity=0.5, method="percentile")
    thresh_high_sens = detector.compute_dynamic_threshold(scores, sensitivity=0.9, method="percentile")

    assert thresh_low_sens > thresh_mid_sens > thresh_high_sens

    # Test MAD method sensitivity scaling
    mad_low_sens = detector.compute_dynamic_threshold(scores, sensitivity=0.1, method="mad")
    mad_high_sens = detector.compute_dynamic_threshold(scores, sensitivity=0.9, method="mad")
    assert mad_low_sens > mad_high_sens


def test_anomaly_categorization_types(detector_and_data):
    detector, test_df = detector_and_data
    results = detector.predict(test_df, sensitivity=0.8)

    # Should detect at least some anomalies
    detected = results[results["predicted_anomaly"] == 1]
    assert len(detected) > 0

    valid_types = {"spike", "stuck", "drift", "dropout", "multivariate", "normal"}
    types_found = set(results["predicted_type"].unique())
    assert types_found.issubset(valid_types)
