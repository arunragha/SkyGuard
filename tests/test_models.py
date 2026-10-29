"""
Unit tests for unsupervised detection models: Isolation Forest, Autoencoder, Statistical, LOF, and Ensemble.
"""

import numpy as np
import pytest

from src.models.autoencoder import AutoencoderDetector
from src.models.ensemble import EnsembleMetaScorer
from src.models.isolation_forest import IsolationForestDetector
from src.models.lof import LOFDetector
from src.models.statistical import StatisticalBaselineDetector


@pytest.fixture
def dummy_features():
    rng = np.random.default_rng(42)
    # Normal cluster
    X = rng.normal(loc=0.0, scale=1.0, size=(200, 10))
    # Inject 5 extreme outliers
    X[10:15, :] += 8.0
    return X


def test_isolation_forest(dummy_features):
    detector = IsolationForestDetector()
    detector.fit(dummy_features)
    scores = detector.predict_scores(dummy_features)

    assert len(scores) == len(dummy_features)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
    # Outliers should have noticeably higher scores than inliers
    assert scores[12] > np.median(scores)


def test_autoencoder(dummy_features):
    from src.config import AutoencoderConfig
    cfg = AutoencoderConfig(epochs=3, batch_size=32)
    detector = AutoencoderDetector(config=cfg)
    detector.fit(dummy_features)
    scores = detector.predict_scores(dummy_features)

    assert len(scores) == len(dummy_features)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
    assert scores[12] > np.median(scores)


def test_statistical_baseline(dummy_features):
    detector = StatisticalBaselineDetector()
    detector.fit(dummy_features)
    scores = detector.predict_scores(dummy_features)

    assert len(scores) == len(dummy_features)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
    assert scores[12] > np.median(scores)


def test_lof_detector(dummy_features):
    detector = LOFDetector()
    detector.fit(dummy_features)
    scores = detector.predict_scores(dummy_features)

    assert len(scores) == len(dummy_features)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()
    assert scores[12] > np.median(scores)


def test_ensemble_meta_scorer():
    ensemble = EnsembleMetaScorer()
    scores = {
        "isolation_forest": np.array([0.1, 0.9, 0.3]),
        "autoencoder": np.array([0.2, 0.8, 0.4]),
        "statistical": np.array([0.05, 0.95, 0.2]),
        "lof": np.array([0.15, 0.85, 0.25]),
    }
    combined = ensemble.combine(scores)
    assert len(combined) == 3
    assert combined[1] > combined[0]
    assert combined[1] > combined[2]
    assert (combined >= 0.0).all() and (combined <= 1.0).all()
