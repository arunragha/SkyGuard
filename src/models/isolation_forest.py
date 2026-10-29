"""
Isolation Forest detector wrapper producing calibrated per-timestamp anomaly scores in [0, 1].
"""

from typing import Optional
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler

from src.config import IsolationForestConfig


class IsolationForestDetector:
    """
    Unsupervised tree-based anomaly detector. Isolates anomalies by recursive random partitioning.
    Outputs a continuous anomaly score in [0, 1].
    """

    def __init__(self, config: Optional[IsolationForestConfig] = None):
        self.config = config or IsolationForestConfig()
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.is_fitted = False
        self._score_min = -0.5
        self._score_max = 0.5

    def fit(self, X: np.ndarray) -> "IsolationForestDetector":
        """Fits scaler and Isolation Forest on training feature matrix."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

        # Calibrate score range on training data
        raw_scores = -self.model.decision_function(X_scaled)
        self._score_min = float(np.percentile(raw_scores, 1.0))
        self._score_max = float(np.percentile(raw_scores, 99.0))
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + 1.0
        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Computes calibrated per-sample anomaly scores in [0, 1], where 1.0 is most anomalous.
        """
        if not self.is_fitted:
            self.fit(X)

        X_scaled = self.scaler.transform(X)
        raw_scores = -self.model.decision_function(X_scaled)

        # Normalize to [0, 1] using calibrated percentiles
        norm_scores = (raw_scores - self._score_min) / (self._score_max - self._score_min)
        return np.clip(norm_scores, 0.0, 1.0)
