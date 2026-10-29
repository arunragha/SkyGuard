"""
Local Outlier Factor (LOF) density-based anomaly detector wrapper in [0, 1].
"""

from typing import Optional
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler

from src.config import LOFConfig


class LOFDetector:
    """
    Local Outlier Factor detector measuring the local density deviation of a sample
    with respect to its k-nearest neighbors.
    """

    def __init__(self, config: Optional[LOFConfig] = None):
        self.config = config or LOFConfig()
        self.scaler = RobustScaler()
        self.model = LocalOutlierFactor(
            n_neighbors=self.config.n_neighbors,
            contamination=self.config.contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.is_fitted = False
        self._score_min = -1.5
        self._score_max = 0.0

    def fit(self, X: np.ndarray) -> "LOFDetector":
        """Fits scaler and LOF model."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.is_fitted = True

        raw_scores = -self.model.decision_function(X_scaled)
        self._score_min = float(np.percentile(raw_scores, 2.0))
        self._score_max = float(np.percentile(raw_scores, 98.0))
        if self._score_max <= self._score_min:
            self._score_max = self._score_min + 1.0

        return self

    def predict_scores(self, X: np.ndarray) -> np.ndarray:
        """Computes calibrated LOF anomaly score in [0, 1]."""
        if not self.is_fitted:
            self.fit(X)

        X_scaled = self.scaler.transform(X)
        raw_scores = -self.model.decision_function(X_scaled)

        norm_scores = (raw_scores - self._score_min) / (self._score_max - self._score_min)
        return np.clip(norm_scores, 0.0, 1.0)
