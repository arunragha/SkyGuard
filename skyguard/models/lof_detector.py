"""
Local Outlier Factor (LOF) Unsupervised Anomaly Detector for Weather Stations
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler


class LOFDetector:
    """
    Density-based anomaly detector using Local Outlier Factor.
    Identifies samples that have substantially lower density than their local neighbors.
    """

    def __init__(
        self,
        n_neighbors: int = 25,
        contamination: float = 0.04,
    ):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.model = LocalOutlierFactor(
            n_neighbors=n_neighbors,
            contamination=contamination,
            novelty=True,
            n_jobs=1,
        )
        self.threshold_: float = 0.5
        self.score_min_: float = 0.0
        self.score_max_: float = 1.0

    def fit(self, X: pd.DataFrame) -> "LOFDetector":
        """Fit scaler and LOF model with novelty detection enabled."""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        raw_scores = -self.model.score_samples(X_scaled)
        self.score_min_ = float(np.percentile(raw_scores, 1))
        self.score_max_ = float(np.percentile(raw_scores, 99))
        if self.score_max_ == self.score_min_:
            self.score_max_ += 1e-6

        norm_scores = np.clip((raw_scores - self.score_min_) / (self.score_max_ - self.score_min_), 0.0, 1.0)
        self.threshold_ = float(np.percentile(norm_scores, (1.0 - self.contamination) * 100))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns normalized anomaly score in [0, 1] range.
        """
        X_scaled = self.scaler.transform(X)
        raw_scores = -self.model.score_samples(X_scaled)
        norm_scores = (raw_scores - self.score_min_) / (self.score_max_ - self.score_min_)
        return np.clip(norm_scores, 0.0, 1.0)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Returns binary anomaly predictions (1 for anomaly, 0 for normal).
        """
        t = self.threshold_ if threshold is None else threshold
        scores = self.score(X)
        return (scores >= t).astype(int)
