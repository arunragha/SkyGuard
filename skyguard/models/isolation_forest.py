"""
Isolation Forest Unsupervised Anomaly Detector for Weather Sensors
"""

from typing import Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler


class IsolationForestDetector:
    """
    Isolation Forest anomaly detector fitted on multi-sensor engineered features.
    Normalizes decision scores into [0, 1] anomaly likelihood.
    """

    def __init__(
        self,
        n_estimators: int = 150,
        contamination: float = 0.04,
        max_samples: str = "auto",
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.scaler = RobustScaler()
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=1,
        )
        self.threshold_: float = 0.5
        self.score_min_: float = 0.0
        self.score_max_: float = 1.0

    def fit(self, X: pd.DataFrame) -> "IsolationForestDetector":
        """Fit scaler and Isolation Forest model, calibrating threshold to contamination."""
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
        Returns continuous anomaly score in range [0, 1] (higher = more anomalous).
        """
        X_scaled = self.scaler.transform(X)
        raw_scores = -self.model.score_samples(X_scaled)
        norm_scores = (raw_scores - self.score_min_) / (self.score_max_ - self.score_min_)
        return np.clip(norm_scores, 0.0, 1.0)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Returns binary anomaly predictions: 1 for anomalous, 0 for normal.
        """
        t = self.threshold_ if threshold is None else threshold
        scores = self.score(X)
        return (scores >= t).astype(int)
