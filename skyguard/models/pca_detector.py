"""
PCA Reconstruction Error Anomaly Detector for Weather Station Sensors
Captures multi-channel correlation breakdown by projecting data onto principal subspace.
"""

from typing import List, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCADetector:
    """
    PCA-based anomaly detector using reconstruction error.
    Points deviating from the normal correlation manifold exhibit high residual norm.
    """

    def __init__(
        self,
        variance_ratio: float = 0.85,
        contamination: float = 0.04,
        random_state: int = 42,
    ):
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=variance_ratio, random_state=random_state)
        self.threshold_: float = 0.5
        self.err_min_: float = 0.0
        self.err_max_: float = 1.0
        self.feature_names_: List[str] = []

    def _get_squared_residuals(self, X: pd.DataFrame) -> np.ndarray:
        """Returns element-wise squared reconstruction residuals."""
        X_scaled = self.scaler.transform(X)
        X_proj = self.pca.inverse_transform(self.pca.transform(X_scaled))
        return (X_scaled - X_proj) ** 2

    def fit(self, X: pd.DataFrame) -> "PCADetector":
        """Fit scaler and PCA components on feature matrix."""
        self.feature_names_ = list(X.columns)
        X_scaled = self.scaler.fit_transform(X)
        self.pca.fit(X_scaled)

        X_proj = self.pca.inverse_transform(self.pca.transform(X_scaled))
        residuals = np.sum((X_scaled - X_proj) ** 2, axis=1)

        self.err_min_ = float(np.percentile(residuals, 1))
        self.err_max_ = float(np.percentile(residuals, 99))
        if self.err_max_ == self.err_min_:
            self.err_max_ += 1e-6

        norm_errors = np.clip((residuals - self.err_min_) / (self.err_max_ - self.err_min_), 0.0, 1.0)
        self.threshold_ = float(np.percentile(norm_errors, (1.0 - self.contamination) * 100))
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Returns normalized reconstruction error in [0, 1] range."""
        residuals = np.sum(self._get_squared_residuals(X), axis=1)
        norm_errors = (residuals - self.err_min_) / (self.err_max_ - self.err_min_)
        return np.clip(norm_errors, 0.0, 1.0)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Returns binary anomaly predictions (1 for anomaly, 0 for normal)."""
        t = self.threshold_ if threshold is None else threshold
        scores = self.score(X)
        return (scores >= t).astype(int)

    def get_feature_reconstruction_errors(self, X: pd.DataFrame) -> pd.DataFrame:
        """Returns per-feature squared reconstruction errors to diagnose which sensor failed."""
        sq_errors = self._get_squared_residuals(X)
        return pd.DataFrame(sq_errors, columns=self.feature_names_, index=X.index)
