"""
SkyGuard End-to-End Anomaly Detection Pipeline
Provides unified fit, predict, batch_process, and export workflows.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd

from skyguard.features.engineer import WeatherFeatureEngineer
from skyguard.models.ensemble import SkyGuardEnsemble


class SkyGuardPipeline:
    """
    End-to-End Automated Pipeline for Weather Station Sensor Anomaly Detection.
    Handles feature engineering, ensemble fitting, inference, and root cause diagnosis.
    """

    def __init__(
        self,
        contamination: float = 0.04,
        rolling_windows: Tuple[int, ...] = (6, 18),
        weights: Optional[Dict[str, float]] = None,
        random_state: int = 42,
    ):
        self.engineer = WeatherFeatureEngineer(rolling_windows=rolling_windows)
        self.ensemble = SkyGuardEnsemble(
            weights=weights,
            contamination=contamination,
            random_state=random_state,
        )
        self.feature_cols_: List[str] = []

    def fit(self, df: pd.DataFrame) -> "SkyGuardPipeline":
        """Fits feature engineer and unsupervised detectors on telemetry data."""
        feat_df = self.engineer.fit_transform(df)
        self.feature_cols_ = self.engineer.get_feature_names()
        self.ensemble.fit(feat_df[self.feature_cols_], df)
        return self

    def process(self, df: pd.DataFrame, threshold: Optional[float] = None) -> pd.DataFrame:
        """Runs feature engineering and anomaly inference, appending results to df."""
        if not self.feature_cols_:
            self.fit(df)

        feat_df = self.engineer.transform(df)
        scores_df = self.ensemble.score(feat_df[self.feature_cols_], df)

        if threshold is not None:
            scores_df["is_anomaly_pred"] = (scores_df["ensemble_score"] >= threshold).astype(int)

        return pd.concat([df.copy(), scores_df], axis=1)

    def diagnose(self, row_idx: int, df: pd.DataFrame) -> Dict[str, Any]:
        """Performs detailed root cause analysis on a specific row."""
        feat_df = self.engineer.transform(df)
        return self.ensemble.diagnose_root_cause(row_idx, feat_df[self.feature_cols_], df)
