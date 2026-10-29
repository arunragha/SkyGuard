"""
SkyGuard Ensemble Meta-Detector and Root Cause Analysis (RCA) Engine
Combines tree-based isolation, subspace reconstruction error, local density, and physics rules.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from skyguard.models.isolation_forest import IsolationForestDetector
from skyguard.models.pca_detector import PCADetector
from skyguard.models.lof_detector import LOFDetector
from skyguard.models.rules import PhysicsRuleDetector


class SkyGuardEnsemble:
    """
    Unified unsupervised anomaly detection ensemble.
    Aggregates diverse detectors into a calibrated severity index and performs root-cause analysis.
    """

    DEFAULT_WEIGHTS = {
        "isolation_forest": 0.35,
        "pca": 0.30,
        "lof": 0.15,
        "physics_rules": 0.20,
    }

    SEVERITY_THRESHOLDS = {
        "NORMAL": 0.45,
        "LOW": 0.60,
        "WARNING": 0.75,
        "CRITICAL": 1.00,
    }

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        contamination: float = 0.04,
        random_state: int = 42,
    ):
        self.weights = weights or self.DEFAULT_WEIGHTS
        # Normalize weights
        total_w = sum(self.weights.values())
        self.weights = {k: v / total_w for k, v in self.weights.items()}

        self.contamination = contamination
        self.random_state = random_state

        self.iforest = IsolationForestDetector(contamination=contamination, random_state=random_state)
        self.pca = PCADetector(variance_ratio=0.85, contamination=contamination, random_state=random_state)
        self.lof = LOFDetector(contamination=contamination)
        self.rules = PhysicsRuleDetector(flatline_window=6)

        self.threshold_: float = 0.45
        self.feature_names_: List[str] = []

    def fit(self, X: pd.DataFrame, raw_df: Optional[pd.DataFrame] = None) -> "SkyGuardEnsemble":
        """
        Fit all constituent unsupervised models.
        """
        self.feature_names_ = list(X.columns)
        self.iforest.fit(X)
        self.pca.fit(X)
        self.lof.fit(X)
        if raw_df is not None:
            self.rules.fit(raw_df)
        else:
            self.rules.fit(X)

        # Calibrate ensemble threshold on fitting data
        scores = self.score(X, raw_df if raw_df is not None else X)["ensemble_score"].values
        self.threshold_ = float(np.percentile(scores, (1.0 - self.contamination) * 100))
        return self

    def score(self, X: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates individual detector scores and the aggregated ensemble score.
        Returns a DataFrame with individual scores, ensemble score, severity, and diagnosis.
        """
        n = len(X)
        s_iforest = self.iforest.score(X)
        s_pca = self.pca.score(X)
        s_lof = self.lof.score(X)
        s_rules, diagnoses = self.rules.score_and_diagnose(raw_df)

        w = self.weights
        ensemble_score = (
            w["isolation_forest"] * s_iforest
            + w["pca"] * s_pca
            + w["lof"] * s_lof
            + w["physics_rules"] * s_rules
        )

        # Classify Severity Levels
        severities = []
        for s in ensemble_score:
            if s < self.SEVERITY_THRESHOLDS["NORMAL"]:
                severities.append("NORMAL")
            elif s < self.SEVERITY_THRESHOLDS["LOW"]:
                severities.append("LOW")
            elif s < self.SEVERITY_THRESHOLDS["WARNING"]:
                severities.append("WARNING")
            else:
                severities.append("CRITICAL")

        res_df = pd.DataFrame({
            "score_iforest": np.round(s_iforest, 4),
            "score_pca": np.round(s_pca, 4),
            "score_lof": np.round(s_lof, 4),
            "score_rules": np.round(s_rules, 4),
            "ensemble_score": np.round(ensemble_score, 4),
            "severity": severities,
            "rule_diagnosis": diagnoses,
            "is_anomaly_pred": (ensemble_score >= self.threshold_).astype(int),
        }, index=X.index)

        return res_df

    def predict(self, X: pd.DataFrame, raw_df: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Returns binary predictions (1 for anomaly, 0 for normal).
        """
        t = self.threshold_ if threshold is None else threshold
        scores = self.score(X, raw_df)["ensemble_score"].values
        return (scores >= t).astype(int)

    def diagnose_root_cause(
        self,
        row_idx: int,
        X: pd.DataFrame,
        raw_df: pd.DataFrame
    ) -> Dict[str, any]:
        """
        Performs Root Cause Analysis (RCA) on a single anomalous timestamp.
        Identifies the primary faulty sensor and failure mechanism.
        """
        row_features = X.iloc[[row_idx]]
        row_raw = raw_df.iloc[[row_idx]]

        scores_df = self.score(row_features, row_raw)
        row_score = scores_df["ensemble_score"].iloc[0]
        rule_diag = scores_df["rule_diagnosis"].iloc[0]

        # 1. Feature attribution using PCA reconstruction residual
        pca_residuals = self.pca.get_feature_reconstruction_errors(row_features).iloc[0]
        top_features = pca_residuals.sort_values(ascending=False).head(5)

        # Map top feature names to physical sensors
        sensor_mapping = {
            "temperature_c": "Temperature Sensor",
            "dew_point_c": "Temperature / Humidity Sensor",
            "dew_point_depression_c": "Psychrometric Humidity / Temp",
            "humidity_pct": "Relative Humidity Sensor",
            "pressure_hpa": "Barometric Pressure Sensor",
            "wind_speed_ms": "Anemometer (Wind Speed)",
            "wind_direction_deg": "Wind Vane (Direction)",
            "solar_radiation_wm2": "Pyranometer (Solar Radiation)",
            "precipitation_mm": "Rain Gauge Sensor",
            "battery_voltage_v": "Station Power / Battery System",
            "internal_temp_c": "Internal Enclosure Electronics",
        }

        primary_sensor = "Unknown"
        for feat in top_features.index:
            base_col = feat.split("_diff")[0].split("_roll")[0]
            if base_col in sensor_mapping:
                primary_sensor = sensor_mapping[base_col]
                break

        # Determine fault mode
        fault_mode = "Multivariate Anomaly"
        if "Physical Boundary" in rule_diag or "Night" in rule_diag:
            fault_mode = "Physical Constraint Violation"
        elif "Sensor Freeze" in rule_diag:
            fault_mode = "Stuck Sensor / Flatline"
        elif "Battery Brownout" in rule_diag:
            fault_mode = "Power Supply Failure / Brownout"
        elif any("diff1" in f for f in top_features.index[:2]):
            fault_mode = "High-Magnitude Spike / Impulse Glitch"
        elif scores_df["score_pca"].iloc[0] > 0.65:
            fault_mode = "Cross-Sensor Correlation Breakdown"

        return {
            "timestamp": str(raw_df["timestamp"].iloc[row_idx]) if "timestamp" in raw_df.columns else str(row_idx),
            "ensemble_score": float(row_score),
            "severity": scores_df["severity"].iloc[0],
            "fault_mode": fault_mode,
            "primary_culprit_sensor": primary_sensor,
            "rule_diagnosis": rule_diag,
            "top_contributing_features": top_features.to_dict(),
        }
