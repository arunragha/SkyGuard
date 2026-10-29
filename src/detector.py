"""
SkyGuard Unified Unsupervised Anomaly Detector and Root Cause Categorizer.
Orchestrates feature extraction, multi-model scoring, dynamic thresholding, and fault archetype categorization.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.config import AppConfig, ThresholdConfig, load_config
from src.features.engineer import FeatureEngineer
from src.models.autoencoder import AutoencoderDetector
from src.models.ensemble import EnsembleMetaScorer
from src.models.isolation_forest import IsolationForestDetector
from src.models.lof import LOFDetector
from src.models.statistical import StatisticalBaselineDetector


class SkyGuardDetector:
    """
    Production-ready unsupervised weather sensor anomaly detection pipeline.
    Combines feature engineering, multi-algorithm inference, dynamic thresholding,
    and physics-informed anomaly classification.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.feature_engineer = FeatureEngineer(self.config)

        # Initialize models
        self.iso_forest = IsolationForestDetector(self.config.isolation_forest)
        self.autoencoder = AutoencoderDetector(self.config.autoencoder)
        self.statistical = StatisticalBaselineDetector(self.config.statistical)
        self.lof = LOFDetector(self.config.lof)
        self.ensemble = EnsembleMetaScorer(self.config.ensemble)

        self.threshold_cfg: ThresholdConfig = self.config.thresholding
        self.feature_cols: List[str] = []
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "SkyGuardDetector":
        """
        Fits unsupervised models on baseline / reference weather station telemetry.
        """
        engineered = self.feature_engineer.transform(df)
        self.feature_cols = self.feature_engineer.get_feature_column_names(engineered)

        X = engineered[self.feature_cols].values

        if self.config.isolation_forest.enabled:
            self.iso_forest.fit(X)
        if self.config.autoencoder.enabled:
            self.autoencoder.fit(X)
        if self.config.statistical.enabled:
            self.statistical.fit(X)
        if self.config.lof.enabled:
            self.lof.fit(X)

        self.is_fitted = True
        return self

    def predict(
        self,
        df: pd.DataFrame,
        sensitivity: Optional[float] = None,
        threshold_method: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Executes end-to-end anomaly detection on input telemetry.
        Returns DataFrame enriched with:
        - individual model scores
        - composite 'anomaly_score' [0, 1]
        - 'anomaly_threshold'
        - 'is_anomaly' (bool)
        - 'anomaly_type' (spike, stuck, drift, dropout, multivariate, normal)
        """
        if not self.is_fitted:
            self.fit(df)

        engineered = self.feature_engineer.transform(df)
        X = engineered[self.feature_cols].values

        # 1. Collect individual model scores in [0, 1]
        model_scores = {}
        out_cols = {}
        if self.config.isolation_forest.enabled:
            model_scores["isolation_forest"] = self.iso_forest.predict_scores(X)
            out_cols["score_isolation_forest"] = np.round(model_scores["isolation_forest"], 4)

        if self.config.autoencoder.enabled:
            model_scores["autoencoder"] = self.autoencoder.predict_scores(X)
            out_cols["score_autoencoder"] = np.round(model_scores["autoencoder"], 4)

        if self.config.statistical.enabled:
            model_scores["statistical"] = self.statistical.predict_scores(X)
            out_cols["score_statistical"] = np.round(model_scores["statistical"], 4)

        if self.config.lof.enabled:
            model_scores["lof"] = self.lof.predict_scores(X)
            out_cols["score_lof"] = np.round(model_scores["lof"], 4)

        # 2. Ensemble meta-scoring
        composite_score = self.ensemble.combine(model_scores)
        out_cols["anomaly_score"] = np.round(composite_score, 4)

        # 3. Dynamic Thresholding
        sens = sensitivity if sensitivity is not None else self.threshold_cfg.default_sensitivity
        method = threshold_method or self.threshold_cfg.method
        threshold = self.compute_dynamic_threshold(composite_score, sensitivity=sens, method=method)

        out_cols["anomaly_threshold"] = round(float(threshold), 4)
        is_anom = composite_score >= threshold
        out_cols["predicted_anomaly"] = is_anom.astype(int)

        # 4. Categorize Anomaly Archetypes
        categories, diagnosed_sensors = self._categorize_anomalies(
            engineered, is_anom, model_scores
        )
        out_cols["predicted_type"] = categories
        out_cols["diagnosed_sensor"] = diagnosed_sensors

        # Concat output columns
        pred_df = pd.DataFrame(out_cols, index=engineered.index)
        return pd.concat([engineered, pred_df], axis=1)

    def compute_dynamic_threshold(
        self,
        scores: np.ndarray,
        sensitivity: float = 0.5,
        method: str = "percentile",
    ) -> float:
        """
        Computes dynamic anomaly threshold based on score distribution and sensitivity parameter.
        Sensitivity in (0, 1]: Higher sensitivity -> lower threshold -> more alarms.
        """
        sens = float(np.clip(sensitivity, 0.01, 1.0))
        scores = np.asarray(scores, dtype=float)

        if method == "mad":
            # Median Absolute Deviation (MAD) dynamic threshold
            med = float(np.median(scores))
            abs_dev = np.abs(scores - med)
            mad = float(np.median(abs_dev)) * 1.4826
            if mad < 1e-4:
                mad = float(np.std(scores)) or 0.05

            # Multiplier scales inversely with sensitivity
            k = self.threshold_cfg.mad_multiplier_base * (1.0 / (sens ** 0.5))
            threshold = med + (k * 0.3 * mad)
        else:
            # Percentile-based dynamic threshold
            # Range: from 99.5th percentile down to 88.0th percentile based on sensitivity
            base_q = self.threshold_cfg.percentile_base_quantile
            effective_quantile = 1.0 - (sens * (1.0 - base_q))
            effective_quantile = np.clip(effective_quantile, 0.50, 0.999)
            threshold = float(np.quantile(scores, effective_quantile))

        return float(np.clip(threshold, 0.05, 0.98))

    def _categorize_anomalies(
        self,
        df: pd.DataFrame,
        anomaly_mask: np.ndarray,
        model_scores: Dict[str, np.ndarray],
    ) -> Tuple[List[str], List[str]]:
        """
        Diagnoses anomaly root cause into:
        'spike', 'stuck', 'drift', 'dropout', 'multivariate', or 'normal'.
        """
        n = len(df)
        categories = ["normal"] * n
        diagnosed_sensors = ["none"] * n

        sensors = list(self.config.sensors.keys())

        for idx in range(n):
            if not anomaly_mask[idx]:
                continue

            # Check individual candidate fault conditions
            is_dropout, drop_sensor = self._check_dropout(df, idx, sensors)
            if is_dropout:
                categories[idx] = "dropout"
                diagnosed_sensors[idx] = drop_sensor
                continue

            is_stuck, stuck_sensor = self._check_stuck(df, idx, sensors)
            if is_stuck:
                categories[idx] = "stuck"
                diagnosed_sensors[idx] = stuck_sensor
                continue

            is_spike, spike_sensor = self._check_spike(df, idx, sensors)
            if is_spike:
                categories[idx] = "spike"
                diagnosed_sensors[idx] = spike_sensor
                continue

            is_drift, drift_sensor = self._check_drift(df, idx, sensors)
            if is_drift:
                categories[idx] = "drift"
                diagnosed_sensors[idx] = drift_sensor
                continue

            # If none of the univariate signatures trigger, or physical breakdown flag is set
            categories[idx] = "multivariate"
            diagnosed_sensors[idx] = self._isolate_multivariate_sensor(df, idx, sensors)

        return categories, diagnosed_sensors

    def _check_dropout(
        self, df: pd.DataFrame, idx: int, sensors: List[str]
    ) -> Tuple[bool, str]:
        for s in sensors:
            if s not in df.columns:
                continue
            val = df.at[idx, s]
            s_cfg = self.config.sensors[s]
            # Zero dropout on sensors that don't rest at zero, or value below physical min
            if val < s_cfg.min:
                return True, s
            if s in ["barometric_pressure", "temperature", "humidity"] and val == 0.0:
                return True, s
        return False, "none"

    def _check_stuck(
        self, df: pd.DataFrame, idx: int, sensors: List[str]
    ) -> Tuple[bool, str]:
        for s in sensors:
            spread_col = f"{s}_roll_spread_6"
            if spread_col in df.columns:
                spread = df.at[idx, spread_col]
                # Flatline: 0 spread over 6 intervals (1 hour)
                if spread < 1e-4 and s in ["temperature", "humidity", "barometric_pressure", "wind_speed"]:
                    return True, s
        return False, "none"

    def _check_spike(
        self, df: pd.DataFrame, idx: int, sensors: List[str]
    ) -> Tuple[bool, str]:
        for s in sensors:
            roc_col = f"{s}_roc"
            z_col = f"{s}_local_zscore_6"
            if roc_col in df.columns:
                roc = abs(df.at[idx, roc_col])
                max_roc = self.config.sensors[s].max_rate_of_change
                if roc > max_roc:
                    return True, s
            if z_col in df.columns and abs(df.at[idx, z_col]) > 3.8:
                return True, s
        return False, "none"

    def _check_drift(
        self, df: pd.DataFrame, idx: int, sensors: List[str]
    ) -> Tuple[bool, str]:
        for s in sensors:
            z_short = abs(df.get(f"{s}_local_zscore_6", pd.Series(0)).iloc[idx])
            z_long = abs(df.get(f"{s}_local_zscore_144", pd.Series(0)).iloc[idx])
            roc = abs(df.get(f"{s}_roc", pd.Series(0)).iloc[idx])
            # Drift signature: persistent deviation on 24h baseline with small instant step change
            if z_long > 2.5 and roc < self.config.sensors[s].max_rate_of_change * 0.5:
                return True, s
        return False, "none"

    def _isolate_multivariate_sensor(
        self, df: pd.DataFrame, idx: int, sensors: List[str]
    ) -> str:
        if df.get("nighttime_solar_anomaly", pd.Series(0)).iloc[idx] == 1.0:
            return "solar_radiation"
        if df.get("physics_rh_violation", pd.Series(0)).iloc[idx] == 1.0:
            return "temperature_humidity"
        if abs(df.get("pressure_anomaly_delta", pd.Series(0)).iloc[idx]) > 15.0:
            return "barometric_pressure"

        # Find sensor with highest local z-score
        max_z = 0.0
        candidate = "multivariate_joint"
        for s in sensors:
            z_val = abs(df.get(f"{s}_local_zscore_36", pd.Series(0)).iloc[idx])
            if z_val > max_z:
                max_z = z_val
                candidate = s
        return candidate
