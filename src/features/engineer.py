"""
Meteorological and time series feature engineering engine for weather station sensors.
Computes rolling window statistics, rates of change, cross-sensor thermodynamic couplings,
and cyclical time-of-day / seasonal encodings.
"""

from typing import List, Optional
import numpy as np
import pandas as pd

from src.config import AppConfig, FeatureConfig, load_config


class FeatureEngineer:
    """
    Transforms raw multivariate weather sensor time series into rich feature matrices
    suitable for unsupervised anomaly detection.
    """

    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or load_config()
        self.feature_cfg: FeatureConfig = self.config.features
        self.sensor_names: List[str] = list(self.config.sensors.keys())

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input DataFrame with sensor readings into an engineered feature DataFrame.
        Preserves original columns and appends engineered features.
        """
        df = df.copy()

        # If multi-station, engineer per station to avoid cross-station boundary leakage
        if "station_id" in df.columns and df["station_id"].nunique() > 1:
            engineered_groups = []
            for _, st_df in df.groupby("station_id", sort=False):
                engineered_groups.append(self._engineer_single_station(st_df))
            res = pd.concat(engineered_groups, axis=0).reset_index(drop=True)
        else:
            res = self._engineer_single_station(df)

        return res

    def _engineer_single_station(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        new_cols = {}

        # Identify existing sensors in df
        present_sensors = [s for s in self.sensor_names if s in df.columns]

        # 1. Rolling Statistics (mean, std, min, max, spread)
        for w in self.feature_cfg.rolling_windows:
            for s in present_sensors:
                series = df[s].astype(float)
                rolling = series.rolling(window=w, min_periods=1)

                mean_val = rolling.mean()
                std_val = rolling.std().fillna(0.0)
                min_val = rolling.min()
                max_val = rolling.max()

                new_cols[f"{s}_roll_mean_{w}"] = mean_val
                new_cols[f"{s}_roll_std_{w}"] = std_val
                new_cols[f"{s}_roll_spread_{w}"] = (max_val - min_val).fillna(0.0)

                # Local z-score relative to rolling mean
                denom = std_val.replace(0.0, 1e-5)
                new_cols[f"{s}_local_zscore_{w}"] = ((series - mean_val) / denom).fillna(0.0)

        # 2. Rate of Change (1st and 2nd differences)
        if self.feature_cfg.calculate_rates_of_change:
            for s in present_sensors:
                s_float = df[s].astype(float)
                diff1 = s_float.diff().fillna(0.0)
                diff2 = diff1.diff().fillna(0.0)
                new_cols[f"{s}_roc"] = diff1
                new_cols[f"{s}_accel"] = diff2

        # 3. Cross-Sensor Thermodynamic & Physical Couplings
        if self.feature_cfg.calculate_thermodynamics:
            has_temp = "temperature" in df.columns
            has_rh = "humidity" in df.columns
            has_pressure = "barometric_pressure" in df.columns
            has_solar = "solar_radiation" in df.columns

            if has_temp and has_rh:
                T = df["temperature"].astype(float).values
                RH = np.clip(df["humidity"].astype(float).values, 0.1, 100.0)

                # Magnus-Tetens Dew Point formula: Td = (b * gamma) / (a - gamma)
                a, b = 17.27, 237.7
                gamma = (a * T) / (b + T) + np.log(RH / 100.0)
                dew_point = (b * gamma) / (a - gamma)
                new_cols["dew_point"] = np.round(dew_point, 2)

                # Dew point depression: T - Td (must be >= 0 physically)
                depression = T - dew_point
                new_cols["dew_point_depression"] = np.round(depression, 2)
                # Physical violation flag (< -0.5°C accounting for sensor tolerances)
                new_cols["physics_rh_violation"] = (depression < -0.5).astype(float)

                # Vapor Pressure Deficit (VPD in kPa)
                # Saturation vapor pressure es = 0.61078 * exp(17.27 * T / (T + 237.3))
                es = 0.61078 * np.exp((17.27 * T) / (T + 237.3))
                ea = es * (RH / 100.0)
                vpd = np.maximum(0.0, es - ea)
                new_cols["vapor_pressure_deficit"] = np.round(vpd, 3)

            if has_solar and has_temp:
                # Solar vs Temp coupling: diurnal heating
                # Solar radiation at night check (solar elevation ~ 0)
                if "timestamp" in df.columns:
                    ts = pd.to_datetime(df["timestamp"])
                    hours = ts.dt.hour + ts.dt.minute / 60.0
                    night_mask = (hours < 4.5) | (hours > 21.0)
                    new_cols["nighttime_solar_anomaly"] = (
                        night_mask & (df["solar_radiation"] > 15.0)
                    ).astype(float)

            if has_pressure:
                # Barometric pressure deviation from 24-hour rolling mean
                w24 = 144  # 24 hours at 10-min cadence
                p_series = df["barometric_pressure"].astype(float)
                p_mean = p_series.rolling(window=w24, min_periods=1).mean()
                new_cols["pressure_anomaly_delta"] = (p_series - p_mean).fillna(0.0)

        # 4. Cyclical Temporal Encodings (diurnal & seasonal)
        if self.feature_cfg.cyclical_encodings and "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            hour_float = ts.dt.hour + ts.dt.minute / 60.0
            day_of_year = ts.dt.dayofyear

            # Diurnal 24-hour cycle
            new_cols["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
            new_cols["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)

            # Annual seasonal cycle
            new_cols["season_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
            new_cols["season_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

        # Concatenate original df and newly engineered columns
        new_df = pd.DataFrame(new_cols, index=df.index)
        combined = pd.concat([df, new_df], axis=1)

        # Fill any remaining NaNs from rolling warm-up with column median or zero
        numeric_cols = combined.select_dtypes(include=[np.number]).columns
        combined[numeric_cols] = combined[numeric_cols].bfill().fillna(0.0)

        return combined

    def get_feature_column_names(self, df: pd.DataFrame) -> List[str]:
        """Extracts the subset of columns that constitute numerical features for model inference."""
        excluded = [
            "timestamp",
            "station_id",
            "is_anomaly",
            "anomaly_type",
            "faulty_sensor",
            "predicted_anomaly",
            "anomaly_score",
            "predicted_type",
        ]
        return [c for c in df.columns if c not in excluded and np.issubdtype(df[c].dtype, np.number)]
