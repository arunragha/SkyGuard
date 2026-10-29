"""
SkyGuard Meteorological and Temporal Feature Engineering Engine
Transforms raw weather station sensor readings into thermodynamic, temporal, and dynamic features
for unsupervised anomaly detection.
"""

from typing import List, Optional, Tuple
import numpy as np
import pandas as pd


class WeatherFeatureEngineer:
    """
    Computes thermodynamic, rate-of-change, rolling statistical, and cyclical features
    designed specifically to expose sensor malfunctions, flatlines, and physical violations.
    """

    RAW_SENSOR_COLS = [
        "temperature_c",
        "humidity_pct",
        "pressure_hpa",
        "wind_speed_ms",
        "wind_direction_deg",
        "solar_radiation_wm2",
        "precipitation_mm",
        "battery_voltage_v",
        "internal_temp_c",
    ]

    def __init__(
        self,
        rolling_windows: Tuple[int, ...] = (6, 18),  # 6 steps = 1 hr, 18 steps = 3 hrs (for 10-min interval)
        include_differencing: bool = True,
        include_thermodynamics: bool = True,
        include_cyclical_time: bool = True,
    ):
        self.rolling_windows = rolling_windows
        self.include_differencing = include_differencing
        self.include_thermodynamics = include_thermodynamics
        self.include_cyclical_time = include_cyclical_time
        self.feature_names_: List[str] = []

    def fit(self, df: pd.DataFrame, y=None):
        """No-op for stateless feature engineering, returns self for scikit-learn compatibility."""
        return self

    DEFAULT_NOMINALS = {
        "temperature_c": 20.0,
        "humidity_pct": 50.0,
        "pressure_hpa": 1013.25,
        "wind_speed_ms": 2.0,
        "wind_direction_deg": 180.0,
        "solar_radiation_wm2": 0.0,
        "precipitation_mm": 0.0,
        "battery_voltage_v": 13.0,
        "internal_temp_c": 22.0,
    }

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms input telemetry DataFrame into an enriched feature matrix.
        Preserves original columns and appends engineered features.
        """
        feat_df = df.copy()

        # Ensure all raw sensor channels exist with fallback nominals
        for col in self.RAW_SENSOR_COLS:
            if col not in feat_df.columns:
                feat_df[col] = self.DEFAULT_NOMINALS.get(col, 0.0)

        # 1. Cyclical Time Features
        if self.include_cyclical_time and "timestamp" in feat_df.columns:
            ts = pd.to_datetime(feat_df["timestamp"])
            hour_float = ts.dt.hour + ts.dt.minute / 60.0
            feat_df["hour_sin"] = np.sin(2 * np.pi * hour_float / 24.0)
            feat_df["hour_cos"] = np.cos(2 * np.pi * hour_float / 24.0)
            feat_df["day_sin"] = np.sin(2 * np.pi * ts.dt.dayofyear / 365.25)
            feat_df["day_cos"] = np.cos(2 * np.pi * ts.dt.dayofyear / 365.25)

        # 2. Thermodynamic and Meteorological Domain Features
        if self.include_thermodynamics:
            T = feat_df["temperature_c"]
            RH = np.clip(feat_df["humidity_pct"], 1e-3, 150.0)
            P = feat_df["pressure_hpa"]

            # Magnus-Tetens formula for Dew Point (approximate)
            a = 17.27
            b = 237.7
            gamma = (a * T) / (b + T) + np.log(RH / 100.0)
            # Avoid division by zero
            denom = a - gamma
            denom = np.where(np.abs(denom) < 1e-4, 1e-4, denom)
            dew_point = (b * gamma) / denom
            feat_df["dew_point_c"] = np.round(dew_point, 2)

            # Dew Point Depression: T - Td (Physically, T >= Td; negative is unphysical!)
            feat_df["dew_point_depression_c"] = T - feat_df["dew_point_c"]

            # Saturated Vapor Pressure (es) and Actual Vapor Pressure (ea) in kPa
            es = 0.61078 * np.exp((17.27 * T) / (T + 237.3))
            ea = es * (RH / 100.0)
            # Vapor Pressure Deficit (VPD)
            feat_df["vpd_kpa"] = np.maximum(0.0, es - ea)

            # Potential Temperature (theta) in Kelvin (dry adiabatic invariant)
            T_kelvin = T + 273.15
            feat_df["potential_temp_k"] = T_kelvin * (1000.0 / np.maximum(P, 500.0)) ** 0.286

            # Temperature gradient: Internal enclosure vs Ambient
            feat_df["temp_enclosure_delta_c"] = feat_df["internal_temp_c"] - T

            # Theoretical night-time solar violation flag
            if "solar_radiation_wm2" in feat_df.columns and "timestamp" in feat_df.columns:
                hours = pd.to_datetime(feat_df["timestamp"]).dt.hour
                is_deep_night = (hours >= 21) | (hours <= 4)
                feat_df["unphysical_night_solar"] = np.where(
                    is_deep_night & (feat_df["solar_radiation_wm2"] > 10.0),
                    feat_df["solar_radiation_wm2"],
                    0.0
                )

        # 3. Dynamic Differencing (Rate-of-Change & Acceleration)
        if self.include_differencing:
            diff_targets = ["temperature_c", "pressure_hpa", "humidity_pct", "wind_speed_ms", "battery_voltage_v"]
            for col in diff_targets:
                if col in feat_df.columns:
                    # 1st order difference (velocity)
                    d1 = feat_df[col].diff().fillna(0.0)
                    feat_df[f"{col}_diff1"] = d1
                    # 2nd order difference (acceleration / impulse)
                    d2 = d1.diff().fillna(0.0)
                    feat_df[f"{col}_diff2"] = d2

        # 4. Rolling Volatility and Spread (Crucial for Flatline & Sudden Outbursts)
        spread_targets = ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms"]
        for w in self.rolling_windows:
            for col in spread_targets:
                if col in feat_df.columns:
                    roll = feat_df[col].rolling(window=w, min_periods=2)
                    # Rolling standard deviation (volatility)
                    std_series = roll.std().bfill().fillna(0.0)
                    feat_df[f"{col}_roll_std_{w}"] = std_series

                    # Rolling range (max - min): exactly 0 or near-0 indicates a stuck-at flatline
                    max_series = roll.max()
                    min_series = roll.min()
                    spread = (max_series - min_series).bfill().fillna(1.0)
                    feat_df[f"{col}_roll_spread_{w}"] = spread

                    # Rolling local z-score: (x - roll_mean) / (roll_std + eps)
                    mean_series = roll.mean().bfill()
                    feat_df[f"{col}_roll_z_{w}"] = (feat_df[col] - mean_series) / (std_series + 1e-5)

        # Forward fill, backward fill, and zero-fill any edge NaNs (e.g. single-row inference)
        feat_df = feat_df.bfill().ffill().fillna(0.0)

        # Record feature names excluding metadata and target columns
        exclude_cols = {"timestamp", "station_id", "is_anomaly", "anomaly_type", "faulty_sensor"}
        self.feature_names_ = [c for c in feat_df.columns if c not in exclude_cols]

        return feat_df

    def fit_transform(self, df: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit and transform convenience method."""
        return self.fit(df, y).transform(df)

    def get_feature_names(self) -> List[str]:
        """Returns the list of engineered feature column names."""
        return self.feature_names_
