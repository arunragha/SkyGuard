"""
Physical Domain Rules and Sensor Freeze (Flatline) Detector
Enforces physical laws, thermodynamic consistency, and mechanical zero-variance checks.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class PhysicsRuleDetector:
    """
    Rule-based physics guardrail and sensor freeze detector for weather station telemetry.
    Checks physical boundaries, thermodynamics, day/night solar envelope, and zero-variance stuck sensors.
    """

    # Reasonable climatological physical limits
    SENSOR_BOUNDS = {
        "temperature_c": (-50.0, 58.0),
        "humidity_pct": (0.0, 100.0),
        "pressure_hpa": (870.0, 1085.0),
        "wind_speed_ms": (0.0, 75.0),
        "solar_radiation_wm2": (0.0, 1400.0),
        "battery_voltage_v": (10.5, 15.5),
    }

    def __init__(self, flatline_window: int = 6):
        self.flatline_window = flatline_window

    def fit(self, df: pd.DataFrame) -> "PhysicsRuleDetector":
        """Stateless rule engine; fit returns self."""
        return self

    def score_and_diagnose(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Evaluate physics rules and return anomaly score in [0, 1] plus diagnostic reasons.
        """
        n = len(df)
        scores = np.zeros(n)
        diagnoses = ["Normal" for _ in range(n)]

        # 1. Climatological Boundary Violations
        for col, (low, high) in self.SENSOR_BOUNDS.items():
            if col in df.columns:
                val = df[col].values
                out_of_bounds = (val < low) | (val > high)
                for idx in np.where(out_of_bounds)[0]:
                    scores[idx] = max(scores[idx], 0.95)
                    diagnoses[idx] = f"Physical Boundary Exceeded: {col} = {val[idx]:.1f} outside [{low}, {high}]"

        # 2. Thermodynamic Consistency: Dew Point Depression >= 0
        if "temperature_c" in df.columns and "dew_point_c" in df.columns:
            # Dew point cannot exceed dry-bulb temperature (allowing 0.5C measurement margin)
            dew_violation = df["dew_point_c"].values > (df["temperature_c"].values + 0.5)
            for idx in np.where(dew_violation)[0]:
                scores[idx] = max(scores[idx], 0.90)
                diff = df["dew_point_c"].iloc[idx] - df["temperature_c"].iloc[idx]
                diagnoses[idx] = f"Thermodynamic Inconsistency: Dew Point exceeds Ambient Temp by {diff:.1f}°C"

        # 3. Night-Time Solar Radiation Violation
        if "solar_radiation_wm2" in df.columns and "timestamp" in df.columns:
            ts = pd.to_datetime(df["timestamp"])
            is_deep_night = (ts.dt.hour >= 21) | (ts.dt.hour <= 4)
            solar_vals = df["solar_radiation_wm2"].values
            night_solar_violation = is_deep_night & (solar_vals > 25.0)
            for idx in np.where(night_solar_violation)[0]:
                scores[idx] = max(scores[idx], 0.95)
                diagnoses[idx] = f"Physical Anomaly: Solar radiation {solar_vals[idx]:.1f} W/m² recorded at night ({ts.iloc[idx].strftime('%H:%M')})"

        # 4. Sensor Freeze / Flatline Check (Zero Variance over window)
        check_freeze_cols = ["humidity_pct", "wind_speed_ms", "temperature_c"]
        for col in check_freeze_cols:
            spread_col = f"{col}_roll_spread_{self.flatline_window}"
            if spread_col in df.columns:
                frozen = df[spread_col].values == 0.0
            elif col in df.columns:
                roll_s = df[col].rolling(window=self.flatline_window, min_periods=2)
                spread = (roll_s.max() - roll_s.min()).bfill().fillna(1.0)
                frozen = spread.values == 0.0
            else:
                frozen = np.zeros(n, dtype=bool)

            for idx in np.where(frozen)[0]:
                scores[idx] = max(scores[idx], 0.88)
                diagnoses[idx] = f"Sensor Freeze (Flatline): {col} unchanged for {self.flatline_window} consecutive steps"

        # 5. Station Battery Critical Brownout
        if "battery_voltage_v" in df.columns:
            batt = df["battery_voltage_v"].values
            batt_critical = batt < 10.8
            for idx in np.where(batt_critical)[0]:
                scores[idx] = max(scores[idx], 0.92)
                diagnoses[idx] = f"Battery Brownout: Supply voltage at {batt[idx]:.2f}V threatens sensor accuracy"

        return scores, diagnoses

    def score(self, df: pd.DataFrame) -> np.ndarray:
        scores, _ = self.score_and_diagnose(df)
        return scores

    def predict(self, df: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        scores = self.score(df)
        return (scores >= threshold).astype(int)
