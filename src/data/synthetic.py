"""
Synthetic weather station data generator with physically consistent atmospheric dynamics
and injected sensor fault archetypes (spikes, stuck flatlines, calibration drift, dropouts, multivariate breaks).
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.config import AppConfig, load_config


class WeatherAnomalyGenerator:
    """
    Simulates high-fidelity automated weather station (AWS) multi-sensor telemetry
    and injects realistic sensor fault modes for unsupervised anomaly detection evaluation.
    """

    def __init__(self, config: Optional[AppConfig] = None, seed: int = 42):
        self.config = config or load_config()
        self.rng = np.random.default_rng(seed)

    def generate(
        self,
        stations: Optional[List[str]] = None,
        days: int = 14,
        freq_minutes: int = 10,
        start_date: str = "2026-06-01 00:00:00",
        anomaly_ratio: float = 0.04,
    ) -> pd.DataFrame:
        """
        Generates continuous time series for specified weather stations with injected anomalies.
        """
        if stations is None:
            stations = ["STATION_ALPHA", "STATION_BETA", "STATION_GAMMA"]

        start_dt = pd.to_datetime(start_date, utc=True)
        periods = int(days * 24 * (60 // freq_minutes))
        timestamps = pd.date_range(start=start_dt, periods=periods, freq=f"{freq_minutes}min")

        all_station_dfs = []
        for st_idx, st_id in enumerate(stations):
            st_df = self._generate_single_station(
                station_id=st_id,
                station_seed_offset=st_idx * 1000,
                timestamps=timestamps,
                anomaly_ratio=anomaly_ratio,
            )
            all_station_dfs.append(st_df)

        full_df = pd.concat(all_station_dfs, axis=0).reset_index(drop=True)
        return full_df

    def _generate_single_station(
        self,
        station_id: str,
        station_seed_offset: int,
        timestamps: pd.DatetimeIndex,
        anomaly_ratio: float,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(station_seed_offset + 42)
        n = len(timestamps)

        # Base station geographic characteristics
        altitude_m = rng.uniform(50.0, 850.0)
        base_temp = rng.uniform(18.0, 26.0) - (altitude_m * 0.0065)  # lapse rate
        base_pressure = 1013.25 * np.exp(-altitude_m / 8434.0)

        # Time vectors
        hours = timestamps.hour.values + timestamps.minute.values / 60.0
        day_of_year = timestamps.dayofyear.values

        # 1. Solar Radiation (W/m²) - solar zenith model
        # Sun rises ~05:30, peaks at 12:30, sets ~19:30
        solar_peak = 12.5
        daylight_mask = (hours >= 5.5) & (hours <= 19.5)
        solar_rad = np.zeros(n)
        sun_angle = np.maximum(0, np.sin(np.pi * (hours[daylight_mask] - 5.5) / 14.0))
        # Clear-sky maximum with cloudiness factor
        cloud_factor = np.clip(1.0 - rng.beta(2, 5, size=np.sum(daylight_mask)), 0.2, 1.0)
        solar_rad[daylight_mask] = (
            sun_angle * rng.uniform(750, 980) * cloud_factor + rng.normal(0, 8, size=np.sum(daylight_mask))
        )
        solar_rad = np.clip(solar_rad, 0.0, 1350.0)

        # 2. Temperature (°C) - Diurnal cycle peaking around 14:30 + synoptic wave
        diurnal_temp = 7.0 * np.sin(2 * np.pi * (hours - 8.5) / 24.0)
        synoptic_temp = 3.5 * np.sin(2 * np.pi * day_of_year / 7.0)
        temp_noise = rng.normal(0, 0.4, size=n)
        temperature = base_temp + diurnal_temp + synoptic_temp + temp_noise

        # 3. Relative Humidity (%) - Inversely coupled to temperature
        # Saturated air warms -> relative humidity drops
        rh_diurnal = -25.0 * np.sin(2 * np.pi * (hours - 8.5) / 24.0)
        rh_synoptic = -5.0 * np.sin(2 * np.pi * day_of_year / 7.0)
        rh_noise = rng.normal(0, 1.5, size=n)
        humidity = np.clip(68.0 + rh_diurnal + rh_synoptic + rh_noise, 12.0, 98.0)

        # 4. Barometric Pressure (hPa) - Atmospheric tides (semi-diurnal) + weather front
        tide_12h = 1.2 * np.cos(2 * np.pi * (hours - 10.0) / 12.0)
        weather_front = 8.0 * np.sin(2 * np.pi * day_of_year / 5.0)
        pressure_noise = rng.normal(0, 0.2, size=n)
        pressure = base_pressure + tide_12h + weather_front + pressure_noise

        # 5. Wind Speed (m/s) - Weibull distribution + afternoon thermal breeze
        thermal_wind = np.maximum(0, 2.5 * np.sin(np.pi * (hours - 8.0) / 12.0))
        wind_speed = rng.weibull(a=1.8, size=n) * 3.2 + thermal_wind
        wind_speed = np.clip(wind_speed, 0.0, 45.0)

        # 6. Wind Direction (degrees) - Markovian smooth rotation
        wind_dir = np.zeros(n)
        current_dir = rng.uniform(0, 360)
        for i in range(n):
            current_dir = (current_dir + rng.normal(0, 4.0)) % 360.0
            wind_dir[i] = current_dir

        # 7. Rainfall (mm) - Sparse rain events
        rainfall = np.zeros(n)
        rain_event_prob = rng.uniform(0, 1, size=n) < 0.02
        rainfall[rain_event_prob] = rng.exponential(scale=2.5, size=np.sum(rain_event_prob))
        # High humidity during rain
        humidity[rain_event_prob] = np.clip(humidity[rain_event_prob] + 20.0, 85.0, 100.0)
        solar_rad[rain_event_prob] *= 0.25

        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "station_id": station_id,
                "temperature": np.round(temperature, 2),
                "humidity": np.round(humidity, 1),
                "barometric_pressure": np.round(pressure, 2),
                "wind_speed": np.round(wind_speed, 2),
                "wind_direction": np.round(wind_dir, 1),
                "rainfall": np.round(rainfall, 2),
                "solar_radiation": np.round(solar_rad, 1),
                "is_anomaly": 0,
                "anomaly_type": "normal",
                "faulty_sensor": "none",
            }
        )

        # Inject Faults
        df = self._inject_faults(df, anomaly_ratio, rng)
        return df

    def _inject_faults(self, df: pd.DataFrame, target_ratio: float, rng: np.random.Generator) -> pd.DataFrame:
        """Injects labeled anomaly archetypes across sensor channels."""
        n = len(df)
        total_anomalies_target = int(n * target_ratio)
        if total_anomalies_target < 5:
            return df

        sensors = [
            "temperature",
            "humidity",
            "barometric_pressure",
            "wind_speed",
            "wind_direction",
            "solar_radiation",
        ]

        # 1. Spikes (Impulse glitches, 1-2 timestamps)
        num_spikes = max(2, int(total_anomalies_target * 0.25))
        spike_indices = rng.choice(np.arange(10, n - 10), size=num_spikes, replace=False)
        for idx in spike_indices:
            sensor = rng.choice(sensors)
            mult = rng.choice([-1.0, 1.0])
            if sensor == "temperature":
                delta = mult * rng.uniform(14.0, 25.0)
                df.loc[idx, sensor] += delta
            elif sensor == "humidity":
                df.loc[idx, sensor] = rng.choice([5.0, 99.0])
            elif sensor == "barometric_pressure":
                delta = mult * rng.uniform(18.0, 35.0)
                df.loc[idx, sensor] += delta
            elif sensor == "wind_speed":
                df.loc[idx, sensor] += rng.uniform(22.0, 40.0)
            elif sensor == "solar_radiation":
                df.loc[idx, sensor] += rng.uniform(400.0, 750.0)

            df.loc[idx, "is_anomaly"] = 1
            df.loc[idx, "anomaly_type"] = "spike"
            df.loc[idx, "faulty_sensor"] = sensor

        # 2. Stuck Values (Sensor freeze / flatline, 6 to 18 consecutive timestamps = 1 to 3 hours)
        num_stuck_blocks = max(1, int(total_anomalies_target * 0.25 // 10))
        for _ in range(num_stuck_blocks):
            start = rng.integers(20, n - 40)
            duration = rng.integers(8, 16)
            sensor = rng.choice(["temperature", "wind_speed", "humidity", "barometric_pressure"])
            stuck_val = df.loc[start, sensor]
            for step in range(duration):
                cur_idx = start + step
                if cur_idx < n:
                    df.loc[cur_idx, sensor] = stuck_val
                    df.loc[cur_idx, "is_anomaly"] = 1
                    df.loc[cur_idx, "anomaly_type"] = "stuck"
                    df.loc[cur_idx, "faulty_sensor"] = sensor

        # 3. Calibration Drift (Monotonic linear bias degradation over 24-48 timesteps)
        num_drifts = max(1, int(total_anomalies_target * 0.20 // 24))
        for _ in range(num_drifts):
            start = rng.integers(30, n - 60)
            duration = rng.integers(24, 48)
            sensor = rng.choice(["barometric_pressure", "temperature", "humidity"])
            max_drift = rng.uniform(12.0, 22.0) * rng.choice([-1.0, 1.0])
            for step in range(duration):
                cur_idx = start + step
                if cur_idx < n:
                    bias = (step / duration) * max_drift
                    df.loc[cur_idx, sensor] += bias
                    # Mark anomaly once drift exceeds subtle baseline
                    if abs(bias) >= 4.0:
                        df.loc[cur_idx, "is_anomaly"] = 1
                        df.loc[cur_idx, "anomaly_type"] = "drift"
                        df.loc[cur_idx, "faulty_sensor"] = sensor

        # 4. Dropouts (Sudden drops to 0.0 or floor)
        num_dropouts = max(2, int(total_anomalies_target * 0.15 // 3))
        for _ in range(num_dropouts):
            start = rng.integers(15, n - 30)
            duration = rng.integers(2, 6)
            sensor = rng.choice(["temperature", "humidity", "barometric_pressure", "solar_radiation"])
            for step in range(duration):
                cur_idx = start + step
                if cur_idx < n:
                    df.loc[cur_idx, sensor] = 0.0
                    df.loc[cur_idx, "is_anomaly"] = 1
                    df.loc[cur_idx, "anomaly_type"] = "dropout"
                    df.loc[cur_idx, "faulty_sensor"] = sensor

        # 5. Multivariate Physical Breakdowns (Individual values normal, but impossible combination)
        num_multivariate = max(2, int(total_anomalies_target * 0.15))
        for _ in range(num_multivariate):
            idx = rng.integers(20, n - 20)
            mode = rng.choice(["night_solar", "dewpoint_inversion", "pressure_temp_decouple"])
            if mode == "night_solar":
                # Find nighttime timestamp (hour < 4 or hour > 22)
                ts = df.loc[idx, "timestamp"]
                night_indices = df.index[(df["timestamp"].dt.hour >= 23) | (df["timestamp"].dt.hour <= 3)]
                if len(night_indices) > 0:
                    chosen_idx = rng.choice(night_indices)
                    df.loc[chosen_idx, "solar_radiation"] = rng.uniform(550.0, 850.0)
                    df.loc[chosen_idx, "is_anomaly"] = 1
                    df.loc[chosen_idx, "anomaly_type"] = "multivariate"
                    df.loc[chosen_idx, "faulty_sensor"] = "solar_radiation"
            elif mode == "dewpoint_inversion":
                # Relative humidity > 100% while temperature is high
                df.loc[idx, "humidity"] = 100.0
                df.loc[idx, "temperature"] = 38.0  # unphysically high heat index
                df.loc[idx, "is_anomaly"] = 1
                df.loc[idx, "anomaly_type"] = "multivariate"
                df.loc[idx, "faulty_sensor"] = "humidity_temperature"
            else:
                # Sudden massive pressure plunge (-25 hPa) with 0 wind and 0 rain
                df.loc[idx, "barometric_pressure"] -= 28.0
                df.loc[idx, "wind_speed"] = 0.2
                df.loc[idx, "rainfall"] = 0.0
                df.loc[idx, "is_anomaly"] = 1
                df.loc[idx, "anomaly_type"] = "multivariate"
                df.loc[idx, "faulty_sensor"] = "barometric_pressure"

        return df
