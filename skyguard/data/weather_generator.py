"""
SkyGuard Weather Station Telemetry & Sensor Fault Generator
Simulates realistic multi-sensor meteorological time series and injects labeled sensor fault modes.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


class WeatherDataGenerator:
    """
    Simulates automated weather station (AWS) IoT telemetry with physically coupled
    meteorological channels, diurnal cycles, synoptic weather trends, and synthetic fault injection.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        np.random.seed(seed)

    def generate_telemetry(
        self,
        start_date: str = "2026-06-01 00:00:00",
        periods: int = 4320,  # 30 days at 10-minute intervals
        freq: str = "10min",
        station_id: str = "AWS-STN-042",
        elevation_m: float = 185.0,
        inject_faults: bool = True,
        anomaly_ratio: float = 0.04,
    ) -> pd.DataFrame:
        """
        Generate time-series weather station data.

        Args:
            start_date: Starting ISO timestamp
            periods: Number of time steps
            freq: Frequency string ('10min', '5min', etc.)
            station_id: Unique station identifier
            elevation_m: Station altitude in meters
            inject_faults: Whether to inject ground-truth anomalies
            anomaly_ratio: Target ratio of anomalous timestamps

        Returns:
            pd.DataFrame with meteorological channels, physical ground truth, and anomaly tags.
        """
        timestamps = pd.date_range(start=start_date, periods=periods, freq=freq)
        n = len(timestamps)

        # Time helpers
        day_of_year = timestamps.dayofyear.values
        hour_of_day = timestamps.hour.values + timestamps.minute.values / 60.0

        # Synoptic weather scale (low/high pressure systems every 4-7 days)
        t_hours = np.arange(n) * (10.0 / 60.0)
        synoptic_wave = 15.0 * np.sin(2 * np.pi * t_hours / (24 * 5.5)) + 6.0 * np.cos(
            2 * np.pi * t_hours / (24 * 2.3)
        )
        barometric_base = 1013.25 - (elevation_m / 8.5)
        pressure_hpa = barometric_base + synoptic_wave + np.random.normal(0, 0.4, n)

        # Solar radiation model (Clear-sky envelope modulated by cloud cover)
        # Solar elevation angle approximation
        declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
        hour_angle = (hour_of_day - 12.0) * 15.0
        # Latitude ~ 40 deg N
        lat_rad = np.radians(40.0)
        dec_rad = np.radians(declination)
        ha_rad = np.radians(hour_angle)
        sin_elev = np.sin(lat_rad) * np.sin(dec_rad) + np.cos(lat_rad) * np.cos(dec_rad) * np.cos(ha_rad)
        sun_up = sin_elev > 0

        # Cloud cover driven by low pressure
        cloud_probability = np.clip(1.0 - (pressure_hpa - 995.0) / 30.0, 0.05, 0.95)
        cloud_noise = np.random.beta(2, 5, n)
        cloud_factor = 1.0 - (cloud_probability * 0.75 * (cloud_noise > 0.3))

        max_solar_wm2 = 1000.0 * np.maximum(0.0, sin_elev)
        solar_radiation_wm2 = np.where(sun_up, max_solar_wm2 * cloud_factor, 0.0)
        solar_radiation_wm2 = np.maximum(0.0, solar_radiation_wm2 + np.random.normal(0, 3.0, n) * sun_up)

        # Temperature model: diurnal solar lag (~2 hr afternoon peak) + synoptic trend
        diurnal_temp = 7.5 * np.sin(2 * np.pi * (hour_of_day - 8.5) / 24.0)
        synoptic_temp = -0.3 * synoptic_wave  # colder in low pressure storms
        base_temp = 22.0
        temperature_c = base_temp + diurnal_temp + synoptic_temp + np.random.normal(0, 0.35, n)

        # Relative Humidity: inversely coupled to temperature (Clausius-Clapeyron approx)
        # Saturated vapor pressure es(T) ~ 0.61078 * exp(17.27 * T / (T + 237.3))
        # Dew point typically lags with moisture variation
        dew_point_base = base_temp - 7.0 + 0.15 * synoptic_wave
        dew_point_c = dew_point_base + np.random.normal(0, 0.3, n)
        # Ensure physical consistency: dew_point <= temperature
        dew_point_c = np.minimum(dew_point_c, temperature_c - 0.5)

        # Actual Vapor Pressure vs Saturation Vapor Pressure
        es_temp = 0.61078 * np.exp(17.27 * temperature_c / (temperature_c + 237.3))
        es_dew = 0.61078 * np.exp(17.27 * dew_point_c / (dew_point_c + 237.3))
        relative_humidity_pct = np.clip((es_dew / es_temp) * 100.0, 10.0, 98.0)

        # Precipitation: occur when humidity is high (>85%) and pressure is dropping
        rain_potential = np.maximum(0.0, (relative_humidity_pct - 85.0) / 15.0) * (pressure_hpa < barometric_base)
        precip_events = (np.random.uniform(0, 1, n) < (rain_potential * 0.45)).astype(float)
        precipitation_mm = np.where(
            precip_events > 0,
            np.random.gamma(shape=1.8, scale=1.5, size=n),
            0.0
        )
        precipitation_mm = np.round(precipitation_mm, 2)

        # Wind speed: Rayleigh/Weibull distributed, higher in afternoons and during storms
        storm_boost = np.maximum(0.0, (barometric_base - pressure_hpa) * 0.4)
        diurnal_wind = 1.2 * np.maximum(0.0, np.sin(2 * np.pi * (hour_of_day - 11.0) / 24.0))
        wind_scale = 2.5 + storm_boost + diurnal_wind
        wind_speed_ms = np.random.weibull(a=1.9, size=n) * wind_scale
        wind_speed_ms = np.clip(wind_speed_ms, 0.0, 45.0)

        # Wind direction: slow random walk around predominant prevailing direction (e.g., 220 deg SW)
        direction_steps = np.random.normal(0, 4.0, n)
        wind_direction_deg = (220.0 + np.cumsum(direction_steps)) % 360.0

        # Battery voltage: solar charged during day (13.6 - 14.2V), draining at night (12.2 - 12.6V)
        battery_base = 12.4 + (solar_radiation_wm2 / 1000.0) * 1.6 - (hour_of_day > 20) * 0.2
        battery_voltage_v = np.clip(battery_base + np.random.normal(0, 0.05, n), 11.8, 14.4)

        # Enclosure internal temperature (higher due to electronics & solar baking)
        internal_temp_c = temperature_c + 2.0 + (solar_radiation_wm2 / 1000.0) * 6.0 + np.random.normal(0, 0.2, n)

        df = pd.DataFrame({
            "timestamp": timestamps,
            "station_id": station_id,
            "temperature_c": np.round(temperature_c, 2),
            "humidity_pct": np.round(relative_humidity_pct, 1),
            "pressure_hpa": np.round(pressure_hpa, 2),
            "wind_speed_ms": np.round(wind_speed_ms, 2),
            "wind_direction_deg": np.round(wind_direction_deg, 1),
            "solar_radiation_wm2": np.round(solar_radiation_wm2, 1),
            "precipitation_mm": precipitation_mm,
            "battery_voltage_v": np.round(battery_voltage_v, 2),
            "internal_temp_c": np.round(internal_temp_c, 2),
            "is_anomaly": 0,
            "anomaly_type": "none",
            "faulty_sensor": "none",
        })

        if inject_faults:
            df = self._inject_synthetic_faults(df, anomaly_ratio)

        return df

    def _inject_synthetic_faults(self, df: pd.DataFrame, target_ratio: float) -> pd.DataFrame:
        """
        Injects realistic, physically distinct sensor fault archetypes.
        """
        n = len(df)
        df = df.copy()

        # Target number of anomaly episodes
        # Average episode length is ~6 time steps (1 hour)
        num_episodes = max(4, int((n * target_ratio) / 6))

        fault_types = [
            "spike",
            "flatline",
            "drift",
            "physical_violation",
            "multivariate_discordance",
            "battery_brownout",
        ]

        # Allocate non-overlapping time windows for each fault
        window_size = 40
        candidate_starts = np.arange(100, n - window_size, window_size)
        chosen_starts = np.random.choice(candidate_starts, size=min(num_episodes, len(candidate_starts)), replace=False)

        for i, start_idx in enumerate(sorted(chosen_starts)):
            fault = fault_types[i % len(fault_types)]

            if fault == "spike":
                # Single or 2-step extreme impulse
                idx = start_idx
                sensor_choice = np.random.choice(["temperature_c", "pressure_hpa", "wind_speed_ms"])
                if sensor_choice == "temperature_c":
                    df.loc[idx, "temperature_c"] += np.random.choice([18.5, -22.0])
                elif sensor_choice == "pressure_hpa":
                    df.loc[idx, "pressure_hpa"] += np.random.choice([45.0, -55.0])
                else:
                    df.loc[idx, "wind_speed_ms"] += 38.0
                df.loc[idx, "is_anomaly"] = 1
                df.loc[idx, "anomaly_type"] = "spike"
                df.loc[idx, "faulty_sensor"] = sensor_choice

            elif fault == "flatline":
                # Stuck-at fault: sensor values freeze exactly for 12 to 24 time steps (2 - 4 hours)
                duration = np.random.randint(14, 28)
                end_idx = min(start_idx + duration, n)
                sensor_choice = np.random.choice(["humidity_pct", "wind_speed_ms", "temperature_c"])
                stuck_val = df.loc[start_idx, sensor_choice]
                df.loc[start_idx:end_idx, sensor_choice] = stuck_val
                df.loc[start_idx:end_idx, "is_anomaly"] = 1
                df.loc[start_idx:end_idx, "anomaly_type"] = "flatline"
                df.loc[start_idx:end_idx, "faulty_sensor"] = sensor_choice

            elif fault == "drift":
                # Calibration drift: gradual ramping bias over 30 to 50 time steps
                duration = np.random.randint(24, 48)
                end_idx = min(start_idx + duration, n)
                sensor_choice = np.random.choice(["temperature_c", "pressure_hpa"])
                drift_slope = np.linspace(0, 8.5 if sensor_choice == "temperature_c" else -18.0, end_idx - start_idx)
                df.loc[start_idx:end_idx-1, sensor_choice] += drift_slope
                df.loc[start_idx:end_idx-1, "is_anomaly"] = 1
                df.loc[start_idx:end_idx-1, "anomaly_type"] = "drift"
                df.loc[start_idx:end_idx-1, "faulty_sensor"] = sensor_choice

            elif fault == "physical_violation":
                # Night-time solar radiation or humidity > 100%
                night_indices = df[(df.index >= start_idx) & (df["solar_radiation_wm2"] == 0)].index
                if len(night_indices) > 5:
                    target_indices = night_indices[:6]
                    df.loc[target_indices, "solar_radiation_wm2"] = np.random.uniform(450, 750, len(target_indices))
                    df.loc[target_indices, "is_anomaly"] = 1
                    df.loc[target_indices, "anomaly_type"] = "physical_violation"
                    df.loc[target_indices, "faulty_sensor"] = "solar_radiation_wm2"
                else:
                    duration = 8
                    end_idx = min(start_idx + duration, n)
                    df.loc[start_idx:end_idx, "humidity_pct"] = np.random.uniform(115.0, 135.0, end_idx - start_idx + 1)
                    df.loc[start_idx:end_idx, "is_anomaly"] = 1
                    df.loc[start_idx:end_idx, "anomaly_type"] = "physical_violation"
                    df.loc[start_idx:end_idx, "faulty_sensor"] = "humidity_pct"

            elif fault == "multivariate_discordance":
                # Temperature surges while humidity surges and pressure plunges without solar heating
                duration = 10
                end_idx = min(start_idx + duration, n)
                df.loc[start_idx:end_idx, "temperature_c"] += 12.0
                df.loc[start_idx:end_idx, "humidity_pct"] = np.clip(df.loc[start_idx:end_idx, "humidity_pct"] + 25.0, 0, 99)
                df.loc[start_idx:end_idx, "pressure_hpa"] -= 16.0
                df.loc[start_idx:end_idx, "is_anomaly"] = 1
                df.loc[start_idx:end_idx, "anomaly_type"] = "multivariate_discordance"
                df.loc[start_idx:end_idx, "faulty_sensor"] = "temp_humidity_pressure_cluster"

            elif fault == "battery_brownout":
                # Station power failure causing noise across multiple sensors
                duration = 15
                end_idx = min(start_idx + duration, n)
                df.loc[start_idx:end_idx, "battery_voltage_v"] = np.random.uniform(9.2, 10.4, end_idx - start_idx + 1)
                # Severe jitter on multiple analog channels
                jitter = np.random.normal(0, 7.0, end_idx - start_idx + 1)
                df.loc[start_idx:end_idx, "temperature_c"] += jitter
                df.loc[start_idx:end_idx, "wind_speed_ms"] += np.abs(jitter * 1.5)
                df.loc[start_idx:end_idx, "is_anomaly"] = 1
                df.loc[start_idx:end_idx, "anomaly_type"] = "battery_brownout"
                df.loc[start_idx:end_idx, "faulty_sensor"] = "multi_sensor_power"

        return df


def generate_benchmark_dataset(output_path: Optional[str] = None) -> pd.DataFrame:
    """Helper to generate and optionally save default 30-day telemetry benchmark dataset."""
    generator = WeatherDataGenerator(seed=42)
    df = generator.generate_telemetry(periods=4320, freq="10min", inject_faults=True, anomaly_ratio=0.04)
    if output_path:
        import os
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        df.to_csv(output_path, index=False)
    return df
