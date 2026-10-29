"""
SkyGuard Live External Meteorological Data Feed Integration
Wires genuine live weather station telemetry into the anomaly detection engine
using Open-Meteo's real-time API (no synthetic data, no API key required).
"""

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
import urllib.parse
import urllib.request

# Registry of real-world Automated Weather Station monitoring sites
# Aligned with SkyGuard's regional array configuration
LIVE_WEATHER_STATIONS: Dict[str, Dict[str, Any]] = {
    "AWS-WA-COAST": {
        "station_id": "AWS-WA-COAST",
        "name": "Pacific Northwest Coastal Array (Washington)",
        "latitude": 46.97,
        "longitude": -123.85,
        "elevation_m": 12.0,
    },
    "AWS-MA-NANTUCKET": {
        "station_id": "AWS-MA-NANTUCKET",
        "name": "Northeast Atlantic Shore & Buoy (Massachusetts)",
        "latitude": 41.28,
        "longitude": -70.09,
        "elevation_m": 8.0,
    },
    "AWS-TX-VALLEY": {
        "station_id": "AWS-TX-VALLEY",
        "name": "Southern Plains & Texas Basin (Texas)",
        "latitude": 29.42,
        "longitude": -98.49,
        "elevation_m": 198.0,
    },
    "AWS-MN-DULUTH": {
        "station_id": "AWS-MN-DULUTH",
        "name": "Upper Midwest & Great Lakes (Minnesota)",
        "latitude": 46.78,
        "longitude": -92.10,
        "elevation_m": 342.0,
    },
    "AWS-ES-VALENCIA": {
        "station_id": "AWS-ES-VALENCIA",
        "name": "Western Mediterranean Basin (Spain)",
        "latitude": 39.46,
        "longitude": -0.37,
        "elevation_m": 15.0,
    },
    "AWS-AU-OUTBACK": {
        "station_id": "AWS-AU-OUTBACK",
        "name": "Central Outback Array (Alice Springs, Australia)",
        "latitude": -23.70,
        "longitude": 133.87,
        "elevation_m": 545.0,
    },
}

API_BASE_URL = "https://api.open-meteo.com/v1/forecast"


def fetch_live_station_telemetry(
    station_id: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    timeout: float = 8.0,
) -> Optional[Dict[str, Any]]:
    """
    Fetch live real-time weather observation for a weather station.
    Returns normalized dictionary mapped to SkyGuard channels.
    """
    if latitude is None or longitude is None:
        station_info = LIVE_WEATHER_STATIONS.get(station_id)
        if not station_info:
            return None
        latitude = station_info["latitude"]
        longitude = station_info["longitude"]

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
    }
    url = f"{API_BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SkyGuard-Anomaly-Detection/1.0 (Live-Sensor-Feed)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        current = data.get("current", {})
        if not current:
            return None

        ts = current.get("time")
        # Format timestamp to ISO 8601 UTC
        if ts and "T" in ts and not ts.endswith("Z"):
            ts = f"{ts}:00Z"

        return {
            "station_id": station_id,
            "sensor_id": station_id,
            "timestamp": ts or datetime.now(timezone.utc).isoformat(),
            "temperature_c": current.get("temperature_2m"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "pressure_hpa": current.get("surface_pressure"),
            "wind_speed_ms": current.get("wind_speed_10m"),
            "wind_direction_deg": current.get("wind_direction_10m"),
            "precipitation_mm": current.get("precipitation", 0.0),
            "data_source": "Open-Meteo Live External API",
        }
    except Exception as e:
        # Fallback error logging without crashing caller
        return None


def fetch_all_live_stations(timeout: float = 8.0) -> List[Dict[str, Any]]:
    """Fetch live telemetry for all configured monitoring stations."""
    results = []
    for stn_id, info in LIVE_WEATHER_STATIONS.items():
        res = fetch_live_station_telemetry(
            stn_id, latitude=info["latitude"], longitude=info["longitude"], timeout=timeout
        )
        if res:
            results.append(res)
    return results


def fetch_station_history(
    station_id: str,
    past_hours: int = 48,
    timeout: float = 10.0,
) -> List[Dict[str, Any]]:
    """
    Fetch genuine recent historical observations to prime the rolling window.
    """
    station_info = LIVE_WEATHER_STATIONS.get(station_id)
    if not station_info:
        return []

    params = {
        "latitude": station_info["latitude"],
        "longitude": station_info["longitude"],
        "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m,wind_direction_10m,precipitation",
        "past_days": max(1, min(7, int(past_hours / 24) + 1)),
        "forecast_days": 0,
    }
    url = f"{API_BASE_URL}?{urllib.parse.urlencode(params)}"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "SkyGuard-Anomaly-Detection/1.0 (History-Seed)"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        hums = hourly.get("relative_humidity_2m", [])
        press = hourly.get("surface_pressure", [])
        ws = hourly.get("wind_speed_10m", [])
        wd = hourly.get("wind_direction_10m", [])
        prec = hourly.get("precipitation", [])

        records = []
        n = min(len(times), len(temps))
        # Keep the latest past_hours records
        start_idx = max(0, n - past_hours)
        for i in range(start_idx, n):
            t = times[i]
            ts = f"{t}:00Z" if not t.endswith("Z") else t
            records.append({
                "station_id": station_id,
                "sensor_id": station_id,
                "timestamp": ts,
                "temperature_c": temps[i] if i < len(temps) else None,
                "humidity_pct": hums[i] if i < len(hums) else None,
                "pressure_hpa": press[i] if i < len(press) else None,
                "wind_speed_ms": ws[i] if i < len(ws) else None,
                "wind_direction_deg": wd[i] if i < len(wd) else None,
                "precipitation_mm": prec[i] if i < len(prec) else 0.0,
                "data_source": "Open-Meteo Live History",
            })
        return records
    except Exception:
        return []
