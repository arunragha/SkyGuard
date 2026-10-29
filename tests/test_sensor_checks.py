"""
Unit tests for SkyGuard per-channel sensor reliability checks.
"""

from datetime import datetime, timedelta
import numpy as np
import pytest

from skyguard.sensor_checks import (
    CHANNEL_CONFIG,
    SUPPORTED_CHANNELS,
    check_drift,
    check_missing_or_invalid,
    check_physical_plausibility,
    check_spike,
    check_stuck_value,
    evaluate_channel,
    evaluate_telemetry,
)


def test_supported_channels():
    expected = [
        "temperature_c",
        "humidity_pct",
        "wind_speed_ms",
        "wind_direction_deg",
        "precipitation_mm",
        "pressure_hpa",
    ]
    for ch in expected:
        assert ch in SUPPORTED_CHANNELS
        assert ch in CHANNEL_CONFIG


# -------------------------------------------------------------------
# 1. SPIKE DETECTION TESTS
# -------------------------------------------------------------------
def test_spike_detection_user_example():
    """31°C -> 31.5°C -> 32°C -> 95°C -> 32.5°C: 95°C should flag, 32.5°C should not."""
    cfg = CHANNEL_CONFIG["temperature_c"]
    series_spike = [31.0, 31.5, 32.0, 95.0]
    is_anom, reason, sev = check_spike(series_spike, config=cfg)
    assert is_anom is True
    assert "Sharp spike detected" in reason
    assert sev == "CRITICAL"

    # Next reading 32.5°C restores normality; should not flag as spike
    series_recovery = [31.0, 31.5, 32.0, 95.0, 32.5]
    is_anom, reason, sev = check_spike(series_recovery, config=cfg)
    assert is_anom is False


def test_spike_detection_across_all_channels():
    """Verify spike check triggers on abrupt jump for every supported channel."""
    channel_spikes = {
        "temperature_c": ([20.0, 20.2, 20.1, 35.0], True),
        "humidity_pct": ([50.0, 50.5, 51.0, 85.0], True),
        "wind_speed_ms": ([3.0, 3.2, 3.1, 28.0], True),
        "wind_direction_deg": ([180.0, 182.0, 181.0, 375.0], True),
        "precipitation_mm": ([0.0, 0.0, 0.0, 50.0], True),
        "pressure_hpa": ([1013.0, 1013.2, 1013.1, 1025.0], True),
    }
    for ch, (vals, expected) in channel_spikes.items():
        is_anom, reason, _ = check_spike(vals, config=CHANNEL_CONFIG[ch])
        assert is_anom == expected, f"Spike check failed for {ch}"


# -------------------------------------------------------------------
# 2. STUCK-AT-VALUE DETECTION TESTS
# -------------------------------------------------------------------
def test_stuck_value_temperature():
    """Temperature frozen at 22.4°C for 8 consecutive readings should flag."""
    cfg = CHANNEL_CONFIG["temperature_c"]
    normal_temp = [22.0, 22.1, 22.3, 22.4, 22.2, 22.5]
    assert check_stuck_value(normal_temp, config=cfg)[0] is False

    stuck_temp = [22.4] * 7
    is_anom, reason, sev = check_stuck_value(stuck_temp, config=cfg)
    assert is_anom is True
    assert "Stuck-at-value freeze" in reason


def test_stuck_value_rainfall_and_wind_zero_exemption():
    """Zero rainfall and zero wind speed in calm weather should NOT be flagged as stuck."""
    rain_cfg = CHANNEL_CONFIG["precipitation_mm"]
    dry_weather = [0.0] * 12
    assert check_stuck_value(dry_weather, config=rain_cfg)[0] is False

    # Non-zero stuck rainfall (stuck gauge at 4.2mm) SHOULD flag
    stuck_gauge = [4.2] * 7
    is_anom, reason, _ = check_stuck_value(stuck_gauge, config=rain_cfg)
    assert is_anom is True


# -------------------------------------------------------------------
# 3. DRIFT DETECTION TESTS
# -------------------------------------------------------------------
def test_drift_detection_temperature():
    """Gradual steady ramp of +0.3°C each step over 24 steps diverges from baseline."""
    cfg = CHANNEL_CONFIG["temperature_c"]
    base = [20.0 + np.random.normal(0, 0.05) for _ in range(18)]
    drift = [20.0 + i * 0.4 for i in range(18, 30)]
    full_series = base + drift
    is_anom, reason, sev = check_drift(full_series, config=cfg)
    assert is_anom is True
    assert "Sustained upward drift" in reason


def test_drift_normal_diurnal_does_not_false_alarm():
    """Stable baseline with minor noise should not trigger drift."""
    cfg = CHANNEL_CONFIG["pressure_hpa"]
    stable_pressure = [1013.0 + 0.1 * np.sin(i / 4.0) for i in range(30)]
    assert check_drift(stable_pressure, config=cfg)[0] is False


# -------------------------------------------------------------------
# 4. MISSING / INVALID DATA DETECTION TESTS
# -------------------------------------------------------------------
def test_missing_and_nan():
    is_anom, reason, sev = check_missing_or_invalid([20.0, None])
    assert is_anom is True
    assert "null or NaN" in reason

    is_anom, reason, sev = check_missing_or_invalid([20.0, np.nan])
    assert is_anom is True

    is_anom, reason, sev = check_missing_or_invalid([20.0, "MALFORMED"])
    assert is_anom is True
    assert "Malformed payload" in reason


def test_reporting_interval_gap():
    """A 35-minute gap between readings when 10-minute expected should flag."""
    t0 = datetime(2026, 6, 1, 10, 0)
    t1 = t0 + timedelta(minutes=35)
    is_anom, reason, sev = check_missing_or_invalid(
        [20.0, 20.5],
        timestamps=[t0, t1],
        config={"max_interval_minutes": 15.0},
    )
    assert is_anom is True
    assert "reporting gap" in reason


# -------------------------------------------------------------------
# 5. PHYSICAL PLAUSIBILITY CHECKS
# -------------------------------------------------------------------
def test_physical_plausibility_bounds():
    # Humidity outside 0-100%
    h_cfg = CHANNEL_CONFIG["humidity_pct"]
    assert check_physical_plausibility([105.0], config=h_cfg)[0] is True
    assert check_physical_plausibility([-2.0], config=h_cfg)[0] is True
    assert check_physical_plausibility([75.0], config=h_cfg)[0] is False

    # Negative rainfall
    r_cfg = CHANNEL_CONFIG["precipitation_mm"]
    assert check_physical_plausibility([-0.5], config=r_cfg)[0] is True
    assert check_physical_plausibility([12.0], config=r_cfg)[0] is False

    # Wind direction outside 0-360
    w_cfg = CHANNEL_CONFIG["wind_direction_deg"]
    assert check_physical_plausibility([385.0], config=w_cfg)[0] is True
    assert check_physical_plausibility([-10.0], config=w_cfg)[0] is True
    assert check_physical_plausibility([270.0], config=w_cfg)[0] is False

    # Impossible pressure
    p_cfg = CHANNEL_CONFIG["pressure_hpa"]
    assert check_physical_plausibility([500.0], config=p_cfg)[0] is True
    assert check_physical_plausibility([1015.0], config=p_cfg)[0] is False


# -------------------------------------------------------------------
# INCIDENT SCHEMA & TELEMETRY EVALUATION
# -------------------------------------------------------------------
def test_evaluate_channel_incident_schema():
    incidents = evaluate_channel(
        channel="temperature_c",
        series=[30.0, 30.2, 30.1, 95.0],
        station_id="AWS-STN-042",
        timestamps=["2026-06-01T12:00:00Z", "2026-06-01T12:10:00Z", "2026-06-01T12:20:00Z", "2026-06-01T12:30:00Z"],
    )
    assert len(incidents) >= 1
    types = {i["type"] for i in incidents}
    assert "SPIKE" in types
    inc = next(i for i in incidents if i["type"] == "SPIKE")
    required_keys = [
        "type",
        "timestamp",
        "channel",
        "station_id",
        "observed_value",
        "severity",
        "description",
        "is_anomaly",
    ]
    for k in required_keys:
        assert k in inc
    assert inc["channel"] == "temperature_c"
    assert inc["station_id"] == "AWS-STN-042"
    assert inc["observed_value"] == 95.0


def test_evaluate_telemetry_batch():
    import pandas as pd

    records = [
        {"timestamp": "2026-06-01 00:00:00", "station_id": "STN-1", "temperature_c": 20.0, "humidity_pct": 50.0},
        {"timestamp": "2026-06-01 00:10:00", "station_id": "STN-1", "temperature_c": 20.2, "humidity_pct": 50.1},
        {"timestamp": "2026-06-01 00:20:00", "station_id": "STN-1", "temperature_c": 20.1, "humidity_pct": 50.2},
        {"timestamp": "2026-06-01 00:30:00", "station_id": "STN-1", "temperature_c": 88.0, "humidity_pct": 140.0},
    ]
    df = pd.DataFrame(records)
    incidents = evaluate_telemetry(df)
    assert len(incidents) >= 2  # Spike on temp, physical violation on humidity
    types = {i["type"] for i in incidents}
    assert "SPIKE" in types
    assert "PHYSICAL_PLAUSIBILITY" in types
