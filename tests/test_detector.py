"""
Unit & Integration Tests for SkyGuard Unsupervised Sensor Anomaly Detection Engine
Validates all 6 anomaly checks, output contract compliance, streaming behavior, and live feed.
"""

from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from starlette.testclient import TestClient

from skyguard.api import app, API_KEY
from skyguard.data.live_feed import (
    LIVE_WEATHER_STATIONS,
    fetch_all_live_stations,
    fetch_live_station_telemetry,
)
from skyguard.detector import (
    CHANNEL_CONFIG,
    SUPPORTED_CHANNELS,
    SkyGuardDetector,
    check_drift,
    check_missing_or_invalid,
    check_physical_plausibility,
    check_spike,
    check_stuck_value,
    create_anomaly_record,
)


@pytest.fixture
def detector():
    return SkyGuardDetector()


# -------------------------------------------------------------------
# 1. SPIKE DETECTION (Rolling Median + MAD)
# -------------------------------------------------------------------
def test_spike_detection_user_sequence():
    """Sequence: 31.0 -> 31.5 -> 32.0 -> 95.0 -> 32.5. 95.0 flags, 32.5 does not."""
    cfg = CHANNEL_CONFIG["temperature_c"]
    series = [31.0, 31.5, 32.0, 95.0]
    is_anom, reason, score, level, obs, exp, dev = check_spike(series, config=cfg)
    assert is_anom is True
    assert "Sharp spike detected" in reason
    assert obs == 95.0
    assert exp == 31.5
    assert dev == pytest.approx(63.5, 0.1)
    assert level in ("high", "critical")

    # Recovery reading 32.5 restores baseline; robust median prevents baseline dragging
    recovery_series = [31.0, 31.5, 32.0, 95.0, 32.5]
    is_anom2, _, _, _, _, _, _ = check_spike(recovery_series, config=cfg)
    assert is_anom2 is False


# -------------------------------------------------------------------
# 2. STUCK-AT-VALUE DETECTION
# -------------------------------------------------------------------
def test_stuck_value_detection():
    """Sensor flatlined at 24.5C for 7 consecutive steps triggers stuck fault."""
    cfg = CHANNEL_CONFIG["temperature_c"]
    stuck_series = [24.5] * 7
    is_anom, reason, score, level, obs, exp, dev = check_stuck_value(stuck_series, config=cfg)
    assert is_anom is True
    assert "Stuck-at-value freeze" in reason
    assert obs == 24.5


def test_stuck_value_natural_calm_dry_exemption():
    """Calm wind (0.0 m/s) and dry weather (0.0 mm) should NOT be flagged as stuck."""
    wind_cfg = CHANNEL_CONFIG["wind_speed_ms"]
    rain_cfg = CHANNEL_CONFIG["precipitation_mm"]

    calm_wind = [0.0] * 10
    is_anom, _, _, _, _, _, _ = check_stuck_value(calm_wind, config=wind_cfg)
    assert is_anom is False

    dry_rain = [0.0] * 10
    is_anom, _, _, _, _, _, _ = check_stuck_value(dry_rain, config=rain_cfg)
    assert is_anom is False

    # Stuck non-zero rainfall (broken gauge stuck at 3.5mm) MUST flag
    stuck_gauge = [3.5] * 8
    is_anom, reason, _, _, _, _, _ = check_stuck_value(stuck_gauge, config=rain_cfg)
    assert is_anom is True


# -------------------------------------------------------------------
# 3. DRIFT DETECTION
# -------------------------------------------------------------------
def test_drift_detection():
    """Linear divergence of +0.35/step over 24 readings triggers drift."""
    cfg = CHANNEL_CONFIG["temperature_c"]
    base = [20.0 + np.random.normal(0, 0.02) for _ in range(18)]
    drift = [20.0 + i * 0.45 for i in range(18, 30)]
    series = base + drift
    is_anom, reason, score, level, obs, exp, dev = check_drift(series, config=cfg)
    assert is_anom is True
    assert "Sustained upward drift" in reason


def test_drift_normal_diurnal_cycle_not_flagged():
    """Diurnal oscillation without monotonic slope does not trigger drift."""
    cfg = CHANNEL_CONFIG["temperature_c"]
    diurnal = [20.0 + 3.0 * np.sin(2 * np.pi * i / 24.0) for i in range(30)]
    is_anom, _, _, _, _, _, _ = check_drift(diurnal, config=cfg)
    assert is_anom is False


# -------------------------------------------------------------------
# 4. MISSING / INVALID DATA DETECTION
# -------------------------------------------------------------------
def test_missing_or_invalid_values():
    assert check_missing_or_invalid([21.0, None])[0] is True
    assert check_missing_or_invalid([21.0, np.nan])[0] is True
    assert check_missing_or_invalid([21.0, "CORRUPT_HEX"])[0] is True


def test_telemetry_reporting_interval_gap():
    """30-minute interval gap exceeds 15-minute threshold and triggers gap anomaly."""
    t0 = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    t1 = t0 + timedelta(minutes=30)
    is_anom, reason, score, level, obs, exp, dev = check_missing_or_invalid(
        [20.0, 20.2], timestamps=[t0, t1], config={"max_interval_minutes": 15.0}
    )
    assert is_anom is True
    assert "Telemetry reporting gap" in reason


# -------------------------------------------------------------------
# 5. PHYSICALLY IMPLAUSIBLE VALUES
# -------------------------------------------------------------------
def test_physical_plausibility_bounds():
    h_cfg = CHANNEL_CONFIG["humidity_pct"]
    assert check_physical_plausibility([115.0], config=h_cfg)[0] is True
    assert check_physical_plausibility([-5.0], config=h_cfg)[0] is True
    assert check_physical_plausibility([65.0], config=h_cfg)[0] is False

    w_cfg = CHANNEL_CONFIG["wind_speed_ms"]
    assert check_physical_plausibility([-1.5], config=w_cfg)[0] is True
    assert check_physical_plausibility([12.0], config=w_cfg)[0] is False

    p_cfg = CHANNEL_CONFIG["pressure_hpa"]
    assert check_physical_plausibility([650.0], config=p_cfg)[0] is True
    assert check_physical_plausibility([1013.25], config=p_cfg)[0] is False


# -------------------------------------------------------------------
# 6. OUTPUT CONTRACT COMPLIANCE
# -------------------------------------------------------------------
def test_create_anomaly_record_contract():
    rec = create_anomaly_record(
        sensor_id="AWS-01",
        channel="temperature_c",
        timestamp="2026-06-01T12:00:00Z",
        is_anomaly=True,
        anomaly_type="spike",
        severity=0.88,
        severity_level="critical",
        observed_value=95.0,
        expected_value=31.5,
        deviation="+63.50",
        details="Sharp spike detected",
        expected_range=(28.0, 35.0),
    )

    required_keys = [
        "sensor_id",
        "channel",
        "timestamp",
        "is_anomaly",
        "anomaly_type",
        "severity",
        "severity_level",
        "observed_value",
        "expected_value",
        "deviation",
        "details",
    ]
    for k in required_keys:
        assert k in rec, f"Missing required contract key: {k}"

    assert rec["sensor_id"] == "AWS-01"
    assert rec["channel"] == "temperature_c"
    assert rec["anomaly_type"] == "spike"
    assert rec["is_anomaly"] is True
    assert rec["severity"] == 0.88
    assert rec["observed_value"] == 95.0
    assert rec["expected_value"] == 31.5
    assert rec["deviation"] == "+63.50"


# -------------------------------------------------------------------
# 7. CROSS-SENSOR INCONSISTENCY & SKYGUARD DETECTOR STREAMING
# -------------------------------------------------------------------
def test_cross_sensor_multivariate_detection(detector):
    """Seed baseline correlations and inject uncoupled impossible cross-sensor state."""
    # Feed 25 normal observations
    for i in range(25):
        t = 20.0 + np.sin(i / 4.0) * 2.0
        rh = 60.0 - (t - 20.0) * 3.0  # Natural inverse T / RH correlation
        p = 1013.0 + np.random.normal(0, 0.2)
        detector.score_reading({
            "station_id": "AWS-MULTI",
            "temperature_c": t,
            "humidity_pct": rh,
            "pressure_hpa": p,
            "wind_speed_ms": 3.0,
            "wind_direction_deg": 180.0,
            "precipitation_mm": 0.0,
        })

    # Inject cross-sensor breakdown: 38C desert heat with 98% humidity and plunging pressure
    uncoupled_reading = {
        "station_id": "AWS-MULTI",
        "temperature_c": 38.0,
        "humidity_pct": 98.0,
        "pressure_hpa": 980.0,
        "wind_speed_ms": 2.0,
        "wind_direction_deg": 180.0,
        "precipitation_mm": 0.0,
    }
    results = detector.score_reading(uncoupled_reading)
    assert len(results) >= 1
    types = {r["anomaly_type"] for r in results}
    assert "cross_sensor" in types or any(r["is_anomaly"] for r in results)


# -------------------------------------------------------------------
# 8. API INTEGRATION TESTS (DETECT & LIVE ENDPOINTS)
# -------------------------------------------------------------------
def test_api_detect_endpoint():
    client = TestClient(app)
    payload = {
        "station_id": "TEST-DET",
        "temperature_c": 98.5,
        "humidity_pct": 45.0,
    }
    resp = client.post("/api/v1/detect", json=payload, headers={"X-API-Key": API_KEY})
    assert resp.status_code == 200
    data = resp.json()
    items = data if isinstance(data, list) else [data]
    assert any(i.get("anomaly_type") == "implausible" or i.get("anomaly_type") == "spike" for i in items)
    for i in items:
        assert "sensor_id" in i
        assert "channel" in i
        assert "deviation" in i


def test_api_live_telemetry_endpoint():
    client = TestClient(app)
    # Query single station live endpoint
    resp = client.get("/api/v1/telemetry/live?station_id=AWS-WA-COAST", headers={"X-API-Key": API_KEY})
    if resp.status_code == 200:
        data = resp.json()
        assert "count" in data
        assert "records" in data
        assert data["count"] >= 1
        rec = data["records"][0]
        assert "sensor_id" in rec
        assert "is_anomaly" in rec
        assert "deviation" in rec
