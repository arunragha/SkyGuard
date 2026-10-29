"""
Unit & Integration Tests for SkyGuard Real-Time Anomaly Monitoring API
"""

import os
import sys
import pytest
from starlette.testclient import TestClient

# Ensure root directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from skyguard.api import (
    app,
    API_KEY,
    CLIENT_REQUEST_LOGS,
    RATE_LIMIT_MAX_REQUESTS,
    RECENT_ANOMALIES,
)


@pytest.fixture(autouse=True)
def clean_state():
    """Reset rate limiter and anomalies buffer before each test."""
    CLIENT_REQUEST_LOGS.clear()
    yield
    CLIENT_REQUEST_LOGS.clear()


@pytest.fixture
def client():
    return TestClient(app)


def test_health_check_unauthenticated(client):
    """GET /health is public and returns 200 with service health stats."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "uptime_seconds" in data
    assert "pipeline_fitted" in data
    assert "active_sse_subscribers" in data


def test_authentication_required(client):
    """Protected endpoints reject requests without valid credentials with 401."""
    # No credentials
    res1 = client.get("/api/v1/anomalies/recent")
    assert res1.status_code == 401
    assert res1.json()["error"] == "Unauthorized"

    # Invalid header
    res2 = client.get("/api/v1/anomalies/recent", headers={"X-API-Key": "wrong-key"})
    assert res2.status_code == 401

    # Valid X-API-Key header
    res3 = client.get("/api/v1/anomalies/recent", headers={"X-API-Key": API_KEY})
    assert res3.status_code == 200

    # Valid Bearer token
    res4 = client.get(
        "/api/v1/anomalies/recent",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    assert res4.status_code == 200

    # Valid query param token (for SSE browser clients)
    res5 = client.get(f"/api/v1/anomalies/recent?token={API_KEY}")
    assert res5.status_code == 200


def test_ingest_telemetry_single(client):
    """POST /api/v1/telemetry evaluates sensor reading and returns anomaly score."""
    sample_payload = {
        "station_id": "TEST-STATION-1",
        "temperature_c": 24.5,
        "humidity_pct": 65.0,
        "pressure_hpa": 1013.2,
        "wind_speed_ms": 3.4,
        "wind_direction_deg": 180.0,
        "solar_radiation_wm2": 450.0,
        "precipitation_mm": 0.0,
        "battery_voltage_v": 13.8,
        "internal_temp_c": 28.1,
    }

    response = client.post(
        "/api/v1/telemetry",
        json=sample_payload,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["station_id"] == "TEST-STATION-1"
    assert "ensemble_score" in data
    assert 0.0 <= data["ensemble_score"] <= 1.0
    assert data["severity"] in ["NORMAL", "LOW", "WARNING", "CRITICAL"]
    assert isinstance(data["is_anomaly"], bool)
    assert "primary_sensor" in data
    assert "fault_mode" in data
    assert "scores" in data
    assert "iforest" in data["scores"]
    assert "pca" in data["scores"]
    assert "lof" in data["scores"]
    assert "rules" in data["scores"]


def test_ingest_telemetry_batch(client):
    """POST /api/v1/telemetry handles batch arrays of observations."""
    batch_payload = [
        {
            "station_id": f"TEST-STATION-{i}",
            "temperature_c": 20.0 + i,
            "humidity_pct": 50.0,
            "pressure_hpa": 1012.0,
            "wind_speed_ms": 2.0,
            "wind_direction_deg": 90.0,
            "solar_radiation_wm2": 300.0,
            "precipitation_mm": 0.0,
            "battery_voltage_v": 13.5,
            "internal_temp_c": 25.0,
        }
        for i in range(3)
    ]

    response = client.post(
        "/api/v1/telemetry",
        json=batch_payload,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) == 3
    assert results[0]["station_id"] == "TEST-STATION-0"
    assert results[2]["station_id"] == "TEST-STATION-2"


def test_ingest_telemetry_validation(client):
    """POST /api/v1/telemetry rejects invalid bodies."""
    # Empty list
    res_empty = client.post(
        "/api/v1/telemetry",
        json=[],
        headers={"X-API-Key": API_KEY},
    )
    assert res_empty.status_code == 400

    # Invalid non-json
    res_invalid = client.post(
        "/api/v1/telemetry",
        content="not-a-json",
        headers={"X-API-Key": API_KEY, "Content-Type": "application/json"},
    )
    assert res_invalid.status_code == 400


def test_get_recent_anomalies(client):
    """GET /api/v1/anomalies/recent retrieves cached anomalies with filters."""
    # Ingest an observation to guarantee at least one item
    client.post(
        "/api/v1/telemetry",
        json={
            "station_id": "CACHE-TEST",
            "temperature_c": 21.0,
            "humidity_pct": 55.0,
            "pressure_hpa": 1013.0,
            "wind_speed_ms": 1.0,
            "wind_direction_deg": 0.0,
            "solar_radiation_wm2": 100.0,
            "precipitation_mm": 0.0,
            "battery_voltage_v": 13.0,
            "internal_temp_c": 22.0,
        },
        headers={"X-API-Key": API_KEY},
    )

    response = client.get(
        "/api/v1/anomalies/recent?limit=10",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert "count" in data
    assert "anomalies" in data
    assert data["count"] >= 1
    assert any(a["station_id"] == "CACHE-TEST" for a in data["anomalies"])


@pytest.mark.anyio
async def test_sse_stream_receives_events():
    """GET /api/v1/anomalies/stream yields SSE frames upon telemetry ingestion."""
    from skyguard.api import stream_anomalies, SUBSCRIBERS
    from starlette.requests import Request

    scope = {"type": "http", "method": "GET", "path": "/api/v1/anomalies/stream", "headers": []}
    async def receive():
        return {"type": "http.request"}
    req = Request(scope, receive)

    resp = await stream_anomalies(req)
    assert resp.media_type == "text/event-stream"
    assert resp.status_code == 200
    assert len(SUBSCRIBERS) >= 1

    # Verify initial connection frame
    gen = resp.body_iterator
    first_chunk = await anext(gen)
    assert ": connected to SkyGuard SSE stream" in first_chunk

    # Push an anomaly to the subscriber queue and verify event frame
    sub_queue = SUBSCRIBERS[-1]
    test_event = {"station_id": "SSE-TEST", "ensemble_score": 0.88, "severity": "CRITICAL"}
    await sub_queue.put(test_event)

    second_chunk = await anext(gen)
    assert "event: anomaly" in second_chunk
    assert "SSE-TEST" in second_chunk

    await gen.aclose()


def test_rate_limiting(client):
    """Sliding-window rate limiter returns 429 once request limit exceeded."""
    import time
    # Simulate client reaching rate limit within active window
    ip = "testclient"
    now = time.monotonic()
    CLIENT_REQUEST_LOGS[ip] = [now] * RATE_LIMIT_MAX_REQUESTS

    response = client.get(
        "/api/v1/anomalies/recent",
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 429
    assert response.json()["error"] == "Too Many Requests"
    assert "Retry-After" in response.headers


def test_sensors_evaluate_endpoint(client):
    """POST /api/v1/sensors/evaluate detects sensor reliability faults."""
    faulty_payload = [
        {"station_id": "TEST-1", "temperature_c": 22.0, "humidity_pct": 55.0},
        {"station_id": "TEST-1", "temperature_c": 22.1, "humidity_pct": 55.0},
        {"station_id": "TEST-1", "temperature_c": 22.2, "humidity_pct": 55.0},
        {"station_id": "TEST-1", "temperature_c": 92.0, "humidity_pct": 140.0},
    ]
    response = client.post(
        "/api/v1/sensors/evaluate",
        json=faulty_payload,
        headers={"X-API-Key": API_KEY},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["count"] >= 2
    types = {i["type"] for i in data["incidents"]}
    assert "SPIKE" in types or "PHYSICAL_PLAUSIBILITY" in types


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
