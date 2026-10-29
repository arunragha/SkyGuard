"""
Integration tests for FastAPI REST API endpoints: /detect, /stations, /anomalies, /health.
"""

from fastapi.testclient import TestClient
import pytest

from src.api import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "endpoints" in data


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["models_fitted"] is True


def test_stations_endpoint():
    response = client.get("/stations")
    assert response.status_code == 200
    stations = response.json()
    assert isinstance(stations, list)
    assert len(stations) > 0
    first_st = stations[0]
    assert "station_id" in first_st
    assert "status" in first_st
    assert "latest_readings" in first_st


def test_detect_endpoint():
    payload = {
        "records": [
            {
                "timestamp": "2026-06-01T12:00:00Z",
                "station_id": "STATION_TEST",
                "temperature": 24.5,
                "humidity": 60.0,
                "barometric_pressure": 1013.25,
                "wind_speed": 4.5,
                "wind_direction": 180.0,
                "rainfall": 0.0,
                "solar_radiation": 800.0,
            },
            {
                "timestamp": "2026-06-01T12:10:00Z",
                "station_id": "STATION_TEST",
                "temperature": 52.0,  # massive spike
                "humidity": 10.0,
                "barometric_pressure": 1013.0,
                "wind_speed": 4.8,
                "wind_direction": 182.0,
                "rainfall": 0.0,
                "solar_radiation": 820.0,
            },
        ],
        "sensitivity": 0.5,
        "threshold_method": "percentile",
    }

    response = client.post("/detect", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["total_records"] == 2
    assert len(data["results"]) == 2

    # Check result fields
    res1 = data["results"][0]
    assert "anomaly_score" in res1
    assert "anomaly_threshold" in res1
    assert "predicted_anomaly" in res1
    assert "predicted_type" in res1
    assert "model_scores" in res1
    assert 0.0 <= res1["anomaly_score"] <= 1.0


def test_anomalies_endpoint():
    response = client.get("/anomalies?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "total_matches" in data
    assert "anomalies" in data
    assert isinstance(data["anomalies"], list)
