"""
SkyGuard Real-Time Anomaly Monitoring API
Lightweight Starlette + Uvicorn server providing REST endpoints and SSE live streaming.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timezone
import json
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))

from skyguard.data.live_feed import (
    LIVE_WEATHER_STATIONS,
    fetch_all_live_stations,
    fetch_live_station_telemetry,
)
from skyguard.data.weather_generator import WeatherDataGenerator
from skyguard.detector import (
    CHANNEL_CONFIG,
    SUPPORTED_CHANNELS,
    SkyGuardDetector,
)
from skyguard.pipeline import SkyGuardPipeline
from skyguard.sensor_checks import evaluate_telemetry

# -------------------------------------------------------------
# Configuration & In-Memory State
# -------------------------------------------------------------
API_KEY = os.environ.get("SKYGUARD_API_KEY", "skyguard-secret-key")
RATE_LIMIT_MAX_REQUESTS = int(os.environ.get("SKYGUARD_RATE_LIMIT", "120"))
RATE_LIMIT_WINDOW_SECONDS = 60.0

START_TIME = time.time()
RECENT_ANOMALIES: deque = deque(maxlen=1000)
SUBSCRIBERS: List[asyncio.Queue] = []
CLIENT_REQUEST_LOGS: Dict[str, List[float]] = defaultdict(list)
STATION_TELEMETRY_HISTORY: Dict[str, deque] = defaultdict(lambda: deque(maxlen=60))

# Pipeline initialization
pipeline = SkyGuardPipeline()


def get_or_init_pipeline() -> SkyGuardPipeline:
    """Ensures pipeline is fitted with baseline benchmark data."""
    if not pipeline.feature_cols_:
        benchmark_csv = os.path.join(
            os.path.dirname(__file__), "..", "data", "weather_station_telemetry.csv"
        )
        if os.path.exists(benchmark_csv):
            df = pd.read_csv(benchmark_csv)
        else:
            gen = WeatherDataGenerator(seed=42)
            df = gen.generate_telemetry(periods=500, freq="10min", inject_faults=False)
        pipeline.fit(df)
    return pipeline


# -------------------------------------------------------------
# Auth & Rate Limit Middleware
# -------------------------------------------------------------
class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Health check, root, and static web UI assets are public
        if request.method == "OPTIONS" or not request.url.path.startswith("/api"):
            return await call_next(request)

        # 1. Rate Limiting (Sliding Window)
        client_ip = request.client.host if request.client else "unknown"
        now = time.monotonic()
        req_times = [t for t in CLIENT_REQUEST_LOGS[client_ip] if now - t < RATE_LIMIT_WINDOW_SECONDS]
        CLIENT_REQUEST_LOGS[client_ip] = req_times

        if len(req_times) >= RATE_LIMIT_MAX_REQUESTS:
            return JSONResponse(
                {"error": "Too Many Requests", "detail": f"Exceeded {RATE_LIMIT_MAX_REQUESTS} req/min limit"},
                status_code=429,
                headers={"Retry-After": "60"},
            )
        CLIENT_REQUEST_LOGS[client_ip].append(now)

        # 2. Authentication
        token = request.headers.get("X-API-Key")
        auth_header = request.headers.get("Authorization", "")
        if not token and auth_header.startswith("Bearer "):
            token = auth_header[7:].strip()
        # Query parameter fallback for SSE browser clients
        if not token and "token" in request.query_params:
            token = request.query_params["token"]

        if not token or token != API_KEY:
            return JSONResponse(
                {"error": "Unauthorized", "detail": "Valid API key required via X-API-Key or Bearer token"},
                status_code=401,
            )

        return await call_next(request)


# -------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------
async def health(request: Request) -> JSONResponse:
    """Health & liveness check."""
    return JSONResponse({
        "status": "healthy",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "pipeline_fitted": len(pipeline.feature_cols_) > 0,
        "active_sse_subscribers": len(SUBSCRIBERS),
        "cached_detections": len(RECENT_ANOMALIES),
    })


detector = SkyGuardDetector()


async def detect_endpoint(request: Request) -> JSONResponse:
    """
    Dedicated unsupervised sensor anomaly detection endpoint conforming to output contract.
    Evaluates spikes, stuck values, drift, missing/invalid data, physical plausibility,
    and cross-sensor multivariate inconsistency.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Bad Request", "detail": "Invalid JSON body"}, status_code=400)

    records = body if isinstance(body, list) else [body]
    if not records:
        return JSONResponse({"error": "Bad Request", "detail": "Empty records array"}, status_code=400)

    all_anomalies = []
    for r in records:
        res = detector.score_reading(r)
        all_anomalies.extend(res)

    return JSONResponse(all_anomalies[0] if not isinstance(body, list) else all_anomalies)


async def live_telemetry_endpoint(request: Request) -> JSONResponse:
    """
    Fetch genuine live real-time weather observations from Open-Meteo external API,
    evaluate via unsupervised sensor detection engine, broadcast to active SSE subscribers,
    and return scored records.
    """
    station_id = request.query_params.get("station_id")
    if station_id:
        reading = fetch_live_station_telemetry(station_id)
        live_records = [reading] if reading else []
    else:
        live_records = fetch_all_live_stations()

    if not live_records:
        return JSONResponse(
            {"error": "Feed Unavailable", "detail": "Could not retrieve live observations from Open-Meteo API"},
            status_code=502,
        )

    scored_records = []
    for rec in live_records:
        detections = detector.score_reading(rec)
        for anom in detections:
            # Sync with SSE broadcast
            for q in list(SUBSCRIBERS):
                try:
                    q.put_nowait(anom)
                except asyncio.QueueFull:
                    try:
                        q.get_nowait()
                        q.put_nowait(anom)
                    except Exception:
                        pass
            if anom.get("is_anomaly"):
                RECENT_ANOMALIES.append(anom)
        scored_records.extend(detections)

    return JSONResponse({"count": len(scored_records), "records": scored_records})


async def ingest_telemetry(request: Request) -> JSONResponse:
    """
    Ingest a telemetry observation, synchronously evaluate anomaly models,
    broadcast to active SSE listeners, and return scored record.
    """
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Bad Request", "detail": "Invalid JSON body"}, status_code=400)

    records = body if isinstance(body, list) else [body]
    if not records:
        return JSONResponse({"error": "Bad Request", "detail": "Empty records array"}, status_code=400)

    pipe = get_or_init_pipeline()
    df_raw = pd.DataFrame(records)

    # Validate basic sensor columns
    if "timestamp" not in df_raw.columns:
        df_raw["timestamp"] = datetime.now(timezone.utc).isoformat()
    if "station_id" not in df_raw.columns:
        df_raw["station_id"] = "AWS-LIVE"

    processed = pipe.process(df_raw)

    response_items = []
    for idx in range(len(processed)):
        row = processed.iloc[idx]
        raw_dict = dict(row)
        stn = str(row["station_id"])
        STATION_TELEMETRY_HISTORY[stn].append(raw_dict)

        # Run primary unsupervised sensor detector
        detector_results = detector.score_reading(raw_dict, station_id=stn)
        flagged_anoms = [a for a in detector_results if a.get("is_anomaly")]

        # Run legacy evaluator for supplementary rule incidents
        sensor_incidents = evaluate_telemetry(list(STATION_TELEMETRY_HISTORY[stn]), station_id=stn)

        is_anom = bool(row.get("is_anomaly_pred", 0) == 1) or len(flagged_anoms) > 0
        primary_sensor = "Normal"
        fault_mode = "None"
        rule_diagnosis = str(row.get("rule_diagnosis", "Normal"))
        row_severity = str(row.get("severity", "NORMAL"))
        observed_val = None
        expected_val = None
        expected_range = None
        deviation_val = "+0.00"

        if flagged_anoms:
            is_anom = True
            primary_anom = flagged_anoms[0]
            primary_sensor = primary_anom.get("channel", "Unknown")
            fault_mode = primary_anom.get("anomaly_type", "fault")
            rule_diagnosis = primary_anom.get("details", "")
            row_severity = primary_anom.get("severity_level", "critical").upper()
            observed_val = primary_anom.get("observed_value")
            expected_val = primary_anom.get("expected_value")
            expected_range = primary_anom.get("expected_range")
            deviation_val = primary_anom.get("deviation", "+0.00")
        elif sensor_incidents:
            is_anom = True
            primary_inc = sensor_incidents[0]
            primary_sensor = primary_inc["channel"]
            fault_mode = primary_inc["type"]
            rule_diagnosis = primary_inc["description"]
            inc_sevs = {inc["severity"] for inc in sensor_incidents}
            if "CRITICAL" in inc_sevs:
                row_severity = "CRITICAL"
            elif "WARNING" in inc_sevs and row_severity in ("NORMAL", "LOW"):
                row_severity = "WARNING"
            observed_val = primary_inc.get("observed_value")
        elif is_anom or row.get("ensemble_score", 0.0) >= 0.45:
            try:
                rca = pipe.diagnose(idx, df_raw)
                primary_sensor = rca.get("primary_culprit_sensor", "Unknown")
                fault_mode = rca.get("fault_mode", "Unspecified")
            except Exception:
                primary_sensor = "Multi-Sensor Coupling"
                fault_mode = "Anomaly"

        item = {
            # Standard Output Contract Fields
            "sensor_id": stn,
            "station_id": stn,
            "channel": primary_sensor,
            "primary_sensor": primary_sensor,
            "timestamp": str(row["timestamp"]),
            "is_anomaly": is_anom,
            "anomaly_type": fault_mode.lower() if fault_mode != "None" else "none",
            "fault_mode": fault_mode,
            "severity": row_severity,
            "severity_score": float(np.round(row["ensemble_score"], 4)) if not flagged_anoms else float(flagged_anoms[0].get("severity", 0.85)),
            "severity_level": row_severity.lower(),
            "observed_value": observed_val,
            "expected_value": expected_val,
            "expected_range": expected_range,
            "deviation": deviation_val,
            "details": rule_diagnosis,
            # Supplementary fields for existing dashboard & SSE components
            "ensemble_score": float(np.round(row["ensemble_score"], 4)),
            "rule_diagnosis": rule_diagnosis,
            "sensor_incidents": sensor_incidents,
            "detector_records": detector_results,
            "scores": {
                "iforest": float(row.get("score_iforest", 0.0)),
                "pca": float(row.get("score_pca", 0.0)),
                "lof": float(row.get("score_lof", 0.0)),
                "rules": float(row.get("score_rules", 0.0)),
            },
        }

        RECENT_ANOMALIES.append(item)
        response_items.append(item)

        # Broadcast to active SSE subscribers (with queue backpressure protection)
        for q in list(SUBSCRIBERS):
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                # Discard oldest to avoid stalling
                try:
                    q.get_nowait()
                    q.put_nowait(item)
                except Exception:
                    pass

    return JSONResponse(response_items[0] if not isinstance(body, list) else response_items)


async def get_recent_anomalies(request: Request) -> JSONResponse:
    """Retrieve historical recent anomalies from ring buffer."""
    limit = min(int(request.query_params.get("limit", "50")), 500)
    min_sev = request.query_params.get("min_severity", "ALL").upper()

    sev_order = {"NORMAL": 0, "LOW": 1, "WARNING": 2, "CRITICAL": 3}
    min_rank = sev_order.get(min_sev, 0)

    filtered = [
        item for item in reversed(RECENT_ANOMALIES)
        if min_sev == "ALL" or sev_order.get(item["severity"], 0) >= min_rank
    ][:limit]

    return JSONResponse({"count": len(filtered), "anomalies": filtered})


async def stream_anomalies(request: Request) -> StreamingResponse:
    """Server-Sent Events (SSE) stream pushing detections live as they occur."""
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    SUBSCRIBERS.append(queue)

    async def event_generator() -> AsyncGenerator[str, None]:
        try:
            # Initial ping to establish connection
            yield f": connected to SkyGuard SSE stream at {datetime.now(timezone.utc).isoformat()}\n\n"
            while True:
                # Check client disconnect
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                    yield f"event: anomaly\ndata: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    # Heartbeat keep-alive comment
                    yield ": keepalive\n\n"
        finally:
            if queue in SUBSCRIBERS:
                SUBSCRIBERS.remove(queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


async def evaluate_sensor_checks_endpoint(request: Request) -> JSONResponse:
    """Evaluates per-channel sensor checks on provided telemetry history."""
    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Bad Request", "detail": "Invalid JSON body"}, status_code=400)

    records = body if isinstance(body, list) else [body]
    if not records:
        return JSONResponse({"error": "Bad Request", "detail": "Empty records array"}, status_code=400)

    df_raw = pd.DataFrame(records)
    incidents = evaluate_telemetry(df_raw)
    return JSONResponse({"count": len(incidents), "incidents": incidents})


# -------------------------------------------------------------
# Application Setup
# -------------------------------------------------------------
routes = [
    Route("/health", health, methods=["GET"]),
    Route("/api/v1/telemetry", ingest_telemetry, methods=["POST"]),
    Route("/api/v1/telemetry/live", live_telemetry_endpoint, methods=["GET", "POST"]),
    Route("/api/v1/detect", detect_endpoint, methods=["POST"]),
    Route("/api/v1/telemetry/evaluate", evaluate_sensor_checks_endpoint, methods=["POST"]),
    Route("/api/v1/sensors/evaluate", evaluate_sensor_checks_endpoint, methods=["POST"]),
    Route("/api/v1/anomalies/recent", get_recent_anomalies, methods=["GET"]),
    Route("/api/v1/anomalies/stream", stream_anomalies, methods=["GET"]),
    Mount("/", app=StaticFiles(directory=WEB_DIR, html=True), name="web"),
]

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    get_or_init_pipeline()
    live_task = None
    if os.environ.get("SKYGUARD_LIVE_POLL", "0") in ("1", "true", "True"):
        async def _background_live_poller():
            poll_interval = float(os.environ.get("SKYGUARD_LIVE_INTERVAL", "60"))
            while True:
                try:
                    await asyncio.sleep(poll_interval)
                    records = await asyncio.to_thread(fetch_all_live_stations)
                    for rec in records:
                        detections = detector.score_reading(rec)
                        for anom in detections:
                            for q in list(SUBSCRIBERS):
                                try:
                                    q.put_nowait(anom)
                                except Exception:
                                    pass
                            if anom.get("is_anomaly"):
                                RECENT_ANOMALIES.append(anom)
                except asyncio.CancelledError:
                    break
                except Exception:
                    pass
        live_task = asyncio.create_task(_background_live_poller())
    yield
    if live_task:
        live_task.cancel()


from starlette.middleware.cors import CORSMiddleware

app = Starlette(
    routes=routes,
    middleware=[
        Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]),
        Middleware(SecurityMiddleware),
    ],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("skyguard.api:app", host="0.0.0.0", port=port, log_level="info")
