"""
SkyGuard Sensor Reliability & Fault Detection Engine
Focused per-channel anomaly detection for weather station sensor reliability.
Implements the 5 core sensor fault checks using stdlib and NumPy.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd

# -------------------------------------------------------------------
# Per-channel physical limits, thresholds, and window parameters
# -------------------------------------------------------------------
CHANNEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "temperature_c": {
        "min_val": -50.0,
        "max_val": 60.0,
        "spike_delta": 5.0,
        "spike_z": 3.5,
        "spike_window": 7,
        "max_stuck_steps": 6,
        "stuck_tolerance": 1e-4,
        "allow_zero_stuck": False,
        "drift_delta": 4.0,
        "drift_long_window": 24,
        "drift_short_window": 6,
        "max_interval_minutes": 15.0,
    },
    "humidity_pct": {
        "min_val": 0.0,
        "max_val": 100.0,
        "spike_delta": 20.0,
        "spike_z": 3.5,
        "spike_window": 7,
        "max_stuck_steps": 12,
        "stuck_tolerance": 1e-4,
        "allow_zero_stuck": False,
        "drift_delta": 25.0,
        "drift_long_window": 24,
        "drift_short_window": 6,
        "max_interval_minutes": 15.0,
    },
    "wind_speed_ms": {
        "min_val": 0.0,
        "max_val": 75.0,
        "spike_delta": 15.0,
        "spike_z": 3.5,
        "spike_window": 7,
        "max_stuck_steps": 6,
        "stuck_tolerance": 1e-4,
        "allow_zero_stuck": True,  # calm conditions can be 0.0 m/s
        "drift_delta": 8.0,
        "drift_long_window": 24,
        "drift_short_window": 6,
        "max_interval_minutes": 15.0,
    },
    "wind_direction_deg": {
        "min_val": 0.0,
        "max_val": 360.0,
        "spike_delta": 180.0,
        "spike_z": 3.5,
        "spike_window": 7,
        "max_stuck_steps": 12,
        "stuck_tolerance": 1e-4,
        "allow_zero_stuck": False,
        "drift_delta": 90.0,
        "drift_long_window": 24,
        "drift_short_window": 6,
        "max_interval_minutes": 15.0,
    },
    "precipitation_mm": {
        "min_val": 0.0,
        "max_val": 200.0,
        "spike_delta": 35.0,
        "spike_z": 3.5,
        "spike_window": 7,
        "max_stuck_steps": 6,
        "stuck_tolerance": 1e-4,
        "allow_zero_stuck": True,  # dry weather can be 0.0 mm
        "drift_delta": 15.0,
        "drift_long_window": 24,
        "drift_short_window": 6,
        "max_interval_minutes": 15.0,
    },
    "pressure_hpa": {
        "min_val": 870.0,
        "max_val": 1085.0,
        "spike_delta": 4.0,
        "spike_z": 3.5,
        "spike_window": 7,
        "max_stuck_steps": 12,
        "stuck_tolerance": 1e-4,
        "allow_zero_stuck": False,
        "drift_delta": 8.0,
        "drift_long_window": 24,
        "drift_short_window": 6,
        "max_interval_minutes": 15.0,
    },
}

SUPPORTED_CHANNELS = list(CHANNEL_CONFIG.keys())


# -------------------------------------------------------------------
# Helper: Parse timestamp
# -------------------------------------------------------------------
def _parse_ts(ts: Any) -> Optional[datetime]:
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    try:
        return pd.to_datetime(ts).to_pydatetime()
    except Exception:
        return None


# -------------------------------------------------------------------
# 1. MISSING / INVALID DATA DETECTION
# -------------------------------------------------------------------
def check_missing_or_invalid(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Flag nulls, NaNs, non-numeric values, or telemetry interval gaps.
    Returns: (is_anomaly, reason, severity)
    """
    if not series or len(series) == 0:
        return True, "Missing payload: channel series is empty", "CRITICAL"

    latest = series[-1]

    # Null or NaN check
    if latest is None or (isinstance(latest, float) and np.isnan(latest)):
        return True, "Invalid reading: null or NaN value received", "CRITICAL"

    # Numeric cast check
    try:
        float(latest)
    except (ValueError, TypeError):
        return True, f"Malformed payload: value '{latest}' cannot be cast to numeric", "CRITICAL"

    # Interval gap check
    if timestamps is not None and len(timestamps) >= 2:
        cfg = config or {}
        max_interval = float(cfg.get("max_interval_minutes", 15.0))
        t1 = _parse_ts(timestamps[-1])
        t0 = _parse_ts(timestamps[-2])
        if t1 and t0:
            diff_min = (t1 - t0).total_seconds() / 60.0
            if diff_min > max_interval:
                sev = "CRITICAL" if diff_min >= 3 * max_interval else "WARNING"
                return (
                    True,
                    f"Telemetry reporting gap: {diff_min:.1f}m interval exceeds {max_interval:.1f}m threshold",
                    sev,
                )

    return False, "Normal", "NORMAL"


# -------------------------------------------------------------------
# 2. PHYSICAL PLAUSIBILITY CHECKS
# -------------------------------------------------------------------
def check_physical_plausibility(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Flag values outside physically possible meteorological bounds.
    Returns: (is_anomaly, reason, severity)
    """
    if not series:
        return False, "Normal", "NORMAL"

    try:
        val = float(series[-1])
    except (ValueError, TypeError):
        return False, "Normal", "NORMAL"  # Handled by missing/invalid check

    if np.isnan(val):
        return False, "Normal", "NORMAL"

    cfg = config or {}
    min_val = cfg.get("min_val")
    max_val = cfg.get("max_val")

    if min_val is not None and val < min_val:
        return True, f"Physical bound violation: {val} is below physical minimum of {min_val}", "CRITICAL"
    if max_val is not None and val > max_val:
        return True, f"Physical bound violation: {val} exceeds physical maximum of {max_val}", "CRITICAL"

    return False, "Normal", "NORMAL"


# -------------------------------------------------------------------
# 3. SPIKE DETECTION
# -------------------------------------------------------------------
def check_spike(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Flag sudden departure from recent neighboring trend using rolling median / MAD.
    Returns: (is_anomaly, reason, severity)
    """
    cfg = config or {}
    window = int(cfg.get("spike_window", 7))
    threshold_delta = float(cfg.get("spike_delta", 5.0))
    z_thresh = float(cfg.get("spike_z", 3.5))

    if len(series) < 3:
        return False, "Normal", "NORMAL"

    # Take context window
    recent = series[-window:]
    try:
        clean = [float(x) for x in recent if x is not None and not np.isnan(float(x))]
    except (ValueError, TypeError):
        return False, "Normal", "NORMAL"

    if len(clean) < 3:
        return False, "Normal", "NORMAL"

    curr_val = clean[-1]
    prev_vals = np.array(clean[:-1], dtype=float)

    med = float(np.median(prev_vals))
    delta = abs(curr_val - med)

    # Median absolute deviation (MAD)
    abs_deviations = np.abs(prev_vals - med)
    mad = float(np.median(abs_deviations))

    # Evaluate spike conditions
    if mad > 1e-6:
        robust_z = delta / (1.4826 * mad)
        is_spike = (delta >= threshold_delta) and (robust_z >= z_thresh)
    else:
        # Zero variance in history, any step >= threshold_delta is a sharp spike
        is_spike = delta >= threshold_delta

    if is_spike:
        sev = "CRITICAL" if delta >= 2.0 * threshold_delta else "WARNING"
        return (
            True,
            f"Sharp spike detected: {curr_val:.2f} deviates by {delta:.2f} from recent median ({med:.2f})",
            sev,
        )

    return False, "Normal", "NORMAL"


# -------------------------------------------------------------------
# 4. STUCK-AT-VALUE DETECTION
# -------------------------------------------------------------------
def check_stuck_value(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Flag sensor reporting identical reading for longer than reasonable duration.
    Returns: (is_anomaly, reason, severity)
    """
    cfg = config or {}
    max_steps = int(cfg.get("max_stuck_steps", 6))
    tol = float(cfg.get("stuck_tolerance", 1e-4))
    allow_zero = bool(cfg.get("allow_zero_stuck", False))

    if len(series) < max_steps:
        return False, "Normal", "NORMAL"

    try:
        window_vals = [float(x) for x in series[-max_steps:] if x is not None]
    except (ValueError, TypeError):
        return False, "Normal", "NORMAL"

    if len(window_vals) < max_steps:
        return False, "Normal", "NORMAL"

    arr = np.array(window_vals, dtype=float)
    if np.any(np.isnan(arr)):
        return False, "Normal", "NORMAL"

    # Natural zero calm/dry exemption
    if allow_zero and np.all(np.abs(arr) <= tol):
        return False, "Normal", "NORMAL"

    val_spread = float(np.max(arr) - np.min(arr))
    if val_spread <= tol:
        val = arr[-1]
        sev = "CRITICAL" if len(series) >= 2 * max_steps else "WARNING"
        return (
            True,
            f"Stuck-at-value freeze: sensor invariant at {val:.2f} across {max_steps} consecutive readings",
            sev,
        )

    return False, "Normal", "NORMAL"


# -------------------------------------------------------------------
# 5. DRIFT DETECTION
# -------------------------------------------------------------------
def check_drift(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, str]:
    """
    Flag gradual, sustained calibration drift from baseline trend over time.
    Returns: (is_anomaly, reason, severity)
    """
    cfg = config or {}
    long_win = int(cfg.get("drift_long_window", 24))
    short_win = int(cfg.get("drift_short_window", 6))
    drift_delta = float(cfg.get("drift_delta", 4.0))

    if len(series) < long_win:
        return False, "Normal", "NORMAL"

    try:
        window_vals = [float(x) for x in series[-long_win:] if x is not None]
    except (ValueError, TypeError):
        return False, "Normal", "NORMAL"

    if len(window_vals) < long_win:
        return False, "Normal", "NORMAL"

    arr = np.array(window_vals, dtype=float)
    if np.any(np.isnan(arr)):
        return False, "Normal", "NORMAL"

    # Baseline: mean of early baseline portion
    baseline = float(np.mean(arr[:-short_win]))
    recent_mean = float(np.mean(arr[-short_win:]))
    divergence = abs(recent_mean - baseline)

    # Monotonic linear slope across window
    x = np.arange(len(arr))
    slope, _ = np.polyfit(x, arr, 1)
    total_drift = abs(slope * len(arr))

    # Drift is flagged when both the short-vs-baseline departure and overall slope drift exceed threshold
    if divergence >= drift_delta and total_drift >= drift_delta:
        sev = "CRITICAL" if divergence >= 1.75 * drift_delta else "WARNING"
        direction = "upward" if recent_mean > baseline else "downward"
        return (
            True,
            f"Sustained {direction} drift: departure of {divergence:.2f} from baseline over {long_win} readings (slope: {slope:+.3f}/step)",
            sev,
        )

    return False, "Normal", "NORMAL"


# -------------------------------------------------------------------
# Incident Generation & Unified Evaluation
# -------------------------------------------------------------------
CHECK_FUNCTIONS = [
    ("MISSING_OR_INVALID", check_missing_or_invalid),
    ("PHYSICAL_PLAUSIBILITY", check_physical_plausibility),
    ("SPIKE", check_spike),
    ("STUCK_AT_VALUE", check_stuck_value),
    ("DRIFT", check_drift),
]


def create_incident(
    check_type: str,
    channel: str,
    station_id: str,
    timestamp: Any,
    observed_value: Any,
    severity: str,
    description: str,
) -> Dict[str, Any]:
    """
    Standard SkyGuard incident schema plugged into SSE stream and dashboard.
    """
    return {
        "type": check_type,
        "timestamp": str(timestamp),
        "channel": channel,
        "station_id": station_id,
        "observed_value": observed_value,
        "severity": severity,
        "description": description,
        "is_anomaly": True,
        # Plug-and-play fields for SkyGuard UI & SSE subscribers
        "primary_sensor": channel,
        "fault_mode": check_type,
        "rule_diagnosis": description,
    }


def evaluate_channel(
    channel: str,
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    station_id: str = "AWS-LIVE",
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """
    Run all 5 checks on a single channel's reading history and return active incidents for latest reading.
    """
    cfg = config or CHANNEL_CONFIG.get(channel, {})
    incidents: List[Dict[str, Any]] = []

    if not series:
        return incidents

    latest_val = series[-1]
    latest_ts = timestamps[-1] if timestamps and len(timestamps) > 0 else datetime.now(timezone.utc).isoformat()

    # 1. Missing / Invalid check
    is_anom, reason, sev = check_missing_or_invalid(series, timestamps, cfg)
    if is_anom:
        incidents.append(create_incident("MISSING_OR_INVALID", channel, station_id, latest_ts, latest_val, sev, reason))
        # Skip subsequent numerical math if reading is missing/invalid
        return incidents

    # 2. Physical Plausibility check
    is_anom, reason, sev = check_physical_plausibility(series, timestamps, cfg)
    if is_anom:
        incidents.append(create_incident("PHYSICAL_PLAUSIBILITY", channel, station_id, latest_ts, latest_val, sev, reason))

    # 3. Spike check
    is_anom, reason, sev = check_spike(series, timestamps, cfg)
    if is_anom:
        incidents.append(create_incident("SPIKE", channel, station_id, latest_ts, latest_val, sev, reason))

    # 4. Stuck-at-value check
    is_anom, reason, sev = check_stuck_value(series, timestamps, cfg)
    if is_anom:
        incidents.append(create_incident("STUCK_AT_VALUE", channel, station_id, latest_ts, latest_val, sev, reason))

    # 5. Drift check
    is_anom, reason, sev = check_drift(series, timestamps, cfg)
    if is_anom:
        incidents.append(create_incident("DRIFT", channel, station_id, latest_ts, latest_val, sev, reason))

    return incidents


def evaluate_telemetry(
    df_or_records: Any,
    channels: Optional[List[str]] = None,
    station_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Convenience function to scan tabular telemetry history and return any sensor incidents for the latest step.
    """
    if isinstance(df_or_records, list):
        df = pd.DataFrame(df_or_records)
    elif isinstance(df_or_records, pd.DataFrame):
        df = df_or_records
    else:
        return []

    if df.empty:
        return []

    target_channels = channels or [c for c in SUPPORTED_CHANNELS if c in df.columns]
    timestamps = df["timestamp"].tolist() if "timestamp" in df.columns else None
    stn = station_id or (df["station_id"].iloc[-1] if "station_id" in df.columns else "AWS-LIVE")

    all_incidents: List[Dict[str, Any]] = []
    for ch in target_channels:
        series = df[ch].tolist()
        incidents = evaluate_channel(ch, series, timestamps=timestamps, station_id=stn)
        all_incidents.extend(incidents)

    return all_incidents
