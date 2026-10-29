"""
SkyGuard — Unsupervised Sensor Anomaly Detection Engine
Scores incoming weather-station sensor readings in real time and flags:
  1. Spikes (rolling median + MAD)
  2. Stuck values (near-zero variance over natural variability window)
  3. Drift (slow divergence from baseline vs diurnal swings)
  4. Missing/invalid data (nulls, NaNs, schema errors, reporting gaps)
  5. Physically implausible values (hard domain bounds)
  6. Cross-sensor inconsistency (multivariate PCA reconstruction error across joint channels)

Adheres strictly to the SkyGuard Output Contract and Ponytail simplicity directives.
"""

from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# -------------------------------------------------------------------
# Channel Configuration & Domain Bounds
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
        "unit": "°C",
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
        "unit": "%",
    },
    "wind_speed_ms": {
        "min_val": 0.0,
        "max_val": 75.0,
        "spike_delta": 15.0,
        "spike_z": 3.5,
        "spike_window": 7,
        "max_stuck_steps": 6,
        "stuck_tolerance": 1e-4,
        "allow_zero_stuck": True,  # Calm air naturally holds 0.0 m/s
        "drift_delta": 8.0,
        "drift_long_window": 24,
        "drift_short_window": 6,
        "max_interval_minutes": 15.0,
        "unit": "m/s",
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
        "unit": "°",
    },
    "precipitation_mm": {
        "min_val": 0.0,
        "max_val": 200.0,
        "spike_delta": 35.0,
        "spike_z": 3.5,
        "spike_window": 7,
        "max_stuck_steps": 6,
        "stuck_tolerance": 1e-4,
        "allow_zero_stuck": True,  # Dry periods naturally hold 0.0 mm
        "drift_delta": 15.0,
        "drift_long_window": 24,
        "drift_short_window": 6,
        "max_interval_minutes": 15.0,
        "unit": "mm",
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
        "unit": "hPa",
    },
}

SUPPORTED_CHANNELS = list(CHANNEL_CONFIG.keys())

CHANNEL_ALIASES: Dict[str, str] = {
    "temperature": "temperature_c",
    "temperature_c": "temperature_c",
    "temp": "temperature_c",
    "temp_c": "temperature_c",
    "temperature_2m": "temperature_c",
    "humidity": "humidity_pct",
    "humidity_pct": "humidity_pct",
    "rh": "humidity_pct",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed": "wind_speed_ms",
    "wind_speed_ms": "wind_speed_ms",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction": "wind_direction_deg",
    "wind_direction_deg": "wind_direction_deg",
    "wind_direction_10m": "wind_direction_deg",
    "precipitation": "precipitation_mm",
    "precipitation_mm": "precipitation_mm",
    "rainfall": "precipitation_mm",
    "rain": "precipitation_mm",
    "pressure": "pressure_hpa",
    "pressure_hpa": "pressure_hpa",
    "surface_pressure": "pressure_hpa",
}


def canonical_channel_name(ch: str) -> str:
    """Resolve alias to standardized channel key."""
    return CHANNEL_ALIASES.get(ch.lower().strip(), ch)


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
# Output Contract Helper
# -------------------------------------------------------------------
def create_anomaly_record(
    sensor_id: str,
    channel: str,
    timestamp: Any,
    is_anomaly: bool,
    anomaly_type: str,
    severity: float,
    severity_level: str,
    observed_value: Any,
    expected_value: Any,
    deviation: Union[str, float],
    details: str,
    expected_range: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> Dict[str, Any]:
    """
    Standardized SkyGuard anomaly output contract.
    Both API, SSE stream, dashboard, and CLI consume this uniform structure.
    """
    ts_str = str(timestamp) if timestamp is not None else datetime.now(timezone.utc).isoformat()
    dev_str = f"{deviation:+.2f}" if isinstance(deviation, (int, float)) else str(deviation)

    return {
        "sensor_id": str(sensor_id),
        "station_id": str(sensor_id),  # Synonymous for backward compatibility
        "channel": channel,
        "primary_sensor": channel,     # Synonymous for dashboard
        "timestamp": ts_str,
        "is_anomaly": bool(is_anomaly),
        "anomaly_type": anomaly_type,
        "fault_mode": anomaly_type,    # Synonymous for dashboard
        "severity": round(float(severity), 4),
        "severity_level": severity_level.lower(),
        "observed_value": observed_value,
        "expected_value": expected_value,
        "expected_range": list(expected_range) if expected_range else None,
        "deviation": dev_str,
        "details": details,
        "description": details,
    }


# -------------------------------------------------------------------
# 1. MISSING / INVALID DATA DETECTION
# -------------------------------------------------------------------
def check_missing_or_invalid(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, float, str, Any, Any, Any]:
    """
    Flag nulls, NaNs, unparseable values, or reporting interval gaps.
    Returns: (is_anomaly, reason, severity_score, severity_level, observed, expected, deviation)
    """
    if not series or len(series) == 0:
        return True, "Missing payload: channel series is empty", 1.0, "critical", None, "Numeric reading", "Empty"

    latest = series[-1]

    # Null / None check
    if latest is None:
        return True, "Invalid reading: null value received", 1.0, "critical", None, "Valid float", "null"

    # NaN check
    if isinstance(latest, float) and np.isnan(latest):
        return True, "Invalid reading: NaN value received", 1.0, "critical", "NaN", "Valid float", "NaN"

    # Numeric castability
    try:
        val = float(latest)
    except (ValueError, TypeError):
        return (
            True,
            f"Malformed payload: '{latest}' cannot be cast to float",
            1.0,
            "critical",
            str(latest),
            "Valid numeric",
            "Type error",
        )

    # Telemetry reporting interval gap check
    if timestamps is not None and len(timestamps) >= 2:
        cfg = config or {}
        max_interval = float(cfg.get("max_interval_minutes", 15.0))
        t1 = _parse_ts(timestamps[-1])
        t0 = _parse_ts(timestamps[-2])
        if t1 and t0:
            diff_min = (t1 - t0).total_seconds() / 60.0
            if diff_min > max_interval:
                sev_score = min(1.0, diff_min / (3.0 * max_interval))
                sev_level = "critical" if diff_min >= 3.0 * max_interval else "med"
                return (
                    True,
                    f"Telemetry reporting gap: {diff_min:.1f}m interval exceeds {max_interval:.1f}m threshold",
                    round(sev_score, 2),
                    sev_level,
                    f"{diff_min:.1f}m gap",
                    f"<= {max_interval:.1f}m",
                    f"+{diff_min - max_interval:.1f}m",
                )

    return False, "Normal", 0.0, "normal", val, val, "0.00"


# -------------------------------------------------------------------
# 2. PHYSICALLY IMPLAUSIBLE VALUES
# -------------------------------------------------------------------
def check_physical_plausibility(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, float, str, Any, Any, Any]:
    """
    Flag hard domain bound violations.
    Returns: (is_anomaly, reason, severity_score, severity_level, observed, expected, deviation)
    """
    if not series:
        return False, "Normal", 0.0, "normal", None, None, 0.0

    try:
        val = float(series[-1])
    except (ValueError, TypeError):
        return False, "Normal", 0.0, "normal", None, None, 0.0

    if np.isnan(val):
        return False, "Normal", 0.0, "normal", val, None, 0.0

    cfg = config or {}
    min_val = cfg.get("min_val")
    max_val = cfg.get("max_val")

    if min_val is not None and val < min_val:
        diff = val - min_val
        return (
            True,
            f"Physical bound violation: {val} is below minimum plausible limit ({min_val})",
            0.95,
            "critical",
            val,
            f">={min_val}",
            diff,
        )

    if max_val is not None and val > max_val:
        diff = val - max_val
        return (
            True,
            f"Physical bound violation: {val} exceeds maximum plausible limit ({max_val})",
            0.95,
            "critical",
            val,
            f"<={max_val}",
            diff,
        )

    return False, "Normal", 0.0, "normal", val, f"[{min_val}, {max_val}]", 0.0


# -------------------------------------------------------------------
# 3. SPIKE DETECTION (Rolling Median + MAD)
# -------------------------------------------------------------------
def check_spike(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, float, str, Any, Any, Any]:
    """
    Flag sudden departure from recent statistical range using robust median + MAD.
    Returns: (is_anomaly, reason, severity_score, severity_level, observed, expected, deviation)
    """
    cfg = config or {}
    window = int(cfg.get("spike_window", 7))
    threshold_delta = float(cfg.get("spike_delta", 5.0))
    z_thresh = float(cfg.get("spike_z", 3.5))

    if len(series) < 3:
        curr = float(series[-1]) if series else None
        return False, "Normal", 0.0, "normal", curr, curr, 0.0

    recent = series[-window:]
    try:
        clean = [float(x) for x in recent if x is not None and not np.isnan(float(x))]
    except (ValueError, TypeError):
        return False, "Normal", 0.0, "normal", None, None, 0.0

    if len(clean) < 3:
        curr = clean[-1] if clean else None
        return False, "Normal", 0.0, "normal", curr, curr, 0.0

    curr_val = clean[-1]
    prev_vals = np.array(clean[:-1], dtype=float)

    med = float(np.median(prev_vals))
    delta = abs(curr_val - med)

    # Median Absolute Deviation (MAD)
    abs_deviations = np.abs(prev_vals - med)
    mad = float(np.median(abs_deviations))

    if mad > 1e-6:
        robust_z = delta / (1.4826 * mad)
        is_spike = (delta >= threshold_delta) and (robust_z >= z_thresh)
    else:
        # Zero variance history: any jump >= threshold_delta is a sharp spike
        robust_z = delta / (threshold_delta if threshold_delta > 0 else 1.0)
        is_spike = delta >= threshold_delta

    raw_dev = curr_val - med

    if is_spike:
        sev_score = min(1.0, max(0.60, float(robust_z / 7.0)))
        sev_level = "critical" if delta >= 2.0 * threshold_delta or robust_z >= 5.0 else "high"
        expected_span = f"[{med - threshold_delta:.1f}, {med + threshold_delta:.1f}]"
        return (
            True,
            f"Sharp spike detected: {curr_val:.2f} deviates by {raw_dev:+.2f} from recent median ({med:.2f})",
            round(sev_score, 2),
            sev_level,
            round(curr_val, 2),
            round(med, 2),
            raw_dev,
        )

    return False, "Normal", 0.0, "normal", round(curr_val, 2), round(med, 2), raw_dev


# -------------------------------------------------------------------
# 4. STUCK-AT-VALUE DETECTION
# -------------------------------------------------------------------
def check_stuck_value(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, float, str, Any, Any, Any]:
    """
    Flag sensor reporting identical reading for longer than reasonable natural variability.
    Returns: (is_anomaly, reason, severity_score, severity_level, observed, expected, deviation)
    """
    cfg = config or {}
    max_steps = int(cfg.get("max_stuck_steps", 6))
    tol = float(cfg.get("stuck_tolerance", 1e-4))
    allow_zero = bool(cfg.get("allow_zero_stuck", False))

    if len(series) < max_steps:
        curr = float(series[-1]) if series else None
        return False, "Normal", 0.0, "normal", curr, curr, 0.0

    try:
        window_vals = [float(x) for x in series[-max_steps:] if x is not None]
    except (ValueError, TypeError):
        return False, "Normal", 0.0, "normal", None, None, 0.0

    if len(window_vals) < max_steps:
        return False, "Normal", 0.0, "normal", None, None, 0.0

    arr = np.array(window_vals, dtype=float)
    if np.any(np.isnan(arr)):
        return False, "Normal", 0.0, "normal", None, None, 0.0

    curr_val = arr[-1]

    # Calm wind or dry rainfall natural zero exemption
    if allow_zero and np.all(np.abs(arr) <= tol):
        return False, "Normal", 0.0, "normal", round(curr_val, 2), "Natural zero (calm/dry)", 0.0

    val_spread = float(np.max(arr) - np.min(arr))
    if val_spread <= tol:
        stuck_ratio = len(series) / (2.0 * max_steps)
        sev_score = min(1.0, max(0.65, stuck_ratio))
        sev_level = "critical" if len(series) >= 2 * max_steps else "high"
        return (
            True,
            f"Stuck-at-value freeze: sensor invariant at {curr_val:.2f} across {max_steps} consecutive readings",
            round(sev_score, 2),
            sev_level,
            round(curr_val, 2),
            f"Expected variability (> {tol})",
            f"Spread {val_spread:.5f}",
        )

    return False, "Normal", 0.0, "normal", round(curr_val, 2), "Normal variability", round(val_spread, 3)


# -------------------------------------------------------------------
# 5. DRIFT DETECTION
# -------------------------------------------------------------------
def check_drift(
    series: Sequence[Any],
    timestamps: Optional[Sequence[Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, str, float, str, Any, Any, Any]:
    """
    Flag gradual, sustained calibration drift from baseline trend over time.
    Distinguishes drift from natural diurnal swings by requiring both departure and monotonic slope.
    Returns: (is_anomaly, reason, severity_score, severity_level, observed, expected, deviation)
    """
    cfg = config or {}
    long_win = int(cfg.get("drift_long_window", 24))
    short_win = int(cfg.get("drift_short_window", 6))
    drift_delta = float(cfg.get("drift_delta", 4.0))

    if len(series) < long_win:
        curr = float(series[-1]) if series else None
        return False, "Normal", 0.0, "normal", curr, curr, 0.0

    try:
        window_vals = [float(x) for x in series[-long_win:] if x is not None]
    except (ValueError, TypeError):
        return False, "Normal", 0.0, "normal", None, None, 0.0

    if len(window_vals) < long_win:
        return False, "Normal", 0.0, "normal", None, None, 0.0

    arr = np.array(window_vals, dtype=float)
    if np.any(np.isnan(arr)):
        return False, "Normal", 0.0, "normal", None, None, 0.0

    curr_val = arr[-1]
    baseline = float(np.mean(arr[:-short_win]))
    recent_mean = float(np.mean(arr[-short_win:]))
    divergence = recent_mean - baseline

    # Linear slope trend across window
    x = np.arange(len(arr))
    slope, _ = np.polyfit(x, arr, 1)
    total_drift = abs(slope * len(arr))

    if abs(divergence) >= drift_delta and total_drift >= drift_delta:
        direction = "upward" if divergence > 0 else "downward"
        sev_score = min(1.0, max(0.55, abs(divergence) / (2.0 * drift_delta)))
        sev_level = "critical" if abs(divergence) >= 1.75 * drift_delta else "med"
        return (
            True,
            f"Sustained {direction} drift: departure of {abs(divergence):.2f} from baseline over {long_win} readings (slope: {slope:+.3f}/step)",
            round(sev_score, 2),
            sev_level,
            round(curr_val, 2),
            round(baseline, 2),
            divergence,
        )

    return False, "Normal", 0.0, "normal", round(curr_val, 2), round(baseline, 2), divergence


# -------------------------------------------------------------------
# 6. CROSS-SENSOR INCONSISTENCY (Multivariate Unsupervised PCA)
# -------------------------------------------------------------------
class RollingMultivariateModel:
    """
    Streaming-friendly PCA model over joint channel vectors.
    Periodically retrains on rolling history to adapt to diurnal and seasonal shifts.
    """

    def __init__(self, channels: Optional[List[str]] = None, n_components: int = 3):
        self.channels = channels or [
            "temperature_c",
            "humidity_pct",
            "pressure_hpa",
            "wind_speed_ms",
            "wind_direction_deg",
            "precipitation_mm",
        ]
        self.n_components = n_components
        self.pca: Optional[PCA] = None
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.error_threshold_: float = 2.5
        self.fitted_: bool = False

    def fit(self, X: np.ndarray) -> "RollingMultivariateModel":
        """Fit standardizer and PCA on rolling historical observations."""
        if X.shape[0] < self.n_components + 2:
            return self

        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0)
        self.std_[self.std_ < 1e-4] = 1.0

        Z = (X - self.mean_) / self.std_
        Z = np.nan_to_num(Z, nan=0.0)

        n_comp = min(self.n_components, Z.shape[1], Z.shape[0])
        self.pca = PCA(n_components=n_comp)
        self.pca.fit(Z)

        # Compute reconstruction error on fit set to establish dynamic 97.5th percentile threshold
        proj = self.pca.transform(Z)
        recon = self.pca.inverse_transform(proj)
        resids = np.linalg.norm(Z - recon, axis=1)
        self.error_threshold_ = max(2.0, float(np.percentile(resids, 97.5)))
        self.fitted_ = True
        return self

    def evaluate_vector(
        self, vector: Dict[str, float]
    ) -> Tuple[bool, str, float, str, str, Any, Any, Any]:
        """
        Calculates multivariate reconstruction error for an incoming reading vector.
        Returns: (is_anomaly, reason, severity_score, severity_level, culprit_ch, observed, expected, deviation)
        """
        if not self.fitted_ or self.pca is None or self.mean_ is None or self.std_ is None:
            return False, "Model not fitted", 0.0, "normal", "cross_sensor", None, None, 0.0

        # Extract values in channel order
        raw_vals = np.array([float(vector.get(ch, 0.0)) for ch in self.channels], dtype=float)
        if np.any(np.isnan(raw_vals)):
            return False, "Missing features", 0.0, "normal", "cross_sensor", None, None, 0.0

        z = (raw_vals - self.mean_) / self.std_
        z = np.nan_to_num(z, nan=0.0).reshape(1, -1)

        proj = self.pca.transform(z)
        recon_z = self.pca.inverse_transform(proj)[0]

        # Per-channel normalized residual error
        channel_errors = np.abs(z[0] - recon_z)
        total_error = float(np.linalg.norm(z[0] - recon_z))

        # Culprit is channel with largest reconstruction divergence
        culprit_idx = int(np.argmax(channel_errors))
        culprit_ch = self.channels[culprit_idx]
        culprit_obs = raw_vals[culprit_idx]
        culprit_exp = float(self.mean_[culprit_idx] + recon_z[culprit_idx] * self.std_[culprit_idx])
        culprit_dev = culprit_obs - culprit_exp

        is_anom = total_error >= self.error_threshold_
        if is_anom:
            sev_ratio = total_error / self.error_threshold_
            sev_score = min(1.0, max(0.60, 0.5 + 0.25 * sev_ratio))
            sev_level = "critical" if sev_ratio >= 1.8 else "high"
            reason = (
                f"Cross-sensor covariance breakdown: reconstruction residual {total_error:.2f} "
                f"exceeds limit ({self.error_threshold_:.2f}). Primary culprit: {culprit_ch} "
                f"(observed {culprit_obs:.2f} vs expected {culprit_exp:.2f})"
            )
            return (
                True,
                reason,
                round(sev_score, 2),
                sev_level,
                culprit_ch,
                round(culprit_obs, 2),
                round(culprit_exp, 2),
                culprit_dev,
            )

        return (
            False,
            "Normal multivariate correlation",
            0.0,
            "normal",
            culprit_ch,
            round(culprit_obs, 2),
            round(culprit_exp, 2),
            culprit_dev,
        )


# -------------------------------------------------------------------
# Unified Streaming Anomaly Detection Engine
# -------------------------------------------------------------------
class SkyGuardDetector:
    """
    Stateful real-time anomaly detection engine.
    Maintains rolling windows per weather station, evaluates all 6 checks,
    periodically retrains multivariate cross-sensor models, and produces output contracts.
    """

    def __init__(
        self,
        channels: Optional[List[str]] = None,
        retrain_interval: int = 50,
        history_window: int = 120,
    ):
        self.channels = channels or list(CHANNEL_CONFIG.keys())
        self.retrain_interval = retrain_interval
        self.history_window = history_window

        # Rolling history per station: deque of raw observation dicts
        self.station_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.history_window))
        self.multivariate_models: Dict[str, RollingMultivariateModel] = {}
        self.station_step_counts: Dict[str, int] = defaultdict(int)

    def _get_or_init_mv_model(self, station_id: str) -> RollingMultivariateModel:
        if station_id not in self.multivariate_models:
            self.multivariate_models[station_id] = RollingMultivariateModel(channels=self.channels)
        return self.multivariate_models[station_id]

    def _maybe_retrain_multivariate(self, station_id: str) -> None:
        hist = self.station_history[station_id]
        if len(hist) < 15:
            return

        model = self._get_or_init_mv_model(station_id)
        steps = self.station_step_counts[station_id]

        if not model.fitted_ or (steps % self.retrain_interval == 0):
            # Extract matrix of available readings
            matrix = []
            for row in hist:
                row_vals = []
                for ch in self.channels:
                    v = row.get(ch)
                    try:
                        row_vals.append(float(v) if v is not None else np.nan)
                    except (ValueError, TypeError):
                        row_vals.append(np.nan)
                matrix.append(row_vals)

            arr = np.array(matrix, dtype=float)
            # Impute column medians for any missing cells in training matrix
            col_med = np.nanmedian(arr, axis=0)
            inds = np.where(np.isnan(arr))
            arr[inds] = np.take(col_med, inds[1])
            model.fit(arr)

    def score_reading(
        self, reading: Dict[str, Any], station_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a single incoming telemetry record and evaluate:
          - Missing/invalid per channel
          - Physical plausibility per channel
          - Spike per channel
          - Stuck values per channel
          - Drift per channel
          - Cross-sensor multivariate inconsistency
        Returns list of anomaly records conforming to output contract.
        If all channels nominal, returns a single nominal record.
        """
        # Normalize keys using aliases
        normalized: Dict[str, Any] = {}
        for k, v in reading.items():
            norm_k = canonical_channel_name(k)
            normalized[norm_k] = v

        stn = str(station_id or normalized.get("station_id") or normalized.get("sensor_id") or "AWS-LIVE")
        ts = normalized.get("timestamp") or datetime.now(timezone.utc).isoformat()

        # Update station history
        self.station_history[stn].append(dict(normalized))
        self.station_step_counts[stn] += 1
        self._maybe_retrain_multivariate(stn)

        hist = list(self.station_history[stn])
        timestamps = [r.get("timestamp") for r in hist if "timestamp" in r]
        if len(timestamps) < len(hist):
            timestamps = None

        anomalies: List[Dict[str, Any]] = []

        # Evaluate active channels: channels present in this reading or historically reported by this station
        active_channels = [ch for ch in self.channels if ch in normalized or any(ch in r for r in hist)]
        if not active_channels:
            active_channels = self.channels

        # 1-5: Evaluate per-channel checks
        for ch in active_channels:
            cfg = CHANNEL_CONFIG.get(ch, {})
            ch_series = [r.get(ch) for r in hist if ch in r or r.get(ch) is not None]
            if not ch_series and ch in normalized:
                ch_series = [normalized.get(ch)]

            # 1. Missing / invalid check
            is_anom, reason, score, level, obs, exp, dev = check_missing_or_invalid(
                ch_series, timestamps=timestamps, config=cfg
            )
            if is_anom:
                anomalies.append(
                    create_anomaly_record(
                        sensor_id=stn,
                        channel=ch,
                        timestamp=ts,
                        is_anomaly=True,
                        anomaly_type="missing",
                        severity=score,
                        severity_level=level,
                        observed_value=obs,
                        expected_value=exp,
                        deviation=dev,
                        details=reason,
                    )
                )
                continue  # Skip numerical math if channel is missing or unparseable

            # 2. Physical plausibility check
            is_anom, reason, score, level, obs, exp, dev = check_physical_plausibility(
                ch_series, timestamps=timestamps, config=cfg
            )
            if is_anom:
                anomalies.append(
                    create_anomaly_record(
                        sensor_id=stn,
                        channel=ch,
                        timestamp=ts,
                        is_anomaly=True,
                        anomaly_type="implausible",
                        severity=score,
                        severity_level=level,
                        observed_value=obs,
                        expected_value=exp,
                        deviation=dev,
                        details=reason,
                        expected_range=(cfg.get("min_val"), cfg.get("max_val")),
                    )
                )

            # 3. Spike check
            is_anom, reason, score, level, obs, exp, dev = check_spike(
                ch_series, timestamps=timestamps, config=cfg
            )
            if is_anom:
                anomalies.append(
                    create_anomaly_record(
                        sensor_id=stn,
                        channel=ch,
                        timestamp=ts,
                        is_anomaly=True,
                        anomaly_type="spike",
                        severity=score,
                        severity_level=level,
                        observed_value=obs,
                        expected_value=exp,
                        deviation=dev,
                        details=reason,
                    )
                )

            # 4. Stuck-at-value check
            is_anom, reason, score, level, obs, exp, dev = check_stuck_value(
                ch_series, timestamps=timestamps, config=cfg
            )
            if is_anom:
                anomalies.append(
                    create_anomaly_record(
                        sensor_id=stn,
                        channel=ch,
                        timestamp=ts,
                        is_anomaly=True,
                        anomaly_type="stuck",
                        severity=score,
                        severity_level=level,
                        observed_value=obs,
                        expected_value=exp,
                        deviation=dev,
                        details=reason,
                    )
                )

            # 5. Drift check
            is_anom, reason, score, level, obs, exp, dev = check_drift(
                ch_series, timestamps=timestamps, config=cfg
            )
            if is_anom:
                anomalies.append(
                    create_anomaly_record(
                        sensor_id=stn,
                        channel=ch,
                        timestamp=ts,
                        is_anomaly=True,
                        anomaly_type="drift",
                        severity=score,
                        severity_level=level,
                        observed_value=obs,
                        expected_value=exp,
                        deviation=dev,
                        details=reason,
                    )
                )

        # 6. Evaluate Cross-sensor multivariate inconsistency
        mv_model = self._get_or_init_mv_model(stn)
        if mv_model.fitted_:
            is_anom, reason, score, level, culprit_ch, obs, exp, dev = mv_model.evaluate_vector(normalized)
            if is_anom:
                anomalies.append(
                    create_anomaly_record(
                        sensor_id=stn,
                        channel=culprit_ch or "cross_sensor",
                        timestamp=ts,
                        is_anomaly=True,
                        anomaly_type="cross_sensor",
                        severity=score,
                        severity_level=level,
                        observed_value=obs,
                        expected_value=exp,
                        deviation=dev,
                        details=reason,
                    )
                )

        # If no anomalies detected, emit a nominal record so real-time monitors know sensor is healthy
        if not anomalies:
            first_ch = self.channels[0] if self.channels else "telemetry"
            first_val = normalized.get(first_ch)
            return [
                create_anomaly_record(
                    sensor_id=stn,
                    channel=first_ch,
                    timestamp=ts,
                    is_anomaly=False,
                    anomaly_type="none",
                    severity=0.0,
                    severity_level="normal",
                    observed_value=first_val,
                    expected_value=first_val,
                    deviation=0.0,
                    details="All channel sensor readings nominal within physical and statistical bounds",
                )
            ]

        return anomalies

    def score_batch(
        self, readings: Union[List[Dict[str, Any]], pd.DataFrame]
    ) -> List[Dict[str, Any]]:
        """Process a series or dataframe of readings sequentially."""
        records = readings.to_dict(orient="records") if isinstance(readings, pd.DataFrame) else readings
        results: List[Dict[str, Any]] = []
        for r in records:
            res = self.score_reading(r)
            results.extend(res)
        return results
