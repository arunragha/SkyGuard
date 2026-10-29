"""
SkyGuard - Weather Station Sensor Anomaly Detection System
Unsupervised Machine Learning & Physical Domain Guardrails
"""

__version__ = "1.0.0"

from skyguard.detector import (
    CHANNEL_CONFIG,
    SUPPORTED_CHANNELS,
    SkyGuardDetector,
    create_anomaly_record,
)
from skyguard.sensor_checks import (
    check_drift,
    check_missing_or_invalid,
    check_physical_plausibility,
    check_spike,
    check_stuck_value,
    evaluate_channel,
    evaluate_telemetry,
)
