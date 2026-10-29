"""
SkyGuard CLI — Standalone Weather Station Sensor Anomaly Detection Tool
Run the unsupervised anomaly detection engine against:
  1. Historical CSV files
  2. Live external weather station feeds (Open-Meteo)
  3. Replayed streams for continuous terminal monitoring

Usage:
  python -m skyguard.cli --csv data/weather_station_telemetry.csv
  python -m skyguard.cli --live --interval 10
  python -m skyguard.cli --csv telemetry.csv --min-severity high --format table
"""

import argparse
from datetime import datetime, timezone
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional
import pandas as pd

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from skyguard.data.live_feed import LIVE_WEATHER_STATIONS, fetch_all_live_stations, fetch_live_station_telemetry
from skyguard.detector import (
    CHANNEL_CONFIG,
    SUPPORTED_CHANNELS,
    SkyGuardDetector,
)

# Load Single Source of Truth from shared/vocabulary.json
VOCAB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shared", "vocabulary.json")
try:
    with open(VOCAB_PATH, "r", encoding="utf-8") as f:
        VOCABULARY = json.load(f)
except Exception:
    VOCABULARY = {}

SHARED_SEVERITIES = VOCABULARY.get("severities", {})
SHARED_ANOMALIES = VOCABULARY.get("anomaly_types", {})
SHARED_CHANNELS = VOCABULARY.get("channels", {})

SEVERITY_ORDER = {
    "info": 0,
    "normal": 0,
    "low": 1,
    "med": 2,
    "medium": 2,
    "warning": 2,
    "high": 3,
    "critical": 4,
}
for s_key, s_val in SHARED_SEVERITIES.items():
    SEVERITY_ORDER[s_key] = s_val.get("rank", 0)


def format_table_row(rec: Dict[str, Any]) -> str:
    """Format an anomaly record into a clean table row using shared vocabulary glyphs."""
    ts = rec.get("timestamp", "")
    if "T" in ts:
        ts = ts.replace("T", " ").replace("Z", "")[:19]

    stn = str(rec.get("sensor_id", rec.get("station_id", "UNKNOWN")))[:16]
    ch = str(rec.get("channel", "unknown"))[:17]
    anom_type = str(rec.get("anomaly_type", rec.get("predicted_type", "none")))
    
    # Retrieve standardized glyph from shared vocabulary
    glyph = SHARED_ANOMALIES.get(anom_type, {}).get("glyph", "•")
    type_display = f"[{glyph}] {anom_type}"[:14]

    sev_lvl = str(rec.get("severity_level", "normal")).lower()
    sev_cfg = SHARED_SEVERITIES.get(sev_lvl, {})
    sev_label = sev_cfg.get("label", sev_lvl.upper())[:8]
    sev_score = f"{rec.get('severity', rec.get('anomaly_score', 0.0)):.2f}"

    obs = str(rec.get("observed_value", "N/A"))[:10]
    exp = str(rec.get("expected_value", "N/A"))[:10]
    dev = str(rec.get("deviation", "0.00"))[:9]
    details = str(rec.get("details", ""))[:45]

    return f"{ts:<19} | {stn:<16} | {ch:<17} | {type_display:<14} | {sev_label:<8} ({sev_score}) | {obs:>10} | {exp:>10} | {dev:>9} | {details}"


def print_table_header():
    hdr = f"{'Timestamp':<19} | {'Station / Sensor':<16} | {'Channel':<17} | {'Type':<12} | {'Severity':<15} | {'Observed':>10} | {'Expected':>10} | {'Deviation':>9} | {'Diagnostic Details'}"
    divider = "-" * len(hdr)
    print(hdr)
    print(divider)


def run_csv_inspection(
    csv_path: str,
    detector: SkyGuardDetector,
    min_severity: str = "all",
    target_channels: Optional[List[str]] = None,
    output_format: str = "table",
    anomalies_only: bool = True,
    limit: Optional[int] = None,
    stream_delay: float = 0.0,
):
    """Run detection engine sequentially through a CSV file."""
    if not os.path.exists(csv_path):
        print(f"Error: CSV file '{csv_path}' not found.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)
    if limit:
        df = df.iloc[:limit]

    records = df.to_dict(orient="records")
    min_rank = SEVERITY_ORDER.get(min_severity.lower(), 0)

    if output_format == "table":
        print(f"\n[SKYGUARD] SENSOR ANOMALY DETECTOR - CSV BATCH INSPECTOR")
        print(f"File: {csv_path} | Observations: {len(records)} | Min Severity: {min_severity.upper()}\n")
        print_table_header()

    total_readings = 0
    anomaly_counts: Dict[str, int] = {
        "spike": 0,
        "stuck": 0,
        "drift": 0,
        "missing": 0,
        "implausible": 0,
        "cross_sensor": 0,
    }
    severity_counts: Dict[str, int] = {"normal": 0, "low": 0, "med": 0, "high": 0, "critical": 0}

    for idx, row in enumerate(records):
        total_readings += 1
        results = detector.score_reading(row)

        for rec in results:
            anom_type = rec.get("anomaly_type", "none")
            sev_lvl = rec.get("severity_level", "normal").lower()
            rank = SEVERITY_ORDER.get(sev_lvl, 0)

            severity_counts[sev_lvl] = severity_counts.get(sev_lvl, 0) + 1
            if rec.get("is_anomaly"):
                anomaly_counts[anom_type] = anomaly_counts.get(anom_type, 0) + 1

            # Filtering
            if anomalies_only and not rec.get("is_anomaly"):
                continue
            if min_severity != "all" and rank < min_rank:
                continue
            if target_channels and rec.get("channel") not in target_channels:
                continue

            if output_format == "json":
                print(json.dumps(rec))
            else:
                print(format_table_row(rec))

        if stream_delay > 0:
            time.sleep(stream_delay)

    if output_format == "table":
        total_anoms = sum(anomaly_counts.values())
        print("\n" + "=" * 80)
        print(f"[SUMMARY] Processed {total_readings} records | Flagged Anomalies: {total_anoms}")
        print("-" * 80)
        print("Anomaly Archetypes Detected:")
        for atype, cnt in anomaly_counts.items():
            print(f"  * {atype:<15}: {cnt:>5}")
        print("\nSeverity Distribution:")
        for slvl, cnt in severity_counts.items():
            if slvl != "normal" or not anomalies_only:
                print(f"  * {slvl.upper():<15}: {cnt:>5}")
        print("=" * 80 + "\n")


def run_live_feed(
    detector: SkyGuardDetector,
    interval: float = 10.0,
    stations: Optional[List[str]] = None,
    min_severity: str = "all",
    target_channels: Optional[List[str]] = None,
    output_format: str = "table",
    limit: Optional[int] = None,
):
    """Continuously poll real live weather station external API and score in real time."""
    target_stations = stations or list(LIVE_WEATHER_STATIONS.keys())
    min_rank = SEVERITY_ORDER.get(min_severity.lower(), 0)

    if output_format == "table":
        print(f"\n[SKYGUARD] LIVE WEATHER SENSOR STREAM MONITOR")
        print(f"Source: Open-Meteo Real Live Meteorological API (Genuinely Live Observations)")
        print(f"Monitoring Stations: {', '.join(target_stations)}")
        print(f"Poll Interval: {interval:.1f}s | Press Ctrl+C to stop\n")
        print_table_header()

    iterations = 0
    try:
        while True:
            iterations += 1
            for stn_id in target_stations:
                obs = fetch_live_station_telemetry(stn_id)
                if not obs:
                    continue

                results = detector.score_reading(obs)
                for rec in results:
                    sev_lvl = rec.get("severity_level", "normal").lower()
                    rank = SEVERITY_ORDER.get(sev_lvl, 0)

                    if min_severity != "all" and rank < min_rank:
                        continue
                    if target_channels and rec.get("channel") not in target_channels:
                        continue

                    if output_format == "json":
                        print(json.dumps(rec))
                    else:
                        print(format_table_row(rec))

            if limit and iterations >= limit:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        if output_format == "table":
            print("\n[SkyGuard Live Monitor Stopped by User]")


def main():
    parser = argparse.ArgumentParser(
        description="SkyGuard — Unsupervised Sensor Anomaly Detection CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--csv", type=str, help="Path to telemetry CSV file for inspection/replay")
    group.add_argument("--live", action="store_true", help="Monitor genuine live external weather API feeds")

    parser.add_argument(
        "--min-severity",
        type=str,
        default="all",
        choices=["all", "low", "med", "warning", "high", "critical"],
        help="Minimum severity threshold to report",
    )
    parser.add_argument(
        "--channel",
        type=str,
        default=None,
        help="Comma-separated channels to evaluate (e.g. temperature_c,humidity_pct)",
    )
    parser.add_argument(
        "--stations",
        type=str,
        default=None,
        help="Comma-separated station IDs for live monitoring",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "json"],
        help="Output display format",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=5.0,
        help="Polling / replay delay in seconds",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of observations to process",
    )
    parser.add_argument(
        "--show-normal",
        action="store_true",
        help="Include nominal healthy readings in output",
    )

    args = parser.parse_args()

    channels = (
        [c.strip() for c in args.channel.split(",")]
        if args.channel
        else None
    )
    stations = (
        [s.strip() for s in args.stations.split(",")]
        if args.stations
        else None
    )

    detector = SkyGuardDetector(channels=channels or SUPPORTED_CHANNELS)

    if args.csv:
        run_csv_inspection(
            csv_path=args.csv,
            detector=detector,
            min_severity=args.min_severity,
            target_channels=channels,
            output_format=args.format,
            anomalies_only=not args.show_normal,
            limit=args.limit,
            stream_delay=0.0,
        )
    elif args.live:
        run_live_feed(
            detector=detector,
            interval=args.interval,
            stations=stations,
            min_severity=args.min_severity,
            target_channels=channels,
            output_format=args.format,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
