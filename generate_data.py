"""
Generates the benchmark weather station telemetry dataset.
"""

import os
import sys

# Ensure skyguard package is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skyguard.data.weather_generator import generate_benchmark_dataset

if __name__ == "__main__":
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    out_path = os.path.join(out_dir, "weather_station_telemetry.csv")
    print(f"Generating benchmark dataset to: {out_path}")
    df = generate_benchmark_dataset(out_path)
    print(f"Successfully generated {len(df)} records with {df['is_anomaly'].sum()} injected anomaly tags.")
    print("Fault breakdown:")
    print(df["anomaly_type"].value_counts())
