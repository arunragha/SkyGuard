"""
Configuration loader and schema validation for SkyGuard.
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml


@dataclass
class SensorConfig:
    unit: str
    min: float
    max: float
    max_rate_of_change: float


@dataclass
class SamplingConfig:
    expected_interval_minutes: int = 10
    fill_method: str = "linear"
    max_fill_gap_intervals: int = 6


@dataclass
class FeatureConfig:
    rolling_windows: List[int] = field(default_factory=lambda: [6, 36, 144])
    calculate_rates_of_change: bool = True
    calculate_thermodynamics: bool = True
    cyclical_encodings: bool = True


@dataclass
class IsolationForestConfig:
    enabled: bool = True
    n_estimators: int = 150
    contamination: float = 0.05
    random_state: int = 42


@dataclass
class AutoencoderConfig:
    enabled: bool = True
    latent_dim: int = 8
    epochs: int = 25
    batch_size: int = 64
    learning_rate: float = 0.003
    architecture: str = "conv1d"


@dataclass
class StatisticalConfig:
    enabled: bool = True
    hampel_window: int = 12
    n_sigmas: float = 3.0
    z_score_window: int = 36


@dataclass
class LOFConfig:
    enabled: bool = True
    n_neighbors: int = 20
    contamination: float = 0.05


@dataclass
class EnsembleConfig:
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "isolation_forest": 0.30,
            "autoencoder": 0.30,
            "statistical": 0.25,
            "lof": 0.15,
        }
    )


@dataclass
class ThresholdConfig:
    method: str = "percentile"  # 'percentile' or 'mad'
    default_sensitivity: float = 0.5
    percentile_base_quantile: float = 0.95
    mad_multiplier_base: float = 3.0


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    title: str = "SkyGuard Weather Station Anomaly Detection API"
    version: str = "2.0.0"


@dataclass
class AppConfig:
    sensors: Dict[str, SensorConfig] = field(default_factory=dict)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    isolation_forest: IsolationForestConfig = field(default_factory=IsolationForestConfig)
    autoencoder: AutoencoderConfig = field(default_factory=AutoencoderConfig)
    statistical: StatisticalConfig = field(default_factory=StatisticalConfig)
    lof: LOFConfig = field(default_factory=LOFConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    thresholding: ThresholdConfig = field(default_factory=ThresholdConfig)
    api: APIConfig = field(default_factory=APIConfig)


def get_default_sensors() -> Dict[str, SensorConfig]:
    return {
        "temperature": SensorConfig(unit="°C", min=-40.0, max=60.0, max_rate_of_change=5.0),
        "humidity": SensorConfig(unit="%", min=0.0, max=100.0, max_rate_of_change=25.0),
        "barometric_pressure": SensorConfig(unit="hPa", min=800.0, max=1100.0, max_rate_of_change=10.0),
        "wind_speed": SensorConfig(unit="m/s", min=0.0, max=65.0, max_rate_of_change=20.0),
        "wind_direction": SensorConfig(unit="degrees", min=0.0, max=360.0, max_rate_of_change=180.0),
        "rainfall": SensorConfig(unit="mm", min=0.0, max=150.0, max_rate_of_change=50.0),
        "solar_radiation": SensorConfig(unit="W/m²", min=0.0, max=1400.0, max_rate_of_change=500.0),
    }


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """Loads configuration from YAML file or falls back to defaults."""
    if config_path is None:
        # Check standard paths
        potential_paths = [
            Path(__file__).resolve().parent.parent / "configs" / "config.yaml",
            Path("configs/config.yaml"),
        ]
        for p in potential_paths:
            if p.is_file():
                config_path = str(p)
                break

    if not config_path or not os.path.exists(config_path):
        return AppConfig(sensors=get_default_sensors())

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    raw_sensors = raw.get("sensors", {})
    sensors = {}
    for name, s_cfg in raw_sensors.items():
        sensors[name] = SensorConfig(
            unit=s_cfg.get("unit", ""),
            min=float(s_cfg.get("min", -1e6)),
            max=float(s_cfg.get("max", 1e6)),
            max_rate_of_change=float(s_cfg.get("max_rate_of_change", 1e6)),
        )
    if not sensors:
        sensors = get_default_sensors()

    raw_sampling = raw.get("sampling", {})
    sampling = SamplingConfig(
        expected_interval_minutes=int(raw_sampling.get("expected_interval_minutes", 10)),
        fill_method=raw_sampling.get("fill_method", "linear"),
        max_fill_gap_intervals=int(raw_sampling.get("max_fill_gap_intervals", 6)),
    )

    raw_feat = raw.get("features", {})
    features = FeatureConfig(
        rolling_windows=raw_feat.get("rolling_windows", [6, 36, 144]),
        calculate_rates_of_change=raw_feat.get("calculate_rates_of_change", True),
        calculate_thermodynamics=raw_feat.get("calculate_thermodynamics", True),
        cyclical_encodings=raw_feat.get("cyclical_encodings", True),
    )

    raw_models = raw.get("models", {})
    raw_if = raw_models.get("isolation_forest", {})
    iso_forest = IsolationForestConfig(
        enabled=raw_if.get("enabled", True),
        n_estimators=int(raw_if.get("n_estimators", 150)),
        contamination=float(raw_if.get("contamination", 0.05)),
        random_state=int(raw_if.get("random_state", 42)),
    )

    raw_ae = raw_models.get("autoencoder", {})
    autoencoder = AutoencoderConfig(
        enabled=raw_ae.get("enabled", True),
        latent_dim=int(raw_ae.get("latent_dim", 8)),
        epochs=int(raw_ae.get("epochs", 25)),
        batch_size=int(raw_ae.get("batch_size", 64)),
        learning_rate=float(raw_ae.get("learning_rate", 0.003)),
        architecture=raw_ae.get("architecture", "conv1d"),
    )

    raw_stat = raw_models.get("statistical", {})
    statistical = StatisticalConfig(
        enabled=raw_stat.get("enabled", True),
        hampel_window=int(raw_stat.get("hampel_window", 12)),
        n_sigmas=float(raw_stat.get("n_sigmas", 3.0)),
        z_score_window=int(raw_stat.get("z_score_window", 36)),
    )

    raw_lof = raw_models.get("lof", {})
    lof = LOFConfig(
        enabled=raw_lof.get("enabled", True),
        n_neighbors=int(raw_lof.get("n_neighbors", 20)),
        contamination=float(raw_lof.get("contamination", 0.05)),
    )

    raw_ens = raw_models.get("ensemble", {})
    ensemble = EnsembleConfig(
        weights=raw_ens.get(
            "weights",
            {
                "isolation_forest": 0.30,
                "autoencoder": 0.30,
                "statistical": 0.25,
                "lof": 0.15,
            },
        )
    )

    raw_thresh = raw.get("thresholding", {})
    thresholding = ThresholdConfig(
        method=raw_thresh.get("method", "percentile"),
        default_sensitivity=float(raw_thresh.get("default_sensitivity", 0.5)),
        percentile_base_quantile=float(raw_thresh.get("percentile_base_quantile", 0.95)),
        mad_multiplier_base=float(raw_thresh.get("mad_multiplier_base", 3.0)),
    )

    raw_api = raw.get("api", {})
    api = APIConfig(
        host=raw_api.get("host", "0.0.0.0"),
        port=int(raw_api.get("port", 8000)),
        title=raw_api.get("title", "SkyGuard Weather Station Anomaly Detection API"),
        version=raw_api.get("version", "2.0.0"),
    )

    return AppConfig(
        sensors=sensors,
        sampling=sampling,
        features=features,
        isolation_forest=iso_forest,
        autoencoder=autoencoder,
        statistical=statistical,
        lof=lof,
        ensemble=ensemble,
        thresholding=thresholding,
        api=api,
    )
