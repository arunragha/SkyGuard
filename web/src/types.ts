/**
 * SkyGuard TypeScript Types Contract
 * Derived directly from shared/vocabulary.json (Single Source of Truth)
 */

export type SeverityLevel = "info" | "low" | "medium" | "high" | "critical";

export type AnomalyType =
  | "spike"
  | "stuck"
  | "drift"
  | "dropout"
  | "out_of_range"
  | "multivariate";

export type ChannelId =
  | "temperature"
  | "humidity"
  | "barometric_pressure"
  | "wind_speed"
  | "wind_direction"
  | "rainfall"
  | "solar_radiation";

export type StationHealth = "ok" | "warning" | "critical" | "offline";

export interface StationMeta {
  station_id: string;
  name: string;
  latitude: number;
  longitude: number;
  elevation_m: number;
  health: StationHealth;
  status?: string;
  records_count: number;
  total_anomalies_detected: number;
  last_timestamp: string;
  latest_readings: Record<string, number>;
}

export interface ContributingSensor {
  channel: string;
  contribution: number;
}

export interface Incident {
  id: string;
  timestamp: string;
  station_id: string;
  station_name: string;
  latitude: number;
  longitude: number;
  lat?: number;
  lng?: number;
  anomaly_score: number;
  score?: number;
  anomaly_threshold: number;
  predicted_type: AnomalyType;
  anomaly_type?: AnomalyType;
  severity_level: SeverityLevel;
  severity?: SeverityLevel;
  glyph: string;
  diagnosed_sensor: string;
  channel?: string;
  contributing_sensors: ContributingSensor[];
  readings: Record<string, number>;
  is_new?: boolean;
  acknowledged?: boolean;
  snoozed?: boolean;
  false_positive?: boolean;
}

export interface SeriesPoint {
  timestamp: string;
  station_id: string;
  anomaly_score: number;
  anomaly_threshold: number;
  is_anomaly: number;
  anomaly_type: string;
  severity_level: SeverityLevel;
  glyph: string;
  diagnosed_sensor: string;
  contributing_sensors: ContributingSensor[];
  readings: Record<string, number>;
  expected_bands: Record<
    string,
    {
      median: number;
      lower: number;
      upper: number;
    }
  >;
}

export interface StationSeriesResponse {
  station_id: string;
  name: string;
  hours: number;
  total_points: number;
  series: SeriesPoint[];
}

export interface FilterState {
  severity: SeverityLevel | "ALL";
  channel: string | "ALL";
  anomaly_type: string | "ALL";
  station_health: StationHealth | "ALL";
  search: string;
}

export type ReplayMode = "live" | "1h" | "24h" | "7d" | "custom";

export interface SystemSettings {
  sensitivity: number;
  models: {
    isolation_forest: boolean;
    autoencoder: boolean;
    statistical: boolean;
    lof: boolean;
  };
  theme: "dark" | "light";
  reduced_motion: boolean;
  sound_enabled: boolean;
  estimated_incidents_24h: number;
  sensitivity_min?: number;
  sensitivity_max?: number;
}

export interface PerformanceMetrics {
  overall: {
    precision: number;
    recall: number;
    f1_score: number;
    roc_auc: number;
    pr_auc: number;
  };
  archetype_recall: Record<string, number>;
  model_comparison: Record<
    string,
    {
      precision: number;
      recall: number;
      f1: number;
    }
  >;
  sampling_interval: string;
  channels_evaluated: number;
}

export interface ToastMessage {
  id: string;
  title: string;
  message: string;
  type: "info" | "success" | "warning" | "critical";
  durationMs?: number;
  undoAction?: () => void;
  undoLabel?: string;
}

export interface RegionalArc {
  startLat: number;
  startLng: number;
  endLat: number;
  endLng: number;
  color: string;
  anomaly_type: string;
  stations: [string, string];
}
