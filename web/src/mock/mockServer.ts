import { Incident, StationMeta, SeriesPoint, StationSeriesResponse } from "../types";
import { SEVERITY_CONFIG, ANOMALY_CONFIG } from "../constants";

export const MOCK_STATIONS: StationMeta[] = [
  {
    station_id: "AWS-WA-COAST",
    name: "Pacific Northwest Coastal Array (Washington)",
    latitude: 46.97,
    longitude: -123.85,
    elevation_m: 12.0,
    health: "critical",
    records_count: 144,
    total_anomalies_detected: 9,
    last_timestamp: new Date().toISOString(),
    latest_readings: {
      temperature: 14.2,
      humidity: 89.4,
      barometric_pressure: 1008.4,
      wind_speed: 12.8,
      wind_direction: 240.0,
      rainfall: 0.0,
      solar_radiation: 420.0,
    },
  },
  {
    station_id: "AWS-MA-NANTUCKET",
    name: "Northeast Atlantic Shore & Buoy (Massachusetts)",
    latitude: 41.28,
    longitude: -70.09,
    elevation_m: 8.0,
    health: "ok",
    records_count: 144,
    total_anomalies_detected: 1,
    last_timestamp: new Date().toISOString(),
    latest_readings: {
      temperature: 19.5,
      humidity: 78.1,
      barometric_pressure: 1014.2,
      wind_speed: 6.4,
      wind_direction: 195.0,
      rainfall: 0.0,
      solar_radiation: 610.0,
    },
  },
  {
    station_id: "AWS-TX-VALLEY",
    name: "Southern Plains & Texas Basin (Texas)",
    latitude: 29.42,
    longitude: -98.49,
    elevation_m: 198.0,
    health: "warning",
    records_count: 144,
    total_anomalies_detected: 4,
    last_timestamp: new Date().toISOString(),
    latest_readings: {
      temperature: 32.8,
      humidity: 45.2,
      barometric_pressure: 1011.0,
      wind_speed: 3.1,
      wind_direction: 150.0,
      rainfall: 0.0,
      solar_radiation: 880.0,
    },
  },
  {
    station_id: "AWS-MN-DULUTH",
    name: "Upper Midwest & Great Lakes (Minnesota)",
    latitude: 46.78,
    longitude: -92.10,
    elevation_m: 342.0,
    health: "warning",
    records_count: 144,
    total_anomalies_detected: 3,
    last_timestamp: new Date().toISOString(),
    latest_readings: {
      temperature: 12.1,
      humidity: 62.0,
      barometric_pressure: 1018.5,
      wind_speed: 16.4,
      wind_direction: 310.0,
      rainfall: 0.0,
      solar_radiation: 530.0,
    },
  },
  {
    station_id: "AWS-ES-VALENCIA",
    name: "Western Mediterranean Basin (Spain)",
    latitude: 39.46,
    longitude: -0.37,
    elevation_m: 15.0,
    health: "ok",
    records_count: 144,
    total_anomalies_detected: 0,
    last_timestamp: new Date().toISOString(),
    latest_readings: {
      temperature: 26.4,
      humidity: 58.0,
      barometric_pressure: 1016.8,
      wind_speed: 4.8,
      wind_direction: 90.0,
      rainfall: 0.0,
      solar_radiation: 740.0,
    },
  },
  {
    station_id: "AWS-AU-OUTBACK",
    name: "Central Outback Array (Alice Springs, Australia)",
    latitude: -23.70,
    longitude: 133.87,
    elevation_m: 545.0,
    health: "ok",
    records_count: 144,
    total_anomalies_detected: 1,
    last_timestamp: new Date().toISOString(),
    latest_readings: {
      temperature: 28.9,
      humidity: 22.0,
      barometric_pressure: 1022.0,
      wind_speed: 5.2,
      wind_direction: 120.0,
      rainfall: 0.0,
      solar_radiation: 910.0,
    },
  },
];

export const MOCK_INITIAL_INCIDENTS: Incident[] = [
  {
    id: "inc_001",
    timestamp: new Date(Date.now() - 4 * 60000).toISOString(),
    station_id: "AWS-WA-COAST",
    station_name: "Pacific Northwest Coastal Array (Washington)",
    latitude: 46.97,
    longitude: -123.85,
    anomaly_score: 0.94,
    anomaly_threshold: 0.52,
    predicted_type: "dropout",
    severity_level: "critical",
    glyph: "o",
    diagnosed_sensor: "rainfall",
    contributing_sensors: [
      { channel: "rainfall", contribution: 0.88 },
      { channel: "humidity", contribution: 0.08 },
      { channel: "pressure", contribution: 0.04 },
    ],
    readings: {
      temperature: 14.2,
      humidity: 89.4,
      barometric_pressure: 1008.4,
      wind_speed: 12.8,
      wind_direction: 240.0,
      rainfall: 0.0,
      solar_radiation: 420.0,
    },
    is_new: true,
  },
  {
    id: "inc_002",
    timestamp: new Date(Date.now() - 12 * 60000).toISOString(),
    station_id: "AWS-MN-DULUTH",
    station_name: "Upper Midwest & Great Lakes (Minnesota)",
    latitude: 46.78,
    longitude: -92.10,
    anomaly_score: 0.84,
    anomaly_threshold: 0.50,
    predicted_type: "spike",
    severity_level: "high",
    glyph: "^",
    diagnosed_sensor: "wind_speed",
    contributing_sensors: [
      { channel: "wind_speed", contribution: 0.76 },
      { channel: "barometric_pressure", contribution: 0.16 },
      { channel: "temperature", contribution: 0.08 },
    ],
    readings: {
      temperature: 12.1,
      humidity: 62.0,
      barometric_pressure: 1018.5,
      wind_speed: 28.4,
      wind_direction: 310.0,
      rainfall: 0.0,
      solar_radiation: 530.0,
    },
    is_new: false,
  },
  {
    id: "inc_003",
    timestamp: new Date(Date.now() - 25 * 60000).toISOString(),
    station_id: "AWS-TX-VALLEY",
    station_name: "Southern Plains & Texas Basin (Texas)",
    latitude: 29.42,
    longitude: -98.49,
    anomaly_score: 0.72,
    anomaly_threshold: 0.51,
    predicted_type: "stuck",
    severity_level: "medium",
    glyph: "=",
    diagnosed_sensor: "humidity",
    contributing_sensors: [
      { channel: "humidity", contribution: 0.68 },
      { channel: "temperature", contribution: 0.22 },
      { channel: "solar_radiation", contribution: 0.10 },
    ],
    readings: {
      temperature: 32.8,
      humidity: 45.0,
      barometric_pressure: 1011.0,
      wind_speed: 3.1,
      wind_direction: 150.0,
      rainfall: 0.0,
      solar_radiation: 880.0,
    },
    is_new: false,
  },
  {
    id: "inc_004",
    timestamp: new Date(Date.now() - 35 * 60000).toISOString(),
    station_id: "AWS-ES-VALENCIA",
    station_name: "Western Mediterranean Basin (Spain)",
    latitude: 39.46,
    longitude: -0.37,
    anomaly_score: 0.74,
    anomaly_threshold: 0.50,
    predicted_type: "drift",
    severity_level: "medium",
    glyph: "/",
    diagnosed_sensor: "temperature",
    contributing_sensors: [
      { channel: "temperature", contribution: 0.71 },
      { channel: "humidity", contribution: 0.19 },
    ],
    readings: {
      temperature: 28.2,
      humidity: 52.0,
      barometric_pressure: 1016.8,
      wind_speed: 4.8,
      wind_direction: 90.0,
      rainfall: 0.0,
      solar_radiation: 740.0,
    },
    is_new: false,
  },
  {
    id: "inc_005",
    timestamp: new Date(Date.now() - 55 * 60000).toISOString(),
    station_id: "AWS-AU-OUTBACK",
    station_name: "Central Outback Array (Alice Springs, Australia)",
    latitude: -23.70,
    longitude: 133.87,
    anomaly_score: 0.96,
    anomaly_threshold: 0.52,
    predicted_type: "out_of_range",
    severity_level: "critical",
    glyph: "!",
    diagnosed_sensor: "solar_radiation",
    contributing_sensors: [
      { channel: "solar_radiation", contribution: 0.89 },
      { channel: "temperature", contribution: 0.11 },
    ],
    readings: {
      temperature: 34.5,
      humidity: 18.0,
      barometric_pressure: 1022.0,
      wind_speed: 6.1,
      wind_direction: 110.0,
      rainfall: 0.0,
      solar_radiation: 1480.0,
    },
    is_new: false,
  },
];

export function generateMockSeries(stationId: string, hours: number = 6): StationSeriesResponse {
  const points = Math.max(6, Math.floor(hours * 6));
  const series: SeriesPoint[] = [];
  const now = Date.now();

  const channels = ["temperature", "humidity", "barometric_pressure", "wind_speed", "wind_direction", "rainfall", "solar_radiation"];
  const baseValues: Record<string, number> = {
    temperature: 18.0,
    humidity: 65.0,
    barometric_pressure: 1013.25,
    wind_speed: 5.5,
    wind_direction: 180.0,
    rainfall: 0.0,
    solar_radiation: 600.0,
  };

  for (let i = points - 1; i >= 0; i--) {
    const timeOffsetMs = i * 10 * 60 * 1000;
    const ts = new Date(now - timeOffsetMs).toISOString();
    const isAnomalyPoint = i === 2 || i === 3; // Recent anomaly

    const readings: Record<string, number> = {};
    const expected_bands: Record<string, { median: number; lower: number; upper: number }> = {};

    channels.forEach((c) => {
      const base = baseValues[c];
      const noise = (Math.sin(i / 3) + Math.cos(i / 5)) * (base * 0.08);
      let val = base + noise;

      if (isAnomalyPoint && (c === "rainfall" || c === "temperature")) {
        val = c === "rainfall" ? 0.0 : base + base * 0.45;
      }

      const med = Math.round(base * 10) / 10;
      const mad = Math.max(0.5, Math.round(base * 0.06 * 10) / 10);
      readings[c] = Math.max(0, Math.round(val * 10) / 10);
      expected_bands[c] = {
        median: med,
        lower: Math.max(0, Math.round((med - 2.5 * mad) * 10) / 10),
        upper: Math.round((med + 2.5 * mad) * 10) / 10,
      };
    });

    const score = isAnomalyPoint ? 0.92 : Math.max(0.05, Math.round((0.15 + Math.random() * 0.25) * 100) / 100);
    const threshold = 0.50;
    const isAnom = score >= threshold ? 1 : 0;
    const pType = isAnom ? "dropout" : "normal";
    const sev = isAnom ? "critical" : "info";

    series.push({
      timestamp: ts,
      station_id: stationId,
      anomaly_score: score,
      anomaly_threshold: threshold,
      is_anomaly: isAnom,
      anomaly_type: pType,
      severity_level: sev,
      glyph: isAnom ? "o" : "•",
      diagnosed_sensor: "rainfall",
      contributing_sensors: [
        { channel: "rainfall", contribution: 0.85 },
        { channel: "humidity", contribution: 0.15 },
      ],
      readings,
      expected_bands,
    });
  }

  return {
    station_id: stationId,
    name: stationId,
    hours,
    total_points: series.length,
    series,
  };
}
