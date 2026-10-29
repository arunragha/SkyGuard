/**
 * SkyGuard Design Tokens & Vocabulary Constants
 * Grounded in shared/vocabulary.json
 */
import vocabulary from "../../shared/vocabulary.json";

export const VOCABULARY = vocabulary;

export const SEVERITY_CONFIG: Record<
  string,
  {
    id: string;
    label: string;
    rank: number;
    color: string;
    bg_rgba: string;
    border: string;
    description: string;
  }
> = vocabulary.severities;

export const ANOMALY_CONFIG: Record<
  string,
  {
    id: string;
    label: string;
    glyph: string;
    description: string;
  }
> = vocabulary.anomaly_types;

export const CHANNEL_CONFIG: Record<
  string,
  {
    id: string;
    name: string;
    unit: string;
    min_valid: number;
    max_valid: number;
    color: string;
  }
> = vocabulary.channels;

export const STATION_HEALTH_CONFIG: Record<
  string,
  {
    id: string;
    label: string;
    color: string;
    description: string;
  }
> = vocabulary.station_health;

export const DEFAULT_SETTINGS = vocabulary.settings_defaults;
