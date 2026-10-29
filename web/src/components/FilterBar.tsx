import React from "react";
import { FilterState, SeverityLevel, StationHealth } from "../types";
import { SEVERITY_CONFIG, ANOMALY_CONFIG, CHANNEL_CONFIG, STATION_HEALTH_CONFIG } from "../constants";

interface FilterBarProps {
  filters: FilterState;
  onFilterChange: (newFilters: Partial<FilterState>) => void;
  onClearAll: () => void;
}

export const FilterBar: React.FC<FilterBarProps> = ({
  filters,
  onFilterChange,
  onClearAll,
}) => {
  const activeChips: { key: keyof FilterState; label: string; value: string }[] = [];

  if (filters.severity !== "ALL") {
    activeChips.push({
      key: "severity",
      label: `Severity: ${SEVERITY_CONFIG[filters.severity]?.label || filters.severity}`,
      value: filters.severity,
    });
  }
  if (filters.channel !== "ALL") {
    activeChips.push({
      key: "channel",
      label: `Channel: ${CHANNEL_CONFIG[filters.channel]?.name || filters.channel}`,
      value: filters.channel,
    });
  }
  if (filters.anomaly_type !== "ALL") {
    activeChips.push({
      key: "anomaly_type",
      label: `Type: ${ANOMALY_CONFIG[filters.anomaly_type]?.label || filters.anomaly_type}`,
      value: filters.anomaly_type,
    });
  }
  if (filters.station_health !== "ALL") {
    activeChips.push({
      key: "station_health",
      label: `Health: ${STATION_HEALTH_CONFIG[filters.station_health]?.label || filters.station_health}`,
      value: filters.station_health,
    });
  }

  return (
    <div className="filter-bar-container" style={{ padding: "8px 16px", background: "rgba(15, 23, 42, 0.85)", borderBottom: "1px solid rgba(255, 255, 255, 0.08)" }}>
      {/* Selection Dropdowns & Filter Chips */}
      <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: "8px" }}>
        {/* Severity Selector */}
        <div style={{ display: "flex", alignItems: "center", gap: "4px" }}>
          <span style={{ fontSize: "11px", color: "#64748b", fontWeight: 600 }}>SEV:</span>
          {(["ALL", "critical", "high", "medium", "low"] as const).map((sev) => {
            const isSelected = filters.severity === sev;
            const color = sev === "ALL" ? "#94a3b8" : SEVERITY_CONFIG[sev]?.color;

            return (
              <button
                key={sev}
                onClick={() => onFilterChange({ severity: sev })}
                style={{
                  padding: "3px 8px",
                  borderRadius: "4px",
                  fontSize: "11px",
                  fontWeight: isSelected ? 700 : 500,
                  cursor: "pointer",
                  border: isSelected ? `1px solid ${color}` : "1px solid rgba(255, 255, 255, 0.08)",
                  background: isSelected ? (sev === "ALL" ? "rgba(255,255,255,0.15)" : `${color}25`) : "rgba(30, 41, 59, 0.5)",
                  color: isSelected ? (sev === "ALL" ? "#F8FAFC" : color) : "#94a3b8",
                  transition: "all 120ms ease",
                }}
              >
                {sev === "ALL" ? "All" : SEVERITY_CONFIG[sev].label}
              </button>
            );
          })}
        </div>

        <div style={{ width: "1px", height: "18px", background: "rgba(255, 255, 255, 0.1)" }} />

        {/* Channel Dropdown */}
        <select
          value={filters.channel}
          onChange={(e) => onFilterChange({ channel: e.target.value })}
          className="micro-select"
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            color: "#cbd5e1",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            padding: "3px 8px",
            borderRadius: "4px",
            fontSize: "11px",
          }}
        >
          <option value="ALL">All Sensor Channels</option>
          {Object.entries(CHANNEL_CONFIG).map(([id, ch]) => (
            <option key={id} value={id}>
              {ch.name} ({ch.unit})
            </option>
          ))}
        </select>

        {/* Anomaly Type Dropdown */}
        <select
          value={filters.anomaly_type}
          onChange={(e) => onFilterChange({ anomaly_type: e.target.value })}
          className="micro-select"
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            color: "#cbd5e1",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            padding: "3px 8px",
            borderRadius: "4px",
            fontSize: "11px",
          }}
        >
          <option value="ALL">All Anomaly Types</option>
          {Object.entries(ANOMALY_CONFIG).map(([id, anom]) => (
            <option key={id} value={id}>
              [{anom.glyph}] {anom.label}
            </option>
          ))}
        </select>

        {/* Station Health Dropdown */}
        <select
          value={filters.station_health}
          onChange={(e) => onFilterChange({ station_health: e.target.value as StationHealth | "ALL" })}
          className="micro-select"
          style={{
            background: "rgba(30, 41, 59, 0.8)",
            color: "#cbd5e1",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            padding: "3px 8px",
            borderRadius: "4px",
            fontSize: "11px",
          }}
        >
          <option value="ALL">All Station Health</option>
          <option value="ok">Operational</option>
          <option value="warning">Warning</option>
          <option value="critical">Critical</option>
          <option value="offline">Offline</option>
        </select>
      </div>

      {/* Active Filter Chips Bar with Clear All */}
      {activeChips.length > 0 && (
        <div style={{ display: "flex", alignItems: "center", gap: "6px", marginTop: "6px", flexWrap: "wrap" }}>
          <span style={{ fontSize: "10px", color: "#64748b", textTransform: "uppercase", letterSpacing: "0.5px" }}>
            Active Filters:
          </span>
          {activeChips.map((chip) => (
            <span
              key={chip.key}
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                padding: "2px 6px",
                background: "rgba(56, 189, 248, 0.15)",
                border: "1px solid rgba(56, 189, 248, 0.3)",
                color: "#38bdf8",
                borderRadius: "4px",
                fontSize: "11px",
              }}
            >
              <span>{chip.label}</span>
              <button
                onClick={() => onFilterChange({ [chip.key]: "ALL" })}
                style={{
                  background: "none",
                  border: "none",
                  color: "#38bdf8",
                  cursor: "pointer",
                  padding: "0 2px",
                  fontWeight: "bold",
                }}
                title="Remove filter"
              >
                ×
              </button>
            </span>
          ))}

          <button
            onClick={onClearAll}
            style={{
              background: "none",
              border: "none",
              color: "#EF4444",
              fontSize: "11px",
              cursor: "pointer",
              marginLeft: "4px",
              textDecoration: "underline",
            }}
          >
            Clear All
          </button>
        </div>
      )}
    </div>
  );
};
