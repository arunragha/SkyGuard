import React, { useEffect, useRef } from "react";
import { Incident } from "../types";
import { SEVERITY_CONFIG, ANOMALY_CONFIG } from "../constants";

interface IncidentPanelProps {
  incidents: Incident[];
  selectedIncident: Incident | null;
  hoveredIncidentId: string | null;
  onSelectIncident: (incident: Incident) => void;
  onHoverIncident: (id: string | null) => void;
}

export const IncidentPanel: React.FC<IncidentPanelProps> = ({
  incidents,
  selectedIncident,
  hoveredIncidentId,
  onSelectIncident,
  onHoverIncident,
}) => {
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});

  // Auto-scroll selected incident into view
  useEffect(() => {
    if (selectedIncident && cardRefs.current[selectedIncident.id]) {
      cardRefs.current[selectedIncident.id]?.scrollIntoView({
        behavior: "smooth",
        block: "nearest",
      });
    }
  }, [selectedIncident]);

  return (
    <aside className="left-panel" style={{ display: "flex", flexDirection: "column", height: "100%" }}>
      <div className="panel-header">
        <div className="panel-title-wrap">
          <h2 className="panel-title">Detected Incidents</h2>
          <span className="incident-badge" id="incidentCountBadge">
            {incidents.length} ACTIVE
          </span>
        </div>
      </div>

      <div className="anomaly-card-list" id="anomalyCardList" style={{ flex: 1, overflowY: "auto" }}>
        {incidents.length === 0 ? (
          <div className="empty-incident-feed" style={{ padding: "24px", textAlign: "center", color: "#94a3b8" }}>
            <div style={{ fontSize: "28px", marginBottom: "8px" }}>🛡️</div>
            <div style={{ fontWeight: 600, color: "#e2e8f0" }}>No Anomalies Detected</div>
            <div style={{ fontSize: "12px", marginTop: "4px" }}>
              All AWS network channels are operating within nominal thresholds.
            </div>
          </div>
        ) : (
          incidents.map((incident) => {
            const isSelected = selectedIncident?.id === incident.id;
            const isHovered = hoveredIncidentId === incident.id;
            const sevCfg = SEVERITY_CONFIG[incident.severity_level] || SEVERITY_CONFIG.medium;
            const anomCfg = ANOMALY_CONFIG[incident.predicted_type] || { label: incident.predicted_type, glyph: incident.glyph };
            const topContrib = incident.contributing_sensors?.[0] || {
              channel: incident.diagnosed_sensor || "temperature",
              contribution: 0.85,
            };
            const channelName = (topContrib?.channel || incident.diagnosed_sensor || "sensor").replace(/_/g, " ");
            const scoreVal = typeof incident.anomaly_score === "number" && !isNaN(incident.anomaly_score) ? incident.anomaly_score : 0;
            const threshVal = typeof incident.anomaly_threshold === "number" && !isNaN(incident.anomaly_threshold) ? incident.anomaly_threshold : 0.5;

            const timeStr = typeof incident.timestamp === "string" && incident.timestamp.includes("T")
              ? incident.timestamp.split("T")[1].slice(0, 8) + " UTC"
              : incident.timestamp || "LIVE";

            return (
              <div
                key={incident.id}
                ref={(el) => { cardRefs.current[incident.id] = el; }}
                tabIndex={0}
                role="button"
                aria-label={`Incident at ${incident.station_name || incident.station_id}: ${incident.predicted_type || incident.anomaly_type}`}
                className={`anomaly-card ${incident.severity_level || "medium"} ${isSelected ? "selected" : ""} ${isHovered ? "hover-linked" : ""} ${incident.is_new ? "new-incident-slide" : ""}`}
                style={{
                  borderLeft: `4px solid ${sevCfg.color}`,
                  background: isSelected ? "rgba(30, 41, 59, 0.95)" : "rgba(15, 23, 42, 0.75)",
                  cursor: "pointer",
                }}
                onMouseEnter={() => onHoverIncident(incident.id)}
                onMouseLeave={() => onHoverIncident(null)}
                onClick={() => onSelectIncident(incident)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    onSelectIncident(incident);
                  }
                }}
              >
                <div className="card-top-row">
                  <div className="card-glyph-tag" style={{ background: sevCfg.bg_rgba, color: sevCfg.color, border: `1px solid ${sevCfg.border}` }}>
                    <span className="glyph-char">{anomCfg.glyph}</span>
                    <span className="glyph-label">{anomCfg.label}</span>
                  </div>
                  <span className="card-timestamp">{timeStr}</span>
                </div>

                <div className="card-station-name">{incident.station_name || incident.station_id}</div>

                <div className="card-metrics-row">
                  <div className="metric-score-group">
                    <span className="metric-score-label">Score / Thresh</span>
                    <div className="score-meter-wrap">
                      <div
                        className="score-meter-fill"
                        style={{
                          width: `${Math.min(100, Math.max(0, Math.round(scoreVal * 100)))}%`,
                          backgroundColor: sevCfg.color,
                        }}
                      />
                    </div>
                    <span className="metric-score-val" style={{ color: sevCfg.color }}>
                      {scoreVal.toFixed(2)}
                      <span style={{ color: "#64748b", fontSize: "10px" }}> / {threshVal.toFixed(2)}</span>
                    </span>
                  </div>

                  <span
                    className={`card-severity-pill ${incident.severity_level || "medium"}`}
                    style={{
                      background: sevCfg.bg_rgba,
                      color: sevCfg.color,
                      borderColor: sevCfg.border,
                    }}
                  >
                    {sevCfg.label.toUpperCase()}
                  </span>
                </div>

                {topContrib && (
                  <div className="card-attribution-row">
                    <span className="attrib-sensor-icon">🎯</span>
                    <span className="attrib-sensor-name">{channelName}:</span>
                    <span className="attrib-sensor-pct" style={{ color: sevCfg.color }}>
                      {Math.round((topContrib?.contribution ?? 0.85) * 100)}% contrib
                    </span>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>
    </aside>
  );
};
