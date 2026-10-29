import React from "react";
import { Incident } from "../types";
import { SEVERITY_CONFIG, ANOMALY_CONFIG } from "../constants";

interface TimelineViewProps {
  incidents: Incident[];
  selectedIncident: Incident | null;
  hoveredIncidentId: string | null;
  onSelectIncident: (incident: Incident) => void;
  onHoverIncident: (id: string | null) => void;
}

export const TimelineView: React.FC<TimelineViewProps> = ({
  incidents,
  selectedIncident,
  hoveredIncidentId,
  onSelectIncident,
  onHoverIncident,
}) => {
  const sorted = [...incidents].sort(
    (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
  );

  return (
    <div
      className="timeline-viewport"
      style={{
        width: "100%",
        height: "100%",
        padding: "24px",
        overflowY: "auto",
        background: "rgba(10, 15, 29, 0.95)",
      }}
    >
      <div style={{ marginBottom: "20px" }}>
        <h3 style={{ margin: 0, fontSize: "16px", color: "#F8FAFC" }}>Incident Temporal Sequence</h3>
        <p style={{ margin: "4px 0 0", fontSize: "12px", color: "#64748b" }}>
          Chronological sequence of sensor anomalies across the network array.
        </p>
      </div>

      <div style={{ position: "relative", paddingLeft: "24px", borderLeft: "2px solid rgba(255, 255, 255, 0.1)" }}>
        {sorted.map((inc) => {
          const isSelected = selectedIncident?.id === inc.id;
          const isHovered = hoveredIncidentId === inc.id;
          const sevCfg = SEVERITY_CONFIG[inc.severity_level] || SEVERITY_CONFIG.medium;
          const anomCfg = ANOMALY_CONFIG[inc.predicted_type] || { label: inc.predicted_type, glyph: inc.glyph };

          return (
            <div
              key={inc.id}
              onClick={() => onSelectIncident(inc)}
              onMouseEnter={() => onHoverIncident(inc.id)}
              onMouseLeave={() => onHoverIncident(null)}
              style={{
                position: "relative",
                marginBottom: "20px",
                cursor: "pointer",
                padding: "12px 16px",
                borderRadius: "8px",
                background: isSelected ? "rgba(56, 189, 248, 0.15)" : "rgba(30, 41, 59, 0.5)",
                border: isSelected ? "1px solid #38bdf8" : isHovered ? "1px solid rgba(255,255,255,0.2)" : "1px solid rgba(255,255,255,0.06)",
                transition: "all 150ms ease",
              }}
            >
              {/* Timeline Dot on axis */}
              <div
                style={{
                  position: "absolute",
                  left: "-31px",
                  top: "16px",
                  width: "12px",
                  height: "12px",
                  borderRadius: "50%",
                  background: sevCfg.color,
                  boxShadow: `0 0 8px ${sevCfg.color}`,
                  border: "2px solid #0F172A",
                }}
              />

              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "6px" }}>
                <span style={{ fontSize: "11px", color: "#94a3b8", fontFamily: "JetBrains Mono" }}>
                  {typeof inc.timestamp === "string" ? inc.timestamp.replace("T", " ").slice(0, 19) : "LIVE"} UTC
                </span>
                <span
                  style={{
                    padding: "2px 6px",
                    borderRadius: "4px",
                    background: sevCfg.bg_rgba,
                    color: sevCfg.color,
                    border: `1px solid ${sevCfg.border}`,
                    fontSize: "10px",
                    fontWeight: 700,
                  }}
                >
                  [{anomCfg.glyph}] {anomCfg.label.toUpperCase()}
                </span>
              </div>

              <div style={{ fontSize: "14px", fontWeight: 600, color: "#E2E8F0", marginBottom: "4px" }}>
                {inc.station_name || inc.station_id}
              </div>

              <div style={{ display: "flex", justifyContent: "space-between", fontSize: "12px", color: "#94a3b8" }}>
                <span>Diagnosed: <span style={{ color: "#38bdf8" }}>{inc.diagnosed_sensor || "temperature"}</span></span>
                <span style={{ color: sevCfg.color, fontWeight: 700 }}>
                  Score: {(typeof inc.anomaly_score === "number" && isFinite(inc.anomaly_score) ? inc.anomaly_score : 0).toFixed(4)}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
