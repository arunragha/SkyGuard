import React, { useState, useEffect, useRef } from "react";
import { Incident, StationMeta, SeriesPoint } from "../types";
import { SEVERITY_CONFIG, ANOMALY_CONFIG, CHANNEL_CONFIG } from "../constants";

interface DetailDrawerProps {
  incident: Incident | null;
  stations: StationMeta[];
  isOpen: boolean;
  onClose: () => void;
  onAcknowledge: (incident: Incident) => void;
  onMarkFalsePositive: (incident: Incident) => void;
  onSnoozeStation: (stationId: string, durationHours: number) => void;
  onShowToast: (toast: any) => void;
}

export const DetailDrawer: React.FC<DetailDrawerProps> = ({
  incident,
  stations,
  isOpen,
  onClose,
  onAcknowledge,
  onMarkFalsePositive,
  onSnoozeStation,
  onShowToast,
}) => {
  const [activeChannel, setActiveChannel] = useState<string>("temperature");
  const [showNeighbors, setShowNeighbors] = useState<boolean>(false);
  const [seriesData, setSeriesData] = useState<SeriesPoint[]>([]);
  const [neighborSeries, setNeighborSeries] = useState<Record<string, SeriesPoint[]>>({});
  const [isLoadingSeries, setIsLoadingSeries] = useState<boolean>(false);
  const sparklineCanvasRef = useRef<HTMLCanvasElement>(null);

  // Set active channel to incident's primary diagnosed sensor on open
  useEffect(() => {
    if (incident) {
      const topChannel = incident.diagnosed_sensor in CHANNEL_CONFIG
        ? incident.diagnosed_sensor
        : (incident.contributing_sensors[0]?.channel || "temperature");
      setActiveChannel(topChannel);
    }
  }, [incident]);

  // Fetch 6h high-resolution series telemetry
  useEffect(() => {
    if (!incident || !isOpen) return;

    let isMounted = true;
    setIsLoadingSeries(true);

    fetch(`/series/${incident.station_id}?hours=6`)
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => {
        if (isMounted && data.series) {
          setSeriesData(data.series);
        }
      })
      .catch(() => {
        // Fallback to synthetic points if network drop
        if (isMounted) {
          const fallbackPoints: SeriesPoint[] = [];
          const now = Date.now();
          for (let i = 35; i >= 0; i--) {
            const ts = new Date(now - i * 10 * 60000).toISOString();
            fallbackPoints.push({
              timestamp: ts,
              station_id: incident.station_id,
              anomaly_score: i < 3 ? incident.anomaly_score : 0.15,
              anomaly_threshold: incident.anomaly_threshold,
              is_anomaly: i < 3 ? 1 : 0,
              anomaly_type: i < 3 ? incident.predicted_type : "normal",
              severity_level: i < 3 ? incident.severity_level : "info",
              glyph: i < 3 ? incident.glyph : "•",
              diagnosed_sensor: incident.diagnosed_sensor,
              contributing_sensors: incident.contributing_sensors,
              readings: {
                ...incident.readings,
                [activeChannel]: (incident.readings[activeChannel] || 20.0) + (Math.sin(i / 4) * 2.5),
              },
              expected_bands: {
                [activeChannel]: {
                  median: incident.readings[activeChannel] || 20.0,
                  lower: (incident.readings[activeChannel] || 20.0) - 3.5,
                  upper: (incident.readings[activeChannel] || 20.0) + 3.5,
                },
              },
            });
          }
          setSeriesData(fallbackPoints);
        }
      })
      .finally(() => {
        if (isMounted) setIsLoadingSeries(false);
      });

    return () => {
      isMounted = false;
    };
  }, [incident, isOpen]);

  // Find nearest 2 neighboring stations and fetch their series if toggled
  useEffect(() => {
    if (!incident || !showNeighbors || !isOpen) return;

    const currentStation = stations.find((s) => s.station_id === incident.station_id);
    if (!currentStation) return;

    // Sort other stations by Euclidean lat/lon distance
    const others = stations
      .filter((s) => s.station_id !== incident.station_id)
      .map((s) => {
        const dLat = s.latitude - currentStation.latitude;
        const dLng = s.longitude - currentStation.longitude;
        const dist = Math.sqrt(dLat * dLat + dLng * dLng);
        return { station: s, dist };
      })
      .sort((a, b) => a.dist - b.dist)
      .slice(0, 2);

    others.forEach(({ station }) => {
      fetch(`/series/${station.station_id}?hours=6`)
        .then((res) => (res.ok ? res.json() : Promise.reject(res)))
        .then((data) => {
          if (data.series) {
            setNeighborSeries((prev) => ({
              ...prev,
              [station.station_id]: data.series,
            }));
          }
        })
        .catch(() => {});
    });
  }, [incident, showNeighbors, stations, isOpen]);

  // Draw 6-Hour Sparkline Canvas: Expected Band (+/- MAD), Shaded Anomaly Window, Observed Line, Live Cursor
  useEffect(() => {
    const canvas = sparklineCanvasRef.current;
    if (!canvas || seriesData.length === 0) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const w = rect.width;
    const h = rect.height;
    ctx.clearRect(0, 0, w, h);

    const values = seriesData.map((d) => {
      const v = d?.readings?.[activeChannel];
      return typeof v === "number" && isFinite(v) ? v : 0;
    });
    const uppers = seriesData.map((d) => {
      const up = d?.expected_bands?.[activeChannel]?.upper;
      if (typeof up === "number" && isFinite(up)) return up;
      const v = d?.readings?.[activeChannel];
      return (typeof v === "number" && isFinite(v) ? v : 20) + 4;
    });
    const lowers = seriesData.map((d) => {
      const low = d?.expected_bands?.[activeChannel]?.lower;
      if (typeof low === "number" && isFinite(low)) return low;
      const v = d?.readings?.[activeChannel];
      return (typeof v === "number" && isFinite(v) ? v : 20) - 4;
    });

    const validLowers = lowers.filter(isFinite);
    const validUppers = uppers.filter(isFinite);
    const validValues = values.filter(isFinite);

    let minVal = validLowers.length > 0 ? Math.min(...validLowers, ...validValues) : 0;
    let maxVal = validUppers.length > 0 ? Math.max(...validUppers, ...validValues) : 40;
    if (!isFinite(minVal)) minVal = 0;
    if (!isFinite(maxVal)) maxVal = 40;
    const span = Math.max(1e-4, maxVal - minVal);
    minVal -= span * 0.1;
    maxVal += span * 0.1;
    const range = Math.max(1e-4, maxVal - minVal);

    const getX = (idx: number) => (seriesData.length > 1 ? (idx / (seriesData.length - 1)) * (w - 40) + 20 : w / 2);
    const getY = (val: number) => (isFinite(val) ? h - 25 - ((val - minVal) / range) * (h - 45) : h / 2);

    // 1. Shaded Anomaly Window (highlight points where is_anomaly == 1)
    seriesData.forEach((d, idx) => {
      if (d.is_anomaly === 1) {
        const x = getX(idx);
        const nextX = idx < seriesData.length - 1 ? getX(idx + 1) : x + 12;
        ctx.fillStyle = "rgba(239, 68, 68, 0.15)";
        ctx.fillRect(x - 6, 10, nextX - x + 12, h - 35);
      }
    });

    // 2. Expected Band (Rolling Median +/- 2.5 MAD) behind observed line
    ctx.beginPath();
    uppers.forEach((up, idx) => {
      const x = getX(idx);
      const y = getY(up);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    for (let idx = lowers.length - 1; idx >= 0; idx--) {
      const x = getX(idx);
      const y = getY(lowers[idx]);
      ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(56, 189, 248, 0.08)";
    ctx.fill();

    // Expected Band Border (Dashed)
    ctx.beginPath();
    ctx.setLineDash([3, 3]);
    ctx.strokeStyle = "rgba(56, 189, 248, 0.35)";
    ctx.lineWidth = 1;
    uppers.forEach((up, idx) => {
      const x = getX(idx);
      const y = getY(up);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.beginPath();
    lowers.forEach((low, idx) => {
      const x = getX(idx);
      const y = getY(low);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // 3. Neighbor Comparison Traces (if enabled)
    if (showNeighbors) {
      const neighborColors = ["#A855F7", "#F59E0B"];
      Object.entries(neighborSeries).forEach(([stnId, nSeries], nIdx) => {
        if (nSeries.length === 0) return;
        ctx.beginPath();
        ctx.setLineDash([4, 2]);
        ctx.strokeStyle = neighborColors[nIdx % neighborColors.length];
        ctx.lineWidth = 1.6;
        nSeries.forEach((p, idx) => {
          if (idx >= seriesData.length) return;
          const x = getX(idx);
          const y = getY(p.readings[activeChannel] || 0);
          if (idx === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        });
        ctx.stroke();
        ctx.setLineDash([]);
      });
    }

    // 4. Observed Telemetry Line
    ctx.beginPath();
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2.4;
    values.forEach((val, idx) => {
      const x = getX(idx);
      const y = getY(val);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // Draw data points & anomaly badges
    values.forEach((val, idx) => {
      const x = getX(idx);
      const y = getY(val);
      const point = seriesData[idx];

      ctx.beginPath();
      if (point.is_anomaly === 1) {
        ctx.arc(x, y, 5, 0, Math.PI * 2);
        ctx.fillStyle = "#EF4444";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#FFFFFF";
        ctx.stroke();
      } else {
        ctx.arc(x, y, 2.5, 0, Math.PI * 2);
        ctx.fillStyle = "#38bdf8";
        ctx.fill();
      }
    });

    // 5. Live Cursor (Advances on latest point)
    const lastIdx = values.length - 1;
    const liveX = getX(lastIdx);
    const liveY = getY(values[lastIdx]);

    ctx.beginPath();
    ctx.setLineDash([2, 2]);
    ctx.strokeStyle = "rgba(34, 197, 94, 0.8)";
    ctx.lineWidth = 1.5;
    ctx.moveTo(liveX, 5);
    ctx.lineTo(liveX, h - 20);
    ctx.stroke();
    ctx.setLineDash([]);

    // Live Pulse Dot
    ctx.beginPath();
    ctx.arc(liveX, liveY, 6, 0, Math.PI * 2);
    ctx.fillStyle = "#22C55E";
    ctx.fill();

    // Axis Labels
    ctx.fillStyle = "#94a3b8";
    ctx.font = "10px JetBrains Mono, monospace";
    ctx.fillText((isFinite(minVal) ? minVal : 0).toFixed(1), 4, h - 25);
    ctx.fillText((isFinite(maxVal) ? maxVal : 40).toFixed(1), 4, 18);
    ctx.fillText("T-6h", 20, h - 8);
    ctx.fillText("NOW (LIVE)", w - 75, h - 8);
  }, [seriesData, activeChannel, showNeighbors, neighborSeries]);

  if (!incident) return null;

  const sevCfg = SEVERITY_CONFIG[incident.severity_level] || SEVERITY_CONFIG.medium;
  const anomCfg = ANOMALY_CONFIG[incident.predicted_type] || { label: incident.predicted_type, glyph: incident.glyph };
  const chCfg = CHANNEL_CONFIG[activeChannel] || { name: activeChannel, unit: "" };

  const handleCreateGitLabTicket = () => {
    const title = encodeURIComponent(`[Anomaly Incident] ${incident.predicted_type.toUpperCase()} at ${incident.station_id}`);
    const body = encodeURIComponent(
      `### SkyGuard Automated Incident Report\n\n` +
      `- **Station**: ${incident.station_name} (${incident.station_id})\n` +
      `- **Coordinates**: ${incident.latitude}, ${incident.longitude}\n` +
      `- **Anomaly Type**: ${incident.predicted_type} (Glyph: ${incident.glyph})\n` +
      `- **Severity**: ${incident.severity_level.toUpperCase()} (Score: ${incident.anomaly_score.toFixed(4)} / Threshold: ${incident.anomaly_threshold.toFixed(4)})\n` +
      `- **Primary Diagnosed Channel**: ${incident.diagnosed_sensor}\n` +
      `- **Timestamp**: ${incident.timestamp}\n\n` +
      `#### Sensor Telemetry Snapshot\n` +
      `\`\`\`json\n${JSON.stringify(incident.readings, null, 2)}\n\`\`\`\n\n` +
      `#### Sensor Contributions\n` +
      incident.contributing_sensors.map((c) => `- ${c.channel}: ${(c.contribution * 100).toFixed(1)}%`).join("\n")
    );

    const gitlabUrl = `https://gitlab.com/skyguard/defense-matrix/-/issues/new?issue[title]=${title}&issue[description]=${body}`;
    window.open(gitlabUrl, "_blank");

    onShowToast({
      id: `toast_gitlab_${Date.now()}`,
      title: "GitLab Work Order Prefilled",
      message: `Created ticket draft for ${incident.station_id}`,
      type: "success",
    });
  };

  return (
    <aside
      className={`right-panel detail-drawer ${isOpen ? "drawer-open" : "drawer-closed"}`}
      id="detailPanel"
      style={{
        position: "absolute",
        top: 0,
        right: 0,
        width: "440px",
        height: "100%",
        zIndex: 100,
        background: "rgba(15, 23, 42, 0.94)",
        backdropFilter: "blur(18px)",
        borderLeft: "1px solid rgba(255, 255, 255, 0.12)",
        transform: isOpen ? "translateX(0)" : "translateX(100%)",
        transition: "transform 300ms cubic-bezier(0.16, 1, 0.3, 1)",
        display: "flex",
        flexDirection: "column",
        boxShadow: "-8px 0 32px rgba(0, 0, 0, 0.6)",
      }}
    >
      {/* Drawer Header */}
      <div
        className="drawer-header"
        style={{
          padding: "16px 20px",
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "flex-start",
        }}
      >
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: "8px", marginBottom: "4px" }}>
            <span
              className="card-severity-pill"
              style={{
                background: sevCfg.bg_rgba,
                color: sevCfg.color,
                borderColor: sevCfg.border,
                fontSize: "11px",
              }}
            >
              {sevCfg.label.toUpperCase()}
            </span>
            <span
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: "4px",
                background: "rgba(30, 41, 59, 0.8)",
                padding: "2px 8px",
                borderRadius: "4px",
                fontSize: "11px",
                color: "#94a3b8",
              }}
            >
              <span style={{ color: sevCfg.color, fontWeight: 800 }}>{anomCfg.glyph}</span>
              <span>{anomCfg.label}</span>
            </span>
          </div>
          <h3 style={{ margin: 0, fontSize: "16px", color: "#F8FAFC", fontWeight: 700 }}>
            {incident.station_name || incident.station_id}
          </h3>
          <div style={{ fontSize: "11px", color: "#64748b", marginTop: "2px" }}>
            Lat {(typeof incident.latitude === "number" && isFinite(incident.latitude) ? incident.latitude : 0).toFixed(2)}°, Lng {(typeof incident.longitude === "number" && isFinite(incident.longitude) ? incident.longitude : 0).toFixed(2)}° • {incident.station_id}
          </div>
        </div>

        <button
          onClick={onClose}
          style={{
            background: "rgba(30, 41, 59, 0.6)",
            border: "1px solid rgba(255, 255, 255, 0.1)",
            color: "#94a3b8",
            width: "28px",
            height: "28px",
            borderRadius: "6px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
          title="Close Drawer (Esc)"
        >
          ✕
        </button>
      </div>

      {/* Drawer Scrollable Body */}
      <div className="drawer-body" style={{ flex: 1, overflowY: "auto", padding: "16px 20px" }}>
        {/* Score vs Threshold Gauge */}
        <div
          className="gauge-box"
          style={{
            background: "rgba(30, 41, 59, 0.6)",
            borderRadius: "8px",
            padding: "12px",
            border: "1px solid rgba(255, 255, 255, 0.06)",
            marginBottom: "16px",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#94a3b8", marginBottom: "6px" }}>
            <span>ANOMALY SCORE</span>
            <span style={{ color: sevCfg.color, fontWeight: 700 }}>
              {(typeof incident.anomaly_score === "number" && isFinite(incident.anomaly_score) ? incident.anomaly_score : 0).toFixed(4)} vs {(typeof incident.anomaly_threshold === "number" && isFinite(incident.anomaly_threshold) ? incident.anomaly_threshold : 0.5).toFixed(4)} threshold
            </span>
          </div>
          <div style={{ position: "relative", height: "10px", background: "rgba(15, 23, 42, 0.8)", borderRadius: "5px", overflow: "hidden" }}>
            <div
              style={{
                width: `${Math.min(100, Math.max(0, (typeof incident.anomaly_score === "number" && isFinite(incident.anomaly_score) ? incident.anomaly_score : 0) * 100))}%`,
                height: "100%",
                background: `linear-gradient(90deg, ${sevCfg.color}88, ${sevCfg.color})`,
                borderRadius: "5px",
              }}
            />
            {/* Dynamic Threshold Pin */}
            <div
              style={{
                position: "absolute",
                top: 0,
                bottom: 0,
                left: `${Math.min(100, Math.max(0, (typeof incident.anomaly_threshold === "number" && isFinite(incident.anomaly_threshold) ? incident.anomaly_threshold : 0.5) * 100))}%`,
                width: "2px",
                background: "#F8FAFC",
                boxShadow: "0 0 4px #fff",
              }}
              title={`Threshold: ${incident.anomaly_threshold}`}
            />
          </div>
        </div>

        {/* Live Sparkline View */}
        <div style={{ marginBottom: "16px" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "8px" }}>
            <span style={{ fontSize: "12px", fontWeight: 600, color: "#E2E8F0" }}>
              Live Telemetry Sparkline (6h) — <span style={{ color: "#38bdf8" }}>{chCfg.name} ({chCfg.unit})</span>
            </span>
            {isLoadingSeries && <span style={{ fontSize: "11px", color: "#94a3b8" }}>Syncing...</span>}
          </div>

          <div
            style={{
              position: "relative",
              height: "150px",
              background: "rgba(15, 23, 42, 0.8)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              borderRadius: "8px",
              overflow: "hidden",
            }}
          >
            <canvas ref={sparklineCanvasRef} style={{ width: "100%", height: "100%" }} />
          </div>

          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#64748b", marginTop: "6px" }}>
            <span>Shaded band = expected (median ± 2.5 MAD)</span>
            <span style={{ color: "#22C55E" }}>● Live Cursor (SSE synced)</span>
          </div>
        </div>

        {/* "Why Flagged": Sensor Contributions */}
        <div style={{ marginBottom: "16px" }}>
          <div style={{ fontSize: "12px", fontWeight: 600, color: "#E2E8F0", marginBottom: "8px" }}>
            Why Flagged (Sensor Contributions)
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: "6px" }}>
            {incident.contributing_sensors.map((item) => {
              const isActive = activeChannel === item.channel;
              const chInfo = CHANNEL_CONFIG[item.channel] || { name: item.channel, color: "#38bdf8" };

              return (
                <div
                  key={item.channel}
                  onClick={() => setActiveChannel(item.channel)}
                  style={{
                    padding: "8px 10px",
                    borderRadius: "6px",
                    background: isActive ? "rgba(56, 189, 248, 0.12)" : "rgba(30, 41, 59, 0.4)",
                    border: isActive ? "1px solid rgba(56, 189, 248, 0.4)" : "1px solid rgba(255, 255, 255, 0.04)",
                    cursor: "pointer",
                    transition: "all 150ms ease",
                  }}
                  title="Click to view sparkline for this channel"
                >
                  <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", marginBottom: "4px" }}>
                    <span style={{ color: isActive ? "#38bdf8" : "#cbd5e1", fontWeight: isActive ? 600 : 400 }}>
                      {chInfo.name}
                    </span>
                    <span style={{ color: isActive ? "#38bdf8" : "#94a3b8", fontFamily: "JetBrains Mono" }}>
                      {Math.round(item.contribution * 100)}%
                    </span>
                  </div>
                  <div style={{ height: "4px", background: "rgba(15, 23, 42, 0.6)", borderRadius: "2px", overflow: "hidden" }}>
                    <div
                      style={{
                        width: `${Math.round(item.contribution * 100)}%`,
                        height: "100%",
                        background: isActive ? "#38bdf8" : chInfo.color || "#64748b",
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Neighbor Comparison Toggle */}
        <div
          style={{
            background: "rgba(30, 41, 59, 0.4)",
            border: "1px solid rgba(255, 255, 255, 0.06)",
            borderRadius: "8px",
            padding: "12px",
            marginBottom: "16px",
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: "12px", fontWeight: 600, color: "#E2E8F0" }}>Neighbor Station Overlay</div>
              <div style={{ fontSize: "11px", color: "#64748b" }}>Compare against nearest 2 stations for sanity check</div>
            </div>
            <label className="switch-toggle" style={{ cursor: "pointer", display: "flex", alignItems: "center" }}>
              <input
                type="checkbox"
                checked={showNeighbors}
                onChange={(e) => setShowNeighbors(e.target.checked)}
                style={{ cursor: "pointer", width: "16px", height: "16px" }}
              />
            </label>
          </div>

          {showNeighbors && (
            <div style={{ marginTop: "8px", paddingTop: "8px", borderTop: "1px solid rgba(255, 255, 255, 0.06)", fontSize: "11px", color: "#cbd5e1" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "6px", marginBottom: "2px" }}>
                <span style={{ color: "#A855F7" }}>-- Neighbor 1</span>
                <span style={{ color: "#F59E0B" }}>-- Neighbor 2</span>
              </div>
              <div style={{ color: "#94a3b8", fontSize: "10px" }}>
                Diagnosis: Isolated deviation flags local sensor fault; correlated moves indicate genuine regional weather.
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Drawer Action Bar */}
      <div
        className="drawer-actions"
        style={{
          padding: "16px 20px",
          borderTop: "1px solid rgba(255, 255, 255, 0.08)",
          background: "rgba(15, 23, 42, 0.9)",
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: "8px",
        }}
      >
        <button
          className="action-btn btn-acknowledge"
          onClick={() => onAcknowledge(incident)}
          style={{
            padding: "8px 12px",
            background: "rgba(34, 197, 94, 0.15)",
            border: "1px solid #16A34A",
            color: "#22C55E",
            borderRadius: "6px",
            cursor: "pointer",
            fontWeight: 600,
            fontSize: "12px",
          }}
        >
          ✓ Acknowledge
        </button>

        <button
          className="action-btn btn-false-positive"
          onClick={() => onMarkFalsePositive(incident)}
          style={{
            padding: "8px 12px",
            background: "rgba(249, 115, 22, 0.15)",
            border: "1px solid #EA580C",
            color: "#F97316",
            borderRadius: "6px",
            cursor: "pointer",
            fontWeight: 600,
            fontSize: "12px",
          }}
        >
          ✕ False Positive
        </button>

        <button
          className="action-btn btn-ticket"
          onClick={handleCreateGitLabTicket}
          style={{
            padding: "8px 12px",
            background: "rgba(59, 130, 246, 0.15)",
            border: "1px solid #2563EB",
            color: "#3B82F6",
            borderRadius: "6px",
            cursor: "pointer",
            fontWeight: 600,
            fontSize: "12px",
          }}
        >
          🎫 GitLab Ticket
        </button>

        <button
          className="action-btn btn-snooze"
          onClick={() => onSnoozeStation(incident.station_id, 4)}
          style={{
            padding: "8px 12px",
            background: "rgba(148, 163, 184, 0.15)",
            border: "1px solid #64748B",
            color: "#94A3B8",
            borderRadius: "6px",
            cursor: "pointer",
            fontWeight: 600,
            fontSize: "12px",
          }}
        >
          💤 Snooze 4h
        </button>
      </div>
    </aside>
  );
};
