import React, { useState, useEffect, useCallback, useMemo, useRef } from "react";
import {
  Incident,
  StationMeta,
  FilterState,
  ReplayMode,
  SystemSettings,
  ToastMessage,
  StationHealth,
} from "./types";
import { SEVERITY_CONFIG, DEFAULT_SETTINGS } from "./constants";
import { GlobeView } from "./components/GlobeView";
import { IncidentPanel } from "./components/IncidentPanel";
import { DetailDrawer } from "./components/DetailDrawer";
import { FilterBar } from "./components/FilterBar";
import { ReplayScrubber } from "./components/ReplayScrubber";
import { CommandPalette } from "./components/CommandPalette";
import { SettingsModal } from "./components/SettingsModal";
import { TimelineView } from "./components/TimelineView";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { useSSE } from "./hooks/useSSE";
import { useGlobeCamera } from "./hooks/useGlobeCamera";
import { MOCK_STATIONS, MOCK_INITIAL_INCIDENTS } from "./mock/mockServer";

export const App: React.FC = () => {
  // State
  const [stations, setStations] = useState<StationMeta[]>(MOCK_STATIONS);
  const [incidents, setIncidents] = useState<Incident[]>(MOCK_INITIAL_INCIDENTS);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);
  const [hoveredIncidentId, setHoveredIncidentId] = useState<string | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const [currentView, setCurrentView] = useState<"globe" | "timeline">("globe");

  // Filter State
  const [filters, setFilters] = useState<FilterState>({
    severity: "ALL",
    channel: "ALL",
    anomaly_type: "ALL",
    station_health: "ALL",
    search: "",
  });

  // Replay State
  const [replayMode, setReplayMode] = useState<ReplayMode>("live");
  const [scrubTimestamp, setScrubTimestamp] = useState<number | null>(null);
  const [pausedAt, setPausedAt] = useState<string | null>(null);

  const [settings, setSettings] = useState<SystemSettings>({
    sensitivity: 1.0,
    models: {
      isolation_forest: true,
      autoencoder: true,
      statistical: true,
      lof: true,
    },
    theme: "dark",
    reduced_motion: false,
    sound_enabled: false,
    estimated_incidents_24h: 8,
  });

  const globeRef = useRef<any>(null);
  const camera = useGlobeCamera(globeRef, {
    reducedMotion: settings.reduced_motion,
  });

  const [isCommandPaletteOpen, setIsCommandPaletteOpen] = useState<boolean>(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState<boolean>(false);
  const [toasts, setToasts] = useState<ToastMessage[]>([]);

  const audioCtxRef = useRef<AudioContext | null>(null);

  // Play synthetic Web Audio chime for critical incident (if sound enabled)
  const playCriticalAlertSound = useCallback(() => {
    if (!settings.sound_enabled) return;
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      const ctx = audioCtxRef.current;
      if (ctx.state === "suspended") ctx.resume();

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = "sine";
      osc.frequency.setValueAtTime(880, ctx.currentTime); // A5
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.35); // A4
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.35);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.35);
    } catch (e) {}
  }, [settings.sound_enabled]);

  // Toast notification helper with optional 5-second undo callback
  const showToast = useCallback((toast: Omit<ToastMessage, "id"> & { id?: string }) => {
    const id = toast.id || `toast_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
    const fullToast: ToastMessage = { ...toast, id, durationMs: toast.durationMs || 5000 };

    setToasts((prev) => [...prev, fullToast]);

    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, fullToast.durationMs);
  }, []);

  // Fetch initial stations and settings from backend
  useEffect(() => {
    fetch("/stations")
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => {
        if (Array.isArray(data) && data.length > 0) {
          setStations(data);
        }
      })
      .catch(() => {});

    fetch("/settings")
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => {
        if (data && data.sensitivity) {
          setSettings((prev) => ({ ...prev, ...data }));
        }
      })
      .catch(() => {});

    fetch("/anomalies?limit=50")
      .then((res) => (res.ok ? res.json() : Promise.reject(res)))
      .then((data) => {
        if (data && Array.isArray(data.anomalies) && data.anomalies.length > 0) {
          const formatted: Incident[] = data.anomalies.map((a: any, idx: number) => ({
            id: `api_anom_${idx}_${a.station_id}`,
            timestamp: a.timestamp,
            station_id: a.station_id,
            station_name: a.station_name || a.station_id,
            latitude: a.latitude,
            longitude: a.longitude,
            anomaly_score: a.anomaly_score,
            anomaly_threshold: a.anomaly_threshold || 0.5,
            predicted_type: a.predicted_type || "spike",
            severity_level: a.severity_level || "medium",
            glyph: a.glyph || "•",
            diagnosed_sensor: a.diagnosed_sensor || "temperature",
            contributing_sensors: a.contributing_sensors || [],
            readings: a.readings || {},
            is_new: false,
          }));
          setIncidents(formatted);
        }
      })
      .catch(() => {});
  }, []);

  // SSE Stream Listeners
  const handleIncomingIncident = useCallback(
    (rawIncident: Incident) => {
      const newIncident: Incident = {
        ...rawIncident,
        anomaly_score: typeof rawIncident.anomaly_score === "number" && isFinite(rawIncident.anomaly_score) ? rawIncident.anomaly_score : 0.15,
        anomaly_threshold: typeof rawIncident.anomaly_threshold === "number" && isFinite(rawIncident.anomaly_threshold) ? rawIncident.anomaly_threshold : 0.5,
        severity_level: rawIncident.severity_level || "medium",
        predicted_type: rawIncident.predicted_type || "spike",
        glyph: rawIncident.glyph || "•",
        diagnosed_sensor: rawIncident.diagnosed_sensor || "temperature",
        contributing_sensors: Array.isArray(rawIncident.contributing_sensors) ? rawIncident.contributing_sensors : [],
        readings: rawIncident.readings || {},
      };

      setIncidents((prev) => [newIncident, ...prev.slice(0, 99)]);

      // Remove "is_new" after 3 seconds so glow fades
      setTimeout(() => {
        setIncidents((prev) =>
          prev.map((inc) => (inc.id === newIncident.id ? { ...inc, is_new: false } : inc))
        );
      }, 3000);

      if (newIncident.severity_level === "critical") {
        showToast({
          title: "Critical Sensor Alert",
          message: `${newIncident.station_name || newIncident.station_id}: ${newIncident.predicted_type.toUpperCase()} on ${newIncident.diagnosed_sensor}`,
          type: "critical",
          durationMs: 5000,
        });
      }
    },
    [showToast]
  );

  const handleStationHealthUpdate = useCallback((stationId: string, health: string) => {
    setStations((prev) =>
      prev.map((s) => (s.station_id === stationId ? { ...s, health: health as StationHealth } : s))
    );
  }, []);

  const sseState = useSSE(handleIncomingIncident, handleStationHealthUpdate);

  // Update Settings API
  const handleUpdateSettings = async (newSettings: Partial<SystemSettings>) => {
    try {
      const res = await fetch("/settings", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newSettings),
      });
      if (res.ok) {
        const data = await res.json();
        setSettings((prev) => ({ ...prev, ...data.settings }));
      } else {
        setSettings((prev) => ({ ...prev, ...newSettings }));
      }
    } catch (e) {
      setSettings((prev) => ({ ...prev, ...newSettings }));
    }
  };

  // Incident Selection Action (Opens Drawer and flies camera)
  const handleSelectIncident = useCallback(
    (incident: Incident) => {
      setSelectedIncident(incident);
      setIsDrawerOpen(true);
      const lat = incident.latitude ?? incident.lat;
      const lng = incident.longitude ?? incident.lng;
      if (lat !== undefined && lng !== undefined) {
        camera.flyToIncident(lat, lng);
      }
    },
    [camera]
  );

  // Deselect / Reset to Overview Action
  const handleDeselect = useCallback(() => {
    setSelectedIncident(null);
    setIsDrawerOpen(false);
    camera.backToOverview();
  }, [camera]);

  const handleCloseDrawer = () => {
    setIsDrawerOpen(false);
  };

  // Optimistic Actions with 5-Second Undo Toast
  const handleAcknowledge = (incident: Incident) => {
    const originalState = [...incidents];
    setIncidents((prev) =>
      prev.map((i) => (i.id === incident.id ? { ...i, acknowledged: true } : i))
    );
    setIsDrawerOpen(false);

    showToast({
      title: "Incident Acknowledged",
      message: `Marked ${incident.station_id} anomaly as reviewed.`,
      type: "success",
      durationMs: 5000,
      undoLabel: "Undo",
      undoAction: () => {
        setIncidents(originalState);
        showToast({ title: "Action Reverted", message: "Incident unacknowledged.", type: "info" });
      },
    });
  };

  const handleMarkFalsePositive = (incident: Incident) => {
    const originalState = [...incidents];
    setIncidents((prev) => prev.filter((i) => i.id !== incident.id));
    setIsDrawerOpen(false);

    showToast({
      title: "Marked False Positive",
      message: `Suppressed incident for ${incident.station_id}. Feedback sent to ensemble.`,
      type: "warning",
      durationMs: 5000,
      undoLabel: "Undo",
      undoAction: () => {
        setIncidents(originalState);
        showToast({ title: "Action Reverted", message: "Incident restored.", type: "info" });
      },
    });
  };

  const handleSnoozeStation = (stationId: string, durationHours: number) => {
    const originalStations = [...stations];
    setStations((prev) =>
      prev.map((s) => (s.station_id === stationId ? { ...s, health: "offline" } : s))
    );
    setIsDrawerOpen(false);

    showToast({
      title: `Station ${stationId} Snoozed`,
      message: `Alerts suppressed for next ${durationHours} hours.`,
      type: "info",
      durationMs: 5000,
      undoLabel: "Undo",
      undoAction: () => {
        setStations(originalStations);
        showToast({ title: "Snooze Cancelled", message: `Alerts re-enabled for ${stationId}.`, type: "info" });
      },
    });
  };

  // Keyboard Shortcuts: J/K to move, Enter to open, Esc to close, F to focus globe, Cmd/Ctrl+K for palette
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setIsCommandPaletteOpen((prev) => !prev);
        return;
      }

      if (e.key === "Escape") {
        if (isCommandPaletteOpen) {
          setIsCommandPaletteOpen(false);
          return;
        }
        if (isSettingsOpen) {
          setIsSettingsOpen(false);
          return;
        }
        if (selectedIncident || isDrawerOpen) {
          handleDeselect();
          return;
        }
      }

      // Ignore input element typing
      const target = e.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "SELECT" || target.tagName === "TEXTAREA") {
        return;
      }

      if (e.key.toLowerCase() === "j") {
        // Next incident
        e.preventDefault();
        if (incidents.length === 0) return;
        const currIdx = selectedIncident ? incidents.findIndex((i) => i.id === selectedIncident.id) : -1;
        const nextIdx = Math.min(incidents.length - 1, currIdx + 1);
        handleSelectIncident(incidents[nextIdx]);
      } else if (e.key.toLowerCase() === "k") {
        // Previous incident
        e.preventDefault();
        if (incidents.length === 0) return;
        const currIdx = selectedIncident ? incidents.findIndex((i) => i.id === selectedIncident.id) : 1;
        const prevIdx = Math.max(0, currIdx - 1);
        handleSelectIncident(incidents[prevIdx]);
      } else if (e.key === "Enter") {
        if (selectedIncident && !isDrawerOpen) {
          setIsDrawerOpen(true);
        }
      } else if (e.key.toLowerCase() === "f") {
        setCurrentView("globe");
        showToast({ title: "Perspective Reset", message: "Focused on global network array.", type: "info", durationMs: 1200 });
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [incidents, selectedIncident, isDrawerOpen, isCommandPaletteOpen, isSettingsOpen, showToast]);

  // Replay Time Handlers
  const handleScrubTimeChange = (ts: number | null) => {
    if (ts === null) {
      setScrubTimestamp(null);
      setPausedAt(null);
    } else {
      setScrubTimestamp(ts);
      setPausedAt(new Date(ts).toISOString());
    }
  };

  const handleResumeLive = () => {
    setReplayMode("live");
    setScrubTimestamp(null);
    setPausedAt(null);
    showToast({ title: "Resumed Live Stream", message: "Synchronized with real-time SSE telemetry feed.", type: "success", durationMs: 2000 });
  };

  // Filtered Incidents
  const filteredIncidents = useMemo(() => {
    return incidents.filter((inc) => {
      if (inc.acknowledged) return false;
      if (filters.severity !== "ALL" && inc.severity_level !== filters.severity) return false;
      if (filters.channel !== "ALL" && inc.diagnosed_sensor !== filters.channel) return false;
      if (filters.anomaly_type !== "ALL" && inc.predicted_type !== filters.anomaly_type) return false;

      if (filters.station_health !== "ALL") {
        const stn = stations.find((s) => s.station_id === inc.station_id);
        if (stn && stn.health !== filters.station_health) return false;
      }

      if (scrubTimestamp !== null) {
        const incTime = new Date(inc.timestamp).getTime();
        // Show incidents occurring around the scrubbed time (+/- 30 min)
        if (Math.abs(incTime - scrubTimestamp) > 30 * 60000) return false;
      }

      return true;
    });
  }, [incidents, filters, scrubTimestamp, stations]);

  // Heartbeat Indicator State
  const heartbeatColor = useMemo(() => {
    if (!sseState.connected) return "#EF4444"; // Red (Disconnected)
    if (sseState.lastUpdatedSecondsAgo > 90) return "#EF4444"; // Red (>90s)
    if (sseState.lastUpdatedSecondsAgo > 30) return "#F59E0B"; // Amber (>30s)
    return "#22C55E"; // Green (<30s)
  }, [sseState]);

  return (
    <div
      className={`app-container ${settings.theme}-theme ${settings.reduced_motion ? "reduced-motion" : ""}`}
      style={{
        display: "flex",
        flexDirection: "column",
        width: "100vw",
        height: "100vh",
        background: "#0A0F1D",
        color: "#F8FAFC",
        overflow: "hidden",
      }}
    >
      {/* ═══════════════════════════════════════════ */}
      {/* TOP NAVIGATION BAR                          */}
      {/* ═══════════════════════════════════════════ */}
      <header
        className="top-bar"
        style={{
          height: "56px",
          padding: "0 20px",
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          background: "rgba(15, 23, 42, 0.95)",
          borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
          zIndex: 20,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          <div style={{ width: "32px", height: "32px", borderRadius: "8px", background: "rgba(56, 189, 248, 0.15)", border: "1px solid #38bdf8", display: "flex", alignItems: "center", justifyContent: "center", color: "#38bdf8", fontSize: "16px" }}>
            🛡️
          </div>
          <div>
            <div style={{ fontSize: "14px", fontWeight: 800, letterSpacing: "0.5px", color: "#F8FAFC" }}>
              SkyGuard <span style={{ fontSize: "10px", padding: "1px 6px", background: "rgba(56, 189, 248, 0.2)", color: "#38bdf8", borderRadius: "4px", marginLeft: "4px" }}>DEFENSE MATRIX</span>
            </div>
            <div style={{ fontSize: "10px", color: "#64748b" }}>
              Unsupervised Sensor Anomaly Engine
            </div>
          </div>
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: "12px" }}>
          {/* Command Palette Trigger */}
          <button
            onClick={() => setIsCommandPaletteOpen(true)}
            style={{
              padding: "6px 12px",
              background: "rgba(30, 41, 59, 0.6)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: "6px",
              color: "#94a3b8",
              fontSize: "12px",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: "8px",
            }}
          >
            <span>🔍 Search station or type...</span>
            <kbd style={{ background: "rgba(255,255,255,0.08)", padding: "1px 5px", borderRadius: "4px", fontSize: "10px" }}>
              Ctrl+K
            </kbd>
          </button>

          {/* View Toggles */}
          <div style={{ display: "flex", background: "rgba(30, 41, 59, 0.6)", borderRadius: "6px", padding: "2px", border: "1px solid rgba(255,255,255,0.08)" }}>
            <button
              onClick={() => setCurrentView("globe")}
              style={{
                padding: "4px 10px",
                borderRadius: "4px",
                fontSize: "11px",
                border: "none",
                cursor: "pointer",
                background: currentView === "globe" ? "#38bdf8" : "transparent",
                color: currentView === "globe" ? "#0F172A" : "#94a3b8",
                fontWeight: currentView === "globe" ? 700 : 500,
              }}
            >
              🌍 3D Globe
            </button>
            <button
              onClick={() => setCurrentView("timeline")}
              style={{
                padding: "4px 10px",
                borderRadius: "4px",
                fontSize: "11px",
                border: "none",
                cursor: "pointer",
                background: currentView === "timeline" ? "#38bdf8" : "transparent",
                color: currentView === "timeline" ? "#0F172A" : "#94a3b8",
                fontWeight: currentView === "timeline" ? 700 : 500,
              }}
            >
              ⏱️ Timeline
            </button>
          </div>

          {/* Real-Time SSE Heartbeat Indicator */}
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "8px",
              padding: "4px 10px",
              background: "rgba(30, 41, 59, 0.5)",
              border: "1px solid rgba(255, 255, 255, 0.08)",
              borderRadius: "6px",
              fontSize: "11px",
            }}
          >
            <div
              style={{
                width: "8px",
                height: "8px",
                borderRadius: "50%",
                background: heartbeatColor,
                boxShadow: `0 0 8px ${heartbeatColor}`,
              }}
            />
            <span style={{ color: "#cbd5e1" }}>
              {sseState.reconnecting
                ? "Reconnecting..."
                : sseState.lastUpdatedSecondsAgo <= 3
                ? "Updated just now"
                : `Updated ${sseState.lastUpdatedSecondsAgo}s ago`}
            </span>
          </div>

          {/* Settings Button */}
          <button
            onClick={() => setIsSettingsOpen(true)}
            style={{
              width: "32px",
              height: "32px",
              borderRadius: "6px",
              background: "rgba(30, 41, 59, 0.6)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              color: "#94a3b8",
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: "14px",
            }}
            title="Settings & Calibration"
          >
            ⚙️
          </button>
        </div>
      </header>

      {/* Filter Bar */}
      <FilterBar
        filters={filters}
        onFilterChange={(newFilters) => setFilters((prev) => ({ ...prev, ...newFilters }))}
        onClearAll={() =>
          setFilters({
            severity: "ALL",
            channel: "ALL",
            anomaly_type: "ALL",
            station_health: "ALL",
            search: "",
          })
        }
      />

      {/* ═══════════════════════════════════════════ */}
      {/* MAIN WORKSPACE: INCIDENT PANEL + GLOBE/TIMELINE */}
      {/* ═══════════════════════════════════════════ */}
      <div
        className="main-workspace"
        style={{
          flex: 1,
          display: "flex",
          position: "relative",
          overflow: "hidden",
        }}
      >
        {/* Left Incident Panel */}
        <div style={{ width: "360px", minWidth: "320px", height: "100%", zIndex: 10 }}>
          <ErrorBoundary>
            <IncidentPanel
              incidents={filteredIncidents}
              selectedIncident={selectedIncident}
              hoveredIncidentId={hoveredIncidentId}
              onSelectIncident={handleSelectIncident}
              onHoverIncident={setHoveredIncidentId}
            />
          </ErrorBoundary>
        </div>

        {/* Center Viewport: Globe or Timeline */}
        <main
          className="center-panel"
          style={{
            flex: 1,
            height: "100%",
            position: "relative",
            transition: "filter 300ms ease",
            filter: isDrawerOpen ? "brightness(0.80)" : "brightness(1)",
          }}
        >
          <ErrorBoundary>
            {currentView === "globe" ? (
              <GlobeView
                stations={stations}
                incidents={filteredIncidents}
                selectedIncident={selectedIncident}
                hoveredIncidentId={hoveredIncidentId}
                onSelectIncident={handleSelectIncident}
                onHoverIncident={setHoveredIncidentId}
                onDeselect={handleDeselect}
                globeRef={globeRef}
                reducedMotion={settings.reduced_motion}
              />
            ) : (
              <TimelineView
                incidents={filteredIncidents}
                selectedIncident={selectedIncident}
                hoveredIncidentId={hoveredIncidentId}
                onSelectIncident={handleSelectIncident}
                onHoverIncident={setHoveredIncidentId}
              />
            )}
          </ErrorBoundary>
        </main>

        {/* Incident Detail Drawer (Slides from Right) */}
        <ErrorBoundary>
          <DetailDrawer
            incident={selectedIncident}
            stations={stations}
            isOpen={isDrawerOpen}
            onClose={handleCloseDrawer}
            onAcknowledge={handleAcknowledge}
            onMarkFalsePositive={handleMarkFalsePositive}
            onSnoozeStation={handleSnoozeStation}
            onShowToast={showToast}
          />
        </ErrorBoundary>
      </div>

      {/* ═══════════════════════════════════════════ */}
      {/* BOTTOM TIME CONTROL & REPLAY SCRUBBER       */}
      {/* ═══════════════════════════════════════════ */}
      <ErrorBoundary>
        <ReplayScrubber
          mode={replayMode}
          incidents={incidents}
          onModeChange={setReplayMode}
          onScrubTimeChange={handleScrubTimeChange}
          isPaused={pausedAt !== null}
          pausedTimestamp={pausedAt}
          onResumeLive={handleResumeLive}
        />
      </ErrorBoundary>

      {/* Command Palette Modal (Ctrl+K) */}
      <CommandPalette
        isOpen={isCommandPaletteOpen}
        onClose={() => setIsCommandPaletteOpen(false)}
        stations={stations}
        onFlyToStation={(st) => {
          const match = incidents.find((i) => i.station_id === st.station_id);
          if (match) {
            handleSelectIncident(match);
          } else {
            setSelectedIncident({
              id: `stn_select_${st.station_id}`,
              timestamp: new Date().toISOString(),
              station_id: st.station_id,
              station_name: st.name,
              latitude: st.latitude,
              longitude: st.longitude,
              anomaly_score: 0.15,
              anomaly_threshold: 0.50,
              predicted_type: "spike",
              severity_level: "low",
              glyph: "•",
              diagnosed_sensor: "temperature",
              contributing_sensors: [],
              readings: st.latest_readings,
            });
            setIsDrawerOpen(true);
          }
        }}
        onFilterAnomalyType={(type) => setFilters((prev) => ({ ...prev, anomaly_type: type }))}
        onToggleTheme={() =>
          setSettings((prev) => ({ ...prev, theme: prev.theme === "dark" ? "light" : "dark" }))
        }
        onToggleReducedMotion={() =>
          setSettings((prev) => ({ ...prev, reduced_motion: !prev.reduced_motion }))
        }
        onSelectReplayWindow={(range) => {
          setReplayMode(range as ReplayMode);
          handleScrubTimeChange(Date.now() - 2 * 3600000);
        }}
        theme={settings.theme}
        reducedMotion={settings.reduced_motion}
      />

      {/* Settings Modal */}
      <SettingsModal
        isOpen={isSettingsOpen}
        onClose={() => setIsSettingsOpen(false)}
        settings={settings}
        onUpdateSettings={handleUpdateSettings}
      />

      {/* Toast Notification Container with Undo Support */}
      <div
        className="toast-container"
        style={{
          position: "fixed",
          bottom: "76px",
          right: "24px",
          zIndex: 99999,
          display: "flex",
          flexDirection: "column",
          gap: "8px",
          pointerEvents: "none",
        }}
      >
        {toasts.map((t) => (
          <div
            key={t.id}
            style={{
              pointerEvents: "auto",
              padding: "12px 16px",
              borderRadius: "8px",
              background: t.type === "critical" ? "rgba(239, 68, 68, 0.95)" : "rgba(15, 23, 42, 0.96)",
              border: t.type === "critical" ? "1px solid #DC2626" : "1px solid rgba(255, 255, 255, 0.15)",
              color: "#FFFFFF",
              boxShadow: "0 8px 24px rgba(0,0,0,0.6)",
              minWidth: "280px",
              maxWidth: "380px",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: "12px",
              animation: "slideInRight 200ms ease",
            }}
          >
            <div>
              <div style={{ fontSize: "12px", fontWeight: 700 }}>{t.title}</div>
              <div style={{ fontSize: "11px", color: t.type === "critical" ? "#FEE2E2" : "#cbd5e1" }}>
                {t.message}
              </div>
            </div>

            {t.undoAction && (
              <button
                onClick={t.undoAction}
                style={{
                  padding: "4px 8px",
                  background: "#F59E0B",
                  border: "none",
                  borderRadius: "4px",
                  color: "#0F172A",
                  fontWeight: 800,
                  fontSize: "11px",
                  cursor: "pointer",
                  whiteSpace: "nowrap",
                }}
              >
                {t.undoLabel || "Undo"}
              </button>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
export default App;
