import React, { useEffect, useRef, useState, useCallback } from "react";
import { Incident, StationMeta, RegionalArc } from "../types";
import { SEVERITY_CONFIG } from "../constants";

interface GlobeViewProps {
  stations: StationMeta[];
  incidents: Incident[];
  selectedIncident: Incident | null;
  hoveredIncidentId: string | null;
  onSelectIncident: (incident: Incident) => void;
  onHoverIncident: (id: string | null) => void;
  onDeselect: () => void;
  globeRef: React.MutableRefObject<any>;
  reducedMotion: boolean;
}

interface ArrivalRing {
  lat: number;
  lng: number;
  color: string;
  maxRadius: number;
  propagationSpeed: number;
  repeatPeriod: number;
}

export const GlobeView: React.FC<GlobeViewProps> = ({
  stations,
  incidents,
  selectedIncident,
  hoveredIncidentId,
  onSelectIncident,
  onHoverIncident,
  onDeselect,
  globeRef,
  reducedMotion,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isAutoRotating, setIsAutoRotating] = useState(true);
  const [regionalArcs, setRegionalArcs] = useState<RegionalArc[]>([]);
  const [arrivalRings, setArrivalRings] = useState<ArrivalRing[]>([]);

  // Compute 30-minute regional co-occurrence arcs
  useEffect(() => {
    const arcs: RegionalArc[] = [];
    const windowMs = 30 * 60 * 1000;

    for (let i = 0; i < incidents.length; i++) {
      for (let j = i + 1; j < incidents.length; j++) {
        const a = incidents[i];
        const b = incidents[j];
        if (a.station_id === b.station_id) continue;

        const timeA = new Date(a.timestamp).getTime();
        const timeB = new Date(b.timestamp).getTime();

        // Same anomaly archetype within 30 minutes
        const typeA = a.predicted_type || a.anomaly_type;
        const typeB = b.predicted_type || b.anomaly_type;
        if (typeA === typeB && Math.abs(timeA - timeB) <= windowMs) {
          const sev = a.severity_level || a.severity || "medium";
          const sevColor = SEVERITY_CONFIG[sev]?.color || "#F59E0B";
          arcs.push({
            startLat: a.latitude ?? a.lat ?? 0,
            startLng: a.longitude ?? a.lng ?? 0,
            endLat: b.latitude ?? b.lat ?? 0,
            endLng: b.longitude ?? b.lng ?? 0,
            color: sevColor,
            anomaly_type: typeA,
            stations: [a.station_id, b.station_id],
          });
        }
      }
    }
    setRegionalArcs(arcs);
  }, [incidents]);

  // Arrival pulse ring on selected target for ~2.5s
  useEffect(() => {
    if (!selectedIncident) {
      setArrivalRings([]);
      return;
    }

    const lat = selectedIncident.latitude ?? selectedIncident.lat;
    const lng = selectedIncident.longitude ?? selectedIncident.lng;
    if (lat === undefined || lng === undefined) return;

    const sev = selectedIncident.severity_level || selectedIncident.severity || "medium";
    const sevColor = SEVERITY_CONFIG[sev]?.color || "#F59E0B";

    setArrivalRings([
      {
        lat,
        lng,
        color: sevColor,
        maxRadius: 3,
        propagationSpeed: reducedMotion ? 0 : 2,
        repeatPeriod: reducedMotion ? 0 : 700,
      },
    ]);

    const timer = setTimeout(() => {
      setArrivalRings([]);
    }, 2500);

    return () => clearTimeout(timer);
  }, [selectedIncident, reducedMotion]);

  // Push updated arrival rings to globe instance
  useEffect(() => {
    if (globeRef.current && globeRef.current.ringsData) {
      globeRef.current.ringsData(arrivalRings);
    }
  }, [arrivalRings, globeRef]);

  // Initialize Globe
  useEffect(() => {
    if (!containerRef.current) return;

    const initGlobe = () => {
      const GlobeFn = (window as any).Globe;
      if (!GlobeFn) {
        setTimeout(initGlobe, 150);
        return;
      }

      if (globeRef.current) return;

      try {
        const globe = GlobeFn()(containerRef.current)
          .globeImageUrl("/vendor/earth-blue-marble.jpg")
          .bumpImageUrl("/vendor/earth-topology.png")
          .backgroundImageUrl("/vendor/night-sky.png")
          .showAtmosphere(true)
          .atmosphereColor("#38bdf8")
          .atmosphereAltitude(0.22)
          .arcsData([])
          .arcColor((d: any) => d.color || "#F59E0B")
          .arcDashLength(0.4)
          .arcDashGap(0.2)
          .arcDashAnimateTime(2000)
          .arcStroke(0.7)
          // Arrival pulse rings configuration
          .ringsData([])
          .ringLat((d: any) => d.lat)
          .ringLng((d: any) => d.lng)
          .ringColor((d: any) => d.color)
          .ringMaxRadius((d: any) => d.maxRadius || 3)
          .ringPropagationSpeed((d: any) => (d.propagationSpeed !== undefined ? d.propagationSpeed : 2))
          .ringRepeatPeriod((d: any) => (d.repeatPeriod !== undefined ? d.repeatPeriod : 700))
          // Points Data configuration (with 1.5x radius scaling when selected)
          .pointsData([])
          .pointLat((d: any) => d.lat)
          .pointLng((d: any) => d.lng)
          .pointColor((d: any) => d.color)
          .pointRadius((d: any) => {
            const isSelected = selectedIncident && (selectedIncident.id === d.id || selectedIncident.station_id === d.station_id);
            return isSelected ? 0.9 : 0.6; // 1.5x enlargement
          })
          .pointAltitude(0.01)
          .onPointClick((p: any) => {
            const inc = incidents.find((i) => i.id === p.id || i.station_id === p.station_id) || p;
            onSelectIncident(inc);
          })
          // Empty space click resets to overview and clears selection
          .onGlobeClick(() => {
            onDeselect();
          })
          .htmlElementsData([])
          .htmlElement((d: any) => {
            const el = document.createElement("div");
            const sevColor = d.color;
            const isSelected = selectedIncident && (selectedIncident.id === d.id || selectedIncident.station_id === d.station_id);
            const isHovered = hoveredIncidentId === d.id;
            const isNew = d.is_new;

            el.className = `globe-pin ${d.severity_level} ${isSelected ? "selected" : ""} ${isHovered ? "hovered" : ""}`;
            el.style.setProperty("--pin-color", sevColor);

            const sizePx = Math.max(28, Math.min(52, Math.round(30 + d.anomaly_score * 20)));

            el.innerHTML = `
              <div class="globe-marker-outer ${isNew ? "pulse-60s" : ""}" style="width: ${sizePx}px; height: ${sizePx}px; border-color: ${sevColor}; background: ${d.bg_rgba || "rgba(15,23,42,0.85)"}">
                <span class="glyph-badge" style="color: ${sevColor}; font-weight: 800; font-size: ${Math.round(sizePx * 0.42)}px;">${d.glyph}</span>
                ${isNew ? `<div class="ring-pulse-beacon" style="border-color: ${sevColor};"></div>` : ""}
              </div>
              <div class="marker-label-chip">
                <span class="label-dot" style="background: ${sevColor};"></span>
                <span class="label-text">${d.station_id}</span>
              </div>
            `;

            el.onmouseenter = () => {
              onHoverIncident(d.id);
            };

            el.onmouseleave = () => {
              onHoverIncident(null);
            };

            el.onclick = (e) => {
              e.stopPropagation();
              onSelectIncident(d);
            };

            return el;
          });

        const controls = typeof globe.controls === "function" ? globe.controls() : globe.controls;
        if (controls) {
          controls.autoRotate = isAutoRotating && !reducedMotion;
          controls.autoRotateSpeed = 0.55;
          controls.enableDamping = true;
          controls.dampingFactor = 0.05;
          controls.minDistance = 120;
          controls.maxDistance = 650;
        }

        // Global Overview Position
        globe.pointOfView({ lat: 20, lng: 0, altitude: 2.2 });
        globeRef.current = globe;

        const handleResize = () => {
          if (!containerRef.current || !globeRef.current) return;
          const rect = containerRef.current.getBoundingClientRect();
          globeRef.current.width(rect.width);
          globeRef.current.height(rect.height);
        };

        window.addEventListener("resize", handleResize);
        setTimeout(handleResize, 100);
        setTimeout(handleResize, 500);
      } catch (err) {
        console.warn("Globe initialization failed, fallback to 2D canvas", err);
      }
    };

    initGlobe();

    return () => {
      if (globeRef.current) {
        globeRef.current._destructor && globeRef.current._destructor();
        globeRef.current = null;
      }
    };
  }, []);

  // Update Markers, Points, and Arcs on state change
  useEffect(() => {
    if (!globeRef.current) return;

    // Combine station coordinates with active incident information
    const markerData = stations.map((st) => {
      const stationIncident = incidents.find((inc) => inc.station_id === st.station_id);
      const sev = stationIncident
        ? (stationIncident.severity_level || stationIncident.severity)
        : (st.health === "critical" ? "critical" : st.health === "warning" ? "medium" : "low");
      const sevConfig = SEVERITY_CONFIG[sev] || SEVERITY_CONFIG.low;
      const glyph = stationIncident ? stationIncident.glyph : (st.health === "critical" ? "!" : st.health === "warning" ? "•" : "✓");
      const score = stationIncident ? (stationIncident.anomaly_score ?? stationIncident.score) : (st.health === "critical" ? 0.92 : 0.15);

      return {
        id: stationIncident ? stationIncident.id : `stn_${st.station_id}`,
        station_id: st.station_id,
        name: st.name,
        lat: st.latitude,
        lng: st.longitude,
        latitude: st.latitude,
        longitude: st.longitude,
        anomaly_score: score,
        score: score,
        anomaly_threshold: stationIncident?.anomaly_threshold || 0.5,
        predicted_type: stationIncident?.predicted_type || stationIncident?.anomaly_type || "normal",
        anomaly_type: stationIncident?.predicted_type || stationIncident?.anomaly_type || "normal",
        severity_level: sev,
        severity: sev,
        glyph: glyph,
        color: sevConfig.color,
        bg_rgba: sevConfig.bg_rgba,
        is_new: stationIncident ? stationIncident.is_new : false,
        diagnosed_sensor: stationIncident?.diagnosed_sensor || stationIncident?.channel || "none",
        channel: stationIncident?.diagnosed_sensor || stationIncident?.channel || "none",
        contributing_sensors: stationIncident?.contributing_sensors || [],
        readings: stationIncident ? stationIncident.readings : st.latest_readings,
      };
    });

    globeRef.current.htmlElementsData(markerData);
    globeRef.current.pointsData(markerData);
    globeRef.current.arcsData(regionalArcs);
  }, [stations, incidents, selectedIncident, hoveredIncidentId, regionalArcs, globeRef]);

  // Handle manual auto-rotation state
  useEffect(() => {
    if (!globeRef.current) return;
    const controls = typeof globeRef.current.controls === "function" ? globeRef.current.controls() : globeRef.current.controls;
    if (controls) {
      controls.autoRotate = isAutoRotating && !reducedMotion;
    }
  }, [isAutoRotating, reducedMotion, globeRef]);

  const toggleAutoRotate = useCallback(() => {
    setIsAutoRotating((prev) => {
      const next = !prev;
      if (globeRef.current) {
        const controls = typeof globeRef.current.controls === "function" ? globeRef.current.controls() : globeRef.current.controls;
        if (controls) {
          controls.autoRotate = next && !reducedMotion;
        }
      }
      return next;
    });
  }, [globeRef, reducedMotion]);

  return (
    <div className="canvas-viewport" style={{ position: "relative", width: "100%", height: "100%" }}>
      <div
        id="globeContainer"
        ref={containerRef}
        className="viewport-surface active"
        style={{ width: "100%", height: "100%", position: "absolute", inset: 0 }}
      />

      {/* Globe Controls & Orbit Badge */}
      <div className="globe-control-badge">
        <button
          className="globe-tool-btn"
          id="btnGlobeAutoOrbit"
          onClick={toggleAutoRotate}
          title="Toggle Earth Auto-Rotation"
        >
          <span>🌍</span>
          <span>Auto-Orbit: {isAutoRotating ? "ON" : "OFF"}</span>
        </button>
        <button
          className="globe-tool-btn"
          id="btnGlobeOverview"
          onClick={onDeselect}
          title="Reset to Global Overview (Esc / Click empty space)"
        >
          <span>🌐</span>
          <span>Overview</span>
        </button>
      </div>

      {/* 3D Defense Matrix Legend */}
      <div className="map-legend-card">
        <div className="legend-header">3D DEFENSE MATRIX</div>
        <div className="legend-items">
          <div className="legend-item">
            <span className="dot dot-critical" style={{ background: SEVERITY_CONFIG.critical.color }}></span>
            <span>Critical Threat [!]</span>
          </div>
          <div className="legend-item">
            <span className="dot dot-high" style={{ background: SEVERITY_CONFIG.high.color }}></span>
            <span>High Anomaly [^]</span>
          </div>
          <div className="legend-item">
            <span className="dot dot-moderate" style={{ background: SEVERITY_CONFIG.medium.color }}></span>
            <span>Moderate Advisory [=]</span>
          </div>
          <div className="legend-item">
            <span className="dot dot-low" style={{ background: SEVERITY_CONFIG.low.color }}></span>
            <span>Low Deviation [/]</span>
          </div>
          {regionalArcs.length > 0 && (
            <div className="legend-item" style={{ borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "4px", marginTop: "4px" }}>
              <span style={{ color: "#38bdf8", fontSize: "11px" }}>⚡ Regional Arcs: {regionalArcs.length} active (30m)</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
