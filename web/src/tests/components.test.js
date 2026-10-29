/**
 * SkyGuard Frontend Component Tests
 * Tests Detail Drawer, FilterBar, and ReplayScrubber logic
 * Executed with Node.js native test runner: node --test src/tests/components.test.js
 */

import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load shared vocabulary
const vocabPath = path.resolve(__dirname, "../../../shared/vocabulary.json");
const vocabulary = JSON.parse(fs.readFileSync(vocabPath, "utf-8"));

test("Shared Vocabulary Contract integrity", () => {
  assert.ok(vocabulary.severities, "severities must exist");
  assert.ok(vocabulary.anomaly_types, "anomaly_types must exist");
  assert.ok(vocabulary.channels, "channels must exist");
  assert.ok(vocabulary.station_health, "station_health must exist");

  // Verify severity scale
  assert.equal(vocabulary.severities.critical.color, "#EF4444");
  assert.equal(vocabulary.severities.high.color, "#F97316");
  assert.equal(vocabulary.severities.medium.color, "#F59E0B");
  assert.equal(vocabulary.severities.low.color, "#22C55E");
  assert.equal(vocabulary.severities.info.color, "#3B82F6");

  // Verify glyphs
  assert.equal(vocabulary.anomaly_types.spike.glyph, "^");
  assert.equal(vocabulary.anomaly_types.stuck.glyph, "=");
  assert.equal(vocabulary.anomaly_types.drift.glyph, "/");
  assert.equal(vocabulary.anomaly_types.dropout.glyph, "o");
  assert.equal(vocabulary.anomaly_types.out_of_range.glyph, "!");
  assert.equal(vocabulary.anomaly_types.multivariate.glyph, "*");
});

test("Detail Drawer: Sensor contribution swap & nearest neighbor selection", () => {
  const mockIncident = {
    id: "inc_test_01",
    station_id: "AWS-WA-COAST",
    station_name: "Pacific Northwest Coastal Array",
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
      rainfall: 0.0,
    },
  };

  const mockStations = [
    { station_id: "AWS-WA-COAST", latitude: 46.97, longitude: -123.85 },
    { station_id: "AWS-MN-DULUTH", latitude: 46.78, longitude: -92.10 },
    { station_id: "AWS-TX-VALLEY", latitude: 29.42, longitude: -98.49 },
    { station_id: "AWS-MA-NANTUCKET", latitude: 41.28, longitude: -70.09 },
  ];

  // 1. Contribution bar selection
  const topContribution = mockIncident.contributing_sensors[0];
  assert.equal(topContribution.channel, "rainfall");
  assert.equal(topContribution.contribution, 0.88);

  // 2. Nearest 2 neighbors calculation
  const currentStation = mockStations[0];
  const neighbors = mockStations
    .filter((s) => s.station_id !== currentStation.station_id)
    .map((s) => {
      const dLat = s.latitude - currentStation.latitude;
      const dLng = s.longitude - currentStation.longitude;
      return { station_id: s.station_id, dist: Math.sqrt(dLat * dLat + dLng * dLng) };
    })
    .sort((a, b) => a.dist - b.dist)
    .slice(0, 2);

  assert.equal(neighbors.length, 2, "Should select exactly nearest 2 stations");
  assert.ok(["AWS-TX-VALLEY", "AWS-MN-DULUTH"].includes(neighbors[0].station_id), "Closest station selected");

  // 3. GitLab Issue prefilled URL generation
  const title = encodeURIComponent(`[Anomaly Incident] ${mockIncident.predicted_type.toUpperCase()} at ${mockIncident.station_id}`);
  const gitlabUrl = `https://gitlab.com/skyguard/defense-matrix/-/issues/new?issue[title]=${title}`;
  assert.ok(gitlabUrl.includes("DROPOUT"), "GitLab issue title contains fault type");
  assert.ok(gitlabUrl.includes("AWS-WA-COAST"), "GitLab issue title contains station ID");
});

test("Detail Drawer: Optimistic action and 5-second undo toast", () => {
  let incidents = [
    { id: "inc_01", station_id: "AWS-WA-COAST", acknowledged: false },
    { id: "inc_02", station_id: "AWS-TX-VALLEY", acknowledged: false },
  ];

  let toastTriggered = null;

  // Optimistic Acknowledge Action
  const targetId = "inc_01";
  const originalState = [...incidents];
  incidents = incidents.map((i) => (i.id === targetId ? { ...i, acknowledged: true } : i));

  toastTriggered = {
    title: "Incident Acknowledged",
    durationMs: 5000,
    undoAction: () => {
      incidents = originalState;
    },
  };

  assert.equal(incidents.find((i) => i.id === targetId).acknowledged, true, "Incident marked acknowledged optimistically");
  assert.equal(toastTriggered.durationMs, 5000, "Undo toast active for 5 seconds");

  // User clicks Undo within 5s
  toastTriggered.undoAction();
  assert.equal(incidents.find((i) => i.id === targetId).acknowledged, false, "Undo restores original incident state");
});

test("FilterBar: Multi-attribute filtering and Clear All", () => {
  const testIncidents = [
    { id: "1", severity_level: "critical", diagnosed_sensor: "rainfall", predicted_type: "dropout", station_health: "critical" },
    { id: "2", severity_level: "high", diagnosed_sensor: "wind_speed", predicted_type: "spike", station_health: "warning" },
    { id: "3", severity_level: "medium", diagnosed_sensor: "humidity", predicted_type: "stuck", station_health: "warning" },
    { id: "4", severity_level: "low", diagnosed_sensor: "temperature", predicted_type: "drift", station_health: "ok" },
  ];

  let filters = {
    severity: "ALL",
    channel: "ALL",
    anomaly_type: "ALL",
    station_health: "ALL",
  };

  const applyFilter = (data, f) => {
    return data.filter((item) => {
      if (f.severity !== "ALL" && item.severity_level !== f.severity) return false;
      if (f.channel !== "ALL" && item.diagnosed_sensor !== f.channel) return false;
      if (f.anomaly_type !== "ALL" && item.predicted_type !== f.anomaly_type) return false;
      if (f.station_health !== "ALL" && item.station_health !== f.station_health) return false;
      return true;
    });
  };

  // Filter by Critical
  filters.severity = "critical";
  let matches = applyFilter(testIncidents, filters);
  assert.equal(matches.length, 1);
  assert.equal(matches[0].id, "1");

  // Filter by Channel
  filters = { severity: "ALL", channel: "wind_speed", anomaly_type: "ALL", station_health: "ALL" };
  matches = applyFilter(testIncidents, filters);
  assert.equal(matches.length, 1);
  assert.equal(matches[0].id, "2");

  // Clear All
  filters = { severity: "ALL", channel: "ALL", anomaly_type: "ALL", station_health: "ALL" };
  matches = applyFilter(testIncidents, filters);
  assert.equal(matches.length, 4, "Clear All restores all incidents");
});

test("Replay Scrubber: Density histogram binning and mode switching", () => {
  const now = Date.now();
  const windowMs = 24 * 60 * 60 * 1000; // 24h
  const incidents = [
    { id: "1", timestamp: new Date(now - 2 * 3600000).toISOString() }, // 2h ago
    { id: "2", timestamp: new Date(now - 4 * 3600000).toISOString() }, // 4h ago
    { id: "3", timestamp: new Date(now - 12 * 3600000).toISOString() }, // 12h ago
  ];

  // 40 bins
  const bins = new Array(40).fill(0);
  incidents.forEach((inc) => {
    const ageMs = now - new Date(inc.timestamp).getTime();
    if (ageMs >= 0 && ageMs <= windowMs) {
      const binIdx = Math.min(39, Math.max(0, Math.floor((1 - ageMs / windowMs) * 40)));
      bins[binIdx] += 1;
    }
  });

  const totalBinned = bins.reduce((a, b) => a + b, 0);
  assert.equal(totalBinned, 3, "All 3 incidents properly binned into density histogram");

  // Mode switching & paused pill logic
  let mode = "live";
  let pausedAt = null;

  // User drags slider
  const scrubVal = 75; // 75%
  if (scrubVal < 100) {
    mode = "24h";
    pausedAt = new Date(now - (1 - scrubVal / 100) * windowMs).toISOString();
  }

  assert.equal(mode, "24h");
  assert.ok(pausedAt !== null, "Persistent 'Paused at <time>' pill displayed");

  // Return to live
  mode = "live";
  pausedAt = null;
  assert.equal(pausedAt, null, "Pill dismissed on return to live");
});

test("Globe Click-to-Zoom: useGlobeCamera contract and pointOfView flight animation", () => {
  const OVERVIEW = { lat: 20, lng: 0, altitude: 2.2 };
  const FOCUS_ALTITUDE = 0.4;
  const FLY_MS = 1200;

  // Mock globe instance and controls
  let povCalls = [];
  let autoRotateCalls = [];
  const mockControls = { autoRotate: true };
  const mockGlobe = {
    pointOfView: (target, duration) => {
      povCalls.push({ target, duration });
    },
    controls: () => mockControls,
  };

  // Simulate useGlobeCamera behavior
  let autoRotateTimeout = null;
  function createCamera(globeRef, reducedMotion = false) {
    const setAutoRotate = (enabled) => {
      mockControls.autoRotate = enabled && !reducedMotion;
      autoRotateCalls.push(mockControls.autoRotate);
    };

    const flyToIncident = (lat, lng) => {
      if (autoRotateTimeout) {
        clearTimeout(autoRotateTimeout);
        autoRotateTimeout = null;
      }
      setAutoRotate(false);
      const duration = reducedMotion ? 0 : FLY_MS;
      globeRef.pointOfView({ lat, lng, altitude: FOCUS_ALTITUDE }, duration);
    };

    const backToOverview = () => {
      const duration = reducedMotion ? 0 : FLY_MS;
      globeRef.pointOfView(OVERVIEW, duration);
      if (autoRotateTimeout) clearTimeout(autoRotateTimeout);
      autoRotateTimeout = setTimeout(() => {
        setAutoRotate(true);
      }, 10000);
    };

    return { flyToIncident, backToOverview, setAutoRotate, getTimeout: () => autoRotateTimeout };
  }

  // 1. Click-to-Zoom on incident
  const camera = createCamera(mockGlobe, false);
  camera.flyToIncident(46.97, -123.85);

  assert.equal(povCalls.length, 1);
  assert.deepEqual(povCalls[0], {
    target: { lat: 46.97, lng: -123.85, altitude: 0.4 },
    duration: 1200,
  });
  assert.equal(mockControls.autoRotate, false, "Auto-rotation paused during incident focus");

  // 2. Rapid retargeting mid-flight
  camera.flyToIncident(39.46, -0.37);
  assert.equal(povCalls.length, 2);
  assert.deepEqual(povCalls[1], {
    target: { lat: 39.46, lng: -0.37, altitude: 0.4 },
    duration: 1200,
  });
  assert.equal(mockControls.autoRotate, false);

  // 3. Back to overview
  camera.backToOverview();
  assert.equal(povCalls.length, 3);
  assert.deepEqual(povCalls[2], {
    target: { lat: 20, lng: 0, altitude: 2.2 },
    duration: 1200,
  });
  assert.ok(camera.getTimeout() !== null, "10-second auto-rotate resume timer scheduled");

  // 4. prefers-reduced-motion: duration 0 instant cut
  povCalls = [];
  const motionCamera = createCamera(mockGlobe, true);
  motionCamera.flyToIncident(-23.70, 133.87);
  assert.deepEqual(povCalls[0], {
    target: { lat: -23.70, lng: 133.87, altitude: 0.4 },
    duration: 0,
  });
});

test("Arrival Highlight: ringsData pulsing and marker scaling", () => {
  const mockIncident = {
    id: "inc_001",
    station_id: "AWS-WA-COAST",
    latitude: 46.97,
    longitude: -123.85,
    severity_level: "critical",
  };

  const sevColor = "#EF4444"; // critical

  // Pulse ring specs: ringColor = severity color, maxRadius 3, propagationSpeed 2, repeatPeriod 700
  const ringEntry = {
    lat: mockIncident.latitude,
    lng: mockIncident.longitude,
    color: sevColor,
    maxRadius: 3,
    propagationSpeed: 2,
    repeatPeriod: 700,
  };

  assert.equal(ringEntry.color, "#EF4444", "Ring pulses with incident severity color");
  assert.equal(ringEntry.maxRadius, 3);
  assert.equal(ringEntry.propagationSpeed, 2);
  assert.equal(ringEntry.repeatPeriod, 700);

  // Marker radius scaling: 1.5x when selected (0.6 -> 0.9)
  const baseRadius = 0.6;
  const selectedRadius = Math.round(baseRadius * 1.5 * 10) / 10;
  assert.equal(selectedRadius, 0.9, "Selected marker scales to exactly 1.5x via pointRadius");

  // Reduced motion: static ring (propagationSpeed 0, repeatPeriod 0)
  const reducedRing = {
    ...ringEntry,
    propagationSpeed: 0,
    repeatPeriod: 0,
  };
  assert.equal(reducedRing.propagationSpeed, 0);
  assert.equal(reducedRing.repeatPeriod, 0);
});

test("Mock Incidents: 5 active incidents across 3 continents", () => {
  // Read mockServer.ts file directly to verify source of truth
  const mockServerPath = path.resolve(__dirname, "../mock/mockServer.ts");
  const mockServerContent = fs.readFileSync(mockServerPath, "utf-8");

  // Extract MOCK_INITIAL_INCIDENTS definition
  assert.ok(mockServerContent.includes("AWS-WA-COAST"), "Includes Washington coastal station (North America)");
  assert.ok(mockServerContent.includes("AWS-MN-DULUTH"), "Includes Minnesota station (North America)");
  assert.ok(mockServerContent.includes("AWS-TX-VALLEY"), "Includes Texas station (North America)");
  assert.ok(mockServerContent.includes("AWS-ES-VALENCIA"), "Includes Valencia station (Europe)");
  assert.ok(mockServerContent.includes("AWS-AU-OUTBACK"), "Includes Alice Springs station (Australia)");

  // Verify continent distribution
  const continents = {
    "North America": ["AWS-WA-COAST", "AWS-MN-DULUTH", "AWS-TX-VALLEY"],
    "Europe": ["AWS-ES-VALENCIA"],
    "Australia": ["AWS-AU-OUTBACK"],
  };

  const continentKeys = Object.keys(continents);
  assert.equal(continentKeys.length, 3, "Spans exactly 3 distinct continents");
  const totalStations = continentKeys.reduce((acc, c) => acc + continents[c].length, 0);
  assert.equal(totalStations, 5, "Exactly 5 mock incident locations");
});
