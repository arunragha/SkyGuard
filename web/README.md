# SkyGuard — Interactive Anomaly Dashboard

Real-time Meteorological Sensor Anomaly Detection Console built with **React + TypeScript**, **Three.js / react-globe.gl**, and **FastAPI REST + SSE**.

---

## Quickstart

### 1. Launch Backend API & Live SSE Stream
Ensure the Python virtual environment is active, then start the FastAPI server:
```bash
.\.venv\Scripts\python.exe -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```
- Interactive Swagger docs: `http://localhost:8000/docs`
- SSE Live stream: `http://localhost:8000/stream`

### 2. Launch React Dashboard
Navigate to the `web/` directory and start the Vite development server:
```bash
cd web
npm install
npm run dev
```
The dashboard will open at **`http://localhost:3000`** with live proxying to the FastAPI backend.

To build the optimized production bundle:
```bash
npm run build
```

### 3. Run Component Tests
Execute the component tests for Detail Drawer, FilterBar, and ReplayScrubber:
```bash
npm test
```

---

## Features & Interaction Model

### 1. Tri-Directional Linking (Globe ↔ Panel ↔ Timeline)
- **Hover**: Hovering a globe marker, left panel incident row, or timeline tick highlights all three simultaneously.
- **Click**: Flies the 3D globe camera to the station (smooth 600ms ease-out), pulses the marker, scrolls the panel row into view, and slides open the Incident Detail Drawer.
- **Marker Encoding**:
  - **Color**: Severity (`Critical #EF4444`, `High #F97316`, `Medium #F59E0B`, `Low #22C55E`, `Info #3B82F6`)
  - **Size**: Proportional to anomaly score ($0.0 \dots 1.0$)
  - **Ring Pulse**: Indicates an incident detected in the last 60 seconds.
  - **Glyph**: Anomaly archetype (`^` Spike, `=` Stuck, `/` Drift, `o` Dropout, `!` Out of Range, `*` Multivariate).
- **Regional Arcs**: Stations flagging the same anomaly archetype within 30 minutes draw a faint correlation arc between them.

### 2. Forensic Detail Drawer
- **6-Hour Sparkline**: High-resolution time-series with the anomaly window shaded, expected band (rolling median $\pm$ 2.5 MAD) drawn behind the observed line, and a live cursor advancing on each SSE tick.
- **"Why Flagged"**: Horizontal bars showing per-sensor contribution. **Clicking any bar swaps the sparkline to that sensor channel.**
- **Neighbor Comparison**: Overlays the nearest 2 stations' telemetry to help technicians distinguish sensor faults from genuine regional weather.
- **Optimistic Work Orders**:
  - *Acknowledge* (optimistic review)
  - *Mark False Positive* (suppresses incident, feeds back to ensemble)
  - *Create GitLab Ticket* (opens prefilled issue URL with telemetry snapshot)
  - *Snooze Station* (suppresses station alerts for 4h)
  - Every action displays an **Undo Toast with a 5-second countdown**.

### 3. Multi-Attribute Filtering & Command Palette
- **Filter Chips**: Severity, Sensor Channel, Anomaly Type, and Station Health applied identically across Globe, Panel, and Timeline, with active chips and "Clear All".
- **Command Palette (`Cmd/Ctrl+K`)**: Jump directly to any station, filter by anomaly archetype, toggle theme, or adjust replay window.
- **Keyboard Navigation**:
  - `J` / `K`: Move down / up through incident cards
  - `Enter`: Open detail drawer for focused incident
  - `Esc`: Close detail drawer or Command Palette
  - `F`: Focus camera on global network array

### 4. Temporal Replay Scrubber
- Modes: `Live | 1h | 24h | 7d | Custom`
- Scrubber with an **incident density histogram** background. Dragging scrubs through historical time and filters the globe and panel to that moment.
- Play / Pause with `1x`, `5x`, and `20x` playback speeds.
- When paused, displays a persistent pill: `"Paused at <time>. Return to live"`.

### 5. Settings & Operational Calibration
- **Sensitivity Slider**: Bound to `PUT /settings` ($0.5 \dots 3.0$) with immediate live calculation of **"Estimated incidents / 24h"**.
- Model toggles (Isolation Forest, Conv1D Autoencoder, Robust Z-Score, LOF).
- Light / Dark console theme toggle.
- Reduced-motion toggle (forces instant transitions and respects `prefers-reduced-motion`).
- Audio alert toggle (synthesizes audio alert chime on Critical anomalies via Web Audio API).

---

## Single Source of Truth (`shared/vocabulary.json`)
The CLI and UI both import from `shared/vocabulary.json` to ensure zero drift in labels, color tokens, and glyphs.
Test CLI output parity:
```bash
python -m skyguard.cli --csv data/weather_station_telemetry.csv --limit 10
```
