import React, { useState, useEffect, useRef, useMemo } from "react";
import { ReplayMode, Incident } from "../types";

interface ReplayScrubberProps {
  mode: ReplayMode;
  incidents: Incident[];
  onModeChange: (mode: ReplayMode) => void;
  onScrubTimeChange: (scrubTimestamp: number | null) => void;
  isPaused: boolean;
  pausedTimestamp: string | null;
  onResumeLive: () => void;
}

export const ReplayScrubber: React.FC<ReplayScrubberProps> = ({
  mode,
  incidents,
  onModeChange,
  onScrubTimeChange,
  isPaused,
  pausedTimestamp,
  onResumeLive,
}) => {
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1); // 1x, 5x, 20x
  const [scrubberValue, setScrubberValue] = useState<number>(100); // 0 to 100%
  const animationFrameRef = useRef<any>(null);

  // Compute time window in milliseconds
  const windowMs = useMemo(() => {
    switch (mode) {
      case "1h":
        return 60 * 60 * 1000;
      case "24h":
        return 24 * 60 * 60 * 1000;
      case "7d":
        return 7 * 24 * 60 * 60 * 1000;
      default:
        return 6 * 60 * 60 * 1000;
    }
  }, [mode]);

  // Compute incident density histogram (40 bins across the window)
  const densityBins = useMemo(() => {
    const bins = new Array(40).fill(0);
    const now = Date.now();

    incidents.forEach((inc) => {
      const t = new Date(inc.timestamp).getTime();
      const ageMs = now - t;
      if (ageMs >= 0 && ageMs <= windowMs) {
        const binIndex = Math.min(39, Math.max(0, Math.floor((1 - ageMs / windowMs) * 40)));
        bins[binIndex] += 1;
      }
    });

    const maxCount = Math.max(1, ...bins);
    return bins.map((count) => count / maxCount);
  }, [incidents, windowMs]);

  // When in live mode, reset scrubber to 100%
  useEffect(() => {
    if (mode === "live") {
      setScrubberValue(100);
      setIsPlaying(false);
      onScrubTimeChange(null);
    }
  }, [mode]);

  // Playback timer loop
  useEffect(() => {
    if (!isPlaying) {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      return;
    }

    let lastTick = performance.now();

    const loop = (currentTick: number) => {
      const deltaSec = (currentTick - lastTick) / 1000;
      lastTick = currentTick;

      setScrubberValue((prev) => {
        // Increment progress scaled by speed
        const increment = (deltaSec * playbackSpeed * 100) / 60; // 60s to scrub full bar at 1x
        const nextVal = prev + increment;
        if (nextVal >= 100) {
          setIsPlaying(false);
          return 100;
        }
        return nextVal;
      });

      animationFrameRef.current = requestAnimationFrame(loop);
    };

    animationFrameRef.current = requestAnimationFrame(loop);

    return () => {
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [isPlaying, playbackSpeed]);

  // Notify parent of scrubbed timestamp whenever slider moves
  const handleSliderChange = (newVal: number) => {
    setScrubberValue(newVal);
    if (newVal >= 100) {
      onResumeLive();
    } else {
      const now = Date.now();
      const targetTime = now - (1 - newVal / 100) * windowMs;
      onScrubTimeChange(targetTime);
    }
  };

  const formattedScrubTime = useMemo(() => {
    if (scrubberValue >= 100 || mode === "live") return "LIVE MONITORING";
    const now = Date.now();
    const targetTime = new Date(now - (1 - scrubberValue / 100) * windowMs);
    return targetTime.toISOString().replace("T", " ").slice(0, 19) + " UTC";
  }, [scrubberValue, mode, windowMs]);

  return (
    <div
      className="replay-scrubber-dock"
      style={{
        padding: "10px 20px",
        background: "rgba(15, 23, 42, 0.92)",
        backdropFilter: "blur(12px)",
        borderTop: "1px solid rgba(255, 255, 255, 0.08)",
        display: "flex",
        alignItems: "center",
        gap: "16px",
      }}
    >
      {/* Mode Selector Tabs */}
      <div style={{ display: "flex", gap: "4px" }}>
        {(["live", "1h", "24h", "7d"] as const).map((m) => {
          const isActive = mode === m;
          return (
            <button
              key={m}
              onClick={() => onModeChange(m)}
              style={{
                padding: "4px 10px",
                borderRadius: "4px",
                fontSize: "11px",
                fontWeight: isActive ? 700 : 500,
                cursor: "pointer",
                background: isActive ? (m === "live" ? "rgba(34, 197, 94, 0.2)" : "rgba(56, 189, 248, 0.2)") : "rgba(30, 41, 59, 0.5)",
                border: isActive ? (m === "live" ? "1px solid #22C55E" : "1px solid #38bdf8") : "1px solid rgba(255, 255, 255, 0.08)",
                color: isActive ? (m === "live" ? "#22C55E" : "#38bdf8") : "#94a3b8",
                transition: "all 120ms ease",
              }}
            >
              {m === "live" ? "● LIVE" : m.toUpperCase()}
            </button>
          );
        })}
      </div>

      <div style={{ width: "1px", height: "24px", background: "rgba(255, 255, 255, 0.1)" }} />

      {/* Play/Pause & Step Controls */}
      <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
        <button
          onClick={() => setIsPlaying(!isPlaying)}
          disabled={mode === "live"}
          style={{
            width: "32px",
            height: "32px",
            borderRadius: "6px",
            background: isPlaying ? "rgba(239, 68, 68, 0.2)" : "rgba(56, 189, 248, 0.2)",
            border: `1px solid ${isPlaying ? "#EF4444" : "#38bdf8"}`,
            color: isPlaying ? "#EF4444" : "#38bdf8",
            cursor: mode === "live" ? "not-allowed" : "pointer",
            fontWeight: "bold",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            opacity: mode === "live" ? 0.5 : 1,
          }}
          title={isPlaying ? "Pause replay" : "Play replay"}
        >
          {isPlaying ? "⏸" : "▶"}
        </button>

        {/* Speed Selector (1x, 5x, 20x) */}
        <div style={{ display: "flex", gap: "2px" }}>
          {[1, 5, 20].map((spd) => (
            <button
              key={spd}
              onClick={() => setPlaybackSpeed(spd)}
              style={{
                padding: "2px 6px",
                borderRadius: "3px",
                fontSize: "10px",
                fontWeight: playbackSpeed === spd ? 700 : 400,
                cursor: "pointer",
                background: playbackSpeed === spd ? "rgba(56, 189, 248, 0.25)" : "transparent",
                border: playbackSpeed === spd ? "1px solid #38bdf8" : "1px solid transparent",
                color: playbackSpeed === spd ? "#38bdf8" : "#64748b",
              }}
            >
              {spd}×
            </button>
          ))}
        </div>
      </div>

      {/* Scrubber Slider with Density Histogram Background */}
      <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: "4px" }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", fontSize: "11px" }}>
          <span style={{ color: "#94a3b8", fontFamily: "JetBrains Mono" }}>{formattedScrubTime}</span>
          {isPaused && (
            <span
              onClick={onResumeLive}
              style={{
                background: "rgba(245, 158, 11, 0.18)",
                border: "1px solid #F59E0B",
                color: "#F59E0B",
                padding: "2px 8px",
                borderRadius: "12px",
                cursor: "pointer",
                fontWeight: 600,
                fontSize: "10px",
              }}
            >
              Paused at {pausedTimestamp ? pausedTimestamp.slice(11, 19) : "time"}. Return to live ↻
            </span>
          )}
        </div>

        {/* Histogram + Slider Container */}
        <div style={{ position: "relative", height: "24px", display: "flex", alignItems: "flex-end" }}>
          {/* Density Histogram Bars behind slider */}
          <div
            style={{
              position: "absolute",
              inset: 0,
              display: "flex",
              alignItems: "flex-end",
              gap: "2px",
              pointerEvents: "none",
              padding: "0 2px",
            }}
          >
            {densityBins.map((heightNorm, bIdx) => (
              <div
                key={bIdx}
                style={{
                  flex: 1,
                  height: `${Math.max(2, Math.round(heightNorm * 18))}px`,
                  background: heightNorm > 0.6 ? "rgba(56, 189, 248, 0.4)" : heightNorm > 0.3 ? "rgba(56, 189, 248, 0.25)" : "rgba(56, 189, 248, 0.12)",
                  borderRadius: "1px",
                }}
              />
            ))}
          </div>

          <input
            type="range"
            min="0"
            max="100"
            step="0.5"
            value={scrubberValue}
            onChange={(e) => handleSliderChange(parseFloat(e.target.value))}
            style={{
              width: "100%",
              position: "relative",
              zIndex: 2,
              cursor: "pointer",
              accentColor: isPaused ? "#F59E0B" : "#22C55E",
            }}
          />
        </div>
      </div>

      {/* Return to Live Button */}
      {(mode !== "live" || scrubberValue < 100) && (
        <button
          onClick={onResumeLive}
          style={{
            padding: "6px 12px",
            borderRadius: "6px",
            background: "rgba(34, 197, 94, 0.2)",
            border: "1px solid #22C55E",
            color: "#22C55E",
            fontWeight: 700,
            fontSize: "11px",
            cursor: "pointer",
            whiteSpace: "nowrap",
          }}
        >
          ● LIVE
        </button>
      )}
    </div>
  );
};
