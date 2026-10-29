import React, { useState, useEffect, useRef } from "react";
import { StationMeta, Incident } from "../types";
import { ANOMALY_CONFIG } from "../constants";

interface CommandPaletteProps {
  isOpen: boolean;
  onClose: () => void;
  stations: StationMeta[];
  onFlyToStation: (station: StationMeta) => void;
  onFilterAnomalyType: (type: string) => void;
  onToggleTheme: () => void;
  onToggleReducedMotion: () => void;
  onSelectReplayWindow: (range: string) => void;
  theme: "dark" | "light";
  reducedMotion: boolean;
}

export const CommandPalette: React.FC<CommandPaletteProps> = ({
  isOpen,
  onClose,
  stations,
  onFlyToStation,
  onFilterAnomalyType,
  onToggleTheme,
  onToggleReducedMotion,
  onSelectReplayWindow,
  theme,
  reducedMotion,
}) => {
  const [query, setQuery] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (isOpen) {
      setQuery("");
      setSelectedIndex(0);
      setTimeout(() => inputRef.current?.focus(), 50);
    }
  }, [isOpen]);

  if (!isOpen) return null;

  // Build command items
  const commands: {
    category: string;
    id: string;
    label: string;
    action: () => void;
    icon: string;
  }[] = [];

  // Station jump commands
  stations.forEach((st) => {
    commands.push({
      category: "Stations",
      id: `station_${st.station_id}`,
      label: `Jump to ${st.name} (${st.station_id})`,
      icon: "📍",
      action: () => {
        onFlyToStation(st);
        onClose();
      },
    });
  });

  // Filter by Anomaly Type
  Object.entries(ANOMALY_CONFIG).forEach(([type, info]) => {
    commands.push({
      category: "Filters",
      id: `filter_${type}`,
      label: `Filter by ${info.label} (${info.glyph})`,
      icon: info.glyph,
      action: () => {
        onFilterAnomalyType(type);
        onClose();
      },
    });
  });

  // System & Window Actions
  commands.push({
    category: "Replay",
    id: "replay_24h",
    label: "Replay window: Last 24 Hours",
    icon: "⏪",
    action: () => {
      onSelectReplayWindow("24h");
      onClose();
    },
  });
  commands.push({
    category: "Replay",
    id: "replay_7d",
    label: "Replay window: Last 7 Days",
    icon: "⏪",
    action: () => {
      onSelectReplayWindow("7d");
      onClose();
    },
  });
  commands.push({
    category: "Preferences",
    id: "toggle_theme",
    label: `Toggle Theme (Current: ${theme.toUpperCase()})`,
    icon: "🎨",
    action: () => {
      onToggleTheme();
      onClose();
    },
  });
  commands.push({
    category: "Preferences",
    id: "toggle_motion",
    label: `Toggle Reduced Motion (Current: ${reducedMotion ? "ON" : "OFF"})`,
    icon: "♿",
    action: () => {
      onToggleReducedMotion();
      onClose();
    },
  });

  const filteredCommands = commands.filter((c) =>
    c.label.toLowerCase().includes(query.toLowerCase()) ||
    c.category.toLowerCase().includes(query.toLowerCase())
  );

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Escape") {
      onClose();
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev + 1) % Math.max(1, filteredCommands.length));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setSelectedIndex((prev) => (prev - 1 + filteredCommands.length) % Math.max(1, filteredCommands.length));
    } else if (e.key === "Enter") {
      e.preventDefault();
      if (filteredCommands[selectedIndex]) {
        filteredCommands[selectedIndex].action();
      }
    }
  };

  return (
    <div
      className="command-palette-backdrop"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(10, 15, 29, 0.75)",
        backdropFilter: "blur(6px)",
        display: "flex",
        justifyContent: "center",
        alignItems: "flex-start",
        paddingTop: "12vh",
      }}
      onClick={onClose}
    >
      <div
        className="command-palette-modal"
        style={{
          width: "560px",
          maxWidth: "92vw",
          background: "rgba(15, 23, 42, 0.96)",
          border: "1px solid rgba(56, 189, 248, 0.3)",
          borderRadius: "12px",
          boxShadow: "0 20px 48px rgba(0, 0, 0, 0.8)",
          overflow: "hidden",
        }}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={handleKeyDown}
      >
        {/* Search Input */}
        <div
          style={{
            padding: "14px 18px",
            borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
            display: "flex",
            alignItems: "center",
            gap: "10px",
          }}
        >
          <span style={{ color: "#38bdf8", fontSize: "16px" }}>🔍</span>
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => {
              setQuery(e.target.value);
              setSelectedIndex(0);
            }}
            placeholder="Type a station name, anomaly archetype, or command..."
            style={{
              flex: 1,
              background: "none",
              border: "none",
              outline: "none",
              color: "#F8FAFC",
              fontSize: "14px",
              fontFamily: "Outfit, sans-serif",
            }}
          />
          <span style={{ fontSize: "11px", color: "#64748b", background: "rgba(255,255,255,0.06)", padding: "2px 6px", borderRadius: "4px" }}>
            ESC to close
          </span>
        </div>

        {/* Results List */}
        <div style={{ maxHeight: "360px", overflowY: "auto", padding: "8px" }}>
          {filteredCommands.length === 0 ? (
            <div style={{ padding: "24px", textAlign: "center", color: "#94a3b8", fontSize: "13px" }}>
              No commands matching "{query}"
            </div>
          ) : (
            filteredCommands.map((cmd, idx) => {
              const isSelected = idx === selectedIndex;
              return (
                <div
                  key={cmd.id}
                  onClick={cmd.action}
                  onMouseEnter={() => setSelectedIndex(idx)}
                  style={{
                    padding: "10px 14px",
                    borderRadius: "6px",
                    display: "flex",
                    alignItems: "center",
                    gap: "10px",
                    cursor: "pointer",
                    background: isSelected ? "rgba(56, 189, 248, 0.16)" : "transparent",
                    border: isSelected ? "1px solid rgba(56, 189, 248, 0.3)" : "1px solid transparent",
                    transition: "all 100ms ease",
                  }}
                >
                  <span style={{ fontSize: "16px", width: "22px", textAlign: "center" }}>{cmd.icon}</span>
                  <div style={{ flex: 1 }}>
                    <div style={{ fontSize: "13px", color: isSelected ? "#38bdf8" : "#E2E8F0", fontWeight: 500 }}>
                      {cmd.label}
                    </div>
                    <div style={{ fontSize: "10px", color: "#64748b", textTransform: "uppercase" }}>
                      {cmd.category}
                    </div>
                  </div>
                  {isSelected && (
                    <span style={{ fontSize: "11px", color: "#38bdf8" }}>↵ Select</span>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
};
