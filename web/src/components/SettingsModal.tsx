import React, { useState } from "react";
import { SystemSettings } from "../types";

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  settings: SystemSettings;
  onUpdateSettings: (newSettings: Partial<SystemSettings>) => Promise<void>;
}

export const SettingsModal: React.FC<SettingsModalProps> = ({
  isOpen,
  onClose,
  settings,
  onUpdateSettings,
}) => {
  const [localSensitivity, setLocalSensitivity] = useState<number>(settings.sensitivity);
  const [isSaving, setIsSaving] = useState<boolean>(false);

  if (!isOpen) return null;

  const estimated24h = Math.max(1, Math.round(8.0 * Math.pow(localSensitivity, 1.6)));

  const handleSensitivityCommit = async (val: number) => {
    setIsSaving(true);
    await onUpdateSettings({ sensitivity: val });
    setIsSaving(false);
  };

  const handleToggleModel = async (modelKey: string, currentVal: boolean) => {
    await onUpdateSettings({
      models: {
        ...settings.models,
        [modelKey]: !currentVal,
      },
    });
  };

  return (
    <div
      className="modal-backdrop"
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 9999,
        background: "rgba(10, 15, 29, 0.75)",
        backdropFilter: "blur(6px)",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
      }}
      onClick={onClose}
    >
      <div
        className="settings-modal"
        style={{
          width: "480px",
          maxWidth: "92vw",
          background: "rgba(15, 23, 42, 0.96)",
          border: "1px solid rgba(255, 255, 255, 0.12)",
          borderRadius: "12px",
          boxShadow: "0 20px 48px rgba(0, 0, 0, 0.8)",
          padding: "24px",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "20px" }}>
          <h3 style={{ margin: 0, fontSize: "18px", color: "#F8FAFC", display: "flex", alignItems: "center", gap: "8px" }}>
            <span>⚙️</span> Detection Engine & Operational Settings
          </h3>
          <button
            onClick={onClose}
            style={{
              background: "none",
              border: "none",
              color: "#94a3b8",
              fontSize: "18px",
              cursor: "pointer",
            }}
          >
            ✕
          </button>
        </div>

        {/* Sensitivity Slider with Live Preview of Estimated Incidents / 24h */}
        <div style={{ marginBottom: "20px", background: "rgba(30, 41, 59, 0.4)", padding: "14px", borderRadius: "8px", border: "1px solid rgba(255, 255, 255, 0.06)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "8px" }}>
            <span style={{ fontSize: "13px", color: "#E2E8F0", fontWeight: 600 }}>Anomaly Sensitivity (Dynamic Threshold)</span>
            <span style={{ fontSize: "13px", color: "#38bdf8", fontFamily: "JetBrains Mono", fontWeight: 700 }}>
              {localSensitivity.toFixed(1)}×
            </span>
          </div>

          <input
            type="range"
            min="0.5"
            max="3.0"
            step="0.1"
            value={localSensitivity}
            onChange={(e) => setLocalSensitivity(parseFloat(e.target.value))}
            onMouseUp={() => handleSensitivityCommit(localSensitivity)}
            onTouchEnd={() => handleSensitivityCommit(localSensitivity)}
            style={{ width: "100%", accentColor: "#38bdf8", cursor: "pointer" }}
          />

          <div style={{ display: "flex", justifyContent: "space-between", fontSize: "11px", color: "#64748b", marginTop: "4px" }}>
            <span>0.5 (Conservative)</span>
            <span>1.0 (Balanced)</span>
            <span>3.0 (Aggressive)</span>
          </div>

          {/* Live Preview Pill */}
          <div
            style={{
              marginTop: "10px",
              padding: "8px 12px",
              borderRadius: "6px",
              background: "rgba(56, 189, 248, 0.12)",
              border: "1px solid rgba(56, 189, 248, 0.25)",
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              fontSize: "12px",
            }}
          >
            <span style={{ color: "#94a3b8" }}>Estimated Anomaly Rate:</span>
            <span style={{ color: "#38bdf8", fontWeight: 700, fontFamily: "JetBrains Mono" }}>
              ~{estimated24h} incidents / 24h {isSaving && "(updating...)"}
            </span>
          </div>
        </div>

        {/* Ensemble Model Toggles */}
        <div style={{ marginBottom: "20px" }}>
          <div style={{ fontSize: "13px", color: "#E2E8F0", fontWeight: 600, marginBottom: "10px" }}>
            Unsupervised Model Ensemble Weights
          </div>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "8px" }}>
            {[
              { id: "isolation_forest", label: "Isolation Forest" },
              { id: "autoencoder", label: "Conv1D Autoencoder" },
              { id: "statistical", label: "Robust Z-Score / MAD" },
              { id: "lof", label: "Local Outlier Factor" },
            ].map((m) => {
              const isChecked = settings.models[m.id as keyof typeof settings.models] ?? true;
              return (
                <label
                  key={m.id}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    padding: "8px 10px",
                    borderRadius: "6px",
                    background: "rgba(30, 41, 59, 0.4)",
                    border: "1px solid rgba(255, 255, 255, 0.06)",
                    cursor: "pointer",
                    fontSize: "12px",
                    color: isChecked ? "#E2E8F0" : "#64748b",
                  }}
                >
                  <input
                    type="checkbox"
                    checked={isChecked}
                    onChange={() => handleToggleModel(m.id, isChecked)}
                    style={{ accentColor: "#38bdf8" }}
                  />
                  <span>{m.label}</span>
                </label>
              );
            })}
          </div>
        </div>

        {/* UI / Accessibility Preferences */}
        <div style={{ borderTop: "1px solid rgba(255, 255, 255, 0.08)", paddingTop: "16px", display: "flex", flexDirection: "column", gap: "10px" }}>
          {/* Theme */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <span style={{ fontSize: "13px", color: "#E2E8F0" }}>Console Theme</span>
            <div style={{ display: "flex", gap: "4px" }}>
              {(["dark", "light"] as const).map((t) => (
                <button
                  key={t}
                  onClick={() => onUpdateSettings({ theme: t })}
                  style={{
                    padding: "4px 10px",
                    borderRadius: "4px",
                    fontSize: "11px",
                    cursor: "pointer",
                    background: settings.theme === t ? "rgba(56, 189, 248, 0.25)" : "rgba(30, 41, 59, 0.5)",
                    border: settings.theme === t ? "1px solid #38bdf8" : "1px solid rgba(255, 255, 255, 0.08)",
                    color: settings.theme === t ? "#38bdf8" : "#94a3b8",
                  }}
                >
                  {t.toUpperCase()}
                </button>
              ))}
            </div>
          </div>

          {/* Reduced Motion */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: "13px", color: "#E2E8F0" }}>Reduced Motion (WCAG)</div>
              <div style={{ fontSize: "11px", color: "#64748b" }}>Disables camera orbit & animation transitions</div>
            </div>
            <input
              type="checkbox"
              checked={settings.reduced_motion}
              onChange={(e) => onUpdateSettings({ reduced_motion: e.target.checked })}
              style={{ width: "16px", height: "16px", accentColor: "#38bdf8", cursor: "pointer" }}
            />
          </div>

          {/* Sound Alert Toggle */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <div>
              <div style={{ fontSize: "13px", color: "#E2E8F0" }}>Audio Chime on Critical Alerts</div>
              <div style={{ fontSize: "11px", color: "#64748b" }}>Synthesizes alert chime via Web Audio API</div>
            </div>
            <input
              type="checkbox"
              checked={settings.sound_enabled}
              onChange={(e) => onUpdateSettings({ sound_enabled: e.target.checked })}
              style={{ width: "16px", height: "16px", accentColor: "#38bdf8", cursor: "pointer" }}
            />
          </div>
        </div>

        <div style={{ marginTop: "24px", display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={onClose}
            style={{
              padding: "8px 18px",
              background: "#38bdf8",
              color: "#0F172A",
              fontWeight: 700,
              borderRadius: "6px",
              border: "none",
              cursor: "pointer",
              fontSize: "12px",
            }}
          >
            Done
          </button>
        </div>
      </div>
    </div>
  );
};
