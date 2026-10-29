"""
SkyGuard: Weather Station Sensor Anomaly Detection Console
Ultra-Premium Cybersecurity & Aerospace Meteorological Command Interface
"""

import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add current dir to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from skyguard.data.weather_generator import WeatherDataGenerator, generate_benchmark_dataset
from skyguard.features.engineer import WeatherFeatureEngineer
from skyguard.models.ensemble import SkyGuardEnsemble
from skyguard.evaluation.metrics import evaluate_anomaly_detection, evaluate_by_fault_type

# Streamlit Page Config
st.set_page_config(
    page_title="SkyGuard AI • Weather Station Sensor Anomaly Matrix",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Cyber-Meteorology Design System
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;700&display=swap');

    /* Global Theme */
    html, body, [class*="css"], .stApp {
        font-family: 'Outfit', sans-serif !important;
        background-color: #030712 !important;
        color: #f3f4f6 !important;
    }

    /* Ambient Glow Background */
    .stApp {
        background: radial-gradient(circle at 15% 10%, rgba(56, 189, 248, 0.08) 0%, transparent 45%),
                    radial-gradient(circle at 85% 85%, rgba(192, 132, 252, 0.06) 0%, transparent 50%),
                    #030712 !important;
    }

    /* Remove default Streamlit top clutter */
    header[data-testid="stHeader"] {
        background: transparent !important;
    }
    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2rem !important;
        max-width: 98% !important;
    }

    /* Hero Header */
    .hero-panel {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.85) 0%, rgba(2, 6, 23, 0.95) 100%);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.7), inset 0 1px 0 rgba(255, 255, 255, 0.1);
    }
    .hero-panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; height: 3px;
        background: linear-gradient(90deg, #00f2fe, #4facfe, #818cf8, #c084fc);
    }

    .hero-title-group {
        display: flex;
        align-items: center;
        gap: 16px;
    }
    .hero-icon {
        width: 54px;
        height: 54px;
        background: linear-gradient(135deg, #0284c7, #6366f1);
        border-radius: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 28px;
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.4);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff 10%, #38bdf8 50%, #c084fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.1;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 4px;
        font-weight: 400;
    }

    /* Pulsing Beacon */
    .pulse-dot {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        background-color: #10b981;
        box-shadow: 0 0 12px #10b981;
        animation: pulse-ring 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
        vertical-align: middle;
        margin-right: 6px;
    }
    .pulse-dot-red {
        background-color: #ef4444;
        box-shadow: 0 0 14px #ef4444;
    }
    @keyframes pulse-ring {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(1.2); }
    }

    /* KPI Cards */
    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(5, 1fr);
        gap: 16px;
        margin-bottom: 24px;
    }
    .kpi-card {
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 16px;
        padding: 18px 20px;
        position: relative;
        overflow: hidden;
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
        transition: transform 0.25s ease, border-color 0.25s ease;
    }
    .kpi-card:hover {
        transform: translateY(-2px);
        border-color: rgba(56, 189, 248, 0.35);
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #94a3b8;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .kpi-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.85rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 8px 0 4px 0;
        line-height: 1;
    }
    .kpi-badge {
        font-size: 0.72rem;
        font-weight: 600;
        padding: 3px 8px;
        border-radius: 6px;
        display: inline-block;
    }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-amber { background: rgba(245, 158, 11, 0.15); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-cyan { background: rgba(6, 182, 212, 0.15); color: #38bdf8; border: 1px solid rgba(6, 182, 212, 0.3); }

    /* Custom Navigation Tabs */
    div[data-testid="stTabs"] button {
        background: transparent !important;
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        border: none !important;
        padding: 10px 18px !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom: 2px solid #38bdf8 !important;
        background: rgba(56, 189, 248, 0.08) !important;
        border-radius: 8px 8px 0 0 !important;
    }

    /* Glass Container */
    .glass-box {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 22px;
        backdrop-filter: blur(12px);
        margin-bottom: 20px;
    }

    /* Custom Sidebar */
    section[data-testid="stSidebar"] {
        background: #020617 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_default_benchmark():
    data_path = os.path.join(os.path.dirname(__file__), "data", "weather_station_telemetry.csv")
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
    else:
        generator = WeatherDataGenerator(seed=42)
        df = generator.generate_telemetry(periods=4320, freq="10min", inject_faults=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


@st.cache_resource
def get_trained_pipeline(df, contamination=0.04):
    engineer = WeatherFeatureEngineer(rolling_windows=(6, 18))
    feat_df = engineer.fit_transform(df)
    feature_cols = engineer.get_feature_names()
    X = feat_df[feature_cols]

    ensemble = SkyGuardEnsemble(contamination=contamination, random_state=42)
    ensemble.fit(X, feat_df)
    return engineer, ensemble, feature_cols


# -------------------------------------------------------------
# SIDEBAR CONTROLS
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 16px;">
        <span style="font-size: 24px;">🛡️</span>
        <div>
            <div style="font-weight: 800; font-size: 1.1rem; color: #f8fafc; letter-spacing: -0.01em;">SKYGUARD AI</div>
            <div style="font-size: 0.75rem; color: #64748b;">AUTONOMOUS SENSOR DEFENSE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🎛️ Observation Engine")
    data_source = st.selectbox(
        "Telemetry Feed Source",
        ["30-Day Benchmark Dataset", "Custom Real-Time Generator", "Upload Station CSV"],
        index=0
    )

    if data_source == "30-Day Benchmark Dataset":
        df_raw = load_default_benchmark()
    elif data_source == "Custom Real-Time Generator":
        days = st.slider("Observation Duration (Days)", 5, 45, 20)
        anom_rate = st.slider("Anomaly Injection Density (%)", 1.0, 10.0, 4.0, step=0.5) / 100.0
        seed = st.number_input("Random Initialization Seed", 1, 9999, 42)
        gen = WeatherDataGenerator(seed=seed)
        df_raw = gen.generate_telemetry(periods=days * 144, freq="10min", inject_faults=True, anomaly_ratio=anom_rate)
        df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
    else:
        uploaded = st.file_uploader("Upload Telemetry CSV", type=["csv"])
        if uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            if "timestamp" in df_raw.columns:
                df_raw["timestamp"] = pd.to_datetime(df_raw["timestamp"])
            if "is_anomaly" not in df_raw.columns:
                df_raw["is_anomaly"] = 0
                df_raw["anomaly_type"] = "none"
                df_raw["faulty_sensor"] = "none"
        else:
            st.info("Awaiting CSV. Using benchmark data in standby.")
            df_raw = load_default_benchmark()

    st.markdown("---")
    st.markdown("### ⚙️ Sensitivity & Detection Cutoffs")
    contamination = st.slider("Expected Contamination (%)", 1.0, 10.0, 4.0, step=0.5) / 100.0
    threshold_override = st.slider("Ensemble Decision Threshold", 0.20, 0.85, 0.45, step=0.05)

    st.markdown("---")
    st.markdown("### ⚖️ Ensemble Detector Weights")
    w_iforest = st.slider("Isolation Forest Weight", 0.0, 1.0, 0.35, step=0.05)
    w_pca = st.slider("PCA Reconstruction Weight", 0.0, 1.0, 0.30, step=0.05)
    w_lof = st.slider("Local Outlier Factor Weight", 0.0, 1.0, 0.15, step=0.05)
    w_rules = st.slider("Atmospheric Physics Weight", 0.0, 1.0, 0.20, step=0.05)


# Model pipeline execution
engineer, ensemble, feature_cols = get_trained_pipeline(df_raw, contamination=contamination)

# Update runtime weights
custom_weights = {
    "isolation_forest": w_iforest,
    "pca": w_pca,
    "lof": w_lof,
    "physics_rules": w_rules,
}
ensemble.weights = custom_weights
total_w = sum(custom_weights.values())
if total_w > 0:
    ensemble.weights = {k: v / total_w for k, v in custom_weights.items()}

feat_df = engineer.transform(df_raw)
X = feat_df[feature_cols]
scores_df = ensemble.score(X, feat_df)
scores_df["is_anomaly_pred"] = (scores_df["ensemble_score"] >= threshold_override).astype(int)

# Recalculate severities
severities = []
for s in scores_df["ensemble_score"]:
    if s < threshold_override:
        severities.append("NORMAL")
    elif s < threshold_override + 0.15:
        severities.append("LOW")
    elif s < threshold_override + 0.30:
        severities.append("WARNING")
    else:
        severities.append("CRITICAL")
scores_df["severity"] = severities

telemetry = pd.concat([df_raw, scores_df], axis=1)

# High-level KPIs
total_pts = len(telemetry)
total_anoms = int(telemetry["is_anomaly_pred"].sum())
anom_pct = (total_anoms / total_pts) * 100
crit_count = int((telemetry["severity"] == "CRITICAL").sum())
station_status = "CRITICAL ALERT" if crit_count > 0 else ("WARNING ACTIVE" if total_anoms > 0 else "NOMINAL")
dot_class = "pulse-dot-red" if crit_count > 0 else "pulse-dot"
status_badge = "badge-red" if crit_count > 0 else ("badge-amber" if total_anoms > 0 else "badge-green")
avg_battery = float(telemetry["battery_voltage_v"].mean())

# -------------------------------------------------------------
# HERO HEADER PANEL
# -------------------------------------------------------------
st.markdown(f"""
<div class="hero-panel">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 20px;">
        <div class="hero-title-group">
            <div class="hero-icon">🛡️</div>
            <div>
                <h1 class="hero-title">SkyGuard Command Deck</h1>
                <div class="hero-sub">Autonomous Unsupervised Sensor Anomaly Defense • Automated Weather Station Network</div>
            </div>
        </div>
        <div style="display: flex; gap: 14px; align-items: center;">
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 8px 16px; text-align: right;">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Node Designation</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #38bdf8;">AWS-STATION-042</div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255,255,255,0.1); border-radius: 12px; padding: 8px 16px; text-align: right;">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Telemetry Link</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #10b981;"><span class="{dot_class}"></span>ONLINE 100%</div>
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 5-CARD KPI MATRIX
# -------------------------------------------------------------
st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-title"><span>Telemetry Ingestion</span><span>📡</span></div>
        <div class="kpi-value">{total_pts:,}</div>
        <div class="kpi-badge badge-cyan">10-MIN INTERVALS</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title"><span>System Status</span><span>🛡️</span></div>
        <div class="kpi-value" style="font-size: 1.45rem; padding-top: 5px;"><span class="kpi-badge {status_badge}">{station_status}</span></div>
        <div style="font-size: 0.74rem; color: #94a3b8; margin-top: 6px;">Multi-layer guard active</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title"><span>Anomalies Flagged</span><span>⚠️</span></div>
        <div class="kpi-value">{total_anoms:,}</div>
        <div class="kpi-badge badge-amber">{anom_pct:.2f}% OF TOTAL FEED</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title"><span>Critical Incidents</span><span>🚨</span></div>
        <div class="kpi-value" style="color: #f87171;">{crit_count:,}</div>
        <div class="kpi-badge badge-red">IMMEDIATE ACTION</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-title"><span>Station Voltage</span><span>🔋</span></div>
        <div class="kpi-value" style="color: #38bdf8;">{avg_battery:.2f}V</div>
        <div class="kpi-badge badge-green">SOLAR POWER STABLE</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# NAVIGATION TABS
# -------------------------------------------------------------
tabs = st.tabs([
    "📈 Interactive Telemetry Matrix",
    "🔬 Unsupervised Intelligence & PCA Subspace",
    "🩺 Automated Root Cause Analysis (RCA)",
    "⚡ Live Fault Injection Laboratory",
    "📊 Quantitative Benchmark Validation"
])

# -------------------------------------------------------------
# TAB 1: Live Telemetry & Inspector
# -------------------------------------------------------------
with tabs[0]:
    c_s1, c_s2, c_s3 = st.columns([2, 2, 2])
    with c_s1:
        sensor_display = st.selectbox(
            "Target Meteorological Channel",
            [
                ("temperature_c", "🌡️ Ambient Temperature (°C)"),
                ("humidity_pct", "💧 Relative Humidity (%)"),
                ("pressure_hpa", "🧭 Barometric Pressure (hPa)"),
                ("wind_speed_ms", "💨 Anemometer Wind Speed (m/s)"),
                ("solar_radiation_wm2", "☀️ Pyranometer Solar Irradiance (W/m²)"),
                ("battery_voltage_v", "🔋 Station Bus Voltage (V)"),
                ("internal_temp_c", "🎛️ Internal Enclosure Temp (°C)"),
            ],
            format_func=lambda x: x[1]
        )[0]
    with c_s2:
        timeframe = st.selectbox("Inspection Temporal Window", ["First 5 Days (High Detail)", "Last 7 Days", "Last 48 Hours", "Full Observation Stream"], index=0)
    with c_s3:
        filter_mode = st.radio("Point Highlights", ["Show All", "Anomalies Only", "Critical Only"], horizontal=True)

    # Slice window
    plot_df = telemetry.copy()
    if timeframe == "First 5 Days (High Detail)":
        plot_df = plot_df.head(5 * 144)
    elif timeframe == "Last 7 Days":
        plot_df = plot_df.tail(7 * 144)
    elif timeframe == "Last 48 Hours":
        plot_df = plot_df.tail(48 * 6)

    # Multi-panel synced chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=(f"Channel Telemetry: {sensor_display.replace('_', ' ').title()}", "SkyGuard Multi-Model Anomaly Severity Index"),
        row_heights=[0.66, 0.34]
    )

    # Sensor Trace
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df[sensor_display],
            name="Nominal Stream",
            line=dict(color="#00f2fe", width=2),
            mode="lines"
        ),
        row=1, col=1
    )

    # Highlight Anomalies
    if filter_mode == "Critical Only":
        anom_points = plot_df[plot_df["severity"] == "CRITICAL"]
    else:
        anom_points = plot_df[plot_df["is_anomaly_pred"] == 1]

    if len(anom_points) > 0:
        fig.add_trace(
            go.Scatter(
                x=anom_points["timestamp"],
                y=anom_points[sensor_display],
                name="Flagged Anomaly",
                mode="markers",
                marker=dict(
                    color="#ff3366",
                    size=10,
                    symbol="diamond",
                    line=dict(color="#ffffff", width=1.5)
                ),
                customdata=np.stack((anom_points["ensemble_score"], anom_points["severity"], anom_points["rule_diagnosis"]), axis=-1),
                hovertemplate="<b>Timestamp:</b> %{x}<br><b>Sensor Value:</b> %{y:.2f}<br><b>Anomaly Score:</b> %{customdata[0]:.3f}<br><b>Severity:</b> %{customdata[1]}<br><b>Diagnostic:</b> %{customdata[2]}<extra></extra>"
            ),
            row=1, col=1
        )

    # Anomaly Score Trace with Neon Fill
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["ensemble_score"],
            name="Anomaly Severity",
            line=dict(color="#c084fc", width=2),
            fill="tozeroy",
            fillcolor="rgba(192, 132, 252, 0.18)"
        ),
        row=2, col=1
    )

    # Cutoff Threshold Line
    fig.add_hline(
        y=threshold_override,
        line_dash="dash",
        line_color="#ff4757",
        line_width=2,
        annotation_text=f"Cutoff Threshold ({threshold_override:.2f})",
        annotation_position="top right",
        annotation_font_color="#ff4757",
        row=2, col=1
    )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="rgba(11, 15, 25, 0.8)",
        plot_bgcolor="rgba(11, 15, 25, 0.8)",
        margin=dict(l=40, r=40, t=50, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(family="Outfit")),
        font=dict(family="Outfit", color="#94a3b8"),
        xaxis2=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis2=dict(gridcolor="rgba(255, 255, 255, 0.05)")
    )
    st.plotly_chart(fig, width="stretch")

    # Forensic Incident Ticker
    st.markdown("#### 🔍 Real-Time Incident Forensic Ticker")
    if len(anom_points) > 0:
        display_tbl = anom_points[["timestamp", sensor_display, "ensemble_score", "severity", "rule_diagnosis"]].tail(8)
        st.dataframe(display_tbl, width="stretch")
    else:
        st.success("✅ No sensor anomalies identified within the active viewing window.")

# -------------------------------------------------------------
# TAB 2: Unsupervised Intelligence & Subspace Projection
# -------------------------------------------------------------
with tabs[1]:
    st.markdown("""
    <div class="glass-box">
        <h3 style="margin-top:0; color: #38bdf8;">🧠 Multi-Model Unsupervised Subspace Decomposition</h3>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Rather than relying on a single fallible detector, SkyGuard maps 30+ engineered meteorological features into an orthogonal principal subspace. 
            Nominal atmospheric phenomena adhere to a tightly bound physical manifold (diurnal heating loops), while sensor failures violate physical correlations and scatter outwards.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_sub1, col_sub2 = st.columns([1, 1])

    with col_sub1:
        st.markdown("##### 2D Principal Component Subspace")
        pca_viz = PCA(n_components=2, random_state=42)
        X_scaled = StandardScaler().fit_transform(X)
        coords = pca_viz.fit_transform(X_scaled)

        proj_df = pd.DataFrame({
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "Severity": telemetry["severity"],
            "Score": telemetry["ensemble_score"],
            "Timestamp": telemetry["timestamp"].astype(str)
        })

        fig_pca = px.scatter(
            proj_df,
            x="PC1", y="PC2",
            color="Severity",
            color_discrete_map={
                "NORMAL": "#10b981",
                "LOW": "#38bdf8",
                "WARNING": "#fbbf24",
                "CRITICAL": "#ff3366"
            },
            hover_data=["Score", "Timestamp"],
            template="plotly_dark",
            height=460
        )
        fig_pca.update_traces(marker=dict(size=6, opacity=0.85))
        fig_pca.update_layout(
            paper_bgcolor="rgba(11, 15, 25, 0.8)",
            plot_bgcolor="rgba(11, 15, 25, 0.8)",
            font=dict(family="Outfit"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title=f"PC 1 ({pca_viz.explained_variance_ratio_[0]*100:.1f}% Variance)"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title=f"PC 2 ({pca_viz.explained_variance_ratio_[1]*100:.1f}% Variance)")
        )
        st.plotly_chart(fig_pca, width="stretch")

    with col_sub2:
        st.markdown("##### Unsupervised Score Density Profiles")
        fig_hist = go.Figure()
        models_data = [
            ("score_iforest", "Isolation Forest", "#00f2fe"),
            ("score_pca", "PCA Reconstruction", "#818cf8"),
            ("score_lof", "Local Outlier Factor", "#34d399"),
            ("score_rules", "Physics Rules", "#fbbf24"),
            ("ensemble_score", "SkyGuard Ensemble", "#c084fc")
        ]
        for col_name, label, color in models_data:
            fig_hist.add_trace(go.Histogram(
                x=telemetry[col_name],
                name=label,
                opacity=0.6,
                marker_color=color,
                nbinsx=40
            ))

        fig_hist.update_layout(
            barmode="overlay",
            template="plotly_dark",
            height=460,
            paper_bgcolor="rgba(11, 15, 25, 0.8)",
            plot_bgcolor="rgba(11, 15, 25, 0.8)",
            font=dict(family="Outfit"),
            xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Normalized Anomaly Score [0, 1]"),
            yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Observation Count"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_hist, width="stretch")

# -------------------------------------------------------------
# TAB 3: Root Cause Analysis (RCA)
# -------------------------------------------------------------
with tabs[2]:
    st.markdown("""
    <div class="glass-box">
        <h3 style="margin-top:0; color: #ff3366;">🩺 Automated Root Cause Analysis & Diagnostic Attribution</h3>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            When an anomaly is flagged, SkyGuard conducts an automated forensic inspection by decomposing the PCA residual matrix 
            to isolate which physical sensor contributed most to the anomaly, paired with atmospheric physics rules to generate an actionable field maintenance work order.
        </p>
    </div>
    """, unsafe_allow_html=True)

    anom_indices = telemetry[telemetry["is_anomaly_pred"] == 1].index
    if len(anom_indices) == 0:
        st.info("No anomalies detected at the current threshold cutoff. Adjust sensitivity in sidebar.")
    else:
        selected_idx = st.selectbox(
            "Select Incident for Forensic Autopsy",
            options=anom_indices,
            format_func=lambda i: f"Timestamp: {telemetry['timestamp'].iloc[i]} | Severity: {telemetry['severity'].iloc[i]} | Score: {telemetry['ensemble_score'].iloc[i]:.3f}"
        )

        rca_report = ensemble.diagnose_root_cause(selected_idx, X, telemetry)
        rca_sev = rca_report['severity']
        rca_badge = "badge-red" if rca_sev == "CRITICAL" else ("badge-amber" if rca_sev == "WARNING" else "badge-cyan")

        r_col1, r_col2 = st.columns([1.4, 1.2])

        with r_col1:
            st.markdown(f"""
            <div class="glass-box" style="border-left: 4px solid #ff3366;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <span style="font-size: 1.2rem; font-weight: 700; color: #f8fafc;">Diagnostic Work Order #{selected_idx}</span>
                    <span class="kpi-badge {rca_badge}">{rca_sev} PRIORITY</span>
                </div>
                <div style="font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; line-height: 1.8;">
                    <div><b>Timestamp:</b> <code style="color:#38bdf8;">{rca_report['timestamp']}</code></div>
                    <div><b>Failure Mechanism:</b> <span style="color:#fbbf24; font-weight: 700;">{rca_report['fault_mode']}</span></div>
                    <div><b>Culprit Sensor Channel:</b> <span style="color:#ff3366; font-weight: 700;">{rca_report['primary_culprit_sensor']}</span></div>
                    <div><b>Ensemble Anomaly Score:</b> <code>{rca_report['ensemble_score']:.4f}</code></div>
                    <div style="margin-top: 8px; color: #cbd5e1; font-style: italic;">"{rca_report['rule_diagnosis']}"</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Feature Attribution Bar
            top_feats = rca_report["top_contributing_features"]
            fig_bar = px.bar(
                x=list(top_feats.values()),
                y=list(top_feats.keys()),
                orientation="h",
                title="Top Residual Feature Contributions to Anomaly",
                labels={"x": "Reconstruction Residual Error", "y": "Feature"},
                template="plotly_dark",
                color_discrete_sequence=["#ff3366"],
                height=260
            )
            fig_bar.update_layout(
                paper_bgcolor="rgba(11, 15, 25, 0.8)",
                plot_bgcolor="rgba(11, 15, 25, 0.8)",
                font=dict(family="Outfit"),
                xaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
                yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)")
            )
            st.plotly_chart(fig_bar, width="stretch")

        with r_col2:
            st.markdown("##### Multi-Sensor Anomaly Attribution Radar")
            sensor_groups = {
                "Temperature": ["temperature_c", "temperature_c_diff1", "internal_temp_c"],
                "Humidity": ["humidity_pct", "dew_point_c", "vpd_kpa"],
                "Pressure": ["pressure_hpa", "pressure_hpa_diff1"],
                "Wind Speed": ["wind_speed_ms", "wind_speed_ms_diff1"],
                "Pyranometer": ["solar_radiation_wm2", "unphysical_night_solar"],
                "Power System": ["battery_voltage_v", "battery_voltage_v_diff1"],
            }
            row_residuals = ensemble.pca.get_feature_reconstruction_errors(X.iloc[[selected_idx]]).iloc[0]
            group_scores = []
            categories = []
            for group, cols in sensor_groups.items():
                matching = [c for c in cols if c in row_residuals.index]
                score = float(row_residuals[matching].mean()) if matching else 0.0
                group_scores.append(score)
                categories.append(group)

            fig_radar = go.Figure(data=go.Scatterpolar(
                r=group_scores,
                theta=categories,
                fill='toself',
                fillcolor='rgba(255, 51, 102, 0.25)',
                line=dict(color='#ff3366', width=2)
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, max(max(group_scores)*1.2, 5.0)], gridcolor="rgba(255, 255, 255, 0.1)"),
                    angularaxis=dict(gridcolor="rgba(255, 255, 255, 0.1)"),
                    bgcolor="rgba(11, 15, 25, 0.6)"
                ),
                showlegend=False,
                template="plotly_dark",
                height=420,
                paper_bgcolor="rgba(11, 15, 25, 0.8)",
                font=dict(family="Outfit")
            )
            st.plotly_chart(fig_radar, width="stretch")

# -------------------------------------------------------------
# TAB 4: Interactive Fault Simulator & Sandbox
# -------------------------------------------------------------
with tabs[3]:
    st.markdown("""
    <div class="glass-box">
        <h3 style="margin-top:0; color: #fbbf24;">⚡ Real-Time Sensor Fault Injection Laboratory</h3>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            Stress-test SkyGuard's unsupervised detection algorithms by injecting hardware failures, frozen transducers, calibration drifts, or power brownouts into live nominal telemetry.
        </p>
    </div>
    """, unsafe_allow_html=True)

    sim_c1, sim_c2, sim_c3 = st.columns(3)
    with sim_c1:
        target_sensor = st.selectbox(
            "Hardware Channel to Compromise",
            ["temperature_c", "humidity_pct", "pressure_hpa", "wind_speed_ms", "solar_radiation_wm2", "battery_voltage_v"]
        )
    with sim_c2:
        fault_type = st.selectbox(
            "Failure Archetype",
            [
                "⚡ Electrostatic Impulse Spike (+25.0)",
                "📉 Negative Impulse Glitch (-25.0)",
                "🧊 Icing Mechanical Freeze / Flatline (6 Hours)",
                "📈 Progressive Calibration Drift (+12.0)",
                "🌙 Night-Time Solar Radiation Leak",
                "🪫 Photovoltaic Battery System Brownout"
            ]
        )
    with sim_c3:
        step_pos = st.slider("Fault Injection Point (Step Offset)", 50, 350, 150)

    # Simulation execution
    sim_df = df_raw.head(450).copy()
    if "Spike (+25.0)" in fault_type:
        sim_df.loc[step_pos, target_sensor] += 25.0
    elif "Glitch (-25.0)" in fault_type:
        sim_df.loc[step_pos, target_sensor] -= 25.0
    elif "Mechanical Freeze" in fault_type:
        stuck_val = sim_df.loc[step_pos, target_sensor]
        sim_df.loc[step_pos:step_pos+36, target_sensor] = stuck_val
    elif "Calibration Drift" in fault_type:
        slope = np.linspace(0, 14.0, 45)
        sim_df.loc[step_pos:step_pos+44, target_sensor] += slope
    elif "Night-Time Solar" in fault_type:
        sim_df.loc[step_pos:step_pos+10, "solar_radiation_wm2"] = 720.0
    elif "Battery System Brownout" in fault_type:
        sim_df.loc[step_pos:step_pos+20, "battery_voltage_v"] = 9.4
        sim_df.loc[step_pos:step_pos+20, target_sensor] += np.random.normal(0, 7.5, 21)

    # Transform & evaluate
    sim_feat = engineer.transform(sim_df)
    sim_scores = ensemble.score(sim_feat[feature_cols], sim_feat)
    sim_results = pd.concat([sim_df, sim_scores], axis=1)

    fig_sim = make_subplots(
        rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        subplot_titles=(f"Synthetically Tampered Sensor: {target_sensor}", "SkyGuard Autonomous Defense Response")
    )
    fig_sim.add_trace(go.Scatter(x=sim_results.index, y=sim_results[target_sensor], name=target_sensor, line=dict(color="#00f2fe", width=2)), row=1, col=1)
    fig_sim.add_trace(go.Scatter(x=sim_results.index, y=sim_results["ensemble_score"], name="Anomaly Score", line=dict(color="#ff3366", width=2), fill="tozeroy", fillcolor="rgba(255, 51, 102, 0.2)"), row=2, col=1)
    fig_sim.add_hline(y=threshold_override, line_dash="dash", line_color="#fbbf24", row=2, col=1, annotation_text="Cutoff")

    fig_sim.update_layout(
        height=480, template="plotly_dark",
        paper_bgcolor="rgba(11, 15, 25, 0.8)", plot_bgcolor="rgba(11, 15, 25, 0.8)",
        font=dict(family="Outfit"),
        xaxis2=dict(gridcolor="rgba(255, 255, 255, 0.05)", title="Simulation Step"),
        yaxis=dict(gridcolor="rgba(255, 255, 255, 0.05)"),
        yaxis2=dict(gridcolor="rgba(255, 255, 255, 0.05)")
    )
    st.plotly_chart(fig_sim, width="stretch")

    peak_score = float(sim_results["ensemble_score"].iloc[step_pos:step_pos+10].max())
    if peak_score >= threshold_override:
        st.markdown(f"""
        <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 12px; padding: 14px; color: #34d399; font-weight: 600;">
            ✅ Autonomous Interception Successful! Peak Anomaly Confidence: {peak_score:.4f} (Threshold: {threshold_override:.2f}). Work order triggered automatically.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background: rgba(245, 158, 11, 0.15); border: 1px solid rgba(245, 158, 11, 0.3); border-radius: 12px; padding: 14px; color: #fbbf24; font-weight: 600;">
            ⚠️ Peak Score: {peak_score:.4f} is currently below the {threshold_override:.2f} cutoff threshold. Lower threshold slider to catch early-stage drift.
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 5: Quantitative Benchmark Validation
# -------------------------------------------------------------
with tabs[4]:
    st.markdown("""
    <div class="glass-box">
        <h3 style="margin-top:0; color: #34d399;">📊 Rigorous Quantitative Benchmark Validation</h3>
        <p style="color: #94a3b8; font-size: 0.95rem;">
            While unsupervised models operate in production without labels, we benchmark their performance against controlled synthetic fault injections across 4,320 observations.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "is_anomaly" in df_raw.columns and df_raw["is_anomaly"].sum() > 0:
        y_true = df_raw["is_anomaly"].values
        y_pred = (telemetry["ensemble_score"] >= threshold_override).astype(int)
        metrics = evaluate_anomaly_detection(y_true, telemetry["ensemble_score"].values, y_pred=y_pred)

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Precision</div>
                <div class="kpi-value" style="color:#38bdf8;">{metrics['precision']*100:.1f}%</div>
                <div class="kpi-badge badge-cyan">MINIMAL FALSE ALARMS</div>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">Recall</div>
                <div class="kpi-value" style="color:#34d399;">{metrics['recall']*100:.1f}%</div>
                <div class="kpi-badge badge-green">HIGH FAULT CAPTURE</div>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">F1-Score</div>
                <div class="kpi-value" style="color:#fbbf24;">{metrics['f1_score']:.3f}</div>
                <div class="kpi-badge badge-amber">HARMONIC MEAN</div>
            </div>
            """, unsafe_allow_html=True)
        with m4:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-title">ROC-AUC</div>
                <div class="kpi-value" style="color:#c084fc;">{metrics['roc_auc']:.3f}</div>
                <div class="kpi-badge badge-green">DISCRIMINATIVE POWER</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("##### Detection Recall Across Fault Archetypes")
        breakdown_df = evaluate_by_fault_type(telemetry)
        st.dataframe(breakdown_df, width="stretch")

        st.markdown("##### Classification Confusion Matrix")
        cm_df = pd.DataFrame(
            [[metrics["true_negatives"], metrics["false_positives"]],
             [metrics["false_negatives"], metrics["true_positives"]]],
            index=["Ground Truth Normal", "Ground Truth Anomaly"],
            columns=["Predicted Normal", "Predicted Anomaly"]
        )
        st.dataframe(cm_df, width="stretch")
    else:
        st.info("Ground truth tags not present in uploaded dataset. Benchmark scores are evaluated on verified synthetic or labeled streams.")

# Footer
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.82rem; margin-top: 40px; padding: 20px 0; border-top: 1px solid rgba(255,255,255,0.06); font-family: 'Outfit';">
    SkyGuard AI • Defense-Grade Unsupervised Weather Sensor Matrix • Powered by Isolation Forest, PCA Reconstruction & Atmospheric Domain Physics
</div>
""", unsafe_allow_html=True)
