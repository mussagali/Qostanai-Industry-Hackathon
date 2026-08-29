from __future__ import annotations

import html
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from simulator import (
    initial_line,
    simulate_step,
    load_config,
    spawn_vehicles,
    advance_vehicles,
    vehicles_at_risk,
    maintenance_window_note,
)
from prediction import train_models, score_line, explain_station, get_model_metrics

# =============================================================================
# PAGE CONFIG
# =============================================================================

st.set_page_config(
    page_title="DigitalTwin.ai — Vehicle Assembly Line",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# DESIGN TOKENS
# =============================================================================

BG_MAIN = "#06101C"
BG_SECONDARY = "#081624"
CARD_BG = "#0C1B2A"
CARD_ELEVATED = "#112438"
BORDER = "#1B344D"
DIVIDER = "#14283C"
TEXT_PRIMARY = "#F8FAFC"
TEXT_SECONDARY = "#B8C4D4"
TEXT_MUTED = "#71839A"
ACCENT_CYAN = "#19C6E6"
ACCENT_BLUE = "#3B82F6"
HEALTHY = "#22C55E"
WARNING = "#F59E0B"
CRITICAL = "#F04444"
MANUAL = "#64748B"

STATUS_COLOR = {
    "Normal": HEALTHY,
    "Warning": WARNING,
    "Critical": CRITICAL,
}

ZONE_LABELS = {
    "Body": "BODY SHOP",
    "Paint": "PAINT SHOP",
    "Final": "FINAL ASSEMBLY",
}
ZONE_ORDER = ["Body", "Paint", "Final"]


# =============================================================================
# GLOBAL CSS
# =============================================================================

def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        html, body, [class*="css"] {
            font-family: 'Inter', 'SF Pro Display', 'SF Pro Text', system-ui, sans-serif;
        }

        .stApp {
            background: #06101C;
        }

        section[data-testid="stSidebar"] {
            background: #081624;
            border-right: 1px solid #1B344D;
        }
        section[data-testid="stSidebar"] .block-container {
            padding-top: 1.4rem;
        }

        #MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; height: 0; }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.4rem;
            max-width: 1400px;
        }

        h1, h2, h3, h4, h5, h6 { color: #F8FAFC !important; }
        p, span, label, div { color: #B8C4D4; }

        hr {
            border: none;
            border-top: 1px solid #14283C;
            margin: 20px 0;
        }

        /* ---- Buttons ---- */
        .stButton > button {
            background: transparent;
            color: #B8C4D4;
            border: 1px solid #334155;
            border-radius: 8px;
            font-weight: 600;
            font-size: 13.5px;
            padding: 0.5rem 1rem;
            transition: all 0.15s ease;
        }
        .stButton > button:hover {
            border-color: #19C6E6;
            color: #19C6E6;
        }
        .stButton > button[kind="primary"] {
            background: #0891B2 !important;
            color: white !important;
            border: none !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #06B6D4 !important;
            color: white !important;
        }

        /* ---- Selectbox / inputs ---- */
        div[data-baseweb="select"] > div {
            background: #0C1B2A !important;
            border: 1px solid #1B344D !important;
            border-radius: 8px !important;
            color: #F8FAFC !important;
        }
        div[data-baseweb="popover"] { z-index: 999; }
        ul[role="listbox"] { background: #112438 !important; }

        /* ---- Tabs ---- */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            border-bottom: 1px solid #1B344D;
        }
        .stTabs [data-baseweb="tab"] {
            background: transparent;
            color: #71839A;
            font-weight: 600;
            font-size: 13.5px;
            letter-spacing: 0.3px;
            padding: 10px 16px;
            border-radius: 8px 8px 0 0;
        }
        .stTabs [aria-selected="true"] {
            color: #19C6E6 !important;
            background: #0C1B2A;
            border-bottom: 2px solid #19C6E6;
        }

        /* ---- Dataframes ---- */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1B344D;
            border-radius: 10px;
            overflow: hidden;
        }

        /* ---- Alerts (st.error/warning/info/success) ---- */
        div[data-testid="stAlert"] {
            background: #0C1B2A;
            border: 1px solid #1B344D;
            border-radius: 10px;
        }

        /* ---- Expander ---- */
        details {
            background: #0C1B2A;
            border: 1px solid #1B344D !important;
            border-radius: 10px !important;
        }
        summary {
            color: #F8FAFC !important;
            font-weight: 600;
        }

        /* ---- Metric (fallback, unused mostly) ---- */
        div[data-testid="stMetric"] {
            background: #0C1B2A;
            border: 1px solid #1B344D;
            border-radius: 12px;
            padding: 14px 16px;
        }

        /* =========================================================
           CUSTOM COMPONENTS
           ========================================================= */

        .dt-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            flex-wrap: wrap;
            gap: 12px;
            padding: 4px 0 18px 0;
        }
        .dt-title { font-size: 30px; font-weight: 700; color: #F8FAFC; line-height: 1.1; margin: 0; }
        .dt-subtitle { font-size: 13.5px; color: #71839A; margin-top: 2px; }
        .dt-header-right {
            display: flex; align-items: center; gap: 18px;
            font-size: 13px; color: #B8C4D4;
        }
        .dt-header-chip {
            color: #B8C4D4; font-weight: 500;
        }
        .dt-live {
            display: inline-flex; align-items: center; gap: 6px;
            color: #22C55E; font-weight: 700; font-size: 12px; letter-spacing: 0.5px;
        }
        .dt-live-dot {
            width: 8px; height: 8px; border-radius: 50%;
            background: #22C55E;
            box-shadow: 0 0 6px #22C55E;
        }

        .kpi-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 6px; }
        .kpi-card {
            flex: 1 1 170px;
            background: #0C1B2A;
            border: 1px solid #1B344D;
            border-radius: 12px;
            padding: 16px 18px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.25);
        }
        .kpi-label {
            font-size: 11px; font-weight: 600; letter-spacing: 0.6px;
            color: #71839A; text-transform: uppercase; margin-bottom: 8px;
        }
        .kpi-value { font-size: 28px; font-weight: 700; color: #F8FAFC; line-height: 1; }
        .kpi-value.accent { color: #19C6E6; }
        .kpi-value.healthy { color: #22C55E; }
        .kpi-value.warning { color: #F59E0B; }
        .kpi-value.critical { color: #F04444; }
        .kpi-unit { font-size: 12.5px; color: #71839A; margin-top: 4px; }

        .badge {
            display: inline-flex; align-items: center; gap: 5px;
            padding: 3px 10px; border-radius: 999px;
            font-size: 11px; font-weight: 700; letter-spacing: 0.4px;
            border: 1px solid; text-transform: uppercase;
        }
        .badge-healthy { color: #22C55E; border-color: #22C55E; background: rgba(34,197,94,0.10); }
        .badge-warning { color: #F59E0B; border-color: #F59E0B; background: rgba(245,158,11,0.10); }
        .badge-critical { color: #F04444; border-color: #F04444; background: rgba(239,68,68,0.10); }
        .badge-manual { color: #64748B; border-color: #64748B; background: rgba(100,116,139,0.10); }

        .section-header {
            font-size: 19px; font-weight: 600; color: #F8FAFC;
            margin: 4px 0 4px 0;
        }
        .section-sub { font-size: 12.5px; color: #71839A; margin-bottom: 14px; }

        .zone-block { margin-bottom: 22px; }
        .zone-title-row {
            display: flex; justify-content: space-between; align-items: baseline;
            margin-bottom: 10px;
        }
        .zone-title {
            font-size: 12px; font-weight: 700; letter-spacing: 0.8px;
            color: #B8C4D4; text-transform: uppercase;
        }
        .zone-meta { font-size: 11.5px; color: #71839A; }

        .zone-flow { display: flex; align-items: center; flex-wrap: wrap; gap: 0; }
        .connector { color: #334155; font-size: 15px; padding: 0 4px; user-select: none; }

        .station-node {
            background: #112438;
            border: 1px solid #1B344D;
            border-radius: 10px;
            padding: 8px 12px;
            min-width: 76px;
            text-align: center;
            margin: 4px 0;
        }
        .station-node.selected {
            border-color: #19C6E6;
            box-shadow: 0 0 0 1px #19C6E6;
        }
        .station-node .sid { font-size: 13px; font-weight: 700; color: #F8FAFC; }
        .station-node .stype { font-size: 9.5px; color: #64748B; letter-spacing: 0.4px; margin-top: 1px; }
        .station-node .sdot {
            width: 7px; height: 7px; border-radius: 50%;
            margin: 6px auto 4px auto;
        }
        .station-node .scycle { font-size: 11px; color: #71839A; }

        .risk-bar-wrap { margin-bottom: 12px; }
        .risk-bar-label {
            display: flex; justify-content: space-between;
            font-size: 12px; color: #71839A; margin-bottom: 5px;
        }
        .risk-bar-track {
            width: 100%; height: 7px; border-radius: 999px;
            background: #14283C; overflow: hidden;
        }
        .risk-bar-fill { height: 100%; border-radius: 999px; }

        .inspector-metric-row {
            display: flex; justify-content: space-between;
            padding: 7px 0; border-bottom: 1px solid #14283C;
            font-size: 13px;
        }
        .inspector-metric-row:last-child { border-bottom: none; }
        .inspector-metric-label { color: #71839A; }
        .inspector-metric-value { color: #F8FAFC; font-weight: 600; }

        .why-list { font-size: 13px; color: #B8C4D4; line-height: 1.85; margin: 6px 0 0 0; padding-left: 18px; }
        .action-box {
            background: #081624; border: 1px solid #1B344D; border-left: 3px solid #19C6E6;
            border-radius: 8px; padding: 10px 14px; font-size: 13px; color: #E2E8F0; margin-top: 10px;
        }

        .alert-card {
            background: #0C1B2A; border: 1px solid #1B344D;
            border-radius: 12px; padding: 16px 20px; margin-bottom: 14px;
        }
        .alert-card.critical { border-left: 3px solid #F04444; }
        .alert-card.warning { border-left: 3px solid #F59E0B; }
        .alert-head { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
        .alert-station { font-size: 15px; font-weight: 700; color: #F8FAFC; }

        .sensor-bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
        .sensor-bar-label { width: 62px; font-size: 11px; color: #71839A; text-transform: uppercase; letter-spacing: 0.4px; }
        .sensor-bar-track { flex: 1; height: 8px; border-radius: 999px; background: #14283C; overflow: hidden; }
        .sensor-bar-fill { height: 100%; border-radius: 999px; }
        .sensor-bar-pct { width: 40px; text-align: right; font-size: 12px; color: #B8C4D4; font-weight: 600; }

        .manual-coverage-box {
            background: #081624; border: 1px solid #1B344D; border-radius: 10px;
            padding: 12px 14px; margin-top: 10px;
        }
        .manual-coverage-title { font-size: 11px; font-weight: 700; letter-spacing: 0.5px; color: #64748B; margin-bottom: 6px; }

        .biz-card {
            background: #0C1B2A; border: 1px solid #1B344D; border-radius: 12px;
            padding: 20px 22px;
        }
        .biz-label { font-size: 11.5px; font-weight: 600; letter-spacing: 0.5px; color: #71839A; text-transform: uppercase; margin-bottom: 10px; }
        .biz-value { font-size: 30px; font-weight: 700; color: #F8FAFC; }
        .biz-note { font-size: 11.5px; color: #64748B; margin-top: 6px; }

        .phase-card {
            background: #0C1B2A; border: 1px solid #1B344D; border-radius: 12px;
            padding: 16px 18px; text-align: left;
        }
        .phase-num { font-size: 10.5px; font-weight: 700; letter-spacing: 0.6px; color: #19C6E6; margin-bottom: 6px; }
        .phase-title { font-size: 14px; font-weight: 600; color: #F8FAFC; }

        .journey-row { display: flex; align-items: center; flex-wrap: wrap; gap: 6px; margin: 14px 0; }
        .journey-step {
            display: flex; align-items: center; gap: 6px;
            background: #112438; border: 1px solid #1B344D; border-radius: 8px;
            padding: 6px 10px; font-size: 12px; color: #B8C4D4;
        }
        .journey-step.flagged { border-color: #F59E0B; color: #F59E0B; }
        .journey-step.current { border-color: #19C6E6; color: #19C6E6; }
        .journey-arrow { color: #334155; font-size: 13px; }

        /* ---- Polished command-center additions ---- */
        .dt-header {
            background: linear-gradient(180deg, rgba(17,36,56,0.35), rgba(6,16,28,0));
            border-radius: 14px;
            padding: 14px 16px 18px 16px;
        }

        .kpi-card {
            position: relative;
            overflow: hidden;
            min-height: 92px;
        }
        .kpi-card::after {
            content: "";
            position: absolute;
            left: 18px;
            right: 18px;
            bottom: 0;
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(25,198,230,0.22), transparent);
        }

        .zone-summary-row {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 8px 0 18px 0;
        }
        .zone-summary-card {
            background: #0C1B2A;
            border: 1px solid #1B344D;
            border-radius: 11px;
            padding: 12px 14px;
            min-height: 74px;
        }
        .zone-summary-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 10px;
        }
        .zone-summary-title {
            font-size: 11px;
            letter-spacing: 0.7px;
            font-weight: 700;
            color: #B8C4D4;
        }
        .zone-summary-meta {
            margin-top: 8px;
            font-size: 12px;
            color: #71839A;
        }
        .zone-health-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            display: inline-block;
            box-shadow: 0 0 8px currentColor;
        }

        .station-node {
            position: relative;
            transition: transform 0.14s ease, border-color 0.14s ease, box-shadow 0.14s ease;
        }
        .station-node:hover {
            transform: translateY(-2px);
            border-color: #315170;
        }
        .station-node.status-warning {
            border-color: rgba(245,158,11,0.48);
        }
        .station-node.status-critical {
            border-color: rgba(240,68,68,0.65);
            box-shadow: 0 0 0 1px rgba(240,68,68,0.10), 0 0 18px rgba(240,68,68,0.08);
        }
        .station-node.status-critical::before {
            content: "";
            position: absolute;
            inset: -1px;
            border: 1px solid rgba(240,68,68,0.38);
            border-radius: 10px;
            animation: dtPulse 1.9s ease-out infinite;
            pointer-events: none;
        }
        @keyframes dtPulse {
            0% { opacity: 0.65; transform: scale(1); }
            70% { opacity: 0; transform: scale(1.07); }
            100% { opacity: 0; transform: scale(1.07); }
        }

        .inspector-shell {
            background: linear-gradient(180deg, rgba(17,36,56,0.60), rgba(12,27,42,0.45));
            border: 1px solid #1B344D;
            border-radius: 14px;
            padding: 15px 16px;
            margin-bottom: 12px;
        }
        .quick-metric-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 8px;
            margin-top: 10px;
        }
        .quick-metric {
            background: #081624;
            border: 1px solid #14283C;
            border-radius: 9px;
            padding: 10px 11px;
        }
        .quick-metric-label {
            font-size: 10px;
            text-transform: uppercase;
            letter-spacing: 0.55px;
            color: #71839A;
            margin-bottom: 5px;
        }
        .quick-metric-value {
            font-size: 16px;
            line-height: 1.1;
            font-weight: 700;
            color: #F8FAFC;
        }
        .quick-metric-delta {
            font-size: 10.5px;
            color: #71839A;
            margin-top: 3px;
        }

        .line-health-banner {
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 16px;
            flex-wrap: wrap;
            background: linear-gradient(90deg, rgba(25,198,230,0.06), rgba(12,27,42,0.85));
            border: 1px solid #1B344D;
            border-radius: 12px;
            padding: 12px 15px;
            margin-bottom: 14px;
        }
        .line-health-title {
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.55px;
            color: #F8FAFC;
        }
        .line-health-sub {
            font-size: 11.5px;
            color: #71839A;
            margin-top: 3px;
        }
        .line-health-risk {
            font-size: 12px;
            color: #B8C4D4;
        }

        .trend-label {
            font-size: 10.5px;
            letter-spacing: 0.5px;
            color: #71839A;
            text-transform: uppercase;
            margin: 12px 0 4px 0;
            font-weight: 700;
        }

        @media (max-width: 900px) {
            .zone-summary-row { grid-template-columns: 1fr; }
            .quick-metric-grid { grid-template-columns: 1fr 1fr; }
        }

        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# RENDER HELPERS
# =============================================================================

def clean_html(fragment: str) -> str:
    """Flatten HTML fragments so Markdown never treats indentation as a code block."""
    return " ".join(
        line.strip()
        for line in fragment.splitlines()
        if line.strip()
    )


def render_header(step: int, n_stations: int) -> None:
    st.markdown(
        clean_html(f"""
        <div class="dt-header">
            <div>
                <p class="dt-title">DigitalTwin.ai</p>
                <p class="dt-subtitle">Predictive Vehicle Assembly Intelligence</p>
            </div>
            <div class="dt-header-right">
                <span class="dt-header-chip">Plant A — Line 1</span>
                <span class="dt-header-chip">{n_stations} Stations</span>
                <span class="dt-header-chip">Step {step}</span>
                <span class="dt-live"><span class="dt-live-dot"></span> LIVE</span>
            </div>
        </div>
        <hr/>
        """),
        unsafe_allow_html=True,
    )


def _kpi_value_class(status: str | None) -> str:
    return {
        None: "accent",
        "healthy": "healthy",
        "warning": "warning",
        "critical": "critical",
    }.get(status, "accent")


def render_kpi_card(label: str, value: str, unit: str = "", status: str | None = None) -> str:
    cls = _kpi_value_class(status)
    return clean_html(f"""
        <div class="kpi-card">
            <div class="kpi-label">{html.escape(label)}</div>
            <div class="kpi-value {cls}">{html.escape(value)}</div>
            <div class="kpi-unit">{html.escape(unit)}</div>
        </div>
    """)


def render_status_badge(status: str) -> str:
    cls = {
        "Normal": "badge-healthy",
        "Warning": "badge-warning",
        "Critical": "badge-critical",
        "Manual": "badge-manual",
    }.get(status, "badge-manual")
    label = "HEALTHY" if status == "Normal" else status.upper()
    return f'<span class="badge {cls}">{label}</span>'


def render_risk_bar(label: str, value: float) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    if pct >= 75:
        color = CRITICAL
    elif pct >= 55:
        color = WARNING
    else:
        color = HEALTHY
    return clean_html(f"""
        <div class="risk-bar-wrap">
            <div class="risk-bar-label"><span>{html.escape(label)}</span><span>{pct:.0f}%</span></div>
            <div class="risk-bar-track"><div class="risk-bar-fill" style="width:{pct:.1f}%; background:{color};"></div></div>
        </div>
    """)


def render_station_node(row: pd.Series, selected: bool) -> str:
    color = STATUS_COLOR.get(row["status"], MANUAL)
    dot_color = MANUAL if row["sensor_level"] == "manual" else color
    sel_class = " selected" if selected else ""
    status_class = f" status-{str(row['status']).lower()}"
    zone_short = str(row["station_type"]).upper()
    return clean_html(f"""
        <div class="station-node{sel_class}{status_class}">
            <div class="sid">{html.escape(str(row['station_id']))}</div>
            <div class="stype">{html.escape(zone_short)}</div>
            <div class="sdot" style="background:{dot_color}; box-shadow:0 0 5px {dot_color};"></div>
            <div class="scycle">{row['cycle_time']:.1f}s</div>
        </div>
    """)


def render_zone(zone_key: str, rows: pd.DataFrame, selected_station: str) -> None:
    if rows.empty:
        return
    warnings = int((rows["status"] != "Normal").sum())
    meta = f"{len(rows)} stations • {warnings} warning" + ("s" if warnings != 1 else "")
    nodes_html = []
    for i, (_, r) in enumerate(rows.iterrows()):
        nodes_html.append(render_station_node(r, r["station_id"] == selected_station))
        if i < len(rows) - 1:
            nodes_html.append('<span class="connector">→</span>')
    st.markdown(
        clean_html(f"""
        <div class="zone-block">
            <div class="zone-title-row">
                <span class="zone-title">{html.escape(ZONE_LABELS.get(zone_key, str(zone_key).upper()))}</span>
                <span class="zone-meta">{html.escape(meta)}</span>
            </div>
            <div class="zone-flow">{''.join(nodes_html)}</div>
        </div>
        """),
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str | None = None) -> None:
    sub_html = f'<div class="section-sub">{html.escape(subtitle)}</div>' if subtitle else ""
    st.markdown(f'<div class="section-header">{html.escape(title)}</div>{sub_html}', unsafe_allow_html=True)


def render_manual_coverage_box(confidence: float) -> str:
    return clean_html(f"""
        <div class="manual-coverage-box">
            <div class="manual-coverage-title">MANUAL COVERAGE</div>
            <div class="inspector-metric-row">
                <span class="inspector-metric-label">Direct sensors</span>
                <span class="inspector-metric-value" style="color:#64748B;">Unavailable</span>
            </div>
            <div class="inspector-metric-row">
                <span class="inspector-metric-label">Twin confidence</span>
                <span class="inspector-metric-value">{confidence*100:.0f}%</span>
            </div>
            <div style="font-size:12px; color:#71839A; margin-top:6px;">
                Inferred using: cycle time • queue behavior • upstream/downstream context • manual inspection
            </div>
        </div>
    """)


def fmt_or_na(value: float, fmt: str) -> str:
    if pd.isna(value):
        return "N/A"
    return fmt.format(value)


def render_zone_summary(scored_df: pd.DataFrame) -> None:
    cards = []
    for zone_key in ZONE_ORDER:
        rows = scored_df.loc[scored_df["station_type"].eq(zone_key)]
        if rows.empty:
            continue
        critical = int((rows["status"] == "Critical").sum())
        warning = int((rows["status"] == "Warning").sum())
        if critical:
            health, color = "CRITICAL", CRITICAL
        elif warning:
            health, color = "WARNING", WARNING
        else:
            health, color = "HEALTHY", HEALTHY
        title = ZONE_LABELS.get(zone_key, str(zone_key).upper())
        cards.append(clean_html(f"""
            <div class="zone-summary-card">
                <div class="zone-summary-top">
                    <span class="zone-summary-title">{html.escape(title)}</span>
                    <span style="display:flex;align-items:center;gap:7px;color:{color};font-size:10.5px;font-weight:700;">
                        <span class="zone-health-dot" style="background:{color};color:{color};"></span>{health}
                    </span>
                </div>
                <div class="zone-summary-meta">
                    {len(rows)} stations · {warning} warning{"s" if warning != 1 else ""} · {critical} critical
                </div>
            </div>
        """))
    if cards:
        st.markdown(
            f'<div class="zone-summary-row">{"".join(cards)}</div>',
            unsafe_allow_html=True,
        )


def render_quick_metric(label: str, value: str, delta: str = "") -> str:
    delta_html = f'<div class="quick-metric-delta">{html.escape(delta)}</div>' if delta else ""
    return clean_html(f"""
        <div class="quick-metric">
            <div class="quick-metric-label">{html.escape(label)}</div>
            <div class="quick-metric-value">{html.escape(value)}</div>
            {delta_html}
        </div>
    """)


def style_sparkline(fig: go.Figure, height: int = 118) -> go.Figure:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=4, b=0),
        showlegend=False,
        hovermode="x unified",
        font=dict(color=TEXT_MUTED, family="Inter, sans-serif", size=10),
    )
    fig.update_xaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )
    fig.update_yaxes(
        visible=False,
        showgrid=False,
        zeroline=False,
        fixedrange=True,
    )
    return fig


# =============================================================================
# APP STATE (unchanged logic)
# =============================================================================

inject_global_css()


@st.cache_resource
def get_models():
    return train_models()


@st.cache_resource
def get_config():
    return load_config()


bottleneck_model, defect_model, anomaly_model = get_models()
config = get_config()

if "step" not in st.session_state:
    st.session_state.step = 0
if "simulation_seed" not in st.session_state:
    st.session_state.simulation_seed = 42
if "line" not in st.session_state:
    st.session_state.line = initial_line(config=config)
if "history" not in st.session_state:
    st.session_state.history = []
if "station_history" not in st.session_state:
    st.session_state.station_history = []
if "vehicles" not in st.session_state:
    st.session_state.vehicles = spawn_vehicles(
        st.session_state.line["station_id"].tolist(), config=config
    )
if "vehicle_log" not in st.session_state:
    st.session_state.vehicle_log = pd.DataFrame()
if "feedback" not in st.session_state:
    st.session_state.feedback = []  # list of dicts: station_id, step, status, verdict

# =============================================================================
# SIDEBAR — SIMULATION CONTROL
# =============================================================================

with st.sidebar:
    st.markdown(
        '<div style="font-size:12px; font-weight:700; letter-spacing:0.6px; color:#71839A; margin-bottom:10px;">'
        "SIMULATION CONTROL</div>",
        unsafe_allow_html=True,
    )

    scenario = st.selectbox(
        "Scenario",
        ["Normal", "Developing bottleneck", "High defect risk", "Sensor-poor station", "Random"],
        help=(
            "Random picks 1-3 stations and a random severity/failure mode each run — "
            "a stress test the model was not tuned against, unlike the scripted scenarios."
        ),
    )

    col1, col2 = st.columns(2)
    with col1:
        advance = st.button("▶ Advance", use_container_width=True, type="primary")
    with col2:
        reset = st.button("↻ Reset", use_container_width=True)

    if reset:
        st.session_state.step = 0
        st.session_state.simulation_seed += 101
        st.session_state.line = initial_line(config=config)
        st.session_state.history = []
        st.session_state.station_history = []
        st.session_state.vehicles = spawn_vehicles(
            st.session_state.line["station_id"].tolist(), config=config
        )
        st.session_state.vehicle_log = pd.DataFrame()
        st.session_state.feedback = []
        st.rerun()

    if advance:
        st.session_state.step += 1
        st.session_state.line = simulate_step(
            st.session_state.line,
            st.session_state.step,
            scenario=scenario,
            seed=st.session_state.simulation_seed,
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px; font-weight:700; letter-spacing:0.6px; color:#71839A; margin-bottom:8px;">'
        "ACTIVE SCENARIO</div>",
        unsafe_allow_html=True,
    )
    scenario_target = {
        "Developing bottleneck": "S14",
        "High defect risk": "S07",
        "Sensor-poor station": "S20",
    }.get(scenario, "—")
    st.markdown(
        clean_html(f"""
        <div style="font-size:14px; color:#F8FAFC; font-weight:600; margin-bottom:6px;">{html.escape(scenario)}</div>
        <div class="inspector-metric-row"><span class="inspector-metric-label">Target station</span><span class="inspector-metric-value">{html.escape(scenario_target)}</span></div>
        <div class="inspector-metric-row"><span class="inspector-metric-label">Simulation step</span><span class="inspector-metric-value">{st.session_state.step}</span></div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown("<hr/>", unsafe_allow_html=True)
    with st.expander("Demo guide"):
        st.markdown("1. Choose **Developing bottleneck**")
        st.markdown("2. Press **Advance** 6–10 times")
        st.markdown("3. Watch Station S14 turn warning/critical")
        st.markdown("4. Switch to **High defect risk** and reset")
        st.markdown("5. Advance 6–10 steps to see S07")
        st.markdown("6. Try **Random** for an unscripted stress test")

    with st.expander("Model validation (held-out test set)"):
        metrics = get_model_metrics()
        if not metrics:
            st.caption("No metrics available yet.")
        for key, m in metrics.items():
            st.markdown(f"**{m['name']}**")
            st.caption(
                f"AUC {m['auc']:.2f} · Precision {m['precision']:.2f} · "
                f"Recall {m['recall']:.2f} · F1 {m['f1']:.2f} "
                f"(n={m['n_test']}, {m['positive_rate']*100:.0f}% positive)"
            )
        st.caption(
            "Computed on a 25% held-out split of the synthetic training "
            "distribution — not the live scenario data. This shows the "
            "model generalizes within its training distribution; the "
            "scenario simulator (below) is a separate, harder test of "
            "generalization to unseen degradation patterns."
        )

# =============================================================================
# SCORING (unchanged logic)
# =============================================================================

scored = score_line(
    st.session_state.line,
    bottleneck_model,
    defect_model,
    anomaly_model,
)

if advance:
    st.session_state.vehicles, new_log = advance_vehicles(
        st.session_state.vehicles, scored, st.session_state.step
    )
    if not new_log.empty:
        st.session_state.vehicle_log = pd.concat(
            [st.session_state.vehicle_log, new_log], ignore_index=True
        )

    top_b = scored.sort_values("bottleneck_risk", ascending=False).iloc[0]
    top_d = scored.sort_values("defect_risk", ascending=False).iloc[0]
    st.session_state.history.append({
        "step": st.session_state.step,
        "scenario": scenario,
        "max_bottleneck_risk": float(top_b["bottleneck_risk"]),
        "max_defect_risk": float(top_d["defect_risk"]),
        "avg_cycle_time": float(scored["cycle_time"].mean()),
        "total_queue": int(scored["queue_length"].sum()),
    })

    snapshot_cols = [
        "station_id", "cycle_time", "queue_length", "throughput",
        "bottleneck_risk", "defect_risk", "confidence",
    ]
    for snapshot in scored[snapshot_cols].to_dict("records"):
        snapshot["step"] = st.session_state.step
        snapshot["scenario"] = scenario
        st.session_state.station_history.append(snapshot)
    st.session_state.station_history = st.session_state.station_history[-3000:]

hist_df = pd.DataFrame(st.session_state.history).drop_duplicates(
    subset=["step", "scenario"], keep="last"
).tail(100)

station_hist_df = pd.DataFrame(st.session_state.station_history)
if not station_hist_df.empty:
    station_hist_df = station_hist_df.drop_duplicates(
        subset=["step", "scenario", "station_id"], keep="last"
    ).tail(3000)

# =============================================================================
# HEADER + KPI ROW
# =============================================================================

render_header(st.session_state.step, len(scored))

critical_count = int((scored["status"] == "Critical").sum())
warning_count = int((scored["status"] == "Warning").sum())
avg_cycle = scored["cycle_time"].mean()
line_throughput = scored["throughput"].median()
mean_confidence = scored["confidence"].mean()

kpi_cards = [
    render_kpi_card("Line Throughput", f"{line_throughput:.1f}", "units/hour"),
    render_kpi_card("Avg Cycle Time", f"{avg_cycle:.1f}", "seconds"),
    render_kpi_card(
        "Active Warnings", f"{warning_count}", "stations",
        status="warning" if warning_count else "healthy",
    ),
    render_kpi_card(
        "Critical Stations", f"{critical_count}", "stations",
        status="critical" if critical_count else "healthy",
    ),
    render_kpi_card("Model Confidence", f"{mean_confidence*100:.0f}%", "avg. across line"),
]
st.markdown(f'<div class="kpi-row">{"".join(kpi_cards)}</div>', unsafe_allow_html=True)
st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

# =============================================================================
# NAVIGATION
# =============================================================================

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "◉  Live Twin",
    "⚠  Alerts",
    "◇  Traceability",
    "▥  Analytics",
    "↗  Business Impact",
])

# -----------------------------------------------------------------------
# TAB 1 — LIVE TWIN
# -----------------------------------------------------------------------
with tab1:
    station_options = scored["station_id"].tolist()

    # High-level line-health banner
    top_risk_row = scored.loc[
        scored[["bottleneck_risk", "defect_risk"]].max(axis=1).idxmax()
    ]
    overall_status = (
        "CRITICAL" if critical_count
        else ("WARNING" if warning_count else "HEALTHY")
    )
    overall_color = (
        CRITICAL if critical_count
        else (WARNING if warning_count else HEALTHY)
    )
    st.markdown(
        clean_html(f"""
        <div class="line-health-banner">
            <div>
                <div class="line-health-title">LINE HEALTH · {overall_status}</div>
                <div class="line-health-sub">
                    {len(scored)}-station mixed-model assembly line · predictive monitoring active
                </div>
            </div>
            <div class="line-health-risk">
                Highest current risk:
                <strong style="color:{overall_color};">{html.escape(str(top_risk_row["station_id"]))}</strong>
                · {max(float(top_risk_row["bottleneck_risk"]), float(top_risk_row["defect_risk"])) * 100:.0f}%
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    render_zone_summary(scored)

    # Give the actual twin most of the width; inspector is intentionally narrower.
    left, right = st.columns([1.72, 0.88], gap="large")

    with left:
        render_section_header(
            "Factory Line",
            "BODY → PAINT → FINAL ASSEMBLY · live station health and process flow",
        )

        # Selection remains a native Streamlit control for reliability, while the
        # map visually highlights the chosen station.
        selected_default = st.session_state.get("live_twin_station_select", station_options[0])
        if selected_default not in station_options:
            selected_default = station_options[0]

        for zone_key in ZONE_ORDER:
            zone_rows = scored.loc[scored["station_type"].eq(zone_key)]
            render_zone(zone_key, zone_rows, selected_default)

        other_types = [t for t in scored["station_type"].unique() if t not in ZONE_ORDER]
        for zone_key in other_types:
            zone_rows = scored.loc[scored["station_type"].eq(zone_key)]
            render_zone(zone_key, zone_rows, selected_default)

        st.markdown("<hr/>", unsafe_allow_html=True)

        render_section_header("Predictive Outlook", "Prototype trend estimate")
        current_risk = float(scored["bottleneck_risk"].max())
        top_row = scored.loc[scored["bottleneck_risk"].idxmax()]
        target_station = str(top_row["station_id"])

        projected_risk = current_risk
        steps_to_critical = None

        if not station_hist_df.empty:
            recent = (
                station_hist_df[
                    station_hist_df["station_id"].eq(target_station)
                ]
                .sort_values("step")
                .tail(8)
            )

            if len(recent) >= 3:
                x = recent["step"].to_numpy(dtype=float)
                y = recent["bottleneck_risk"].to_numpy(dtype=float)

                if len(set(x)) > 1:
                    slope, intercept = np.polyfit(x, y, 1)
                    next_step = x[-1] + 5
                    projected_risk = float(
                        np.clip(slope * next_step + intercept, 0.0, 1.0)
                    )

                    if slope > 0.001:
                        remaining = (0.75 - y[-1]) / slope
                        if remaining > 0:
                            steps_to_critical = max(
                                1,
                                int(round(remaining)),
                            )

        pcol1, pcol2, pcol3 = st.columns(3)
        pcol1.markdown(
            render_kpi_card(
                "Current Bottleneck Risk",
                f"{current_risk*100:.0f}%",
                str(top_row["station_id"]),
            ),
            unsafe_allow_html=True,
        )
        pcol2.markdown(
            render_kpi_card(
                "Projected Risk",
                f"{projected_risk*100:.0f}%",
                "+5 steps ahead",
                status=(
                    "critical" if projected_risk >= 0.75
                    else ("warning" if projected_risk >= 0.55 else "healthy")
                ),
            ),
            unsafe_allow_html=True,
        )
        eta_text = (
            f"~{steps_to_critical} sim. steps"
            if steps_to_critical
            else "Stable"
        )
        pcol3.markdown(
            render_kpi_card(
                "Critical ETA",
                eta_text,
                "at current trend",
                status="warning" if steps_to_critical else "healthy",
            ),
            unsafe_allow_html=True,
        )
        st.caption(
            "Prototype trend estimate — a simple linear projection of recent "
            "bottleneck-risk history, not a production-grade forecasting model."
        )

    with right:
        render_section_header(
            "Station Inspector",
            "Select a station to inspect risk, process conditions and recent trend",
        )
        station = st.selectbox(
            "Select station",
            station_options,
            key="live_twin_station_select",
            label_visibility="collapsed",
        )
        row = scored.loc[scored["station_id"].eq(station)].iloc[0]
        reasons, action = explain_station(row)

        baseline_cycle = 48.0
        cycle_delta = ((float(row["cycle_time"]) - baseline_cycle) / baseline_cycle) * 100
        cycle_delta_text = f"{cycle_delta:+.0f}% vs baseline"
        queue_delta_text = "Upstream WIP"
        throughput_delta_text = "Current line rate"
        vibration_text = fmt_or_na(row["vibration"], "{:.2f} mm/s")

        quick_metrics = "".join([
            render_quick_metric("Cycle Time", f'{row["cycle_time"]:.1f} sec', cycle_delta_text),
            render_quick_metric("Queue", f'{int(row["queue_length"])} units', queue_delta_text),
            render_quick_metric("Throughput", f'{row["throughput"]:.1f}/hr', throughput_delta_text),
            render_quick_metric("Vibration", vibration_text, "Machine condition"),
        ])

        st.markdown(
            clean_html(f"""
            <div class="inspector-shell">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
                    <div>
                        <div style="font-size:19px; font-weight:700; color:#F8FAFC;">{html.escape(str(row["station_id"]))}</div>
                        <div style="font-size:12px; color:#71839A;">
                            {html.escape(ZONE_LABELS.get(row["station_type"], str(row["station_type"])))}
                            · {html.escape(str(row["sensor_level"]).upper())} sensing
                        </div>
                    </div>
                    {render_status_badge(row["status"])}
                </div>
                <div class="quick-metric-grid">{quick_metrics}</div>
            </div>
            """),
            unsafe_allow_html=True,
        )

        st.markdown(render_risk_bar("Bottleneck Risk", row["bottleneck_risk"]), unsafe_allow_html=True)
        st.markdown(render_risk_bar("Defect Risk", row["defect_risk"]), unsafe_allow_html=True)
        st.markdown(render_risk_bar("Prediction Confidence", row["confidence"]), unsafe_allow_html=True)

        # Recent selected-station sparklines
        st.markdown('<div class="trend-label">Recent station trend</div>', unsafe_allow_html=True)
        selected_hist = (
            station_hist_df.loc[station_hist_df["station_id"].eq(station)].sort_values("step")
            if not station_hist_df.empty
            else pd.DataFrame()
        )

        if len(selected_hist) >= 2:
            sc1, sc2 = st.columns(2)
            with sc1:
                fig_cycle_spark = go.Figure()
                fig_cycle_spark.add_trace(go.Scatter(
                    x=selected_hist["step"],
                    y=selected_hist["cycle_time"],
                    mode="lines+markers",
                    line=dict(color=ACCENT_CYAN, width=2),
                    marker=dict(size=4),
                    fill="tozeroy",
                    fillcolor="rgba(25,198,230,0.05)",
                    hovertemplate="Step %{x}<br>%{y:.1f}s<extra></extra>",
                ))
                st.caption("Cycle time")
                st.plotly_chart(
                    style_sparkline(fig_cycle_spark),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"cycle_spark_{station}",
                )
            with sc2:
                risk_color = (
                    CRITICAL if row["bottleneck_risk"] >= 0.75
                    else (WARNING if row["bottleneck_risk"] >= 0.55 else HEALTHY)
                )
                fig_risk_spark = go.Figure()
                fig_risk_spark.add_trace(go.Scatter(
                    x=selected_hist["step"],
                    y=selected_hist["bottleneck_risk"],
                    mode="lines+markers",
                    line=dict(color=risk_color, width=2),
                    marker=dict(size=4),
                    fill="tozeroy",
                    fillcolor="rgba(240,68,68,0.04)",
                    hovertemplate="Step %{x}<br>%{y:.0%}<extra></extra>",
                ))
                fig_risk_spark.update_yaxes(range=[0, 1])
                st.caption("Bottleneck risk")
                st.plotly_chart(
                    style_sparkline(fig_risk_spark),
                    use_container_width=True,
                    config={"displayModeBar": False},
                    key=f"risk_spark_{station}",
                )
        else:
            st.caption("Advance the simulation at least twice to build station trend history.")

        why_items = "".join(
            f"<li>{html.escape(r.capitalize())}.</li>" for r in reasons[:3]
        )
        st.markdown(
            '<div style="font-size:11px;font-weight:700;letter-spacing:0.5px;color:#71839A;margin:12px 0 4px 0;">'
            f'WHY IT WAS FLAGGED</div><ul class="why-list">{why_items}</ul>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div style="font-size:11px;font-weight:700;letter-spacing:0.5px;color:#71839A;margin:12px 0 4px 0;">'
            f'RECOMMENDED ACTION</div><div class="action-box">{html.escape(action)}</div>',
            unsafe_allow_html=True,
        )

        with st.expander("More process details"):
            detailed_metrics = pd.DataFrame({
                "Metric": [
                    "Temperature",
                    "Torque",
                    "Downtime",
                    "Anomaly score",
                    "Sensor coverage",
                ],
                "Value": [
                    fmt_or_na(row["temperature"], "{:.1f} °C"),
                    fmt_or_na(row["torque"], "{:.1f} Nm"),
                    f'{row["downtime_min"]:.1f} min',
                    f'{row["anomaly_score"]*100:.0f}%',
                    str(row["sensor_level"]).title(),
                ],
            })
            st.dataframe(
                detailed_metrics,
                hide_index=True,
                use_container_width=True,
            )

        if row["sensor_level"] in ("partial", "manual"):
            st.markdown(
                render_manual_coverage_box(row["confidence"]),
                unsafe_allow_html=True,
            )
            st.caption("🔧 " + maintenance_window_note(config))

# -----------------------------------------------------------------------
# TAB 2 — ALERTS
# -----------------------------------------------------------------------
with tab2:
    render_section_header("Active Incidents")

    fb = pd.DataFrame(st.session_state.feedback)
    confirmed = false_alarm = total = 0
    acceptance_rate = None
    if not fb.empty:
        confirmed = int((fb["verdict"] == "confirmed").sum())
        false_alarm = int((fb["verdict"] == "false_alarm").sum())
        total = confirmed + false_alarm
        acceptance_rate = (confirmed / total * 100) if total else None

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(render_kpi_card("Alerts Reviewed", f"{total}"), unsafe_allow_html=True)
    m2.markdown(render_kpi_card("Confirmed", f"{confirmed}", status="critical" if confirmed else None), unsafe_allow_html=True)
    m3.markdown(render_kpi_card("False Alarms", f"{false_alarm}"), unsafe_allow_html=True)
    m4.markdown(
        render_kpi_card("Acceptance Rate", f"{acceptance_rate:.0f}%" if acceptance_rate is not None else "—"),
        unsafe_allow_html=True,
    )
    st.caption(
        "Acceptance rate = confirmed issues ÷ reviewed alerts. This is the feedback "
        "loop that would calibrate thresholds against real outcomes before production use."
    )
    st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)

    alerts = scored[
        (scored["status"] != "Normal")
        | (scored["anomaly_score"] >= 0.62)
    ].copy()

    if alerts.empty:
        st.success("No active high-priority alerts.")
    else:
        alerts["priority"] = alerts[["bottleneck_risk", "defect_risk", "anomaly_score"]].max(axis=1)
        alerts = alerts.sort_values("priority", ascending=False)

        for _, row in alerts.head(8).iterrows():
            reasons, action = explain_station(row)
            css_class = "critical" if row["status"] == "Critical" else "warning"

            why_items = "".join(f"<li>{html.escape(r.capitalize())}.</li>" for r in reasons[:3])
            st.markdown(
                clean_html(f"""
                <div class="alert-card {css_class}">
                    <div class="alert-head">
                        <span class="alert-station">{html.escape(row['station_id'])} — {html.escape(row['station_type'])}</span>
                        {render_status_badge(row['status'])}
                    </div>
                    {render_risk_bar("Predicted Bottleneck Risk", row['bottleneck_risk'])}
                    {render_risk_bar("Prediction Confidence", row['confidence'])}
                    <div style="font-size:11px; font-weight:700; letter-spacing:0.5px; color:#71839A; margin:10px 0 4px 0;">WHY</div>
                    <ul class="why-list">{why_items}</ul>
                    <div style="font-size:11px; font-weight:700; letter-spacing:0.5px; color:#71839A; margin:12px 0 4px 0;">RECOMMENDED ACTION</div>
                    <div class="action-box">{html.escape(action)}</div>
                </div>
                """),
                unsafe_allow_html=True,
            )

            key_base = f"{row['station_id']}_{st.session_state.step}"
            fcol1, fcol2, _ = st.columns([1, 1, 3])
            with fcol1:
                if st.button("Confirm Issue", key=f"confirm_{key_base}"):
                    st.session_state.feedback.append({
                        "station_id": row["station_id"],
                        "step": st.session_state.step,
                        "status": row["status"],
                        "verdict": "confirmed",
                    })
                    st.rerun()
            with fcol2:
                if st.button("False Alarm", key=f"false_{key_base}"):
                    st.session_state.feedback.append({
                        "station_id": row["station_id"],
                        "step": st.session_state.step,
                        "status": row["status"],
                        "verdict": "false_alarm",
                    })
                    st.rerun()
            st.markdown("<div style='height:6px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------
# TAB 3 — TRACEABILITY
# -----------------------------------------------------------------------
with tab3:
    render_section_header(
        "Vehicle Genealogy & Quality Traceability",
        "A defect introduced early on the line may not surface until a much later inspection point — "
        "every unit is tagged with the stations it passed and the risk at each, so a problem can be "
        "traced back even after the unit has left the affected station.",
    )

    log = st.session_state.vehicle_log
    total_vehicles = int(log["vehicle_id"].nunique()) if not log.empty else 0

    if log.empty:
        st.info("Press **Advance** in the sidebar to move vehicles through the line and build genealogy history.")
    else:
        at_risk = vehicles_at_risk(log, defect_risk_threshold=0.55)
        units_at_risk = len(at_risk)
        highest_exposure = f"{at_risk['max_defect_risk'].max()*100:.0f}%" if not at_risk.empty else "—"
        stations_involved = (
            len(set(s for stations in at_risk["flagged_stations"] for s in stations.split(", ")))
            if not at_risk.empty else 0
        )

        t1, t2, t3, t4 = st.columns(4)
        t1.markdown(render_kpi_card("Vehicles Tracked", f"{total_vehicles}"), unsafe_allow_html=True)
        t2.markdown(
            render_kpi_card("Units At Risk", f"{units_at_risk}", status="warning" if units_at_risk else "healthy"),
            unsafe_allow_html=True,
        )
        t3.markdown(render_kpi_card("Highest Defect Exposure", highest_exposure), unsafe_allow_html=True)
        t4.markdown(render_kpi_card("Stations Involved", f"{stations_involved}"), unsafe_allow_html=True)

        st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
        render_section_header(f"Units flagged for inspection ({units_at_risk})")
        if at_risk.empty:
            st.success("No units have passed through an elevated-defect-risk station yet.")
        else:
            display_at_risk = at_risk.rename(columns={
                "vehicle_id": "Vehicle",
                "model": "Model",
                "flagged_stations": "Flagged Station",
                "max_defect_risk": "Peak Defect Risk",
                "first_flagged_step": "First Exposure",
            }).copy()
            display_at_risk["Peak Defect Risk"] = (display_at_risk["Peak Defect Risk"] * 100).round(0).astype(int).astype(str) + "%"
            display_at_risk["Inspection Status"] = "Pending"
            st.dataframe(display_at_risk, hide_index=True, use_container_width=True)
            st.caption(
                "These units should be held or inspected downstream even though they may "
                "already have left the flagged station — this is the audit trail a QA team "
                "would use for root-cause tracing."
            )

        st.markdown("<hr/>", unsafe_allow_html=True)
        render_section_header("Vehicle Journey")
        vehicle_ids = sorted(log["vehicle_id"].unique().tolist())
        selected_vehicle = st.selectbox("Vehicle ID", vehicle_ids, key="traceability_vehicle_select")
        v_log = log.loc[log["vehicle_id"].eq(selected_vehicle)].sort_values("step")

        journey_html = ['<div class="journey-row">']
        rows_list = list(v_log.iterrows())
        for i, (_, jr) in enumerate(rows_list):
            is_last = i == len(rows_list) - 1
            defect_risk = jr["defect_risk"]
            if is_last:
                cls, label = "current", f'{jr["station_id"]} · CURRENT'
            elif defect_risk >= 0.55:
                cls, label = "flagged", f'{jr["station_id"]} · {defect_risk*100:.0f}% defect exposure'
            else:
                cls, label = "", f'{jr["station_id"]} ✓'
            journey_html.append(f'<div class="journey-step {cls}">{html.escape(label)}</div>')
            if not is_last:
                journey_html.append('<span class="journey-arrow">→</span>')
        journey_html.append("</div>")
        st.markdown("".join(journey_html), unsafe_allow_html=True)

        has_flag = (v_log["defect_risk"] >= 0.55).any()
        if has_flag:
            st.markdown(
                '<div class="action-box">Recommended QA action: Inspect vehicle at next available quality gate.</div>',
                unsafe_allow_html=True,
            )

        st.dataframe(
            v_log[["step", "station_id", "sensor_level", "defect_risk", "bottleneck_risk", "confidence"]].rename(
                columns={
                    "step": "Step", "station_id": "Station", "sensor_level": "Sensor Level",
                    "defect_risk": "Defect Risk", "bottleneck_risk": "Bottleneck Risk", "confidence": "Confidence",
                }
            ),
            hide_index=True,
            use_container_width=True,
        )

# -----------------------------------------------------------------------
# TAB 4 — ANALYTICS
# -----------------------------------------------------------------------
with tab4:
    render_section_header("Plant Manager Analytics")

    a1, a2, a3, a4, a5 = st.columns(5)
    a1.markdown(render_kpi_card("Throughput", f"{line_throughput:.1f}", "units/hr"), unsafe_allow_html=True)
    a2.markdown(render_kpi_card("Avg Cycle Time", f"{avg_cycle:.1f}", "sec"), unsafe_allow_html=True)
    a3.markdown(
        render_kpi_card("Critical Stations", f"{critical_count}", status="critical" if critical_count else "healthy"),
        unsafe_allow_html=True,
    )
    a4.markdown(render_kpi_card("Total Queue", f"{int(scored['queue_length'].sum())}", "units"), unsafe_allow_html=True)
    a5.markdown(render_kpi_card("Data Confidence", f"{mean_confidence*100:.0f}%", "avg. across line"), unsafe_allow_html=True)

    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    def _style_fig(fig: go.Figure) -> go.Figure:
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_SECONDARY, family="Inter, sans-serif"),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            margin=dict(t=48, l=10, r=10, b=10),
            title_font=dict(color=TEXT_PRIMARY, size=16),
        )
        fig.update_xaxes(gridcolor=DIVIDER, linecolor=BORDER, tickfont=dict(color="#94A3B8"))
        fig.update_yaxes(gridcolor=DIVIDER, linecolor=BORDER, tickfont=dict(color="#94A3B8"))
        return fig

    plot_df = scored.sort_values("station_id")

    bar_colors = plot_df["status"].map(STATUS_COLOR)
    fig_bottleneck = go.Figure(go.Bar(
        x=plot_df["station_id"], y=plot_df["bottleneck_risk"], marker_color=bar_colors,
        hovertext=[
            f"Cycle {c:.1f}s · Queue {q} · Throughput {t:.1f} · {s}"
            for c, q, t, s in zip(plot_df["cycle_time"], plot_df["queue_length"], plot_df["throughput"], plot_df["sensor_level"])
        ],
    ))
    fig_bottleneck.update_layout(title="Bottleneck Risk by Station")
    fig_bottleneck.update_yaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(_style_fig(fig_bottleneck), use_container_width=True)

    if not hist_df.empty:
        long = hist_df.melt(
            id_vars=["step"],
            value_vars=["max_bottleneck_risk", "max_defect_risk"],
            var_name="metric",
            value_name="risk",
        )
        fig_trend = px.line(
            long, x="step", y="risk", color="metric", markers=True,
            title="Prediction Risk Over Time",
            color_discrete_map={"max_bottleneck_risk": ACCENT_CYAN, "max_defect_risk": WARNING},
        )
        fig_trend.update_yaxes(tickformat=".0%", range=[0, 1])
        st.plotly_chart(_style_fig(fig_trend), use_container_width=True)

    c_left, c_right = st.columns(2)
    with c_left:
        fig_cycle = go.Figure(go.Bar(
            x=plot_df["station_id"], y=plot_df["cycle_time"], marker_color=ACCENT_CYAN,
        ))
        fig_cycle.update_layout(title="Cycle Time by Station")
        st.plotly_chart(_style_fig(fig_cycle), use_container_width=True)

    with c_right:
        cov_counts = scored["sensor_level"].value_counts(normalize=True)
        cov_labels = {"rich": "Rich", "partial": "Partial", "manual": "Manual"}
        cov_colors = {"rich": ACCENT_CYAN, "partial": WARNING, "manual": MANUAL}
        fig_cov = go.Figure(go.Bar(
            x=[cov_labels.get(k, k) for k in cov_counts.index],
            y=[v * 100 for v in cov_counts.values],
            marker_color=[cov_colors.get(k, MANUAL) for k in cov_counts.index],
        ))
        fig_cov.update_layout(title="Sensor Coverage")
        fig_cov.update_yaxes(ticksuffix="%")
        st.plotly_chart(_style_fig(fig_cov), use_container_width=True)

    render_section_header("Highest-Risk Stations")
    top_risk = plot_df.sort_values("bottleneck_risk", ascending=False).head(6)
    fig_top = go.Figure(go.Bar(
        x=top_risk["bottleneck_risk"], y=top_risk["station_id"], orientation="h",
        marker_color=top_risk["status"].map(STATUS_COLOR),
    ))
    fig_top.update_layout(title=None, yaxis=dict(autorange="reversed"))
    fig_top.update_xaxes(tickformat=".0%", range=[0, 1])
    st.plotly_chart(_style_fig(fig_top), use_container_width=True)

    st.markdown("<hr/>", unsafe_allow_html=True)
    render_section_header("Sensor Maturity")
    cov_pct = scored["sensor_level"].value_counts(normalize=True)
    bars_html = []
    for level in ["rich", "partial", "manual"]:
        pct = cov_pct.get(level, 0.0) * 100
        color = {"rich": ACCENT_CYAN, "partial": WARNING, "manual": MANUAL}[level]
        bars_html.append(clean_html(f"""
            <div class="sensor-bar-row">
                <span class="sensor-bar-label">{level}</span>
                <div class="sensor-bar-track"><div class="sensor-bar-fill" style="width:{pct:.0f}%; background:{color};"></div></div>
                <span class="sensor-bar-pct">{pct:.0f}%</span>
            </div>
        """))
    st.markdown("".join(bars_html), unsafe_allow_html=True)

    render_section_header("Lowest Sensor Coverage")
    sensor_poor = scored.loc[
        scored["sensor_level"].isin(["partial", "manual"]),
        ["station_id", "station_type", "sensor_level", "confidence", "status"]
    ].copy()
    sensor_poor["confidence"] = (sensor_poor["confidence"] * 100).round(0).astype(int).astype(str) + "%"
    sensor_poor.columns = ["Station", "Zone", "Sensor Level", "Confidence", "Status"]
    st.dataframe(sensor_poor, hide_index=True, use_container_width=True)
    if not sensor_poor.empty:
        st.caption("🔧 " + maintenance_window_note(config))

# -----------------------------------------------------------------------
# TAB 5 — BUSINESS IMPACT
# -----------------------------------------------------------------------
with tab5:
    render_section_header("Business Impact")

    high_risk = int(((scored["bottleneck_risk"] >= 0.65) | (scored["defect_risk"] >= 0.65)).sum())
    estimated_downtime_avoided = high_risk * 18
    estimated_units_protected = high_risk * 7
    illustrative_value = estimated_downtime_avoided * 150 + estimated_units_protected * 400

    b1, b2, b3, b4 = st.columns(4)
    with b1:
        st.markdown(
            clean_html(f"""<div class="biz-card"><div class="biz-label">High-Risk Stations Detected</div>
            <div class="biz-value">{high_risk}</div>
            <div class="biz-note">Illustrative prototype estimate</div></div>"""),
            unsafe_allow_html=True,
        )
    with b2:
        st.markdown(
            clean_html(f"""<div class="biz-card"><div class="biz-label">Potential Downtime Avoided</div>
            <div class="biz-value">{estimated_downtime_avoided} min</div>
            <div class="biz-note">Illustrative prototype estimate</div></div>"""),
            unsafe_allow_html=True,
        )
    with b3:
        st.markdown(
            clean_html(f"""<div class="biz-card"><div class="biz-label">Production Units Protected</div>
            <div class="biz-value">{estimated_units_protected}</div>
            <div class="biz-note">Illustrative prototype estimate</div></div>"""),
            unsafe_allow_html=True,
        )
    with b4:
        st.markdown(
            clean_html(f"""<div class="biz-card"><div class="biz-label">Estimated Business Value</div>
            <div class="biz-value">${illustrative_value:,.0f}</div>
            <div class="biz-note">Illustrative prototype estimate</div></div>"""),
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

    metrics = get_model_metrics()
    if metrics:
        render_section_header("Model Validation (held-out test set)")
        met_df = pd.DataFrame([
            {
                "Model": m["name"], "AUC": f"{m['auc']:.2f}", "Precision": f"{m['precision']:.2f}",
                "Recall": f"{m['recall']:.2f}", "F1": f"{m['f1']:.2f}",
            }
            for m in metrics.values()
        ])
        st.dataframe(met_df, hide_index=True, use_container_width=True)
        st.caption(
            "These are accuracy metrics for the underlying classifiers on held-out "
            "synthetic data — a production rollout would replace this training set "
            "with labeled historical plant outcomes and re-validate before go-live."
        )

    st.markdown("<hr/>", unsafe_allow_html=True)
    render_section_header("Rollout Plan")
    phases = [
        ("PHASE 1", "Pilot Line", "LIVE"),
        ("PHASE 2", "Read-only plant integration", None),
        ("PHASE 3", "Human-in-the-loop validation", None),
        ("PHASE 4", "Multi-plant rollout", None),
    ]
    p_cols = st.columns(4)
    for col, (num, title, tag) in zip(p_cols, phases):
        tag_html = '<span class="badge badge-healthy" style="margin-top:8px; display:inline-block;">LIVE</span>' if tag == "LIVE" else ""
        col.markdown(
            clean_html(f"""<div class="phase-card"><div class="phase-num">{num}</div>
            <div class="phase-title">{html.escape(title)}</div>{tag_html}</div>"""),
            unsafe_allow_html=True,
        )
    st.caption(
        "🔧 Instrumentation upgrades for partial/manual stations are constrained to "
        f"~{config.get('maintenance_windows_per_year', 4)} scheduled maintenance windows per year — "
        "the rollout plan above assumes sensor gaps are closed opportunistically during those windows, "
        "not on demand."
    )

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    "Prototype note: predictions are trained on synthetic data for demonstration. "
    "A production deployment would require validation against real plant outcomes before operational use."
)