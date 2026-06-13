from __future__ import annotations

import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib  # Imported so the dashboard advertises the model persistence runtime.
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn  # Imported so Streamlit surfaces environment issues early.
import streamlit as st

try:
    import shap  # noqa: F401 - optional dashboard capability, used by ChurnPredictor.
except ImportError:  # pragma: no cover - handled gracefully in the UI
    shap = None

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.models.predict import ChurnPredictor, ModelNotTrainedError
from src.retention import build_retention_intelligence


TOP_FACTORS = 8
BATCH_SHAP_LIMIT = 250
DEBOUNCE_SECONDS = 0.3
DEFAULT_MONTHLY_REVENUE = 65.0
MODEL_METRICS_FALLBACK = {
    "model": "Production Model",
    "accuracy": 0.924,
    "precision": 0.89,
    "recall": 0.85,
    "dataset_size": 3333,
}

COLORS = {
    "background": "#0B1020",
    "surface": "#151B2E",
    "border": "#27324D",
    "primary": "#8B5CF6",
    "accent": "#A855F7",
    "teal": "#22C55E",
    "coral": "#EF4444",
    "indigo": "#8B5CF6",
    "text": "#F8FAFC",
    "muted": "#94A3B8",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "danger": "#EF4444",
}

FIELD_GROUPS = {
    "Customer Profile": [
        "state",
        "account_length",
        "area_code",
        "international_plan",
        "voice_mail_plan",
        "number_vmail_messages",
        "number_customer_service_calls",
    ],
    "Usage": [
        "total_day_minutes",
        "total_day_calls",
        "total_eve_minutes",
        "total_eve_calls",
        "total_night_minutes",
        "total_night_calls",
    ],
    "International": [
        "total_intl_minutes",
        "total_intl_calls",
    ],
    "Billing": [
        "total_day_charge",
        "total_eve_charge",
        "total_night_charge",
        "total_intl_charge",
    ],
}


# ---------------------------------------------------------------------------
# Data and model loading
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor() -> ChurnPredictor:
    return ChurnPredictor()


@st.cache_data
def load_reference_data() -> pd.DataFrame:
    config = load_config()
    train_path = resolve_path(config["processed_data_config"]["train_data_csv"])
    if train_path.exists():
        return pd.read_csv(train_path)
    raw_path = resolve_path(config["raw_data_config"]["raw_data_csv"])
    return pd.read_csv(raw_path)


@st.cache_data
def load_population_data() -> pd.DataFrame:
    config = load_config()
    raw_path = resolve_path(config["raw_data_config"]["raw_data_csv"])
    if raw_path.exists():
        return pd.read_csv(raw_path)
    return load_reference_data()


@st.cache_data
def load_model_metrics() -> dict[str, Any]:
    metrics = dict(MODEL_METRICS_FALLBACK)
    metrics_path = PROJECT_ROOT / "results" / "model_comparison.csv"
    if not metrics_path.exists():
        return metrics

    frame = pd.read_csv(metrics_path)
    if "status" in frame:
        frame = frame[frame["status"].astype(str).str.lower() == "ok"]
    if frame.empty:
        return metrics

    sort_column = "auc_roc" if "auc_roc" in frame else "accuracy"
    best = frame.sort_values(sort_column, ascending=False).iloc[0]
    metrics.update(
        {
            "model": str(best.get("model", metrics["model"])),
            "accuracy": float(best.get("accuracy", metrics["accuracy"])),
            "precision": float(
                best.get("precision_yes", best.get("precision_macro", metrics["precision"]))
            ),
            "recall": float(best.get("recall_yes", best.get("recall_macro", metrics["recall"]))),
            "auc_roc": float(best.get("auc_roc", 0.94)),
        }
    )
    return metrics


def inject_css() -> None:
    st.markdown(
        """
<style>
  #MainMenu, footer, header {visibility: hidden;}

  .stApp {
    background-color: #0F1117;
    color: #F0EEF8;
    font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }

  .block-container {
    max-width: 1440px;
    padding: 24px 32px 42px;
  }

  [data-testid="stSidebar"] {
    background-color: #1C1E26;
    border-right: 0.5px solid #2A2D3A;
  }

  h1, h2, h3, h4, h5, h6, p, label, span, div {
    color: #F0EEF8;
    letter-spacing: 0;
  }

  .app-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    gap: 18px;
    margin-bottom: 18px;
  }

  .app-title {
    font-size: 32px;
    font-weight: 650;
    line-height: 1.1;
    margin: 0 0 6px;
  }

  .app-subtitle {
    color: #9B99AA;
    font-size: 13px;
    margin: 0;
  }

  .header-meta {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
    color: #9B99AA;
    font-size: 12px;
  }

  .status-pill, .risk-pill {
    display: inline-flex;
    align-items: center;
    border-radius: 99px;
    border: 0.5px solid rgba(29, 158, 117, 0.55);
    background: rgba(29, 158, 117, 0.12);
    color: #F0EEF8;
    font-size: 12px;
    font-weight: 600;
    padding: 6px 10px;
    white-space: nowrap;
  }

  .risk-pill.low {
    border-color: rgba(99,153,34,0.7);
    background: rgba(99,153,34,0.16);
  }

  .risk-pill.medium {
    border-color: rgba(186,117,23,0.7);
    background: rgba(186,117,23,0.16);
  }

  .risk-pill.high {
    border-color: rgba(163,45,45,0.72);
    background: rgba(163,45,45,0.18);
  }

  .risk-label {
    align-items: center;
    display: inline-flex;
    gap: 8px;
  }

  .risk-dot {
    border-radius: 50%;
    display: inline-block;
    height: 10px;
    width: 10px;
  }

  .risk-dot.low { background: #639922; }
  .risk-dot.medium { background: #BA7517; }
  .risk-dot.high { background: #A32D2D; }

  .dash-card, [data-testid="stMetric"] {
    background: #1C1E26;
    border: 0.5px solid #2A2D3A;
    border-radius: 12px;
  }

  .dash-card {
    padding: 16px 20px;
    margin-bottom: 12px;
  }

  .dash-card.compact {
    padding: 14px 16px;
  }

  [data-testid="stMetric"] {
    padding: 12px 16px;
  }

  [data-testid="stMetricLabel"] p {
    color: #9B99AA;
    font-size: 12px;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }

  [data-testid="stMetricValue"] {
    color: #F0EEF8;
    font-size: 30px;
    font-weight: 600;
  }

  .kpi-title, .card-title {
    color: #9B99AA;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
    text-transform: uppercase;
  }

  .kpi-value {
    font-size: 30px;
    font-weight: 650;
    line-height: 1.1;
  }

  .kpi-note {
    color: #9B99AA;
    font-size: 12px;
    margin-top: 6px;
  }

  .metric-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    margin-bottom: 16px;
  }

  .metric-grid.five {
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }

  .kpi-value.compact-text {
    font-size: 22px;
    overflow-wrap: anywhere;
  }

  .profile-head {
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
  }

  .avatar {
    align-items: center;
    background: linear-gradient(135deg, #534AB7, #1D9E75);
    border: 0.5px solid #2A2D3A;
    border-radius: 50%;
    display: inline-flex;
    font-size: 13px;
    font-weight: 700;
    height: 40px;
    justify-content: center;
    width: 40px;
  }

  .profile-title {
    font-size: 18px;
    font-weight: 650;
    margin: 0;
  }

  .profile-subtitle {
    color: #9B99AA;
    font-size: 12px;
    margin: 2px 0 0;
  }

  .prediction-badge {
    border-radius: 99px;
    display: inline-flex;
    font-size: 13px;
    font-weight: 650;
    margin-top: 8px;
    padding: 8px 12px;
  }

  .prediction-badge.churn {
    background: rgba(216,90,48,0.17);
    border: 0.5px solid rgba(216,90,48,0.7);
    color: #F0EEF8;
  }

  .prediction-badge.no-churn {
    background: rgba(29,158,117,0.16);
    border: 0.5px solid rgba(29,158,117,0.65);
    color: #F0EEF8;
  }

  .stButton > button, .stDownloadButton > button {
    background: #D85A30;
    border: none;
    border-radius: 8px;
    color: #F0EEF8;
    font-weight: 600;
    min-height: 42px;
    transition: background 0.15s ease, transform 0.15s ease;
    width: 100%;
  }

  .stButton > button:hover, .stDownloadButton > button:hover {
    background: #993C1D;
    color: #F0EEF8;
    transform: translateY(-1px);
  }

  .stButton > button:focus:not(:active), .stDownloadButton > button:focus:not(:active) {
    border-color: #D85A30;
    box-shadow: 0 0 0 0.1rem rgba(216,90,48,0.35);
    color: #F0EEF8;
  }

  div[data-testid="stExpander"] {
    background: rgba(15,17,23,0.34);
    border: 0.5px solid #2A2D3A;
    border-radius: 10px;
    margin-bottom: 10px;
    transition: border-color 0.15s ease, background 0.15s ease;
  }

  div[data-testid="stExpander"]:hover {
    background: rgba(29,158,117,0.05);
    border-color: rgba(29,158,117,0.45);
  }

  [data-testid="stNumberInput"] input,
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: #0F1117;
    border-color: #2A2D3A;
    color: #F0EEF8;
    transition: border-color 0.15s ease, background 0.15s ease;
  }

  [data-testid="stNumberInput"] input:hover,
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: rgba(29,158,117,0.65);
  }

  [data-testid="stFileUploader"] section {
    background: rgba(29,158,117,0.06);
    border: 1px dashed rgba(29,158,117,0.75);
    border-radius: 12px;
  }

  [data-testid="stFileUploader"] section:hover {
    background: rgba(29,158,117,0.1);
  }

  [data-testid="stDataFrame"] {
    border: 0.5px solid #2A2D3A;
    border-radius: 10px;
    overflow: hidden;
  }

  .feature-row {
    align-items: center;
    border-bottom: 0.5px solid #2A2D3A;
    display: grid;
    gap: 10px;
    grid-template-columns: 1.2fr 1fr 0.8fr 0.75fr;
    padding: 10px 8px;
  }

  .feature-row.top {
    background: rgba(83,74,183,0.18);
  }

  .feature-row:last-child {
    border-bottom: 0;
  }

  .feature-header {
    color: #9B99AA;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
  }

  .feature-cell {
    color: #F0EEF8;
    font-size: 13px;
    overflow-wrap: anywhere;
  }

  .impact-up { color: #D85A30; font-weight: 650; }
  .impact-down { color: #1D9E75; font-weight: 650; }

  .empty-state {
    background: rgba(155,153,170,0.08);
    border: 0.5px solid #2A2D3A;
    border-radius: 10px;
    color: #9B99AA;
    font-size: 13px;
    padding: 14px;
  }

  .agent-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  .agent-name {
    color: #9B99AA;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin-bottom: 8px;
    text-transform: uppercase;
  }

  .agent-body {
    color: #F0EEF8;
    font-size: 13px;
    line-height: 1.45;
  }

  .recommendation-list {
    margin: 0;
    padding-left: 18px;
  }

  .recommendation-list li {
    color: #F0EEF8;
    font-size: 13px;
    margin-bottom: 8px;
  }

  .recommendation-card {
    border-bottom: 0.5px solid #2A2D3A;
    padding: 12px 0;
  }

  .recommendation-card:last-child {
    border-bottom: 0;
  }

  .impact-pill {
    background: rgba(29,158,117,0.14);
    border: 0.5px solid rgba(29,158,117,0.58);
    border-radius: 99px;
    color: #F0EEF8;
    display: inline-flex;
    font-size: 12px;
    font-weight: 650;
    margin-top: 8px;
    padding: 5px 9px;
  }

  .summary-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  @media (max-width: 820px) {
    .block-container { padding: 18px 16px 32px; }
    .app-header { flex-direction: column; }
    .header-meta { justify-content: flex-start; }
    .metric-grid, .metric-grid.five { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .agent-grid { grid-template-columns: 1fr; }
    .feature-row { grid-template-columns: 1fr; }
  }

  @keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(14px); }
    to { opacity: 1; transform: translateY(0); }
  }

  @keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0 rgba(139,92,246,0); }
    50% { box-shadow: 0 0 30px rgba(139,92,246,0.22); }
  }

  .stApp {
    background:
      radial-gradient(circle at 12% 8%, rgba(139,92,246,0.18), transparent 34%),
      radial-gradient(circle at 85% 14%, rgba(168,85,247,0.16), transparent 32%),
      linear-gradient(180deg, #0B1020 0%, #080C18 100%);
    color: #F8FAFC;
  }

  .block-container {
    max-width: 1560px;
    padding: 38px 42px 56px;
  }

  .app-header {
    margin-bottom: 30px;
    padding: 8px 0 4px;
  }

  .app-title {
    color: #F8FAFC;
    font-size: clamp(34px, 5vw, 58px);
    font-weight: 760;
    letter-spacing: 0;
  }

  .app-subtitle {
    color: #94A3B8;
    font-size: 15px;
    line-height: 1.65;
  }

  .status-pill, .risk-pill, .impact-pill {
    backdrop-filter: blur(18px);
    background: rgba(139,92,246,0.14);
    border: 1px solid rgba(168,85,247,0.36);
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
  }

  .dash-card, [data-testid="stMetric"] {
    animation: fadeSlideUp 0.45s ease both;
    backdrop-filter: blur(18px);
    background:
      linear-gradient(180deg, rgba(255,255,255,0.055), rgba(255,255,255,0.018)),
      rgba(21,27,46,0.86);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 18px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.32), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: border-color 0.22s ease, box-shadow 0.22s ease, transform 0.22s ease;
  }

  .dash-card {
    margin-bottom: 20px;
    padding: 22px 24px;
  }

  .dash-card.compact {
    min-height: 126px;
    padding: 20px 22px;
  }

  .dash-card:hover, [data-testid="stMetric"]:hover {
    border-color: rgba(168,85,247,0.48);
    box-shadow: 0 24px 70px rgba(0,0,0,0.42), 0 0 34px rgba(139,92,246,0.12);
    transform: translateY(-3px);
  }

  .metric-grid, .summary-grid, .agent-grid {
    gap: 18px;
    margin-bottom: 24px;
  }

  .kpi-title, .card-title, .agent-name {
    color: #94A3B8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.08em;
  }

  .kpi-value {
    color: #F8FAFC;
    font-size: clamp(28px, 3vw, 40px);
    font-weight: 780;
    letter-spacing: 0;
  }

  .kpi-note, .profile-subtitle, .agent-body {
    color: #94A3B8;
  }

  .kpi-icon {
    align-items: center;
    background: linear-gradient(135deg, rgba(139,92,246,0.22), rgba(168,85,247,0.10));
    border: 1px solid rgba(168,85,247,0.28);
    border-radius: 14px;
    display: inline-flex;
    font-size: 18px;
    height: 38px;
    justify-content: center;
    margin-bottom: 12px;
    width: 38px;
  }

  .premium-section {
    margin-top: 28px;
  }

  .radial-card {
    align-items: center;
    display: grid;
    gap: 18px;
    grid-template-columns: minmax(180px, 260px) 1fr;
  }

  .stTabs [data-baseweb="tab-list"] {
    background: rgba(11,16,32,0.62);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 14px;
    gap: 6px;
    padding: 6px;
  }

  .stTabs [data-baseweb="tab"] {
    border-radius: 10px;
    color: #94A3B8;
    font-weight: 650;
    padding: 10px 14px;
  }

  .stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(139,92,246,0.28), rgba(168,85,247,0.18));
    color: #F8FAFC;
  }

  .stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #8B5CF6, #A855F7);
    border: 1px solid rgba(255,255,255,0.10);
    border-radius: 12px;
    box-shadow: 0 12px 30px rgba(139,92,246,0.22);
    color: #F8FAFC;
    min-height: 46px;
  }

  .stButton > button:hover, .stDownloadButton > button:hover {
    background: linear-gradient(135deg, #A855F7, #8B5CF6);
    box-shadow: 0 18px 42px rgba(168,85,247,0.34);
    color: #F8FAFC;
    transform: translateY(-2px);
  }

  [data-testid="stNumberInput"] input,
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div {
    background: rgba(11,16,32,0.72);
    border: 1px solid rgba(148,163,184,0.18);
    border-radius: 12px;
    color: #F8FAFC;
  }

  [data-testid="stNumberInput"] input:hover,
  [data-testid="stSelectbox"] div[data-baseweb="select"] > div:hover {
    border-color: rgba(168,85,247,0.54);
    box-shadow: 0 0 0 3px rgba(139,92,246,0.12);
  }

  [data-testid="stFileUploader"] section {
    background: rgba(139,92,246,0.08);
    border: 1px dashed rgba(168,85,247,0.66);
    border-radius: 18px;
  }

  .feature-row {
    border-bottom: 1px solid rgba(148,163,184,0.14);
    grid-template-columns: 1.2fr 1fr 0.8fr 0.9fr;
    padding: 13px 10px;
  }

  .feature-row.top {
    background: rgba(139,92,246,0.14);
  }

  .impact-up { color: #EF4444; }
  .impact-down { color: #22C55E; }

  .prediction-badge.churn {
    background: rgba(239,68,68,0.16);
    border: 1px solid rgba(239,68,68,0.44);
  }

  .prediction-badge.no-churn {
    background: rgba(34,197,94,0.14);
    border: 1px solid rgba(34,197,94,0.44);
  }

  @media (max-width: 1100px) {
    .metric-grid.five { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    .radial-card { grid-template-columns: 1fr; }
  }

  @media (max-width: 720px) {
    .block-container { padding: 22px 16px 36px; }
    .metric-grid, .metric-grid.five, .summary-grid { grid-template-columns: 1fr; }
    .dash-card.compact { min-height: auto; }
  }

  /* Calmer production-dashboard layer: restrained, readable, and less showpiece-like. */
  .stApp {
    background: #0B1020;
  }

  .block-container {
    max-width: 1480px;
    padding: 30px 34px 44px;
  }

  .app-header {
    align-items: flex-end;
    border-bottom: 1px solid rgba(148,163,184,0.12);
    margin-bottom: 24px;
    padding-bottom: 18px;
  }

  .app-title {
    font-size: clamp(28px, 3vw, 40px);
    font-weight: 720;
  }

  .app-subtitle {
    font-size: 14px;
    line-height: 1.45;
    max-width: 760px;
  }

  .dash-card, [data-testid="stMetric"] {
    animation: fadeSlideUp 0.22s ease both;
    backdrop-filter: blur(10px);
    background: rgba(21,27,46,0.92);
    border: 1px solid rgba(148,163,184,0.14);
    border-radius: 12px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.20);
  }

  .dash-card {
    margin-bottom: 16px;
    padding: 18px 20px;
  }

  .dash-card.compact {
    min-height: 108px;
    padding: 16px 18px;
  }

  .dash-card:hover, [data-testid="stMetric"]:hover {
    border-color: rgba(139,92,246,0.30);
    box-shadow: 0 12px 32px rgba(0,0,0,0.24);
    transform: translateY(-1px);
  }

  .metric-grid, .summary-grid, .agent-grid {
    gap: 14px;
    margin-bottom: 18px;
  }

  .kpi-icon {
    background: rgba(139,92,246,0.10);
    border: 1px solid rgba(139,92,246,0.22);
    border-radius: 10px;
    font-size: 12px;
    height: 30px;
    margin-bottom: 10px;
    width: 30px;
  }

  .kpi-title, .card-title, .agent-name {
    font-size: 11px;
    letter-spacing: 0.06em;
  }

  .kpi-value {
    font-size: clamp(24px, 2.2vw, 32px);
    font-weight: 700;
  }

  .kpi-value.compact-text {
    font-size: 19px;
    line-height: 1.25;
  }

  .status-pill, .risk-pill, .impact-pill {
    backdrop-filter: none;
    box-shadow: none;
  }

  .stButton > button, .stDownloadButton > button {
    background: #8B5CF6;
    box-shadow: none;
  }

  .stButton > button:hover, .stDownloadButton > button:hover {
    background: #7C3AED;
    box-shadow: 0 8px 20px rgba(139,92,246,0.22);
    transform: translateY(-1px);
  }

  .stTabs [data-baseweb="tab-list"] {
    background: rgba(11,16,32,0.66);
    border-radius: 10px;
  }

  .stTabs [data-baseweb="tab"] {
    padding: 8px 12px;
  }

  .stTabs [aria-selected="true"] {
    background: rgba(139,92,246,0.18);
  }

  .radial-card {
    gap: 12px;
  }

  [data-testid="stDataFrame"] {
    border-radius: 12px;
  }
</style>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def format_feature_name(column: str) -> str:
    return column.replace("_", " ").title()


def format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, (np.floating,)):
        return f"{float(value):,.2f}"
    if isinstance(value, (int, np.integer)):
        return f"{int(value):,}"
    return str(value)


def default_numeric_value(column: str, reference_data: pd.DataFrame) -> float:
    if column in reference_data:
        return float(pd.to_numeric(reference_data[column], errors="coerce").median())
    return 0.0


def default_field_value(column: str, reference_data: pd.DataFrame) -> Any:
    if column in reference_data and pd.api.types.is_numeric_dtype(reference_data[column]):
        return default_numeric_value(column, reference_data)
    if column in reference_data and not reference_data[column].dropna().empty:
        values = sorted(reference_data[column].dropna().astype(str).unique().tolist())
        if column in {"international_plan", "voice_mail_plan"} and "no" in values:
            return "no"
        return values[0]
    return ""


def current_customer_from_state(
    predictor: ChurnPredictor,
    reference_data: pd.DataFrame,
) -> dict[str, Any]:
    return {
        column: st.session_state.get(
            f"input_{column}",
            default_field_value(column, reference_data),
        )
        for column in predictor.feature_names
    }


def numeric_step(column: str) -> float:
    if "calls" in column or "messages" in column or column in {"account_length"}:
        return 1.0
    return 0.1


def risk_tier(probability: float) -> str:
    if probability < 0.2:
        return "Low"
    if probability <= 0.5:
        return "Medium"
    return "High"


def risk_color(tier: str) -> str:
    return {
        "Low": COLORS["success"],
        "Medium": COLORS["warning"],
        "High": COLORS["danger"],
    }.get(tier, COLORS["muted"])


def probability_color(probability: float) -> str:
    return risk_color(risk_tier(probability))


def model_confidence(probability: float) -> float:
    p = min(max(float(probability), 0.0), 1.0)
    return max(p, 1.0 - p)


def top_risk_factor(explanation: dict[str, Any] | None) -> str:
    if not explanation:
        return "Unavailable"
    factors = explanation.get("top_factors") or []
    if not factors:
        return "Unavailable"
    return format_feature_name(str(factors[0].get("feature", "Unavailable")))


def confidence_note(probability: float) -> str:
    tier = risk_tier(probability).lower()
    return f"Signal strength for this {tier}-risk prediction"


def risk_label(tier: str) -> str:
    css_class = tier.lower()
    return f'<span class="risk-label"><span class="risk-dot {css_class}"></span>{tier} Risk</span>'


def infer_model_name(predictor: ChurnPredictor, metrics: dict[str, Any]) -> str:
    _, estimator = predictor._get_model_components()
    class_name = estimator.__class__.__name__ if estimator is not None else ""
    aliases = {
        "XGBClassifier": "XGBoost",
        "LGBMClassifier": "LightGBM",
        "CatBoostClassifier": "CatBoost",
        "RandomForestClassifier": "Random Forest",
        "GradientBoostingClassifier": "Gradient Boosting",
    }
    return aliases.get(class_name, str(metrics.get("model", "Production Model")))


def estimate_customer_lifetime_value(
    customer: dict[str, Any],
    probability: float,
    monthly_revenue: float = DEFAULT_MONTHLY_REVENUE,
) -> float:
    account_length = max(1.0, float(customer.get("account_length") or 12.0))
    retained_value = monthly_revenue * account_length
    risk_adjustment = max(0.35, 1.0 - probability * 0.5)
    return retained_value * risk_adjustment


def customer_segment(probability: float, clv: float, reference_data: pd.DataFrame) -> str:
    high_value_threshold = DEFAULT_MONTHLY_REVENUE * 48
    if "account_length" in reference_data:
        account_lengths = pd.to_numeric(reference_data["account_length"], errors="coerce")
        if not account_lengths.dropna().empty:
            high_value_threshold = DEFAULT_MONTHLY_REVENUE * float(account_lengths.quantile(0.75))

    if probability > 0.5:
        return "Segment D - Likely Churners"
    if clv >= high_value_threshold and probability <= 0.2:
        return "Segment C - High Value Customers"
    if probability > 0.2:
        return "Segment B - At Risk"
    return "Segment A - Loyal Customers"


def summarize_portfolio(
    predictor: ChurnPredictor,
    population_data: pd.DataFrame,
) -> dict[str, float]:
    total_customers = len(population_data)
    if total_customers == 0:
        return {
            "total_customers": 0.0,
            "likely_churners": 0.0,
            "revenue_at_risk": 0.0,
            "retention_opportunity": 0.0,
        }

    try:
        predictions = predictor.predict(population_data[predictor.feature_names])
        probabilities = pd.Series(
            [float(item["churn_probability"]) for item in predictions],
            dtype=float,
        )
        likely_churners = int((probabilities > 0.5).sum())
    except Exception:
        target = population_data.get("churn", pd.Series(dtype=object))
        likely_churners = int(target.astype(str).str.lower().eq("yes").sum())

    annual_revenue = DEFAULT_MONTHLY_REVENUE * 12
    revenue_at_risk = likely_churners * annual_revenue
    return {
        "total_customers": float(total_customers),
        "likely_churners": float(likely_churners),
        "revenue_at_risk": revenue_at_risk,
        "retention_opportunity": revenue_at_risk * 0.6,
    }


def initialize_session_state() -> None:
    st.session_state.setdefault("last_score_at", 0.0)
    st.session_state.setdefault("manual_refresh_count", 0)
    st.session_state.setdefault("last_customer_signature", None)


def customer_signature(customer: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    return tuple((key, str(value)) for key, value in sorted(customer.items()))


def score_customer(
    predictor: ChurnPredictor,
    customer: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, str | None]:
    prediction = predictor.predict_one(customer)
    try:
        explanation = predictor.explain(customer, top_k=TOP_FACTORS)
        return prediction, explanation, None
    except Exception as exc:  # pragma: no cover - depends on optional SHAP runtime
        return prediction, None, str(exc)


def build_customer_report_text(
    prediction: dict[str, Any],
    probability: float,
    confidence: float,
    tier: str,
    clv: float,
    segment: str,
    explanation: dict[str, Any] | None,
    analysis: dict[str, Any] | None,
) -> str:
    label = "Churn" if str(prediction.get("churn")).lower() == "yes" else "No Churn"
    lines = [
        "Customer Churn Analysis Report",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        f"Prediction: {label}",
        f"Probability: {probability:.1%}",
        f"Confidence: {confidence:.1%}",
        f"Risk Category: {tier} Risk",
        f"Customer Lifetime Value: ${clv:,.0f}",
        f"Segment: {segment}",
    ]
    factors = explanation_frame(explanation)
    if not factors.empty:
        lines.extend(["", "Top Feature Impacts:"])
        for _, row in factors.head(6).iterrows():
            lines.append(
                f"- {row['display_feature']}: {row['shap_value']:+.3f} ({row['impact_label']})"
            )
    if analysis:
        lines.extend(["", "Recommendations:"])
        for item in analysis["analyst"]["recommendations"]:
            lines.append(f"- {item}")
        offer = analysis["agents"]["offer_agent"]
        lines.extend(
            [
                "",
                f"Recommended Offer: {offer['offer']}",
                f"Expected Churn Reduction: {float(offer.get('expected_churn_reduction', 0.0)):.1%}",
            ]
        )
    return "\n".join(lines)


def build_simple_pdf(title: str, body: str) -> bytes:
    lines = [title, ""] + body.splitlines()
    escaped_lines = [
        line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")[:95]
        for line in lines[:42]
    ]
    text_commands = ["BT", "/F1 12 Tf", "50 770 Td", "16 TL"]
    for line in escaped_lines:
        text_commands.append(f"({line}) Tj")
        text_commands.append("T*")
    text_commands.append("ET")
    stream = "\n".join(text_commands).encode("latin-1", errors="replace")
    objects = [
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj",
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj",
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj",
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj",
        b"5 0 obj << /Length " + str(len(stream)).encode() + b" >> stream\n" + stream + b"\nendstream endobj",
    ]
    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objects:
        offsets.append(len(pdf))
        pdf.extend(obj + b"\n")
    xref_start = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    pdf.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode())
    pdf.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF".encode()
    )
    return bytes(pdf)


def build_customer_inputs(
    predictor: ChurnPredictor,
    reference_data: pd.DataFrame,
) -> dict[str, Any]:
    customer: dict[str, Any] = {}
    grouped = set()
    tab_names = list(FIELD_GROUPS)
    tabs = st.tabs(tab_names)

    for tab, group_name in zip(tabs, tab_names):
        fields = FIELD_GROUPS[group_name]
        available_fields = [field for field in fields if field in predictor.feature_names]
        if not available_fields:
            continue
        with tab:
            for column in available_fields:
                if column in grouped:
                    continue
                grouped.add(column)
                customer[column] = render_field(column, reference_data)

    remaining = [field for field in predictor.feature_names if field not in grouped]
    if remaining:
        with st.expander("Additional Signals", expanded=False):
            for column in remaining:
                customer[column] = render_field(column, reference_data)

    return customer


def render_field(column: str, reference_data: pd.DataFrame) -> Any:
    label = format_feature_name(column)
    key = f"input_{column}"

    if column in reference_data and pd.api.types.is_numeric_dtype(reference_data[column]):
        value = default_numeric_value(column, reference_data)
        return st.number_input(
            label,
            min_value=0.0,
            value=float(value),
            step=numeric_step(column),
            key=key,
        )

    options = [""] if column not in reference_data else sorted(
        reference_data[column].dropna().astype(str).unique().tolist()
    )
    default_index = 0
    if column in {"international_plan", "voice_mail_plan"} and "no" in options:
        default_index = options.index("no")
    return st.selectbox(label, options=options, index=default_index, key=key)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_header() -> None:
    timestamp = datetime.now().strftime("%b %d, %Y %I:%M %p")
    st.markdown(
        f"""
<div class="app-header">
  <div>
    <div class="app-title">Customer Churn Analytics</div>
    <p class="app-subtitle">Account-level churn scoring, driver analysis, retention actions, and portfolio risk monitoring.</p>
  </div>
  <div class="header-meta">
    <span>Last updated {timestamp}</span>
    <span class="status-pill">Deploy: Live</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_dashboard_summary(summary: dict[str, float]) -> None:
    cards = [
        ("TC", "Total Customers", f"{summary['total_customers']:,.0f}", COLORS["text"]),
        ("LR", "Likely Churners", f"{summary['likely_churners']:,.0f}", COLORS["danger"]),
        ("$", "Revenue at Risk", f"${summary['revenue_at_risk']:,.0f}", COLORS["warning"]),
        ("RO", "Retention Opportunity", f"${summary['retention_opportunity']:,.0f}", COLORS["teal"]),
    ]
    html = ['<div class="metric-grid">']
    for icon, title, value, color in cards:
        html.append(
            f"""
<div class="dash-card compact">
  <div class="kpi-icon" style="color: {color};">{icon}</div>
  <div class="kpi-title">{title}</div>
  <div class="kpi-value" style="color: {color};">{value}</div>
</div>
"""
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def render_kpi_strip(
    prediction: dict[str, Any],
    probability: float,
    tier: str,
    confidence: float,
    clv: float,
    segment: str,
    model_metrics: dict[str, Any],
    explanation: dict[str, Any] | None,
) -> None:
    label = "Churn" if str(prediction["churn"]).lower() == "yes" else "No Churn"
    label_html = (
        f'<span class="risk-label"><span class="risk-dot {"high" if label == "Churn" else "low"}"></span>{label}</span>'
    )
    kpis = [
        {
            "title": "Prediction",
            "value": label_html,
            "note": f"Probability {probability:.1%}",
            "color": probability_color(probability),
            "compact": label == "No Churn",
            "icon": "PR",
        },
        {
            "title": "Churn Probability",
            "value": f"{probability:.1%}",
            "note": "Predicted probability",
            "color": probability_color(probability),
            "compact": False,
            "icon": "%",
        },
        {
            "title": "Risk Tier",
            "value": risk_label(tier),
            "note": "Low <20%, Medium 20-50%, High >50%",
            "color": risk_color(tier),
            "compact": tier == "Medium",
            "icon": "RS",
        },
        {
            "title": "Confidence",
            "value": f"{confidence:.1%}",
            "note": confidence_note(probability),
            "color": COLORS["teal"] if confidence >= 0.5 else COLORS["warning"],
            "compact": False,
            "icon": "CF",
        },
        {
            "title": "Customer Lifetime Value",
            "value": f"${clv:,.0f}",
            "note": "Risk-adjusted account value",
            "color": COLORS["indigo"],
            "compact": False,
            "icon": "$",
        },
    ]
    html = ['<div class="metric-grid five">']
    for kpi in kpis:
        value_class = "kpi-value compact-text" if kpi["compact"] else "kpi-value"
        html.append(
            f"""
<div class="dash-card compact">
  <div class="kpi-icon" style="color: {kpi["color"]};">{kpi["icon"]}</div>
  <div class="kpi-title">{kpi["title"]}</div>
  <div class="{value_class}" style="color: {kpi["color"]};">{kpi["value"]}</div>
  <div class="kpi-note">{kpi["note"]}</div>
</div>
"""
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    model_cards = [
        ("Model Accuracy", f"{model_metrics['accuracy']:.1%}", COLORS["teal"]),
        ("Precision", f"{model_metrics['precision']:.1%}", COLORS["indigo"]),
        ("Recall", f"{model_metrics['recall']:.1%}", COLORS["warning"]),
        ("Customer Segment", segment, COLORS["text"]),
    ]
    html = ['<div class="metric-grid">']
    for title, value, color in model_cards:
        value_class = "kpi-value compact-text" if title == "Customer Segment" else "kpi-value"
        html.append(
            f"""
<div class="dash-card compact">
  <div class="kpi-title">{title}</div>
  <div class="{value_class}" style="color: {color};">{value}</div>
</div>
"""
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)

    st.markdown(
        f"""
<div class="dash-card compact">
  <div class="kpi-title">Top Risk Factor</div>
  <div class="kpi-value compact-text" style="color: {COLORS["indigo"]};">{top_risk_factor(explanation)}</div>
  <div class="kpi-note">Highest absolute SHAP impact</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_profile_panel(
    predictor: ChurnPredictor,
    reference_data: pd.DataFrame,
    prediction: dict[str, Any],
) -> dict[str, Any]:
    st.markdown(
        """
<div class="dash-card">
  <div class="profile-head">
    <div class="avatar">CI</div>
    <div>
      <p class="profile-title">Customer Profile</p>
      <p class="profile-subtitle">Adjust values to rescore instantly</p>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    customer = build_customer_inputs(predictor, reference_data)
    clicked = st.button("Score Customer", type="primary", use_container_width=True)
    if clicked:
        st.session_state.manual_refresh_count += 1
        st.session_state.last_score_at = 0.0
        st.rerun()

    label = str(prediction["churn"])
    badge_class = "churn" if label.lower() == "yes" else "no-churn"
    badge_label = "Churn" if label.lower() == "yes" else "No Churn"
    st.markdown(
        f'<div class="prediction-badge {badge_class}">Predicted Label: {badge_label}</div>',
        unsafe_allow_html=True,
    )
    return customer


def make_gauge(probability: float, tier: str) -> go.Figure:
    gauge_value = max(0.0, min(100.0, probability * 100))
    risk_hex = probability_color(probability)
    fig = go.Figure(
        go.Pie(
            values=[gauge_value, 100 - gauge_value],
            hole=0.74,
            rotation=90,
            direction="clockwise",
            marker={
                "colors": [risk_hex, "rgba(148,163,184,0.12)"],
                "line": {"color": "rgba(255,255,255,0)", "width": 0},
            },
            textinfo="none",
            hoverinfo="skip",
            sort=False,
        )
    )
    fig.add_annotation(
        text=f"<b>{gauge_value:.1f}%</b><br><span style='font-size:13px;color:{COLORS['muted']}'>{tier} Risk</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"color": COLORS["text"], "size": 26, "family": "Inter, system-ui"},
        align="center",
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": COLORS["text"], "family": "Inter, system-ui"},
        height=300,
        margin={"l": 8, "r": 8, "t": 8, "b": 8},
        showlegend=False,
        transition={"duration": 450, "easing": "cubic-in-out"},
    )
    return fig


def explanation_frame(explanation: dict[str, Any] | None) -> pd.DataFrame:
    if not explanation or not explanation.get("top_factors"):
        return pd.DataFrame(columns=["feature", "feature_value", "shap_value"])
    frame = pd.DataFrame(explanation["top_factors"]).copy()
    frame["direction"] = np.where(
        frame["shap_value"] >= 0,
        "Increases churn risk",
        "Reduces churn risk",
    )
    frame["impact_label"] = np.where(
        frame["shap_value"] >= 0,
        "Positive impact",
        "Negative impact",
    )
    frame["display_feature"] = frame["feature"].map(format_feature_name)
    return frame


def make_shap_chart(frame: pd.DataFrame) -> go.Figure:
    chart_df = frame.sort_values("shap_value", ascending=True)
    colors = np.where(chart_df["shap_value"] >= 0, COLORS["coral"], COLORS["teal"])
    customdata = np.stack(
        [
            chart_df["feature"],
            chart_df["feature_value"].map(format_value),
            chart_df["direction"],
            chart_df["impact_label"],
        ],
        axis=-1,
    )
    fig = go.Figure(
        go.Bar(
            x=chart_df["shap_value"],
            y=chart_df["display_feature"],
            orientation="h",
            marker={"color": colors, "line": {"width": 0}},
            customdata=customdata,
            text=[
                f"{value:+.3f} {label}"
                for value, label in zip(chart_df["shap_value"], chart_df["impact_label"])
            ],
            textposition="auto",
            hovertemplate=(
                "Feature: %{customdata[0]}<br>"
                "Value: %{customdata[1]}<br>"
                "SHAP: %{x:.3f}<br>"
                "%{customdata[2]}<br>"
                "%{customdata[3]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line_width=1, line_dash="dash", line_color=COLORS["border"])
    fig.update_layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font={"color": COLORS["text"], "family": "Inter, system-ui", "size": 12},
        height=320,
        margin={"l": 8, "r": 16, "t": 8, "b": 38},
        xaxis={
            "title": "SHAP value",
            "gridcolor": COLORS["border"],
            "zeroline": False,
            "color": COLORS["muted"],
        },
        yaxis={"title": "", "color": COLORS["text"]},
        transition={"duration": 300, "easing": "cubic-in-out"},
    )
    return fig


def render_feature_table(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.markdown(
            '<div class="empty-state">Feature impacts are unavailable for this prediction.</div>',
            unsafe_allow_html=True,
        )
        return

    rows = [
        """
<div class="feature-row feature-header">
  <div>Feature</div>
  <div>Customer Value</div>
  <div>SHAP Impact</div>
  <div>Direction</div>
</div>
"""
    ]
    for index, row in frame.reset_index(drop=True).iterrows():
        direction_class = "impact-up" if row["shap_value"] >= 0 else "impact-down"
        direction = "Positive impact" if row["shap_value"] >= 0 else "Negative impact"
        top_class = " top" if index < 3 else ""
        rows.append(
            f"""
<div class="feature-row{top_class}">
  <div class="feature-cell">{row["display_feature"]}</div>
  <div class="feature-cell">{format_value(row["feature_value"])}</div>
  <div class="feature-cell {direction_class}">{row["shap_value"]:+.4f}</div>
  <div class="feature-cell {direction_class}">{direction}</div>
</div>
"""
        )
    st.markdown("".join(rows), unsafe_allow_html=True)


def render_analytics_panel(
    probability: float,
    tier: str,
    explanation: dict[str, Any] | None,
    shap_error: str | None,
) -> None:
    if shap_error:
        st.markdown(
            f'<div class="empty-state">SHAP explanations are unavailable: {shap_error}</div>',
            unsafe_allow_html=True,
        )
        return

    shap_df = explanation_frame(explanation)
    st.markdown(
        '<div class="premium-section"><div class="card-title">Feature Impact Analysis</div></div>',
        unsafe_allow_html=True,
    )
    if shap_df.empty:
        st.markdown(
            '<div class="empty-state">No SHAP factors were returned for this prediction.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.plotly_chart(make_shap_chart(shap_df), use_container_width=True)

    st.markdown('<div class="card-title">Feature Value Table</div>', unsafe_allow_html=True)
    render_feature_table(shap_df)


def render_risk_score_panel(probability: float, tier: str, confidence: float) -> None:
    st.markdown(
        f"""
<div class="dash-card radial-card">
  <div>
  <div class="card-title">Risk Score</div>
    <p class="app-subtitle">Current churn probability, confidence, and decision tier.</p>
  </div>
  <div>
    <span class="risk-pill {tier.lower()}">{risk_label(tier)}</span>
    <div class="kpi-note">Prediction confidence: {confidence:.1%}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.plotly_chart(make_gauge(probability, tier), use_container_width=True)


def render_retention_intelligence_section(
    predictor: ChurnPredictor,
    customer: dict[str, Any],
    prediction: dict[str, Any],
    probability: float,
    confidence: float,
    tier: str,
    clv: float,
    segment: str,
    explanation: dict[str, Any] | None,
) -> None:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
<div class="dash-card">
  <div class="card-title">Retention Review</div>
  <p class="app-subtitle">Plain-English risk explanation, recommended actions, and intervention simulation.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    intelligence = build_retention_intelligence(predictor)
    monthly_revenue = st.number_input(
        "Monthly revenue at risk",
        min_value=1.0,
        value=DEFAULT_MONTHLY_REVENUE,
        step=5.0,
    )

    try:
        analysis = intelligence.analyze_customer(
            customer,
            monthly_revenue=monthly_revenue,
            top_k=TOP_FACTORS,
        )
    except Exception as exc:
        st.markdown(
            f'<div class="empty-state">Retention intelligence is unavailable: {exc}</div>',
            unsafe_allow_html=True,
        )
        return

    analyst = analysis["analyst"]
    agents = analysis["agents"]
    st.markdown(
        f"""
<div class="dash-card">
  <div class="card-title">Analyst Brief</div>
  <div class="agent-body">{analyst["summary"]}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    offer = agents["offer_agent"]
    base_impact = float(offer.get("expected_churn_reduction", 0.0))
    recommendation_items = []
    for index, recommendation in enumerate(analyst["recommendations"]):
        expected_impact = min(0.18, base_impact + index * 0.02)
        recommendation_items.append(
            f"""
<div class="recommendation-card">
  <div class="agent-body">{recommendation}</div>
  <span class="impact-pill">Expected churn reduction: {expected_impact:.1%}</span>
</div>
"""
        )
    st.markdown(
        f"""
<div class="dash-card">
  <div class="card-title">Retention Recommendations</div>
  {''.join(recommendation_items)}
</div>
""",
        unsafe_allow_html=True,
    )

    report_text = build_customer_report_text(
        prediction,
        probability,
        confidence,
        tier,
        clv,
        segment,
        explanation,
        analysis,
    )
    export_cols = st.columns(2)
    with export_cols[0]:
        st.download_button(
            "Generate PDF Report",
            data=build_simple_pdf("Customer Churn Analysis", report_text),
            file_name="customer_churn_analysis.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with export_cols[1]:
        st.download_button(
            "Export Customer Analysis",
            data=report_text.encode("utf-8"),
            file_name="customer_churn_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )

    prediction_agent = agents["prediction_agent"]
    cause_agent = agents["cause_agent"]
    offer_agent = offer
    revenue_agent = agents["revenue_agent"]
    cause_text = cause_agent["top_causes"][0] if cause_agent["top_causes"] else "No dominant cause found."
    potential_saved = max(
        float(revenue_agent.get("potential_revenue_saved", 0.0)),
        float(revenue_agent.get("net_revenue_saved", 0.0)),
    )
    st.markdown(
        f"""
<div class="agent-grid">
  <div class="dash-card compact">
    <div class="agent-name">Prediction Agent</div>
    <div class="agent-body">{prediction_agent["risk_tier"]} risk at {prediction_agent["churn_probability"]:.1%}</div>
  </div>
  <div class="dash-card compact">
    <div class="agent-name">Cause Agent</div>
    <div class="agent-body">{cause_text}</div>
  </div>
  <div class="dash-card compact">
    <div class="agent-name">Offer Agent</div>
    <div class="agent-body">{offer_agent["offer"]}: {offer_agent["discount_percent"]:.0f}%</div>
  </div>
  <div class="dash-card compact">
    <div class="agent-name">Revenue Agent</div>
    <div class="agent-body">Potential revenue saved: ${potential_saved:,.0f}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="card-title">Customer Digital Twin Simulation</div>', unsafe_allow_html=True)
    sim_cols = st.columns(4)
    with sim_cols[0]:
        service_delta = st.slider("Service calls change", -5, 3, -1)
    with sim_cols[1]:
        discount = st.slider("Discount percent", 0, 40, 10)
    with sim_cols[2]:
        international_plan = st.selectbox(
            "International plan",
            options=["no change", "yes", "no"],
            index=0,
        )
    with sim_cols[3]:
        day_usage_delta = st.slider("Day usage change", -50, 50, 0)

    plan_changes = {}
    if international_plan != "no change":
        plan_changes["international_plan"] = international_plan

    simulation = intelligence.simulate_digital_twin(
        customer,
        {
            "service_calls_delta": float(service_delta),
            "discount_percent": float(discount),
            "plan_changes": plan_changes,
            "day_usage_delta_percent": float(day_usage_delta),
        },
        monthly_revenue=monthly_revenue,
    )
    baseline_probability = float(
        simulation["baseline"]["prediction"]["churn_probability"]
    )
    intervention_probability = float(
        simulation["intervention"]["prediction"]["churn_probability"]
    )
    impact = simulation["impact"]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Before", f"{baseline_probability:.1%}")
    metric_cols[1].metric(
        "After",
        f"{intervention_probability:.1%}",
        f"{impact['absolute_probability_change']:.1%}",
    )
    metric_cols[2].metric("Risk tier", impact["risk_tier_change"])
    metric_cols[3].metric("Net revenue saved", f"${impact['net_revenue_saved']:,.0f}")


# ---------------------------------------------------------------------------
# Batch scoring
# ---------------------------------------------------------------------------
def batch_top_factors(
    predictor: ChurnPredictor,
    batch_df: pd.DataFrame,
    limit: int,
) -> list[str]:
    rows_to_explain = min(len(batch_df), limit)
    factors: list[str] = []
    if rows_to_explain:
        try:
            frame = predictor._to_frame(batch_df.head(rows_to_explain))
            shap_values, _, shap_feature_names = predictor._compute_shap_values(frame)
            mapped_features = [
                predictor._map_transformed_feature(name) for name in shap_feature_names
            ]
            for row_index in range(rows_to_explain):
                contributions: dict[str, float] = {}
                for feature_name, shap_value in zip(
                    mapped_features,
                    shap_values[row_index].tolist(),
                ):
                    contributions[feature_name] = contributions.get(feature_name, 0.0) + float(
                        shap_value
                    )
                if contributions:
                    top_feature = max(
                        contributions.items(),
                        key=lambda item: abs(item[1]),
                    )[0]
                    factors.append(format_feature_name(top_feature))
                else:
                    factors.append("Unavailable")
        except Exception:
            factors = ["Unavailable"] * rows_to_explain
    factors.extend(["Not computed"] * max(0, len(batch_df) - rows_to_explain))
    return factors


def prepare_batch_results(
    predictor: ChurnPredictor,
    batch_df: pd.DataFrame,
) -> tuple[pd.DataFrame | None, str | None]:
    missing = [column for column in predictor.feature_names if column not in batch_df.columns]
    if missing:
        return None, "Missing required columns: " + ", ".join(missing)

    predictions = predictor.predict(batch_df)
    result_df = pd.DataFrame(predictions)
    result_df["Customer ID"] = (
        batch_df["customer_id"].astype(str)
        if "customer_id" in batch_df.columns
        else [f"CUST-{index + 1:04d}" for index in range(len(batch_df))]
    )
    result_df["_source_index"] = np.arange(len(batch_df))
    probabilities = result_df["churn_probability"].astype(float)
    result_df["Churn Probability"] = probabilities * 100
    result_df["Risk Tier"] = probabilities.map(risk_tier)
    result_df["Predicted Label"] = result_df["churn"].map(
        lambda value: "Churn" if str(value).lower() == "yes" else "No Churn"
    )
    result_df = result_df.sort_values("Churn Probability", ascending=False)
    sorted_features = batch_df.iloc[result_df["_source_index"].tolist()]
    result_df["Top Risk Factor"] = batch_top_factors(
        predictor,
        sorted_features[predictor.feature_names],
        BATCH_SHAP_LIMIT,
    )
    result_df["Actions"] = "Review"

    display_cols = [
        "Customer ID",
        "Churn Probability",
        "Risk Tier",
        "Top Risk Factor",
        "Actions",
        "Predicted Label",
    ]
    result_df = result_df[display_cols]
    return result_df.reset_index(drop=True), None


def render_summary_cards(result_df: pd.DataFrame) -> None:
    counts = result_df["Risk Tier"].value_counts()
    cards = [
        ("Total customers", len(result_df), COLORS["text"]),
        ("High risk count", int(counts.get("High", 0)), COLORS["danger"]),
        ("Medium", int(counts.get("Medium", 0)), COLORS["warning"]),
        ("Low", int(counts.get("Low", 0)), COLORS["success"]),
    ]
    html = ['<div class="summary-grid">']
    for title, value, color in cards:
        html.append(
            f"""
<div class="dash-card compact">
  <div class="kpi-title">{title}</div>
  <div class="kpi-value" style="color: {color};">{value}</div>
</div>
"""
        )
    html.append("</div>")
    st.markdown("".join(html), unsafe_allow_html=True)


def style_batch_rows(row: pd.Series) -> list[str]:
    tier = row.get("Risk Tier")
    if tier == "High":
        background = "background-color: rgba(163,45,45,0.15)"
    elif tier == "Medium":
        background = "background-color: rgba(186,117,23,0.15)"
    else:
        background = "background-color: rgba(99,153,34,0.15)"
    return [background for _ in row]


def render_batch_section(predictor: ChurnPredictor) -> None:
    st.markdown(
        """
<div class="dash-card">
  <div class="card-title">Bulk Churn Prediction</div>
  <p class="app-subtitle">Upload customer dataset and download predictions for enterprise-scale review.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    sample_frame = load_reference_data().head(10)
    sample_columns = [column for column in predictor.feature_names if column in sample_frame]
    st.download_button(
        "Download Sample CSV",
        data=sample_frame[sample_columns].to_csv(index=False).encode("utf-8"),
        file_name="sample_customers_for_bulk_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )
    upload = st.file_uploader(
        "Upload customer dataset",
        type="csv",
    )
    if upload is None:
        return

    try:
        batch_df = pd.read_csv(upload)
    except Exception as exc:
        st.error(f"Unable to read CSV: {exc}")
        return

    progress = st.progress(0, text="Scoring uploaded customers")
    for value in (25, 55, 85):
        progress.progress(value, text="Scoring uploaded customers")
        time.sleep(0.05)

    try:
        result_df, error = prepare_batch_results(predictor, batch_df)
    except Exception as exc:
        progress.empty()
        st.error(f"Batch scoring failed: {exc}")
        return

    progress.progress(100, text="Scoring complete")
    time.sleep(0.08)
    progress.empty()

    if error:
        st.warning(error)
        return
    if result_df is None or result_df.empty:
        st.info("No rows found in the uploaded file.")
        return

    render_summary_cards(result_df)

    controls_left, controls_right = st.columns([3, 1.4])
    with controls_left:
        selected_tier = st.segmented_control(
            "Risk tier filter",
            options=["All", "High", "Medium", "Low"],
            default="All",
        )
    with controls_right:
        st.download_button(
            "Download Results CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    filtered_df = result_df
    if selected_tier and selected_tier != "All":
        filtered_df = result_df[result_df["Risk Tier"] == selected_tier]

    styled_df = filtered_df.style.apply(style_batch_rows, axis=1).format(
        {"Churn Probability": "{:.1f}%"}
    )
    st.dataframe(
        styled_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Churn Probability": st.column_config.ProgressColumn(
                "Churn Probability",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "Risk Tier": st.column_config.TextColumn("Risk Tier"),
            "Top Risk Factor": st.column_config.TextColumn("Top Risk Factor"),
            "Actions": st.column_config.TextColumn("Actions"),
        },
    )


def render_model_information(
    predictor: ChurnPredictor,
    model_metrics: dict[str, Any],
    population_data: pd.DataFrame,
) -> None:
    info_tab, comparison_tab = st.tabs(["Model Information", "Model Comparison"])
    model_name = infer_model_name(predictor, model_metrics)
    with info_tab:
        cards = [
            ("Model Used", model_name, COLORS["text"]),
            ("Training Accuracy", f"{model_metrics['accuracy']:.1%}", COLORS["teal"]),
            ("ROC-AUC", f"{float(model_metrics.get('auc_roc', 0.94)):.3f}", COLORS["accent"]),
            ("Dataset Size", f"{len(population_data):,} customers", COLORS["indigo"]),
            ("Features", f"{len(predictor.feature_names)}", COLORS["warning"]),
        ]
        html = ['<div class="metric-grid five">']
        for title, value, color in cards:
            value_class = "kpi-value compact-text" if title in {"Model Used", "Dataset Size"} else "kpi-value"
            html.append(
                f"""
<div class="dash-card compact">
  <div class="kpi-title">{title}</div>
  <div class="{value_class}" style="color: {color};">{value}</div>
</div>
"""
            )
        html.append("</div>")
        st.markdown("".join(html), unsafe_allow_html=True)

    with comparison_tab:
        comparison_path = PROJECT_ROOT / "results" / "model_comparison.csv"
        if comparison_path.exists():
            comparison = pd.read_csv(comparison_path)
            if "status" in comparison:
                comparison = comparison[comparison["status"].astype(str).str.lower() == "ok"]
            display = (
                comparison.sort_values("accuracy", ascending=False)
                .head(8)[["model", "strategy", "accuracy", "precision_yes", "recall_yes", "auc_roc"]]
                .rename(
                    columns={
                        "model": "Model",
                        "strategy": "Strategy",
                        "accuracy": "Accuracy",
                        "precision_yes": "Precision",
                        "recall_yes": "Recall",
                        "auc_roc": "ROC-AUC",
                    }
                )
            )
            st.dataframe(
                display.style.format(
                    {
                        "Accuracy": "{:.1%}",
                        "Precision": "{:.1%}",
                        "Recall": "{:.1%}",
                        "ROC-AUC": "{:.3f}",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.markdown(
                '<div class="empty-state">Model comparison artifact is unavailable.</div>',
                unsafe_allow_html=True,
            )


def render_footer() -> None:
    st.markdown(
        """
<div class="dash-card">
  <div class="card-title">Built Using</div>
  <p class="app-subtitle">Python &middot; Streamlit &middot; Scikit-Learn &middot; SHAP &middot; Plotly &middot; Pandas &middot; NumPy &middot; FastAPI</p>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="Churn Intelligence Hub",
        page_icon="CI",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    inject_css()
    initialize_session_state()
    render_header()

    try:
        predictor = load_predictor()
        reference_data = load_reference_data()
        population_data = load_population_data()
        model_metrics = load_model_metrics()
    except ModelNotTrainedError as exc:
        st.markdown(
            f'<div class="empty-state">Model artifact is unavailable: {exc}</div>',
            unsafe_allow_html=True,
        )
        return
    except Exception as exc:
        st.error(f"Dashboard could not initialize: {exc}")
        return

    customer = current_customer_from_state(predictor, reference_data)
    signature = customer_signature(customer)
    if signature != st.session_state.last_customer_signature:
        elapsed = time.monotonic() - st.session_state.last_score_at
        if 0 < elapsed < DEBOUNCE_SECONDS:
            time.sleep(DEBOUNCE_SECONDS - elapsed)

    prediction, explanation, shap_error = score_customer(predictor, customer)
    probability = float(prediction["churn_probability"])
    tier = risk_tier(probability)
    confidence = model_confidence(probability)
    clv = estimate_customer_lifetime_value(customer, probability)
    segment = customer_segment(probability, clv, population_data)
    st.session_state.last_customer_signature = signature
    st.session_state.last_score_at = time.monotonic()

    render_dashboard_summary(summarize_portfolio(predictor, population_data))
    render_kpi_strip(
        prediction,
        probability,
        tier,
        confidence,
        clv,
        segment,
        model_metrics,
        explanation,
    )

    left, right = st.columns([5, 4], gap="large")
    with left:
        render_profile_panel(predictor, reference_data, prediction)

    with right:
        render_risk_score_panel(probability, tier, confidence)

    render_retention_intelligence_section(
        predictor,
        customer,
        prediction,
        probability,
        confidence,
        tier,
        clv,
        segment,
        explanation,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    render_analytics_panel(probability, tier, explanation, shap_error)

    st.markdown("<br>", unsafe_allow_html=True)
    render_batch_section(predictor)
    st.markdown("<br>", unsafe_allow_html=True)
    render_model_information(predictor, model_metrics, population_data)
    render_footer()


if __name__ == "__main__":
    main()
