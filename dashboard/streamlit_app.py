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


TOP_FACTORS = 8
BATCH_SHAP_LIMIT = 250
DEBOUNCE_SECONDS = 0.3

COLORS = {
    "background": "#0F1117",
    "surface": "#1C1E26",
    "border": "#2A2D3A",
    "teal": "#1D9E75",
    "coral": "#D85A30",
    "indigo": "#534AB7",
    "text": "#F0EEF8",
    "muted": "#9B99AA",
    "success": "#639922",
    "warning": "#BA7517",
    "danger": "#A32D2D",
}

FIELD_GROUPS = {
    "Account Info": [
        "state",
        "account_length",
        "area_code",
        "international_plan",
        "voice_mail_plan",
        "number_vmail_messages",
        "number_customer_service_calls",
    ],
    "Usage - Day": [
        "total_day_minutes",
        "total_day_calls",
        "total_day_charge",
    ],
    "Usage - Eve/Night": [
        "total_eve_minutes",
        "total_eve_calls",
        "total_eve_charge",
        "total_night_minutes",
        "total_night_calls",
        "total_night_charge",
    ],
    "International": [
        "total_intl_minutes",
        "total_intl_calls",
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

  .summary-grid {
    display: grid;
    gap: 12px;
    grid-template-columns: repeat(4, minmax(0, 1fr));
  }

  @media (max-width: 820px) {
    .block-container { padding: 18px 16px 32px; }
    .app-header { flex-direction: column; }
    .header-meta { justify-content: flex-start; }
    .summary-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .feature-row { grid-template-columns: 1fr; }
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
    p = min(max(float(probability), 1e-12), 1.0 - 1e-12)
    entropy = -(p * math.log(p) + (1 - p) * math.log(1 - p))
    confidence = 1 - (entropy / math.log(2))
    return max(0.0, min(1.0, confidence))


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


def build_customer_inputs(
    predictor: ChurnPredictor,
    reference_data: pd.DataFrame,
) -> dict[str, Any]:
    customer: dict[str, Any] = {}
    grouped = set()

    for group_name, fields in FIELD_GROUPS.items():
        available_fields = [field for field in fields if field in predictor.feature_names]
        if not available_fields:
            continue
        with st.expander(group_name, expanded=group_name == "Account Info"):
            for column in available_fields:
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
    <div class="app-title">Churn Intelligence Hub</div>
    <p class="app-subtitle">Real-time churn scoring &middot; SHAP explainability &middot; Batch analytics</p>
  </div>
  <div class="header-meta">
    <span>Last updated {timestamp}</span>
    <span class="status-pill">Deploy: Live</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_kpi_strip(
    probability: float,
    tier: str,
    explanation: dict[str, Any] | None,
) -> None:
    kpi_cols = st.columns(4)
    confidence = model_confidence(probability)
    kpis = [
        {
            "title": "Churn Probability",
            "value": f"{probability:.1%}",
            "note": "Predicted probability",
            "color": probability_color(probability),
        },
        {
            "title": "Risk Tier",
            "value": tier,
            "note": "Low <20%, Medium 20-50%, High >50%",
            "color": risk_color(tier),
        },
        {
            "title": "Top Risk Factor",
            "value": top_risk_factor(explanation),
            "note": "Highest absolute SHAP impact",
            "color": COLORS["indigo"],
        },
        {
            "title": "Model Confidence",
            "value": f"{confidence:.1%}",
            "note": confidence_note(probability),
            "color": COLORS["teal"] if confidence >= 0.5 else COLORS["warning"],
        },
    ]
    for col, kpi in zip(kpi_cols, kpis):
        with col:
            st.markdown(
                f"""
<div class="dash-card compact">
  <div class="kpi-title">{kpi["title"]}</div>
  <div class="kpi-value" style="color: {kpi["color"]};">{kpi["value"]}</div>
  <div class="kpi-note">{kpi["note"]}</div>
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
    gauge_value = probability * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=gauge_value,
            number={
                "suffix": "%",
                "font": {"size": 42, "color": COLORS["text"]},
                "valueformat": ".1f",
            },
            title={
                "text": f"{tier} Risk",
                "font": {"size": 16, "color": COLORS["muted"]},
            },
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 1,
                    "tickcolor": COLORS["muted"],
                    "tickfont": {"color": COLORS["muted"]},
                },
                "bar": {"color": probability_color(probability), "thickness": 0.18},
                "bgcolor": COLORS["surface"],
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30], "color": "rgba(99,153,34,0.32)"},
                    {"range": [30, 60], "color": "rgba(186,117,23,0.32)"},
                    {"range": [60, 100], "color": "rgba(163,45,45,0.36)"},
                ],
                "threshold": {
                    "line": {"color": COLORS["text"], "width": 4},
                    "thickness": 0.85,
                    "value": gauge_value,
                },
                "shape": "angular",
            },
        )
    )
    fig.update_layout(
        paper_bgcolor=COLORS["surface"],
        plot_bgcolor=COLORS["surface"],
        font={"color": COLORS["text"], "family": "Inter, system-ui"},
        height=280,
        margin={"l": 24, "r": 24, "t": 34, "b": 14},
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
            hovertemplate=(
                "Feature: %{customdata[0]}<br>"
                "Value: %{customdata[1]}<br>"
                "SHAP: %{x:.3f}<br>"
                "%{customdata[2]}<extra></extra>"
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
        direction = "&uarr; Risk" if row["shap_value"] >= 0 else "&darr; Risk"
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
    st.markdown('<div class="card-title">Risk Gauge</div>', unsafe_allow_html=True)
    st.plotly_chart(make_gauge(probability, tier), use_container_width=True)

    if shap_error:
        st.markdown(
            f'<div class="empty-state">SHAP explanations are unavailable: {shap_error}</div>',
            unsafe_allow_html=True,
        )
        return

    shap_df = explanation_frame(explanation)
    st.markdown(
        '<div class="card-title">What&#39;s driving this prediction?</div>',
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
  <div class="card-title">Batch Prediction</div>
  <p class="app-subtitle">Upload a CSV with the model feature columns to score customers at scale.</p>
</div>
""",
        unsafe_allow_html=True,
    )
    upload = st.file_uploader(
        "Drop a customer CSV here",
        type="csv",
        label_visibility="collapsed",
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
    st.session_state.last_customer_signature = signature
    st.session_state.last_score_at = time.monotonic()

    render_kpi_strip(probability, tier, explanation)

    left, right = st.columns([4, 6], gap="large")
    with left:
        render_profile_panel(predictor, reference_data, prediction)

    with right:
        render_analytics_panel(probability, tier, explanation, shap_error)

    st.markdown("<br>", unsafe_allow_html=True)
    render_batch_section(predictor)


if __name__ == "__main__":
    main()
