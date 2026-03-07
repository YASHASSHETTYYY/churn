from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config, resolve_path
from src.models.predict import ChurnPredictor


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


def default_value(column: str, reference_data: pd.DataFrame) -> float:
    if column in reference_data:
        return float(reference_data[column].median())
    return 0.0


def render_sidebar(reference_data: pd.DataFrame) -> dict[str, float]:
    st.sidebar.header("Customer Profile")
    return {
        "number_vmail_messages": st.sidebar.number_input(
            "Voice mail messages",
            min_value=0.0,
            value=default_value("number_vmail_messages", reference_data),
            step=1.0,
        ),
        "total_day_calls": st.sidebar.number_input(
            "Day calls",
            min_value=0.0,
            value=default_value("total_day_calls", reference_data),
            step=1.0,
        ),
        "total_eve_minutes": st.sidebar.number_input(
            "Evening minutes",
            min_value=0.0,
            value=default_value("total_eve_minutes", reference_data),
        ),
        "total_eve_charge": st.sidebar.number_input(
            "Evening charge",
            min_value=0.0,
            value=default_value("total_eve_charge", reference_data),
        ),
        "total_intl_minutes": st.sidebar.number_input(
            "International minutes",
            min_value=0.0,
            value=default_value("total_intl_minutes", reference_data),
        ),
        "number_customer_service_calls": st.sidebar.number_input(
            "Customer service calls",
            min_value=0.0,
            value=default_value("number_customer_service_calls", reference_data),
            step=1.0,
        ),
    }


def main():
    st.set_page_config(page_title="Customer Churn Workbench", layout="wide")
    st.title("Customer Churn Workbench")
    st.caption(
        "Interactive scoring, batch prediction, and SHAP-based explanations "
        "for the production churn model."
    )

    predictor = load_predictor()
    reference_data = load_reference_data()
    customer = render_sidebar(reference_data)

    left, right = st.columns([1.1, 1.4])

    with left:
        st.subheader("Single Prediction")
        st.dataframe(pd.DataFrame([customer]), hide_index=True, use_container_width=True)

        if st.button("Predict churn", type="primary"):
            prediction = predictor.predict_one(customer)
            explanation = predictor.explain(customer)

            st.metric("Predicted label", prediction["churn"])
            st.metric(
                "Churn probability",
                f'{prediction["churn_probability"]:.2%}',
            )

            top_factors = pd.DataFrame(explanation["top_factors"])
            st.subheader("Top churn factors")
            st.bar_chart(
                top_factors.set_index("feature")["shap_value"],
                use_container_width=True,
            )
            st.dataframe(top_factors, hide_index=True, use_container_width=True)

    with right:
        st.subheader("Batch Prediction")
        st.write(
            "Upload a CSV with the model feature columns to score multiple "
            "customers at once."
        )
        upload = st.file_uploader("Batch CSV", type="csv")
        if upload is not None:
            batch_df = pd.read_csv(upload)
            predictions = predictor.predict(batch_df)
            result_df = pd.concat(
                [batch_df.reset_index(drop=True), pd.DataFrame(predictions)],
                axis=1,
            )
            st.dataframe(result_df, use_container_width=True)
            st.download_button(
                label="Download predictions",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="churn_predictions.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
