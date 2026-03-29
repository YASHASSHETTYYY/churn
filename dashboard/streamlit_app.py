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


def render_sidebar(
    predictor: ChurnPredictor,
    reference_data: pd.DataFrame,
) -> dict[str, object]:
    st.sidebar.header("Customer Profile")
    customer: dict[str, object] = {}
    for column in predictor.feature_names:
        if column in reference_data and pd.api.types.is_numeric_dtype(reference_data[column]):
            customer[column] = st.sidebar.number_input(
                column.replace("_", " ").title(),
                min_value=0.0,
                value=default_value(column, reference_data),
            )
        else:
            options = [""] if column not in reference_data else sorted(
                reference_data[column].dropna().astype(str).unique().tolist()
            )
            customer[column] = st.sidebar.selectbox(
                column.replace("_", " ").title(),
                options=options,
                index=0,
            )
    return customer


def main():
    st.set_page_config(page_title="Customer Churn Workbench", layout="wide")
    st.title("Customer Churn Workbench")
    st.caption(
        "Interactive scoring, batch prediction, and SHAP-based explanations "
        "for the production churn model."
    )

    predictor = load_predictor()
    reference_data = load_reference_data()
    customer = render_sidebar(predictor, reference_data)

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
