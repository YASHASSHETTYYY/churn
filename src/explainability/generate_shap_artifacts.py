from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import shap

from src.config import load_config, resolve_path


def _get_model_components(model) -> tuple[Any | None, Any]:
    if hasattr(model, "named_steps"):
        return model.named_steps.get("preprocessor"), model.named_steps.get("classifier")
    return None, model


def _positive_class_index(model, positive_label: str) -> int:
    classes = list(getattr(model, "classes_", []))
    if positive_label in classes:
        return classes.index(positive_label)
    if len(classes) > 1:
        return 1
    return 0


def _map_transformed_feature(transformed_name: str, raw_feature_names: list[str]) -> str:
    suffix = transformed_name.split("__", 1)[-1]
    for feature_name in sorted(raw_feature_names, key=len, reverse=True):
        if suffix == feature_name or suffix.startswith(f"{feature_name}_"):
            return feature_name
    return suffix


def _aggregate_shap_values(
    shap_values: np.ndarray,
    transformed_feature_names: list[str],
    raw_feature_names: list[str],
) -> np.ndarray:
    feature_index = {name: idx for idx, name in enumerate(raw_feature_names)}
    aggregated = np.zeros((shap_values.shape[0], len(raw_feature_names)), dtype=float)

    for transformed_idx, transformed_name in enumerate(transformed_feature_names):
        raw_feature = _map_transformed_feature(transformed_name, raw_feature_names)
        raw_idx = feature_index.get(raw_feature)
        if raw_idx is not None:
            aggregated[:, raw_idx] += shap_values[:, transformed_idx]
    return aggregated


def _normalize_shap_output(
    shap_output,
    expected_value,
    positive_index: int,
) -> tuple[np.ndarray, float]:
    if isinstance(shap_output, list):
        values = np.asarray(shap_output[positive_index], dtype=float)
        base_value = float(np.asarray(expected_value)[positive_index])
        return values, base_value

    values = np.asarray(shap_output, dtype=float)
    if values.ndim == 3:
        values = values[:, :, positive_index]
        base_value = float(np.asarray(expected_value)[positive_index])
        return values, base_value

    base_value_array = np.asarray(expected_value, dtype=float)
    base_value = float(base_value_array.ravel()[0]) if base_value_array.size else 0.0
    return values, base_value


def generate_shap_summary(
    *,
    model_bundle_path: str | Path,
    feature_frame: pd.DataFrame,
    output_path: str | Path,
) -> dict[str, str | float]:
    bundle = joblib.load(model_bundle_path)
    model = bundle["model"]
    metadata = bundle.get("metadata", {})
    raw_feature_names = metadata.get("feature_names", feature_frame.columns.tolist())
    raw_feature_names = [name for name in raw_feature_names if name in feature_frame.columns]
    positive_label = str(metadata.get("positive_label", "yes"))

    frame = feature_frame[raw_feature_names].copy()
    preprocessor, estimator = _get_model_components(model)
    transformed = frame
    transformed_feature_names = raw_feature_names

    if preprocessor is not None:
        transformed = preprocessor.transform(frame)
        transformed_feature_names = list(preprocessor.get_feature_names_out())
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    explainer = shap.TreeExplainer(estimator)
    shap_output = explainer.shap_values(transformed)
    positive_index = _positive_class_index(model, positive_label)
    shap_values, base_value = _normalize_shap_output(
        shap_output,
        getattr(explainer, "expected_value", 0.0),
        positive_index,
    )
    aggregated = _aggregate_shap_values(
        np.asarray(shap_values, dtype=float),
        transformed_feature_names,
        raw_feature_names,
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(11, 7))
    shap.summary_plot(
        aggregated,
        features=frame,
        feature_names=raw_feature_names,
        plot_type="bar",
        max_display=min(20, len(raw_feature_names)),
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close("all")

    return {
        "plot_path": str(output),
        "base_value": float(base_value),
    }


def run_shap_stage(config_path: str | Path = "params.yaml") -> dict[str, str | float]:
    config = load_config(config_path)
    test_data_path = resolve_path(config["processed_data_config"]["test_data_csv"], config_path)
    artifact_path = resolve_path(config["training"]["artifact_path"], config_path)
    shap_plot_path = resolve_path("plots/shap_summary_train.png", config_path)

    test_df = pd.read_csv(test_data_path)
    target = config["raw_data_config"]["target"]
    feature_frame = test_df.drop(columns=[target])

    return generate_shap_summary(
        model_bundle_path=artifact_path,
        feature_frame=feature_frame,
        output_path=shap_plot_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    run_shap_stage(config_path=args.config)
