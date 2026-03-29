from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.config import load_config, resolve_path

try:
    import shap
except ImportError:  # pragma: no cover - optional in some local envs
    shap = None


class ModelNotTrainedError(FileNotFoundError):
    """Raised when prediction is requested before a model artifact exists."""


class ChurnPredictor:
    """Shared prediction service used by the API, dashboard, and monitoring."""

    def __init__(
        self,
        artifact_path: str | Path | None = None,
        config_path: str | Path = "params.yaml",
    ) -> None:
        self.config_path = config_path
        config = load_config(config_path)
        training_config = config["training"]
        raw_config = config["raw_data_config"]

        artifact = artifact_path or training_config["artifact_path"]
        self.artifact_path = resolve_path(artifact, config_path)
        if not self.artifact_path.exists():
            raise ModelNotTrainedError(
                f"Model artifact not found at {self.artifact_path}."
            )

        bundle = joblib.load(self.artifact_path)
        self.model = bundle["model"]
        self.metadata = bundle.get("metadata", {})
        self.target = self.metadata.get("target", raw_config["target"])
        feature_names = self.metadata.get("feature_names", raw_config["model_var"])
        self.feature_names = [name for name in feature_names if name != self.target]
        self.feature_dtypes = self.metadata.get("feature_dtypes", {})
        self.positive_label = self.metadata.get(
            "positive_label",
            raw_config.get("positive_class", "yes"),
        )
        self._explainer = None

    def _to_frame(self, records: Any) -> pd.DataFrame:
        if isinstance(records, pd.DataFrame):
            frame = records.copy()
        elif isinstance(records, dict):
            frame = pd.DataFrame([records])
        else:
            frame = pd.DataFrame(records)

        missing = [column for column in self.feature_names if column not in frame]
        if missing:
            raise ValueError(f"Missing required features: {', '.join(missing)}")

        frame = frame[self.feature_names].copy()
        for column in self.feature_names:
            dtype_name = self.feature_dtypes.get(column, "")
            if any(token in dtype_name for token in ["int", "float"]):
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
            else:
                frame[column] = frame[column].astype(str)

        return frame

    def _get_model_components(self):
        if isinstance(self.model, Pipeline):
            preprocessor = self.model.named_steps.get("preprocessor")
            estimator = self.model.named_steps.get("classifier")
            return preprocessor, estimator
        return None, self.model

    def _positive_class_index(self) -> int:
        classes = list(getattr(self.model, "classes_", []))
        if self.positive_label in classes:
            return classes.index(self.positive_label)
        if len(classes) > 1:
            return 1
        return 0

    def predict_proba(self, records: Any) -> np.ndarray:
        frame = self._to_frame(records)
        if hasattr(self.model, "predict_proba"):
            probabilities = self.model.predict_proba(frame)
            return probabilities[:, self._positive_class_index()]

        predictions = self.model.predict(frame)
        return np.asarray(
            [
                1.0 if str(prediction) == str(self.positive_label) else 0.0
                for prediction in predictions
            ],
            dtype=float,
        )

    def predict(self, records: Any) -> list[dict[str, float | str]]:
        frame = self._to_frame(records)
        labels = self.model.predict(frame)
        probabilities = self.predict_proba(frame)
        return [
            {
                "churn": str(label),
                "churn_probability": float(probability),
            }
            for label, probability in zip(labels, probabilities)
        ]

    def predict_one(self, record: dict[str, float]) -> dict[str, float | str]:
        return self.predict(record)[0]

    def _get_explainer(self):
        if shap is None:
            raise RuntimeError("SHAP is not installed.")
        if self._explainer is None:
            _, estimator = self._get_model_components()
            self._explainer = shap.TreeExplainer(estimator)
        return self._explainer

    def _compute_shap_values(
        self,
        frame: pd.DataFrame,
    ) -> tuple[np.ndarray, float | None, list[str]]:
        explainer = self._get_explainer()
        preprocessor, _ = self._get_model_components()
        transformed = frame
        shap_feature_names = self.feature_names

        if preprocessor is not None:
            transformed = preprocessor.transform(frame)
            shap_feature_names = list(preprocessor.get_feature_names_out())

        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        shap_values = explainer.shap_values(transformed)
        expected_value = getattr(explainer, "expected_value", None)

        if isinstance(shap_values, list):
            class_index = self._positive_class_index()
            values = np.asarray(shap_values[class_index])
            if isinstance(expected_value, (list, tuple, np.ndarray)):
                expected_value = float(np.asarray(expected_value)[class_index])
            return values, expected_value, shap_feature_names

        values = np.asarray(shap_values)
        if values.ndim == 3:
            class_index = self._positive_class_index()
            values = values[:, :, class_index]
            if isinstance(expected_value, (list, tuple, np.ndarray)):
                expected_value = float(np.asarray(expected_value)[class_index])
        elif isinstance(expected_value, (list, tuple, np.ndarray)):
            expected_value = float(np.asarray(expected_value).ravel()[0])

        return values, expected_value, shap_feature_names

    def _map_transformed_feature(self, transformed_name: str) -> str:
        suffix = transformed_name.split("__", 1)[-1]
        for feature_name in sorted(self.feature_names, key=len, reverse=True):
            if suffix == feature_name or suffix.startswith(f"{feature_name}_"):
                return feature_name
        return suffix

    def explain(
        self,
        record: dict[str, Any] | pd.DataFrame,
        top_k: int = 5,
    ) -> dict[str, Any]:
        frame = self._to_frame(record)
        prediction = self.predict(frame)[0]
        shap_values, expected_value, shap_feature_names = self._compute_shap_values(frame)

        contributions_by_feature: dict[str, dict[str, Any]] = {}
        row_values = frame.iloc[0]
        for transformed_name, shap_value in zip(
            shap_feature_names,
            shap_values[0].tolist(),
        ):
            feature_name = self._map_transformed_feature(transformed_name)
            if feature_name not in contributions_by_feature:
                feature_value = row_values[feature_name]
                contributions_by_feature[feature_name] = {
                    "feature": feature_name,
                    "feature_value": feature_value.item()
                    if hasattr(feature_value, "item")
                    else feature_value,
                    "shap_value": 0.0,
                }
            contributions_by_feature[feature_name]["shap_value"] += float(shap_value)

        ranked = sorted(
            contributions_by_feature.values(),
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )

        return {
            "prediction": prediction,
            "base_value": expected_value,
            "top_factors": ranked[:top_k],
        }
