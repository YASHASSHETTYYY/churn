from __future__ import annotations

import json
import warnings
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _coerce_binary_target(y_true: Any, positive_label: int | str = 1) -> np.ndarray:
    y_array = np.asarray(y_true)
    if y_array.dtype.kind in {"b", "i", "u", "f"}:
        return y_array.astype(int)
    return (y_array == positive_label).astype(int)


def _get_model_classes(model) -> list[Any]:
    if hasattr(model, "classes_"):
        return list(model.classes_)
    if hasattr(model, "named_steps"):
        classifier = model.named_steps.get("classifier")
        if classifier is not None and hasattr(classifier, "classes_"):
            return list(classifier.classes_)
    return [0, 1]


def get_positive_class_scores(model, x_data, positive_label: int | str = 1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)

        if hasattr(model, "predict_proba"):
            probabilities = np.asarray(model.predict_proba(x_data))
            classes = _get_model_classes(model)
            positive_index = classes.index(positive_label) if positive_label in classes else -1
            return probabilities[:, positive_index]

        if hasattr(model, "decision_function"):
            return np.asarray(model.decision_function(x_data), dtype=float)

        return np.asarray(model.predict(x_data), dtype=float)


def predict_labels_from_scores(
    y_scores: np.ndarray,
    threshold: float = 0.5,
    positive_label: int | str = 1,
    negative_label: int | str = 0,
) -> np.ndarray:
    return np.where(np.asarray(y_scores) >= threshold, positive_label, negative_label)


def find_optimal_f1_threshold(
    y_true,
    y_scores,
    positive_label: int | str = 1,
    default_threshold: float = 0.5,
) -> float:
    y_binary = _coerce_binary_target(y_true, positive_label=positive_label)
    y_scores_array = np.asarray(y_scores, dtype=float)

    precision, recall, thresholds = precision_recall_curve(y_binary, y_scores_array)
    if thresholds.size == 0:
        return default_threshold

    candidate_precision = precision[:-1]
    candidate_recall = recall[:-1]
    f1_scores = np.divide(
        2 * candidate_precision * candidate_recall,
        candidate_precision + candidate_recall,
        out=np.zeros_like(candidate_precision),
        where=(candidate_precision + candidate_recall) > 0,
    )

    best_index = int(np.nanargmax(f1_scores))
    return float(thresholds[best_index])


def compute_binary_classification_metrics(
    y_true,
    y_scores,
    *,
    threshold: float = 0.5,
    positive_label: int | str = 1,
    negative_label: int | str = 0,
    class_name_map: dict[int | str, str] | None = None,
) -> dict[str, Any]:
    y_true_binary = _coerce_binary_target(y_true, positive_label=positive_label)
    y_scores_array = np.asarray(y_scores, dtype=float)
    y_pred = predict_labels_from_scores(
        y_scores_array,
        threshold=threshold,
        positive_label=positive_label,
        negative_label=negative_label,
    )
    y_pred_binary = _coerce_binary_target(y_pred, positive_label=positive_label)

    label_order = [negative_label, positive_label]
    class_name_map = class_name_map or {
        negative_label: str(negative_label),
        positive_label: str(positive_label),
    }

    per_class_precision, per_class_recall, per_class_f1, supports = (
        precision_recall_fscore_support(
            y_true_binary,
            y_pred_binary,
            labels=[0, 1],
            zero_division=0,
        )
    )
    cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true_binary, y_pred_binary)),
        "precision_macro": float(
            precision_score(
                y_true_binary,
                y_pred_binary,
                average="macro",
                zero_division=0,
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true_binary,
                y_pred_binary,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true_binary,
                y_pred_binary,
                average="macro",
                zero_division=0,
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true_binary,
                y_pred_binary,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_positive_class": float(
            f1_score(
                y_true_binary,
                y_pred_binary,
                pos_label=1,
                zero_division=0,
            )
        ),
        "auc_roc": float(roc_auc_score(y_true_binary, y_scores_array)),
        "pr_auc": float(average_precision_score(y_true_binary, y_scores_array)),
        "confusion_matrix": json.dumps(cm.tolist()),
    }

    for index, label in enumerate(label_order):
        label_name = class_name_map.get(label, str(label))
        metrics[f"precision_{label_name}"] = float(per_class_precision[index])
        metrics[f"recall_{label_name}"] = float(per_class_recall[index])
        metrics[f"f1_{label_name}"] = float(per_class_f1[index])
        metrics[f"support_{label_name}"] = int(supports[index])

    return metrics
