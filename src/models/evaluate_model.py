from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd

from src.config import load_config, resolve_path
from src.evaluation.metrics import (
    compute_binary_classification_metrics,
    get_positive_class_scores,
)


def evaluate_saved_model(config_path: str | Path = "params.yaml") -> dict:
    config = load_config(config_path)
    test_data_path = resolve_path(config["processed_data_config"]["test_data_csv"], config_path)
    artifact_path = resolve_path(config["training"]["artifact_path"], config_path)
    metrics_path = resolve_path(config["training"]["metrics_path"], config_path)

    test_df = pd.read_csv(test_data_path)
    target = config["raw_data_config"]["target"]
    positive_label = config["raw_data_config"].get("positive_class", "yes")

    x_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    bundle = joblib.load(artifact_path)
    model = bundle["model"]
    scores = get_positive_class_scores(model, x_test, positive_label=positive_label)
    metrics = compute_binary_classification_metrics(
        y_test,
        scores,
        threshold=0.5,
        positive_label=positive_label,
        negative_label="no" if positive_label == "yes" else 0,
        class_name_map={0: "no", 1: "yes"},
    )

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml")
    args = parser.parse_args()
    evaluate_saved_model(config_path=args.config)
